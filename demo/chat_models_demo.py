import argparse
import json
import pathlib
import sys
import time


def check_line_direction(baseline_seg):
    """Ensures baseline points run top to bottom for vertical text."""
    for line in baseline_seg.lines:
        if line.baseline and line.baseline[0][1] > line.baseline[-1][1]:
            line.baseline.reverse()
    return baseline_seg


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CHAT Kraken OCR demo in debug-friendly mode."
    )
    parser.add_argument(
        "--image",
        action="append",
        help="Specific PNG filename or path to process. May be used multiple times.",
    )
    parser.add_argument(
        "--list-images",
        action="store_true",
        help="List available demo/test PNGs and exit.",
    )
    parser.add_argument(
        "--segmentation-mode",
        choices=("baseline", "bbox"),
        default="baseline",
        help="Use CHAT baseline segmentation or Kraken bbox fallback.",
    )
    parser.add_argument(
        "--binarize",
        dest="binarize",
        action="store_true",
        default=True,
        help="Binarize image before OCR. Default: on.",
    )
    parser.add_argument(
        "--no-binarize",
        dest="binarize",
        action="store_false",
        help="Keep grayscale input. Useful when hard threshold hurts segmentation.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Binarization threshold. Default: 128.",
    )
    parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="Raise baseline polygonizer errors instead of logging warnings.",
    )
    parser.add_argument(
        "--legacy-polygons",
        dest="no_legacy_polygons",
        action="store_false",
        help="Use the old legacy polygon extractor in recognition for debugging.",
    )
    parser.set_defaults(no_legacy_polygons=True)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Optional directory for OCR text, segmentation overlay, and geometry dumps.",
    )
    return parser.parse_args()


def resolve_image_paths(image_args, image_paths):
    if not image_args:
        return image_paths

    image_map = {path.name: path for path in image_paths}
    selected = []
    for image_arg in image_args:
        image_path = pathlib.Path(image_arg)
        if image_path.exists():
            selected.append(image_path.resolve())
        elif image_arg in image_map:
            selected.append(image_map[image_arg])
        else:
            raise FileNotFoundError(f"Image not found: {image_arg}")
    return selected


def maybe_binarize(img, threshold):
    gray = img.convert("L")
    return gray.point(lambda x: 0 if x < threshold else 255, "1")


def dump_text(output_dir, image_path, records):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}.kraken.txt"
    output_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    log(f"wrote OCR text to {output_path}")


def dump_segmentation_data(output_dir, image_path, seg):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}.segmentation.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, line in enumerate(seg.lines, start=1):
            payload = {
                "index": idx,
                "type": seg.type,
            }
            if getattr(line, "baseline", None) is not None:
                payload["baseline"] = line.baseline
            if getattr(line, "boundary", None) is not None:
                payload["boundary"] = line.boundary
            if getattr(line, "bbox", None) is not None:
                payload["bbox"] = line.bbox
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    log(f"wrote segmentation data to {output_path}")


def dump_segmentation_overlay(output_dir, image_path, src_img, seg):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}.overlay.png"
    overlay = src_img.convert("RGB").copy()

    from PIL import ImageDraw

    draw = ImageDraw.Draw(overlay)
    for line in seg.lines:
        bbox = getattr(line, "bbox", None)
        boundary = getattr(line, "boundary", None)
        baseline = getattr(line, "baseline", None)

        if bbox is not None:
            draw.rectangle(bbox, outline=(255, 0, 0), width=2)
        if boundary:
            draw.line([tuple(point) for point in boundary] + [tuple(boundary[0])], fill=(255, 0, 0), width=2)
        if baseline:
            draw.line([tuple(point) for point in baseline], fill=(0, 200, 255), width=2)

    overlay.save(output_path)
    log(f"wrote segmentation overlay to {output_path}")


def collect_raw_baselines(proc_img, seg_model):
    from kraken import blla
    from kraken.lib.segmentation import scale_regions, vectorize_lines

    rets = blla.compute_segmentation_map(proc_img, model=seg_model)
    cls_map = rets["cls_map"]
    start_sep = cls_map["aux"]["_start_separator"]
    end_sep = cls_map["aux"]["_end_separator"]
    baselines = []

    for baseline_type, idx in cls_map["baselines"].items():
        candidates = vectorize_lines(
            rets["heatmap"][(start_sep, end_sep, idx), :, :],
            text_direction="vertical",
        )
        scaled_candidates = scale_regions(candidates, rets["scale"])
        for baseline in scaled_candidates:
            baselines.append(
                {
                    "baseline_type": baseline_type,
                    "baseline": baseline,
                }
            )

    return baselines


def dump_raw_baselines(output_dir, image_path, src_img, baselines):
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / f"{image_path.stem}.raw-baselines.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for idx, baseline in enumerate(baselines, start=1):
            payload = {
                "index": idx,
                "baseline_type": baseline["baseline_type"],
                "baseline": baseline["baseline"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    log(f"wrote raw baselines to {data_path}")

    overlay_path = output_dir / f"{image_path.stem}.raw-baselines.overlay.png"
    overlay = src_img.convert("RGB").copy()

    from PIL import ImageDraw

    draw = ImageDraw.Draw(overlay)
    palette = {
        "default": (255, 80, 80),
        "DoubleLine": (80, 200, 255),
    }
    for baseline in baselines:
        color = palette.get(baseline["baseline_type"], (255, 170, 0))
        draw.line([tuple(point) for point in baseline["baseline"]], fill=color, width=2)

    overlay.save(overlay_path)
    log(f"wrote raw baseline overlay to {overlay_path}")

    counts = {}
    for baseline in baselines:
        counts[baseline["baseline_type"]] = counts.get(baseline["baseline_type"], 0) + 1
    log(f"raw baseline counts: {counts}")


if __name__ == "__main__":
    args = parse_args()

    try:
        from kraken import blla, pageseg, rpred
        from kraken.lib import models, vgsl
    except ImportError:
        print("Install kraken OCR engine to run this script: https://github.com/mittagessen/kraken#installation.")
        sys.exit(1)

    import torch
    from PIL import Image

    torch.set_num_threads(1)

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    models_dir = repo_root / "models"
    test_dir = repo_root / "test"
    if not test_dir.exists():
        test_dir = repo_root / "demo"

    seg_model_path = models_dir / "chat_seg.mlmodel"
    rec_model_path = models_dir / "chat_rec.mlmodel"
    assert seg_model_path.exists()
    assert rec_model_path.exists()

    image_paths = sorted(test_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {test_dir}")

    if args.list_images:
        for image_path in image_paths:
            print(image_path)
        sys.exit(0)

    image_paths = resolve_image_paths(args.image, image_paths)

    log(f"loading segmentation model: {seg_model_path}")
    seg_model = vgsl.TorchVGSLModel.load_model(seg_model_path)
    log(f"loading recognition model: {rec_model_path}")
    rec_model = models.load_any(rec_model_path)
    log("models loaded")
    if args.no_legacy_polygons:
        log("recognition uses modern polygon extraction by default for CHAT models")
    else:
        log("recognition uses legacy polygon extraction")
    log(f"processing {len(image_paths)} image(s) with {args.segmentation_mode} segmentation")

    for img_path in image_paths:
        start = time.time()
        log(f"opening {img_path}")
        src_img = Image.open(img_path)

        if args.binarize:
            log(f"binarizing {img_path.name} with threshold={args.threshold}")
            proc_img = maybe_binarize(src_img, args.threshold)
        else:
            log(f"keeping grayscale input for {img_path.name}")
            proc_img = src_img.convert("L")

        log(f"segmenting {img_path.name}")
        if args.segmentation_mode == "baseline":
            if args.output_dir:
                log(f"collecting raw baselines for {img_path.name}")
                raw_baselines = collect_raw_baselines(proc_img, seg_model)
                dump_raw_baselines(args.output_dir, pathlib.Path(img_path), src_img, raw_baselines)
            seg = blla.segment(
                proc_img,
                text_direction="vertical-rl",
                model=seg_model,
                raise_on_error=args.raise_on_error,
            )
            seg = check_line_direction(seg)
        else:
            seg = pageseg.segment(proc_img, text_direction="vertical-rl")
        log(f"segmentation done: type={seg.type}, lines={len(seg.lines)}")
        if args.output_dir:
            dump_segmentation_data(args.output_dir, pathlib.Path(img_path), seg)
            dump_segmentation_overlay(args.output_dir, pathlib.Path(img_path), src_img, seg)

        records = []
        log(f"recognizing {img_path.name}")
        for idx, record in enumerate(
            rpred.rpred(
                rec_model,
                proc_img,
                seg,
                no_legacy_polygons=args.no_legacy_polygons,
            ),
            start=1,
        ):
            text = str(record)
            records.append(text)
            print(text)
            log(f"record {idx}: {text}")

        log(f"finished {img_path.name} in {time.time() - start:.1f}s with {len(records)} record(s)")
        if args.output_dir:
            dump_text(args.output_dir, pathlib.Path(img_path), records)

    print("Done!")
