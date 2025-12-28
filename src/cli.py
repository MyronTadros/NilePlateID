"""CLI entrypoints for the pipeline."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import subprocess
from pathlib import Path

import cv2

from src.pipeline.associate import match_plates_to_cars
from src.pipeline.clean import execute_deletions, plan_deletions
from src.pipeline.crop import clamp_bbox, crop_image, validate_bbox
from src.pipeline.detect import run_detection
from src.pipeline.normalize import normalize_plate_id
from src.pipeline.ocr import read_plate_text
from src.pipeline.visualize import draw_debug, draw_plate_id_debug

LOGGER = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser(
        "detect",
        help="Run YOLO detection on an image or directory.",
    )
    detect_parser.add_argument(
        "--weights",
        default="models/best.pt",
        help="Path to YOLO weights (repo-relative by default).",
    )
    detect_parser.add_argument(
        "--input",
        default="data/incoming",
        help="Image file or directory (repo-relative by default).",
    )
    detect_parser.add_argument(
        "--out",
        default="data/meta/detections.json",
        help="Output JSON file (repo-relative by default).",
    )
    detect_parser.add_argument(
        "--debug_dir",
        default="data/meta/debug",
        help="Directory for annotated debug images (repo-relative by default).",
    )
    detect_parser.add_argument(
        "--crops_preview",
        action="store_true",
        help="Save a small sample of cropped detections for sanity checks.",
    )
    detect_parser.add_argument(
        "--crops_preview_dir",
        default="data/meta/crops_preview",
        help="Directory for crop previews (repo-relative by default).",
    )
    detect_parser.add_argument(
        "--preview_limit",
        type=int,
        default=10,
        help="Maximum number of preview crops to save.",
    )
    detect_parser.add_argument(
        "--pad",
        type=float,
        default=0.05,
        help="Padding fraction applied to crop-ready bounding boxes.",
    )
    detect_parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    detect_parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold.",
    )
    detect_parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device string (e.g., 'cpu', '0').",
    )
    detect_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite outputs if they already exist.",
    )

    clean_parser = subparsers.add_parser(
        "clean",
        help="Remove generated artifacts (safe list only).",
    )
    clean_parser.add_argument(
        "--repo_root",
        default=str(_repo_root()),
        help="Repository root (defaults to project root).",
    )
    clean_group = clean_parser.add_mutually_exclusive_group()
    clean_group.add_argument(
        "--dry_run",
        action="store_true",
        help="Print deletion plan without deleting (default).",
    )
    clean_group.add_argument(
        "--force",
        action="store_true",
        help="Delete planned artifacts.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run full detection, OCR, and crop pipeline.",
    )
    run_parser.add_argument(
        "--weights",
        default="models/best.pt",
        help="Path to YOLO weights (repo-relative by default).",
    )
    run_parser.add_argument(
        "--input",
        default="data/incoming",
        help="Image file or directory (repo-relative by default).",
    )
    run_parser.add_argument(
        "--gallery",
        default="data/gallery",
        help="Directory for car crops (repo-relative by default).",
    )
    run_parser.add_argument(
        "--plates",
        default="data/plates",
        help="Directory for plate crops (repo-relative by default).",
    )
    run_parser.add_argument(
        "--index",
        default="data/meta/index.csv",
        help="CSV index output path (repo-relative by default).",
    )
    run_parser.add_argument(
        "--debug_dir",
        default="data/meta/debug",
        help="Directory for annotated debug images (repo-relative by default).",
    )
    run_parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    run_parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold.",
    )
    run_parser.add_argument(
        "--pad",
        type=float,
        default=0.05,
        help="Padding fraction applied to crops.",
    )
    run_parser.add_argument(
        "--ocr_min_conf",
        type=float,
        default=0.05,
        help="Minimum OCR confidence for accepting a plate ID.",
    )
    run_parser.add_argument(
        "--max_debug",
        action="store_true",
        help="Enable maximum debugging outputs.",
    )
    run_parser.add_argument(
        "--preview_limit",
        type=int,
        default=10,
        help="Maximum number of preview crops to save in max debug mode.",
    )
    run_parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device string (e.g., 'cpu', '0').",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite outputs if they already exist.",
    )

    return parser.parse_args()


def _collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            [
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS
            ],
            key=lambda path: path.name,
        )
    return []


def _write_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def _split_by_role(
    detections: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cars: list[dict[str, object]] = []
    plates: list[dict[str, object]] = []

    for detection in detections:
        cls_name = str(detection.get("cls_name", "")).lower()
        if "plate" in cls_name:
            plates.append(detection)
        elif "car" in cls_name:
            cars.append(detection)

    return cars, plates


def _debug_output_path(debug_dir: Path, image_path: Path) -> Path:
    return debug_dir / f"{image_path.stem}_debug.jpg"


def _sanitize_label(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label)
    return safe.strip("_") or "unknown"


def _preview_output_path(
    preview_dir: Path, image_path: Path, index: int, cls_name: str
) -> Path:
    label = _sanitize_label(cls_name)
    return preview_dir / f"{image_path.stem}_{index:03d}_{label}.jpg"


def _run_debug_output_path(debug_dir: Path, image_path: Path) -> Path:
    return debug_dir / f"{image_path.stem}_run_debug.jpg"


def _det_sort_key(detection: dict[str, object]) -> tuple[object, ...]:
    bbox = detection.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
    try:
        coords = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        coords = (0.0, 0.0, 0.0, 0.0)
    return (
        coords[0],
        coords[1],
        coords[2],
        coords[3],
        str(detection.get("cls_name", "")),
        float(detection.get("conf", 0.0)),
    )


def _sorted_detections(
    detections: list[dict[str, object]],
) -> list[dict[str, object]]:
    return sorted(detections, key=_det_sort_key)


def _unknown_plate_id(seed: str) -> str:
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
    return f"unknown_{digest}"


def _plate_dir(base_dir: Path, plate_id: str) -> Path:
    return base_dir / _sanitize_label(plate_id)


def _image_timestamp(image_path: Path) -> str:
    mtime = image_path.stat().st_mtime
    return dt.datetime.fromtimestamp(mtime, tz=dt.timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _write_run_config(config_path: Path, args: argparse.Namespace) -> int:
    payload = {
        "timestamp": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(_repo_root()),
        "args": vars(args),
    }
    if config_path.exists() and not args.force:
        LOGGER.error("Run config exists (use --force): %s", config_path)
        return 2
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    LOGGER.info("Wrote run config: %s", config_path)
    return 0


def _run_detect(args: argparse.Namespace) -> int:
    weights_path = Path(args.weights)
    input_path = Path(args.input)
    json_path = Path(args.out)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    preview_dir = Path(args.crops_preview_dir) if args.crops_preview else None

    if not weights_path.is_file():
        LOGGER.error("Weights not found: %s", weights_path)
        return 2
    if json_path.exists() and not args.force:
        LOGGER.error("Output exists (use --force): %s", json_path)
        return 2

    image_paths = _collect_images(input_path)
    if not image_paths:
        LOGGER.error("No images found in: %s", input_path)
        return 2
    if debug_dir is not None and not args.force:
        existing = [
            _debug_output_path(debug_dir, path)
            for path in image_paths
            if _debug_output_path(debug_dir, path).exists()
        ]
        if existing:
            LOGGER.error("Debug outputs exist (use --force): %s", existing[0])
            return 2

    LOGGER.info("Detecting on %d image(s)", len(image_paths))
    detections = run_detection(
        weights_path=weights_path,
        image_paths=image_paths,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
    total_detections = 0
    invalid_bboxes = 0
    empty_crops = 0
    previews_saved = 0
    debug_saved = 0
    preview_remaining = args.preview_limit if preview_dir is not None else 0
    if debug_dir is not None:
        LOGGER.info("Writing debug images to: %s", debug_dir)
    if preview_dir is not None:
        LOGGER.info("Writing crop previews to: %s", preview_dir)

    for image_result in detections:
        image_path = Path(str(image_result["image_path"]))
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.error("Failed to read image: %s", image_path)
            return 2

        per_image = list(image_result.get("detections", []))
        total_detections += len(per_image)
        valid_detections: list[dict[str, object]] = []
        for index, detection in enumerate(per_image):
            bbox = detection.get("bbox_xyxy")
            if not bbox:
                invalid_bboxes += 1
                continue
            ok, reason = validate_bbox(bbox, image.shape)
            if not ok:
                invalid_bboxes += 1
                LOGGER.warning("Skipping bbox for %s (%s)", image_path, reason)
                continue
            if reason == "clamped":
                LOGGER.warning("Clamping bbox for %s", image_path)
            clamped = clamp_bbox(bbox, image.shape, pad=args.pad)
            detection["bbox_xyxy_crop"] = list(clamped)
            crop = crop_image(image, bbox, pad=args.pad)
            if crop.size == 0:
                empty_crops += 1
                LOGGER.warning("Empty crop for %s", image_path)
                continue
            valid_detections.append(detection)

            if preview_dir is not None and preview_remaining > 0:
                cls_name = str(detection.get("cls_name", "unknown"))
                preview_path = _preview_output_path(
                    preview_dir, image_path, index, cls_name
                )
                if preview_path.exists() and not args.force:
                    LOGGER.error("Preview exists (use --force): %s", preview_path)
                    return 2
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                if crop.size == 0:
                    LOGGER.error("Empty crop for %s", image_path)
                    return 2
                if not cv2.imwrite(str(preview_path), crop):
                    LOGGER.error("Failed to write preview crop: %s", preview_path)
                    return 2
                preview_remaining -= 1
                previews_saved += 1

        if debug_dir is not None:
            cars, plates = _split_by_role(valid_detections)
            matches = match_plates_to_cars(cars, plates)
            annotated = draw_debug(image, cars, plates, matches)
            output_image = _debug_output_path(debug_dir, image_path)
            output_image.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_image), annotated):
                LOGGER.error("Failed to write debug image: %s", output_image)
                return 2
            LOGGER.info("Wrote debug image: %s", output_image)
            debug_saved += 1
    payload = {
        "weights": str(weights_path),
        "input": str(input_path),
        "conf": args.conf,
        "iou": args.iou,
        "pad": args.pad,
        "images": detections,
    }
    _write_json(payload, json_path)
    LOGGER.info("Wrote detections: %s", json_path)
    LOGGER.info(
        "Summary: images=%d detections=%d invalid_bboxes=%d empty_crops=%d previews_saved=%d debug_images=%d",
        len(image_paths),
        total_detections,
        invalid_bboxes,
        empty_crops,
        previews_saved,
        debug_saved,
    )
    return 0


def _run_clean(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    deletions = plan_deletions(repo_root)

    if not deletions:
        LOGGER.info("Nothing to delete.")
        return 0

    LOGGER.info("Deletion plan (%d paths):", len(deletions))
    for path in deletions:
        LOGGER.info(" - %s", path)

    if not args.force:
        LOGGER.info("Dry run only. Use --force to delete.")
        return 0

    return execute_deletions(deletions, repo_root)


def _write_index(index_path: Path, rows: list[dict[str, object]], force: bool) -> int:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "plate_id",
        "plate_text_raw",
        "ocr_conf",
        "car_crop_path",
        "plate_crop_path",
        "car_bbox_xyxy",
        "plate_bbox_xyxy",
        "car_det_conf",
        "plate_det_conf",
        "timestamp",
    ]
    mode = "w" if force or not index_path.exists() else "a"
    write_header = mode == "w" or index_path.stat().st_size == 0
    with index_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote index: %s", index_path)
    return 0


def _run_pipeline(args: argparse.Namespace) -> int:
    weights_path = Path(args.weights)
    input_path = Path(args.input)
    gallery_dir = Path(args.gallery)
    plates_dir = Path(args.plates)
    index_path = Path(args.index)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    repo_root = _repo_root()
    max_debug = bool(args.max_debug)
    detections_json_path: Path | None = None
    preview_dir: Path | None = None
    preview_limit = args.preview_limit
    run_config_path: Path | None = None

    if max_debug:
        logging.getLogger().setLevel(logging.DEBUG)
        debug_dir = repo_root / "data" / "meta" / "debug"
        index_path = repo_root / "data" / "meta" / "index.csv"
        detections_json_path = repo_root / "data" / "meta" / "detections.json"
        preview_dir = repo_root / "data" / "meta" / "crops_preview"
        run_config_path = repo_root / "data" / "meta" / "run_config.json"

    if not weights_path.is_file():
        LOGGER.error("Weights not found: %s", weights_path)
        return 2

    image_paths = _collect_images(input_path)
    if not image_paths:
        LOGGER.error("No images found in: %s", input_path)
        return 2

    if debug_dir is not None and not args.force:
        existing = [
            _run_debug_output_path(debug_dir, path)
            for path in image_paths
            if _run_debug_output_path(debug_dir, path).exists()
        ]
        if existing:
            LOGGER.error("Debug outputs exist (use --force): %s", existing[0])
            return 2
    if detections_json_path is not None and detections_json_path.exists() and not args.force:
        LOGGER.error("Detections JSON exists (use --force): %s", detections_json_path)
        return 2
    if run_config_path is not None and run_config_path.exists() and not args.force:
        LOGGER.error("Run config exists (use --force): %s", run_config_path)
        return 2
    if preview_dir is not None and preview_dir.exists() and not args.force:
        existing_preview = next(preview_dir.iterdir(), None)
        if existing_preview is not None:
            LOGGER.error("Preview outputs exist (use --force): %s", preview_dir)
            return 2

    LOGGER.info("Detecting on %d image(s)", len(image_paths))
    detections = run_detection(
        weights_path=weights_path,
        image_paths=image_paths,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )

    rows: list[dict[str, object]] = []
    total_detections = 0
    total_matches = 0
    invalid_bboxes = 0
    empty_crops = 0
    unknown_ids = 0
    debug_saved = 0
    previews_saved = 0
    preview_remaining = preview_limit if preview_dir is not None else 0
    for image_result in detections:
        image_path = Path(str(image_result["image_path"]))
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.error("Failed to read image: %s", image_path)
            return 2

        per_image = _sorted_detections(list(image_result.get("detections", [])))
        total_detections += len(per_image)
        valid_detections: list[dict[str, object]] = []
        for det_index, detection in enumerate(per_image):
            bbox = detection.get("bbox_xyxy")
            if not bbox:
                invalid_bboxes += 1
                continue
            ok, reason = validate_bbox(bbox, image.shape)
            if not ok:
                invalid_bboxes += 1
                LOGGER.warning("Skipping bbox for %s (%s)", image_path, reason)
                continue
            if reason == "clamped":
                LOGGER.warning("Clamping bbox for %s", image_path)
            clamped = clamp_bbox(bbox, image.shape, pad=args.pad)
            detection["bbox_xyxy_crop"] = list(clamped)
            valid_detections.append(detection)

            if preview_dir is not None and preview_remaining > 0:
                crop = crop_image(image, bbox, pad=args.pad)
                if crop.size == 0:
                    empty_crops += 1
                    LOGGER.warning("Empty preview crop for %s", image_path)
                    continue
                cls_name = str(detection.get("cls_name", "unknown"))
                preview_path = _preview_output_path(
                    preview_dir, image_path, det_index, cls_name
                )
                if preview_path.exists() and not args.force:
                    LOGGER.error("Preview exists (use --force): %s", preview_path)
                    return 2
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(preview_path), crop):
                    LOGGER.error("Failed to write preview crop: %s", preview_path)
                    return 2
                preview_remaining -= 1
                previews_saved += 1

        cars, plates = _split_by_role(valid_detections)
        cars = _sorted_detections(cars)
        plates = _sorted_detections(plates)
        matches = match_plates_to_cars(cars, plates)
        total_matches += len(matches)
        timestamp = _image_timestamp(image_path)
        matches_with_ids: list[dict[str, object]] = []
        rows_added = 0

        for match_index, match in enumerate(matches):
            car = match.get("car")
            plate = match.get("plate")
            if not isinstance(car, dict) or not isinstance(plate, dict):
                continue

            car_bbox = car.get("bbox_xyxy")
            plate_bbox = plate.get("bbox_xyxy")
            if not car_bbox or not plate_bbox:
                continue
            car_ok, car_reason = validate_bbox(car_bbox, image.shape)
            if not car_ok:
                invalid_bboxes += 1
                LOGGER.warning("Skipping car bbox for %s (%s)", image_path, car_reason)
                continue
            if car_reason == "clamped":
                LOGGER.warning("Clamping car bbox for %s", image_path)
            plate_ok, plate_reason = validate_bbox(plate_bbox, image.shape)
            if not plate_ok:
                invalid_bboxes += 1
                LOGGER.warning("Skipping plate bbox for %s (%s)", image_path, plate_reason)
                continue
            if plate_reason == "clamped":
                LOGGER.warning("Clamping plate bbox for %s", image_path)

            car_crop = crop_image(image, car_bbox, pad=args.pad)
            plate_crop = crop_image(image, plate_bbox, pad=args.pad)
            if car_crop.size == 0 or plate_crop.size == 0:
                empty_crops += 1
                LOGGER.warning("Empty crop for %s", image_path)
                continue

            raw_text, ocr_conf = read_plate_text(plate_crop)
            if raw_text.strip() and ocr_conf >= args.ocr_min_conf:
                plate_id = normalize_plate_id(raw_text)
            else:
                plate_id = _unknown_plate_id(f"{image_path}|{match_index}")
                unknown_ids += 1

            plate_dir = _plate_dir(plates_dir, plate_id)
            car_dir = _plate_dir(gallery_dir, plate_id)
            car_path = car_dir / f"{image_path.stem}_{match_index:03d}_car.jpg"
            plate_path = plate_dir / f"{image_path.stem}_{match_index:03d}_plate.jpg"
            if (car_path.exists() or plate_path.exists()) and not args.force:
                LOGGER.error("Crop exists (use --force): %s", car_path)
                return 2

            car_dir.mkdir(parents=True, exist_ok=True)
            plate_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(car_path), car_crop):
                LOGGER.error("Failed to write car crop: %s", car_path)
                return 2
            if not cv2.imwrite(str(plate_path), plate_crop):
                LOGGER.error("Failed to write plate crop: %s", plate_path)
                return 2

            rows.append(
                {
                    "image_path": str(image_path),
                    "plate_id": plate_id,
                    "plate_text_raw": raw_text,
                    "ocr_conf": f"{ocr_conf:.4f}",
                    "car_crop_path": str(car_path),
                    "plate_crop_path": str(plate_path),
                    "car_bbox_xyxy": json.dumps([float(value) for value in car_bbox]),
                    "plate_bbox_xyxy": json.dumps([float(value) for value in plate_bbox]),
                    "car_det_conf": f"{float(car.get('conf', 0.0)):.4f}",
                    "plate_det_conf": f"{float(plate.get('conf', 0.0)):.4f}",
                    "timestamp": timestamp,
                }
            )
            rows_added += 1
            matches_with_ids.append(
                {
                    "car": car,
                    "plate": plate,
                    "iou": match.get("iou", 0.0),
                    "plate_id": plate_id,
                }
            )

        if debug_dir is not None:
            annotated = draw_plate_id_debug(image, cars, plates, matches_with_ids)
            debug_path = _run_debug_output_path(debug_dir, image_path)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            if debug_path.exists() and not args.force:
                LOGGER.error("Debug output exists (use --force): %s", debug_path)
                return 2
            if not cv2.imwrite(str(debug_path), annotated):
                LOGGER.error("Failed to write debug image: %s", debug_path)
                return 2
            LOGGER.info("Wrote debug image: %s", debug_path)
            debug_saved += 1

        if max_debug:
            LOGGER.info(
                "Image %s: detections=%d cars=%d plates=%d matches=%d rows=%d",
                image_path.name,
                len(per_image),
                len(cars),
                len(plates),
                len(matches),
                rows_added,
            )

    if detections_json_path is not None:
        payload = {
            "weights": str(weights_path),
            "input": str(input_path),
            "conf": args.conf,
            "iou": args.iou,
            "pad": args.pad,
            "images": detections,
        }
        _write_json(payload, detections_json_path)

    if run_config_path is not None:
        result = _write_run_config(run_config_path, args)
        if result != 0:
            return result

    result = _write_index(index_path, rows, args.force)
    LOGGER.info(
        "Summary: images=%d detections=%d matches=%d rows=%d unknown_ids=%d invalid_bboxes=%d empty_crops=%d debug_images=%d previews=%d",
        len(image_paths),
        total_detections,
        total_matches,
        len(rows),
        unknown_ids,
        invalid_bboxes,
        empty_crops,
        debug_saved,
        previews_saved,
    )
    return result


def main() -> int:
    args = parse_args()
    if args.command == "detect":
        return _run_detect(args)
    if args.command == "clean":
        return _run_clean(args)
    if args.command == "run":
        return _run_pipeline(args)
    LOGGER.error("Unknown command: %s", args.command)
    return 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
