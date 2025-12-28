"""Load a YOLO model and optionally run a small prediction on one image."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a YOLO weights file and optionally run a smoke-test prediction."
        )
    )
    parser.add_argument(
        "--weights",
        default="models/best.pt",
        help="Path to YOLO weights (repo-relative by default).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to a sample image (repo-relative by default).",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Directory to scan for the first jpg/png/jpeg when --image is omitted.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Save an annotated preview image with detections.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the annotated image (repo-relative by default).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    return parser.parse_args()


def _format_class_names(names: object) -> str:
    if isinstance(names, dict):
        items = [f"{idx}: {name}" for idx, name in sorted(names.items())]
        return ", ".join(items) if items else "none"
    if isinstance(names, (list, tuple)):
        return ", ".join(str(name) for name in names) if names else "none"
    return "none"


def _class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)):
        return str(names[class_id]) if 0 <= class_id < len(names) else str(class_id)
    return str(class_id)


def _pick_first_image(input_dir: Path) -> Path | None:
    if not input_dir.is_dir():
        return None
    candidates = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name.lower())[0]


def _resolve_image_path(args: argparse.Namespace) -> Path | None:
    if args.image:
        return Path(args.image)
    if args.input_dir:
        return _pick_first_image(Path(args.input_dir))
    return None


def _default_output_path(image_path: Path) -> Path:
    return Path("outputs") / "vis" / f"{image_path.stem}_pred.jpg"


def _resolve_output_path(args: argparse.Namespace, image_path: Path) -> Path:
    return Path(args.output) if args.output else _default_output_path(image_path)


def main() -> int:
    args = parse_args()
    weights_path = Path(args.weights)
    image_path = _resolve_image_path(args)

    if not weights_path.is_file():
        LOGGER.error("Weights not found: %s", weights_path)
        return 2

    LOGGER.info("Loading model: %s", weights_path)
    model = YOLO(str(weights_path))
    print(f"class_names: {_format_class_names(getattr(model, 'names', {}))}")
    print(f"model_task: {getattr(model, 'task', 'unknown')}")

    if image_path is None:
        if args.show:
            LOGGER.error("--show requires --image or --input_dir.")
            return 2
        return 0
    if not image_path.is_file():
        LOGGER.error("Image not found: %s", image_path)
        return 2

    LOGGER.info("Reading image: %s", image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        LOGGER.error("Failed to read image: %s", image_path)
        return 2

    LOGGER.info("Running prediction")
    results = model.predict(
        source=image,
        conf=args.conf,
        imgsz=args.imgsz,
        save=False,
        verbose=False,
    )
    counts: Counter[int] = Counter()
    if results and results[0].boxes is not None:
        class_ids = results[0].boxes.cls.tolist()
        counts.update(int(class_id) for class_id in class_ids)

    if not counts:
        print("detections_per_class: none")
        return 0

    names = getattr(model, "names", {})
    parts = [
        f"{_class_name(names, class_id)}={counts[class_id]}"
        for class_id in sorted(counts)
    ]
    print(f"detections_per_class: {', '.join(parts)}")

    if args.show:
        output_path = _resolve_output_path(args, image_path)
        if output_path.exists() and not args.force:
            LOGGER.error("Output exists (use --force): %s", output_path)
            return 2
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = results[0].plot()
        if not cv2.imwrite(str(output_path), annotated):
            LOGGER.error("Failed to write output image: %s", output_path)
            return 2
        LOGGER.info("Wrote annotated image: %s", output_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
