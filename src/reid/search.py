"""Vehicle ReID search utilities."""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from src.pipeline.crop import clamp_bbox, validate_bbox
from src.pipeline.detect import run_detection
from src.reid.visualize import draw_reid_debug

LOGGER = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class GalleryEntry:
    path: Path
    feature: torch.Tensor


@dataclass
class Candidate:
    image_path: Path
    bbox_xyxy: tuple[float, float, float, float]
    det_conf: float
    crop: np.ndarray


@dataclass
class Match:
    candidate_id: int
    image_path: Path
    bbox_xyxy: tuple[float, float, float, float]
    det_conf: float
    score: float
    best_gallery_path: Path
    kept: bool = False


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


def _sanitize_label(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label)
    return safe.strip("_") or "unknown"


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


def _resolve_device(requested: str | None) -> torch.device:
    if not requested:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.isdigit():
        return torch.device(f"cuda:{requested}")
    return torch.device(requested)


def _patch_torch_hub_for_cpu() -> None:
    if torch.cuda.is_available():
        return
    original = torch.hub.load_state_dict_from_url
    if getattr(original, "_patched_for_cpu", False):
        return

    def _wrapped(url, *args, **kwargs):
        if kwargs.get("map_location") is None:
            kwargs["map_location"] = torch.device("cpu")
        return original(url, *args, **kwargs)

    _wrapped._patched_for_cpu = True
    torch.hub.load_state_dict_from_url = _wrapped


def _load_reid_model(
    opts_path: Path,
    ckpt_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    reid_dir = Path(__file__).resolve().parents[2] / "third_party" / "vehicle_reid"
    if not reid_dir.is_dir():
        raise FileNotFoundError(f"ReID vendor dir missing: {reid_dir}")

    sys.path.append(str(reid_dir))
    try:
        from load_model import load_model_from_opts
    finally:
        sys.path.remove(str(reid_dir))

    _patch_torch_hub_for_cpu()
    model = load_model_from_opts(str(opts_path), ckpt=str(ckpt_path), remove_classifier=True)
    model.eval()
    model.to(device)
    return model


def _prepare_crop_tensor(
    crop: np.ndarray,
    input_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    elif crop.shape[2] == 1:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    elif crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)
    return (tensor - mean) / std


def _extract_features(
    model: torch.nn.Module,
    crops: list[np.ndarray],
    input_size: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if not crops:
        return torch.empty((0, 0), dtype=torch.float32)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensors = [_prepare_crop_tensor(crop, input_size, mean, std) for crop in crops]

    features: list[torch.Tensor] = []
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[start : start + batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[-1]
            if not isinstance(outputs, torch.Tensor):
                raise ValueError("Model output is not a tensor.")
            flipped = torch.flip(batch, dims=[3])
            flipped_out = model(flipped)
            if isinstance(flipped_out, (list, tuple)):
                flipped_out = flipped_out[-1]
            if not isinstance(flipped_out, torch.Tensor):
                raise ValueError("Flipped output is not a tensor.")
            outputs = outputs + flipped_out
            norm = torch.norm(outputs, p=2, dim=1, keepdim=True).clamp_min(1e-12)
            outputs = outputs.div(norm)
            features.append(outputs.cpu())

    return torch.cat(features, dim=0)


def _load_gallery_embeddings(
    gallery_dir: Path,
    plate_id: str,
    model: torch.nn.Module,
    input_size: int,
    device: torch.device,
    batch_size: int,
) -> list[GalleryEntry]:
    plate_dir = gallery_dir / _sanitize_label(plate_id)
    if not plate_dir.is_dir():
        return []

    image_paths = sorted(
        [path for path in plate_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS],
        key=lambda path: path.name,
    )
    crops: list[np.ndarray] = []
    valid_paths: list[Path] = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            LOGGER.warning("Failed to read gallery image: %s", path)
            continue
        crops.append(image)
        valid_paths.append(path)

    if not crops:
        return []

    features = _extract_features(model, crops, input_size, device, batch_size)
    return [GalleryEntry(path=path, feature=features[idx]) for idx, path in enumerate(valid_paths)]


def _collect_candidates(
    weights_path: Path,
    image_paths: list[Path],
    conf: float,
    iou: float,
    device: str | None,
    pad: float,
    car_label: str,
) -> list[Candidate]:
    detections = run_detection(
        weights_path=weights_path,
        image_paths=image_paths,
        conf=conf,
        iou=iou,
        device=device,
    )

    candidates: list[Candidate] = []
    car_label_lower = car_label.lower()

    for image_result in detections:
        image_path = Path(str(image_result["image_path"]))
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        per_image = sorted(
            list(image_result.get("detections", [])),
            key=_det_sort_key,
        )
        for detection in per_image:
            cls_name = str(detection.get("cls_name", "")).lower()
            if car_label_lower not in cls_name:
                continue
            bbox = detection.get("bbox_xyxy")
            if not bbox:
                continue
            ok, _reason = validate_bbox(bbox, image.shape)
            if not ok:
                continue
            clamped = clamp_bbox(bbox, image.shape, pad=pad)
            x_min, y_min, x_max, y_max = clamped
            crop = image[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue
            candidates.append(
                Candidate(
                    image_path=image_path,
                    bbox_xyxy=tuple(float(value) for value in clamped),
                    det_conf=float(detection.get("conf", 0.0)),
                    crop=crop,
                )
            )

    return candidates


def _score_candidates(
    gallery_entries: list[GalleryEntry],
    candidates: list[Candidate],
    model: torch.nn.Module,
    input_size: int,
    device: torch.device,
    batch_size: int,
) -> list[Match]:
    if not candidates:
        return []

    gallery_features = torch.stack([entry.feature for entry in gallery_entries])
    candidate_features = _extract_features(
        model,
        [candidate.crop for candidate in candidates],
        input_size,
        device,
        batch_size,
    )
    scores = torch.matmul(gallery_features, candidate_features.T)
    best_scores, best_indices = torch.max(scores, dim=0)

    matches: list[Match] = []
    for idx, candidate in enumerate(candidates):
        best_index = int(best_indices[idx].item())
        matches.append(
            Match(
                candidate_id=idx,
                image_path=candidate.image_path,
                bbox_xyxy=candidate.bbox_xyxy,
                det_conf=candidate.det_conf,
                score=float(best_scores[idx].item()),
                best_gallery_path=gallery_entries[best_index].path,
            )
        )

    return matches


def _filter_matches(
    matches: list[Match],
    min_score: float,
    top_k: int,
) -> list[Match]:
    per_image: dict[Path, list[Match]] = {}
    for match in matches:
        per_image.setdefault(match.image_path, []).append(match)

    for image_path, items in per_image.items():
        sorted_items = sorted(items, key=lambda m: (-m.score, m.bbox_xyxy))
        kept_ids: set[int] = set()
        for item in sorted_items:
            if item.score < min_score:
                continue
            if top_k > 0 and len(kept_ids) >= top_k:
                break
            kept_ids.add(item.candidate_id)
        for item in items:
            item.kept = item.candidate_id in kept_ids

    return matches


def _write_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def _write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_image_path",
        "query_bbox_xyxy",
        "query_det_conf",
        "plate_id",
        "score",
        "best_gallery_path",
        "kept",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_reid_search(
    *,
    plate_id: str,
    weights_path: Path,
    input_path: Path,
    gallery_dir: Path,
    reid_opts: Path,
    reid_ckpt: Path,
    conf: float,
    iou: float,
    pad: float,
    input_size: int,
    batch_size: int,
    device: str | None,
    reid_device: str | None,
    debug_dir: Path,
    min_score: float,
    top_k: int,
    force: bool,
    car_label: str,
) -> int:
    if not weights_path.is_file():
        LOGGER.error("Weights not found: %s", weights_path)
        return 2
    if not reid_opts.is_file():
        LOGGER.error("ReID opts not found: %s", reid_opts)
        return 2
    if not reid_ckpt.is_file():
        LOGGER.error("ReID checkpoint not found: %s", reid_ckpt)
        return 2

    image_paths = _collect_images(input_path)
    if not image_paths:
        LOGGER.error("No images found in: %s", input_path)
        return 2

    results_csv = debug_dir / "results.csv"
    results_json = debug_dir / "results.json"
    annotated_dir = debug_dir / "annotated"

    if not force:
        if results_csv.exists() or results_json.exists():
            LOGGER.error("Results exist (use --force): %s", debug_dir)
            return 2
        if annotated_dir.exists():
            existing = next(annotated_dir.iterdir(), None)
            if existing is not None:
                LOGGER.error("Annotated outputs exist (use --force): %s", annotated_dir)
                return 2

    LOGGER.info("Loading ReID model")
    device_t = _resolve_device(reid_device)
    try:
        model = _load_reid_model(reid_opts, reid_ckpt, device_t)
    except Exception as exc:
        LOGGER.error("Failed to load ReID model: %s", exc)
        return 2

    gallery_entries = _load_gallery_embeddings(
        gallery_dir,
        plate_id,
        model,
        input_size,
        device_t,
        batch_size,
    )
    if not gallery_entries:
        LOGGER.error("No gallery images found for plate_id: %s", plate_id)
        return 2

    LOGGER.info("Gallery images loaded: %d", len(gallery_entries))
    LOGGER.info("Detecting cars in %d image(s)", len(image_paths))
    try:
        candidates = _collect_candidates(
            weights_path,
            image_paths,
            conf,
            iou,
            device,
            pad,
            car_label,
        )
    except ValueError as exc:
        LOGGER.error("Detection failed: %s", exc)
        return 2

    LOGGER.info("Car candidates: %d", len(candidates))
    matches = _score_candidates(
        gallery_entries,
        candidates,
        model,
        input_size,
        device_t,
        batch_size,
    )
    matches = _filter_matches(matches, min_score=min_score, top_k=top_k)

    rows: list[dict[str, object]] = []
    for match in matches:
        rows.append(
            {
                "query_image_path": str(match.image_path),
                "query_bbox_xyxy": json.dumps([float(value) for value in match.bbox_xyxy]),
                "query_det_conf": f"{match.det_conf:.4f}",
                "plate_id": plate_id,
                "score": f"{match.score:.6f}",
                "best_gallery_path": str(match.best_gallery_path),
                "kept": str(match.kept),
            }
        )

    _write_csv(rows, results_csv)

    payload = {
        "plate_id": plate_id,
        "weights": str(weights_path),
        "input": str(input_path),
        "gallery": str(gallery_dir),
        "reid_opts": str(reid_opts),
        "reid_ckpt": str(reid_ckpt),
        "conf": conf,
        "iou": iou,
        "pad": pad,
        "input_size": input_size,
        "batch_size": batch_size,
        "min_score": min_score,
        "top_k": top_k,
        "matches": [
            {
                "query_image_path": str(match.image_path),
                "query_bbox_xyxy": [float(value) for value in match.bbox_xyxy],
                "query_det_conf": float(match.det_conf),
                "score": float(match.score),
                "best_gallery_path": str(match.best_gallery_path),
                "kept": bool(match.kept),
            }
            for match in matches
        ],
    }
    _write_json(payload, results_json)

    annotated_dir.mkdir(parents=True, exist_ok=True)
    per_image: dict[Path, list[Match]] = {}
    for match in matches:
        per_image.setdefault(match.image_path, []).append(match)

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.warning("Failed to read image for debug: %s", image_path)
            continue
        annotated = draw_reid_debug(
            image,
            [
                {
                    "bbox_xyxy": match.bbox_xyxy,
                    "score": match.score,
                    "kept": match.kept,
                }
                for match in per_image.get(image_path, [])
            ],
            plate_id,
        )
        output_path = annotated_dir / f"{image_path.stem}_reid.jpg"
        if output_path.exists() and not force:
            LOGGER.error("Debug image exists (use --force): %s", output_path)
            return 2
        if not cv2.imwrite(str(output_path), annotated):
            LOGGER.error("Failed to write debug image: %s", output_path)
            return 2

    kept_count = sum(1 for match in matches if match.kept)
    LOGGER.info(
        "ReID summary: images=%d gallery=%d candidates=%d kept=%d",
        len(image_paths),
        len(gallery_entries),
        len(candidates),
        kept_count,
    )
    LOGGER.info("Wrote results: %s", results_csv)
    LOGGER.info("Wrote debug images: %s", annotated_dir)
    return 0



def _gather_gallery_entries(gallery_dir: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    if not gallery_dir.is_dir():
        return entries

    for plate_dir in sorted(gallery_dir.iterdir(), key=lambda p: p.name):
        if not plate_dir.is_dir():
            continue
        plate_id = plate_dir.name
        for image_path in sorted(plate_dir.iterdir(), key=lambda p: p.name):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            entries.append((plate_id, image_path))

    return entries


def _write_index_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind",
        "plate_id",
        "image_path",
        "feature_index",
        "centroid_index",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_reid_index(
    *,
    gallery_dir: Path,
    reid_opts: Path,
    reid_ckpt: Path,
    output_dir: Path,
    input_size: int,
    batch_size: int,
    reid_device: str | None,
    force: bool,
) -> int:
    if not gallery_dir.is_dir():
        LOGGER.error("Gallery directory not found: %s", gallery_dir)
        return 2
    if not reid_opts.is_file():
        LOGGER.error("ReID opts not found: %s", reid_opts)
        return 2
    if not reid_ckpt.is_file():
        LOGGER.error("ReID checkpoint not found: %s", reid_ckpt)
        return 2

    index_npz = output_dir / "index.npz"
    index_csv = output_dir / "index.csv"
    if not force and (index_npz.exists() or index_csv.exists()):
        LOGGER.error("Index exists (use --force): %s", output_dir)
        return 2

    entries = _gather_gallery_entries(gallery_dir)
    if not entries:
        LOGGER.error("No gallery images found in: %s", gallery_dir)
        return 2

    crops: list[np.ndarray] = []
    plate_ids: list[str] = []
    paths: list[Path] = []
    for plate_id, image_path in entries:
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.warning("Failed to read gallery image: %s", image_path)
            continue
        crops.append(image)
        plate_ids.append(plate_id)
        paths.append(image_path)

    if not crops:
        LOGGER.error("No readable gallery images in: %s", gallery_dir)
        return 2

    device = _resolve_device(reid_device)
    LOGGER.info("Loading ReID model on %s", device)
    model = _load_reid_model(reid_opts, reid_ckpt, device)

    features = _extract_features(model, crops, input_size, device, batch_size)
    if features.shape[0] != len(paths):
        LOGGER.error("Feature count mismatch (features=%d paths=%d)", features.shape[0], len(paths))
        return 2

    features_np = features.detach().cpu().numpy()
    unique_ids = sorted(set(plate_ids))
    centroid_features: list[np.ndarray] = []
    centroid_ids: list[str] = []

    for plate_id in unique_ids:
        indices = [idx for idx, value in enumerate(plate_ids) if value == plate_id]
        if not indices:
            continue
        centroid = features_np[indices].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm
        centroid_features.append(centroid.astype(np.float32))
        centroid_ids.append(plate_id)

    centroid_array = (
        np.stack(centroid_features, axis=0)
        if centroid_features
        else np.empty((0, features_np.shape[1]), dtype=np.float32)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        index_npz,
        features=features_np,
        paths=np.array([str(path) for path in paths]),
        plate_ids=np.array(plate_ids),
        centroid_features=centroid_array,
        centroid_ids=np.array(centroid_ids),
        input_size=np.array([input_size]),
        reid_ckpt=np.array([str(reid_ckpt)]),
        reid_opts=np.array([str(reid_opts)]),
    )

    rows: list[dict[str, object]] = []
    for idx, (plate_id, image_path) in enumerate(zip(plate_ids, paths)):
        rows.append(
            {
                "kind": "gallery",
                "plate_id": plate_id,
                "image_path": str(image_path),
                "feature_index": idx,
                "centroid_index": "",
            }
        )
    for idx, plate_id in enumerate(centroid_ids):
        rows.append(
            {
                "kind": "centroid",
                "plate_id": plate_id,
                "image_path": "",
                "feature_index": "",
                "centroid_index": idx,
            }
        )

    _write_index_csv(rows, index_csv)
    LOGGER.info("Wrote ReID index: %s", index_npz)
    LOGGER.info("Wrote ReID index CSV: %s", index_csv)
    return 0


def _load_reid_index(index_path: Path) -> dict[str, object]:
    data = np.load(index_path, allow_pickle=True)
    features = data["features"].astype(np.float32)
    paths = [str(value) for value in data["paths"]]
    plate_ids = [str(value) for value in data["plate_ids"]]
    centroid_features = data["centroid_features"].astype(np.float32)
    centroid_ids = [str(value) for value in data["centroid_ids"]]
    input_size = int(data["input_size"][0]) if "input_size" in data else None
    reid_opts = None
    reid_ckpt = None
    if "reid_opts" in data:
        reid_opts = str(data["reid_opts"][0])
    if "reid_ckpt" in data:
        reid_ckpt = str(data["reid_ckpt"][0])
    return {
        "features": features,
        "paths": paths,
        "plate_ids": plate_ids,
        "centroid_features": centroid_features,
        "centroid_ids": centroid_ids,
        "input_size": input_size,
        "reid_opts": reid_opts,
        "reid_ckpt": reid_ckpt,
    }


def run_reid_search_cached(
    *,
    plate_id: str,
    weights_path: Path,
    input_dir: Path,
    index_dir: Path,
    output_dir: Path,
    conf: float,
    iou: float,
    pad: float,
    input_size: int,
    batch_size: int,
    device: str | None,
    reid_device: str | None,
    min_score: float,
    top_k: int,
    force: bool,
    car_label: str,
) -> int:
    if not weights_path.is_file():
        LOGGER.error("Weights not found: %s", weights_path)
        return 2

    index_npz = index_dir / "index.npz"
    if not index_npz.is_file():
        LOGGER.error("ReID index not found: %s", index_npz)
        return 2

    image_paths = _collect_images(input_dir)
    if not image_paths:
        LOGGER.error("No images found in: %s", input_dir)
        return 2

    results_csv = output_dir / "results.csv"
    results_json = output_dir / "results.json"
    annotated_dir = output_dir / "annotated"

    if not force:
        if results_csv.exists() or results_json.exists():
            LOGGER.error("Results exist (use --force): %s", output_dir)
            return 2
        if annotated_dir.exists():
            existing = next(annotated_dir.iterdir(), None)
            if existing is not None:
                LOGGER.error("Annotated outputs exist (use --force): %s", annotated_dir)
                return 2

    index = _load_reid_index(index_npz)
    centroid_ids = index["centroid_ids"]
    if plate_id not in centroid_ids:
        LOGGER.error("Plate ID not found in index: %s", plate_id)
        return 2

    centroid_index = centroid_ids.index(plate_id)
    centroid_features = index["centroid_features"]
    centroid = torch.from_numpy(centroid_features[centroid_index]).float()

    index_input_size = index.get("input_size")
    if index_input_size and index_input_size != input_size:
        LOGGER.warning(
            "Index input_size=%s differs from requested input_size=%s",
            index_input_size,
            input_size,
        )

    LOGGER.info("Loading ReID model")
    device_t = _resolve_device(reid_device)
    reid_opts_value = index.get("reid_opts")
    reid_ckpt_value = index.get("reid_ckpt")
    if not reid_opts_value or not reid_ckpt_value:
        LOGGER.error("ReID opts/checkpoint not found in index. Rebuild the index.")
        return 2
    repo_root = Path(__file__).resolve().parents[2]
    reid_opts_path = Path(reid_opts_value)
    reid_ckpt_path = Path(reid_ckpt_value)
    if not reid_opts_path.is_absolute():
        reid_opts_path = repo_root / reid_opts_path
    if not reid_ckpt_path.is_absolute():
        reid_ckpt_path = repo_root / reid_ckpt_path
    if not reid_opts_path.is_file():
        LOGGER.error("ReID opts not found: %s", reid_opts_path)
        return 2
    if not reid_ckpt_path.is_file():
        LOGGER.error("ReID checkpoint not found: %s", reid_ckpt_path)
        return 2
    model = _load_reid_model(reid_opts_path, reid_ckpt_path, device_t)

    LOGGER.info("Detecting cars in %d image(s)", len(image_paths))
    try:
        candidates = _collect_candidates(
            weights_path,
            image_paths,
            conf,
            iou,
            device,
            pad,
            car_label,
        )
    except ValueError as exc:
        LOGGER.error("Detection failed: %s", exc)
        return 2

    if not candidates:
        LOGGER.warning("No car detections found in inputs.")

    candidate_features = _extract_features(
        model,
        [candidate.crop for candidate in candidates],
        input_size,
        device_t,
        batch_size,
    )

    if candidates:
        scores = candidate_features.matmul(centroid)
    else:
        scores = torch.empty((0,), dtype=torch.float32)

    gallery_plate_ids = index["plate_ids"]
    gallery_paths = index["paths"]
    gallery_features = index["features"]
    mask = [value == plate_id for value in gallery_plate_ids]
    best_gallery_paths: list[str] = [""] * len(candidates)
    if any(mask):
        gallery_features_t = torch.from_numpy(gallery_features[mask]).float()
        if candidates:
            gallery_scores = gallery_features_t.matmul(candidate_features.T)
            best_scores, best_indices = torch.max(gallery_scores, dim=0)
            filtered_paths = [path for path, keep in zip(gallery_paths, mask) if keep]
            for idx in range(len(candidates)):
                best_gallery_paths[idx] = filtered_paths[int(best_indices[idx].item())]
    else:
        LOGGER.warning("No gallery embeddings found for plate_id: %s", plate_id)

    matches: list[Match] = []
    for idx, candidate in enumerate(candidates):
        score = float(scores[idx].item()) if idx < len(scores) else 0.0
        best_path = Path(best_gallery_paths[idx]) if best_gallery_paths[idx] else Path("")
        matches.append(
            Match(
                candidate_id=idx,
                image_path=candidate.image_path,
                bbox_xyxy=candidate.bbox_xyxy,
                det_conf=candidate.det_conf,
                score=score,
                best_gallery_path=best_path,
            )
        )

    matches = _filter_matches(matches, min_score=min_score, top_k=top_k)

    rows: list[dict[str, object]] = []
    for match in matches:
        rows.append(
            {
                "query_image_path": str(match.image_path),
                "query_bbox_xyxy": json.dumps([float(value) for value in match.bbox_xyxy]),
                "query_det_conf": f"{match.det_conf:.4f}",
                "plate_id": plate_id,
                "score": f"{match.score:.6f}",
                "best_gallery_path": str(match.best_gallery_path) if match.best_gallery_path else "",
                "kept": str(match.kept),
            }
        )

    _write_csv(rows, results_csv)

    payload = {
        "plate_id": plate_id,
        "weights": str(weights_path),
        "input": str(input_dir),
        "index": str(index_npz),
        "conf": conf,
        "iou": iou,
        "pad": pad,
        "input_size": input_size,
        "batch_size": batch_size,
        "min_score": min_score,
        "top_k": top_k,
        "matches": [
            {
                "query_image_path": str(match.image_path),
                "query_bbox_xyxy": [float(value) for value in match.bbox_xyxy],
                "query_det_conf": float(match.det_conf),
                "score": float(match.score),
                "best_gallery_path": str(match.best_gallery_path) if match.best_gallery_path else "",
                "kept": bool(match.kept),
            }
            for match in matches
        ],
    }
    _write_json(payload, results_json)

    annotated_dir.mkdir(parents=True, exist_ok=True)
    per_image: dict[Path, list[Match]] = {}
    for match in matches:
        per_image.setdefault(match.image_path, []).append(match)

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.warning("Failed to read image for debug: %s", image_path)
            continue
        annotated = draw_reid_debug(
            image,
            [
                {
                    "bbox_xyxy": match.bbox_xyxy,
                    "score": match.score,
                    "kept": match.kept,
                }
                for match in per_image.get(image_path, [])
            ],
            plate_id,
        )
        output_path = annotated_dir / f"{image_path.stem}_reid.jpg"
        if output_path.exists() and not force:
            LOGGER.error("Debug image exists (use --force): %s", output_path)
            return 2
        if not cv2.imwrite(str(output_path), annotated):
            LOGGER.error("Failed to write debug image: %s", output_path)
            return 2

    kept_count = sum(1 for match in matches if match.kept)
    LOGGER.info(
        "ReID summary: images=%d candidates=%d kept=%d",
        len(image_paths),
        len(candidates),
        kept_count,
    )
    LOGGER.info("Wrote results: %s", results_csv)
    LOGGER.info("Wrote debug images: %s", annotated_dir)
    return 0
