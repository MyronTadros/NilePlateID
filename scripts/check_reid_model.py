"""Validate ReID model files and run a single embedding."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import torch

LOGGER = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_opts_path(opts_arg: str | None, ckpt_path: Path, repo_root: Path) -> Path | None:
    if opts_arg:
        opt_path = Path(opts_arg)
        if not opt_path.is_absolute():
            opt_path = repo_root / opt_path
        return opt_path
    candidate = ckpt_path.parent / "opts.yaml"
    if candidate.is_file():
        return candidate
    return None


def _resolve_device(device: str | None) -> torch.device:
    if not device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


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


def _prepare_tensor(image, input_size: int) -> torch.Tensor:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def _embed_image(model: torch.nn.Module, image, input_size: int, device: torch.device) -> torch.Tensor:
    tensor = _prepare_tensor(image, input_size)
    batch = tensor.unsqueeze(0).to(device)
    output = model(batch)
    if isinstance(output, (list, tuple)):
        output = output[-1]
    if not isinstance(output, torch.Tensor):
        raise ValueError("Model output is not a tensor.")
    flipped = torch.flip(batch, dims=[3])
    flipped_out = model(flipped)
    if isinstance(flipped_out, (list, tuple)):
        flipped_out = flipped_out[-1]
    if not isinstance(flipped_out, torch.Tensor):
        raise ValueError("Flipped output is not a tensor.")
    output = output + flipped_out
    norm = torch.norm(output, p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return output.div(norm).cpu().squeeze(0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load a ReID model and compute one embedding."
    )
    parser.add_argument(
        "--reid_ckpt",
        default="models/reid/net.pth",
        help="Path to ReID checkpoint (repo-relative by default).",
    )
    parser.add_argument(
        "--reid_opts",
        default=None,
        help="Path to opts.yaml (defaults to sibling of checkpoint).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to an image to embed.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (e.g., cpu, cuda, 0).",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input image size expected by the ReID model.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    ckpt_path = (repo_root / args.reid_ckpt).resolve() if not Path(args.reid_ckpt).is_absolute() else Path(args.reid_ckpt)
    opts_path = _resolve_opts_path(args.reid_opts, ckpt_path, repo_root)

    if not ckpt_path.is_file():
        LOGGER.error("ReID checkpoint not found: %s", ckpt_path)
        return 2
    if not opts_path or not opts_path.is_file():
        LOGGER.error("ReID opts.yaml not found. Pass --reid_opts or place opts.yaml next to the checkpoint.")
        return 2

    reid_dir = repo_root / "third_party" / "vehicle_reid"
    if not reid_dir.is_dir():
        LOGGER.error("Vendor directory missing: %s", reid_dir)
        return 2

    sys.path.insert(0, str(reid_dir))
    try:
        from load_model import load_model_from_opts
    finally:
        if str(reid_dir) in sys.path:
            sys.path.remove(str(reid_dir))

    device = _resolve_device(args.device)
    _patch_torch_hub_for_cpu()
    LOGGER.info("Loading ReID model on %s", device)
    model = load_model_from_opts(str(opts_path), ckpt=str(ckpt_path), remove_classifier=True)
    model.eval()
    model.to(device)

    if not args.image:
        LOGGER.error("No image provided. Pass --image to compute an embedding.")
        return 2

    image_path = (repo_root / args.image).resolve() if not Path(args.image).is_absolute() else Path(args.image)
    if not image_path.is_file():
        LOGGER.error("Image not found: %s", image_path)
        return 2

    image = cv2.imread(str(image_path))
    if image is None:
        LOGGER.error("Failed to read image: %s", image_path)
        return 2

    with torch.no_grad():
        embedding = _embed_image(model, image, args.input_size, device)

    LOGGER.info("Embedding shape: %s", tuple(embedding.shape))
    LOGGER.info("Embedding L2 norm: %.6f", float(torch.norm(embedding)))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
