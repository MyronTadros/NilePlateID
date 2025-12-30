"""Download model weights for NilePlateID."""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

try:
    import gdown
except ImportError:
    gdown = None

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadItem:
    name: str
    url: str
    rel_path: Path


ITEMS = [
    DownloadItem(
        name="yolo-best",
        url="https://drive.google.com/file/d/11-tN3GAnxLvrSUNimJzKSldJzzXosE9d/view?usp=drive_link",
        rel_path=Path("models/best.pt"),
    ),
    DownloadItem(
        name="reid-ckpt",
        url="https://drive.google.com/file/d/1vJwbdxL8bq3XH7SW5k52YFEk38O7frdw/view?usp=sharing",
        rel_path=Path("models/reid/net.pth"),
    ),
    DownloadItem(
        name="reid-opts",
        url="https://drive.google.com/file/d/1m0-YfUm3OjgdhjbrOz_Wr5nOQT7zwB8-/view?usp=drive_link",
        rel_path=Path("models/reid/opts.yaml"),
    ),
    DownloadItem(
        name="yolo-ocr",
        url="https://drive.google.com/file/d/1OHouC1XBjIa0qv3hF4eT45QS28Xevvfy/view?usp=sharing",
        rel_path=Path("models/yolo11m_car_plate_ocr.pt"),
    ),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model weights.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files if they already exist.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with nonzero status if any download fails.",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _download_item(item: DownloadItem, repo_root: Path, force: bool) -> tuple[str, Path]:
    target = repo_root / item.rel_path
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        if not force:
            LOGGER.info("Skip %s (exists): %s", item.name, target)
            return "skipped", target
        target.unlink()

    LOGGER.info("Downloading %s -> %s", item.name, target)
    try:
        result = gdown.download(item.url, str(target), quiet=False, fuzzy=True)
        if not result or not target.exists():
            raise RuntimeError("download failed")
        return "downloaded", target
    except Exception as exc:
        LOGGER.error("Failed %s: %s", item.name, exc)
        return "failed", target


def _format_paths(paths: list[Path], repo_root: Path) -> str:
    if not paths:
        return "none"
    return ", ".join(str(path.relative_to(repo_root)) for path in paths)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if gdown is None:
        LOGGER.error("gdown is not installed. Install with: pip install gdown")
        return 1

    repo_root = _repo_root()
    downloaded: list[Path] = []
    skipped: list[Path] = []
    failed: list[Path] = []

    for item in ITEMS:
        status, path = _download_item(item, repo_root, args.force)
        if status == "downloaded":
            downloaded.append(path)
        elif status == "skipped":
            skipped.append(path)
        else:
            failed.append(path)

    LOGGER.info("Summary")
    LOGGER.info("Downloaded: %s", _format_paths(downloaded, repo_root))
    LOGGER.info("Skipped: %s", _format_paths(skipped, repo_root))
    LOGGER.info("Failed: %s", _format_paths(failed, repo_root))

    if failed and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
