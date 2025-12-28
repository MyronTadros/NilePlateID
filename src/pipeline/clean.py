"""Cleanup helpers for generated artifacts."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _is_within_root(path: Path, repo_root: Path) -> bool:
    try:
        path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    return True


def _add_children(paths: list[Path], directory: Path, repo_root: Path) -> None:
    if not directory.exists():
        return
    for child in directory.iterdir():
        if _is_within_root(child, repo_root):
            paths.append(child)
        else:
            LOGGER.warning("Skipping outside root: %s", child)


def plan_deletions(repo_root: Path) -> list[Path]:
    """Return a list of paths to delete, scoped to known safe targets."""
    repo_root = repo_root.resolve()
    deletions: list[Path] = []

    gallery_dir = repo_root / "data" / "gallery"
    plates_dir = repo_root / "data" / "plates"
    meta_dir = repo_root / "data" / "meta"
    outputs_dir = repo_root / "outputs"
    runs_dir = repo_root / "runs"

    _add_children(deletions, gallery_dir, repo_root)
    _add_children(deletions, plates_dir, repo_root)
    _add_children(deletions, meta_dir, repo_root)

    for path in [outputs_dir, runs_dir]:
        if path.exists():
            if _is_within_root(path, repo_root):
                deletions.append(path)
            else:
                LOGGER.warning("Skipping outside root: %s", path)

    unique = sorted({path.resolve() for path in deletions}, key=lambda p: str(p))
    return unique


def execute_deletions(paths: list[Path], repo_root: Path) -> int:
    """Delete the planned paths."""
    repo_root = repo_root.resolve()
    failures = 0
    for path in paths:
        if not _is_within_root(path, repo_root):
            LOGGER.warning("Refusing to delete outside root: %s", path)
            failures += 1
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            LOGGER.info("Deleted: %s", path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            LOGGER.error("Failed to delete %s: %s", path, exc)
            failures += 1
    return 0 if failures == 0 else 2
