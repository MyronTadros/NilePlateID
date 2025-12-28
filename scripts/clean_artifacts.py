"""Clean generated artifacts in the repository."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.clean import execute_deletions, plan_deletions

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan and delete generated artifacts (safe list only)."
    )
    parser.add_argument(
        "--repo_root",
        default=str(REPO_ROOT),
        help="Repository root (defaults to project root).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dry_run",
        action="store_true",
        help="Print deletion plan without deleting (default).",
    )
    group.add_argument(
        "--force",
        action="store_true",
        help="Delete planned artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
