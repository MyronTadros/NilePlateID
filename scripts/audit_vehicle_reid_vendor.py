"""Audit local vendor imports for third_party/vehicle_reid."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from contextlib import contextmanager
from pathlib import Path

LOGGER = logging.getLogger(__name__)

VENDOR_MODULES = [
    "third_party.vehicle_reid.model",
    "third_party.vehicle_reid.load_model",
    "third_party.vehicle_reid.extract_features",
    "third_party.vehicle_reid.match_one_query",
]


class _CliParseAbort(Exception):
    """Raised to stop CLI parsing during import checks."""


@contextmanager
def _block_argparse() -> None:
    original = argparse.ArgumentParser.parse_args

    def _blocked_parse_args(self, *args, **kwargs):
        raise _CliParseAbort("CLI parsing blocked during import")

    argparse.ArgumentParser.parse_args = _blocked_parse_args
    try:
        yield
    finally:
        argparse.ArgumentParser.parse_args = original


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_requirement_name(line: str) -> str | None:
    trimmed = line.strip()
    if not trimmed or trimmed.startswith("#"):
        return None
    for sep in ("==", ">=", "<=", "~=", "!="):
        if sep in trimmed:
            trimmed = trimmed.split(sep, 1)[0].strip()
            break
    return trimmed or None


def _load_vendor_deps(vendor_root: Path) -> set[str]:
    req_path = vendor_root / "requirements.txt"
    deps: set[str] = set()
    if not req_path.exists():
        return deps
    for line in req_path.read_text(encoding="utf-8").splitlines():
        name = _parse_requirement_name(line)
        if name:
            deps.add(name.lower())
    return deps


def _categorize_missing(name: str, vendor_root: Path, deps: set[str]) -> str:
    lowered = name.lower()
    alias_map = {
        "pyyaml": "yaml",
    }
    dep_names = set(deps)
    dep_names.update(alias_map.values())

    if lowered in dep_names:
        return "external"

    local_file = vendor_root / f"{name}.py"
    local_dir = vendor_root / name
    if local_file.exists() or local_dir.exists():
        return "local"

    return "local"


def _attempt_import(module_name: str) -> str | None:
    try:
        with _block_argparse():
            importlib.import_module(module_name)
        LOGGER.info("Import ok: %s", module_name)
        return None
    except ModuleNotFoundError as exc:
        LOGGER.error("Import failed: %s (missing module: %s)", module_name, exc.name)
        return exc.name
    except _CliParseAbort:
        LOGGER.warning("Import skipped CLI parsing: %s", module_name)
        return None
    except SystemExit:
        LOGGER.warning("Import exited early: %s", module_name)
        return None
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Import failed: %s (%s)", module_name, exc.__class__.__name__)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit third_party/vehicle_reid vendor imports."
    )
    parser.add_argument(
        "--repo_root",
        default=str(_repo_root()),
        help="Repository root (defaults to project root).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    vendor_root = repo_root / "third_party" / "vehicle_reid"
    if not vendor_root.is_dir():
        LOGGER.error("Vendor directory missing: %s", vendor_root)
        return 2

    deps = _load_vendor_deps(vendor_root)

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(vendor_root))
    missing_local: set[str] = set()
    missing_external: set[str] = set()
    try:
        for module_name in VENDOR_MODULES:
            missing = _attempt_import(module_name)
            if missing:
                category = _categorize_missing(missing, vendor_root, deps)
                if category == "local":
                    missing_local.add(missing)
                else:
                    missing_external.add(missing)
    finally:
        if str(vendor_root) in sys.path:
            sys.path.remove(str(vendor_root))
        if str(repo_root) in sys.path:
            sys.path.remove(str(repo_root))

    LOGGER.info("Missing local modules: %s", ", ".join(sorted(missing_local)) or "none")
    LOGGER.info("Missing external modules: %s", ", ".join(sorted(missing_external)) or "none")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
