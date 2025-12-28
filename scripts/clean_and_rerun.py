"""Clean artifacts and rerun the pipeline with max debug enabled."""

from __future__ import annotations

import subprocess
from pathlib import Path
import sys


def _format_command(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    python = sys.executable

    clean_cmd = [python, "scripts/clean_artifacts.py", "--force"]
    run_cmd = [
        python,
        "-m",
        "src.cli",
        "run",
        "--weights",
        "models/best.pt",
        "--input",
        "data/incoming",
        "--max_debug",
    ]

    print("Running:")
    print(_format_command(clean_cmd))
    result = subprocess.run(clean_cmd, cwd=repo_root)
    if result.returncode != 0:
        return result.returncode

    print("Running:")
    print(_format_command(run_cmd))
    result = subprocess.run(run_cmd, cwd=repo_root)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
