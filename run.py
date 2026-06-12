#!/usr/bin/env python3
"""Run the main experiment scripts from the repository root.

This launcher simply invokes the existing shell scripts in sequence:
- optimizer comparison experiments
- Adam and Muon ablations
- linear regression experiments
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

SCRIPTS = {
    "comparison": REPO_ROOT / "scripts" / "run_optimizer_comparison.sh",
    "ablation_adam": REPO_ROOT / "scripts" / "run_ablation_adam.sh",
    "ablation_muon": REPO_ROOT / "scripts" / "run_ablation_muon.sh",
    "linear": REPO_ROOT / "linear_regression" / "run_linear.sh",
}


def run_script(script_path: Path) -> None:
    print(f"\n=== Running {script_path.relative_to(REPO_ROOT)} ===")
    subprocess.run(["bash", str(script_path)], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OptML experiment scripts.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(SCRIPTS.keys()),
        help="Run only the selected script groups.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = args.only if args.only else list(SCRIPTS.keys())

    for key in selected:
        script_path = SCRIPTS[key]
        if not script_path.exists():
            raise FileNotFoundError(f"Missing script: {script_path}")
        run_script(script_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())