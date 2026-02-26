"""Helper runner for Morris phase-1 and phase-2 sensitivity."""

import argparse

from simulation.sensitivity.morris_phase1 import run_morris_phase1
from simulation.sensitivity.morris_phase2 import run_morris_phase2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run/resume Morris phase-1 or phase-2 sensitivity analysis."
    )
    parser.add_argument("--output-dir", default="outputs", help="Output directory.")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=12345,
        help="SALib sampling seed for reproducible Morris design.",
    )
    parser.add_argument(
        "--morris-phase2",
        action="store_true",
        help="Run/resume Morris phase-2 pipeline (default is phase-1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.morris_phase2:
        run_morris_phase2(output_dir=args.output_dir, sample_seed=args.sample_seed)
    else:
        run_morris_phase1(output_dir=args.output_dir, sample_seed=args.sample_seed)


if __name__ == "__main__":
    main()
