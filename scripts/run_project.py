#!/usr/bin/env python
"""Run the full MADELON high-dimensional inference project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hddi26_madelon.pipeline import RunConfig, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-folds", type=int, default=3, help="Cross-validation folds on the official training split.")
    parser.add_argument("--bootstrap-runs", type=int, default=1000, help="Bootstrap resamples for final-test accuracy CIs.")
    parser.add_argument("--sample-repeats", type=int, default=5, help="Repeats per sample size in the learning-curve experiment.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs used by GridSearchCV where applicable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        cv_folds=args.cv_folds,
        bootstrap_runs=args.bootstrap_runs,
        sample_repeats=args.sample_repeats,
        n_jobs=args.n_jobs,
    )
    run(config)


if __name__ == "__main__":
    main()

