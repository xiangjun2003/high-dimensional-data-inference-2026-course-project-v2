#!/usr/bin/env python
"""Run advanced MADELON extension experiments without touching main outputs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CACHE = PROJECT_ROOT / ".cache"
PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
(PROJECT_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
(PROJECT_CACHE / "joblib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(PROJECT_CACHE / "joblib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hddi26_madelon.extensions import ExtensionConfig, run_extensions


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scad-lambdas", default="0.5,1.0,2.0,5.0,10.0,20.0,50.0,100.0,200.0")
    parser.add_argument("--scad-iterations", type=int, default=5)
    parser.add_argument("--kernel-components", default="50,100,200,500,1000")
    parser.add_argument("--bayes-top-k", type=int, default=20)
    parser.add_argument("--bayes-draws", type=int, default=600)
    parser.add_argument("--gmm-pca-dims", default="5,10,20")
    parser.add_argument("--gmm-components", default="1,2,3,5,8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExtensionConfig(
        scad_lambdas=_parse_csv_floats(args.scad_lambdas),
        scad_iterations=args.scad_iterations,
        kernel_components=_parse_csv_ints(args.kernel_components),
        bayes_top_k=args.bayes_top_k,
        bayes_draws=args.bayes_draws,
        gmm_pca_dims=_parse_csv_ints(args.gmm_pca_dims),
        gmm_components=_parse_csv_ints(args.gmm_components),
    )
    run_extensions(config)


if __name__ == "__main__":
    main()
