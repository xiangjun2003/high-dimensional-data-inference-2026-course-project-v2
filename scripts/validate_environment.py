#!/usr/bin/env python
"""Validate the Python environment required for the MADELON project."""

from __future__ import annotations

import importlib
import sys


REQUIRED = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("umap", "umap-learn"),
    ("requests", "requests"),
    ("tqdm", "tqdm"),
]


def main() -> None:
    missing = []
    print(f"python {sys.version.split()[0]}")
    for import_name, package_name in REQUIRED:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"{package_name} {version}")
        except Exception as exc:
            missing.append((package_name, str(exc)))
    if missing:
        print("missing packages:")
        for package_name, reason in missing:
            print(f"- {package_name}: {reason}")
        raise SystemExit(1)
    print("environment smoke_test ok")


if __name__ == "__main__":
    main()

