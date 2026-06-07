"""Download and load the UCI MADELON dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import zipfile

import pandas as pd
import requests
from tqdm import tqdm

from .settings import PROCESSED_DIR, RAW_DIR, UCI_BASE_URL, UCI_FILES, UCI_ZIP_URL, ensure_runtime_dirs


def _download_file(filename: str, force: bool = False) -> Path:
    ensure_runtime_dirs()
    path = RAW_DIR / filename
    if path.exists() and path.stat().st_size > 0 and not force:
        return path

    url = f"{UCI_BASE_URL}/{filename}"
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=90) as response:
        if response.status_code == 404:
            download_madelon_zip(force=force)
            found = _find_raw_file(filename)
            if found is not None:
                return found
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with tmp.open("wb") as f, tqdm(
            total=total if total else None,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {filename}",
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
    tmp.replace(path)
    return path


def download_madelon_zip(force: bool = False) -> Path:
    """Download and extract the official UCI static archive."""
    ensure_runtime_dirs()
    zip_path = RAW_DIR / "madelon.zip"
    if force or not zip_path.exists() or zip_path.stat().st_size == 0:
        tmp = zip_path.with_suffix(".zip.part")
        with requests.get(UCI_ZIP_URL, stream=True, timeout=90) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with tmp.open("wb") as f, tqdm(
                total=total if total else None,
                unit="B",
                unit_scale=True,
                desc="Downloading official madelon.zip",
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))
        tmp.replace(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    return zip_path


def _find_raw_file(filename: str) -> Path | None:
    candidates = [RAW_DIR / filename, RAW_DIR / "MADELON" / filename]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def download_madelon(force: bool = False) -> None:
    """Download the official train/validation MADELON files."""
    for filename in UCI_FILES.values():
        _download_file(filename, force=force)


def _read_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df.columns = [f"V{idx:03d}" for idx in range(1, df.shape[1] + 1)]
    return df.astype(float)


def _read_labels(path: Path) -> pd.Series:
    labels = pd.read_csv(path, sep=r"\s+", header=None, engine="python")[0].astype(int)
    return labels.rename("target")


def load_madelon(force_download: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return official training and validation/test splits.

    The UCI files call the held-out labeled split `validation`; for this course
    project it is used only as the final test set.
    """
    ensure_runtime_dirs()
    download_madelon(force=force_download)

    paths = {key: _find_raw_file(filename) for key, filename in UCI_FILES.items()}
    if any(path is None for path in paths.values()):
        download_madelon_zip(force=force_download)
        paths = {key: _find_raw_file(filename) for key, filename in UCI_FILES.items()}
    missing = [key for key, path in paths.items() if path is None]
    if missing:
        raise FileNotFoundError(f"Missing MADELON files after download: {missing}")

    X_train = _read_data(paths["train_data"])
    y_train = _read_labels(paths["train_labels"])
    X_test = _read_data(paths["valid_data"])
    y_test = _read_labels(paths["valid_labels"])

    X_train.to_csv(PROCESSED_DIR / "madelon_train_features.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "madelon_train_labels.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "madelon_test_features.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "madelon_test_labels.csv", index=False)
    return X_train, y_train, X_test, y_test
