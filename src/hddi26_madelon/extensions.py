"""Advanced extension experiments for the MADELON project.

These experiments intentionally write only to outputs/extensions so the main
pipeline results remain reproducible and untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable
import json
import os
import warnings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _PROJECT_ROOT / ".cache"
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "joblib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(_CACHE_DIR / "joblib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from .data import load_madelon
from .plotting import METHOD_COLORS, PALETTE, apply_style
from .settings import OUTPUT_DIR, RANDOM_STATE, ensure_runtime_dirs


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


EXTENSION_DIR = OUTPUT_DIR / "extensions"
EXTENSION_RESULT_DIR = EXTENSION_DIR / "results"
EXTENSION_FIGURE_DIR = EXTENSION_DIR / "figures"


@dataclass(frozen=True)
class ExtensionConfig:
    random_state: int = RANDOM_STATE
    scad_lambdas: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
    scad_iterations: int = 5
    kernel_components: tuple[int, ...] = (50, 100, 200, 500, 1000)
    bayes_top_k: int = 20
    bayes_draws: int = 600
    bayes_prior_variance: float = 4.0
    gmm_pca_dims: tuple[int, ...] = (5, 10, 20)
    gmm_components: tuple[int, ...] = (1, 2, 3, 5, 8)
    gmm_covariance_types: tuple[str, ...] = ("diag", "tied", "full")


def ensure_extension_dirs() -> None:
    ensure_runtime_dirs()
    EXTENSION_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    EXTENSION_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, name: str) -> Path:
    ensure_extension_dirs()
    path = EXTENSION_RESULT_DIR / name
    df.to_csv(path, index=False)
    return path


def _save_extension_figure(fig, name: str) -> None:
    ensure_extension_dirs()
    stem = EXTENSION_FIGURE_DIR / name
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=450, bbox_inches="tight")
    plt.close(fig)


def _positive_score(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        classes = list(estimator.classes_)
        pos_idx = classes.index(1) if 1 in classes else -1
        return proba[:, pos_idx]
    if hasattr(estimator, "decision_function"):
        score = estimator.decision_function(X)
        if np.ndim(score) == 2:
            classes = list(estimator.classes_) if hasattr(estimator, "classes_") else [-1, 1]
            pos_idx = classes.index(1) if 1 in classes else -1
            score = score[:, pos_idx]
        return np.asarray(score)
    return estimator.predict(X)


def _metrics(y_true: Iterable[int], pred: Iterable[int], score: Iterable[float]) -> dict[str, float]:
    try:
        auc = float(roc_auc_score(y_true, score))
    except Exception:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "auc": auc,
        "f1": float(f1_score(y_true, pred, pos_label=1)),
    }


def _probability_metrics(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    pred = np.where(proba >= 0.5, 1, -1)
    out = _metrics(y_true, pred, proba)
    out["brier"] = float(brier_score_loss((y_true == 1).astype(int), proba))
    out["ece_10bin"] = _expected_calibration_error((y_true == 1).astype(int).to_numpy(), proba, n_bins=10)
    return out


def _expected_calibration_error(y01: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (proba >= lo) & (proba <= hi)
        else:
            mask = (proba >= lo) & (proba < hi)
        if mask.any():
            ece += mask.mean() * abs(float(proba[mask].mean()) - float(y01[mask].mean()))
    return float(ece)


def _logistic_log_likelihood(y: pd.Series, proba: np.ndarray) -> float:
    y01 = (y == 1).astype(float).to_numpy()
    p = np.clip(proba, 1e-8, 1.0 - 1e-8)
    return float(np.sum(y01 * np.log(p) + (1.0 - y01) * np.log(1.0 - p)))


def _bic_for_sparse_logistic(y: pd.Series, proba: np.ndarray, nonzero: int) -> float:
    return float(-2.0 * _logistic_log_likelihood(y, proba) + (nonzero + 1) * np.log(len(y)))


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _display_number(value) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "NA"
    if pd.isna(value):
        return "NA"
    return str(value)


def _weighted_l1_fit(X: np.ndarray, y: pd.Series, weights: np.ndarray, random_state: int) -> tuple[np.ndarray, float]:
    weights = np.maximum(np.asarray(weights, dtype=float), 1e-4)
    X_weighted = X / weights
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=1.0,
        fit_intercept=True,
        max_iter=3000,
        random_state=random_state,
    )
    model.fit(X_weighted, y)
    beta = model.coef_.ravel() / weights
    intercept = float(model.intercept_[0])
    return beta, intercept


def _scad_derivative(abs_beta: np.ndarray, lam: float, a: float = 3.7) -> np.ndarray:
    deriv = np.zeros_like(abs_beta, dtype=float)
    small = abs_beta <= lam
    middle = (abs_beta > lam) & (abs_beta <= a * lam)
    deriv[small] = lam
    deriv[middle] = (a * lam - abs_beta[middle]) / (a - 1.0)
    return np.maximum(deriv, 0.02 * lam)


def _predict_from_linear(X: np.ndarray, beta: np.ndarray, intercept: float) -> tuple[np.ndarray, np.ndarray]:
    proba = expit(intercept + X @ beta)
    pred = np.where(proba >= 0.5, 1, -1)
    return pred, proba


def run_scad_experiment(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, config: ExtensionConfig) -> pd.DataFrame:
    """Compare L1, Elastic Net, and SCAD-LLA sparse logistic paths."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    rows: list[dict] = []
    selected_sets: dict[tuple[str, float], set[str]] = {}
    coef_rows: list[dict] = []

    for lam in config.scad_lambdas:
        t0 = perf_counter()
        beta_l1, intercept_l1 = _weighted_l1_fit(Xtr, y_train, np.full(Xtr.shape[1], lam), config.random_state)
        pred_l1, proba_l1 = _predict_from_linear(Xte, beta_l1, intercept_l1)
        train_proba_l1 = _predict_from_linear(Xtr, beta_l1, intercept_l1)[1]
        nonzero_l1 = int(np.sum(np.abs(beta_l1) > 1e-6))
        rows.append(
            {
                "method": "L1 Logistic",
                "lambda": lam,
                "nonzero_features": nonzero_l1,
                "bic": _bic_for_sparse_logistic(y_train, train_proba_l1, nonzero_l1),
                "train_time_sec": perf_counter() - t0,
                **_metrics(y_test, pred_l1, proba_l1),
            }
        )
        selected_sets[("L1 Logistic", lam)] = set(X_train.columns[np.abs(beta_l1) > 1e-6])

        t0 = perf_counter()
        enet = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=max(1.0 / lam, 1e-3),
            max_iter=4000,
            random_state=config.random_state,
        )
        enet.fit(Xtr, y_train)
        proba_enet = enet.predict_proba(Xte)[:, list(enet.classes_).index(1)]
        train_proba_enet = enet.predict_proba(Xtr)[:, list(enet.classes_).index(1)]
        pred_enet = np.where(proba_enet >= 0.5, 1, -1)
        beta_enet = enet.coef_.ravel()
        nonzero_enet = int(np.sum(np.abs(beta_enet) > 1e-6))
        rows.append(
            {
                "method": "Elastic Net Logistic",
                "lambda": lam,
                "nonzero_features": nonzero_enet,
                "bic": _bic_for_sparse_logistic(y_train, train_proba_enet, nonzero_enet),
                "train_time_sec": perf_counter() - t0,
                **_metrics(y_test, pred_enet, proba_enet),
            }
        )
        selected_sets[("Elastic Net Logistic", lam)] = set(X_train.columns[np.abs(beta_enet) > 1e-6])

        t0 = perf_counter()
        beta = beta_l1.copy()
        intercept = intercept_l1
        for _ in range(config.scad_iterations):
            weights = _scad_derivative(np.abs(beta), lam)
            beta, intercept = _weighted_l1_fit(Xtr, y_train, weights, config.random_state)
        pred_scad, proba_scad = _predict_from_linear(Xte, beta, intercept)
        train_proba_scad = _predict_from_linear(Xtr, beta, intercept)[1]
        nonzero_scad = int(np.sum(np.abs(beta) > 1e-6))
        rows.append(
            {
                "method": "SCAD Logistic via LLA",
                "lambda": lam,
                "nonzero_features": nonzero_scad,
                "bic": _bic_for_sparse_logistic(y_train, train_proba_scad, nonzero_scad),
                "train_time_sec": perf_counter() - t0,
                **_metrics(y_test, pred_scad, proba_scad),
            }
        )
        selected_sets[("SCAD Logistic via LLA", lam)] = set(X_train.columns[np.abs(beta) > 1e-6])
        for feature, value in zip(X_train.columns, beta):
            if abs(value) > 1e-6:
                coef_rows.append({"method": "SCAD Logistic via LLA", "lambda": lam, "feature": feature, "coefficient": float(value)})

    result = pd.DataFrame(rows)
    _write_csv(result, "extension_A_scad_sparse_logistic.csv")
    _write_csv(pd.DataFrame(coef_rows), "extension_A_scad_coefficients.csv")

    overlap_rows = []
    for lam in config.scad_lambdas:
        l1 = selected_sets[("L1 Logistic", lam)]
        scad = selected_sets[("SCAD Logistic via LLA", lam)]
        union = l1 | scad
        overlap_rows.append(
            {
                "lambda": lam,
                "l1_selected": len(l1),
                "scad_selected": len(scad),
                "intersection": len(l1 & scad),
                "jaccard": len(l1 & scad) / len(union) if union else 1.0,
            }
        )
    _write_csv(pd.DataFrame(overlap_rows), "extension_A_l1_scad_overlap.csv")
    _plot_scad_results(result)
    _plot_scad_coefficients(pd.DataFrame(coef_rows))
    return result


def _plot_scad_results(result: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7), sharex=True)
    metrics = [("accuracy", "Accuracy"), ("nonzero_features", "Selected features"), ("bic", "BIC")]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for idx, (method, group) in enumerate(result.groupby("method")):
            group = group.sort_values("lambda")
            ax.plot(group["lambda"], group[metric], marker="o", lw=1.1, ms=3.0, color=METHOD_COLORS[idx % len(METHOD_COLORS)], label=method)
        ax.set_xscale("log")
        ax.set_xlabel("Penalty lambda")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, loc="left", fontweight="bold")
    axes[0].legend(fontsize=5.5)
    fig.suptitle("Extension A: SCAD/LLA sparse logistic path", x=0.02, y=1.05, ha="left", fontsize=8, fontweight="bold")
    _save_extension_figure(fig, "extension_A_scad_sparse_path")


def _plot_scad_coefficients(coef_df: pd.DataFrame) -> None:
    if coef_df.empty:
        return
    top_features = (
        coef_df.assign(abs_coef=lambda d: d["coefficient"].abs())
        .groupby("feature", as_index=False)["abs_coef"]
        .max()
        .sort_values("abs_coef", ascending=False)
        .head(12)["feature"]
        .tolist()
    )
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for idx, feature in enumerate(top_features):
        subset = coef_df[coef_df["feature"] == feature].sort_values("lambda")
        ax.plot(subset["lambda"], subset["coefficient"], marker="o", lw=0.9, ms=2.5, color=METHOD_COLORS[idx % len(METHOD_COLORS)], label=feature)
    ax.axhline(0.0, color=PALETTE["black"], lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Penalty lambda")
    ax.set_ylabel("SCAD coefficient")
    ax.set_title("Largest SCAD coefficients across penalty strengths", loc="left", fontweight="bold")
    ax.legend(fontsize=5.3, ncol=2)
    _save_extension_figure(fig, "extension_A_scad_coefficient_path")


def run_kernel_approximation_experiment(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, config: ExtensionConfig
) -> pd.DataFrame:
    """Compare exact RBF-SVM with finite-dimensional RBF approximations."""
    gamma = 1.0 / X_train.shape[1]
    rows: list[dict] = []

    exact = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVC(C=10.0, gamma=gamma, kernel="rbf", probability=True, random_state=config.random_state)),
        ]
    )
    t0 = perf_counter()
    exact.fit(X_train, y_train)
    exact_train_time = perf_counter() - t0
    pred = exact.predict(X_test)
    score = _positive_score(exact, X_test)
    rows.append(
        {
            "method": "Exact RBF-SVM",
            "feature_map": "implicit RBF",
            "classifier": "SVC",
            "components": np.nan,
            "train_time_sec": exact_train_time,
            **_metrics(y_test, pred, score),
        }
    )

    for m in config.kernel_components:
        specs = [
            ("RFF + Logistic", RBFSampler(gamma=gamma, n_components=m, random_state=config.random_state), LogisticRegression(max_iter=2500, random_state=config.random_state)),
            ("RFF + Linear SVM", RBFSampler(gamma=gamma, n_components=m, random_state=config.random_state), LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=config.random_state)),
            ("Nystroem + Logistic", Nystroem(kernel="rbf", gamma=gamma, n_components=m, random_state=config.random_state), LogisticRegression(max_iter=2500, random_state=config.random_state)),
            ("Nystroem + Linear SVM", Nystroem(kernel="rbf", gamma=gamma, n_components=m, random_state=config.random_state), LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=config.random_state)),
        ]
        for method, mapper, classifier in specs:
            estimator = Pipeline([("scaler", StandardScaler()), ("map", mapper), ("model", classifier)])
            t0 = perf_counter()
            estimator.fit(X_train, y_train)
            train_time = perf_counter() - t0
            pred = estimator.predict(X_test)
            score = _positive_score(estimator, X_test)
            rows.append(
                {
                    "method": method,
                    "feature_map": "RFF" if method.startswith("RFF") else "Nystroem",
                    "classifier": "Logistic" if "Logistic" in method else "Linear SVM",
                    "components": m,
                    "train_time_sec": train_time,
                    **_metrics(y_test, pred, score),
                }
            )

    result = pd.DataFrame(rows)
    _write_csv(result, "extension_B_kernel_approximation.csv")
    _plot_kernel_approximation(result)
    return result


def _plot_kernel_approximation(result: pd.DataFrame) -> None:
    apply_style()
    exact = result[result["method"] == "Exact RBF-SVM"].iloc[0]
    approx = result[result["method"] != "Exact RBF-SVM"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7))
    for idx, (method, group) in enumerate(approx.groupby("method")):
        group = group.sort_values("components")
        color = METHOD_COLORS[idx % len(METHOD_COLORS)]
        axes[0].plot(group["components"], group["accuracy"], marker="o", ms=3, lw=1.0, color=color, label=method)
        axes[1].plot(group["components"], group["auc"], marker="o", ms=3, lw=1.0, color=color, label=method)
        axes[2].plot(group["train_time_sec"], group["accuracy"], marker="o", ms=3, lw=0.0, color=color, label=method)
    axes[0].axhline(exact["accuracy"], color=PALETTE["black"], lw=0.8, ls="--", label="Exact RBF-SVM")
    axes[1].axhline(exact["auc"], color=PALETTE["black"], lw=0.8, ls="--")
    axes[2].scatter([exact["train_time_sec"]], [exact["accuracy"]], color=PALETTE["black"], s=18, marker="x", label="Exact RBF-SVM")
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("Components / landmarks")
    axes[1].set_xlabel("Components / landmarks")
    axes[2].set_xlabel("Training time (s)")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_ylabel("AUC")
    axes[2].set_ylabel("Accuracy")
    axes[0].set_title("Approximation dimension", loc="left", fontweight="bold")
    axes[1].set_title("AUC recovery", loc="left", fontweight="bold")
    axes[2].set_title("Accuracy-time trade-off", loc="left", fontweight="bold")
    axes[0].legend(fontsize=5.2)
    _save_extension_figure(fig, "extension_B_kernel_approximation")


def _rf_top_features(X_train: pd.DataFrame, y_train: pd.Series, top_k: int, random_state: int) -> list[str]:
    ranking_path = OUTPUT_DIR / "results" / "feature_rankings.csv"
    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path)
        return ranking.sort_values("rank_Random Forest")["feature"].head(top_k).tolist()
    model = RandomForestClassifier(n_estimators=220, max_features="sqrt", min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    order = np.argsort(model.feature_importances_)[::-1][:top_k]
    return X_train.columns[order].tolist()


def run_bayesian_logistic_experiment(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, config: ExtensionConfig
) -> pd.DataFrame:
    """Laplace approximation for Bayesian logistic uncertainty on RF-top features."""
    rng = np.random.default_rng(config.random_state)
    features = _rf_top_features(X_train, y_train, config.bayes_top_k, config.random_state)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train[features])
    Xte = scaler.transform(X_test[features])

    point = LogisticRegression(C=config.bayes_prior_variance, penalty="l2", solver="lbfgs", max_iter=3000, random_state=config.random_state)
    point.fit(Xtr, y_train)
    point_proba = point.predict_proba(Xte)[:, list(point.classes_).index(1)]

    beta_map = np.concatenate([point.intercept_, point.coef_.ravel()])
    X_aug = np.column_stack([np.ones(Xtr.shape[0]), Xtr])
    p_train = expit(X_aug @ beta_map)
    weights = p_train * (1.0 - p_train)
    prior_prec = np.diag(np.r_[0.0, np.full(Xtr.shape[1], 1.0 / config.bayes_prior_variance)])
    hessian = X_aug.T @ (weights[:, None] * X_aug) + prior_prec
    cov = np.linalg.pinv(hessian)
    cov = 0.5 * (cov + cov.T) + np.eye(cov.shape[0]) * 1e-8

    draws = rng.multivariate_normal(beta_map, cov, size=config.bayes_draws, check_valid="ignore")
    Xte_aug = np.column_stack([np.ones(Xte.shape[0]), Xte])
    posterior_probs = expit(Xte_aug @ draws.T)
    laplace_proba = posterior_probs.mean(axis=1)

    point_metrics = {
        "method": "L2 Logistic MAP",
        "features": config.bayes_top_k,
        "posterior_draws": 0,
        **_probability_metrics(y_test, point_proba),
    }
    laplace_metrics = {
        "method": "Bayesian Logistic Laplace",
        "features": config.bayes_top_k,
        "posterior_draws": config.bayes_draws,
        **_probability_metrics(y_test, laplace_proba),
    }
    result = pd.DataFrame([point_metrics, laplace_metrics])
    _write_csv(result, "extension_C_bayesian_logistic_metrics.csv")

    entropy = -(laplace_proba * np.log(np.clip(laplace_proba, 1e-8, 1.0)) + (1.0 - laplace_proba) * np.log(np.clip(1.0 - laplace_proba, 1e-8, 1.0)))
    correct = np.where((laplace_proba >= 0.5) == (y_test.to_numpy() == 1), "correct", "incorrect")
    entropy_df = pd.DataFrame({"test_index": np.arange(len(y_test)), "posterior_probability": laplace_proba, "predictive_entropy": entropy, "status": correct})
    _write_csv(entropy_df, "extension_C_bayesian_entropy_by_sample.csv")

    coef_draws = draws[:, 1:]
    coef_df = pd.DataFrame(
        {
            "feature": features,
            "posterior_mean": coef_draws.mean(axis=0),
            "posterior_sd": coef_draws.std(axis=0),
            "ci_low": np.quantile(coef_draws, 0.025, axis=0),
            "ci_high": np.quantile(coef_draws, 0.975, axis=0),
            "map_estimate": point.coef_.ravel(),
        }
    ).sort_values("posterior_mean", key=lambda s: s.abs(), ascending=False)
    _write_csv(coef_df, "extension_C_bayesian_coefficient_intervals.csv")

    calibration_rows = []
    for method, proba in [("L2 Logistic MAP", point_proba), ("Bayesian Logistic Laplace", laplace_proba)]:
        y01 = (y_test == 1).astype(int).to_numpy()
        bins = np.linspace(0.0, 1.0, 11)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (proba >= lo) & (proba < hi) if hi < 1.0 else (proba >= lo) & (proba <= hi)
            if mask.any():
                calibration_rows.append(
                    {
                        "method": method,
                        "bin_low": lo,
                        "bin_high": hi,
                        "mean_predicted_probability": float(proba[mask].mean()),
                        "observed_positive_rate": float(y01[mask].mean()),
                        "n": int(mask.sum()),
                    }
                )
    calibration_df = pd.DataFrame(calibration_rows)
    _write_csv(calibration_df, "extension_C_calibration_bins.csv")
    _plot_bayesian_logistic(calibration_df, entropy_df, coef_df)
    return result


def _plot_bayesian_logistic(calibration_df: pd.DataFrame, entropy_df: pd.DataFrame, coef_df: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7))
    axes[0].plot([0, 1], [0, 1], color=PALETTE["black"], lw=0.8, ls="--")
    for idx, (method, group) in enumerate(calibration_df.groupby("method")):
        axes[0].plot(group["mean_predicted_probability"], group["observed_positive_rate"], marker="o", lw=1.0, ms=3, color=METHOD_COLORS[idx], label=method)
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed positive rate")
    axes[0].set_title("Calibration", loc="left", fontweight="bold")
    axes[0].legend(fontsize=5.5)

    order = ["correct", "incorrect"]
    values = [entropy_df.loc[entropy_df["status"] == status, "predictive_entropy"].to_numpy() for status in order]
    axes[1].boxplot(values, labels=order, patch_artist=True, boxprops={"facecolor": PALETTE["light_blue"], "edgecolor": PALETTE["black"]}, medianprops={"color": PALETTE["red"]})
    axes[1].set_ylabel("Predictive entropy")
    axes[1].set_title("Uncertainty separates errors", loc="left", fontweight="bold")

    top = coef_df.head(10).iloc[::-1]
    xerr = np.vstack([top["posterior_mean"] - top["ci_low"], top["ci_high"] - top["posterior_mean"]])
    axes[2].errorbar(top["posterior_mean"], top["feature"], xerr=xerr, fmt="o", color=PALETTE["blue"], ecolor=PALETTE["gray"], lw=0.8, ms=3)
    axes[2].axvline(0.0, color=PALETTE["black"], lw=0.7)
    axes[2].set_xlabel("Posterior coefficient")
    axes[2].set_title("95% credible intervals", loc="left", fontweight="bold")
    _save_extension_figure(fig, "extension_C_bayesian_logistic_uncertainty")


def run_gmm_em_experiment(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, config: ExtensionConfig) -> pd.DataFrame:
    """Class-conditional EM-GMM classifier in low-dimensional PCA spaces."""
    rows: list[dict] = []
    classes = np.array([-1, 1])
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(X_train)
    Xte_scaled = scaler.transform(X_test)

    for dim in config.gmm_pca_dims:
        pca = PCA(n_components=dim, random_state=config.random_state)
        Xtr = pca.fit_transform(Xtr_scaled)
        Xte = pca.transform(Xte_scaled)
        for cov_type in config.gmm_covariance_types:
            for k in config.gmm_components:
                t0 = perf_counter()
                models = {}
                bic = 0.0
                converged = True
                lower_bounds = []
                for cls in classes:
                    X_cls = Xtr[y_train.to_numpy() == cls]
                    model = GaussianMixture(
                        n_components=k,
                        covariance_type=cov_type,
                        reg_covar=1e-5,
                        max_iter=250,
                        n_init=3,
                        random_state=config.random_state,
                    )
                    model.fit(X_cls)
                    models[cls] = model
                    bic += float(model.bic(X_cls))
                    converged = converged and bool(model.converged_)
                    lower_bounds.append(float(model.lower_bound_))
                log_scores = []
                for cls in classes:
                    prior = np.log(float(np.mean(y_train.to_numpy() == cls)))
                    log_scores.append(models[cls].score_samples(Xte) + prior)
                log_scores_arr = np.vstack(log_scores).T
                pred = classes[np.argmax(log_scores_arr, axis=1)]
                score = log_scores_arr[:, list(classes).index(1)] - log_scores_arr[:, list(classes).index(-1)]
                rows.append(
                    {
                        "method": "Class-conditional EM-GMM",
                        "pca_components": dim,
                        "gmm_components": k,
                        "covariance_type": cov_type,
                        "bic": bic,
                        "mean_em_lower_bound": float(np.mean(lower_bounds)),
                        "converged": converged,
                        "train_time_sec": perf_counter() - t0,
                        **_metrics(y_test, pred, score),
                    }
                )

    result = pd.DataFrame(rows)
    _write_csv(result, "extension_D_em_gmm_bic.csv")
    best = result.sort_values(["accuracy", "auc"], ascending=False).iloc[0].to_dict()
    _write_csv(_gmm_lower_bound_trace(Xtr_scaled, y_train, best, config), "extension_D_em_lower_bound_trace.csv")
    _plot_gmm_results(result)
    _plot_gmm_responsibilities(Xtr_scaled, y_train, Xte_scaled, y_test, best, config)
    return result


def _gmm_lower_bound_trace(Xtr_scaled: np.ndarray, y_train: pd.Series, best: dict, config: ExtensionConfig) -> pd.DataFrame:
    dim = int(best["pca_components"])
    k = int(best["gmm_components"])
    cov_type = str(best["covariance_type"])
    pca = PCA(n_components=dim, random_state=config.random_state)
    Xtr = pca.fit_transform(Xtr_scaled)
    rows = []
    for cls in [-1, 1]:
        X_cls = Xtr[y_train.to_numpy() == cls]
        model = GaussianMixture(
            n_components=k,
            covariance_type=cov_type,
            reg_covar=1e-5,
            max_iter=1,
            n_init=1,
            warm_start=True,
            random_state=config.random_state,
        )
        for iteration in range(1, 31):
            model.fit(X_cls)
            rows.append({"class": cls, "iteration": iteration, "lower_bound": float(model.lower_bound_)})
    return pd.DataFrame(rows)


def _plot_gmm_results(result: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
    for idx, (dim, group) in enumerate(result.groupby("pca_components")):
        grouped = group.groupby("gmm_components", as_index=False).agg({"accuracy": "max", "bic": "min"})
        color = METHOD_COLORS[idx % len(METHOD_COLORS)]
        axes[0].plot(grouped["gmm_components"], grouped["accuracy"], marker="o", lw=1.0, ms=3, color=color, label=f"PCA-{dim}")
        axes[1].plot(grouped["gmm_components"], grouped["bic"], marker="o", lw=1.0, ms=3, color=color, label=f"PCA-{dim}")
    axes[0].set_xlabel("Mixture components K")
    axes[1].set_xlabel("Mixture components K")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_ylabel("BIC")
    axes[0].set_title("Best EM-GMM accuracy over covariance types", loc="left", fontweight="bold")
    axes[1].set_title("Minimum BIC over covariance types", loc="left", fontweight="bold")
    axes[0].legend(fontsize=5.5)
    _save_extension_figure(fig, "extension_D_em_gmm_bic")


def _plot_gmm_responsibilities(
    Xtr_scaled: np.ndarray,
    y_train: pd.Series,
    Xte_scaled: np.ndarray,
    y_test: pd.Series,
    best: dict,
    config: ExtensionConfig,
) -> None:
    dim = int(best["pca_components"])
    k = int(best["gmm_components"])
    cov_type = str(best["covariance_type"])
    pca = PCA(n_components=max(2, dim), random_state=config.random_state)
    Xtr = pca.fit_transform(Xtr_scaled)
    Xte = pca.transform(Xte_scaled)
    classes = np.array([-1, 1])
    models = {}
    for cls in classes:
        model = GaussianMixture(
            n_components=k,
            covariance_type=cov_type,
            reg_covar=1e-5,
            max_iter=250,
            n_init=3,
            random_state=config.random_state,
        )
        model.fit(Xtr[y_train.to_numpy() == cls, :dim])
        models[cls] = model
    log_scores = []
    resp_strength = []
    for cls in classes:
        prior = np.log(float(np.mean(y_train.to_numpy() == cls)))
        log_scores.append(models[cls].score_samples(Xte[:, :dim]) + prior)
    log_scores_arr = np.vstack(log_scores).T
    pred = classes[np.argmax(log_scores_arr, axis=1)]
    for i, cls in enumerate(pred):
        resp_strength.append(float(models[cls].predict_proba(Xte[i : i + 1, :dim]).max()))
    resp_strength = np.asarray(resp_strength)
    correct = pred == y_test.to_numpy()

    apply_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for cls, color in [(-1, PALETTE["red"]), (1, PALETTE["blue"])]:
        mask = pred == cls
        ax.scatter(
            Xte[mask, 0],
            Xte[mask, 1],
            s=12 + 28 * resp_strength[mask],
            color=color,
            alpha=0.68,
            edgecolor="white",
            linewidth=0.25,
            label=f"pred {cls:+d}",
        )
    if (~correct).any():
        ax.scatter(Xte[~correct, 0], Xte[~correct, 1], s=50, facecolor="none", edgecolor=PALETTE["black"], linewidth=0.9, label="error")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("GMM responsibilities in PCA space", loc="left", fontweight="bold")
    ax.legend(fontsize=5.5)
    _save_extension_figure(fig, "extension_D_gmm_responsibilities")


def write_extension_summary(
    scad: pd.DataFrame,
    kernel: pd.DataFrame,
    bayes: pd.DataFrame,
    gmm: pd.DataFrame,
    config: ExtensionConfig,
) -> Path:
    best_sparse = scad.sort_values("accuracy", ascending=False).iloc[0]
    best_scad = scad[scad["method"] == "SCAD Logistic via LLA"].sort_values("accuracy", ascending=False).iloc[0]
    bic_sparse = scad.sort_values("bic", ascending=True).iloc[0]
    best_kernel = kernel.sort_values("accuracy", ascending=False).iloc[0]
    best_bayes = bayes.sort_values("auc", ascending=False).iloc[0]
    best_gmm_acc = gmm.sort_values("accuracy", ascending=False).iloc[0]
    best_gmm_bic = gmm.sort_values("bic", ascending=True).iloc[0]
    for stale_name in ["extension_best_scad_family.csv"]:
        stale_path = EXTENSION_RESULT_DIR / stale_name
        if stale_path.exists():
            stale_path.unlink()
    payload = {
        "config": {
            "scad_lambdas": config.scad_lambdas,
            "kernel_components": config.kernel_components,
            "bayes_top_k": config.bayes_top_k,
            "bayes_draws": config.bayes_draws,
            "gmm_pca_dims": config.gmm_pca_dims,
            "gmm_components": config.gmm_components,
            "gmm_covariance_types": config.gmm_covariance_types,
        },
        "best_sparse_family": best_sparse.to_dict(),
        "best_scad_only": best_scad.to_dict(),
        "bic_selected_sparse_logistic": bic_sparse.to_dict(),
        "best_kernel_family": best_kernel.to_dict(),
        "best_bayesian_family": best_bayes.to_dict(),
        "best_gmm_by_accuracy": best_gmm_acc.to_dict(),
        "best_gmm_by_bic": best_gmm_bic.to_dict(),
    }
    _write_csv(pd.DataFrame([best_sparse.to_dict()]), "extension_best_sparse_family.csv")
    _write_csv(pd.DataFrame([best_scad.to_dict()]), "extension_best_scad_only.csv")
    _write_csv(pd.DataFrame([bic_sparse.to_dict()]), "extension_bic_selected_sparse_logistic.csv")
    _write_csv(pd.DataFrame([best_kernel.to_dict()]), "extension_best_kernel_family.csv")
    _write_csv(pd.DataFrame([best_bayes.to_dict()]), "extension_best_bayesian_family.csv")
    _write_csv(pd.DataFrame([best_gmm_acc.to_dict()]), "extension_best_gmm_by_accuracy.csv")
    _write_csv(pd.DataFrame([best_gmm_bic.to_dict()]), "extension_best_gmm_by_bic.csv")

    summary = f"""# Advanced extension experiment summary

These experiments are isolated from the main project pipeline. They read the same
official MADELON train/validation split, but write only to `outputs/extensions`.

## Extension A: SCAD / nonconvex sparse logistic regression

Best sparse-logistic-family result by validation accuracy:
**{best_sparse['method']}**, lambda = {best_sparse['lambda']}, accuracy =
{best_sparse['accuracy']:.3f}, AUC = {best_sparse['auc']:.3f}, selected
features = {int(best_sparse['nonzero_features'])}, BIC =
{best_sparse['bic']:.1f}.

Best SCAD-only result: lambda = {best_scad['lambda']}, accuracy =
{best_scad['accuracy']:.3f}, AUC = {best_scad['auc']:.3f}, selected features =
{int(best_scad['nonzero_features'])}, BIC = {best_scad['bic']:.1f}.

BIC-selected sparse logistic result: **{bic_sparse['method']}**, lambda =
{bic_sparse['lambda']}, accuracy = {bic_sparse['accuracy']:.3f}, selected
features = {int(bic_sparse['nonzero_features'])}, BIC =
{bic_sparse['bic']:.1f}.

## Extension B: Random Fourier Features / Nystroem kernel approximation

Best kernel-family result: **{best_kernel['method']}**, components =
{_display_number(best_kernel['components'])}, accuracy = {best_kernel['accuracy']:.3f}, AUC =
{best_kernel['auc']:.3f}, train time = {best_kernel['train_time_sec']:.3f}s.

## Extension C: Bayesian logistic regression with Laplace approximation

Best uncertainty result by AUC: **{best_bayes['method']}**, accuracy =
{best_bayes['accuracy']:.3f}, AUC = {best_bayes['auc']:.3f}, Brier =
{best_bayes['brier']:.3f}, ECE = {best_bayes['ece_10bin']:.3f}.

## Extension D: EM-GMM classifier and BIC model selection

Best GMM by validation accuracy: PCA-{int(best_gmm_acc['pca_components'])},
K={int(best_gmm_acc['gmm_components'])}, covariance =
{best_gmm_acc['covariance_type']}, accuracy = {best_gmm_acc['accuracy']:.3f},
BIC = {best_gmm_acc['bic']:.1f}.

BIC-selected GMM: PCA-{int(best_gmm_bic['pca_components'])},
K={int(best_gmm_bic['gmm_components'])}, covariance =
{best_gmm_bic['covariance_type']}, accuracy = {best_gmm_bic['accuracy']:.3f},
BIC = {best_gmm_bic['bic']:.1f}.

```json
{json.dumps(_json_ready(payload), indent=2, default=str)}
```
"""
    path = EXTENSION_RESULT_DIR / "extension_summary.md"
    path.write_text(summary, encoding="utf-8")
    return path


def run_extensions(config: ExtensionConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or ExtensionConfig()
    ensure_extension_dirs()
    apply_style()
    X_train, y_train, X_test, y_test = load_madelon()
    print("Running extension A: SCAD sparse logistic path", flush=True)
    scad = run_scad_experiment(X_train, y_train, X_test, y_test, config)
    print("Running extension B: RFF/Nystroem kernel approximation", flush=True)
    kernel = run_kernel_approximation_experiment(X_train, y_train, X_test, y_test, config)
    print("Running extension C: Bayesian logistic Laplace uncertainty", flush=True)
    bayes = run_bayesian_logistic_experiment(X_train, y_train, X_test, y_test, config)
    print("Running extension D: EM-GMM with BIC", flush=True)
    gmm = run_gmm_em_experiment(X_train, y_train, X_test, y_test, config)
    write_extension_summary(scad, kernel, bayes, gmm, config)
    print(f"Extension outputs written to {EXTENSION_DIR}", flush=True)
    return {"scad": scad, "kernel": kernel, "bayes": bayes, "gmm": gmm}
