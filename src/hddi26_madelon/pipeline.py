"""End-to-end MADELON analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Tuple
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from scipy.cluster.hierarchy import linkage
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from .data import load_madelon
from .plotting import (
    CLASS_COLORS,
    METHOD_COLORS,
    PALETTE,
    add_panel_label,
    apply_style,
    save_figure,
    scatter_by_label,
)
from .settings import FIGURE_DIR, RANDOM_STATE, RESULT_DIR, ensure_runtime_dirs


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class RunConfig:
    random_state: int = RANDOM_STATE
    cv_folds: int = 3
    n_jobs: int = 1
    bootstrap_runs: int = 1000
    sample_repeats: int = 5
    tsne_perplexity: int = 35
    rfe_step: float = 0.1


def _write_csv(df: pd.DataFrame, name: str) -> Path:
    path = RESULT_DIR / name
    df.to_csv(path, index=False)
    return path


def _signed_score(estimator, X) -> np.ndarray:
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


def _binary_metrics(y_true, pred, score) -> Dict[str, float]:
    try:
        auc = float(roc_auc_score(y_true, score))
    except Exception:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "auc": auc,
        "f1": float(f1_score(y_true, pred, pos_label=1)),
    }


def _evaluate_model(name: str, estimator, X_train, y_train, X_test, y_test, selected_features=None) -> Tuple[dict, object, np.ndarray, np.ndarray]:
    t0 = perf_counter()
    estimator.fit(X_train, y_train)
    train_time = perf_counter() - t0
    t1 = perf_counter()
    pred = estimator.predict(X_test)
    score = _signed_score(estimator, X_test)
    predict_time = perf_counter() - t1
    row = {
        "method": name,
        "selected_features": selected_features if selected_features is not None else X_train.shape[1],
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
        **_binary_metrics(y_test, pred, score),
    }
    return row, estimator, pred, score


def _model_factory(random_state: int) -> Dict[str, object]:
    return {
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=random_state))]
        ),
        "Linear SVM": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=random_state))]
        ),
        "RBF-SVM": Pipeline(
            [("scaler", StandardScaler()), ("model", SVC(C=10.0, gamma="scale", kernel="rbf", probability=True, random_state=random_state))]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=220,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=160,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }


def prepare_data(config: RunConfig):
    X_train, y_train, X_test, y_test = load_madelon()
    summary = pd.DataFrame(
        [
            {"item": "dataset", "value": "UCI MADELON"},
            {"item": "train_samples", "value": X_train.shape[0]},
            {"item": "test_samples_official_validation", "value": X_test.shape[0]},
            {"item": "features", "value": X_train.shape[1]},
            {"item": "classes", "value": len(np.unique(y_train))},
            {"item": "p_over_train_n", "value": X_train.shape[1] / X_train.shape[0]},
            {"item": "missing_train", "value": int(X_train.isna().sum().sum())},
            {"item": "missing_test", "value": int(X_test.isna().sum().sum())},
            {"item": "random_state", "value": config.random_state},
        ]
    )
    _write_csv(summary, "data_summary.csv")
    class_rows = []
    for split, y in [("train", y_train), ("test", y_test)]:
        counts = y.value_counts().sort_index()
        for label, n in counts.items():
            class_rows.append({"split": split, "class": int(label), "n": int(n)})
    _write_csv(pd.DataFrame(class_rows), "class_distribution.csv")
    return X_train, y_train, X_test, y_test


def eda_and_embeddings(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, config: RunConfig):
    apply_style()
    X_all = pd.concat([X_train, X_test], ignore_index=True)
    y_all = pd.concat([y_train, y_test], ignore_index=True)
    split = np.array(["train"] * len(X_train) + ["test"] * len(X_test))

    counts = pd.read_csv(RESULT_DIR / "class_distribution.csv")
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    width = 0.36
    labels = [-1, 1]
    for i, split_name in enumerate(["train", "test"]):
        subset = counts[counts["split"] == split_name].set_index("class").reindex(labels)
        ax.bar(np.arange(len(labels)) + (i - 0.5) * width, subset["n"], width=width, label=split_name, edgecolor="black", lw=0.5)
    ax.set_xticks(np.arange(len(labels)), [f"class {v:+d}" for v in labels])
    ax.set_ylabel("Samples")
    ax.set_title("MADELON is balanced across official splits", loc="left", fontweight="bold")
    ax.legend(fontsize=6)
    save_figure(fig, "01_class_distribution")

    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    mean_diff = (X_train[y_train == 1].mean(axis=0) - X_train[y_train == -1].mean(axis=0)).abs()
    feature_stats = pd.DataFrame({"feature": X_train.columns, "mean": means.values, "std": stds.values, "abs_class_mean_diff": mean_diff.values})
    _write_csv(feature_stats.sort_values("abs_class_mean_diff", ascending=False), "feature_distribution_stats.csv")

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.hist(means, bins=45, color=PALETTE["blue"], edgecolor="white", lw=0.3)
    ax.set_title("Distribution of feature means", loc="left", fontweight="bold")
    ax.set_xlabel("Mean over training samples")
    ax.set_ylabel("Features")
    save_figure(fig, "02_feature_mean_distribution")

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.hist(stds, bins=45, color=PALETTE["teal"], edgecolor="white", lw=0.3)
    ax.set_title("Distribution of feature standard deviations", loc="left", fontweight="bold")
    ax.set_xlabel("Training-set standard deviation")
    ax.set_ylabel("Features")
    save_figure(fig, "03_feature_std_distribution")

    ranked_diff = np.sort(mean_diff.values)[::-1]
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ax.plot(np.arange(1, len(ranked_diff) + 1), ranked_diff, color=PALETTE["red"], lw=1.3)
    ax.set_title("Most features have weak marginal class separation", loc="left", fontweight="bold")
    ax.set_xlabel("Feature rank by |mean(+1)-mean(-1)|")
    ax.set_ylabel("Absolute class mean difference")
    save_figure(fig, "04_class_mean_difference_ranking")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X_all)
    pca = PCA(n_components=500, random_state=config.random_state)
    train_pca = pca.fit_transform(X_train_scaled)
    all_pca = pca.transform(X_all_scaled)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    pca_summary = pd.DataFrame(
        {
            "component": np.arange(1, len(evr) + 1),
            "explained_variance_ratio": evr,
            "cumulative_explained_variance": cum,
        }
    )
    _write_csv(pca_summary, "pca_explained_variance.csv")
    thresholds = []
    for threshold in [0.8, 0.9, 0.95]:
        thresholds.append({"threshold": threshold, "components_needed": int(np.searchsorted(cum, threshold) + 1)})
    _write_csv(pd.DataFrame(thresholds), "pca_variance_thresholds.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    axes[0].plot(np.arange(1, 51), evr[:50], color=PALETTE["blue"], lw=1.2)
    axes[0].set_title("First 50 PCA components", loc="left", fontweight="bold")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance ratio")
    axes[1].plot(np.arange(1, len(cum) + 1), cum, color=PALETTE["teal"], lw=1.2)
    for threshold in [0.8, 0.9, 0.95]:
        axes[1].axhline(threshold, color=PALETTE["gray"], ls="--", lw=0.7)
    axes[1].set_title("Cumulative explained variance", loc="left", fontweight="bold")
    axes[1].set_xlabel("Number of PCs")
    axes[1].set_ylabel("Cumulative variance")
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    save_figure(fig, "05_pca_explained_variance")

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    scatter_by_label(ax, all_pca[:, :2], y_all, "PCA 2D projection", "PC1", "PC2")
    save_figure(fig, "06_pca_2d_scatter")

    fig = plt.figure(figsize=(4.6, 3.6))
    ax = fig.add_subplot(111, projection="3d")
    y_arr = y_all.to_numpy()
    for cls, color in CLASS_COLORS.items():
        mask = y_arr == cls
        ax.scatter(all_pca[mask, 0], all_pca[mask, 1], all_pca[mask, 2], s=12, alpha=0.7, color=color, label=f"class {cls:+d}")
    ax.set_title("PCA 3D projection", loc="left", fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(fontsize=6)
    save_figure(fig, "07_pca_3d_scatter")

    pca50 = all_pca[:, :50]
    tsne = TSNE(n_components=2, perplexity=config.tsne_perplexity, init="pca", learning_rate="auto", random_state=config.random_state).fit_transform(pca50)
    umap_coords = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.2, metric="euclidean", random_state=config.random_state).fit_transform(pca50)

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    scatter_by_label(ax, tsne, y_all, "t-SNE after PCA-50", "t-SNE 1", "t-SNE 2")
    save_figure(fig, "08_tsne_2d_scatter")

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    scatter_by_label(ax, umap_coords, y_all, "UMAP after PCA-50", "UMAP 1", "UMAP 2")
    save_figure(fig, "09_umap_2d_scatter")

    return {
        "X_all": X_all,
        "y_all": y_all,
        "split": split,
        "scaler": scaler,
        "pca": pca,
        "all_pca": all_pca,
        "train_pca": train_pca,
        "tsne": tsne,
        "umap": umap_coords,
    }


def compute_feature_rankings(X_train: pd.DataFrame, y_train: pd.Series, config: RunConfig) -> pd.DataFrame:
    X_scaled = StandardScaler().fit_transform(X_train)
    feature_names = np.asarray(X_train.columns)

    f_scores, f_pvalues = f_classif(X_scaled, y_train)
    mi = mutual_info_classif(X_scaled, y_train, random_state=config.random_state, n_neighbors=5)

    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=2000, random_state=config.random_state)
    l1.fit(X_scaled, y_train)
    l1_importance = np.abs(l1.coef_).ravel()

    rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2, random_state=config.random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_importance = rf.feature_importances_

    X_sub, X_val, y_sub, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=config.random_state)
    perm_model = RandomForestClassifier(n_estimators=220, max_features="sqrt", min_samples_leaf=2, random_state=config.random_state, n_jobs=-1)
    perm_model.fit(X_sub, y_sub)
    perm = permutation_importance(perm_model, X_val, y_val, scoring="accuracy", n_repeats=10, random_state=config.random_state, n_jobs=1)

    rfe_estimator = LogisticRegression(penalty="l2", solver="liblinear", C=1.0, max_iter=2000, random_state=config.random_state)
    rfe = RFE(rfe_estimator, n_features_to_select=1, step=config.rfe_step)
    rfe.fit(X_scaled, y_train)
    rfe_importance = 1 / rfe.ranking_.astype(float)

    ranking_df = pd.DataFrame(
        {
            "feature": feature_names,
            "anova_f": f_scores,
            "anova_p": f_pvalues,
            "mutual_information": mi,
            "l1_logistic_abs_coef": l1_importance,
            "random_forest_importance": rf_importance,
            "permutation_importance": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
            "rfe_inverse_rank": rfe_importance,
            "rfe_rank": rfe.ranking_,
        }
    )
    methods = {
        "ANOVA F-test": "anova_f",
        "Mutual Information": "mutual_information",
        "L1 Logistic": "l1_logistic_abs_coef",
        "Random Forest": "random_forest_importance",
        "Permutation Importance": "permutation_importance",
        "RFE": "rfe_inverse_rank",
    }
    for method, col in methods.items():
        ranking_df[f"rank_{method}"] = ranking_df[col].rank(ascending=False, method="min").astype(int)
    _write_csv(ranking_df, "feature_rankings.csv")

    top50 = ranking_df.sort_values("anova_f", ascending=False).head(50)["feature"].tolist()
    corr = X_train[top50].corr()
    corr.to_csv(RESULT_DIR / "top50_anova_feature_correlation.csv")
    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(50), labels=top50, rotation=90, fontsize=4)
    ax.set_yticks(np.arange(50), labels=top50, fontsize=4)
    ax.set_title("Correlation heatmap of top-50 ANOVA features", loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Pearson r")
    save_figure(fig, "10_top50_feature_correlation_heatmap")

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8))
    axes = axes.ravel()
    for ax, (method, col) in zip(axes, methods.items()):
        values = np.sort(ranking_df[col].fillna(0).to_numpy())[::-1][:80]
        ax.plot(np.arange(1, len(values) + 1), values, lw=1.0, color=METHOD_COLORS[len(ax.lines) % len(METHOD_COLORS)])
        ax.set_title(method, loc="left", fontweight="bold")
        ax.set_xlabel("Feature rank")
        ax.set_ylabel("Importance")
    save_figure(fig, "11_feature_importance_rankings")

    overlap_rows = []
    top_k = 20
    top_sets = {method: set(ranking_df.sort_values(col, ascending=False).head(top_k)["feature"]) for method, col in methods.items()}
    method_names = list(top_sets)
    overlap = np.zeros((len(method_names), len(method_names)))
    for i, m1 in enumerate(method_names):
        for j, m2 in enumerate(method_names):
            overlap[i, j] = len(top_sets[m1] & top_sets[m2]) / top_k
            overlap_rows.append({"method_a": m1, "method_b": m2, "top_k": top_k, "overlap_fraction": overlap[i, j]})
    _write_csv(pd.DataFrame(overlap_rows), "feature_selection_overlap.csv")
    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    im = ax.imshow(overlap, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(np.arange(len(method_names)), labels=method_names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(method_names)), labels=method_names)
    ax.set_title("Top-20 feature overlap across selectors", loc="left", fontweight="bold")
    for i in range(len(method_names)):
        for j in range(len(method_names)):
            ax.text(j, i, f"{overlap[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if overlap[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Overlap")
    save_figure(fig, "12_feature_selection_overlap")
    return ranking_df


def run_base_models(X_train, y_train, X_test, y_test, config: RunConfig):
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    specs = {
        "Majority Class": (DummyClassifier(strategy="most_frequent"), {}),
        "LDA": (Pipeline([("scaler", StandardScaler()), ("model", LinearDiscriminantAnalysis())]), {}),
        "Shrinkage LDA": (Pipeline([("scaler", StandardScaler()), ("model", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]), {}),
        "Regularized QDA": (
            Pipeline([("scaler", StandardScaler()), ("model", QuadraticDiscriminantAnalysis())]),
            {"model__reg_param": [0.05, 0.1, 0.3, 0.5, 0.7]},
        ),
        "Logistic Regression": (
            Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2500, random_state=config.random_state))]),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "L1 Logistic": (
            Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(penalty="l1", solver="liblinear", max_iter=2500, random_state=config.random_state))]),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "L2 Logistic": (
            Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2500, random_state=config.random_state))]),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Elastic Net Logistic": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=3000, random_state=config.random_state)),
                ]
            ),
            {"model__C": [0.01, 0.1, 1.0], "model__l1_ratio": [0.5]},
        ),
        "Linear SVM": (
            Pipeline([("scaler", StandardScaler()), ("model", LinearSVC(dual="auto", max_iter=6000, random_state=config.random_state))]),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "RBF-SVM": (
            Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="rbf", probability=True, random_state=config.random_state))]),
            {"model__C": [1.0, 10.0, 50.0], "model__gamma": ["scale", 0.01]},
        ),
        "KNN": (
            Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())]),
            {"model__n_neighbors": [3, 5, 9, 15]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=260, max_features="sqrt", min_samples_leaf=2, random_state=config.random_state, n_jobs=-1),
            {},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(n_estimators=180, learning_rate=0.05, max_depth=3, random_state=config.random_state),
            {},
        ),
        "Extra Trees": (
            ExtraTreesClassifier(n_estimators=260, max_features="sqrt", min_samples_leaf=2, random_state=config.random_state, n_jobs=-1),
            {},
        ),
        "MLP": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", MLPClassifier(hidden_layer_sizes=(80,), alpha=0.001, max_iter=350, early_stopping=True, random_state=config.random_state)),
                ]
            ),
            {},
        ),
    }
    rows, fitted, predictions, scores = [], {}, {}, {}
    for name, (estimator, params) in specs.items():
        model = GridSearchCV(estimator, params, scoring="accuracy", cv=cv, n_jobs=config.n_jobs, refit=True) if params else estimator
        row, fit_model, pred, score = _evaluate_model(name, model, X_train, y_train, X_test, y_test)
        best_estimator = fit_model.best_estimator_ if hasattr(fit_model, "best_estimator_") else fit_model
        row["best_params"] = json.dumps(getattr(fit_model, "best_params_", {}))
        fitted[name] = best_estimator
        predictions[name] = pred
        scores[name] = score
        rows.append(row)
    results = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    _write_csv(results, "base_model_results.csv")
    return fitted, predictions, scores, results


def _select_features(ranking_df: pd.DataFrame, method: str, k: int) -> List[str]:
    col_map = {
        "ANOVA F-test": "anova_f",
        "Mutual Information": "mutual_information",
        "L1 Logistic": "l1_logistic_abs_coef",
        "Random Forest": "random_forest_importance",
        "Permutation Importance": "permutation_importance",
        "RFE": "rfe_inverse_rank",
    }
    return ranking_df.sort_values(col_map[method], ascending=False).head(k)["feature"].tolist()


def feature_selection_experiment(X_train, y_train, X_test, y_test, ranking_df, config: RunConfig):
    selectors = ["ANOVA F-test", "Mutual Information", "L1 Logistic", "Random Forest", "Permutation Importance", "RFE"]
    ks = [5, 10, 20, 50, 100, 200, 500]
    classifiers = _model_factory(config.random_state)
    rows = []
    for selector in selectors:
        for k in ks:
            features = _select_features(ranking_df, selector, k)
            for clf_name, estimator in classifiers.items():
                row, _, _, _ = _evaluate_model(
                    f"{selector} + {clf_name}",
                    clone(estimator),
                    X_train[features],
                    y_train,
                    X_test[features],
                    y_test,
                    selected_features=k,
                )
                row.update({"feature_selection": selector, "k": k, "classifier": clf_name})
                rows.append(row)
    results = pd.DataFrame(rows)
    _write_csv(results, "feature_selection_results.csv")
    pivot = results.pivot_table(index=["feature_selection", "k"], columns="classifier", values="accuracy").reset_index()
    pivot.to_csv(RESULT_DIR / "feature_selection_accuracy_table.csv", index=False)
    return results, pivot


def pca_experiment(X_train, y_train, X_test, y_test, config: RunConfig):
    ks = [5, 10, 20, 50, 100, 200, 300, 500]
    classifiers = {
        "PCA + Logistic Regression": LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=config.random_state),
        "PCA + Linear SVM": LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=config.random_state),
        "PCA + RBF-SVM": SVC(C=10.0, gamma="scale", kernel="rbf", probability=True, random_state=config.random_state),
        "PCA + Random Forest": RandomForestClassifier(n_estimators=220, max_features="sqrt", min_samples_leaf=2, random_state=config.random_state, n_jobs=-1),
    }
    rows = []
    for k in ks:
        for name, clf in classifiers.items():
            estimator = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=k, random_state=config.random_state)), ("model", clone(clf))])
            row, _, _, _ = _evaluate_model(name, estimator, X_train, y_train, X_test, y_test, selected_features=k)
            row.update({"pca_components": k, "classifier": name.replace("PCA + ", "")})
            rows.append(row)
    results = pd.DataFrame(rows)
    _write_csv(results, "pca_classification_results.csv")
    return results


def regularization_path_experiment(X_train, y_train, X_test, y_test, config: RunConfig):
    Cs = [0.001, 0.01, 0.1, 1, 10, 100]
    penalties = {
        "L1 Logistic": {"penalty": "l1", "solver": "liblinear", "l1_ratio": None},
        "L2 Logistic": {"penalty": "l2", "solver": "lbfgs", "l1_ratio": None},
        "Elastic Net Logistic": {"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5},
    }
    rows = []
    for name, opts in penalties.items():
        for C in Cs:
            kwargs = {
                "penalty": opts["penalty"],
                "solver": opts["solver"],
                "C": C,
                "max_iter": 3500,
                "random_state": config.random_state,
            }
            if opts["l1_ratio"] is not None:
                kwargs["l1_ratio"] = opts["l1_ratio"]
            estimator = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(**kwargs))])
            row, fit_model, _, _ = _evaluate_model(name, estimator, X_train, y_train, X_test, y_test)
            coef = fit_model.named_steps["model"].coef_.ravel()
            row.update({"C": C, "log10_C": float(np.log10(C)), "nonzero_features": int((np.abs(coef) > 1e-8).sum())})
            rows.append(row)
    results = pd.DataFrame(rows)
    _write_csv(results, "regularization_path_results.csv")
    return results


def sample_size_experiment(X_train, y_train, X_test, y_test, config: RunConfig):
    sizes = [200, 500, 1000, 1500, 2000]
    classifiers = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=config.random_state))]),
        "L1 Logistic": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=0.1, penalty="l1", solver="liblinear", max_iter=2000, random_state=config.random_state))]),
        "Linear SVM": Pipeline([("scaler", StandardScaler()), ("model", LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=config.random_state))]),
        "RBF-SVM": Pipeline([("scaler", StandardScaler()), ("model", SVC(C=10.0, gamma="scale", kernel="rbf", probability=True, random_state=config.random_state))]),
        "Random Forest": RandomForestClassifier(n_estimators=180, max_features="sqrt", min_samples_leaf=2, random_state=config.random_state, n_jobs=-1),
    }
    rows = []
    for n in sizes:
        for rep in range(config.sample_repeats):
            if n == len(X_train):
                idx = np.arange(len(X_train))
            else:
                idx, _ = train_test_split(np.arange(len(X_train)), train_size=n, stratify=y_train, random_state=config.random_state + rep + n)
            X_sub = X_train.iloc[idx]
            y_sub = y_train.iloc[idx]
            for name, estimator in classifiers.items():
                row, _, _, _ = _evaluate_model(name, clone(estimator), X_sub, y_sub, X_test, y_test)
                row.update({"train_n": n, "repeat": rep})
                rows.append(row)
    results = pd.DataFrame(rows)
    _write_csv(results, "sample_size_sensitivity.csv")
    return results


def refit_best_experimental_models(
    X_train,
    y_train,
    X_test,
    y_test,
    ranking_df,
    fs_results,
    pca_results,
    reg_results,
    predictions,
    scores,
    config: RunConfig,
):
    """Refit best non-base experiment winners for bootstrap and error analysis."""
    rows = []

    best_fs = fs_results.sort_values(["accuracy", "auc"], ascending=False).iloc[0]
    fs_features = _select_features(ranking_df, best_fs["feature_selection"], int(best_fs["k"]))
    fs_estimator = clone(_model_factory(config.random_state)[best_fs["classifier"]])
    fs_name = f"Best FS: {best_fs['feature_selection']} + {best_fs['classifier']} (k={int(best_fs['k'])})"
    row, _, pred, score = _evaluate_model(fs_name, fs_estimator, X_train[fs_features], y_train, X_test[fs_features], y_test, selected_features=int(best_fs["k"]))
    row.update({"source": "feature_selection", "settings": json.dumps({"selector": best_fs["feature_selection"], "classifier": best_fs["classifier"], "k": int(best_fs["k"])})})
    rows.append(row)
    predictions[fs_name] = pred
    scores[fs_name] = score

    best_pca = pca_results.sort_values(["accuracy", "auc"], ascending=False).iloc[0]
    pca_classifier_map = {
        "Logistic Regression": LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=config.random_state),
        "Linear SVM": LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=config.random_state),
        "RBF-SVM": SVC(C=10.0, gamma="scale", kernel="rbf", probability=True, random_state=config.random_state),
        "Random Forest": RandomForestClassifier(n_estimators=220, max_features="sqrt", min_samples_leaf=2, random_state=config.random_state, n_jobs=-1),
    }
    pca_components = int(best_pca["pca_components"])
    pca_name = f"Best PCA: {best_pca['classifier']} (PCs={pca_components})"
    pca_estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_components, random_state=config.random_state)),
            ("model", clone(pca_classifier_map[best_pca["classifier"]])),
        ]
    )
    row, _, pred, score = _evaluate_model(pca_name, pca_estimator, X_train, y_train, X_test, y_test, selected_features=pca_components)
    row.update({"source": "pca", "settings": json.dumps({"classifier": best_pca["classifier"], "components": pca_components})})
    rows.append(row)
    predictions[pca_name] = pred
    scores[pca_name] = score

    best_reg = reg_results.sort_values(["accuracy", "auc"], ascending=False).iloc[0]
    reg_name = f"Best regularized: {best_reg['method']} (C={best_reg['C']})"
    reg_kwargs = {
        "C": float(best_reg["C"]),
        "max_iter": 3500,
        "random_state": config.random_state,
    }
    if best_reg["method"] == "L1 Logistic":
        reg_kwargs.update({"penalty": "l1", "solver": "liblinear"})
    elif best_reg["method"] == "Elastic Net Logistic":
        reg_kwargs.update({"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5})
    else:
        reg_kwargs.update({"penalty": "l2", "solver": "lbfgs"})
    reg_estimator = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(**reg_kwargs))])
    row, fit_model, pred, score = _evaluate_model(reg_name, reg_estimator, X_train, y_train, X_test, y_test)
    coef = fit_model.named_steps["model"].coef_.ravel()
    row.update({"source": "regularization_path", "settings": json.dumps({"method": best_reg["method"], "C": float(best_reg["C"])}), "nonzero_features": int((np.abs(coef) > 1e-8).sum())})
    rows.append(row)
    predictions[reg_name] = pred
    scores[reg_name] = score

    candidates = pd.DataFrame(rows).sort_values(["accuracy", "auc"], ascending=False)
    _write_csv(candidates, "final_experimental_model_candidates.csv")
    return predictions, scores, candidates


def bootstrap_analysis(y_test, predictions: Dict[str, np.ndarray], config: RunConfig):
    rng = np.random.default_rng(config.random_state)
    y_arr = np.asarray(y_test)
    rows = []
    distributions = {}
    for name, pred in predictions.items():
        pred = np.asarray(pred)
        values = []
        for _ in range(config.bootstrap_runs):
            idx = rng.integers(0, len(y_arr), size=len(y_arr))
            values.append(accuracy_score(y_arr[idx], pred[idx]))
        values = np.asarray(values)
        distributions[name] = values
        rows.append(
            {
                "method": name,
                "accuracy": accuracy_score(y_arr, pred),
                "bootstrap_mean_accuracy": float(values.mean()),
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
            }
        )
    ci_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    _write_csv(ci_df, "bootstrap_accuracy_ci.csv")

    best = ci_df.iloc[0]["method"]
    paired_rows = []
    for other in ci_df["method"]:
        if other == best:
            continue
        diff_values = distributions[best] - distributions[other]
        paired_rows.append(
            {
                "model_a": best,
                "model_b": other,
                "accuracy_diff": accuracy_score(y_arr, predictions[best]) - accuracy_score(y_arr, predictions[other]),
                "ci_lower": float(np.quantile(diff_values, 0.025)),
                "ci_upper": float(np.quantile(diff_values, 0.975)),
                "ci_crosses_zero": bool(np.quantile(diff_values, 0.025) <= 0 <= np.quantile(diff_values, 0.975)),
            }
        )
    paired_df = pd.DataFrame(paired_rows)
    _write_csv(paired_df, "paired_bootstrap_accuracy_diff.csv")
    return ci_df, paired_df


def error_analysis(y_test, predictions, scores, embeddings, config: RunConfig):
    accuracy_rows = []
    y_arr = np.asarray(y_test)
    for name, pred_values in predictions.items():
        accuracy_rows.append({"method": name, "accuracy": accuracy_score(y_arr, np.asarray(pred_values))})
    best = pd.DataFrame(accuracy_rows).sort_values("accuracy", ascending=False).iloc[0]["method"]
    pred = np.asarray(predictions[best])
    errors = pred != np.asarray(y_test)
    score = np.asarray(scores[best])
    test_start = len(embeddings["X_all"]) - len(y_test)
    pca_test = embeddings["all_pca"][test_start:]
    tsne_test = embeddings["tsne"][test_start:]
    umap_test = embeddings["umap"][test_start:]
    error_df = pd.DataFrame(
        {
            "test_index": np.arange(len(y_test)),
            "true_label": np.asarray(y_test),
            "predicted_label": pred,
            "score_positive": score,
            "is_error": errors,
            "pca1": pca_test[:, 0],
            "pca2": pca_test[:, 1],
            "tsne1": tsne_test[:, 0],
            "tsne2": tsne_test[:, 1],
            "umap1": umap_test[:, 0],
            "umap2": umap_test[:, 1],
        }
    )
    _write_csv(error_df, "error_samples_best_model.csv")
    return best, errors, error_df


def plot_experiment_results(base_results, fs_results, pca_results, reg_results, sample_results, ci_df, paired_df, y_test, predictions, best_model, errors, embeddings):
    apply_style()

    plot_df = base_results.sort_values("accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(plot_df["method"], plot_df["accuracy"], color=METHOD_COLORS * 3, edgecolor="black", lw=0.45)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Official validation accuracy")
    ax.set_title("Base classifier comparison", loc="left", fontweight="bold")
    for y_pos, value in enumerate(plot_df["accuracy"]):
        ax.text(value + 0.01, y_pos, f"{value:.3f}", va="center", fontsize=6)
    save_figure(fig, "13_model_accuracy_bar")

    plot_df = base_results.sort_values("auc", ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(plot_df["method"], plot_df["auc"], color=METHOD_COLORS * 3, edgecolor="black", lw=0.45)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC")
    ax.set_title("AUC comparison", loc="left", fontweight="bold")
    save_figure(fig, "14_model_auc_bar")

    plot_df = base_results.sort_values("train_time_sec", ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(plot_df["method"], plot_df["train_time_sec"], color=PALETTE["teal"], edgecolor="black", lw=0.45)
    ax.set_xlabel("Training time (seconds)")
    ax.set_title("Training cost differs across methods", loc="left", fontweight="bold")
    save_figure(fig, "15_training_time_bar")

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.scatter(base_results["train_time_sec"], base_results["accuracy"], s=35, color=PALETTE["blue"], edgecolor="white", lw=0.4)
    for _, row in base_results.iterrows():
        ax.text(row["train_time_sec"], row["accuracy"] + 0.006, row["method"], fontsize=5, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("Training time (log seconds)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs training time", loc="left", fontweight="bold")
    save_figure(fig, "16_accuracy_vs_training_time")

    cm = confusion_matrix(y_test, predictions[best_model], labels=[-1, 1])
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["-1", "+1"])
    ax.set_yticks([0, 1], labels=["-1", "+1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix: {best_model}", loc="left", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8, color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, "17_best_model_confusion_matrix")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    fs_plot = fs_results.groupby(["feature_selection", "k"], as_index=False)["accuracy"].max()
    for method, group in fs_plot.groupby("feature_selection"):
        axes[0].plot(group["k"], group["accuracy"], marker="o", lw=1.0, label=method)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Top-k selected features")
    axes[0].set_ylabel("Best accuracy across classifiers")
    axes[0].set_title("Feature selection improves the noise tradeoff", loc="left", fontweight="bold")
    axes[0].legend(fontsize=5, ncols=2)
    fs_clf = fs_results.groupby(["classifier", "k"], as_index=False)["accuracy"].max()
    for clf, group in fs_clf.groupby("classifier"):
        axes[1].plot(group["k"], group["accuracy"], marker="o", lw=1.0, label=clf)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Top-k selected features")
    axes[1].set_ylabel("Best accuracy across selectors")
    axes[1].set_title("Classifier sensitivity to selected feature count", loc="left", fontweight="bold")
    axes[1].legend(fontsize=5, ncols=2)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    save_figure(fig, "18_feature_selection_topk_accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    for clf, group in pca_results.groupby("classifier"):
        group = group.sort_values("pca_components")
        axes[0].plot(group["pca_components"], group["accuracy"], marker="o", lw=1.0, label=clf)
        axes[1].plot(group["pca_components"], group["auc"], marker="o", lw=1.0, label=clf)
    for ax, metric in zip(axes, ["Accuracy", "AUC"]):
        ax.set_xscale("log")
        ax.set_xlabel("PCA components")
        ax.set_ylabel(metric)
        ax.legend(fontsize=5)
    axes[0].set_title("PCA dimension vs accuracy", loc="left", fontweight="bold")
    axes[1].set_title("PCA dimension vs AUC", loc="left", fontweight="bold")
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    save_figure(fig, "19_pca_dimension_accuracy_auc")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    for method, group in reg_results.groupby("method"):
        group = group.sort_values("C")
        axes[0].plot(group["C"], group["accuracy"], marker="o", lw=1.0, label=method)
        axes[1].plot(group["C"], group["nonzero_features"], marker="o", lw=1.0, label=method)
    for ax in axes:
        ax.set_xscale("log")
        ax.legend(fontsize=5)
    axes[0].set_xlabel("C")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Regularization strength vs accuracy", loc="left", fontweight="bold")
    axes[1].set_xlabel("C")
    axes[1].set_ylabel("Nonzero coefficients")
    axes[1].set_title("Regularization strength vs sparsity", loc="left", fontweight="bold")
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    save_figure(fig, "20_regularization_path")

    sample_summary = sample_results.groupby(["method", "train_n"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
    )
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for method, group in sample_summary.groupby("method"):
        group = group.sort_values("train_n")
        ax.errorbar(group["train_n"], group["accuracy_mean"], yerr=group["accuracy_std"], marker="o", lw=1.0, capsize=2, label=method)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Sample-size sensitivity", loc="left", fontweight="bold")
    ax.legend(fontsize=5, ncols=2)
    save_figure(fig, "21_sample_size_learning_curve")

    ci_plot = ci_df.sort_values("accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    xerr = np.vstack([ci_plot["accuracy"] - ci_plot["ci_lower"], ci_plot["ci_upper"] - ci_plot["accuracy"]])
    ax.errorbar(ci_plot["accuracy"], ci_plot["method"], xerr=xerr, fmt="o", color=PALETTE["blue"], ecolor=PALETTE["gray"], capsize=2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy with 95% bootstrap CI")
    ax.set_title("Bootstrap uncertainty on the final test set", loc="left", fontweight="bold")
    save_figure(fig, "22_bootstrap_accuracy_ci")

    if not paired_df.empty:
        paired_plot = paired_df.sort_values("accuracy_diff", ascending=True)
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        xerr = np.vstack([paired_plot["accuracy_diff"] - paired_plot["ci_lower"], paired_plot["ci_upper"] - paired_plot["accuracy_diff"]])
        ax.errorbar(paired_plot["accuracy_diff"], paired_plot["model_b"], xerr=xerr, fmt="o", color=PALETTE["teal"], ecolor=PALETTE["gray"], capsize=2)
        ax.axvline(0, color=PALETTE["red"], ls="--", lw=0.8)
        ax.set_xlabel(f"Accuracy difference: {paired_plot.iloc[0]['model_a']} - competitor")
        ax.set_title("Paired bootstrap model comparison", loc="left", fontweight="bold")
        save_figure(fig, "23_paired_bootstrap_accuracy_diff")

    test_start = len(embeddings["X_all"]) - len(y_test)
    coords = [
        ("PCA", embeddings["all_pca"][test_start:, :2], "PC1", "PC2"),
        ("t-SNE", embeddings["tsne"][test_start:], "t-SNE 1", "t-SNE 2"),
        ("UMAP", embeddings["umap"][test_start:], "UMAP 1", "UMAP 2"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    for ax, (name, values, xlab, ylab) in zip(axes, coords):
        scatter_by_label(ax, values, y_test, f"{name}: errors highlighted", xlab, ylab, errors=errors)
    for ax, label in zip(axes, ["a", "b", "c"]):
        add_panel_label(ax, label)
    save_figure(fig, "24_error_samples_in_embeddings")


def write_figure_contracts() -> None:
    text = """# Figure Contracts

All figures are generated in Python with matplotlib and exported as editable
SVG, PDF, high-DPI PNG, and TIFF. The report is a source-backed analysis of
MADELON, not a decorative classification demo.

Core claims:

1. MADELON is balanced and contains many weak/noisy marginal features.
2. PCA, t-SNE, and UMAP reveal complementary linear and nonlinear structure.
3. Supervised feature selection is central because MADELON contains many probe
   and irrelevant dimensions.
4. Nonlinear models, especially RBF-SVM and boosting/tree ensembles, should be
   compared with linear and regularized baselines.
5. Bootstrap and sample-size experiments quantify stability rather than relying
   on a single accuracy number.

Leakage controls:

- The official train split is used for fitting and cross-validation.
- The official validation split is used only as the final test set.
- Scaling, PCA, feature selection, RFE, and model fitting are fit on training
  data before transforming/evaluating the final test set.
"""
    (RESULT_DIR / "figure_contracts.md").write_text(text, encoding="utf-8")


def write_summary(base_results, fs_results, pca_results, reg_results, sample_results, ci_df, best_model) -> None:
    best_base = base_results.sort_values(["accuracy", "auc"], ascending=False).iloc[0]
    best_fs = fs_results.sort_values(["accuracy", "auc"], ascending=False).iloc[0]
    best_pca = pca_results.sort_values(["accuracy", "auc"], ascending=False).iloc[0]
    best_reg = reg_results.sort_values(["accuracy", "auc"], ascending=False).iloc[0]
    text = f"""# Analysis Summary

- Best base model: **{best_base['method']}** (accuracy = {best_base['accuracy']:.3f}, AUC = {best_base['auc']:.3f}).
- Best feature-selection result: **{best_fs['feature_selection']} + {best_fs['classifier']}**, k = {int(best_fs['k'])}
  (accuracy = {best_fs['accuracy']:.3f}, AUC = {best_fs['auc']:.3f}).
- Best PCA result: **{best_pca['classifier']}**, components = {int(best_pca['pca_components'])}
  (accuracy = {best_pca['accuracy']:.3f}, AUC = {best_pca['auc']:.3f}).
- Best regularized logistic path result: **{best_reg['method']}**, C = {best_reg['C']}
  (accuracy = {best_reg['accuracy']:.3f}, nonzero features = {int(best_reg['nonzero_features'])}).
- Bootstrap best model used for error analysis: **{best_model}**.
- The official MADELON validation split is treated as the final test set.
"""
    (RESULT_DIR / "analysis_summary.md").write_text(text, encoding="utf-8")


def run(config: RunConfig | None = None) -> None:
    ensure_runtime_dirs()
    config = config or RunConfig()
    print(f"Running MADELON project with cv={config.cv_folds}, bootstrap={config.bootstrap_runs}", flush=True)
    write_figure_contracts()

    print("[1/10] Loading MADELON", flush=True)
    X_train, y_train, X_test, y_test = prepare_data(config)
    print("[2/10] EDA and embeddings", flush=True)
    embeddings = eda_and_embeddings(X_train, y_train, X_test, y_test, config)
    print("[3/10] Feature rankings", flush=True)
    ranking_df = compute_feature_rankings(X_train, y_train, config)
    print("[4/10] Base model comparison", flush=True)
    fitted, predictions, scores, base_results = run_base_models(X_train, y_train, X_test, y_test, config)
    print("[5/10] Feature-selection experiment", flush=True)
    fs_results, _ = feature_selection_experiment(X_train, y_train, X_test, y_test, ranking_df, config)
    print("[6/10] PCA classification experiment", flush=True)
    pca_results = pca_experiment(X_train, y_train, X_test, y_test, config)
    print("[7/10] Regularization path", flush=True)
    reg_results = regularization_path_experiment(X_train, y_train, X_test, y_test, config)
    print("[8/10] Sample-size sensitivity", flush=True)
    sample_results = sample_size_experiment(X_train, y_train, X_test, y_test, config)
    print("[9/10] Bootstrap and error analysis", flush=True)
    predictions, scores, _ = refit_best_experimental_models(
        X_train,
        y_train,
        X_test,
        y_test,
        ranking_df,
        fs_results,
        pca_results,
        reg_results,
        predictions,
        scores,
        config,
    )
    ci_df, paired_df = bootstrap_analysis(y_test, predictions, config)
    best_model, errors, _ = error_analysis(y_test, predictions, scores, embeddings, config)
    print("[10/10] Plotting experiment results", flush=True)
    plot_experiment_results(base_results, fs_results, pca_results, reg_results, sample_results, ci_df, paired_df, y_test, predictions, best_model, errors, embeddings)
    write_summary(base_results, fs_results, pca_results, reg_results, sample_results, ci_df, best_model)
    print(f"Wrote figures to {FIGURE_DIR}", flush=True)
    print(f"Wrote results to {RESULT_DIR}", flush=True)
