#!/usr/bin/env python3
"""
WUSTL-IIoT-2021 — standalone post-training analysis (IEEE-style figures & tables).

Run after your training notebook, or use --fit to train + analyze in one pass.
Outputs: publication figures (no axis grids), CSV/LaTeX tables, optional model cache.

Example:
  python wustl_iiot_post_analysis.py --fit
  python wustl_iiot_post_analysis.py   # uses cache from previous --fit
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    cohen_kappa_score,
    det_curve,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Publication style — no grid on plots (IEEE-friendly)
# ---------------------------------------------------------------------------
def _set_pub_style() -> None:
    sns.set_theme(style="white", rc={"axes.grid": False})
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "lines.linewidth": 1.8,
        }
    )


COLORS = sns.color_palette("colorblind", as_cmap=False)


# ---------------------------------------------------------------------------
# Config (aligned with wustl_iiot_intrusion_detection.ipynb)
# ---------------------------------------------------------------------------
# Repo root = parent of "Source Code" folder in this project layout.
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "dataset" / "wustl_iiot_2021.csv"
LEAKAGE_COLS = ["StartTime", "LastTime", "SrcAddr", "DstAddr", "sIpId", "dIpId"]
RANDOM_STATE = 42
TUNE_SAMPLE_SIZE = 150_000
FINAL_TRAIN_CAP = 500_000
KNN_TRAIN_CAP = 80_000
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def stratified_cap(
    X_df: pd.DataFrame, y_s: pd.Series, max_rows: int, seed: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.Series]:
    if len(X_df) <= max_rows:
        return X_df, y_s
    X_sub, _, y_sub, _ = train_test_split(
        X_df, y_s, train_size=max_rows, stratify=y_s, random_state=seed
    )
    return X_sub, y_sub


def load_data(path: Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Returns X (features), y (binary Target), traffic (full Traffic column aligned with X)."""
    df = pd.read_csv(path, low_memory=False)
    for c in LEAKAGE_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")
    df_model = df.drop(columns=LEAKAGE_COLS).copy()
    traffic = df_model["Traffic"].copy()
    feature_cols = [c for c in df_model.columns if c not in ("Traffic", "Target")]
    X = df_model[feature_cols].copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col], _ = X[col].factorize()
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.fillna(0.0)
    y = df_model["Target"].astype(int)
    chk = (traffic.str.lower() != "normal").astype(int)
    assert (y.values == chk.values).all()
    return X, y, traffic


def build_pipelines() -> Dict[str, Any]:
    """Unfitted estimators + metadata for training data size."""
    pipe_lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    solver="saga",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipe_nb = Pipeline(
        [("scaler", StandardScaler()), ("clf", GaussianNB())]
    )
    pipe_knn = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15, n_jobs=1)),
        ]
    )
    pipe_dt = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                DecisionTreeClassifier(
                    class_weight="balanced",
                    max_depth=24,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipe_rf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    n_estimators=200,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipe_svc = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LinearSVC(
                    class_weight="balanced",
                    dual=False,
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return {
        "LogisticRegression": {"est": pipe_lr, "train": "full"},
        "GaussianNB": {"est": pipe_nb, "train": "full"},
        "kNN": {"est": pipe_knn, "train": "knn"},
        "DecisionTree": {"est": pipe_dt, "train": "full"},
        "RandomForest": {"est": pipe_rf, "train": "full"},
        "LinearSVC": {"est": pipe_svc, "train": "full"},
    }


def tune_random_forest(
    X_tune: pd.DataFrame, y_tune: pd.Series, seed: int
) -> RandomizedSearchCV:
    pipe_rf = build_pipelines()["RandomForest"]["est"]
    param_dist = {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [None, 24, 32],
        "clf__min_samples_leaf": [1, 2, 4],
    }
    search = RandomizedSearchCV(
        pipe_rf,
        param_distributions=param_dist,
        n_iter=12,
        cv=CV,
        scoring="average_precision",
        random_state=seed,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_tune, y_tune)
    return search


def fit_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_tune: pd.DataFrame,
    y_tune: pd.Series,
) -> Dict[str, Any]:
    """Returns dict with 'models', 'rf_search', 'X_fit', 'y_fit', 'X_knn_fit', 'y_knn_fit'."""
    specs = build_pipelines()
    rf_search = tune_random_forest(X_tune, y_tune, RANDOM_STATE)
    best_rf = rf_search.best_estimator_

    ensemble = VotingClassifier(
        estimators=[
            ("lr", clone(specs["LogisticRegression"]["est"])),
            ("rf", clone(best_rf)),
            ("nb", clone(specs["GaussianNB"]["est"])),
        ],
        voting="soft",
        n_jobs=1,
    )

    X_fit, y_fit = stratified_cap(X_train, y_train, FINAL_TRAIN_CAP)
    X_knn_fit, y_knn_fit = stratified_cap(X_train, y_train, KNN_TRAIN_CAP)

    models: Dict[str, Any] = {
        "LogisticRegression": clone(specs["LogisticRegression"]["est"]),
        "GaussianNB": clone(specs["GaussianNB"]["est"]),
        "kNN": clone(specs["kNN"]["est"]),
        "DecisionTree": clone(specs["DecisionTree"]["est"]),
        "RandomForest (tuned)": clone(best_rf),
        "LinearSVC": clone(specs["LinearSVC"]["est"]),
        "Voting LR+RF+NB": clone(ensemble),
    }

    print("Fitting models on capped training data…")
    for name, est in models.items():
        if name == "kNN":
            est.fit(X_knn_fit, y_knn_fit)
        else:
            est.fit(X_fit, y_fit)
        print(f"  [ok] {name}")

    return {
        "models": models,
        "rf_search": rf_search,
        "X_fit": X_fit,
        "y_fit": y_fit,
        "X_knn_fit": X_knn_fit,
        "y_knn_fit": y_knn_fit,
        "feature_names": list(X_train.columns),
    }


def scores_and_pred(
    model: Any, X_te: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict_proba"):
        s = model.predict_proba(X_te)[:, 1]
    else:
        s = model.decision_function(X_te)
    y_hat = model.predict(X_te)
    return s, y_hat


def mcnemar_midp(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Tuple[float, float]:
    """McNemar test (mid-p optional — here classic chi2). Returns statistic, p-value."""
    b = np.sum((pred_a == y_true) & (pred_b != y_true))
    c = np.sum((pred_a != y_true) & (pred_b == y_true))
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(stats.chi2.sf(chi2, df=1))
    return float(chi2), p


def bootstrap_pr_auc(
    y_true: np.ndarray, y_score: np.ndarray, n_resamples: int = 400, seed: int = RANDOM_STATE
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)

    def stat(ix: np.ndarray) -> float:
        return average_precision_score(y_true[ix], y_score[ix])

    n = len(y_true)
    idx = rng.integers(0, n, size=(n_resamples, n))
    vals = np.array([stat(idx[i]) for i in range(n_resamples)])
    point = average_precision_score(y_true, y_score)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def force_single_thread(est: Any) -> None:
    """Force nested estimators onto one worker so Windows joblib does not spawn processes."""
    if isinstance(est, Pipeline):
        for _, step in est.steps:
            force_single_thread(step)
    elif isinstance(est, VotingClassifier):
        try:
            est.n_jobs = 1
        except Exception:
            pass
        for _, sub_est in est.estimators:
            force_single_thread(sub_est)
        if hasattr(est, "estimators_"):
            for sub_est in est.estimators_:
                force_single_thread(sub_est)

    if hasattr(est, "n_jobs"):
        try:
            est.n_jobs = 1
        except Exception:
            pass
        try:
            if hasattr(est, "set_params") and "n_jobs" in est.get_params(deep=False):
                est.set_params(n_jobs=1)
        except Exception:
            pass


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="WUSTL-IIoT post-training analysis")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("wustl_analysis_output"))
    parser.add_argument("--fit", action="store_true", help="Train models (otherwise load cache)")
    parser.add_argument("--cache-dir", type=Path, default=Path("wustl_analysis_cache"))
    args = parser.parse_args()

    _set_pub_style()
    out_dir: Path = args.output_dir
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    cache_dir: Path = args.cache_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / "analysis_bundle.joblib"

    if not args.data_path.exists():
        print(f"Data not found: {args.data_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading data…")
    X, y, traffic = load_data(args.data_path)
    # ------------------------------------------------------------------
    # Figure 0: dataset overview — class balance + attack-family skew
    # ------------------------------------------------------------------
    class_counts = pd.Series(
        {"Normal": int((y == 0).sum()), "Attack": int((y == 1).sum())}
    )
    class_pct = class_counts / class_counts.sum() * 100.0
    attack_counts = traffic[y == 1].value_counts()
    top_n_attack = 10
    if len(attack_counts) > top_n_attack:
        attack_plot = attack_counts.head(top_n_attack).copy()
        other_attacks = int(attack_counts.iloc[top_n_attack:].sum())
        if other_attacks > 0:
            attack_plot.loc["Other attack families"] = other_attacks
    else:
        attack_plot = attack_counts.copy()
    attack_plot = attack_plot.sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), constrained_layout=True)
    class_colors = [COLORS[0], COLORS[3]]
    bars = axes[0].bar(class_counts.index, class_counts.values, color=class_colors, width=0.55)
    axes[0].set_title("Binary class balance")
    axes[0].set_ylabel("Flow count")
    axes[0].tick_params(axis="x", labelsize=11)
    axes[0].tick_params(axis="y", labelsize=10)
    axes[0].set_ylim(0, class_counts.max() * 1.25)
    axes[0].bar_label(
        bars,
        labels=[f"{int(v):,}\n({p:.1f}%)" for v, p in zip(class_counts.values, class_pct.values)],
        padding=4,
        fontsize=10,
    )

    attack_y = np.arange(len(attack_plot))
    attack_bars = axes[1].barh(
        attack_y,
        attack_plot.values,
        color=sns.color_palette("Blues", n_colors=len(attack_plot) + 3)[2:],
        height=0.7,
    )
    axes[1].set_yticks(attack_y)
    axes[1].set_yticklabels([textwrap.fill(str(x), width=18) for x in attack_plot.index], fontsize=9)
    axes[1].set_xlabel("Attack-flow count")
    axes[1].set_title("Attack-family skew within the positive class")
    axes[1].tick_params(axis="x", labelsize=10)
    axes[1].bar_label(
        attack_bars,
        labels=[f"{int(v):,}" for v in attack_plot.values],
        padding=3,
        fontsize=9,
    )
    fig.savefig(fig_dir / "fig00_dataset_overview.pdf")
    fig.savefig(fig_dir / "fig00_dataset_overview.png")
    plt.close(fig)

    X_train, X_test, y_train, y_test, tr_train, tr_test = train_test_split(
        X, y, traffic, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    if len(X_train) > TUNE_SAMPLE_SIZE:
        X_tune, _, y_tune, _ = train_test_split(
            X_train, y_train, train_size=TUNE_SAMPLE_SIZE, stratify=y_train, random_state=RANDOM_STATE
        )
    else:
        X_tune, y_tune = X_train, y_train

    if args.fit or not cache_file.exists():
        print("Training / tuning models (this may take a while)…")
        bundle = fit_all_models(X_train, y_train, X_tune, y_tune)
        bundle["X_train_shape"] = X_train.shape
        bundle["y_test"] = y_test
        bundle["traffic_test"] = tr_test.reset_index(drop=True)
        joblib.dump(bundle, cache_file)
        print(f"Saved cache: {cache_file}")
    else:
        print(f"Loading cached models: {cache_file}")
        bundle = joblib.load(cache_file)

    models: Dict[str, Any] = bundle["models"]
    feature_names: List[str] = bundle["feature_names"]
    for model in models.values():
        force_single_thread(model)

    # --- Collect scores & preds on test ---
    scores: Dict[str, np.ndarray] = {}
    preds: Dict[str, np.ndarray] = {}
    for name, m in models.items():
        s, p = scores_and_pred(m, X_test)
        scores[name] = s
        preds[name] = p

    y_test_arr = y_test.values
    traffic_test = bundle["traffic_test"]

    # ------------------------------------------------------------------
    # Table 0: dataset summary
    # ------------------------------------------------------------------
    dataset_summary = pd.DataFrame(
        [
            {"metric": "Total flows", "value": int(len(y))},
            {"metric": "Normal flows", "value": int((y == 0).sum())},
            {"metric": "Attack flows", "value": int((y == 1).sum())},
            {"metric": "Attack rate (%)", "value": float(y.mean() * 100.0)},
            {"metric": "Distinct attack families", "value": int(attack_counts.shape[0])},
        ]
    )
    dataset_summary.to_csv(tab_dir / "table0_dataset_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Table 1: scalar metrics (all models)
    # ------------------------------------------------------------------
    rows = []
    for name in models:
        s, p = scores[name], preds[name]
        rows.append(
            {
                "model": name,
                "PR_AUC": average_precision_score(y_test_arr, s),
                "ROC_AUC": roc_auc_score(y_test_arr, s),
                "F1_default": f1_score(y_test_arr, p, pos_label=1),
                "MCC": matthews_corrcoef(y_test_arr, p),
                "Precision": precision_score(y_test_arr, p, pos_label=1, zero_division=0),
                "Recall": recall_score(y_test_arr, p, pos_label=1, zero_division=0),
            }
        )
    df_metrics = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False).reset_index(drop=True)
    df_metrics.to_csv(tab_dir / "table1_scalar_metrics.csv", index=False)
    df_metrics.round(4).to_latex(tab_dir / "table1_scalar_metrics.tex", index=False, float_format="%.4f")
    print("\nTable 1 (top by PR-AUC):\n", df_metrics.head(8).to_string(index=False))

    # ------------------------------------------------------------------
    # Figure 1–2: ROC & PR overlays (primary comparison)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.9, 6.1))
    for i, name in enumerate(df_metrics["model"]):
        s = scores[name]
        fpr, tpr, _ = roc_curve(y_test_arr, s)
        auc_v = roc_auc_score(y_test_arr, s)
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], lw=1.6, label=f"{name} ({auc_v:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.45)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_roc_overlay.pdf")
    fig.savefig(fig_dir / "fig01_roc_overlay.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.9, 6.1))
    for i, name in enumerate(df_metrics["model"]):
        s = scores[name]
        prec, rec, _ = precision_recall_curve(y_test_arr, s)
        ap = average_precision_score(y_test_arr, s)
        ax.plot(rec, prec, color=COLORS[i % len(COLORS)], lw=1.6, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall (attack)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curves (test set)")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_pr_overlay.pdf")
    fig.savefig(fig_dir / "fig02_pr_overlay.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: DET curves (error trade-off, common in detection literature)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 5.9))
    for i, name in enumerate(df_metrics["model"][:6]):  # top 6 by table order — already sorted
        s = scores[name]
        fpr_det, fnr_det, _ = det_curve(y_test_arr, s)
        ax.plot(fpr_det, fnr_det, color=COLORS[i % len(COLORS)], lw=1.6, label=name)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("False negative rate")
    ax.set_title("DET curves (lower-left is better)")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_det_curves.pdf")
    fig.savefig(fig_dir / "fig03_det_curves.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: Threshold analysis (F1, precision, recall vs threshold) — probabilistic models
    # ------------------------------------------------------------------
    prob_models = [n for n in models if hasattr(models[n], "predict_proba")]
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2), sharex=True)
    thresholds = np.linspace(0.01, 0.99, 120)
    for ax, title, metric_fn in zip(
        axes,
        ["Precision", "Recall", "F1 (attack)"],
        [
            lambda t, s: precision_score(y_test_arr, (s >= t).astype(int), pos_label=1, zero_division=0),
            lambda t, s: recall_score(y_test_arr, (s >= t).astype(int), pos_label=1, zero_division=0),
            lambda t, s: f1_score(y_test_arr, (s >= t).astype(int), pos_label=1, zero_division=0),
        ],
    ):
        for j, name in enumerate(prob_models[:5]):
            s = scores[name]
            ys = [metric_fn(t, s) for t in thresholds]
            ax.plot(thresholds, ys, color=COLORS[j % len(COLORS)], lw=1.4, label=name)
        ax.set_title(title, pad=8)
        ax.set_xlabel("Threshold (probability)")
        ax.set_xlim(0, 1)
        ax.tick_params(axis="both", labelsize=10)
    axes[0].legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle("Operating-point sensitivity (probabilistic models)", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / "fig04_threshold_sweep.pdf")
    fig.savefig(fig_dir / "fig04_threshold_sweep.png")
    plt.close(fig)

    # Best F1 threshold table (prob models)
    thr_rows = []
    for name in prob_models:
        s = scores[name]
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.05, 0.95, 180):
            f1v = f1_score(y_test_arr, (s >= t).astype(int), pos_label=1, zero_division=0)
            if f1v > best_f1:
                best_f1, best_t = f1v, t
        thr_rows.append({"model": name, "best_threshold_f1": best_t, "F1_at_best": best_f1})
    pd.DataFrame(thr_rows).to_csv(tab_dir / "table_threshold_f1_opt.csv", index=False)

    # ------------------------------------------------------------------
    # Figure 5: Reliability diagrams (calibration) — test-set bins
    # ------------------------------------------------------------------
    n_cal = min(len(y_test_arr), 80_000)
    rng = np.random.default_rng(RANDOM_STATE + 3)
    ix = rng.choice(np.arange(len(y_test_arr)), size=n_cal, replace=False)
    y_c = y_test_arr[ix]

    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.45)
    for j, name in enumerate(prob_models[:4]):
        s = scores[name][ix]
        prob_true, prob_pred = calibration_curve(y_c, s, n_bins=12, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", ms=3, color=COLORS[j % len(COLORS)], label=name)
    ax.set_xlabel("Mean predicted probability (bins)")
    ax.set_ylabel("Fraction of positives (empirical)")
    ax.set_title("Reliability diagrams (test subsample)")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig05_calibration.pdf")
    fig.savefig(fig_dir / "fig05_calibration.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 6: Score separation — KDE for champion model
    # ------------------------------------------------------------------
    champ = df_metrics.iloc[0]["model"]
    s_ch = scores[champ]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sns.kdeplot(x=s_ch[y_test_arr == 0], ax=ax, fill=True, alpha=0.35, label="Normal", color=COLORS[0])
    sns.kdeplot(x=s_ch[y_test_arr == 1], ax=ax, fill=True, alpha=0.35, label="Attack", color=COLORS[3])
    ax.set_xlabel(f"Score ({champ})")
    ax.set_title("Score distributions on test set (kernel density)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig06_score_separation_kde.pdf")
    fig.savefig(fig_dir / "fig06_score_separation_kde.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 7: Per–attack-type recall (Traffic != normal)
    # ------------------------------------------------------------------
    atk_mask = y_test_arr == 1
    atk_types = traffic_test.values[atk_mask]
    unique_atk = pd.Series(atk_types).value_counts().index.tolist()
    rec_rows = []
    heat = []
    model_subset = [m for m in df_metrics["model"].head(5)]
    for mname in model_subset:
        row = []
        p = preds[mname][atk_mask]
        for at in unique_atk:
            sub = atk_types == at
            if sub.sum() == 0:
                r = np.nan
            else:
                r = (p[sub] == 1).mean()
            row.append(r)
            rec_rows.append({"model": mname, "attack_type": at, "recall_on_slice": r})
        heat.append(row)
    pd.DataFrame(rec_rows).to_csv(tab_dir / "table_per_attack_recall.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    H = np.array(heat)
    im = ax.imshow(H, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(model_subset)))
    ax.set_yticklabels(model_subset, fontsize=9)
    ax.set_xticks(range(len(unique_atk)))
    ax.set_xticklabels(unique_atk, rotation=35, ha="right", fontsize=9)
    ax.set_title("Detection rate on attack slices (Traffic label, test)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall on slice")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig07_per_attack_heatmap.pdf")
    fig.savefig(fig_dir / "fig07_per_attack_heatmap.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 8: Bootstrap PR-AUC 95% CI (top 4 models)
    # ------------------------------------------------------------------
    boot_rows = []
    fig, ax = plt.subplots(figsize=(7.9, 5.5))
    top4 = df_metrics["model"].head(4).tolist()
    xpos = np.arange(len(top4))
    means, los, his = [], [], []
    for name in top4:
        s = scores[name]
        m, lo, hi = bootstrap_pr_auc(y_test_arr, s, n_resamples=300)
        means.append(m)
        los.append(lo)
        his.append(hi)
        boot_rows.append({"model": name, "PR_AUC": m, "CI95_low": lo, "CI95_high": hi})
    pd.DataFrame(boot_rows).to_csv(tab_dir / "table_bootstrap_pr_auc.csv", index=False)
    ax.bar(xpos, means, yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)], color=[COLORS[i] for i in range(len(top4))], capsize=4, width=0.65)
    ax.set_xticks(xpos)
    ax.set_xticklabels(top4, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Bootstrap 95% CI for PR-AUC (test, n=300)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig08_bootstrap_pr_auc.pdf")
    fig.savefig(fig_dir / "fig08_bootstrap_pr_auc.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Table: McNemar + Cohen kappa (pairwise, top models)
    # ------------------------------------------------------------------
    top_m = df_metrics["model"].head(4).tolist()
    mc = []
    for i in range(len(top_m)):
        for j in range(i + 1, len(top_m)):
            a, b = top_m[i], top_m[j]
            chi2, p = mcnemar_midp(y_test_arr, preds[a], preds[b])
            kap = cohen_kappa_score(preds[a], preds[b])
            mc.append({"model_A": a, "model_B": b, "McNemar_chi2": chi2, "p_value": p, "Cohen_kappa": kap})
    pd.DataFrame(mc).to_csv(tab_dir / "table_mcnemar_kappa.csv", index=False)

    # ------------------------------------------------------------------
    # Figure 9: Cost-sensitive threshold sweep (FN cost multiplier)
    # ------------------------------------------------------------------
    # Normalised expected cost = (c_fp * FP + c_fn * FN) / n  for varying c_fn : c_fp
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    cost_ratios = [1, 5, 15, 40]  # c_fn / c_fp
    linestyles = ["-", "--", ":", "-."]
    t_grid = np.linspace(0.02, 0.98, 120)
    cost_models = [m for m in df_metrics["model"].head(4).tolist() if m in prob_models][:2]
    if not cost_models:
        cost_models = prob_models[:2]
    for j, name in enumerate(cost_models):
        s = scores[name]
        for k, cr in enumerate(cost_ratios):
            c_fn, c_fp = float(cr), 1.0
            losses = []
            for t in t_grid:
                pred = (s >= t).astype(int)
                fp = int(np.sum((pred == 1) & (y_test_arr == 0)))
                fn = int(np.sum((pred == 0) & (y_test_arr == 1)))
                losses.append((c_fp * fp + c_fn * fn) / len(y_test_arr))
            ax.plot(
                t_grid,
                losses,
                lw=1.35,
                color=COLORS[j],
                linestyle=linestyles[k % len(linestyles)],
                label=f"{name} · {cr}:1",
            )
    ax.set_xlabel("Classification threshold (probability)")
    ax.set_ylabel("Normalised expected misclassification cost")
    ax.set_title("Cost–threshold trade-offs (missed attacks weighted higher)")
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig09_cost_sensitive_threshold.pdf")
    fig.savefig(fig_dir / "fig09_cost_sensitive_threshold.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 10: Normalised confusion matrices (small multiples, top 4)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.3))
    axes = axes.ravel()
    for ax, name in zip(axes, df_metrics["model"].head(4)):
        ConfusionMatrixDisplay.from_predictions(
            y_test_arr,
            preds[name],
            normalize="true",
            display_labels=["Normal", "Attack"],
            ax=ax,
            colorbar=False,
            im_kw={"cmap": "Blues"},
        )
        ax.set_title(name, fontsize=10)
    fig.suptitle("Row-normalised confusion matrices (test)", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fig_dir / "fig10_confusion_grid.pdf")
    fig.savefig(fig_dir / "fig10_confusion_grid.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 11: Partial dependence (Random Forest, top 2 features by impurity)
    # ------------------------------------------------------------------
    rf_model = models.get("RandomForest (tuned)")
    if rf_model is not None:
        clf = rf_model.named_steps["clf"]
        imps = clf.feature_importances_
        top_idx = np.argsort(imps)[-2:][::-1]
        top_feats = [feature_names[i] for i in top_idx]
        X_fit = bundle["X_fit"]
        n_pdp = min(60_000, len(X_fit))
        rng = np.random.default_rng(7)
        ix = rng.choice(len(X_fit), size=n_pdp, replace=False)
        X_pdp = X_fit.iloc[ix]
        fig, ax = plt.subplots(1, 2, figsize=(11.2, 5.0))
        for a, fi in zip(ax, top_feats):
            PartialDependenceDisplay.from_estimator(
                rf_model,
                X_pdp,
                features=[fi],
                ax=a,
                line_kw={"color": COLORS[0], "lw": 2},
            )
            a.set_title(f"PDP: {fi}")
            a.grid(False)
        fig.suptitle("Partial dependence (Random Forest, train subsample)", y=1.02)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(fig_dir / "fig11_partial_dependence_rf.pdf")
        fig.savefig(fig_dir / "fig11_partial_dependence_rf.png")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 12: Precision@k for ranked alerts (operations-style)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    k_fracs = np.linspace(0.001, 0.15, 40)
    for j, name in enumerate(df_metrics["model"].head(4)):
        if name not in prob_models:
            continue
        s = scores[name]
        order = np.argsort(-s)
        y_sorted = y_test_arr[order]
        p_at_k = []
        for frac in k_fracs:
            k = max(1, int(frac * len(y_sorted)))
            p_at_k.append(y_sorted[:k].mean())
        ax.plot(k_fracs * 100, p_at_k, color=COLORS[j], lw=1.6, label=name)
    ax.set_xlabel("Top fraction of flows flagged (%)")
    ax.set_ylabel("Precision in top-k (attack proportion)")
    ax.set_title("Precision@k under score ranking (test)")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig12_precision_at_k.pdf")
    fig.savefig(fig_dir / "fig12_precision_at_k.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 13: Cumulative capture of attacks (gain chart vs random ordering)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    frac = np.linspace(0, 1, 400)
    n_test = len(y_test_arr)
    n_pos = max(int(y_test_arr.sum()), 1)
    # Optimal: all positives appear before any negative → attacks captured = min(n_pos, frac * n_test)
    ideal_pct = np.minimum(frac * n_test, n_pos) / n_pos * 100.0
    ax.plot(frac * 100, ideal_pct, "k--", lw=1.0, alpha=0.55, label="Perfect ranker")
    ax.plot(frac * 100, frac * 100, color="gray", lw=1.0, alpha=0.65, label="Random order")
    for j, name in enumerate(df_metrics["model"].head(4)):
        s = scores[name]
        order = np.argsort(-s)
        y_ord = y_test_arr[order]
        cum = np.cumsum(y_ord)
        xs = (np.arange(1, len(cum) + 1) / len(cum)) * 100
        ax.plot(xs, cum / max(n_pos, 1) * 100, color=COLORS[j], lw=1.5, label=name)
    ax.set_xlabel("Fraction of test flows examined (%)")
    ax.set_ylabel("Cumulative % of all attacks captured")
    ax.set_title("Attack capture curves (score-sorted vs baselines)")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig13_attack_capture_gain.pdf")
    fig.savefig(fig_dir / "fig13_attack_gain.png")
    plt.close(fig)

    # Summary JSON for paper
    summary = {
        "champion_model": champ,
        "test_n": int(len(y_test_arr)),
        "attack_rate_test": float(y_test_arr.mean()),
        "tables": [str(tab_dir / "table1_scalar_metrics.csv")],
        "figures": [str(p) for p in sorted(fig_dir.glob("fig*.pdf"))],
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Outputs -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
