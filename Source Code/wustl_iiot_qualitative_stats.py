#!/usr/bin/env python3
"""
Qualitative error analysis + extra statistical tests for WUSTL-IIoT (post-hoc).

Requires:
  - Same CSV + random split as wustl_iiot_post_analysis.py (RANDOM_STATE=42)
  - Cached trained models: wustl_analysis_cache/analysis_bundle.joblib

Outputs (under --output-dir):
  qualitative/   — FP/FN samples, feature deviation tables, report snippets
  stats_extra/   — paired bootstrap, DeLong ROC test, effect sizes

Run:
  python wustl_iiot_qualitative_stats.py
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Reuse data protocol from main analysis script
from wustl_iiot_post_analysis import (
    DEFAULT_DATA,
    RANDOM_STATE,
    load_data,
)


# ---------------------------------------------------------------------------
# DeLong et al. (1988) — correlated ROC AUC comparison (two models, same test set)
# Implementation follows the covariance formulation used in biostatistics packages.
# ---------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of vector x (for ties)."""
    J = np.argsort(x)
    Z = x[J]
    n = len(x)
    T = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(n, dtype=float)
    T2[J] = T
    return T2


def _fast_aucs(s: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """AUC and structural components for DeLong (binary y in {0,1})."""
    pos = s[y == 1]
    neg = s[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        raise ValueError("Need both classes for ROC.")
    v_pos = _midrank(pos)
    v_neg = _midrank(neg)
    w = _midrank(np.concatenate([pos, neg]))
    auc = (w[m:].sum() - n * (n + 1) / 2.0) / (m * n)
    v10 = (w[:m] - v_pos - (m + 1) / 2.0) / n
    v01 = -(w[m:] - v_neg - (n + 1) / 2.0) / m
    return float(auc), v01, v10


def delong_roc_test(
    y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray
) -> Dict[str, float]:
    """
    DeLong test for difference in ROC AUC between two scoring systems on the same subjects.
    Returns auc_a, auc_b, z, two-sided p-value.
    """
    y = np.asarray(y_true).astype(int)
    sa = np.asarray(scores_a).ravel()
    sb = np.asarray(scores_b).ravel()
    auc_a, v10a, v01a = _fast_aucs(sa, y)
    auc_b, v10b, v01b = _fast_aucs(sb, y)
    na = int((y == 0).sum())
    nn = int((y == 1).sum())
    t01 = v01a - v01b
    t10 = v10a - v10b
    s01 = t01.var(ddof=1) / nn if nn > 1 else 0.0
    s10 = t10.var(ddof=1) / na if na > 1 else 0.0
    se = float(np.sqrt(s01 + s10))
    z = (auc_a - auc_b) / se if se > 1e-15 else 0.0
    p = float(2 * (1 - norm.cdf(abs(z))))
    return {"auc_a": auc_a, "auc_b": auc_b, "delta_auc": auc_a - auc_b, "z": z, "p_two_sided": p}


def bootstrap_paired_metric_diff(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric_fn,
    n_bootstrap: int = 800,
    seed: int = RANDOM_STATE,
) -> Dict[str, float]:
    """Bootstrap distribution of (metric_a - metric_b) on resampled test rows (paired)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    sa = np.asarray(score_a)
    sb = np.asarray(score_b)
    n = len(y)
    diffs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        ix = rng.integers(0, n, size=n)
        ma = metric_fn(y[ix], sa[ix])
        mb = metric_fn(y[ix], sb[ix])
        diffs[b] = ma - mb
    obs = metric_fn(y, sa) - metric_fn(y, sb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_two = 2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))
    p_two = min(p_two, 1.0)
    return {
        "observed_delta": float(obs),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_approx_two_sided": float(p_two),
    }


def cliffs_delta(
    x: np.ndarray, y: np.ndarray, max_pairs: int = 500_000, seed: int = RANDOM_STATE
) -> float:
    """Cliff's delta (effect size). For large arrays uses random subset of pairs (unbiased estimate)."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    total_pairs = nx * ny
    rng = np.random.default_rng(seed)
    if total_pairs <= max_pairs:
        more = np.sum(x[:, None] > y[None, :])
        less = np.sum(x[:, None] < y[None, :])
    else:
        xi = rng.integers(0, nx, size=max_pairs)
        yi = rng.integers(0, ny, size=max_pairs)
        more = np.sum(x[xi] > y[yi])
        less = np.sum(x[xi] < y[yi])
        return float((more - less) / max_pairs)
    return float((more - less) / total_pairs)


def scores_from_model(model: Any, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict_proba"):
        s = model.predict_proba(X)[:, 1]
    else:
        s = model.decision_function(X)
    pred = model.predict(X)
    return s, pred


def top_feature_deviations(
    X_row: pd.Series,
    ref_normal: pd.DataFrame,
    k: int = 5,
) -> List[Tuple[str, float, float]]:
    """Return top-k features by absolute z-score vs normal-only training reference."""
    out = []
    for col in X_row.index:
        ref = ref_normal[col].dropna()
        if len(ref) < 2:
            continue
        mu, sig = ref.mean(), ref.std(ddof=0)
        if sig < 1e-12:
            continue
        z = (float(X_row[col]) - mu) / sig
        out.append((col, float(X_row[col]), z))
    out.sort(key=lambda t: abs(t[2]), reverse=True)
    return out[:k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("wustl_analysis_cache") / "analysis_bundle.joblib",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("wustl_analysis_output"))
    parser.add_argument("--max-samples", type=int, default=40, help="FP/FN rows each to export")
    parser.add_argument("--bootstrap", type=int, default=800)
    args = parser.parse_args()

    if not args.cache.exists():
        print(f"Cache not found: {args.cache}", file=sys.stderr)
        sys.exit(1)
    if not args.data_path.exists():
        print(f"Data not found: {args.data_path}", file=sys.stderr)
        sys.exit(1)

    bundle = joblib.load(args.cache)
    models: Dict[str, Any] = bundle["models"]

    print("Loading data and reproducing train/test split...")
    X, y, traffic = load_data(args.data_path)
    X_train, X_test, y_train, y_test, tr_train, tr_test = train_test_split(
        X, y, traffic, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    y_test_arr = y_test.values
    if not np.array_equal(y_test_arr, bundle["y_test"].values):
        print("Warning: y_test mismatch with cache — check DATA_PATH and RANDOM_STATE.", file=sys.stderr)

    ref_normal = X_train[y_train == 0]

    out_root = args.output_dir
    qdir = out_root / "qualitative"
    sdir = out_root / "stats_extra"
    qdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)

    # --- Rank models by PR-AUC on test (same as main script) ---
    scores_all: Dict[str, np.ndarray] = {}
    preds_all: Dict[str, np.ndarray] = {}
    for name, m in models.items():
        s, p = scores_from_model(m, X_test)
        scores_all[name] = s
        preds_all[name] = p

    rows_rank = []
    for name in models:
        s = scores_all[name]
        rows_rank.append(
            (name, average_precision_score(y_test_arr, s), roc_auc_score(y_test_arr, s))
        )
    rows_rank.sort(key=lambda t: t[1], reverse=True)
    champ = rows_rank[0][0]
    print(f"Champion (PR-AUC): {champ}")

    s_ch = scores_all[champ]
    p_ch = preds_all[champ]

    # --- Qualitative: FP / FN for champion ---
    fp_mask = (y_test_arr == 0) & (p_ch == 1)
    fn_mask = (y_test_arr == 1) & (p_ch == 0)
    fp_idx = np.flatnonzero(fp_mask)
    fn_idx = np.flatnonzero(fn_mask)
    rng = np.random.default_rng(RANDOM_STATE + 99)
    if len(fp_idx) > args.max_samples:
        fp_idx = rng.choice(fp_idx, size=args.max_samples, replace=False)
    if len(fn_idx) > args.max_samples:
        fn_idx = rng.choice(fn_idx, size=args.max_samples, replace=False)

    def build_table(indices: np.ndarray, kind: str) -> pd.DataFrame:
        recs = []
        for i in indices:
            row = X_test.iloc[int(i)]
            tr = tr_test.iloc[int(i)]
            devs = top_feature_deviations(row, ref_normal, k=5)
            recs.append(
                {
                    "row_index": int(i),
                    "Traffic_label": str(tr),
                    "y_true": int(y_test_arr[int(i)]),
                    "pred": int(p_ch[int(i)]),
                    "score": float(s_ch[int(i)]),
                    "top1_feature": devs[0][0] if devs else "",
                    "top1_z": devs[0][2] if devs else np.nan,
                    "deviation_summary": "; ".join(f"{c}(z={z:.2f})" for c, _, z in devs[:3]),
                }
            )
        return pd.DataFrame(recs)

    df_fp = build_table(fp_idx, "FP")
    df_fn = build_table(fn_idx, "FN")
    df_fp.to_csv(qdir / "false_positives_sample.csv", index=False)
    df_fn.to_csv(qdir / "false_negatives_sample.csv", index=False)

    # Narrative snippet for report (qualitative discussion)
    snippet = f"""\
    Qualitative error inspection ({champ} on test set)
    - False positives (normal labelled attack): {int(fp_mask.sum())} total; exported {len(df_fp)} rows.
    - False negatives (attack labelled normal): {int(fn_mask.sum())} total; exported {len(df_fn)} rows.
    - Columns `top1_z` and `deviation_summary` show which flow features deviate most from **normal-only**
      training reference (z-scores). Large |z| suggests the flow resembles attack-like volume/timing
      statistics even when the label is normal (operational overlap), or the opposite for FN.
    - Cross-check `Traffic_label` for FN to see whether misses concentrate in rare classes (e.g. reconnaissance).
    """
    (qdir / "README_qualitative_snippet.txt").write_text(textwrap.dedent(snippet), encoding="utf-8")

    # --- Effect size: separation of scores (attacks vs normal) ---
    s_att = s_ch[y_test_arr == 1]
    s_norm = s_ch[y_test_arr == 0]
    delta_cliff = cliffs_delta(s_att, s_norm)
    cohen_d = (np.mean(s_att) - np.mean(s_norm)) / np.sqrt(
        0.5 * (np.var(s_att, ddof=1) + np.var(s_norm, ddof=1))
    ) if len(s_att) > 1 and len(s_norm) > 1 else np.nan

    pd.DataFrame(
        [
            {
                "model": champ,
                "cliffs_delta_attack_vs_normal_scores": delta_cliff,
                "cohen_d_attack_vs_normal_scores": float(cohen_d),
            }
        ]
    ).to_csv(sdir / "effect_size_score_separation.csv", index=False)

    # --- Paired bootstrap: PR-AUC and ROC-AUC differences (top models) ---
    top_names = [t[0] for t in rows_rank[:4]]
    pr_rows = []
    roc_rows = []
    for i in range(len(top_names)):
        for j in range(i + 1, len(top_names)):
            a, b = top_names[i], top_names[j]
            sa, sb = scores_all[a], scores_all[b]
            pr = bootstrap_paired_metric_diff(
                y_test_arr,
                sa,
                sb,
                lambda y, s: average_precision_score(y, s),
                n_bootstrap=args.bootstrap,
            )
            pr["model_a"] = a
            pr["model_b"] = b
            pr["metric"] = "PR_AUC"
            pr_rows.append(pr)
            roc_b = bootstrap_paired_metric_diff(
                y_test_arr,
                sa,
                sb,
                lambda y, s: roc_auc_score(y, s),
                n_bootstrap=args.bootstrap,
            )
            roc_b["model_a"] = a
            roc_b["model_b"] = b
            roc_b["metric"] = "ROC_AUC"
            roc_rows.append(roc_b)

    pd.DataFrame(pr_rows).to_csv(sdir / "paired_bootstrap_pr_auc_diff.csv", index=False)
    pd.DataFrame(roc_rows).to_csv(sdir / "paired_bootstrap_roc_auc_diff.csv", index=False)

    # --- DeLong tests (top 4 pairwise) ---
    delong_rows = []
    for i in range(len(top_names)):
        for j in range(i + 1, len(top_names)):
            a, b = top_names[i], top_names[j]
            d = delong_roc_test(y_test_arr, scores_all[a], scores_all[b])
            d["model_a"] = a
            d["model_b"] = b
            delong_rows.append(d)
    pd.DataFrame(delong_rows).to_csv(sdir / "delong_roc_auc_test.csv", index=False)

    # --- Mann–Whitney: are attack scores stochastically larger than normal? (descriptive) ---
    u_stat, p_mw = stats.mannwhitneyu(s_att, s_norm, alternative="greater")
    pd.DataFrame(
        [
            {
                "model": champ,
                "mannwhitney_u": float(u_stat),
                "p_one_sided_attack_scores_greater": float(p_mw),
                "n_attack": len(s_att),
                "n_normal": len(s_norm),
            }
        ]
    ).to_csv(sdir / "mann_whitney_scores_attack_vs_normal.csv", index=False)

    summary = {
        "champion": champ,
        "qualitative_exports": str(qdir),
        "stats_exports": str(sdir),
        "notes": [
            "DeLong: compares correlated ROC AUCs on the same test subjects (two-sided p).",
            "Bootstrap: paired resampling of test rows for PR-AUC / ROC-AUC differences.",
            "Mann–Whitney: attack vs normal score distributions (descriptive separation).",
        ],
    }
    (out_root / "qualitative_stats_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Qualitative tables -> {qdir.resolve()}")
    print(f"Extra statistics -> {sdir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
