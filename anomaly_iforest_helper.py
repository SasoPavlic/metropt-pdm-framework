#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-series IsolationForest anomaly explorer (native timeline plot, MetroPT-3).

- Loads CSV/TXT specific to this project (no MAT/Parquet to keep it simple).
- Optional pre-downsample (time median) to regularize cadence.
- Rolling stats over a compact feature set.
- Train on first N% (chronological), score all points.
- Label by a robust (Q3 + 3*IQR) threshold on (optionally LPF-smoothed) IF scores.
- Plot ORIGINAL timeline with point colors (normal/anomalous) + shaded/marked failure windows with labels.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from iforest_data import (
    build_plot_pca_feature,
    build_operation_phase,
    build_rolling_features,
    load_csv,
    parse_maintenance_windows,
    pre_downsample,
    select_numeric_features,
    top_k_by_variance,
)
from iforest_metrics import confusion_and_scores, evaluate_risk_thresholds, window_mask
from iforest_model import train_iforest_and_score
from iforest_plotting import prepare_plot_frame, plot_raw_timeline


def parse_risk_grid_spec(spec: str) -> List[float]:
    parts = (spec or "").split(":")
    if len(parts) != 3:
        raise ValueError("Risk grid must be provided as 'start:stop:step'.")
    start, stop, step = map(float, parts)
    if step <= 0:
        raise ValueError("Risk grid step must be positive.")
    if stop < start:
        raise ValueError("Risk grid stop must be greater than or equal to start.")
    values: List[float] = []
    val = start
    while val <= stop + 1e-9:
        values.append(round(val, 6))
        val += step
    return values


# -----------------------------
# Configuration constants
# -----------------------------
# Timestamp column name in the input CSV; set to None to auto-detect.
INPUT_TIMESTAMP_COL: Optional[str] = None
# Whether to drop unnamed index columns commonly created by pandas when saving CSVs.
DROP_UNNAMED_INDEX: bool = True
# Optional time-based pre-downsampling rule (e.g., '60s') to regularize cadence before feature building; None disables.
PRE_DOWNSAMPLE_RULE: Optional[str] = None
# Rolling window for feature aggregation (e.g., '600s' = 10 minutes).
ROLLING_WINDOW: str = "600s"
# Fraction of earliest data used to train the IsolationForest.
TRAIN_FRAC: float = 0.33
# Limit of most-variable base numeric features to keep before rolling aggregation.
MAX_BASE_FEATURES: int = 12
# Exclude near-binary numeric columns from features to focus on informative signals.
EXCLUDE_QUASI_BINARY: bool = True
# PCA components computed for plotting (training always uses the full feature set).
PCA_COMPONENTS: int = 8
# Whether to project all features to a PCA component for visualization.
USE_PCA_FOR_PLOTTING: bool = True
# 1-based index of the PCA component to show on the timeline when PCA plotting is enabled.
PLOT_PCA_COMPONENT_INDEX: int = 1
# Column name used to store the PCA-derived plot feature.
PLOT_PCA_FEATURE_NAME: str = "plot_pca_component"
# Hours before maintenance start considered pre-maintenance (operation_phase=1).
PRE_MAINTENANCE_HOURS: int = 2
# Rolling risk default window (minutes).
DEFAULT_RISK_WINDOW_MINUTES: int = 120
# Default risk evaluation grid specification (start:stop:step).
DEFAULT_RISK_GRID: str = "0.1:0.6:0.05"
# Early-warning horizon for risk evaluation (minutes).
EARLY_WARNING_MINUTES: int = 120
# Exponential low-pass filter alpha for anomaly scores; 0 disables smoothing.
LPF_ALPHA: float = 0
# Name of the column to plot as context; None chooses the first numeric column.
PLOT_FEATURE: Optional[str] = "DV_pressure"
# Plot every Nth point to reduce density; 1 means no thinning.
PLOT_STRIDE: int = 10
# Optional rolling median rule for the plotted Y only (e.g., '60s'); None disables.
PLOT_ROLLING: Optional[str] = None
# Short windows (minutes) threshold: windows with duration <= this are drawn as vertical lines.
SHORT_WINDOW_MINUTES: float = 100.0
# Show labels for maintenance windows (IDs/severity) on the plot.
SHOW_WINDOW_LABELS: bool = True
# Font size for maintenance window labels.
WINDOW_LABEL_FONTSIZE: int = 9
# Label format; placeholders: {id}, {severity}, {dur_min}.
WINDOW_LABEL_FORMAT: str = "{id}"
# Use the built-in MetroPT-3 failure windows from Davari et al.
USE_DEFAULT_METROPT_WINDOWS: bool = True
# Preferential feature names for MetroPT-3; if present, they are prioritized.
LIKELY_METROPT_FEATURES = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Motor_current",
    "Oil_temperature",
    "Caudal_impulses",
]

# ===== MetroPT-3 failure windows (Davari et al., 2021) =====
# Table II intervals normalized to ISO (YYYY-MM-DD HH:MM:SS)
# Format: (start, end, id, severity)
DEFAULT_METROPT_WINDOWS: List[Tuple[str, str, str, str]] = [
    ("2020-04-12 11:50:00", "2020-04-12 23:30:00", "#1", "high"),
    ("2020-04-18 00:00:00", "2020-04-18 23:59:00", "#2", "high"),
    ("2020-04-19 00:00:00", "2020-04-19 01:30:00", "#3", "high"),
    ("2020-04-29 03:20:00", "2020-04-29 04:00:00", "#4", "high"),
    ("2020-04-29 22:00:00", "2020-04-29 22:20:00", "#5", "high"),
    ("2020-05-13 14:00:00", "2020-05-13 23:59:00", "#6", "high"),
    ("2020-05-18 05:00:00", "2020-05-18 05:30:00", "#7", "high"),
    ("2020-05-19 10:10:00", "2020-05-19 11:00:00", "#8", "high"),
    ("2020-05-19 22:10:00", "2020-05-19 23:59:00", "#9", "high"),
    ("2020-05-20 00:00:00", "2020-05-20 20:00:00", "#10", "high"),
    ("2020-05-23 09:50:00", "2020-05-23 10:10:00", "#11", "high"),
    ("2020-05-29 23:30:00", "2020-05-29 23:59:00", "#12", "high"),
    ("2020-05-30 00:00:00", "2020-05-30 06:00:00", "#13", "high"),
    ("2020-06-01 15:00:00", "2020-06-01 15:40:00", "#14", "high"),
    ("2020-06-03 10:00:00", "2020-06-03 11:00:00", "#15", "high"),
    ("2020-06-05 10:00:00", "2020-06-05 23:59:00", "#16", "high"),
    ("2020-06-06 00:00:00", "2020-06-06 23:59:00", "#17", "high"),
    ("2020-06-07 00:00:00", "2020-06-07 14:30:00", "#18", "high"),
    ("2020-07-08 17:30:00", "2020-07-08 19:00:00", "#19", "high"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00", "#20", "medium"),
    ("2020-07-17 04:30:00", "2020-07-17 05:30:00", "#21", "high"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="IsolationForest anomaly explorer (MetroPT-3, CSV-only).")
    ap.add_argument("--input", required=True, help="Path to input CSV/TXT file.")
    ap.add_argument("--save_fig", default=None, help="Optional path to save the plotted figure.")
    ap.add_argument("--save_pred_csv", default=None, help="Optional path to save per-point predictions.")
    ap.add_argument(
        "--risk_window_minutes",
        type=int,
        default=DEFAULT_RISK_WINDOW_MINUTES,
        help="Rolling window (minutes) for maintenance_risk calculation.",
    )
    ap.add_argument(
        "--risk_eval_grid",
        default=DEFAULT_RISK_GRID,
        help="Grid spec 'start:stop:step' for maintenance_risk event-level evaluation.",
    )
    args = ap.parse_args()

    # 1) Load raw (CSV-only)
    df_raw = load_csv(args.input, INPUT_TIMESTAMP_COL, drop_unnamed=DROP_UNNAMED_INDEX)

    # 2) Pre-downsample (optional) before feature building
    df_ds = pre_downsample(df_raw, PRE_DOWNSAMPLE_RULE)

    # 3) Base numeric features (exclude quasi-binary, then top-K by variance)
    base_feats = select_numeric_features(
        df_ds, prefer=LIKELY_METROPT_FEATURES, exclude_quasi_binary=EXCLUDE_QUASI_BINARY
    )
    if not base_feats:
        raise ValueError("No numeric features found after binary exclusion.")
    df_base = top_k_by_variance(df_ds[base_feats].copy(), MAX_BASE_FEATURES)

    # 4) Rolling features on the compact set
    X = build_rolling_features(df_base, rolling_window=ROLLING_WINDOW)

    # 5) Train IF + score (with optional LPF on scores)
    pred, info = train_iforest_and_score(
        X=X,
        train_frac=TRAIN_FRAC,
        lpf_alpha=LPF_ALPHA,
    )

    # Rolling maintenance risk from exceedance of the training threshold
    risk_window_minutes = max(1, int(args.risk_window_minutes))
    score_for_risk = (
        pred["anom_score_lpf"] if "anom_score_lpf" in pred.columns else pred["anom_score"]
    ).astype(float).fillna(0.0)
    tau = info.get("threshold")
    if tau is None:
        q3 = info.get("q3")
        iqr = info.get("iqr")
        if q3 is not None and iqr is not None:
            tau = float(q3 + 3.0 * iqr)
        else:
            finite = score_for_risk.to_numpy()
            finite = finite[np.isfinite(finite)]
            tau = float(np.percentile(finite, 75)) if finite.size else 0.0
    exceedance = (score_for_risk >= float(tau)).astype(float)
    maintenance_risk = (
        exceedance.rolling(f"{risk_window_minutes}min", min_periods=1).mean().astype(np.float32).fillna(0.0)
    ).rename("maintenance_risk")
    pred["maintenance_risk"] = maintenance_risk

    # 6) Align predictions back to ORIGINAL (raw) time index
    plot_feature_name = PLOT_FEATURE
    plot_feature_label: Optional[str] = plot_feature_name
    plot_pca_meta: Optional[dict] = None
    raw_num = df_raw.select_dtypes(include=[np.number])
    if PLOT_FEATURE and PLOT_FEATURE in df_raw.columns and PLOT_FEATURE not in raw_num.columns:
        raw_num = raw_num.join(df_raw[[PLOT_FEATURE]], how="left")
    raw_reindexed = raw_num.reindex(pred.index)

    if USE_PCA_FOR_PLOTTING:
        try:
            n_comp_for_plot = PCA_COMPONENTS if PCA_COMPONENTS and PCA_COMPONENTS > 0 else 1
            plot_pca_series, plot_pca_meta = build_plot_pca_feature(
                df_num=X,
                n_components=n_comp_for_plot,
                component_index=PLOT_PCA_COMPONENT_INDEX,
                feature_name=PLOT_PCA_FEATURE_NAME,
            )
            raw_reindexed[plot_pca_series.name] = plot_pca_series.reindex(raw_reindexed.index)
            plot_feature_name = plot_pca_series.name
            plot_feature_label = plot_feature_name
        except Exception as exc:
            plot_pca_meta = None
            print(f"[WARN] Plot PCA feature generation failed ({exc}); falling back to configured plot feature.")
    if plot_pca_meta:
        comp_idx = plot_pca_meta.get("component_index", 1)
        ratios = plot_pca_meta.get("explained_variance_ratio", [])
        ratio_val = ratios[comp_idx - 1] if 0 <= comp_idx - 1 < len(ratios) else None
        if ratio_val is not None:
            plot_feature_label = f"{plot_feature_name} (PCA component #{comp_idx}, var={ratio_val:.2%})"
            print(
                f"[INFO] Plot PCA component #{comp_idx} derived from {plot_pca_meta.get('n_components')} total components "
                f"(explained variance ratio={ratio_val:.4f})."
            )
        else:
            plot_feature_label = f"{plot_feature_name} (PCA component #{comp_idx})"
            print(
                f"[INFO] Plot PCA component #{comp_idx} derived from {plot_pca_meta.get('n_components')} total components."
            )

    if not plot_feature_name or plot_feature_name not in raw_reindexed.columns:
        fallback_cols = raw_reindexed.select_dtypes(include=[np.number]).columns.tolist()
        if not fallback_cols:
            raise ValueError("No numeric column available for plotting after alignment.")
        fallback_name = fallback_cols[0]
        if plot_feature_name:
            print(f"[WARN] Plot feature '{plot_feature_name}' unavailable after alignment; using '{fallback_name}' instead.")
        plot_feature_name = fallback_name
        plot_feature_label = f"{plot_feature_name} (auto-selected)"

    if not plot_feature_label:
        plot_feature_label = plot_feature_name

    # 7) Maintenance windows + operation phase feature
    maint_windows = parse_maintenance_windows(
        windows=None,
        maintenance_csv=None,
        use_default_windows=USE_DEFAULT_METROPT_WINDOWS,
        default_windows=DEFAULT_METROPT_WINDOWS,
    )
    operation_phase = build_operation_phase(
        index=pred.index,
        windows=maint_windows,
        pre_hours=PRE_MAINTENANCE_HOURS,
    )
    operation_phase = operation_phase.astype(np.int8)
    pred["operation_phase"] = operation_phase
    extra_features = pd.concat([operation_phase, maintenance_risk], axis=1)
    raw_reindexed = raw_reindexed.join(extra_features, how="left")

    # Event-level evaluation of maintenance_risk thresholds
    risk_thresholds: List[float] = []
    try:
        risk_thresholds = parse_risk_grid_spec(args.risk_eval_grid)
    except ValueError as exc:
        print(f"[WARN] Skipping maintenance_risk evaluation: {exc}")
    if risk_thresholds:
        risk_results = evaluate_risk_thresholds(
            risk=maintenance_risk,
            maintenance_windows=maint_windows,
            thresholds=risk_thresholds,
            early_warning_minutes=EARLY_WARNING_MINUTES,
        )
        if risk_results:
            print(
                f"[RISK] Rolling window={risk_window_minutes} min, early-warning horizon={EARLY_WARNING_MINUTES} min"
            )
            for res in risk_results:
                print(
                    f"[RISK] θ={res['threshold']:.2f}  Precision={res['precision']:.4f}  "
                    f"Recall={res['recall']:.4f}  F1={res['f1']:.4f}  TP={res['tp']}  FP={res['fp']}  FN={res['fn']}"
                )
            best = max(risk_results, key=lambda r: (r["f1"], r["precision"], -r["threshold"]))
            print(
                f"[RISK] Best θ={best['threshold']:.2f}: Precision={best['precision']:.4f}  "
                f"Recall={best['recall']:.4f}  F1={best['f1']:.4f}"
            )

    # 8) Prepare & plot
    df_plot = prepare_plot_frame(
        raw_df=raw_reindexed,
        pred_df=pred,
        feature_to_plot=plot_feature_name,
        plot_stride=PLOT_STRIDE,
        plot_rolling=PLOT_ROLLING,
    )

    # Exact training cutoff timestamp on the prediction index
    train_cutoff_ts = None
    try:
        if info.get("n_train", 0) > 0 and info["n_train"] <= len(pred.index):
            train_cutoff_ts = pred.index[info["n_train"] - 1]
    except Exception:
        train_cutoff_ts = None

    # Decide labeling default: constants-based
    effective_show_labels = bool(SHOW_WINDOW_LABELS or USE_DEFAULT_METROPT_WINDOWS)
    plot_raw_timeline(
        df_plot,
        maint_windows,
        save_fig=args.save_fig,
        train_frac=TRAIN_FRAC,
        train_cutoff_time=train_cutoff_ts,
        min_window_minutes=SHORT_WINDOW_MINUTES,
        show_window_labels=effective_show_labels,
        window_label_fontsize=WINDOW_LABEL_FONTSIZE,
        window_label_format=WINDOW_LABEL_FORMAT,
        feature_label=plot_feature_label,
    )

    # 9) Optional: save per-point predictions (timestamp, score, is_anomaly)
    if args.save_pred_csv:
        out = pred.copy()
        out.index.name = "timestamp"
        out.to_csv(args.save_pred_csv)

    # --------- 10) Console summary + metrics ----------
    total_pts = int(pred["is_anomaly"].sum())
    pct_pts = float(100.0 * pred["is_anomaly"].mean())
    print(f"[INFO] Model rule: {info['label_rule']}, features={info['n_features']}, LPF alpha={info['lpf_alpha']}")
    print(f"[INFO] Train size: {info['n_train']}/{info['n_total']}")
    print(f"[INFO] Point anomalies: {total_pts} ({pct_pts:.2f}%)")
    if info["threshold"] is not None:
        print(f"[INFO] Train-based threshold: {info['threshold']:.4f}")

    # ---- Build ground-truth labels from windows and compute metrics ----
    y_true_all = window_mask(pred.index, maint_windows)
    y_pred_all = pd.Series(pred["is_anomaly"].astype(int).values, index=pred.index, name="is_anomaly")

    m_all = confusion_and_scores(y_true_all, y_pred_all)
    print("[METRIC] Full timeline:")
    print(f"         TP={m_all['TP']}  FP={m_all['FP']}  FN={m_all['FN']}  TN={m_all['TN']}")
    print(f"         Precision={m_all['precision']:.4f}  Recall={m_all['recall']:.4f}  F1={m_all['f1']:.4f}  Accuracy={m_all['accuracy']:.4f}")

    if train_cutoff_ts is not None:
        post_idx = pred.index >= train_cutoff_ts
        if post_idx.any():
            m_post = confusion_and_scores(y_true_all[post_idx], y_pred_all[post_idx])
            print("[METRIC] Post-training window only:")
            print(f"         TP={m_post['TP']}  FP={m_post['FP']}  FN={m_post['FN']}  TN={m_post['TN']}")
            print(f"         Precision={m_post['precision']:.4f}  Recall={m_post['recall']:.4f}  F1={m_post['f1']:.4f}  Accuracy={m_post['accuracy']:.4f}")

    if maint_windows:
        def _fmt_win(w):
            try:
                if len(w) >= 4:
                    s, e, wid, sev = w[0], w[1], w[2], w[3]
                else:
                    s, e = w[0], w[1]
                    wid, sev = None, None
                dur_min = int(round(max(0.0, (pd.to_datetime(e) - pd.to_datetime(s)).total_seconds() / 60.0)))
                label = f"{pd.to_datetime(s)} → {pd.to_datetime(e)}"
                meta = []
                if wid:
                    meta.append(str(wid))
                if sev:
                    meta.append(str(sev))
                meta.append(f"{dur_min}min")
                return f"{label} ({', '.join(meta)})"
            except Exception:
                return str(w)

        print("[INFO] Failure windows:", [_fmt_win(w) for w in maint_windows])


if __name__ == "__main__":
    main()
