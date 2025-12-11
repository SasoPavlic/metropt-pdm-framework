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
    build_operation_phase,
    build_rolling_features,
    load_csv,
    parse_maintenance_windows,
    pre_downsample,
    select_numeric_features,
    top_k_by_variance,
)
from iforest_metrics import confusion_and_scores, evaluate_risk_thresholds
from iforest_model import train_iforest_and_score, train_iforest_on_slices
from iforest_plotting import plot_raw_timeline


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
# --- Data loading / preprocessing ---
# Timestamp column name in the input CSV; set to None to auto-detect.
INPUT_TIMESTAMP_COL: Optional[str] = None
# Whether to drop unnamed index columns commonly created by pandas when saving CSVs.
DROP_UNNAMED_INDEX: bool = True
# Input/outputs
INPUT_PATH: str = "MetroPT3.csv"
SAVE_FIG_PATH: Optional[str] = "metropt3_iforest_raw.png"
SAVE_PRED_CSV_PATH: Optional[str] = "metropt3_iforest_pred.csv"
SAVE_FEATURES_CSV_PATH: Optional[str] = "metropt3_iforest_features.csv"

# Experiment mode:
# - "single": one global model trained once on an early slice.
# - "daily": per-day adaptive models trained on all history up to the previous day.
# - "per_maint": per-inter-maintenance models trained on accumulated normal data
#   up to the end of the previous maintenance window.
EXPERIMENT_MODE: str = "single"

# Optional time-based pre-downsampling rule (e.g., '60s') to regularize cadence before feature building; None disables.
PRE_DOWNSAMPLE_RULE: Optional[str] = None
# Rolling window for feature aggregation (e.g., '600s' = 10 minutes).
ROLLING_WINDOW: str = "60s"
# Fraction of earliest data used to train the IsolationForest.
TRAIN_FRAC: float = 0.03
# Limit of most-variable base numeric features to keep before rolling aggregation.
MAX_BASE_FEATURES: int = 12
# Exclude near-binary numeric columns from features to focus on informative signals.
EXCLUDE_QUASI_BINARY: bool = False

# --- Modeling / scoring ---
# Exponential low-pass filter alpha for anomaly scores; 0 disables smoothing.
LPF_ALPHA: float = 0.4
# Rolling risk window (minutes).
RISK_WINDOW_MINUTES: int = 1920
# Risk evaluation grid specification (start:stop:step).
RISK_EVAL_GRID_SPEC: str = "0.05:0.6:0.01"
# Early-warning horizon for risk evaluation (minutes).
EARLY_WARNING_MINUTES: int = 120

# --- Maintenance context / plotting ---
# Hours before maintenance start considered pre-maintenance (operation_phase=1).
PRE_MAINTENANCE_HOURS: int = 2
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


def _build_features_and_context() -> Tuple[pd.DataFrame, List[Tuple], pd.Series]:
    """Load raw data, engineer rolling features, and build maintenance context."""
    df_raw = load_csv(INPUT_PATH, INPUT_TIMESTAMP_COL, drop_unnamed=DROP_UNNAMED_INDEX)

    # Optional pre-downsample before feature building
    df_ds = pre_downsample(df_raw, PRE_DOWNSAMPLE_RULE)

    # Base numeric features (optionally exclude quasi-binary, then top-K by variance)
    base_feats = select_numeric_features(
        df_ds,
        prefer=LIKELY_METROPT_FEATURES,
        exclude_quasi_binary=EXCLUDE_QUASI_BINARY,
    )
    if not base_feats:
        raise ValueError("No numeric features found after binary exclusion.")
    df_base = top_k_by_variance(df_ds[base_feats].copy(), MAX_BASE_FEATURES)

    # Rolling features on the compact set
    X = build_rolling_features(df_base, rolling_window=ROLLING_WINDOW)
    if SAVE_FEATURES_CSV_PATH:
        feats_out = X.copy()
        feats_out.index.name = "timestamp"
        feats_out.to_csv(SAVE_FEATURES_CSV_PATH)

    # Maintenance windows + operation phase feature (0=normal,1=pre-maint,2=maintenance)
    maint_windows = parse_maintenance_windows(
        windows=None,
        maintenance_csv=None,
        use_default_windows=USE_DEFAULT_METROPT_WINDOWS,
        default_windows=DEFAULT_METROPT_WINDOWS,
    )
    operation_phase = build_operation_phase(
        index=X.index,
        windows=maint_windows,
        pre_hours=PRE_MAINTENANCE_HOURS,
    ).astype(np.int8)

    return X, maint_windows, operation_phase


def _compute_pointwise_metrics(
    is_anomaly: pd.Series,
    operation_phase: pd.Series,
    eval_mask: Optional[pd.Series] = None,
) -> dict:
    """Compute confusion metrics with labels from operation_phase (1=pre-maint,0=normal)."""
    op_phase = operation_phase.reindex(is_anomaly.index)
    base_mask = op_phase.isin([0, 1]) & is_anomaly.notna()
    if eval_mask is not None:
        eval_mask = eval_mask.reindex(is_anomaly.index).fillna(False)
        mask = base_mask & eval_mask
    else:
        mask = base_mask

    if not mask.any():
        return {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "TN": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
        }

    y_true = (op_phase[mask] == 1).astype(int)
    y_pred = is_anomaly[mask].astype(int)
    return confusion_and_scores(y_true, y_pred)


def _evaluate_risk_series(
    maintenance_risk: pd.Series,
    maint_windows: List[Tuple],
    tag: str,
) -> Tuple[Optional[float], Optional[pd.Series], List[dict]]:
    """Grid-search risk thresholds and print event-level metrics."""
    risk_thresholds: List[float] = []
    risk_alarm_mask: Optional[pd.Series] = None
    best_risk_threshold: Optional[float] = None
    try:
        risk_thresholds = parse_risk_grid_spec(RISK_EVAL_GRID_SPEC)
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
                f"[{tag}] Rolling window={RISK_WINDOW_MINUTES} min, early-warning horizon={EARLY_WARNING_MINUTES} min"
            )
            for res in risk_results:
                print(
                    f"[{tag}] θ={res['threshold']:.2f}  Precision={res['precision']:.4f}  "
                    f"Recall={res['recall']:.4f}  F1={res['f1']:.4f}  TP={res['tp']}  FP={res['fp']}  FN={res['fn']}"
                )
            best = max(risk_results, key=lambda r: (r["f1"], r["precision"], -r["threshold"]))
            print(
                f"[{tag}] Best θ={best['threshold']:.2f}: Precision={best['precision']:.4f}  "
                f"Recall={best['recall']:.4f}  F1={best['f1']:.4f}  "
                f"TP={best['tp']}  FP={best['fp']}  FN={best['fn']}"
            )
            best_risk_threshold = float(best["threshold"])
            risk_alarm_mask = (maintenance_risk >= best_risk_threshold).astype(bool)
    else:
        risk_results = []

    return best_risk_threshold, risk_alarm_mask, risk_results


def _run_single_model_experiment(
    X: pd.DataFrame,
    maint_windows: List[Tuple],
    operation_phase: pd.Series,
) -> Tuple[pd.DataFrame, dict, Optional[pd.Timestamp], Optional[float], Optional[pd.Series]]:
    """Baseline: single global IF model trained once and scored over the full timeline."""
    pred_if, info = train_iforest_and_score(
        X=X,
        train_frac=TRAIN_FRAC,
        lpf_alpha=LPF_ALPHA,
    )

    pred = pred_if.copy()

    # Rolling maintenance risk from exceedance of the training threshold
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
        exceedance.rolling(f"{RISK_WINDOW_MINUTES}min", min_periods=1)
        .mean()
        .astype(np.float32)
        .fillna(0.0)
    ).rename("maintenance_risk")
    pred["maintenance_risk"] = maintenance_risk
    pred["operation_phase"] = operation_phase.reindex(pred.index).astype(np.int8)

    # Event-level evaluation of maintenance_risk thresholds
    best_risk_threshold, risk_alarm_mask, _ = _evaluate_risk_series(
        maintenance_risk, maint_windows, tag="RISK"
    )

    # Exact training cutoff timestamp on the prediction index
    train_cutoff_ts: Optional[pd.Timestamp] = None
    try:
        if info.get("n_train", 0) > 0 and info["n_train"] <= len(pred.index):
            train_cutoff_ts = pred.index[info["n_train"] - 1]
    except Exception:
        train_cutoff_ts = None

    # Point-wise metrics on normal + pre-maintenance
    m_all = _compute_pointwise_metrics(pred["is_anomaly"], pred["operation_phase"])
    print("[METRIC] Single-model (all evaluated rows):")
    print(
        f"         TP={m_all['TP']}  FP={m_all['FP']}  FN={m_all['FN']}  TN={m_all['TN']}"
    )
    print(
        f"         Precision={m_all['precision']:.4f}  Recall={m_all['recall']:.4f}  "
        f"F1={m_all['f1']:.4f}  Accuracy={m_all['accuracy']:.4f}"
    )

    # Console summary
    total_pts = int(pred["is_anomaly"].sum())
    pct_pts = float(100.0 * pred["is_anomaly"].mean())
    print(
        f"[INFO] Model rule: {info['label_rule']}, features={info['n_features']}, LPF alpha={info['lpf_alpha']}"
    )
    print(f"[INFO] Train size: {info['n_train']}/{info['n_total']}")
    print(f"[INFO] Point anomalies: {total_pts} ({pct_pts:.2f}%)")
    if info.get("threshold") is not None:
        print(f"[INFO] Train-based threshold: {info['threshold']:.4f}")

    return pred, info, train_cutoff_ts, best_risk_threshold, risk_alarm_mask


def _run_daily_models_experiment(
    X: pd.DataFrame,
    maint_windows: List[Tuple],
    operation_phase: pd.Series,
) -> Tuple[pd.DataFrame, dict, Optional[float], Optional[pd.Series]]:
    """
    Adaptive regime: train a new IF model for each calendar day, using all data
    up to the previous day as training, and evaluate day-by-day.
    """
    index = X.index
    day_periods = index.to_period("D")
    unique_days = day_periods.unique().sort_values()
    if len(unique_days) < 2:
        raise ValueError("Daily-mode requires at least two calendar days of data.")

    pred = pd.DataFrame(index=index)
    exceedance = pd.Series(index=index, dtype=float)

    daily_infos: List[dict] = []

    # First day is used only as training history; daily evaluation starts at second day.
    n_test_days = len(unique_days) - 1
    print(
        f"[INFO] Daily mode: {n_test_days} test days "
        f"(skipping initial training-only day {unique_days[0]})"
    )

    for day_idx, day in enumerate(unique_days[1:], start=1):
        train_mask = day_periods < day
        test_mask = day_periods == day
        if not train_mask.any() or not test_mask.any():
            continue

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]

        print(
            f"[INFO] Daily model {day_idx}/{n_test_days} for day {day}: "
            f"train_size={X_train.shape[0]}, test_size={X_test.shape[0]}"
        )

        # Train on all history up to previous day, score only current day.
        slice_pred, info = train_iforest_on_slices(
            X_train=X_train,
            X_all=X_test,
            lpf_alpha=LPF_ALPHA,
            random_state=42,
        )
        daily_infos.append({"day": str(day), **info})

        pred.loc[test_mask, "anom_score"] = slice_pred["anom_score"]
        if "anom_score_lpf" in slice_pred.columns:
            pred.loc[test_mask, "anom_score_lpf"] = slice_pred["anom_score_lpf"]
        pred.loc[test_mask, "is_anomaly"] = slice_pred["is_anomaly"]

        score_for_risk = (
            slice_pred["anom_score_lpf"]
            if "anom_score_lpf" in slice_pred.columns
            else slice_pred["anom_score"]
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
        exceedance_slice = (score_for_risk >= float(tau)).astype(float)
        exceedance.loc[test_mask] = exceedance_slice

    # Any points with no exceedance value (e.g., initial training-only days) are treated as zero risk.
    exceedance = exceedance.fillna(0.0)
    maintenance_risk = (
        exceedance.rolling(f"{RISK_WINDOW_MINUTES}min", min_periods=1)
        .mean()
        .astype(np.float32)
        .fillna(0.0)
    ).rename("maintenance_risk")
    pred["maintenance_risk"] = maintenance_risk
    pred["operation_phase"] = operation_phase.reindex(pred.index).astype(np.int8)

    # Event-level evaluation of maintenance_risk thresholds
    best_risk_threshold, risk_alarm_mask, _ = _evaluate_risk_series(
        maintenance_risk, maint_windows, tag="RISK-DAILY"
    )

    # Point-wise metrics on days with predictions (normal + pre-maintenance only)
    eval_mask = pred["is_anomaly"].notna()
    m_all = _compute_pointwise_metrics(pred["is_anomaly"], pred["operation_phase"], eval_mask=eval_mask)
    print("[METRIC] Daily-model (all evaluated rows across days):")
    print(
        f"         TP={m_all['TP']}  FP={m_all['FP']}  FN={m_all['FN']}  TN={m_all['TN']}"
    )
    print(
        f"         Precision={m_all['precision']:.4f}  Recall={m_all['recall']:.4f}  "
        f"F1={m_all['f1']:.4f}  Accuracy={m_all['accuracy']:.4f}"
    )

    # Summary of daily training sizes
    if daily_infos:
        n_days = len(daily_infos)
        min_train = min(info["n_train"] for info in daily_infos)
        max_train = max(info["n_train"] for info in daily_infos)
        print(
            f"[INFO] Daily models: {n_days} test days, training size range={min_train}..{max_train}"
        )

    summary_info = {
        "mode": "daily",
        "n_days": len(daily_infos),
    }
    return pred, summary_info, best_risk_threshold, risk_alarm_mask


def _run_per_maintenance_experiment(
    X: pd.DataFrame,
    maint_windows: List[Tuple],
    operation_phase: pd.Series,
) -> Tuple[pd.DataFrame, dict, Optional[float], Optional[pd.Series]]:
    """
    Adaptive regime by inter-maintenance period:

    - Define periods between consecutive maintenance windows (plus the initial
      period before the first and the final period after the last).
    - For period P_j (after maintenance window W_j), train on all historical
      rows with operation_phase==0 (normal) and timestamps up to the end of W_j.
    - For the initial period (before W1), train on normal rows strictly before
      the first maintenance start.
    - Evaluate on each period's rows, then aggregate metrics across periods.
    """
    if not maint_windows:
        raise ValueError("Per-maintenance mode requires maintenance windows.")

    # Ensure maintenance windows are sorted by start time
    mw_sorted = sorted(maint_windows, key=lambda w: pd.to_datetime(w[0]))
    starts = [pd.to_datetime(w[0]) for w in mw_sorted]
    ends = [pd.to_datetime(w[1]) for w in mw_sorted]

    index = X.index
    data_start = pd.to_datetime(index.min())
    data_end = pd.to_datetime(index.max())

    # Build inter-maintenance periods P0..Pk
    periods: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    # P0: start of data to start(W1)
    periods.append((data_start, starts[0], 0))
    # Intermediate periods: end(Wj) to start(Wj+1)
    for j in range(len(mw_sorted) - 1):
        periods.append((ends[j], starts[j + 1], j + 1))
    # Final period: end(last W) to end of data
    periods.append((ends[-1], data_end, len(mw_sorted)))

    pred = pd.DataFrame(index=index)
    exceedance = pd.Series(index=index, dtype=float)

    period_infos: List[dict] = []

    n_periods = len(periods)
    print(f"[INFO] Per-maint mode: {n_periods} inter-maintenance periods")

    # Precompute operation_phase aligned to X
    op_phase = operation_phase.reindex(index).astype(np.int8)

    for p_idx, (p_start, p_end, w_idx) in enumerate(periods):
        # Test mask: rows in this inter-maintenance period
        if w_idx == len(mw_sorted):
            # Last period includes end boundary
            test_mask = (index >= p_start) & (index <= p_end)
        else:
            test_mask = (index >= p_start) & (index < p_end)
        if not test_mask.any():
            continue

        # Training mask: accumulated normal history up to end of previous window
        if w_idx == 0:
            # For P0 (before first maintenance), train on normal rows before W1 start
            train_cutoff = starts[0]
            train_mask = (index < train_cutoff) & (op_phase == 0)
        else:
            # For P_j (after W_j), train on normal rows up to end(W_j)
            train_cutoff = ends[w_idx - 1]
            train_mask = (index <= train_cutoff) & (op_phase == 0)

        if not train_mask.any():
            continue

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]

        print(
            f"[INFO] Per-maint model {p_idx + 1}/{n_periods} "
            f"for period {p_start} → {p_end}: "
            f"train_size={X_train.shape[0]}, test_size={X_test.shape[0]}"
        )

        slice_pred, info = train_iforest_on_slices(
            X_train=X_train,
            X_all=X_test,
            lpf_alpha=LPF_ALPHA,
            random_state=42,
        )
        period_infos.append(
            {
                "period_index": p_idx,
                "train_size": info["n_train"],
                "test_size": X_test.shape[0],
            }
        )

        pred.loc[test_mask, "anom_score"] = slice_pred["anom_score"]
        if "anom_score_lpf" in slice_pred.columns:
            pred.loc[test_mask, "anom_score_lpf"] = slice_pred["anom_score_lpf"]
        pred.loc[test_mask, "is_anomaly"] = slice_pred["is_anomaly"]

        score_for_risk = (
            slice_pred["anom_score_lpf"]
            if "anom_score_lpf" in slice_pred.columns
            else slice_pred["anom_score"]
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
        exceedance_slice = (score_for_risk >= float(tau)).astype(float)
        exceedance.loc[test_mask] = exceedance_slice

    # Any points with no exceedance value (e.g., unscored periods) are treated as zero risk.
    exceedance = exceedance.fillna(0.0)
    maintenance_risk = (
        exceedance.rolling(f"{RISK_WINDOW_MINUTES}min", min_periods=1)
        .mean()
        .astype(np.float32)
        .fillna(0.0)
    ).rename("maintenance_risk")
    pred["maintenance_risk"] = maintenance_risk
    pred["operation_phase"] = op_phase

    # Event-level evaluation of maintenance_risk thresholds
    best_risk_threshold, risk_alarm_mask, _ = _evaluate_risk_series(
        maintenance_risk, mw_sorted, tag="RISK-PERMAINT"
    )

    # Point-wise metrics on rows with predictions (normal + pre-maintenance only)
    eval_mask = pred["is_anomaly"].notna()
    m_all = _compute_pointwise_metrics(
        pred["is_anomaly"], pred["operation_phase"], eval_mask=eval_mask
    )
    print("[METRIC] Per-maint-model (all evaluated rows across periods):")
    print(
        f"         TP={m_all['TP']}  FP={m_all['FP']}  FN={m_all['FN']}  TN={m_all['TN']}"
    )
    print(
        f"         Precision={m_all['precision']:.4f}  Recall={m_all['recall']:.4f}  "
        f"F1={m_all['f1']:.4f}  Accuracy={m_all['accuracy']:.4f}"
    )

    if period_infos:
        n_used = len(period_infos)
        min_train = min(info["train_size"] for info in period_infos)
        max_train = max(info["train_size"] for info in period_infos)
        print(
            f"[INFO] Per-maint models: {n_used} periods with models, "
            f"training size range={min_train}..{max_train}"
        )

    summary_info = {
        "mode": "per_maint",
        "n_periods": len(period_infos),
    }
    return pred, summary_info, best_risk_threshold, risk_alarm_mask


def main() -> None:
    # 1) Build rolling feature matrix and maintenance context
    X, maint_windows, operation_phase = _build_features_and_context()

    # 2) Run the selected experiment mode
    mode = EXPERIMENT_MODE.lower().strip()
    if mode not in {"single", "daily", "per_maint"}:
        raise ValueError(
            f"Unsupported EXPERIMENT_MODE={EXPERIMENT_MODE!r}; "
            f"use 'single', 'daily', or 'per_maint'."
        )

    if mode == "single":
        pred, info, train_cutoff_ts, best_risk_threshold, risk_alarm_mask = _run_single_model_experiment(
            X, maint_windows, operation_phase
        )
    elif mode == "daily":
        pred, info, best_risk_threshold, risk_alarm_mask = _run_daily_models_experiment(
            X, maint_windows, operation_phase
        )
        train_cutoff_ts = None
    else:
        pred, info, best_risk_threshold, risk_alarm_mask = _run_per_maintenance_experiment(
            X, maint_windows, operation_phase
        )
        train_cutoff_ts = None

    # 3) Plot risk timeline
    df_plot = pred[["maintenance_risk"]].copy()

    effective_show_labels = bool(SHOW_WINDOW_LABELS or USE_DEFAULT_METROPT_WINDOWS)
    plot_raw_timeline(
        df_plot,
        maint_windows,
        save_fig=SAVE_FIG_PATH,
        train_frac=TRAIN_FRAC if mode == "single" else None,
        train_cutoff_time=train_cutoff_ts,
        show_window_labels=effective_show_labels,
        window_label_fontsize=WINDOW_LABEL_FONTSIZE,
        window_label_format=WINDOW_LABEL_FORMAT,
        risk_alarm_mask=risk_alarm_mask,
        risk_threshold=best_risk_threshold,
        early_warning_minutes=EARLY_WARNING_MINUTES,
    )

    # 4) Optional: save per-point predictions (timestamp, score, labels, risk)
    if SAVE_PRED_CSV_PATH:
        out = pred.copy()
        out.index.name = "timestamp"
        out.to_csv(SAVE_PRED_CSV_PATH)

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
