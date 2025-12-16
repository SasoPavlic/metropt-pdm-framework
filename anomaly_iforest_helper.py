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
from iforest_model import train_iforest_and_score, train_iforest_on_slices, _time_based_train_mask
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
# - "per_maint": per-maintenance models trained on fixed post-maintenance
#   training days for each cycle (plus an initial pre-W1 model).
EXPERIMENT_MODE: str = "per_maint"

# Optional time-based pre-downsampling rule (e.g., '60s') to regularize cadence before feature building; None disables.
PRE_DOWNSAMPLE_RULE: Optional[str] = None
# Rolling window for feature aggregation (e.g., '600s' = 10 minutes).
ROLLING_WINDOW: str = "60s"
# Fraction of earliest data used to train the IsolationForest.
# Duration (minutes) of the initial training window (chronological from the start).
# Previously this was a fraction; now it is interpreted as minutes.
TRAIN_FRAC: float = 1440
# Limit of most-variable base numeric features to keep before rolling aggregation.
MAX_BASE_FEATURES: int = 12
# Exclude near-binary numeric columns from features to focus on informative signals.
EXCLUDE_QUASI_BINARY: bool = False

# --- Modeling / scoring ---
# Exponential low-pass filter alpha for anomaly scores; 0 disables smoothing.
LPF_ALPHA: float = 0.4
# Rolling risk window (minutes).
RISK_WINDOW_MINUTES: int = 480
# Risk evaluation grid specification (start:stop:step).
RISK_EVAL_GRID_SPEC: str = "0.05:0.6:0.01"
# Early-warning horizon for risk evaluation (minutes).
EARLY_WARNING_MINUTES: int = 120
# Length of the post-maintenance training interval (in minutes) for the
# per-maintenance regime. For each maintenance window W_j, the next model
# (if any) is trained on the interval [end_j, end_j + POST_MAINT_TRAIN_MINUTES),
# clipped so it does not intrude into the next maintenance window. The same
# time mask is excluded from point-wise evaluation in the single-model regime
# for fair comparison.
POST_MAINT_TRAIN_MINUTES: int = 1440

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


def _build_post_maintenance_train_mask(
    index: pd.DatetimeIndex,
    maint_windows: List[Tuple],
    train_minutes: int,
) -> pd.Series:
    """
    Build a boolean mask marking post-maintenance training intervals.

    For each maintenance window W_j with end time e_j, we mark the interval
    [e_j, e_j + train_minutes) as training-only, clipped so that it does not
    intrude into the next maintenance window. This mask is used to exclude
    post-maintenance training time from point-wise evaluation in the
    single-model regime. The per-maintenance regime may further refine which
    parts of these intervals are actually used for training when gaps are very
    short.
    """
    mask = pd.Series(False, index=index)
    if not maint_windows or train_minutes <= 0:
        return mask

    mw_sorted = sorted(maint_windows, key=lambda w: pd.to_datetime(w[0]))
    starts = [pd.to_datetime(w[0]) for w in mw_sorted]
    ends = [pd.to_datetime(w[1]) for w in mw_sorted]

    for j, end_ts in enumerate(ends):
        t_start = pd.to_datetime(end_ts)
        t_stop = t_start + pd.Timedelta(minutes=float(train_minutes))
        if j < len(starts) - 1:
            next_start = pd.to_datetime(starts[j + 1])
            if t_stop > next_start:
                t_stop = next_start
        slice_mask = (index >= t_start) & (index < t_stop)
        mask |= slice_mask

    return mask


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
            # Diagnostics: for each failure window, show max risk inside the
            # early-warning horizon. This helps understand why TP may be zero.
            try:
                horizon = pd.to_timedelta(float(max(0.0, EARLY_WARNING_MINUTES)), unit="m")
                windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
                for w in maint_windows or []:
                    if len(w) < 2:
                        continue
                    s = pd.to_datetime(w[0])
                    e = pd.to_datetime(w[1])
                    if pd.isna(s) or pd.isna(e) or e < s:
                        continue
                    windows.append((s, e))
                if windows:
                    print(f"[{tag}] Per-window max risk within early-warning horizon:")
                    for idx, (s, e) in enumerate(windows, start=1):
                        ew_start = s - horizon
                        rslice = maintenance_risk.loc[ew_start:e]
                        if rslice.empty:
                            print(f"[{tag}]   window #{idx}: no risk data in horizon")
                        else:
                            max_val = float(rslice.max())
                            t_max = rslice.idxmax()
                            print(
                                f"[{tag}]   window #{idx}: max_risk={max_val:.4f} at {t_max}"
                            )
            except Exception as exc:
                print(f"[WARN] Risk diagnostics failed for tag {tag!r}: {exc}")
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
        train_minutes=TRAIN_FRAC,
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

    # Training-only rows used in any regime:
    # - the initial chronological TRAIN_FRAC minutes (same as train_iforest_and_score)
    # - fixed post-maintenance training intervals for per-maint models
    initial_train_time_mask = _time_based_train_mask(pred.index, TRAIN_FRAC)
    post_maint_train_mask = _build_post_maintenance_train_mask(
        pred.index, maint_windows, POST_MAINT_TRAIN_MINUTES
    )
    eval_exclude_mask = initial_train_time_mask | post_maint_train_mask

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

    # Point-wise metrics on normal + pre-maintenance, excluding training-only rows
    eval_mask = ~eval_exclude_mask
    m_all = _compute_pointwise_metrics(
        pred["is_anomaly"], pred["operation_phase"], eval_mask=eval_mask
    )
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




def _run_per_maintenance_experiment(
    X: pd.DataFrame,
    maint_windows: List[Tuple],
    operation_phase: pd.Series,
) -> Tuple[pd.DataFrame, dict, Optional[float], Optional[pd.Series]]:
    """
    Adaptive regime by maintenance cycles.

    - Model 0 (pre-W1) is trained on the same earliest TRAIN_FRAC fraction of
      the timeline as the single-model regime (chronological order), but with
      maintenance rows (operation_phase==2) excluded from the training set.
    - After each maintenance window W_j, we reserve a fixed amount of time
      immediately after the window (POST_MAINT_TRAIN_MINUTES) as a
      training-only interval for the *next* model, provided the gap to the
      next maintenance is long enough.
    - That model is then used to score the remaining time until the next
      maintenance. If the gap is shorter than POST_MAINT_TRAIN_MINUTES, no
      new model is trained and the previous model is reused to score the gap.
    - All training-only intervals (initial TRAIN_FRAC slice and
      post-maintenance training intervals) are excluded from point-wise
      evaluation in all regimes.
    """
    if not maint_windows:
        raise ValueError("Per-maintenance mode requires maintenance windows.")

    # Ensure maintenance windows are sorted by start time
    mw_sorted = sorted(maint_windows, key=lambda w: pd.to_datetime(w[0]))
    starts = [pd.to_datetime(w[0]) for w in mw_sorted]
    ends = [pd.to_datetime(w[1]) for w in mw_sorted]

    index = X.index
    op_phase = operation_phase.reindex(index).astype(np.int8)
    if index.empty:
        raise ValueError("Empty index in per-maintenance experiment.")

    # Initial training window (shared with the single-model regime) – this is
    # the global baseline that every per-maintenance model sees.
    initial_train_time_mask = _time_based_train_mask(index, TRAIN_FRAC)
    initial_train_order_mask = initial_train_time_mask.copy()
    # For regime 2 we only exclude maintenance rows from training.
    initial_train_mask = initial_train_time_mask & (op_phase != 2)

    # Post-maintenance training schedule (time-based) and global training-only mask
    post_maint_train_mask = _build_post_maintenance_train_mask(
        index, mw_sorted, POST_MAINT_TRAIN_MINUTES
    )
    global_train_for_eval = initial_train_order_mask | post_maint_train_mask

    pred = pd.DataFrame(index=index)
    exceedance = pd.Series(index=index, dtype=float)

    period_infos: List[dict] = []

    print(
        f"[INFO] Per-maint mode: {len(mw_sorted) + 1} logical cycles "
        f"(pre-W1 plus one cycle per maintenance window)"
    )

    # Helper: fit IF on X_train and score X_test; update pred/exceedance.
    def _fit_and_score_segment(
        seg_label: str, train_mask: pd.Series, test_mask: pd.Series
    ) -> Optional[dict]:
        nonlocal pred, exceedance

        if not test_mask.any():
            return None

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        if X_train.shape[0] < 2 or X_test.shape[0] < 2:
            return None

        print(
            f"[INFO] Per-maint segment {seg_label}: "
            f"train_size={X_train.shape[0]}, test_size={X_test.shape[0]}"
        )

        slice_pred, info = train_iforest_on_slices(
            X_train=X_train,
            X_all=X_test,
            lpf_alpha=LPF_ALPHA,
            random_state=42,
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

        return {
            "segment": seg_label,
            "train_size": info["n_train"],
            "test_size": X_test.shape[0],
        }

    # Model 0: pre-W1 region, trained on the initial TRAIN_FRAC slice only.
    first_start = starts[0]
    pre_w1_test_mask = (index < first_start) & (~initial_train_order_mask)
    current_train_mask = initial_train_mask

    seg_info = _fit_and_score_segment("pre_W1", current_train_mask, pre_w1_test_mask)
    if seg_info:
        period_infos.append(seg_info)

    # Iterate over maintenance windows; after each, either train a new model on
    # its post-maintenance training interval (plus the global baseline) or
    # reuse the previous model for short gaps.
    data_start = index.min()
    data_end = index.max()

    for j, end_ts in enumerate(ends):
        gap_start = pd.to_datetime(end_ts)
        if j < len(starts) - 1:
            gap_end = pd.to_datetime(starts[j + 1])
        else:
            gap_end = data_end

        # Gap between end of this maintenance and the next maintenance (or end of data)
        if gap_end <= gap_start:
            print(f"[INFO] No gap after maint #{j + 1} (gap_end <= gap_start).")
            continue

        gap_minutes = (gap_end - gap_start).total_seconds() / 60.0
        gap_mask = (index > gap_start) & (index < gap_end if j < len(starts) - 1 else index <= gap_end)
        if not gap_mask.any():
            print(f"[INFO] No candidate points after maint #{j + 1} (gap has no data).")
            continue

        if POST_MAINT_TRAIN_MINUTES <= 0 or gap_minutes <= POST_MAINT_TRAIN_MINUTES:
            # Gap too short for a dedicated post-maintenance training block:
            # reuse the previous model (which already includes the global
            # baseline and its last post-maint block in its training mask).
            seg_label = f"gap_after_maint_{j + 1}"
            seg_info = _fit_and_score_segment(seg_label, current_train_mask, gap_mask)
            if seg_info:
                period_infos.append(seg_info)
            continue

        # Enough gap to fit a new model: split into a training sub-interval and the remaining test interval.
        train_end = gap_start + pd.Timedelta(minutes=float(POST_MAINT_TRAIN_MINUTES))
        if train_end > gap_end:
            train_end = gap_end

        # Local post-maintenance block for this cycle (non-maintenance points)
        local_post_mask = (
            (index > gap_start)
            & (index <= train_end)
            & (index <= gap_end)
            & (op_phase != 2)
        )
        # Per-cycle training = global baseline + local post-maint block.
        train_mask_new = initial_train_mask | local_post_mask
        test_mask_new = (index > train_end) & (index < gap_end if j < len(starts) - 1 else index <= gap_end)

        seg_label = f"after_maint_{j + 1}"
        seg_info = _fit_and_score_segment(seg_label, train_mask_new, test_mask_new)
        if seg_info:
            period_infos.append(seg_info)
            current_train_mask = train_mask_new

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

    # Point-wise metrics on rows with predictions (normal + pre-maintenance only),
    # excluding all training-only rows (initial slice + post-maint training days).
    eval_mask = pred["is_anomaly"].notna() & (~global_train_for_eval)
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
            f"[INFO] Per-maint models: {n_used} segments with models, "
            f"training size range={min_train}..{max_train}"
        )

    summary_info = {
        "mode": "per_maint",
        "n_segments": len(period_infos),
    }
    return pred, summary_info, best_risk_threshold, risk_alarm_mask


def main() -> None:
    # 1) Build rolling feature matrix and maintenance context
    X, maint_windows, operation_phase = _build_features_and_context()

    # 2) Run the selected experiment mode
    mode = EXPERIMENT_MODE.lower().strip()
    if mode not in {"single", "per_maint"}:
        raise ValueError(
            f"Unsupported EXPERIMENT_MODE={EXPERIMENT_MODE!r}; "
            f"use 'single' or 'per_maint'."
        )

    if mode == "single":
        pred, info, train_cutoff_ts, best_risk_threshold, risk_alarm_mask = _run_single_model_experiment(
            X, maint_windows, operation_phase
        )
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
