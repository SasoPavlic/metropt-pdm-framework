#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-series anomaly explorer (native timeline plot, MetroPT-3).

- Loads CSV/TXT specific to this project (no MAT/Parquet to keep it simple).
- Rolling stats over all numeric features.
- Train on the first N minutes (chronological), score all points.
- Label by a robust (Q3 + 3*IQR) threshold on model scores.
- Plot ORIGINAL timeline with point colors (normal/anomalous) + shaded/marked failure windows with labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from data_utils import (
    build_operation_phase,
    build_rolling_features,
    load_csv,
    parse_maintenance_windows,
    select_numeric_features,
)
from metrics_event import evaluate_maintenance_prediction
from metrics_point import confusion_and_scores, evaluate_risk_thresholds
from detector_model import train_iforest_on_slices, _time_based_train_mask
from plotting import plot_raw_timeline, plot_lead_time_distribution, plot_pr_vs_lead_time


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


def _mode_output_path(path: Optional[str], mode: str) -> Optional[str]:
    """Prefix output filename with the experiment mode while preserving directory."""
    if not path:
        return None
    p = Path(path)
    if p.suffix:
        return str(p.with_name(f"{mode}_{p.name}"))
    return str(p / f"{mode}.png")


def _print_section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(title)
    print(bar)


# -----------------------------
# Configuration constants
# -----------------------------
# --- Data loading / preprocessing ---
# Timestamp column name in the input CSV; set to None to auto-detect.
INPUT_TIMESTAMP_COL: Optional[str] = None
# Whether to drop unnamed index columns commonly created by pandas when saving CSVs.
DROP_UNNAMED_INDEX: bool = True
# Input/outputs
INPUT_PATH: str = "datasets/MetroPT3.csv"
SAVE_FIG_PATH: Optional[str] = "plots/metropt3_raw.png"
SAVE_LEAD_TIME_DIST_PATH: Optional[str] = "plots/lead_time_distribution.png"
SAVE_PR_LEADTIME_PATH: Optional[str] = "plots/pr_vs_lead_time.png"
SAVE_PRED_CSV_PATH: Optional[str] = "datasets/metropt3_predictions.csv"
SAVE_FEATURES_CSV_PATH: Optional[str] = "datasets/metropt3_features.csv"

# Experiment mode:
# - "single": one global model trained once on an early slice.
# - "per_maint": per-maintenance models trained on fixed post-maintenance
#   training days for each cycle (plus an initial pre-W1 model).
EXPERIMENT_MODE: str = "per_maint"

# Rolling window for feature aggregation (e.g., '600s' = 10 minutes).
ROLLING_WINDOW: str = "60s"
# Duration (minutes) of the initial training window (chronological from the start).
TRAIN_FRAC: float = 1440
# --- Modeling / scoring ---
# Rolling risk window (minutes).
RISK_WINDOW_MINUTES: int = 480
# Quantile for extreme-point exceedance when building maintenance risk.
RISK_EXCEEDANCE_QUANTILE: float = 0.95
# Risk evaluation grid specification (start:stop:step).
RISK_EVAL_GRID_SPEC: str = "0.30:0.90:0.10"
# Lead-time step size for event-level curves (minutes).
LEAD_STEP_MINUTES: int = 10
# Length of the post-maintenance training interval (in minutes) for the
# per-maintenance regime. For each maintenance window W_j, the next model
# (if any) is trained on the interval [end_j, end_j + POST_MAINT_TRAIN_MINUTES),
# clipped so it does not intrude into the next maintenance window. The same
# time mask is excluded from point-wise evaluation in the single-model regime
# for fair comparison.
POST_MAINT_TRAIN_MINUTES: int = 1440

# --- Maintenance context / plotting ---
# Minutes before maintenance start considered pre-maintenance (operation_phase=1)
# and used as the early-warning horizon for event-level evaluation.
PRE_MAINTENANCE_MINUTES: int = 120
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

    # Base numeric features (all numeric columns, ordered by preference if provided)
    base_feats = select_numeric_features(
        df_raw,
        prefer=LIKELY_METROPT_FEATURES,
    )
    if not base_feats:
        raise ValueError("No numeric features found in the input data.")
    df_base = df_raw[base_feats].copy()

    # Rolling features on all numeric signals
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
        pre_minutes=PRE_MAINTENANCE_MINUTES,
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


def _compute_segmented_risk(
    risk_base: pd.Series,
    segment_masks: List[pd.Series],
) -> pd.Series:
    """Compute rolling risk within each segment, resetting at boundaries."""
    risk = pd.Series(0.0, index=risk_base.index, name="maintenance_risk")
    if risk_base.empty or not segment_masks:
        return risk

    for seg_mask in segment_masks:
        seg_mask = seg_mask.reindex(risk_base.index).fillna(False)
        if not seg_mask.any():
            continue
        seg_vals = risk_base[seg_mask].astype(float).fillna(0.0)
        seg_risk = (
            seg_vals.rolling(f"{RISK_WINDOW_MINUTES}min", min_periods=1)
            .mean()
            .astype(np.float32)
            .fillna(0.0)
        )
        risk.loc[seg_mask] = seg_risk

    return risk


def _ensure_bool_mask(mask: object, index: pd.Index) -> pd.Series:
    """Coerce a mask into a boolean Series aligned to index."""
    if isinstance(mask, pd.Series):
        out = mask.reindex(index).fillna(False)
    else:
        arr = np.asarray(mask, dtype=bool)
        if arr.shape[0] != len(index):
            raise ValueError("Mask length does not match index length.")
        out = pd.Series(arr, index=index)
    return out.astype(bool)


def _evaluate_risk_series(
    maintenance_risk: pd.Series,
    maint_windows: List[Tuple],
    tag: str,
) -> Tuple[Optional[float], Optional[pd.Series], List[dict], Optional[dict]]:
    """Grid-search risk thresholds and return best stats."""
    risk_thresholds: List[float] = []
    predicted_phase: Optional[pd.Series] = None
    best_risk_threshold: Optional[float] = None
    best_stats: Optional[dict] = None

    try:
        risk_thresholds = parse_risk_grid_spec(RISK_EVAL_GRID_SPEC)
    except ValueError as exc:
        print(f"[WARN] Skipping maintenance_risk evaluation: {exc}")

    if not risk_thresholds:
        return best_risk_threshold, predicted_phase, [], best_stats

    risk_results = evaluate_risk_thresholds(
        risk=maintenance_risk,
        maintenance_windows=maint_windows,
        thresholds=risk_thresholds,
        early_warning_minutes=PRE_MAINTENANCE_MINUTES,
    )
    if not risk_results:
        return best_risk_threshold, predicted_phase, [], best_stats

    best = max(risk_results, key=lambda r: (r["f1"], r["precision"], -r["threshold"]))
    best_stats = dict(best)
    best_risk_threshold = float(best["threshold"])
    predicted_phase = (maintenance_risk >= best_risk_threshold).astype(bool)
    return best_risk_threshold, predicted_phase, risk_results, best_stats


def _assign_predicted_phase(pred: pd.DataFrame, predicted_phase: Optional[pd.Series]) -> None:
    if predicted_phase is None:
        col = pd.Series(False, index=pred.index, dtype=bool)
    else:
        col = predicted_phase.reindex(pred.index).fillna(False).astype(bool)
    col = col.astype(np.int8)
    if "predicted_phase" in pred.columns:
        pred["predicted_phase"] = col
    elif "maintenance_risk" in pred.columns:
        insert_at = pred.columns.get_loc("maintenance_risk") + 1
        pred.insert(insert_at, "predicted_phase", col)
    else:
        pred["predicted_phase"] = col


def _assign_exceedance(pred: pd.DataFrame) -> None:
    """Add exceedance column based on risk_score and RISK_EXCEEDANCE_QUANTILE."""
    if "risk_score" not in pred.columns:
        return
    exceedance = (pred["risk_score"].astype(float) >= float(RISK_EXCEEDANCE_QUANTILE)).astype(np.int8)
    if "exceedance" in pred.columns:
        pred["exceedance"] = exceedance
    else:
        insert_at = pred.columns.get_loc("risk_score") + 1
        pred.insert(insert_at, "exceedance", exceedance)


def _print_pointwise_metrics(label: str, metrics: dict, threshold_info: Optional[str] = None) -> None:
    _print_section(f"POINT-WISE METRICS ({label})")
    if threshold_info:
        print(f"Threshold: {threshold_info}")
    print(
        f"TP={metrics['TP']}  FP={metrics['FP']}  FN={metrics['FN']}  TN={metrics['TN']}"
    )
    print(
        f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  "
        f"F1={metrics['f1']:.4f}  Accuracy={metrics['accuracy']:.4f}"
    )


def _print_event_level_metrics(
    label: str,
    best_stats: Optional[dict],
    risk_results: List[dict],
) -> None:
    _print_section(f"EVENT-LEVEL METRICS ({label})")
    print(
        f"Risk window={RISK_WINDOW_MINUTES} min  |  "
        f"Pre-maint horizon={PRE_MAINTENANCE_MINUTES} min  |  "
        f"Exceedance quantile={RISK_EXCEEDANCE_QUANTILE:.2f}"
    )
    if best_stats:
        print(
            f"Best θ={best_stats['threshold']:.2f}  "
            f"TP={best_stats['tp']}  FP={best_stats['fp']}  FN={best_stats['fn']}  "
            f"Precision={best_stats['precision']:.4f}  "
            f"Recall={best_stats['recall']:.4f}  F1={best_stats['f1']:.4f}"
        )
    else:
        print("Best θ=N/A (no risk results)")


def _print_event_extra_metrics(label: str, event_results: dict) -> None:
    _print_section(f"ADDITIONAL EVENT METRICS ({label})")
    ttd = event_results.get("ttd", {})
    faa = event_results.get("first_alarm_accuracy", {})
    far = event_results.get("far", {})
    cov = event_results.get("coverage", {})
    mtia = event_results.get("mtia", {})
    pr = event_results.get("pr_leadtime", {})

    if ttd.get("mean_ttd") is not None:
        print(
            f"TTD mean={ttd['mean_ttd']:.1f} min (std={ttd['std_ttd']:.1f}, "
            f"min={ttd['min_ttd']:.1f}, max={ttd['max_ttd']:.1f})"
        )
    else:
        print("TTD mean=N/A (no detections)")
    print(f"TTD detected={ttd.get('detected_events', 0)} missed={ttd.get('missed_events', 0)}")

    if faa.get("first_alarm_accuracy") is not None:
        print(
            f"FAA={faa['first_alarm_accuracy']:.3f} "
            f"({faa['first_alarm_in_window']}/{faa['tp_events']})"
        )
    else:
        print("FAA=N/A")

    if cov:
        print(
            f"Coverage={cov['alarm_coverage_percent']:.2f}% "
            f"({cov['alarm_points']}/{cov['total_points']})"
        )

    if mtia.get("mtia_minutes") is not None:
        print(
            f"MTIA mean={mtia['mtia_minutes']:.1f} min "
            f"(std={mtia['std_minutes']:.1f}, intervals={mtia['num_intervals']})"
        )
    else:
        print("MTIA mean=N/A (no alarm intervals)")

    if far.get("far_per_week") is not None:
        print(
            f"FAR per_day={far['far_per_day']:.3f}  "
            f"per_week={far['far_per_week']:.3f}  "
            f"FP intervals={far['fp_intervals']}"
        )

    if pr:
        print("PR vs Lead Time:")
        for i, lt in enumerate(pr.get("lead_times", [])):
            print(
                f"  {lt} min: P={pr['precision'][i]:.3f}, "
                f"R={pr['recall'][i]:.3f}, F1={pr['f1'][i]:.3f}"
            ) 


def _save_event_plots(event_results: dict, mode: str) -> None:
    if not event_results:
        return
    dist = event_results.get("lead_time_distribution", {})
    pr = event_results.get("pr_leadtime", {})

    dist_path = _mode_output_path(SAVE_LEAD_TIME_DIST_PATH, mode)
    pr_path = _mode_output_path(SAVE_PR_LEADTIME_PATH, mode)

    if dist_path:
        Path(dist_path).parent.mkdir(parents=True, exist_ok=True)
        plot_lead_time_distribution(
            dist,
            save_fig=dist_path,
            title=f"Lead Time Distribution ({mode})",
        )

    if pr_path:
        Path(pr_path).parent.mkdir(parents=True, exist_ok=True)
        plot_pr_vs_lead_time(
            pr,
            save_fig=pr_path,
            title=f"Precision-Recall vs Lead Time ({mode})",
        )


def _run_single_model_experiment(
    X: pd.DataFrame,
    maint_windows: List[Tuple],
    operation_phase: pd.Series,
) -> Tuple[pd.DataFrame, dict, Optional[pd.Timestamp], Optional[float], Optional[pd.Series]]:
    """Baseline: single global model trained once and scored over the full timeline."""
    # Time-based training window, restricted to non-maintenance rows (phases 0/1)
    train_time_mask = _time_based_train_mask(X.index, TRAIN_FRAC)
    op_phase = operation_phase.reindex(X.index).astype(np.int8)
    train_mask = train_time_mask & (op_phase != 2)
    if train_mask.sum() < 2:
        raise ValueError("Single-model training set has fewer than 2 samples after filtering.")

    pred_if, info = train_iforest_on_slices(
        X_train=X.loc[train_mask],
        X_all=X,
        random_state=42,
    )

    pred = pred_if.copy()

    # Rolling maintenance risk from normalized risk scores (extreme-point exceedance)
    risk_base = pred["risk_score"].astype(float).fillna(0.0)
    risk_base = (risk_base >= float(RISK_EXCEEDANCE_QUANTILE)).astype(float)
    maintenance_risk = _compute_segmented_risk(
        risk_base, [pd.Series(True, index=pred.index)]
    )
    pred["maintenance_risk"] = maintenance_risk
    pred["operation_phase"] = op_phase

    # Training-only rows used in any regime:
    # - the initial chronological TRAIN_FRAC minutes
    # - fixed post-maintenance training intervals for per-maint models
    initial_train_time_mask = train_time_mask.reindex(pred.index).fillna(False)
    post_maint_train_mask = _build_post_maintenance_train_mask(
        pred.index, maint_windows, POST_MAINT_TRAIN_MINUTES
    )
    eval_exclude_mask = initial_train_time_mask | post_maint_train_mask

    # Event-level evaluation of maintenance_risk thresholds
    best_risk_threshold, predicted_phase, risk_results, best_stats = _evaluate_risk_series(
        maintenance_risk, maint_windows, tag="RISK"
    )
    _assign_exceedance(pred)
    _assign_predicted_phase(pred, predicted_phase)

    # Exact training cutoff timestamp on the prediction index
    train_cutoff_ts: Optional[pd.Timestamp] = None
    try:
        if train_mask.any():
            train_cutoff_ts = train_mask[train_mask].index.max()
    except Exception:
        train_cutoff_ts = None

    # Point-wise metrics on normal + pre-maintenance, excluding training-only rows
    eval_mask = ~eval_exclude_mask
    m_all = _compute_pointwise_metrics(pred["is_anomaly"], pred["operation_phase"], eval_mask=eval_mask)
    threshold_info = None
    if info.get("threshold") is not None:
        threshold_info = f"{info.get('label_rule')} | value={info['threshold']:.4f}"
    _print_pointwise_metrics("Single", m_all, threshold_info=threshold_info)

    _print_event_level_metrics("Single", best_stats, risk_results)

    event_results = evaluate_maintenance_prediction(
        predictions=pred["predicted_phase"],
        maintenance_windows=maint_windows,
        early_warning_minutes=PRE_MAINTENANCE_MINUTES,
        method_name="Single",
        eval_mask=eval_mask,
        lead_step_minutes=LEAD_STEP_MINUTES,
    )
    info["event_metrics"] = event_results
    _print_event_extra_metrics("Single", event_results)
    _save_event_plots(event_results, mode="single")

    # Console summary
    total_pts = int(pred["is_anomaly"].sum())
    pct_pts = float(100.0 * pred["is_anomaly"].mean())
    print(f"[INFO] Model rule: {info['label_rule']}, features={info['n_features']}")
    print(f"[INFO] Train size: {info['n_train']}/{info['n_total']}")
    print(f"[INFO] Point anomalies: {total_pts} ({pct_pts:.2f}%)")
    if info.get("threshold") is not None:
        print(f"[INFO] Train-based threshold: {info['threshold']:.4f}")

    return pred, info, train_cutoff_ts, best_risk_threshold, predicted_phase




def _run_per_maintenance_experiment(
    X: pd.DataFrame,
    maint_windows: List[Tuple],
    operation_phase: pd.Series,
) -> Tuple[pd.DataFrame, dict, Optional[float], Optional[pd.Series]]:
    """
    Adaptive regime by maintenance cycles.

    - Model 0 (pre-W1) is trained on the same earliest TRAIN_FRAC minutes of
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
    risk_segments: List[pd.Series] = []

    period_infos: List[dict] = []

    print(
        f"[INFO] Per-maint mode: {len(mw_sorted) + 1} logical cycles "
        f"(pre-W1 plus one cycle per maintenance window)"
    )

    # Helper: fit the detector on X_train and score X_test; update per-point predictions.
    def _fit_and_score_segment(
        seg_label: str, train_mask: pd.Series, test_mask: pd.Series
    ) -> Optional[dict]:
        nonlocal pred, risk_segments

        train_mask = _ensure_bool_mask(train_mask, index)
        test_mask = _ensure_bool_mask(test_mask, index)

        if not test_mask.any():
            return None

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        if X_train.shape[0] < 2 or X_test.shape[0] < 2:
            return None


        slice_pred, info = train_iforest_on_slices(
            X_train=X_train,
            X_all=X_test,
            random_state=42,
        )

        pred.loc[test_mask, "anom_score"] = slice_pred["anom_score"]
        pred.loc[test_mask, "is_anomaly"] = slice_pred["is_anomaly"]
        if "risk_score" in slice_pred.columns:
            pred.loc[test_mask, "risk_score"] = slice_pred["risk_score"]

        risk_segments.append(test_mask)

        return {
            "segment": seg_label,
            "train_size": info["n_train"],
            "test_size": X_test.shape[0],
            "threshold": info.get("threshold"),
            "pct_anom": info.get("pct_anom"),
            "train_start": X_train.index.min(),
            "train_end": X_train.index.max(),
            "test_start": X_test.index.min(),
            "test_end": X_test.index.max(),
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

    for j, (start_ts, end_ts) in enumerate(zip(starts, ends)):
        # Score the maintenance window itself with the current model (risk only).
        maint_mask = (index >= pd.to_datetime(start_ts)) & (index <= pd.to_datetime(end_ts))
        seg_info = _fit_and_score_segment(f"maint_{j + 1}", current_train_mask, maint_mask)
        if seg_info:
            period_infos.append(seg_info)

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

    # Compute maintenance risk per segment (reset at segment boundaries).
    if "risk_score" in pred.columns:
        risk_base = pred["risk_score"].astype(float).fillna(0.0)
        risk_base = (risk_base >= float(RISK_EXCEEDANCE_QUANTILE)).astype(float)
    else:
        risk_base = pd.Series(0.0, index=index)
    maintenance_risk = _compute_segmented_risk(risk_base, risk_segments)
    pred["maintenance_risk"] = maintenance_risk
    pred["operation_phase"] = op_phase

    # Event-level evaluation of maintenance_risk thresholds
    best_risk_threshold, predicted_phase, risk_results, best_stats = _evaluate_risk_series(
        maintenance_risk, mw_sorted, tag="RISK-PERMAINT"
    )
    _assign_exceedance(pred)
    _assign_predicted_phase(pred, predicted_phase)

    # Point-wise metrics on rows with predictions (normal + pre-maintenance only),
    # excluding all training-only rows (initial slice + post-maint training days).
    eval_mask = pred["is_anomaly"].notna() & (~global_train_for_eval)
    m_all = _compute_pointwise_metrics(pred["is_anomaly"], pred["operation_phase"], eval_mask=eval_mask)
    threshold_info = None
    thresholds = [info["threshold"] for info in period_infos if info.get("threshold") is not None]
    if thresholds:
        threshold_info = (
            f"per-segment thresholds: min={min(thresholds):.4f}, "
            f"max={max(thresholds):.4f}, mean={np.mean(thresholds):.4f}"
        )
    _print_pointwise_metrics("Per-maint", m_all, threshold_info=threshold_info)

    _print_event_level_metrics("Per-maint", best_stats, risk_results)

    event_results = evaluate_maintenance_prediction(
        predictions=pred["predicted_phase"],
        maintenance_windows=mw_sorted,
        early_warning_minutes=PRE_MAINTENANCE_MINUTES,
        method_name="Per-maint",
        eval_mask=eval_mask,
        lead_step_minutes=LEAD_STEP_MINUTES,
    )
    _print_event_extra_metrics("Per-maint", event_results)
    _save_event_plots(event_results, mode="per_maint")

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
    summary_info["event_metrics"] = event_results
    return pred, summary_info, best_risk_threshold, predicted_phase


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
        pred, info, train_cutoff_ts, best_risk_threshold, predicted_phase = _run_single_model_experiment(
            X, maint_windows, operation_phase
        )
    else:
        pred, info, best_risk_threshold, predicted_phase = _run_per_maintenance_experiment(
            X, maint_windows, operation_phase
        )
        train_cutoff_ts = None

    # 3) Plot risk timeline
    df_plot = pred[["maintenance_risk"]].copy()

    effective_show_labels = bool(SHOW_WINDOW_LABELS or USE_DEFAULT_METROPT_WINDOWS)
    save_fig_path = _mode_output_path(SAVE_FIG_PATH, EXPERIMENT_MODE)
    if save_fig_path:
        Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plot_raw_timeline(
        df_plot,
        maint_windows,
        save_fig=save_fig_path,
        train_frac=TRAIN_FRAC if mode == "single" else None,
        train_cutoff_time=train_cutoff_ts,
        show_window_labels=effective_show_labels,
        window_label_fontsize=WINDOW_LABEL_FONTSIZE,
        window_label_format=WINDOW_LABEL_FORMAT,
        predicted_phase=predicted_phase,
        risk_threshold=best_risk_threshold,
        early_warning_minutes=PRE_MAINTENANCE_MINUTES,
    )

    # 4) Optional: save per-point predictions (timestamp, score, labels, risk)
    if SAVE_PRED_CSV_PATH:
        out = pred.copy()
        out.index.name = "timestamp"
        out.to_csv(SAVE_PRED_CSV_PATH)

    if maint_windows:
        rows = []
        for w in maint_windows:
            try:
                if len(w) >= 4:
                    s, e, wid, sev = w[0], w[1], w[2], w[3]
                else:
                    s, e = w[0], w[1]
                    wid, sev = None, None
                start = pd.to_datetime(s)
                end = pd.to_datetime(e)
                dur_min = int(round(max(0.0, (end - start).total_seconds() / 60.0)))
                rows.append(
                    {
                        "id": str(wid) if wid is not None else "",
                        "severity": str(sev) if sev is not None else "",
                        "start": start,
                        "end": end,
                        "duration_min": dur_min,
                    }
                )
            except Exception:
                rows.append(
                    {"id": "", "severity": "", "start": "", "end": "", "duration_min": ""}
                )
        print("[INFO] Failure windows:")
        df_windows = pd.DataFrame(rows, columns=["id", "severity", "start", "end", "duration_min"])
        try:
            print(df_windows.to_markdown(index=False))
        except Exception:
            print(df_windows.to_string(index=False))


if __name__ == "__main__":
    main()
