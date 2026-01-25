#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event-level metrics for maintenance prediction using alarm intervals.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _normalize_windows(
    maintenance_windows: List[Tuple],
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for w in maintenance_windows or []:
        if len(w) < 2:
            continue
        s = pd.to_datetime(w[0])
        e = pd.to_datetime(w[1])
        if pd.isna(s) or pd.isna(e) or e < s:
            continue
        windows.append((s, e))
    windows.sort(key=lambda t: t[0])
    return windows


def _alarm_intervals_from_mask(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return contiguous intervals where mask is True."""
    if mask.empty:
        return []
    idx = mask.index
    arr = mask.to_numpy(dtype=bool)
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    in_run = False
    run_start: Optional[pd.Timestamp] = None
    for i, flag in enumerate(arr):
        if flag and not in_run:
            in_run = True
            run_start = idx[i]
        elif not flag and in_run:
            intervals.append((run_start, idx[i - 1]))
            in_run = False
            run_start = None
    if in_run and run_start is not None:
        intervals.append((run_start, idx[-1]))
    return intervals


def _mask_total_minutes(mask: pd.Series) -> float:
    """Total duration (minutes) covered by True mask intervals."""
    mask = mask.astype(bool)
    intervals = _alarm_intervals_from_mask(mask)
    total_minutes = 0.0
    for start, end in intervals:
        total_minutes += max(0.0, (end - start).total_seconds() / 60.0)
    return total_minutes


def calculate_ttd(
    predictions: pd.Series,
    maintenance_windows: List[Tuple],
    early_warning_minutes: int,
) -> Dict:
    """Strict TTD: first alarm in [start - early_warning, start]."""
    windows = _normalize_windows(maintenance_windows)
    ttd_values: List[float] = []
    detected_events = 0
    missed_events = 0

    for maint_start, _maint_end in windows:
        warning_start = maint_start - pd.Timedelta(minutes=early_warning_minutes)
        window_predictions = predictions.loc[warning_start:maint_start]
        alarms_in_window = window_predictions[window_predictions == 1]
        if not alarms_in_window.empty:
            first_alarm_time = alarms_in_window.index[0]
            ttd_minutes = (maint_start - first_alarm_time).total_seconds() / 60.0
            ttd_values.append(ttd_minutes)
            detected_events += 1
        else:
            missed_events += 1

    if ttd_values:
        return {
            "ttd_values": ttd_values,
            "mean_ttd": float(np.mean(ttd_values)),
            "std_ttd": float(np.std(ttd_values)),
            "min_ttd": float(np.min(ttd_values)),
            "max_ttd": float(np.max(ttd_values)),
            "median_ttd": float(np.median(ttd_values)),
            "detected_events": detected_events,
            "missed_events": missed_events,
        }
    return {
        "ttd_values": [],
        "mean_ttd": None,
        "std_ttd": None,
        "min_ttd": None,
        "max_ttd": None,
        "median_ttd": None,
        "detected_events": 0,
        "missed_events": missed_events,
    }


def lead_time_distribution(ttd_values: List[float], bins: List[int]) -> Dict:
    """Return histogram counts and bin edges for TTD values."""
    if not ttd_values:
        return {"bins": bins, "counts": [0] * (len(bins) - 1)}
    counts, edges = np.histogram(ttd_values, bins=bins)
    return {"bins": list(edges), "counts": counts.tolist()}


def calculate_first_alarm_accuracy(
    predictions: pd.Series,
    maintenance_windows: List[Tuple],
    early_warning_minutes: int,
) -> Dict:
    """Fraction of TP events where first alarm starts within strict pre-window."""
    windows = _normalize_windows(maintenance_windows)
    tp_events = 0
    first_alarm_in_window = 0

    for maint_start, maint_end in windows:
        warning_start = maint_start - pd.Timedelta(minutes=early_warning_minutes)
        window_predictions = predictions.loc[warning_start:maint_start]
        alarms_extended = window_predictions[window_predictions == 1]
        if alarms_extended.empty:
            continue
        tp_events += 1
        first_alarm_time = alarms_extended.index[0]
        if warning_start <= first_alarm_time <= maint_start:
            first_alarm_in_window += 1

    accuracy = first_alarm_in_window / tp_events if tp_events > 0 else None
    return {
        "first_alarm_accuracy": accuracy,
        "tp_events": tp_events,
        "first_alarm_in_window": first_alarm_in_window,
    }


def calculate_far(
    predictions: pd.Series,
    maintenance_windows: List[Tuple],
    early_warning_minutes: int,
    eval_mask: Optional[pd.Series] = None,
) -> Dict:
    """False Alarm Rate per day/week on evaluation intervals only."""
    windows = _normalize_windows(maintenance_windows)
    alarm_intervals = _alarm_intervals_from_mask(predictions.astype(bool))

    fp_intervals = 0
    for alarm_start, alarm_end in alarm_intervals:
        is_tp = False
        for maint_start, maint_end in windows:
            warning_start = maint_start - pd.Timedelta(minutes=early_warning_minutes)
            window_end = maint_start
            if alarm_start <= window_end and alarm_end >= warning_start:
                is_tp = True
                break
        if not is_tp:
            fp_intervals += 1

    if eval_mask is not None:
        total_minutes = _mask_total_minutes(eval_mask)
    else:
        total_minutes = _mask_total_minutes(pd.Series(True, index=predictions.index))

    total_days = total_minutes / (24.0 * 60.0) if total_minutes > 0 else 0.0
    total_weeks = total_days / 7.0 if total_days > 0 else 0.0

    return {
        "total_alarm_intervals": len(alarm_intervals),
        "fp_intervals": fp_intervals,
        "far_per_day": fp_intervals / total_days if total_days > 0 else None,
        "far_per_week": fp_intervals / total_weeks if total_weeks > 0 else None,
        "total_days": total_days,
        "total_weeks": total_weeks,
    }


def calculate_alarm_coverage(
    predictions: pd.Series,
    eval_mask: Optional[pd.Series] = None,
) -> Dict:
    """Share of evaluation points with prediction == 1."""
    if eval_mask is not None:
        mask = eval_mask.astype(bool)
        total_points = int(mask.sum())
        alarm_points = int((predictions.astype(int) & mask.astype(int)).sum())
    else:
        total_points = int(len(predictions))
        alarm_points = int((predictions == 1).sum())

    coverage = alarm_points / total_points if total_points > 0 else 0.0
    return {
        "alarm_coverage": coverage,
        "alarm_coverage_percent": coverage * 100.0,
        "alarm_points": alarm_points,
        "total_points": total_points,
    }


def calculate_mtia(predictions: pd.Series) -> Dict:
    """Mean time in alarm (minutes) based on alarm intervals."""
    alarm_intervals = _alarm_intervals_from_mask(predictions.astype(bool))
    if not alarm_intervals:
        return {
            "mtia_minutes": None,
            "std_minutes": None,
            "min_minutes": None,
            "max_minutes": None,
            "median_minutes": None,
            "num_intervals": 0,
            "durations": [],
        }

    durations: List[float] = []
    for start, end in alarm_intervals:
        duration_minutes = (end - start).total_seconds() / 60.0
        durations.append(duration_minutes)

    return {
        "mtia_minutes": float(np.mean(durations)),
        "std_minutes": float(np.std(durations)),
        "min_minutes": float(np.min(durations)),
        "max_minutes": float(np.max(durations)),
        "median_minutes": float(np.median(durations)),
        "num_intervals": len(durations),
        "durations": durations,
    }


def calculate_nab_score(
    predictions: pd.Series,
    maintenance_windows: List[Tuple],
    early_warning_minutes: int,
    profile: str = "standard",
) -> Dict:
    """
    Calculate NAB score using a strict pre-maintenance window:
    [maintenance_start - early_warning, maintenance_start].
    """
    profiles = {
        "standard": {"A_tp": 1.0, "A_fp": 0.11, "A_fn": 1.0},
        "low_fp": {"A_tp": 1.0, "A_fp": 0.22, "A_fn": 1.0},
        "low_fn": {"A_tp": 1.0, "A_fp": 0.11, "A_fn": 2.0},
    }
    if profile not in profiles:
        raise ValueError(f"Unknown NAB profile: {profile!r}")
    params = profiles[profile]

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    windows = _normalize_windows(maintenance_windows)
    alarm_intervals = _alarm_intervals_from_mask(predictions.astype(bool))

    total_score = 0.0
    window_scores: List[float] = []
    used_alarms: set[int] = set()

    for maint_start, maint_end in windows:
        warning_start = maint_start - pd.Timedelta(minutes=early_warning_minutes)
        window_size = float(early_warning_minutes)

        best_score = -params["A_fn"]
        best_alarm_idx: Optional[int] = None

        for idx, (alarm_start, alarm_end) in enumerate(alarm_intervals):
            if idx in used_alarms:
                continue
            # Strict window overlap: [warning_start, maint_start]
            if alarm_start <= maint_start and alarm_end >= warning_start:
                detection_time = max(alarm_start, warning_start)
                relative_pos = (
                    (detection_time - warning_start).total_seconds() / 60.0 / window_size
                )
                score = params["A_tp"] * sigmoid(-5.0 * (relative_pos - 0.5))
                if score > best_score:
                    best_score = score
                    best_alarm_idx = idx

        if best_alarm_idx is not None:
            used_alarms.add(best_alarm_idx)

        total_score += best_score
        window_scores.append(best_score)

    # Penalize unused alarm intervals as FP
    for idx in range(len(alarm_intervals)):
        if idx not in used_alarms:
            total_score -= params["A_fp"]

    max_possible = len(windows) * params["A_tp"]
    min_possible = -len(windows) * params["A_fn"] - len(alarm_intervals) * params["A_fp"]
    if max_possible != min_possible:
        normalized_score = 100.0 * (total_score - min_possible) / (max_possible - min_possible)
    else:
        normalized_score = 0.0

    return {
        "nab_score_raw": float(total_score),
        "nab_score_normalized": float(normalized_score),
        "profile": profile,
        "window_scores": window_scores,
        "num_fp": len(alarm_intervals) - len(used_alarms),
    }


def calculate_precision_recall_vs_leadtime(
    predictions: pd.Series,
    maintenance_windows: List[Tuple],
    lead_times: List[int],
    base_early_warning: int,
) -> Dict:
    """Precision/Recall for multiple required lead times."""
    windows = _normalize_windows(maintenance_windows)
    alarm_intervals = _alarm_intervals_from_mask(predictions.astype(bool))

    results = {
        "lead_times": lead_times,
        "precision": [],
        "recall": [],
        "f1": [],
        "tp": [],
        "fp": [],
        "fn": [],
    }

    for required_lead in lead_times:
        tp = 0
        fn = 0
        used_alarms: set[int] = set()

        for maint_start, maint_end in windows:
            warning_start = maint_start - pd.Timedelta(minutes=base_early_warning)
            required_deadline = maint_start - pd.Timedelta(minutes=required_lead)

            detected = False
            for idx, (alarm_start, alarm_end) in enumerate(alarm_intervals):
                if idx in used_alarms:
                    continue
                if alarm_start <= maint_end and alarm_end >= warning_start:
                    if alarm_start <= required_deadline:
                        detected = True
                        used_alarms.add(idx)
                        break

            if detected:
                tp += 1
            else:
                fn += 1

        fp = len(alarm_intervals) - len(used_alarms)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1"].append(f1)
        results["tp"].append(tp)
        results["fp"].append(fp)
        results["fn"].append(fn)

    return results


def calculate_event_level_scores(
    predictions: pd.Series,
    maintenance_windows: List[Tuple],
    early_warning_minutes: int,
) -> Dict:
    """Event-level TP/FP/FN/precision/recall/f1 based on alarm intervals."""
    windows = _normalize_windows(maintenance_windows)
    alarm_intervals = _alarm_intervals_from_mask(predictions.astype(bool))
    horizon = pd.Timedelta(minutes=early_warning_minutes)
    alarm_used = [False] * len(alarm_intervals)
    tp = 0
    fn = 0

    for maint_start, maint_end in windows:
        matched = False
        window_start = maint_start - horizon
        window_end = maint_start
        for i, (alarm_start, alarm_end) in enumerate(alarm_intervals):
            if alarm_used[i]:
                continue
            if alarm_start <= window_end and alarm_end >= window_start:
                alarm_used[i] = True
                tp += 1
                matched = True
                break
        if not matched:
            fn += 1

    fp = sum(1 for used in alarm_used if not used)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_maintenance_prediction(
    predictions: pd.Series,
    maintenance_windows: List[Tuple],
    early_warning_minutes: int,
    method_name: str = "Model",
    eval_mask: Optional[pd.Series] = None,
    lead_step_minutes: int = 30,
) -> Dict:
    """
    Evaluate event-level maintenance prediction metrics (no NAB).
    """
    if predictions.empty:
        return {"method_name": method_name}

    predictions = predictions.astype(bool)
    if eval_mask is not None:
        eval_mask = eval_mask.reindex(predictions.index).fillna(False).astype(bool)
        predictions = predictions & eval_mask

    lead_times = list(range(lead_step_minutes, early_warning_minutes + 1, lead_step_minutes))
    if lead_times[-1] != early_warning_minutes:
        lead_times.append(early_warning_minutes)
    bins = list(range(0, early_warning_minutes + lead_step_minutes, lead_step_minutes))
    if bins[-1] != early_warning_minutes:
        bins.append(early_warning_minutes)
    bins = sorted(set(bins))

    results = {"method_name": method_name}
    ttd = calculate_ttd(predictions, maintenance_windows, early_warning_minutes)
    results["ttd"] = ttd
    results["lead_time_distribution"] = lead_time_distribution(ttd["ttd_values"], bins)

    faa = calculate_first_alarm_accuracy(predictions, maintenance_windows, early_warning_minutes)
    results["first_alarm_accuracy"] = faa

    far = calculate_far(predictions, maintenance_windows, early_warning_minutes, eval_mask=eval_mask)
    results["far"] = far

    coverage = calculate_alarm_coverage(predictions, eval_mask=eval_mask)
    results["coverage"] = coverage

    mtia = calculate_mtia(predictions)
    results["mtia"] = mtia

    nab_standard = calculate_nab_score(
        predictions, maintenance_windows, early_warning_minutes, "standard"
    )
    nab_low_fp = calculate_nab_score(
        predictions, maintenance_windows, early_warning_minutes, "low_fp"
    )
    nab_low_fn = calculate_nab_score(
        predictions, maintenance_windows, early_warning_minutes, "low_fn"
    )
    results["nab"] = {
        "standard": nab_standard,
        "low_fp": nab_low_fp,
        "low_fn": nab_low_fn,
    }

    pr_leadtime = calculate_precision_recall_vs_leadtime(
        predictions, maintenance_windows, lead_times, early_warning_minutes
    )
    results["pr_leadtime"] = pr_leadtime

    event_scores = calculate_event_level_scores(predictions, maintenance_windows, early_warning_minutes)
    results["event_scores"] = event_scores

    return results


def generate_results_table(results_list: List[Dict]) -> pd.DataFrame:
    """Generate article-ready table of key metrics."""
    rows = []
    for r in results_list:
        row = {
            "Method": r.get("method_name", "Model"),
            "Mean TTD (min)": f"{r['ttd']['mean_ttd']:.1f}"
            if r.get("ttd", {}).get("mean_ttd") is not None
            else "N/A",
            "Std TTD": f"±{r['ttd']['std_ttd']:.1f}"
            if r.get("ttd", {}).get("std_ttd") is not None
            else "",
            "FAR/week": f"{r['far']['far_per_week']:.2f}"
            if r.get("far", {}).get("far_per_week") is not None
            else "N/A",
            "Coverage (%)": f"{r['coverage']['alarm_coverage_percent']:.1f}"
            if r.get("coverage")
            else "N/A",
            "MTIA (min)": f"{r['mtia']['mtia_minutes']:.1f}"
            if r.get("mtia", {}).get("mtia_minutes") is not None
            else "N/A",
            "NAB Score": f"{r['nab']['standard']['nab_score_normalized']:.1f}"
            if r.get("nab", {}).get("standard")
            else "N/A",
            "FAA": f"{r['first_alarm_accuracy']['first_alarm_accuracy']:.2f}"
            if r.get("first_alarm_accuracy", {}).get("first_alarm_accuracy") is not None
            else "N/A",
        }
        rows.append(row)
    return pd.DataFrame(rows)
