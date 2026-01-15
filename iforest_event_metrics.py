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
        extended_predictions = predictions.loc[warning_start:maint_end]
        alarms_extended = extended_predictions[extended_predictions == 1]
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
            if alarm_start <= maint_end and alarm_end >= warning_start:
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
        window_end = maint_end
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

    pr_leadtime = calculate_precision_recall_vs_leadtime(
        predictions, maintenance_windows, lead_times, early_warning_minutes
    )
    results["pr_leadtime"] = pr_leadtime

    event_scores = calculate_event_level_scores(predictions, maintenance_windows, early_warning_minutes)
    results["event_scores"] = event_scores

    print(f"\n{'=' * 60}")
    print(f"EVALUATION: {method_name}")
    print(f"{'=' * 60}")

    if ttd["mean_ttd"] is not None:
        print(
            f"[TTD] mean={ttd['mean_ttd']:.1f} min, std={ttd['std_ttd']:.1f}, "
            f"min={ttd['min_ttd']:.1f}, max={ttd['max_ttd']:.1f}"
        )
    else:
        print("[TTD] mean=N/A (no detections)")
    print(f"[TTD] detected={ttd['detected_events']} missed={ttd['missed_events']}")

    if faa["first_alarm_accuracy"] is not None:
        print(
            f"[FAA] accuracy={faa['first_alarm_accuracy']:.3f} "
            f"({faa['first_alarm_in_window']}/{faa['tp_events']})"
        )
    else:
        print("[FAA] accuracy=N/A")

    if far["far_per_day"] is not None:
        print(
            f"[FAR] per_day={far['far_per_day']:.3f}, per_week={far['far_per_week']:.3f}, "
            f"fp_intervals={far['fp_intervals']}"
        )
    else:
        print("[FAR] per_day=N/A per_week=N/A")

    print(
        f"[COV] coverage={coverage['alarm_coverage_percent']:.2f}% "
        f"({coverage['alarm_points']}/{coverage['total_points']})"
    )

    if mtia["mtia_minutes"] is not None:
        print(
            f"[MTIA] mean={mtia['mtia_minutes']:.1f} min, std={mtia['std_minutes']:.1f}, "
            f"intervals={mtia['num_intervals']}"
        )
    else:
        print("[MTIA] mean=N/A (no alarm intervals)")

    print("[PR-LT]")
    for i, lt in enumerate(pr_leadtime["lead_times"]):
        print(
            f"  {lt} min: P={pr_leadtime['precision'][i]:.3f}, "
            f"R={pr_leadtime['recall'][i]:.3f}, F1={pr_leadtime['f1'][i]:.3f}"
        )

    print(
        "[EVENT] TP={tp} FP={fp} FN={fn} P={p:.3f} R={r:.3f} F1={f:.3f}".format(
            tp=event_scores["tp"],
            fp=event_scores["fp"],
            fn=event_scores["fn"],
            p=event_scores["precision"],
            r=event_scores["recall"],
            f=event_scores["f1"],
        )
    )
    print(f"{'=' * 60}\n")

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
            "FAA": f"{r['first_alarm_accuracy']['first_alarm_accuracy']:.2f}"
            if r.get("first_alarm_accuracy", {}).get("first_alarm_accuracy") is not None
            else "N/A",
        }
        rows.append(row)
    return pd.DataFrame(rows)
