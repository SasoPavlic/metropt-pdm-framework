#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alarm-island diagnostics for selected maintenance-risk alarms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _normalize_windows(maintenance_windows: List[Tuple]) -> List[Dict]:
    windows: List[Dict] = []
    for pos, window in enumerate(maintenance_windows or [], start=1):
        if len(window) < 2:
            continue
        start = pd.to_datetime(window[0])
        end = pd.to_datetime(window[1])
        if pd.isna(start) or pd.isna(end) or end < start:
            continue
        windows.append(
            {
                "window_id": str(window[2]) if len(window) >= 3 else f"W{pos}",
                "severity": str(window[3]) if len(window) >= 4 else "",
                "start": start,
                "end": end,
            }
        )
    windows.sort(key=lambda item: item["start"])
    return windows


def _alarm_intervals_from_mask(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Return contiguous True intervals with point counts."""
    if mask.empty:
        return []
    idx = mask.index
    arr = mask.to_numpy(dtype=bool)
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    in_run = False
    run_start: Optional[pd.Timestamp] = None
    run_points = 0

    for i, flag in enumerate(arr):
        if flag and not in_run:
            in_run = True
            run_start = idx[i]
            run_points = 1
        elif flag and in_run:
            run_points += 1
        elif not flag and in_run:
            intervals.append((run_start, idx[i - 1], run_points))
            in_run = False
            run_start = None
            run_points = 0

    if in_run and run_start is not None:
        intervals.append((run_start, idx[-1], run_points))
    return intervals


def _duration_minutes(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return max(0.0, (end - start).total_seconds() / 60.0)


def _prepare_prediction_mask(
    predictions: pd.Series,
    eval_mask: Optional[pd.Series],
) -> Tuple[pd.Series, int]:
    prediction_mask = predictions.sort_index().fillna(False).astype(bool)
    if eval_mask is None:
        return prediction_mask, int(len(prediction_mask))

    aligned_eval_mask = eval_mask.reindex(prediction_mask.index).fillna(False).astype(bool)
    return prediction_mask & aligned_eval_mask, int(aligned_eval_mask.sum())


def build_alarm_island_table(
    predictions: pd.Series,
    maintenance_windows: List[Tuple],
    early_warning_minutes: int,
    eval_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Build one row per contiguous selected-threshold alarm block."""
    prediction_mask, _total_eval_points = _prepare_prediction_mask(predictions, eval_mask)
    intervals = _alarm_intervals_from_mask(prediction_mask)
    windows = _normalize_windows(maintenance_windows)
    horizon = pd.Timedelta(minutes=int(early_warning_minutes))

    rows: List[Dict] = []
    previous_end: Optional[pd.Timestamp] = None
    for island_id, (start, end, points) in enumerate(intervals, start=1):
        warning_matches: List[Dict] = []
        maintenance_matches: List[Dict] = []

        for window in windows:
            maint_start = window["start"]
            maint_end = window["end"]
            warning_start = maint_start - horizon
            warning_end = maint_start

            if start <= warning_end and end >= warning_start:
                detection_time = max(start, warning_start)
                lead_minutes = (maint_start - detection_time).total_seconds() / 60.0
                warning_matches.append(
                    {
                        **window,
                        "lead_minutes": max(0.0, lead_minutes),
                        "starts_in_warning_window": warning_start <= start <= warning_end,
                        "starts_before_warning_window": start < warning_start,
                    }
                )

            if start <= maint_end and end >= maint_start:
                maintenance_matches.append(window)

        relation = "false_alarm"
        if warning_matches:
            relation = "early_warning_overlap"
        elif maintenance_matches:
            relation = "maintenance_overlap_only"

        first_warning = warning_matches[0] if warning_matches else None
        primary_match = (warning_matches or maintenance_matches or [None])[0]
        matched_ids = [match["window_id"] for match in warning_matches or maintenance_matches]
        matched_starts = [match["start"].isoformat() for match in warning_matches or maintenance_matches]
        gap_minutes = None
        if previous_end is not None:
            gap_minutes = max(0.0, (start - previous_end).total_seconds() / 60.0)

        rows.append(
            {
                "island_id": island_id,
                "start_time": start,
                "end_time": end,
                "duration_minutes": _duration_minutes(start, end),
                "points": int(points),
                "gap_since_previous_minutes": gap_minutes,
                "relation": relation,
                "overlaps_warning_window": bool(warning_matches),
                "overlaps_maintenance_window": bool(maintenance_matches),
                "starts_in_warning_window": bool(
                    first_warning.get("starts_in_warning_window") if first_warning else False
                ),
                "starts_before_warning_window": bool(
                    first_warning.get("starts_before_warning_window") if first_warning else False
                ),
                "first_matched_window_id": primary_match["window_id"] if primary_match else "",
                "first_matched_window_start": primary_match["start"] if primary_match else pd.NaT,
                "first_warning_lead_minutes": first_warning["lead_minutes"] if first_warning else np.nan,
                "matched_window_ids": ";".join(matched_ids),
                "matched_window_starts": ";".join(matched_starts),
            }
        )
        previous_end = end

    return pd.DataFrame(rows)


def summarize_alarm_islands(
    islands: pd.DataFrame,
    predictions: pd.Series,
    eval_mask: Optional[pd.Series] = None,
) -> Dict:
    """Summarize selected-threshold alarm-block shape."""
    prediction_mask, total_eval_points = _prepare_prediction_mask(predictions, eval_mask)
    alarm_points = int(prediction_mask.sum())
    coverage = alarm_points / total_eval_points if total_eval_points > 0 else 0.0

    summary: Dict = {
        "island_count": int(len(islands)),
        "alarm_points": alarm_points,
        "total_eval_points": int(total_eval_points),
        "coverage": float(coverage),
        "coverage_percent": float(coverage * 100.0),
    }

    if islands.empty:
        summary.update(
            {
                "early_warning_overlap_islands": 0,
                "maintenance_overlap_only_islands": 0,
                "false_alarm_islands": 0,
                "total_alarm_minutes": 0.0,
                "mean_duration_minutes": None,
                "median_duration_minutes": None,
                "p90_duration_minutes": None,
                "max_duration_minutes": None,
                "longest_island_id": None,
                "longest_start_time": None,
                "longest_end_time": None,
                "longest_relation": None,
                "mean_gap_minutes": None,
                "median_gap_minutes": None,
            }
        )
        return summary

    durations = pd.to_numeric(islands["duration_minutes"], errors="coerce").dropna()
    gaps = pd.to_numeric(islands["gap_since_previous_minutes"], errors="coerce").dropna()
    longest_idx = durations.idxmax()
    longest = islands.loc[longest_idx]

    summary.update(
        {
            "early_warning_overlap_islands": int(islands["overlaps_warning_window"].sum()),
            "maintenance_overlap_only_islands": int(
                (islands["relation"] == "maintenance_overlap_only").sum()
            ),
            "false_alarm_islands": int((islands["relation"] == "false_alarm").sum()),
            "total_alarm_minutes": float(durations.sum()),
            "mean_duration_minutes": float(durations.mean()),
            "median_duration_minutes": float(durations.median()),
            "p90_duration_minutes": float(durations.quantile(0.90)),
            "max_duration_minutes": float(durations.max()),
            "longest_island_id": int(longest["island_id"]),
            "longest_start_time": longest["start_time"],
            "longest_end_time": longest["end_time"],
            "longest_relation": str(longest["relation"]),
            "mean_gap_minutes": float(gaps.mean()) if not gaps.empty else None,
            "median_gap_minutes": float(gaps.median()) if not gaps.empty else None,
        }
    )
    return summary


def build_alarm_island_report(
    predictions: pd.Series,
    maintenance_windows: List[Tuple],
    early_warning_minutes: int,
    eval_mask: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, Dict]:
    islands = build_alarm_island_table(
        predictions=predictions,
        maintenance_windows=maintenance_windows,
        early_warning_minutes=early_warning_minutes,
        eval_mask=eval_mask,
    )
    summary = summarize_alarm_islands(
        islands=islands,
        predictions=predictions,
        eval_mask=eval_mask,
    )
    return islands, summary


def save_alarm_island_report(
    islands: pd.DataFrame,
    summary: Dict,
    islands_path: str,
    summary_path: str,
) -> None:
    Path(islands_path).parent.mkdir(parents=True, exist_ok=True)
    islands.to_csv(islands_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"[INFO] Saved alarm island table: {islands_path}")
    print(f"[INFO] Saved alarm island summary: {summary_path}")


def print_alarm_island_summary(label: str, summary: Dict) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"ALARM ISLANDS ({label})")
    print(bar)
    print(
        f"Islands={summary.get('island_count', 0)}  "
        f"early-warning overlap={summary.get('early_warning_overlap_islands', 0)}  "
        f"maintenance-only overlap={summary.get('maintenance_overlap_only_islands', 0)}  "
        f"false alarms={summary.get('false_alarm_islands', 0)}"
    )
    print(
        f"Coverage={summary.get('coverage_percent', 0.0):.2f}%  "
        f"alarm_points={summary.get('alarm_points', 0)}/"
        f"{summary.get('total_eval_points', 0)}"
    )

    if summary.get("island_count", 0) <= 0:
        print("No selected-threshold alarm islands.")
        return

    print(
        "Duration minutes: "
        f"mean={summary.get('mean_duration_minutes', 0.0):.1f}  "
        f"median={summary.get('median_duration_minutes', 0.0):.1f}  "
        f"p90={summary.get('p90_duration_minutes', 0.0):.1f}  "
        f"max={summary.get('max_duration_minutes', 0.0):.1f}"
    )
    if summary.get("mean_gap_minutes") is not None:
        print(
            "Gap minutes: "
            f"mean={summary['mean_gap_minutes']:.1f}  "
            f"median={summary['median_gap_minutes']:.1f}"
        )
    print(
        f"Longest island #{summary.get('longest_island_id')} "
        f"{summary.get('longest_start_time')} -> {summary.get('longest_end_time')} "
        f"relation={summary.get('longest_relation')}"
    )
