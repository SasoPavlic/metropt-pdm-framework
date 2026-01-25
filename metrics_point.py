#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point-wise and risk-grid metrics for the anomaly detection pipeline.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def _alarm_intervals_from_mask(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return contiguous intervals where mask is True."""
    if mask.empty:
        return []
    idx = mask.index
    arr = mask.to_numpy(dtype=bool)
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    in_run = False
    run_start = None
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


def evaluate_risk_thresholds(
    risk: pd.Series,
    maintenance_windows: List[Tuple],
    thresholds: List[float],
    early_warning_minutes: float = 120.0,
) -> List[dict]:
    """Evaluate maintenance_risk thresholds for early-warning detection."""
    risk = risk.sort_index().astype(float).fillna(0.0)
    windows = []
    for w in maintenance_windows or []:
        if len(w) < 2:
            continue
        s = pd.to_datetime(w[0])
        e = pd.to_datetime(w[1])
        if pd.isna(s) or pd.isna(e) or e < s:
            continue
        windows.append((s, e))
    horizon = pd.to_timedelta(float(max(0.0, early_warning_minutes)), unit="m")

    results: List[dict] = []
    for theta in thresholds:
        if theta is None:
            continue
        mask = (risk >= float(theta))
        intervals = _alarm_intervals_from_mask(mask)
        stats = _risk_scores(intervals, windows, horizon)
        stats.update({"threshold": float(theta)})
        results.append(stats)
    return results


def _risk_scores(
    alarm_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]],
    failure_windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    horizon: pd.Timedelta,
) -> dict:
    alarm_used = [False] * len(alarm_intervals)
    tp = 0
    fn = 0

    for fst, fend in failure_windows:
        matched = False
        window_start = fst - horizon
        window_end = fst
        for i, (ast, aend) in enumerate(alarm_intervals):
            if alarm_used[i]:
                continue
            # Any overlap between alarm interval and [window_start, window_end]
            if ast <= window_end and aend >= window_start:
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


def confusion_and_scores(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute TP, FP, FN, TN, precision, recall, f1, accuracy."""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
    }
