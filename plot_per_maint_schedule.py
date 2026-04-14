#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone visualization of the MetroPT per-maintenance dataset.

The output is intentionally paper-oriented rather than a dense debug dashboard.
It creates two focused figures:
  - a dataset timeline explaining recorded coverage, phase-1 pre-maintenance
    labels, phase-2 maintenance labels, and data gaps,
  - a NiaNetVAE cycle-level label availability summary explaining which cycles
    form a valid PdM ranking problem for the window-AUPRC objective.

This script does not modify the main pipeline or any model artifacts.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils import build_operation_phase, load_csv, parse_maintenance_windows
from detectors.utils import time_based_train_mask

# -----------------------------
# Defaults (fallbacks)
# -----------------------------
INPUT_PATH: str = "MetroPT3.csv"
INPUT_TIMESTAMP_COL: Optional[str] = None
DROP_UNNAMED_INDEX: bool = True

# Training schedule defaults (minutes)
TRAIN_FRAC: float = 1440
POST_MAINT_TRAIN_MINUTES: int = 1440
PRE_MAINTENANCE_MINUTES: int = 120

# NiaNetVAE window defaults used for objective-label availability.
SEQUENCE_LENGTH: int = 200
STRIDE: int = 1

# Maintenance windows (Davari et al., 2021)
USE_DEFAULT_METROPT_WINDOWS: bool = True
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


@dataclass(frozen=True)
class CycleSummary:
    cycle_id: int
    label: str
    maint_start: Optional[pd.Timestamp]
    maint_end: Optional[pd.Timestamp]
    next_start: Optional[pd.Timestamp]
    post_train_end: Optional[pd.Timestamp]
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    test_rows: int
    test_phase0_rows: int
    test_phase1_rows: int
    test_phase2_rows: int
    test_windows: int
    positive_windows: int
    negative_windows: int
    category: str
    note: str


def _build_post_maintenance_train_mask(
    index: pd.DatetimeIndex,
    maint_windows: List[Tuple],
    train_minutes: int,
) -> pd.Series:
    """Mark [end_j, end_j + train_minutes) as training-only per maintenance window."""
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


def _mask_to_spans(mask: pd.Series) -> List[Tuple[float, float]]:
    """Convert a boolean mask into matplotlib date spans."""
    if mask.empty:
        return []
    idx = mask.index
    values = mask.to_numpy(dtype=bool)
    spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    in_span = False
    start: Optional[pd.Timestamp] = None
    for i, flag in enumerate(values):
        if flag and not in_span:
            start = idx[i]
            in_span = True
        elif not flag and in_span:
            end = idx[i - 1]
            spans.append((start, end))
            in_span = False
    if in_span and start is not None:
        spans.append((start, idx[-1]))
    out: List[Tuple[float, float]] = []
    for s, e in spans:
        s_num = mdates.date2num(s)
        e_num = mdates.date2num(e)
        dur = max(0.0, e_num - s_num)
        out.append((s_num, dur))
    return out


def _intervals_to_broken_barh(
    intervals: Iterable[Tuple[pd.Timestamp, pd.Timestamp]],
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for start, end in intervals:
        s_num = mdates.date2num(pd.to_datetime(start))
        e_num = mdates.date2num(pd.to_datetime(end))
        dur = max(0.00001, e_num - s_num)
        out.append((s_num, dur))
    return out


def _single_interval(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[float, float]]:
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if pd.isna(start) or pd.isna(end) or end < start:
        return []
    return _intervals_to_broken_barh([(start, end)])


def _window_label_counts(
    phase_values: np.ndarray,
    mask: np.ndarray,
    seq_len: int,
    stride: int,
) -> Tuple[int, int, int]:
    """Return total, positive, negative end-anchor window counts for a phase mask."""
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != phase_values.shape[0]:
        raise ValueError("mask and phase arrays must have the same length")

    total = 0
    positives = 0
    negatives = 0
    start: Optional[int] = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            t, p, n = _window_label_counts_for_slice(phase_values[start:i], seq_len, stride)
            total += t
            positives += p
            negatives += n
            start = None
    if start is not None:
        t, p, n = _window_label_counts_for_slice(phase_values[start:], seq_len, stride)
        total += t
        positives += p
        negatives += n
    return total, positives, negatives


def _window_label_counts_for_slice(
    phases: np.ndarray,
    seq_len: int,
    stride: int,
) -> Tuple[int, int, int]:
    window_count = max(0, (len(phases) - seq_len) // stride + 1)
    if window_count <= 0:
        return 0, 0, 0
    anchors = np.arange(seq_len - 1, seq_len - 1 + window_count * stride, stride, dtype=np.int64)
    positives = int(np.sum(phases[anchors] == 1))
    negatives = int(window_count - positives)
    return int(window_count), positives, negatives


def _cycle_summaries(
    index: pd.DatetimeIndex,
    operation_phase: pd.Series,
    maint_windows: List[Tuple],
    train_minutes: float,
    post_train_minutes: int,
    seq_len: int,
    stride: int,
) -> List[CycleSummary]:
    mw_sorted = sorted(maint_windows, key=lambda w: pd.to_datetime(w[0]))
    if not mw_sorted:
        return []

    phase_values = operation_phase.to_numpy(dtype=np.int8, copy=False)
    base_end = index.min() + pd.Timedelta(minutes=float(train_minutes))
    summaries: List[CycleSummary] = []

    for cycle_id in range(0, len(mw_sorted) + 1):
        if cycle_id == 0:
            test_start = base_end
            test_end = pd.to_datetime(mw_sorted[0][0])
            test_time_mask = (index > test_start) & (index < test_end)
            label = "pre_W1"
            maint_start = None
            maint_end = None
            next_start = pd.to_datetime(mw_sorted[0][0])
            post_train_end = None
        else:
            j = cycle_id - 1
            maint_start = pd.to_datetime(mw_sorted[j][0])
            maint_end = pd.to_datetime(mw_sorted[j][1])
            label = str(mw_sorted[j][2] or f"#{cycle_id}")
            is_last = j == len(mw_sorted) - 1
            next_start = pd.to_datetime(mw_sorted[j + 1][0]) if not is_last else pd.to_datetime(index.max())
            post_train_end = maint_end + pd.Timedelta(minutes=float(post_train_minutes))
            if post_train_end > next_start:
                post_train_end = next_start
            test_start = post_train_end
            test_end = next_start
            if is_last:
                test_time_mask = (index >= test_start) & (index <= test_end)
            else:
                test_time_mask = (index >= test_start) & (index < test_end)

        test_mask = test_time_mask & operation_phase.isin([0, 1]).to_numpy(dtype=bool)
        test_windows, pos_windows, neg_windows = _window_label_counts(
            phase_values, test_mask, seq_len=seq_len, stride=stride
        )
        test_rows = int(test_mask.sum())
        p0_rows = int(np.sum(test_mask & (phase_values == 0)))
        p1_rows = int(np.sum(test_mask & (phase_values == 1)))
        p2_rows = int(np.sum(test_time_mask & (phase_values == 2)))

        note = "valid: phase 0 + phase 1 test windows"
        if test_rows <= 0:
            category = "no_test_rows"
            note = "no phase 0/1 test rows after split"
        elif test_windows <= 0:
            category = "too_short"
            note = f"test slice shorter than seq_len={seq_len}"
        elif pos_windows <= 0:
            category = "no_positive_windows"
            note = "no positive phase-1 test windows"
        elif neg_windows <= 0:
            category = "no_negative_windows"
            note = "no negative phase-0 test windows"
        else:
            category = "valid_mixed"

        summaries.append(
            CycleSummary(
                cycle_id=cycle_id,
                label=label,
                maint_start=maint_start,
                maint_end=maint_end,
                next_start=next_start,
                post_train_end=post_train_end,
                test_start=pd.to_datetime(test_start),
                test_end=pd.to_datetime(test_end),
                test_rows=test_rows,
                test_phase0_rows=p0_rows,
                test_phase1_rows=p1_rows,
                test_phase2_rows=p2_rows,
                test_windows=test_windows,
                positive_windows=pos_windows,
                negative_windows=neg_windows,
                category=category,
                note=note,
            )
        )
    return summaries


def _sibling_output_path(save_path: str, suffix: str) -> str:
    base, ext = os.path.splitext(save_path)
    return f"{base}_{suffix}{ext or '.png'}"


def _median_sample_period_seconds(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return float("nan")
    deltas = pd.Series(index).diff().dt.total_seconds().dropna()
    return float(deltas.median()) if not deltas.empty else float("nan")


def _pre_maintenance_row_counts(
    index: pd.DatetimeIndex,
    maint_windows: List[Tuple],
    pre_maint_minutes: int,
) -> pd.DataFrame:
    rows = []
    horizon = pd.Timedelta(minutes=float(pre_maint_minutes))
    for start, end, label, severity in sorted(maint_windows, key=lambda w: pd.to_datetime(w[0])):
        maint_start = pd.to_datetime(start)
        pre_start = maint_start - horizon
        pre_mask = (index >= pre_start) & (index < maint_start)
        maint_mask = (index >= maint_start) & (index <= pd.to_datetime(end))
        rows.append(
            {
                "label": str(label),
                "severity": str(severity),
                "pre_start": pre_start,
                "maint_start": maint_start,
                "maint_end": pd.to_datetime(end),
                "pre_rows": int(pre_mask.sum()),
                "maint_rows": int(maint_mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def _category_label(category: str) -> str:
    labels = {
        "valid_mixed": "valid: has positive and negative windows",
        "no_positive_windows": "not valid: no positive windows",
        "no_negative_windows": "not valid: no negative windows",
        "too_short": "not valid: test period too short",
        "no_test_rows": "not valid: no usable test rows",
    }
    return labels.get(category, category)


def _style_time_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, axis="x", color="#D0D7DE", linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)


def _plot_dataset_overview(
    index: pd.DatetimeIndex,
    maint_windows: List[Tuple],
    save_path: str,
    train_minutes: float,
    post_train_minutes: int,
    pre_maint_minutes: int,
) -> None:
    """Paper-style figure: schedule blocks (A) + pre-maint row availability (B)."""
    pre_counts = _pre_maintenance_row_counts(index, maint_windows, pre_maint_minutes)
    median_period_s = _median_sample_period_seconds(index)
    expected_pre_rows = (
        pre_maint_minutes * 60.0 / median_period_s
        if np.isfinite(median_period_s) and median_period_s > 0
        else float("nan")
    )

    fig, (ax_timeline, ax_prewindow) = plt.subplots(
        2,
        1,
        figsize=(19, 9.5),
        gridspec_kw={"height_ratios": [1.0, 1.7]},
        constrained_layout=False,
    )

    # Panel A: schedule blocks (same semantics as classic legacy view).
    initial_train_mask = time_based_train_mask(index, train_minutes)
    post_train_mask = _build_post_maintenance_train_mask(index, maint_windows, post_train_minutes)
    training_mask = initial_train_mask | post_train_mask
    mw_sorted = sorted(maint_windows, key=lambda w: pd.to_datetime(w[0]))
    starts = [pd.to_datetime(w[0]) for w in mw_sorted]
    ends = [pd.to_datetime(w[1]) for w in mw_sorted]
    data_end = index.max()
    first_start = starts[0]

    pre_w1_mask = (index <= first_start) & (~training_mask)
    maint_mask = pd.Series(False, index=index)
    for s, e in zip(starts, ends):
        maint_mask |= (index >= s) & (index <= e)

    gap_short_mask = pd.Series(False, index=index)
    after_maint_mask = pd.Series(False, index=index)
    for j, (_start_ts, end_ts) in enumerate(zip(starts, ends)):
        gap_start = pd.to_datetime(end_ts)
        gap_end = pd.to_datetime(starts[j + 1]) if j < len(starts) - 1 else data_end
        if gap_end <= gap_start:
            continue
        gap_minutes = (gap_end - gap_start).total_seconds() / 60.0
        is_last = j == len(starts) - 1
        if post_train_minutes <= 0 or gap_minutes <= post_train_minutes:
            if is_last:
                gap_short_mask |= (index >= gap_start) & (index <= gap_end)
            else:
                gap_short_mask |= (index >= gap_start) & (index < gap_end)
            continue
        train_end = gap_start + pd.Timedelta(minutes=float(post_train_minutes))
        if train_end > gap_end:
            train_end = gap_end
        if is_last:
            after_maint_mask |= (index >= train_end) & (index <= gap_end)
        else:
            after_maint_mask |= (index >= train_end) & (index < gap_end)

    assigned = (
        initial_train_mask
        | post_train_mask
        | pre_w1_mask
        | maint_mask
        | gap_short_mask
        | after_maint_mask
    )
    remainder_mask = ~assigned

    band_y = 0.20
    band_h = 0.60
    ax_timeline.broken_barh(_mask_to_spans(initial_train_mask), (band_y, band_h), facecolors="#66BB6A", label="train: initial baseline")
    ax_timeline.broken_barh(_mask_to_spans(post_train_mask), (band_y, band_h), facecolors="#43A047", label="train: post-maint block")
    ax_timeline.broken_barh(_mask_to_spans(pre_w1_mask), (band_y, band_h), facecolors="#90CAF9", label="test: pre_W1")
    ax_timeline.broken_barh(_mask_to_spans(gap_short_mask), (band_y, band_h), facecolors="#64B5F6", label="test: gaps between maint")
    ax_timeline.broken_barh(_mask_to_spans(after_maint_mask), (band_y, band_h), facecolors="#1E88E5", label="test: after_maint")
    ax_timeline.broken_barh(_mask_to_spans(maint_mask), (band_y, band_h), facecolors="#EF5350", label="test: maintenance windows")
    if remainder_mask.any():
        ax_timeline.broken_barh(_mask_to_spans(remainder_mask), (band_y, band_h), facecolors="#BDBDBD", label="test: other")
    ax_timeline.set_yticks([])
    ax_timeline.set_title("A. Training and testing blocks over time", loc="left", fontsize=16, fontweight="bold", pad=10)
    _style_time_axis(ax_timeline)
    ax_timeline.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#B0BEC5",
        framealpha=0.92,
        fontsize=10.5,
        ncol=2,
    )

    # Panel B: whether each labelled pre-maintenance horizon has recorded rows.
    x = np.arange(len(pre_counts))
    colors = np.where(pre_counts["pre_rows"].to_numpy() == 0, "#BDBDBD", "#F4A340")
    ax_prewindow.bar(x, pre_counts["pre_rows"], color=colors, edgecolor="#4A4A4A", linewidth=0.6)
    if np.isfinite(expected_pre_rows):
        ax_prewindow.axhline(
            expected_pre_rows,
            color="#455A64",
            linestyle="--",
            linewidth=1.0,
            label=f"full {pre_maint_minutes} min period",
        )
    ax_prewindow.set_xticks(x)
    ax_prewindow.set_xticklabels([f"C{i+1}" for i in range(len(pre_counts))], rotation=0, fontsize=12)
    ax_prewindow.tick_params(axis="y", labelsize=11)
    ax_prewindow.set_ylabel("Rows in warning\nperiod", fontsize=14)
    ax_prewindow.set_title("B. Available rows before each maintenance", loc="left", fontsize=16, fontweight="bold", pad=10)
    ax_prewindow.grid(True, axis="y", color="#D0D7DE", linewidth=0.6, alpha=0.8)
    ax_prewindow.spines["top"].set_visible(False)
    ax_prewindow.spines["right"].set_visible(False)
    if np.isfinite(expected_pre_rows):
        ax_prewindow.legend(
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="#B0BEC5",
            framealpha=0.92,
            fontsize=10.5,
        )

    for i, row in pre_counts.iterrows():
        if row["pre_rows"] == 0:
            ax_prewindow.annotate(
                "no data\nin warning period",
                xy=(i, 0),
                xytext=(i, max(10.0, float(np.nanmax(pre_counts["pre_rows"])) * 0.15)),
                ha="center",
                va="bottom",
                fontsize=11,
                color="#6D4C41",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#D9B26F",
                    "alpha": 0.95,
                },
                arrowprops={"arrowstyle": "-|>", "color": "#6D4C41", "lw": 0.8},
            )

    fig.suptitle("MetroPT-3: schedule blocks and warning-label availability", fontsize=20, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.86, hspace=0.40, left=0.08, right=0.98, bottom=0.08)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_cycle_label_summary(
    cycle_summaries: List[CycleSummary],
    save_path: str,
    seq_len: int,
    stride: int,
) -> None:
    """Paper-style summary of whether each cycle can support window-AUPRC."""
    category_colors = {
        "valid_mixed": "#2A6FBB",
        "no_positive_windows": "#E69F00",
        "no_negative_windows": "#8A5FBF",
        "too_short": "#8C8C8C",
        "no_test_rows": "#C9C1B8",
    }
    cycles = np.array([s.cycle_id for s in cycle_summaries])
    total = np.array([s.test_windows for s in cycle_summaries], dtype=float)
    positive = np.array([s.positive_windows for s in cycle_summaries], dtype=float)
    negative = np.array([s.negative_windows for s in cycle_summaries], dtype=float)
    categories = [s.category for s in cycle_summaries]
    colors = [category_colors.get(cat, "#999999") for cat in categories]
    positive_rate = np.divide(positive, total, out=np.full_like(positive, np.nan), where=total > 0) * 100.0

    fig, (ax_count, ax_rate) = plt.subplots(
        2,
        1,
        figsize=(8.2, 4.7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0]},
        constrained_layout=True,
    )

    ax_count.bar(cycles, np.maximum(total, 1.0), color=colors, edgecolor="#333333", linewidth=0.55)
    ax_count.set_yscale("log")
    ax_count.tick_params(axis="both", labelsize=7.5)
    ax_count.set_ylabel("Test windows\n(log scale)", fontsize=8)
    ax_count.set_title("A. Test data size and validity by cycle", loc="left", fontsize=9, fontweight="bold")
    ax_count.set_xticks(cycles)
    ax_count.set_xticklabels([f"C{int(c)}" for c in cycles], rotation=0, fontsize=7.0)
    ax_count.grid(True, axis="y", color="#D0D7DE", linewidth=0.6, alpha=0.8)
    ax_count.spines["top"].set_visible(False)
    ax_count.spines["right"].set_visible(False)

    for summary in cycle_summaries:
        if summary.category == "no_positive_windows":
            ax_count.annotate(
                f"C{summary.cycle_id}\nno positive labels",
                xy=(summary.cycle_id, max(summary.test_windows, 1)),
                xytext=(summary.cycle_id - 0.9, max(summary.test_windows, 1) * 2.5),
                ha="center",
                fontsize=6.5,
                color="#8A4B00",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#D9B26F",
                    "alpha": 0.95,
                },
                arrowprops={"arrowstyle": "-|>", "color": "#8A4B00", "lw": 0.8},
            )

    valid_mask = np.array([cat == "valid_mixed" for cat in categories])
    ax_rate.bar(cycles[valid_mask], positive_rate[valid_mask], color="#F4A340", edgecolor="#8A4B00", linewidth=0.55)
    ax_rate.scatter(cycles[~valid_mask], np.zeros(np.sum(~valid_mask)), marker="x", color="#7A7A7A", s=35, label="score is not valid")
    ax_rate.tick_params(axis="both", labelsize=7.5)
    ax_rate.set_ylabel("Positive warning\nwindows (%)", fontsize=8)
    ax_rate.set_xlabel(f"NiaNetVAE cycle (window end label, seq_len={seq_len}, stride={stride})", fontsize=8)
    ax_rate.set_xticks(cycles)
    ax_rate.set_xticklabels([f"C{int(c)}" for c in cycles], rotation=0, fontsize=7.0)
    ax_rate.set_title("B. Positive-label share in test windows", loc="left", fontsize=9, fontweight="bold")
    ax_rate.grid(True, axis="y", color="#D0D7DE", linewidth=0.6, alpha=0.8)
    ax_rate.spines["top"].set_visible(False)
    ax_rate.spines["right"].set_visible(False)
    ax_rate.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#B0BEC5",
        framealpha=0.92,
        fontsize=7.0,
    )

    legend_handles = [
        mpatches.Patch(color=color, label=_category_label(category))
        for category, color in category_colors.items()
        if category in categories
    ]
    ax_count.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#B0BEC5",
        framealpha=0.92,
        fontsize=7.0,
    )

    fig.suptitle("NiaNetVAE: where cycle-level PdM scoring is possible", fontsize=10.5, fontweight="bold")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_train_test_blocks(
    index: pd.DatetimeIndex,
    maint_windows: List[Tuple],
    train_minutes: float,
    post_train_minutes: int,
    save_path: str,
) -> None:
    """Legacy-style schedule plot with explicit training/testing block categories."""
    initial_train_mask = time_based_train_mask(index, train_minutes)
    post_train_mask = _build_post_maintenance_train_mask(index, maint_windows, post_train_minutes)
    training_mask = initial_train_mask | post_train_mask

    mw_sorted = sorted(maint_windows, key=lambda w: pd.to_datetime(w[0]))
    starts = [pd.to_datetime(w[0]) for w in mw_sorted]
    ends = [pd.to_datetime(w[1]) for w in mw_sorted]

    data_end = index.max()
    first_start = starts[0]

    pre_w1_mask = (index <= first_start) & (~training_mask)
    maint_mask = pd.Series(False, index=index)
    for s, e in zip(starts, ends):
        maint_mask |= (index >= s) & (index <= e)

    gap_short_mask = pd.Series(False, index=index)
    after_maint_mask = pd.Series(False, index=index)
    for j, (_start_ts, end_ts) in enumerate(zip(starts, ends)):
        gap_start = pd.to_datetime(end_ts)
        gap_end = pd.to_datetime(starts[j + 1]) if j < len(starts) - 1 else data_end
        if gap_end <= gap_start:
            continue
        gap_minutes = (gap_end - gap_start).total_seconds() / 60.0
        is_last = j == len(starts) - 1
        if post_train_minutes <= 0 or gap_minutes <= post_train_minutes:
            if is_last:
                gap_short_mask |= (index >= gap_start) & (index <= gap_end)
            else:
                gap_short_mask |= (index >= gap_start) & (index < gap_end)
            continue
        train_end = gap_start + pd.Timedelta(minutes=float(post_train_minutes))
        if train_end > gap_end:
            train_end = gap_end
        if is_last:
            after_maint_mask |= (index >= train_end) & (index <= gap_end)
        else:
            after_maint_mask |= (index >= train_end) & (index < gap_end)

    assigned = (
        initial_train_mask
        | post_train_mask
        | pre_w1_mask
        | maint_mask
        | gap_short_mask
        | after_maint_mask
    )
    remainder_mask = ~assigned

    y = 0.5
    h = 0.6
    fig, ax = plt.subplots(figsize=(12, 3.2))

    spans_initial = _mask_to_spans(initial_train_mask)
    spans_post = _mask_to_spans(post_train_mask)
    spans_pre = _mask_to_spans(pre_w1_mask)
    spans_maint = _mask_to_spans(maint_mask)
    spans_gap = _mask_to_spans(gap_short_mask)
    spans_after = _mask_to_spans(after_maint_mask)
    spans_remainder = _mask_to_spans(remainder_mask)

    if spans_initial:
        ax.broken_barh(spans_initial, (y, h), facecolors="#66BB6A", label="Training: initial baseline")
    if spans_post:
        ax.broken_barh(spans_post, (y, h), facecolors="#43A047", label="Training: post-maint block")
    if spans_pre:
        ax.broken_barh(spans_pre, (y, h), facecolors="#90CAF9", label="Testing: pre_W1")
    if spans_gap:
        ax.broken_barh(spans_gap, (y, h), facecolors="#64B5F6", label="Testing: gaps between maint")
    if spans_after:
        ax.broken_barh(spans_after, (y, h), facecolors="#1E88E5", label="Testing: after_maint")
    if spans_maint:
        ax.broken_barh(spans_maint, (y, h), facecolors="#EF5350", label="Testing: maintenance windows")
    for w in maint_windows:
        try:
            s_num = mdates.date2num(pd.to_datetime(w[0]))
            ax.axvline(s_num, color="#E53935", linewidth=0.8, alpha=0.7)
        except Exception:
            continue
    if spans_remainder:
        ax.broken_barh(spans_remainder, (y, h), facecolors="#BDBDBD", label="Testing: other (unassigned)")

    ax.set_ylim(0, 1.8)
    ax.set_yticks([])
    ax.set_xlabel("Time")
    ax.set_title("Per-maintenance schedule (training vs testing blocks)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_schedule(
    index: pd.DatetimeIndex,
    operation_phase: pd.Series,
    maint_windows: List[Tuple],
    cycle_summaries: List[CycleSummary],
    save_path: str,
    train_minutes: float,
    post_train_minutes: int,
    pre_maint_minutes: int,
    seq_len: int,
    stride: int,
) -> List[str]:
    """Create paper-oriented figures plus classic train/test block overview."""
    cycle_path = _sibling_output_path(save_path, "cycle_labels")
    blocks_path = _sibling_output_path(save_path, "schedule_blocks")
    _plot_dataset_overview(
        index=index,
        maint_windows=maint_windows,
        save_path=save_path,
        train_minutes=train_minutes,
        post_train_minutes=post_train_minutes,
        pre_maint_minutes=pre_maint_minutes,
    )
    _plot_cycle_label_summary(
        cycle_summaries=cycle_summaries,
        save_path=cycle_path,
        seq_len=seq_len,
        stride=stride,
    )
    _plot_train_test_blocks(
        index=index,
        maint_windows=maint_windows,
        train_minutes=train_minutes,
        post_train_minutes=post_train_minutes,
        save_path=blocks_path,
    )
    return [save_path, cycle_path, blocks_path]


def _build_demo_index_and_windows():
    """Create a small synthetic timeline and maintenance windows for illustration."""
    index = pd.date_range("2020-01-01", periods=60 * 24 * 360, freq="10s")
    maint_windows = [
        (pd.Timestamp("2020-01-05 08:00"), pd.Timestamp("2020-01-05 12:00"), "#A", "high"),
        (pd.Timestamp("2020-01-10 06:00"), pd.Timestamp("2020-01-10 08:00"), "#B", "high"),
        (pd.Timestamp("2020-01-12 06:00"), pd.Timestamp("2020-01-12 07:00"), "#B2", "high"),
        (pd.Timestamp("2020-01-12 09:00"), pd.Timestamp("2020-01-12 10:00"), "#B3", "high"),
        (pd.Timestamp("2020-01-15 00:00"), pd.Timestamp("2020-01-15 06:00"), "#C", "high"),
        (pd.Timestamp("2020-01-20 18:00"), pd.Timestamp("2020-01-20 22:00"), "#D", "high"),
        (pd.Timestamp("2020-01-26 10:00"), pd.Timestamp("2020-01-26 15:00"), "#E", "medium"),
        (pd.Timestamp("2020-02-08 08:00"), pd.Timestamp("2020-02-08 15:00"), "#F", "medium"),
        (pd.Timestamp("2020-02-15 10:00"), pd.Timestamp("2020-02-15 19:00"), "#G", "medium"),
    ]
    return index, maint_windows


def _resolve_input_path(path: str) -> str:
    if os.path.exists(path):
        return path
    fallback = os.path.join("datasets", "MetroPT3.csv")
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(
        f"Input CSV not found: {path!r}. Provide --input or place MetroPT3.csv under datasets/."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MetroPT per-maintenance schedule and phase labels.")
    parser.add_argument("--input", default=INPUT_PATH, help="Input CSV path.")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data for a clean illustration.")
    parser.add_argument("--out", default="plots/per_maint_schedule.png", help="Output PNG path.")
    parser.add_argument("--train_minutes", type=float, default=TRAIN_FRAC, help="Initial training minutes.")
    parser.add_argument(
        "--post_train_minutes",
        type=int,
        default=POST_MAINT_TRAIN_MINUTES,
        help="Post-maintenance training minutes.",
    )
    parser.add_argument(
        "--pre_maint_minutes",
        type=int,
        default=PRE_MAINTENANCE_MINUTES,
        help="Minutes before maintenance labeled as phase 1.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=SEQUENCE_LENGTH,
        help="NiaNetVAE sequence length used for end-anchor window label counts.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=STRIDE,
        help="NiaNetVAE test-window stride used for label counts.",
    )
    args = parser.parse_args()

    out_path = args.out
    if out_path.endswith(".png"):
        base = out_path[:-4]
        suffix = "_demo" if args.demo else "_real"
        out_path = f"{base}{suffix}.png"

    if args.demo:
        index, maint_windows = _build_demo_index_and_windows()
    else:
        input_path = _resolve_input_path(args.input)
        df = load_csv(input_path, INPUT_TIMESTAMP_COL, drop_unnamed=DROP_UNNAMED_INDEX)
        index = df.index
        if index.empty:
            raise ValueError("Empty dataset index.")
        maint_windows = parse_maintenance_windows(
            windows=None,
            maintenance_csv=None,
            use_default_windows=USE_DEFAULT_METROPT_WINDOWS,
            default_windows=DEFAULT_METROPT_WINDOWS,
        )

    if not maint_windows:
        raise ValueError("No maintenance windows available to plot.")

    operation_phase = build_operation_phase(
        index=index,
        windows=maint_windows,
        pre_minutes=args.pre_maint_minutes,
    ).astype(np.int8)

    cycle_summaries = _cycle_summaries(
        index=index,
        operation_phase=operation_phase,
        maint_windows=maint_windows,
        train_minutes=args.train_minutes,
        post_train_minutes=args.post_train_minutes,
        seq_len=args.seq_len,
        stride=args.stride,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    saved_paths = plot_schedule(
        index=index,
        operation_phase=operation_phase,
        maint_windows=maint_windows,
        cycle_summaries=cycle_summaries,
        save_path=out_path,
        train_minutes=args.train_minutes,
        post_train_minutes=args.post_train_minutes,
        pre_maint_minutes=args.pre_maint_minutes,
        seq_len=args.seq_len,
        stride=args.stride,
    )

    invalid = [s.cycle_id for s in cycle_summaries if s.category != "valid_mixed"]
    no_positive = [s.cycle_id for s in cycle_summaries if s.category == "no_positive_windows"]
    for path in saved_paths:
        print(f"[INFO] Saved figure to {path}")
    print(f"[INFO] Cycles without valid window_auprc: {invalid}")
    print(f"[INFO] Cycles with test windows but no positives: {no_positive}")


if __name__ == "__main__":
    main()
