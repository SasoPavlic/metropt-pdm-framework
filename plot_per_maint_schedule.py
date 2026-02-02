#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone visualization of the per-maintenance training/testing schedule.

This script does NOT modify the main pipeline; it only reads the dataset,
derives the training-only intervals (initial TRAIN_FRAC minutes + post-maint
training minutes), and plots a single horizontal bar segmented into:
  - Training: initial baseline
  - Training: post-maint blocks
  - Testing: pre_W1
  - Testing: maintenance windows
  - Testing: gaps between maintenance (too short for post-train)
  - Testing: after_maint (post-train evaluation)
  - Testing: other (unassigned)
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from data_utils import load_csv, parse_maintenance_windows
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
        slice_mask = (index >= t_start) & (index <= t_stop)
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
    start = None
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


def plot_schedule(
    index: pd.DatetimeIndex,
    initial_train_mask: pd.Series,
    post_train_mask: pd.Series,
    pre_w1_mask: pd.Series,
    maint_mask: pd.Series,
    gap_short_mask: pd.Series,
    after_maint_mask: pd.Series,
    remainder_mask: pd.Series,
    maint_windows: List[Tuple],
    save_path: str,
    title: str,
) -> None:
    """Plot a single horizontal bar segmented by training/testing intervals."""
    y = 0.5
    h = 0.6

    def _spans(mask: pd.Series):
        return _mask_to_spans(mask)

    fig, ax = plt.subplots(figsize=(12, 3.2))

    # Training blocks
    spans_initial = _spans(initial_train_mask)
    spans_post = _spans(post_train_mask)
    if spans_initial:
        ax.broken_barh(spans_initial, (y, h), facecolors="#66BB6A", label="Training: initial baseline")
    if spans_post:
        ax.broken_barh(spans_post, (y, h), facecolors="#43A047", label="Training: post-maint block")

    # Testing blocks
    spans_pre = _spans(pre_w1_mask)
    spans_maint = _spans(maint_mask)
    spans_gap = _spans(gap_short_mask)
    spans_after = _spans(after_maint_mask)
    spans_remainder = _spans(remainder_mask)
    if spans_pre:
        ax.broken_barh(spans_pre, (y, h), facecolors="#90CAF9", label="Testing: pre_W1")
    if spans_gap:
        ax.broken_barh(spans_gap, (y, h), facecolors="#64B5F6", label="Testing: gaps between maint")
    if spans_after:
        ax.broken_barh(spans_after, (y, h), facecolors="#1E88E5", label="Testing: after_maint")
    if spans_maint:
        ax.broken_barh(spans_maint, (y, h), facecolors="#EF5350", label="Testing: maintenance windows")
    # Add vertical markers for maintenance starts (visibility for short windows)
    for w in maint_windows:
        s = w[0]
        e = w[1]
        try:
            s_num = mdates.date2num(pd.to_datetime(s))
            ax.axvline(s_num, color="#E53935", linewidth=0.8, alpha=0.7)
        except Exception:
            continue
    if spans_remainder:
        ax.broken_barh(spans_remainder, (y, h), facecolors="#BDBDBD", label="Testing: other (unassigned)")

    ax.set_ylim(0, 1.8)
    ax.set_yticks([])
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _build_demo_index_and_windows():
    """Create a small synthetic timeline and maintenance windows for illustration."""
    # 30 days of 10-second samples
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-maintenance training/testing schedule.")
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
    args = parser.parse_args()
    # Auto-suffix output based on demo vs real mode
    out_path = args.out
    if out_path.endswith(".png"):
        base = out_path[:-4]
        suffix = "_demo" if args.demo else "_real"
        out_path = f"{base}{suffix}.png"

    input_path = args.input
    if not os.path.exists(input_path):
        fallback = os.path.join("datasets", "MetroPT3.csv")
        if os.path.exists(fallback):
            input_path = fallback
        else:
            raise FileNotFoundError(
                f"Input CSV not found: {args.input!r}. "
                f"Provide --input or place MetroPT3.csv under datasets/."
            )

    if args.demo:
        index, maint_windows = _build_demo_index_and_windows()
    else:
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

    initial_train_mask = time_based_train_mask(index, args.train_minutes)
    post_train_mask = _build_post_maintenance_train_mask(index, maint_windows, args.post_train_minutes)
    training_mask = initial_train_mask | post_train_mask

    # Testing categories
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
    for j, (start_ts, end_ts) in enumerate(zip(starts, ends)):
        gap_start = pd.to_datetime(end_ts)
        gap_end = pd.to_datetime(starts[j + 1]) if j < len(starts) - 1 else data_end
        if gap_end <= gap_start:
            continue
        gap_minutes = (gap_end - gap_start).total_seconds() / 60.0
        if args.post_train_minutes <= 0 or gap_minutes <= args.post_train_minutes:
            gap_short_mask |= (index >= gap_start) & (index <= gap_end if j < len(starts) - 1 else index <= gap_end)
            continue
        train_end = gap_start + pd.Timedelta(minutes=float(args.post_train_minutes))
        if train_end > gap_end:
            train_end = gap_end
        after_maint_mask |= (index >= train_end) & (index <= gap_end if j < len(starts) - 1 else index <= gap_end)

    # Fill any remaining gaps so the bar has no blank slices
    assigned = (
        initial_train_mask
        | post_train_mask
        | pre_w1_mask
        | maint_mask
        | gap_short_mask
        | after_maint_mask
    )
    remainder_mask = ~assigned

    plot_schedule(
        index=index,
        initial_train_mask=initial_train_mask,
        post_train_mask=post_train_mask,
        pre_w1_mask=pre_w1_mask,
        maint_mask=maint_mask,
        gap_short_mask=gap_short_mask,
        after_maint_mask=after_maint_mask,
        remainder_mask=remainder_mask,
        maint_windows=maint_windows,
        save_path=out_path,
        title="Per-maintenance schedule (training vs testing blocks)",
    )

    print(f"[INFO] Saved schedule plot to {out_path}")


if __name__ == "__main__":
    main()
