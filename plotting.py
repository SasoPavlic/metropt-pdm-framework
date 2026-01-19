#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting helpers for the anomaly detection pipeline.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def plot_raw_timeline(
        df_plot: pd.DataFrame,
        maintenance_windows: List[Tuple],
        save_fig: Optional[str],
        train_frac: float = None,
        train_cutoff_time: Optional[pd.Timestamp] = None,
        show_window_labels: bool = True,
        window_label_fontsize: int = 9,
        window_label_format: str = "{id}",
        predicted_phase: Optional[pd.Series] = None,
        risk_threshold: Optional[float] = None,
        early_warning_minutes: float = 120.0,
):
    fig, ax = plt.subplots(figsize=(14, 5.5))

    if "maintenance_risk" not in df_plot.columns:
        raise ValueError("maintenance_risk column required for plotting.")

    state = (
        predicted_phase.reindex(df_plot.index).fillna(0).astype(bool)
        if predicted_phase is not None
        else pd.Series(False, index=df_plot.index)
    )

    ax.fill_between(
        df_plot.index,
        0,
        1,
        where=~state.values,
        color="#F4F9F4",
        alpha=0.35,
        step="post",
        label="Normal",
        zorder=0,
    )
    coverage = float(state.mean() * 100.0) if len(state) else 0.0
    risk_label = "Risk alarm"
    if risk_threshold is not None:
        risk_label = f"Risk alarm (≥ θ={risk_threshold:.2f}, {coverage:.1f}% coverage)"
    ax.fill_between(
        df_plot.index,
        0,
        1,
        where=state.values,
        color="#FF7043",
        alpha=0.25,
        label=risk_label,
        zorder=1,
    )

    # Plot risk curve for visibility of threshold vs signal
    risk_series = df_plot["maintenance_risk"].astype(float).fillna(0.0)
    ax.plot(
        df_plot.index,
        risk_series.values,
        color="#263238",
        linewidth=1.0,
        alpha=0.7,
        label="Maintenance risk",
        zorder=2,
    )
    if risk_threshold is not None:
        ax.axhline(
            float(risk_threshold),
            color="#FF7043",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            label="Risk threshold",
            zorder=3,
        )

    # Training cutoff
    cutoff_ts = train_cutoff_time
    if cutoff_ts is None and train_frac is not None:
        train_size = int(len(df_plot) * train_frac)
        if 0 < train_size < len(df_plot):
            cutoff_ts = df_plot.index[train_size - 1]
    if cutoff_ts is not None:
        ax.axvline(cutoff_ts, color="#7B1FA2", linestyle="-", linewidth=2.0, alpha=0.9, label="Training cutoff")

    # Determine plot x-range to clip maintenance windows
    if len(df_plot.index) > 0:
        xmin = pd.to_datetime(df_plot.index.min())
        xmax = pd.to_datetime(df_plot.index.max())
    else:
        xmin = xmax = None

    # Visualize maintenance windows
    span_drawn = False
    horizon = pd.to_timedelta(float(max(0.0, early_warning_minutes)), unit="m")

    # Label placement lanes
    lanes_y = [1.02, 1.06, 1.10]
    last_x_in_lane = [float('-inf')] * len(lanes_y)
    xaxis_transform = ax.get_xaxis_transform()
    xnum_min = mdates.date2num(xmin) if xmin is not None else None
    xnum_max = mdates.date2num(xmax) if xmax is not None else None
    xspan_num = (xnum_max - xnum_min) if (xnum_min is not None and xnum_max is not None) else None
    sep_thresh = (xspan_num / 40.0) if xspan_num else 0.0

    for item in maintenance_windows or []:
        if len(item) >= 4:
            s_raw, e_raw, wid, sev = item[0], item[1], item[2], item[3]
        else:
            s_raw, e_raw = item[0], item[1]
            wid, sev = None, None
        s = pd.to_datetime(s_raw)
        e = pd.to_datetime(e_raw)
        if xmin is not None and xmax is not None:
            if e < xmin or s > xmax:
                continue
            s_clip = max(s, xmin)
            e_clip = min(e, xmax)
        else:
            s_clip, e_clip = s, e

        dur_min_real = max(0.0, (e - s).total_seconds() / 60.0)

        ax.axvspan(
            s_clip,
            e_clip,
            facecolor="#1E88E5",
            edgecolor="#0D47A1",
            alpha=0.55,
            linewidth=1.2,
            zorder=4,
        )
        span_drawn = True
        x_for_label = s_clip + (e_clip - s_clip) / 2

        # Lead windows omitted for clarity

        if show_window_labels:
            try:
                label_id = (wid if wid is not None else "")
                label_sev = (sev if sev is not None else "")
                label = window_label_format.format(id=label_id, severity=label_sev, dur_min=int(round(dur_min_real)))
                if str(label).strip() == "":
                    continue
                xnum = mdates.date2num(pd.to_datetime(x_for_label))
                lane_idx = 0
                if xspan_num:
                    for j in range(len(lanes_y)):
                        idx = j % len(lanes_y)
                        if xnum - last_x_in_lane[idx] >= sep_thresh:
                            lane_idx = idx
                            break
                        lane_idx = idx
                    last_x_in_lane[lane_idx] = xnum
                ax.text(
                    x_for_label,
                    lanes_y[lane_idx],
                    label,
                    transform=xaxis_transform,
                    fontsize=window_label_fontsize,
                    color="#5D4037",
                    ha="center",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#FFC107", alpha=0.9, linewidth=0.8),
                )
            except Exception:
                pass

    handles, labels = ax.get_legend_handles_labels()
    if span_drawn:
        handles.append(Patch(facecolor="#1565C0", alpha=0.35, label="Failure window"))
        labels.append("Failure window")
    # Lead window legend removed as spans are not drawn
    ax.legend(handles, labels, loc="best")

    ax.set_xlabel("Time")
    ax.set_ylabel("Maintenance risk")
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0.0", "0.5", "1.0"])
    ax.set_ylim(0, 1)
    ax.grid(True, axis="x", alpha=0.2)

    fig.tight_layout()
    if save_fig:
        fig.savefig(save_fig, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_lead_time_distribution(
        dist: dict,
        save_fig: Optional[str],
        title: Optional[str] = None,
):
    """Plot histogram of lead-time-to-detection values."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = dist.get("bins", []) if isinstance(dist, dict) else []
    counts = dist.get("counts", []) if isinstance(dist, dict) else []

    if len(bins) < 2 or not counts:
        ax.text(0.5, 0.5, "No lead-time data", ha="center", va="center", fontsize=10)
        ax.set_axis_off()
    else:
        edges = np.asarray(bins, dtype=float)
        counts_arr = np.asarray(counts, dtype=float)
        n = min(len(edges) - 1, len(counts_arr))
        edges = edges[: n + 1]
        counts_arr = counts_arr[:n]
        widths = edges[1:] - edges[:-1]
        ax.bar(
            edges[:-1],
            counts_arr,
            width=widths,
            align="edge",
            color="#90CAF9",
            edgecolor="#1E88E5",
            alpha=0.8,
        )
        bin_label = "Count of detected events by lead-time bucket (minutes)"
        ax.legend(handles=[Patch(facecolor="#90CAF9", edgecolor="#1E88E5", label=bin_label)], loc="best")
        ax.set_xlabel("Lead time (min)")
        ax.set_ylabel("Count")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, max(counts_arr) * 1.1 if counts_arr.size else 1.0)
        tick_start = int(np.floor(edges[0]))
        tick_end = int(np.ceil(edges[-1]))
        ax.set_xticks(np.arange(tick_start, tick_end + 1, 10))

    ax.set_title(title or "Lead Time Distribution")
    fig.tight_layout()
    if save_fig:
        fig.savefig(save_fig, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pr_vs_lead_time(
        pr: dict,
        save_fig: Optional[str],
        title: Optional[str] = None,
):
    """Plot precision/recall as a function of lead time."""
    fig, ax = plt.subplots(figsize=(7, 4))
    lead_times = pr.get("lead_times", []) if isinstance(pr, dict) else []
    precision = pr.get("precision", []) if isinstance(pr, dict) else []
    recall = pr.get("recall", []) if isinstance(pr, dict) else []

    if not lead_times:
        ax.text(0.5, 0.5, "No PR-lead-time data", ha="center", va="center", fontsize=10)
        ax.set_axis_off()
    else:
        n = min(len(lead_times), len(precision), len(recall))
        x = np.asarray(lead_times[:n], dtype=float)
        p = np.asarray(precision[:n], dtype=float)
        r = np.asarray(recall[:n], dtype=float)
        ax.plot(x, p, marker="o", color="#1E88E5", label="Precision")
        ax.plot(x, r, marker="o", color="#43A047", label="Recall")
        ax.set_xlabel("Lead time (min)")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)
        tick_start = int(np.floor(x.min()))
        tick_end = int(np.ceil(x.max()))
        ax.set_xticks(np.arange(tick_start, tick_end + 1, 10))
        ax.set_yticks(np.arange(0.0, 1.01, 0.1))
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(loc="best")

    ax.set_title(title or "Precision-Recall vs Lead Time")
    fig.tight_layout()
    if save_fig:
        fig.savefig(save_fig, dpi=160, bbox_inches="tight")
    plt.close(fig)
