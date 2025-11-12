#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading and preprocessing utilities for the IsolationForest anomaly helper.
"""

from __future__ import annotations

import os
import platform
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def convert_wsl_to_windows_path(path: str) -> str:
    """Convert /mnt/<drive>/... to Windows-style paths when running on Windows."""
    if platform.system() != "Windows":
        return path
    m = re.match(r"^/mnt/([a-zA-Z])(/.*)?$", path)
    if not m:
        return path
    drive = m.group(1).upper()
    rest = m.group(2) or ""
    return f"{drive}:{rest}".replace("/", "\\")


def infer_timestamp_column(df: pd.DataFrame, user_ts: Optional[str]) -> str:
    if user_ts and user_ts in df.columns:
        return user_ts
    for c in ["timestamp", "time", "datetime", "date", "Date", "Timestamp", "Time"]:
        if c in df.columns:
            return c
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    raise ValueError("Could not infer timestamp column. Provide --timestamp_col.")


def load_csv(input_path: str, timestamp_col: Optional[str], drop_unnamed: bool) -> pd.DataFrame:
    """Load a CSV/TXT file, parse timestamp column, and index by time."""
    input_path = convert_wsl_to_windows_path(input_path)  # noop on non-Windows
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in [".csv", ".txt"]:
        raise ValueError(f"Only CSV/TXT files are supported for this script (got: {ext}).")

    df = pd.read_csv(input_path)

    if drop_unnamed:
        for c in list(df.columns):
            if str(c).lower().startswith("unnamed"):
                df = df.drop(columns=[c])

    ts_col = infer_timestamp_column(df, timestamp_col)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True).set_index(ts_col)
    return df


def select_numeric_features(
    df: pd.DataFrame,
    prefer: Optional[List[str]] = None,
    exclude_quasi_binary: bool = True,
    quasi_unique_threshold: int = 3
) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude_quasi_binary:
        nb = []
        for c in num_cols:
            nunq = int(min(50_000, df[c].nunique(dropna=True)))
            if nunq >= quasi_unique_threshold:
                nb.append(c)
        num_cols = nb
    if prefer:
        chosen = [c for c in prefer if c in num_cols]
        chosen += [c for c in num_cols if c not in chosen]
        return chosen
    return num_cols


def pre_downsample(df: pd.DataFrame, rule: Optional[str]) -> pd.DataFrame:
    if not rule:
        return df
    num = df.select_dtypes(include=[np.number])
    agg = num.resample(rule).median()
    return agg.dropna(how="all")


def top_k_by_variance(df_num: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 0 or k >= df_num.shape[1]:
        return df_num
    vars_ = df_num.var(axis=0, skipna=True)
    topk = vars_.sort_values(ascending=False).head(k).index.tolist()
    return df_num[topk]


def build_rolling_features(
    df_num: pd.DataFrame,
    rolling_window: str = "600s",
    min_periods: int = 1
) -> pd.DataFrame:
    rolled = df_num.rolling(rolling_window, min_periods=min_periods)
    agg = rolled.aggregate(["mean", "median", "std", "skew", "min", "max"])
    if isinstance(agg.columns, pd.MultiIndex):
        agg.columns = ["__".join(map(str, col)).strip() for col in agg.columns.values]
    else:
        agg.columns = [str(col) for col in agg.columns]
    agg = agg.ffill().bfill()
    return agg



def parse_maintenance_windows(
    windows: Optional[List[str]],
    maintenance_csv: Optional[str],
    use_default_windows: bool,
    default_windows: Optional[List[Tuple[str, str, str, str]]] = None,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, Optional[str], Optional[str]]]:
    """
    Returns a list of (start, end, id, severity).
    - windows: flat list like ["start1","end1","start2","end2",...] → ids auto-assigned
    - maintenance_csv: CSV with columns (start,end[,id][,severity]) with flexible header names
    - default_windows: fallback list of tuples when use_default_windows is True
    """
    out: List[Tuple[pd.Timestamp, pd.Timestamp, Optional[str], Optional[str]]] = []

    # CSV
    if maintenance_csv and os.path.exists(maintenance_csv):
        mdf = pd.read_csv(maintenance_csv)
        cols = {c.lower(): c for c in mdf.columns}
        start_key = next((cols[k] for k in ["start", "begin", "from", "t_start", "start_time"] if k in cols), None)
        end_key = next((cols[k] for k in ["end", "finish", "to", "t_end", "end_time"] if k in cols), None)
        id_key = next((cols[k] for k in ["id", "name", "label", "nr", "number"] if k in cols), None)
        sev_key = next((cols[k] for k in ["severity", "level", "prio"] if k in cols), None)
        if start_key and end_key:
            s = pd.to_datetime(mdf[start_key], errors="coerce")
            e = pd.to_datetime(mdf[end_key], errors="coerce")
            ids = mdf[id_key].astype(str) if id_key else pd.Series([None] * len(mdf))
            sevs = mdf[sev_key].astype(str) if sev_key else pd.Series([None] * len(mdf))
            m = pd.DataFrame({"s": s, "e": e, "id": ids, "sev": sevs}).dropna(subset=["s", "e"])
            for _, row in m.iterrows():
                if row["e"] >= row["s"]:
                    rid = row["id"] if pd.notna(row["id"]) else None
                    rsev = row["sev"] if pd.notna(row["sev"]) else None
                    out.append((row["s"], row["e"], rid, rsev))

    # Manual list: pairs of start/end
    if windows:
        if len(windows) % 2 != 0:
            raise ValueError("--maintenance_windows expects an even number of timestamps (start/end pairs).")
        pair_idx = 1
        for i in range(0, len(windows), 2):
            s = pd.to_datetime(windows[i], errors="coerce")
            e = pd.to_datetime(windows[i + 1], errors="coerce")
            if pd.notna(s) and pd.notna(e) and e >= s:
                out.append((s, e, f"win{pair_idx}", None))
                pair_idx += 1

    # Defaults
    if use_default_windows and default_windows:
        for s, e, wid, sev in default_windows:
            out.append((pd.to_datetime(s), pd.to_datetime(e), wid, sev))

    # Normalize & sort
    cleaned: List[Tuple[pd.Timestamp, pd.Timestamp, Optional[str], Optional[str]]] = []
    seen = set()
    for item in out:
        s, e, wid, sev = item
        if pd.isna(s) or pd.isna(e) or e < s:
            continue
        key = (pd.to_datetime(s), pd.to_datetime(e), wid, sev)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((pd.to_datetime(s), pd.to_datetime(e), wid, sev))
    cleaned.sort(key=lambda t: t[0])
    return cleaned


def build_operation_phase(
    index: pd.DatetimeIndex,
    windows: List[Tuple],
    pre_hours: float = 2.0,
) -> pd.Series:
    """
    Build an operation phase indicator:
    0=normal, 1=pre-maintenance, 2=maintenance.
    Maintenance overrides pre-maintenance when overlapping.
    """
    phase = pd.Series(np.zeros(len(index), dtype=np.int8), index=index, name="operation_phase")
    if index.size == 0 or not windows:
        return phase
    try:
        pre_delta = pd.to_timedelta(float(pre_hours), unit="h")
    except Exception:
        pre_delta = pd.to_timedelta(0, unit="h")

    arr = phase.to_numpy()
    for item in windows:
        if len(item) < 2:
            continue
        try:
            start = pd.to_datetime(item[0])
            end = pd.to_datetime(item[1])
        except Exception:
            continue
        if pd.isna(start) or pd.isna(end):
            continue
        if end < start:
            continue

        maint_mask = (index >= start) & (index <= end)
        if maint_mask.any():
            arr[maint_mask] = np.int8(2)

        if pre_delta is not None and pre_delta > pd.Timedelta(0):
            pre_start = start - pre_delta
            pre_mask = (index >= pre_start) & (index < start)
            if pre_mask.any():
                zero_mask = arr == 0
                combined = pre_mask & zero_mask
                if combined.any():
                    arr[combined] = np.int8(1)

    phase[:] = arr
    return phase
