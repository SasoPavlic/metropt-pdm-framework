#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector-agnostic utilities.
"""

from __future__ import annotations

import pandas as pd


def time_based_train_mask(index: pd.Index, train_minutes: float, min_rows: int = 30) -> pd.Series:
    """
    Build a boolean mask selecting rows within the first `train_minutes` minutes
    from the start of the index. Falls back to the first `min_rows` rows if the
    time window contains too few points or if the index is not datetime-like.
    """
    mask = pd.Series(False, index=index)
    if len(index) == 0 or train_minutes is None or train_minutes <= 0:
        return mask
    try:
        start = pd.to_datetime(index.min())
        cutoff = start + pd.Timedelta(minutes=float(train_minutes))
        mask = pd.Series(
            (pd.to_datetime(index) >= start) & (pd.to_datetime(index) <= cutoff),
            index=index,
        )
    except Exception:
        mask = pd.Series(False, index=index)
    if mask.sum() < 2:
        n_train = min(len(index), max(2, min_rows))
        mask.iloc[:n_train] = True
    return mask
