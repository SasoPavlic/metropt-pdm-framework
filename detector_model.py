#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modeling utilities for the anomaly detection pipeline.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def _train_iforest_core(
    X_train: pd.DataFrame,
    X_all: pd.DataFrame,
    random_state: int,
) -> Tuple[pd.DataFrame, dict]:
    """
    Fit the detector on X_train and score X_all using a shared scaler.

    Threshold is derived from the training scores only (Q3 + 3*IQR) and then
    applied to all scored points in X_all.
    """
    if X_train is None or X_all is None:
        raise ValueError("X_train and X_all must not be None.")
    if X_train.shape[0] < 2 or X_all.shape[0] < 2:
        raise ValueError("Need at least 2 samples in training and scoring sets.")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_all_s = scaler.transform(X_all)

    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_s)

    # Larger score => more anomalous
    scores_all = -model.decision_function(X_all_s)
    scores_all = pd.Series(scores_all, index=X_all.index, name="anom_score")

    # Also score the training part explicitly for stable thresholding
    scores_train = -model.decision_function(X_train_s)
    scores_train = pd.Series(scores_train, index=X_train.index, name="anom_score_train")

    # Risk-normalized score based on training distribution (percentile rank)
    train_vals = scores_train.values.astype(float)
    train_vals = train_vals[np.isfinite(train_vals)]
    if train_vals.size:
        sorted_train = np.sort(train_vals)
        all_vals = scores_all.values.astype(float)
        all_vals = np.where(np.isfinite(all_vals), all_vals, -np.inf)
        ranks = np.searchsorted(sorted_train, all_vals, side="right")
        risk_score = ranks / float(sorted_train.size)
        risk_score = np.clip(risk_score, 0.0, 1.0)
    else:
        risk_score = np.zeros_like(scores_all.values, dtype=float)

    # Thresholding via Q3 + 3*IQR on the training scores only
    train_vals = scores_train.values.astype(float)
    q1, q3 = np.nanpercentile(train_vals, [25, 75])
    if np.isnan(q1) or np.isnan(q3):
        finite = train_vals[np.isfinite(train_vals)]
        fallback = float(finite.mean()) if finite.size else 0.0
        q1 = q3 = fallback
    iqr = max(0.0, float(q3 - q1))
    thr = float(q3 + 3.0 * iqr)
    if not np.isfinite(thr):
        finite = train_vals[np.isfinite(train_vals)]
        thr = float(finite.max()) if finite.size else float(0.0)
    is_anom = np.where(scores_all.values > thr, 1, 0)
    rule = f"Q3+3*IQR (Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f})"

    out = pd.DataFrame(index=X_all.index)
    out["anom_score"] = scores_all.values
    out["risk_score"] = risk_score
    out["is_anomaly"] = is_anom

    info = {
        "n_total": int(X_all.shape[0]),
        "n_train": int(X_train.shape[0]),
        "pct_anom": float(out["is_anomaly"].mean()),
        "threshold": thr,
        "label_rule": rule,
        "pca_components": None,
        "n_features": int(X_all.shape[1]),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
    }
    return out, info


def _time_based_train_mask(index: pd.Index, train_minutes: float, min_rows: int = 30) -> pd.Series:
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


def train_iforest_and_score(
    X: pd.DataFrame,
    train_minutes: float = 1440.0,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    """
    Train on the first `train_minutes` of X (chronological) and score all rows.
    If the time window yields too few rows, fall back to the first `min_rows`.
    """
    if len(X) < 2:
        raise ValueError("Need at least 2 samples to train the detector.")
    train_mask = _time_based_train_mask(X.index, train_minutes=train_minutes)
    X_train = X.loc[train_mask]
    if X_train.shape[0] < 2:
        raise ValueError("Time-based training selection produced fewer than 2 samples.")
    return _train_iforest_core(X_train, X, random_state=random_state)


def train_iforest_on_slices(
    X_train: pd.DataFrame,
    X_all: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    """
    Train the detector on an arbitrary training slice and score an arbitrary
    (typically disjoint) slice X_all that shares the same feature schema.
    """
    return _train_iforest_core(X_train, X_all, random_state=random_state)
