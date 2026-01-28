#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-processing for detector scores (risk score, thresholds, labels).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .base import BaseDetector


def _as_series(values, index: pd.Index, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reindex(index).astype(float).rename(name)
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] != len(index):
        raise ValueError("Score length does not match index length.")
    return pd.Series(arr, index=index, name=name)


def build_risk_score(scores_train: pd.Series, scores_all: pd.Series) -> pd.Series:
    """Percentile rank of scores_all relative to scores_train (0..1)."""
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
    return pd.Series(risk_score, index=scores_all.index, name="risk_score")


def compute_threshold(scores_train: pd.Series) -> Tuple[float, dict]:
    """Compute Q3 + 3*IQR threshold from training scores."""
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
        thr = float(finite.max()) if finite.size else 0.0
    info = {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "label_rule": f"Q3+3*IQR (Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f})",
    }
    return thr, info


def train_and_score(
    detector: BaseDetector,
    X_train: pd.DataFrame,
    X_all: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """Fit detector on X_train and score X_all with shared post-processing."""
    if X_train is None or X_all is None:
        raise ValueError("X_train and X_all must not be None.")
    if X_train.shape[0] < 2 or X_all.shape[0] < 2:
        raise ValueError("Need at least 2 samples in training and scoring sets.")

    detector.fit(X_train)
    scores_all = _as_series(detector.score(X_all), X_all.index, "anom_score")
    scores_train = _as_series(detector.score(X_train), X_train.index, "anom_score_train")

    risk_score = build_risk_score(scores_train, scores_all)
    thr, thr_info = compute_threshold(scores_train)
    is_anom = np.where(scores_all.values > thr, 1, 0)

    out = pd.DataFrame(index=X_all.index)
    out["anom_score"] = scores_all.values
    out["risk_score"] = risk_score.values
    out["is_anomaly"] = is_anom

    info = {
        "n_total": int(X_all.shape[0]),
        "n_train": int(X_train.shape[0]),
        "pct_anom": float(out["is_anomaly"].mean()),
        "threshold": float(thr),
        "label_rule": thr_info["label_rule"],
        "pca_components": None,
        "n_features": int(X_all.shape[1]),
        "q1": thr_info["q1"],
        "q3": thr_info["q3"],
        "iqr": thr_info["iqr"],
        "detector": getattr(detector, "name", type(detector).__name__),
    }
    return out, info
