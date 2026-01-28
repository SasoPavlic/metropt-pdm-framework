#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolation Forest detector wrapper.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .base import BaseDetector


class IsolationForestDetector(BaseDetector):
    name = "iforest"

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: str = "auto",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[IsolationForest] = None

    def fit(self, X_train: pd.DataFrame) -> "IsolationForestDetector":
        if X_train is None or X_train.shape[0] < 2:
            raise ValueError("Need at least 2 samples to fit IsolationForest.")
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X_train_s)
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None or self.scaler is None:
            raise ValueError("Detector must be fitted before scoring.")
        X_s = self.scaler.transform(X)
        scores = -self.model.decision_function(X_s)
        return pd.Series(scores, index=X.index, name="anom_score")
