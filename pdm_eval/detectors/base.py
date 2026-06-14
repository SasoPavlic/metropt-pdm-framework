#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base detector interface for the anomaly detection pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseDetector(ABC):
    """Abstract detector interface (fit + score)."""

    name: str = "base"

    @abstractmethod
    def fit(self, X_train: pd.DataFrame) -> "BaseDetector":
        """Fit the detector on training data."""
        raise NotImplementedError

    @abstractmethod
    def score(self, X: pd.DataFrame) -> pd.Series:
        """Return anomaly scores for X (higher = more anomalous)."""
        raise NotImplementedError
