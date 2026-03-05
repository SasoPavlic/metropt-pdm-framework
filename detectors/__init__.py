#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector factory and re-exports.
"""

from __future__ import annotations

from .autoencoder_detector import AutoencoderDetector
from .base import BaseDetector
from .iforest_detector import IsolationForestDetector
from .nianetvae_detector import NiaNetVAEPretrainedDetector


def get_detector(detector_type: str, **kwargs) -> BaseDetector:
    """Return a detector instance by type name."""
    if not detector_type:
        raise ValueError("detector_type must be a non-empty string.")
    key = str(detector_type).strip().lower()
    if key in {"iforest", "isolation_forest", "isolationforest"}:
        return IsolationForestDetector(**kwargs)
    if key in {"autoencoder", "ae"}:
        return AutoencoderDetector(**kwargs)
    if key in {"nianetvae", "nianetvae_pretrained"}:
        return NiaNetVAEPretrainedDetector(**kwargs)
    raise ValueError(f"Unsupported detector_type={detector_type!r}.")


__all__ = [
    "BaseDetector",
    "AutoencoderDetector",
    "IsolationForestDetector",
    "NiaNetVAEPretrainedDetector",
    "get_detector",
]
