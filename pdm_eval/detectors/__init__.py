#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector factory and re-exports.
"""

from __future__ import annotations

from .base import BaseDetector
from .iforest_detector import IsolationForestDetector
from .imported_recurrent_autoencoder_detector import ImportedRecurrentAutoencoderDetector
from .recurrent_autoencoder_detector import (
    RecurrentSAEDetector,
    RecurrentVAEDetector,
    is_recurrent_sae_type,
    is_recurrent_vae_type,
)


def get_detector(detector_type: str, **kwargs) -> BaseDetector:
    """Return a detector instance by type name."""
    if not detector_type:
        raise ValueError("detector_type must be a non-empty string.")
    key = str(detector_type).strip().lower()
    if key in {"iforest", "isolation_forest", "isolationforest"}:
        return IsolationForestDetector(**kwargs)
    if is_recurrent_vae_type(key):
        return RecurrentVAEDetector(**kwargs)
    if is_recurrent_sae_type(key):
        return RecurrentSAEDetector(**kwargs)
    if key == "imported-recurrent-autoencoder":
        return ImportedRecurrentAutoencoderDetector(**kwargs)
    raise ValueError(f"Unsupported detector_type={detector_type!r}.")


__all__ = [
    "BaseDetector",
    "ImportedRecurrentAutoencoderDetector",
    "IsolationForestDetector",
    "RecurrentSAEDetector",
    "RecurrentVAEDetector",
    "get_detector",
]
