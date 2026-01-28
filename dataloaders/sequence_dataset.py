#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequence dataset for sliding-window time series.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MetroPTSequenceDataset(Dataset):
    """Return sliding windows of shape (seq_len, n_features)."""

    def __init__(self, data: np.ndarray, seq_len: int, stride: int = 1) -> None:
        if data is None or len(data) == 0:
            raise ValueError("Empty data passed to MetroPTSequenceDataset.")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")
        if stride < 1:
            raise ValueError("stride must be >= 1.")
        self.data = data.astype(np.float32, copy=False)
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.n_samples = self.data.shape[0]
        self.n_windows = max(0, (self.n_samples - self.seq_len) // self.stride + 1)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx < 0 or idx >= self.n_windows:
            raise IndexError("Index out of range in MetroPTSequenceDataset.")
        start = idx * self.stride
        end = start + self.seq_len
        window = self.data[start:end]
        anchor = end - 1
        return torch.from_numpy(window), anchor
