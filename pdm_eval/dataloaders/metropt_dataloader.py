#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning DataModule + sliding-window dataset for MetroPT sequences.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("This dataset is empty.")


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


class MetroPTDataModule(LightningDataModule):
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        batch_size: int,
        stride: int = 1,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.stride = stride
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.train_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = MetroPTSequenceDataset(
            data=self.data, seq_len=self.seq_len, stride=self.stride
        )

    def _empty_dataloader(self):
        return DataLoader(
            EmptyDataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup()
        if not self.train_dataset:
            return self._empty_dataloader()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
