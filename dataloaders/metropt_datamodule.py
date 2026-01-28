#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning DataModule for MetroPT sequence windows.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .sequence_dataset import MetroPTSequenceDataset


class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("This dataset is empty.")


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
