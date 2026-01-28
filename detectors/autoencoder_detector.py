#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoencoder detector (PyTorch Lightning).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import BaseDetector


@dataclass
class AEConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    use_scaler: bool = True
    model_path: Optional[str] = None
    load_pretrained: bool = True
    sequence_len: int = 200
    stride: int = 1
    beta: float = 0.1
    rnn_type: str = "LSTM"
    hidden_size: int = 32
    num_layers: int = 3
    latent_dim: Optional[int] = None
    bidirectional: bool = False
    num_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = False


class AutoencoderDetector(BaseDetector):
    name = "autoencoder"

    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cpu",
        use_scaler: bool = True,
        model_path: Optional[str] = None,
        load_pretrained: bool = True,
        sequence_len: int = 200,
        stride: int = 1,
        beta: float = 0.1,
        rnn_type: str = "LSTM",
        hidden_size: int = 32,
        num_layers: int = 3,
        latent_dim: Optional[int] = None,
        bidirectional: bool = False,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        **kwargs,
    ) -> None:
        self.extra_kwargs = dict(kwargs)
        self.cfg = AEConfig(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            use_scaler=use_scaler,
            model_path=model_path,
            load_pretrained=load_pretrained,
            sequence_len=sequence_len,
            stride=stride,
            beta=beta,
            rnn_type=rnn_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            latent_dim=latent_dim,
            bidirectional=bidirectional,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
        )
        self.scaler: Optional[StandardScaler] = None
        self.model = None
        self.input_dim: Optional[int] = None

    def _require_torch(self):
        try:
            import torch  # noqa: F401
            import lightning  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "PyTorch + Lightning are required for AutoencoderDetector. "
                "Install lightning/torch or switch DETECTOR_TYPE back to 'iforest'."
            ) from exc

    def _load_model(self, input_dim: int):
        if not self.cfg.model_path:
            return None
        path = Path(self.cfg.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        import torch
        obj = torch.load(path, map_location="cpu")
        return obj

    def fit(self, X_train: pd.DataFrame) -> "AutoencoderDetector":
        self._require_torch()
        import torch
        from lightning import Trainer

        from dataloaders.metropt_datamodule import MetroPTDataModule
        from detectors.lightning_vae_module import RecurrentVAELightningModule

        if X_train is None or X_train.shape[0] < 2:
            raise ValueError("Need at least 2 samples to fit AutoencoderDetector.")

        X_vals = X_train.to_numpy(dtype=np.float32, copy=True)
        if self.cfg.use_scaler:
            self.scaler = StandardScaler()
            X_vals = self.scaler.fit_transform(X_vals)

        self.input_dim = X_vals.shape[1]
        latent_dim = self.cfg.latent_dim or max(1, self.input_dim // 2)

        self.model = RecurrentVAELightningModule(
            input_dim=self.input_dim,
            seq_len=self.cfg.sequence_len,
            rnn_type=self.cfg.rnn_type,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            latent_dim=latent_dim,
            bidirectional=self.cfg.bidirectional,
            beta=self.cfg.beta,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        if self.cfg.model_path and self.cfg.load_pretrained:
            obj = self._load_model(self.input_dim)
            if hasattr(obj, "state_dict"):
                self.model.load_state_dict(obj.state_dict())
            elif isinstance(obj, dict):
                self.model.load_state_dict(obj)

        if self.cfg.epochs <= 0:
            return self

        dm = MetroPTDataModule(
            data=X_vals,
            seq_len=self.cfg.sequence_len,
            batch_size=self.cfg.batch_size,
            stride=self.cfg.stride,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.persistent_workers,
            pin_memory=self.cfg.pin_memory,
        )
        dm.setup()

        accelerator = "gpu" if self.cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
        trainer = Trainer(
            max_epochs=self.cfg.epochs,
            accelerator=accelerator,
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
        )
        trainer.fit(self.model, datamodule=dm)

        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        self._require_torch()
        import torch
        from torch.utils.data import DataLoader

        from dataloaders.sequence_dataset import MetroPTSequenceDataset

        if self.model is None:
            raise ValueError("Detector must be fitted or loaded before scoring.")

        X_vals = X.to_numpy(dtype=np.float32, copy=True)
        if self.scaler is not None:
            X_vals = self.scaler.transform(X_vals)

        dataset = MetroPTSequenceDataset(
            data=X_vals, seq_len=self.cfg.sequence_len, stride=self.cfg.stride
        )
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.persistent_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=False,
        )

        scores = np.full((X_vals.shape[0],), np.nan, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for windows, anchors in loader:
                windows = windows.to(self.model.device)
                recon, _, _ = self.model(windows)
                err = (recon - windows) ** 2
                window_scores = err.mean(dim=(1, 2)).detach().cpu().numpy()
                scores[anchors.numpy()] = window_scores

        # TODO: consider alternative aggregation (max/last) and alignment strategy.
        return pd.Series(scores, index=X.index, name="anom_score")
