#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pretrained NiaNetVAE detector (inference-only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import BaseDetector


@dataclass
class NiaNetVAEConfig:
    model_meta_path: str
    model_path: Optional[str] = None
    device: str = "cpu"
    use_scaler: bool = True
    batch_size: int = 256
    sequence_len: Optional[int] = None
    stride: int = 1
    score_stride: Optional[int] = None
    num_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = False


class NiaNetVAEPretrainedDetector(BaseDetector):
    name = "nianetvae_pretrained"

    def __init__(
        self,
        model_meta_path: str,
        model_path: Optional[str] = None,
        device: str = "cpu",
        use_scaler: bool = True,
        batch_size: int = 256,
        sequence_len: Optional[int] = None,
        stride: int = 1,
        score_stride: Optional[int] = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        **kwargs,
    ) -> None:
        self.extra_kwargs = dict(kwargs)
        self.cfg = NiaNetVAEConfig(
            model_meta_path=model_meta_path,
            model_path=model_path,
            device=device,
            use_scaler=use_scaler,
            batch_size=batch_size,
            sequence_len=sequence_len,
            stride=stride,
            score_stride=score_stride,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
        )
        self.scaler: Optional[StandardScaler] = None
        self.meta = self._load_meta(self.cfg.model_meta_path)
        self.model = self._build_model()
        self.model.eval()
        self.input_dim = int(self.meta.get("n_features"))
        self.seq_len = int(self.cfg.sequence_len or self.meta.get("seq_len", 200))

    @staticmethod
    def _load_meta(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"NiaNetVAE meta file not found: {p}")
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse NiaNetVAE meta file: {p}") from exc

    def _require_nianetvae(self):
        try:
            import torch  # noqa: F401
            from nianetvae.models.rnn_vae import RNNVAE  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "NiaNetVAE package import failed. Install local dependency with: "
                "`pip install -e ../NiaNetVAE`"
            ) from exc

    def _resolve_weights_path(self) -> Path:
        if self.cfg.model_path:
            candidate = Path(self.cfg.model_path)
        else:
            weights_file = self.meta.get("weights_file", "model.pt")
            candidate = Path(self.cfg.model_meta_path).parent / str(weights_file)
        if not candidate.exists():
            raise FileNotFoundError(f"NiaNetVAE model weights not found: {candidate}")
        return candidate

    def _build_model(self):
        self._require_nianetvae()
        import torch
        from nianetvae.models.rnn_vae import RNNVAE

        solution = np.asarray(self.meta.get("solution"), dtype=float)
        if solution.shape[0] != 7:
            raise ValueError("NiaNetVAE model_meta.json must contain a 7-gene solution vector.")

        model_config = {
            "data_params": {
                "n_features": int(self.meta.get("n_features")),
                "seq_len": int(self.meta.get("seq_len")),
                "batch_size": int(self.cfg.batch_size),
            },
        }
        model = RNNVAE(solution, **model_config)
        if not getattr(model, "is_valid", False):
            raise ValueError("Resolved NiaNetVAE architecture is invalid; cannot run inference.")

        weights_path = self._resolve_weights_path()
        loaded = torch.load(weights_path, map_location="cpu")
        state_dict = loaded.state_dict() if hasattr(loaded, "state_dict") else loaded
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported model weights format at {weights_path}")
        model.load_state_dict(state_dict, strict=True)

        device = "cuda" if self.cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
        model.to(device)
        self._device = device
        return model

    def fit(self, X_train: pd.DataFrame) -> "NiaNetVAEPretrainedDetector":
        if X_train is None or X_train.shape[0] < 2:
            raise ValueError("Need at least 2 samples to fit NiaNetVAEPretrainedDetector.")
        if X_train.shape[1] != self.input_dim:
            raise ValueError(
                f"Feature mismatch: input has {X_train.shape[1]} features, "
                f"model expects {self.input_dim}."
            )
        if self.cfg.use_scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(X_train.to_numpy(dtype=np.float32, copy=False))
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        self._require_nianetvae()
        import torch
        from torch.utils.data import DataLoader

        from dataloaders.metropt_dataloader import MetroPTSequenceDataset

        if X is None or X.shape[0] < 2:
            raise ValueError("Need at least 2 samples for scoring.")
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Feature mismatch: input has {X.shape[1]} features, "
                f"model expects {self.input_dim}."
            )

        X_vals = X.to_numpy(dtype=np.float32, copy=True)
        if self.scaler is not None:
            X_vals = self.scaler.transform(X_vals)

        score_stride = self.cfg.score_stride or self.cfg.stride
        dataset = MetroPTSequenceDataset(data=X_vals, seq_len=self.seq_len, stride=score_stride)
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
                windows = windows.to(self._device)
                outputs = self.model({"signal": windows})
                recon = outputs["reconstructed"]
                err = (recon - windows) ** 2
                window_scores = err.mean(dim=(1, 2)).detach().cpu().numpy()
                scores[anchors.numpy()] = window_scores

        return pd.Series(scores, index=X.index, name="anom_score")
