#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local recurrent SAE/VAE detectors for MetroPT single-mode experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import BaseDetector


RECURRENT_VAE_TYPE = "recurrent-vae"
RECURRENT_SAE_TYPE = "recurrent-sae"
LOCAL_RECURRENT_TYPES = {RECURRENT_VAE_TYPE, RECURRENT_SAE_TYPE}


@dataclass
class RecurrentAutoencoderConfig:
    sequence_len: int = 200
    stride: int = 1
    score_stride: Optional[int] = 1
    hidden_size: int = 64
    num_layers: int = 2
    latent_dim: int = 32
    epochs: int = 30
    batch_size: int = 64
    lr: float = 0.003
    weight_decay: float = 1e-5
    device: str = "cpu"
    use_scaler: bool = True
    num_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = False
    random_state: int = 42


def normalize_detector_type(detector_type: str) -> str:
    return str(detector_type or "").strip().lower()


def is_recurrent_vae_type(detector_type: str) -> bool:
    return normalize_detector_type(detector_type) == RECURRENT_VAE_TYPE


def is_recurrent_sae_type(detector_type: str) -> bool:
    return normalize_detector_type(detector_type) == RECURRENT_SAE_TYPE


def is_local_recurrent_type(detector_type: str) -> bool:
    return normalize_detector_type(detector_type) in LOCAL_RECURRENT_TYPES


class _RecurrentVAEModel:
    def __new__(cls, *args, **kwargs):
        import torch.nn as nn

        class RecurrentVAEModel(nn.Module):
            def __init__(
                self,
                input_dim: int,
                seq_len: int,
                hidden_size: int,
                num_layers: int,
                latent_dim: int,
            ) -> None:
                super().__init__()
                self.seq_len = int(seq_len)
                self.encoder = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                )
                self.fc_mu = nn.Linear(hidden_size, latent_dim)
                self.fc_logvar = nn.Linear(hidden_size, latent_dim)
                self.decoder = nn.LSTM(
                    input_size=latent_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                )
                self.output = nn.Linear(hidden_size, input_dim)

            def encode(self, x):
                _, (hidden, _) = self.encoder(x)
                h_last = hidden[-1]
                return self.fc_mu(h_last), self.fc_logvar(h_last)

            def reparameterize(self, mu, logvar):
                import torch

                if not self.training:
                    return mu
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)
                decoded, _ = self.decoder(repeated)
                return self.output(decoded)

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar

            def reconstruct_from_mean(self, x):
                mu, _ = self.encode(x)
                return self.decode(mu)

        return RecurrentVAEModel(*args, **kwargs)


class _RecurrentSAEModel:
    def __new__(cls, *args, **kwargs):
        import torch.nn as nn

        class RecurrentSAEModel(nn.Module):
            def __init__(
                self,
                input_dim: int,
                seq_len: int,
                hidden_size: int,
                num_layers: int,
                latent_dim: int,
            ) -> None:
                super().__init__()
                self.seq_len = int(seq_len)
                self.encoder = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                )
                self.fc_latent = nn.Linear(hidden_size, latent_dim)
                self.latent_activation = nn.Sigmoid()
                self.decoder = nn.LSTM(
                    input_size=latent_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                )
                self.output = nn.Linear(hidden_size, input_dim)

            def encode(self, x):
                _, (hidden, _) = self.encoder(x)
                h_last = hidden[-1]
                return self.latent_activation(self.fc_latent(h_last))

            def decode(self, z):
                repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)
                decoded, _ = self.decoder(repeated)
                return self.output(decoded)

            def forward(self, x):
                z = self.encode(x)
                return self.decode(z), z

        return RecurrentSAEModel(*args, **kwargs)


class _BaseRecurrentDetector(BaseDetector):
    name = "recurrent-base"

    def __init__(
        self,
        sequence_len: int = 200,
        stride: int = 1,
        score_stride: Optional[int] = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        latent_dim: int = 32,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 0.003,
        weight_decay: float = 1e-5,
        device: str = "cpu",
        use_scaler: bool = True,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self.extra_kwargs = dict(kwargs)
        self.cfg = RecurrentAutoencoderConfig(
            sequence_len=int(sequence_len),
            stride=int(stride),
            score_stride=None if score_stride is None else int(score_stride),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            latent_dim=int(latent_dim),
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            weight_decay=float(weight_decay),
            device=str(device),
            use_scaler=bool(use_scaler),
            num_workers=int(num_workers),
            persistent_workers=bool(persistent_workers),
            pin_memory=bool(pin_memory),
            random_state=int(random_state),
        )
        if self.cfg.sequence_len < 1:
            raise ValueError("sequence_len must be >= 1.")
        if self.cfg.stride < 1:
            raise ValueError("stride must be >= 1.")
        if self.cfg.score_stride is not None and self.cfg.score_stride < 1:
            raise ValueError("score_stride must be >= 1.")
        if self.cfg.hidden_size < 1 or self.cfg.num_layers < 1 or self.cfg.latent_dim < 1:
            raise ValueError("hidden_size, num_layers, and latent_dim must be positive.")
        if self.cfg.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        self.scaler: Optional[StandardScaler] = None
        self.model = None
        self.input_dim: Optional[int] = None
        self.device_ = "cpu"

    def _require_torch(self):
        try:
            import torch  # noqa: F401
        except Exception as exc:
            raise ImportError("PyTorch is required for recurrent SAE/VAE detectors.") from exc

    def _set_seed(self) -> None:
        import torch

        np.random.seed(self.cfg.random_state)
        torch.manual_seed(self.cfg.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.random_state)

    def _resolve_device(self):
        import torch

        if self.cfg.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _prepare_values(self, X: pd.DataFrame, *, fit_scaler: bool) -> np.ndarray:
        X_vals = X.to_numpy(dtype=np.float32, copy=True)
        if self.cfg.use_scaler:
            if fit_scaler:
                self.scaler = StandardScaler()
                X_vals = self.scaler.fit_transform(X_vals)
            elif self.scaler is not None:
                X_vals = self.scaler.transform(X_vals)
        return X_vals.astype(np.float32, copy=False)

    def _make_loader(self, X_vals: np.ndarray, *, stride: int, shuffle: bool):
        import torch
        from torch.utils.data import DataLoader

        from pdm_eval.dataloaders.metropt_dataloader import MetroPTSequenceDataset

        dataset = MetroPTSequenceDataset(
            data=X_vals,
            seq_len=self.cfg.sequence_len,
            stride=stride,
        )
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.cfg.random_state)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.persistent_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=False,
            generator=generator,
        )
        return dataset, loader

    def _build_model(self, input_dim: int):
        raise NotImplementedError

    def _training_loss(self, batch):
        raise NotImplementedError

    def _score_batch(self, batch):
        raise NotImplementedError

    def fit(self, X_train: pd.DataFrame) -> "_BaseRecurrentDetector":
        self._require_torch()
        import torch

        if X_train is None or X_train.shape[0] < self.cfg.sequence_len:
            raise ValueError(
                f"Need at least sequence_len={self.cfg.sequence_len} samples to fit {type(self).__name__}."
            )

        self._set_seed()
        X_vals = self._prepare_values(X_train, fit_scaler=True)
        self.input_dim = int(X_vals.shape[1])
        self.device_ = self._resolve_device()
        self.model = self._build_model(self.input_dim).to(self.device_)

        if self.cfg.epochs <= 0:
            return self

        dataset, loader = self._make_loader(X_vals, stride=self.cfg.stride, shuffle=True)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        print(
            f"[INFO] {self.name} train windows={len(dataset)} "
            f"(seq_len={self.cfg.sequence_len}, stride={self.cfg.stride}), "
            f"epochs={self.cfg.epochs}, batch_size={self.cfg.batch_size}, device={self.device_}"
        )

        self.model.train()
        log_every = max(1, self.cfg.epochs // 5)
        for epoch in range(1, self.cfg.epochs + 1):
            losses = []
            for windows, _anchors in loader:
                windows = windows.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                loss = self._training_loss(windows)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if epoch == 1 or epoch == self.cfg.epochs or epoch % log_every == 0:
                mean_loss = float(np.mean(losses)) if losses else float("nan")
                print(f"[INFO] {self.name} epoch={epoch}/{self.cfg.epochs} loss={mean_loss:.6f}")

        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        self._require_torch()
        import torch

        if self.model is None:
            raise ValueError("Detector must be fitted before scoring.")

        X_vals = self._prepare_values(X, fit_scaler=False)
        score_stride = self.cfg.score_stride or self.cfg.stride
        dataset, loader = self._make_loader(X_vals, stride=score_stride, shuffle=False)
        print(
            f"[INFO] {self.name} score windows={len(dataset)} "
            f"(seq_len={self.cfg.sequence_len}, stride={score_stride}), "
            f"batch_size={self.cfg.batch_size}"
        )

        scores = np.full((X_vals.shape[0],), np.nan, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for windows, anchors in loader:
                windows = windows.to(self.device_)
                window_scores = self._score_batch(windows).detach().cpu().numpy()
                scores[anchors.numpy()] = window_scores
        return pd.Series(scores, index=X.index, name="anom_score")


class RecurrentVAEDetector(_BaseRecurrentDetector):
    name = RECURRENT_VAE_TYPE

    def __init__(self, kl_beta: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.kl_beta = float(kl_beta)

    def _build_model(self, input_dim: int):
        return _RecurrentVAEModel(
            input_dim=input_dim,
            seq_len=self.cfg.sequence_len,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            latent_dim=self.cfg.latent_dim,
        )

    def _training_loss(self, batch):
        import torch
        import torch.nn.functional as F

        recon, mu, logvar = self.model(batch)
        recon_loss = F.mse_loss(recon, batch, reduction="mean")
        kl_loss = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return recon_loss + self.kl_beta * kl_loss

    def _score_batch(self, batch):
        recon = self.model.reconstruct_from_mean(batch)
        return ((recon - batch) ** 2).mean(dim=(1, 2))


class RecurrentSAEDetector(_BaseRecurrentDetector):
    name = RECURRENT_SAE_TYPE

    def __init__(
        self,
        sparsity_beta: float = 0.05,
        sparsity_rho: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sparsity_beta = float(sparsity_beta)
        self.sparsity_rho = float(sparsity_rho)
        if not 0.0 < self.sparsity_rho < 1.0:
            raise ValueError("sparsity_rho must be in (0, 1).")

    def _build_model(self, input_dim: int):
        return _RecurrentSAEModel(
            input_dim=input_dim,
            seq_len=self.cfg.sequence_len,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            latent_dim=self.cfg.latent_dim,
        )

    def _sparsity_loss(self, latent):
        import torch

        eps = 1e-7
        rho = torch.tensor(self.sparsity_rho, device=latent.device, dtype=latent.dtype)
        rho_hat = torch.clamp(latent.mean(dim=0), eps, 1.0 - eps)
        kl = rho * torch.log(rho / rho_hat) + (1.0 - rho) * torch.log((1.0 - rho) / (1.0 - rho_hat))
        return kl.sum()

    def _training_loss(self, batch):
        import torch.nn.functional as F

        recon, latent = self.model(batch)
        recon_loss = F.mse_loss(recon, batch, reduction="mean")
        return recon_loss + self.sparsity_beta * self._sparsity_loss(latent)

    def _score_batch(self, batch):
        recon, _latent = self.model(batch)
        return ((recon - batch) ** 2).mean(dim=(1, 2))


def build_recurrent_detector_kwargs(
    *,
    sequence_len: int = 200,
    stride: int = 1,
    score_stride: Optional[int] = 1,
    hidden_size: int = 64,
    num_layers: int = 2,
    latent_dim: int = 32,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.003,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    use_scaler: bool = True,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    random_state: int = 42,
) -> dict:
    return {
        "sequence_len": sequence_len,
        "stride": stride,
        "score_stride": score_stride,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "latent_dim": latent_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "device": device,
        "use_scaler": use_scaler,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "pin_memory": pin_memory,
        "random_state": random_state,
    }


__all__ = [
    "LOCAL_RECURRENT_TYPES",
    "RECURRENT_SAE_TYPE",
    "RECURRENT_VAE_TYPE",
    "RecurrentAutoencoderConfig",
    "RecurrentSAEDetector",
    "RecurrentVAEDetector",
    "build_recurrent_detector_kwargs",
    "is_local_recurrent_type",
    "is_recurrent_sae_type",
    "is_recurrent_vae_type",
    "normalize_detector_type",
]
