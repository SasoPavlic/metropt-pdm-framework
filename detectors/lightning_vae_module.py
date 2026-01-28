#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightningModule wrapper for the recurrent VAE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from lightning import LightningModule

from .vae_models import build_recurrent_vae


class RecurrentVAELightningModule(LightningModule):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        rnn_type: str,
        hidden_size: int,
        num_layers: int,
        latent_dim: int,
        bidirectional: bool,
        beta: float,
        lr: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = build_recurrent_vae(
            input_dim=input_dim,
            seq_len=seq_len,
            rnn_type=rnn_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            latent_dim=latent_dim,
            bidirectional=bidirectional,
        )
        self.beta = float(beta)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon, mu, logvar = self.model(x)
        recon_loss = self.criterion(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl
        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
