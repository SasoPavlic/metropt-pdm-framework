#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recurrent VAE models for sequence anomaly detection.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _build_rnn(
    rnn_type: str,
    input_dim: int,
    hidden_size: int,
    num_layers: int,
    bidirectional: bool,
) -> nn.Module:
    key = str(rnn_type).strip().upper()
    if key == "LSTM":
        return nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
    if key == "GRU":
        return nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
    if key in {"RNN", "RNN_TANH"}:
        return nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            bidirectional=bidirectional,
        )
    raise ValueError(f"Unsupported rnn_type={rnn_type!r} (use LSTM/GRU/RNN_TANH)")


class RecurrentVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        rnn_type: str,
        hidden_size: int,
        num_layers: int,
        latent_dim: int,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.rnn_type = str(rnn_type)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.latent_dim = int(latent_dim)
        self.bidirectional = bool(bidirectional)

        self.encoder_rnn = _build_rnn(
            rnn_type=self.rnn_type,
            input_dim=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        )

        enc_dir = 2 if self.bidirectional else 1
        enc_out_dim = self.hidden_size * enc_dir
        self.fc_mu = nn.Linear(enc_out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, self.latent_dim)

        self.fc_dec_init = nn.Linear(
            self.latent_dim, self.hidden_size * self.num_layers * enc_dir
        )
        self.decoder_rnn = _build_rnn(
            rnn_type=self.rnn_type,
            input_dim=self.latent_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        )
        self.fc_out = nn.Linear(self.hidden_size * enc_dir, self.input_dim)

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        if self.rnn_type.upper() == "LSTM":
            _, (h_n, _) = self.encoder_rnn(x)
        else:
            _, h_n = self.encoder_rnn(x)

        if self.bidirectional:
            # last layer forward/backward
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_last = h_n[-1]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent)
        bsz = z.size(0)
        enc_dir = 2 if self.bidirectional else 1
        h0 = self.fc_dec_init(z)
        h0 = h0.view(self.num_layers * enc_dir, bsz, self.hidden_size)
        if self.rnn_type.upper() == "LSTM":
            c0 = torch.zeros_like(h0)
            init = (h0, c0)
        else:
            init = h0

        z_seq = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder_rnn(z_seq, init)
        recon = self.fc_out(out)
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        recon = self._decode(z)
        return recon, mu, logvar


def build_recurrent_vae(
    input_dim: int,
    seq_len: int,
    rnn_type: str,
    hidden_size: int,
    num_layers: int,
    latent_dim: int,
    bidirectional: bool = False,
) -> RecurrentVAE:
    return RecurrentVAE(
        input_dim=input_dim,
        seq_len=seq_len,
        rnn_type=rnn_type,
        hidden_size=hidden_size,
        num_layers=num_layers,
        latent_dim=latent_dim,
        bidirectional=bidirectional,
    )
