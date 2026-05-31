#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Imported recurrent autoencoder detector (manifest artifact inference).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from .base import BaseDetector


ARTIFACT_CONTRACT_VERSION = "2.0"


class ImportedArtifactContractError(ValueError):
    """Raised when an imported artifact violates the MetroPT v2 contract."""


def _json_hash(payload: object) -> str:
    raw = json.dumps(_as_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _feature_hash(feature_names: list[str]) -> str:
    raw = json.dumps(list(feature_names), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _as_jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    return value


def _scaler_state_payload(scaler: object) -> dict:
    payload = {
        "class": f"{type(scaler).__module__}.{type(scaler).__name__}",
    }
    for attr in ("mean_", "scale_", "var_", "n_features_in_", "n_samples_seen_"):
        if hasattr(scaler, attr):
            payload[attr] = _as_jsonable(getattr(scaler, attr))
    return payload


def _normalize_list(values: object) -> list:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        return [values]
    try:
        return list(values)  # type: ignore[arg-type]
    except TypeError:
        return [values]


def _normalize_int_list(values: object) -> list[int]:
    out: list[int] = []
    for value in _normalize_list(values):
        out.append(int(value))
    return sorted(out)


def _values_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return True
    if isinstance(left, (int, float, np.integer, np.floating)) or isinstance(right, (int, float, np.integer, np.floating)):
        try:
            return float(left) == float(right)
        except Exception:
            return False
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        try:
            return _normalize_int_list(left) == _normalize_int_list(right)
        except Exception:
            return list(_normalize_list(left)) == list(_normalize_list(right))
    return str(left) == str(right)


def _require_contract_value(
    *,
    label: str,
    expected: object,
    observed: object,
) -> None:
    if expected is None:
        return
    if not _values_equal(expected, observed):
        raise ImportedArtifactContractError(
            f"Imported artifact contract mismatch for {label}: "
            f"runtime={expected!r}, artifact={observed!r}."
        )


@dataclass
class ImportedRecurrentAutoencoderConfig:
    model_meta_path: str
    model_path: Optional[str] = None
    scaler_path: Optional[str] = None
    device: str = "cpu"
    use_scaler: bool = True
    batch_size: int = 256
    sequence_len: Optional[int] = None
    stride: int = 1
    score_stride: Optional[int] = None
    num_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = False
    expected_contract: Optional[dict[str, Any]] = None


class ImportedRecurrentAutoencoderDetector(BaseDetector):
    """Inference-only detector for per-cycle artifacts exported by NiaNetVAE."""

    name = "imported-recurrent-autoencoder"

    def __init__(
        self,
        model_meta_path: str,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        device: str = "cpu",
        use_scaler: bool = True,
        batch_size: int = 256,
        sequence_len: Optional[int] = None,
        stride: int = 1,
        score_stride: Optional[int] = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        expected_contract: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.extra_kwargs = dict(kwargs)
        if not use_scaler:
            raise ImportedArtifactContractError(
                "Imported recurrent autoencoder contract v2 requires the exported scaler; "
                "use_scaler=False is not supported."
            )
        self.cfg = ImportedRecurrentAutoencoderConfig(
            model_meta_path=model_meta_path,
            model_path=model_path,
            scaler_path=scaler_path,
            device=device,
            use_scaler=use_scaler,
            batch_size=batch_size,
            sequence_len=sequence_len,
            stride=stride,
            score_stride=score_stride,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            expected_contract=expected_contract,
        )
        self.meta = self._load_meta(self.cfg.model_meta_path)
        self._validate_contract_metadata()
        self.feature_contract = self.meta["feature_contract"]
        self.preprocessing_contract = self.meta["preprocessing_contract"]
        self.sequence_contract = self.meta["sequence_contract"]
        self.split_contract = self.meta["split_contract"]
        self.feature_names = [str(col) for col in self.feature_contract["rolling_feature_names"]]
        self.input_dim = int(self.feature_contract.get("n_features") or len(self.feature_names))
        self.seq_len = int(self.sequence_contract["seq_len"])
        self.score_stride = int(self.sequence_contract.get("score_stride") or self.sequence_contract["stride"])
        self.scaler_path: Optional[Path] = None
        self.scaler = self._load_scaler()
        self._validate_scaler_contract()
        self._validate_runtime_contract()
        print(
            "[INFO] Imported recurrent artifact contract v2 validated; "
            f"using exported scaler={self.scaler_path}"
        )
        self.model = self._build_model()
        self.model.eval()

    @staticmethod
    def _load_meta(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Imported model meta file not found: {p}")
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse imported model meta file: {p}") from exc

    def _validate_contract_metadata(self) -> None:
        schema_version = str(self.meta.get("schema_version"))
        contract_version = str(self.meta.get("contract_version"))
        if schema_version != ARTIFACT_CONTRACT_VERSION or contract_version != ARTIFACT_CONTRACT_VERSION:
            raise ImportedArtifactContractError(
                "Imported model_meta.json must use artifact contract v2 "
                f"(schema_version={ARTIFACT_CONTRACT_VERSION!r}, "
                f"contract_version={ARTIFACT_CONTRACT_VERSION!r}); "
                f"got schema_version={self.meta.get('schema_version')!r}, "
                f"contract_version={self.meta.get('contract_version')!r}."
            )

        required_objects = (
            "feature_contract",
            "preprocessing_contract",
            "sequence_contract",
            "split_contract",
        )
        for key in required_objects:
            if not isinstance(self.meta.get(key), dict):
                raise ImportedArtifactContractError(
                    f"Imported model_meta.json missing required v2 object: {key}."
                )

        feature_contract = self.meta["feature_contract"]
        rolling_feature_names = feature_contract.get("rolling_feature_names")
        if not isinstance(rolling_feature_names, list) or not rolling_feature_names:
            raise ImportedArtifactContractError(
                "Imported model_meta.json feature_contract must include non-empty rolling_feature_names."
            )
        expected_hash = feature_contract.get("feature_hash")
        observed_hash = _feature_hash([str(col) for col in rolling_feature_names])
        if expected_hash and str(expected_hash) != observed_hash:
            raise ImportedArtifactContractError(
                "Imported artifact feature_contract.feature_hash does not match rolling_feature_names."
            )

        preprocessing_contract = self.meta["preprocessing_contract"]
        if not preprocessing_contract.get("scaler_file") and not self.cfg.scaler_path:
            raise ImportedArtifactContractError(
                "Imported model_meta.json preprocessing_contract must include scaler_file."
            )
        if int(preprocessing_contract.get("scaler_feature_count", len(rolling_feature_names))) != len(rolling_feature_names):
            raise ImportedArtifactContractError(
                "Imported artifact scaler_feature_count does not match rolling_feature_names length."
            )

        sequence_contract = self.meta["sequence_contract"]
        if not sequence_contract.get("seq_len") or not sequence_contract.get("stride"):
            raise ImportedArtifactContractError(
                "Imported model_meta.json sequence_contract must include seq_len and stride."
            )
        if bool(sequence_contract.get("cross_gap_windows_allowed", True)):
            raise ImportedArtifactContractError(
                "Imported artifact must declare cross_gap_windows_allowed=False."
            )

    def _resolve_scaler_path(self) -> Path:
        if self.cfg.scaler_path:
            candidate = Path(self.cfg.scaler_path)
        else:
            scaler_file = self.meta["preprocessing_contract"].get("scaler_file") or self.meta.get("scaler_file")
            candidate = Path(self.cfg.model_meta_path).parent / str(scaler_file)
        if not candidate.exists():
            raise FileNotFoundError(f"Imported v2 scaler artifact not found: {candidate}")
        return candidate

    def _load_scaler(self):
        scaler_path = self._resolve_scaler_path()
        self.scaler_path = scaler_path
        try:
            return joblib.load(scaler_path)
        except Exception as exc:
            raise ImportedArtifactContractError(
                f"Failed to load imported scaler artifact: {scaler_path}"
            ) from exc

    def _validate_scaler_contract(self) -> None:
        if not hasattr(self.scaler, "transform"):
            raise ImportedArtifactContractError(
                "Imported scaler artifact must provide a transform() method."
            )
        scaler_features = int(getattr(self.scaler, "n_features_in_", self.input_dim))
        if scaler_features != self.input_dim:
            raise ImportedArtifactContractError(
                f"Imported scaler feature count mismatch: scaler={scaler_features}, "
                f"feature_contract={self.input_dim}."
            )
        expected_hash = self.preprocessing_contract.get("scaler_hash")
        if expected_hash:
            observed_hash = _json_hash(_scaler_state_payload(self.scaler))
            if str(expected_hash) != observed_hash:
                raise ImportedArtifactContractError(
                    "Imported scaler hash does not match preprocessing_contract.scaler_hash."
                )

    def _validate_runtime_contract(self) -> None:
        expected = self.cfg.expected_contract or {}
        if self.cfg.sequence_len is not None:
            _require_contract_value(
                label="sequence_len",
                expected=int(self.cfg.sequence_len),
                observed=self.sequence_contract.get("seq_len"),
            )
        _require_contract_value(
            label="stride",
            expected=int(self.cfg.stride),
            observed=self.sequence_contract.get("stride"),
        )
        if self.cfg.score_stride is not None:
            _require_contract_value(
                label="score_stride",
                expected=int(self.cfg.score_stride),
                observed=self.sequence_contract.get("score_stride") or self.sequence_contract.get("stride"),
            )

        _require_contract_value(
            label="rolling_window",
            expected=expected.get("rolling_window"),
            observed=self.feature_contract.get("rolling_window"),
        )
        _require_contract_value(
            label="seq_len",
            expected=expected.get("seq_len"),
            observed=self.sequence_contract.get("seq_len"),
        )
        _require_contract_value(
            label="score_stride",
            expected=expected.get("score_stride"),
            observed=self.sequence_contract.get("score_stride") or self.sequence_contract.get("stride"),
        )
        _require_contract_value(
            label="train_minutes",
            expected=expected.get("train_minutes"),
            observed=self.split_contract.get("train_minutes"),
        )
        _require_contract_value(
            label="post_train_minutes",
            expected=expected.get("post_train_minutes"),
            observed=self.split_contract.get("post_train_minutes"),
        )
        _require_contract_value(
            label="pre_maint_minutes",
            expected=expected.get("pre_maint_minutes"),
            observed=self.split_contract.get("pre_maint_minutes"),
        )
        _require_contract_value(
            label="train_phases",
            expected=expected.get("train_phases"),
            observed=self.split_contract.get("train_phases"),
        )
        _require_contract_value(
            label="test_phases",
            expected=expected.get("test_phases"),
            observed=self.split_contract.get("test_phases"),
        )
        _require_contract_value(
            label="phase_policy",
            expected=expected.get("phase_policy"),
            observed=self.sequence_contract.get("window_label_policy"),
        )
        _require_contract_value(
            label="regime",
            expected=expected.get("regime"),
            observed=self.split_contract.get("regime"),
        )
        _require_contract_value(
            label="cycle_id",
            expected=expected.get("cycle_id"),
            observed=self.split_contract.get("cycle_id"),
        )

    def _require_export_runtime(self):
        try:
            import torch  # noqa: F401
            from nianetvae.models.rnn_vae import RNNVAE  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "Imported recurrent autoencoder artifacts currently require the local "
                "NiaNetVAE package. Install with: `pip install -e ../NiaNetVAE`"
            ) from exc

    def _resolve_weights_path(self) -> Path:
        if self.cfg.model_path:
            candidate = Path(self.cfg.model_path)
        else:
            weights_file = self.meta.get("weights_file", "model.pt")
            candidate = Path(self.cfg.model_meta_path).parent / str(weights_file)
        if not candidate.exists():
            raise FileNotFoundError(f"Imported model weights not found: {candidate}")
        return candidate

    def _build_model(self):
        self._require_export_runtime()
        import torch
        from nianetvae.models.rnn_vae import RNNVAE

        model_class = str(self.meta.get("model_class", "nianetvae.models.rnn_vae.RNNVAE"))
        if model_class not in {"nianetvae.models.rnn_vae.RNNVAE", "RNNVAE"}:
            raise ValueError(
                f"Unsupported imported model_class={model_class!r}. "
                "Current metropt loader supports NiaNetVAE RNNVAE exports only."
            )

        solution = np.asarray(self.meta.get("solution"), dtype=float)
        if solution.shape[0] != int(RNNVAE.GENE_DIMENSION):
            raise ValueError(
                "Imported model_meta.json must contain a "
                f"{RNNVAE.GENE_DIMENSION}-gene solution vector."
            )

        model_config = {
            "data_params": {
                "n_features": self.input_dim,
                "seq_len": self.seq_len,
                "batch_size": int(self.cfg.batch_size),
            },
        }
        model = RNNVAE(solution, **model_config)
        if not getattr(model, "is_valid", False):
            raise ValueError("Resolved imported architecture is invalid; cannot run inference.")

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

    def fit(self, X_train: pd.DataFrame) -> "ImportedRecurrentAutoencoderDetector":
        if X_train is None or X_train.shape[0] < 2:
            raise ValueError("Need at least 2 samples to calibrate imported recurrent autoencoder.")
        self._validate_frame_features(X_train)
        return self

    def _validate_frame_features(self, X: pd.DataFrame) -> None:
        if X.shape[1] != self.input_dim:
            raise ImportedArtifactContractError(
                f"Feature mismatch: input has {X.shape[1]} features, model expects {self.input_dim}."
            )
        columns = [str(col) for col in X.columns]
        if columns != self.feature_names:
            first_mismatch = None
            for idx, (observed, expected) in enumerate(zip(columns, self.feature_names)):
                if observed != expected:
                    first_mismatch = f"at position {idx}: runtime={observed!r}, artifact={expected!r}"
                    break
            if first_mismatch is None and len(columns) != len(self.feature_names):
                first_mismatch = f"length runtime={len(columns)}, artifact={len(self.feature_names)}"
            raise ImportedArtifactContractError(
                "Runtime rolling feature columns do not exactly match imported artifact "
                f"rolling_feature_names ({first_mismatch})."
            )

    def score(self, X: pd.DataFrame) -> pd.Series:
        self._require_export_runtime()
        import torch
        from torch.utils.data import DataLoader

        from dataloaders.metropt_dataloader import MetroPTSequenceDataset

        if X is None or X.shape[0] < 2:
            raise ValueError("Need at least 2 samples for scoring.")
        self._validate_frame_features(X)

        X_vals = X.to_numpy(dtype=np.float32, copy=True)
        X_vals = self.scaler.transform(X_vals).astype(np.float32, copy=False)

        dataset = MetroPTSequenceDataset(data=X_vals, seq_len=self.seq_len, stride=self.score_stride)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.persistent_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=False,
        )
        print(
            f"[INFO] imported-recurrent-autoencoder score windows={len(dataset)} "
            f"(seq_len={self.seq_len}, stride={self.score_stride}), batch_size={self.cfg.batch_size}"
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


__all__ = [
    "ARTIFACT_CONTRACT_VERSION",
    "ImportedArtifactContractError",
    "ImportedRecurrentAutoencoderConfig",
    "ImportedRecurrentAutoencoderDetector",
]
