import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import main as metropt_main
import detectors.imported_recurrent_autoencoder_detector as imported_detector_module
from detectors import (
    RecurrentSAEDetector,
    RecurrentVAEDetector,
    get_detector,
)
from detectors.imported_recurrent_autoencoder_detector import (
    ImportedArtifactContractError,
    ImportedRecurrentAutoencoderDetector,
)
from detectors.postprocess import train_and_score


def _synthetic_frame(rows: int = 16, cols: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    idx = pd.date_range("2020-01-01", periods=rows, freq="1min")
    return pd.DataFrame(
        rng.normal(size=(rows, cols)),
        index=idx,
        columns=[f"f{i}" for i in range(cols)],
    )


def _smoke_kwargs() -> dict:
    return {
        "sequence_len": 5,
        "stride": 1,
        "score_stride": 1,
        "hidden_size": 8,
        "num_layers": 1,
        "latent_dim": 3,
        "epochs": 1,
        "batch_size": 4,
        "lr": 0.003,
        "weight_decay": 1e-5,
        "device": "cpu",
        "num_workers": 0,
        "persistent_workers": False,
        "pin_memory": False,
        "random_state": 123,
    }


def _write_v2_imported_artifact(tmp_path: Path, feature_names: list[str]):
    torch = pytest.importorskip("torch")
    rnn_module = pytest.importorskip("nianetvae.models.rnn_vae")
    dataloader_module = pytest.importorskip("nianetvae.dataloaders.metropt_dataloader")
    runtime_artifacts_module = pytest.importorskip("nianetvae.search.runtime_artifacts")
    RNNVAE = rnn_module.RNNVAE

    artifact_dir = tmp_path / "cycle_00"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(321)
    scaler = StandardScaler().fit(rng.normal(size=(16, len(feature_names))))
    scaler_path = artifact_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    solution = np.asarray([0.25, 0.8, 0.5, 0.75, 0.4, 0.2], dtype=float)
    model = RNNVAE(
        solution,
        data_params={
            "n_features": len(feature_names),
            "seq_len": 5,
            "batch_size": 4,
        },
    )
    assert model.is_valid
    model_path = artifact_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    meta = {
        "schema_version": "2.0",
        "contract_version": "2.0",
        "model_class": "nianetvae.models.rnn_vae.RNNVAE",
        "solution": solution.tolist(),
        "n_features": len(feature_names),
        "seq_len": 5,
        "stride": 1,
        "weights_file": "model.pt",
        "scaler_file": "scaler.joblib",
        "feature_contract": {
            "base_feature_names": ["sensor"],
            "rolling_feature_names": feature_names,
            "rolling_aggregations": ["mean"],
            "rolling_window": "60s",
            "feature_hash": dataloader_module.build_feature_hash(feature_names),
            "n_features": len(feature_names),
        },
        "preprocessing_contract": {
            "scaler_type": f"{type(scaler).__module__}.{type(scaler).__name__}",
            "scaler_file": "scaler.joblib",
            "scaler_feature_count": len(feature_names),
            "scaler_hash": runtime_artifacts_module._hash_json_payload(
                runtime_artifacts_module._scaler_state_payload(scaler)
            ),
        },
        "sequence_contract": {
            "seq_len": 5,
            "stride": 1,
            "score_stride": 1,
            "window_label_policy": "end_anchor_phase",
            "cross_gap_windows_allowed": False,
        },
        "split_contract": {
            "regime": "per_maint",
            "cycle_id": 0,
            "train_minutes": 1440,
            "post_train_minutes": 1440,
            "pre_maint_minutes": 120,
            "train_phases": [0, 1],
            "test_phases": [0, 1],
            "train_segments": [],
            "test_segments": [],
        },
    }
    meta_path = artifact_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta_path, model_path, scaler_path, scaler


def _expected_contract() -> dict:
    return {
        "rolling_window": "60s",
        "seq_len": 5,
        "score_stride": 1,
        "train_minutes": 1440,
        "post_train_minutes": 1440,
        "pre_maint_minutes": 120,
        "train_phases": [0, 1],
        "test_phases": [0, 1],
        "phase_policy": "end_anchor_phase",
        "regime": "per_maint",
        "cycle_id": 0,
    }


def test_factory_registers_clean_recurrent_detector_names():
    assert isinstance(get_detector("iforest"), object)
    assert isinstance(get_detector("recurrent-vae"), RecurrentVAEDetector)
    assert isinstance(get_detector("recurrent-sae"), RecurrentSAEDetector)


@pytest.mark.parametrize("legacy_name", ["autoencoder", "ae", "nianetvae", "nianetvae_pretrained"])
def test_factory_rejects_removed_legacy_detector_names(legacy_name):
    with pytest.raises(ValueError, match="Unsupported detector_type"):
        get_detector(legacy_name)


@pytest.mark.parametrize("detector_type", ["recurrent-vae", "recurrent-sae"])
def test_recurrent_autoencoder_smoke_fit_and_score(detector_type):
    pytest.importorskip("torch")
    X = _synthetic_frame()
    detector = get_detector(detector_type, **_smoke_kwargs())

    detector.fit(X.iloc[:10])
    scores = detector.score(X)

    assert scores.name == "anom_score"
    assert scores.index.equals(X.index)
    assert scores.iloc[:4].isna().all()
    assert np.isfinite(scores.iloc[4:].to_numpy()).all()


def test_imported_per_maint_mode_uses_flag_not_detector_type(monkeypatch, tmp_path):
    manifest_path = tmp_path / "cycle_manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "2.0", "contract_version": "2.0", "cycles": {}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(metropt_main, "PER_MAINT_USE_IMPORTED_MODELS", True)
    monkeypatch.setattr(metropt_main, "PER_MAINT_MODEL_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setattr(metropt_main, "DETECTOR_TYPE", "ignored-in-imported-mode")

    metropt_main._validate_runtime_configuration("per_maint")

    assert (
        metropt_main._effective_detector_type("per_maint")
        == metropt_main.IMPORTED_RECURRENT_AUTOENCODER_TYPE
    )


@pytest.mark.parametrize("detector_type", ["recurrent-vae", "recurrent-sae"])
def test_recurrent_detectors_are_single_mode_only_for_local_training(monkeypatch, detector_type):
    monkeypatch.setattr(metropt_main, "PER_MAINT_USE_IMPORTED_MODELS", False)
    monkeypatch.setattr(metropt_main, "DETECTOR_TYPE", detector_type)

    with pytest.raises(ValueError, match="local per-maint training"):
        metropt_main._validate_runtime_configuration("per_maint")


def test_imported_detector_rejects_legacy_schema(tmp_path: Path):
    meta_path = tmp_path / "model_meta.json"
    meta_path.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")

    with pytest.raises(ImportedArtifactContractError, match="contract v2"):
        ImportedRecurrentAutoencoderDetector(model_meta_path=str(meta_path))


def test_imported_detector_rejects_missing_scaler(tmp_path: Path):
    feature_names = ["f0", "f1", "f2"]
    meta_path = tmp_path / "model_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "contract_version": "2.0",
                "feature_contract": {
                    "rolling_feature_names": feature_names,
                    "feature_hash": imported_detector_module._feature_hash(feature_names),
                    "n_features": 3,
                },
                "preprocessing_contract": {
                    "scaler_file": "scaler.joblib",
                    "scaler_feature_count": 3,
                },
                "sequence_contract": {
                    "seq_len": 5,
                    "stride": 1,
                    "cross_gap_windows_allowed": False,
                },
                "split_contract": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="scaler"):
        ImportedRecurrentAutoencoderDetector(model_meta_path=str(meta_path))


def test_imported_detector_uses_exported_scaler_and_validates_feature_order(tmp_path: Path):
    feature_names = [f"f{i}" for i in range(6)]
    meta_path, model_path, scaler_path, exported_scaler = _write_v2_imported_artifact(
        tmp_path,
        feature_names,
    )
    X = _synthetic_frame(rows=12, cols=6)
    X.columns = feature_names

    detector = ImportedRecurrentAutoencoderDetector(
        model_meta_path=str(meta_path),
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        device="cpu",
        batch_size=4,
        sequence_len=5,
        stride=1,
        score_stride=1,
        expected_contract=_expected_contract(),
    )
    detector.fit(X.iloc[:8])
    assert np.allclose(detector.scaler.mean_, exported_scaler.mean_)

    scores = detector.score(X)
    assert scores.index.equals(X.index)
    assert scores.iloc[:4].isna().all()
    assert np.isfinite(scores.iloc[4:].to_numpy()).all()

    reordered = X[list(reversed(feature_names))]
    with pytest.raises(ImportedArtifactContractError, match="feature columns"):
        detector.score(reordered)


def test_train_and_score_can_calibrate_on_contiguous_train_segments_only():
    class RecordingDetector:
        name = "recording"

        def __init__(self):
            self.score_lengths = []

        def fit(self, X_train):
            self.train_length = len(X_train)
            return self

        def score(self, X):
            self.score_lengths.append(len(X))
            values = np.arange(len(X), dtype=float)
            return pd.Series(values, index=X.index, name="anom_score")

    X = _synthetic_frame(rows=8, cols=2)
    train_a = X.iloc[:3]
    train_b = X.iloc[5:]
    X_train = pd.concat([train_a, train_b])
    detector = RecordingDetector()

    _pred, _info = train_and_score(
        detector,
        X_train,
        X.iloc[3:5],
        train_score_segments=[train_a, train_b],
    )

    assert detector.train_length == 6
    assert detector.score_lengths == [2, 3, 3]
