#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-series anomaly explorer (native timeline plot, MetroPT-3).

- Loads CSV/TXT specific to this project (no MAT/Parquet to keep it simple).
- Rolling stats over all numeric features.
- Train on the first N minutes (chronological), score all points.
- Label by a robust (Q3 + 3*IQR) threshold on model scores.
- Plot ORIGINAL timeline with point colors (normal/anomalous) + shaded/marked failure windows with labels.
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_utils import (
    build_operation_phase,
    build_rolling_features,
    load_csv,
    parse_maintenance_windows,
    select_numeric_features,
)
from metrics_alarm_islands import (
    build_alarm_island_report,
    print_alarm_island_summary,
    save_alarm_island_report,
)
from metrics_event import evaluate_maintenance_prediction
from metrics_point import confusion_and_scores, evaluate_risk_thresholds
from detectors import get_detector
from detectors.imported_recurrent_autoencoder_detector import ARTIFACT_CONTRACT_VERSION
from detectors.recurrent_autoencoder_detector import (
    LOCAL_RECURRENT_TYPES,
    RECURRENT_SAE_TYPE,
    RECURRENT_VAE_TYPE,
    build_recurrent_detector_kwargs,
    is_local_recurrent_type,
    is_recurrent_sae_type,
    is_recurrent_vae_type,
)
from detectors.postprocess import train_and_score
from detectors.utils import time_based_train_mask
from plotting import plot_raw_timeline, plot_lead_time_distribution, plot_pr_vs_lead_time
from logging_utils import log_to_file


IMPORTED_RECURRENT_AUTOENCODER_TYPE = "imported-recurrent-autoencoder"


def parse_risk_grid_spec(spec: str) -> List[float]:
    parts = (spec or "").split(":")
    if len(parts) != 3:
        raise ValueError("Risk grid must be provided as 'start:stop:step'.")
    start, stop, step = map(float, parts)
    if step <= 0:
        raise ValueError("Risk grid step must be positive.")
    if stop < start:
        raise ValueError("Risk grid stop must be greater than or equal to start.")
    values: List[float] = []
    val = start
    while val <= stop + 1e-9:
        values.append(round(val, 6))
        val += step
    return values


def build_risk_threshold_grid(
    grid_spec: str,
    extra_thresholds: Optional[List[float]] = None,
) -> List[float]:
    """Build the maintenance-risk theta grid from coarse range + explicit probes."""
    thresholds = parse_risk_grid_spec(grid_spec)
    for raw_value in extra_thresholds or []:
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError("Risk grid thresholds must be finite.")
        if value < 0.0 or value > 1.0:
            raise ValueError("Risk grid thresholds must be within [0, 1].")
        thresholds.append(value)
    return sorted({round(float(value), 6) for value in thresholds})


def _set_global_seed(seed: int) -> None:
    """Best-effort deterministic seeding for fair/reproducible comparisons."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass

    try:
        from lightning import seed_everything

        seed_everything(seed, workers=True)
    except Exception:
        pass






def _normalize_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    if key in {"per_maint", "per-maint", "permaint"}:
        return "per_maint"
    if key == "single":
        return "single"
    return key or "unknown"


def _detector_artifact_group(detector: str) -> str:
    key = str(detector).strip().lower()
    if key in {"iforest", "isolation_forest", "isolationforest"}:
        return "iforest"
    if is_recurrent_vae_type(key):
        return RECURRENT_VAE_TYPE
    if is_recurrent_sae_type(key):
        return RECURRENT_SAE_TYPE
    if key == IMPORTED_RECURRENT_AUTOENCODER_TYPE:
        return IMPORTED_RECURRENT_AUTOENCODER_TYPE
    return key or "unknown"


def _detector_display_label(detector: str) -> str:
    key = str(detector).strip().lower()
    if is_recurrent_vae_type(key):
        return "RecurrentVAE"
    if is_recurrent_sae_type(key):
        return "RecurrentSAE"
    if key == IMPORTED_RECURRENT_AUTOENCODER_TYPE:
        return "ImportedRecurrentAE"
    return "IsolationForest"


def _artifact_subdir(mode: str, detector: str, kind: str) -> Path:
    return Path(ARTIFACTS_ROOT) / _detector_artifact_group(detector) / _resolve_artifact_mode(mode) / kind


def _ensure_artifact_tree(mode: str, detector: str) -> None:
    # Create the full per-run artifact directory structure up-front.
    root = Path(ARTIFACTS_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    for kind in ("logs", "plots", "predictions"):
        _artifact_subdir(mode, detector, kind).mkdir(parents=True, exist_ok=True)


def _artifact_file_path(
    path: Optional[str],
    mode: str,
    detector: str,
    kind: str,
    default_suffix: str,
) -> Optional[str]:
    if not path:
        return None
    src = Path(path)
    filename = src.name
    if not Path(filename).suffix:
        filename = f"{filename}{default_suffix}"
    return str(_artifact_subdir(mode, detector, kind) / filename)


def _output_path_with_detector(path: Optional[str], mode: str, detector: str) -> Optional[str]:
    return _artifact_file_path(
        path=path,
        mode=mode,
        detector=detector,
        kind="plots",
        default_suffix=".png",
    )


def _pred_output_path(path: Optional[str], mode: str, detector: str) -> Optional[str]:
    return _artifact_file_path(
        path=path,
        mode=mode,
        detector=detector,
        kind="predictions",
        default_suffix=".csv",
    )


def _map_manifest_workflow_to_artifact_mode(workflow_mode: Optional[str]) -> Optional[str]:
    if not workflow_mode:
        return None
    key = str(workflow_mode).strip().lower()
    if key == "per_maint_warmstart_search":
        return "per_maint_warmstart_search"
    if key == "per_maint_finetune_search":
        return "per_maint_finetune_search"
    if key == "per_maint_baseline_search":
        return "per_maint_baseline_search"
    if key in {"per_maint_warmstart", "warmstart_search"} or "warmstart" in key:
        return "per_maint_warmstart"
    if key in {"per_maint_finetune", "finetune_search"} or "finetune" in key:
        return "per_maint_finetune"
    if key in {"per_maint_baseline", "baseline_search"} or "baseline" in key:
        return "per_maint_baseline"
    return None


@lru_cache(maxsize=4)
def _resolve_artifact_mode(mode: str) -> str:
    mode_norm = _normalize_mode(mode)
    if mode_norm != "per_maint" or not PER_MAINT_USE_IMPORTED_MODELS:
        return mode_norm

    manifest_path = str(PER_MAINT_MODEL_MANIFEST_PATH or "").strip()
    if not manifest_path:
        return mode_norm
    try:
        payload = json.loads(Path(manifest_path).expanduser().resolve().read_text(encoding="utf-8"))
    except Exception:
        return mode_norm

    manifest_workflow = _map_manifest_workflow_to_artifact_mode(payload.get("workflow_mode"))
    if manifest_workflow:
        return manifest_workflow

    cycles = payload.get("cycles")
    if isinstance(cycles, dict):
        for entry in cycles.values():
            if not isinstance(entry, dict):
                continue
            cycle_mode = _map_manifest_workflow_to_artifact_mode(entry.get("experiment_mode"))
            if cycle_mode:
                return cycle_mode
    return mode_norm


def _effective_detector_type(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "per_maint" and PER_MAINT_USE_IMPORTED_MODELS:
        return IMPORTED_RECURRENT_AUTOENCODER_TYPE
    return DETECTOR_TYPE


def _config_error(message: str, hints: Optional[List[str]] = None) -> None:
    print(f"[CONFIG-ERROR] {message}")
    for hint in hints or []:
        print(f"[CONFIG-ERROR] Hint: {hint}")
    raise ValueError(message)


def _validate_runtime_configuration(mode: str) -> None:
    mode_norm = _normalize_mode(mode)
    detector = str(DETECTOR_TYPE).strip().lower()
    imported = bool(PER_MAINT_USE_IMPORTED_MODELS)

    if int(PRE_MAINTENANCE_MINUTES) != int(THRESHOLD_POLICY_FIXED_LEAD_MINUTES):
        _config_error(
            "PRE_MAINTENANCE_MINUTES must equal THRESHOLD_POLICY_FIXED_LEAD_MINUTES "
            "for locked S1-T4 evaluation policy.",
            [
                f"Set PRE_MAINTENANCE_MINUTES={THRESHOLD_POLICY_FIXED_LEAD_MINUTES}.",
                "Or deliberately update both values together if policy changes.",
            ],
        )

    if mode_norm not in {"single", "per_maint"}:
        _config_error(
            f"Unsupported EXPERIMENT_MODE={mode!r}.",
            ["Use EXPERIMENT_MODE='single' or EXPERIMENT_MODE='per_maint'."],
        )

    local_single_detectors = {"iforest", "isolation_forest", "isolationforest"} | LOCAL_RECURRENT_TYPES

    if mode_norm == "single":
        if detector not in local_single_detectors:
            _config_error(
                f"Unsupported DETECTOR_TYPE={DETECTOR_TYPE!r} for EXPERIMENT_MODE='single'.",
                [
                    "Use DETECTOR_TYPE='iforest', 'recurrent-vae', or 'recurrent-sae'.",
                ],
            )
        if imported:
            print("[CONFIG-WARN] PER_MAINT_USE_IMPORTED_MODELS=True is ignored in single mode.")
        return

    if imported:
        if not PER_MAINT_MODEL_MANIFEST_PATH:
            _config_error(
                "PER_MAINT_MODEL_MANIFEST_PATH is required when PER_MAINT_USE_IMPORTED_MODELS=True.",
                ["Set PER_MAINT_MODEL_MANIFEST_PATH to cycle_manifest.json exported by NiaNetVAE."],
            )
        manifest_path = Path(str(PER_MAINT_MODEL_MANIFEST_PATH)).expanduser()
        if not manifest_path.exists():
            _config_error(
                f"Manifest path does not exist: {manifest_path}",
                ["Verify path and that cycle artifacts + cycle_manifest.json were copied locally."],
            )
        _load_cycle_manifest(str(manifest_path))
        return

    local_per_maint_detectors = {"iforest", "isolation_forest", "isolationforest"}
    if detector not in local_per_maint_detectors:
        _config_error(
            f"Unsupported DETECTOR_TYPE={DETECTOR_TYPE!r} for local per-maint training.",
            [
                "Use DETECTOR_TYPE='iforest' with PER_MAINT_USE_IMPORTED_MODELS=False.",
                "Set PER_MAINT_USE_IMPORTED_MODELS=True to evaluate imported per-cycle recurrent artifacts.",
            ],
        )




def _log_pipeline_overview() -> None:
    steps = [
        "Load data",
        "Build rolling features",
        "Build maintenance context",
        "Train/score detector (mode-specific)",
        "Evaluate point-wise + event metrics",
        "Plot + save outputs",
    ]
    print("\nPIPELINE OVERVIEW")
    for i, step in enumerate(steps, start=1):
        print(f"  {i}. {step}")


def _log_step_start(name: str) -> float:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[STEP] {name} ... start {ts}")
    return time.perf_counter()


def _log_step_end(name: str, t0: float) -> None:
    dt = time.perf_counter() - t0
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[STEP] {name} ... done in {dt:.2f}s at {ts}")

def _print_section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(title)
    print(bar)


# -----------------------------
# Configuration constants
# -----------------------------
# --- Data loading / preprocessing ---
# Timestamp column name in the input CSV; set to None to auto-detect.
INPUT_TIMESTAMP_COL: Optional[str] = None
# Whether to drop unnamed index columns commonly created by pandas when saving CSVs.
DROP_UNNAMED_INDEX: bool = True
# Unified output root:
# artifacts/<detector-group>/<regime>/{logs,plots,predictions}
ARTIFACTS_ROOT: str = "artifacts"
# Input/outputs
INPUT_PATH: str = "datasets/MetroPT3.csv"
SAVE_FIG_PATH: Optional[str] = "metropt3_raw.png"
SAVE_LEAD_TIME_DIST_PATH: Optional[str] = "lead_time_distribution.png"
SAVE_PR_LEADTIME_PATH: Optional[str] = "pr_vs_lead_time.png"
SAVE_PRED_CSV_PATH: Optional[str] = "metropt3_predictions.csv"
SAVE_FEATURES_CSV_PATH: Optional[str] = "datasets/metropt3_features.csv"

# Experiment mode:
# - "single": one global model trained once on an early slice.
# - "per_maint": per-maintenance models trained on fixed post-maintenance
#   training days for each cycle (plus an initial pre-W1 model).
EXPERIMENT_MODE: str = "per_maint"

# Pretrained per-maint model handoff (manifest-driven)
PER_MAINT_USE_IMPORTED_MODELS: bool = True
PER_MAINT_MODEL_MANIFEST_PATH: Optional[str] = r'C:\Users\sasop\CodexProjects\nianet\NiaNetVAE\logs\per_maint_vae_finetune\MetroPT\cycle_manifest.json'
PER_MAINT_MODEL_STRICT: bool = True

# Detector backend for local training ("iforest", "recurrent-vae", "recurrent-sae")
DETECTOR_TYPE: str = "recurrent-vae"
GLOBAL_SEED: int = 42

# Recurrent SAE/VAE settings (used when DETECTOR_TYPE is recurrent-vae/recurrent-sae)
REC_SEQUENCE_LEN: int = 200
REC_STRIDE: int = 1
REC_SCORE_STRIDE: int = 1
REC_HIDDEN_SIZE: int = 64
REC_NUM_LAYERS: int = 2
REC_LATENT_DIM: int = 32
REC_EPOCHS: int = 30
REC_BATCH_SIZE: int = 64
REC_NUM_WORKERS: int = 1
REC_PIN_MEMORY: bool = True
REC_PERSISTENT_WORKERS: bool = True
REC_LR: float = 0.003
REC_WEIGHT_DECAY: float = 1e-5
REC_DEVICE: str = "cuda"
REC_VAE_KL_BETA: float = 0.001
REC_SAE_SPARSITY_BETA: float = 0.05
REC_SAE_SPARSITY_RHO: float = 0.05

# Detector hyperparameters
RECURRENT_DETECTOR_KWARGS = build_recurrent_detector_kwargs(
    sequence_len=REC_SEQUENCE_LEN,
    stride=REC_STRIDE,
    score_stride=REC_SCORE_STRIDE,
    hidden_size=REC_HIDDEN_SIZE,
    num_layers=REC_NUM_LAYERS,
    latent_dim=REC_LATENT_DIM,
    epochs=REC_EPOCHS,
    batch_size=REC_BATCH_SIZE,
    lr=REC_LR,
    weight_decay=REC_WEIGHT_DECAY,
    device=REC_DEVICE,
    num_workers=REC_NUM_WORKERS,
    persistent_workers=REC_PERSISTENT_WORKERS,
    pin_memory=REC_PIN_MEMORY,
    random_state=GLOBAL_SEED,
)
DETECTOR_KWARGS = {"random_state": GLOBAL_SEED}
if is_recurrent_vae_type(DETECTOR_TYPE):
    DETECTOR_KWARGS = {
        **RECURRENT_DETECTOR_KWARGS,
        "kl_beta": REC_VAE_KL_BETA,
    }
elif is_recurrent_sae_type(DETECTOR_TYPE):
    DETECTOR_KWARGS = {
        **RECURRENT_DETECTOR_KWARGS,
        "sparsity_beta": REC_SAE_SPARSITY_BETA,
        "sparsity_rho": REC_SAE_SPARSITY_RHO,
    }



# Rolling window for feature aggregation (e.g., '600s' = 10 minutes).
ROLLING_WINDOW: str = "60s"
# Duration (minutes) of the initial training window (chronological from the start).
TRAIN_FRAC: float = 43200
# --- Modeling / scoring ---
# Rolling risk window (minutes).
RISK_WINDOW_MINUTES: int = 120
# Quantile for extreme-point exceedance when building maintenance risk.
RISK_EXCEEDANCE_QUANTILE: float = 0.95
# Risk evaluation grid specification (start:stop:step).
RISK_EVAL_GRID_SPEC: str = "0.10:0.90:0.05"
# Extra strict theta probes for diagnosing high alarm coverage.
RISK_EVAL_EXTRA_THRESHOLDS: List[float] = [0.925, 0.95, 0.975, 0.985, 0.99, 0.995]
# Locked deterministic theta-selection policy (S1-T4).
THRESHOLD_POLICY_FIXED_LEAD_MINUTES: int = 120
THRESHOLD_POLICY_TARGET_RECALL: float = 0.60
THRESHOLD_POLICY_TARGET_COVERAGE: float = 0.20
THRESHOLD_POLICY_SENSITIVITY_LEADS: List[int] = [30, 60, 90]
# Lead-time step size for event-level curves (minutes).
LEAD_STEP_MINUTES: int = 30
# Length of the post-maintenance training interval (in minutes) for the
# per-maintenance regime. For each maintenance window W_j, the next model
# (if any) is trained on the interval [end_j, end_j + POST_MAINT_TRAIN_MINUTES),
# clipped so it does not intrude into the next maintenance window. The same
# time mask is excluded from point-wise evaluation in the single-model regime
# for fair comparison.
POST_MAINT_TRAIN_MINUTES: int = 540

# --- Maintenance context / plotting ---
# Minutes before maintenance start considered pre-maintenance (operation_phase=1)
# and used as the early-warning horizon for event-level evaluation.
PRE_MAINTENANCE_MINUTES: int = 120
# Show labels for maintenance windows (IDs/severity) on the plot.
SHOW_WINDOW_LABELS: bool = True
# Font size for maintenance window labels.
WINDOW_LABEL_FONTSIZE: int = 9
# Label format; placeholders: {id}, {severity}, {dur_min}.
WINDOW_LABEL_FORMAT: str = "{id}"
# Use the built-in MetroPT-3 failure windows from Davari et al.
USE_DEFAULT_METROPT_WINDOWS: bool = True
# Preferential feature names for MetroPT-3; if present, they are prioritized.
LIKELY_METROPT_FEATURES = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Motor_current",
    "Oil_temperature",
    "Caudal_impulses",
]

# ===== MetroPT-3 failure windows (Davari et al., 2021) =====
# Table II intervals normalized to ISO (YYYY-MM-DD HH:MM:SS)
# Format: (start, end, id, severity)
DEFAULT_METROPT_WINDOWS: List[Tuple[str, str, str, str]] = [
    ("2020-04-12 11:50:00", "2020-04-12 23:30:00", "#1", "high"),
    ("2020-04-18 00:00:00", "2020-04-18 23:59:00", "#2", "high"),
    ("2020-04-19 00:00:00", "2020-04-19 01:30:00", "#3", "high"),
    ("2020-04-29 03:20:00", "2020-04-29 04:00:00", "#4", "high"),
    ("2020-04-29 22:00:00", "2020-04-29 22:20:00", "#5", "high"),
    ("2020-05-13 14:00:00", "2020-05-13 23:59:00", "#6", "high"),
    ("2020-05-18 05:00:00", "2020-05-18 05:30:00", "#7", "high"),
    ("2020-05-19 10:10:00", "2020-05-19 11:00:00", "#8", "high"),
    ("2020-05-19 22:10:00", "2020-05-19 23:59:00", "#9", "high"),
    ("2020-05-20 00:00:00", "2020-05-20 20:00:00", "#10", "high"),
    ("2020-05-23 09:50:00", "2020-05-23 10:10:00", "#11", "high"),
    ("2020-05-29 23:30:00", "2020-05-29 23:59:00", "#12", "high"),
    ("2020-05-30 00:00:00", "2020-05-30 06:00:00", "#13", "high"),
    ("2020-06-01 15:00:00", "2020-06-01 15:40:00", "#14", "high"),
    ("2020-06-03 10:00:00", "2020-06-03 11:00:00", "#15", "high"),
    ("2020-06-05 10:00:00", "2020-06-05 23:59:00", "#16", "high"),
    ("2020-06-06 00:00:00", "2020-06-06 23:59:00", "#17", "high"),
    ("2020-06-07 00:00:00", "2020-06-07 14:30:00", "#18", "high"),
    ("2020-07-08 17:30:00", "2020-07-08 19:00:00", "#19", "high"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00", "#20", "medium"),
    ("2020-07-17 04:30:00", "2020-07-17 05:30:00", "#21", "high"),
]


def _build_features_and_context() -> Tuple[pd.DataFrame, List[Tuple], pd.Series]:
    """Load raw data, engineer rolling features, and build maintenance context."""
    df_raw = load_csv(INPUT_PATH, INPUT_TIMESTAMP_COL, drop_unnamed=DROP_UNNAMED_INDEX)

    # Base numeric features (all numeric columns, ordered by preference if provided)
    base_feats = select_numeric_features(
        df_raw,
        prefer=LIKELY_METROPT_FEATURES,
    )
    if not base_feats:
        raise ValueError("No numeric features found in the input data.")
    df_base = df_raw[base_feats].copy()

    # Rolling features on all numeric signals
    X = build_rolling_features(df_base, rolling_window=ROLLING_WINDOW)
    if SAVE_FEATURES_CSV_PATH:
        feats_out = X.copy()
        feats_out.index.name = "timestamp"
        feats_out.to_csv(SAVE_FEATURES_CSV_PATH)

    # Maintenance windows + operation phase feature (0=normal,1=pre-maint,2=maintenance)
    maint_windows = parse_maintenance_windows(
        windows=None,
        maintenance_csv=None,
        use_default_windows=USE_DEFAULT_METROPT_WINDOWS,
        default_windows=DEFAULT_METROPT_WINDOWS,
    )
    operation_phase = build_operation_phase(
        index=X.index,
        windows=maint_windows,
        pre_minutes=PRE_MAINTENANCE_MINUTES,
    ).astype(np.int8)

    return X, maint_windows, operation_phase


def _build_post_maintenance_train_mask(
    index: pd.DatetimeIndex,
    maint_windows: List[Tuple],
    train_minutes: int,
) -> pd.Series:
    """
    Build a boolean mask marking post-maintenance training intervals.

    For each maintenance window W_j with end time e_j, we mark the interval
    [e_j, e_j + train_minutes) as training-only, clipped so that it does not
    intrude into the next maintenance window. This mask is used to exclude
    post-maintenance training time from point-wise evaluation in the
    single-model regime. The per-maintenance regime may further refine which
    parts of these intervals are actually used for training when gaps are very
    short.
    """
    mask = pd.Series(False, index=index)
    if not maint_windows or train_minutes <= 0:
        return mask

    mw_sorted = sorted(maint_windows, key=lambda w: pd.to_datetime(w[0]))
    starts = [pd.to_datetime(w[0]) for w in mw_sorted]
    ends = [pd.to_datetime(w[1]) for w in mw_sorted]

    for j, end_ts in enumerate(ends):
        t_start = pd.to_datetime(end_ts)
        t_stop = t_start + pd.Timedelta(minutes=float(train_minutes))
        if j < len(starts) - 1:
            next_start = pd.to_datetime(starts[j + 1])
            if t_stop > next_start:
                t_stop = next_start
        slice_mask = (index >= t_start) & (index < t_stop)
        mask |= slice_mask

    return mask


def _compute_pointwise_metrics(
    is_anomaly: pd.Series,
    operation_phase: pd.Series,
    eval_mask: Optional[pd.Series] = None,
) -> dict:
    """Compute confusion metrics with labels from operation_phase (1=pre-maint,0=normal)."""
    op_phase = operation_phase.reindex(is_anomaly.index)
    base_mask = op_phase.isin([0, 1]) & is_anomaly.notna()
    if eval_mask is not None:
        eval_mask = eval_mask.reindex(is_anomaly.index).fillna(False)
        mask = base_mask & eval_mask
    else:
        mask = base_mask

    if not mask.any():
        return {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "TN": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
        }

    y_true = (op_phase[mask] == 1).astype(int)
    y_pred = is_anomaly[mask].astype(int)
    return confusion_and_scores(y_true, y_pred)


def _compute_segmented_risk(
    risk_base: pd.Series,
    segment_masks: List[pd.Series],
) -> pd.Series:
    """Compute rolling risk within each segment, resetting at boundaries."""
    risk = pd.Series(0.0, index=risk_base.index, name="maintenance_risk")
    if risk_base.empty or not segment_masks:
        return risk

    for seg_mask in segment_masks:
        seg_mask = seg_mask.reindex(risk_base.index).fillna(False)
        if not seg_mask.any():
            continue
        seg_vals = risk_base[seg_mask].astype(float).fillna(0.0)
        seg_risk = (
            seg_vals.rolling(f"{RISK_WINDOW_MINUTES}min", min_periods=1)
            .mean()
            .astype(np.float32)
            .fillna(0.0)
        )
        risk.loc[seg_mask] = seg_risk

    return risk


def _ensure_bool_mask(mask: object, index: pd.Index) -> pd.Series:
    """Coerce a mask into a boolean Series aligned to index."""
    if isinstance(mask, pd.Series):
        out = mask.reindex(index).fillna(False)
    else:
        arr = np.asarray(mask, dtype=bool)
        if arr.shape[0] != len(index):
            raise ValueError("Mask length does not match index length.")
        out = pd.Series(arr, index=index)
    return out.astype(bool)


def _dataframe_segments_from_mask(X: pd.DataFrame, mask: pd.Series) -> List[pd.DataFrame]:
    """Return contiguous dataframe segments for True-runs in a full-index mask."""
    aligned_mask = _ensure_bool_mask(mask, X.index)
    flags = aligned_mask.to_numpy(dtype=bool)
    segments: List[pd.DataFrame] = []
    start: Optional[int] = None
    for pos, flag in enumerate(flags):
        if flag and start is None:
            start = pos
        elif not flag and start is not None:
            segments.append(X.iloc[start:pos])
            start = None
    if start is not None:
        segments.append(X.iloc[start:])
    return [segment for segment in segments if not segment.empty]


def _cycle_key(cycle_id: int) -> str:
    return f"{int(cycle_id):02d}"


def _load_cycle_manifest(path: str) -> tuple[dict, Path]:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"PER_MAINT_MODEL_MANIFEST_PATH not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if str(payload.get("schema_version")) != ARTIFACT_CONTRACT_VERSION:
        raise ValueError(
            "Imported cycle_manifest.json must use contract v2 "
            f"(schema_version={ARTIFACT_CONTRACT_VERSION!r}); "
            f"got schema_version={payload.get('schema_version')!r}."
        )
    if str(payload.get("contract_version")) != ARTIFACT_CONTRACT_VERSION:
        raise ValueError(
            "Imported cycle_manifest.json must declare "
            f"contract_version={ARTIFACT_CONTRACT_VERSION!r}; "
            f"got contract_version={payload.get('contract_version')!r}."
        )
    if "cycles" not in payload or not isinstance(payload["cycles"], dict):
        raise ValueError(f"Invalid cycle manifest format at {p}: missing 'cycles' object.")
    for key, entry in payload["cycles"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid cycle manifest format at {p}: cycle {key} entry is not an object.")
        status = str(entry.get("status", "")).strip().lower()
        if status == "trained":
            if str(entry.get("contract_version")) != ARTIFACT_CONTRACT_VERSION:
                raise ValueError(
                    f"Cycle {key} must use contract_version={ARTIFACT_CONTRACT_VERSION!r}; "
                    f"got {entry.get('contract_version')!r}."
                )
            missing = [field for field in ("model_path", "meta_path", "scaler_path") if not entry.get(field)]
            if missing:
                raise ValueError(
                    f"Cycle {key} status=trained is missing required v2 fields: {', '.join(missing)}."
                )
        elif status == "alias":
            if entry.get("alias_to") is None:
                raise ValueError(f"Cycle {key} status=alias is missing alias_to.")
        elif status != "missing":
            raise ValueError(f"Cycle {key} has unsupported status={status!r}.")
    return payload, p.parent


def _resolve_manifest_path(
    raw_path: Optional[str],
    manifest_dir: Path,
    cycle_id: Optional[int] = None,
) -> Optional[str]:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        return str((manifest_dir / candidate).resolve())

    if candidate.exists():
        return str(candidate.resolve())

    # Backward compatibility for old HPC-generated absolute Linux paths copied to another machine.
    fallbacks = []
    cycle_fragment = None
    for idx, part in enumerate(candidate.parts):
        if str(part).startswith("cycle_"):
            cycle_fragment = Path(*candidate.parts[idx:])
            break
    if cycle_fragment is not None:
        fallbacks.append((manifest_dir / cycle_fragment).resolve())

    if cycle_id is not None:
        fallbacks.append((manifest_dir / f"cycle_{_cycle_key(cycle_id)}" / candidate.name).resolve())

    fallbacks.append((manifest_dir / candidate.name).resolve())
    for fallback in fallbacks:
        if fallback.exists():
            return str(fallback)

    return str(candidate)


def _resolve_manifest_cycle(
    manifest: dict,
    manifest_dir: Path,
    cycle_id: int,
    strict: bool,
    visited: Optional[set] = None,
) -> Optional[dict]:
    visited = visited or set()
    key = _cycle_key(cycle_id)
    if key in visited:
        raise ValueError(f"Cycle alias loop detected while resolving cycle={key}.")
    visited.add(key)

    entry = manifest.get("cycles", {}).get(key)
    if entry is None:
        if strict:
            raise KeyError(f"Cycle {key} not present in manifest.")
        return None

    status = str(entry.get("status", "")).strip().lower()
    if status == "trained":
        model_path = _resolve_manifest_path(entry.get("model_path"), manifest_dir, cycle_id=int(cycle_id))
        meta_path = _resolve_manifest_path(entry.get("meta_path"), manifest_dir, cycle_id=int(cycle_id))
        scaler_path = _resolve_manifest_path(entry.get("scaler_path"), manifest_dir, cycle_id=int(cycle_id))
        if not model_path or not meta_path or not scaler_path:
            if strict:
                raise ValueError(f"Cycle {key} is trained but model_path/meta_path/scaler_path are missing.")
            return None
        model_exists = Path(model_path).exists()
        meta_exists = Path(meta_path).exists()
        scaler_exists = Path(scaler_path).exists()
        if strict and (not model_exists or not meta_exists or not scaler_exists):
            raise FileNotFoundError(
                f"Cycle {key} paths do not exist after resolution: "
                f"model_path={model_path} (exists={model_exists}), "
                f"meta_path={meta_path} (exists={meta_exists}), "
                f"scaler_path={scaler_path} (exists={scaler_exists})"
            )
        if not strict and (not model_exists or not meta_exists or not scaler_exists):
            return None
        resolved = dict(entry)
        resolved["resolved_cycle_id"] = int(entry.get("cycle_id", int(cycle_id)))
        resolved["model_path"] = model_path
        resolved["meta_path"] = meta_path
        resolved["scaler_path"] = scaler_path
        return resolved

    if status == "alias":
        alias_to = entry.get("alias_to")
        if alias_to is None:
            if strict:
                raise ValueError(f"Cycle {key} has status=alias but no alias_to.")
            return None
        return _resolve_manifest_cycle(
            manifest=manifest,
            manifest_dir=manifest_dir,
            cycle_id=int(alias_to),
            strict=strict,
            visited=visited,
        )

    if strict:
        raise ValueError(f"Cycle {key} unavailable in manifest (status={status!r}).")
    return None


def _build_imported_expected_contract(cycle_id: int) -> Dict[str, object]:
    """Runtime constants that must match NiaNetVAE v2 artifact metadata."""
    return {
        "rolling_window": ROLLING_WINDOW,
        "seq_len": REC_SEQUENCE_LEN,
        "score_stride": REC_SCORE_STRIDE,
        "train_minutes": TRAIN_FRAC,
        "post_train_minutes": POST_MAINT_TRAIN_MINUTES,
        "pre_maint_minutes": PRE_MAINTENANCE_MINUTES,
        "train_phases": [0, 1],
        "test_phases": [0, 1],
        "phase_policy": "end_anchor_phase",
        "regime": "per_maint",
        "cycle_id": int(cycle_id),
    }


def _evaluate_risk_series(
    maintenance_risk: pd.Series,
    maint_windows: List[Tuple],
    tag: str,
    eval_mask: Optional[pd.Series] = None,
) -> Tuple[Optional[float], Optional[pd.Series], List[dict], Optional[dict]]:
    """Grid-search risk thresholds and return best stats."""
    risk_thresholds: List[float] = []
    predicted_phase: Optional[pd.Series] = None
    best_risk_threshold: Optional[float] = None
    best_stats: Optional[dict] = None

    try:
        risk_thresholds = build_risk_threshold_grid(
            RISK_EVAL_GRID_SPEC,
            RISK_EVAL_EXTRA_THRESHOLDS,
        )
    except ValueError as exc:
        print(f"[WARN] Skipping maintenance_risk evaluation: {exc}")

    if not risk_thresholds:
        return best_risk_threshold, predicted_phase, [], best_stats

    risk_results = evaluate_risk_thresholds(
        risk=maintenance_risk,
        maintenance_windows=maint_windows,
        thresholds=risk_thresholds,
        early_warning_minutes=THRESHOLD_POLICY_FIXED_LEAD_MINUTES,
        eval_mask=eval_mask,
    )
    if not risk_results:
        return best_risk_threshold, predicted_phase, [], best_stats

    for row in risk_results:
        row["target_gap"] = (
            max(0.0, float(THRESHOLD_POLICY_TARGET_RECALL) - float(row.get("recall", 0.0)))
            + max(0.0, float(row.get("coverage", 0.0)) - float(THRESHOLD_POLICY_TARGET_COVERAGE))
        )

    feasible = [
        row
        for row in risk_results
        if float(row.get("recall", 0.0)) >= float(THRESHOLD_POLICY_TARGET_RECALL)
        and float(row.get("coverage", 1.0)) < float(THRESHOLD_POLICY_TARGET_COVERAGE)
    ]
    if feasible:
        best = max(
            feasible,
            key=lambda row: (
                float(row.get("f1", 0.0)),
                float(row.get("precision", 0.0)),
                float(row.get("recall", 0.0)),
                -float(row.get("threshold", 1e9)),
            ),
        )
        selection_mode = "feasible"
    else:
        best = min(
            risk_results,
            key=lambda row: (
                float(row.get("target_gap", 1e9)),
                -float(row.get("f1", 0.0)),
                -float(row.get("precision", 0.0)),
                -float(row.get("recall", 0.0)),
                float(row.get("threshold", 1e9)),
            ),
        )
        selection_mode = "fallback"

    best_stats = dict(best)
    best_stats["selection_mode"] = selection_mode
    best_stats["fixed_lead_minutes"] = int(THRESHOLD_POLICY_FIXED_LEAD_MINUTES)
    best_stats["target_recall"] = float(THRESHOLD_POLICY_TARGET_RECALL)
    best_stats["target_coverage"] = float(THRESHOLD_POLICY_TARGET_COVERAGE)
    best_risk_threshold = float(best["threshold"])
    predicted_phase = (maintenance_risk >= best_risk_threshold).astype(bool)
    return best_risk_threshold, predicted_phase, risk_results, best_stats


def _assign_predicted_phase(pred: pd.DataFrame, predicted_phase: Optional[pd.Series]) -> None:
    if predicted_phase is None:
        col = pd.Series(False, index=pred.index, dtype=bool)
    else:
        col = predicted_phase.reindex(pred.index).fillna(False).astype(bool)
    col = col.astype(np.int8)
    if "predicted_phase" in pred.columns:
        pred["predicted_phase"] = col
    elif "maintenance_risk" in pred.columns:
        insert_at = pred.columns.get_loc("maintenance_risk") + 1
        pred.insert(insert_at, "predicted_phase", col)
    else:
        pred["predicted_phase"] = col


def _assign_exceedance(pred: pd.DataFrame) -> None:
    """Add exceedance column based on risk_score and RISK_EXCEEDANCE_QUANTILE."""
    if "risk_score" not in pred.columns:
        return
    exceedance = (pred["risk_score"].astype(float) >= float(RISK_EXCEEDANCE_QUANTILE)).astype(np.int8)
    if "exceedance" in pred.columns:
        pred["exceedance"] = exceedance
    else:
        insert_at = pred.columns.get_loc("risk_score") + 1
        pred.insert(insert_at, "exceedance", exceedance)


def _print_pointwise_metrics(label: str, metrics: dict, threshold_info: Optional[str] = None) -> None:
    _print_section(f"POINT-WISE METRICS ({label})")
    if threshold_info:
        print(f"Threshold: {threshold_info}")
    print(
        f"TP={metrics['TP']}  FP={metrics['FP']}  FN={metrics['FN']}  TN={metrics['TN']}"
    )
    print(
        f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  "
        f"F1={metrics['f1']:.4f}  Accuracy={metrics['accuracy']:.4f}"
    )


def _print_event_level_metrics(
    label: str,
    best_stats: Optional[dict],
    risk_results: List[dict],
) -> None:
    _print_section(f"EVENT-LEVEL METRICS ({label})")
    print(
        f"Risk window={RISK_WINDOW_MINUTES} min  |  "
        f"Fixed lead={THRESHOLD_POLICY_FIXED_LEAD_MINUTES} min  |  "
        f"Exceedance quantile={RISK_EXCEEDANCE_QUANTILE:.2f}"
    )
    print(
        "maintenance_risk_theta policy: feasible-first "
        f"(recall>={THRESHOLD_POLICY_TARGET_RECALL:.2f}, "
        f"coverage<{THRESHOLD_POLICY_TARGET_COVERAGE:.2f}), "
        "fallback=lowest target_gap."
    )
    if best_stats:
        print(
            f"Best maintenance_risk_theta={best_stats['threshold']:.2f}  "
            f"mode={best_stats.get('selection_mode', 'n/a')}  "
            f"TP={best_stats['tp']}  FP={best_stats['fp']}  FN={best_stats['fn']}  "
            f"Precision={best_stats['precision']:.4f}  "
            f"Recall={best_stats['recall']:.4f}  F1={best_stats['f1']:.4f}  "
            f"Coverage={best_stats.get('coverage_percent', 0.0):.2f}%  "
            f"target_gap={best_stats.get('target_gap', 0.0):.4f}"
        )
    else:
        print("Best maintenance_risk_theta=N/A (no risk results)")


def _extract_policy_metric_block(event_results: dict) -> dict:
    event_scores = event_results.get("event_scores", {})
    coverage = event_results.get("coverage", {})
    far = event_results.get("far", {})
    faa = event_results.get("first_alarm_accuracy", {})
    nab = event_results.get("nab", {}).get("standard", {})
    return {
        "precision": float(event_scores.get("precision", 0.0)),
        "recall": float(event_scores.get("recall", 0.0)),
        "f1": float(event_scores.get("f1", 0.0)),
        "coverage_percent": float(coverage.get("alarm_coverage_percent", 0.0)),
        "far_per_week": far.get("far_per_week"),
        "faa": faa.get("first_alarm_accuracy"),
        "nab_standard": nab.get("nab_score_normalized"),
        "tp": int(event_scores.get("tp", 0)),
        "fp": int(event_scores.get("fp", 0)),
        "fn": int(event_scores.get("fn", 0)),
    }


THETA_SWEEP_SUMMARY_METRICS: List[str] = [
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "coverage",
    "far_per_week",
    "faa",
    "nab_standard",
    "target_gap",
]

THETA_SWEEP_HIGHER_IS_BETTER = {
    "tp",
    "precision",
    "recall",
    "f1",
    "faa",
    "nab_standard",
}


def _build_theta_sweep_summary(rows: List[dict]) -> pd.DataFrame:
    """Summarize all theta-sweep numeric metrics without changing raw sweep rows."""
    df = pd.DataFrame(rows)
    if df.empty or "maintenance_risk_theta" not in df.columns:
        return pd.DataFrame()

    theta = pd.to_numeric(df["maintenance_risk_theta"], errors="coerce")
    summary_rows = []
    for metric in THETA_SWEEP_SUMMARY_METRICS:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        metric_df = pd.DataFrame(
            {
                "maintenance_risk_theta": theta,
                "value": values,
            }
        ).dropna()
        if metric_df.empty:
            continue

        higher_is_better = metric in THETA_SWEEP_HIGHER_IS_BETTER
        best_idx = metric_df["value"].idxmax() if higher_is_better else metric_df["value"].idxmin()
        worst_idx = metric_df["value"].idxmin() if higher_is_better else metric_df["value"].idxmax()
        metric_values = metric_df["value"]

        summary_rows.append(
            {
                "metric": metric,
                "direction": "higher_is_better" if higher_is_better else "lower_is_better",
                "count": int(metric_values.count()),
                "mean": float(metric_values.mean()),
                "median": float(metric_values.median()),
                "std": float(metric_values.std(ddof=0)),
                "min": float(metric_values.min()),
                "max": float(metric_values.max()),
                "q25": float(metric_values.quantile(0.25)),
                "q75": float(metric_values.quantile(0.75)),
                "best_value": float(metric_df.loc[best_idx, "value"]),
                "best_theta": float(metric_df.loc[best_idx, "maintenance_risk_theta"]),
                "worst_value": float(metric_df.loc[worst_idx, "value"]),
                "worst_theta": float(metric_df.loc[worst_idx, "maintenance_risk_theta"]),
            }
        )

    return pd.DataFrame(summary_rows)


def _print_theta_sweep_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    by_metric = {str(row["metric"]): row for _, row in summary.iterrows()}

    def row(metric: str) -> Optional[pd.Series]:
        return by_metric.get(metric)

    _print_section("THETA-SWEEP SUMMARY")
    for metric, label, third_stat in [
        ("f1", "F1", "max"),
        ("recall", "Recall", "max"),
        ("coverage", "Coverage", "min"),
        ("far_per_week", "FAR/week", "min"),
        ("nab_standard", "NAB", "max"),
    ]:
        metric_row = row(metric)
        if metric_row is None:
            continue
        print(
            f"{label} mean={metric_row['mean']:.4f}  "
            f"median={metric_row['median']:.4f}  "
            f"{third_stat}={metric_row[third_stat]:.4f}  "
            f"best={metric_row['best_value']:.4f} at theta={metric_row['best_theta']:.2f}"
        )


def _save_theta_sweep_summary(rows: List[dict], *, mode: str, detector: str) -> pd.DataFrame:
    summary = _build_theta_sweep_summary(rows)
    if summary.empty:
        return summary

    out_path = _pred_output_path("theta_sweep_summary.csv", mode, detector)
    if not out_path:
        return summary

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"[INFO] Saved maintenance_risk_theta sweep summary: {out_path}")
    return summary


def _build_threshold_policy_report(
    predicted_phase: Optional[pd.Series],
    maint_windows: List[Tuple],
    eval_mask: pd.Series,
    best_stats: Optional[dict],
    primary_event_results: dict,
) -> Optional[dict]:
    if predicted_phase is None or best_stats is None:
        return None

    selected_maintenance_risk_theta = float(best_stats.get("threshold", 0.0))
    report = {
        "fixed_lead_minutes": int(THRESHOLD_POLICY_FIXED_LEAD_MINUTES),
        "selected_maintenance_risk_theta": selected_maintenance_risk_theta,
        "selection_mode": best_stats.get("selection_mode"),
        "target_recall": float(THRESHOLD_POLICY_TARGET_RECALL),
        "target_coverage": float(THRESHOLD_POLICY_TARGET_COVERAGE),
        "target_gap": float(best_stats.get("target_gap", 0.0)),
        "primary": _extract_policy_metric_block(primary_event_results),
        "sensitivity": [],
    }

    sensitivity_leads = sorted(
        {
            int(lead)
            for lead in THRESHOLD_POLICY_SENSITIVITY_LEADS
            if int(lead) > 0 and int(lead) != int(THRESHOLD_POLICY_FIXED_LEAD_MINUTES)
        }
    )
    for lead in sensitivity_leads:
        lead_results = evaluate_maintenance_prediction(
            predictions=predicted_phase,
            maintenance_windows=maint_windows,
            early_warning_minutes=int(lead),
            method_name=f"Sensitivity-{lead}",
            eval_mask=eval_mask,
            lead_step_minutes=max(1, min(int(LEAD_STEP_MINUTES), int(lead))),
        )
        row = _extract_policy_metric_block(lead_results)
        row["lead_time_min"] = int(lead)
        report["sensitivity"].append(row)

    return report


def _save_maintenance_risk_theta_sweep(
    maintenance_risk: pd.Series,
    maint_windows: List[Tuple],
    eval_mask: pd.Series,
    risk_results: List[dict],
    best_stats: Optional[dict],
    *,
    mode: str,
    detector: str,
) -> None:
    """Export the maintenance_risk threshold frontier used by the locked policy."""
    if not risk_results:
        return
    out_path = _pred_output_path("theta_sweep_maintenance_risk.csv", mode, detector)
    if not out_path:
        return

    selected_maintenance_risk_theta = None
    selected_mode = None
    if best_stats:
        selected_maintenance_risk_theta = float(
            best_stats.get("threshold", best_stats.get("maintenance_risk_theta", 0.0))
        )
        selected_mode = str(best_stats.get("selection_mode") or "")

    rows = []
    for row in risk_results:
        theta = float(row.get("threshold", row.get("maintenance_risk_theta", 0.0)))
        predicted = (maintenance_risk >= theta).astype(bool)
        event_results = evaluate_maintenance_prediction(
            predictions=predicted,
            maintenance_windows=maint_windows,
            early_warning_minutes=THRESHOLD_POLICY_FIXED_LEAD_MINUTES,
            method_name=f"maintenance_risk_theta={theta:.6f}",
            eval_mask=eval_mask,
            lead_step_minutes=LEAD_STEP_MINUTES,
        )
        block = _extract_policy_metric_block(event_results)
        feasible = (
            float(row.get("recall", 0.0)) >= float(THRESHOLD_POLICY_TARGET_RECALL)
            and float(row.get("coverage", 1.0)) < float(THRESHOLD_POLICY_TARGET_COVERAGE)
        )
        is_selected = (
            selected_maintenance_risk_theta is not None
            and abs(theta - selected_maintenance_risk_theta) <= 1e-12
        )
        selection_mode = selected_mode if is_selected else ("feasible" if feasible else "fallback_candidate")
        rows.append(
            {
                "maintenance_risk_theta": theta,
                "tp": int(row.get("tp", 0)),
                "fp": int(row.get("fp", 0)),
                "fn": int(row.get("fn", 0)),
                "precision": float(row.get("precision", 0.0)),
                "recall": float(row.get("recall", 0.0)),
                "f1": float(row.get("f1", 0.0)),
                "coverage": float(row.get("coverage", 0.0)),
                "far_per_week": block.get("far_per_week"),
                "faa": block.get("faa"),
                "nab_standard": block.get("nab_standard"),
                "target_gap": float(row.get("target_gap", 0.0)),
                "selection_mode": selection_mode,
            }
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[INFO] Saved maintenance_risk_theta sweep: {out_path}")
    summary = _save_theta_sweep_summary(rows, mode=mode, detector=detector)
    _print_theta_sweep_summary(summary)


def _save_alarm_island_analysis(
    predictions: pd.Series,
    maint_windows: List[Tuple],
    eval_mask: pd.Series,
    *,
    mode: str,
    detector: str,
    label: str,
) -> dict:
    islands, summary = build_alarm_island_report(
        predictions=predictions,
        maintenance_windows=maint_windows,
        early_warning_minutes=THRESHOLD_POLICY_FIXED_LEAD_MINUTES,
        eval_mask=eval_mask,
    )
    islands_path = _pred_output_path("alarm_islands.csv", mode, detector)
    summary_path = _pred_output_path("alarm_island_summary.csv", mode, detector)
    if islands_path and summary_path:
        save_alarm_island_report(islands, summary, islands_path, summary_path)
    print_alarm_island_summary(label, summary)
    return summary


def _print_event_extra_metrics(label: str, event_results: dict) -> None:
    _print_section(f"ADDITIONAL EVENT METRICS ({label})")
    ttd = event_results.get("ttd", {})
    faa = event_results.get("first_alarm_accuracy", {})
    far = event_results.get("far", {})
    cov = event_results.get("coverage", {})
    mtia = event_results.get("mtia", {})
    pr = event_results.get("pr_leadtime", {})
    nab = event_results.get("nab", {})

    if ttd.get("mean_ttd") is not None:
        print(
            f"TTD mean={ttd['mean_ttd']:.1f} min (std={ttd['std_ttd']:.1f}, "
            f"min={ttd['min_ttd']:.1f}, max={ttd['max_ttd']:.1f})  |  "
            f"higher is better (earlier), max={PRE_MAINTENANCE_MINUTES} min"
        )
    else:
        print("TTD mean=N/A (no detections)  |  higher is better (earlier)")
    print(
        f"TTD detected={ttd.get('detected_events', 0)} "
        f"missed={ttd.get('missed_events', 0)}  |  higher detected is better"
    )

    if faa.get("first_alarm_accuracy") is not None:
        print(
            f"FAA={faa['first_alarm_accuracy']:.3f} "
            f"({faa['first_alarm_in_window']}/{faa['tp_events']})  |  "
            f"higher is better (ideal=1.0)"
        )
    else:
        print("FAA=N/A  |  higher is better (ideal=1.0)")

    if cov:
        print(
            f"Coverage={cov['alarm_coverage_percent']:.2f}% "
            f"({cov['alarm_points']}/{cov['total_points']})  |  "
            f"lower is better (ideal~0%)"
        )

    if mtia.get("mtia_minutes") is not None:
        print(
            f"MTIA mean={mtia['mtia_minutes']:.1f} min "
            f"(std={mtia['std_minutes']:.1f}, intervals={mtia['num_intervals']})  |  "
            f"lower is better"
        )
    else:
        print("MTIA mean=N/A (no alarm intervals)  |  lower is better")

    if far.get("far_per_week") is not None:
        print(
            f"FAR per_day={far['far_per_day']:.3f}  "
            f"per_week={far['far_per_week']:.3f}  "
            f"FP intervals={far['fp_intervals']}  |  "
            f"lower is better (ideal=0)"
        )

    if pr:
        print("PR vs Lead Time (higher is better):")
        for i, lt in enumerate(pr.get("lead_times", [])):
            print(
                f"  {lt} min: P={pr['precision'][i]:.3f}, "
                f"R={pr['recall'][i]:.3f}, F1={pr['f1'][i]:.3f}"
            )

    if nab:
        std = nab.get("standard", {}).get("nab_score_normalized")
        low_fp = nab.get("low_fp", {}).get("nab_score_normalized")
        low_fn = nab.get("low_fn", {}).get("nab_score_normalized")
        if std is not None or low_fp is not None or low_fn is not None:
            print(
                "NAB: "
                f"standard={std:.1f} "
                f"low_fp={low_fp:.1f} "
                f"low_fn={low_fn:.1f}  |  "
                f"higher is better (ideal=100)"
            )


def _save_event_plots(event_results: dict, mode: str, detector: str) -> None:
    if not event_results:
        return
    dist = event_results.get("lead_time_distribution", {})
    pr = event_results.get("pr_leadtime", {})

    dist_path = _output_path_with_detector(SAVE_LEAD_TIME_DIST_PATH, mode, detector)
    pr_path = _output_path_with_detector(SAVE_PR_LEADTIME_PATH, mode, detector)

    if dist_path:
        plot_lead_time_distribution(
            dist,
            save_fig=dist_path,
            title=f"Lead Time Distribution ({detector}, {mode})",
        )
        print(f"[INFO] Saved plot: {Path(dist_path).resolve()}")

    if pr_path:
        plot_pr_vs_lead_time(
            pr,
            save_fig=pr_path,
            title=f"Precision-Recall vs Lead Time ({detector}, {mode})",
        )
        print(f"[INFO] Saved plot: {Path(pr_path).resolve()}")


def _run_single_model_experiment(
    X: pd.DataFrame,
    maint_windows: List[Tuple],
    operation_phase: pd.Series,
) -> Tuple[pd.DataFrame, dict, Optional[pd.Timestamp], Optional[float], Optional[pd.Series]]:
    """Baseline: single global model trained once and scored over the full timeline."""
    # Time-based training window, restricted to non-maintenance rows (phases 0/1)
    train_time_mask = time_based_train_mask(X.index, TRAIN_FRAC)
    op_phase = operation_phase.reindex(X.index).astype(np.int8)
    train_mask = train_time_mask & (op_phase != 2)
    if train_mask.sum() < 2:
        raise ValueError("Single-model training set has fewer than 2 samples after filtering.")
    detector = get_detector(DETECTOR_TYPE, **DETECTOR_KWARGS)
    pred_if, info = train_and_score(detector, X.loc[train_mask], X)

    pred = pred_if.copy()

    # Rolling maintenance risk from normalized risk scores (extreme-point exceedance)
    risk_base = pred["risk_score"].astype(float).fillna(0.0)
    risk_base = (risk_base >= float(RISK_EXCEEDANCE_QUANTILE)).astype(float)
    maintenance_risk = _compute_segmented_risk(
        risk_base, [pd.Series(True, index=pred.index)]
    )
    pred["maintenance_risk"] = maintenance_risk
    pred["operation_phase"] = op_phase

    # Training-only rows used in any regime:
    # - the initial chronological TRAIN_FRAC minutes
    # - fixed post-maintenance training intervals for per-maint models
    initial_train_time_mask = train_time_mask.reindex(pred.index).fillna(False)
    post_maint_train_mask = _build_post_maintenance_train_mask(
        pred.index, maint_windows, POST_MAINT_TRAIN_MINUTES
    )
    eval_exclude_mask = initial_train_time_mask | post_maint_train_mask

    # Event-level evaluation of maintenance_risk thresholds
    eval_mask = ~eval_exclude_mask
    best_risk_threshold, predicted_phase, risk_results, best_stats = _evaluate_risk_series(
        maintenance_risk,
        maint_windows,
        tag="RISK",
        eval_mask=eval_mask,
    )
    _assign_exceedance(pred)
    _assign_predicted_phase(pred, predicted_phase)

    # Exact training cutoff timestamp on the prediction index
    train_cutoff_ts: Optional[pd.Timestamp] = None
    try:
        if train_mask.any():
            train_cutoff_ts = train_mask[train_mask].index.max()
    except Exception:
        train_cutoff_ts = None

    # Point-wise metrics on normal + pre-maintenance, excluding training-only rows
    m_all = _compute_pointwise_metrics(pred["is_anomaly"], pred["operation_phase"], eval_mask=eval_mask)
    threshold_info = None
    if info.get("threshold") is not None:
        threshold_info = f"{info.get('label_rule')} | value={info['threshold']:.4f}"
    _print_pointwise_metrics("Single", m_all, threshold_info=threshold_info)

    _print_event_level_metrics("Single", best_stats, risk_results)

    event_results = evaluate_maintenance_prediction(
        predictions=pred["predicted_phase"],
        maintenance_windows=maint_windows,
        early_warning_minutes=THRESHOLD_POLICY_FIXED_LEAD_MINUTES,
        method_name="Single",
        eval_mask=eval_mask,
        lead_step_minutes=LEAD_STEP_MINUTES,
    )
    info["event_metrics"] = event_results
    info["threshold_policy"] = _build_threshold_policy_report(
        predicted_phase=predicted_phase,
        maint_windows=maint_windows,
        eval_mask=eval_mask,
        best_stats=best_stats,
        primary_event_results=event_results,
    )
    _save_maintenance_risk_theta_sweep(
        maintenance_risk=maintenance_risk,
        maint_windows=maint_windows,
        eval_mask=eval_mask,
        risk_results=risk_results,
        best_stats=best_stats,
        mode="single",
        detector=DETECTOR_TYPE,
    )
    info["alarm_island_summary"] = _save_alarm_island_analysis(
        predictions=pred["predicted_phase"],
        maint_windows=maint_windows,
        eval_mask=eval_mask,
        mode="single",
        detector=DETECTOR_TYPE,
        label="Single",
    )
    _print_event_extra_metrics("Single", event_results)
    _save_event_plots(event_results, mode="single", detector=DETECTOR_TYPE)

    # Console summary
    total_pts = int(pred["is_anomaly"].sum())
    pct_pts = float(100.0 * pred["is_anomaly"].mean())
    print(f"[INFO] Model rule: {info['label_rule']}, features={info['n_features']}")
    print(f"[INFO] Train size: {info['n_train']}/{info['n_total']}")
    print(f"[INFO] Point anomalies: {total_pts} ({pct_pts:.2f}%)")
    if info.get("threshold") is not None:
        print(f"[INFO] Train-based threshold: {info['threshold']:.4f}")

    return pred, info, train_cutoff_ts, best_risk_threshold, predicted_phase




def _run_per_maintenance_experiment(
    X: pd.DataFrame,
    maint_windows: List[Tuple],
    operation_phase: pd.Series,
) -> Tuple[pd.DataFrame, dict, Optional[float], Optional[pd.Series]]:
    """
    Adaptive regime by maintenance cycles.

    - Model 0 (pre-W1) is trained on the same earliest TRAIN_FRAC minutes of
      the timeline as the single-model regime (chronological order), but with
      maintenance rows (operation_phase==2) excluded from the training set.
    - After each maintenance window W_j, we reserve a fixed amount of time
      immediately after the window (POST_MAINT_TRAIN_MINUTES) as a
      training-only interval for the *next* model, provided the gap to the
      next maintenance is long enough.
    - That model is then used to score the remaining time until the next
      maintenance. If the gap is shorter than POST_MAINT_TRAIN_MINUTES, no
      new model is trained and the previous model is reused to score the gap.
    - All training-only intervals (initial TRAIN_FRAC slice and
      post-maintenance training intervals) are excluded from point-wise
      evaluation in all regimes.
    """
    if not maint_windows:
        raise ValueError("Per-maintenance mode requires maintenance windows.")

    # Ensure maintenance windows are sorted by start time
    mw_sorted = sorted(maint_windows, key=lambda w: pd.to_datetime(w[0]))
    starts = [pd.to_datetime(w[0]) for w in mw_sorted]
    ends = [pd.to_datetime(w[1]) for w in mw_sorted]

    index = X.index
    op_phase = operation_phase.reindex(index).astype(np.int8)
    if index.empty:
        raise ValueError("Empty index in per-maintenance experiment.")

    # Initial training window (shared with the single-model regime) - this is
    # the global baseline that every per-maintenance model sees.
    initial_train_time_mask = time_based_train_mask(index, TRAIN_FRAC)
    initial_train_order_mask = initial_train_time_mask.copy()
    # For regime 2 we only exclude maintenance rows from training.
    initial_train_mask = initial_train_time_mask & (op_phase != 2)

    # Post-maintenance training schedule (time-based) and global training-only mask
    post_maint_train_mask = _build_post_maintenance_train_mask(
        index, mw_sorted, POST_MAINT_TRAIN_MINUTES
    )
    global_train_for_eval = initial_train_order_mask | post_maint_train_mask

    pred = pd.DataFrame(index=index)
    risk_segments: List[pd.Series] = []

    period_infos: List[dict] = []

    data_end = index.max()
    first_start = starts[0]
    pre_w1_test_mask = (index < first_start) & (~initial_train_order_mask)
    current_train_mask = initial_train_mask

    use_imported_models = bool(PER_MAINT_USE_IMPORTED_MODELS)
    detector_type_runtime = DETECTOR_TYPE
    detector_kwargs_runtime: Dict[str, object] = dict(DETECTOR_KWARGS)
    manifest_payload: Optional[dict] = None
    manifest_dir: Optional[Path] = None
    if use_imported_models:
        if not PER_MAINT_MODEL_MANIFEST_PATH:
            raise ValueError(
                "PER_MAINT_USE_IMPORTED_MODELS=True requires PER_MAINT_MODEL_MANIFEST_PATH."
            )
        manifest_payload, manifest_dir = _load_cycle_manifest(PER_MAINT_MODEL_MANIFEST_PATH)
        detector_type_runtime = IMPORTED_RECURRENT_AUTOENCODER_TYPE
        detector_kwargs_runtime = {
            "device": REC_DEVICE,
            "use_scaler": True,
            "batch_size": REC_BATCH_SIZE,
            "sequence_len": REC_SEQUENCE_LEN,
            "stride": REC_STRIDE,
            "score_stride": REC_SCORE_STRIDE,
            "num_workers": REC_NUM_WORKERS,
            "persistent_workers": REC_PERSISTENT_WORKERS,
            "pin_memory": REC_PIN_MEMORY,
        }
        print(
            f"[INFO] Per-maint imported recurrent artifact mode enabled. "
            f"Manifest={Path(PER_MAINT_MODEL_MANIFEST_PATH).resolve()}"
        )

    def _is_trainable(train_mask: pd.Series, test_mask: pd.Series) -> bool:
        train_mask = _ensure_bool_mask(train_mask, index)
        test_mask = _ensure_bool_mask(test_mask, index)
        if not test_mask.any():
            return False
        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        return X_train.shape[0] >= 2 and X_test.shape[0] >= 2

    def _count_trainable_segments() -> int:
        count = 0
        current_mask = initial_train_mask
        # pre_W1
        if _is_trainable(current_mask, pre_w1_test_mask):
            count += 1
        # per maintenance cycle
        for j, (start_ts, end_ts) in enumerate(zip(starts, ends)):
            maint_mask = (index >= pd.to_datetime(start_ts)) & (index <= pd.to_datetime(end_ts))
            if _is_trainable(current_mask, maint_mask):
                count += 1
            gap_start = pd.to_datetime(end_ts)
            gap_end = pd.to_datetime(starts[j + 1]) if j < len(starts) - 1 else data_end
            if gap_end <= gap_start:
                continue
            gap_minutes = (gap_end - gap_start).total_seconds() / 60.0
            gap_mask = (index > gap_start) & (index < gap_end if j < len(starts) - 1 else index <= gap_end)
            if not gap_mask.any():
                continue
            if POST_MAINT_TRAIN_MINUTES <= 0 or gap_minutes <= POST_MAINT_TRAIN_MINUTES:
                if _is_trainable(current_mask, gap_mask):
                    count += 1
                continue
            train_end = gap_start + pd.Timedelta(minutes=float(POST_MAINT_TRAIN_MINUTES))
            if train_end > gap_end:
                train_end = gap_end
            local_post_mask = (
                (index > gap_start)
                & (index <= train_end)
                & (index <= gap_end)
                & (op_phase != 2)
            )
            train_mask_new = initial_train_mask | local_post_mask
            test_mask_new = (index > train_end) & (index < gap_end if j < len(starts) - 1 else index <= gap_end)
            if _is_trainable(train_mask_new, test_mask_new):
                count += 1
                current_mask = train_mask_new
        return count

    total_models = _count_trainable_segments()
    model_idx = 0

    print(
        f"[INFO] Per-maint mode: {len(mw_sorted) + 1} logical cycles "
        f"(pre-W1 plus one cycle per maintenance window)"
    )

    # Helper: fit the detector on X_train and score X_test; update per-point predictions.
    def _fit_and_score_segment(
        seg_label: str,
        train_mask: pd.Series,
        test_mask: pd.Series,
        requested_cycle_id: Optional[int] = None,
    ) -> Optional[dict]:
        nonlocal pred, risk_segments

        train_mask = _ensure_bool_mask(train_mask, index)
        test_mask = _ensure_bool_mask(test_mask, index)

        if not test_mask.any():
            return None

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        if X_train.shape[0] < 2 or X_test.shape[0] < 2:
            return None
        nonlocal model_idx
        model_idx += 1
        segment_action = "Calibrate+score" if use_imported_models else "Train+score"
        print(
            f"[INFO] {segment_action} segment {model_idx}/{total_models}: "
            f"{seg_label} (train_size={X_train.shape[0]}, test_size={X_test.shape[0]})"
        )
        detector_kwargs = dict(detector_kwargs_runtime)
        resolved_cycle_id = None
        if use_imported_models:
            if manifest_payload is None or manifest_dir is None:
                raise RuntimeError("Manifest resolver is not initialized.")
            if requested_cycle_id is None:
                raise ValueError(f"Imported mode requires requested_cycle_id for segment {seg_label}.")
            resolved_entry = _resolve_manifest_cycle(
                manifest=manifest_payload,
                manifest_dir=manifest_dir,
                cycle_id=int(requested_cycle_id),
                strict=bool(PER_MAINT_MODEL_STRICT),
            )
            if resolved_entry is None:
                print(
                    f"[WARN] No manifest artifact for segment={seg_label} requested_cycle={requested_cycle_id}; "
                    "skipping segment in non-strict mode."
                )
                return None
            detector_kwargs["model_meta_path"] = resolved_entry["meta_path"]
            detector_kwargs["model_path"] = resolved_entry["model_path"]
            detector_kwargs["scaler_path"] = resolved_entry["scaler_path"]
            resolved_cycle_id = int(resolved_entry.get("resolved_cycle_id", requested_cycle_id))
            detector_kwargs["expected_contract"] = _build_imported_expected_contract(resolved_cycle_id)
            print(
                f"[INFO] Imported model: segment={seg_label} requested_cycle={requested_cycle_id} "
                f"resolved_cycle={resolved_cycle_id}"
            )

        detector = get_detector(detector_type_runtime, **detector_kwargs)
        t_seg = None
        if detector_type_runtime == IMPORTED_RECURRENT_AUTOENCODER_TYPE or is_local_recurrent_type(detector_type_runtime):
            run_label = "Infer segment" if use_imported_models else "Run segment"
            t_seg = _log_step_start(
                f"{run_label} {seg_label} (train={X_train.shape[0]}, test={X_test.shape[0]})"
            )
        train_score_segments = _dataframe_segments_from_mask(X, train_mask) if use_imported_models else None
        slice_pred, info = train_and_score(
            detector,
            X_train,
            X_test,
            train_score_segments=train_score_segments,
        )
        if t_seg is not None:
            run_label = "Infer segment" if use_imported_models else "Run segment"
            _log_step_end(f"{run_label} {seg_label}", t_seg)

        pred.loc[test_mask, "anom_score"] = slice_pred["anom_score"]
        pred.loc[test_mask, "is_anomaly"] = slice_pred["is_anomaly"]
        if "risk_score" in slice_pred.columns:
            pred.loc[test_mask, "risk_score"] = slice_pred["risk_score"]

        risk_segments.append(test_mask)

        return {
            "segment": seg_label,
            "train_size": info["n_train"],
            "test_size": X_test.shape[0],
            "threshold": info.get("threshold"),
            "pct_anom": info.get("pct_anom"),
            "train_start": X_train.index.min(),
            "train_end": X_train.index.max(),
            "test_start": X_test.index.min(),
            "test_end": X_test.index.max(),
            "requested_cycle_id": requested_cycle_id,
            "resolved_cycle_id": resolved_cycle_id,
        }

    # Model 0: pre-W1 region, trained on the initial TRAIN_FRAC slice only.
    seg_info = _fit_and_score_segment(
        "pre_W1",
        current_train_mask,
        pre_w1_test_mask,
        requested_cycle_id=0,
    )
    if seg_info:
        period_infos.append(seg_info)

    # Iterate over maintenance windows; after each, either train a new model on
    # its post-maintenance training interval (plus the global baseline) or
    # reuse the previous model for short gaps.
    for j, (start_ts, end_ts) in enumerate(zip(starts, ends)):
        # Score the maintenance window itself with the current model (risk only).
        maint_mask = (index >= pd.to_datetime(start_ts)) & (index <= pd.to_datetime(end_ts))
        seg_info = _fit_and_score_segment(
            f"maint_{j + 1}",
            current_train_mask,
            maint_mask,
            requested_cycle_id=j,
        )
        if seg_info:
            period_infos.append(seg_info)

        gap_start = pd.to_datetime(end_ts)
        if j < len(starts) - 1:
            gap_end = pd.to_datetime(starts[j + 1])
        else:
            gap_end = data_end

        # Gap between end of this maintenance and the next maintenance (or end of data)
        if gap_end <= gap_start:
            print(f"[INFO] No gap after maint #{j + 1} (gap_end <= gap_start).")
            continue

        gap_minutes = (gap_end - gap_start).total_seconds() / 60.0
        gap_mask = (index > gap_start) & (index < gap_end if j < len(starts) - 1 else index <= gap_end)
        if not gap_mask.any():
            print(f"[INFO] No candidate points after maint #{j + 1} (gap has no data).")
            continue

        if POST_MAINT_TRAIN_MINUTES <= 0 or gap_minutes <= POST_MAINT_TRAIN_MINUTES:
            # Gap too short for a dedicated post-maintenance training block:
            # reuse the previous model (which already includes the global
            # baseline and its last post-maint block in its training mask).
            seg_label = f"gap_after_maint_{j + 1}"
            seg_info = _fit_and_score_segment(
                seg_label,
                current_train_mask,
                gap_mask,
                requested_cycle_id=j,
            )
            if seg_info:
                period_infos.append(seg_info)
            continue

        # Enough gap to fit a new model: split into a training sub-interval and the remaining test interval.
        train_end = gap_start + pd.Timedelta(minutes=float(POST_MAINT_TRAIN_MINUTES))
        if train_end > gap_end:
            train_end = gap_end

        # Local post-maintenance block for this cycle (non-maintenance points)
        local_post_mask = (
            (index > gap_start)
            & (index <= train_end)
            & (index <= gap_end)
            & (op_phase != 2)
        )
        # Per-cycle training = global baseline + local post-maint block.
        train_mask_new = initial_train_mask | local_post_mask
        test_mask_new = (index > train_end) & (index < gap_end if j < len(starts) - 1 else index <= gap_end)

        seg_label = f"after_maint_{j + 1}"
        seg_info = _fit_and_score_segment(
            seg_label,
            train_mask_new,
            test_mask_new,
            requested_cycle_id=j + 1,
        )
        if seg_info:
            period_infos.append(seg_info)
            current_train_mask = train_mask_new

    # Compute maintenance risk per segment (reset at segment boundaries).
    if "risk_score" in pred.columns:
        risk_base = pred["risk_score"].astype(float).fillna(0.0)
        risk_base = (risk_base >= float(RISK_EXCEEDANCE_QUANTILE)).astype(float)
    else:
        risk_base = pd.Series(0.0, index=index)
    maintenance_risk = _compute_segmented_risk(risk_base, risk_segments)
    pred["maintenance_risk"] = maintenance_risk
    pred["operation_phase"] = op_phase

    # Event-level evaluation of maintenance_risk thresholds
    eval_mask = pred["is_anomaly"].notna() & (~global_train_for_eval)
    best_risk_threshold, predicted_phase, risk_results, best_stats = _evaluate_risk_series(
        maintenance_risk,
        mw_sorted,
        tag="RISK-PERMAINT",
        eval_mask=eval_mask,
    )
    _assign_exceedance(pred)
    _assign_predicted_phase(pred, predicted_phase)

    # Point-wise metrics on rows with predictions (normal + pre-maintenance only),
    # excluding all training-only rows (initial slice + post-maint training days).
    m_all = _compute_pointwise_metrics(pred["is_anomaly"], pred["operation_phase"], eval_mask=eval_mask)
    threshold_info = None
    thresholds = [info["threshold"] for info in period_infos if info.get("threshold") is not None]
    if thresholds:
        threshold_info = (
            f"per-segment thresholds: min={min(thresholds):.4f}, "
            f"max={max(thresholds):.4f}, mean={np.mean(thresholds):.4f}"
        )
    _print_pointwise_metrics("Per-maint", m_all, threshold_info=threshold_info)

    _print_event_level_metrics("Per-maint", best_stats, risk_results)

    event_results = evaluate_maintenance_prediction(
        predictions=pred["predicted_phase"],
        maintenance_windows=mw_sorted,
        early_warning_minutes=THRESHOLD_POLICY_FIXED_LEAD_MINUTES,
        method_name="Per-maint",
        eval_mask=eval_mask,
        lead_step_minutes=LEAD_STEP_MINUTES,
    )
    threshold_policy_report = _build_threshold_policy_report(
        predicted_phase=predicted_phase,
        maint_windows=mw_sorted,
        eval_mask=eval_mask,
        best_stats=best_stats,
        primary_event_results=event_results,
    )
    _save_maintenance_risk_theta_sweep(
        maintenance_risk=maintenance_risk,
        maint_windows=mw_sorted,
        eval_mask=eval_mask,
        risk_results=risk_results,
        best_stats=best_stats,
        mode="per_maint",
        detector=detector_type_runtime,
    )
    alarm_island_summary = _save_alarm_island_analysis(
        predictions=pred["predicted_phase"],
        maint_windows=mw_sorted,
        eval_mask=eval_mask,
        mode="per_maint",
        detector=detector_type_runtime,
        label="Per-maint",
    )
    _print_event_extra_metrics("Per-maint", event_results)
    _save_event_plots(event_results, mode="per_maint", detector=detector_type_runtime)

    if period_infos:
        n_used = len(period_infos)
        min_train = min(info["train_size"] for info in period_infos)
        max_train = max(info["train_size"] for info in period_infos)
        print(
            f"[INFO] Per-maint models: {n_used} segments with models, "
            f"training size range={min_train}..{max_train}"
        )

    summary_info = {
        "mode": "per_maint",
        "n_segments": len(period_infos),
        "imported_models": bool(use_imported_models),
        "manifest_path": str(Path(PER_MAINT_MODEL_MANIFEST_PATH).resolve()) if use_imported_models else None,
        "detector_type": detector_type_runtime,
    }
    summary_info["event_metrics"] = event_results
    summary_info["threshold_policy"] = threshold_policy_report
    summary_info["alarm_island_summary"] = alarm_island_summary
    return pred, summary_info, best_risk_threshold, predicted_phase


def main() -> None:
    # 1) Resolve run mode and initialize artifact directories.
    mode = EXPERIMENT_MODE.lower().strip()
    if mode not in {"single", "per_maint"}:
        raise ValueError(
            f"Unsupported EXPERIMENT_MODE={EXPERIMENT_MODE!r}; "
            f"use 'single' or 'per_maint'."
        )
    _validate_runtime_configuration(mode)
    _set_global_seed(GLOBAL_SEED)
    effective_detector = _effective_detector_type(mode)
    artifact_mode = _resolve_artifact_mode(mode)
    _ensure_artifact_tree(mode=mode, detector=effective_detector)
    if artifact_mode != _normalize_mode(mode):
        print(
            "[INFO] Artifact routing override from manifest workflow: "
            f"mode={_normalize_mode(mode)} -> {artifact_mode}"
        )

    # 2) Build rolling feature matrix and maintenance context
    _log_pipeline_overview()
    t_build = _log_step_start("Build features + context")
    X, maint_windows, operation_phase = _build_features_and_context()
    _log_step_end("Build features + context", t_build)

    # 3) Run the selected experiment mode
    t_run = _log_step_start("Run experiment")
    if mode == "single":
        pred, info, train_cutoff_ts, best_risk_threshold, predicted_phase = _run_single_model_experiment(
            X, maint_windows, operation_phase
        )
    else:
        pred, info, best_risk_threshold, predicted_phase = _run_per_maintenance_experiment(
            X, maint_windows, operation_phase
        )
        train_cutoff_ts = None
    _log_step_end("Run experiment", t_run)

    # 4) Plot risk timeline
    t_plot = _log_step_start("Plot + save outputs")
    df_plot = pred[["maintenance_risk"]].copy()

    effective_show_labels = bool(SHOW_WINDOW_LABELS or USE_DEFAULT_METROPT_WINDOWS)
    save_fig_path = _output_path_with_detector(SAVE_FIG_PATH, mode, effective_detector)
    detector_label = _detector_display_label(effective_detector)
    plot_raw_timeline(
        df_plot,
        maint_windows,
        save_fig=save_fig_path,
        train_frac=TRAIN_FRAC if mode == "single" else None,
        train_cutoff_time=train_cutoff_ts,
        show_window_labels=effective_show_labels,
        window_label_fontsize=WINDOW_LABEL_FONTSIZE,
        window_label_format=WINDOW_LABEL_FORMAT,
        predicted_phase=predicted_phase,
        risk_threshold=best_risk_threshold,
        early_warning_minutes=PRE_MAINTENANCE_MINUTES,
        detector_name=detector_label,
    )
    if save_fig_path:
        print(f"[INFO] Saved plot: {Path(save_fig_path).resolve()}")

    # 5) Optional: save per-point predictions (timestamp, score, labels, risk)
    if SAVE_PRED_CSV_PATH:
        out = pred.copy()
        if "anom_score" in out.columns:
            no_score = out["anom_score"].isna()
            null_cols = [c for c in ["risk_score", "exceedance", "maintenance_risk", "predicted_phase", "is_anomaly"] if c in out.columns]
            if null_cols:
                out.loc[no_score, null_cols] = np.nan
        out.index.name = "timestamp"
        pred_path = _pred_output_path(SAVE_PRED_CSV_PATH, mode, effective_detector)
        out.to_csv(pred_path)

    _log_step_end("Plot + save outputs", t_plot)

    if maint_windows:
        rows = []
        for w in maint_windows:
            try:
                if len(w) >= 4:
                    s, e, wid, sev = w[0], w[1], w[2], w[3]
                else:
                    s, e = w[0], w[1]
                    wid, sev = None, None
                start = pd.to_datetime(s)
                end = pd.to_datetime(e)
                dur_min = int(round(max(0.0, (end - start).total_seconds() / 60.0)))
                rows.append(
                    {
                        "id": str(wid) if wid is not None else "",
                        "severity": str(sev) if sev is not None else "",
                        "start": start,
                        "end": end,
                        "duration_min": dur_min,
                    }
                )
            except Exception:
                rows.append(
                    {"id": "", "severity": "", "start": "", "end": "", "duration_min": ""}
                )
        print("[INFO] Failure windows:")
        df_windows = pd.DataFrame(rows, columns=["id", "severity", "start", "end", "duration_min"])
        try:
            print(df_windows.to_markdown(index=False))
        except Exception:
            print(df_windows.to_string(index=False))


if __name__ == "__main__":
    with log_to_file(
        _effective_detector_type(EXPERIMENT_MODE),
        _resolve_artifact_mode(EXPERIMENT_MODE),
        artifacts_root=ARTIFACTS_ROOT,
    ):
        main()
