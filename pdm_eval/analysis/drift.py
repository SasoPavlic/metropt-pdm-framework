"""MetroPT maintenance-cycle drift analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class TargetEventSegment:
    """Rows assigned to one target maintenance event or to the post-final tail."""

    cycle_id: Optional[int]
    cycle_label: str
    segment_type: str
    segment_start: Optional[pd.Timestamp]
    segment_end: Optional[pd.Timestamp]
    maintenance_start: Optional[pd.Timestamp]
    maintenance_end: Optional[pd.Timestamp]
    mask: np.ndarray
    phase0_rows: int
    phase1_rows: int
    phase2_rows: int

    @property
    def row_count(self) -> int:
        return int(np.sum(self.mask))


def _timestamp_or_none(value: object) -> Optional[pd.Timestamp]:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _format_ts(value: Optional[pd.Timestamp]) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _phase_counts(mask: np.ndarray, phases: np.ndarray) -> tuple[int, int, int]:
    if mask.shape[0] != phases.shape[0]:
        raise ValueError("Mask and phase arrays must have the same length.")
    selected = phases[mask]
    return (
        int(np.sum(selected == 0)),
        int(np.sum(selected == 1)),
        int(np.sum(selected == 2)),
    )


def _actual_bounds(index: pd.DatetimeIndex, mask: np.ndarray) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if not mask.any():
        return None, None
    selected = index[mask]
    return pd.Timestamp(selected.min()), pd.Timestamp(selected.max())


def build_target_event_segments(
    index: pd.DatetimeIndex,
    maintenance_windows: Iterable[tuple],
    operation_phase: pd.Series,
) -> tuple[list[TargetEventSegment], Optional[TargetEventSegment]]:
    """
    Build target-event cycle segments.

    Cycle #1 spans dataset start through maintenance #1 end. Cycle #k spans
    after maintenance #(k-1) end through maintenance #k end. Rows after the
    final maintenance are returned as an optional tail segment and excluded from
    pairwise drift comparisons by callers.
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(pd.to_datetime(index))
    if len(index) == 0:
        return [], None

    phases = operation_phase.reindex(index).fillna(0).astype(np.int8).to_numpy()
    windows = sorted(maintenance_windows, key=lambda w: pd.to_datetime(w[0]))
    segments: list[TargetEventSegment] = []
    previous_end: Optional[pd.Timestamp] = None

    for cycle_idx, window in enumerate(windows, start=1):
        maint_start = _timestamp_or_none(window[0])
        maint_end = _timestamp_or_none(window[1])
        label = str(window[2]) if len(window) >= 3 and window[2] is not None else f"#{cycle_idx}"
        if maint_start is None or maint_end is None:
            continue

        if previous_end is None:
            mask = (index >= index.min()) & (index <= maint_end)
        else:
            mask = (index > previous_end) & (index <= maint_end)
        mask = np.asarray(mask, dtype=bool)
        seg_start, seg_end = _actual_bounds(index, mask)
        p0, p1, p2 = _phase_counts(mask, phases)
        segments.append(
            TargetEventSegment(
                cycle_id=cycle_idx,
                cycle_label=label,
                segment_type="target_event_cycle",
                segment_start=seg_start,
                segment_end=seg_end,
                maintenance_start=maint_start,
                maintenance_end=maint_end,
                mask=mask,
                phase0_rows=p0,
                phase1_rows=p1,
                phase2_rows=p2,
            )
        )
        previous_end = maint_end

    tail: Optional[TargetEventSegment] = None
    if previous_end is not None:
        tail_mask = np.asarray(index > previous_end, dtype=bool)
        if tail_mask.any():
            seg_start, seg_end = _actual_bounds(index, tail_mask)
            p0, p1, p2 = _phase_counts(tail_mask, phases)
            tail = TargetEventSegment(
                cycle_id=None,
                cycle_label="post_last_tail",
                segment_type="post_last_tail",
                segment_start=seg_start,
                segment_end=seg_end,
                maintenance_start=None,
                maintenance_end=None,
                mask=tail_mask,
                phase0_rows=p0,
                phase1_rows=p1,
                phase2_rows=p2,
            )
    return segments, tail


def phase_counts_table(
    segments: Iterable[TargetEventSegment],
    tail_segment: Optional[TargetEventSegment] = None,
) -> pd.DataFrame:
    rows = [_segment_record(segment) for segment in segments]
    if tail_segment is not None:
        rows.append(_segment_record(tail_segment))
    return pd.DataFrame(rows)


def _segment_record(segment: TargetEventSegment) -> dict:
    return {
        "cycle_id": segment.cycle_id,
        "cycle_label": segment.cycle_label,
        "segment_type": segment.segment_type,
        "segment_start": _format_ts(segment.segment_start),
        "segment_end": _format_ts(segment.segment_end),
        "maintenance_start": _format_ts(segment.maintenance_start),
        "maintenance_end": _format_ts(segment.maintenance_end),
        "row_count": segment.row_count,
        "phase0_rows": segment.phase0_rows,
        "phase1_rows": segment.phase1_rows,
        "phase2_rows": segment.phase2_rows,
    }


def robust_scale_frame(
    frame: pd.DataFrame,
    epsilon: float = 1e-9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robust-scale numeric columns with global median/IQR and safe constants."""
    numeric = frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    if numeric.empty:
        raise ValueError("No numeric columns available for drift analysis.")

    med = numeric.median(axis=0, skipna=True).fillna(0.0)
    q75 = numeric.quantile(0.75, axis=0, interpolation="linear")
    q25 = numeric.quantile(0.25, axis=0, interpolation="linear")
    iqr = (q75 - q25).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    safe_iqr = iqr.where(iqr.abs() >= float(epsilon), 1.0)
    filled = numeric.fillna(med)
    scaled = (filled - med) / safe_iqr
    params = pd.DataFrame(
        {
            "feature": numeric.columns,
            "median": med.reindex(numeric.columns).to_numpy(dtype=float),
            "iqr": iqr.reindex(numeric.columns).to_numpy(dtype=float),
            "scale_used": safe_iqr.reindex(numeric.columns).to_numpy(dtype=float),
        }
    )
    return scaled.astype(float), params


def summarize_cycle_features(
    features: pd.DataFrame,
    segments: Iterable[TargetEventSegment],
) -> pd.DataFrame:
    """Create long-form per-cycle feature summary statistics."""
    rows: list[dict] = []
    numeric = features.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    for segment in segments:
        subset = numeric.loc[segment.mask]
        for feature in numeric.columns:
            values = subset[feature].dropna()
            if values.empty:
                stats = {
                    "valid_count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "iqr": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }
            else:
                q75 = float(values.quantile(0.75))
                q25 = float(values.quantile(0.25))
                stats = {
                    "valid_count": int(values.shape[0]),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0,
                    "median": float(values.median()),
                    "iqr": float(q75 - q25),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            rows.append(
                {
                    "cycle_id": segment.cycle_id,
                    "cycle_label": segment.cycle_label,
                    "feature": feature,
                    "row_count": segment.row_count,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def compute_feature_pairwise_wasserstein(
    scaled_features: pd.DataFrame,
    segments: list[TargetEventSegment],
    quantile_count: int = 201,
) -> pd.DataFrame:
    """Compute feature-wise Wasserstein distances for unordered cycle pairs.

    For CPU-friendly full MetroPT runs, W1 is estimated from each cycle's
    empirical quantile function. In 1D, Wasserstein-1 is the integral of the
    absolute difference between quantile functions.
    """
    if quantile_count < 2:
        raise ValueError("quantile_count must be at least 2.")

    rows: list[dict] = []
    features = list(scaled_features.columns)
    quantiles = np.linspace(0.0, 1.0, int(quantile_count))
    profiles: dict[tuple[int, str], Optional[np.ndarray]] = {}

    for segment in segments:
        segment_frame = scaled_features.loc[segment.mask, features]
        for feature in features:
            values = segment_frame[feature].dropna().to_numpy(dtype=float)
            if values.size == 0:
                profiles[(int(segment.cycle_id), feature)] = None
            else:
                profiles[(int(segment.cycle_id), feature)] = np.quantile(values, quantiles)

    for i, left in enumerate(segments):
        for right in segments[i + 1 :]:
            for feature in features:
                left_profile = profiles.get((int(left.cycle_id), feature))
                right_profile = profiles.get((int(right.cycle_id), feature))
                if left_profile is None or right_profile is None:
                    distance = np.nan
                else:
                    distance = float(np.trapz(np.abs(left_profile - right_profile), quantiles))
                rows.append(
                    {
                        "cycle_id_a": left.cycle_id,
                        "cycle_label_a": left.cycle_label,
                        "cycle_id_b": right.cycle_id,
                        "cycle_label_b": right.cycle_label,
                        "feature": feature,
                        "wasserstein": distance,
                        "wasserstein_method": "quantile_grid",
                        "quantile_count": int(quantile_count),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_cycle_wasserstein(
    feature_pairwise: pd.DataFrame,
    segments: list[TargetEventSegment],
) -> pd.DataFrame:
    """Aggregate feature distances into a symmetric cycle-distance table."""
    pair_lookup: dict[tuple[int, int], pd.Series] = {}
    if not feature_pairwise.empty:
        grouped = feature_pairwise.groupby(["cycle_id_a", "cycle_id_b"], dropna=False)["wasserstein"]
        for key, values in grouped:
            pair_lookup[(int(key[0]), int(key[1]))] = values

    rows: list[dict] = []
    for left in segments:
        for right in segments:
            if left.cycle_id == right.cycle_id:
                median_distance = 0.0
                feature_count = len(feature_pairwise["feature"].unique()) if not feature_pairwise.empty else 0
            else:
                key = (min(int(left.cycle_id), int(right.cycle_id)), max(int(left.cycle_id), int(right.cycle_id)))
                values = pair_lookup.get(key, pd.Series(dtype=float)).dropna()
                feature_count = int(values.shape[0])
                median_distance = float(values.median()) if feature_count else np.nan
            rows.append(
                {
                    "cycle_id_a": left.cycle_id,
                    "cycle_label_a": left.cycle_label,
                    "cycle_id_b": right.cycle_id,
                    "cycle_label_b": right.cycle_label,
                    "median_wasserstein": median_distance,
                    "feature_count": feature_count,
                }
            )
    return pd.DataFrame(rows)


def build_pca_projection(
    scaled_features: pd.DataFrame,
    segments: list[TargetEventSegment],
    max_samples_per_cycle: int = 2000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, float]]:
    """Sample rows per cycle and project them into two PCA dimensions."""
    if max_samples_per_cycle <= 0:
        raise ValueError("max_samples_per_cycle must be positive.")

    rng = np.random.default_rng(int(random_state))
    sampled_frames: list[pd.DataFrame] = []
    for segment in segments:
        positions = np.flatnonzero(segment.mask)
        if positions.size == 0:
            continue
        if positions.size > max_samples_per_cycle:
            positions = np.sort(rng.choice(positions, size=max_samples_per_cycle, replace=False))
        subset = scaled_features.iloc[positions].copy()
        subset.insert(0, "timestamp", scaled_features.index[positions])
        subset.insert(1, "cycle_id", segment.cycle_id)
        subset.insert(2, "cycle_label", segment.cycle_label)
        sampled_frames.append(subset)

    if not sampled_frames:
        return pd.DataFrame(), pd.DataFrame(), (0.0, 0.0)

    sampled = pd.concat(sampled_frames, ignore_index=True)
    feature_cols = [c for c in sampled.columns if c not in {"timestamp", "cycle_id", "cycle_label"}]
    matrix = sampled[feature_cols].to_numpy(dtype=float)
    n_components = min(2, matrix.shape[0], matrix.shape[1])
    if n_components <= 0:
        return pd.DataFrame(), pd.DataFrame(), (0.0, 0.0)

    pca = PCA(n_components=n_components, random_state=int(random_state))
    coords = pca.fit_transform(matrix)
    sampled["pc1"] = coords[:, 0]
    sampled["pc2"] = coords[:, 1] if n_components >= 2 else 0.0
    scores = sampled[["timestamp", "cycle_id", "cycle_label", "pc1", "pc2"]].copy()
    scores["timestamp"] = pd.to_datetime(scores["timestamp"]).map(lambda ts: ts.isoformat())

    centroids = (
        scores.groupby(["cycle_id", "cycle_label"], as_index=False)[["pc1", "pc2"]]
        .mean()
        .sort_values("cycle_id")
        .reset_index(drop=True)
    )
    explained = list(pca.explained_variance_ratio_)
    while len(explained) < 2:
        explained.append(0.0)
    return scores, centroids, (float(explained[0]), float(explained[1]))


def save_cycle_wasserstein_heatmap(cycle_distances: pd.DataFrame, output_path: str | Path) -> None:
    """Save a symmetric cycle-distance heatmap."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if cycle_distances.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No cycle distances available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    labels = (
        cycle_distances[["cycle_id_a", "cycle_label_a"]]
        .drop_duplicates()
        .sort_values("cycle_id_a")["cycle_label_a"]
        .tolist()
    )
    matrix = (
        cycle_distances.pivot(index="cycle_label_a", columns="cycle_label_b", values="median_wasserstein")
        .reindex(index=labels, columns=labels)
        .to_numpy(dtype=float)
    )
    masked = np.ma.masked_invalid(matrix)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(masked, cmap="viridis")
    ax.set_title("MetroPT target-event cycle drift (median Wasserstein)")
    ax.set_xlabel("Target maintenance cycle")
    ax.set_ylabel("Target maintenance cycle")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Median feature-wise Wasserstein")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_cycle_pca_scatter(
    scores: pd.DataFrame,
    centroids: pd.DataFrame,
    explained_variance: tuple[float, float],
    output_path: str | Path,
) -> None:
    """Save a PCA scatter plot with sampled rows and cycle centroids."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    if scores.empty:
        ax.text(0.5, 0.5, "No PCA samples available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    labels = sorted(scores["cycle_label"].unique(), key=lambda label: int(scores.loc[scores["cycle_label"] == label, "cycle_id"].iloc[0]))
    cmap = plt.get_cmap("tab20")
    for idx, label in enumerate(labels):
        subset = scores[scores["cycle_label"] == label]
        color = cmap(idx % cmap.N)
        ax.scatter(subset["pc1"], subset["pc2"], s=8, alpha=0.18, color=color, edgecolors="none")

    for idx, row in centroids.iterrows():
        color = cmap(idx % cmap.N)
        ax.scatter(row["pc1"], row["pc2"], s=90, color=color, edgecolors="black", linewidths=0.8)
        ax.text(row["pc1"], row["pc2"], str(row["cycle_label"]), fontsize=8, ha="left", va="bottom")

    ax.set_title("MetroPT rolling-feature PCA by target maintenance cycle")
    ax.set_xlabel(f"PC1 ({explained_variance[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_variance[1] * 100:.1f}% variance)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
