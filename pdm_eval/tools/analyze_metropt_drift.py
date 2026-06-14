"""Standalone MetroPT drift/non-stationarity analysis CLI."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from pdm_eval import pipeline
from pdm_eval.analysis.drift import (
    aggregate_cycle_wasserstein,
    build_pca_projection,
    build_target_event_segments,
    compute_feature_pairwise_wasserstein,
    phase_counts_table,
    robust_scale_frame,
    save_cycle_pca_scatter,
    save_cycle_wasserstein_heatmap,
    summarize_cycle_features,
)
from pdm_eval.data.preprocessing import (
    build_operation_phase,
    build_rolling_features,
    load_csv,
    parse_maintenance_windows,
    select_numeric_features,
)


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MetroPT target-event cycle drift using rolling detector features."
    )
    parser.add_argument("--input", default=pipeline.INPUT_PATH, help="Input MetroPT CSV path.")
    parser.add_argument(
        "--timestamp-col",
        default=pipeline.INPUT_TIMESTAMP_COL,
        help="Timestamp column name. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/analysis/drift",
        help="Output directory for drift tables, plots, and run_config.json.",
    )
    parser.add_argument(
        "--rolling-window",
        default=pipeline.ROLLING_WINDOW,
        help="Pandas rolling window used to build detector-style features.",
    )
    parser.add_argument(
        "--pre-maintenance-minutes",
        type=float,
        default=float(pipeline.PRE_MAINTENANCE_MINUTES),
        help="Minutes before maintenance start labelled as phase 1.",
    )
    parser.add_argument(
        "--max-samples-per-cycle",
        type=int,
        default=2000,
        help="Maximum sampled rows per cycle for PCA.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic PCA sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    tables_dir = output_root / "tables"
    plots_dir = output_root / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("[DRIFT] Loading MetroPT data ...")
    raw = load_csv(args.input, args.timestamp_col, drop_unnamed=pipeline.DROP_UNNAMED_INDEX)
    feature_names = select_numeric_features(raw, prefer=pipeline.LIKELY_METROPT_FEATURES)
    if not feature_names:
        raise ValueError("No numeric MetroPT features found.")

    print(f"[DRIFT] Building rolling features ({args.rolling_window}) for {len(feature_names)} base features ...")
    rolling = build_rolling_features(raw[feature_names].copy(), rolling_window=args.rolling_window)

    maintenance_windows = parse_maintenance_windows(
        windows=None,
        maintenance_csv=None,
        use_default_windows=pipeline.USE_DEFAULT_METROPT_WINDOWS,
        default_windows=pipeline.DEFAULT_METROPT_WINDOWS,
    )
    operation_phase = build_operation_phase(
        index=rolling.index,
        windows=maintenance_windows,
        pre_minutes=args.pre_maintenance_minutes,
    ).astype(np.int8)

    print("[DRIFT] Building target-event cycle segments ...")
    segments, tail_segment = build_target_event_segments(
        index=rolling.index,
        maintenance_windows=maintenance_windows,
        operation_phase=operation_phase,
    )
    if len(segments) < 2:
        raise ValueError("Need at least two target-event cycles for pairwise drift analysis.")

    print("[DRIFT] Robust-scaling rolling features ...")
    scaled, scaling_params = robust_scale_frame(rolling)
    scaling_params.to_csv(tables_dir / "feature_scaling_params.csv", index=False)

    print("[DRIFT] Writing summary tables ...")
    summarize_cycle_features(rolling, segments).to_csv(tables_dir / "cycle_feature_summary.csv", index=False)
    phase_counts_table(segments, tail_segment).to_csv(tables_dir / "cycle_phase_counts.csv", index=False)

    print("[DRIFT] Computing feature-wise Wasserstein distances ...")
    feature_pairwise = compute_feature_pairwise_wasserstein(scaled, segments)
    cycle_pairwise = aggregate_cycle_wasserstein(feature_pairwise, segments)
    feature_pairwise.to_csv(tables_dir / "feature_pairwise_wasserstein.csv", index=False)
    cycle_pairwise.to_csv(tables_dir / "cycle_pairwise_wasserstein.csv", index=False)

    print("[DRIFT] Building PCA projection ...")
    pca_scores, pca_centroids, explained_variance = build_pca_projection(
        scaled,
        segments,
        max_samples_per_cycle=args.max_samples_per_cycle,
        random_state=args.random_state,
    )
    pca_scores.to_csv(tables_dir / "pca_cycle_scores.csv", index=False)
    pca_centroids.to_csv(tables_dir / "pca_cycle_centroids.csv", index=False)

    print("[DRIFT] Saving plots ...")
    save_cycle_wasserstein_heatmap(cycle_pairwise, plots_dir / "cycle_wasserstein_heatmap.png")
    save_cycle_pca_scatter(
        pca_scores,
        pca_centroids,
        explained_variance,
        plots_dir / "cycle_pca_scatter.png",
    )

    run_config = {
        "generated_at": datetime.now().isoformat(),
        "input": str(Path(args.input).resolve()),
        "output_root": str(output_root.resolve()),
        "timestamp_col": args.timestamp_col,
        "rolling_window": args.rolling_window,
        "pre_maintenance_minutes": args.pre_maintenance_minutes,
        "max_samples_per_cycle": args.max_samples_per_cycle,
        "random_state": args.random_state,
        "base_feature_count": len(feature_names),
        "rolling_feature_count": int(rolling.shape[1]),
        "row_count": int(rolling.shape[0]),
        "target_event_cycle_count": len(segments),
        "tail_row_count": int(tail_segment.row_count) if tail_segment is not None else 0,
        "wasserstein_method": "quantile_grid",
        "wasserstein_quantile_count": 201,
        "pca_explained_variance": {
            "pc1": explained_variance[0],
            "pc2": explained_variance[1],
        },
        "interpretation_boundary": (
            "This is dataset drift evidence over all operation phases, not a detector benchmark."
        ),
    }
    (output_root / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print(f"[DRIFT] Done. Outputs written to: {output_root.resolve()}")


if __name__ == "__main__":
    main()
