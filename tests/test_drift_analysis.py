from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from pdm_eval.analysis.drift import (
    aggregate_cycle_wasserstein,
    build_pca_projection,
    build_target_event_segments,
    compute_feature_pairwise_wasserstein,
    robust_scale_frame,
)


def test_target_event_segments_and_phase_counts() -> None:
    index = pd.date_range("2020-01-01 00:00:00", periods=13, freq="h")
    windows = [
        (index[3], index[4], "#1", "high"),
        (index[8], index[9], "#2", "high"),
    ]
    phases = pd.Series(np.zeros(len(index), dtype=np.int8), index=index)
    phases.iloc[[2, 7]] = 1
    phases.iloc[[3, 4, 8, 9]] = 2

    segments, tail = build_target_event_segments(index, windows, phases)

    assert len(segments) == 2
    assert segments[0].cycle_id == 1
    assert segments[0].segment_start == index[0]
    assert segments[0].segment_end == index[4]
    assert segments[0].row_count == 5
    assert segments[0].phase0_rows == 2
    assert segments[0].phase1_rows == 1
    assert segments[0].phase2_rows == 2

    assert segments[1].cycle_id == 2
    assert segments[1].segment_start == index[5]
    assert segments[1].segment_end == index[9]
    assert segments[1].row_count == 5
    assert segments[1].phase0_rows == 2
    assert segments[1].phase1_rows == 1
    assert segments[1].phase2_rows == 2

    assert tail is not None
    assert tail.segment_type == "post_last_tail"
    assert tail.row_count == 3


def test_robust_scale_handles_constant_features() -> None:
    frame = pd.DataFrame(
        {
            "constant": [5.0, 5.0, 5.0, 5.0],
            "varying": [1.0, 2.0, 3.0, 4.0],
        }
    )

    scaled, params = robust_scale_frame(frame)

    assert np.isfinite(scaled.to_numpy()).all()
    assert np.allclose(scaled["constant"], 0.0)
    assert float(params.loc[params["feature"] == "constant", "scale_used"].iloc[0]) == 1.0


def test_wasserstein_detects_shifted_distribution() -> None:
    index = pd.date_range("2020-01-01 00:00:00", periods=6, freq="h")
    windows = [
        (index[2], index[2], "#1", "high"),
        (index[5], index[5], "#2", "high"),
    ]
    phases = pd.Series(np.zeros(len(index), dtype=np.int8), index=index)
    segments, _tail = build_target_event_segments(index, windows, phases)

    identical = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]}, index=index)
    shifted = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]}, index=index)

    identical_feature = compute_feature_pairwise_wasserstein(identical, segments)
    shifted_feature = compute_feature_pairwise_wasserstein(shifted, segments)
    identical_cycle = aggregate_cycle_wasserstein(identical_feature, segments)
    shifted_cycle = aggregate_cycle_wasserstein(shifted_feature, segments)

    identical_distance = float(
        identical_cycle.query("cycle_id_a == 1 and cycle_id_b == 2")["median_wasserstein"].iloc[0]
    )
    shifted_distance = float(
        shifted_cycle.query("cycle_id_a == 1 and cycle_id_b == 2")["median_wasserstein"].iloc[0]
    )
    assert identical_distance == 0.0
    assert shifted_distance > identical_distance


def test_pca_sampling_is_deterministic() -> None:
    index = pd.date_range("2020-01-01 00:00:00", periods=20, freq="h")
    windows = [
        (index[9], index[9], "#1", "high"),
        (index[19], index[19], "#2", "high"),
    ]
    phases = pd.Series(np.zeros(len(index), dtype=np.int8), index=index)
    segments, _tail = build_target_event_segments(index, windows, phases)
    frame = pd.DataFrame(
        {
            "feature_a": np.arange(20, dtype=float),
            "feature_b": np.arange(20, dtype=float) ** 2,
        },
        index=index,
    )
    scaled, _params = robust_scale_frame(frame)

    scores_a, centroids_a, explained_a = build_pca_projection(
        scaled,
        segments,
        max_samples_per_cycle=4,
        random_state=42,
    )
    scores_b, centroids_b, explained_b = build_pca_projection(
        scaled,
        segments,
        max_samples_per_cycle=4,
        random_state=42,
    )

    pdt.assert_frame_equal(scores_a, scores_b)
    pdt.assert_frame_equal(centroids_a, centroids_b)
    assert explained_a == explained_b

