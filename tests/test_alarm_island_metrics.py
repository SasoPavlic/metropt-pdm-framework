import pandas as pd
import pytest

from metrics_alarm_islands import build_alarm_island_report


def test_alarm_island_report_classifies_and_summarizes_blocks():
    idx = pd.date_range("2020-01-01 00:00:00", periods=10, freq="1min")
    predictions = pd.Series([0, 1, 1, 0, 1, 1, 1, 0, 1, 0], index=idx)
    maintenance_windows = [(idx[6], idx[7], "#1", "high")]

    islands, summary = build_alarm_island_report(
        predictions=predictions,
        maintenance_windows=maintenance_windows,
        early_warning_minutes=2,
    )

    assert list(islands["relation"]) == [
        "false_alarm",
        "early_warning_overlap",
        "false_alarm",
    ]
    assert list(islands["points"]) == [2, 3, 1]
    assert summary["island_count"] == 3
    assert summary["early_warning_overlap_islands"] == 1
    assert summary["false_alarm_islands"] == 2
    assert summary["coverage"] == pytest.approx(6 / 10)
    assert summary["longest_island_id"] == 2


def test_alarm_island_report_applies_eval_mask_and_splits_blocks():
    idx = pd.date_range("2020-01-01 00:00:00", periods=6, freq="1min")
    predictions = pd.Series([1, 1, 1, 1, 0, 1], index=idx)
    eval_mask = pd.Series([True, True, False, True, True, False], index=idx)

    islands, summary = build_alarm_island_report(
        predictions=predictions,
        maintenance_windows=[],
        early_warning_minutes=2,
        eval_mask=eval_mask,
    )

    assert list(islands["points"]) == [2, 1]
    assert summary["alarm_points"] == 3
    assert summary["total_eval_points"] == 4
    assert summary["coverage"] == pytest.approx(3 / 4)
