import pandas as pd

from pdm_eval.metrics.event import calculate_first_alarm_accuracy


def test_faa_penalizes_too_early_alarm_start():
    idx = pd.date_range("2020-01-01 00:00:00", periods=300, freq="1min")
    predictions = pd.Series(False, index=idx)

    # Alarm starts before warning window and stays active into it.
    predictions.loc["2020-01-01 00:30:00":"2020-01-01 02:10:00"] = True
    windows = [(pd.Timestamp("2020-01-01 02:00:00"), pd.Timestamp("2020-01-01 02:10:00"))]

    result = calculate_first_alarm_accuracy(predictions, windows, early_warning_minutes=60)

    assert result["tp_events"] == 1
    assert result["first_alarm_in_window"] == 0
    assert result["first_alarm_accuracy"] == 0.0


def test_faa_scores_inside_window_alarm_start_as_correct():
    idx = pd.date_range("2020-01-01 00:00:00", periods=300, freq="1min")
    predictions = pd.Series(False, index=idx)

    # Alarm starts inside strict warning window.
    predictions.loc["2020-01-01 01:20:00":"2020-01-01 02:10:00"] = True
    windows = [(pd.Timestamp("2020-01-01 02:00:00"), pd.Timestamp("2020-01-01 02:10:00"))]

    result = calculate_first_alarm_accuracy(predictions, windows, early_warning_minutes=60)

    assert result["tp_events"] == 1
    assert result["first_alarm_in_window"] == 1
    assert result["first_alarm_accuracy"] == 1.0
