from pathlib import Path

import pandas as pd
import pytest

import main as metropt_main
from metrics_point import evaluate_risk_thresholds


def test_evaluate_risk_thresholds_respects_eval_mask_for_coverage():
    idx = pd.date_range("2020-01-01 00:00:00", periods=6, freq="1min")
    risk = pd.Series([0.2, 0.7, 0.8, 0.1, 0.9, 0.2], index=idx)
    eval_mask = pd.Series([False, True, True, True, False, False], index=idx)
    windows = [(idx[4], idx[5])]

    out = evaluate_risk_thresholds(
        risk=risk,
        maintenance_windows=windows,
        thresholds=[0.5],
        early_warning_minutes=120,
        eval_mask=eval_mask,
    )

    assert len(out) == 1
    row = out[0]
    assert row["total_points"] == 3
    assert row["alarm_points"] == 2
    assert row["coverage"] == 2 / 3
    assert row["maintenance_risk_theta"] == 0.5


def test_risk_threshold_grid_includes_explicit_high_theta_probes():
    thresholds = metropt_main.build_risk_threshold_grid(
        "0.10:0.90:0.05",
        [0.925, 0.95, 0.975, 0.985, 0.99, 0.995],
    )

    assert 0.90 in thresholds
    assert thresholds[-6:] == [0.925, 0.95, 0.975, 0.985, 0.99, 0.995]
    assert len(thresholds) == len(set(thresholds))


def test_locked_theta_policy_prefers_feasible_then_tiebreaks_by_lower_theta(monkeypatch):
    monkeypatch.setattr(metropt_main, "parse_risk_grid_spec", lambda _spec: [0.4, 0.5, 0.6])
    monkeypatch.setattr(
        metropt_main,
        "evaluate_risk_thresholds",
        lambda **_kwargs: [
            {"threshold": 0.4, "precision": 0.40, "recall": 0.70, "f1": 0.50, "tp": 1, "fp": 1, "fn": 1, "coverage": 0.19, "coverage_percent": 19.0},
            {"threshold": 0.5, "precision": 0.40, "recall": 0.70, "f1": 0.50, "tp": 1, "fp": 1, "fn": 1, "coverage": 0.15, "coverage_percent": 15.0},
            {"threshold": 0.6, "precision": 0.90, "recall": 0.30, "f1": 0.45, "tp": 1, "fp": 0, "fn": 2, "coverage": 0.05, "coverage_percent": 5.0},
        ],
    )

    idx = pd.date_range("2020-01-01", periods=5, freq="1min")
    risk = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=idx)
    eval_mask = pd.Series([True] * len(idx), index=idx)
    windows = [(idx[3], idx[4])]

    theta, _pred, _rows, best = metropt_main._evaluate_risk_series(
        maintenance_risk=risk,
        maint_windows=windows,
        tag="TEST",
        eval_mask=eval_mask,
    )

    assert theta == 0.4
    assert best["selection_mode"] == "feasible"
    assert best["target_gap"] == 0.0


def test_locked_theta_policy_fallback_uses_target_gap(monkeypatch):
    monkeypatch.setattr(metropt_main, "parse_risk_grid_spec", lambda _spec: [0.4, 0.5, 0.6])
    monkeypatch.setattr(
        metropt_main,
        "evaluate_risk_thresholds",
        lambda **_kwargs: [
            {"threshold": 0.4, "precision": 0.60, "recall": 0.50, "f1": 0.55, "tp": 1, "fp": 1, "fn": 1, "coverage": 0.22, "coverage_percent": 22.0},
            {"threshold": 0.5, "precision": 0.59, "recall": 0.59, "f1": 0.59, "tp": 1, "fp": 1, "fn": 1, "coverage": 0.25, "coverage_percent": 25.0},
            {"threshold": 0.6, "precision": 0.70, "recall": 0.45, "f1": 0.55, "tp": 1, "fp": 1, "fn": 1, "coverage": 0.18, "coverage_percent": 18.0},
        ],
    )

    idx = pd.date_range("2020-01-01", periods=5, freq="1min")
    risk = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=idx)
    eval_mask = pd.Series([True] * len(idx), index=idx)
    windows = [(idx[3], idx[4])]

    theta, _pred, _rows, best = metropt_main._evaluate_risk_series(
        maintenance_risk=risk,
        maint_windows=windows,
        tag="TEST",
        eval_mask=eval_mask,
    )

    # target_gap:
    # 0.4 => max(0, 0.60-0.50)+max(0, 0.22-0.20) = 0.12
    # 0.5 => max(0, 0.60-0.59)+max(0, 0.25-0.20) = 0.06
    # 0.6 => max(0, 0.60-0.45)+max(0, 0.18-0.20) = 0.15
    assert theta == 0.5
    assert best["selection_mode"] == "fallback"
    assert best["target_gap"] == 0.06


def test_threshold_policy_report_uses_maintenance_risk_theta_name():
    idx = pd.date_range("2020-01-01", periods=5, freq="1min")
    predicted = pd.Series([0, 1, 1, 0, 0], index=idx)
    eval_mask = pd.Series([True] * len(idx), index=idx)
    windows = [(idx[3], idx[4])]
    primary_event_results = {
        "event_scores": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 1, "fp": 0, "fn": 0},
        "coverage": {"alarm_coverage_percent": 40.0},
        "far": {"far_per_week": 0.0},
        "first_alarm_accuracy": {"first_alarm_accuracy": 1.0},
        "nab": {"standard": {"nab_score_normalized": 100.0}},
    }

    report = metropt_main._build_threshold_policy_report(
        predicted_phase=predicted,
        maint_windows=windows,
        eval_mask=eval_mask,
        best_stats={
            "threshold": 0.615,
            "selection_mode": "fallback",
            "target_gap": 0.1,
        },
        primary_event_results=primary_event_results,
    )

    assert report["selected_maintenance_risk_theta"] == 0.615
    assert "selected_theta" not in report


def test_theta_sweep_summary_statistics_and_metric_directions():
    rows = [
        {
            "maintenance_risk_theta": 0.1,
            "tp": 2,
            "fp": 10,
            "fn": 8,
            "precision": 0.10,
            "recall": 0.20,
            "f1": 0.13,
            "coverage": 0.50,
            "far_per_week": 5.0,
            "faa": 0.2,
            "nab_standard": 10.0,
            "target_gap": 0.40,
        },
        {
            "maintenance_risk_theta": 0.2,
            "tp": 4,
            "fp": 6,
            "fn": 6,
            "precision": 0.20,
            "recall": 0.40,
            "f1": 0.27,
            "coverage": 0.30,
            "far_per_week": 3.0,
            "faa": 0.4,
            "nab_standard": 20.0,
            "target_gap": 0.20,
        },
        {
            "maintenance_risk_theta": 0.3,
            "tp": 3,
            "fp": 2,
            "fn": 7,
            "precision": 0.30,
            "recall": 0.30,
            "f1": 0.30,
            "coverage": 0.10,
            "far_per_week": 1.0,
            "faa": 0.1,
            "nab_standard": 15.0,
            "target_gap": 0.10,
        },
    ]

    summary = metropt_main._build_theta_sweep_summary(rows).set_index("metric")

    assert summary.loc["precision", "count"] == 3
    assert summary.loc["precision", "mean"] == pytest.approx(0.20)
    assert summary.loc["precision", "median"] == pytest.approx(0.20)
    assert summary.loc["precision", "std"] == pytest.approx(0.0816496581)
    assert summary.loc["precision", "q25"] == pytest.approx(0.15)
    assert summary.loc["precision", "q75"] == pytest.approx(0.25)
    assert summary.loc["f1", "best_value"] == pytest.approx(0.30)
    assert summary.loc["f1", "best_theta"] == pytest.approx(0.3)
    assert summary.loc["fp", "direction"] == "lower_is_better"
    assert summary.loc["fp", "best_value"] == pytest.approx(2)
    assert summary.loc["fp", "best_theta"] == pytest.approx(0.3)
    assert summary.loc["recall", "direction"] == "higher_is_better"
    assert summary.loc["recall", "best_theta"] == pytest.approx(0.2)
    assert summary.loc["coverage", "best_theta"] == pytest.approx(0.3)
    assert summary.loc["target_gap", "best_theta"] == pytest.approx(0.3)


def test_theta_sweep_export_writes_summary_and_preserves_raw_sweep(monkeypatch, tmp_path):
    def fake_pred_output_path(path, mode, detector):
        return str(tmp_path / path)

    monkeypatch.setattr(metropt_main, "_pred_output_path", fake_pred_output_path)
    monkeypatch.setattr(
        metropt_main,
        "evaluate_maintenance_prediction",
        lambda **_kwargs: {
            "event_scores": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "tp": 1, "fp": 1, "fn": 1},
            "coverage": {"alarm_coverage_percent": 10.0},
            "far": {"far_per_week": 2.0},
            "first_alarm_accuracy": {"first_alarm_accuracy": 0.25},
            "nab": {"standard": {"nab_score_normalized": 30.0}},
        },
    )

    idx = pd.date_range("2020-01-01", periods=4, freq="1min")
    maintenance_risk = pd.Series([0.1, 0.4, 0.7, 0.9], index=idx)
    eval_mask = pd.Series([True] * len(idx), index=idx)
    risk_results = [
        {"threshold": 0.2, "tp": 1, "fp": 4, "fn": 3, "precision": 0.20, "recall": 0.25, "f1": 0.22, "coverage": 0.75, "target_gap": 0.55},
        {"threshold": 0.6, "tp": 2, "fp": 1, "fn": 2, "precision": 0.67, "recall": 0.50, "f1": 0.57, "coverage": 0.25, "target_gap": 0.15},
    ]

    metropt_main._save_maintenance_risk_theta_sweep(
        maintenance_risk=maintenance_risk,
        maint_windows=[(idx[2], idx[3])],
        eval_mask=eval_mask,
        risk_results=risk_results,
        best_stats={"threshold": 0.6, "selection_mode": "fallback"},
        mode="single",
        detector="iforest",
    )

    sweep_path = tmp_path / "theta_sweep_maintenance_risk.csv"
    summary_path = tmp_path / "theta_sweep_summary.csv"
    assert sweep_path.exists()
    assert summary_path.exists()

    sweep = pd.read_csv(sweep_path)
    assert list(sweep.columns) == [
        "maintenance_risk_theta",
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
        "selection_mode",
    ]
    assert len(sweep) == 2

    summary = pd.read_csv(summary_path).set_index("metric")
    assert "f1" in summary.index
    assert "coverage" in summary.index
    assert summary.loc["f1", "best_theta"] == pytest.approx(0.6)
    assert summary.loc["coverage", "best_theta"] == pytest.approx(0.6)


def test_resolve_manifest_cycle_trained_and_alias(tmp_path: Path):
    model_file = tmp_path / "cycle_01" / "model.pt"
    meta_file = tmp_path / "cycle_01" / "model_meta.json"
    scaler_file = tmp_path / "cycle_01" / "scaler.joblib"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_text("x", encoding="utf-8")
    meta_file.write_text("{}", encoding="utf-8")
    scaler_file.write_text("x", encoding="utf-8")

    manifest = {
        "cycles": {
            "01": {
                "status": "trained",
                "cycle_id": 1,
                "model_path": str(model_file),
                "meta_path": str(meta_file),
                "scaler_path": str(scaler_file),
            },
            "02": {
                "status": "alias",
                "alias_to": 1,
            },
        }
    }

    resolved_trained = metropt_main._resolve_manifest_cycle(
        manifest=manifest,
        manifest_dir=tmp_path,
        cycle_id=1,
        strict=True,
    )
    resolved_alias = metropt_main._resolve_manifest_cycle(
        manifest=manifest,
        manifest_dir=tmp_path,
        cycle_id=2,
        strict=True,
    )

    assert resolved_trained is not None
    assert resolved_alias is not None
    assert resolved_trained["resolved_cycle_id"] == 1
    assert resolved_alias["resolved_cycle_id"] == 1
    assert Path(resolved_alias["model_path"]).exists()
    assert Path(resolved_alias["meta_path"]).exists()
    assert Path(resolved_alias["scaler_path"]).exists()
