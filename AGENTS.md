# AGENTS.md

## Purpose and Workflow Role
`metropt-pdm-framework` is the evaluation repository for MetroPT-style predictive maintenance:
- Loads telemetry and engineers rolling features.
- Runs detector backends in `single` or `per_maint` mode.
- Computes point-wise and event/risk metrics.
- Produces prediction artifacts and plots.
- Optionally consumes per-cycle exported artifacts + manifest from `NiaNetVAE`.

## Main Entry Points
- `main.py`: orchestration, mode selection, manifest resolution, evaluation, outputs.
- `data_utils.py`: CSV load, timestamp handling, rolling feature engineering, maintenance phase labels.
- `detectors/`: backend implementations and factory (`get_detector`).
- `metrics_point.py`: point-wise confusion + risk-threshold grid scoring.
- `metrics_event.py`: event-level maintenance metrics (TTD/FAR/FAA/MTIA/PR-vs-lead/NAB).
- `plotting.py`: risk timeline and event summary plots.

## Pipeline Stages (Keep Stable)
1. Load + clean telemetry (`load_csv`).
2. Build rolling features (`build_rolling_features`).
3. Build maintenance context (`parse_maintenance_windows`, `build_operation_phase`).
4. Fit/score detector by mode (`_run_single_model_experiment` / `_run_per_maintenance_experiment`).
5. Convert scores to `risk_score`, `exceedance`, `maintenance_risk`, `predicted_phase`.
6. Evaluate point-wise and event metrics.
7. Save predictions + plots and log run output.

## Maintenance / Label Semantics Rules
- Preserve `operation_phase` meaning: `0=normal`, `1=pre-maintenance`, `2=maintenance`.
- Preserve default window semantics from `DEFAULT_METROPT_WINDOWS` unless task explicitly changes dataset definition.
- Preserve risk interpretation: `maintenance_risk` is rolling average of exceedance points, and `predicted_phase` is thresholded risk state.
- Do not silently change event-metric meanings (TTD/FAR/FAA/MTIA/NAB) or threshold-selection logic.

## Imported `NiaNetVAE` Compatibility Rules
- Imported mode is enabled only in `per_maint` when `PER_MAINT_USE_IMPORTED_MODELS=True`.
- Manifest contract expected by resolver:
  - top-level `cycles` object,
  - per-cycle `status` in `{trained, alias, missing}`,
  - `trained` entries provide `model_path` + `meta_path`,
  - `alias` entries provide `alias_to`.
- Preserve support for relative manifest paths and backward-compatible absolute-path fallback.
- Keep `PER_MAINT_MODEL_STRICT` fail-fast behavior for missing/unresolvable artifacts.
- Do not add workaround hacks for manifest/model mismatches; fix and document the true contract issue (and coordinate with `NiaNetVAE` when required).

## Detector / Metric Extension Guidelines
- New detector backends must implement `BaseDetector.fit()` and `BaseDetector.score()` and be registered in `detectors/__init__.py`.
- Keep anomaly score contract: higher score means more anomalous.
- Preserve `train_and_score` postprocessing contract unless explicitly migrating all call sites.
- Add new metrics in dedicated metric modules and keep existing outputs backward compatible.

## Validation Expectations
Run from repository root:
- Main run smoke test:
  - `python main.py`
- Imported-model smoke test:
  - set `EXPERIMENT_MODE="per_maint"`, `PER_MAINT_USE_IMPORTED_MODELS=True`, and valid `PER_MAINT_MODEL_MANIFEST_PATH`, then run `python main.py`.
- Verify outputs after run:
  - log file in `logs/<detector>_<mode>.log`,
  - predictions CSV in `datasets/` (if enabled),
  - plots in `plots/` (if enabled),
  - no unresolved manifest cycles when strict mode is enabled.

## Generated Files and Large Artifacts
- Treat `datasets/metropt3_predictions*.csv`, `plots/*.png`, and `logs/*.log` as generated outputs.
- Avoid committing large generated datasets/predictions unless explicitly requested.
- Keep source dataset immutable during implementation tasks.

## Explicit Assumptions
- Assumption: `main.py` constants are the active configuration mechanism (no separate runtime config file currently in use).
- Assumption: imported `NiaNetVAE` artifacts follow current `model_meta.json` + `model.pt` + manifest schema used by resolver.
- Assumption: MetroPT maintenance windows and pre-maintenance horizon definitions are the evaluation ground truth unless a task explicitly redefines them.
