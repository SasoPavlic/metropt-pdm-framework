## MetroPT PdM Framework

General-purpose anomaly and early-warning pipeline for the MetroPT‑3 tram maintenance dataset.  
The project ingests raw MetroPT telemetry, engineers rolling statistical features, trains an unsupervised detector, and evaluates alarms in the context of the 21 documented maintenance windows. The goal is to inspect how well an unsupervised model can warn about failures within the configured pre-maintenance horizon, while keeping the pipeline reusable for different detector backends.

### Dataset
- `datasets/MetroPT3.csv`: 16 sensor channels (pressures, temperatures, currents, actuator states) with a timestamp column.
- `DEFAULT_METROPT_WINDOWS` in `metropt_pdm_framework.pipeline` encodes 21 maintenance intervals from Davari et al. (2021). Each window is treated as the ground-truth failure period; the `PRE_MAINTENANCE_MINUTES` before a window are labelled as “pre-maintenance”.

### Pipeline Overview
1. **Load & clean** – `metropt_pdm_framework.data.preprocessing.load_csv` removes “Unnamed” columns, infers the timestamp column, and sorts chronologically.
2. **Feature selection** – `select_numeric_features` keeps all numeric sensors and orders them by domain preference when available.
3. **Rolling aggregation** – `build_rolling_features` computes mean/median/std/skew/min/max over a configurable window (`ROLLING_WINDOW`, default `60s`) to produce the model matrix.
4. **Detector training** – supports two regimes: a single global model (`EXPERIMENT_MODE="single"`) trained on the first `TRAIN_FRAC` minutes, or a sequence of per‑maintenance models (`"per_maint"`) trained on the initial baseline plus a short post‑maintenance interval for each cycle. Local single-mode detectors use `DETECTOR_TYPE` (`iforest`, `recurrent-vae`, `recurrent-sae`); imported per-maint evaluation is controlled by `PER_MAINT_USE_IMPORTED_MODELS`.
5. **Maintenance context** – `build_operation_phase` encodes states (0 normal, 1 pre‑maintenance, 2 maintenance). `maintenance_risk` is the rolling average of extreme‑point exceedance (`risk_score >= RISK_EXCEEDANCE_QUANTILE`) over `RISK_WINDOW_MINUTES` minutes and serves as the early‑warning signal.
6. **Risk threshold search** – `metropt_pdm_framework.metrics.point.evaluate_risk_thresholds` tries a grid (`RISK_EVAL_GRID_SPEC` plus strict probes from `RISK_EVAL_EXTRA_THRESHOLDS`) and reports precision/recall/F1 along with TP/FP/FN counts for alarms versus maintenance windows.
7. **Outputs** – `datasets/metropt3_features.csv` (opt.) with the engineered rolling stats, plus unified runtime artifacts in `artifacts/<detector-group>/<mode>/`:
   - `logs/run.log`
   - `predictions/metropt3_predictions.csv` (opt.)
   - `predictions/theta_sweep_maintenance_risk.csv`, `predictions/theta_sweep_summary.csv`
   - `predictions/alarm_islands.csv`, `predictions/alarm_island_summary.csv`
   - `plots/metropt3_raw.png`, `plots/lead_time_distribution.png`, `plots/pr_vs_lead_time.png`
   together with console `[INFO]` model settings and `[RISK]` summaries.

### Requirements
```
python >= 3.9
numpy
pandas
scikit-learn
matplotlib
```
Optional (only when using PyTorch recurrent detectors such as `recurrent-vae` or `recurrent-sae`, or imported recurrent artifacts):
```
torch
```
Optional (only when using imported NiaNetVAE pretrained models):
```
pip install -e ../NiaNetVAE
```

### Usage
1. Place `MetroPT3.csv` in `datasets/` (or update `INPUT_PATH` in `metropt_pdm_framework/pipeline.py`).
2. Create a virtual environment and install dependencies, e.g.:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt  # or pip install numpy pandas scikit-learn matplotlib
   ```
3. Run the helper script:
   ```bash
   python main.py
   ```
   If installed as a Poetry package, the equivalent console command is:
   ```bash
   metropt-pdm
   ```
   The script will emit `[INFO]`, `[METRIC]`, and `[RISK]` summaries and write outputs to `artifacts/<detector-group>/<mode>/` (logs, predictions, and plots).

By default `EXPERIMENT_MODE` in `metropt_pdm_framework/pipeline.py` controls the selected regime. Set it to `"per_maint"` to enable per‑maintenance models, where each cycle is trained on the initial baseline plus a post‑maintenance training interval.

Command-line arguments are not required, but you can tweak configuration constants in `metropt_pdm_framework/pipeline.py` (paths, rolling windows, training window, maintenance windows, labelling options).

### Local recurrent SAE/VAE detectors (single)
For local recurrent reconstruction baselines inside the current framework, set:
1. `EXPERIMENT_MODE="single"`.
2. `DETECTOR_TYPE="recurrent-vae"` or `DETECTOR_TYPE="recurrent-sae"`.

Both detectors use the same combined analog+digital rolling feature matrix and a shared LSTM sequence setup: sequence length `200`, stride `1`, hidden size `64`, two recurrent layers, latent dimension `32`, Adam, learning rate `0.003`, batch size `64`, and training-split standardization. The VAE uses stochastic latent sampling with KL beta `0.1`; the SAE uses deterministic sparse latent activations with KL sparsity beta/rho defaults in `metropt_pdm_framework/pipeline.py`.

### Imported per-cycle recurrent artifacts (per_maint)
To consume pretrained per-cycle recurrent artifacts exported from `NiaNetVAE`:
1. Set `EXPERIMENT_MODE="per_maint"`.
2. Set `PER_MAINT_USE_IMPORTED_MODELS=True`.
3. Set `PER_MAINT_MODEL_MANIFEST_PATH` to exported `cycle_manifest.json`.
4. Keep `PER_MAINT_MODEL_STRICT=True` for production (fails fast on missing model artifacts).

In this mode, metropt resolves cycle models from the manifest (including alias cycles) and uses them in fixed inference mode (no model weight updates in this repository). `DETECTOR_TYPE` is ignored for imported per-maint evaluation.

#### Recommended local setup (Windows dedicated venv)
Use a dedicated Python 3.11 virtual environment for metropt imported-mode evaluation to avoid changing your existing interpreter:

```powershell
cd <path-to>\metropt-pdm-framework
py -3.11 -m venv .venv-metropt
.\.venv-metropt\Scripts\Activate.ps1
pip install numpy pandas scikit-learn matplotlib torch lightning
pip install -e ..\NiaNetVAE
```

Then in PyCharm:
- create a dedicated run configuration for imported mode,
- select interpreter `.\.venv-metropt\Scripts\python.exe`,
- keep defaults in code unchanged, and set imported-mode flags only in that run profile.

#### Manifest ownership and sync workflow
1. Generate `cycle_manifest.json` on HPC using `python -m nianetvae.tools.generate_cycle_manifest`.
2. Copy from HPC to laptop:
   - `logs/per_maint_models/MetroPT/cycle_XX/*`
   - `logs/per_maint_models/MetroPT/cycle_manifest.json`
3. Point `PER_MAINT_MODEL_MANIFEST_PATH` to the local copied manifest.

The manifest contains explicit `trained` / `alias` / `missing` cycle states; strict mode requires all needed cycles to resolve.

### Drift / non-stationarity analysis
To generate CPU-only evidence of MetroPT target-event cycle drift, run:
```bash
python -m pdm_eval.tools.analyze_metropt_drift --input datasets/MetroPT3.csv --output-root artifacts/analysis/drift
```
or, when installed through Poetry:
```bash
metropt-drift --input datasets/MetroPT3.csv --output-root artifacts/analysis/drift
```

This standalone analysis uses the same rolling feature matrix as the detectors, groups rows by target maintenance event cycles, computes median feature-wise Wasserstein distances using a CPU-friendly quantile-grid estimate, and writes CSV tables plus a heatmap/PCA plot under `artifacts/analysis/drift/`. It is dataset drift evidence for the paper methodology, not a detector benchmark and not a replacement for event-level alarm evaluation.

### Key Files
- `main.py` – thin root runner for the package CLI.
- `metropt_pdm_framework/pipeline.py` – main workflow runner (loading → features → model → risk → plotting).
- `metropt_pdm_framework/manifest.py` – imported NiaNetVAE cycle manifest validation and cycle artifact resolution.
- `metropt_pdm_framework/data/preprocessing.py` – CSV ingestion, feature engineering, maintenance-window parsing.
- `metropt_pdm_framework/detectors/` – detector backends and shared post‑processing (`iforest_detector.py`, `recurrent_autoencoder_detector.py`, `imported_recurrent_autoencoder_detector.py`, `postprocess.py`).
- `metropt_pdm_framework/metrics/point.py` – point‑wise and risk‑grid evaluation.
- `metropt_pdm_framework/metrics/alarm_islands.py` – selected-threshold alarm block diagnostics.
- `metropt_pdm_framework/metrics/event.py` – event‑level evaluation (TTD, FAR, FAA, MTIA, PR‑LT, etc.).
- `metropt_pdm_framework/visualization/plots.py` – visualisation of risk states, lead-time distribution, and precision/recall vs lead time.
- `metropt_pdm_framework/tools/plot_per_maint_schedule.py` – standalone per-maintenance schedule visualizer (`metropt-plot-schedule` when installed).

### Output Interpretation
- `operation_phase`: 0 normal, 1 within `PRE_MAINTENANCE_MINUTES` before a known maintenance start, 2 during maintenance.
- `exceedance`: 1 when `risk_score >= RISK_EXCEEDANCE_QUANTILE`, otherwise 0.
- `maintenance_risk`: fraction of the last `RISK_WINDOW_MINUTES` with extreme risk points (`risk_score >= RISK_EXCEEDANCE_QUANTILE`); higher values indicate sustained alarms and are compared against the risk threshold grid.
- `predicted_phase`: 1 when `maintenance_risk >= θ` (alarm state), otherwise 0; `θ` is the best threshold from the risk grid.
- `[METRIC]` console block: point-wise performance of `is_anomaly` when predicting pre-maintenance horizon (phase 1) vs normal operation (phase 0). Maintenance rows (phase 2) are ignored for these metrics.
- `[RISK]` / `[RISK-PERMAINT]` console blocks: event-level performance of the rolling risk alarm (TP/FP/FN refer to maintenance events and alarm intervals, not individual rows).
- `alarm_islands.csv`: one row per contiguous selected-threshold alarm block, including duration, point count, gap from previous block, and whether it overlaps the strict early-warning window.

### Large Files
`datasets/MetroPT3.csv`, prediction CSVs under `artifacts/`, and generated plots/logs under `artifacts/` can grow quickly and should remain untracked (add to `.gitignore` or use Git LFS if they must be shared).

### Roadmap
- Parameter sweeps to balance precision vs recall (e.g., `TRAIN_FRAC`, `RISK_WINDOW_MINUTES`).
- Optional CLI arguments for key hyperparameters.
- Support for additional data formats (Parquet, MAT) if needed.
