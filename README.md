## MetroPT PdM Framework

General-purpose anomaly and early-warning pipeline for the MetroPT‑3 tram maintenance dataset.  
The project ingests raw MetroPT telemetry, engineers rolling statistical features, trains an unsupervised detector, and evaluates alarms in the context of the 21 documented maintenance windows. The goal is to inspect how well an unsupervised model can warn about failures within the configured pre-maintenance horizon, while keeping the pipeline reusable for different detector backends (e.g., Isolation Forest now, autoencoders later).

### Dataset
- `datasets/MetroPT3.csv`: 16 sensor channels (pressures, temperatures, currents, actuator states) with a timestamp column.
- `DEFAULT_METROPT_WINDOWS` in `main.py` encodes 21 maintenance intervals from Davari et al. (2021). Each window is treated as the ground-truth failure period; the `PRE_MAINTENANCE_MINUTES` before a window are labelled as “pre-maintenance”.

### Pipeline Overview
1. **Load & clean** – `data_utils.load_csv` removes “Unnamed” columns, infers the timestamp column, and sorts chronologically.
2. **Feature selection** – `select_numeric_features` keeps all numeric sensors and orders them by domain preference when available.
3. **Rolling aggregation** – `build_rolling_features` computes mean/median/std/skew/min/max over a configurable window (`ROLLING_WINDOW`, default `60s`) to produce the model matrix.
4. **Detector training** – supports two regimes: a single global model (`EXPERIMENT_MODE="single"`) trained on the first `TRAIN_FRAC` minutes, or a sequence of per‑maintenance models (`"per_maint"`) trained on the initial baseline plus a short post‑maintenance interval for each cycle. The current implementation uses Isolation Forest via `detectors/iforest_detector.py`, but the pipeline is detector‑agnostic (`DETECTOR_TYPE`).
5. **Maintenance context** – `build_operation_phase` encodes states (0 normal, 1 pre‑maintenance, 2 maintenance). `maintenance_risk` is the rolling average of extreme‑point exceedance (`risk_score >= RISK_EXCEEDANCE_QUANTILE`) over `RISK_WINDOW_MINUTES` minutes and serves as the early‑warning signal.
6. **Risk threshold search** – `metrics_point.evaluate_risk_thresholds` tries a grid (`RISK_EVAL_GRID_SPEC`) and reports precision/recall/F1 along with TP/FP/FN counts for alarms versus maintenance windows.
7. **Outputs** – `datasets/metropt3_features.csv` (opt.) with the engineered rolling stats, `datasets/metropt3_predictions.csv` (opt.) with scores/risk/phase, `<mode>_metropt3_raw.png` showing the risk timeline with failures, `<mode>_lead_time_distribution.png` for the lead-time histogram, `<mode>_pr_vs_lead_time.png` for precision/recall vs lead time, plus console `[INFO]` model settings and `[RISK]` summaries.

### Requirements
```
python >= 3.9
numpy
pandas
scikit-learn
matplotlib
```
Optional (only when using `DETECTOR_TYPE="autoencoder"`):
```
torch
```
Optional (only when using imported NiaNetVAE pretrained models):
```
pip install -e ../NiaNetVAE
```

### Usage
1. Place `MetroPT3.csv` in `datasets/` (or update `INPUT_PATH` in `main.py`).
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
   The script will emit `[INFO]`, `[METRIC]`, and `[RISK]` summaries, produce `datasets/metropt3_predictions.csv`, and save plots to `<mode>_metropt3_raw.png`, `<mode>_lead_time_distribution.png`, and `<mode>_pr_vs_lead_time.png`.

By default `EXPERIMENT_MODE` in `main.py` is set to `"single"` (one global model). Set it to `"per_maint"` to enable per‑maintenance models, where each cycle is trained on the initial baseline plus a post‑maintenance training interval.

Command-line arguments are not required, but you can tweak configuration constants at the top of `main.py` (paths, rolling windows, training window, maintenance windows, labelling options).

### Imported NiaNetVAE per-cycle models (per_maint)
To consume pretrained artifacts exported from `NiaNetVAE`:
1. Set `EXPERIMENT_MODE="per_maint"`.
2. Set `PER_MAINT_USE_IMPORTED_MODELS=True`.
3. Set `PER_MAINT_MODEL_MANIFEST_PATH` to exported `cycle_manifest.json`.
4. Keep `PER_MAINT_MODEL_STRICT=True` for production (fails fast on missing model artifacts).

In this mode, metropt resolves cycle models from the manifest (including alias cycles) and uses them in fixed inference mode (no model weight updates in this repository).

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

### Key Files
- `main.py` – main workflow runner (loading → features → model → risk → plotting).
- `data_utils.py` – CSV ingestion, feature engineering, maintenance-window parsing.
- `detectors/` – detector backends and shared post‑processing (`iforest_detector.py`, `autoencoder_detector.py`, `postprocess.py`).
- `metrics_point.py` – point‑wise and risk‑grid evaluation.
- `metrics_event.py` – event‑level evaluation (TTD, FAR, FAA, MTIA, PR‑LT, etc.).
- `plotting.py` – visualisation of risk states, lead-time distribution, and precision/recall vs lead time.

### Output Interpretation
- `operation_phase`: 0 normal, 1 within `PRE_MAINTENANCE_MINUTES` before a known maintenance start, 2 during maintenance.
- `exceedance`: 1 when `risk_score >= RISK_EXCEEDANCE_QUANTILE`, otherwise 0.
- `maintenance_risk`: fraction of the last `RISK_WINDOW_MINUTES` with extreme risk points (`risk_score >= RISK_EXCEEDANCE_QUANTILE`); higher values indicate sustained alarms and are compared against the risk threshold grid.
- `predicted_phase`: 1 when `maintenance_risk >= θ` (alarm state), otherwise 0; `θ` is the best threshold from the risk grid.
- `[METRIC]` console block: point-wise performance of `is_anomaly` when predicting pre-maintenance horizon (phase 1) vs normal operation (phase 0). Maintenance rows (phase 2) are ignored for these metrics.
- `[RISK]` / `[RISK-PERMAINT]` console blocks: event-level performance of the rolling risk alarm (TP/FP/FN refer to maintenance events and alarm intervals, not individual rows).

### Large Files
`datasets/MetroPT3.csv` and `datasets/metropt3_predictions.csv` can exceed GitHub’s 100 MB limit and should remain untracked (add to `.gitignore` or use Git LFS if they must be shared).

### Roadmap
- Parameter sweeps to balance precision vs recall (e.g., `TRAIN_FRAC`, `RISK_WINDOW_MINUTES`).
- Optional CLI arguments for key hyperparameters.
- Support for additional data formats (Parquet, MAT) if needed.
