## MetroPT PdM Framework

General-purpose anomaly and early-warning pipeline for the MetroPT‑3 tram maintenance dataset.  
The project ingests raw MetroPT telemetry, engineers rolling statistical features, trains an unsupervised detector (currently Isolation Forest), and evaluates alarms in the context of the 21 documented maintenance windows. The goal is to inspect how well an unsupervised model can warn about failures within the configured pre-maintenance horizon, while keeping the pipeline reusable for other detectors (e.g., autoencoders).

### Dataset
- `datasets/MetroPT3.csv`: 16 sensor channels (pressures, temperatures, currents, actuator states) with a timestamp column.
- `DEFAULT_METROPT_WINDOWS` in `pipeline_runner.py` encodes 21 maintenance intervals from Davari et al. (2021). Each window is treated as the ground-truth failure period; the `PRE_MAINTENANCE_MINUTES` before a window are labelled as “pre-maintenance”.

### Pipeline Overview
1. **Load & clean** – `data_utils.load_csv` removes “Unnamed” columns, infers the timestamp column, and sorts chronologically.
2. **Feature selection** – `select_numeric_features` keeps all numeric sensors and orders them by domain preference when available.
3. **Rolling aggregation** – `build_rolling_features` computes mean/median/std/skew/min/max over a configurable window (`ROLLING_WINDOW`, default `60s`) to produce the model matrix.
4. **Detector training** – supports two regimes: a single global model (`EXPERIMENT_MODE="single"`) trained on the first `TRAIN_FRAC` minutes, or a sequence of per‑maintenance models (`"per_maint"`) trained on the initial baseline plus a short post‑maintenance interval for each cycle. The current implementation uses Isolation Forest (`detector_model.py`), but the pipeline structure is detector‑agnostic.
5. **Maintenance context** – `build_operation_phase` encodes states (0 normal, 1 pre‑maintenance, 2 maintenance). `maintenance_risk` is the rolling average of extreme‑point exceedance (`risk_score >= RISK_EXCEEDANCE_QUANTILE`) over `RISK_WINDOW_MINUTES` minutes and serves as the early‑warning signal.
6. **Risk threshold search** – `metrics_point.evaluate_risk_thresholds` tries a grid (`RISK_EVAL_GRID_SPEC`) and reports precision/recall/F1 along with TP/FP/FN counts for alarms versus maintenance windows.
7. **Outputs** – `datasets/metropt3_features.csv` (opt.) with the engineered rolling stats, `datasets/metropt3_predictions.csv` (opt.) with scores/risk/phase, `<mode>_metropt3_raw.png` showing the risk timeline with failures, plus console `[INFO]` model settings and `[RISK]` summaries.

### Requirements
```
python >= 3.9
numpy
pandas
scikit-learn
matplotlib
```

### Usage
1. Place `MetroPT3.csv` in `datasets/` (or update `INPUT_PATH` in `pipeline_runner.py`).
2. Create a virtual environment and install dependencies, e.g.:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt  # or pip install numpy pandas scikit-learn matplotlib
   ```
3. Run the helper script:
   ```bash
   python pipeline_runner.py
   ```
   The script will emit `[INFO]`, `[METRIC]`, and `[RISK]` summaries, produce `datasets/metropt3_predictions.csv`, and save the timeline plot to `<mode>_metropt3_raw.png`.

By default `EXPERIMENT_MODE` in `pipeline_runner.py` is set to `"single"` (one global model). Set it to `"per_maint"` to enable per‑maintenance models, where each cycle is trained on the initial baseline plus a post‑maintenance training interval.

Command-line arguments are not required, but you can tweak configuration constants at the top of `pipeline_runner.py` (paths, rolling windows, training window, maintenance windows, labelling options).

### Key Files
- `pipeline_runner.py` – main workflow runner (loading → features → model → risk → plotting).
- `data_utils.py` – CSV ingestion, feature engineering, maintenance-window parsing.
- `detector_model.py` – detector wrapper (Isolation Forest today, extensible for autoencoders).
- `metrics_point.py` – point‑wise and risk‑grid evaluation.
- `metrics_event.py` – event‑level evaluation (TTD, FAR, FAA, MTIA, PR‑LT, etc.).
- `plotting.py` – visualisation of risk states, training cutoff, and maintenance windows.

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
