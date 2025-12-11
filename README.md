## MetroPT-IForest Early Warning

Single-series anomaly exploration for the MetroPT‑3 tram maintenance dataset.  
This project ingests the raw MetroPT telemetry, engineers rolling statistical features, trains an IsolationForest on the earliest data, and evaluates alarms in the context of the 21 documented maintenance windows. The goal is to inspect how well an unsupervised model can warn about failures up to two hours in advance.

### Dataset
- `MetroPT3.csv` (not stored in Git): 16 sensor channels (pressures, temperatures, currents, actuator states) with a timestamp column.
- `DEFAULT_METROPT_WINDOWS` in `anomaly_iforest_helper.py` encodes 21 maintenance intervals from Davari et al. (2021). Each window is treated as the ground-truth failure period; the two hours before a window are labelled as “pre-maintenance”.

### Pipeline Overview
1. **Load & clean** – `iforest_data.load_csv` removes “Unnamed” columns, infers the timestamp column, sorts chronologically, and optionally downsamples (Pandas resample/median).
2. **Feature selection** – `select_numeric_features` keeps numeric sensors, drops quasi-binary channels, and orders them by domain preference; `top_k_by_variance` retains the `MAX_BASE_FEATURES` most variant sensors.
3. **Rolling aggregation** – `build_rolling_features` computes mean/median/std/skew/min/max over a configurable window (`ROLLING_WINDOW`, default `600s`) to produce the model matrix.
4. **Isolation Forest** – supports two regimes: a single global model (`EXPERIMENT_MODE="single"`) trained on the first `TRAIN_FRAC` slice, or a sequence of per-inter-maintenance models (`"per_maint"`) trained on accumulated normal data up to the previous maintenance window. Both standardise features, smooth anomaly scores (optional LPF), and label points using `Q3 + 3·IQR`.
5. **Maintenance context** – `build_operation_phase` encodes states (0 normal, 1 pre-maintenance, 2 maintenance). `maintenance_risk` is the rolling average of anomaly exceedances over `RISK_WINDOW_MINUTES` minutes and serves as the early-warning signal.
6. **Risk threshold search** – `iforest_metrics.evaluate_risk_thresholds` tries a grid (`RISK_EVAL_GRID_SPEC`) and reports precision/recall/F1 along with TP/FP/FN counts for alarms versus maintenance windows.
7. **Outputs** – `metropt3_iforest_features.csv` (opt.) with the engineered rolling stats, `metropt3_iforest_pred.csv` (opt.) with scores/risk/phase, `metropt3_iforest_raw.png` showing the risk timeline with failures, plus console `[INFO]` model settings and `[RISK]` summaries.

### Requirements
```
python >= 3.9
numpy
pandas
scikit-learn
matplotlib
```

### Usage
1. Place `MetroPT3.csv` in the project root (or update `INPUT_PATH` in `anomaly_iforest_helper.py`).
2. Create a virtual environment and install dependencies, e.g.:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt  # or pip install numpy pandas scikit-learn matplotlib
   ```
3. Run the helper script:
   ```bash
   python anomaly_iforest_helper.py
   ```
   The script will emit `[INFO]`, `[METRIC]`, and `[RISK]` summaries, produce `metropt3_iforest_pred.csv`, and save the timeline plot to `metropt3_iforest_raw.png`.

By default `EXPERIMENT_MODE` in `anomaly_iforest_helper.py` is set to `"single"` (one global model). Set it to `"per_maint"` to enable per-inter-maintenance models, where each period between failures is served by a model trained on accumulated normal data up to the previous maintenance.

Command-line arguments are not required, but you can tweak configuration constants at the top of `anomaly_iforest_helper.py` (paths, rolling windows, training fraction, maintenance windows, labelling options).

### Key Files
- `anomaly_iforest_helper.py` – main workflow runner (loading → features → model → risk → plotting).
- `iforest_data.py` – CSV ingestion, resampling, feature engineering, maintenance-window parsing.
- `iforest_model.py` – IsolationForest model wrapper and low-pass filtering.
- `iforest_metrics.py` – event-level maintenance risk evaluation.
- `iforest_plotting.py` – visualisation of risk states, training cutoff, and maintenance windows.

### Output Interpretation
- `operation_phase`: 0 normal, 1 within 2 h before a known maintenance start, 2 during maintenance.
- `maintenance_risk`: fraction of the last `RISK_WINDOW_MINUTES` with anomaly scores above the learned threshold; higher values indicate sustained alarms and are compared against the risk threshold grid.
- `[METRIC]` console block: point-wise performance of `is_anomaly` when predicting pre-maintenance horizon (phase 1) vs normal operation (phase 0). Maintenance rows (phase 2) are ignored for these metrics.
- `[RISK]` / `[RISK-PERMAINT]` console blocks: event-level performance of the rolling risk alarm (TP/FP/FN refer to maintenance events and alarm intervals, not individual rows).

### Large Files
`MetroPT3.csv` and `metropt3_iforest_pred.csv` exceed GitHub’s 100 MB limit and should remain untracked (add to `.gitignore` or use Git LFS if they must be shared).

### Roadmap
- Parameter sweeps to balance precision vs recall (e.g., `TRAIN_FRAC`, `LPF_ALPHA`, `RISK_WINDOW_MINUTES`).
- Optional CLI arguments for key hyperparameters.
- Support for additional data formats (Parquet, MAT) if needed.
