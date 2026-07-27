# Bike-Sharing Link Prediction on Hybrid Graph + Time Series

Hybrid link prediction on the NYC / Jersey City bike-sharing network: for a
station pair and a future time window, predict **whether** a ride happens
(binary) and **how many** rides (count). Combines graph structure with
continuous node time series (availability).

Big Data Praktikum, Leipzig University.

## Task & comparisons

- **Our hybrid model**: graph branch (GCN/GraphSAGE) + time-series branch
  (1D-CNN/GRU) + fusion; two outputs (binary + count).
- **Comparison 1 (binary)**: our model vs. **GraphMixer** (temporal-graph baseline) — AUC, AP.
- **Comparison 2 (count)**: our model vs. **LSTM** (time-series baseline) — MSE, MAE.
- Ground truth for the count = difference of the cumulative `num_rides` series.

## Layout

```
.
├── evaluation/
│   └── shared_eval.py          # MODEL-AGNOSTIC eval (binary + count), one GT / one split for all
├── prepared Data/              # prepared inputs for ALL models (small) + README
├── graphmixer/
│   ├── model/                  # GraphMixer (PyTorch) + local runner
│   └── prepare_hybrid_inputs.py
├── lstm/                       # LSTM baseline (count) + local runner
├── hybrid_model/               # iteration1 (ablation) + iteration2 (GraphSAGE+GRU+hurdle)
├── ablation/                   # grid-search HPO for all models + GraphSAGE+1D-CNN ablation
├── compare_models.py           # collects predictions → final comparison tables
└── docs/                       # concept (DE/EN), data analysis, methods assessment, explanations
```

## Data

The **prepared** files live in `prepared Data/` (see the README there for the
schema and conventions) and are used by every model.

**Count ground truth = super-edge `num_rides` difference** (`superedge_counts.csv`),
as required by the assignment. The LSTM uses the same super-edge series as input;
GraphMixer uses the individual, timestamped edges (`ml_citibike.*`).

The **raw data** is intentionally NOT in the repo (too large), but publicly
reproducible:

- Hybrid dataset (super-edge, time series): Zenodo DOI `10.5281/zenodo.13846868`
- Temporal graph (individual trips): Citi Bike System Data, files
  `JC-202405-citibike-tripdata.csv`, `JC-202406-citibike-tripdata.csv`
  (Jersey City / Hoboken, May + June 2024), filtered to the 232 active stations.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

## Running the models locally

Everything runs locally (CPU is enough, the dataset is small; a GPU is used if
available). Each model has a runner notebook that is executed from its own
folder, or can be started via script:

```bash
# GraphMixer (binary)
cd graphmixer/model && python train_graphmixer.py        # or run_graphmixer.ipynb
# LSTM (count)
cd lstm && python lstm_count.py                          # or run_lstm.ipynb
# Hybrid model (binary + count)
#   hybrid_model/iteration2_graphsage_gru_hurdle.ipynb
```
Quick smoke test via the configs, e.g. `GMConfig(epochs=2)` /
`LSTMConfig(epochs=2)`. Every run writes `predictions/*.csv` in the respective
model folder.

## Final comparison tables

```bash
python compare_models.py     # collects all predictions/*.csv -> results/comparison.md
```

## Ablation study (grid-search HPO)

`ablation/` tunes every model with the same search depth (grid search, selection
on validation only) and adds a GraphSAGE + 1D-CNN encoder ablation. See
`ablation/results/ablation_comparison.md`.

```bash
cd ablation && bash run_all_grids.sh     # runs all four grids, then:
python make_ablation_comparison.py       # writes ablation_comparison.md
```

## Evaluation (identical for every model)

Each model exports predictions with the columns `u, i, bin_idx` plus `score`
(binary) and/or `pred_count` (count), node IDs **canonically 0-indexed**. Then:
```python
from evaluation.shared_eval import SharedLinkEval
ev = SharedLinkEval()
ev.score_binary(pred_df, split="test")   # AUC, AP, F1, Acc
ev.score_count(pred_df,  split="test")   # MSE, MAE, RMSE
```
This guarantees identical ground truth, splits and candidate pairs across all
methods.

## Status

- [x] Data preparation + sanity check
- [x] Shared evaluation module
- [x] GraphMixer baseline (code + local runner)
- [x] LSTM baseline (count) (code + local runner)
- [x] Hybrid model GraphSAGE+GRU+hurdle (binary & count), iteration 2
- [x] Model runs + final comparison tables (`results/comparison.md`)
- [x] Ablation study: grid-search HPO + GraphSAGE+1D-CNN (`ablation/results/`)

## Sharing via GitLab

```bash
git remote add origin <YOUR-GITLAB-REPO-URL>
git push -u origin main
```
Raw data and model outputs (`predictions/`, `results/`) are excluded via
`.gitignore`; the small prepared data in `prepared Data/` is kept in the repo.
