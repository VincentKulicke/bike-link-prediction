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
├── ablation/                   # grid-search HPO, encoder ablation, seed & factor experiments
│   └── results/                # comparison reports (ablation, ranking, seeds, factors)
├── compare_models.py           # collects predictions → final comparison tables
└── results/                    # comparison.md across all models
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

## Hyperparameter search and ablations

`ablation/` tunes every model with its own grid, selection on validation only,
and then compares the winners over 5 seeds. It also contains the encoder
ablation (GRU vs 1D-CNN) and the branch ablation (graph / temporal / pair).

```bash
cd ablation
bash run_hpo_final.sh          # all four grids, restarts on transient CUDA faults
python hpo_final_report.py     # -> results/hpo_final_comparison.md
python final_eval.py           # winners x 5 seeds -> results/final_eval_summary.csv
python eval_branches.py        # -> results/branches_comparison.md
python eval_ranking.py         # 1-vs-99 protocol -> results/ranking_comparison.md
python runtime_analysis.py --phase b    # -> results/runtime_comparison.md
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
