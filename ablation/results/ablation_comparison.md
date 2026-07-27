# Ablation study – grid-search hyperparameter optimization

Every model was tuned with the **same search-depth philosophy** via grid search. Hyperparameters were selected **from the validation metric only**; the test split was computed **once** per model (final config). All runs use the same `shared_eval` protocol and seed 42.

> **Read together with `seeds_comparison.md`.** The numbers below are single runs (seed 42) and carry no error bars. Across 5 seeds two of the differences shown here turn out to be seed noise: the LSTM tuning gain and the hybrid default-vs-HPO gap. Only the GraphMixer tuning gain survives (+0.106 AP, 3.4 σ). GraphMixer is also by far the least stable model (±0.023 AP vs. ±0.0007 for the hybrid), so its single-seed numbers should be read with that spread in mind.

## Comparison 1 – binary (link yes/no), test set

| Model | Tuning | AUC | AP | F1 | Accuracy | Best config |
|---|---|---|---|---|---|---|
| GraphMixer | default | 0.895 | 0.563 | 0.554 | 0.758 | — |
| GraphMixer | **HPO** | 0.925 | 0.756 | 0.503 | 0.694 | lr0.0001_h64_L1 |
| Hybrid GraphSAGE+GRU | default | 0.985 | 0.923 | 0.859 | 0.952 | — |
| Hybrid GraphSAGE+GRU | **HPO** | 0.985 | 0.923 | 0.858 | 0.952 | gru_lr0.001_h128_lam0.5 |
| Hybrid GraphSAGE+1D-CNN | **HPO** | 0.985 | 0.922 | 0.859 | 0.952 | cnn_lr0.001_h128_lam1_k3 |

## Comparison 2 – count (number of rides), test set

| Model | Tuning | MSE | MAE | RMSE | Best config |
|---|---|---|---|---|---|
| LSTM | default | 0.239 | 0.187 | 0.489 | — |
| LSTM | **HPO** | 0.233 | 0.190 | 0.483 | lr0.0003_h32_L1 |
| Hybrid GraphSAGE+GRU | default | 0.082 | 0.092 | 0.287 | — |
| Hybrid GraphSAGE+GRU | **HPO** | 0.082 | 0.092 | 0.287 | gru_lr0.001_h128_lam0.5 |
| Hybrid GraphSAGE+1D-CNN | **HPO** | 0.082 | 0.092 | 0.287 | cnn_lr0.001_h128_lam1_k3 |

## Search spaces

| Model | Search space | # configs | Selection metric (val) |
|---|---|---|---|
| GraphMixer | lr {1e-3, 3e-4, 1e-4} × hidden {64,128,256} × mixer_layers {1,2} | 18 | AP |
| LSTM | lr {1e-3, 3e-4, 1e-4} × hidden {32,64,128} × num_layers {1,2} | 18 | MSE |
| Hybrid GRU | lr {1e-3, 3e-4, 1e-4} × hidden {32,64,128} × λ_count {0.5,1,2} | 27 | AP |
| Hybrid 1D-CNN | lr {1e-3, 3e-4, 1e-4} × hidden {32,64,128} × kernel {3,5} | 18 | AP |

## Detailed logs

Full per-config validation metrics: `grid_graphmixer.csv`, `grid_lstm.csv`, `grid_hybrid_gru.csv`, `grid_hybrid_cnn.csv` (all in `results/`).
