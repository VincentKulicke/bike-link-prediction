# Model comparison

Available predictions: GraphMixer, Hybrid, LSTM

> Single run per model (seed 42). For error bars see `ablation/results/seeds_comparison.md` — across 5 seeds the hybrid sits at AP 0.923 ± 0.001 and GraphMixer at 0.596 ± 0.023, so GraphMixer in particular varies noticeably from seed to seed.

## Comparison 1 – binary (link yes/no)

| Model | Split | AUC | AP | F1 | Accuracy |
|---|---|---|---|---|---|
| Hybrid | val | 0.985 | 0.916 | 0.855 | 0.951 |
| Hybrid | test | 0.985 | 0.923 | 0.859 | 0.952 |
| GraphMixer | val | 0.894 | 0.555 | 0.545 | 0.749 |
| GraphMixer | test | 0.895 | 0.563 | 0.554 | 0.758 |

## Comparison 2 – count (number of rides)

| Model | Split | MSE | MAE | RMSE |
|---|---|---|---|---|
| Hybrid | val | 0.091 | 0.099 | 0.302 |
| Hybrid | test | 0.082 | 0.092 | 0.287 |
| LSTM | val | 0.257 | 0.194 | 0.507 |
| LSTM | test | 0.239 | 0.187 | 0.489 |
