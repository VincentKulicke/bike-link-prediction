# Model comparison

Available predictions: GraphMixer, Hybrid, LSTM

## Comparison 1 – binary (link yes/no)

| Model | Split | AUC | AP | F1 | Accuracy |
|---|---|---|---|---|---|
| Hybrid | val | 0.985 | 0.916 | 0.855 | 0.951 |
| Hybrid | test | 0.985 | 0.923 | 0.859 | 0.952 |
| GraphMixer | val | 0.909 | 0.656 | 0.512 | 0.704 |
| GraphMixer | test | 0.909 | 0.653 | 0.515 | 0.708 |

## Comparison 2 – count (number of rides)

| Model | Split | MSE | MAE | RMSE |
|---|---|---|---|---|
| Hybrid | val | 0.091 | 0.099 | 0.302 |
| Hybrid | test | 0.082 | 0.092 | 0.287 |
| LSTM | val | 0.257 | 0.194 | 0.507 |
| LSTM | test | 0.239 | 0.187 | 0.489 |
