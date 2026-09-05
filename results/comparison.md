# Model comparison

Mean ± std over 5 seeds (42–46) of each model's grid winner. Configurations are read from `hpo_final_*.csv`, so they cannot drift from the search that produced them. Selection on validation, everything below is test.

## Winning configurations

- **Hybrid GRU**: `{'lr': 0.001, 'hidden': 256.0, 'ts_lookback': 48.0, 'fusion_hidden': 256.0}`
- **Hybrid CNN**: `{'lr': 0.001, 'hidden': 128.0, 'ts_lookback': 48.0, 'dropout': 0.0}`
- **LSTM**: `{'lr': 0.001, 'hidden_dim': 64, 'lookback': 192, 'layers_dropout': '(2, 0.2)'}`
- **GraphMixer**: `{'lr': 0.001, 'hidden_dim': 128.0, 'num_neighbors': 40.0, 'mixer_layers': 1.0}`

## Binary (link yes/no)

| Model | AP | AUC | F1 |
|---|---|---|---|
| Hybrid GRU | 0.9238 ± 0.0004 | 0.9851 ± 0.0001 | 0.8596 ± 0.0011 |
| Hybrid CNN | 0.9231 ± 0.0003 | 0.9851 ± 0.0001 | 0.8600 ± 0.0005 |
| GraphMixer | 0.9019 ± 0.0014 | 0.9831 ± 0.0002 | 0.8469 ± 0.0011 |

## Count (number of rides)

| Model | MSE |
|---|---|
| Hybrid GRU | 0.0826 ± 0.0004 |
| Hybrid CNN | 0.0831 ± 0.0004 |
| LSTM | 0.0954 ± 0.0013 |

## Pairwise, in pooled seed sigmas

| Comparison | Difference | sigma | |
|---|---|---|---|
| Hybrid GRU vs GraphMixer | +0.0219 AP | 15.0 | significant |
| Hybrid GRU vs Hybrid CNN | +0.0007 AP | 1.4 | **not significant** |
| LSTM vs Hybrid GRU | +0.0128 MSE | 9.4 | significant |

The hybrid leads on both tasks, but by far less than earlier reports suggested: fixing GraphMixer's training distribution moved it from AP 0.70 to 0.90, and tuning the LSTM's never-searched `lookback` moved it from MSE 0.116 to 0.095. GRU and CNN are statistically indistinguishable.
