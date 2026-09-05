# Final hyperparameter search

One run, four grids. Selection on validation, numbers reported on test.

## hybrid_gru

- configs: 72, seeds each: 1
- best vs worst on test: +0.0131 (32.8 sigma)
- best: {'lr': 0.001, 'hidden': 256.0, 'ts_lookback': 48.0, 'fusion_hidden': 256.0} -> test 0.9239

## hybrid_cnn

- configs: 108, seeds each: 1
- best vs worst on test: +0.0060 (20.0 sigma)
- best: {'lr': 0.001, 'hidden': 128.0, 'ts_lookback': 48.0, 'dropout': 0.0} -> test 0.9225

## lstm

- configs: 90, seeds each: 1
- best vs worst on test: +0.0658 (50.6 sigma)
- best: {'lr': 0.001, 'hidden_dim': 64, 'lookback': 192, 'layers_dropout': '(2, 0.2)'} -> test 0.0935

## graphmixer

- configs: 46, seeds each: 2
- best vs worst on test: +0.0490 (35.0 sigma)
- best: {'lr': 0.001, 'hidden_dim': 128.0, 'num_neighbors': 40.0, 'mixer_layers': 1.0} -> test 0.9021
