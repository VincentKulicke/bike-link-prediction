# Runtime and cost

What each model costs to train and to run, measured under one protocol.

## Hardware

```
GPU     NVIDIA GeForce RTX 4070 Laptop, 8.6 GB VRAM
CPU     Intel Core i9-13900HX, 24 cores / 32 threads
RAM     31.7 GB
OS      Windows 11
Stack   Python 3.12.8 · PyTorch 2.12.0+cu126 · CUDA 12.6
```

All numbers below come from this machine. On a datacentre GPU the ratios would
shift: GraphMixer's per-event Python loops barely benefit from faster hardware,
while the hybrid's tensor operations do.

## Controlled measurement (best config per model, 5 seeds, medians)

| Model | Train total | Train / epoch | Inference | µs / pair | Peak GPU | Parameters |
|---|---|---|---|---|---|---|
| **Hybrid CNN** | **21.7 s** | **1.45 s** | **0.05 s** | **0.73** | **331 MB** | 151,682 |
| **Hybrid GRU** | 29.6 s | 1.98 s | 0.17 s | 2.37 | 706 MB | 152,194 |
| LSTM | 53.7 s | 5.37 s | 0.34 s | 4.75 | 2,501 MB | 17,217 |
| GraphMixer | 89.7 s | 4.48 s | 3.18 s | 43.76 | 606 MB | 60,857 |

Inference = scoring all 72,570 test candidates. For GraphMixer the CSV write is
included (its export function does both), so that number is an upper bound; the
gap to the hybrid is far larger than any I/O could explain.

## The main result

**The hybrid is Pareto-dominant: more accurate *and* cheaper.**

| | Hybrid GRU | GraphMixer | Factor |
|---|---|---|---|
| Test AP | **0.923** | 0.701 | +0.22 |
| Inference | **0.17 s** | 3.18 s | **19× faster** |
| Train / epoch | **1.98 s** | 4.48 s | 2.3× faster |

There is no accuracy-for-speed trade-off to discuss here — one model wins on both
axes. The same holds on the count task against the LSTM (MSE 0.082 vs. 0.116 at
2× the inference speed).

## Two observations worth a sentence on the slide

**Parameter count says little about cost.** The LSTM has the fewest parameters
(17k, a ninth of the hybrid) but the largest memory footprint by far (2.5 GB vs.
706 MB). It materialises 48 timesteps × 64 hidden units per batch element;
the hybrid's windows are only 12 steps long. Model size and memory cost decouple.

**The CNN encoder is the cheaper half of an otherwise identical model.** Same
parameter count as the GRU variant (151,682 vs. 152,194), same accuracy
(AP 0.9226 vs. 0.9233, 1.3 σ), but 27 % less training time, 3× faster inference
and half the memory. A CNN processes the timesteps in parallel where the GRU
walks them sequentially. If inference cost ever mattered, this is the variant to
ship.

## Why medians, not means

A single run on this laptop can be badly wrong. Two examples from these very
measurements:

| | median | mean | worst single run |
|---|---|---|---|
| Hybrid GRU | 29.6 s | 94.6 s | **355.4 s** (12×) |
| GraphMixer | 89.7 s | 98.8 s | 136.2 s |

Both outliers landed in the last seed. Inference time was unaffected in the same
run (0.18 s vs. 0.16–0.17 s), so this is not thermal throttling of the GPU but
something taking CPU away from the batch sampling. Reporting means would have
overstated hybrid training by a factor of 3.2.

The interquartile range in `runtime_summary.csv` shows how tight the rest is:
±0.02 s/epoch for the hybrid GRU across five seeds.

## The grid timings cannot be used for this

`grid_*.csv` also records a `sec` per config, but those numbers are confounded.
Runtime drifts upward over a long grid session — on the hybrid GRU grid the
correlation between run order and runtime is **0.87**. Since no model uses early
stopping and every run trains a fixed number of epochs, no hyperparameter can
change compute cost; the apparent 2× "learning-rate effect" is the machine
slowing down, because `lr` was the outer loop in every grid (confounding 0.94).

After controlling for run order only one real driver survives:

| Model | Parameter | raw ratio | within-block | confounded with order |
|---|---|---|---|---|
| Hybrid GRU | lr | 1.99× | not estimable | 0.94 |
| Hybrid GRU | **hidden** | 1.32× | **1.33×** | 0.31 |
| Hybrid GRU | λ_count | 1.02× | 1.02× | 0.10 |
| GraphMixer | mixer_layers | 1.24× | 1.25× | 0.10 |

`hidden` costs real time, `λ_count` and `kernel_size` do not — as expected, since
neither changes the amount of computation.

This is also why phase B interleaves the models inside each seed instead of
running them in blocks: otherwise the drift would be handed to whichever model
goes last.

## Reproduce

```
python ablation/runtime_analysis.py --phase a          # grid timings + drift diagnostics
python ablation/runtime_analysis.py --phase b --seeds 5 # controlled measurement (~35 min)
```

Raw: `runtime_controlled.csv` · summaries: `runtime_summary.csv`,
`runtime_grid_summary.csv`, `runtime_hyperparams.csv`
