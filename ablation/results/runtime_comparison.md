# Runtime and cost

What each model costs to train and to run, measured under one protocol.
Configs are the winners of the final grid search, read from
`hpo_final_*.csv` rather than hard-coded.

## Hardware

```
GPU     NVIDIA GeForce RTX 4070 Laptop, 8.6 GB VRAM (8188 MiB)
CPU     Intel Core i9-13900HX, 24 cores / 32 threads
RAM     31.7 GB
OS      Windows 11
Stack   Python 3.12.8, PyTorch 2.12.0+cu126, CUDA 12.6
```

Ratios are hardware-dependent: GraphMixer's per-event Python loops
barely benefit from a faster GPU, the hybrid's tensor operations do.

## Controlled measurement (5 seeds, medians)

| Model | Training | s/epoch | Inference | us/pair | Peak memory | Params |
|---|---|---|---|---|---|---|
| Hybrid CNN | 99.0 s | 3.30 | 0.228 s | 3.15 | 1,171 MB | 151,682 |
| Hybrid GRU | 158.8 s | 5.29 | 0.410 s | 5.64 | 1,938 MB | 599,298 |
| LSTM | 158.5 s | 5.28 | 0.481 s | 6.62 | 9,845 MB | 50,497 |
| GraphMixer | 69.8 s | 2.33 | 3.724 s | 51.31 | 1,585 MB | 115,085 |

Inference = forward pass over the test candidates, excluding file I/O.

## The main result

The hybrid infers **9.1x faster** than GraphMixer (0.410 s vs. 3.724 s) at AP 0.9238 vs. 0.9019.

GraphMixer, however, now **trains faster** (70 s vs. 159 s), which reverses the
earlier picture. Since training is one-off and inference is the
running cost, the hybrid remains the cheaper choice in operation --
but it is a trade-off now, not Pareto dominance.

## Two observations worth a slide

**The CNN variant is the efficiency winner.** Fastest inference in the
field, lowest memory among the hybrids, a quarter of the GRU's
parameters -- at statistically indistinguishable accuracy (1.4 sigma).

**Parameter count says nothing about cost.** The LSTM has the fewest
parameters and by far the largest memory peak.

> **The best LSTM config does not fit in VRAM.** Its peak of 9,845 MB exceeds the card's 8,188 MiB.
> It only runs because the Windows driver spills CUDA allocations to
> system memory; on a card without that fallback it would raise OOM.

## Why medians, not means

Single measurements on this machine occasionally spike by an order of
magnitude (transient thermal or driver effects). Models are measured
interleaved within each seed, not in blocks, so drift cannot favour
whichever model runs last.

## The grid timings cannot be used for this

In `grid_hybrid_gru.csv` runtime correlates 0.87 with run order. Since
no model uses early stopping there, no hyperparameter can change
runtime causally -- the apparent 2x learning-rate effect is thermal
drift, because lr was the outer loop in every grid.

## Reproduce

```bash
python ablation/runtime_analysis.py --phase a   # grid diagnostics
python ablation/runtime_analysis.py --phase b   # controlled measurement
```

Measured in ~53 min on cuda.
Raw data: `runtime_controlled.csv`, aggregated: `runtime_summary.csv`
