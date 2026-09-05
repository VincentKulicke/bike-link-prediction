# Branch ablations

Swapping the encoder (GRU <-> CNN) only shows that the *encoding* is
interchangeable. These runs remove each component by zeroing its
contribution in `HybridHurdle` - parameter count is identical across
variants, so a difference cannot come from model size.

5 variants x 5 seeds (42-46) x 2 encoders, ~28 min on cuda.
Both encoders use their own grid-search winner. Only the *model* seed
varies; EvalConfig stays at 42, so every run is scored on the same
candidate set.

## GRU - results (test split)

| Variant | AP | MSE | F1 |
|---|---|---|---|
| **full** | **0.9238 +- 0.0004** | 0.0826 +- 0.0004 | 0.8596 +- 0.0011 |
| no_graph | 0.9214 +- 0.0006 | 0.0833 +- 0.0003 | 0.8600 +- 0.0011 |
| no_temporal | 0.9236 +- 0.0003 | 0.0827 +- 0.0002 | 0.8593 +- 0.0005 |
| no_pair | 0.9112 +- 0.0019 | 0.0861 +- 0.0009 | 0.8497 +- 0.0030 |
| pair_only | 0.9158 +- 0.0009 | 0.0847 +- 0.0003 | 0.8565 +- 0.0010 |

### GRU - difference vs. full model

Pooled standard deviations sqrt((sigma_abl^2 + sigma_full^2)/2); below ~2 sigma is not distinguishable from seed noise. AP and MSE can disagree — a branch may look irrelevant under binary AP while still feeding the count head.

| Variant | d AP | sigma | Verdict (AP) | d MSE | sigma | Verdict (MSE) |
|---|---|---|---|---|---|---|
| no_graph | -0.0024 | 4.7 | **real effect** | +0.0007 | 2.0 | not distinguishable from seed noise |
| no_temporal | -0.0002 | 0.6 | not distinguishable from seed noise | +0.0001 | 0.3 | not distinguishable from seed noise |
| no_pair | -0.0126 | 9.2 | **real effect** | +0.0035 | 5.0 | **real effect** |
| pair_only | -0.0080 | 11.5 | **real effect** | +0.0021 | 5.9 | **real effect** |

## CNN - results (test split)

| Variant | AP | MSE | F1 |
|---|---|---|---|
| **full** | **0.9230 +- 0.0005** | 0.0828 +- 0.0004 | 0.8603 +- 0.0007 |
| no_graph | 0.9198 +- 0.0011 | 0.0842 +- 0.0003 | 0.8582 +- 0.0006 |
| no_temporal | 0.9233 +- 0.0011 | 0.0826 +- 0.0017 | 0.8590 +- 0.0010 |
| no_pair | 0.8950 +- 0.0025 | 0.0913 +- 0.0015 | 0.8409 +- 0.0011 |
| pair_only | 0.9156 +- 0.0019 | 0.0845 +- 0.0006 | 0.8567 +- 0.0007 |

### CNN - difference vs. full model

Pooled standard deviations sqrt((sigma_abl^2 + sigma_full^2)/2); below ~2 sigma is not distinguishable from seed noise. AP and MSE can disagree — a branch may look irrelevant under binary AP while still feeding the count head.

| Variant | d AP | sigma | Verdict (AP) | d MSE | sigma | Verdict (MSE) |
|---|---|---|---|---|---|---|
| no_graph | -0.0032 | 3.7 | **real effect** | +0.0014 | 4.0 | **real effect** |
| no_temporal | +0.0003 | 0.4 | not distinguishable from seed noise | -0.0002 | 0.2 | not distinguishable from seed noise |
| no_pair | -0.0280 | 15.5 | **real effect** | +0.0085 | 7.7 | **real effect** |
| pair_only | -0.0074 | 5.3 | **real effect** | +0.0017 | 3.3 | **real effect** |


Reproduce: `python eval_branches.py --seeds 5`
Raw per-run data: `branches_raw.csv`, aggregated: `branches_summary.csv`
