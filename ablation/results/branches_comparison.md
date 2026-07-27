# Branch ablations (Hybrid GraphSAGE+GRU)

Encoder swap (GRU ↔ CNN) only shows interchangeable *encoding*.
These runs remove each component by zeroing its contribution in
`HybridHurdle` — parameter count is identical across variants.

4 variants × 3 seeds (42–44), ~22 min on cpu.
Only the *model* seed varies; EvalConfig stays at 42.
Full-model reference is measured on this machine (not copied from CUDA).

## Results (test split)

| Variant | AP | MSE | F1 | Seed std (AP) |
|---|---|---|---|---|
| **full** | **0.9232 ± 0.0008** | 0.0826 ± 0.0001 | 0.8606 ± 0.0017 | 0.0008 |
| no_graph | 0.9203 ± 0.0010 | 0.0843 ± 0.0005 | 0.8589 ± 0.0024 | 0.0010 |
| no_temporal | 0.9227 ± 0.0008 | 0.0831 ± 0.0002 | 0.8600 ± 0.0018 | 0.0008 |
| no_pair | 0.9017 ± 0.0014 | 0.0889 ± 0.0006 | 0.8422 ± 0.0029 | 0.0014 |

## Difference vs. full model

Expressed in pooled standard deviations (√((σ_abl² + σ_full²) / 2)) — anything below ~2σ is not distinguishable from seed noise.

AP and MSE can disagree: a branch may look irrelevant under binary AP while still contributing to the count head. Read both columns.

| Variant | Δ AP | σ (AP) | Verdict (AP) | Δ MSE | σ (MSE) | Verdict (MSE) |
|---|---|---|---|---|---|---|
| no_graph | -0.0029 | 3.2 | **real effect** | +0.0017 | 4.7 | **real effect** |
| no_temporal | -0.0005 | 0.6 | not distinguishable from seed noise | +0.0005 | 3.2 | **real effect** |
| no_pair | -0.0215 | 18.9 | **real effect** | +0.0063 | 14.6 | **real effect** |

Reproduce: `python eval_branches.py --seeds 3`
Raw per-run data: `branches_raw.csv` · aggregated: `branches_summary.csv`
