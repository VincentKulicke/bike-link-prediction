# Multi-seed robustness check

Every other result in this study came from a single run, so no difference had an
error bar. This re-runs the final selected configurations across 5 seeds
(42–46) and reports mean ± std. Only the *model* seed varies (weight init,
batch shuffling); the evaluation seed stays at 42, so all runs are scored on
exactly the same candidate set.

35 runs total, ~30 min on an RTX 4070 Laptop.

## Results (test split, 5 seeds)

| Model | Variant | AP | MSE | Seed std (AP) |
|---|---|---|---|---|
| Hybrid GraphSAGE+GRU | default | **0.9233 ± 0.0008** | 0.0824 ± 0.0004 | 0.0008 |
| Hybrid GraphSAGE+GRU | HPO | **0.9233 ± 0.0006** | 0.0820 ± 0.0003 | 0.0006 |
| Hybrid GraphSAGE+1D-CNN | HPO | 0.9226 ± 0.0005 | 0.0821 ± 0.0002 | 0.0005 |
| GraphMixer | default | 0.5957 ± 0.0233 | — | 0.0233 |
| GraphMixer | HPO | 0.7012 ± 0.0375 | — | 0.0375 |
| LSTM | default | — | 0.2404 ± 0.0038 | — |
| LSTM | HPO | — | 0.2400 ± 0.0047 | — |

## Which differences are real?

Expressed in pooled standard deviations — anything below ~2σ is not
distinguishable from seed noise.

| Comparison | Difference | σ | Verdict |
|---|---|---|---|
| Hybrid: HPO vs. default | +0.0000 AP | 0.04 | **no effect** |
| LSTM: HPO vs. default | −0.0004 MSE | 0.09 | **no effect** |
| Hybrid: GRU vs. 1D-CNN | +0.0007 AP | 1.3 | not significant, practically irrelevant |
| GraphMixer: HPO vs. default | +0.1056 AP | 3.4 | **real improvement** |
| Hybrid vs. GraphMixer (AP) | +0.327 AP | ~12 | **decisive** |
| Hybrid vs. LSTM (MSE) | −0.158 MSE | ~38 | **decisive** |

## What this changes

1. **"HPO does not improve the hybrid" is now a measured result, not a guess.**
   The difference is 0.04σ — the tuned config (h128, λ 0.5) and the default
   (h64, λ 1.0) are indistinguishable despite twice the width.

2. **The LSTM tuning gain was noise.** The single-seed run suggested MSE
   0.239 → 0.233 (~2.5 % better). Across seeds the difference is 0.09σ, and the
   spread alone (±0.004) is ten times the supposed gain. That claim is dropped.

3. **The GraphMixer tuning gain is real** (+0.106 AP, 3.4σ). Of all four HPO
   experiments, only the weakest baseline actually benefited from tuning.

4. **Stability differs enormously between architectures.** Seed noise is
   ±0.0007 AP for the hybrid but ±0.023 (default) to ±0.038 (HPO) for
   GraphMixer — a factor of 33–54. The hybrid is not just more accurate but
   far more reproducible, which matters as much in practice.

5. **The hyperparameter effect is systematic but tiny.** Note that the two
   factors are measured separately in `factors_comparison.md` — the grid spreads
   quoted elsewhere in this repo come from single-seed runs and are inflated by
   seed noise. Cleanly separated, the hybrid's hyperparameter effect is 8.2×
   its seed noise (so the grid ranking is meaningful) but spans only 0.0064 AP
   in total. The architecture gap to GraphMixer is roughly 50× that range. The
   performance comes from the architecture, not from tuning.

## Correction to earlier GraphMixer numbers

`results/comparison.md` reports GraphMixer default at AP 0.653 / AUC 0.909.
That number comes from a prediction file dated 22 June and **does not
reproduce**. Two independent current measurements agree with each other and
disagree with it:

| Source | AP | AUC |
|---|---|---|
| `predictions/graphmixer_pred_test.csv` (22 Jun, stale) | 0.653 | 0.909 |
| grid search, same config (today) | 0.555 (val) | 0.894 (val) |
| seed sweep, 5 seeds (today) | **0.596 ± 0.023** | 0.898 ± 0.006 |

The evaluation protocol is not the cause — `shared_eval.py` only gained the
ranking helpers today; `build_candidates` and `score_binary` are unchanged. The
old file most likely came from an interactive notebook run with different
settings (the notebook contains both `GMConfig(epochs=20)` and
`GMConfig(epochs=2)`). Use the reproducible values.

This makes the hybrid's lead **larger**, not smaller: 0.923 vs. 0.596 instead of
0.923 vs. 0.653.

Reproduce: `python eval_seeds.py --models all --seeds 5`
Raw per-run data: `seeds_raw.csv` · aggregated: `seeds_summary.csv`
