# Grid-search effect vs. seed effect — measured separately

The grid search used one seed per configuration, so each config's score also
contained that single random draw; the seed sweep did the opposite (many seeds,
but only for the selected configs). Neither isolates its own factor, and the
grid spreads reported earlier were therefore inflated by seed noise.

This is a two-factor design that separates them: **4 configurations spanning
each grid × 5 seeds each** (60 runs total). The configurations are taken from
the existing grid logs at ranks 1, ~1/3, ~2/3 and last, so they cover the range
instead of clustering at the top.

- **σ_seed** — spread from the random seed alone (pooled within-config std)
- **σ_grid** — spread from the hyperparameters alone (between-config std,
  corrected for the seed noise remaining in each config mean:
  σ_grid² = var(means) − σ_seed²/n)

Metric is each model's grid selection metric on validation (AP for the binary
models, MSE for LSTM).

## Result

| Model | σ_seed | σ_grid | σ_grid / σ_seed | range of config means |
|---|---|---|---|---|
| Hybrid GraphSAGE+GRU | 0.0003 | 0.0028 | **8.2×** | 0.0064 |
| LSTM | 0.0034 | 0.0074 | **2.2×** | 0.0181 |
| GraphMixer | 0.0700 | 0.0919 | **1.3×** | 0.2240 |

## Per configuration

**Hybrid** (val AP, higher is better)

| Config | lr, hidden, λ | mean | seed std |
|---|---|---|---|
| rank 1 (best) | 1e-3, 128, 0.5 | 0.9160 | ±0.0003 |
| rank 10 | 1e-3, 32, 2.0 | 0.9148 | ±0.0006 |
| rank 19 | 1e-4, 128, 0.5 | 0.9124 | ±0.0001 |
| rank 27 (worst) | 1e-4, 32, 2.0 | 0.9096 | ±0.0002 |

**LSTM** (val MSE, lower is better)

| Config | lr, hidden, layers | mean | seed std |
|---|---|---|---|
| rank 1 (best) | 3e-4, 32, 1 | 0.2577 | ±0.0046 |
| rank 7 | 1e-4, 128, 1 | 0.2641 | ±0.0045 |
| rank 13 | 3e-4, 32, 2 | 0.2643 | ±0.0021 |
| rank 18 (worst) | 1e-4, 32, 1 | 0.2758 | ±0.0002 |

**GraphMixer** (val AP, higher is better)

| Config | lr, hidden, layers | mean | seed std |
|---|---|---|---|
| rank 1 (best) | 1e-4, 64, 1 | 0.7087 | ±0.0360 |
| rank 7 | 1e-3, 256, 2 | 0.5438 | ±0.1000 |
| rank 13 | 3e-4, 128, 2 | 0.6208 | ±0.0585 |
| rank 18 (worst) | 1e-3, 128, 1 | 0.4848 | ±0.0698 |

## How much the earlier single-seed numbers were inflated

| Model | grid range (1 seed, full grid) | corrected (5 seeds, 4 configs) | overstated by |
|---|---|---|---|
| Hybrid | 0.0065 | 0.0064 | ~2 % |
| LSTM | 0.0250 | 0.0181 | **28 %** |
| GraphMixer | 0.3383 | 0.2240 | **34 %** |

## What this means

1. **The ratio σ_grid/σ_seed says how trustworthy a grid search can be at all.**
   For the hybrid (8.2×) hyperparameters clearly dominate the noise and the grid
   ranking is meaningful. For the LSTM (2.2×) it is borderline. For GraphMixer
   (1.3×) a single-seed grid search is barely informative.

2. **GraphMixer's ranking is partly random.** Configs at grid ranks 7 and 13 swap
   places once averaged over 5 seeds (0.544 vs. 0.621). Only the extremes stay
   separable (0.709 best vs. 0.485 worst). One config varies by ±0.100 AP across
   seeds — a single GraphMixer run is essentially uninterpretable on its own.

3. **The hybrid's hyperparameter effect is systematic but tiny.** 8× above noise,
   yet the whole reachable range is 0.0064 AP. For comparison, the gap to the
   trivial frequency heuristic is 0.032 AP — five times larger than anything
   tuning can move.

4. **Stability differs by two orders of magnitude.** σ_seed is 0.0003 for the
   hybrid and 0.0700 for GraphMixer (233×). The hybrid produces reproducible
   results; GraphMixer does not. That is a quality argument in its own right and
   independent of accuracy.

5. **Consequence for the GraphMixer HPO result.** Its tuning gain (+0.106 AP over
   5 seeds) is real, but the *selection* was made on a single seed with
   σ_seed ≈ 0.07 — that it picked a genuinely good region was partly luck. Any
   single-seed GraphMixer comparison (including the 1-vs-99 default-vs-HPO
   difference in `ranking_comparison.md`) should be treated as inconclusive.

Reproduce: `python eval_factors.py --models all --seeds 5`
Raw data: `factors_raw.csv` · summary: `factors_summary.csv`
