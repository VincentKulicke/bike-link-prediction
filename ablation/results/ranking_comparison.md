# 1-vs-99 ranking evaluation

Harder protocol than the 1:5 default: each positive is ranked against 99 random destinations (same source & bin), seed 42, 3000 queries per split. MRR / Hits are the ranking view; AUC / AP are pooled under 1:99 imbalance. Frequency heuristic = mean train trips/bin per pair (no learning).

## Test set

| Model | Tuning | MRR | Hits@1 | Hits@5 | AUC | AP |
|---|---|---|---|---|---|---|
| Frequency heuristic | — | 0.409 | 0.256 | 0.577 | 0.929 | 0.126 |
| GraphMixer | final | 0.343 | 0.196 | 0.487 | 0.915 | 0.073 |
| Hybrid GRU | final | 0.408 | 0.255 | 0.573 | 0.930 | 0.108 |
| Hybrid CNN | final | 0.402 | 0.249 | 0.566 | 0.930 | 0.109 |

## Validation set

| Model | Tuning | MRR | Hits@1 | Hits@5 | AUC | AP |
|---|---|---|---|---|---|---|
| Frequency heuristic | — | 0.393 | 0.237 | 0.563 | 0.926 | 0.121 |
| GraphMixer | final | 0.337 | 0.189 | 0.480 | 0.913 | 0.074 |
| Hybrid GRU | final | 0.392 | 0.238 | 0.556 | 0.928 | 0.109 |
| Hybrid CNN | final | 0.390 | 0.234 | 0.559 | 0.927 | 0.108 |

## Paired significance (per-query reciprocal ranks)

All models scored on the **same** seeded 1:99 queries (test, n=12095). For each query we store RR = 1/rank(true destination), then form the paired difference Δ = RR(A) − RR(B). Mean ± SE and a Wald 95 % CI; if the CI excludes 0 the difference is distinguishable from a tie.

| Comparison | MRR(A) | MRR(B) | Mean Δ RR | SE | 95 % CI | Verdict |
|---|---|---|---|---|---|---|
| Hybrid − Frequency | 0.4061 | 0.4098 | -0.0037 | 0.0017 | [-0.0071, -0.0003] | **distinguishable from zero** |
| Hybrid − GraphMixer | 0.4061 | 0.3467 | +0.0594 | 0.0031 | [+0.0534, +0.0654] | **distinguishable from zero** |

## Stratified by pair history (train trip count)

Each ranking query’s true pair (u, i) is binned by the **sum of train-split trips** for that pair (`build_targets()`, split=train). Expectation: frequency ≈ / ≥ hybrid on high-history pairs; hybrid ahead on rare/unseen pairs (frequency scores 0 → near-random).

| Stratum | n | Frequency MRR | Hybrid MRR | GraphMixer MRR | Frequency H@1 | Hybrid H@1 | GraphMixer H@1 |
|---|---|---|---|---|---|---|---|
| 0 | 252 | 0.057 | 0.027 | 0.064 | 0.016 | 0.000 | 0.008 |
| 1-5 | 1097 | 0.074 | 0.074 | 0.107 | 0.004 | 0.004 | 0.025 |
| 6-20 | 3000 | 0.168 | 0.174 | 0.200 | 0.029 | 0.032 | 0.061 |
| 21-100 | 5638 | 0.467 | 0.467 | 0.377 | 0.266 | 0.270 | 0.204 |
| >100 | 2108 | 0.818 | 0.792 | 0.635 | 0.718 | 0.690 | 0.496 |
| all | 12095 | 0.410 | 0.406 | 0.347 | 0.257 | 0.254 | 0.199 |

Raw stratified metrics: `ranking_stratified.csv`

## Findings


1. **The 1:99 protocol does create the headroom the 1:5 metric hid.** The hybrid
   sits at MRR ≈ 0.40 / Hits@1 ≈ 0.25 here, far from the ceiling — whereas its
   1:5 AP was 0.92. So the earlier "saturation" was partly an artifact of the
   easy 1:5 protocol.

2. **But HPO still does not help the hybrid, even with that headroom.** Default
   and HPO are identical (test MRR 0.402 = 0.402, Hits@1 0.250 ≈ 0.249). The
   headroom exists but the tuned axes (lr, hidden, λ_count) cannot capture it.
   The limitation is architectural / informational, not a tuning deficiency —
   to improve the hybrid you must change what the model sees (features,
   architecture), not how it is tuned.

3. **The hybrid still beats GraphMixer, but by a narrower, more honest margin.**
   MRR 0.402 vs. 0.342 (test) — a real lead, but far less dramatic than the
   1:5 AP gap (0.92 vs. 0.65) suggested. The hard metric gives a fairer picture.

4. ~~**The HPO selection objective matters.**~~ *Retracted.* The original reading
   was that GraphMixer's HPO config, chosen on 1:5 val AP, transfers poorly to
   1-vs-99 ranking (MRR 0.342 → 0.315). A later two-factor experiment measured
   GraphMixer's seed noise at σ ≈ 0.07 AP under the then-broken training —
   far larger than this gap. Both numbers here are single runs, so
   the difference is not distinguishable from noise and the claim does not hold.
   Confirming or refuting it would require running the ranking evaluation across
   several seeds.

5. AUC is ~0.925 for every model (insensitive to imbalance) and pooled AP sits
   near the 1/100 base rate for all; MRR / Hits@1 are the discriminating metrics
   under this protocol.

6. **Under 1:99 ranking the frequency heuristic matches or slightly beats the
   hybrid** (test MRR 0.409 vs. 0.402; Hits@1 0.256 vs. 0.250). Pair frequency
   alone is already a strong ranker; the hybrid's clear advantage remains on the
   count task (and on 1:5 F1), not on destination ranking.

Reproduce: `python eval_ranking.py --max_queries 3000 --gm_epochs 20`
(3000 seeded queries per split; full metrics in `ranking_eval.csv`).
Frequency-only refresh: `python eval_ranking.py --freq_only`
