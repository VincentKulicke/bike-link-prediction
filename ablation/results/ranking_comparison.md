# 1-vs-99 ranking evaluation

Harder protocol than the 1:5 default: each positive is ranked against 99 random destinations (same source & bin), seed 42, 3000 queries per split. MRR / Hits are the ranking view; AUC / AP are pooled under 1:99 imbalance.

## Test set

| Model | Tuning | MRR | Hits@1 | Hits@5 | AUC | AP |
|---|---|---|---|---|---|---|
| Hybrid GraphSAGE+GRU | default | 0.402 | 0.250 | 0.567 | 0.928 | 0.102 |
| Hybrid GraphSAGE+GRU | HPO | 0.402 | 0.249 | 0.567 | 0.927 | 0.102 |
| GraphMixer | default | 0.342 | 0.196 | 0.490 | 0.927 | 0.115 |
| GraphMixer | HPO | 0.315 | 0.173 | 0.449 | 0.924 | 0.095 |

## Validation set

| Model | Tuning | MRR | Hits@1 | Hits@5 | AUC | AP |
|---|---|---|---|---|---|---|
| Hybrid GraphSAGE+GRU | default | 0.383 | 0.229 | 0.555 | 0.926 | 0.104 |
| Hybrid GraphSAGE+GRU | HPO | 0.386 | 0.231 | 0.554 | 0.926 | 0.107 |
| GraphMixer | default | 0.334 | 0.182 | 0.488 | 0.926 | 0.109 |
| GraphMixer | HPO | 0.317 | 0.166 | 0.466 | 0.925 | 0.095 |

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
   1-vs-99 ranking (MRR 0.342 → 0.315). The two-factor experiment
   (`factors_comparison.md`) later measured GraphMixer's seed noise at
   σ ≈ 0.07 AP — far larger than this gap. Both numbers here are single runs, so
   the difference is not distinguishable from noise and the claim does not hold.
   Confirming or refuting it would require running the ranking evaluation across
   several seeds.

5. AUC is ~0.925 for every model (insensitive to imbalance) and pooled AP sits
   near the 1/100 base rate for all; MRR / Hits@1 are the discriminating metrics
   under this protocol.

Reproduce: `python eval_ranking.py --max_queries 3000 --gm_epochs 20`
(3000 seeded queries per split; full metrics in `ranking_eval.csv`).
