# -*- coding: utf-8 -*-
"""
How much easier is the 1:5 evaluation protocol than scanning every pair?

Scores the full station grid for every test window (175 bins x 53,592 ordered
pairs) and compares against the sampled protocol the metrics are reported on.
Self-loops (u == i, round trips) are excluded on both sides so the two
protocols are measured on the same universe of pairs.

    python ablation/protocol_gap.py
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "evaluation"))
from hybrid_core import HybridCfg, HybridData, run_hybrid

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(os.path.dirname(HERE), "prepared Data")
RES = os.path.join(HERE, "results")


def main():
    # config from the final grid search, not hard-coded, so this cannot drift
    from final_eval import best_cfg
    c = best_cfg("hybrid_gru", "val_ap", True)
    lb = int(c["ts_lookback"])
    data = HybridData(lookback=lb)
    cfg = HybridCfg(encoder="gru", hidden=int(c["hidden"]), lr=float(c["lr"]),
                    lambda_count=0.5, ts_lookback=lb,
                    fusion_hidden=int(c["fusion_hidden"]), seed=42,
                    epochs=30, patience=3)
    print(f"Hybrid-Konfiguration: {c}")
    print("training the best hybrid config ...")
    res = run_hybrid(cfg, data, eval_splits=(), return_model=True)
    model = res["model"]; model.eval()
    print(f"  done in {res['train_s']} s")

    se = pd.read_csv(os.path.join(PREP, "superedge_counts.csv"))
    se = se[se.u != se.i]                      # drop self-loops everywhere
    test_bins = sorted(se[se.bin_idx >= 1200].bin_idx.unique())
    print(f"test windows: {len(test_bins)}")

    N = data.N
    uu, ii = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    keep = uu != ii
    base_u, base_i = uu[keep].ravel(), ii[keep].ravel()
    P = len(base_u)

    tot_pairs = tot_pos = 0
    tp50 = fp50 = 0
    tpK = totK = 0
    per_bin = []

    with torch.no_grad():
        sage = model.sage(data.static_x_t, data.A_norm)
        for n, b in enumerate(test_bins, 1):
            grid = pd.DataFrame({"u": base_u, "i": base_i, "bin_idx": b})
            t = data._precompute(grid)
            sc = np.zeros(P, dtype=np.float32)
            for s in range(0, P, 16384):
                sl = slice(s, s + 16384)
                logit, _ = model(sage, t["u"][sl], t["i"][sl],
                                 t["win_u"][sl], t["win_i"][sl], t["pf"][sl])
                sc[sl] = torch.sigmoid(logit).cpu().numpy()

            truth = se[se.bin_idx == b]
            lab = np.zeros(P, dtype=np.int8)
            key = {(u, i): 1 for u, i in zip(truth.u, truth.i)}
            if key:
                idx = pd.MultiIndex.from_arrays([base_u, base_i])
                lab = idx.isin(key.keys()).astype(np.int8)

            K = int(lab.sum())
            tot_pairs += P; tot_pos += K
            hit50 = (sc >= 0.5)
            tp50 += int((hit50 & (lab == 1)).sum())
            fp50 += int((hit50 & (lab == 0)).sum())
            if K:
                topk = np.argpartition(-sc, K)[:K]
                h = int(lab[topk].sum())
                tpK += h; totK += K
                per_bin.append((b, K, h, h / K))
            if n % 25 == 0:
                print(f"  {n}/{len(test_bins)} windows")

    prior = tot_pos / tot_pairs
    print("\n" + "=" * 66)
    print("FULL GRID over the whole test split (self-loops excluded)")
    print(f"  pairs scored      : {tot_pairs:,}")
    print(f"  real connections  : {tot_pos:,}")
    print(f"  positive rate     : {100*prior:.4f} %")
    print(f"  threshold 0.5     : TP={tp50:,}  FP={fp50:,}  "
          f"precision={tp50/max(1,tp50+fp50):.4f}  recall={tp50/tot_pos:.4f}")
    print(f"  top-K per window  : {tpK:,}/{totK:,}  "
          f"precision@K={tpK/max(1,totK):.4f}")
    print(f"  random baseline   : {prior:.6f}  "
          f"-> top-K is {(tpK/max(1,totK))/prior:.0f}x better than chance")

    # the sampled protocol, same model, same universe
    ev = data.ev
    cand = ev.build_candidates("test")
    cand = cand[cand.u != cand.i]
    # Score with the model trained above, not with a prediction CSV lying
    # around from an earlier run. The whole point of this comparison is that
    # both halves use the SAME model; reading a stale export silently broke
    # that and made the 1:5 numbers freeze across config changes.
    from hybrid_core import _predict
    pred = _predict(model, data, cfg, "test")
    m = cand.merge(pred[["u", "i", "bin_idx", "score"]],
                   on=["u", "i", "bin_idx"], how="left")
    m["score"] = m["score"].fillna(0.0)
    hit = m.score >= 0.5
    tp = int((hit & (m.label == 1)).sum()); fp = int((hit & (m.label == 0)).sum())
    print("\n1:5 SAMPLED PROTOCOL (what the reported metrics use)")
    print(f"  pairs             : {len(m):,}")
    print(f"  positive rate     : {100*m.label.mean():.2f} %")
    print(f"  threshold 0.5     : TP={tp:,}  FP={fp:,}  "
          f"precision={tp/max(1,tp+fp):.4f}  recall={tp/int(m.label.sum()):.4f}")
    print(f"\n  prior shift       : {m.label.mean()/prior:.0f}x")
    print("=" * 66)

    pd.DataFrame(per_bin, columns=["bin_idx", "K", "hits", "precision_at_K"]) \
        .to_csv(os.path.join(RES, "protocol_gap_per_bin.csv"), index=False)
    pd.DataFrame([dict(
        protocol="full grid @0.5", pairs=tot_pairs, positive_rate=prior,
        precision=tp50 / max(1, tp50 + fp50), recall=tp50 / tot_pos),
        dict(protocol="full grid top-K", pairs=tot_pairs, positive_rate=prior,
             precision=tpK / max(1, totK), recall=tpK / max(1, totK)),
        dict(protocol="1:5 sampled @0.5", pairs=len(m),
             positive_rate=float(m.label.mean()),
             precision=tp / max(1, tp + fp), recall=tp / int(m.label.sum())),
        dict(protocol="random", pairs=tot_pairs, positive_rate=prior,
             precision=prior, recall=np.nan),
    ]).to_csv(os.path.join(RES, "protocol_gap.csv"), index=False)
    print(f"wrote {os.path.join(RES, 'protocol_gap.csv')}")


if __name__ == "__main__":
    main()
