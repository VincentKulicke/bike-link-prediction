# -*- coding: utf-8 -*-
"""
Score every 30-minute window of one day on the full station grid.

Same idea as demo_score_bin.py, but for all 48 bins of a day. Writing all
48 x 53,592 rows would be a 150 MB CSV of mostly near-zero scores, so we keep
only what the animation needs per bin: the edges that actually happened, plus
the model's top-K pairs (K = number of real edges in that window).

    python ablation/demo_score_day.py --day 27      # Wed 2024-06-12
"""
import os
import sys
import argparse
import datetime as dt

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_core import HybridCfg, HybridData, run_hybrid

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(os.path.dirname(HERE), "prepared Data")
RES = os.path.join(HERE, "results")
T0 = dt.date(2024, 5, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=27,
                    help="day index since 2024-05-16 (27 = Wed 2024-06-12)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_k", type=int, default=3,
                    help="score at least this many pairs even in dead windows")
    args = ap.parse_args()

    b0, b1 = args.day * 48, args.day * 48 + 47
    date = T0 + dt.timedelta(days=args.day)
    print(f"Day {args.day} = {date} ({date.strftime('%A')}), bins {b0}..{b1}")
    if b0 < 1200:
        print("WARNING: this day is not fully inside the test split (starts at 1200).")

    # config from the final grid search, so the demo shows the model the talk
    # actually presents rather than an earlier one
    from final_eval import best_cfg
    c = best_cfg("hybrid_gru", "val_ap", True)
    lb = int(c["ts_lookback"])
    print(f"Hybrid-Konfiguration: {c}")
    data = HybridData(lookback=lb)
    cfg = HybridCfg(encoder="gru", hidden=int(c["hidden"]), lr=float(c["lr"]),
                    lambda_count=0.5, ts_lookback=lb,
                    fusion_hidden=int(c["fusion_hidden"]), seed=args.seed,
                    epochs=30, patience=3)
    print("training the best hybrid config ...")
    res = run_hybrid(cfg, data, eval_splits=(), return_model=True)
    model = res["model"]; model.eval()
    print(f"  done in {res['train_s']} s")

    se = pd.read_csv(os.path.join(PREP, "superedge_counts.csv"))
    N = data.N
    uu, ii = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    keep = uu != ii
    base_u, base_i = uu[keep].ravel(), ii[keep].ravel()

    rows = []
    with torch.no_grad():
        sage_emb = model.sage(data.static_x_t, data.A_norm)
        for b in range(b0, b1 + 1):
            grid = pd.DataFrame({"u": base_u, "i": base_i, "bin_idx": b})
            t = data._precompute(grid)
            sc = np.zeros(len(grid), dtype=np.float32)
            pc = np.zeros(len(grid), dtype=np.float32)
            for s in range(0, len(grid), 8192):
                sl = slice(s, s + 8192)
                logit, cnt = model(sage_emb, t["u"][sl], t["i"][sl],
                                   t["win_u"][sl], t["win_i"][sl], t["pf"][sl])
                sc[sl] = torch.sigmoid(logit).cpu().numpy()
                pc[sl] = cnt.cpu().numpy()
            grid["score"] = sc; grid["pred_count"] = pc

            truth = se[se["bin_idx"] == b][["u", "i", "count"]]
            grid = grid.merge(truth, on=["u", "i"], how="left")
            grid["count"] = grid["count"].fillna(0.0)
            grid["label"] = (grid["count"] > 0).astype(int)

            K = max(args.min_k, int(grid.label.sum()))
            top = grid.nlargest(K, "score")
            real = grid[grid.label == 1]
            # keep the union; a row can be both real and predicted
            sub = pd.concat([real, top]).drop_duplicates(subset=["u", "i"])
            sub = sub[["u", "i", "bin_idx", "score", "pred_count", "count", "label"]]
            sub["in_topk"] = sub.set_index(["u", "i"]).index.isin(
                top.set_index(["u", "i"]).index).astype(int)
            rows.append(sub)

            hits = int(top.label.sum())
            print(f"  bin {b} {(b%48)//2:02d}:{(b%48)%2*30:02d}  "
                  f"real={int(grid.label.sum()):3d}  K={K:3d}  hits={hits:3d}")

    out = pd.concat(rows, ignore_index=True)
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, f"demo_day{args.day}_scored.csv")
    out.to_csv(path, index=False)
    print(f"\nwrote {path}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
