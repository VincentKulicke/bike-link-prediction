# -*- coding: utf-8 -*-
"""
Score the full station grid for one 30-minute window and dump it to CSV.

The evaluation files only hold the 1:5 candidate sample (~411 pairs per bin),
which is fine for metrics but leaves holes in a map. Here we score every
ordered pair instead: 232 * 231 = 53,592 rows, roughly 0.13 s of inference.

Stage 1 of the demo. Stage 2 (demo_map.py) draws the picture from the CSV so
the plot can be tweaked without retraining.

    python ablation/demo_score_bin.py --bin 1332
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

PREP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "prepared Data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# window start, from prepared Data/README.md
T0 = dt.date(2024, 5, 16)
BIN_MIN = 30


def bin_label(b):
    """1332 -> 'Mi 12.06.2024, 18:00'"""
    day = T0 + dt.timedelta(days=b // 48)
    hh, mm = (b % 48) // 2, (b % 48) % 2 * 30
    wd = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"][day.weekday()]
    return f"{wd} {day.strftime('%d.%m.%Y')}, {hh:02d}:{mm:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=int, default=1332,
                    help="bin index to score (default: Wed 18:00, busiest test window)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    b = args.bin

    # the split boundaries live in shared_eval: train 21d, val 4d, test 4d
    if b < 1200:
        print(f"WARNING: bin {b} is not in the test split (test starts at 1200).")

    print(f"Scoring bin {b}  =  {bin_label(b)}")

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
    model = res["model"]
    print(f"  done in {res['train_s']} s")

    # --- full ordered grid for this one bin ---------------------------------
    N = data.N
    uu, ii = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    keep = uu != ii
    grid = pd.DataFrame({"u": uu[keep].ravel(), "i": ii[keep].ravel()})
    grid["bin_idx"] = b
    print(f"  grid: {len(grid):,} ordered pairs")

    t = data._precompute(grid)
    model.eval()
    scores = np.zeros(len(grid), dtype=np.float32)
    counts = np.zeros(len(grid), dtype=np.float32)
    with torch.no_grad():
        sage_emb = model.sage(data.static_x_t, data.A_norm)
        for s in range(0, len(grid), 8192):
            sl = slice(s, s + 8192)
            logit, cnt = model(sage_emb, t["u"][sl], t["i"][sl],
                               t["win_u"][sl], t["win_i"][sl], t["pf"][sl])
            scores[sl] = torch.sigmoid(logit).cpu().numpy()
            counts[sl] = cnt.cpu().numpy()
    grid["score"] = scores
    grid["pred_count"] = counts

    # --- ground truth for the same window -----------------------------------
    se = pd.read_csv(os.path.join(PREP, "superedge_counts.csv"))
    truth = se[se["bin_idx"] == b][["u", "i", "count"]]
    grid = grid.merge(truth, on=["u", "i"], how="left")
    grid["count"] = grid["count"].fillna(0.0)
    grid["label"] = (grid["count"] > 0).astype(int)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"demo_bin{b}_scored.csv")
    grid.to_csv(path, index=False)

    # --- diagnostics --------------------------------------------------------
    n_true = int(grid.label.sum())
    print(f"\n  true edges in this window : {n_true}")
    print(f"  trips in this window      : {int(grid['count'].sum())}")
    print("\n  threshold   predicted   TP    FP    FN   precision  recall")
    for thr in (0.3, 0.5, 0.7, 0.9):
        pred = grid.score >= thr
        tp = int((pred & (grid.label == 1)).sum())
        fp = int((pred & (grid.label == 0)).sum())
        fn = n_true - tp
        pr = tp / max(1, tp + fp)
        rc = tp / max(1, n_true)
        print(f"    {thr:.1f}       {int(pred.sum()):6d}  {tp:4d}  {fp:4d}  {fn:4d}"
              f"     {pr:.3f}    {rc:.3f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
