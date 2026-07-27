# -*- coding: utf-8 -*-
"""
grid_hybrid.py — grid-search HPO for the hybrid model (ablation study).
=======================================================================

Two encoder variants, same search-depth philosophy:

  --encoder gru   GraphSAGE + GRU     (reference, iteration 2)
                  grid: lr x hidden x lambda_count = 3x3x3 = 27 configs
  --encoder cnn   GraphSAGE + 1D-CNN  (ablation, alternative temporal encoder)
                  grid: lr x hidden x kernel_size = 3x3x2 = 18 configs

Protocol:
  - Selection uses the validation metric only (default: AP, higher is better).
  - The best config is evaluated on the test split exactly once.
  - All runs share the same precomputed inputs (HybridData) and the same seed,
    so the grid search is reproducible and fair.

Usage:
  python grid_hybrid.py --encoder gru
  python grid_hybrid.py --encoder cnn
  python grid_hybrid.py --encoder both      (default)
"""
from __future__ import annotations
import os, sys, time, argparse, itertools
import numpy as np
import pandas as pd

import hybrid_core as hc

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PRED_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions")

# --- search spaces ---------------------------------------------------------
LR_GRID     = [1e-3, 3e-4, 1e-4]
HIDDEN_GRID = [32, 64, 128]
LAMBDA_GRID = [0.5, 1.0, 2.0]     # GRU only
KERNEL_GRID = [3, 5]              # CNN only


def configs_for(encoder: str):
    if encoder == "gru":
        for lr, h, lam in itertools.product(LR_GRID, HIDDEN_GRID, LAMBDA_GRID):
            yield hc.HybridCfg(encoder="gru", lr=lr, hidden=h,
                               fusion_hidden=2 * h, lambda_count=lam)
    elif encoder == "cnn":
        for lr, h, k in itertools.product(LR_GRID, HIDDEN_GRID, KERNEL_GRID):
            yield hc.HybridCfg(encoder="cnn", lr=lr, hidden=h,
                               fusion_hidden=2 * h, kernel_size=k, lambda_count=1.0)
    else:
        raise ValueError(encoder)


def run_grid(encoder: str, data: hc.HybridData, select_by: str = "ap") -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfgs = list(configs_for(encoder))
    print(f"\n=== GRID [{encoder}] : {len(cfgs)} configs | select by val {select_by} ===")

    rows = []
    for j, cfg in enumerate(cfgs, 1):
        t0 = time.time()
        r = hc.run_hybrid(cfg, data, eval_splits=("val",))
        v = r["val"]
        rows.append({"encoder": encoder, "lr": cfg.lr, "hidden": cfg.hidden,
                     "lambda_count": cfg.lambda_count, "kernel_size": cfg.kernel_size,
                     "val_auc": v["auc"], "val_ap": v["ap"], "val_f1": v["f1"],
                     "val_acc": v["accuracy"], "val_mse": v["mse"], "val_mae": v["mae"],
                     "val_rmse": v["rmse"], "sec": round(time.time() - t0, 1)})
        print(f"[{j:2d}/{len(cfgs)}] {cfg.tag():28s} "
              f"val AUC={v['auc']:.3f} AP={v['ap']:.3f} MSE={v['mse']:.3f} "
              f"({rows[-1]['sec']:.0f}s)")

    df = pd.DataFrame(rows)
    log_csv = os.path.join(RESULTS_DIR, f"grid_hybrid_{encoder}.csv")
    df.to_csv(log_csv, index=False)

    # best config by validation metric (AP/AUC higher is better; mse lower is better)
    ascending = select_by in ("val_mse", "val_mae", "val_rmse")
    key = select_by if select_by.startswith("val_") else f"val_{select_by}"
    best_row = df.sort_values(key, ascending=ascending).iloc[0]
    best_cfg = next(c for c in cfgs
                    if c.lr == best_row["lr"] and c.hidden == best_row["hidden"]
                    and c.lambda_count == best_row["lambda_count"]
                    and c.kernel_size == best_row["kernel_size"])

    print(f"\n>>> BEST [{encoder}] by val {select_by}: {best_cfg.tag()}")
    print(">>> Final run with test evaluation (test is touched only NOW) ...")
    final = hc.run_hybrid(best_cfg, data, eval_splits=("val", "test"),
                          export_dir=PRED_DIR)
    t = final["test"]
    print(f">>> TEST [{encoder}]: AUC={t['auc']:.3f} AP={t['ap']:.3f} "
          f"F1={t['f1']:.3f} Acc={t['accuracy']:.3f} | "
          f"MSE={t['mse']:.3f} MAE={t['mae']:.3f} RMSE={t['rmse']:.3f}")

    return {"encoder": encoder, "log_csv": log_csv, "best_cfg": best_cfg,
            "val": final["val"], "test": final["test"], "n_configs": len(cfgs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["gru", "cnn", "both"], default="both")
    ap.add_argument("--select_by", default="ap",
                    help="validation metric used for selection (ap|auc|f1|mse|mae|rmse)")
    args = ap.parse_args()

    data = hc.HybridData()   # load + precompute ONCE for both encoders
    encoders = ["gru", "cnn"] if args.encoder == "both" else [args.encoder]

    summary = {}
    t0 = time.time()
    for enc in encoders:
        summary[enc] = run_grid(enc, data, select_by=args.select_by)

    # compact summary CSV (feeds make_ablation_comparison.py)
    srows = []
    for enc, s in summary.items():
        bc = s["best_cfg"]
        srows.append({"model": f"Hybrid-{enc.upper()} (HPO)",
                      "best_config": bc.tag(),
                      "test_auc": s["test"]["auc"], "test_ap": s["test"]["ap"],
                      "test_f1": s["test"]["f1"], "test_acc": s["test"]["accuracy"],
                      "test_mse": s["test"]["mse"], "test_mae": s["test"]["mae"],
                      "test_rmse": s["test"]["rmse"]})
    out = os.path.join(RESULTS_DIR, "grid_hybrid_best.csv")
    pd.DataFrame(srows).to_csv(out, index=False)
    print(f"\n=== hybrid grid done in {(time.time()-t0)/60:.1f} min -> {out} ===")


if __name__ == "__main__":
    main()
