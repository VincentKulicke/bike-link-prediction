# -*- coding: utf-8 -*-
"""
grid_lstm.py — grid-search HPO for the LSTM count baseline (ablation study).
============================================================================

Tunes the pure time-series baseline with the same search depth as the hybrid
model, so the comparison isn't skewed by unequal tuning.

  grid: lr x hidden_dim x num_layers = 3x3x2 = 18 configs
  selection: validation MSE only (lower is better).
  The best config is evaluated on the test split exactly once.

Reuses the model/training/export code from  ../lstm/lstm_count.py  as-is.

Usage:  python grid_lstm.py
"""
from __future__ import annotations
import os, sys, time, itertools
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "lstm"))
sys.path.insert(0, os.path.join(_HERE, "..", "evaluation"))
from lstm_count import LSTMConfig, CountSeries, LSTMForecaster, train, export  # noqa: E402
from shared_eval import SharedLinkEval                                          # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")
PRED_DIR    = os.path.join(_HERE, "predictions")

LR_GRID     = [1e-3, 3e-4, 1e-4]
HIDDEN_GRID = [32, 64, 128]
LAYERS_GRID = [1, 2]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(PRED_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # shared structures, built once
    ev = SharedLinkEval()
    cs = CountSeries(ev)                          # expensive -> build only once
    bins_per_day = (24 * 60) // ev.cfg.bin_minutes
    train_end_bin = ev.cfg.train_days * bins_per_day

    cfgs = [LSTMConfig(lr=lr, hidden_dim=h, num_layers=nl)
            for lr, h, nl in itertools.product(LR_GRID, HIDDEN_GRID, LAYERS_GRID)]
    print(f"\n=== GRID [lstm] : {len(cfgs)} configs | select by val MSE ===")

    rows = []; t_all = time.time()
    for j, cfg in enumerate(cfgs, 1):
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        t0 = time.time()
        model = LSTMForecaster(cfg).to(device)
        model = train(cfg, cs, model, train_end_bin, device, rng)
        val_csv = os.path.join(PRED_DIR, "_tmp_lstm_val.csv")
        pred = export(cfg, cs, model, ev, "val", device, val_csv)
        res = ev.score_count(pred, split="val")
        rows.append({"model": "lstm", "lr": cfg.lr, "hidden_dim": cfg.hidden_dim,
                     "num_layers": cfg.num_layers, "val_mse": res["mse"],
                     "val_mae": res["mae"], "val_rmse": res["rmse"],
                     "sec": round(time.time() - t0, 1)})
        print(f"[{j:2d}/{len(cfgs)}] lr{cfg.lr:g}_h{cfg.hidden_dim}_L{cfg.num_layers:d}  "
              f"val MSE={res['mse']:.4f} MAE={res['mae']:.4f} ({rows[-1]['sec']:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "grid_lstm.csv"), index=False)

    best = df.sort_values("val_mse").iloc[0]
    best_cfg = LSTMConfig(lr=float(best["lr"]), hidden_dim=int(best["hidden_dim"]),
                          num_layers=int(best["num_layers"]))
    print(f"\n>>> BEST [lstm] by val MSE: lr{best_cfg.lr:g}_h{best_cfg.hidden_dim}_"
          f"L{best_cfg.num_layers}")
    print(">>> Final run with test evaluation ...")
    torch.manual_seed(best_cfg.seed); np.random.seed(best_cfg.seed)
    rng = np.random.default_rng(best_cfg.seed)
    model = LSTMForecaster(best_cfg).to(device)
    model = train(best_cfg, cs, model, train_end_bin, device, rng)
    summary = {}
    for split in ["val", "test"]:
        out_csv = os.path.join(PRED_DIR, f"lstm_hpo_pred_{split}.csv")
        pred = export(best_cfg, cs, model, ev, split, device, out_csv)
        res = ev.score_count(pred, split=split)
        summary[split] = res
        print(f">>> {split.upper()} [lstm]: MSE={res['mse']:.4f} MAE={res['mae']:.4f} "
              f"RMSE={res['rmse']:.4f}")

    pd.DataFrame([{"model": "LSTM (HPO)",
                   "best_config": f"lr{best_cfg.lr:g}_h{best_cfg.hidden_dim}_L{best_cfg.num_layers}",
                   "test_mse": summary["test"]["mse"], "test_mae": summary["test"]["mae"],
                   "test_rmse": summary["test"]["rmse"]}]
                 ).to_csv(os.path.join(RESULTS_DIR, "grid_lstm_best.csv"), index=False)
    print(f"\n=== lstm grid done in {(time.time()-t_all)/60:.1f} min ===")


if __name__ == "__main__":
    main()
