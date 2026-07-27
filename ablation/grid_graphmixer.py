# -*- coding: utf-8 -*-
"""
grid_graphmixer.py — grid-search HPO for the GraphMixer baseline (ablation).
============================================================================

Tunes the temporal-graph baseline with the same search depth as the hybrid
model (binary link prediction).

  grid: lr x hidden_dim x mixer_layers = 3x3x2 = 18 configs
  selection: validation AP only (higher is better).
  The best config is evaluated on the test split exactly once.

Reuses the model/training/export code from  ../graphmixer/model/  as-is.
Note: GraphMixer trains event-by-event and is slower than the tensor-based
models. epochs is controllable via EPOCHS.

Usage:  python grid_graphmixer.py [--epochs 20]
"""
from __future__ import annotations
import os, sys, time, itertools, argparse
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "graphmixer", "model"))
sys.path.insert(0, os.path.join(_HERE, "..", "evaluation"))
from graphmixer import GraphMixer, GMConfig                       # noqa: E402
from graphmixer_data import GraphMixerData                        # noqa: E402
from train_graphmixer import train, export_predictions           # noqa: E402
from shared_eval import SharedLinkEval, EvalConfig                # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")
PRED_DIR    = os.path.join(_HERE, "predictions")

LR_GRID     = [1e-3, 3e-4, 1e-4]
HIDDEN_GRID = [64, 128, 256]
LAYERS_GRID = [1, 2]


def build(cfg, data, device):
    model = GraphMixer(cfg, edge_feat_dim=data.d_edge, node_feat_dim=data.d_node).to(device)
    return train(cfg, data, model, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(PRED_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | epochs={args.epochs}")

    base = GMConfig()
    data = GraphMixerData(base.prep_dir)          # expensive -> build only once
    ev = SharedLinkEval(EvalConfig(bin_minutes=base.bin_minutes,
                                   train_days=base.train_days, val_days=base.val_days))
    print(f"Nodes: {data.num_nodes} | edges: {len(data.edges):,}")

    cfgs = [GMConfig(lr=lr, hidden_dim=h, mixer_layers=ml, epochs=args.epochs)
            for lr, h, ml in itertools.product(LR_GRID, HIDDEN_GRID, LAYERS_GRID)]
    print(f"\n=== GRID [graphmixer] : {len(cfgs)} configs | select by val AP ===")

    rows = []; t_all = time.time()
    for j, cfg in enumerate(cfgs, 1):
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        t0 = time.time()
        model = build(cfg, data, device)
        val_csv = os.path.join(PRED_DIR, "_tmp_gm_val.csv")
        pred = export_predictions(cfg, data, model, ev, "val", device, val_csv)
        res = ev.score_binary(pred, split="val")
        rows.append({"model": "graphmixer", "lr": cfg.lr, "hidden_dim": cfg.hidden_dim,
                     "mixer_layers": cfg.mixer_layers, "val_auc": res["auc"],
                     "val_ap": res["ap"], "val_f1": res["f1"], "val_acc": res["accuracy"],
                     "sec": round(time.time() - t0, 1)})
        print(f"[{j:2d}/{len(cfgs)}] lr{cfg.lr:g}_h{cfg.hidden_dim}_L{cfg.mixer_layers}  "
              f"val AUC={res['auc']:.3f} AP={res['ap']:.3f} ({rows[-1]['sec']:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "grid_graphmixer.csv"), index=False)

    best = df.sort_values("val_ap", ascending=False).iloc[0]
    best_cfg = GMConfig(lr=float(best["lr"]), hidden_dim=int(best["hidden_dim"]),
                        mixer_layers=int(best["mixer_layers"]), epochs=args.epochs)
    print(f"\n>>> BEST [graphmixer] by val AP: lr{best_cfg.lr:g}_h{best_cfg.hidden_dim}_"
          f"L{best_cfg.mixer_layers}")
    print(">>> Final run with test evaluation ...")
    torch.manual_seed(best_cfg.seed); np.random.seed(best_cfg.seed)
    model = build(best_cfg, data, device)
    summary = {}
    for split in ["val", "test"]:
        out_csv = os.path.join(PRED_DIR, f"graphmixer_hpo_pred_{split}.csv")
        pred = export_predictions(best_cfg, data, model, ev, split, device, out_csv)
        res = ev.score_binary(pred, split=split)
        summary[split] = res
        print(f">>> {split.upper()} [graphmixer]: AUC={res['auc']:.3f} AP={res['ap']:.3f} "
              f"F1={res['f1']:.3f} Acc={res['accuracy']:.3f}")

    pd.DataFrame([{"model": "GraphMixer (HPO)",
                   "best_config": f"lr{best_cfg.lr:g}_h{best_cfg.hidden_dim}_L{best_cfg.mixer_layers}",
                   "test_auc": summary["test"]["auc"], "test_ap": summary["test"]["ap"],
                   "test_f1": summary["test"]["f1"], "test_acc": summary["test"]["accuracy"]}]
                 ).to_csv(os.path.join(RESULTS_DIR, "grid_graphmixer_best.csv"), index=False)
    print(f"\n=== graphmixer grid done in {(time.time()-t_all)/60:.1f} min ===")


if __name__ == "__main__":
    main()
