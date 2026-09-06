# -*- coding: utf-8 -*-
"""
Final hyperparameter search -- one run per model, then done.

Grid design follows two things the earlier search told us:

1. Almost every previous optimum sat on the EDGE of its tested range
   (hybrid wanted the largest lr and hidden, GraphMixer the smallest), which
   is the classic sign of a grid that was cut too narrow. The ranges below
   extend past those edges.

2. No model ever tuned the one parameter that controls how much history it
   sees -- ts_lookback (hybrid), lookback (LSTM), num_neighbors (GraphMixer).
   Those change the information available, not just capacity, so they get an
   axis each.

Dropped on evidence: lambda_count (span 0.0005 AP, below seed noise) and,
for the GRU variant, dropout (it only touches the graph branch and the fusion,
and there is no overfitting signal -- test AP sits above val AP).

GraphMixer note: its old grid was measured while it trained on a 50 % positive
rate against a 1:5 evaluation. That is fixed now, so the old "wants a small lr"
finding no longer applies and the lr range is opened in both directions.

Every finished config is appended to CSV immediately and skipped on restart,
so an interrupted run costs one config rather than the whole night.

    python ablation/hpo_final.py --model hybrid_gru
    python ablation/hpo_final.py --model all
"""
import os
import sys
import time
import argparse
import itertools

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "lstm"),
          os.path.join(ROOT, "graphmixer", "model")):
    sys.path.insert(0, p)

RES = os.path.join(HERE, "results")
PATIENCE = 3
MAX_EPOCHS = 30

# ---------------------------------------------------------------------------
# grids
# ---------------------------------------------------------------------------
GRIDS = {
    "hybrid_gru": dict(
        lr=[1e-3, 3e-3, 1e-2],           # old best 1e-3 was the top edge
        hidden=[128, 256, 384],          # old best 128 was the top edge
        ts_lookback=[6, 12, 24, 48],     # never tuned
        fusion_hidden=[128, 256],        # never tuned
        seeds=[42],
    ),
    "hybrid_cnn": dict(
        lr=[1e-3, 3e-3, 1e-2],
        hidden=[128, 256, 384],
        ts_lookback=[6, 12, 24, 48],
        dropout=[0.0, 0.1, 0.3],         # CNN applies it at 4 sites, GRU at 2
        seeds=[42],
    ),
    "lstm": dict(
        lr=[1e-3, 3e-3, 1e-2],           # old best 1e-3 was the top edge
        hidden_dim=[32, 64, 128],        # old best 64 sat inside -> keep
        # Never tuned before, and the one axis in this whole search with a
        # large effect: val MSE falls monotonically 0.168 -> 0.114 from 12 to
        # 96 with no sign of flattening, so 192 (4 days of history) is added
        # to find out whether the optimum is inside the range at all.
        lookback=[12, 24, 48, 96, 192],
        layers_dropout=[(1, 0.0), (2, 0.2)],   # coupled: nn.LSTM ignores
        seeds=[42],                            # dropout when num_layers == 1
    ),
    "graphmixer": dict(
        lr=[3e-5, 1e-4, 3e-4, 1e-3],     # opened both ways after the fix
        hidden_dim=[32, 64, 128],
        num_neighbors=[10, 20, 40],      # never tuned
        # Cut to 1 partway through the run: across the 36 configs measured at
        # lr=3e-5 the two levels differed by 0.0032 AP on average -- far under
        # the ~0.02 seed noise -- while L=2 cost 571 s against 427 s per run.
        # Dropping it saved ~10.7 h of the remaining budget. The L=2 rows that
        # had already finished are kept, so that axis is covered at lr=3e-5
        # only and its main effect must be read as such.
        mixer_layers=[1],
        seeds=[42, 43],                  # sigma ~0.05 AP -> 1 seed selects noise
    ),
}


def combos(spec):
    keys = [k for k in spec if k != "seeds"]
    for vals in itertools.product(*[spec[k] for k in keys]):
        for seed in spec["seeds"]:
            yield dict(zip(keys, vals), seed=seed)


def _k(v):
    """Canonical key text. Needed because a CSV round-trip turns 128 into
    128.0, and df.iterrows() upcasts every row to one dtype on top of that."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f == int(f) else repr(f)


def done_keys(path, keys):
    """Configs that finished *successfully*.

    Rows with an error are deliberately not counted as done: this box throws
    the occasional transient CUDA fault under sustained load, and a config
    that hit one must be retried rather than silently dropped from the grid.
    """
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    if not all(k in df.columns for k in keys):
        return set()
    if "error" in df.columns:
        df = df[df["error"].isna()]
    if not len(df):
        return set()
    # column-wise, so dtypes survive; iterrows() would not
    cols = [df[k].map(_k).tolist() for k in keys]
    return set(zip(*cols))


def append(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path),
                               index=False)


# ---------------------------------------------------------------------------
# per-model runners
# ---------------------------------------------------------------------------
def run_hybrid_grid(encoder):
    from hybrid_core import HybridCfg, HybridData, run_hybrid
    name = f"hybrid_{encoder}"
    spec = GRIDS[name]
    out = os.path.join(RES, f"hpo_final_{name}.csv")
    keys = [k for k in spec if k != "seeds"] + ["seed"]
    done = done_keys(out, keys)
    data_cache = {}

    todo = [c for c in combos(spec)
            if tuple(_k(c[k]) for k in keys) not in done]
    print(f"[{name}] {len(todo)} open of {sum(1 for _ in combos(spec))}")

    for n, c in enumerate(todo, 1):
        lb = c["ts_lookback"]
        t0 = time.time()
        try:
            # inside the guard: a poisoned CUDA context surfaces here, and the
            # crash must not take the whole grid down with it
            if lb not in data_cache:
                data_cache.clear()             # keep at most one on the GPU
                data_cache[lb] = HybridData(lookback=lb)
            data = data_cache[lb]
            cfg = HybridCfg(encoder=encoder, lr=c["lr"], hidden=c["hidden"],
                            ts_lookback=lb, seed=c["seed"],
                            fusion_hidden=c.get("fusion_hidden", 128),
                            dropout=c.get("dropout", 0.1),
                            lambda_count=0.5 if encoder == "gru" else 1.0,
                            epochs=MAX_EPOCHS, patience=PATIENCE)
            r = run_hybrid(cfg, data, eval_splits=("val", "test"))
            row = dict(c, val_ap=r["val"]["ap"], test_ap=r["test"]["ap"],
                       val_mse=r["val"]["mse"], test_mse=r["test"]["mse"],
                       test_f1=r["test"]["f1"], epochs_run=r["epochs_run"],
                       sec=round(time.time() - t0, 1), error="")
        except Exception as e:                                  # keep going
            data_cache.clear()      # cached tensors may sit on a dead context
            row = dict(c, val_ap=np.nan, test_ap=np.nan, val_mse=np.nan,
                       test_mse=np.nan, test_f1=np.nan, epochs_run=0,
                       sec=round(time.time() - t0, 1), error=str(e)[:200])
        append(out, row)
        print(f"  [{name} {n}/{len(todo)}] lr={c['lr']:g} h={c['hidden']} "
              f"lb={lb} val_ap={row['val_ap']:.4f} ep={row['epochs_run']} "
              f"{row['sec']:.0f}s")


def run_lstm_grid():
    from lstm_count import LSTMConfig, CountSeries, LSTMForecaster, train, export
    from shared_eval import SharedLinkEval
    spec = GRIDS["lstm"]
    out = os.path.join(RES, "hpo_final_lstm.csv")
    keys = ["lr", "hidden_dim", "lookback", "layers_dropout", "seed"]
    done = done_keys(out, keys)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = SharedLinkEval(); cs = CountSeries(ev)

    todo = [c for c in combos(spec)
            if tuple(_k(c[k]) for k in keys) not in done]
    print(f"[lstm] {len(todo)} open of {sum(1 for _ in combos(spec))}")

    for n, c in enumerate(todo, 1):
        layers, drop = c["layers_dropout"]
        cfg = LSTMConfig(lr=c["lr"], hidden_dim=c["hidden_dim"],
                         lookback=c["lookback"], num_layers=layers,
                         dropout=drop, seed=c["seed"], epochs=MAX_EPOCHS)
        rng = np.random.default_rng(c["seed"]); torch.manual_seed(c["seed"])
        t0 = time.time()
        try:
            m = train(cfg, cs, LSTMForecaster(cfg).to(dev), 21 * 48, dev, rng,
                      patience=PATIENCE, verbose=False)
            tmp = os.path.join(HERE, "predictions", "_tmp_hpo_lstm.csv")
            res = {}
            for split in ("val", "test"):
                pred = export(cfg, cs, m, ev, split, dev, tmp)
                res[split] = ev.score_count(pred, split=split)["mse"]
            row = dict(c, val_mse=res["val"], test_mse=res["test"],
                       epochs_run=m.epochs_run, sec=round(time.time() - t0, 1),
                       error="")
        except Exception as e:
            row = dict(c, val_mse=np.nan, test_mse=np.nan, epochs_run=0,
                       sec=round(time.time() - t0, 1), error=str(e)[:200])
        append(out, row)
        print(f"  [lstm {n}/{len(todo)}] lr={c['lr']:g} h={c['hidden_dim']} "
              f"lb={c['lookback']} L={layers} val_mse={row['val_mse']:.4f} "
              f"ep={row['epochs_run']} {row['sec']:.0f}s")


def run_gm_grid():
    from graphmixer import GMConfig, GraphMixer
    from graphmixer_data import GraphMixerData
    from train_graphmixer import train as gm_train, export_predictions
    from shared_eval import SharedLinkEval
    spec = GRIDS["graphmixer"]
    out = os.path.join(RES, "hpo_final_graphmixer.csv")
    keys = ["lr", "hidden_dim", "num_neighbors", "mixer_layers", "seed"]
    done = done_keys(out, keys)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = SharedLinkEval()
    base = GMConfig()
    data = GraphMixerData(base.prep_dir)

    todo = [c for c in combos(spec)
            if tuple(_k(c[k]) for k in keys) not in done]
    print(f"[graphmixer] {len(todo)} open of {sum(1 for _ in combos(spec))}")

    for n, c in enumerate(todo, 1):
        cfg = GMConfig(lr=c["lr"], hidden_dim=c["hidden_dim"],
                       num_neighbors=c["num_neighbors"],
                       mixer_layers=c["mixer_layers"], seed=c["seed"],
                       epochs=MAX_EPOCHS)
        torch.manual_seed(c["seed"])
        t0 = time.time()
        try:
            model = GraphMixer(cfg, edge_feat_dim=data.d_edge,
                               node_feat_dim=data.d_node).to(dev)
            model = gm_train(cfg, data, model, dev, ev=ev, patience=PATIENCE,
                             verbose=False)
            tmp = os.path.join(HERE, "predictions", "_tmp_hpo_gm.csv")
            res = {}
            for split in ("val", "test"):
                pred = export_predictions(cfg, data, model, ev, split, dev, tmp)
                res[split] = ev.score_binary(pred, split=split)
            row = dict(c, val_ap=res["val"]["ap"], test_ap=res["test"]["ap"],
                       test_auc=res["test"]["auc"], test_f1=res["test"]["f1"],
                       sec=round(time.time() - t0, 1), error="")
        except Exception as e:
            row = dict(c, val_ap=np.nan, test_ap=np.nan, test_auc=np.nan,
                       test_f1=np.nan, sec=round(time.time() - t0, 1),
                       error=str(e)[:200])
        append(out, row)
        print(f"  [gm {n}/{len(todo)}] lr={c['lr']:g} h={c['hidden_dim']} "
              f"K={c['num_neighbors']} L={c['mixer_layers']} s={c['seed']} "
              f"val_ap={row['val_ap']:.4f} {row['sec']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    choices=["all", "hybrid_gru", "hybrid_cnn", "lstm", "graphmixer"])
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(os.path.join(HERE, "predictions"), exist_ok=True)

    jobs = ({"hybrid_gru": lambda: run_hybrid_grid("gru"),
             "hybrid_cnn": lambda: run_hybrid_grid("cnn"),
             "lstm": run_lstm_grid,
             "graphmixer": run_gm_grid})
    order = ["hybrid_gru", "hybrid_cnn", "lstm", "graphmixer"] \
        if args.model == "all" else [args.model]

    t0 = time.time()
    for k in order:
        print(f"\n{'='*70}\n{k}\n{'='*70}")
        jobs[k]()
    print(f"\nfertig in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
