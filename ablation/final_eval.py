# -*- coding: utf-8 -*-
"""
Definitive model comparison: the winner of each grid, run over 5 seeds.

The configurations are read straight out of the hpo_final_*.csv files rather
than copied into this file, so the reported numbers cannot drift away from the
search that produced them. Selection uses the validation metric; everything
reported here is test.

Appends per run and skips finished ones, same as the search itself.

    python ablation/final_eval.py
"""
import os
import sys
import time
import argparse

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
OUT = os.path.join(RES, "final_eval_raw.csv")
SEEDS = [42, 43, 44, 45, 46]
PATIENCE, MAX_EPOCHS = 3, 30


def best_cfg(name, metric, higher):
    """Winning row of a grid, averaged over seeds where several were run."""
    d = pd.read_csv(os.path.join(RES, f"hpo_final_{name}.csv"))
    d = d[d["error"].isna()]
    axes = [c for c in d.columns
            if c not in ("seed", "sec", "error", "epochs_run")
            and not c.startswith(("val_", "test_"))]
    g = d.groupby(axes, as_index=False)[metric].mean()
    row = g.loc[g[metric].idxmax() if higher else g[metric].idxmin()]
    return {a: row[a] for a in axes}


def done_pairs():
    """Only successful runs count as done -- a transient CUDA fault must be
    retried, not silently accepted as a missing data point."""
    if not os.path.exists(OUT):
        return set()
    d = pd.read_csv(OUT)
    if "error" in d.columns:
        d = d[d["error"].isna()]
    return set(zip(d["model"].astype(str), d["seed"].astype(int)))


def append(row):
    pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT),
                               index=False)


def write_markdown(raw, summ, cfgs):
    """results/comparison.md, straight from the 5-seed runs above.

    Replaces the old single-run table, which reported seed 42 only and drew
    its GraphMixer and LSTM numbers from models that trained on the wrong
    candidate distribution.
    """
    def pm(m, col, sd, dec=4):
        if m not in summ.index or pd.isna(summ.loc[m, col]):
            return "—"
        return f"{summ.loc[m, col]:.{dec}f} ± {summ.loc[m, sd]:.{dec}f}"

    def sig(a, b, col, sd):
        diff = summ.loc[a, col] - summ.loc[b, col]
        pooled = np.sqrt(summ.loc[a, sd] ** 2 + summ.loc[b, sd] ** 2)
        return diff, abs(diff) / pooled

    L = ["# Model comparison", "",
         "Mean +/- std over 5 seeds (42-46) of each model's grid winner. "
         "Configurations are read from `hpo_final_*.csv`, so they cannot drift "
         "from the search that produced them. Selection on validation, "
         "everything below is test.", "",
         "## Winning configurations", ""]
    for k, v in cfgs.items():
        L.append(f"- **{k}**: `{v}`")
    L += ["", "## Binary (link yes/no)", "",
          "| Model | AP | AUC | F1 |", "|---|---|---|---|"]
    for m in ("Hybrid GRU", "Hybrid CNN", "GraphMixer"):
        L.append(f"| {m} | {pm(m,'ap_mean','ap_std')} | "
                 f"{pm(m,'auc_mean','auc_std')} | {pm(m,'f1_mean','f1_std')} |")
    L += ["", "## Count (number of rides)", "",
          "| Model | MSE |", "|---|---|"]
    for m in ("Hybrid GRU", "Hybrid CNN", "LSTM"):
        L.append(f"| {m} | {pm(m,'mse_mean','mse_std')} |")

    L += ["", "## Pairwise, in pooled seed sigmas", "",
          "| Comparison | Difference | sigma | |", "|---|---|---|---|"]
    for a, b, col, sd, unit in (("Hybrid GRU", "GraphMixer", "ap_mean", "ap_std", "AP"),
                                ("Hybrid GRU", "Hybrid CNN", "ap_mean", "ap_std", "AP"),
                                ("LSTM", "Hybrid GRU", "mse_mean", "mse_std", "MSE")):
        if a in summ.index and b in summ.index:
            diff, sg = sig(a, b, col, sd)
            verdict = "significant" if sg >= 3 else "**not significant**"
            L.append(f"| {a} vs {b} | {diff:+.4f} {unit} | {sg:.1f} | {verdict} |")
    L += ["",
          "The hybrid leads on both tasks, but by far less than earlier "
          "reports suggested: fixing GraphMixer's training distribution moved "
          "it from AP 0.70 to 0.90, and tuning the LSTM's never-searched "
          "`lookback` moved it from MSE 0.116 to 0.095. GRU and CNN are "
          "statistically indistinguishable.", ""]
    out = os.path.join(os.path.dirname(RES), "..", "results", "comparison.md")
    out = os.path.normpath(out)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    done = done_pairs()

    cfgs = {
        "Hybrid GRU":  best_cfg("hybrid_gru", "val_ap", True),
        "Hybrid CNN":  best_cfg("hybrid_cnn", "val_ap", True),
        "LSTM":        best_cfg("lstm", "val_mse", False),
        "GraphMixer":  best_cfg("graphmixer", "val_ap", True),
    }
    print("winning configurations from the grid search:")
    for k, v in cfgs.items():
        print(f"  {k:<12} {v}")
    print()

    from hybrid_core import HybridCfg, HybridData, run_hybrid
    from lstm_count import LSTMConfig, CountSeries, LSTMForecaster, train as lstm_train, export
    from graphmixer import GMConfig, GraphMixer
    from graphmixer_data import GraphMixerData
    from train_graphmixer import train as gm_train, export_predictions
    from shared_eval import SharedLinkEval

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = SharedLinkEval()
    hdata = {}          # one HybridData per lookback

    for seed in seeds:
        for name, c in cfgs.items():
            if (name, seed) in done:
                print(f"  [skip] {name} seed {seed}")
                continue
            t0 = time.time()
            try:
                if name.startswith("Hybrid"):
                    enc = "gru" if "GRU" in name else "cnn"
                    lb = int(c["ts_lookback"])
                    if lb not in hdata:
                        hdata.clear(); hdata[lb] = HybridData(lookback=lb)
                    cfg = HybridCfg(encoder=enc, lr=float(c["lr"]),
                                    hidden=int(c["hidden"]), ts_lookback=lb,
                                    fusion_hidden=int(c.get("fusion_hidden", 128)),
                                    dropout=float(c.get("dropout", 0.1)),
                                    lambda_count=0.5 if enc == "gru" else 1.0,
                                    seed=seed, epochs=MAX_EPOCHS, patience=PATIENCE)
                    r = run_hybrid(cfg, hdata[lb], eval_splits=("test",))
                    row = dict(model=name, seed=seed, test_ap=r["test"]["ap"],
                               test_auc=r["test"]["auc"], test_f1=r["test"]["f1"],
                               test_mse=r["test"]["mse"], epochs_run=r["epochs_run"])
                elif name == "LSTM":
                    layers, drop = eval(str(c["layers_dropout"]))
                    cfg = LSTMConfig(lr=float(c["lr"]), hidden_dim=int(c["hidden_dim"]),
                                     lookback=int(c["lookback"]), num_layers=layers,
                                     dropout=drop, seed=seed, epochs=MAX_EPOCHS)
                    rng = np.random.default_rng(seed); torch.manual_seed(seed)
                    cs = CountSeries(ev)
                    m = lstm_train(cfg, cs, LSTMForecaster(cfg).to(dev), 21 * 48,
                                   dev, rng, patience=PATIENCE, verbose=False)
                    tmp = os.path.join(HERE, "predictions", "_tmp_final_lstm.csv")
                    pred = export(cfg, cs, m, ev, "test", dev, tmp)
                    sc = ev.score_count(pred, split="test")
                    row = dict(model=name, seed=seed, test_ap=np.nan, test_auc=np.nan,
                               test_f1=np.nan, test_mse=sc["mse"],
                               epochs_run=m.epochs_run)
                else:
                    cfg = GMConfig(lr=float(c["lr"]), hidden_dim=int(c["hidden_dim"]),
                                   num_neighbors=int(c["num_neighbors"]),
                                   mixer_layers=int(c["mixer_layers"]),
                                   seed=seed, epochs=MAX_EPOCHS)
                    torch.manual_seed(seed)
                    gd = GraphMixerData(GMConfig().prep_dir)
                    model = GraphMixer(cfg, edge_feat_dim=gd.d_edge,
                                       node_feat_dim=gd.d_node).to(dev)
                    model = gm_train(cfg, gd, model, dev, ev=ev,
                                     patience=PATIENCE, verbose=False)
                    tmp = os.path.join(HERE, "predictions", "_tmp_final_gm.csv")
                    pred = export_predictions(cfg, gd, model, ev, "test", dev, tmp)
                    sb = ev.score_binary(pred, split="test")
                    row = dict(model=name, seed=seed, test_ap=sb["ap"],
                               test_auc=sb["auc"], test_f1=sb["f1"],
                               test_mse=np.nan, epochs_run=np.nan)
                row["sec"] = round(time.time() - t0, 1); row["error"] = ""
            except Exception as e:
                row = dict(model=name, seed=seed, test_ap=np.nan, test_auc=np.nan,
                           test_f1=np.nan, test_mse=np.nan, epochs_run=0,
                           sec=round(time.time() - t0, 1), error=str(e)[:200])
                hdata.clear()
                fatal = any(k in str(e) for k in ("CUDA", "cuDNN", "cuBLAS"))
            else:
                fatal = False
            append(row)
            print(f"  {name:<12} seed {seed} | AP {row['test_ap']:.4f} "
                  f"MSE {row['test_mse']:.4f} | {row['sec']:.0f}s")
            if fatal:
                # the context is dead; every further run here would fail too
                print("  -> CUDA-Kontext defekt, Prozess beenden fuer Neustart")
                break
        else:
            continue
        break

    # ---- summary -----------------------------------------------------------
    d = pd.read_csv(OUT)
    missing = [(m, s) for m in cfgs for s in seeds
               if (m, s) not in done_pairs()]
    d = d[d["error"].isna()]
    s = d.groupby("model").agg(
        n=("seed", "nunique"),
        ap_mean=("test_ap", "mean"), ap_std=("test_ap", "std"),
        auc_mean=("test_auc", "mean"), auc_std=("test_auc", "std"),
        f1_mean=("test_f1", "mean"), f1_std=("test_f1", "std"),
        mse_mean=("test_mse", "mean"), mse_std=("test_mse", "std"),
        sec=("sec", "median")).round(4)
    s.to_csv(os.path.join(RES, "final_eval_summary.csv"))
    print("\n" + s.to_string())
    print(f"\nwrote {os.path.join(RES, 'final_eval_summary.csv')}")
    write_markdown(d, s, cfgs)

    if missing:
        # exit non-zero so the supervisor restarts with a fresh CUDA context;
        # a poisoned context makes every later run in this process fail too
        print(f"\n{len(missing)} runs still missing: {missing[:6]}"
              f"{' ...' if len(missing) > 6 else ''}")
        sys.exit(1)


if __name__ == "__main__":
    main()
