# -*- coding: utf-8 -*-
"""
eval_seeds.py — multi-seed robustness check for the final configurations.
=========================================================================

Everything else in this study was run with a single seed, so none of the
reported differences had an error bar. This script re-runs only the FINAL
selected configs across several seeds and reports mean +/- std, which gives
the yardstick the rest of the study was missing: a difference is only
meaningful if it is larger than the spread caused by the random seed alone.

Important: only the MODEL seed varies (weight init + batch shuffling). The
evaluation seed in EvalConfig stays at 42, so every run is scored on exactly
the same candidate set. Otherwise model variance and test-set variance would
be mixed together.

Configs covered (the ones the grids selected):
  hybrid  : GRU default (h64, lam 1.0) | GRU HPO (h128, lam 0.5) | CNN HPO (h128, k3)
  lstm    : default (lr 1e-3, h64, L1) | HPO (lr 3e-4, h32, L1)
  gm      : default                    | HPO (lr 1e-4, h64, L1)

Usage:
  python eval_seeds.py --models hybrid            # fast (~15 min)
  python eval_seeds.py --models lstm,gm           # slower, run in background
  python eval_seeds.py --models all --seeds 5
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "evaluation"))
sys.path.insert(0, os.path.join(_HERE, "..", "lstm"))
sys.path.insert(0, os.path.join(_HERE, "..", "graphmixer", "model"))

RESULTS_DIR = os.path.join(_HERE, "results")
SEEDS = [42, 43, 44, 45, 46]

# One shared raw file so partial progress survives a crash and can be resumed.
# (A long GPU sweep occasionally dies on a transient CUDA fault; losing an hour
# of finished runs to that is avoidable.)
RAW_CSV = os.path.join(RESULTS_DIR, "seeds_raw.csv")
SUMM_CSV = os.path.join(RESULTS_DIR, "seeds_summary.csv")


# ===========================================================================
# per-model runners: each returns one dict of metrics for a given seed
# ===========================================================================
def run_hybrid_seed(variant: str, seed: int, data) -> dict:
    import hybrid_core as hc
    cfgs = {
        "GRU default": hc.HybridCfg(seed=seed),
        "GRU HPO":     hc.HybridCfg(seed=seed, hidden=128, fusion_hidden=256,
                                    lambda_count=0.5),
        "CNN HPO":     hc.HybridCfg(seed=seed, encoder="cnn", hidden=128,
                                    fusion_hidden=256, kernel_size=3),
    }
    r = hc.run_hybrid(cfgs[variant], data, eval_splits=("val", "test"))
    return {"val_ap": r["val"]["ap"], "val_auc": r["val"]["auc"],
            "test_ap": r["test"]["ap"], "test_auc": r["test"]["auc"],
            "test_f1": r["test"]["f1"], "test_mse": r["test"]["mse"],
            "test_mae": r["test"]["mae"]}


def run_lstm_seed(variant: str, seed: int, shared) -> dict:
    from lstm_count import LSTMConfig, LSTMForecaster, train, export
    ev, cs, train_end_bin, device = shared
    cfg = (LSTMConfig(seed=seed) if variant == "default"
           else LSTMConfig(seed=seed, lr=3e-4, hidden_dim=32, num_layers=1))
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = LSTMForecaster(cfg).to(device)
    model = train(cfg, cs, model, train_end_bin, device, rng)
    out = {}
    for split in ["val", "test"]:
        tmp = os.path.join(_HERE, "predictions", f"_tmp_seed_lstm_{split}.csv")
        res = ev.score_count(export(cfg, cs, model, ev, split, device, tmp), split=split)
        out[f"{split}_mse"] = res["mse"]; out[f"{split}_mae"] = res["mae"]
    return out


def run_gm_seed(variant: str, seed: int, shared) -> dict:
    from graphmixer import GraphMixer, GMConfig
    from train_graphmixer import train as gm_train, export_predictions
    ev, data, device, epochs = shared
    cfg = (GMConfig(seed=seed, epochs=epochs) if variant == "default"
           else GMConfig(seed=seed, epochs=epochs, lr=1e-4, hidden_dim=64, mixer_layers=1))
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    model = GraphMixer(cfg, edge_feat_dim=data.d_edge, node_feat_dim=data.d_node).to(device)
    model = gm_train(cfg, data, model, device)
    out = {}
    for split in ["val", "test"]:
        tmp = os.path.join(_HERE, "predictions", f"_tmp_seed_gm_{split}.csv")
        res = ev.score_binary(export_predictions(cfg, data, model, ev, split, device, tmp),
                              split=split)
        out[f"{split}_ap"] = res["ap"]; out[f"{split}_auc"] = res["auc"]
        out[f"{split}_f1"] = res["f1"]
    return out


# ===========================================================================
def _load_done() -> set:
    """(model, variant, seed) triples already present in the raw file."""
    if not os.path.exists(RAW_CSV):
        return set()
    d = pd.read_csv(RAW_CSV)
    return {(r["model"], r["variant"], int(r["seed"])) for _, r in d.iterrows()}


def _append(row: dict) -> None:
    """Write one finished run immediately, so a crash costs at most one run.

    Rewrites the whole file instead of appending a line: the models report
    different metric keys (AP/AUC for the binary ones, MSE/MAE for the count
    ones), and a plain append would write each dict in its own column order and
    silently shift values into the wrong columns. The file is tiny, so a full
    rewrite is the cheap and safe option.
    """
    new = pd.DataFrame([row])
    if os.path.exists(RAW_CSV):
        new = pd.concat([pd.read_csv(RAW_CSV), new], ignore_index=True)
    new.to_csv(RAW_CSV, index=False)


def sweep(model_name: str, variants: list[str], runner, shared, seeds,
          done: set, retries: int = 1) -> None:
    for variant in variants:
        print(f"\n--- {model_name}: {variant} ---")
        for s in seeds:
            if (model_name, variant, s) in done:
                print(f"  seed {s}: skipped (already in {os.path.basename(RAW_CSV)})")
                continue
            for attempt in range(retries + 1):
                try:
                    t0 = time.time()
                    m = runner(variant, s, shared)
                    _append({"model": model_name, "variant": variant, "seed": s, **m})
                    head = ", ".join(f"{k}={v:.4f}" for k, v in list(m.items())[:3])
                    print(f"  seed {s}: {head}  ({time.time()-t0:.0f}s)")
                    break
                except Exception as e:                      # noqa: BLE001
                    kind = type(e).__name__
                    if attempt < retries:
                        print(f"  seed {s}: {kind} — retrying once ({e})")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        time.sleep(5)
                    else:
                        print(f"  seed {s}: FAILED after retry ({kind}: {e}) — skipping")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c not in ("model", "variant", "seed")]
    g = df.groupby(["model", "variant"])[metric_cols]
    out = g.agg(["mean", "std"]).round(4)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="hybrid", help="hybrid,lstm,gm or all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--gm_epochs", type=int, default=20)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    which = ["hybrid", "lstm", "gm"] if args.models == "all" else \
            [m.strip() for m in args.models.split(",")]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(_HERE, "predictions"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | seeds: {seeds} | models: {which}")

    done = _load_done()
    if done:
        print(f"resuming: {len(done)} runs already recorded")
    t_all = time.time()

    if "hybrid" in which:
        import hybrid_core as hc
        data = hc.HybridData()
        sweep("Hybrid", ["GRU default", "GRU HPO", "CNN HPO"],
              run_hybrid_seed, data, seeds, done)

    if "lstm" in which:
        from shared_eval import SharedLinkEval
        from lstm_count import CountSeries
        ev = SharedLinkEval(); cs = CountSeries(ev)
        teb = ev.cfg.train_days * ((24 * 60) // ev.cfg.bin_minutes)
        # after the training-distribution fix the grid picks lr1e-3/h64/L1,
        # which is the default - so "HPO" would be the same run twice
        sweep("LSTM", ["default"], run_lstm_seed,
              (ev, cs, teb, device), seeds, done)

    if "gm" in which:
        from shared_eval import SharedLinkEval, EvalConfig
        from graphmixer import GMConfig
        from graphmixer_data import GraphMixerData
        base = GMConfig()
        gdata = GraphMixerData(base.prep_dir)
        gev = SharedLinkEval(EvalConfig(bin_minutes=base.bin_minutes,
                                        train_days=base.train_days,
                                        val_days=base.val_days))
        sweep("GraphMixer", ["default", "HPO"], run_gm_seed,
              (gev, gdata, device, args.gm_epochs), seeds, done)

    if not os.path.exists(RAW_CSV):
        print("no results recorded — nothing to summarize")
        return
    df = pd.read_csv(RAW_CSV)
    summ = summarize(df)
    summ.to_csv(SUMM_CSV, index=False)
    raw_csv, summ_csv = RAW_CSV, SUMM_CSV

    print("\n" + "=" * 70)
    print(f"SUMMARY (mean +/- std over {len(seeds)} seeds)")
    print("=" * 70)
    for _, r in summ.iterrows():
        parts = []
        for c in summ.columns:
            if c.endswith("_mean"):
                base = c[:-5]
                parts.append(f"{base}={r[c]:.4f}+/-{r.get(base+'_std', float('nan')):.4f}")
        print(f"{r['model']:11s} {r['variant']:12s} " + "  ".join(parts))
    print(f"\nwritten: {raw_csv}\n         {summ_csv}")
    print(f"total {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
