# -*- coding: utf-8 -*-
"""
eval_factors.py — separating the grid-search effect from the seed effect.
==========================================================================

The grid search ran one seed per configuration, so every config's score also
contains that one random draw. The seed sweep did the opposite: many seeds, but
only for the selected configs. Neither number isolates its own factor, and the
grid spread reported earlier is therefore inflated by seed noise.

This runs a proper two-factor design — 4 configurations spanning the grid range
× 5 seeds each — and separates the two sources of variation:

  sigma_seed : spread caused by the random seed alone
               (pooled within-configuration std)
  sigma_grid : spread caused by the hyperparameters alone
               (between-configuration std, corrected for the seed noise that
                still sits in each configuration mean)

Correction used: var(config means) = sigma_grid^2 + sigma_seed^2 / n_seeds,
so sigma_grid^2 = var(means) - sigma_seed^2 / n_seeds  (clipped at 0).

The configurations are taken from the existing grid logs at ranks
1, ~1/3, ~2/3 and last, so they span the observed range rather than clustering.

Usage:
  python eval_factors.py --models hybrid          # ~17 min
  python eval_factors.py --models lstm            # ~50 min
  python eval_factors.py --models gm              # ~110 min
  python eval_factors.py --models all
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
RAW_CSV = os.path.join(RESULTS_DIR, "factors_raw.csv")
SEEDS = [42, 43, 44, 45, 46]

# --- configurations spanning each grid (rank 1 / ~1/3 / ~2/3 / last) ---------
HYBRID_CFGS = [   # (tag, lr, hidden, lambda_count)
    ("rank01_best",  1e-3, 128, 0.5),
    ("rank10",       1e-3,  32, 2.0),
    ("rank19",       1e-4, 128, 0.5),
    ("rank27_worst", 1e-4,  32, 2.0),
]
LSTM_CFGS = [     # (tag, lr, hidden_dim, num_layers)
    ("rank01_best",  3e-4,  32, 1),
    ("rank07",       1e-4, 128, 1),
    ("rank13",       3e-4,  32, 2),
    ("rank18_worst", 1e-4,  32, 1),
]
GM_CFGS = [       # (tag, lr, hidden_dim, mixer_layers)
    ("rank01_best",  1e-4,  64, 1),
    ("rank07",       1e-3, 256, 2),
    ("rank13",       3e-4, 128, 2),
    ("rank18_worst", 1e-3, 128, 1),
]


def _load_done() -> set:
    if not os.path.exists(RAW_CSV):
        return set()
    d = pd.read_csv(RAW_CSV)
    return {(r["model"], r["config"], int(r["seed"])) for _, r in d.iterrows()}


def _append(row: dict) -> None:
    new = pd.DataFrame([row])
    if os.path.exists(RAW_CSV):
        new = pd.concat([pd.read_csv(RAW_CSV), new], ignore_index=True)
    new.to_csv(RAW_CSV, index=False)


# ===========================================================================
# runners — each returns the grid's selection metric (val) plus the test value
# ===========================================================================
def run_hybrid(cfg_spec, seed, shared):
    import hybrid_core as hc
    _tag, lr, hidden, lam = cfg_spec
    cfg = hc.HybridCfg(seed=seed, lr=lr, hidden=hidden,
                       fusion_hidden=2 * hidden, lambda_count=lam)
    r = hc.run_hybrid(cfg, shared, eval_splits=("val", "test"))
    return {"val_metric": r["val"]["ap"], "test_metric": r["test"]["ap"]}


def run_lstm(cfg_spec, seed, shared):
    from lstm_count import LSTMConfig, LSTMForecaster, train, export
    _tag, lr, hid, nl = cfg_spec
    ev, cs, train_end_bin, device = shared
    cfg = LSTMConfig(seed=seed, lr=lr, hidden_dim=hid, num_layers=nl)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = train(cfg, cs, LSTMForecaster(cfg).to(device), train_end_bin, device, rng)
    out = {}
    for split, key in [("val", "val_metric"), ("test", "test_metric")]:
        tmp = os.path.join(_HERE, "predictions", f"_tmp_fac_lstm_{split}.csv")
        out[key] = ev.score_count(export(cfg, cs, model, ev, split, device, tmp),
                                  split=split)["mse"]
    return out


def run_gm(cfg_spec, seed, shared):
    from graphmixer import GraphMixer, GMConfig
    from train_graphmixer import train as gm_train, export_predictions
    _tag, lr, hid, ml = cfg_spec
    ev, data, device, epochs = shared
    cfg = GMConfig(seed=seed, epochs=epochs, lr=lr, hidden_dim=hid, mixer_layers=ml)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    model = GraphMixer(cfg, edge_feat_dim=data.d_edge, node_feat_dim=data.d_node).to(device)
    model = gm_train(cfg, data, model, device)
    out = {}
    for split, key in [("val", "val_metric"), ("test", "test_metric")]:
        tmp = os.path.join(_HERE, "predictions", f"_tmp_fac_gm_{split}.csv")
        out[key] = ev.score_binary(export_predictions(cfg, data, model, ev, split,
                                                      device, tmp), split=split)["ap"]
    return out


def sweep(model_name, cfgs, runner, shared, seeds, done):
    for spec in cfgs:
        tag = spec[0]
        print(f"\n--- {model_name}: {tag} {spec[1:]} ---")
        for s in seeds:
            if (model_name, tag, s) in done:
                print(f"  seed {s}: skipped"); continue
            try:
                t0 = time.time()
                m = runner(spec, s, shared)
                _append({"model": model_name, "config": tag, "seed": s,
                         "params": str(spec[1:]), **m})
                print(f"  seed {s}: val={m['val_metric']:.4f} "
                      f"test={m['test_metric']:.4f}  ({time.time()-t0:.0f}s)")
            except Exception as e:                          # noqa: BLE001
                print(f"  seed {s}: FAILED ({type(e).__name__}: {e})")


# ===========================================================================
# variance decomposition
# ===========================================================================
def decompose(df: pd.DataFrame, model: str, col: str = "val_metric") -> dict:
    sub = df[df["model"] == model]
    groups = [g[col].values for _, g in sub.groupby("config")]
    n_seeds = int(np.mean([len(g) for g in groups]))
    # seed effect: pooled within-config variance
    var_seed = float(np.mean([g.var(ddof=1) for g in groups if len(g) > 1]))
    sigma_seed = np.sqrt(var_seed)
    # grid effect: between-config variance, minus the seed noise still in each mean
    means = np.array([g.mean() for g in groups])
    var_means = float(means.var(ddof=1))
    sigma_grid = np.sqrt(max(var_means - var_seed / max(n_seeds, 1), 0.0))
    return {"model": model, "n_configs": len(groups), "n_seeds": n_seeds,
            "sigma_seed": sigma_seed, "sigma_grid": sigma_grid,
            "ratio_grid_over_seed": (sigma_grid / sigma_seed) if sigma_seed > 0 else float("nan"),
            "range_config_means": float(means.max() - means.min()),
            "best_mean": float(means.max()), "worst_mean": float(means.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="hybrid")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--gm_epochs", type=int, default=20)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    which = ["hybrid", "lstm", "gm"] if args.models == "all" else \
            [m.strip() for m in args.models.split(",")]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(_HERE, "predictions"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | seeds {seeds} | models {which}")
    done = _load_done()
    if done:
        print(f"resuming: {len(done)} runs already recorded")
    t0 = time.time()

    if "hybrid" in which:
        import hybrid_core as hc
        sweep("Hybrid", HYBRID_CFGS, run_hybrid, hc.HybridData(), seeds, done)
    if "lstm" in which:
        from shared_eval import SharedLinkEval
        from lstm_count import CountSeries
        ev = SharedLinkEval(); cs = CountSeries(ev)
        teb = ev.cfg.train_days * ((24 * 60) // ev.cfg.bin_minutes)
        sweep("LSTM", LSTM_CFGS, run_lstm, (ev, cs, teb, device), seeds, done)
    if "gm" in which:
        from shared_eval import SharedLinkEval, EvalConfig
        from graphmixer import GMConfig
        from graphmixer_data import GraphMixerData
        base = GMConfig()
        gdata = GraphMixerData(base.prep_dir)
        gev = SharedLinkEval(EvalConfig(bin_minutes=base.bin_minutes,
                                        train_days=base.train_days,
                                        val_days=base.val_days))
        sweep("GraphMixer", GM_CFGS, run_gm, (gev, gdata, device, args.gm_epochs),
              seeds, done)

    df = pd.read_csv(RAW_CSV)
    rows = [decompose(df, m) for m in df["model"].unique()]
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(RESULTS_DIR, "factors_summary.csv"), index=False)
    print("\n" + "=" * 78)
    print("GRID EFFECT vs. SEED EFFECT  (validation metric, 4 configs x N seeds)")
    print("=" * 78)
    for r in rows:
        print(f"{r['model']:11s} sigma_seed={r['sigma_seed']:.4f}  "
              f"sigma_grid={r['sigma_grid']:.4f}  "
              f"ratio={r['ratio_grid_over_seed']:.1f}x  "
              f"range(config means)={r['range_config_means']:.4f}")
    print(f"\ntotal {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
