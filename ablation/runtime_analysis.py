# -*- coding: utf-8 -*-
"""
runtime_analysis.py - how long the models take, and what you get for it.

Two parts:

  Phase A (--phase a, no new compute)
      Re-uses the `sec` column already recorded in the grid CSVs (81 configs).
      Within one model the epoch count and implementation are constant, so this
      shows which hyperparameter drives cost. Across models the raw totals are
      NOT comparable (different epoch counts, different implementations), which
      is why phase B exists.

  Phase B (--phase b, ~70-90 min)
      Controlled measurement of the four best configs, 5 seeds each:
      train time per epoch, inference time on the test candidates, peak GPU
      memory and parameter count.

Why medians and not means: a single run on this laptop can be off by a large
factor. The LSTM grid once recorded 3849 s for a config that takes ~58 s when
repeated - a thermal/background artefact, not a property of the config. Means
would carry that straight into the slides.

Usage
    python runtime_analysis.py --phase a
    python runtime_analysis.py --phase b --seeds 5
"""
import os
import sys
import json
import time
import argparse

# anaconda ships two OpenMP runtimes; numpy+torch in one process trips over it
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_HERE, "results")
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "evaluation"))
sys.path.insert(0, os.path.join(_HERE, "..", "lstm"))
sys.path.insert(0, os.path.join(_HERE, "..", "graphmixer", "model"))

RAW_B = os.path.join(RES, "runtime_controlled.csv")

# epochs per model - needed to make the grid totals comparable
EPOCHS = {"Hybrid GRU": 15, "Hybrid CNN": 15, "LSTM": 10, "GraphMixer": 20}

GRIDS = [
    ("Hybrid GRU",  "grid_hybrid_gru.csv",  ["lr", "hidden", "lambda_count"], "val_ap",  True),
    ("Hybrid CNN",  "grid_hybrid_cnn.csv",  ["lr", "hidden", "kernel_size"],  "val_ap",  True),
    ("LSTM",        "grid_lstm.csv",        ["lr", "hidden_dim", "num_layers"], "val_mse", False),
    ("GraphMixer",  "grid_graphmixer.csv",  ["lr", "hidden_dim", "mixer_layers"], "val_ap", True),
]


# ===========================================================================
# Phase A - what the existing grid runs already tell us
# ===========================================================================
def phase_a():
    rows, per_hp = [], []
    for name, fn, hps, metric, higher_better in GRIDS:
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            print(f"  [skip] {fn} missing")
            continue
        d = pd.read_csv(p)
        ep = EPOCHS[name]
        rows.append(dict(
            model=name, n_configs=len(d), epochs=ep,
            sec_median=d.sec.median(), sec_min=d.sec.min(), sec_max=d.sec.max(),
            sec_per_epoch_median=d.sec.median() / ep,
            spread_factor=d.sec.max() / d.sec.min(),
            best_metric=(d[metric].max() if higher_better else d[metric].min()),
        ))
        # Runtimes drift upward over a grid (the laptop GPU throttles), so a
        # hyperparameter whose levels were swept in blocks picks that drift up
        # as a fake effect. Report the raw ratio, but also the ratio measured
        # *within* run-order blocks, and how strongly the parameter is
        # confounded with position in the grid.
        drift = d.index.to_series().corr(d.sec)
        rows[-1]["drift_corr"] = drift
        for hp in hps:
            if hp not in d.columns:
                continue
            g = d.groupby(hp)["sec"].median()
            conf = abs(d[hp].rank(method="dense").corr(d.index.to_series()))
            # within-block: compare levels only among runs that sit close together
            blocks = pd.qcut(d.index, q=min(3, d[hp].nunique()), labels=False, duplicates="drop")
            within = []
            for _, sub in d.groupby(blocks):
                gg = sub.groupby(hp)["sec"].median()
                if len(gg) > 1:
                    within.append(gg.max() / gg.min())
            per_hp.append(dict(model=name, hyperparam=hp,
                               levels="; ".join(f"{k}:{v:.0f}s" for k, v in g.items()),
                               ratio_raw=g.max() / g.min(),
                               ratio_within_block=(np.median(within) if within else np.nan),
                               confounded_with_order=conf))

    summary = pd.DataFrame(rows)
    hp_tab = pd.DataFrame(per_hp)
    summary.to_csv(os.path.join(RES, "runtime_grid_summary.csv"), index=False)
    hp_tab.to_csv(os.path.join(RES, "runtime_hyperparams.csv"), index=False)

    print("\n=== Phase A: grid runtimes (existing runs) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\n  drift_corr = correlation between run order and runtime.")
    print("  High values mean the machine slowed down over the grid, so raw")
    print("  per-hyperparameter ratios below are not causal on their own.")
    print("\n=== Which hyperparameter drives the runtime? ===")
    print(hp_tab.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n  ratio_within_block controls for the drift; trust it over ratio_raw")
    print("  when confounded_with_order is high.")
    print(f"\n[written] runtime_grid_summary.csv, runtime_hyperparams.csv")
    return summary, hp_tab


# ===========================================================================
# Phase B - controlled measurement
# ===========================================================================
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _peak_mb():
    return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else float("nan")


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


def measure_hybrid(encoder, seed, shared):
    """encoder: 'gru' or 'cnn' - best config from the grid."""
    from hybrid_core import HybridCfg, run_hybrid, _predict
    data, device = shared
    from final_eval import best_cfg
    c = best_cfg(f"hybrid_{encoder}", "val_ap", True)
    cfg = HybridCfg(seed=seed, encoder=encoder, lr=float(c["lr"]),
                    hidden=int(c["hidden"]), ts_lookback=int(c["ts_lookback"]),
                    fusion_hidden=int(c.get("fusion_hidden", 128)),
                    dropout=float(c.get("dropout", 0.1)),
                    lambda_count=0.5 if encoder == "gru" else 1.0,
                    epochs=30, patience=3)

    _reset_mem(); _sync()
    t0 = time.time()
    # eval_splits=() -> pure training; scoring is timed separately below
    res = run_hybrid(cfg, data, eval_splits=(), export_dir=None, return_model=True)
    _sync()
    t_train = time.time() - t0
    model = res["model"]

    # inference: forward pass over the test candidates (no CSV write)
    _sync(); t0 = time.time()
    pred = _predict(model, data, cfg, "test")
    _sync()
    t_infer = time.time() - t0

    return dict(t_train_total=t_train, t_train_epoch=t_train / cfg.epochs,
                t_infer=t_infer, n_cand=len(pred), peak_mb=_peak_mb(),
                n_params=_n_params(model))


def measure_lstm(seed, shared):
    from lstm_count import LSTMConfig, LSTMForecaster, train
    ev, cs, teb, device = shared
    from final_eval import best_cfg
    c = best_cfg("lstm", "val_mse", False)
    layers, drop = eval(str(c["layers_dropout"]))
    cfg = LSTMConfig(seed=seed, lr=float(c["lr"]), hidden_dim=int(c["hidden_dim"]),
                     lookback=int(c["lookback"]), num_layers=layers,
                     dropout=drop, epochs=30)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    _reset_mem(); _sync()
    model = LSTMForecaster(cfg).to(device)
    t0 = time.time()
    model = train(cfg, cs, model, teb, device, rng, patience=3, verbose=False)
    _sync()
    t_train = time.time() - t0

    # inference: build windows + forward, no CSV write
    cand = ev.build_candidates("test")[["u", "i", "bin_idx"]]
    u = cand["u"].to_numpy(); i = cand["i"].to_numpy(); b = cand["bin_idx"].to_numpy()
    _sync(); t0 = time.time()
    X = np.zeros((len(cand), cfg.lookback), dtype=np.float32)
    for k in range(len(cand)):
        X[k] = cs.window(int(u[k]), int(i[k]), int(b[k]), cfg.lookback)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(X), 8192):
            xb = torch.from_numpy(X[s:s + 8192]).unsqueeze(-1).to(device)
            model(xb)
    _sync()
    t_infer = time.time() - t0

    return dict(t_train_total=t_train, t_train_epoch=t_train / cfg.epochs,
                t_infer=t_infer, n_cand=len(cand), peak_mb=_peak_mb(),
                n_params=_n_params(model))


def measure_gm(seed, shared):
    from graphmixer import GMConfig, GraphMixer
    from train_graphmixer import train as gm_train, export_predictions
    data, gev, device = shared
    from final_eval import best_cfg
    c = best_cfg("graphmixer", "val_ap", True)
    cfg = GMConfig(seed=seed, lr=float(c["lr"]), hidden_dim=int(c["hidden_dim"]),
                   num_neighbors=int(c["num_neighbors"]),
                   mixer_layers=int(c["mixer_layers"]), epochs=30)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    _reset_mem(); _sync()
    model = GraphMixer(cfg, edge_feat_dim=data.d_edge,
                       node_feat_dim=data.d_node).to(device)
    t0 = time.time()
    model = gm_train(cfg, data, model, device, ev=gev, patience=3, verbose=False)
    _sync()
    t_train = time.time() - t0

    # export_predictions also writes a CSV; the write is a few MB and small
    # next to the forward pass, but it means this number is an upper bound.
    tmp = os.path.join(_HERE, "predictions", "_tmp_rt_gm.csv")
    _sync(); t0 = time.time()
    pred = export_predictions(cfg, data, model, gev, "test", device, tmp)
    _sync()
    t_infer = time.time() - t0

    return dict(t_train_total=t_train, t_train_epoch=t_train / cfg.epochs,
                t_infer=t_infer, n_cand=len(pred), peak_mb=_peak_mb(),
                n_params=_n_params(model))


def phase_b(seeds):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    done = set()
    if os.path.exists(RAW_B):
        prev = pd.read_csv(RAW_B)
        done = {(r.model, int(r.seed)) for r in prev.itertuples()}
        print(f"resuming: {len(done)} runs already recorded")

    def record(row):
        df = pd.DataFrame([row])
        if os.path.exists(RAW_B):
            old = pd.read_csv(RAW_B)
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(RAW_B, index=False)

    # --- shared data, built once -------------------------------------------
    from hybrid_core import HybridData
    from final_eval import best_cfg
    # both hybrid variants won at the same lookback; HybridData asserts that
    # cfg.ts_lookback matches, so it has to be built for that value
    _lb = int(best_cfg("hybrid_gru", "val_ap", True)["ts_lookback"])
    _lb_cnn = int(best_cfg("hybrid_cnn", "val_ap", True)["ts_lookback"])
    assert _lb == _lb_cnn, (
        f"GRU and CNN won at different lookbacks ({_lb} vs {_lb_cnn}); "
        "the shared HybridData below would need splitting per encoder.")
    hyb_data = HybridData(device=device, lookback=_lb)
    print(f"HybridData lookback={_lb} (aus der Gittersuche)")

    from shared_eval import SharedLinkEval
    from lstm_count import CountSeries
    ev = SharedLinkEval(); cs = CountSeries(ev)
    teb = ev.cfg.train_days * ((24 * 60) // ev.cfg.bin_minutes)

    from graphmixer import GMConfig
    from graphmixer_data import GraphMixerData
    from shared_eval import EvalConfig
    base = GMConfig()
    gm_data = GraphMixerData(base.prep_dir)
    gev = SharedLinkEval(EvalConfig(bin_minutes=base.bin_minutes,
                                    train_days=base.train_days,
                                    val_days=base.val_days))

    jobs = [
        ("Hybrid GRU",  lambda s: measure_hybrid("gru", s, (hyb_data, device))),
        ("Hybrid CNN",  lambda s: measure_hybrid("cnn", s, (hyb_data, device))),
        ("LSTM",        lambda s: measure_lstm(s, (ev, cs, teb, device))),
        ("GraphMixer",  lambda s: measure_gm(s, (gm_data, gev, device))),
    ]

    # One warm-up before anything is recorded: the first CUDA call of a process
    # pays context initialisation, which would otherwise land on whichever model
    # happens to run first.
    print("\nwarm-up run (not recorded) ...")
    try:
        jobs[0][1](seeds[0])
    except Exception as e:
        print(f"  warm-up failed: {type(e).__name__}: {e}")

    # Models are interleaved *within* each seed rather than run in blocks.
    # The grid CSVs show runtime drifting upward over a long session
    # (r = 0.87 with run order on the hybrid grid - the GPU throttles). Running
    # model after model would hand that drift to whichever model goes last.
    order = 0
    for s in seeds:
        print(f"\n--- seed {s} ---")
        for name, fn in jobs:
            if (name, s) in done:
                print(f"  {name}: skipped (already recorded)")
                continue
            try:
                r = fn(s)
                r.update(model=name, seed=s, run_order=order)
                order += 1
                record(r)
                print(f"  {name:12s} train {r['t_train_total']:6.1f}s "
                      f"({r['t_train_epoch']:5.1f}s/ep) | infer {r['t_infer']:5.2f}s | "
                      f"{r['peak_mb']:6.0f} MB | {r['n_params']:,} params")
            except Exception as e:
                print(f"  {name}: FAILED ({type(e).__name__}: {e}) - skipping")

    summarize_b()


def summarize_b():
    if not os.path.exists(RAW_B):
        print("no controlled runs recorded yet")
        return
    d = pd.read_csv(RAW_B)
    g = d.groupby("model").agg(
        n=("seed", "count"),
        train_total_med=("t_train_total", "median"),
        train_epoch_med=("t_train_epoch", "median"),
        train_epoch_iqr=("t_train_epoch", lambda x: x.quantile(.75) - x.quantile(.25)),
        infer_med=("t_infer", "median"),
        peak_mb_med=("peak_mb", "median"),
        n_params=("n_params", "max"),
        n_cand=("n_cand", "max"),
    ).reset_index()
    g["us_per_pair"] = g.infer_med / g.n_cand * 1e6
    g.to_csv(os.path.join(RES, "runtime_summary.csv"), index=False)
    print("\n=== Phase B: controlled measurement (medians over seeds) ===")
    print(g.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n[written] runtime_summary.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="a", choices=["a", "b", "summary"])
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    if a.phase == "a":
        phase_a()
    elif a.phase == "b":
        phase_b(list(range(42, 42 + a.seeds)))
    else:
        summarize_b()
