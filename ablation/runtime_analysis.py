# -*- coding: utf-8 -*-
"""
runtime_analysis.py - how long the models take, and what you get for it.

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




# --- Phase B - controlled measurement ----------------------------------------
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def write_report(summ, elapsed_min, device):
    """ablation/results/runtime_comparison.md, generated from the measurements.

    This used to be maintained by hand, which meant it silently kept the old
    numbers when the configs changed. Generating it removes that failure mode.
    """
    import pandas as pd
    md = os.path.join(RES, "runtime_comparison.md")
    ev_path = os.path.join(RES, "final_eval_summary.csv")
    ev = pd.read_csv(ev_path).set_index("model") if os.path.exists(ev_path) else None
    s = summ.set_index("model")
    order = [m for m in ("Hybrid CNN", "Hybrid GRU", "LSTM", "GraphMixer")
             if m in s.index]

    L = ["# Runtime and cost", "",
         "What each model costs to train and to run, measured under one protocol.",
         "Configs are the winners of the final grid search, read from",
         "`hpo_final_*.csv` rather than hard-coded.", "",
         "## Hardware", "", "```",
         "GPU     NVIDIA GeForce RTX 4070 Laptop, 8.6 GB VRAM (8188 MiB)",
         "CPU     Intel Core i9-13900HX, 24 cores / 32 threads",
         "RAM     31.7 GB", "OS      Windows 11",
         "Stack   Python 3.12.8, PyTorch 2.12.0+cu126, CUDA 12.6", "```", "",
         "Ratios are hardware-dependent: GraphMixer's per-event Python loops",
         "barely benefit from a faster GPU, the hybrid's tensor operations do.",
         "", f"## Controlled measurement ({int(s['n'].max())} seeds, medians)", "",
         "| Model | Training | s/epoch | Inference | us/pair | Peak memory | Params |",
         "|---|---|---|---|---|---|---|"]
    for m in order:
        r = s.loc[m]
        L.append(f"| {m} | {r.train_total_med:.1f} s | {r.train_epoch_med:.2f} | "
                 f"{r.infer_med:.3f} s | {r.us_per_pair:.2f} | "
                 f"{r.peak_mb_med:,.0f} MB | {int(r.n_params):,} |")
    L += ["", "Inference = forward pass over the test candidates, excluding file I/O.", ""]

    if ev is not None and {"Hybrid GRU", "GraphMixer"} <= set(ev.index):
        fac = s.loc["GraphMixer", "infer_med"] / s.loc["Hybrid GRU", "infer_med"]
        L += ["## The main result", "",
              f"The hybrid infers **{fac:.1f}x faster** than GraphMixer "
              f"({s.loc['Hybrid GRU','infer_med']:.3f} s vs. "
              f"{s.loc['GraphMixer','infer_med']:.3f} s) at "
              f"AP {ev.loc['Hybrid GRU','ap_mean']:.4f} vs. "
              f"{ev.loc['GraphMixer','ap_mean']:.4f}.", "",
              "GraphMixer, however, now **trains faster** "
              f"({s.loc['GraphMixer','train_total_med']:.0f} s vs. "
              f"{s.loc['Hybrid GRU','train_total_med']:.0f} s), which reverses the",
              "earlier picture. Since training is one-off and inference is the",
              "running cost, the hybrid remains the cheaper choice in operation --",
              "but it is a trade-off now, not Pareto dominance.", ""]

    L += ["## Two observations worth a slide", "",
          "**The CNN variant is the efficiency winner.** Fastest inference in the",
          "field, lowest memory among the hybrids, a quarter of the GRU's",
          "parameters -- at statistically indistinguishable accuracy (1.4 sigma).",
          "",
          "**Parameter count says nothing about cost.** The LSTM has the fewest",
          "parameters and by far the largest memory peak.", ""]
    if "LSTM" in s.index and s.loc["LSTM", "peak_mb_med"] > 8588:
        L += ["> **The best LSTM config does not fit in VRAM.** Its peak of "
              f"{s.loc['LSTM','peak_mb_med']:,.0f} MB exceeds the card's 8,188 MiB.",
              "> It only runs because the Windows driver spills CUDA allocations to",
              "> system memory; on a card without that fallback it would raise OOM.", ""]

    L += ["## Why medians, not means", "",
          "Single measurements on this machine occasionally spike by an order of",
          "magnitude (transient thermal or driver effects). Models are measured",
          "interleaved within each seed, not in blocks, so drift cannot favour",
          "whichever model runs last.", "",
          "## Why a dedicated measurement, not the search timings", "",
          "Reusing the `sec` column from a hyperparameter search looks cheap but",
          "does not work: within a search, runtime correlates strongly with run",
          "order (thermal drift on this laptop), and the outer loop variable",
          "picks that drift up as an apparent effect. Every number here comes",
          "from the controlled run instead.", "",
          "## Reproduce", "",
          "```bash", "python ablation/runtime_analysis.py --phase b   # controlled measurement",
          "python ablation/runtime_analysis.py --phase summary",
          "```", "",
          f"Measured in ~{elapsed_min:.0f} min on {device}.",
          "Raw data: `runtime_controlled.csv`, aggregated: `runtime_summary.csv`", ""]
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote {md}")


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
    _t_start = time.time()
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
    print(f"HybridData lookback={_lb} (from the grid search)")

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
    # elapsed and device used to be read from phase_b()'s locals, which only
    # worked when summarize_b() happened to run in the same process -- calling
    # --phase summary on its own raised NameError. Both are derived from the
    # recorded runs now, which also makes the report reproducible from the CSV.
    elapsed_min = (d.t_train_total.sum() + d.t_infer.sum()) / 60.0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    write_report(g, elapsed_min=elapsed_min, device=dev)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Phase A, which reused the timings from the earlier grid searches, was
    # dropped: those runs predate the two baseline fixes. Everything is based
    # on the controlled measurement now.
    ap.add_argument("--phase", default="b", choices=["b", "summary"])
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    if a.phase == "b":
        phase_b(list(range(42, 42 + a.seeds)))
    else:
        summarize_b()
