# -*- coding: utf-8 -*-
"""
eval_branches.py: component ablations for the hybrid model.
============================================================

Encoder swap (GRU ↔ CNN) only shows that the *encoding* is interchangeable.
It does not show whether the temporal branch, the graph branch, or the pair
features carry the signal. This script removes each component in turn by
zeroing its contribution (same parameter count, see HybridHurdle).

Variants:
  full         all branches on
  no_graph     use_graph=False
  no_temporal  use_temporal=False
  no_pair      use_pair=False

Runs both encoders, each on its own winner from the final grid search, so the
branch contributions are measured on the models the talk actually reports.
Configs are read from hpo_final_*.csv rather than hard-coded.

Only the model seed varies (42-46); EvalConfig stays at 42 so every run is
scored on the same candidate set. Full-model numbers are measured on this
machine, so do not copy CUDA reference numbers into the delta.

Usage:
  python eval_branches.py              # 2 encoders × 4 variants × 5 seeds
  python eval_branches.py --seeds 3
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

RESULTS_DIR = os.path.join(_HERE, "results")
SEEDS = [42, 43, 44, 45, 46]
RAW_CSV = os.path.join(RESULTS_DIR, "branches_raw.csv")
SUMM_CSV = os.path.join(RESULTS_DIR, "branches_summary.csv")
MD_PATH = os.path.join(RESULTS_DIR, "branches_comparison.md")

# (name, HybridCfg kwargs overrides relative to GRU default)
ENCODERS = ["gru", "cnn"]

VARIANTS = [
    ("full",        {}),
    ("no_graph",    {"use_graph": False}),
    ("no_temporal", {"use_temporal": False}),
    ("no_pair",     {"use_pair": False}),
    # Leave-one-out only measures the *marginal* value of a branch while the
    # others are present, so it cannot tell "useless" from "redundant". This
    # variant keeps only the pair features and answers the question the
    # leave-one-out runs raise: would a non-hybrid model do the same job?
    ("pair_only",   {"use_graph": False, "use_temporal": False}),
]


def _base_cfg_kwargs(encoder: str = "gru"):
    """Winning hybrid config from the final grid search, per encoder.

    The ablation used to run on HybridCfg() defaults; those are no longer the
    configuration we report, so the branch contributions were measured on a
    different model than the headline numbers.
    """
    from final_eval import best_cfg
    c = best_cfg(f"hybrid_{encoder}", "val_ap", True)
    kw = dict(encoder=encoder, lr=float(c["lr"]), hidden=int(c["hidden"]),
              ts_lookback=int(c["ts_lookback"]),
              lambda_count=0.5 if encoder == "gru" else 1.0,
              epochs=30, patience=3)
    # the two grids searched different fourth axes
    if "fusion_hidden" in c:
        kw["fusion_hidden"] = int(c["fusion_hidden"])
    if "dropout" in c:
        kw["dropout"] = float(c["dropout"])
    return kw


def run_variant(variant: str, seed: int, data, encoder: str) -> dict:
    import hybrid_core as hc
    overrides = dict(next(o for n, o in VARIANTS if n == variant))
    cfg = hc.HybridCfg(seed=seed, **_base_cfg_kwargs(encoder), **overrides)
    r = hc.run_hybrid(cfg, data, eval_splits=("val", "test"))
    return {"val_ap": r["val"]["ap"], "val_auc": r["val"]["auc"],
            "val_mse": r["val"]["mse"], "val_mae": r["val"]["mae"],
            "test_ap": r["test"]["ap"], "test_auc": r["test"]["auc"],
            "test_f1": r["test"]["f1"],
            "test_mse": r["test"]["mse"], "test_mae": r["test"]["mae"]}


def _load_done() -> set:
    if not os.path.exists(RAW_CSV):
        return set()
    d = pd.read_csv(RAW_CSV)
    if "encoder" not in d.columns:      # rows from before the encoder split
        return set()
    return {(str(e), str(v), int(s))
            for e, v, s in zip(d["encoder"], d["variant"], d["seed"])}


def _append(row: dict) -> None:
    new = pd.DataFrame([row])
    if os.path.exists(RAW_CSV):
        new = pd.concat([pd.read_csv(RAW_CSV), new], ignore_index=True)
    new.to_csv(RAW_CSV, index=False)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("encoder", "variant") if c in df.columns]
    metric_cols = [c for c in df.columns if c not in ("encoder", "variant", "seed")]
    g = df.groupby(keys)[metric_cols]
    out = g.agg(["mean", "std"]).round(4)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def _pooled_sigma(std_a: float, std_b: float) -> float:
    """Pooled SD: sqrt((sigma_a^2 + sigma_b^2) / 2)."""
    return float(np.sqrt((std_a ** 2 + std_b ** 2) / 2.0))


def _verdict(n_sigma: float) -> str:
    if n_sigma < 2:
        return "not distinguishable from seed noise"
    if n_sigma < 3:
        return "borderline"
    return "**real effect**"


def write_comparison_md(summ: pd.DataFrame, n_seeds: int, device: str,
                        elapsed_min: float) -> None:
    order = [n for n, _ in VARIANTS]
    lines = [
        "# Branch ablations",
        "",
        "Swapping the encoder (GRU <-> CNN) only shows that the *encoding* is",
        "interchangeable. These runs remove each component by zeroing its",
        "contribution in `HybridHurdle` - parameter count is identical across",
        "variants, so a difference cannot come from model size.",
        "",
        f"{len(order)} variants x {n_seeds} seeds ({SEEDS[0]}-{SEEDS[n_seeds-1]}) "
        f"x {len(ENCODERS)} encoders, ~{elapsed_min:.0f} min on {device}.",
        "Both encoders use their own grid-search winner. Only the *model* seed",
        "varies; EvalConfig stays at 42, so every run is scored on the same",
        "candidate set.",
        "",
    ]

    for enc in ENCODERS:
        sub = summ[summ["encoder"] == enc] if "encoder" in summ.columns else summ
        if not len(sub):
            continue
        sub = sub.set_index("variant").reindex(order).reset_index()
        full = sub[sub["variant"] == "full"].iloc[0]
        lines += [f"## {enc.upper()} - results (test split)", "",
                  "| Variant | AP | MSE | F1 |", "|---|---|---|---|"]
        for _, r in sub.iterrows():
            mark = "**" if r["variant"] == "full" else ""
            lines.append(
                f"| {mark}{r['variant']}{mark} "
                f"| {mark}{r['test_ap_mean']:.4f} +- {r['test_ap_std']:.4f}{mark} "
                f"| {r['test_mse_mean']:.4f} +- {r['test_mse_std']:.4f} "
                f"| {r['test_f1_mean']:.4f} +- {r['test_f1_std']:.4f} |")
        lines += ["", f"### {enc.upper()} - difference vs. full model", "",
                  "Pooled standard deviations sqrt((sigma_abl^2 + sigma_full^2)/2); "
                  "below ~2 sigma is not distinguishable from seed noise. "
                  "AP and MSE can disagree: a branch may look irrelevant under "
                  "binary AP while still feeding the count head.", "",
                  "| Variant | d AP | sigma | Verdict (AP) | d MSE | sigma | Verdict (MSE) |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            if r["variant"] == "full":
                continue
            d_ap = r["test_ap_mean"] - full["test_ap_mean"]
            d_mse = r["test_mse_mean"] - full["test_mse_mean"]
            p_ap = _pooled_sigma(r["test_ap_std"], full["test_ap_std"])
            p_mse = _pooled_sigma(r["test_mse_std"], full["test_mse_std"])
            n_ap = abs(d_ap) / p_ap if p_ap > 0 else float("inf")
            n_mse = abs(d_mse) / p_mse if p_mse > 0 else float("inf")
            lines.append(
                f"| {r['variant']} | {d_ap:+.4f} | {n_ap:.1f} | {_verdict(n_ap)} | "
                f"{d_mse:+.4f} | {n_mse:.1f} | {_verdict(n_mse)} |")
        lines.append("")

    lines += [
        "",
        "Reproduce: `python eval_branches.py --seeds 5`",
        "Raw per-run data: `branches_raw.csv`, aggregated: `branches_summary.csv`",
        "",
    ]
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | seeds: {seeds} | variants: {[n for n, _ in VARIANTS]}")

    # Sanity: same param count across ablation flags
    import hybrid_core as hc
    def _nparams(cfg):
        m = hc.HybridHurdle(cfg, n_static=3, n_channels=4, n_pair=6)
        return sum(p.numel() for p in m.parameters())
    for enc in ENCODERS:
        base = _base_cfg_kwargs(enc)
        n_full = _nparams(hc.HybridCfg(**base))
        for name, ov in VARIANTS:
            n = _nparams(hc.HybridCfg(**base, **ov))
            assert n == n_full, f"{enc}/{name}: {n} params != full {n_full}"
        print(f"  {enc}: param count identical across variants: {n_full:,}")

    done = _load_done()
    if done:
        print(f"resuming: {len(done)} runs already recorded")
    t_all = time.time()

    for enc in ENCODERS:
        base = _base_cfg_kwargs(enc)
        print(f"\n{'='*60}\nEncoder: {enc}  ({base})\n{'='*60}")
        data = hc.HybridData(lookback=base["ts_lookback"])
        for variant, _ in VARIANTS:
            print(f"\n--- {enc} / {variant} ---")
            for s in seeds:
                if (enc, variant, s) in done:
                    print(f"  seed {s}: skipped")
                    continue
                t0 = time.time()
                m = run_variant(variant, s, data, enc)
                _append({"encoder": enc, "variant": variant, "seed": s, **m})
                print(f"  seed {s}: test_ap={m['test_ap']:.4f} "
                      f"test_mse={m['test_mse']:.4f}  ({time.time()-t0:.0f}s)")
        del data

    df = pd.read_csv(RAW_CSV)
    # only keep the requested seeds/variants for this summary
    df = df[df["variant"].isin([n for n, _ in VARIANTS]) & df["seed"].isin(seeds)]
    if "encoder" in df.columns:
        df = df[df["encoder"].isin(ENCODERS)]
    summ = summarize(df)
    # stable row order
    summ["_ord"] = summ["variant"].map({n: i for i, (n, _) in enumerate(VARIANTS)})
    sort_by = (["encoder", "_ord"] if "encoder" in summ.columns else ["_ord"])
    summ = summ.sort_values(sort_by).drop(columns="_ord")
    summ.to_csv(SUMM_CSV, index=False)

    elapsed = (time.time() - t_all) / 60
    write_comparison_md(summ, len(seeds), device, elapsed)

    print("\n" + "=" * 70)
    print(f"SUMMARY (mean +/- std over {len(seeds)} seeds)")
    print("=" * 70)
    for _, r in summ.iterrows():
        print(f"{r['variant']:12s}  test_ap={r['test_ap_mean']:.4f}+/-{r['test_ap_std']:.4f}  "
              f"test_mse={r['test_mse_mean']:.4f}+/-{r['test_mse_std']:.4f}  "
              f"test_f1={r['test_f1_mean']:.4f}+/-{r['test_f1_std']:.4f}")
    print(f"\nwritten: {RAW_CSV}\n         {SUMM_CSV}\n         {MD_PATH}")
    print(f"total {elapsed:.1f} min")


if __name__ == "__main__":
    main()
