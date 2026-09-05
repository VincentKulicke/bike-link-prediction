# -*- coding: utf-8 -*-
"""
Summarise the final hyperparameter search.

Answers the three questions the talk needs, per model:
  1. Does tuning matter at all?      -> best vs WORST config, in seed sigmas
  2. Did the hand-picked default already sit near the top?  -> its rank
  3. Which axis actually moves the metric?  -> main effect per axis

Point 1 is the honest version of the old "default vs HPO" slide: comparing
against the default answers whether our starting guess was lucky, not whether
searching helps. Best vs worst answers the latter.

    python ablation/hpo_final_report.py
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# metric, direction, grid axes, and the config the code ships as default
MODELS = {
    "hybrid_gru": dict(
        metric="val_ap", test="test_ap", higher=True,
        axes=["lr", "hidden", "ts_lookback", "fusion_hidden"],
        default=dict(lr=1e-3, hidden=64, ts_lookback=12, fusion_hidden=128),
        seed_sigma=0.0004),
    "hybrid_cnn": dict(
        metric="val_ap", test="test_ap", higher=True,
        axes=["lr", "hidden", "ts_lookback", "dropout"],
        default=dict(lr=1e-3, hidden=64, ts_lookback=12, dropout=0.1),
        seed_sigma=0.0003),
    "lstm": dict(
        metric="val_mse", test="test_mse", higher=False,
        axes=["lr", "hidden_dim", "lookback", "layers_dropout"],
        default=dict(lr=1e-3, hidden_dim=64, lookback=48,
                     layers_dropout="(1, 0.0)"),
        seed_sigma=0.0013),
    "graphmixer": dict(
        metric="val_ap", test="test_ap", higher=True,
        axes=["lr", "hidden_dim", "num_neighbors", "mixer_layers"],
        default=dict(lr=1e-3, hidden_dim=128, num_neighbors=20,
                     mixer_layers=2),
        # 0.0014, measured over 5 seeds of the winning config in
        # final_eval.py. The old 0.0375 came from the broken training regime;
        # pooled across the whole grid it is 0.0079, because badly converging
        # configs (lr=3e-5) are far noisier than the winner.
        seed_sigma=0.0014),
}


def load(name, spec):
    p = os.path.join(RES, f"hpo_final_{name}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[d["error"].isna()]
    keys = spec["axes"] + ["seed"]
    d = d.drop_duplicates(subset=keys, keep="last")
    if not len(d):
        return None
    # average over seeds where a grid used more than one
    g = d.groupby(spec["axes"], as_index=False).agg(
        val=(spec["metric"], "mean"), test=(spec["test"], "mean"),
        n_seeds=("seed", "nunique"), sec=("sec", "mean"))
    return g.sort_values("val", ascending=not spec["higher"]).reset_index(drop=True)


def main():
    lines = ["# Final hyperparameter search", "",
             "One run, four grids. Selection on validation, numbers reported on test.",
             ""]
    for name, spec in MODELS.items():
        g = load(name, spec)
        print(f"\n{'='*72}\n{name}\n{'='*72}")
        if g is None:
            print("  noch keine Ergebnisse"); continue
        g["rank"] = np.arange(1, len(g) + 1)
        best, worst = g.iloc[0], g.iloc[-1]
        sign = 1.0 if spec["higher"] else -1.0
        gap = sign * (best["test"] - worst["test"])
        sig = gap / spec["seed_sigma"]

        print(f"  Konfigs: {len(g)} | Seeds je Konfig: {g.n_seeds.max()}")
        print(f"  bester      {dict(best[spec['axes']])}  val {best['val']:.4f}  test {best['test']:.4f}")
        print(f"  schlechtest {dict(worst[spec['axes']])}  val {worst['val']:.4f}  test {worst['test']:.4f}")
        print(f"  bester vs schlechtester: {gap:+.4f} = {sig:.1f} sigma")

        # where does the shipped default sit?
        m = pd.Series(True, index=g.index)
        for k, v in spec["default"].items():
            if k in g.columns:
                col = g[k].astype(str) if isinstance(v, str) else g[k]
                m &= (col == v)
        if m.any():
            r = g[m].iloc[0]
            print(f"  Default liegt auf Rang {int(r['rank'])}/{len(g)} "
                  f"(val {r['val']:.4f}, test {r['test']:.4f})")
        else:
            print("  Default liegt ausserhalb des neuen Gitters")

        print("  Haupteffekte:")
        for ax in spec["axes"]:
            mm = g.groupby(ax)["val"].mean()
            print(f"    {ax:<15} Spanne {mm.max()-mm.min():.4f}  " +
                  " ".join(f"{k}:{v:.4f}" for k, v in mm.items()))

        lines += [f"## {name}", "",
                  f"- configs: {len(g)}, seeds each: {g.n_seeds.max()}",
                  f"- best vs worst on test: {gap:+.4f} ({sig:.1f} sigma)",
                  f"- best: {dict(best[spec['axes']])} -> test {best['test']:.4f}",
                  ""]
    out = os.path.join(RES, "hpo_final_comparison.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
