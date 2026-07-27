# -*- coding: utf-8 -*-
"""
eval_ranking.py — the harder 1-vs-99 ranking protocol.
=======================================================

The default shared_eval uses a 1:5 negative ratio, which is relatively easy and
pushes both the hybrid metrics against the ceiling (AP ~0.92). This script scores
the models under the 1-vs-99 destination-ranking protocol the concept document
originally intended (MRR + AP under 1:99 imbalance). That lowers the artificial
plateau and gives HPO / architecture room to show a difference.

It answers one question directly: under the harder metric, does the HPO-tuned
hybrid beat the default hybrid?

Models scored on the SAME seeded ranking set:
  - Hybrid GraphSAGE+GRU (default:  h64,  lambda 1.0)
  - Hybrid GraphSAGE+GRU (HPO best: h128, lambda 0.5)
  - GraphMixer (default) and GraphMixer (HPO best: lr 1e-4, h64, 1 layer)

Usage:  python eval_ranking.py [--max_queries 3000] [--gm_epochs 20]
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "evaluation"))
sys.path.insert(0, os.path.join(_HERE, "..", "graphmixer", "model"))
import hybrid_core as hc                                              # noqa: E402
from shared_eval import SharedLinkEval, EvalConfig, ranking_metrics  # noqa: E402
from graphmixer import GraphMixer, GMConfig                          # noqa: E402
from graphmixer_data import GraphMixerData                           # noqa: E402
from train_graphmixer import train as gm_train, make_pack           # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")
SPLITS = ["val", "test"]


# --- scoring the ranking set per model (unique (u,i,bin) tuples) --------------
@torch.no_grad()
def score_hybrid(uniq: pd.DataFrame, data: hc.HybridData, model) -> np.ndarray:
    dev = data.device
    u = uniq["u"].to_numpy(); i = uniq["i"].to_numpy(); b = uniq["bin_idx"].to_numpy()
    sage = model.sage(data.static_x_t, data.A_norm)
    out = np.zeros(len(uniq), dtype=np.float32)
    for s in range(0, len(uniq), 8192):
        sl = slice(s, s + 8192)
        wu = torch.tensor(data._windows(u[sl], b[sl]), device=dev)
        wi = torch.tensor(data._windows(i[sl], b[sl]), device=dev)
        pf = torch.tensor(data._pair_feats(u[sl], i[sl], b[sl]), device=dev)
        ut = torch.tensor(u[sl], dtype=torch.long, device=dev)
        it = torch.tensor(i[sl], dtype=torch.long, device=dev)
        logit, _ = model(sage, ut, it, wu, wi, pf)
        out[sl] = torch.sigmoid(logit).cpu().numpy()
    return out


@torch.no_grad()
def score_graphmixer(uniq: pd.DataFrame, data: GraphMixerData, model, cfg, dev) -> np.ndarray:
    u = uniq["u"].to_numpy(); i = uniq["i"].to_numpy(); b = uniq["bin_idx"].to_numpy()
    tq = b * cfg.bin_seconds
    out = np.zeros(len(uniq), dtype=np.float32)
    for s in range(0, len(uniq), 4096):
        sl = slice(s, s + 4096)
        up = make_pack(model, *data.get_batch(u[sl] + 1, tq[sl], cfg.num_neighbors), dev)  # +1: 1-indexed
        ip = make_pack(model, *data.get_batch(i[sl] + 1, tq[sl], cfg.num_neighbors), dev)
        out[sl] = torch.sigmoid(model(up, ip)).cpu().numpy()
    return out


def evaluate(name, tuning, score_fn, cand_by_split):
    rows = []
    for split in SPLITS:
        cand = cand_by_split[split]
        uniq = cand[["u", "i", "bin_idx"]].drop_duplicates().reset_index(drop=True)
        uniq["score"] = score_fn(uniq)
        merged = cand.merge(uniq, on=["u", "i", "bin_idx"], how="left")
        m = ranking_metrics(merged)
        rows.append({"model": name, "tuning": tuning, "split": split,
                     "mrr": m["mrr"], "hits@1": m["hits@1"], "hits@5": m["hits@5"],
                     "auc": m["auc"], "ap": m["ap"], "n_queries": m["n_queries"]})
        print(f"  [{split}] {name} ({tuning}): MRR={m['mrr']:.3f} "
              f"H@1={m['hits@1']:.3f} H@5={m['hits@5']:.3f} | "
              f"AUC={m['auc']:.3f} AP={m['ap']:.3f}  (1:99, n={m['n_queries']})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_queries", type=int, default=3000)
    ap.add_argument("--gm_epochs", type=int, default=20)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ev = SharedLinkEval()
    cand_by_split = {s: ev.build_ranking_candidates(s, n_neg=99, max_queries=args.max_queries)
                     for s in SPLITS}
    for s in SPLITS:
        n_q = cand_by_split[s]["query_id"].nunique()
        print(f"[ranking set] {s}: {n_q} queries x 100 = {len(cand_by_split[s]):,} rows")

    all_rows = []
    t0 = time.time()

    # --- Hybrid (shares one HybridData for both configs) ---
    print("\n=== Hybrid ===")
    data = hc.HybridData()
    for tuning, cfg in [("default", hc.HybridCfg()),
                        ("HPO", hc.HybridCfg(hidden=128, fusion_hidden=256, lambda_count=0.5))]:
        r = hc.run_hybrid(cfg, data, eval_splits=(), return_model=True)
        model = r["model"]
        all_rows += evaluate("Hybrid GraphSAGE+GRU", tuning,
                             lambda uniq, m=model: score_hybrid(uniq, data, m), cand_by_split)

    # --- GraphMixer (shares one GraphMixerData) ---
    print("\n=== GraphMixer ===")
    gm_base = GMConfig()
    gm_data = GraphMixerData(gm_base.prep_dir)
    for tuning, cfg in [("default", GMConfig(epochs=args.gm_epochs)),
                        ("HPO", GMConfig(lr=1e-4, hidden_dim=64, mixer_layers=1, epochs=args.gm_epochs))]:
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        model = GraphMixer(cfg, edge_feat_dim=gm_data.d_edge, node_feat_dim=gm_data.d_node).to(dev)
        model = gm_train(cfg, gm_data, model, dev)
        all_rows += evaluate("GraphMixer", tuning,
                             lambda uniq, m=model, c=cfg: score_graphmixer(uniq, gm_data, m, c, dev),
                             cand_by_split)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "ranking_eval.csv"), index=False)

    # --- markdown summary (test split) ---
    def f(x): return f"{x:.3f}"
    lines = ["# 1-vs-99 ranking evaluation\n",
             f"Harder protocol than the 1:5 default: each positive is ranked against 99 random "
             f"destinations (same source & bin), seed 42, {args.max_queries} queries per split. "
             f"MRR / Hits are the ranking view; AUC / AP are pooled under 1:99 imbalance.\n",
             "## Test set\n",
             "| Model | Tuning | MRR | Hits@1 | Hits@5 | AUC | AP |",
             "|---|---|---|---|---|---|---|"]
    for r in [x for x in all_rows if x["split"] == "test"]:
        lines.append(f"| {r['model']} | {r['tuning']} | {f(r['mrr'])} | {f(r['hits@1'])} | "
                     f"{f(r['hits@5'])} | {f(r['auc'])} | {f(r['ap'])} |")
    lines += ["\n## Validation set\n",
              "| Model | Tuning | MRR | Hits@1 | Hits@5 | AUC | AP |",
              "|---|---|---|---|---|---|---|"]
    for r in [x for x in all_rows if x["split"] == "val"]:
        lines.append(f"| {r['model']} | {r['tuning']} | {f(r['mrr'])} | {f(r['hits@1'])} | "
                     f"{f(r['hits@5'])} | {f(r['auc'])} | {f(r['ap'])} |")
    out = os.path.join(RESULTS_DIR, "ranking_comparison.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n=== done in {(time.time()-t0)/60:.1f} min -> {out} ===")


if __name__ == "__main__":
    main()
