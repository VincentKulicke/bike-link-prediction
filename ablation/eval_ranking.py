# -*- coding: utf-8 -*-
"""
eval_ranking.py: the harder 1-vs-99 ranking protocol.
=======================================================

The default shared_eval uses a 1:5 negative ratio, which is relatively easy and
pushes both the hybrid metrics against the ceiling (AP ~0.92). This script scores
the models under the 1-vs-99 destination-ranking protocol the concept document
originally intended (MRR + AP under 1:99 imbalance). That lowers the artificial
plateau and gives HPO / architecture room to show a difference.

It answers one question directly: under the harder metric, does the HPO-tuned
hybrid beat the default hybrid?

Models scored on the SAME seeded ranking set:
  - Frequency heuristic (train rate per pair, no training)
  - Hybrid GraphSAGE+GRU (default:  h64,  lambda 1.0)
  - Hybrid GraphSAGE+GRU (HPO best: h128, lambda 0.5)
  - GraphMixer (default) and GraphMixer (HPO best: lr 1e-4, h64, 1 layer)

Analysis mode (--analysis):
  Paired per-query RR tests (Hybrid vs Frequency, Hybrid vs GraphMixer) and
  stratification by train pair-trip count. Uses ALL test positives by default
  (max_queries=0) so rare/unseen strata are large enough.

Usage:
  python eval_ranking.py [--max_queries 3000] [--gm_epochs 20]
  python eval_ranking.py --analysis                 # all test positives
  python eval_ranking.py --analysis --max_queries 0 # same (0 = all)
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
from shared_eval import (SharedLinkEval, EvalConfig, ranking_metrics,  # noqa: E402
                         per_query_ranks, _frequency_baseline)
from graphmixer import GraphMixer, GMConfig                          # noqa: E402
from graphmixer_data import GraphMixerData                           # noqa: E402
from train_graphmixer import train as gm_train, make_pack           # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")
SPLITS = ["val", "test"]
STRAT_CSV = os.path.join(RESULTS_DIR, "ranking_stratified.csv")
MD_PATH = os.path.join(RESULTS_DIR, "ranking_comparison.md")

# Train-trip-count bins for the true positive pair of each ranking query.
STRATA = [
    ("0", 0, 0),
    ("1-5", 1, 5),
    ("6-20", 6, 20),
    ("21-100", 21, 100),
    (">100", 101, None),
]


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


# --- Analysis: paired RR test + stratification by pair history ---------------
def _train_trip_counts(ev: SharedLinkEval) -> pd.Series:
    """Sum of train-split trips per (u, i), the pair history used for strata."""
    tg = ev.build_targets()
    train = tg[tg["split"] == "train"]
    return train.groupby(["u", "i"])["count"].sum()


def _assign_stratum(n_trips: float, strata=STRATA) -> str:
    for name, lo, hi in strata:
        if hi is None:
            if n_trips >= lo:
                return name
        elif lo <= n_trips <= hi:
            return name
    return "other"


def paired_diff_ci(rr_a: np.ndarray, rr_b: np.ndarray, label_a: str, label_b: str,
                   z: float = 1.96) -> dict:
    """Paired mean of (rr_a - rr_b) with SE and Wald 95% CI."""
    d = np.asarray(rr_a, dtype=float) - np.asarray(rr_b, dtype=float)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    lo, hi = mean - z * se, mean + z * se
    return {
        "comparison": f"{label_a} − {label_b}",
        "n": n,
        "mean_delta_rr": mean,
        "se": se,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "mrr_a": float(np.mean(rr_a)),
        "mrr_b": float(np.mean(rr_b)),
    }


def score_cand(cand: pd.DataFrame, score_fn) -> pd.DataFrame:
    uniq = cand[["u", "i", "bin_idx"]].drop_duplicates().reset_index(drop=True)
    uniq["score"] = score_fn(uniq)
    return cand.merge(uniq, on=["u", "i", "bin_idx"], how="left")


def run_analysis(ev: SharedLinkEval, cand: pd.DataFrame, pq: dict[str, pd.DataFrame],
                 split: str = "test") -> tuple[pd.DataFrame, list[dict], list[str]]:
    """Build stratified table + paired tests from per-query rank frames in `pq`.

    pq keys: 'Frequency', 'Hybrid', 'GraphMixer'; each a per_query_ranks frame.
    """
    trips = _train_trip_counts(ev)
    base = pq["Hybrid"][["query_id", "u", "i"]].copy()
    base["train_trips"] = [float(trips.get((int(u), int(i)), 0.0))
                           for u, i in zip(base["u"], base["i"])]
    base["stratum"] = base["train_trips"].map(_assign_stratum)

    counts = base["stratum"].value_counts()
    merged_note = None
    active = list(STRATA)
    if counts.get("0", 0) < 100:
        merged_note = ("Stratum 0 has only "
                       f"{int(counts.get('0', 0))} queries, merged with 1-5 "
                       "into 0-5.")
        active = [("0-5", 0, 5), ("6-20", 6, 20), ("21-100", 21, 100),
                  (">100", 101, None)]
        base["stratum"] = base["train_trips"].map(
            lambda n: _assign_stratum(n, active))

    strat_rows = []
    order = [n for n, _, _ in active]
    for model_name, frame in pq.items():
        m = frame.merge(base[["query_id", "stratum", "train_trips"]], on="query_id")
        for st in order:
            sub = m[m["stratum"] == st]
            strat_rows.append({
                "split": split, "model": model_name, "stratum": st,
                "n_queries": int(len(sub)),
                "mrr": float(sub["rr"].mean()) if len(sub) else float("nan"),
                "hits@1": float(sub["hits@1"].mean()) if len(sub) else float("nan"),
            })
        strat_rows.append({
            "split": split, "model": model_name, "stratum": "all",
            "n_queries": int(len(m)),
            "mrr": float(m["rr"].mean()),
            "hits@1": float(m["hits@1"].mean()),
        })
    strat_df = pd.DataFrame(strat_rows)

    h = pq["Hybrid"].set_index("query_id").sort_index()
    f = pq["Frequency"].set_index("query_id").sort_index()
    g = pq["GraphMixer"].set_index("query_id").sort_index()
    assert h.index.equals(f.index) and h.index.equals(g.index)
    paired = [
        paired_diff_ci(h["rr"].values, f["rr"].values, "Hybrid", "Frequency"),
        paired_diff_ci(h["rr"].values, g["rr"].values, "Hybrid", "GraphMixer"),
    ]

    md_lines = _format_analysis_md(strat_df, paired, order, len(base), merged_note)
    return strat_df, paired, md_lines


def _format_analysis_md(strat_df: pd.DataFrame, paired: list[dict],
                        strata_order: list[str], n_queries: int,
                        merged_note: str | None) -> list[str]:
    lines = [
        "## Paired significance (per-query reciprocal ranks)",
        "",
        f"All models scored on the **same** seeded 1:99 queries (test, n={n_queries}). "
        "For each query we store RR = 1/rank(true destination), then form the "
        "paired difference Δ = RR(A) − RR(B). Mean ± SE and a Wald 95 % CI; "
        "if the CI excludes 0 the difference is distinguishable from a tie.",
        "",
        "| Comparison | MRR(A) | MRR(B) | Mean Δ RR | SE | 95 % CI | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in paired:
        verd = ("**distinguishable from zero**" if p["ci_excludes_zero"]
                else "tie (CI includes 0)")
        lines.append(
            f"| {p['comparison']} | {p['mrr_a']:.4f} | {p['mrr_b']:.4f} | "
            f"{p['mean_delta_rr']:+.4f} | {p['se']:.4f} | "
            f"[{p['ci95_lo']:+.4f}, {p['ci95_hi']:+.4f}] | {verd} |"
        )

    lines += [
        "",
        "## Stratified by pair history (train trip count)",
        "",
        "Each ranking query’s true pair (u, i) is binned by the **sum of train-split "
        "trips** for that pair (`build_targets()`, split=train). "
        "Expectation: frequency ≈ / ≥ hybrid on high-history pairs; hybrid ahead "
        "on rare/unseen pairs (frequency scores 0 → near-random).",
        "",
    ]
    if merged_note:
        lines += [f"*Note: {merged_note}*", ""]

    models = ["Frequency", "Hybrid", "GraphMixer"]
    lines += [
        "| Stratum | n | Frequency MRR | Hybrid MRR | GraphMixer MRR | "
        "Frequency H@1 | Hybrid H@1 | GraphMixer H@1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for st in strata_order + ["all"]:
        cells = {}
        n = None
        for m in models:
            row = strat_df[(strat_df["model"] == m) & (strat_df["stratum"] == st)]
            if len(row) == 0:
                cells[m] = ("—", "—")
                continue
            r = row.iloc[0]
            n = int(r["n_queries"])
            cells[m] = (f"{r['mrr']:.3f}", f"{r['hits@1']:.3f}")
        lines.append(
            f"| {st} | {n if n is not None else '—'} | "
            f"{cells['Frequency'][0]} | {cells['Hybrid'][0]} | {cells['GraphMixer'][0]} | "
            f"{cells['Frequency'][1]} | {cells['Hybrid'][1]} | {cells['GraphMixer'][1]} |"
        )
    lines.append("")
    lines.append("Raw stratified metrics: `ranking_stratified.csv`")
    lines.append("")
    return lines


def _upsert_analysis_section(md_path: str, analysis_lines: list[str]) -> None:
    """Replace or append the analysis block in ranking_comparison.md."""
    marker = "## Paired significance (per-query reciprocal ranks)"
    block = "\n".join(analysis_lines).rstrip() + "\n"
    findings = ""
    if os.path.exists(md_path):
        text = open(md_path, encoding="utf-8").read()
        if "## Findings" in text:
            findings = "## Findings" + text.split("## Findings", 1)[1]
            # drop trailing reproduce-only duplicates later
        if marker in text:
            pre = text.split(marker, 1)[0].rstrip()
        else:
            pre = text.split("## Findings", 1)[0].rstrip() if "## Findings" in text \
                else text.rstrip()
        text = pre + "\n\n" + block
        if findings:
            # keep a single Findings block
            if "## Findings" in findings:
                text = text.rstrip() + "\n\n" + findings.lstrip()
            if not text.endswith("\n"):
                text += "\n"
    else:
        text = block
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")


def analysis_main(args, ev: SharedLinkEval, dev: str) -> None:
    mq = None if args.max_queries == 0 else args.max_queries
    split = "test"
    print(f"\n=== Analysis mode (split={split}, max_queries="
          f"{'ALL' if mq is None else mq}) ===")
    cand = ev.build_ranking_candidates(split, n_neg=99, max_queries=mq)
    n_q = cand["query_id"].nunique()
    print(f"[ranking set] {split}: {n_q} queries x 100 = {len(cand):,} rows")

    print("\n--- Frequency ---")
    t0 = time.time()
    m_freq = score_cand(
        cand, lambda uniq: _frequency_baseline(ev, split, cand=uniq)["score"].to_numpy())
    pq_freq = per_query_ranks(m_freq)
    print(f"  MRR={pq_freq['rr'].mean():.4f} H@1={pq_freq['hits@1'].mean():.4f}  "
          f"({time.time()-t0:.1f}s)")

    print("\n--- Hybrid default ---")
    t0 = time.time()
    data = hc.HybridData()
    r = hc.run_hybrid(hc.HybridCfg(), data, eval_splits=(), return_model=True)
    model = r["model"]
    print(f"  trained in {r['train_s']:.0f}s")
    m_hyb = score_cand(cand, lambda uniq, m=model: score_hybrid(uniq, data, m))
    pq_hyb = per_query_ranks(m_hyb)
    print(f"  MRR={pq_hyb['rr'].mean():.4f} H@1={pq_hyb['hits@1'].mean():.4f}  "
          f"(score+ranks {time.time()-t0 - r['train_s']:.1f}s)")

    print("\n--- GraphMixer default ---")
    t0 = time.time()
    gm_cfg = GMConfig(epochs=args.gm_epochs)
    gm_data = GraphMixerData(gm_cfg.prep_dir)
    torch.manual_seed(gm_cfg.seed); np.random.seed(gm_cfg.seed)
    gm_model = GraphMixer(gm_cfg, edge_feat_dim=gm_data.d_edge,
                          node_feat_dim=gm_data.d_node).to(dev)
    gm_model = gm_train(gm_cfg, gm_data, gm_model, dev)
    print(f"  trained in {time.time()-t0:.0f}s")
    t1 = time.time()
    m_gm = score_cand(
        cand, lambda uniq, m=gm_model, c=gm_cfg: score_graphmixer(uniq, gm_data, m, c, dev))
    pq_gm = per_query_ranks(m_gm)
    print(f"  MRR={pq_gm['rr'].mean():.4f} H@1={pq_gm['hits@1'].mean():.4f}  "
          f"(score {time.time()-t1:.1f}s)")

    pq = {"Frequency": pq_freq, "Hybrid": pq_hyb, "GraphMixer": pq_gm}
    strat_df, paired, md_lines = run_analysis(ev, cand, pq, split=split)
    strat_df.to_csv(STRAT_CSV, index=False)
    _upsert_analysis_section(MD_PATH, md_lines)

    print("\n" + "=" * 70)
    for p in paired:
        verd = "≠ 0" if p["ci_excludes_zero"] else "tie"
        print(f"{p['comparison']}: Δ={p['mean_delta_rr']:+.4f} "
              f"SE={p['se']:.4f} CI=[{p['ci95_lo']:+.4f},{p['ci95_hi']:+.4f}]  {verd}")
    print(f"\nwritten: {STRAT_CSV}\n         {MD_PATH} (analysis sections)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_queries", type=int, default=3000,
                    help="ranking query cap; 0 = all positives (analysis default)")
    ap.add_argument("--gm_epochs", type=int, default=20)
    ap.add_argument("--freq_only", action="store_true",
                    help="score only the frequency heuristic; merge into existing ranking_eval.csv")
    ap.add_argument("--analysis", action="store_true",
                    help="paired RR test + pair-history stratification (test split)")
    args = ap.parse_args()
    if args.analysis and "--max_queries" not in sys.argv:
        args.max_queries = 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ev = SharedLinkEval()

    if args.analysis:
        analysis_main(args, ev, dev)
        return

    mq = None if args.max_queries == 0 else args.max_queries
    cand_by_split = {s: ev.build_ranking_candidates(s, n_neg=99, max_queries=mq)
                     for s in SPLITS}
    for s in SPLITS:
        n_q = cand_by_split[s]["query_id"].nunique()
        print(f"[ranking set] {s}: {n_q} queries x 100 = {len(cand_by_split[s]):,} rows")

    all_rows = []
    t0 = time.time()

    print("\n=== Frequency heuristic ===")

    def _freq_score_fn(split):
        return lambda uniq: _frequency_baseline(ev, split, cand=uniq)["score"].to_numpy()

    for split in SPLITS:
        cand = cand_by_split[split]
        uniq = cand[["u", "i", "bin_idx"]].drop_duplicates().reset_index(drop=True)
        uniq["score"] = _freq_score_fn(split)(uniq)
        merged = cand.merge(uniq, on=["u", "i", "bin_idx"], how="left")
        m = ranking_metrics(merged)
        row = {"model": "Frequency heuristic", "tuning": "—", "split": split,
               "mrr": m["mrr"], "hits@1": m["hits@1"], "hits@5": m["hits@5"],
               "auc": m["auc"], "ap": m["ap"], "n_queries": m["n_queries"]}
        all_rows.append(row)
        print(f"  [{split}] Frequency heuristic: MRR={m['mrr']:.3f} "
              f"H@1={m['hits@1']:.3f} H@5={m['hits@5']:.3f} | "
              f"AUC={m['auc']:.3f} AP={m['ap']:.3f}  (1:99, n={m['n_queries']})")

    if not args.freq_only:
        # The old "default vs HPO" split is dropped: the final grid search
        # settled that question. The open one now is whether the corrected
        # baselines keep up under the harder protocol -- and whether any
        # learned model beats the frequency heuristic, which won last time.
        from final_eval import best_cfg

        for enc in ("gru", "cnn"):
            c = best_cfg(f"hybrid_{enc}", "val_ap", True)
            lb = int(c["ts_lookback"])
            print(f"\n=== Hybrid {enc.upper()} ({c}) ===")
            data = hc.HybridData(lookback=lb)
            cfg = hc.HybridCfg(encoder=enc, lr=float(c["lr"]),
                               hidden=int(c["hidden"]), ts_lookback=lb,
                               fusion_hidden=int(c.get("fusion_hidden", 128)),
                               dropout=float(c.get("dropout", 0.1)),
                               lambda_count=0.5 if enc == "gru" else 1.0,
                               epochs=30, patience=3)
            r = hc.run_hybrid(cfg, data, eval_splits=(), return_model=True)
            model = r["model"]
            all_rows += evaluate(f"Hybrid {enc.upper()}", "final",
                                 lambda uniq, m=model, d=data: score_hybrid(uniq, d, m),
                                 cand_by_split)

        c = best_cfg("graphmixer", "val_ap", True)
        print(f"\n=== GraphMixer ({c}) ===")
        gm_data = GraphMixerData(GMConfig().prep_dir)
        cfg = GMConfig(lr=float(c["lr"]), hidden_dim=int(c["hidden_dim"]),
                       num_neighbors=int(c["num_neighbors"]),
                       mixer_layers=int(c["mixer_layers"]), epochs=30)
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        model = GraphMixer(cfg, edge_feat_dim=gm_data.d_edge,
                           node_feat_dim=gm_data.d_node).to(dev)
        model = gm_train(cfg, gm_data, model, dev, ev=ev, patience=3, verbose=False)
        all_rows += evaluate(
            "GraphMixer", "final",
            lambda uniq, m=model, c=cfg: score_graphmixer(uniq, gm_data, m, c, dev),
            cand_by_split)

    csv_path = os.path.join(RESULTS_DIR, "ranking_eval.csv")
    analysis_keep = ""
    if args.freq_only and os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        prev = prev[prev["model"] != "Frequency heuristic"]
        df = pd.concat([pd.DataFrame(all_rows), prev], ignore_index=True)
    else:
        df = pd.DataFrame(all_rows)
    _mord = {"Frequency heuristic": 0, "Hybrid GraphSAGE+GRU": 1, "GraphMixer": 2}
    _tord = {"—": 0, "default": 1, "HPO": 2}
    df["_mo"] = df["model"].map(_mord).fillna(9)
    df["_to"] = df["tuning"].map(_tord).fillna(9)
    df = (df.sort_values(["_mo", "_to", "split"])
            .drop(columns=["_mo", "_to"]).reset_index(drop=True))
    df.to_csv(csv_path, index=False)

    def f(x): return f"{x:.3f}"
    all_rows = df.to_dict("records")
    findings = ""
    out = MD_PATH
    if os.path.exists(out):
        prev = open(out, encoding="utf-8").read()
        if "## Findings" in prev:
            findings = "\n" + prev.split("## Findings", 1)[1]
            if not findings.startswith("## Findings"):
                findings = "## Findings" + findings
        if "## Paired significance" in prev:
            chunk = prev.split("## Paired significance", 1)[1]
            if "## Findings" in chunk:
                chunk = chunk.split("## Findings", 1)[0]
            analysis_keep = "## Paired significance" + chunk

    lines = ["# 1-vs-99 ranking evaluation\n",
             f"Harder protocol than the 1:5 default: each positive is ranked against 99 random "
             f"destinations (same source & bin), seed 42, "
             f"{'ALL' if mq is None else args.max_queries} queries per split. "
             f"MRR / Hits are the ranking view; AUC / AP are pooled under 1:99 imbalance. "
             f"Frequency heuristic = mean train trips/bin per pair (no learning).\n",
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
    body = "\n".join(lines) + "\n"
    if analysis_keep:
        body = body.rstrip() + "\n\n" + analysis_keep.strip() + "\n"
    if findings:
        body = body.rstrip() + "\n\n" + findings.lstrip()
        if not body.endswith("\n"):
            body += "\n"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(f"\n=== done in {(time.time()-t0)/60:.1f} min -> {out} ===")


if __name__ == "__main__":
    main()
