# -*- coding: utf-8 -*-
"""
Training & prediction export for the GraphMixer baseline.
=========================================================

Wires the model (graphmixer.py) and the data (graphmixer_data.py) together,
trains with negative sampling on the training edges, and exports predictions
for the SHARED candidate set from the eval module.

Steps:
  1. set up data + model + optimizer
  2. training loop (binary cross-entropy, 1 negative per positive)
  3. produce predictions for the shared_eval candidates (val/test)
  4. export as CSV (u, i, bin_idx, score) – CANONICAL 0-indexed
  5. score directly via shared_eval.SharedLinkEval.score_binary

Run:  python train_graphmixer.py
"""

from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# our own modules
from graphmixer import GraphMixer, GMConfig
from graphmixer_data import GraphMixerData

# shared eval module (lives under ../../evaluation)
_EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation")
sys.path.insert(0, _EVAL_DIR)
from shared_eval import SharedLinkEval, EvalConfig   # noqa: E402


# ===========================================================================
# Helper: numpy batch -> model "pack" (tokens, own_feat, neigh_mean)
# ===========================================================================
def make_pack(model: GraphMixer, ef, dt, msk, own, neigh_mean, device):
    """Builds the input tensors from the collected raw data.

    tokens = [edge feature || fixed time encoding(dt)], padding rows = 0.
    """
    ef = torch.as_tensor(ef, dtype=torch.float32, device=device)        # (B,K,d_edge)
    dt = torch.as_tensor(dt, dtype=torch.float32, device=device)        # (B,K)
    msk = torch.as_tensor(msk, dtype=torch.float32, device=device)      # (B,K)
    own = torch.as_tensor(own, dtype=torch.float32, device=device)      # (B,d_node)
    neigh_mean = torch.as_tensor(neigh_mean, dtype=torch.float32, device=device)

    time_feat = model.time_enc(dt)                                      # (B,K,time_dim)
    tokens = torch.cat([ef, time_feat], dim=-1)                         # (B,K,d_edge+time_dim)
    tokens = tokens * msk.unsqueeze(-1)                                 # zero out padding rows
    return tokens, own, neigh_mean


# ===========================================================================
# 1) TRAINING
# ===========================================================================
def train(cfg: GMConfig, data: GraphMixerData, model: GraphMixer, device) -> GraphMixer:
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    tr_mask, _, _ = data.split_masks(cfg.train_end_s, cfg.val_end_s)
    edges = data.edges
    u_all = edges["u"].to_numpy(); i_all = edges["i"].to_numpy(); ts_all = edges["ts"].to_numpy(float)
    tr_idx = np.where(tr_mask)[0]
    nodes = np.arange(1, data.num_nodes + 1)
    print(f"Training edges: {len(tr_idx):,}")

    bs = cfg.batch_size
    for ep in range(1, cfg.epochs + 1):
        model.train()
        perm = rng.permutation(tr_idx)
        total, nb = 0.0, 0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            u = u_all[b]; i = i_all[b]; t = ts_all[b]
            # negatives: same source node u, random target i_neg != u
            i_neg = rng.choice(nodes, size=len(b))
            resample = (i_neg == u)
            while resample.any():
                i_neg[resample] = rng.choice(nodes, size=int(resample.sum()))
                resample = (i_neg == u)

            # collect raw data (query time = event time t; only events < t)
            u_pack = make_pack(model, *data.get_batch(u, t, cfg.num_neighbors), device)
            ipos_pack = make_pack(model, *data.get_batch(i, t, cfg.num_neighbors), device)
            ineg_pack = make_pack(model, *data.get_batch(i_neg, t, cfg.num_neighbors), device)

            pos_logit = model(u_pack, ipos_pack)
            neg_logit = model(u_pack, ineg_pack)
            logits = torch.cat([pos_logit, neg_logit])
            labels = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])

            opt.zero_grad()
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            total += loss.item(); nb += 1
        print(f"Epoch {ep:2d}/{cfg.epochs} | loss {total/max(1,nb):.4f}")
    return model


# ===========================================================================
# 2) PREDICTION EXPORT for the shared_eval candidates
# ===========================================================================
@torch.no_grad()
def export_predictions(cfg: GMConfig, data: GraphMixerData, model: GraphMixer,
                       ev: SharedLinkEval, split: str, device, out_csv: str) -> pd.DataFrame:
    """Computes a link score for every candidate cell (u, i, bin_idx).

    Query time = start of the bin (bin_idx * bin_seconds).
    Candidate node IDs are CANONICAL (0..231); internally +1 (1-indexed).
    """
    model.eval()
    cand = ev.build_candidates(split)[["u", "i", "bin_idx"]].copy()
    u_can = cand["u"].to_numpy(); i_can = cand["i"].to_numpy()
    bin_idx = cand["bin_idx"].to_numpy()
    t_query = bin_idx * cfg.bin_seconds

    scores = np.zeros(len(cand), dtype=np.float32)
    bs = 4096
    for s in range(0, len(cand), bs):
        sl = slice(s, s + bs)
        u_node = u_can[sl] + 1            # canonical -> 1-indexed
        i_node = i_can[sl] + 1
        tq = t_query[sl]
        u_pack = make_pack(model, *data.get_batch(u_node, tq, cfg.num_neighbors), device)
        i_pack = make_pack(model, *data.get_batch(i_node, tq, cfg.num_neighbors), device)
        logit = model(u_pack, i_pack)
        scores[sl] = torch.sigmoid(logit).cpu().numpy()

    pred = cand.copy()
    pred["score"] = scores            # u, i stay canonical 0-indexed (for shared_eval)
    pred.to_csv(out_csv, index=False)
    return pred


# ===========================================================================
# 3) MAIN
# ===========================================================================
def main(cfg: GMConfig | None = None):
    cfg = cfg or GMConfig()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = GraphMixerData(cfg.prep_dir)
    print(f"Nodes: {data.num_nodes} | edges: {len(data.edges):,} | "
          f"d_edge={data.d_edge} d_node={data.d_node}")

    model = GraphMixer(cfg, edge_feat_dim=data.d_edge, node_feat_dim=data.d_node).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_param:,}")

    model = train(cfg, data, model, device)

    # shared eval protocol (identical bins/split/candidates as the main model)
    ev = SharedLinkEval(EvalConfig(bin_minutes=cfg.bin_minutes,
                                   train_days=cfg.train_days, val_days=cfg.val_days))
    out_dir = os.path.join(os.path.dirname(__file__), "predictions")
    os.makedirs(out_dir, exist_ok=True)
    for split in ["val", "test"]:
        out_csv = os.path.join(out_dir, f"graphmixer_pred_{split}.csv")
        pred = export_predictions(cfg, data, model, ev, split, device, out_csv)
        res = ev.score_binary(pred, split=split)
        print(f"[{split}] GraphMixer  AUC={res['auc']:.3f}  AP={res['ap']:.3f}  "
              f"F1={res['f1']:.3f}  Acc={res['accuracy']:.3f}  "
              f"(n_pos={res['n_pos']}/{res['n_total']})  -> {out_csv}")


if __name__ == "__main__":
    main()
