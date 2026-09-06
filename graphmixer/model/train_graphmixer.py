# -*- coding: utf-8 -*-
"""
Training and prediction export for the GraphMixer baseline.

Ties the model (graphmixer.py) to the data (graphmixer_data.py), trains on the
training candidates and exports predictions for the shared candidate set.

Steps:
  1. set up data, model, optimizer
  2. training loop (binary cross-entropy)
  3. predict on the shared_eval candidates for val and test
  4. export as CSV (u, i, bin_idx, score), canonical 0-indexing
  5. score via shared_eval.SharedLinkEval.score_binary

Training draws from ev.build_candidates("train"), not one negative per positive.
The earlier 1:1 sampling left the model calibrated for a 50 % prior while the
evaluation runs at 1:5, which cost about 0.22 F1.

Usage:  python train_graphmixer.py
"""

from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# local modules
from graphmixer import GraphMixer, GMConfig
from graphmixer_data import GraphMixerData

# shared evaluation module, ../../evaluation
_EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation")
sys.path.insert(0, _EVAL_DIR)
from shared_eval import SharedLinkEval, EvalConfig   # noqa: E402


# --- numpy batch -> model pack (tokens, own_feat, neigh_mean) ---------------
def make_pack(model: GraphMixer, ef, dt, msk, own, neigh_mean, device):
    """Builds the input tensors from the collected raw arrays.

    tokens = [edge feature || fixed time encoding(dt)], padding rows zeroed.
    """
    ef = torch.as_tensor(ef, dtype=torch.float32, device=device)        # (B,K,d_edge)
    dt = torch.as_tensor(dt, dtype=torch.float32, device=device)        # (B,K)
    msk = torch.as_tensor(msk, dtype=torch.float32, device=device)      # (B,K)
    own = torch.as_tensor(own, dtype=torch.float32, device=device)      # (B,d_node)
    neigh_mean = torch.as_tensor(neigh_mean, dtype=torch.float32, device=device)

    time_feat = model.time_enc(dt)                                      # (B,K,time_dim)
    tokens = torch.cat([ef, time_feat], dim=-1)                         # (B,K,d_edge+time_dim)
    tokens = tokens * msk.unsqueeze(-1)                                 # zero the padding rows
    return tokens, own, neigh_mean


# --- training ----------------------------------------------------------------
def train(cfg: GMConfig, data: GraphMixerData, model: GraphMixer, device,
          ev: SharedLinkEval | None = None, patience: int = 0,
          verbose: bool = True) -> GraphMixer:
    """Train GraphMixer on the same candidate distribution it is scored on.

    Earlier versions sampled one negative per positive edge event (50 % positive)
    while shared_eval scores a 1:5 mix (16.7 %), and queried at event timestamps
    instead of bin starts. Both are fixed by drawing from ev.build_candidates().

    patience > 0 enables early stopping on validation AP.
    """
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    ev = ev or SharedLinkEval()

    cand = ev.build_candidates("train")
    if len(cand) > cfg.max_train_pairs:                 # uniform -> prior kept
        cand = cand.iloc[rng.choice(len(cand), cfg.max_train_pairs,
                                    replace=False)]
    u_all = cand["u"].to_numpy() + 1                    # canonical -> 1-indexed
    i_all = cand["i"].to_numpy() + 1
    t_all = cand["bin_idx"].to_numpy() * cfg.bin_seconds
    y_all = cand["label"].to_numpy(np.float32)
    if verbose:
        print(f"training pairs: {len(u_all):,} | positive {100*y_all.mean():.1f}%")

    best, best_state, bad, bs = -1.0, None, 0, cfg.batch_size
    for ep in range(1, cfg.epochs + 1):
        model.train()
        perm = rng.permutation(len(u_all))
        total, nb = 0.0, 0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            tq = t_all[b]
            u_pack = make_pack(model, *data.get_batch(u_all[b], tq, cfg.num_neighbors), device)
            i_pack = make_pack(model, *data.get_batch(i_all[b], tq, cfg.num_neighbors), device)
            logit = model(u_pack, i_pack)
            labels = torch.as_tensor(y_all[b], dtype=torch.float32, device=device)

            opt.zero_grad()
            loss = loss_fn(logit, labels)
            loss.backward()
            opt.step()
            total += loss.item(); nb += 1

        msg = f"epoch {ep:2d}/{cfg.epochs} | loss {total/max(1,nb):.4f}"
        if patience > 0:
            ap = _val_ap(cfg, data, model, ev, device)
            msg += f" | val AP {ap:.4f}"
            if ap > best:
                best, bad = ap, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(msg + f"  -> early stop (best {best:.4f})")
                    break
        if verbose:
            print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _val_ap(cfg: GMConfig, data: GraphMixerData, model: GraphMixer,
            ev: SharedLinkEval, device) -> float:
    """Validation AP, used only for early stopping."""
    from sklearn.metrics import average_precision_score
    model.eval()
    cand = ev.build_candidates("val")
    u = cand["u"].to_numpy() + 1; i = cand["i"].to_numpy() + 1
    tq = cand["bin_idx"].to_numpy() * cfg.bin_seconds
    out = np.zeros(len(cand), dtype=np.float32)
    for s in range(0, len(cand), 4096):
        sl = slice(s, s + 4096)
        up = make_pack(model, *data.get_batch(u[sl], tq[sl], cfg.num_neighbors), device)
        ip = make_pack(model, *data.get_batch(i[sl], tq[sl], cfg.num_neighbors), device)
        out[sl] = torch.sigmoid(model(up, ip)).cpu().numpy()
    return float(average_precision_score(cand["label"].to_numpy(), out))


# --- prediction export for the shared_eval candidates -----------------------
@torch.no_grad()
def export_predictions(cfg: GMConfig, data: GraphMixerData, model: GraphMixer,
                       ev: SharedLinkEval, split: str, device, out_csv: str) -> pd.DataFrame:
    """Scores every candidate cell (u, i, bin_idx).

    Query time is the start of the bin (bin_idx * bin_seconds). Candidate node
    IDs are canonical (0..231); internally shifted by +1.
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
    pred["score"] = scores            # u, i stay 0-indexed for shared_eval
    pred.to_csv(out_csv, index=False)
    return pred


# --- main --------------------------------------------------------------------
def main(cfg: GMConfig | None = None):
    cfg = cfg or GMConfig()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = GraphMixerData(cfg.prep_dir)
    print(f"nodes: {data.num_nodes} | edges: {len(data.edges):,} | "
          f"d_edge={data.d_edge} d_node={data.d_node}")

    model = GraphMixer(cfg, edge_feat_dim=data.d_edge, node_feat_dim=data.d_node).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_param:,}")

    model = train(cfg, data, model, device)

    # shared protocol: same bins, split and candidates as the main model
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
