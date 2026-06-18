# -*- coding: utf-8 -*-
"""
Training & Vorhersage-Export für die GraphMixer-Baseline.
=========================================================

Bindet Modell (graphmixer.py) und Daten (graphmixer_data.py) zusammen,
trainiert mit Negative Sampling auf den Trainingskanten und exportiert
Vorhersagen für die GEMEINSAME Kandidatenmenge des Eval-Moduls.

Ablauf:
  1. Daten + Modell + Optimizer aufsetzen
  2. Trainings-Schleife (binäre Cross-Entropy, 1 Negative je Positive)
  3. Vorhersagen für die shared_eval-Kandidaten (val/test) erzeugen
  4. Export als CSV (u, i, bin_idx, score) – KANONISCH 0-indiziert
  5. direkte Bewertung über shared_eval.SharedLinkEval.score_binary

Aufruf (z. B. in Colab):  python train_graphmixer.py
"""

from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# eigene Module
from graphmixer import GraphMixer, GMConfig
from graphmixer_data import GraphMixerData

# gemeinsames Eval-Modul (liegt unter ../../evaluation)
_EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation")
sys.path.insert(0, _EVAL_DIR)
from shared_eval import SharedLinkEval, EvalConfig   # noqa: E402


# ===========================================================================
# Hilfsfunktion: numpy-Batch -> Modell-"Pack" (tokens, own_feat, neigh_mean)
# ===========================================================================
def make_pack(model: GraphMixer, ef, dt, msk, own, neigh_mean, device):
    """Baut aus den gesammelten Rohdaten die Eingabe-Tensoren.

    tokens = [Kanten-Feature || feste Zeit-Kodierung(Δt)], Padding-Zeilen = 0.
    """
    ef = torch.as_tensor(ef, dtype=torch.float32, device=device)        # (B,K,d_edge)
    dt = torch.as_tensor(dt, dtype=torch.float32, device=device)        # (B,K)
    msk = torch.as_tensor(msk, dtype=torch.float32, device=device)      # (B,K)
    own = torch.as_tensor(own, dtype=torch.float32, device=device)      # (B,d_node)
    neigh_mean = torch.as_tensor(neigh_mean, dtype=torch.float32, device=device)

    time_feat = model.time_enc(dt)                                      # (B,K,time_dim)
    tokens = torch.cat([ef, time_feat], dim=-1)                         # (B,K,d_edge+time_dim)
    tokens = tokens * msk.unsqueeze(-1)                                 # Padding-Zeilen nullen
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
    print(f"Trainingskanten: {len(tr_idx):,}")

    bs = cfg.batch_size
    for ep in range(1, cfg.epochs + 1):
        model.train()
        perm = rng.permutation(tr_idx)
        total, nb = 0.0, 0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            u = u_all[b]; i = i_all[b]; t = ts_all[b]
            # Negative: gleicher Quellknoten u, zufälliges Ziel i_neg != u
            i_neg = rng.choice(nodes, size=len(b))
            resample = (i_neg == u)
            while resample.any():
                i_neg[resample] = rng.choice(nodes, size=int(resample.sum()))
                resample = (i_neg == u)

            # Rohdaten sammeln (Abfragezeit = Ereigniszeit t; nur Ereignisse < t)
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
        print(f"Epoche {ep:2d}/{cfg.epochs} | Loss {total/max(1,nb):.4f}")
    return model


# ===========================================================================
# 2) VORHERSAGE-EXPORT für die shared_eval-Kandidaten
# ===========================================================================
@torch.no_grad()
def export_predictions(cfg: GMConfig, data: GraphMixerData, model: GraphMixer,
                       ev: SharedLinkEval, split: str, device, out_csv: str) -> pd.DataFrame:
    """Berechnet für jede Kandidaten-Zelle (u, i, bin_idx) einen Link-Score.

    Abfragezeit = Beginn des Bins (bin_idx * bin_seconds).
    Knoten-IDs der Kandidaten sind KANONISCH (0..231); intern +1 (1-indiziert).
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
        u_node = u_can[sl] + 1            # kanonisch -> 1-indiziert
        i_node = i_can[sl] + 1
        tq = t_query[sl]
        u_pack = make_pack(model, *data.get_batch(u_node, tq, cfg.num_neighbors), device)
        i_pack = make_pack(model, *data.get_batch(i_node, tq, cfg.num_neighbors), device)
        logit = model(u_pack, i_pack)
        scores[sl] = torch.sigmoid(logit).cpu().numpy()

    pred = cand.copy()
    pred["score"] = scores            # u, i bleiben kanonisch 0-indiziert (für shared_eval)
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
    print(f"Knoten: {data.num_nodes} | Kanten: {len(data.edges):,} | "
          f"d_edge={data.d_edge} d_node={data.d_node}")

    model = GraphMixer(cfg, edge_feat_dim=data.d_edge, node_feat_dim=data.d_node).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Modellparameter: {n_param:,}")

    model = train(cfg, data, model, device)

    # gemeinsames Eval-Protokoll (identische Bins/Split/Kandidaten wie das Hauptmodell)
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
