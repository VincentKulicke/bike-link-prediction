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

Aufruf:  python train_graphmixer.py
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
def train(cfg: GMConfig, data: GraphMixerData, model: GraphMixer, device,
          ev: SharedLinkEval | None = None, patience: int = 0,
          verbose: bool = True) -> GraphMixer:
    """Train GraphMixer on the same candidate distribution it is scored on.

    The previous version drew one negative per positive *edge event*, i.e. a
    50 % positive rate, while shared_eval scores a 1:5 mix (16.7 %). That left
    the model ~2.5x over-confident (mean score 0.409 against a true rate of
    0.166) and cost 0.219 F1 at threshold 0.5 -- almost half of its apparent
    gap to the hybrid was calibration, not modelling. It also queried at event
    timestamps while evaluation queries at bin starts.

    Now both come from ev.build_candidates(), exactly like the hybrid.

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
        print(f"Trainingspaare: {len(u_all):,} | positiv {100*y_all.mean():.1f}%")

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

        msg = f"Epoche {ep:2d}/{cfg.epochs} | Loss {total/max(1,nb):.4f}"
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
