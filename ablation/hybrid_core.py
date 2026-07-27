# -*- coding: utf-8 -*-
"""
hybrid_core.py — reusable hybrid model for the ablation study.
==============================================================

Pulls the logic out of  hybrid_model/iteration2_graphsage_gru_hurdle.ipynb
into a module we can drive with a grid search. The time-series encoder is
swappable:

    encoder = "gru"   -> nn.GRU        (iteration 2, the reference)
    encoder = "cnn"   -> 1D convolution (ablation: GraphSAGE + 1D-CNN)

Both encoders share the same interface:  (B, L, C) -> (B, hidden). That keeps
the rest of the architecture (GraphSAGE, fusion, hurdle heads) identical, so
the comparison stays fair.

Selection protocol (important):
  - Hyperparameters are picked from VALIDATION metrics only.
  - The test split is touched exactly once, for the final (best) config.
  - Every model runs through the same shared_eval protocol.

Usually called from grid_hybrid.py.
"""
from __future__ import annotations
import os, sys, time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# --- locate prepared Data / evaluation relative to this script ---------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PREP = os.path.abspath(os.path.join(_HERE, "..", "prepared Data"))
_EVAL = os.path.abspath(os.path.join(_HERE, "..", "evaluation"))
sys.path.insert(0, _EVAL)
from shared_eval import SharedLinkEval, EvalConfig   # noqa: E402


# ===========================================================================
# 1) CONFIG  (tuned: lr, hidden, lambda_count, kernel_size)
# ===========================================================================
@dataclass
class HybridCfg:
    encoder: str = "gru"          # "gru" | "cnn"
    ts_lookback: int = 12         # GRU/CNN input: last 12 bins (6 h)
    hidden: int = 64              # couples sage_out == enc_hidden
    fusion_hidden: int = 128
    kernel_size: int = 3          # only used for encoder == "cnn"
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 15
    batch_size: int = 1024
    lambda_count: float = 1.0
    seed: int = 42

    def tag(self) -> str:
        base = f"{self.encoder}_lr{self.lr:g}_h{self.hidden}_lam{self.lambda_count:g}"
        if self.encoder == "cnn":
            base += f"_k{self.kernel_size}"
        return base


# ===========================================================================
# 2) LOAD DATA once  (shared across all grid runs)
# ===========================================================================
class HybridData:
    """Loads and normalizes every input exactly once, then gets handed to each
    run_hybrid() call so the grid doesn't reload the data 40 times."""

    def __init__(self, device: str | None = None, lookback: int = 12):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lookback = lookback
        self.ev = SharedLinkEval()
        bins_per_day = (24 * 60) // self.ev.cfg.bin_minutes
        self.train_end_bin = self.ev.cfg.train_days * bins_per_day

        node_static = np.load(os.path.join(_PREP, "node_static.npy"))   # (N,3)
        node_avail  = np.load(os.path.join(_PREP, "node_avail.npy"))    # (N,T,4)
        edge_index  = np.load(os.path.join(_PREP, "edge_index.npy"))    # (2,E)
        edge_weight = np.load(os.path.join(_PREP, "edge_weight.npy"))   # (E,)
        self.N, self.T, self.C = node_avail.shape

        # z-score the static features
        mu_s, sd_s = node_static.mean(0), node_static.std(0) + 1e-6
        self.static_x = ((node_static - mu_s) / sd_s).astype(np.float32)
        self.node_static = node_static

        # z-score availability per channel (train bins only -> no leakage)
        tr = node_avail[:, :self.train_end_bin, :]
        mu_a = tr.mean((0, 1)); sd_a = tr.std((0, 1)) + 1e-6
        self.avail_n = ((node_avail - mu_a) / sd_a).astype(np.float32)

        # symmetric, row-normalized adjacency (weight = trip frequency)
        A = np.zeros((self.N, self.N), dtype=np.float32)
        for (u, i), w in zip(edge_index.T, edge_weight):
            A[u, i] += w; A[i, u] += w
        deg = A.sum(1, keepdims=True) + 1e-6
        self.A_norm = torch.tensor(A / deg, device=self.device)
        self.static_x_t = torch.tensor(self.static_x, device=self.device)

        # per-pair training frequency, used as a pair feature
        self.freq = {}
        for (u, i), w in zip(edge_index.T, edge_weight):
            self.freq[(int(u), int(i))] = float(w)
        self.WD_START = 3   # 2024-05-16 is a Thursday (Mon=0)

        # cache the fixed, seeded candidate set per split
        self.cand = {s: self.ev.build_candidates(s) for s in ["train", "val", "test"]}

        # --- precompute windows + pair features per split, exactly ONCE --------
        # These only depend on (u, i, bin) and the lookback, not on the model.
        # Computing them once and sharing them as GPU tensors speeds the grid up
        # by ~15-40x (no per-epoch, per-run Python loop anymore).
        t0 = time.time()
        self.tensors = {s: self._precompute(self.cand[s]) for s in ["train", "val", "test"]}
        print(f"[HybridData] N,T,C = {self.N},{self.T},{self.C} | device={self.device} "
              f"| lookback={self.lookback} | train_cands={len(self.cand['train'])} "
              f"| precompute {time.time()-t0:.1f}s")

    # vectorized windows (B, L, C), zero-padded for early bins
    def _windows(self, nodes: np.ndarray, bins: np.ndarray) -> np.ndarray:
        L = self.lookback
        idx = bins[:, None] - L + np.arange(L)[None, :]     # (B, L) time indices
        valid = idx >= 0
        idx_c = np.clip(idx, 0, self.T - 1)
        nodes_rep = np.repeat(nodes[:, None], L, axis=1)     # (B, L)
        out = self.avail_n[nodes_rep, idx_c]                 # (B, L, C)
        out[~valid] = 0.0
        return out.astype(np.float32)

    # vectorized pair features: log1p(frequency), distance, cyclic time
    def _pair_feats(self, u: np.ndarray, i: np.ndarray, b: np.ndarray) -> np.ndarray:
        bm = self.ev.cfg.bin_minutes
        fr = np.array([self.freq.get((int(u[k]), int(i[k])), 0.0) for k in range(len(u))],
                      dtype=np.float32)
        dx = self.node_static[u, 1] - self.node_static[i, 1]
        dy = self.node_static[u, 2] - self.node_static[i, 2]
        hour = ((b * bm) // 60) % 24
        dow = (self.WD_START + (b * bm) // (60 * 24)) % 7
        f = np.stack([np.log1p(fr), np.sqrt(dx * dx + dy * dy),
                      np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                      np.sin(2 * np.pi * dow / 7),  np.cos(2 * np.pi * dow / 7)], axis=1)
        return f.astype(np.float32)

    def _precompute(self, cand: pd.DataFrame) -> dict:
        u = cand["u"].to_numpy(); i = cand["i"].to_numpy(); b = cand["bin_idx"].to_numpy()
        d = self.device
        out = {
            "u":    torch.tensor(u, dtype=torch.long, device=d),
            "i":    torch.tensor(i, dtype=torch.long, device=d),
            "win_u": torch.tensor(self._windows(u, b), device=d),
            "win_i": torch.tensor(self._windows(i, b), device=d),
            "pf":   torch.tensor(self._pair_feats(u, i, b), device=d),
        }
        if "label" in cand.columns:
            out["y"] = torch.tensor(cand["label"].to_numpy(np.float32), device=d)
        if "count" in cand.columns:
            out["c"] = torch.tensor(cand["count"].to_numpy(np.float32), device=d)
        return out


# ===========================================================================
# 3) GRAPH BRANCH (GraphSAGE) — same as in the notebook
# ===========================================================================
class SAGELayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_neigh = nn.Linear(d_in, d_out)

    def forward(self, x, A_norm):
        neigh = A_norm @ x
        return self.lin_self(x) + self.lin_neigh(neigh)


class GraphSAGE(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout):
        super().__init__()
        self.l1 = SAGELayer(d_in, d_hidden); self.l2 = SAGELayer(d_hidden, d_out)
        self.act = nn.ReLU(); self.do = nn.Dropout(dropout)

    def forward(self, x, A_norm):
        h = self.do(self.act(self.l1(x, A_norm)))
        return self.act(self.l2(h, A_norm))


# ===========================================================================
# 4) TIME-SERIES ENCODER — swappable (GRU vs. 1D-CNN)
#    Both map (B, L, C) -> (B, hidden)
# ===========================================================================
class GRUEncoder(nn.Module):
    def __init__(self, n_channels, hidden):
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden, batch_first=True)

    def forward(self, win):
        g, _ = self.gru(win)          # (B, L, H)
        return g[:, -1, :]            # last hidden state


class CNN1DEncoder(nn.Module):
    """1D convolution over the time axis. An odd kernel_size with padding=k//2
    keeps the sequence length fixed; global average pooling collapses to (B, H)."""
    def __init__(self, n_channels, hidden, kernel_size=3, dropout=0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd (padding='same')"
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(n_channels, hidden, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size, padding=pad)
        self.act = nn.ReLU(); self.do = nn.Dropout(dropout)

    def forward(self, win):
        x = win.transpose(1, 2)                 # (B, C, L)
        x = self.do(self.act(self.conv1(x)))    # (B, H, L)
        x = self.do(self.act(self.conv2(x)))    # (B, H, L)
        return x.mean(dim=-1)                    # global avg pool -> (B, H)


def build_encoder(cfg: HybridCfg, n_channels: int) -> nn.Module:
    if cfg.encoder == "gru":
        return GRUEncoder(n_channels, cfg.hidden)
    if cfg.encoder == "cnn":
        return CNN1DEncoder(n_channels, cfg.hidden, cfg.kernel_size, cfg.dropout)
    raise ValueError(f"Unknown encoder: {cfg.encoder!r}")


# ===========================================================================
# 5) HYBRID MODEL (GraphSAGE + encoder + fusion + hurdle heads)
# ===========================================================================
class HybridHurdle(nn.Module):
    def __init__(self, cfg: HybridCfg, n_static, n_channels, n_pair):
        super().__init__()
        self.sage = GraphSAGE(n_static, cfg.hidden, cfg.hidden, cfg.dropout)
        self.enc  = build_encoder(cfg, n_channels)
        node_dim  = cfg.hidden + cfg.hidden          # sage_out + enc_hidden
        fuse_in   = 2 * node_dim + n_pair
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, cfg.fusion_hidden), nn.ReLU(), nn.Dropout(cfg.dropout))
        self.head_bin   = nn.Linear(cfg.fusion_hidden, 1)
        self.head_count = nn.Linear(cfg.fusion_hidden, 1)
        self.softplus = nn.Softplus()

    def node_repr(self, sage_emb, idx, win):
        return torch.cat([sage_emb[idx], self.enc(win)], dim=-1)

    def forward(self, sage_emb, u, i, win_u, win_i, pf):
        hu = self.node_repr(sage_emb, u, win_u)
        hi = self.node_repr(sage_emb, i, win_i)
        z = self.fusion(torch.cat([hu, hi, pf], dim=-1))
        logit = self.head_bin(z).squeeze(-1)
        count = self.softplus(self.head_count(z)).squeeze(-1)
        return logit, count


# ===========================================================================
# 6) TRAIN + PREDICT + SCORE
# ===========================================================================
def _predict(model, data: HybridData, cfg: HybridCfg, split: str) -> pd.DataFrame:
    model.eval()
    cand = data.cand[split]; t = data.tensors[split]
    n = len(cand)
    with torch.no_grad():
        sage_emb = model.sage(data.static_x_t, data.A_norm)
        scores = np.zeros(n, dtype=np.float32)
        counts = np.zeros(n, dtype=np.float32)
        for s in range(0, n, 4096):
            sl = slice(s, s + 4096)
            logit, count = model(sage_emb, t["u"][sl], t["i"][sl],
                                 t["win_u"][sl], t["win_i"][sl], t["pf"][sl])
            scores[sl] = torch.sigmoid(logit).cpu().numpy()
            counts[sl] = count.cpu().numpy()
    out = cand[["u", "i", "bin_idx"]].copy()
    out["score"] = scores; out["pred_count"] = counts
    return out


def run_hybrid(cfg: HybridCfg, data: HybridData,
               eval_splits=("val",), export_dir: str | None = None,
               verbose: bool = False, return_model: bool = False) -> dict:
    """Train one hybrid model and score it on the given splits.

    Grid search:  eval_splits=("val",)  -> validation only.
    Final run:    eval_splits=("val","test")  and set export_dir.
    return_model=True adds the trained model to the result (e.g. for the
    ranking evaluation, which scores its own candidate set). Pass
    eval_splits=() to only train.
    """
    assert cfg.ts_lookback == data.lookback, (
        f"cfg.ts_lookback={cfg.ts_lookback} != data.lookback={data.lookback}; "
        f"re-create HybridData with a matching lookback.")
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = data.device
    model = HybridHurdle(cfg, n_static=3, n_channels=data.C, n_pair=6).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    bce = nn.BCEWithLogitsLoss(); mse = nn.MSELoss()

    tr = data.tensors["train"]
    n = len(data.cand["train"]); rng = np.random.default_rng(cfg.seed)

    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train(); perm = torch.tensor(rng.permutation(n), device=device); tot = 0.0; nb = 0
        for s in range(0, n, cfg.batch_size):
            bi = perm[s:s + cfg.batch_size]
            sage_emb = model.sage(data.static_x_t, data.A_norm)
            logit, count = model(sage_emb, tr["u"][bi], tr["i"][bi],
                                 tr["win_u"][bi], tr["win_i"][bi], tr["pf"][bi])
            loss = bce(logit, tr["y"][bi]) + cfg.lambda_count * mse(count, tr["c"][bi])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if verbose:
            print(f"  epoch {ep:2d}/{cfg.epochs} | loss {tot/max(1,nb):.4f}")

    res = {"cfg": cfg, "train_s": round(time.time() - t0, 1)}
    for split in eval_splits:
        pred = _predict(model, data, cfg, split)
        rb = data.ev.score_binary(pred, split=split)
        rc = data.ev.score_count(pred, split=split)
        res[split] = {"auc": rb["auc"], "ap": rb["ap"], "f1": rb["f1"],
                      "accuracy": rb["accuracy"], "mse": rc["mse"],
                      "mae": rc["mae"], "rmse": rc["rmse"]}
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
            pred.to_csv(os.path.join(export_dir, f"{cfg.encoder}_pred_{split}.csv"),
                        index=False)
    if return_model:
        res["model"] = model
    return res
