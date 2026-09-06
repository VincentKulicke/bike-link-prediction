# -*- coding: utf-8 -*-
"""
LSTM count baseline: pure time-series prediction, no graph structure.

Comparison 2 of the task. For each station pair (u->i) it predicts the number
of trips in the next 30-minute bin from that pair's own past count series,
i.e. the differenced num_rides series.

Design:
  - one global LSTM over the count series of all edges (multi-series)
  - univariate input, past counts only
  - MSE objective, matching the evaluation metric

The defaults below (lookback 48, hidden 64, 1 layer) are the starting point;
the tuned configuration from the final search is wider and deeper
(lookback 192, 2 layers, dropout 0.2) and lives in
ablation/results/hpo_final_lstm.csv.

Exports predictions as (u, i, bin_idx, pred_count) and scores them through
SharedLinkEval.score_count.

Needs PyTorch. A GPU is optional.
"""
from __future__ import annotations
import os, sys, time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# gemeinsames Eval-Modul (liegt unter ../evaluation)
_EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "evaluation")
sys.path.insert(0, _EVAL_DIR)
from shared_eval import SharedLinkEval, EvalConfig   # noqa: E402


# --- config ------------------------------------------------------------------
@dataclass
class LSTMConfig:
    lookback: int = 48             # Eingabefenster (Bins) = 24 h bei 30-min-Bins
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 512
    max_train_samples: int = 400_000   # Subsampling der Trainingsfenster (Tempo)
    seed: int = 42


# --- data: build the count matrix (pair x bin) from the superedge series ----
class CountSeries:
    """Dense count matrix (pairs x bins) built from the aggregated superedge
    series (num_rides per bin). This series is the LSTM input."""

    def __init__(self, ev: SharedLinkEval):
        self.ev = ev                                   # kept so the training set can use the same candidates
        raw = ev._load_raw()                           # columns: u, i, bin_idx, count (super-edge)
        self.n_bins = int(raw["bin_idx"].max()) + 1
        # eindeutige Paare -> Zeilenindex
        pairs = raw[["u", "i"]].drop_duplicates().reset_index(drop=True)
        self.pair_to_row = {(int(u), int(i)): r for r, (u, i) in
                            enumerate(zip(pairs["u"], pairs["i"]))}
        self.counts = np.zeros((len(pairs), self.n_bins), dtype=np.float32)
        for u, i, b, c in zip(raw["u"], raw["i"], raw["bin_idx"], raw["count"]):
            self.counts[self.pair_to_row[(int(u), int(i))], int(b)] = c
        print(f"Count-Matrix (Superedge): {self.counts.shape[0]} Paare x {self.n_bins} Bins "
              f"| belegte Zellen: {int((self.counts > 0).sum()):,}")

    def window(self, u: int, i: int, bin_idx: int, lookback: int) -> np.ndarray:
        """Vergangene `lookback` Counts vor bin_idx (zero-gepolstert, falls Paar/Anfang fehlt)."""
        row = self.pair_to_row.get((int(u), int(i)))
        w = np.zeros(lookback, dtype=np.float32)
        if row is None:
            return w
        lo = max(0, bin_idx - lookback)
        seg = self.counts[row, lo:bin_idx]
        if len(seg) > 0:
            w[lookback - len(seg):] = seg
        return w


# --- training set: sliding windows over all pairs in the training period -----
class WindowDataset(Dataset):
    """Training windows drawn from the SAME candidate set that shared_eval uses
    for scoring (positives + negatives at neg_ratio).

    Sampling uniformly over the raw count matrix instead would train on a very
    different distribution than we evaluate on: the full matrix is 98.9% zeros
    (mean 0.013), the 1:5 candidate set is 83.3% zeros (mean 0.189). A model
    fitted on the first one minimises its MSE by predicting near zero, which is
    systematically too low on the second - the LSTM ended up worse than simply
    predicting the mean. Using the candidates here keeps the baseline on the
    same footing as the hybrid, which trains on these candidates too.
    """

    def __init__(self, cs: CountSeries, cfg: LSTMConfig, train_end_bin: int, rng,
                 split: str = "train"):
        self.cs = cs; self.lb = cfg.lookback
        cand = cs.ev.build_candidates(split)
        u = cand["u"].to_numpy(); i = cand["i"].to_numpy()
        b = cand["bin_idx"].to_numpy(); y = cand["count"].to_numpy(dtype=np.float32)

        if split == "train" and len(cand) > cfg.max_train_samples:
            sel = rng.choice(len(cand), cfg.max_train_samples, replace=False)
            u, i, b, y = u[sel], i[sel], b[sel], y[sel]

        # precompute the lookback windows (vectorised; the per-item dict lookup
        # was the bottleneck in the old version)
        rows = np.array([cs.pair_to_row.get((int(a), int(c)), -1)
                         for a, c in zip(u, i)], dtype=np.int64)
        t_idx = b[:, None] - self.lb + np.arange(self.lb)[None, :]     # (N, lb)
        ok = (t_idx >= 0) & (rows[:, None] >= 0)
        X = cs.counts[np.clip(rows, 0, None)[:, None],
                      np.clip(t_idx, 0, cs.n_bins - 1)]
        self.X = np.where(ok, X, 0.0).astype(np.float32)
        self.y = y

        if split == "train":
            print(f"Training windows: {len(self.y):,} from shared_eval candidates "
                  f"| zeros {100.0 * (y == 0).mean():.1f}% | mean target {y.mean():.4f}")

    def __len__(self): return len(self.y)

    def __getitem__(self, k):
        return (torch.from_numpy(self.X[k]).unsqueeze(-1),
                torch.tensor(self.y[k], dtype=torch.float32))


# --- model -------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=cfg.hidden_dim,
                            num_layers=cfg.num_layers, batch_first=True,
                            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0)
        self.head = nn.Linear(cfg.hidden_dim, 1)
        self.softplus = nn.Softplus()                        # erzwingt nicht-negative Counts

    def forward(self, x):
        out, _ = self.lstm(x)            # (B, L, H)
        last = out[:, -1, :]             # letzter Zeitschritt
        return self.softplus(self.head(last)).squeeze(-1)    # (B,)


# --- training ----------------------------------------------------------------
def train(cfg, cs, model, train_end_bin, device, rng, patience: int = 0,
          verbose: bool = True):
    """patience > 0 enables early stopping on validation MSE."""
    ds = WindowDataset(cs, cfg, train_end_bin, rng)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    vX = vy = None
    if patience > 0:                      # windows built once, not per epoch
        vds = WindowDataset(cs, cfg, train_end_bin, rng, split="val")
        vX = torch.from_numpy(vds.X).unsqueeze(-1).to(device)
        vy = torch.from_numpy(vds.y).to(device)

    best, best_state, bad, stopped_at = float("inf"), None, 0, cfg.epochs
    for ep in range(1, cfg.epochs + 1):
        model.train(); total, nb = 0.0, 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
            total += loss.item(); nb += 1
        msg = f"Epoche {ep:2d}/{cfg.epochs} | MSE {total/max(1,nb):.4f}"

        if patience > 0:
            model.eval()
            with torch.no_grad():
                out = torch.cat([model(vX[s:s + 8192]) for s in range(0, len(vX), 8192)])
                vmse = float(((out - vy) ** 2).mean())
            msg += f" | val MSE {vmse:.4f}"
            if vmse < best:
                best, bad = vmse, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    stopped_at = ep
                    if verbose:
                        print(msg + f"  -> early stop (best {best:.4f})")
                    break
        if verbose:
            print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.epochs_run = stopped_at
    return model


# --- prediction export for the shared_eval candidates -----------------------
@torch.no_grad()
def export(cfg, cs, model, ev, split, device, out_csv):
    model.eval()
    cand = ev.build_candidates(split)[["u", "i", "bin_idx"]].copy()
    N = len(cand)
    X = np.zeros((N, cfg.lookback), dtype=np.float32)
    u = cand["u"].to_numpy(); i = cand["i"].to_numpy(); b = cand["bin_idx"].to_numpy()
    for k in range(N):
        X[k] = cs.window(u[k], i[k], int(b[k]), cfg.lookback)
    preds = np.zeros(N, dtype=np.float32)
    bs = 8192
    for s in range(0, N, bs):
        xb = torch.from_numpy(X[s:s+bs]).unsqueeze(-1).to(device)
        preds[s:s+bs] = model(xb).cpu().numpy()
    pred = cand.copy(); pred["pred_count"] = preds
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pred.to_csv(out_csv, index=False)
    return pred


# --- main --------------------------------------------------------------------
def main(cfg: LSTMConfig | None = None):
    cfg = cfg or LSTMConfig()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ev = SharedLinkEval()                                  # identisches Protokoll wie alle Modelle
    cs = CountSeries(ev)
    bins_per_day = (24 * 60) // ev.cfg.bin_minutes
    train_end_bin = ev.cfg.train_days * bins_per_day
    print(f"Train-Bins < {train_end_bin} | Lookback {cfg.lookback}")

    model = LSTMForecaster(cfg).to(device)
    print(f"Parameter: {sum(p.numel() for p in model.parameters()):,}")
    model = train(cfg, cs, model, train_end_bin, device, rng)

    out_dir = os.path.join(os.path.dirname(__file__), "predictions")
    for split in ["val", "test"]:
        out_csv = os.path.join(out_dir, f"lstm_pred_{split}.csv")
        pred = export(cfg, cs, model, ev, split, device, out_csv)
        res = ev.score_count(pred, split=split)
        print(f"[{split}] LSTM  MSE={res['mse']:.4f}  MAE={res['mae']:.4f}  "
              f"RMSE={res['rmse']:.4f}  (n={res['n_total']})  -> {out_csv}")


if __name__ == "__main__":
    main()
