# -*- coding: utf-8 -*-
"""
LSTM-Baseline (Count) – reine Zeitreihen-Vorhersage.
====================================================

Vergleich 2 der Aufgabenstellung: prognostiziert pro Stationspaar (u->i) die
Fahrtenzahl im naechsten 30-Minuten-Bin aus der vergangenen Count-Reihe dieses
Paares (= Differenz der num_rides-Zeitreihe). KEINE Graphstruktur.

Designentscheidungen (abgestimmt):
  - EIN globales LSTM ueber die Count-Reihen aller Kanten (Multi-Series).
  - Lookback = 48 Bins (24 h).
  - Univariate Eingabe (nur vergangene Counts).
  - Trainingsziel = MSE (entspricht der Evaluationsmetrik).

Bindet das gemeinsame Eval-Modul an: exportiert Vorhersagen als
(u, i, bin_idx, pred_count) und bewertet via SharedLinkEval.score_count.

Lokal ist PyTorch optional installierbar; ansonsten Colab/GPU.
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


# ===========================================================================
# 1) KONFIGURATION
# ===========================================================================
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


# ===========================================================================
# 2) DATEN: Count-Matrix (Paar x Bin) aus der SUPEREDGE-Zeitreihe bauen
# ===========================================================================
class CountSeries:
    """Baut aus der aggregierten Superedge-Zeitreihe (num_rides je Bin) eine
    dichte Count-Matrix (Paare x Bins). Genau diese Reihe ist der LSTM-Input."""

    def __init__(self, ev: SharedLinkEval):
        raw = ev._load_raw()                           # Spalten: u, i, bin_idx, count (Superedge)
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


# ===========================================================================
# 3) TRAININGS-DATENSATZ (Sliding Windows ueber alle Paare im Trainingszeitraum)
# ===========================================================================
class WindowDataset(Dataset):
    def __init__(self, cs: CountSeries, cfg: LSTMConfig, train_end_bin: int, rng):
        self.cs = cs; self.lb = cfg.lookback
        # alle (Zeilen-, t)-Indizes mit gueltigem Fenster im Trainingszeitraum
        n_rows = cs.counts.shape[0]
        ts = np.arange(cfg.lookback, train_end_bin)
        rows = np.arange(n_rows)
        # kartesisches Produkt, dann subsamplen
        total = n_rows * len(ts)
        if total > cfg.max_train_samples:
            ridx = rng.integers(0, n_rows, size=cfg.max_train_samples)
            tidx = rng.integers(cfg.lookback, train_end_bin, size=cfg.max_train_samples)
        else:
            ridx = np.repeat(rows, len(ts)); tidx = np.tile(ts, n_rows)
        self.row = ridx.astype(np.int32); self.t = tidx.astype(np.int32)
        print(f"Trainingsfenster: {len(self.row):,} (von theoretisch {total:,})")

    def __len__(self): return len(self.row)

    def __getitem__(self, k):
        r = int(self.row[k]); t = int(self.t[k])
        x = self.cs.counts[r, t - self.lb:t].copy()          # (lookback,)
        y = self.cs.counts[r, t]                              # Skalar
        return torch.from_numpy(x).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)


# ===========================================================================
# 4) MODELL
# ===========================================================================
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


# ===========================================================================
# 5) TRAINING
# ===========================================================================
def train(cfg, cs, model, train_end_bin, device, rng):
    ds = WindowDataset(cs, cfg, train_end_bin, rng)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    for ep in range(1, cfg.epochs + 1):
        model.train(); total, nb = 0.0, 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
            total += loss.item(); nb += 1
        print(f"Epoche {ep:2d}/{cfg.epochs} | MSE {total/max(1,nb):.4f}")
    return model


# ===========================================================================
# 6) VORHERSAGE-EXPORT fuer die shared_eval-Kandidaten
# ===========================================================================
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


# ===========================================================================
# 7) MAIN
# ===========================================================================
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
