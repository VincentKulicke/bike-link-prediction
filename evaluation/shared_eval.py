# -*- coding: utf-8 -*-
"""
Gemeinsames, MODELL-AGNOSTISCHES Evaluationsmodul fuer das Link-Prediction-Projekt.

Zweck
-----
Definiert Ground Truth, temporalen Split und Kandidatenpaare GENAU EINMAL, sodass
jedes Modell (GraphMixer-Baseline, LSTM-Baseline, euer Hybridmodell) auf exakt
demselben Protokoll bewertet wird. Jedes Modell liefert nur Vorhersagen in einem
gemeinsamen Format zurueck; dieses Modul rechnet die Metriken.

Zwei Aufgaben (gemaess Aufgabenstellung)
---------------------------------------
1. BINAER  : Gibt es im Zeitfenster eine Verbindung (u->v)? -> AUC, AP, F1, Accuracy
             (Vergleich: euer Modell vs. GraphMixer)
2. COUNT   : Wie viele Fahrten zwischen u->v im Fenster?     -> MSE, MAE, RMSE
             (Vergleich: euer Modell vs. LSTM)
             Ground Truth = Differenz der num_rides-Zeitreihe (jede Fahrt = +1).

Schnittstelle fuer Modelle
--------------------------
Ein Modell erzeugt eine Vorhersage-Tabelle mit Spalten:
    u, i, bin_idx, score        (binaer: Wahrscheinlichkeit/Logit-Sigmoid in [0,1])
    u, i, bin_idx, pred_count   (count : nicht-negative Zahl)
Knoten-IDs in KANONISCHER Form (0..231, siehe node_index.csv).
GraphMixer-Ausgaben (1-indiziert) vorher um -1 verschieben.

Dann:
    ev = SharedLinkEval()
    targets = ev.build_targets()
    cand    = ev.build_candidates(split="test")          # feste, geseedete Kandidaten
    # ... Modell erzeugt Scores fuer genau diese (u,i,bin_idx) ...
    print(ev.score_binary(pred_df, split="test"))
    print(ev.score_count(pred_df,  split="test"))

Das Modul hat KEINE Abhaengigkeit zu sklearn/scipy (Metriken sind selbst
implementiert), damit es ueberall laeuft.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Konfiguration des Protokolls (fuer ALLE Modelle identisch)
# ---------------------------------------------------------------------------
PREP_DIR = os.path.join(os.path.dirname(__file__), "..", "prepared Data")

@dataclass
class EvalConfig:
    edges_csv: str = os.path.join(PREP_DIR, "graphmixer_edges.csv")  # kanonisch 0-indiziert
    bin_minutes: int = 30          # Laenge eines Vorhersage-Fensters
    # Temporaler Split in Tagen (Fenster ~29 Tage): Train 21 / Val 4 / Test 4
    train_days: int = 21
    val_days: int = 4
    neg_ratio: int = 5             # negative je positive Zelle (binaere Eval)
    seed: int = 42

    @property
    def bin_seconds(self) -> int:
        return self.bin_minutes * 60


# ---------------------------------------------------------------------------
# Metriken (ohne externe Abhaengigkeiten)
# ---------------------------------------------------------------------------
def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC via Mann-Whitney-U (rangbasiert)."""
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    n_pos = int((y_true == 1).sum()); n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # Bindungen (gleiche Scores) mitteln
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    sum_pos = ranks[y_true == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    yt = y_true[order]
    tp = np.cumsum(yt)
    precision = tp / np.arange(1, len(yt) + 1)
    recall = tp / yt.sum()
    rprev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - rprev) * precision))

def binary_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> dict:
    y_true = np.asarray(y_true); pred = (np.asarray(y_score) >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc  = (tp + tn) / max(1, tp + fp + fn + tn)
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}

def count_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt = np.asarray(y_true, dtype=float); yp = np.asarray(y_pred, dtype=float)
    err = yp - yt
    return {"mse": float(np.mean(err**2)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2)))}


# ---------------------------------------------------------------------------
# Gemeinsames Eval-Objekt
# ---------------------------------------------------------------------------
class SharedLinkEval:
    def __init__(self, config: EvalConfig | None = None):
        self.cfg = config or EvalConfig()
        self._trips = None
        self._targets = None

    # ---- Rohtrips laden (kanonisch 0-indiziert) ----
    def _load_trips(self) -> pd.DataFrame:
        if self._trips is None:
            df = pd.read_csv(self.cfg.edges_csv, usecols=["u", "i", "ts"])
            df["bin_idx"] = (df["ts"].values // self.cfg.bin_seconds).astype(int)
            self._trips = df
        return self._trips

    @property
    def n_bins(self) -> int:
        df = self._load_trips()
        return int(df["bin_idx"].max()) + 1

    def _split_of_bin(self, bin_idx: np.ndarray) -> np.ndarray:
        bins_per_day = (24 * 60) // self.cfg.bin_minutes
        train_end = self.cfg.train_days * bins_per_day
        val_end = train_end + self.cfg.val_days * bins_per_day
        split = np.where(bin_idx < train_end, "train",
                 np.where(bin_idx < val_end, "val", "test"))
        return split

    # ---- Ground Truth: count & label je (u,i,bin) ----
    def build_targets(self) -> pd.DataFrame:
        """Positive Zellen: count = Anzahl Trips in (u,i,bin) = Delta num_rides.
        label = 1 (count>0). Negative werden separat in build_candidates gezogen."""
        if self._targets is not None:
            return self._targets
        df = self._load_trips()
        g = (df.groupby(["u", "i", "bin_idx"]).size()
               .reset_index(name="count"))
        g["label"] = 1
        g["split"] = self._split_of_bin(g["bin_idx"].values)
        self._targets = g
        return g

    # ---- Kandidatenmenge (Positives + geseedete Negatives) ----
    def build_candidates(self, split: str = "test") -> pd.DataFrame:
        """Feste, reproduzierbare Kandidatenmenge fuer einen Split.
        Positives = beobachtete (u,i,bin); Negatives = zufaellige (u,i,bin) mit count=0.
        Alle Modelle MUESSEN auf genau diese (u,i,bin_idx) Vorhersagen liefern."""
        tg = self.build_targets()
        pos = tg[tg["split"] == split][["u", "i", "bin_idx", "count"]].copy()
        pos["label"] = 1
        if len(pos) == 0:
            raise ValueError(f"Keine Positives im Split '{split}'.")

        df = self._load_trips()
        nodes = np.unique(np.concatenate([df["u"].values, df["i"].values]))
        bins = pos["bin_idx"].unique()
        rng = np.random.default_rng(self.cfg.seed)
        pos_set = set(zip(pos["u"], pos["i"], pos["bin_idx"]))
        n_neg = self.cfg.neg_ratio * len(pos)

        neg = []
        attempts = 0
        max_attempts = n_neg * 20
        while len(neg) < n_neg and attempts < max_attempts:
            u = rng.choice(nodes); i = rng.choice(nodes); b = int(rng.choice(bins))
            attempts += 1
            if u == i:
                continue
            if (u, i, b) in pos_set:
                continue
            neg.append((u, i, b))
        neg = pd.DataFrame(neg, columns=["u", "i", "bin_idx"])
        neg["count"] = 0; neg["label"] = 0

        cand = pd.concat([pos, neg], ignore_index=True)
        return cand.sample(frac=1.0, random_state=self.cfg.seed).reset_index(drop=True)

    # ---- Bewertung: Vorhersagen einklinken ----
    def _join(self, pred_df: pd.DataFrame, split: str, value_col: str) -> pd.DataFrame:
        cand = self.build_candidates(split)
        keys = ["u", "i", "bin_idx"]
        merged = cand.merge(pred_df[keys + [value_col]], on=keys, how="left")
        missing = merged[value_col].isna().sum()
        if missing:
            print(f"[WARN] {missing} Kandidaten ohne Vorhersage -> mit 0 aufgefuellt.")
        merged[value_col] = merged[value_col].fillna(0.0)
        return merged

    def score_binary(self, pred_df: pd.DataFrame, split: str = "test",
                     score_col: str = "score") -> dict:
        m = self._join(pred_df, split, score_col)
        out = {"split": split, "n_pos": int(m["label"].sum()), "n_total": len(m),
               "auc": auc_roc(m["label"].values, m[score_col].values),
               "ap":  average_precision(m["label"].values, m[score_col].values)}
        out.update(binary_at_threshold(m["label"].values, m[score_col].values))
        return out

    def score_count(self, pred_df: pd.DataFrame, split: str = "test",
                    count_col: str = "pred_count") -> dict:
        m = self._join(pred_df, split, count_col)
        out = {"split": split, "n_total": len(m)}
        out.update(count_metrics(m["count"].values, m[count_col].values))
        return out


# ---------------------------------------------------------------------------
# Demo / Selbsttest: Frequenz-Heuristik als triviale Referenz-Baseline
# ---------------------------------------------------------------------------
def _frequency_baseline(ev: SharedLinkEval, split: str) -> pd.DataFrame:
    """Einfachste denkbare Baseline: Score/Count eines Paares = mittlere
    Trip-Anzahl pro Bin im TRAININGSZEITRAUM. Dient nur zum Testen der Pipeline."""
    tg = ev.build_targets()
    train = tg[tg["split"] == "train"]
    n_train_bins = max(1, train["bin_idx"].nunique())
    rate = (train.groupby(["u", "i"])["count"].sum() / n_train_bins)
    cand = ev.build_candidates(split)
    pred = cand[["u", "i", "bin_idx"]].copy()
    key = list(zip(pred["u"], pred["i"]))
    vals = np.array([rate.get(k, 0.0) for k in key])
    pred["pred_count"] = vals
    pred["score"] = 1 - np.exp(-vals)   # Rate -> Wahrscheinlichkeit (mind. 1 Trip)
    return pred


if __name__ == "__main__":
    ev = SharedLinkEval()
    tg = ev.build_targets()
    print("=== Protokoll ===")
    print(f"Bins gesamt: {ev.n_bins} (a {ev.cfg.bin_minutes} min)")
    for s in ["train", "val", "test"]:
        sub = tg[tg["split"] == s]
        print(f"  {s:5s}: {len(sub):6d} positive Zellen | "
              f"{sub['bin_idx'].nunique():4d} Bins | {int(sub['count'].sum()):6d} Trips")

    print("\n=== Demo: Frequenz-Heuristik (Referenz) ===")
    for split in ["val", "test"]:
        pred = _frequency_baseline(ev, split)
        b = ev.score_binary(pred, split=split)
        c = ev.score_count(pred, split=split)
        print(f"[{split}] BINAER  AUC={b['auc']:.3f} AP={b['ap']:.3f} "
              f"F1={b['f1']:.3f} Acc={b['accuracy']:.3f} (n_pos={b['n_pos']}/{b['n_total']})")
        print(f"[{split}] COUNT   MSE={c['mse']:.3f} MAE={c['mae']:.3f} RMSE={c['rmse']:.3f}")
