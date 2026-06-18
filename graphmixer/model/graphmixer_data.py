# -*- coding: utf-8 -*-
"""
Daten- und Sampling-Schicht für GraphMixer.
===========================================

Lädt die aufbereiteten Dateien (ml_citibike.csv/.npy/_node.npy) und stellt den
TEMPORALEN NACHBARSCHAFTS-SAMPLER bereit: Für einen Knoten und einen Abfrage-
zeitpunkt t liefert er dessen letzte K Kanten-Ereignisse VOR t. Genau diese
"jüngste Vergangenheit" verarbeitet der Link-Encoder.

Wichtig (Kausalität): Es werden ausschließlich Ereignisse mit ts < t_query
verwendet. So kann beim Training/Test keine Zukunftsinformation einfließen.
"""

from __future__ import annotations
import os
import bisect
import numpy as np
import pandas as pd


class GraphMixerData:
    """Hält Kanten/Features und beantwortet temporale Nachbarschaftsanfragen."""

    def __init__(self, prep_dir: str):
        # --- Dateien laden ---
        self.edges = pd.read_csv(os.path.join(prep_dir, "ml_citibike.csv"))
        self.edge_feat = np.load(os.path.join(prep_dir, "ml_citibike.npy"))        # (n+1, d_edge)
        self.node_feat = np.load(os.path.join(prep_dir, "ml_citibike_node.npy"))   # (N+1, d_node)
        self.d_edge = self.edge_feat.shape[1]
        self.d_node = self.node_feat.shape[1]
        self.num_nodes = self.node_feat.shape[0] - 1   # ohne Padding-Zeile 0

        u = self.edges["u"].to_numpy()
        i = self.edges["i"].to_numpy()
        ts = self.edges["ts"].to_numpy(dtype=float)
        eidx = self.edges["idx"].to_numpy()            # 1..n, Zeile in edge_feat

        # --- Pro Knoten eine zeitlich sortierte Ereignisliste aufbauen ---
        # Ein Ereignis zählt für BEIDE beteiligten Knoten (gerichtete Trips,
        # aber die "Aktivitätshistorie" eines Knotens umfasst Ein- und Ausgänge).
        self._ev_ts = {n: [] for n in range(1, self.num_nodes + 1)}
        self._ev_other = {n: [] for n in range(1, self.num_nodes + 1)}
        self._ev_efeat = {n: [] for n in range(1, self.num_nodes + 1)}

        order = np.argsort(ts, kind="mergesort")       # chronologisch
        for k in order:
            a, b, t, e = int(u[k]), int(i[k]), float(ts[k]), int(eidx[k])
            self._ev_ts[a].append(t); self._ev_other[a].append(b); self._ev_efeat[a].append(e)
            self._ev_ts[b].append(t); self._ev_other[b].append(a); self._ev_efeat[b].append(e)

        # in Arrays umwandeln (für schnelles Slicing)
        for n in range(1, self.num_nodes + 1):
            self._ev_ts[n] = np.asarray(self._ev_ts[n], dtype=float)
            self._ev_other[n] = np.asarray(self._ev_other[n], dtype=int)
            self._ev_efeat[n] = np.asarray(self._ev_efeat[n], dtype=int)

    # ------------------------------------------------------------------
    def get_recent(self, node: int, t_query: float, K: int):
        """Letzte K Ereignisse des Knotens VOR t_query.

        Rückgabe (jeweils Länge K, vorne mit Nullen aufgefüllt falls < K):
          ef  : (K, d_edge)  Kanten-Features der Ereignisse
          dt  : (K,)         t_query - t_event   (Zeitdifferenz)
          msk : (K,)         1 = gültiges Ereignis, 0 = Padding
          nbf : (K, d_node)  Knoten-Features der jeweiligen Nachbarn
        """
        tarr = self._ev_ts[node]
        pos = bisect.bisect_left(tarr, t_query)         # erstes Ereignis mit ts >= t_query
        lo = max(0, pos - K)
        sel = slice(lo, pos)                            # die letzten <=K Ereignisse < t_query

        sel_e = self._ev_efeat[node][sel]
        sel_o = self._ev_other[node][sel]
        sel_t = tarr[sel]
        m = len(sel_t)

        ef = np.zeros((K, self.d_edge), dtype=np.float32)
        dt = np.zeros((K,), dtype=np.float32)
        msk = np.zeros((K,), dtype=np.float32)
        nbf = np.zeros((K, self.d_node), dtype=np.float32)
        if m > 0:
            ef[K - m:] = self.edge_feat[sel_e]
            dt[K - m:] = (t_query - sel_t)
            msk[K - m:] = 1.0
            nbf[K - m:] = self.node_feat[sel_o]
        return ef, dt, msk, nbf

    # ------------------------------------------------------------------
    def get_batch(self, nodes, t_queries, K: int):
        """Vektorisierte Sammlung für eine Liste von (node, t_query)."""
        B = len(nodes)
        ef = np.zeros((B, K, self.d_edge), dtype=np.float32)
        dt = np.zeros((B, K), dtype=np.float32)
        msk = np.zeros((B, K), dtype=np.float32)
        nbf = np.zeros((B, K, self.d_node), dtype=np.float32)
        own = np.zeros((B, self.d_node), dtype=np.float32)
        for b, (nd, tq) in enumerate(zip(nodes, t_queries)):
            e, d, m, nf = self.get_recent(int(nd), float(tq), K)
            ef[b], dt[b], msk[b], nbf[b] = e, d, m, nf
            own[b] = self.node_feat[int(nd)]
        # mittlere Nachbar-Knoten-Features (nur gültige Ereignisse)
        denom = np.clip(msk.sum(axis=1, keepdims=True), 1.0, None)
        neigh_mean = (nbf * msk[:, :, None]).sum(axis=1) / denom
        return ef, dt, msk, own, neigh_mean

    # ------------------------------------------------------------------
    def split_masks(self, train_end_s: float, val_end_s: float):
        """Bool-Masken der Kanten für Train/Val/Test nach Zeit."""
        ts = self.edges["ts"].to_numpy(dtype=float)
        return (ts < train_end_s,
                (ts >= train_end_s) & (ts < val_end_s),
                ts >= val_end_s)
