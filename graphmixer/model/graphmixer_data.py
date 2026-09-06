# -*- coding: utf-8 -*-
"""
Data and sampling layer for GraphMixer.
=======================================

Loads the prepared files (ml_citibike.csv/.npy/_node.npy) and provides the
TEMPORAL NEIGHBORHOOD SAMPLER: for a node and a query time t, it returns that
node's last K edge events BEFORE t. This "recent past" is exactly what the link
encoder processes.

Important (causality): only events with ts < t_query are used, so no future
information can leak in during training/testing.
"""

from __future__ import annotations
import os
import bisect
import numpy as np
import pandas as pd


class GraphMixerData:
    """Holds edges/features and answers temporal neighborhood queries."""

    def __init__(self, prep_dir: str):
        # --- load files ---
        self.edges = pd.read_csv(os.path.join(prep_dir, "ml_citibike.csv"))
        self.edge_feat = np.load(os.path.join(prep_dir, "ml_citibike.npy"))        # (n+1, d_edge)
        self.node_feat = np.load(os.path.join(prep_dir, "ml_citibike_node.npy"))   # (N+1, d_node)
        self.d_edge = self.edge_feat.shape[1]
        self.d_node = self.node_feat.shape[1]
        self.num_nodes = self.node_feat.shape[0] - 1   # excluding padding row 0

        u = self.edges["u"].to_numpy()
        i = self.edges["i"].to_numpy()
        ts = self.edges["ts"].to_numpy(dtype=float)
        eidx = self.edges["idx"].to_numpy()            # 1..n, row in edge_feat

        # --- build a time-sorted event list per node ---
        # An event counts for BOTH involved nodes (directed trips, but a node's
        # "activity history" covers both inbound and outbound edges).
        self._ev_ts = {n: [] for n in range(1, self.num_nodes + 1)}
        self._ev_other = {n: [] for n in range(1, self.num_nodes + 1)}
        self._ev_efeat = {n: [] for n in range(1, self.num_nodes + 1)}

        order = np.argsort(ts, kind="mergesort")       # chronological
        for k in order:
            a, b, t, e = int(u[k]), int(i[k]), float(ts[k]), int(eidx[k])
            self._ev_ts[a].append(t); self._ev_other[a].append(b); self._ev_efeat[a].append(e)
            self._ev_ts[b].append(t); self._ev_other[b].append(a); self._ev_efeat[b].append(e)

        # convert to arrays (for fast slicing)
        for n in range(1, self.num_nodes + 1):
            self._ev_ts[n] = np.asarray(self._ev_ts[n], dtype=float)
            self._ev_other[n] = np.asarray(self._ev_other[n], dtype=int)
            self._ev_efeat[n] = np.asarray(self._ev_efeat[n], dtype=int)

    # ------------------------------------------------------------------
    def get_recent(self, node: int, t_query: float, K: int):
        """The node's last K events BEFORE t_query.

        Returns (each of length K, front-padded with zeros if < K):
          ef  : (K, d_edge)  edge features of the events
          dt  : (K,)         t_query - t_event   (time difference)
          msk : (K,)         1 = valid event, 0 = padding
          nbf : (K, d_node)  node features of the respective neighbors
        """
        tarr = self._ev_ts[node]
        pos = bisect.bisect_left(tarr, t_query)         # first event with ts >= t_query
        lo = max(0, pos - K)
        sel = slice(lo, pos)                            # the last <=K events < t_query

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
        """Vectorized collection for a list of (node, t_query)."""
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
        # mean neighbor node features (valid events only)
        denom = np.clip(msk.sum(axis=1, keepdims=True), 1.0, None)
        neigh_mean = (nbf * msk[:, :, None]).sum(axis=1) / denom
        return ef, dt, msk, own, neigh_mean

    # ------------------------------------------------------------------
    def split_masks(self, train_end_s: float, val_end_s: float):
        """Boolean edge masks for train/val/test by time."""
        ts = self.edges["ts"].to_numpy(dtype=float)
        return (ts < train_end_s,
                (ts >= train_end_s) & (ts < val_end_s),
                ts >= val_end_s)
