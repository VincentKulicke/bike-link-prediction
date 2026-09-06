# -*- coding: utf-8 -*-
"""
Preprocessing of the HYBRID node inputs for the hybrid model (iteration 2).
===========================================================================

Extracts small, versionable input files from the Zenodo dataset
(graph_nodes.json / graph_edges.json), aligned to the 30-min grid of
shared_eval:

  node_static.npy  (N, 3)        capacity, lat, lon
  node_avail.npy   (N, T, 4)     availability per 30-min bin (forward-fill):
                                 num_bikes_available, num_ebikes_available,
                                 num_bikes_disabled, num_docks_disabled
  edge_index.npy   (2, E)        directed edges (canonical node IDs)
  edge_weight.npy  (E,)          trip frequency in the training period (adjacency weight)

Raw data lives outside the repo (see README). Adjust the DATA path if needed.
"""
import os
import numpy as np
import pandas as pd
import ijson

# --- paths -------------------------------------------------------------------
DATA = r"C:/Users/user/Data/nyc-bike-sharing"             # raw data (external, adjust)
PREP = os.path.join(os.path.dirname(__file__), "..", "prepared Data")  # target (in repo)

BIN_SECONDS = 1800
TRAIN_DAYS = 21
SERIES = ["num_bikes_available", "num_ebikes_available",
          "num_bikes_disabled", "num_docks_disabled"]


def main():
    # canonical mapping short_name (= station_id) -> idx 0..N-1
    nodes_idx = pd.read_csv(os.path.join(PREP, "node_index.csv"))
    short2idx = {str(s): int(i) for s, i in zip(nodes_idx["station_id"], nodes_idx["idx"])}
    N = len(short2idx)

    # derive the number of bins from the prepared trips (same grid as shared_eval)
    trips = pd.read_csv(os.path.join(PREP, "graphmixer_edges.csv"), usecols=["u", "i", "ts"])
    n_bins = int(trips["ts"].max() // BIN_SECONDS) + 1

    # window start from the first node
    with open(f"{DATA}/graph_nodes.json", "rb") as f:
        for n in ijson.items(f, "item"):
            win_start_dt = pd.to_datetime(n.get("start")); break
    bin_starts_s = np.arange(n_bins) * BIN_SECONDS

    node_static = np.zeros((N, 3), dtype=np.float32)
    node_avail = np.zeros((N, n_bins, 4), dtype=np.float32)

    def sample(ts_list):
        if not ts_list:
            return np.zeros(n_bins, dtype=np.float32)
        starts = pd.to_datetime([r["Start"] for r in ts_list], format="ISO8601")
        starts_s = (starts - win_start_dt).total_seconds().to_numpy()
        vals = np.array([r["Value"] for r in ts_list], dtype=np.float32)
        order = np.argsort(starts_s); starts_s, vals = starts_s[order], vals[order]
        pos = np.searchsorted(starts_s, bin_starts_s, side="right") - 1
        return np.where(pos >= 0, vals[np.clip(pos, 0, len(vals) - 1)], 0.0).astype(np.float32)

    found = 0
    with open(f"{DATA}/graph_nodes.json", "rb") as f:
        for n in ijson.items(f, "item"):
            sn = str(n.get("short_name", ""))
            if sn not in short2idx:
                continue
            idx = short2idx[sn]; found += 1
            node_static[idx] = [float(n.get("capacity", 0) or 0),
                                float(n.get("lat", 0) or 0), float(n.get("lon", 0) or 0)]
            ts = n.get("ts", {})
            for c, key in enumerate(SERIES):
                node_avail[idx, :, c] = sample(ts.get(key, []))
    print(f"Nodes found/expected: {found}/{N} | bins: {n_bins}")

    # adjacency from the training period (canonical) – weight = super-edge num_rides
    se = pd.read_csv(os.path.join(PREP, "superedge_counts.csv"))
    train_end_bin = TRAIN_DAYS * ((24 * 60) // (BIN_SECONDS // 60))
    tr = se[se["bin_idx"] < train_end_bin]
    agg = tr.groupby(["u", "i"])["count"].sum().reset_index(name="w")
    edge_index = np.vstack([agg["u"].to_numpy(), agg["i"].to_numpy()]).astype(np.int64)
    edge_weight = agg["w"].to_numpy().astype(np.float32)

    np.save(os.path.join(PREP, "node_static.npy"), node_static)
    np.save(os.path.join(PREP, "node_avail.npy"), node_avail)
    np.save(os.path.join(PREP, "edge_index.npy"), edge_index)
    np.save(os.path.join(PREP, "edge_weight.npy"), edge_weight)
    print(f"written to {PREP}: node_static {node_static.shape}, "
          f"node_avail {node_avail.shape}, edge_index {edge_index.shape}")


if __name__ == "__main__":
    main()
