# Prepared data – temporal graph (GraphMixer baseline)

This folder holds the **temporal graph** of the NYC/JC bike-sharing trips,
prepared as input for the GraphMixer baseline and for the shared evaluation.
Built from the raw Citi Bike trips (Jersey City / Hoboken, May + June 2024),
filtered to the **active 232-node subgraph** of the super-edge dataset.

## Provenance

- **Raw source**: Citi Bike System Data, files `JC-202405-citibike-tripdata.csv`
  and `JC-202406-citibike-tripdata.csv` (Jersey City / Hoboken sub-system).
- **Filter**: only trips whose start and end station belong to the 232 active
  stations of the super-edge dataset (matched via `short_name` = Citi Bike
  `station_id`, 232/232 = 100 % match).
- **Observation window**: 2024-05-16 to 2024-06-14 (≈ 29 days), identical to the
  super-edge dataset (Zenodo DOI 10.5281/zenodo.13846868).
- **Sanity check**: 104,681 filtered trips vs. 102,594 super-edge trips; all
  5,626 super-edge pairs present, median per-pair deviation = 0 (window boundary
  effects explain the ~2 % difference).

## Files

| File | Contents | Used by |
|---|---|---|
| `superedge_counts.csv` | **Ground-truth source**: aggregated super-edge `num_rides` per 30-min bin `u, i, bin_idx, count` (= Δnum_rides). | `shared_eval` (targets / count GT, all models), **LSTM** (input series) |
| `graphmixer_edges.csv` | Individual trips (temporal graph): `u, i, ts, ts_iso, rideable_type, member_casual` | building `ml_citibike.*` |
| `node_index.csv` | Canonical node mapping: `idx` (0…231) ↔ `station_id` ↔ `name` | all |
| `node_static.npy` | Static node features `(232, 3)`: capacity, lat, lon | **Hybrid** (GraphSAGE) |
| `node_avail.npy` | Availability time series `(232, T, 4)` per 30-min bin | **Hybrid** (GRU) |
| `edge_index.npy` / `edge_weight.npy` | Adjacency (directed) + weight = super-edge `num_rides` in training | **Hybrid** (GraphSAGE) |
| `ml_citibike.csv` | DyGLib edge list: `u, i, ts, label, idx` (**nodes 1-indexed** 1…232, homogeneous) | **GraphMixer** |
| `ml_citibike.npy` | Edge features `(n+1, 4)`: one-hot `classic, electric, member, casual`; row 0 = padding | **GraphMixer** |
| `ml_citibike_node.npy` | Node features `(233, 3)`: `capacity, lat, lon` (z-normalized); row 0 = padding | **GraphMixer** |

## Key conventions

- **Canonical node ID** = `idx` from `node_index.csv` (**0-indexed**, 0…231).
  Used by the evaluation module and by `graphmixer_edges.csv`.
- **DyGLib/GraphMixer files** (`ml_*`) are **1-indexed** (canonical ID + 1),
  row 0 is padding. Before scoring, shift GraphMixer predictions back by **−1**
  so they line up with the canonical ID.
- **`ts`**: seconds since the window start (2024-05-16), sorted chronologically.
- **Direction**: edges are directed (`u` = start, `i` = end).

## Reproduction

Produced by the scripts (against the external data path):
1. `build_graphmixer.py` → `graphmixer_edges.csv`, `node_index.csv` (+ sanity check)
2. `to_dyglib.py` → `ml_citibike.csv`, `ml_citibike.npy`, `ml_citibike_node.npy`
