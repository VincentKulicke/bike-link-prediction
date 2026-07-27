---
tags: [project, data-analysis, link-prediction]
status: active
created: 2026-05-12
project: Link Prediction on Hybrid Graph + Time Series Data
---

# Data analysis: NYC Bike Sharing Network

Full data exploration of the two JSON files from the Zenodo record [10.5281/zenodo.13846868](https://zenodo.org/records/13846868). This note documents the schemas and the most important statistical properties. The task and the method design live in the [[Link Prediction on Hybrid Graph + Time Series Data.md|project's main file]].

## Method

The analysis was run directly on the full files, stored locally under `C:\Users\user\Data\nyc-bike-sharing\` (outside the vault). `graph_edges.json` was loaded fully into memory; `graph_nodes.json` was streamed with `ijson` because of its size. All numbers are reproducible with the script at the end of this note.

## Overview

| Metric | Value |
|---|---|
| Number of nodes (stations) | **2,213** |
| Number of directed edges (super_edges) | **5,626** |
| Observation period | 2024-05-16 to 2024-06-14 (about 4 weeks) |
| Nodes with an identical observation window | 2,213 / 2,213 (100%) |
| Total node time-series points | **7,521,171** |
| Total trips (sum of last num_rides values) | **102,594** |

## 1. Node schema (`graph_nodes.json`)

Each element is a Citi Bike station.

### Static fields per node

| Field | Type | Description |
|---|---|---|
| `station_id` | UUID | Primary key, identical to `nodeid` |
| `nodeid` | UUID | Duplicate of `station_id`, used as the graph key |
| `name` | string | Human-readable station name (e.g. "Whitehall St & Bridge St") |
| `short_name` | string | Internal short code (e.g. "4962.02") |
| `region_id` | string \| null | **Not constant.** Distribution: `"71"` 1,740, `null` 393, `"70"` 53, `"311"` 27 |
| `capacity` | int | Number of docks. Observed range **1..123**, median 24, mean 31.2 |
| `lat`, `lon` | float | Geo-coordinates |
| `start`, `end` | ISO datetime | Observation window, **constant for all nodes** (`2024-05-16T00:00:00` to `2024-06-14T00:00:00`) |
| `labels` | list[string] | Constant `["station"]` for all nodes |

> **Correction to an earlier assumption**: `region_id` is not constant. There are four distinct values plus `null`. Treat it as a categorical feature in preprocessing and encode missing values explicitly.

### Dynamic fields — the `ts` block

Each node carries **four** time series:

| Series                 | Meaning                                                     |
| ---------------------- | ----------------------------------------------------------- |
| `num_bikes_available`  | Currently dockable, rentable bikes                          |
| `num_ebikes_available` | Subset of available bikes that are e-bikes                  |
| `num_bikes_disabled`   | Bikes physically at the station but marked unusable         |
| `num_docks_disabled`   | Docks marked as broken                                      |

Format: a list of `{Start: ISO_datetime, Value: int}` records.

### Sampling semantics: change-point compression

The publisher (Zenodo description) confirms it: the GBFS station-status feed is polled every 5 minutes, but a record is written **only when at least one of the four counters has changed since the last poll**. This is by design, not a data-quality issue.

Our own measurements confirm the 5-minute cadence:

- Minimum inter-event delta: **294 s** (≈ 5 min), matching the poll frequency.
- Median inter-event delta over all series and all nodes: **600 s (10 min)**.
- Per series:
  - `num_bikes_available`: median 9.9 min (n=3,917,759 events)
  - `num_ebikes_available`: median 10.0 min (n=3,043,035 events)
  - `num_bikes_disabled`: median 24.9 min (n=520,308 events) — rarer, because breakdowns are rare
  - `num_docks_disabled`: median 10.1 min (n=31,217 events) — very rare in absolute frequency

> **Consequence for the model**: to use the series as regular time series, resample onto a fixed grid (e.g. 5-minute bins via forward fill). The "current value" at time t is the most recent record with `Start ≤ t`.

### Length distribution of the node time series

| Statistic | Value |
|---|---|
| Total ts points over all nodes | 7,521,171 |
| Min length per series | 1 |
| Median length | 359 |
| Mean length | 849.7 |
| Max length | 5,407 |

The spread is large: active stations have several thousand events, dormant stations almost none. This affects the train/val split strategy (see below).

## 2. Edge schema (`graph_edges.json`)

Each element is a directed `super_edge` between two stations and aggregates **all trips** from `from` to `to` in the observation window.

### Static fields per edge

| Field | Type | Description |
|---|---|---|
| `from` | UUID | Source station (references `station_id`) |
| `to` | UUID | Target station (references `station_id`) |
| `label` | string | Constant `"super_edge"` for all edges |
| `start` | ISO datetime | Time of the first ride on this edge |
| `end` | ISO datetime | Global window end (`2024-06-14T00:00:00`) |

### Dynamic fields — the `ts` block

Each edge carries **six** time series:

| Series           | Meaning                                        | Monotone?             |
| ---------------- | ---------------------------------------------- | --------------------- |
| `num_rides`      | Cumulative total of all rides on the edge      | yes, non-decreasing   |
| `classic_rides`  | Cumulative, rides on a classic bike            | yes, non-decreasing   |
| `electric_rides` | Cumulative, rides on an e-bike                 | yes, non-decreasing   |
| `member_rides`   | Cumulative, rides from member accounts         | yes, non-decreasing   |
| `casual_rides`   | Cumulative, rides from casual accounts         | yes, non-decreasing   |
| `active_trips`   | Currently ongoing trips on this edge           | **no, oscillates**    |

> **Important**: the five `_rides` series are **cumulative counters**. Each event increments by exactly 1. The identity `num_rides = classic_rides + electric_rides = member_rides + casual_rides` can presumably be used for data validation.
>
> `active_trips` is an **inventory level**, not a counter — the value goes up by 1 when a trip starts and down by 1 when it ends. Observed value-range sample over 500 edges: `[0, 85]`.

### Counter semantics for the prediction target

Since `num_rides` is cumulative, an edge's per-window volume follows from the difference:

```
rides_in_window(u, v, t, Δ) = num_rides[u→v] @ (t+Δ)  −  num_rides[u→v] @ t
```

The binary target per the assignment is then: `Label = 1` if `rides_in_window > 0`, else `0`.

### Length distribution per edge (based on `num_rides`)

| Statistic | Value |
|---|---|
| Total trips (sum of last counter values) | 102,594 |
| Min trips per edge | 1 |
| Median trips per edge | 6 |
| Mean trips per edge | 18.2 |
| Max trips per edge | 468 |
| Edges with < 5 trips | 2,375 (**42.2 %**) |

Almost half of all edges are very sparsely populated. This is a clear sign of **sparsity and a long tail**: a few edges dominate the trip volume, the rest is noise or rare connections.

### Inter-event spacing on `num_rides`

| Sample | Median | Mean |
|---|---|---|
| All edges, all events (n=96,968) | **5.81 h** | 22.9 h |
| Only edges with ≥ 20 trips (n=77,807) | **3.50 h** | 11.3 h |

Very long-tailed: the means sit far above the medians, because inactive edges and overnight pauses pull the distribution upward. Active edges have a median of 3.5 h between two trips.

## 3. Implications for method and modelling

### Preprocessing

- **Resample node ts onto a fixed grid** (e.g. 5- or 15-min bins via forward fill).
- **Edge ts: use differences, not cumulative values** to get a window volume.
- **Station identity** via `station_id` (or equivalently `nodeid`).
- **`region_id` missing handling**: encode explicitly; don't misinterpret it as a second "71" category.

### Targets

- **Binary**: `Δnum_rides > 0` in the future window per (u, v).
- **Regression** would also be possible (`Δnum_rides` directly), but is not required by the assignment.

### Class imbalance and negative sampling

42.2 % of edges have < 5 trips over the whole period. On a window of, say, 30 minutes there are almost always 0 trips per `(u, v, t)` → class imbalance is extreme. The classic fix: **negative sampling** (e.g. 1 positive sample : k negative samples) plus AP / MRR instead of accuracy alone.

### Train/val/test split

- **Temporal**: first 3 weeks training, second-to-last week validation, last week test. A random split would be methodologically wrong (leakage through temporal correlation).
- **Observation**: stations with short series (length=1 or very small) may be excluded, since they are below the information-recovery threshold.

### Feature proposal

- **Static node features**: `capacity`, `lat`, `lon`, `region_id` (one-hot), optionally derived spatial features (distance to the centroid, cluster ID).
- **Dynamic node features**: a short window (e.g. last 30 minutes) of the four ts series, e.g. mean, std, trend, plus a time-of-day / weekday feature.
- **Edge features**: historical trip frequency (`num_rides` total over the known window), bike-type ratio (`electric_rides / num_rides`), rider-type ratio (`member_rides / num_rides`).

### Capacity vs. trip volume: geographic confounding

A full analysis on the 232 active stations shows a **negative** correlation between station capacity and trip volume:

| Measure | Value |
|---|---|
| Pearson r | −0.44 |
| Spearman r | −0.58 |

This counterintuitive finding is **not causal** but confounded by geographic location. The active subgraph contains two spatially separate populations:

- **Small stations (11–30 docks) in Hoboken / Jersey City**: highly active, median trips/dock ~90.
- **Large stations (61–123 docks) from Manhattan**: nearly inactive, median trips/dock ~0.

The top-5 stations by trips-per-dock (Newport Pkwy: 293.8; Newport PATH: 273.6; Hoboken Terminal: 216.9) confirm the pattern. At the same time, the largest stations (E 40 St & Park Ave: 123 docks, 11 trips; West St & Chambers St: 115 docks, 8 trips) have practically no activity despite their huge capacity.

**Consequence for feature engineering**: `capacity` alone is not a reliable activity indicator. `lat`/`lon` separate the two clusters cleanly and are the stronger predictor. In the GCN branch this doesn't matter directly — the adjacency matrix with historical trip frequencies as edge weights captures the cluster structure on its own.

## 4. Open questions for the next iteration

- Does the identity `num_rides == classic_rides + electric_rides == member_rides + casual_rides` hold for all edges? → Quick validation possible.
- How does `active_trips` distribute over time (time-of-day pattern, weekday pattern)? Could serve as an extra node feature.
- Is the 4-week window enough for a valid validation/test split, or is there too little data in the test period?
- What does the spatial cluster pattern look like (Manhattan, Brooklyn, Queens, …)?

## 5. Reproducibility

Prerequisites:

```bash
pip install ijson
```

Dataset locally:

```bash
mkdir -p ~/Data/nyc-bike-sharing
cd ~/Data/nyc-bike-sharing
curl -L -o graph_edges.json "https://zenodo.org/records/13846868/files/graph_edges.json?download=1"
curl -L -o graph_nodes.json "https://zenodo.org/records/13846868/files/graph_nodes.json?download=1"
```

Script skeleton:

```python
import json, ijson, statistics
from collections import Counter
from datetime import datetime

DATA = r"C:\Users\user\Data\nyc-bike-sharing"

# Edges fit in memory
with open(f"{DATA}/graph_edges.json", "rb") as f:
    edges = json.load(f)

# Nodes via streaming
with open(f"{DATA}/graph_nodes.json", "rb") as f:
    for node in ijson.items(f, "item"):
        # process node
        pass
```

The full analysis scripts (structure checks, length distributions, inter-event deltas) will be added to the project code repo once it exists.
