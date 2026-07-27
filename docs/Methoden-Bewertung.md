---
tags: [project, methodology, link-prediction]
status: active
created: 2026-05-12
project: Link Prediction on Hybrid Graph + Time Series Data
---

# Methods assessment: GCN vs. GraphSAGE and GRU vs. 1D-CNN

The reasoning behind the choice of the two branches in the proposed architecture. It builds on the [[Datenanalyse.md|data analysis]] (after filtering: 232 active stations, 5,626 edges, mean degree 48.5, four node time series per station, six edge time series, a 4-week observation window). The task is described in the [[Link Prediction on Hybrid Graph + Time Series Data.md|main file]].

> **Decision (as of 2026-06-03):** the team chose **GraphSAGE + GRU** as the primary architecture, in line with the [[Konzeptdokument.md|concept document]]. GCN and 1D-CNN are planned as documented alternatives and ablation variants. This document records the full trade-off so the choice is well argued in the concept and the presentation.

## 1. Graph branch: GCN vs. GraphSAGE

### Graph facts

- 232 active nodes after filtering out isolated stations (small by GNN standards).
- 5,626 directed edges, mean total degree ≈ 48.5 — a **small but dense** subgraph.
- Edge weights from historical trip frequency are available and explicitly required.
- The set of stations stays practically constant over the observation window.

### GCN — strengths in this setup

- **Edge weights supported natively** via the weighted adjacency matrix. The signal required by the assignment (trip frequency as an edge weight) can be used without a workaround.
- **Full-batch trainable**: with 2,213 nodes the whole graph fits in any GPU memory. No sampling overhead, deterministic training.
- **Few hyperparameters**: quick to spec, quick to debug.
- **Smooths strongly**, which helps under strong spatial homophily (nearby stations behave similarly).

### GCN — weaknesses

- **Transductive**: new stations are hard. Irrelevant here, since the station set is closed.
- **Over-smoothing** in deeper architectures. Barely noticeable at two hops.
- **Treats all neighbors equally** modulo the edge weight. No learned weighting of neighbors.

### GraphSAGE — strengths in this setup

- **Inductive capability**. Works with new nodes. No added value here.
- **Several aggregators** (mean, max-pool, LSTM): modelling flexibility if the exact aggregation pattern matters.
- **Mini-batching with neighbor sampling**: scales to large graphs. Unnecessary at this size.
- **Self-concat**: keeps a node's own feature explicitly separate from the neighbor aggregate, slightly reducing over-smoothing.

### GraphSAGE — weaknesses in this setup

- **Edge weights not native**. Requires manual feature engineering to bring in the trip frequency.
- **More hyperparameters** (per-layer sample sizes, aggregator choice).
- **Stochastic training** through sampling, which makes reproducible results harder.

### Decision, graph branch

**GraphSAGE as the chosen primary architecture, GCN (and optionally GAT) as the ablation.**

Reasoning:
1. **Close to the literature**: GraphSAGE (Will et al., 2017) is the canonical inductive GNN for link prediction and the standard method for this task in the relevant literature. That supports the argument in the concept and the presentation.
2. **Fits the dense subgraph**: the active graph is small but dense (mean degree 48.5). GraphSAGE's neighbor sampling keeps aggregation manageable even at high-degree hubs, and the self-concat structure keeps a station's own feature separate from the neighbor aggregate — which reduces over-smoothing precisely at high node density.
3. **Aggregator flexibility**: the choice between mean, max-pool and LSTM aggregation lets us target a specific aggregation pattern (e.g. "strong inflow" vs. "balanced").
4. **Inductivity as a robustness bonus**: even though the station set is closed, the inductive formulation makes the model insensitive to small changes in the node set.

Handling the edge weights: unlike GCN, GraphSAGE does not use edge weights natively. The historical trip frequency is therefore brought in via a **weighted mean aggregation** or as an extra edge feature — slightly more implementation effort than GCN, but standard.

**GCN** stays the obvious ablation: native edge weights, full-batch training (trivial at 232 nodes), fewer hyperparameters, deterministic. It is the simpler reference against which GraphSAGE's added value is measured. Optionally, as a third variant: **GAT / GATv2**, which learns edge importance from the features — an interesting ablation for mobility data with time-of-day modulation.

## 2. Node time-series branch: GRU vs. 1D-CNN

### Node time-series facts

- Four series per station: `num_bikes_available`, `num_ebikes_available`, `num_bikes_disabled`, `num_docks_disabled`.
- Median sampling after resampling to 5-minute bins: 5–10 minutes depending on the series.
- Realistic input window: 30 minutes to 6 hours, i.e. 6 to 72 time steps.
- Prediction horizon per the assignment: one future time window, typically 15–60 minutes.

### GRU — strengths in this setup

- **Variable sequence lengths** possible.
- **Implicit state tracking**: a station that just ran empty keeps the "empty phase" in its hidden state.
- **Rich literature** specifically for bike-sharing forecasting.

### GRU — weaknesses in this setup

- **Sequential**. Trains slowly; for short sequences the compute overhead is disproportionate.
- **Vanishing-gradient risk** from ~50 steps on.
- **More tuning** (hidden size, number of layers, dropout placement).

### 1D-CNN — strengths in this setup

- **Parallel and fast**. For short windows (6–72 steps) a 1D-CNN trains orders of magnitude faster than a GRU of the same size.
- **Local patterns are exactly what matters here**: emptying spikes, refill spikes, short-term time-of-day waves. Kernel sizes of 3–5 are enough.
- **Dilated convolutions** give a larger receptive field on demand without a depth explosion.
- **Simple tuning**: kernel size, number of filters, pooling.
- **Robust to padding and masking**, important for stations with shorter series.

### 1D-CNN — weaknesses in this setup

- **Fixed window size**: the receptive field is an architectural commitment.
- **Less natural state tracking** for very long dependencies (several days). Irrelevant given the short prediction horizon.

### Decision, node time-series branch

**GRU as the chosen primary architecture, 1D-CNN as the ablation.**

Reasoning:
1. **Close to the literature**: the GRU is the standard encoder in bike-sharing forecasting (Chen et al., 2021; Cini et al., 2025). It also fits naturally into the GCRNN/DCGRU framework, should graph and time-series processing be coupled more tightly in a later iteration.
2. **State tracking**: the GRU's implicit hidden state captures operating states like "station just ran empty" or "refill in progress" naturally — exactly the dynamics relevant to trip prediction.
3. **Manageable compute**: the GRU's main drawback (sequential, slower training) weighs less here, because the active graph has only 232 stations after filtering. The total volume stays manageable.

**1D-CNN** stays the obvious ablation: much faster, parallel training, good for local patterns (emptying spikes, refill waves) and easier to interpret (conv-filter inspection). It is the more efficient reference against which the GRU's state-tracking value can be measured.

If the prediction horizon is extended to several hours in a later iteration, it is also worth looking at a **TCN** (temporal convolutional network with dilated kernels) or a **small Transformer encoder**.

## 3. Should I try both?

Yes, but **sequentially and hypothesis-driven**, not in parallel and open-ended.

### Recommended order

1. **Iteration 1**: GraphSAGE + GRU + fusion MLP. The chosen primary path, close to the literature, end-to-end implementation.
2. **Iteration 2 (architecture ablation)**:
   - Swap the graph branch: GraphSAGE → GCN (or GAT).
   - Swap the time-series branch: GRU → 1D-CNN.
   - Do each swap independently, not combined in parallel.
3. **Iteration 2 (component ablation)**:
   - Graph branch only.
   - Time-series branch only.
   - Both without the fusion MLP (simple concatenation into a linear layer).
   - Shows whether fusion adds real value.
4. **Baselines**: **TGN** (Temporal Graph Networks) as a temporal-graph method and **VSTD** (variational-autoencoder-based spatio-temporal disentanglement) as a spatio-temporal method; plus a frequency heuristic and a logistic regression over static features as simple reference points.
5. **Final results table** for the concept and the presentation.

### Anti-recommendation

Don't grid-search all 4 combinations (GraphSAGE/GCN × GRU/1D-CNN). The treatments overlap, compute is burned, and narrative clarity is lost. Instead, start from the chosen path (GraphSAGE + GRU) and swap one variable at a time in a controlled way.

## 4. Argument snippet for the concept document

> For the graph branch we chose **GraphSAGE**, because it is the established inductive aggregation framework in the link-prediction literature and, via its self-concat structure and neighbor sampling, fits the small but dense active station graph well (232 nodes, mean degree 48.5). Historical trip frequency is included as a weighted edge signal in the aggregation. GCN serves as the simpler reference with native edge weights and is kept as an ablation variant.
>
> For the node time-series branch we chose **GRU**, because recurrent encoders are established in bike-sharing forecasting and the hidden state implicitly tracks station operating states (emptying, refill). The GRU also fits naturally into the GCRNN/DCGRU framework, should graph and time-series processing be coupled more tightly later. The 1D-CNN is the faster, local-pattern-focused alternative and is kept as an ablation variant.

## 5. Open questions

- If the prediction window is enlarged in a later iteration (e.g. 24 h), the time-series branch should be re-evaluated (TCN, Transformer encoder).
- Is a **directed graph variant** worthwhile? Trip flow is directed. GraphSAGE can run directed by sampling and aggregating in- and out-neighbors separately; alternatively two separate adjacencies (forward, backward) with parallel branches, or an explicitly directed GNN variant (Directed GCN, DGCN).
- If GAT is chosen as the alternative: is multi-head attention worth it, or is single-head enough?
