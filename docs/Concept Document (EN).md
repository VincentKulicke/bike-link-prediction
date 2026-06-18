---
tags: [projekt, konzept, link-prediction, english]
status: aktiv
erstellt: 2026-05-29
projekt: Link Prediction on Hybrid Graph + Time Series Data
---

# Hybrid GNN-Based Link Prediction for Bike-Sharing Networks

**Authors**

| Name | E-Mail |
|---|---|
| Moritz Rau | wx60iroz@studserv.uni-leipzig.de |
| Vincent Kulicke | ok09ined@studserv.uni-leipzig.de |
| Henning Hesselbarth | hh95copi@studserv.uni-leipzig.de |

**Contents**

- Introduction
- Theoretical Foundations
  - Graph Neural Networks
  - Evaluation Metrics
- Dataset
  - Node Structure
  - Edge Structure
  - Dataset Statistics and Data Preparation Consequences
- Proposed Method
  - Model Architecture
  - Evaluation
- References

---

## Introduction

Bike sharing systems generate both graph-structured and temporal data. Stations can be represented as nodes connected by trips, while additional time series such as bike availability describe the dynamic state of the network. Predicting future trips between stations is important for applications such as demand forecasting and bike rebalancing.

Existing temporal link prediction methods mainly focus on event streams and often ignore continuous node-level measurements. In contrast, many spatiotemporal approaches process time series data but make limited use of graph structure and node attributes. Therefore, current methods are not fully suited for hybrid graph and time series datasets.

This project proposes a hybrid link prediction framework that combines Graph Neural Networks (GraphSAGE) with temporal encoders such as GRUs. The model jointly learns structural and temporal representations of bike stations to predict whether trips between station pairs will occur in a future time window. The approach will be evaluated on the NYC Bike Sharing Network dataset and compared against existing baseline methods using standard link prediction metrics such as AUC, AP, and MRR.

## Theoretical Foundations

### Graph Neural Networks

A graph is formally defined as G = (V, E, F, E), where V is a set of N nodes, E is a set of edges, and F and E represent node and edge attributes, respectively (Guo et al., 2022). Graph Neural Networks (GNNs) are designed to process data contained in these graph-based structures, which are more flexible than traditional grids or sequences (Chen et al., 2021). The core mechanism of most GNNs follows the message-passing (MP) framework, where node representations are updated by aggregating information from their local neighborhood. In spatiotemporal contexts, GNNs model dependencies as pairwise relationships among time series, where each time series is associated with a node and functional relationships are represented as edges (Cini et al., 2025).

GraphSAGE is a general inductive framework that leverages node feature information to generate embeddings for previously unseen data. Unlike transductive methods that train individual embeddings for each node, GraphSAGE learns a set of aggregator functions that sample and aggregate features from a node's local neighborhood. To maintain computational efficiency, GraphSAGE uniformly samples a fixed-size set of neighbors rather than using the full neighborhood (Will et al., 2017).

The Gated Recurrent Unit (GRU) is a variant of the Recurrent Neural Network (RNN) designed to capture temporal information and handle long-term dependencies while mitigating vanishing gradient issues. A GRU cell utilizes an update gate to determine how much of the past information to retain and a reset gate to decide how much to forget (Kontopoulos et al., 2023). In graph-based forecasting, GRUs are often integrated with graph convolutions to form Graph Convolutional Recurrent Neural Networks (GCRNNs) or Diffusion Convolutional GRUs (DCGRU). In these architectures, standard matrix multiplications within the GRU gates are replaced by graph convolutional operators, allowing the model to capture spatial and temporal patterns simultaneously (Cini et al., 2025; Guo et al., 2022).

### Evaluation Metrics

The following table provides a comprehensive overview of the three evaluation metrics Area Under the Curve (AUC), Average Precision (AP), and Mean Reciprocal Rank (MRR) commonly used to evaluate machine learning models in classification and ranking tasks. These measures go beyond standard metrics like accuracy, which can often be misleading when dealing with imbalanced datasets or when the specific order of results is critical (Beddar-Wiesing et al., 2025).

| Metric | Definition | Formula |
|---|---|---|
| Area Under the Curve (AUC) | A graphical metric that evaluates a model's performance across all decision thresholds by calculating the area under a performance curve. | Area under the curve (e.g., the integral of the ROC curve plotting TPR against FPR) |
| Average Precision (AP) | A metric for ranking and recommendation quality that calculates the weighted mean of precisions achieved at each threshold, where the weight is the increase in recall from the previous threshold. | AP = Σ_n (R_n − R_n−1) · P_n  (where P_n and R_n are precision and recall at the n-th threshold) |
| Mean Reciprocal Rank (MRR) | A measure designed for tasks where only the first relevant result is of primary interest. It computes the average of the reciprocal ranks of the first correct answer across a sample of queries. | MRR = (1 / \|D\|) · Σ_x∈D (1 / k_x)  (\|D\|: number of queries, k_x: rank of the first relevant element for query x) |

## Dataset

The dataset is the NYC Citi Bike Sharing Network (Constantin Urbainsky & Lyft Bikes & Scooters, 2024), covering four weeks from 16 May to 14 June 2024, provided as graph_nodes.json (530.6 MB, 2,213 stations) and graph_edges.json (26.5 MB, 5,626 directed edges).

### Node Structure

Each station node combines static attributes (station_id, capacity, lat/lon, region_id) with four continuous availability time series under ts (num_bikes_available, num_ebikes_available, num_bikes_disabled, num_docks_disabled). The series use change-point compression — a new record is written only when a counter changes — yielding irregular inter-event intervals (median ≈ 10 min, minimum 294 s, 7.5 M events total).

### Edge Structure

Each of the 5,626 directed edges (super-edge) carries six time series attributes in addition to the static attributes "from" (starting station) and "to" (end station):

| Series | Meaning | Monotone? |
|---|---|---|
| num_rides | Cumulative total trip count | Yes |
| classic_rides / electric_rides | Cumulative by bike type | Yes |
| member_rides / casual_rides | Cumulative by rider type | Yes |
| active_trips | Trips currently in progress | No — oscillates |

Trip volume for any window [t, t+Δ] and the binary prediction target follow from differencing the cumulative counter:

> rides(u, v, t, Δ) = num_rides(t+Δ) − num_rides(t)
>
> y(u, v, t) = 1 if rides(u, v, t, Δ) > 0, else 0

### Dataset Statistics and Data Preparation Consequences

| Metric | Value | Preprocessing / Modelling Consequence |
|---|---|---|
| Total stations | 2,213 | — |
| Active stations (≥ 1 edge) | 232 | Filter 1,981 isolated stations before all steps |
| Directed edges | 5,626 | — |
| Total trips | 102,594 | — |
| Edges with < 5 trips | 2,375 (42.2%) | Severe class imbalance → optional negative sampling (1:5 train, 1:99 eval) |

| Metric | Before Filtering | After Filtering |
|---|---|---|
| Stations | 2,213 | 232 |
| Edges | 5,626 | 5,626 |
| Mean total degree | 5.1 | 48.5 |
| Usable negative space | approx. 460,000 pairs | approx. 48,000 pairs |

**1. Filter isolated stations.** All 1,981 stations without edges are removed. After filtering: 232 active stations remain, mean degree rises from 5.1 to 48.5, and the usable negative pair space contracts from ≈ 460,000 to ≈ 48,000 pairs.

**2. Resampling node time series.** The change-point-compressed availability time series are resampled onto a uniform 5-minute grid. The resampling strategy is forward fill (last observation carried forward): the value of a series at time t is taken as the most recent event with a timestamp less than or equal to t.

**3. Derive prediction target.** For each active edge (u, v) and each 5-minute timestamp, label = 1 if num_rides increases within [t, t+30 min], else 0. The 30-minute horizon (6 bins, configurable) yields ≈ 1,008 snapshots per edge over the three-week training period, roughly 96,000 positive samples before negative sampling.

**4. Temporal split.** The dataset is split strictly by time to prevent leakage.

| Split | Period | Duration |
|---|---|---|
| Training | 16 May – 5 June 2024 | 21 days |
| Validation | 6 – 9 June 2024 | 4 days |
| Test | 10 – 14 June 2024 | 5 days |

The adjacency matrix and all edge weights are computed exclusively from the training period and remain frozen during validation and testing.

**5. Negative sampling (optional).** Each positive sample is paired with 5 randomly drawn negative pairs (inactive station pairs in the same time bin). For evaluation, 99 negatives per positive construct the 1-vs-99 ranking problem.

## Proposed Method

### Model Architecture

The proposed model is a dual-branch architecture that jointly learns structural and temporal representations of bike stations to predict whether a trip between two stations will occur within the next 30 minutes.

The **Graph Branch** applies GraphSAGE over the directed station graph, using static station attributes (capacity, coordinates, region) as node features and historical trip frequencies as edge weights. Through two message-passing layers, each station aggregates information from its direct neighbors, producing a structural embedding that captures the station's role and connectivity within the network.

The **Time-Series Branch** encodes the recent bike availability dynamics of each station using a GRU. Applied independently to both the source and destination station with shared weights, it takes the last 30 minutes of availability measurements (num_bikes_available, num_ebikes_available, num_bikes_disabled, num_docks_disabled) as input and produces a compact representation of the current operational state.

For a candidate station pair, the structural and temporal embeddings of both stations are concatenated with pair-level features — geographic distance, historical trip frequency of the pair, and cyclical time encodings (hour of day, day of week) — and passed through a three-layer MLP that outputs the link probability.

Two baselines from the temporal graph and spatiotemporal literature are included for comparison: **TGN** (Temporal Graph Networks), which models link dynamics through a node memory updated by observed trip events, and **VSTD** (Variational Autoencoder-based Spatio-Temporal Disentanglement), which learns disentangled spatial and temporal node representations via a variational approach. A frequency heuristic and logistic regression over static features serve as simple reference points. To isolate the contribution of individual components, three model variants are trained: the full model, a variant without the time-series branch, and a variant without the historical activity rate.

### Evaluation

All models are evaluated under a unified temporal split — Training (21 days), Validation (4 days), and Test (5 days) — strictly partitioned by time to prevent information leakage. Normalisation parameters are computed exclusively on the training split and applied without recomputation to validation and test sets.

Two query-set protocols are used. The **sampled** protocol provides a balanced set of positive and negative station pairs and is used primarily during training. The **rank_all** protocol serves as the headline evaluation: for each query station at a given time step, 99 randomly drawn negative station pairs are ranked alongside the positive candidates, constructing a standardised 1-vs-99 ranking problem for consistent and reproducible evaluation.

Performance is reported using the three metrics introduced in Section 2:

- AUC-ROC
- Average Precision (AP)
- Mean Reciprocal Rank (MRR)

The primary comparison metric is AP on the rank_all test set, as it is most robust to class imbalance and directly captures ranking quality across all query groups.

## References

Beddar-Wiesing, S., Moallemy-Oureh, A., Kempkes, M., & Thomas, J. M. (2025). *Absolute Evaluation Measures for Machine Learning: A Survey* (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2507.03392

Chen, Z., Wu, H., O'Connor, N. E., & Liu, M. (2021). A Comparative Study of Using Spatial-Temporal Graph Convolutional Networks for Predicting Availability in Bike Sharing Schemes. *2021 IEEE International Intelligent Transportation Systems Conference (ITSC)*, 1299–1305. https://doi.org/10.1109/ITSC48978.2021.9564831

Cini, A., Marisca, I., Zambon, D., & Alippi, C. (2025). Graph Deep Learning for Time Series Forecasting. *ACM Computing Surveys*, 57(12), 1–34. https://doi.org/10.1145/3742784

Constantin Urbainsky & Lyft Bikes & Scooters. (2024). *NYC Bike Sharing Network: Time-Series Enhanced Nodes and Edges Dataset* [Dataset]. Zenodo. https://doi.org/10.5281/ZENODO.13846868

Guo, X., Wang, S., & Zhao, L. (2022). Graph Neural Networks: Graph Transformation. In L. Wu, P. Cui, J. Pei, & L. Zhao (Eds.), *Graph Neural Networks: Foundations, Frontiers, and Applications* (pp. 251–275). Springer Nature Singapore. https://doi.org/10.1007/978-981-16-6054-2_12

Kontopoulos, I., Makris, A., Tserpes, K., & Varvarigou, T. (2023). *An evaluation of time series forecasting models on water consumption data: A case study of Greece* (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2303.17617

Will, H., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems*, 30.
