---
tags: [project, explainer, didactic]
status: active
created: 2026-05-17
project: Link Prediction on Hybrid Graph + Time Series Data
---

# Iteration 1: notebook explained for non-specialists

A companion explanation for [[iteration1_gcn_cnn_fusion.ipynb|Iteration 1: GCN + 1D-CNN + fusion MLP]]

## 1. What is this about?

New York City has the **Citi Bike system**: nearly 2,200 stations where you pick up and drop off rental bikes. Every day, thousands of rides happen between different station pairs.

The question in this project is, roughly:

> If I'm standing at station A and looking at station B: how likely is it that within the next time unit (30 minutes) at least one person rides from A to B?

This prediction is useful for:

- **Rebalancing**: if the system knows where bikes will soon be needed, it can redistribute them preemptively.
- **Demand planning**: the city and the operator can plan capacity better.
- **Anomaly detection**: notable deviations from the usual pattern are often signs of events, weather changes, or technical problems.

So we build an algorithm that learns: *"From the past behavior of the stations and how they're connected, I can estimate where activity will happen next."*

## 2. What data do we have?

Two files:

### `graph_nodes.json` (the stations)

For each station there is:

- **Static information** (does not change): name, geo-coordinates, number of docks, region.
- **Time series**: every 5 minutes it is measured how many bikes are currently available, how many of those are e-bikes, how many bikes are broken, how many docks are broken. Four series in total per station.

### `graph_edges.json` (the rides between station pairs)

For each station pair (A → B) between which at least one ride happened, there is a so-called **aggregate edge** with:

- **Timestamp of the first ride** between that pair.
- **Six counter time series**: total rides, classic bikes, e-bikes, member rides, casual rides, currently ongoing rides.

### Neural network

A particular kind of machine-learning model, loosely inspired by biological neurons. At its core it's many small compute functions chained together. Each function has "weights" — numbers that are tuned during training so the model produces the right output from the input.

### Link prediction

The technical term for our task: predicting whether an edge (a "link") between two nodes forms at a given time.

## 4. What do we actually build?

Our model has three components:

```
                ┌─────────────────────────-┐
                │   Graph branch (GCN)     │
station data ───┤                          ├─-┐
                │  produces one "embedding"│  │
                │  vector per station      │  │
                └────────────────────────-─┘  │
                                              ├── fusion MLP ── probability
                ┌─────────────────────────-┐  │
                │   Time-series branch     │  │
availability ───┤   (1D-CNN)               ├──┘
ts per station  │                          │
                │  produces one "embedding"│
                │  vector per station      │
                └────────────────────────-─┘
```

1. **Graph branch (GCN)**: for each station, it looks at its neighbors in the station network and summarizes: *"who are you, based on your position in the network and the stations around you?"* The answer is a list of numbers — the **embedding vector**.

2. **Time-series branch (1D-CNN)**: for each station, it looks at what happened to availability over the last 60 minutes and distills a different profile: *"how active is the station right now? Which pattern is it showing?"* This also produces an embedding vector.

3. **Fusion MLP**: receives four embedding vectors — from source station A and target station B, each once from the graph branch and once from the time-series branch. It merges them and says: *"with what probability will someone ride from A to B in the next 30 minutes?"*

## 5. The notebook, cell by cell

### Cell 0: setup note

A commented-out line that, as the very first step, installs the required software libraries:

```python
# %pip install --quiet torch pandas numpy scikit-learn ijson tqdm matplotlib
```

The `%` is a Jupyter feature: it lets you run commands **inside** a notebook that normally run on the command line. `pip install` is the standard Python command to install libraries from the internet.

- **torch** = PyTorch, the framework for neural networks.
- **pandas** = library for working with tabular data.
- **numpy** = library for numeric arrays and math operations.
- **scikit-learn** = classic machine-learning library, used here for evaluation metrics.
- **ijson** = library to read large JSON files piece by piece without loading everything into memory at once.
- **tqdm** = shows nice progress bars.
- **matplotlib** = for drawing charts.

### Cell 1 (`In[1]`): imports

```python
import json
import math
import time
import random
from dataclasses import dataclass
from pathlib import Path

import ijson
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
...
```

```python
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
```

**Seed** = a starting value for the random-number generators. We fix it to 42 so the experiment is **reproducible**. Without it, every training run would use different random initial values and give slightly different results.

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Checks whether a **GPU** (graphics card) is available. GPUs can train neural networks orders of magnitude faster than CPUs. If there's no GPU, the code falls back to the **CPU**.

### Cell 2 (`In[2]`): configuration

```python
@dataclass
class Config:
    bin_minutes: int = 5
    window_bins: int = 12
    horizon_bins: int = 6
    sample_stride_bins: int = 6
    ...
    static_hidden: int = 32
    static_out: int = 32
    cnn_channels: tuple = (16, 32)
    ...
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3
```

Setting the **hyperparameters**:

- `bin_minutes: 5` = we split time into 5-minute chunks.
- `window_bins: 12` = the input window is 12 chunks long, i.e. 60 minutes.
- `horizon_bins: 6` = we predict 6 chunks ahead, i.e. the next 30 minutes.
- `epochs: 10` = we let the model pass over the full training data 10 times.
- `batch_size: 256` = per training step we process 256 examples at once.
- `lr: 1e-3` = the **learning rate**. It controls how strongly the model adjusts its internal parameters per training step. `1e-3` means 0.001.

### Cell 3 (`In[3]`): raw data — load edges

```python
with open(CFG.data_dir / "graph_edges.json", "rb") as f:
    edges_raw = json.load(f)
```

### Cell 4 (`In[4]`): raw data — load nodes (streaming)

```python
with open(CFG.data_dir / "graph_nodes.json", "rb") as f:
    for node in ijson.items(f, "item"):
        ...
```

The nodes file is 530 MB. Reading it fully with `json.load` like the edges file would keep it in memory as a giant nested data structure — that can get tight.
--> We use `ijson` as a **streaming library**: it reads the file piece by piece, returns each top-level element individually, and forgets it afterward --> memory-efficient.

While streaming, we fill two tables (called **DataFrames** in pandas):

- `stations_df` with the static station data (one row per station).
- `ts_df` with the time-series data (one row per measurement point, i.e. millions of rows).

### Cell 5 (`In[5]`): index the stations

```python
station_ids = sorted(stations_df.index.unique())
sid_to_idx = {sid: i for i, sid in enumerate(station_ids)}
N = len(station_ids)
```

Each station has a long **UUID** (universally unique identifier), e.g. `c1a4d909-0a00-475a-8e82-18ed13a4eb01`.

We build a **translation table** (`sid_to_idx`): each UUID gets a running integer from 0 to N-1.

### Cell 6 (`In[6]`): static node features

Here we turn the station properties into **features**. **Feature** is the ML term for *"a piece of input information for the model"*.

```python
stations_df["capacity_z"] = zscore(stations_df["capacity"])
stations_df["lat_z"] = zscore(stations_df["lat"])
stations_df["lon_z"] = zscore(stations_df["lon"])
```

**Z-score** = a form of **normalization**: subtract the mean from the value, divide by the standard deviation. Afterward the column has mean 0 and spread 1. Why? Neural networks train more stably when all features are on a similar scale. Otherwise a few large numbers (e.g. longitude with a range of 100) would dominate the model.

```python
region_dummies = pd.get_dummies(region_filled, prefix="region").astype(np.float32)
```

Here `region_id` is represented as a **one-hot encoding**. Instead of a single category column there are now four columns (`region_71`, `region_70`, `region_311`, `region_MISSING`). For each station exactly one of them is `1`, the rest `0`. This is the standard way to feed categories to a neural network.

### Cell 7 (`In[7]`): static adjacency matrix

Here we build the mathematical description of the graph.

```python
A = torch.zeros(N, N, dtype=torch.float32)
for u, v, w in edge_rows:
    A[u, v] += w
    A[v, u] += w
A.diagonal().add_(1.0)
```

`A` is an **N×N matrix** (a square table with N=2,213 rows and columns). Row *u*, column *v* holds the weight of the connection between station *u* and station *v*. This table is the **adjacency matrix**.

The weight `w` is the **number of rides** between the two stations during the training period. Frequently ridden routes get a high weight, rarely ridden ones a low weight.

`A.diagonal().add_(1.0)` adds **self-loops**: in each row *i*, column *i* is increased by 1. This means every station is connected to itself. It's a technical necessity for GCN: without a self-loop, a station would not "see" its own features in the first layer.

```python
deg = A.sum(dim=1)
D_inv_sqrt = torch.diag(1.0 / (deg.sqrt() + 1e-9))
A_norm = D_inv_sqrt @ A @ D_inv_sqrt
```

This is the **symmetric normalization** of the adjacency matrix. Why normalize?

Stations have different numbers of neighbors (their **degree**). A station in the middle of Manhattan might have 200 neighbors, one on the outskirts 20. If, during aggregation, we simply sum all neighbors, high-degree nodes systematically get larger values. We don't want that. Normalizing with `1/√degree` keeps all nodes on a similar scale.

`@` is Python's operator for **matrix multiplication**.

### Cells 8 + 9 (`In[8]`, `In[9]`): resample the time series

The availability series are **event-based**: a new entry only appears when the value changes. That's space-efficient, but impractical for an ML model: the model wants to see a regular cadence.

```python
time_index = pd.date_range(global_start, global_end, freq=freq, inclusive="left")
```

We build a **regular time grid** with 5-minute steps over the whole observation period (4 weeks → 8,064 time points).

Then, for each station and each of the four series:

```python
resampled = grp_series.reindex(time_index, method="ffill").fillna(0).astype(np.float32)
```

`ffill` means **forward fill**: the last measured value is carried forward until a new value arrives. Example: if "5 bikes available" was measured at 17:10 and "3 bikes available" at 17:25, then the time slots 17:10, 17:15, 17:20 all hold the value `5`.

Result: a 3D tensor `X_ts` of shape `[2,213 stations, 8,064 time steps, 4 channels]`. That's about 70 million numbers — but it fits in 285 MB of memory because we use 32-bit floats.

**Tensor** is the generalization of a matrix to any number of dimensions. A 1D tensor is a list, a 2D tensor a table, a 3D tensor a "cube", a 4D tensor a stack of cubes, and so on.

Then z-score normalization of the time series, for the same reason as with the static features.

### Cell 10 (`In[10]`): build the targets

What exactly do we want to predict?

```python
H = CFG.horizon_bins  # 6 bins = 30 minutes
diff = timeline[H:] - timeline[:-H]
pos_bins = np.where(diff > 0)[0]
```

For each edge (u, v) and each time point *t*: is the number of rides in `[t, t+30min]` greater than 0?

Because `num_rides` is a **cumulative counter**, this works via a difference: `count at end − count at start = trips in the window`. If the difference is greater than 0, there was activity → a **positive example** (label = 1). We collect these positive examples as triples `(u, v, t_bin)`.

**Negative examples** (label = 0) are assembled later during training, because there are far more of them than positives (see cell 13).

### Cell 11 (`In[11]`): train/val/test split

```python
train_pos = [s for s in positive_samples if in_range(s[2], min_t, train_end_idx)]
val_pos   = [s for s in positive_samples if in_range(s[2], train_end_idx, val_end_idx)]
test_pos  = [s for s in positive_samples if in_range(s[2], val_end_idx, test_end_idx)]
```

We split the positive examples into three groups:

- **Train** (training set): the first 3 weeks → the model learns on this.
- **Validation** (val set): the next 4 days → during training we check how well the model does on unseen data. Helps tune hyperparameters without burning the test set.
- **Test** (test set): the last 5 days → evaluated **once** at the very end. This is the final assessment.

Important: the split is **temporal** (along the time axis), not random. Why? For time-series data a random split would mean the model learns from the future and predicts the past. That would be cheating (**information leakage**).

### Cell 12 (`In[12]`): model definition

Three small building blocks, then assembled.

#### `GCNBranch` — the graph part

```python
class GCNBranch(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden, bias=False)
        self.W2 = nn.Linear(hidden, out_dim, bias=False)
        self.drop = nn.Dropout(0.2)

    def forward(self, X, A_norm):
        H = A_norm @ self.W1(X)
        H = F.relu(H)
        H = self.drop(H)
        H = A_norm @ self.W2(H)
        return H
```

What happens here, in words?

1. `X` is the table of all static station features (shape `[2,213, 7]`).
2. `self.W1(X)` is a **linear transformation**: each station is replaced by a learned mix of its original features. 7 values become 32 values. `W1` holds weights optimized during training.
3. `A_norm @ ...` is the **neighbor aggregation**: each station additionally gets a mix of its neighbors' values blended in — weighted by the normalized adjacency matrix.
4. `F.relu(H)` = **ReLU activation** (rectified linear unit): negative values are set to 0, positive ones stay. This is an **activation function** and gives the network the ability to learn non-linear relationships.
5. `self.drop` = **dropout**: during training, a random 20 % of the values are set to 0. This is an anti-overfitting technique. **Overfitting** = the model memorizes the training data without generalizing the underlying patterns.
6. Steps 2–4 are repeated a second time → a "two-layer GCN". One layer means: each station sees its direct neighbors. Two layers: each station also sees the neighbors' neighbors (the 2-hop range).

Result: for each of the 2,213 stations, a 32-dimensional **embedding vector** summarizing its position and role in the network.

#### `CNN1DBranch` — the time-series part

```python
class CNN1DBranch(nn.Module):
    def __init__(self, in_channels=4, channels=(16, 32), kernel=3):
        super().__init__()
        c1, c2 = channels
        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel, padding=kernel // 2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return x
```

**1D-CNN** = one-dimensional convolutional neural network. Picture a **filter** like a small sliding window (in our case 3 time steps wide) that glides over the time series. At each position it does a little computation and outputs a number. Different filters detect different patterns: one might react to "fast rise", another to "stable value", yet another to "sudden drop".

The input has shape `[batch size, 12 time steps, 4 channels]`. The channels are our four availability series. After two conv layers and an **average-pooling** operation (averages over time), each station produces a 32-dimensional embedding vector.

#### `LinkPredictor` — the assembly

```python
class LinkPredictor(nn.Module):
    def __init__(self, cfg, in_dim_static, c_ts):
        super().__init__()
        self.gcn = GCNBranch(...)
        self.cnn = CNN1DBranch(...)
        self.mlp = nn.Sequential(
            nn.Linear(fusion_in, cfg.fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(cfg.fusion_hidden, 1),
        )

    def forward(self, X_static, A_norm, ts_u, ts_v, u_idx, v_idx):
        node_emb = self.gcn(X_static, A_norm)
        eu = node_emb[u_idx]
        ev = node_emb[v_idx]
        cu = self.cnn(ts_u)
        cv = self.cnn(ts_v)
        feat = torch.cat([eu, ev, cu, cv], dim=1)
        logits = self.mlp(feat).squeeze(-1)
        return logits
```

For each station pair (u, v):

1. Fetch the GCN embeddings of both stations → `eu`, `ev`.
2. Fetch the CNN embeddings of both stations → `cu`, `cv`.
3. Concatenate all four into a 128-dimensional vector.
4. Send it through a two-stage **MLP** (multi-layer perceptron), a classic small neural network.
5. The MLP outputs a single value, the **logit**. The logit is the "raw score": positive values = the model believes in activity, negative values = it doesn't. Later it's turned into a probability between 0 and 1 via the **sigmoid function**.

### Cell 13 (`In[13]`): batch sampler

Per training step we don't take all examples but a small selection — a **batch**.

```python
def sample_batch(positives, n_pos, neg_per_pos):
    pos = random.sample(positives, k=min(n_pos, len(positives)))
    samples = []
    labels = []
    for u, v, tb in pos:
        samples.append((u, v, tb))
        labels.append(1.0)
        for _ in range(neg_per_pos):
            while True:
                u2 = random.randrange(N)
                v2 = random.randrange(N)
                if u2 == v2: continue
                if (u2, v2, tb) in positive_set: continue
                break
            samples.append((u2, v2, tb))
            labels.append(0.0)
    return samples, labels
```

In words:

1. Draw a few positive examples at random from the training set.
2. For each positive example: create **5 negative examples** by picking random station pairs at the same time point where **no** ride happened.

This method is called **negative sampling**. Why do we need it?

In total there are `N × N × T` possible `(u, v, t)` triples, i.e. well over 100 million. But only a tiny fraction of them has positive activity. This is **extreme class imbalance**. If you showed the model all examples, it would simply learn "always say 0" and have 99.9 % accuracy without being useful. Negative sampling artificially balances positive and negative examples.

### Cell 14 (`In[14]`): training

```python
opt = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(1, CFG.epochs + 1):
    for step in range(steps_per_epoch):
        samples, labels = sample_batch(...)
        ...
        logits = model(...)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    val_auc, val_ap = evaluate(val_pos)
```

What happens per step?

1. **Draw a batch**: 42 positive + 210 negative = 252 training examples.
2. **Forward pass**: run it through the model → the model estimates a logit value for each example.
3. **Compute the loss**: `BCEWithLogitsLoss` is the **binary cross-entropy** loss. It measures how far the estimated logits are from the actual labels. High loss = the model is off, low loss = the model is right.
4. **Backward pass** (`loss.backward()`): the loss is "differentiated" backward through the network. This yields, for every learnable parameter, a **gradient** — a direction in which the parameter should move to reduce the loss.
5. **Optimizer step** (`opt.step()`): the parameters are moved a tiny amount in the direction the gradient points. How big "tiny" is is set by the **learning rate** (`lr`).

This is repeated thousands of times. With every iteration the model gets a little better.

**Adam** is a modern optimizer algorithm that adapts the learning rate per parameter individually. The default choice in most cases.

**Weight decay** is an extra regularization: the weights are nudged slightly toward 0 at every step. It prevents individual weights from becoming too extreme, which again reduces overfitting.

At the end of each epoch, the metrics on the validation set are computed and printed.

### Cell 15 (`In[15]`): test evaluation

After training, the model is evaluated **once** on the test set. There are three metrics for this:

#### AUC (area under the curve)

In full: *area under the ROC curve*. The answer to: *"If I put a random positive and a random negative example side by side — how likely is my model to give the positive the higher probability?"*

- AUC 0.5 = chance (a coin flip).
- AUC 0.8 = decent, clearly better than chance.
- AUC 0.95 = very good.
- AUC 1.0 = perfect.

#### AP (average precision)

A different view: *"If I sort my examples by model score and take the top-K — how many of them are actually positive?"* Aggregated over all possible K. This metric is more robust than plain accuracy on heavily imbalanced data.

#### MRR@100 (mean reciprocal rank)

For each positive test example we generate 99 random negative examples at the same time point and let the model score all 100. We look at the position the positive example lands at when everything is sorted by score.

- Rank 1 = perfect, **reciprocal rank** = 1/1 = 1.0
- Rank 2 = very good, RR = 1/2 = 0.5
- Rank 50 = poor, RR = 1/50 = 0.02

The average over all test examples is the **mean reciprocal rank**. Higher is better. A good value is in the 0.3 – 0.7 range, a top value above 0.8.

### Cell 16 (`In[16]`): learning curve

Draws two plots:

- **Train loss**: should fall continuously over the epochs.
- **Validation AUC and AP**: should rise over the epochs and plateau at the end.

If the val metrics start falling while the train loss keeps falling: **overfitting** — the model memorizes the training data but no longer generalizes. Countermeasure: fewer epochs or more regularization.

## 6. Glossary of the most important terms

| Term | Meaning |
|---|---|
| **Adjacency matrix** | Square table that compactly represents all connections in a graph. Row *u*, column *v* = the connection weight between nodes *u* and *v*. |
| **AP (average precision)** | Metric for classifiers: measures how well the positive examples land among the top-scored items. |
| **AUC (area under the curve)** | Metric for classifiers: the probability that the model scores a random positive example higher than a random negative one. |
| **Backpropagation** | Algorithm that sends the error backward from the output through the network and thus computes the right gradient for each parameter. |
| **Batch** | A subset of the training data processed in one optimizer step. |
| **CNN (convolutional neural network)** | Neural network with filters that slide like a window over the input. Originally for images, also works for time series (then "1D-CNN"). |
| **DataFrame** | A pandas data structure for tabular data. Like an Excel sheet, but programmatic. |
| **Dropout** | Anti-overfitting trick: during training, random neuron activations are set to 0. |
| **Edge** | Connection between two nodes in a graph. |
| **Edge weight** | Numeric weight of an edge. In our case, the number of historical rides. |
| **Embedding** | A learned vector that summarizes an entity (node, word, image) as a list of numbers. |
| **Epoch** | One full pass through the training dataset. |
| **Feature** | A piece of input information for a model. Also: "variable", "attribute". |
| **Forward fill** | Method for closing gaps in time series by carrying the last known value forward. |
| **GCN (graph convolutional network)** | Neural network that exchanges and aggregates information between neighboring nodes in a graph. |
| **Gradient** | Vector that indicates, for each parameter, the direction in which it should move to reduce the loss. |
| **GPU** | Graphics card. Massively accelerates matrix operations, which helps neural networks a lot. |
| **Graph** | Mathematical description of a network of nodes and edges. |
| **Hyperparameter** | Knobs the human chooses and the model does **not** learn itself (learning rate, number of layers, batch size, etc.). |
| **JSON** | File format for storing structured data as text. Readable by humans and machines. |
| **Node** | A single point in a graph. For us: a bike station. |
| **Layer** | One layer in a neural network. Several layers in a row = a "deep" network. |
| **Learning rate** | How strongly the model adjusts its parameters per optimizer step. Too high = unstable. Too low = learns slowly. |
| **Link prediction** | The task of predicting whether (or when) an edge forms between two nodes. |
| **Logit** | The raw output of a classification model before it's turned into a probability via sigmoid. Range (-∞, +∞). |
| **Loss / loss function** | Mathematical function that measures how wrong the model is. Training tries to minimize it. |
| **Matrix** | A table of numbers. Two-dimensional. |
| **MLP (multi-layer perceptron)** | Classic simple neural network of several fully connected layers. |
| **MRR (mean reciprocal rank)** | Metric for ranking tasks: on average, at which position the correct item lands in the sort order. |
| **Negative sampling** | Trick for training on classification tasks with extreme imbalance: for each positive example, negatives are drawn at random. |
| **Neural network** | A model of many small, chained compute functions that together can learn complex relationships. |
| **One-hot encoding** | Representing a category as a vector: for N possible values, an N-dimensional vector with exactly one 1 and the rest 0. |
| **Optimizer** | Algorithm that updates a neural network's parameters based on the gradients (e.g. Adam, SGD). |
| **Overfitting** | The model memorizes the training data but doesn't generalize to new data. |
| **Padding** | Filling a sequence or an image with dummy values so operations work cleanly at the edges. |
| **Pandas** | Python library for tabular data. |
| **Parameter** | The internal numbers of a neural network, optimized during training. Also "weights". |
| **PyTorch** | Open-source library for neural networks, developed by Meta. |
| **Reproducibility** | An experiment yields the same result when run again. |
| **ReLU (rectified linear unit)** | Activation function. `ReLU(x) = max(0, x)`. Brings non-linearity into the network. |
| **Resampling** | Mapping a time series onto a different time grid. |
| **Seed** | Starting value for random generators. Fixed for reproducible results. |
| **Self-loop** | Edge from a node to itself. Technically needed in GCN so nodes see their own features. |
| **Sigmoid** | Mathematical function that maps values from (-∞, +∞) into (0, 1). Used to convert logits into probabilities. |
| **Streaming** | Processing data piece by piece instead of loading everything at once. |
| **Tensor** | Generalization of a matrix to any number of dimensions. PyTorch's base data type. |
| **Test set** | The part of the data used only once, at the end, for the final evaluation. |
| **Time series** | A sequence of measured values with timestamps. |
| **Train set** | The part of the data the model learns on. |
| **UUID** | Universally unique identifier. A long string as a unique ID, e.g. `c1a4d909-0a00-475a-8e82-18ed13a4eb01`. |
| **Validation set** | The part of the data on which performance is checked during training. Helps with model selection without burning the test set. |
| **Vector** | A list of numbers. One-dimensional. |
| **Weight decay** | A form of regularization: weights are nudged slightly toward 0 at every step. |
| **Z-score** | Normalization: `(value − mean) / standard deviation`. Result: mean 0, spread 1. |

## 7. If someone wants to read the notebook: the order to explain it

If you want to present the project to someone, here is a sensible didactic order:

1. **What do we want to predict?** The probability of a ride between two stations in the next 30-minute window.
2. **What data do we have?** A graph of stations, plus an availability time series per station, plus a cumulative trip series per station pair.
3. **What's the difficulty?** Class imbalance (rides between any given pair are rare) and mixed data types (graph plus time series).
4. **How do we model it?** Two specialized neural networks (one for the graph, one for the time series), whose outputs come together in a fusion MLP.
5. **How do we evaluate?** Three metrics (AUC, AP, MRR), a temporal train/val/test split, negative sampling.
6. **What's the status?** A first end-to-end iteration, default choice of components, later ablations and baselines.

Anyone who wants to go deeper proceeds from there into the specific notebook sections.
