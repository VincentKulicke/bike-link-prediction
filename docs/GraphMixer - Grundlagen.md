---
tags: [project, baseline, link-prediction, explainer, graphmixer]
status: in progress
created: 2026-06-13
belongs-to: "[[Link Prediction on Hybrid Graph + Time Series Data.md|Link Prediction project]]"
---

# GraphMixer – the basics, explained plainly

This document explains the **GraphMixer model** from the ground up, so it makes sense even without a background in graph deep learning. In our project GraphMixer is the **temporal-graph baseline** that we pit our own hybrid model against in the binary comparison (link yes/no).

> Source: Cong et al., *"Do We Really Need Complicated Model Architectures for Temporal Networks?"*, ICLR 2023. The paper's core message is already in the title: a **deliberately simple** model surprisingly often beats the complicated ones.

---

## 1. The problem GraphMixer solves

Picture a network where **connections keep forming** — for us: rides between bike-sharing stations. The question is:

> *Will there be a ride between station A and station B in the next half hour — yes or no?*

This is called **link prediction**. What makes it special: the network is **dynamic**, it changes over time. A connection that is typical at 8 a.m. (commuters to the station) doesn't exist at 3 a.m. So a model has to learn not only *who-with-whom* but also *when*.

---

## 2. The basics (step by step)

### Graph
A **graph** is a collection of points and connections.
- **Node**: the points. For us: the bike stations.
- **Edge**: a connection between two nodes. For us: a ride from station A to station B.

Visually: a map where stations are points and each ride is a line between two points.

### Static vs. temporal graph
- **Static graph**: the connections are "frozen". There is a line between A and B or not — with no time information.
- **Temporal graph**: every edge has a **timestamp**. An edge isn't "A–B", it's "A–B at 08:14", "A–B at 08:21", and so on. Each individual ride is its own event with a time.

GraphMixer works with a **temporal graph**. That is exactly why, for this baseline, we need the **individual, timestamped rides** from the original Citi Bike dataset — not our aggregated "super-edge", which collapses all rides between A and B into a single number.

### Node features
Every node can carry **properties** expressed as numbers. For a station, e.g. capacity (number of docks) and geographic location. Such lists of numbers are called a **feature vector**.

### Embedding
An **embedding** is a compact list of numbers that a model learns itself to describe something "in its own words". Instead of saying "Grove St PATH station", the model describes the station with, say, 100 numbers that summarize its behavior (e.g. "very active commuter hub, high demand in the morning"). Two stations with similar behavior get similar embeddings.

---

## 3. GraphMixer's core idea in one sentence

> To predict whether A and B will connect soon, GraphMixer looks at the **recent past of both stations** — *with whom* and *when* they last had connections — and infers from that how likely a new A–B connection is.

The clever bit: GraphMixer deliberately skips complicated components (no "attention", no "memory module", no recurrent networks). It uses almost exclusively the **simplest neural building block there is** — the MLP. And it is still very competitive.

### What is an MLP?
**MLP** = *multilayer perceptron*, the classic, simplest neural network: it takes a list of numbers as input, multiplies and adds them across several "layers", and outputs a new list of numbers. Think of it as a **flexible mathematical function** that learns from examples how to translate inputs into useful outputs.

---

## 4. The build: three components

GraphMixer has three parts. We'll go through them in order.

### Component 1 – the link encoder ("what happened recently?")

This part summarizes a node's **most recent connections**.

How it works:
1. For a station, take its **last K rides** (e.g. the last 20 events).
2. Each of those rides becomes a row in a table. The row contains:
   - the **features** of the ride (e.g. to which station, plus any further properties),
   - a **time encoding** — i.e. "how long ago was this?" translated into a number form the model can read.
3. This gives a **table**: rows = the last K rides, columns = the features + time info.
4. An **MLP-Mixer** is applied to this table (see section 5), condensing it into a single embedding: "this is what this station's recent activity looked like."

**Key design choice – the fixed time encoding:** how "how long ago" is turned into numbers is **fixed** in GraphMixer and is *not* learned. The authors found that learnable time encodings destabilize training. The fixed version (a mathematical function with cosine waves of different speeds) is more robust. This is one reason GraphMixer is so stable and simple.

### Component 2 – the node encoder ("how active is the station in general?")

This part describes a node's **general identity and recent activity**, independent of the exact order of events.

How it works:
- Look at the station's **neighbors** in the last time window (all stations it was recently connected to) and take an **average** of their features.
- This gives a compact summary: "this station was heavily/lightly used recently and is connected to these kinds of stations."

Where the link encoder emphasizes the *temporal sequence*, the node encoder gives the *big picture* of the station.

### Component 3 – the link classifier ("connection yes or no?")

Now the actual prediction for a pair (A, B):
1. Take the embeddings of both stations (each from the link and node encoders).
2. **Join them** (concatenate the number lists of A and B).
3. A final **MLP** reads this combined description and outputs a **probability**: how likely is a connection between A and B?
4. If the probability exceeds a threshold, the prediction is "link = yes".

---

## 5. The heart: the MLP-Mixer

The name "GraphMixer" comes from this component. It originally comes from computer vision (MLP-Mixer, 2021), where it replaces more complicated mechanisms with two simple MLPs applied in alternation.

Recall: the link encoder built a **table** — rows = recent rides, columns = features. The MLP-Mixer mixes this table in **two ways**:

1. **Token mixing ("mix column-wise")**
Mixes information **across the different rides** (across the rows). Answers: *how do the individual recent events relate to each other?* Example: "three rides in quick succession to the same station" becomes a recognizable pattern.

2. **Channel mixing ("mix row-wise")**
Mixes information **across the different features** (across the columns). Answers: *how do the properties within a single ride relate to each other?*

These two mixing steps are applied in alternation. The result is a compact summary of the entire recent activity — produced **with MLPs only**, without the compute-heavy mechanisms of other models.

**Analogy:** picture a table of notes. Token mixing reads *column-wise* (compares the same property across all events), channel mixing reads *row-wise* (looks at all properties of an event together). By reading in both directions in turn, an overall understanding of the table emerges.

---

## 6. Why GraphMixer is simpler than TGN

In our assignment GraphMixer was explicitly recommended because it is **easier to implement** than TGN. The difference:

| Property | TGN (complex) | GraphMixer (simple) |
|---|---|---|
| **Memory module** | yes – stores a continuously updated state per node | **no** |
| **Recurrent networks (RNN/GRU)** | yes | **no** |
| **Attention** | yes | **no** |
| **Time encoding** | learned | **fixed** (more stable) |
| **Main building block** | several interacting modules | almost only **MLPs** |

Fewer moving parts means: less that can go wrong, faster training, easier debugging. That is exactly what makes GraphMixer the ideal, solid baseline.

---

## 7. How GraphMixer fits our project

- **Role:** temporal-graph baseline for the **binary comparison** (link yes/no).
- **Input:** the **temporal graph** of individual, timestamped rides `(start, end, time)` — from the original Citi Bike dataset for May–June 2024, filtered to the same active stations as our main dataset.
- **Output:** per station pair and time window, a probability for "connection yes/no".
- **Comparison:** GraphMixer (binary only) against the **binary head of our hybrid model**. For our model, `number of rides > 0` ⇒ "link = yes".
- **Metrics:** AUC, AP, Accuracy/F1 (see the [[Konzeptdokument.md|concept document]]).

Important: GraphMixer makes **no count prediction** (how many rides). The count comparison is handled by the separate **LSTM baseline**. So GraphMixer deliberately covers only *one* of the two tasks.

---

## 8. Strengths and limits

**Strengths**
- Easy to implement and train (hardly any complex modules).
- Stable thanks to the fixed time encoding.
- Very competitive despite its simplicity — a fair, serious baseline.
- Uses the temporal information of real individual events, not just aggregates.

**Limits**
- No explicit long-term memory per node (only looks at the last K events).
- Purely **binary** — doesn't answer "how many rides".
- Uses **no continuous node time series** (like bike availability). This is exactly the gap our hybrid model fills — and the reason we expect to beat GraphMixer on the hybrid problem.

---

## 9. Quick summary (for the reader in a hurry)

- GraphMixer predicts whether a connection between two nodes will form soon.
- It looks at the **recent events** of both nodes, encodes "when" with a **fixed** time function, and condenses everything with an **MLP-Mixer** (two simple mixing steps: across events and across features).
- Three components: **link encoder** (recent activity), **node encoder** (big picture), **link classifier** (prediction).
- Deliberately **simpler than TGN** (no memory, no attention, no RNNs) — ideal as a baseline.
- In our project: the **binary temporal-graph baseline** on the timestamped individual rides.
