---
tags: [projekt, erklärung, didaktik]
status: aktiv
erstellt: 2026-05-17
projekt: Link Prediction on Hybrid Graph + Time Series Data
---

# Iteration 1: Notebook-Erklärung für Fachfremde

Begleitende Erklärung zu [[iteration1_gcn_cnn_fusion.ipynb|Iteration 1: GCN + 1D-CNN + Fusion-MLP]]
## 1. Worum geht es?

In New York City gibt es das **Citi-Bike-System**: knapp 2.200 Stationen, an denen man Leihräder abholt und wieder abgibt. Pro Tag finden Tausende von Fahrten zwischen verschiedenen Stationspaaren statt.

Die Fragestellung dieses Projekts lautet sinngemäß:

> Wenn ich gerade an Station A stehe und Station B betrachte: wie wahrscheinlich ist es, dass innerhalb der nächsten Zeiteinheit (30 Minuten) mindestens eine Person von A nach B fährt?

Diese Vorhersage ist nützlich für:

- **Rebalancing**: Wenn das System weiß, wo gleich viele Räder gebraucht werden, kann es vorbeugend Räder umverteilen.
- **Demand Planning**: Stadt und Betreiber können die Auslastung besser planen.
- **Anomalie-Erkennung**: Auffällige Abweichungen vom üblichen Muster sind oft Hinweise auf Events, Wetterveränderungen oder technische Probleme.

Wir bauen also einen Algorithmus, der lernt: *"Aus dem bisherigen Verhalten der Stationen und ihrer Vernetzung kann ich abschätzen, wo gleich Aktivität stattfinden wird."*

## 2. Welche Daten haben wir?

Zwei Dateien:

### `graph_nodes.json` (die Stationen)

Pro Station gibt es:

- **Statische Informationen** (ändern sich nicht): Name, Geokoordinaten, Anzahl Docks (Stellplätze), Region.
- **Zeitreihen**: alle 5 Minuten wird gemessen, wie viele Räder gerade verfügbar sind, wie viele E-Bikes davon, wie viele Räder defekt sind, wie viele Docks defekt sind. Insgesamt vier Messreihen pro Station.

### `graph_edges.json` (die Fahrten zwischen Stationspaaren)

Pro Station-Paar (A → B), zwischen denen mindestens eine Fahrt stattgefunden hat, gibt es eine sogenannte **Aggregat-Kante** mit:

- **Zeitstempel der ersten Fahrt** zwischen diesem Paar.
- **Sechs Zähler-Zeitreihen**: Gesamt-Fahrten, klassische Räder, E-Bikes, Member-Fahrten, Casual-Fahrten, gerade laufende Fahrten.
### Neuronales Netz

Eine bestimmte Art von Machine-Learning-Modell, die lose von biologischen Neuronen inspiriert ist. Im Kern besteht es aus vielen kleinen Rechenfunktionen, die hintereinandergeschaltet sind. Jede Funktion hat "Gewichte" — Zahlen, die beim Training so eingestellt werden, dass das Modell aus dem Eingangswert den richtigen Ausgangswert produziert.

### Link Prediction

Der Fachbegriff für unsere Aufgabe: vorhersagen, ob eine Kante (ein "Link") zwischen zwei Knoten zu einem bestimmten Zeitpunkt entsteht.

## 4. Was bauen wir konkret?

Unser Modell besteht aus drei Komponenten:

```
                ┌─────────────────────────-┐
                │   Graph-Branch (GCN)     │
Stationsdaten ──┤                          ├─-┐
                │  liefert pro Station     │  │
                │  einen "Embedding"-Vektor│  │
                └────────────────────────-─┘  │
                                              ├── Fusion-MLP ── Wahrscheinlichkeit
                ┌─────────────────────────-┐  │
                │   Zeitreihen-Branch      │  │
Verfügbarkeit ──┤   (1D-CNN)               ├──┘
ts pro Station  │                          │
                │  liefert pro Station     │
                │  einen "Embedding"-Vektor│
                └────────────────────────-─┘
```

1. **Graph-Branch (GCN)**: Schaut sich für jede Station ihre Nachbarn im Stationsnetzwerk an und fasst zusammen: *"Wer bist du, basierend auf deiner Position im Netzwerk und den Stationen um dich herum?"* Antwort ist eine Zahlenliste — der **Embedding-Vektor**.

2. **Zeitreihen-Branch (1D-CNN)**: Schaut sich für jede Station an, was in den letzten 60 Minuten an Verfügbarkeit passiert ist, und destilliert daraus ein anderes Profil: *"Wie aktiv ist die Station gerade? Welches Muster zeigt sie aktuell?"* Auch hier kommt ein Embedding-Vektor heraus.

3. **Fusion-MLP**: Bekommt vier Embedding-Vektoren — von Start-Station A und Ziel-Station B, jeweils einmal aus dem Graph-Branch und einmal aus dem Zeitreihen-Branch. Verschmilzt sie und sagt: *"Mit welcher Wahrscheinlichkeit fährt in den nächsten 30 Minuten jemand von A nach B?"*

## 5. Das Notebook Zelle für Zelle

### Zelle 0: Setup-Hinweis

Hier steht eine kommentierte Zeile, mit der man im allerersten Schritt die benötigten Software-Bibliotheken installiert:

```python
# %pip install --quiet torch pandas numpy scikit-learn ijson tqdm matplotlib
```

Das `%` ist eine Jupyter-Spezialität: damit kann man **innerhalb** eines Notebooks Befehle ausführen, die normalerweise auf der Kommandozeile laufen. `pip install` ist der Standard-Befehl in Python, um Bibliotheken aus dem Internet zu installieren.

- **torch** = PyTorch, das Framework für neuronale Netze.
- **pandas** = Bibliothek zum Arbeiten mit Tabellen-Daten.
- **numpy** = Bibliothek für Zahlen-Arrays und mathematische Operationen.
- **scikit-learn** = klassische Machine-Learning-Bibliothek, hier für Bewertungsmetriken.
- **ijson** = Bibliothek, um große JSON-Dateien Stück für Stück zu lesen ohne alles auf einmal in den Speicher zu laden.
- **tqdm** = zeigt schöne Fortschrittsbalken.
- **matplotlib** = zum Zeichnen von Diagrammen.

### Zelle 1 (`In[1]`): Imports

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

**Seed** = Startwert für Zufallszahlengeneratoren. Wir setzen ihn fest auf 42, damit das Experiment **reproduzierbar** ist. Wenn man das nicht macht, würde jedes Training andere zufällige Initialwerte verwenden und etwas andere Ergebnisse liefern.

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Prüft, ob eine **GPU** (Grafikkarte) verfügbar ist. GPUs können neuronale Netze um Größenordnungen schneller trainieren als CPUs. Falls keine GPU da ist, fällt der Code auf die **CPU** zurück.

### Zelle 2 (`In[2]`): Konfiguration

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

Einstellen/ Konfigurieren der **Hyperparameter** :

- `bin_minutes: 5` = wir teilen die Zeit in 5-Minuten-Häppchen ein.
- `window_bins: 12` = das Eingabe-Fenster ist 12 Häppchen lang, also 60 Minuten.
- `horizon_bins: 6` = wir sagen 6 Häppchen voraus, also die nächsten 30 Minuten.
- `epochs: 10` = wir lassen das Modell 10 Mal über die kompletten Trainingsdaten laufen.
- `batch_size: 256` = pro Trainings-Schritt verarbeiten wir 256 Beispiele gleichzeitig.
- `lr: 1e-3` = die **Lernrate** (Learning Rate). Steuert, wie stark das Modell pro Trainings-Schritt seine internen Parameter anpasst. `1e-3` heißt 0.001.

### Zelle 3 (`In[3]`): Rohdaten — Edges laden

```python
with open(CFG.data_dir / "graph_edges.json", "rb") as f:
    edges_raw = json.load(f)
```

### Zelle 4 (`In[4]`): Rohdaten — Nodes laden (Streaming)

```python
with open(CFG.data_dir / "graph_nodes.json", "rb") as f:
    for node in ijson.items(f, "item"):
        ...
```

Die Nodes-Datei ist 530 MB groß. Würde man sie wie die Edges-Datei mit `json.load` komplett einlesen, würde Python sie als gigantische verschachtelte Datenstruktur im Speicher halten — kann eng werden. 
--> Verwendung von `ijson` als  **Streaming-Bibliothek**: sie liest die Datei Stück für Stück, gibt jedes Top-Level-Element einzeln zurück und vergisst es danach wieder --> Speicher-effizient.

Während des Streamings füllen wir zwei Tabellen (in Pandas heißen die **DataFrames**):

- `stations_df` mit den statischen Stationsdaten (eine Zeile pro Station).
- `ts_df` mit den Zeitreihen-Daten (eine Zeile pro Messpunkt, also Millionen Zeilen).

### Zelle 5 (`In[5]`): Stationen indizieren

```python
station_ids = sorted(stations_df.index.unique())
sid_to_idx = {sid: i for i, sid in enumerate(station_ids)}
N = len(station_ids)
```

Jede Station hat eine lange **UUID** (Universally Unique Identifier), z.B. `c1a4d909-0a00-475a-8e82-18ed13a4eb01`. 

Wir bauen eine **Übersetzungstabelle** (`sid_to_idx`): jede UUID bekommt eine fortlaufende Integer-Zahl von 0 bis N-1.

### Zelle 6 (`In[6]`): Statische Node-Features

Hier wandeln wir die Stations-Eigenschaften in **Features** um. **Feature** ist der ML-Fachbegriff für *"eine Eingangsinformation für das Modell"*.

```python
stations_df["capacity_z"] = zscore(stations_df["capacity"])
stations_df["lat_z"] = zscore(stations_df["lat"])
stations_df["lon_z"] = zscore(stations_df["lon"])
```

**Z-Score** = eine Form der **Normalisierung**: vom Wert wird der Mittelwert abgezogen, das Ergebnis durch die Standardabweichung geteilt. Danach hat die Spalte Mittelwert 0 und Streuung 1. Warum? Neuronale Netze trainieren stabiler, wenn alle Features auf einer ähnlichen Skala liegen. Sonst würden ein paar große Zahlen (z.B. Längengrad mit 100 Schritten Spannweite) das Modell dominieren.

```python
region_dummies = pd.get_dummies(region_filled, prefix="region").astype(np.float32)
```

Hier wird die `region_id` als **One-Hot-Encoding** dargestellt. Statt einer einzigen Kategorie-Spalte gibt es jetzt vier Spalten (`region_71`, `region_70`, `region_311`, `region_MISSING`). Für jede Station ist genau eine davon `1`, der Rest `0`. Das ist die Standard-Form, in der man Kategorien einem neuronalen Netz übergibt.

### Zelle 7 (`In[7]`): Statische Adjazenzmatrix

Hier bauen wir die mathematische Beschreibung des Graphen.

```python
A = torch.zeros(N, N, dtype=torch.float32)
for u, v, w in edge_rows:
    A[u, v] += w
    A[v, u] += w
A.diagonal().add_(1.0)
```

`A` ist eine **N×N-Matrix** (eine quadratische Tabelle mit N=2.213 Zeilen und Spalten). Zeile *u* und Spalte *v* enthält das Gewicht der Verbindung zwischen Station *u* und Station *v*. Diese Tabelle heißt **Adjazenzmatrix**.

Das Gewicht `w` ist die **Anzahl Fahrten** zwischen den beiden Stationen während der Trainingsperiode. Häufig befahrene Routen bekommen ein hohes Gewicht, selten befahrene ein niedriges.

`A.diagonal().add_(1.0)` fügt **Self-Loops** hinzu: in jeder Zeile *i* wird Spalte *i* um 1 erhöht. Das bedeutet, jede Station ist mit sich selbst verbunden. Das ist eine technische Notwendigkeit für GCN: ohne Self-Loop würde eine Station ihre eigenen Features in der ersten Layer nicht "sehen".

```python
deg = A.sum(dim=1)
D_inv_sqrt = torch.diag(1.0 / (deg.sqrt() + 1e-9))
A_norm = D_inv_sqrt @ A @ D_inv_sqrt
```

Dies ist die **symmetrische Normalisierung** der Adjazenzmatrix. Warum normalisieren?

Stationen haben unterschiedlich viele Nachbarn (sogenannter **Grad**, englisch *degree*). Eine Station mitten in Manhattan hat vielleicht 200 Nachbarn, eine am Stadtrand 20. Wenn wir später bei der Aggregation einfach alle Nachbarn aufsummieren, bekommen Knoten mit hohem Grad systematisch größere Werte. Das wollen wir nicht. Die Normalisierung mit `1/√Grad` sorgt dafür, dass alle Knoten auf einer ähnlichen Skala bleiben.

`@` ist in Python der Operator für **Matrixmultiplikation**.

### Zelle 8 + 9 (`In[8]`, `In[9]`): Resampling der Zeitreihen

Die Verfügbarkeits-Zeitreihen sind **event-basiert**: ein neuer Eintrag entsteht nur, wenn sich der Wert ändert. Das ist platzsparend, aber für ein ML-Modell unpraktisch: das Modell will einen geregelten Takt sehen.

```python
time_index = pd.date_range(global_start, global_end, freq=freq, inclusive="left")
```

Wir bauen ein **regelmäßiges Zeit-Raster** mit 5-Minuten-Schritten über den gesamten Beobachtungszeitraum (4 Wochen → 8.064 Zeitpunkte).

Dann für jede Station und jede der vier Messreihen:

```python
resampled = grp_series.reindex(time_index, method="ffill").fillna(0).astype(np.float32)
```

`ffill` heißt **Forward Fill**: der letzte gemessene Wert wird so lange fortgeschrieben, bis ein neuer Wert kommt. Beispiel: Wenn um 17:10 gemessen wurde "5 Räder verfügbar" und um 17:25 "3 Räder verfügbar", dann steht in den Zeit-Slots 17:10, 17:15, 17:20 jeweils der Wert `5`.

Resultat: ein 3D-Tensor `X_ts` der Form `[2.213 Stationen, 8.064 Zeitschritte, 4 Kanäle]`. Das sind etwa 70 Millionen Zahlen — passt aber in 285 MB Speicher, weil wir 32-Bit-Floats verwenden.

**Tensor** ist die Verallgemeinerung von Matrix auf beliebig viele Dimensionen. Ein 1D-Tensor ist eine Liste, ein 2D-Tensor eine Tabelle, ein 3D-Tensor ein "Würfel", ein 4D-Tensor ein Stapel von Würfeln, und so weiter.

Anschließend Z-Score-Normalisierung der Zeitreihen, gleicher Grund wie bei den statischen Features.

### Zelle 10 (`In[10]`): Targets aufbauen

Was wollen wir genau vorhersagen?

```python
H = CFG.horizon_bins  # 6 Bins = 30 Minuten
diff = timeline[H:] - timeline[:-H]
pos_bins = np.where(diff > 0)[0]
```

Für jede Edge (u, v) und jeden Zeitpunkt *t*: ist die Anzahl der Fahrten in `[t, t+30min]` größer als 0?

Weil `num_rides` ein **kumulativer Zähler** ist, geht das per Differenz: `Anzahl am Ende − Anzahl am Anfang = Trips im Fenster`. Wenn die Differenz größer 0 ist, war Aktivität → **positives Beispiel** (label = 1). Diese positiven Beispiele sammeln wir als Tripel `(u, v, t_bin)`.

**Negative Beispiele** (Label = 0) bauen wir später im Training selbst zusammen, weil es davon viel mehr gibt als positive (siehe Zelle 13).

### Zelle 11 (`In[11]`): Train/Val/Test-Split

```python
train_pos = [s for s in positive_samples if in_range(s[2], min_t, train_end_idx)]
val_pos   = [s for s in positive_samples if in_range(s[2], train_end_idx, val_end_idx)]
test_pos  = [s for s in positive_samples if in_range(s[2], val_end_idx, test_end_idx)]
```

Wir teilen die positiven Beispiele in drei Gruppen:

- **Train** (Trainings-Set): die ersten 3 Wochen → das Modell lernt darauf.
- **Validation** (Val-Set): die nächsten 4 Tage → wir prüfen während des Trainings, wie gut das Modell auf ungesehene Daten ist. Hilft, Hyperparameter zu tunen, ohne den Test-Set zu verbrennen.
- **Test** (Test-Set): die letzten 5 Tage → ganz am Ende **einmal** auswerten. Das ist die finale Bewertung.

Wichtig: der Split ist **temporal** (entlang der Zeitachse), nicht zufällig. Warum? Bei Zeitreihen-Daten würde ein zufälliger Split bedeuten, dass das Modell aus der Zukunft lernt und die Vergangenheit vorhersagt. Das wäre Schummelei (**Information Leakage**).

### Zelle 12 (`In[12]`): Modell-Definition

Drei kleine Bausteine, dann zusammengebaut.

#### `GCNBranch` — der Graph-Teil

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

Was passiert hier in Worten?

1. `X` ist die Tabelle aller statischen Station-Features (Form `[2.213, 7]`).
2. `self.W1(X)` ist eine **lineare Transformation**: jede Station wird durch eine erlernte Mischung ihrer Original-Features ersetzt. Aus 7 Werten werden 32 Werte. Das `W1` enthält Gewichte, die im Training optimiert werden.
3. `A_norm @ ...` ist die **Nachbar-Aggregation**: jede Station bekommt zusätzlich einen Mix aus den Werten ihrer Nachbarn dazugemischt — gewichtet nach der normalisierten Adjazenzmatrix.
4. `F.relu(H)` = **ReLU-Aktivierung** (Rectified Linear Unit): negative Werte werden auf 0 gesetzt, positive bleiben. Das ist eine sogenannte **Aktivierungsfunktion** und gibt dem Netz die Möglichkeit, nichtlineare Zusammenhänge zu lernen.
5. `self.drop` = **Dropout**: während des Trainings werden zufällig 20 % der Werte auf 0 gesetzt. Das ist eine Anti-Overfitting-Technik. **Overfitting** = das Modell lernt die Trainingsdaten auswendig, ohne die zugrunde liegenden Muster zu generalisieren.
6. Schritt 2-4 wird ein zweites Mal wiederholt → "zwei-Layer-GCN". Eine Layer bedeutet: jede Station sieht ihre direkten Nachbarn. Zwei Layers: jede Station sieht auch die Nachbarn-Nachbarn (also den 2-Hop-Bereich).

Ergebnis: für jede der 2.213 Stationen ein 32-dimensionaler **Embedding-Vektor**, der ihre Position und Rolle im Netzwerk zusammenfasst.

#### `CNN1DBranch` — der Zeitreihen-Teil

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

**1D-CNN** = eindimensionales Convolutional Neural Network. Stell dir einen **Filter** wie ein kleines Schiebefenster vor (in unserem Fall 3 Zeitschritte breit), das über die Zeitreihe gleitet. An jeder Position macht es eine kleine Rechnung und gibt eine Zahl aus. Verschiedene Filter erkennen verschiedene Muster: einer könnte auf "schneller Anstieg" reagieren, ein anderer auf "stabiler Wert", wieder ein anderer auf "plötzlicher Drop".

Der Eingang ist die Form `[Batch-Größe, 12 Zeitschritte, 4 Kanäle]`. Die Kanäle sind unsere vier Verfügbarkeits-Serien. Nach zwei Conv-Layern und einer **Average-Pooling**-Operation (mittelt über die Zeit) kommt pro Station ein 32-dimensionaler Embedding-Vektor heraus.

#### `LinkPredictor` — der Zusammenbau

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

Für jedes Stationspaar (u, v):

1. Hol die GCN-Embeddings beider Stationen → `eu`, `ev`.
2. Hol die CNN-Embeddings beider Stationen → `cu`, `cv`.
3. Hänge alle vier hintereinander zu einem 128-dimensionalen Vektor.
4. Schick den durch ein zweistufiges **MLP** (Multi-Layer Perceptron), ein klassisches kleines neuronales Netz.
5. Das MLP gibt einen einzelnen Wert aus, den **Logit**. Logit ist die "Rohpunktzahl": positive Werte = Modell glaubt an Aktivität, negative Werte = glaubt nicht dran. Übersetzt sich später per **Sigmoid-Funktion** in eine Wahrscheinlichkeit zwischen 0 und 1.

### Zelle 13 (`In[13]`): Batch-Sampler

Pro Trainings-Schritt nehmen wir nicht alle Beispiele, sondern eine kleine Auswahl — einen **Batch**.

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

In Worten:

1. Zieh zufällig ein paar positive Beispiele aus dem Trainings-Set.
2. Für jedes positive Beispiel: erzeuge **5 negative Beispiele** durch zufällige Stationspaar-Wahl zum gleichen Zeitpunkt, bei denen aber **keine** Fahrt stattgefunden hat.

Diese Methode heißt **Negative Sampling**. Warum brauchen wir das?

Insgesamt gibt es `N × N × T` mögliche `(u, v, t)`-Tripel, also weit über 100 Millionen. Aber nur ein winziger Bruchteil davon hat positive Aktivität. Das ist **extreme Klassen-Unbalance**. Wenn man dem Modell alle Beispiele zeigen würde, würde es einfach lernen "sag immer 0" und hätte 99,9 % Genauigkeit, ohne irgendwas Nützliches zu können. Negative Sampling balanciert positive und negative Beispiele künstlich aus.

### Zelle 14 (`In[14]`): Training

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

Was passiert pro Schritt?

1. **Batch ziehen**: 42 positive + 210 negative = 252 Trainings-Beispiele.
2. **Forward Pass**: durch das Modell jagen → das Modell schätzt für jedes Beispiel einen Logit-Wert.
3. **Loss berechnen**: `BCEWithLogitsLoss` ist die **Binary Cross-Entropy**-Verlustfunktion. Sie misst, wie weit die geschätzten Logits von den tatsächlichen Labels abweichen. Hoher Loss = Modell liegt daneben, niedriger Loss = Modell liegt richtig.
4. **Backward Pass** (`loss.backward()`): die Verlustfunktion wird rückwärts durch das Netz "abgeleitet". Daraus ergibt sich für jeden lernbaren Parameter im Netz ein **Gradient** — eine Richtung, in die der Parameter verschoben werden müsste, um den Loss zu verkleinern.
5. **Optimizer-Schritt** (`opt.step()`): die Parameter werden ein winziges Stückchen in die Richtung verschoben, die der Gradient vorgibt. Wie groß "winzig" ist, regelt die **Lernrate** (`lr`).

Das wird Tausende Male wiederholt. Mit jeder Iteration wird das Modell ein kleines bisschen besser.

**Adam** ist ein moderner Optimizer-Algorithmus, der die Lernrate pro Parameter individuell anpasst. Standard-Wahl in den meisten Fällen.

**Weight Decay** ist eine zusätzliche Regularisierung: die Gewichte werden bei jedem Schritt minimal nach 0 gezogen. Verhindert, dass einzelne Gewichte zu extrem werden, was wieder Overfitting reduziert.

Am Ende jeder Epoche werden die Metriken auf dem Validation-Set berechnet und ausgegeben.

### Zelle 15 (`In[15]`): Test-Auswertung

Nach dem Training wird das Modell **einmal** auf dem Test-Set bewertet. Dafür gibt es drei Metriken:

#### AUC (Area Under the Curve)

Im Detail: *Area under the ROC Curve*. Antwort auf die Frage: *"Wenn ich ein zufälliges positives und ein zufälliges negatives Beispiel nebeneinanderlege — wie wahrscheinlich gibt mein Modell dem positiven die höhere Wahrscheinlichkeit?"*

- AUC 0.5 = Zufall (Münzwurf).
- AUC 0.8 = ordentlich, klar besser als Zufall.
- AUC 0.95 = sehr gut.
- AUC 1.0 = perfekt.

#### AP (Average Precision)

Eine andere Sicht: *"Wenn ich meine Beispiele nach Modell-Score sortiere und die Top-K nehme — wie viele davon sind tatsächlich positiv?"* Aggregiert über alle möglichen K. Diese Metrik ist robuster als reine Accuracy bei stark unbalancierten Daten.

#### MRR@100 (Mean Reciprocal Rank)

Für jedes positive Test-Beispiel erzeugen wir 99 zufällige negative Beispiele zum gleichen Zeitpunkt und lassen das Modell alle 100 bewerten. Wir schauen, an welcher Position das positive Beispiel landet, wenn man alle nach Score sortiert.

- Rang 1 = perfekt, **reciprocal rank** = 1/1 = 1.0
- Rang 2 = sehr gut, RR = 1/2 = 0.5
- Rang 50 = mies, RR = 1/50 = 0.02

Der Durchschnitt über alle Test-Beispiele ist die **Mean Reciprocal Rank**. Höher ist besser. Ein guter Wert liegt im Bereich 0.3 - 0.7, ein Top-Wert über 0.8.

### Zelle 16 (`In[16]`): Lernkurve

Zeichnet zwei Plots:

- **Train Loss**: sollte über die Epochen kontinuierlich fallen.
- **Validation AUC und AP**: sollte über die Epochen steigen und am Ende stagnieren.

Falls Val-Metriken irgendwann fallen, während Train-Loss weiterfällt: **Overfitting** — das Modell lernt die Trainings-Daten auswendig, generalisiert aber nicht mehr. Gegenmaßnahme: weniger Epochen oder mehr Regularisierung.

## 6. Glossar der wichtigsten Begriffe

| Begriff | Bedeutung |
|---|---|
| **Adjazenzmatrix** | Quadratische Tabelle, die alle Verbindungen eines Graphen kompakt darstellt. Zeile *u*, Spalte *v* = Verbindungsgewicht zwischen Knoten *u* und *v*. |
| **AP (Average Precision)** | Metrik für Klassifikatoren: misst, wie gut die positiven Beispiele unter den Top-Scored Items landen. |
| **AUC (Area Under the Curve)** | Metrik für Klassifikatoren: Wahrscheinlichkeit, dass das Modell ein zufälliges positives Beispiel höher bewertet als ein zufälliges negatives. |
| **Backpropagation** | Algorithmus, der den Fehler vom Ausgang rückwärts durch das Netz schickt und so für jeden Parameter den passenden Gradienten berechnet. |
| **Batch** | Eine Untermenge der Trainingsdaten, die in einem Optimizer-Schritt verarbeitet wird. |
| **CNN (Convolutional Neural Network)** | Neuronales Netz mit Filtern, die wie Schiebefenster über die Eingabe gleiten. Ursprünglich für Bilder, funktioniert auch für Zeitreihen (dann "1D-CNN"). |
| **DataFrame** | Pandas-Datenstruktur für tabellarische Daten. Wie eine Excel-Tabelle, aber programmatisch. |
| **Dropout** | Anti-Overfitting-Trick: während des Trainings werden zufällig Neuronen-Aktivierungen auf 0 gesetzt. |
| **Edge / Kante** | Verbindung zwischen zwei Knoten in einem Graphen. |
| **Edge Weight** | Numerisches Gewicht einer Kante. In unserem Fall die Anzahl der historischen Fahrten. |
| **Embedding** | Ein erlernter Vektor, der eine Entität (Knoten, Wort, Bild) als Zahlenliste zusammenfasst. |
| **Epoch / Epoche** | Eine vollständige Runde durch den Trainings-Datensatz. |
| **Feature** | Eine Eingangsinformation für ein Modell. Auch: "Variable", "Attribut". |
| **Forward Fill** | Methode, um Lücken in Zeitreihen zu schließen, indem der letzte bekannte Wert fortgeschrieben wird. |
| **GCN (Graph Convolutional Network)** | Neuronales Netz, das Information zwischen benachbarten Knoten in einem Graphen austauscht und aggregiert. |
| **Gradient** | Vektor, der für jeden Parameter angibt, in welche Richtung er verschoben werden sollte, um den Loss zu verringern. |
| **GPU** | Grafikkarte. Beschleunigt Matrix-Operationen massiv, was neuronalen Netzen extrem hilft. |
| **Graph** | Mathematische Beschreibung eines Netzwerks aus Knoten und Kanten. |
| **Hyperparameter** | Stellschrauben, die der Mensch wählt und das Modell **nicht** selbst lernt (Lernrate, Anzahl Schichten, Batch-Größe etc.). |
| **JSON** | Dateiformat zum Speichern strukturierter Daten als Text. Lesbar für Menschen und Maschinen. |
| **Knoten / Node** | Einzelner Punkt in einem Graphen. Bei uns: eine Bike-Station. |
| **Layer** | Eine Schicht in einem neuronalen Netz. Mehrere Schichten hintereinander = "tiefes" Netz. |
| **Lernrate (Learning Rate)** | Wie stark das Modell pro Optimizer-Schritt seine Parameter anpasst. Zu hoch = instabil. Zu niedrig = lernt langsam. |
| **Link Prediction** | Aufgabe, vorherzusagen, ob (oder wann) zwischen zwei Knoten eine Kante entsteht. |
| **Logit** | Roh-Ausgabe eines Klassifikationsmodells, bevor sie per Sigmoid in eine Wahrscheinlichkeit umgerechnet wird. Wertebereich (-∞, +∞). |
| **Loss / Verlustfunktion** | Mathematische Funktion, die misst, wie falsch das Modell liegt. Das Training versucht, sie zu minimieren. |
| **Matrix** | Tabelle aus Zahlen. Zweidimensional. |
| **MLP (Multi-Layer Perceptron)** | Klassisches einfaches neuronales Netz aus mehreren vollständig verbundenen Schichten. |
| **MRR (Mean Reciprocal Rank)** | Metrik für Ranking-Aufgaben: durchschnittlich an welcher Stelle landet das richtige Element in der Sortier-Reihenfolge. |
| **Negative Sampling** | Trick, um in Klassifikationsaufgaben mit extremer Unbalance zu trainieren: zu jedem positiven Beispiel werden zufällig negative gezogen. |
| **Neuronales Netz** | Modell aus vielen kleinen, hintereinandergeschalteten Rechenfunktionen, die zusammen komplexe Zusammenhänge lernen können. |
| **One-Hot-Encoding** | Darstellung einer Kategorie als Vektor: für N mögliche Werte ein N-dimensionaler Vektor, in dem genau eine Stelle 1 ist, der Rest 0. |
| **Optimizer** | Algorithmus, der die Parameter eines neuronalen Netzes anhand der Gradienten aktualisiert (z.B. Adam, SGD). |
| **Overfitting** | Das Modell lernt die Trainingsdaten auswendig, generalisiert aber nicht auf neue Daten. |
| **Padding** | Auffüllen einer Sequenz oder eines Bildes mit Dummy-Werten, damit die Operationen am Rand sauber funktionieren. |
| **Pandas** | Python-Bibliothek für tabellarische Daten. |
| **Parameter** | Die internen Zahlen eines neuronalen Netzes, die im Training optimiert werden. Auch "Gewichte". |
| **PyTorch** | Open-Source-Bibliothek für neuronale Netze, entwickelt von Meta. |
| **Reproduzierbarkeit** | Ein Experiment liefert beim erneuten Ausführen dasselbe Ergebnis. |
| **ReLU (Rectified Linear Unit)** | Aktivierungsfunktion. `ReLU(x) = max(0, x)`. Bringt Nicht-Linearität ins Netz. |
| **Resampling** | Übertragen einer Zeitreihe auf ein anderes Zeit-Raster. |
| **Seed** | Startwert für Zufallsgeneratoren. Festgesetzt für reproduzierbare Ergebnisse. |
| **Self-Loop** | Kante von einem Knoten zu sich selbst. Bei GCN technisch nötig, damit Knoten ihre eigenen Features sehen. |
| **Sigmoid** | Mathematische Funktion, die Werte aus (-∞, +∞) in (0, 1) umrechnet. Wird benutzt, um Logits in Wahrscheinlichkeiten zu konvertieren. |
| **Streaming** | Daten Stück für Stück verarbeiten statt alles auf einmal zu laden. |
| **Tensor** | Verallgemeinerung von Matrix auf beliebig viele Dimensionen. PyTorchs Basis-Datentyp. |
| **Test-Set** | Teil der Daten, der nur am Ende einmal zur finalen Bewertung verwendet wird. |
| **Time Series / Zeitreihe** | Eine Folge von Messwerten mit Zeitstempel. |
| **Train-Set** | Teil der Daten, auf dem das Modell lernt. |
| **UUID** | Universally Unique Identifier. Lange Zeichenkette als eindeutige ID, z.B. `c1a4d909-0a00-475a-8e82-18ed13a4eb01`. |
| **Validation-Set** | Teil der Daten, auf dem während des Trainings die Performance geprüft wird. Hilft bei Modell-Auswahl, ohne den Test-Set zu verbrennen. |
| **Vektor** | Liste von Zahlen. Eindimensional. |
| **Weight Decay** | Form der Regularisierung: Gewichte werden bei jedem Schritt leicht in Richtung 0 gezogen. |
| **Z-Score** | Normalisierung: `(Wert − Mittelwert) / Standardabweichung`. Resultat: Mittelwert 0, Streuung 1. |

## 7. Wenn jemand das Notebook lesen will: Reihenfolge der Erklärung

Wenn du jemandem das Projekt vorstellen willst, hier eine sinnvolle didaktische Reihenfolge:

1. **Was wollen wir vorhersagen?** Wahrscheinlichkeit einer Fahrt zwischen zwei Stationen im nächsten 30-Minuten-Fenster.
2. **Welche Daten haben wir?** Ein Graph aus Stationen plus pro Station eine Verfügbarkeits-Zeitreihe plus pro Stationspaar eine kumulative Trip-Zeitreihe.
3. **Was ist die Schwierigkeit?** Klassen-Unbalance (selten kommt es zu Fahrten zwischen jedem beliebigen Paar) und gemischte Datentypen (Graph plus Zeitreihen).
4. **Wie modellieren wir das?** Zwei spezialisierte neuronale Netze (eines für den Graphen, eines für die Zeitreihen), deren Ausgaben in einem Fusion-MLP zusammenkommen.
5. **Wie evaluieren wir?** Drei Metriken (AUC, AP, MRR), temporaler Train/Val/Test-Split, Negative Sampling.
6. **Was ist der Stand?** Erste End-to-End-Iteration, Default-Wahl der Komponenten, später Ablations und Baselines.

Wer sich vertiefen will, geht von dort in die spezifischen Notebook-Sektionen.
