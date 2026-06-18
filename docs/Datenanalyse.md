---
tags: [projekt, datenanalyse, link-prediction]
status: aktiv
erstellt: 2026-05-12
projekt: Link Prediction on Hybrid Graph + Time Series Data
---

# Datenanalyse: NYC Bike Sharing Network

Vollständige Datenexploration der beiden JSON-Dateien aus dem Zenodo-Record [10.5281/zenodo.13846868](https://zenodo.org/records/13846868). Diese Notiz dokumentiert die Schemata und die wichtigsten statistischen Eigenschaften. Aufgabenstellung und Methodendesign liegen in [[Link Prediction on Hybrid Graph + Time Series Data.md|Hauptdatei des Projekts]].

## Methodik

Die Analyse wurde direkt auf den vollständigen Dateien ausgeführt, lokal abgelegt unter `C:\Users\user\Data\nyc-bike-sharing\` (außerhalb des Vaults). `graph_edges.json` wurde komplett in den Speicher geladen, `graph_nodes.json` wurde wegen seiner Größe mit `ijson` gestreamt. Alle Zahlen sind reproduzierbar mit dem Skript am Ende dieser Notiz.

## Überblick

| Kennzahl | Wert |
|---|---|
| Anzahl Knoten (Stationen) | **2.213** |
| Anzahl gerichteter Kanten (super_edges) | **5.626** |
| Beobachtungszeitraum | 2024-05-16 bis 2024-06-14 (ca. 4 Wochen) |
| Knoten mit identischem Beobachtungsfenster | 2.213 / 2.213 (100%) |
| Knoten-Zeitreihen-Punkte insgesamt | **7.521.171** |
| Trips insgesamt (Summe letzter num_rides-Werte) | **102.594** |

## 1. Knoten-Schema (`graph_nodes.json`)

Jedes Element ist eine Citi-Bike-Station.

### Statische Felder pro Knoten

| Feld | Typ | Beschreibung |
|---|---|---|
| `station_id` | UUID | Primärschlüssel, identisch mit `nodeid` |
| `nodeid` | UUID | Duplikat von `station_id`, wird als Graph-Key verwendet |
| `name` | string | Klartext-Stationsname (z.B. "Whitehall St & Bridge St") |
| `short_name` | string | Interner Kurzcode (z.B. "4962.02") |
| `region_id` | string \| null | **Nicht konstant.** Verteilung: `"71"` 1.740, `null` 393, `"70"` 53, `"311"` 27 |
| `capacity` | int | Anzahl Docks. Beobachteter Bereich **1..123**, Median 24, Mittelwert 31.2 |
| `lat`, `lon` | float | Geokoordinaten |
| `start`, `end` | ISO datetime | Beobachtungsfenster, **konstant für alle Knoten** (`2024-05-16T00:00:00` bis `2024-06-14T00:00:00`) |
| `labels` | list[string] | Konstant `["station"]` für alle Knoten |

> **Korrektur zu einer früheren Annahme**: `region_id` ist nicht konstant. Es gibt vier verschiedene Werte plus `null`. Beim Preprocessing als kategorisches Feature behandeln und Missing-Values explizit kodieren.

### Dynamische Felder — `ts`-Block

Jeder Knoten trägt **vier** Zeitreihen-Serien:

| Serie                  | Bedeutung                                                    |
| ---------------------- | ------------------------------------------------------------ |
| `num_bikes_available`  | Aktuell andockfähige, verleihbare Räder                      |
| `num_ebikes_available` | Teilmenge der verfügbaren Räder, die E-Bikes sind            |
| `num_bikes_disabled`   | Räder physisch an der Station, aber als unbenutzbar markiert |
| `num_docks_disabled`   | Docks, die als defekt markiert sind                          |

Format: Liste von `{Start: ISO_datetime, Value: int}`-Records.

### Sampling-Semantik: Change-Point-Kompression

Der Publisher (Zenodo-Beschreibung) bestätigt: der GBFS-Station-Status-Feed wird alle 5 Minuten gepollt, ein Record wird aber **nur dann geschrieben, wenn sich mindestens einer der vier Zähler seit dem letzten Poll geändert hat**. Das ist Absicht, kein Datenqualitätsproblem.

Eigene Messungen bestätigen die 5-Minuten-Cadence:

- Minimales Inter-Event-Delta: **294 s** (≈ 5 min), entspricht der Poll-Frequenz.
- Median-Inter-Event-Delta über alle Serien und alle Knoten: **600 s (10 min)**.
- Pro Serie:
  - `num_bikes_available`: Median 9,9 min (n=3.917.759 Events)
  - `num_ebikes_available`: Median 10,0 min (n=3.043.035 Events)
  - `num_bikes_disabled`: Median 24,9 min (n=520.308 Events) — seltener, weil Defekte selten passieren
  - `num_docks_disabled`: Median 10,1 min (n=31.217 Events) — sehr selten in absoluter Häufigkeit

> **Konsequenz fürs Modell**: Um die Zeitreihen als reguläre Time Series zu nutzen, auf ein festes Raster resamplen (z.B. 5-Minuten-Bins per Forward Fill). Der "aktuelle Wert" zum Zeitpunkt t ist der jüngste Record mit `Start ≤ t`.

### Längen-Verteilung der Knoten-Zeitreihen

| Statistik | Wert |
|---|---|
| Total ts-Points über alle Knoten | 7.521.171 |
| Min Länge pro Serie | 1 |
| Median Länge | 359 |
| Mean Länge | 849,7 |
| Max Länge | 5.407 |

Die Streuung ist groß: aktive Stationen haben mehrere Tausend Events, schlafende Stationen fast keine. Das wirkt sich auf die Train/Val-Split-Strategie aus (siehe unten).

## 2. Kanten-Schema (`graph_edges.json`)

Jedes Element ist eine gerichtete `super_edge` zwischen zwei Stationen und aggregiert **alle Trips** im Beobachtungsfenster von `from` nach `to`.

### Statische Felder pro Kante

| Feld | Typ | Beschreibung |
|---|---|---|
| `from` | UUID | Quell-Station (verweist auf `station_id`) |
| `to` | UUID | Ziel-Station (verweist auf `station_id`) |
| `label` | string | Konstant `"super_edge"` für alle Kanten |
| `start` | ISO datetime | Zeitpunkt der ersten Fahrt auf dieser Kante |
| `end` | ISO datetime | Globales Fenster-Ende (`2024-06-14T00:00:00`) |

### Dynamische Felder — `ts`-Block

Jede Kante trägt **sechs** Zeitreihen:

| Serie            | Bedeutung                                         | Monoton?             |
| ---------------- | ------------------------------------------------- | -------------------- |
| `num_rides`      | Kumulative Gesamtzahl aller Fahrten auf der Kante | ja, nicht-fallend    |
| `classic_rides`  | Kumulativ, Fahrten mit klassischem Rad            | ja, nicht-fallend    |
| `electric_rides` | Kumulativ, Fahrten mit E-Bike                     | ja, nicht-fallend    |
| `member_rides`   | Kumulativ, Fahrten von Member-Accounts            | ja, nicht-fallend    |
| `casual_rides`   | Kumulativ, Fahrten von Casual-Accounts            | ja, nicht-fallend    |
| `active_trips`   | Aktuell laufende Trips auf dieser Kante           | **nein, oszilliert** |

> **Wichtig**: Die fünf `_rides`-Serien sind **kumulative Zähler**. Pro Event wird um genau 1 hochgezählt. Die Identität `num_rides = classic_rides + electric_rides = member_rides + casual_rides` lässt sich vermutlich zur Datenvalidierung nutzen.
>
> `active_trips` ist ein **Inventur-Stand**, kein Zähler — der Wert geht bei Start eines Trips um 1 hoch und bei Ende um 1 runter. Beobachtetes Wertebereich-Sample über 500 Kanten: `[0, 85]`.

### Counter-Semantik für das Prediction Target

Da `num_rides` kumulativ ist, ergibt sich das pro-Fenster-Volumen einer Kante über die Differenz:

```
rides_in_window(u, v, t, Δ) = num_rides[u→v] @ (t+Δ)  −  num_rides[u→v] @ t
```

Das binäre Target laut Aufgabenstellung wird daraus: `Label = 1` wenn `rides_in_window > 0`, sonst `0`.

### Längen-Verteilung pro Kante (basierend auf `num_rides`)

| Statistik | Wert |
|---|---|
| Total Trips (Summe letzter Counter-Werte) | 102.594 |
| Min Trips pro Kante | 1 |
| Median Trips pro Kante | 6 |
| Mean Trips pro Kante | 18,2 |
| Max Trips pro Kante | 468 |
| Kanten mit < 5 Trips | 2.375 (**42,2 %**) |

Knapp die Hälfte aller Kanten ist sehr schwach besetzt. Das ist ein deutlicher Hinweis auf **Sparsity und Long-Tail**: wenige Kanten dominieren das Trip-Volumen, der Rest ist Rauschen oder seltene Verbindungen.

### Inter-Event-Spacing auf `num_rides`

| Stichprobe | Median | Mean |
|---|---|---|
| Alle Kanten, alle Events (n=96.968) | **5,81 h** | 22,9 h |
| Nur Kanten mit ≥ 20 Trips (n=77.807) | **3,50 h** | 11,3 h |

Sehr long-tailed: Mittelwerte liegen weit über den Medianen, weil inaktive Kanten und Nacht-Pausen die Verteilung nach oben ziehen. Aktive Kanten haben einen Median von 3,5 h zwischen zwei Trips.

## 3. Implikationen für Methodik und Modellierung

### Preprocessing

- **Knoten-ts auf festes Raster resamplen** (z.B. 5- oder 15-min Bins per Forward Fill).
- **Kanten-ts: Differenzen statt Kumulativwerte** für ein Fenster-Volumen berechnen.
- **Stations-Identität** über `station_id` (oder gleichwertig `nodeid`).
- **`region_id`-Missing-Handling** explizit kodieren; nicht als zweite "71"-Kategorie missinterpretieren.

### Targets

- **Binär**: `Δnum_rides > 0` im Zukunftsfenster pro (u, v).
- **Regression** wäre ebenfalls möglich (`Δnum_rides` direkt), aber von der Aufgabenstellung nicht gefordert.

### Class Imbalance und Negative Sampling

42,2 % der Kanten haben < 5 Trips im Gesamtzeitraum. Auf einem Fenster von z.B. 30 Minuten gibt es pro `(u, v, t)` fast immer 0 Trips → Klassen-Imbalance ist extrem. Klassische Lösung: **Negative Sampling** (z.B. 1 positives Sample : k negative Samples) plus AP / MRR statt nur Accuracy.

### Train/Val/Test-Split

- **Temporal**: erste 3 Wochen Training, vorletzte Woche Validation, letzte Woche Test. Random-Split wäre methodisch falsch (Leakage durch zeitliche Korrelation).
- **Beobachtung**: Stationen mit kurzen Zeitreihen (length=1 oder sehr klein) eventuell ausschließen, da unter Information-Recovery-Threshold.

### Feature-Vorschlag

- **Statische Knoten-Features**: `capacity`, `lat`, `lon`, `region_id` (one-hot), optional abgeleitete räumliche Features (Distanz zum Schwerpunkt, Cluster-ID).
- **Dynamische Knoten-Features**: Kurz-Fenster (z.B. letzte 30 Minuten) der vier ts-Serien, z.B. Mittelwert, Std, Trend, plus Tageszeit / Wochentag-Feature.
- **Edge-Features**: historische Trip-Frequenz (`num_rides` total über bekanntes Fenster), Bike-Typ-Verhältnis (`electric_rides / num_rides`), Rider-Typ-Verhältnis (`member_rides / num_rides`).

### Kapazität vs. Trip-Volumen: geografische Konfundierung

Vollständige Analyse auf den 232 aktiven Stationen ergibt eine **negative** Korrelation zwischen Stationskapazität und Trip-Volumen:

| Maß | Wert |
|---|---|
| Pearson r | −0,44 |
| Spearman r | −0,58 |

Dieser auf den ersten Blick kontraintuitive Befund ist **nicht kausal**, sondern durch den geografischen Standort konfundiert. Der aktive Teilgraph enthält zwei räumlich getrennte Populationen:

- **Kleine Stationen (11–30 Docks) in Hoboken / Jersey City**: hochaktiv, Trips/Dock-Median ~90.
- **Große Stationen (61–123 Docks) aus Manhattan**: nahezu inaktiv, Trips/Dock-Median ~0.

Die Top-5-Stationen nach Trips-pro-Dock (Newport Pkwy: 293,8; Newport PATH: 273,6; Hoboken Terminal: 216,9) bestätigen das Muster. Gleichzeitig haben die größten Stationen (E 40 St & Park Ave: 123 Docks, 11 Trips; West St & Chambers St: 115 Docks, 8 Trips) trotz enormer Kapazität praktisch keine Aktivität.

**Konsequenz für Feature-Engineering**: `capacity` allein ist kein zuverlässiger Aktivitäts-Indikator. `lat`/`lon` trennen die beiden Cluster sauber und sind der stärkere Prädiktor. Im GCN-Branch spielt dies keine direkte Rolle — die Adjazenzmatrix mit historischen Trip-Frequenzen als Edge-Weights bildet die Cluster-Struktur von selbst ab.

## 4. Offene Fragen für die nächste Iteration

- Stimmt die Identität `num_rides == classic_rides + electric_rides == member_rides + casual_rides` für alle Edges? → Schnelle Validierung möglich.
- Wie verteilt sich `active_trips` über die Zeit (Tageszeit-Muster, Wochentag-Muster)? Könnte als zusätzliches Knoten-Feature dienen.
- Reicht das 4-Wochen-Fenster für eine valide Validation/Test-Trennung, oder droht zu wenig Daten in der Test-Periode?
- Wie sieht das räumliche Cluster-Muster aus (Manhattan, Brooklyn, Queens, …)?

## 5. Reproduzierbarkeit

Voraussetzungen:

```bash
pip install ijson
```

Datensatz lokal:

```bash
mkdir -p ~/Data/nyc-bike-sharing
cd ~/Data/nyc-bike-sharing
curl -L -o graph_edges.json "https://zenodo.org/records/13846868/files/graph_edges.json?download=1"
curl -L -o graph_nodes.json "https://zenodo.org/records/13846868/files/graph_nodes.json?download=1"
```

Skript-Skelett:

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

Die vollständigen Analyse-Skripte (Strukturchecks, Längen-Verteilungen, Inter-Event-Deltas) werden ins Projekt-Code-Repo aufgenommen, sobald dieses angelegt ist.
