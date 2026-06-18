# Bike-Sharing Link Prediction on Hybrid Graph + Time Series

Hybride Link-Vorhersage auf dem NYC/Jersey-City-Bike-Sharing-Netz: Sage für ein
Stationspaar in einem zukünftigen Zeitfenster vorher, **ob** eine Fahrt
stattfindet (binär) und **wie viele** Fahrten (count). Kombiniert Graphstruktur
mit kontinuierlichen Knoten-Zeitreihen (Verfügbarkeit).

Big-Data-Praktikum, Universität Leipzig.

## Aufgabe & Vergleiche

- **Eigenes Hybridmodell**: Graph-Branch (GCN/GraphSAGE) + Zeitreihen-Branch
  (1D-CNN/GRU) + Fusion; zwei Ausgaben (binär + count).
- **Vergleich 1 (binär)**: eigenes Modell vs. **GraphMixer** (Temporal-Graph-Baseline) — AUC, AP.
- **Vergleich 2 (count)**: eigenes Modell vs. **LSTM** (Zeitreihen-Baseline) — MSE, MAE.
- Ground Truth des Counts = Differenz der kumulativen `num_rides`-Zeitreihe.

## Struktur

```
.
├── evaluation/
│   └── shared_eval.py          # MODELL-AGNOSTISCHES Eval (binär + count), eine GT/ein Split für alle
├── graphmixer/
│   ├── prepared/               # aufbereitete Eingaben (klein) + README
│   └── model/                  # GraphMixer (PyTorch) + Colab-Runner
├── lstm/                       # LSTM-Baseline (count) + Colab-Runner
├── hybrid_model/
│   └── iteration1_gcn_cnn_fusion.ipynb   # eigenes Modell, Iteration 1
└── docs/                       # Konzept (DE/EN), Datenanalyse, Methoden-Bewertung, Erklärungen
```

## Daten

Die **aufbereiteten** Dateien liegen in `graphmixer/prepared/` (siehe das dortige
README für Schema und Konventionen). Die **Rohdaten** sind absichtlich NICHT im
Repo (zu groß), aber öffentlich reproduzierbar:

- Hybrid-Datensatz (Superedge, Zeitreihen): Zenodo DOI `10.5281/zenodo.13846868`
- Temporaler Graph (Einzelfahrten): Citi Bike System Data, Dateien
  `JC-202405-citibike-tripdata.csv`, `JC-202406-citibike-tripdata.csv`
  (Jersey City/Hoboken, Mai+Juni 2024), gefiltert auf die 232 aktiven Stationen.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

## GraphMixer-Baseline ausführen

```bash
cd graphmixer/model
python train_graphmixer.py                 # nutzt GPU falls vorhanden, sonst CPU
```
Schneller Smoke-Test: in `train_graphmixer.py` `main(GMConfig(epochs=2))`.
Ergebnis: `graphmixer/model/predictions/graphmixer_pred_{val,test}.csv` + AUC/AP/F1.
Alternativ Colab: `graphmixer/model/run_graphmixer.ipynb` (Pfade anpassen).

## Bewertung (für alle Modelle gleich)

Jedes Modell exportiert Vorhersagen mit den Spalten `u, i, bin_idx` plus
`score` (binär) und/oder `pred_count` (count), Knoten-IDs **kanonisch 0-indiziert**.
Dann:
```python
from evaluation.shared_eval import SharedLinkEval
ev = SharedLinkEval()
ev.score_binary(pred_df, split="test")   # AUC, AP, F1, Acc
ev.score_count(pred_df,  split="test")   # MSE, MAE, RMSE
```
Das garantiert identische Ground Truth, Splits und Kandidatenpaare über alle
Verfahren hinweg.

## Stand

- [x] Datenaufbereitung + Sanity-Check
- [x] Gemeinsames Evaluationsmodul
- [x] GraphMixer-Baseline (Code + Colab-Runner)
- [x] LSTM-Baseline (count) (Code + Colab-Runner)
- [ ] Hybridmodell mit Count-Kopf + Anbindung an `shared_eval`
- [ ] Finale Vergleichstabellen
