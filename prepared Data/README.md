# Aufbereitete Daten – Temporal Graph (GraphMixer-Baseline)

Dieser Ordner enthält den **temporalen Graphen** der NYC/JC-Bike-Sharing-Fahrten,
aufbereitet als Eingabe für die GraphMixer-Baseline und für die gemeinsame
Evaluation. Erzeugt aus den Citi-Bike-Rohfahrten (Jersey City/Hoboken,
Mai + Juni 2024), gefiltert auf den **aktiven 232-Knoten-Teilgraphen** des
Superedge-Datensatzes.

## Herkunft

- **Rohquelle**: Citi Bike System Data, Dateien `JC-202405-citibike-tripdata.csv`
  und `JC-202406-citibike-tripdata.csv` (Jersey City/Hoboken-Teilsystem).
- **Filter**: nur Fahrten, deren Start- und Zielstation zu den 232 aktiven
  Stationen des Superedge-Datensatzes gehören (Abgleich über `short_name` =
  Citi-Bike-`station_id`, 232/232 = 100 % Übereinstimmung).
- **Beobachtungsfenster**: 2024-05-16 bis 2024-06-14 (≈ 29 Tage), identisch zum
  Superedge-Datensatz (Zenodo DOI 10.5281/zenodo.13846868).
- **Sanity-Check**: 104.681 gefilterte Trips vs. 102.594 Superedge-Trips; alle
  5.626 Superedge-Paare vorhanden, Median-Abweichung pro Paar = 0 (Randeffekte
  des Fensters erklären die ~2 % Differenz).

## Dateien

| Datei | Inhalt | genutzt von |
|---|---|---|
| `superedge_counts.csv` | **Ground-Truth-Quelle**: aggregierte Superedge-`num_rides` je 30-min-Bin `u, i, bin_idx, count` (= Δnum_rides). | `shared_eval` (Targets/Count-GT, alle Modelle), **LSTM** (Eingabe-Zeitreihe) |
| `graphmixer_edges.csv` | Einzelfahrten (temporaler Graph): `u, i, ts, ts_iso, rideable_type, member_casual` | Bau von `ml_citibike.*` |
| `node_index.csv` | Kanonisches Knoten-Mapping: `idx` (0…231) ↔ `station_id` ↔ `name` | alle |
| `node_static.npy` | statische Knoten-Features `(232, 3)`: capacity, lat, lon | **Hybrid** (GraphSAGE) |
| `node_avail.npy` | Verfügbarkeits-Zeitreihen `(232, T, 4)` je 30-min-Bin | **Hybrid** (GRU) |
| `edge_index.npy` / `edge_weight.npy` | Adjazenz (gerichtet) + Gewicht = Superedge-`num_rides` im Training | **Hybrid** (GraphSAGE) |
| `ml_citibike.csv` | DyGLib-Kantenliste: `u, i, ts, label, idx` (**Knoten 1-indiziert** 1…232, homogen) | **GraphMixer** |
| `ml_citibike.npy` | Kanten-Features `(n+1, 4)`: One-Hot `classic, electric, member, casual`; Zeile 0 = Padding | **GraphMixer** |
| `ml_citibike_node.npy` | Knoten-Features `(233, 3)`: `capacity, lat, lon` (z-normalisiert); Zeile 0 = Padding | **GraphMixer** |

## Wichtige Konventionen

- **Kanonische Knoten-ID** = `idx` aus `node_index.csv` (**0-indiziert**, 0…231).
  Wird vom Evaluationsmodul und von `graphmixer_edges.csv` verwendet.
- **DyGLib/GraphMixer-Dateien** (`ml_*`) sind **1-indiziert** (kanonische ID + 1),
  Zeile 0 ist Padding. Vor der Bewertung GraphMixer-Vorhersagen um **−1**
  zurückrechnen, damit sie zur kanonischen ID passen.
- **`ts`**: Sekunden seit Fensterbeginn (2024-05-16), chronologisch sortiert.
- **Richtung**: Kanten sind gerichtet (`u` = Start, `i` = Ziel).

## Reproduktion

Erzeugt durch die Skripte (gegen den externen Datenpfad):
1. `build_graphmixer.py` → `graphmixer_edges.csv`, `node_index.csv` (+ Sanity-Check)
2. `to_dyglib.py` → `ml_citibike.csv`, `ml_citibike.npy`, `ml_citibike_node.npy`
