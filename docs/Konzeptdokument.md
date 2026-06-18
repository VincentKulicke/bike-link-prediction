---
tags: [projekt, konzept, link-prediction]
status: aktiv
erstellt: 2026-05-28
projekt: Link Prediction on Hybrid Graph + Time Series Data
---

# Hybride GNN-basierte Link Prediction für Bike-Sharing-Netzwerke

**Autoren**

| Name | E-Mail |
|---|---|
| Moritz Rau | wx60iroz@studserv.uni-leipzig.de |
| Vincent Kulicke | ok09ined@studserv.uni-leipzig.de |
| Henning Hesselbarth | hh95copi@studserv.uni-leipzig.de |

**Inhalt**

- Einleitung
- Theoretische Grundlagen
  - Graph Neural Networks
  - Evaluationsmetriken
- Datensatz
  - Knotenstruktur
  - Kantenstruktur
  - Datensatzstatistik und Konsequenzen für die Datenvorbereitung
- Vorgeschlagene Methode
  - Modellarchitektur
  - Evaluation
- Referenzen

---

## Einleitung

Bike-Sharing-Systeme erzeugen sowohl graphstrukturierte als auch zeitliche Daten. Stationen lassen sich als Knoten darstellen, die über Fahrten verbunden sind, während zusätzliche Zeitreihen wie die Fahrrad-Verfügbarkeit den dynamischen Zustand des Netzwerks beschreiben. Die Vorhersage zukünftiger Fahrten zwischen Stationen ist wichtig für Anwendungen wie Nachfrageprognose und Fahrrad-Rebalancing.

Bestehende Verfahren zur temporalen Link Prediction konzentrieren sich überwiegend auf Event-Streams und ignorieren kontinuierliche Messungen auf Knotenebene häufig. Umgekehrt verarbeiten viele spatiotemporale Ansätze zwar Zeitreihendaten, nutzen die Graphstruktur und Knotenattribute aber nur begrenzt. Aktuelle Methoden sind daher für hybride Graph- und Zeitreihen-Datensätze nicht vollständig geeignet.

Dieses Projekt schlägt ein hybrides Link-Prediction-Framework vor, das Graph Neural Networks (GraphSAGE) mit temporalen Encodern wie GRUs kombiniert. Das Modell lernt strukturelle und temporale Repräsentationen der Bike-Stationen gemeinsam, um vorherzusagen, ob in einem zukünftigen Zeitfenster Fahrten zwischen Stationspaaren auftreten. Der Ansatz wird auf dem NYC Bike Sharing Network Datensatz evaluiert und anhand gängiger Link-Prediction-Metriken (AUC, AP und MRR) mit bestehenden Baseline-Methoden verglichen.

## Theoretische Grundlagen

### Graph Neural Networks

Ein Graph ist formal definiert als G = (V, E, F, E), wobei V eine Menge von N Knoten, E eine Menge von Kanten und F sowie E die Knoten- bzw. Kantenattribute repräsentieren (Guo et al., 2022). Graph Neural Networks (GNNs) sind darauf ausgelegt, Daten in solchen graphbasierten Strukturen zu verarbeiten, die flexibler sind als klassische Gitter oder Sequenzen (Chen et al., 2021). Der Kernmechanismus der meisten GNNs folgt dem Message-Passing-Framework (MP), bei dem Knotenrepräsentationen durch Aggregation von Informationen aus ihrer lokalen Nachbarschaft aktualisiert werden. In spatiotemporalen Kontexten modellieren GNNs Abhängigkeiten als paarweise Beziehungen zwischen Zeitreihen, wobei jede Zeitreihe einem Knoten zugeordnet ist und funktionale Beziehungen als Kanten dargestellt werden (Cini et al., 2025).

GraphSAGE ist ein allgemeines induktives Framework, das Knoten-Feature-Informationen nutzt, um Embeddings für zuvor ungesehene Daten zu erzeugen. Anders als transduktive Verfahren, die individuelle Embeddings pro Knoten trainieren, lernt GraphSAGE eine Menge von Aggregator-Funktionen, die Features aus der lokalen Nachbarschaft eines Knotens sampeln und aggregieren. Aus Effizienzgründen sampelt GraphSAGE eine fixe Anzahl von Nachbarn gleichverteilt, statt die vollständige Nachbarschaft zu nutzen (Will et al., 2017).

Die Gated Recurrent Unit (GRU) ist eine Variante des Recurrent Neural Network (RNN), die temporale Informationen erfasst und langfristige Abhängigkeiten handhabt, während sie das Problem verschwindender Gradienten abmildert. Eine GRU-Zelle verwendet ein Update-Gate, um zu bestimmen, wie viel der vergangenen Information erhalten bleibt, und ein Reset-Gate, um zu entscheiden, wie viel vergessen wird (Kontopoulos et al., 2023). In der graphbasierten Prognose werden GRUs häufig mit Graph-Convolutions kombiniert und bilden so Graph Convolutional Recurrent Neural Networks (GCRNNs) oder Diffusion Convolutional GRUs (DCGRU). In diesen Architekturen werden die Standard-Matrixmultiplikationen innerhalb der GRU-Gates durch graph-konvolutionale Operatoren ersetzt, sodass das Modell räumliche und zeitliche Muster gleichzeitig erfassen kann (Cini et al., 2025; Guo et al., 2022).

### Evaluationsmetriken

Die folgende Tabelle gibt einen umfassenden Überblick über die drei Evaluationsmetriken Area Under the Curve (AUC), Average Precision (AP) und Mean Reciprocal Rank (MRR), die häufig zur Bewertung von Machine-Learning-Modellen in Klassifikations- und Ranking-Aufgaben verwendet werden. Diese Maße gehen über Standardmetriken wie Accuracy hinaus, die bei unausgewogenen Datensätzen oder wenn die konkrete Reihenfolge der Ergebnisse entscheidend ist, oft irreführend sein können (Beddar-Wiesing et al., 2025).

| Metrik | Definition | Formel |
|---|---|---|
| Area Under the Curve (AUC) | Eine grafische Metrik, die die Leistung eines Modells über alle Entscheidungsschwellen hinweg bewertet, indem sie die Fläche unter einer Leistungskurve berechnet. | Fläche unter der Kurve (z. B. das Integral der ROC-Kurve, die TPR gegen FPR aufträgt) |
| Average Precision (AP) | Eine Metrik für Ranking- und Empfehlungsqualität, die den gewichteten Mittelwert der Präzisionen an jeder Schwelle berechnet, wobei das Gewicht der Recall-Zuwachs gegenüber der vorherigen Schwelle ist. | AP = Σ_n (R_n − R_n−1) · P_n  (wobei P_n und R_n Präzision und Recall an der n-ten Schwelle sind) |
| Mean Reciprocal Rank (MRR) | Ein Maß für Aufgaben, bei denen nur das erste relevante Ergebnis von primärem Interesse ist. Es berechnet den Durchschnitt der reziproken Ränge der ersten korrekten Antwort über eine Stichprobe von Anfragen. | MRR = (1 / \|D\|) · Σ_x∈D (1 / k_x)  (\|D\|: Anzahl der Anfragen, k_x: Rang des ersten relevanten Elements für Anfrage x) |

## Datensatz

Der Datensatz ist das NYC Citi Bike Sharing Network (Constantin Urbainsky & Lyft Bikes & Scooters, 2024) und umfasst vier Wochen vom 16. Mai bis 14. Juni 2024, bereitgestellt als graph_nodes.json (530,6 MB, 2.213 Stationen) und graph_edges.json (26,5 MB, 5.626 gerichtete Kanten).

### Knotenstruktur

Jeder Stationsknoten kombiniert statische Attribute (station_id, capacity, lat/lon, region_id) mit vier kontinuierlichen Verfügbarkeits-Zeitreihen unter ts (num_bikes_available, num_ebikes_available, num_bikes_disabled, num_docks_disabled). Die Reihen verwenden Change-Point-Kompression — ein neuer Eintrag wird nur geschrieben, wenn sich ein Zähler ändert — was zu unregelmäßigen Inter-Event-Abständen führt (Median ≈ 10 min, Minimum 294 s, insgesamt 7,5 Mio. Ereignisse).

### Kantenstruktur

Jede der 5.626 gerichteten Kanten (Super-Edge) trägt zusätzlich zu den statischen Attributen „from" (Startstation) und „to" (Zielstation) sechs Zeitreihen-Attribute:

| Reihe | Bedeutung | Monoton? |
|---|---|---|
| num_rides | Kumulative Gesamtzahl der Fahrten | Ja |
| classic_rides / electric_rides | Kumulativ nach Fahrradtyp | Ja |
| member_rides / casual_rides | Kumulativ nach Nutzertyp | Ja |
| active_trips | Aktuell laufende Fahrten | Nein — oszilliert |

Das Fahrtvolumen für ein beliebiges Fenster [t, t+Δ] und das binäre Vorhersageziel ergeben sich aus der Differenzbildung des kumulativen Zählers:

> rides(u, v, t, Δ) = num_rides(t+Δ) − num_rides(t)
>
> y(u, v, t) = 1, falls rides(u, v, t, Δ) > 0, sonst 0

### Datensatzstatistik und Konsequenzen für die Datenvorbereitung

| Kennzahl | Wert | Konsequenz für Vorverarbeitung / Modellierung |
|---|---|---|
| Stationen gesamt | 2.213 | — |
| Aktive Stationen (≥ 1 Kante) | 232 | 1.981 isolierte Stationen vor allen Schritten herausfiltern |
| Gerichtete Kanten | 5.626 | — |
| Fahrten gesamt | 102.594 | — |
| Kanten mit < 5 Fahrten | 2.375 (42,2 %) | Starkes Klassenungleichgewicht → optionales Negative Sampling (1:5 Training, 1:99 Evaluation) |

| Kennzahl | Vor Filterung | Nach Filterung |
|---|---|---|
| Stationen | 2.213 | 232 |
| Kanten | 5.626 | 5.626 |
| Mittlerer Gesamtgrad | 5,1 | 48,5 |
| Nutzbarer Negativraum | ca. 460.000 Paare | ca. 48.000 Paare |

**1. Isolierte Stationen filtern.** Alle 1.981 Stationen ohne Kanten werden entfernt. Nach der Filterung bleiben 232 aktive Stationen, der mittlere Grad steigt von 5,1 auf 48,5, und der nutzbare Negativ-Paarraum schrumpft von ≈ 460.000 auf ≈ 48.000 Paare.

**2. Resampling der Knoten-Zeitreihen.** Die change-point-komprimierten Verfügbarkeits-Zeitreihen werden auf ein gleichmäßiges 5-Minuten-Raster resampled. Die Resampling-Strategie ist Forward Fill (letzte Beobachtung wird fortgeschrieben): Der Wert einer Reihe zum Zeitpunkt t ist das jüngste Ereignis mit einem Zeitstempel kleiner oder gleich t.

**3. Vorhersageziel ableiten.** Für jede aktive Kante (u, v) und jeden 5-Minuten-Zeitstempel ist label = 1, falls num_rides innerhalb von [t, t+30 min] steigt, sonst 0. Der 30-Minuten-Horizont (6 Bins, konfigurierbar) liefert ≈ 1.008 Snapshots pro Kante über den dreiwöchigen Trainingszeitraum, also rund 96.000 positive Samples vor dem Negative Sampling.

**4. Temporaler Split.** Der Datensatz wird strikt nach Zeit aufgeteilt, um Leakage zu verhindern.

| Split | Zeitraum | Dauer |
|---|---|---|
| Training | 16. Mai – 5. Juni 2024 | 21 Tage |
| Validierung | 6. – 9. Juni 2024 | 4 Tage |
| Test | 10. – 14. Juni 2024 | 5 Tage |

Die Adjazenzmatrix und alle Kantengewichte werden ausschließlich aus dem Trainingszeitraum berechnet und bleiben während Validierung und Test eingefroren.

**5. Negative Sampling (optional).** Jedes positive Sample wird mit 5 zufällig gezogenen Negativpaaren kombiniert (inaktive Stationspaare im selben Zeit-Bin). Für die Evaluation werden pro positivem Sample 99 Negative gezogen, um das 1-vs-99-Ranking-Problem zu konstruieren.

## Vorgeschlagene Methode

### Modellarchitektur

Das vorgeschlagene Modell ist eine Zwei-Branch-Architektur, die strukturelle und temporale Repräsentationen der Bike-Stationen gemeinsam lernt, um vorherzusagen, ob innerhalb der nächsten 30 Minuten eine Fahrt zwischen zwei Stationen auftritt.

Der **Graph-Branch** wendet GraphSAGE auf den gerichteten Stationsgraphen an und nutzt statische Stationsattribute (Kapazität, Koordinaten, Region) als Knoten-Features sowie historische Fahrtfrequenzen als Kantengewichte. Über zwei Message-Passing-Layer aggregiert jede Station Informationen aus ihren direkten Nachbarn und erzeugt ein strukturelles Embedding, das die Rolle und Konnektivität der Station im Netzwerk erfasst.

Der **Zeitreihen-Branch** kodiert die jüngste Verfügbarkeitsdynamik jeder Station mithilfe einer GRU. Unabhängig auf Start- und Zielstation mit geteilten Gewichten angewandt, nimmt er die letzten 30 Minuten der Verfügbarkeitsmessungen (num_bikes_available, num_ebikes_available, num_bikes_disabled, num_docks_disabled) als Eingabe und erzeugt eine kompakte Repräsentation des aktuellen Betriebszustands.

Für ein Kandidaten-Stationspaar werden die strukturellen und temporalen Embeddings beider Stationen mit paarbezogenen Features verkettet — geografische Distanz, historische Fahrtfrequenz des Paares sowie zyklische Zeitkodierungen (Stunde des Tages, Wochentag) — und durch ein dreischichtiges MLP geleitet, das die Link-Wahrscheinlichkeit ausgibt.

Zum Vergleich werden zwei Baselines aus der Temporal-Graph- und spatiotemporalen Literatur einbezogen: **TGN** (Temporal Graph Networks), das Link-Dynamiken über einen Knotenspeicher modelliert, der durch beobachtete Fahrt-Events aktualisiert wird, und **VSTD** (Variational Autoencoder-based Spatio-Temporal Disentanglement), das über einen variationalen Ansatz entflochtene räumliche und zeitliche Knotenrepräsentationen lernt. Eine Frequenz-Heuristik und eine logistische Regression über statische Features dienen als einfache Referenzpunkte. Um den Beitrag einzelner Komponenten zu isolieren, werden drei Modellvarianten trainiert: das vollständige Modell, eine Variante ohne Zeitreihen-Branch und eine Variante ohne historische Aktivitätsrate.

### Evaluation

Alle Modelle werden unter einem einheitlichen temporalen Split evaluiert — Training (21 Tage), Validierung (4 Tage) und Test (5 Tage) — strikt nach Zeit partitioniert, um Information Leakage zu verhindern. Normalisierungsparameter werden ausschließlich auf dem Trainings-Split berechnet und ohne Neuberechnung auf Validierungs- und Test-Sets angewandt.

Es werden zwei Query-Set-Protokolle verwendet. Das **sampled**-Protokoll liefert eine ausgewogene Menge positiver und negativer Stationspaare und wird primär während des Trainings genutzt. Das **rank_all**-Protokoll dient als zentrale Evaluation: Für jede Query-Station zu einem gegebenen Zeitschritt werden 99 zufällig gezogene Negativ-Stationspaare zusammen mit den positiven Kandidaten gerankt, was ein standardisiertes 1-vs-99-Ranking-Problem für eine konsistente und reproduzierbare Evaluation konstruiert.

Die Leistung wird anhand der drei in Abschnitt 2 eingeführten Metriken berichtet:

- AUC-ROC
- Average Precision (AP)
- Mean Reciprocal Rank (MRR)

Die primäre Vergleichsmetrik ist AP auf dem rank_all-Test-Set, da sie am robustesten gegenüber Klassenungleichgewicht ist und die Ranking-Qualität über alle Query-Gruppen hinweg direkt erfasst.

## Referenzen

Beddar-Wiesing, S., Moallemy-Oureh, A., Kempkes, M., & Thomas, J. M. (2025). *Absolute Evaluation Measures for Machine Learning: A Survey* (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2507.03392

Chen, Z., Wu, H., O'Connor, N. E., & Liu, M. (2021). A Comparative Study of Using Spatial-Temporal Graph Convolutional Networks for Predicting Availability in Bike Sharing Schemes. *2021 IEEE International Intelligent Transportation Systems Conference (ITSC)*, 1299–1305. https://doi.org/10.1109/ITSC48978.2021.9564831

Cini, A., Marisca, I., Zambon, D., & Alippi, C. (2025). Graph Deep Learning for Time Series Forecasting. *ACM Computing Surveys*, 57(12), 1–34. https://doi.org/10.1145/3742784

Constantin Urbainsky & Lyft Bikes & Scooters. (2024). *NYC Bike Sharing Network: Time-Series Enhanced Nodes and Edges Dataset* [Dataset]. Zenodo. https://doi.org/10.5281/ZENODO.13846868

Guo, X., Wang, S., & Zhao, L. (2022). Graph Neural Networks: Graph Transformation. In L. Wu, P. Cui, J. Pei, & L. Zhao (Hrsg.), *Graph Neural Networks: Foundations, Frontiers, and Applications* (S. 251–275). Springer Nature Singapore. https://doi.org/10.1007/978-981-16-6054-2_12

Kontopoulos, I., Makris, A., Tserpes, K., & Varvarigou, T. (2023). *An evaluation of time series forecasting models on water consumption data: A case study of Greece* (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2303.17617

Will, H., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems*, 30.
