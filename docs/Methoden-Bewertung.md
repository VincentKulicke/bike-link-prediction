---
tags: [projekt, methodik, link-prediction]
status: aktiv
erstellt: 2026-05-12
projekt: Link Prediction on Hybrid Graph + Time Series Data
---

# Methoden-Bewertung: GCN vs. GraphSAGE und GRU vs. 1D-CNN

Methodische Argumentation zur Wahl der beiden Branches in der vorgeschlagenen Architektur. Grundlage ist die [[Datenanalyse.md|Datenanalyse]] (nach Filterung 232 aktive Stationen, 5.626 Kanten, mittlerer Grad 48,5, vier Knoten-Zeitreihen pro Station, sechs Kanten-Zeitreihen, Beobachtungsfenster 4 Wochen). Aufgabenstellung in [[Link Prediction on Hybrid Graph + Time Series Data.md|Hauptdatei]].

> **Entscheidung (Stand 2026-06-03):** Das Team hat sich für **GraphSAGE + GRU** als primäre Architektur entschieden, übereinstimmend mit dem [[Konzeptdokument.md|Konzeptdokument]]. GCN und 1D-CNN sind als dokumentierte Alternativen und Ablations-Varianten eingeplant. Dieses Dokument hält den vollständigen Abwägungsprozess fest, damit die Wahl im Konzept und in der Präsentation belastbar begründet ist.

## 1. Graph-Branch: GCN vs. GraphSAGE

### Eckdaten des Graphen

- 232 aktive Knoten nach Filterung der isolierten Stationen (klein nach GNN-Maßstäben).
- 5.626 gerichtete Kanten, mittlerer Gesamtgrad ≈ 48,5 — ein **kleiner, dichter** Teilgraph.
- Edge Weights aus historischer Trip-Frequenz sind verfügbar und explizit gefordert.
- Stationen-Set bleibt über den Beobachtungszeitraum praktisch konstant.

### GCN — Stärken in diesem Setup

- **Edge Weights nativ unterstützt** über die gewichtete Adjazenzmatrix. Das aus der Aufgabenstellung geforderte Signal (Trip-Frequenz als Edge Weight) lässt sich ohne Workaround verwenden.
- **Voll-Batch-trainierbar**: Bei 2.213 Knoten passt der gesamte Graph in jedes GPU-Memory. Kein Sampling-Overhead, deterministisches Training.
- **Wenige Hyperparameter**: schnelles Konzept, schnelles Debugging.
- **Glättet stark**, was bei starker räumlicher Homophilie (nahe Stationen verhalten sich ähnlich) hilft.

### GCN — Schwächen

- **Transduktiv**: neue Stationen schwierig. Hier irrelevant, da geschlossenes Stationen-Set.
- **Über-Glättung** bei tieferen Architekturen. Bei zwei Hops kaum spürbar.
- **Behandelt alle Nachbarn gleich** modulo Edge Weight. Keine gelernte Bewertung der Nachbarn.

### GraphSAGE — Stärken in diesem Setup

- **Inductive Capability**. Funktioniert mit neuen Knoten. Hier kein Mehrwert.
- **Verschiedene Aggregatoren** (mean, max-pool, LSTM): bietet Modellierungs-Flexibilität, falls das konkrete Aggregat-Muster wichtig ist.
- **Mini-Batching mit Neighbor Sampling**: skaliert auf große Graphen. Bei dieser Größe unnötig.
- **Self-Concat**: trennt eigenes Feature explizit von Nachbar-Aggregat, reduziert Über-Glättung leicht.

### GraphSAGE — Schwächen in diesem Setup

- **Edge Weights nicht nativ**. Erfordert manuelles Feature-Engineering, um die Trip-Frequenz einzubringen.
- **Mehr Hyperparameter** (Sample-Größen pro Layer, Aggregator-Wahl).
- **Stochastik im Training** durch Sampling, erschwert reproduzierbare Ergebnisse.

### Entscheidung Graph-Branch

**GraphSAGE als gewählte primäre Architektur, GCN (und optional GAT) als Ablation.**

Begründung:
1. **Literaturnähe**: GraphSAGE (Will et al., 2017) ist der kanonische induktive GNN für Link Prediction und in der einschlägigen Literatur das Standard-Verfahren für diese Aufgabe. Das stützt die Argumentation im Konzept und in der Präsentation.
2. **Passt zum dichten Teilgraphen**: Der aktive Graph ist klein, aber dicht (mittlerer Grad 48,5). GraphSAGEs Neighbor-Sampling hält die Aggregation auch an hochgradigen Hubs beherrschbar, und die Self-Concat-Struktur trennt das eigene Feature einer Station explizit vom Nachbar-Aggregat — das reduziert Über-Glättung gerade bei hoher Knotendichte.
3. **Aggregator-Flexibilität**: Die Wahl zwischen mean-, max-pool- und LSTM-Aggregation erlaubt es, das Aggregat-Muster (z.B. „starker Inflow" vs. „ausgewogen") gezielt zu modellieren.
4. **Induktivität als Robustheits-Bonus**: Auch wenn das Stationen-Set geschlossen ist, macht die induktive Formulierung das Modell unempfindlich gegenüber kleinen Änderungen im Knoten-Set.

Umgang mit den Edge Weights: Anders als GCN nutzt GraphSAGE Kantengewichte nicht nativ. Die historische Trip-Frequenz wird daher über eine **gewichtete Mean-Aggregation** bzw. als zusätzliches Kanten-Feature eingebracht — etwas mehr Implementierungsaufwand als bei GCN, aber Standard.

**GCN** bleibt die naheliegende Ablation: native Edge Weights, Voll-Batch-Training (bei 232 Knoten trivial), weniger Hyperparameter, deterministisch. Es ist die einfachere Referenz, gegen die der Mehrwert von GraphSAGE gemessen wird. Optional als dritte Variante: **GAT / GATv2**, das Kanten-Wichtigkeit aus den Features lernt — bei Mobilitätsdaten mit Tageszeit-Modulation eine interessante Ablation.

## 2. Node-Zeitreihen-Branch: GRU vs. 1D-CNN

### Eckdaten der Knoten-Zeitreihen

- Vier Serien pro Station: `num_bikes_available`, `num_ebikes_available`, `num_bikes_disabled`, `num_docks_disabled`.
- Sampling-Median nach Resampling auf 5-Minuten-Bins: 5-10 Minuten je nach Serie.
- Realistisches Eingabefenster: 30 Minuten bis 6 Stunden, also 6 bis 72 Zeitschritte.
- Vorhersage-Horizont laut Aufgabenstellung: ein zukünftiges Zeitfenster, typischerweise 15-60 Minuten.

### GRU — Stärken in diesem Setup

- **Variable Sequenzlängen** möglich.
- **Implizites State-Tracking**: eine gerade leer-gelaufene Station behält die "Leerlauf-Phase" als hidden state.
- **Reichhaltige Literatur** speziell für Bike-Sharing-Forecasting.

### GRU — Schwächen in diesem Setup

- **Sequenziell**. Trainiert langsam, gerade bei kurzen Sequenzen ist der Compute-Overhead überproportional.
- **Vanishing-Gradient-Risiko** ab ~50 Schritten.
- **Mehr Tuning** (hidden size, Anzahl Layer, Dropout-Stellen).

### 1D-CNN — Stärken in diesem Setup

- **Parallel und schnell**. Bei kurzen Fenstern (6-72 Schritte) trainiert eine 1D-CNN um Größenordnungen schneller als eine GRU bei gleicher Modellgröße.
- **Lokale Muster sind exakt das, was hier zählt**: Entleerungs-Spikes, Refill-Spikes, kurzfristige Tageszeit-Wellen. Kernelgrößen von 3-5 reichen.
- **Dilated Convolutions** geben bei Bedarf einen größeren Receptive Field ohne Tiefen-Explosion.
- **Einfaches Tuning**: Kernelgröße, Filteranzahl, Pooling.
- **Robust gegen Padding und Maskierung**, wichtig bei Stationen mit kürzeren Zeitreihen.

### 1D-CNN — Schwächen in diesem Setup

- **Fixe Fenstergröße**: Receptive Field ist ein Architektur-Commitment.
- **Weniger natürliches State-Tracking** für sehr lange Abhängigkeiten (mehrere Tage). Bei kurzem Vorhersage-Horizont irrelevant.

### Entscheidung Node-Zeitreihen-Branch

**GRU als gewählte primäre Architektur, 1D-CNN als Ablation.**

Begründung:
1. **Literaturnähe**: GRU ist der Standard-Encoder im Bike-Sharing-Forecasting (Chen et al., 2021; Cini et al., 2025). Es fügt sich zudem natürlich in das GCRNN/DCGRU-Framework ein, falls Graph- und Zeitreihen-Verarbeitung in einer späteren Iteration enger verschränkt werden sollen.
2. **State-Tracking**: Das implizite hidden state der GRU erfasst Betriebszustände wie „Station gerade leer-gelaufen" oder „Refill läuft" auf natürliche Weise — genau die Dynamik, die für die Trip-Vorhersage relevant ist.
3. **Beherrschbarer Compute**: Der Hauptnachteil der GRU (sequenzielles, langsameres Training) wiegt hier weniger schwer, weil der aktive Graph nach Filterung nur 232 Stationen umfasst. Das Gesamtvolumen bleibt handhabbar.

**1D-CNN** bleibt die naheliegende Ablation: deutlich schnelleres, paralleles Training, gut für lokale Muster (Entleerungs-Spikes, Refill-Wellen) und einfacher zu interpretieren (Conv-Filter-Inspektion). Es ist die effizientere Referenz, gegen die sich der Mehrwert des State-Trackings der GRU messen lässt.

Falls der Vorhersage-Horizont in einer späteren Iteration auf mehrere Stunden ausgedehnt wird, lohnt sich zusätzlich der Blick auf **TCN** (Temporal Convolutional Network mit dilated kernels) oder einen **kleinen Transformer-Encoder**.

## 3. Sollte ich beides ausprobieren?

Ja, aber **sequenziell und hypothesengetrieben**, nicht parallel und ergebnisoffen.

### Empfohlene Vorgehens-Reihenfolge

1. **Iteration 1**: GraphSAGE + GRU + Fusion-MLP. Gewählter primärer Pfad, literaturnah, End-to-End-Implementierung.
2. **Iteration 2 (Ablation Architektur)**:
   - Tausch Graph-Branch: GraphSAGE → GCN (oder GAT).
   - Tausch Zeitreihen-Branch: GRU → 1D-CNN.
   - Beide Tauschs unabhängig durchführen, nicht parallel kombinieren.
3. **Iteration 2 (Ablation Komponenten)**:
   - Nur Graph-Branch.
   - Nur Zeitreihen-Branch.
   - Beide ohne Fusion-MLP (einfache Verkettung als Linear-Layer).
   - Zeigt, ob Fusion echten Mehrwert bringt.
4. **Baselines**: **TGN** (Temporal Graph Networks) als temporales Graph-Verfahren und **VSTD** (Variational Autoencoder-based Spatio-Temporal Disentanglement) als spatiotemporales Verfahren; zusätzlich eine Frequenz-Heuristik und eine logistische Regression über statische Features als einfache Referenzpunkte.
5. **Finale Ergebnistabelle** für Konzept und Präsentation.

### Anti-Empfehlung

Nicht alle 4 Kombinationen (GraphSAGE/GCN × GRU/1D-CNN) im Grid durchprobieren. Treatments überlagern sich, Compute wird verbrannt, narrative Klarheit geht verloren. Stattdessen vom gewählten Pfad (GraphSAGE + GRU) ausgehend jeweils eine Variable kontrolliert tauschen.

## 4. Argumentations-Snippet für das Konzept-Dokument

> Für den Graph-Branch wurde **GraphSAGE** gewählt, weil es das in der Link-Prediction-Literatur etablierte induktive Aggregations-Framework ist und über seine Self-Concat-Struktur und Neighbor-Sampling gut zum kleinen, aber dichten aktiven Stationsgraphen (232 Knoten, mittlerer Grad 48,5) passt. Die historische Trip-Frequenz wird als gewichtetes Kanten-Signal in die Aggregation einbezogen. GCN dient als einfachere Referenz mit nativen Edge Weights und wird als Ablations-Variante geführt.
>
> Für den Node-Zeitreihen-Branch wurde **GRU** gewählt, weil rekurrente Encoder im Bike-Sharing-Forecasting etabliert sind und der hidden state Betriebszustände der Stationen (Entleerung, Refill) implizit nachverfolgt. GRU fügt sich zudem natürlich in das GCRNN/DCGRU-Framework ein, falls Graph- und Zeitreihen-Verarbeitung später enger verschränkt werden. 1D-CNN ist die schnellere, auf lokale Muster fokussierte Alternative und wird als Ablations-Variante geführt.

## 5. Offene Fragen

- Falls in einer späteren Iteration das Vorhersage-Fenster vergrößert wird (z.B. 24 h), sollte der Zeitreihen-Branch neu evaluiert werden (TCN, Transformer-Encoder).
- Ist eine **gerichtete Graph-Variante** sinnvoll? Trip-Flow ist gerichtet. GraphSAGE lässt sich gerichtet betreiben, indem In- und Out-Nachbarn getrennt gesampelt und aggregiert werden; alternativ zwei separate Adjazenzen (forward, backward) mit parallelen Branches oder eine explizit gerichtete GNN-Variante (Directed GCN, DGCN).
- Falls GAT als Alternative gewählt wird: lohnt sich Multi-Head-Attention, oder reicht Single-Head?
