# Ergebnisgrundlage für die Präsentation

Basis für alle Zahlen, Aussagen und Schlussfolgerungen der Abschlusspräsentation.
Stand: **05.09.2026**, nach der finalen Hyperparameter-Suche (361 Läufe, 21 h)
und zwei Korrekturen an den Baselines.

---

## 0. Welche Zahlen gelten — Quellenhierarchie

1. **`final_eval_summary.csv`** — 4 Modelle × 5 Seeds mit den Siegerkonfigurationen
   der finalen Suche. Diese Zahlen tragen die Präsentation.
2. `hpo_final_comparison.md` — für Aussagen über Hyperparameter-Wirkung.
3. `ranking_comparison.md` — 1:99-Protokoll.
4. `branches_comparison.md` — Komponenten-Ablation.
5. `protocol_gap.csv` — Verhalten auf allen Paaren statt auf der Stichprobe.
6. `runtime_summary.csv` — Kosten.

Die Konfigurationen werden von allen Auswertungsskripten aus `hpo_final_*.csv`
**gelesen**, nicht im Code hinterlegt. Zahlen und Konfiguration können damit
nicht auseinanderlaufen.

### ⚠️ Zahlen, die NICHT mehr verwendet werden dürfen

| Veraltet | Korrekt | Grund |
|---|---|---|
| GraphMixer AP **0,701** | **0,9019 ± 0,0014** | Trainierte auf 50 % Positivrate, bewertet wurde 1:5 (16,7 %) |
| LSTM MSE **0,116** | **0,0954 ± 0,0013** | `lookback` war nie getunt; 192 statt 48 |
| Hybrid AP **0,9233** | **0,9238 ± 0,0004** | neue Siegerkonfiguration, 5 Seeds |
| GraphMixer Seed-σ **0,0375** | **0,0014** | Artefakt des defekten Trainings |
| „Hybrid ist pareto-dominant, 19× schneller" | **9× bei der Inferenz** | GraphMixer trainiert jetzt schneller |
| „Der temporale Branch trägt nichts bei" | **redundant, nicht nutzlos** | siehe Abschnitt 10 |
| σ-Angaben aus `factors_comparison.md` | überholt | altes Seed-Rauschen, siehe Abschnitt 7 |

> Anders als bei der letzten Korrekturrunde macht diese Runde den Vorsprung des
> Hybridmodells **kleiner**, nicht größer. Der Abstand zu GraphMixer schrumpft
> von 0,222 auf 0,022 AP.

---

## 1. Versuchsaufbau (unverändert)

- **Daten:** Citi Bike, 16.05.–14.06.2024, faktisch Jersey City / Hoboken. 232 Stationen.
- **Zeitraster:** 30-Minuten-Bins, 1.379 Bins.
- **Zielgröße:** `count = Δ num_rides` je (u, i, bin); `label = 1` wenn `count > 0`.
- **Split** strikt chronologisch: Train 965 Bins / Val 192 / Test 175.
- **Kandidaten:** Positive + geseedete Negative im Verhältnis **1:5** (Seed 42),
  identisch für alle Modelle über `shared_eval`.
- **Zusätzliche Protokolle:** 1-vs-99-Ranking (3.000 Queries) und das
  **vollständige Gitter** (alle 232 × 231 Paare je Fenster).

Testsplit: 12.095 positive Zellen inklusive Selbstschleifen, **11.552** ohne.
Selbstschleifen (Rundfahrten, 5,4 % der Zellen) sind in der Gitter-Auswertung
beidseitig ausgeschlossen.

---

## 2. Referenzpunkte — womit die Ergebnisse zu vergleichen sind

**Ohne diese Zeilen ist keine Zahl interpretierbar.**

| Referenz | 1:5 AP | 1:5 AUC | 1:5 F1 | Count MSE | 1:99 MRR |
|---|---|---|---|---|---|
| **Zufall** | 0,166 | 0,500 | — | — | 0,052 |
| **Frequenz-Heuristik** | **0,8914** | 0,9732 | 0,000 | 0,238 | **0,409** |

Die Heuristik ist der einfachste denkbare Ansatz: *Score = mittlere Fahrten pro
Bin im Trainingszeitraum.* Kein Lernen, drei Zeilen Code.

**Was das bedeutet:**
- Die AP-Zufallsbasis liegt bei **0,166**, nicht bei 0.
- Binär schlägt die Heuristik den Hybrid nicht (0,891 gegenüber 0,924) — der
  Abstand beträgt aber nur **+0,032 AP**.
- **Unter 1:99 gewinnt die Heuristik** (MRR 0,409 gegenüber 0,408). Siehe Abschnitt 5.
- Ihr F1 = 0,000: Sie rankt gut, trifft aber keine brauchbare Schwellenentscheidung.

---

## 3. Hauptergebnis binär (5 Seeds, Test)

| Modell | AP | AUC | F1 |
|---|---|---|---|
| **Hybrid GRU** | **0,9238 ± 0,0004** | 0,9851 ± 0,0001 | 0,8596 ± 0,0011 |
| Hybrid CNN | 0,9231 ± 0,0003 | 0,9851 ± 0,0001 | 0,8600 ± 0,0005 |
| GraphMixer | 0,9019 ± 0,0014 | 0,9831 ± 0,0002 | 0,8469 ± 0,0011 |
| Frequenz-Heuristik | 0,8914 | 0,9732 | 0,000 |

| Vergleich | Differenz | σ | |
|---|---|---|---|
| Hybrid GRU vs. GraphMixer | +0,0219 | 15,3 | signifikant |
| Hybrid GRU vs. Hybrid CNN | +0,0007 | 1,4 | **nicht signifikant** |

Bemerkenswert: Die **AUC ist nahezu gleich** (0,9851 gegenüber 0,9831). Der
Unterschied steckt fast vollständig in der AP, also im hochbewerteten Bereich —
genau dort, wo es bei Link Prediction zählt.

---

## 4. Hauptergebnis Count (5 Seeds, Test)

| Modell | MSE |
|---|---|
| **Hybrid GRU** | **0,0826 ± 0,0004** |
| Hybrid CNN | 0,0831 ± 0,0004 |
| LSTM | 0,0954 ± 0,0013 |
| Frequenz-Heuristik | 0,238 |
| konstant null | 0,263 |

Hybrid gegenüber LSTM: −0,0128 MSE = **9,4 σ**, signifikant. Der Abstand ist
gegenüber dem früheren Stand (0,0344) auf ein Drittel geschrumpft, weil die
Fensterlänge des LSTM erstmals getunt wurde.

---

## 5. Hartes Protokoll: 1-vs-99-Ranking (Test)

Jedes Ziel wird gegen 99 Alternativen **mit demselben Quellknoten und
Zeitfenster** gerankt — die Frage lautet „welches Ziel?", nicht „ob überhaupt?".

| Modell | MRR | Hits@1 | Hits@5 | AUC | AP |
|---|---|---|---|---|---|
| **Frequenz-Heuristik** | **0,409** | **0,256** | **0,577** | 0,929 | **0,126** |
| Hybrid GRU | 0,408 | 0,255 | 0,573 | 0,930 | 0,108 |
| Hybrid CNN | 0,402 | 0,249 | 0,566 | 0,930 | 0,109 |
| GraphMixer | 0,343 | 0,196 | 0,487 | 0,915 | 0,073 |

> **Der wichtigste Vorbehalt der Arbeit.** Kein gelerntes Modell schlägt die
> triviale Heuristik. Der Hybrid liegt beim MRR gleichauf und bei der AP
> darunter — nach 361 durchsuchten Konfigurationen.

**Und ein Befund, der den GraphMixer-Fix einordnet:** Der Sprung von AP 0,70 auf
0,90 unter 1:5 überträgt sich **nicht** (MRR 0,342 → 0,343). Die Protokolle
testen Verschiedenes: 1:5 zieht Negative zufällig über alle Paare und Bins,
1:99 gegen Alternativen desselben Startpunkts. Der Fix hat die leichtere
Fähigkeit verbessert.

---

## 5b. Vollständiges Gitter statt Stichprobe

`protocol_gap.py`, 175 Testfenster × 53.592 Paare = 9,38 Mio., selbes Modell
für beide Hälften, Selbstschleifen beidseitig ausgeschlossen.

| Protokoll | Paare | Positivrate | TP | FP | Precision | Recall |
|---|---|---|---|---|---|---|
| 1:5-Stichprobe | 72.393 | 15,96 % | **10.276** | 2.219 | 0,8224 | **0,8895** |
| volles Gitter @0,5 | 9.378.600 | 0,12 % | **10.276** | 343.180 | 0,0291 | **0,8895** |
| volles Gitter, Top-K | 9.378.600 | 0,12 % | 2.448 | — | 0,2119 | — |
| Zufall | — | 0,12 % | — | — | 0,0012 | — |

Treffer und Recall sind **bitidentisch** — dasselbe Modell findet dieselben
Fahrten. Nur die Fehlalarme unterscheiden sich (2.219 gegenüber 343.180). Das
Ranking bleibt stark: Top-K ist **172× besser als Zufall**.

---

## 6. Hyperparameter-Optimierung (finale Suche)

361 Konfigurationen, 21,2 h, Auswahl auf Validierung, Zahlen auf Test.

| Modell | Configs | bester | schlechtester | Δ | σ |
|---|---|---|---|---|---|
| LSTM (MSE) | 90 | 0,0935 | 0,1593 | 0,0658 | **51** |
| Hybrid GRU (AP) | 72 | 0,9239 | 0,9107 | 0,0131 | **33** |
| GraphMixer (AP) | 91 | 0,9021 | 0,8531 | 0,0490 | **35** |
| Hybrid CNN (AP) | 108 | 0,9225 | 0,9165 | 0,0060 | 20 |

**Hyperparameter wirken bei jedem Modell hochsignifikant.** Der übliche
Vergleich „Default gegen HPO" beantwortet eine *andere* Frage — ob die
Ausgangswahl gut war. Sie war es: Der Default lag im alten Gitter auf Rang 3
von 27.

**Je Modell dominiert genau eine Achse:**

| Modell | dominante Achse | Spanne | alle übrigen |
|---|---|---|---|
| LSTM | **`lookback`** | 0,0619 | < 0,0011 |
| GraphMixer | `lr` | 0,0167 | ≤ 0,0105 |
| Hybrid GRU | `lr` | 0,0024 | ≤ 0,0008 |
| Hybrid CNN | `lr` | 0,0015 | ≤ 0,0010 |

Beim LSTM wirkt die Fensterlänge **56× stärker** als jede andere Achse. Der alte
Default liegt im neuen Gitter auf **Rang 38 von 90**.

**Siegerkonfigurationen:**

| Modell | Konfiguration |
|---|---|
| Hybrid GRU | `lr=1e-3, hidden=256, ts_lookback=48, fusion_hidden=256` |
| Hybrid CNN | `lr=1e-3, hidden=128, ts_lookback=48, dropout=0` |
| LSTM | `lr=1e-3, hidden=64, lookback=192, layers=2, dropout=0,2` |
| GraphMixer | `lr=1e-3, hidden=128, num_neighbors=40, mixer_layers=1` |

**Offene Randlage:** Bei GraphMixer liegt das Optimum bei **allen drei** Achsen
am oberen Rand — das Optimum ist nicht erreicht. Beim LSTM ist `lookback=192`
ebenfalls Randwert, die Kurve flacht aber sichtbar ab (48→96: −0,0125;
96→192: −0,0084).

---

## 7. Seed-Rauschen (ersetzt die alte Zwei-Faktor-Analyse)

Gemessen über 5 Seeds der jeweiligen Siegerkonfiguration:

| Modell | Seed-σ | früher |
|---|---|---|
| Hybrid CNN | 0,0003 | 0,0005 |
| Hybrid GRU | 0,0004 | 0,0006 |
| LSTM | 0,0013 | 0,0008 |
| GraphMixer | **0,0014** | **0,0375** |

GraphMixers Rauschen fiel um **Faktor 27** — dasselbe Muster wie beim LSTM-Fix
(±0,0038 → ±0,0008 damals). Ein Modell, das auf der falschen Verteilung
trainiert, sitzt nahe an einer degenerierten Lösung, wo Zufall stark durchschlägt.

Das LSTM rauscht jetzt *stärker* als zuvor, weil die Siegerkonfiguration zwei
Schichten mit Dropout nutzt statt einer ohne.

> `factors_comparison.md` und `seeds_comparison.md` beruhen auf dem alten
> Rauschen und sind **überholt**.

---

## 8. Kosten (5 Seeds, Mediane)

| Modell | Training | s/Epoche | Inferenz | Speicher | Parameter |
|---|---|---|---|---|---|
| **Hybrid CNN** | 99,0 s | 3,30 | **0,228 s** | **1.171 MB** | 151.682 |
| Hybrid GRU | 158,8 s | 5,29 | 0,410 s | 1.938 MB | 599.298 |
| LSTM | 158,5 s | 5,28 | 0,481 s | **9.845 MB** | 50.497 |
| **GraphMixer** | **69,8 s** | **2,33** | 3,724 s | 1.585 MB | 115.085 |

- GraphMixer **trainiert am schnellsten**, der Hybrid **inferiert 9× schneller**.
  Da Training einmalig und Inferenz Dauerbetrieb ist, bleibt der Hybrid im
  Betrieb günstiger — die frühere Aussage „19× und pareto-dominant" gilt nicht mehr.
- Die **CNN-Variante** ist der Effizienzgewinner: ein Viertel der GRU-Parameter,
  schnellste Inferenz, bei statistisch gleicher Genauigkeit.
- **Die beste LSTM-Konfiguration passt nicht in den Grafikspeicher.** 9.845 MB
  gegenüber 8.188 MiB der Karte; sie läuft nur, weil der Treiber in den
  Systemspeicher auslagert. `lookback=96` (MSE 0,1020) wäre der betreibbare
  Kompromiss.

---

## 9. Ehrliche Grenzen (Limitations-Folie)

- **Kein gelerntes Modell schlägt die Frequenz-Heuristik unter 1:99.** Der
  wichtigste Vorbehalt — Abschnitt 5.
- **Auf allen Paaren fällt die Precision von 0,822 auf 0,029** bei unverändertem
  Recall. Die berichteten Zahlen beschreiben ein gestütztes Protokoll.
- **GraphMixers Optimum ist nicht erreicht** (drei Achsen am Rand).
- **Datensatz ist faktisch Jersey City / Hoboken**, nicht Manhattan.
- **Count-Kopf ist nicht konditioniert:** unmaskierter Loss über alle
  Kandidatenpaare. Sprachlich sauber: „dual-head", nicht „Hurdle" im engeren Sinne.
- **Laufzeiten gelten für diese Hardware.** GraphMixers Python-Schleifen
  profitieren kaum von schnellerer Hardware, die Tensor-Operationen des Hybrids schon.

---

## 10. Welche Komponente trägt das Signal?

2 Encoder × 5 Varianten × 5 Seeds, Parameterzahl über alle Varianten identisch
(per Zusicherung geprüft: GRU 599.298, CNN 151.682).

| entfernt | GRU: Δ AP (σ) | CNN: Δ AP (σ) |
|---|---|---|
| **Paar-Features** | **−0,0126 (9,4)** | **−0,0281 (15,4)** |
| Graph-Branch | −0,0024 (5,0) | −0,0033 (3,7) |
| temporaler Branch | −0,0002 (0,6) | +0,0002 (0,3) |
| **Graph *und* Zeitreihe** (`pair_only`) | **−0,0080 (11,2)** | **−0,0075 (5,4)** |

**Die letzte Zeile ist die entscheidende.** Leave-one-out misst nur den
*marginalen* Beitrag und kann „nutzlos" nicht von „redundant" unterscheiden:

| | Δ AP (GRU) |
|---|---|
| Graph allein entfernen | −0,0024 |
| Zeitreihe allein entfernen | −0,0002 |
| Summe der Einzelbeiträge | −0,0026 |
| **beide zusammen entfernen** | **−0,0080** |

Der gemeinsame Effekt ist **dreimal so groß wie die Summe der Einzeleffekte** —
der Fingerabdruck von Redundanz. Fällt der Zeitreihen-Zweig weg, kompensiert der
Graph, und umgekehrt.

> **Korrektur:** „Der temporale Branch trägt nichts bei" war eine
> Fehlinterpretation. Korrekt: *Er trägt nichts bei, das der Graph nicht auch
> liefern könnte.* Dass zeitliche Information wertvoll ist, zeigt das LSTM
> allein (MSE 0,0954 gegenüber 0,0826).

**Rechtfertigt sich der Hybrid?**

| Modell | test AP |
|---|---|
| voller Hybrid | **0,9238** |
| nur Paar-Features (nicht-hybrid) | 0,9158 |
| ohne Paar-Features | 0,9112 |
| Frequenz-Heuristik | 0,8914 |

Ja — mit **+0,0080 AP (11,2 σ)** gegenüber einer nicht-hybriden Variante. Klein,
aber hochsignifikant. Die Architektur rechtfertigt sich knapper als ursprünglich
dargestellt.

---

## 11. Reproduzierbarkeit

```bash
python evaluation/shared_eval.py                        # Protokoll + Frequenz-Heuristik
bash   ablation/run_hpo_final.sh                        # finale Suche (361 Läufe, ~21 h)
python ablation/hpo_final_report.py                     # Gitter-Auswertung
python ablation/final_eval.py                           # 4 Modelle × 5 Seeds
python ablation/runtime_analysis.py --phase b           # Kosten, 5 Seeds interleaved
python ablation/protocol_gap.py                         # volles Gitter vs. 1:5
python ablation/eval_ranking.py --max_queries 3000      # 1-vs-99
python ablation/eval_branches.py --seeds 5              # Ablation, 2 Encoder × 5 Varianten
python ablation/demo_score_day.py --day 27              # Demo-Daten
python ablation/demo_animate.py --day 27                # Demo-Animation
```

Alle Skripte lesen ihre Modellkonfiguration über `final_eval.best_cfg()` aus
`hpo_final_*.csv`. Die Suche selbst hängt jede fertige Konfiguration sofort an
die CSV an und überspringt sie beim Neustart; `run_hpo_final.sh` startet den
Prozess nach einem CUDA-Fault neu.

Rohdaten: `ablation/results/*.csv` · Zusammenfassungen: `ablation/results/*_comparison.md`
