# Ergebnisgrundlage für die Präsentation

Basis für alle Zahlen, Aussagen und Schlussfolgerungen der Abschlusspräsentation.
Stand: 27.07.2026. Quelle: `bike-link-prediction-Final_Update/`.

---

## 0. Welche Zahlen gelten — Quellenhierarchie

Es existieren mehrere Ergebnisstände, die sich teilweise widersprechen. Verbindlich ist:

1. **`seeds_comparison.md` (5 Seeds, mit ± Streuung)** — höchste Priorität. Diese Zahlen tragen die Präsentation.
2. `ranking_comparison.md` (1:99-Protokoll) — für die harte Metrik.
3. `factors_comparison.md` — für Aussagen über Tuning vs. Rauschen.
4. `ablation_comparison.md` / `comparison.md` (Einzellauf) — **nur** wo keine Seed-Zahl existiert.

### ⚠️ Zahlen, die NICHT mehr verwendet werden dürfen

| Veraltet | Korrekt | Grund |
|---|---|---|
| GraphMixer default AP **0,653** | **0,596 ± 0,023** | Alte Prediction-Datei (22. Juni) reproduziert nicht; zwei unabhängige aktuelle Messungen stimmen überein |
| GraphMixer HPO AP **0,756** | **0,701 ± 0,038** | Single-Seed-Wert, durch Seed-Sweep korrigiert |
| „LSTM-Tuning verbessert MSE 0,239 → 0,233" | **kein Effekt (0,09σ)** | Differenz liegt unter dem Seed-Rauschen (±0,004) |
| Grid-Spreads aus Einzelläufen | um 28–34 % nach unten korrigiert | Enthielten Seed-Rauschen (siehe `factors_comparison.md`) |

Die Korrektur macht den Vorsprung des Hybridmodells **größer**, nicht kleiner.

---

## 1. Versuchsaufbau (Grundlage jeder Zahl)

- **Daten:** NYC Citi Bike, 16.05.–14.06.2024. Faktisch Jersey City / Hoboken (99,7 % des Verkehrs). 232 aktive Stationen.
- **Zeitraster:** 30-Minuten-Bins, insgesamt 1.379 Bins.
- **Zielgröße:** `count = Δ num_rides` je (u, i, bin); `label = 1` wenn `count > 0`.
- **Split** (strikt chronologisch, leakage-frei):

| Split | Bins | Positive Zellen | Fahrten |
|---|---|---|---|
| Train | 965 | 62.343 | 72.752 |
| Val | 192 | 13.617 | 16.099 |
| Test | 175 | 12.095 | 13.743 |

- **Kandidaten:** Positive + geseedete Negative im Verhältnis **1:5** (Seed 42), identisch für alle Modelle über `shared_eval`.
- **Zusätzliches hartes Protokoll:** 1-vs-99-Ranking, 3.000 Queries pro Split.

---

## 2. Referenzpunkte — womit die Ergebnisse zu vergleichen sind

**Ohne diese beiden Zeilen ist keine Zahl interpretierbar. Sie gehören auf die Ergebnisfolie.**

| Referenz | 1:5 AP | 1:5 AUC | 1:5 F1 | Count MSE | 1:99 MRR |
|---|---|---|---|---|---|
| **Zufall** | 0,166 | 0,500 | — | — | 0,052 |
| **Frequenz-Heuristik** | **0,891** | 0,973 | **0,000** | **0,238** | noch nicht gemessen |

Die Frequenz-Heuristik ist der einfachste denkbare Ansatz: *Score eines Paares = mittlere Fahrten pro Bin im Trainingszeitraum.* Kein Lernen, drei Zeilen Code (`_frequency_baseline` in `shared_eval.py`). Test-Werte oben, Validation: AP 0,885 / MSE 0,258.

**Warum das zentral ist:**
- Die AP-Zufallsbasis liegt bei **0,166**, nicht bei 0. Eine AP von 0,92 muss dagegen gelesen werden.
- Die Heuristik erreicht binär AP 0,891 — **sie schlägt GraphMixer (0,596) deutlich.**
- Beim Count ist sie mit MSE 0,238 **so gut wie das LSTM (0,240)**.
- Aber: Ihr **F1 = 0,000** — sie rankt gut, trifft aber keine brauchbare Entscheidung. Das Hybridmodell hat F1 0,860.

---

## 3. Hauptergebnis binär (5 Seeds, Test)

| Modell | AP | AUC | F1 |
|---|---|---|---|
| **Hybrid GraphSAGE+GRU (default)** | **0,9233 ± 0,0008** | 0,9851 ± 0,0003 | 0,8603 ± 0,0011 |
| Hybrid GraphSAGE+GRU (HPO) | 0,9233 ± 0,0006 | 0,9849 ± 0,0002 | 0,8597 ± 0,0010 |
| Hybrid GraphSAGE+1D-CNN (HPO) | 0,9226 ± 0,0005 | 0,9848 ± 0,0002 | 0,8599 ± 0,0012 |
| Frequenz-Heuristik | 0,891 | 0,973 | 0,000 |
| GraphMixer (HPO) | 0,7012 ± 0,0375 | 0,9095 ± 0,0100 | 0,4842 ± 0,0173 |
| GraphMixer (default) | 0,5957 ± 0,0233 | 0,8979 ± 0,0063 | 0,5589 ± 0,0130 |
| Zufall | 0,166 | 0,500 | — |

## 4. Hauptergebnis Count (5 Seeds, Test)

| Modell | MSE | MAE |
|---|---|---|
| **Hybrid GraphSAGE+GRU (HPO)** | **0,0820 ± 0,0003** | 0,0927 ± 0,0014 |
| Hybrid GraphSAGE+GRU (default) | 0,0824 ± 0,0004 | 0,0928 ± 0,0008 |
| Hybrid GraphSAGE+1D-CNN (HPO) | 0,0821 ± 0,0002 | 0,0922 ± 0,0012 |
| LSTM (default) | 0,2404 ± 0,0038 | 0,1875 ± 0,0020 |
| LSTM (HPO) | 0,2400 ± 0,0047 | 0,1890 ± 0,0016 |
| Frequenz-Heuristik | 0,238 | 0,180 |

**Der Count-Task ist das stärkste Ergebnis der Arbeit:** Faktor ~2,9 besser als Heuristik und LSTM.
Und: Das LSTM ist **nicht besser als eine konstante Rate pro Paar** — eine rein univariate Zeitreihe trägt hier nichts bei.

---

## 5. Hartes Protokoll: 1-vs-99-Ranking (Test)

| Modell | MRR | Hits@1 | Hits@5 |
|---|---|---|---|
| Hybrid GRU (default) | **0,402** | 0,250 | 0,567 |
| Hybrid GRU (HPO) | 0,402 | 0,249 | 0,567 |
| GraphMixer (default) | 0,342 | 0,196 | 0,490 |
| GraphMixer (HPO) | 0,315 | 0,173 | 0,449 |
| Zufall | 0,052 | 0,010 | — |

**Erkenntnisse:**
1. Das 1:5-Protokoll verdeckte Spielraum. MRR 0,402 statt AP 0,92 — die „Sättigung" war teils Artefakt des leichten Protokolls.
2. Der Vorsprung bleibt real, ist aber ehrlicher: 0,402 vs. 0,342 statt 0,92 vs. 0,60.
3. **Die HPO-Konfiguration von GraphMixer, ausgewählt auf 1:5-AP, wird unter 1:99 schlechter** (0,342 → 0,315). Eine auf leichte Diskriminierung optimierte Konfiguration überträgt sich nicht auf das harte Ranking-Ziel.

---

## 6. Hyperparameter-Optimierung

Suchräume (jeweils drei einflussreichste Achsen, Selektion nur auf Validation, Test genau einmal):

| Modell | Suchraum | Configs | Selektionsmetrik |
|---|---|---|---|
| GraphMixer | lr × hidden {64,128,256} × mixer_layers {1,2} | 18 | val AP |
| LSTM | lr × hidden {32,64,128} × num_layers {1,2} | 18 | val MSE |
| Hybrid GRU | lr × hidden {32,64,128} × λ_count {0,5;1;2} | 27 | val AP |
| Hybrid CNN | lr × hidden {32,64,128} × kernel {3,5} | 18 | val AP |

**Welche Unterschiede real sind (in gepoolten Standardabweichungen):**

| Vergleich | Differenz | σ | Bewertung |
|---|---|---|---|
| Hybrid: HPO vs. default | +0,0000 AP | 0,04 | **kein Effekt** |
| LSTM: HPO vs. default | −0,0004 MSE | 0,09 | **kein Effekt** |
| Hybrid: GRU vs. 1D-CNN | +0,0007 AP | 1,3 | nicht signifikant |
| GraphMixer: HPO vs. default | +0,1056 AP | 3,4 | **echte Verbesserung** |
| Hybrid vs. GraphMixer | +0,327 AP | ~12 | **eindeutig** |
| Hybrid vs. LSTM | −0,158 MSE | ~38 | **eindeutig** |

Nur das schwächste Modell profitiert vom Tuning. Beim Hybridmodell bewegt Tuning nichts.

---

## 7. Zwei-Faktor-Analyse: Tuning-Effekt vs. Seed-Rauschen

4 Konfigurationen über den Grid-Bereich × 5 Seeds = 60 Läufe.
σ_grid korrigiert um das verbleibende Seed-Rauschen: `σ_grid² = var(means) − σ_seed²/n`.

| Modell | σ_seed | σ_grid | σ_grid/σ_seed | Aussagekraft des Grids |
|---|---|---|---|---|
| Hybrid GRU | 0,0003 | 0,0028 | **8,2×** | Ranking aussagekräftig |
| LSTM | 0,0034 | 0,0074 | 2,2× | grenzwertig |
| GraphMixer | 0,0700 | 0,0919 | 1,3× | **kaum informativ** |

**Erkenntnisse:**
1. Das Verhältnis σ_grid/σ_seed sagt, ob eine Grid-Suche überhaupt vertrauenswürdig sein kann.
2. **GraphMixers Ranking ist teilweise zufällig** — Configs auf Rang 7 und 13 tauschen die Plätze, wenn über 5 Seeds gemittelt wird. Ein einzelner GraphMixer-Lauf ist für sich genommen nicht interpretierbar (±0,100 AP).
3. **Stabilität unterscheidet sich um zwei Größenordnungen:** σ_seed 0,0003 (Hybrid) vs. 0,0700 (GraphMixer) = Faktor 233. Reproduzierbarkeit ist ein eigenständiges Qualitätsargument.
4. Der Hyperparameter-Effekt des Hybrids ist zwar systematisch (8,2× über Rauschen), umfasst aber nur 0,0064 AP — **weniger als der Abstand zur Frequenz-Heuristik (0,032 AP).**

---

## 8. Kernaussagen für die Präsentation

**① Das Hybridmodell gewinnt auf beiden Aufgaben — auch bei fairem Tuning.**
Binär AP 0,923 vs. 0,701 (GraphMixer, getunt). Count MSE 0,082 vs. 0,240 (LSTM). ~12σ bzw. ~38σ.

**② Der eigentliche Beitrag liegt beim Count.**
Binär schlägt das Modell die triviale Frequenz-Heuristik nur um +0,032 AP. Beim Count ist es ~2,9× besser als Heuristik und LSTM. Das ist der Ort, an dem die Fusion nachweislich etwas leistet.

**③ Beide Baselines aus der Literatur sind schwächer als eine triviale Heuristik.**
GraphMixer (AP 0,596–0,701) liegt unter der Frequenz-Heuristik (0,891). Das LSTM (MSE 0,240) ist nicht besser als eine konstante Rate (0,238). Ein wichtiger Befund über die Aufgabe: Paar-Historie ist ein sehr starkes Signal.

**④ Das Modell ist robust, die Baseline ist es nicht.**
σ_seed 0,0003 vs. 0,0700 — Faktor 233. Über 27 Konfigurationen bewegt sich die Validation-AP nur um 0,006; selbst die schlechteste Konfiguration schlägt GraphMixer deutlich.

**⑤ Tuning ist nicht die Stellschraube.**
HPO bewegt das Hybridmodell nicht (0,04σ), auch nicht unter dem harten 1:99-Protokoll mit echtem Spielraum (MRR 0,402 = 0,402). Verbesserungen müssen über Features/Architektur kommen, nicht über Hyperparameter.

**⑥ Das Evaluationsprotokoll bestimmt das Bild.**
Unter 1:5 wirkt der Vorsprung dramatisch (0,92 vs. 0,60), unter 1:99 ist er ehrlicher (MRR 0,402 vs. 0,342). Und eine auf 1:5 optimierte Konfiguration kann unter 1:99 schlechter werden (GraphMixer 0,342 → 0,315).

---

## 9. Ehrliche Grenzen (gehören auf die Limitations-Folie)

- **Fokussierte Grid-Suche** über drei Achsen pro Modell, nicht erschöpfend. Fest blieben u. a. `epochs`, `dropout`, `batch_size`, `ts_lookback`, `num_neighbors`. Die Lernrate war nachweislich die dominante Achse.
- **Branch-Ablationen fehlen.** Es wurde nur der temporale Encoder getauscht (GRU ↔ CNN), keine Komponente entfernt. → siehe offener Punkt unten.
- **1:99-Ranking nur für die binären Modelle**, die Frequenz-Heuristik ist dort noch nicht gemessen.
- **Datensatz ist faktisch Jersey City / Hoboken**, nicht Manhattan — Übertragbarkeit auf größere Netze offen.
- **Count-Kopf ist nicht konditioniert:** Der Loss läuft unmaskiert über alle Kandidatenpaare. Es ist ein Dual-Head-Multi-Task-Modell, keine Hurdle-Formulierung im engeren Sinne (die E[Y | Y>0] modelliert). Sprachlich sauber: „dual-head" oder „simplified hurdle-style".

---

## 10. Offener Punkt mit dem größten Hebel

**Die Aussage „das Signal kommt aus dem Graph-Branch und den Pair-Features, nicht aus dem temporalen Modul" ist derzeit nicht belegt.**

Aus GRU ≈ CNN (0,9233 vs. 0,9226) folgt nur, dass die *Kodierung* austauschbar ist — nicht, dass der Branch nichts beiträgt. Beide Encoder könnten gleich viel oder gleich wenig beitragen. Der Beleg erfordert das **Entfernen** einer Komponente:

- ohne temporalen Branch (nur Graph + Pair-Features)
- ohne Graph-Branch (nur temporal + Pair-Features)
- ohne Pair-Features (ohne `log1p(frequency)`)

Der dritte ist wegen der Frequenz-Heuristik (AP 0,891) besonders relevant.

**In beide Richtungen verwertbar:** Bricht die Leistung ein, ist die Hybrid-These belegt. Bleibt sie gleich, ist es ein ehrliches negatives Ergebnis („in diesem Datensatz trägt die Verfügbarkeits-Zeitreihe nicht bei"). Solange der Lauf fehlt, sollte die Behauptung auf den Folien **nicht** aufgestellt werden.

---

## 11. Reproduzierbarkeit

```
python evaluation/shared_eval.py                       # Protokoll + Frequenz-Heuristik
python ablation/run_all_grids.sh                       # HPO (81 Configs)
python ablation/eval_seeds.py   --models all --seeds 5 # Multi-Seed (35 Läufe)
python ablation/eval_factors.py --models all --seeds 5 # Zwei-Faktor (60 Läufe)
python ablation/eval_ranking.py --max_queries 3000     # 1-vs-99-Ranking
```

Rohdaten: `ablation/results/*.csv` · Zusammenfassungen: `ablation/results/*_comparison.md`
