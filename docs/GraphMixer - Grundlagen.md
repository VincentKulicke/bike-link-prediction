---
tags: [projekt, baseline, link-prediction, didaktik, graphmixer]
status: in Bearbeitung
erstellt: 2026-06-13
gehört-zu: "[[Link Prediction on Hybrid Graph + Time Series Data.md|Link Prediction Projekt]]"
---

# GraphMixer – Grundlagen verständlich erklärt

Dieses Dokument erklärt das **GraphMixer-Modell** von Grund auf, sodass es auch ohne Vorwissen in Graph-Deep-Learning verständlich ist. GraphMixer ist in unserem Projekt die **Temporal-Graph-Baseline**, gegen die wir unser eigenes Hybridmodell beim binären Vergleich (Link ja/nein) antreten lassen.

> Quelle: Cong et al., *„Do We Really Need Complicated Model Architectures for Temporal Networks?"*, ICLR 2023. Die Kernbotschaft des Papers steckt schon im Titel: Ein **bewusst einfaches** Modell schlägt überraschend oft die komplizierten.

---

## 1. Das Problem, das GraphMixer löst

Stell dir ein Netzwerk vor, in dem ständig **Verbindungen entstehen** – bei uns: Fahrten zwischen Bike-Sharing-Stationen. Die Frage lautet:

> *Wird zwischen Station A und Station B in der nächsten halben Stunde eine Fahrt stattfinden – ja oder nein?*

Das nennt man **Link Prediction** (Verbindungsvorhersage). Das Besondere: Das Netzwerk ist **dynamisch**, es verändert sich über die Zeit. Eine Verbindung, die morgens um 8 Uhr typisch ist (Pendler zum Bahnhof), gibt es nachts um 3 Uhr nicht. Ein Modell muss also nicht nur *wer-mit-wem* lernen, sondern auch *wann*.

---

## 2. Die Grundbegriffe (Schritt für Schritt)

### Graph
Ein **Graph** ist eine Sammlung von Punkten und Verbindungen.
- **Knoten** (engl. *node*): die Punkte. Bei uns: die Bike-Stationen.
- **Kante** (engl. *edge*): eine Verbindung zwischen zwei Knoten. Bei uns: eine Fahrt von Station A zu Station B.

Bildlich: eine Landkarte, auf der Stationen Punkte sind und jede gefahrene Strecke eine Linie zwischen zwei Punkten.

### Statischer vs. temporaler Graph
- **Statischer Graph**: Die Verbindungen sind „eingefroren". Es gibt eine Linie zwischen A und B oder nicht – ohne Zeitangabe.
- **Temporaler Graph**: Jede Kante hat einen **Zeitstempel**. Eine Kante ist nicht „A–B", sondern „A–B um 08:14 Uhr", „A–B um 08:21 Uhr", usw. Jede einzelne Fahrt ist ein eigenes Ereignis mit Uhrzeit.

GraphMixer arbeitet mit einem **temporalen Graphen**. Genau deshalb brauchen wir für diese Baseline die **einzelnen, zeitgestempelten Fahrten** aus dem Original-Citi-Bike-Datensatz – nicht unsere aggregierte „Superedge", die alle Fahrten zwischen A und B nur zu einer Zahl zusammenfasst.

### Knoten-Features
Jeder Knoten kann **Eigenschaften** mitbringen, in Zahlen ausgedrückt. Bei einer Station z. B. Kapazität (Anzahl Docks), geografische Lage. Solche Zahlenlisten nennt man **Feature-Vektor**.

### Embedding
Ein **Embedding** ist eine kompakte Zahlenliste, die ein Modell selbst lernt, um etwas „in eigenen Worten" zu beschreiben. Statt „Station Grove St PATH" zu sagen, beschreibt das Modell die Station durch z. B. 100 Zahlen, die ihr Verhalten zusammenfassen (etwa „sehr aktiver Pendler-Hub, morgens stark nachgefragt"). Zwei Stationen mit ähnlichem Verhalten bekommen ähnliche Embeddings.

---

## 3. Die Grundidee von GraphMixer in einem Satz

> Um vorherzusagen, ob A und B sich bald verbinden, schaut GraphMixer sich die **jüngste Vergangenheit beider Stationen** an – *mit wem* und *wann* hatten sie zuletzt Verbindungen – und leitet daraus ab, wie wahrscheinlich eine neue Verbindung A–B ist.

Das Clevere: GraphMixer verzichtet bewusst auf komplizierte Bauteile (keine „Aufmerksamkeit", kein „Gedächtnis-Modul", keine wiederkehrenden Netze). Es nutzt fast nur das **einfachste neuronale Bauteil überhaupt** – das MLP. Trotzdem ist es sehr konkurrenzfähig.

### Was ist ein MLP?
**MLP** = *Multilayer Perceptron*, das klassische, einfachste neuronale Netz: Es nimmt eine Zahlenliste als Eingabe, multipliziert und addiert sie über mehrere „Schichten" und gibt eine neue Zahlenliste aus. Man kann es sich als eine **flexible mathematische Funktion** vorstellen, die aus Beispielen lernt, Eingaben in nützliche Ausgaben zu übersetzen.

---

## 4. Der Aufbau: drei Bausteine

GraphMixer besteht aus drei Teilen. Wir gehen sie der Reihe nach durch.

### Baustein 1 – Der Link-Encoder („Was ist zuletzt passiert?")

Dieser Teil fasst die **jüngsten Verbindungen** eines Knotens zusammen.

So läuft es ab:
1. Für eine Station nimmt man ihre **letzten K Fahrten** (z. B. die letzten 20 Ereignisse).
2. Jede dieser Fahrten wird zu einer Zeile in einer Tabelle. Die Zeile enthält:
   - die **Merkmale** der Fahrt (z. B. zu welcher Station, ggf. weitere Eigenschaften),
   - eine **Zeit-Kodierung** – also „wie lange ist das her?" in eine für das Modell lesbare Zahlenform übersetzt.
3. So entsteht eine **Tabelle**: Zeilen = die letzten K Fahrten, Spalten = die Merkmale + Zeitinfo.
4. Auf diese Tabelle wird ein **MLP-Mixer** angewendet (siehe Abschnitt 5), der die Tabelle zu einem einzigen Embedding verdichtet: „So sah die jüngste Aktivität dieser Station aus."

**Wichtige Design-Entscheidung – die feste Zeit-Kodierung:** Wie „wie lange her" in Zahlen übersetzt wird, ist bei GraphMixer **fest vorgegeben** und wird *nicht* mitgelernt. Die Autoren fanden heraus, dass lernbare Zeit-Kodierungen das Training instabil machen. Die feste Variante (eine mathematische Funktion mit Kosinus-Schwingungen unterschiedlicher Geschwindigkeit) ist robuster. Das ist einer der Gründe, warum GraphMixer so stabil und einfach ist.

### Baustein 2 – Der Node-Encoder („Wie aktiv ist die Station generell?")

Dieser Teil beschreibt die **allgemeine Identität und jüngste Aktivität** eines Knotens, unabhängig von der genauen Reihenfolge der Ereignisse.

So läuft es ab:
- Man schaut die **Nachbarn** der Station im letzten Zeitfenster an (alle Stationen, mit denen sie kürzlich verbunden war) und bildet einen **Durchschnitt** ihrer Merkmale.
- Das ergibt eine kompakte Zusammenfassung: „Diese Station ist zuletzt stark/wenig genutzt worden und hängt mit diesen Arten von Stationen zusammen."

Während der Link-Encoder die *zeitliche Abfolge* betont, liefert der Node-Encoder das *Gesamtbild* der Station.

### Baustein 3 – Der Link-Klassifikator („Verbindung ja oder nein?")

Jetzt kommt die eigentliche Vorhersage für ein Paar (A, B):
1. Man nimmt die Embeddings beider Stationen (jeweils aus Link- und Node-Encoder).
2. Man **fügt sie zusammen** (verkettet die Zahlenlisten von A und B).
3. Ein abschließendes **MLP** liest diese kombinierte Beschreibung und gibt eine **Wahrscheinlichkeit** aus: Wie wahrscheinlich entsteht zwischen A und B eine Verbindung?
4. Liegt die Wahrscheinlichkeit über einer Schwelle, lautet die Vorhersage „Link = ja".

---

## 5. Das Herzstück: der MLP-Mixer

Der Name „GraphMixer" kommt von diesem Bauteil. Es stammt ursprünglich aus der Bildverarbeitung (MLP-Mixer, 2021) und ersetzt dort kompliziertere Mechanismen durch zwei einfache, abwechselnd angewandte MLPs.

Erinnerung: Der Link-Encoder hat eine **Tabelle** gebaut – Zeilen = letzte Fahrten, Spalten = Merkmale. Der MLP-Mixer mischt diese Tabelle auf **zwei Arten**:

1. **Token-Mixing („spaltenweise mischen")**
Mischt Information **über die verschiedenen Fahrten hinweg** (über die Zeilen). Beantwortet: *Wie hängen die einzelnen jüngsten Ereignisse miteinander zusammen?* Beispiel: „Drei Fahrten kurz hintereinander zur selben Station" wird als Muster erkennbar.

2. **Channel-Mixing („zeilenweise mischen")**
Mischt Information **über die verschiedenen Merkmale hinweg** (über die Spalten). Beantwortet: *Wie hängen die Eigenschaften innerhalb einer Fahrt zusammen?*

Diese beiden Misch-Schritte werden abwechselnd ausgeführt. Das Ergebnis ist eine kompakte Zusammenfassung der gesamten jüngsten Aktivität – erzeugt **nur mit MLPs**, ohne die rechenintensiven Mechanismen anderer Modelle.

**Analogie:** Stell dir eine Tabelle mit Notizen vor. Token-Mixing liest *spaltenweise* (vergleicht dieselbe Eigenschaft über alle Ereignisse), Channel-Mixing liest *zeilenweise* (betrachtet alle Eigenschaften eines Ereignisses zusammen). Durch wechselndes Lesen in beide Richtungen entsteht ein Gesamtverständnis der Tabelle.

---

## 6. Warum GraphMixer einfacher ist als TGN

In unserer Aufgabe wurde GraphMixer ausdrücklich empfohlen, weil es **leichter zu implementieren** ist als TGN. Der Unterschied:

| Eigenschaft | TGN (komplex) | GraphMixer (einfach) |
|---|---|---|
| **Gedächtnis-Modul** | ja – speichert pro Knoten einen fortlaufend aktualisierten Zustand | **nein** |
| **Wiederkehrende Netze (RNN/GRU)** | ja | **nein** |
| **Aufmerksamkeit (Attention)** | ja | **nein** |
| **Zeit-Kodierung** | gelernt | **fest** (stabiler) |
| **Hauptbauteil** | mehrere zusammenspielende Module | fast nur **MLPs** |

Weniger bewegliche Teile bedeutet: weniger, was schiefgehen kann, schnelleres Training, einfacheres Debugging. Genau das macht GraphMixer zur idealen, soliden Baseline.

---

## 7. Wie GraphMixer in unser Projekt passt

- **Rolle:** Temporal-Graph-Baseline für den **binären Vergleich** (Link ja/nein).
- **Eingabe:** der **temporale Graph** aus einzelnen, zeitgestempelten Fahrten `(Start, Ziel, Zeitpunkt)` – aus dem Original-Citi-Bike-Datensatz für Mai–Juni 2024, gefiltert auf dieselben aktiven Stationen wie unser Hauptdatensatz.
- **Ausgabe:** pro Stationspaar und Zeitfenster eine Wahrscheinlichkeit für „Verbindung ja/nein".
- **Vergleich:** GraphMixer (nur binär) gegen den **Binär-Kopf unseres Hybridmodells**. Für unser Modell gilt dabei: `Anzahl Fahrten > 0` ⇒ „Link = ja".
- **Metriken:** AUC, AP, Accuracy/F1 (siehe [[Konzeptdokument.md|Konzeptdokument]]).

Wichtig: GraphMixer macht **keine Zählvorhersage** (wie viele Fahrten). Den Count-Vergleich übernimmt die separate **LSTM-Baseline**. GraphMixer ist also bewusst nur für die *eine* der beiden Aufgaben zuständig.

---

## 8. Stärken und Grenzen

**Stärken**
- Einfach zu implementieren und zu trainieren (kaum komplexe Module).
- Stabil dank fester Zeit-Kodierung.
- Trotz Einfachheit sehr konkurrenzfähig – eine faire, ernstzunehmende Baseline.
- Nutzt die zeitliche Information echter Einzel-Ereignisse, nicht nur Aggregate.

**Grenzen**
- Kein explizites Langzeit-Gedächtnis pro Knoten (betrachtet nur die letzten K Ereignisse).
- Rein **binär** – beantwortet nicht „wie viele Fahrten".
- Nutzt **keine kontinuierlichen Knoten-Zeitreihen** (wie Fahrrad-Verfügbarkeit). Genau diese Lücke füllt unser Hybridmodell – und ist der Grund, warum wir erwarten, GraphMixer beim hybriden Problem zu übertreffen.

---

## 9. Kurz-Zusammenfassung (für den eiligen Leser)

- GraphMixer sagt voraus, ob zwischen zwei Knoten bald eine Verbindung entsteht.
- Es schaut auf die **letzten Ereignisse** beider Knoten, kodiert „wann" mit einer **festen** Zeitfunktion und verdichtet alles mit einem **MLP-Mixer** (zwei einfache Misch-Schritte: über Ereignisse und über Merkmale).
- Drei Bausteine: **Link-Encoder** (jüngste Aktivität), **Node-Encoder** (Gesamtbild), **Link-Klassifikator** (Vorhersage).
- Bewusst **einfacher als TGN** (kein Gedächtnis, keine Attention, keine RNNs) – ideal als Baseline.
- In unserem Projekt: **binäre Temporal-Graph-Baseline** auf den zeitgestempelten Einzelfahrten.
