# -*- coding: utf-8 -*-
"""
Vergleichs-/Reporting-Skript fuer alle Modelle.
===============================================

Sammelt die exportierten Vorhersagen (predictions/*.csv) von GraphMixer, LSTM und
Hybridmodell ein und erzeugt die zwei finalen Vergleichstabellen ueber das
gemeinsame Eval-Modul:

  Vergleich 1 (binaer): Hybrid vs. GraphMixer   -> AUC, AP, F1, Accuracy
  Vergleich 2 (count) : Hybrid vs. LSTM         -> MSE, MAE, RMSE

Fehlende Vorhersagen (Modell noch nicht gelaufen) werden uebersprungen und als
"-" markiert. Ergebnis: Konsole + results/comparison.md

Aufruf:  python compare_models.py
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "evaluation"))
from shared_eval import SharedLinkEval   # noqa: E402

SPLITS = ["val", "test"]

# Modell -> (Pfadvorlage relativ zum Repo, welche Aufgaben es liefert)
MODELS = {
    "GraphMixer": ("graphmixer/model/predictions/graphmixer_pred_{split}.csv", {"binary"}),
    "LSTM":       ("lstm/predictions/lstm_pred_{split}.csv",                    {"count"}),
    "Hybrid":     ("hybrid_model/predictions/hybrid_pred_{split}.csv",          {"binary", "count"}),
}


def _load(path_tmpl, split):
    fp = os.path.join(HERE, path_tmpl.format(split=split))
    return pd.read_csv(fp) if os.path.isfile(fp) else None


def evaluate_all(ev):
    res = {}   # (model, split) -> dict mit Metriken
    for name, (tmpl, tasks) in MODELS.items():
        for split in SPLITS:
            df = _load(tmpl, split)
            if df is None:
                continue
            entry = {}
            if "binary" in tasks and "score" in df.columns:
                entry.update(ev.score_binary(df, split=split))
            if "count" in tasks and "pred_count" in df.columns:
                entry.update(ev.score_count(df, split=split))
            res[(name, split)] = entry
    return res


def _fmt(res, model, split, key, nd=3):
    e = res.get((model, split))
    if not e or key not in e:
        return "  -  "
    return f"{e[key]:.{nd}f}"


def binary_table(res):
    rows = ["## Vergleich 1 – Binär (Link ja/nein)", "",
            "| Modell | Split | AUC | AP | F1 | Accuracy |",
            "|---|---|---|---|---|---|"]
    for model in ["Hybrid", "GraphMixer"]:
        for split in SPLITS:
            rows.append(f"| {model} | {split} | {_fmt(res,model,split,'auc')} | "
                        f"{_fmt(res,model,split,'ap')} | {_fmt(res,model,split,'f1')} | "
                        f"{_fmt(res,model,split,'accuracy')} |")
    return "\n".join(rows)


def count_table(res):
    rows = ["## Vergleich 2 – Count (Fahrtenzahl)", "",
            "| Modell | Split | MSE | MAE | RMSE |",
            "|---|---|---|---|---|"]
    for model in ["Hybrid", "LSTM"]:
        for split in SPLITS:
            rows.append(f"| {model} | {split} | {_fmt(res,model,split,'mse')} | "
                        f"{_fmt(res,model,split,'mae')} | {_fmt(res,model,split,'rmse')} |")
    return "\n".join(rows)


def main():
    ev = SharedLinkEval()
    res = evaluate_all(ev)

    present = sorted({m for (m, _s) in res})
    missing = [m for m in MODELS if m not in present]
    status = (f"Vorhandene Vorhersagen: {', '.join(present) if present else 'keine'}"
              + (f" | fehlend: {', '.join(missing)}" if missing else ""))

    out = "# Modellvergleich\n\n" + status + "\n\n" + binary_table(res) + "\n\n" + count_table(res) + "\n"
    print(out)

    res_dir = os.path.join(HERE, "results"); os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(out)
    print(f"[geschrieben] {os.path.join(res_dir, 'comparison.md')}")


if __name__ == "__main__":
    main()
