# -*- coding: utf-8 -*-
"""
make_ablation_comparison.py — summarizes the grid-search ablation.
==================================================================

Reads the *_best.csv files from results/ (best config per model, evaluated on
the test split) and puts them next to the DEFAULT results (the untuned
reference). Writes results/ablation_comparison.md with two tables (binary /
count) plus the selected best configurations.

Usage:  python make_ablation_comparison.py
"""
from __future__ import annotations
import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_HERE, "results")

# --- DEFAULT reference (untuned) from results/comparison.md, test split -------
DEFAULT = {
    "Hybrid-GRU": dict(auc=0.985, ap=0.923, f1=0.859, acc=0.952,
                       mse=0.082, mae=0.092, rmse=0.287),
    "GraphMixer": dict(auc=0.909, ap=0.653, f1=0.515, acc=0.708),
    "LSTM":       dict(mse=0.239, mae=0.187, rmse=0.489),
}


def _read(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _f(x):
    return "—" if x is None else f"{x:.3f}"


def main():
    hyb = _read("grid_hybrid_best.csv")     # Hybrid-GRU (HPO), Hybrid-CNN (HPO)
    lstm = _read("grid_lstm_best.csv")
    gm = _read("grid_graphmixer_best.csv")

    def hyb_row(tag):
        if hyb is None:
            return None
        m = hyb[hyb["model"].str.contains(tag)]
        return m.iloc[0] if len(m) else None

    gru_hpo = hyb_row("GRU")
    cnn_hpo = hyb_row("CNN")
    lstm_hpo = None if lstm is None else lstm.iloc[0]
    gm_hpo = None if gm is None else gm.iloc[0]

    lines = []
    lines.append("# Ablation study – grid-search hyperparameter optimization\n")
    lines.append("Every model was tuned with the **same search-depth philosophy** via "
                 "grid search. Hyperparameters were selected **from the validation "
                 "metric only**; the test split was computed **once** per model "
                 "(final config). All runs use the same `shared_eval` protocol and "
                 "seed 42.\n")

    # --- binary ---------------------------------------------------------------
    lines.append("## Comparison 1 – binary (link yes/no), test set\n")
    lines.append("| Model | Tuning | AUC | AP | F1 | Accuracy | Best config |")
    lines.append("|---|---|---|---|---|---|---|")
    d = DEFAULT["GraphMixer"]
    lines.append(f"| GraphMixer | default | {_f(d['auc'])} | {_f(d['ap'])} | "
                 f"{_f(d['f1'])} | {_f(d['acc'])} | — |")
    if gm_hpo is not None:
        lines.append(f"| GraphMixer | **HPO** | {_f(gm_hpo['test_auc'])} | "
                     f"{_f(gm_hpo['test_ap'])} | {_f(gm_hpo['test_f1'])} | "
                     f"{_f(gm_hpo['test_acc'])} | {gm_hpo['best_config']} |")
    d = DEFAULT["Hybrid-GRU"]
    lines.append(f"| Hybrid GraphSAGE+GRU | default | {_f(d['auc'])} | {_f(d['ap'])} | "
                 f"{_f(d['f1'])} | {_f(d['acc'])} | — |")
    if gru_hpo is not None:
        lines.append(f"| Hybrid GraphSAGE+GRU | **HPO** | {_f(gru_hpo['test_auc'])} | "
                     f"{_f(gru_hpo['test_ap'])} | {_f(gru_hpo['test_f1'])} | "
                     f"{_f(gru_hpo['test_acc'])} | {gru_hpo['best_config']} |")
    if cnn_hpo is not None:
        lines.append(f"| Hybrid GraphSAGE+1D-CNN | **HPO** | {_f(cnn_hpo['test_auc'])} | "
                     f"{_f(cnn_hpo['test_ap'])} | {_f(cnn_hpo['test_f1'])} | "
                     f"{_f(cnn_hpo['test_acc'])} | {cnn_hpo['best_config']} |")

    # --- count ----------------------------------------------------------------
    lines.append("\n## Comparison 2 – count (number of rides), test set\n")
    lines.append("| Model | Tuning | MSE | MAE | RMSE | Best config |")
    lines.append("|---|---|---|---|---|---|")
    d = DEFAULT["LSTM"]
    lines.append(f"| LSTM | default | {_f(d['mse'])} | {_f(d['mae'])} | "
                 f"{_f(d['rmse'])} | — |")
    if lstm_hpo is not None:
        lines.append(f"| LSTM | **HPO** | {_f(lstm_hpo['test_mse'])} | "
                     f"{_f(lstm_hpo['test_mae'])} | {_f(lstm_hpo['test_rmse'])} | "
                     f"{lstm_hpo['best_config']} |")
    d = DEFAULT["Hybrid-GRU"]
    lines.append(f"| Hybrid GraphSAGE+GRU | default | {_f(d['mse'])} | {_f(d['mae'])} | "
                 f"{_f(d['rmse'])} | — |")
    if gru_hpo is not None:
        lines.append(f"| Hybrid GraphSAGE+GRU | **HPO** | {_f(gru_hpo['test_mse'])} | "
                     f"{_f(gru_hpo['test_mae'])} | {_f(gru_hpo['test_rmse'])} | "
                     f"{gru_hpo['best_config']} |")
    if cnn_hpo is not None:
        lines.append(f"| Hybrid GraphSAGE+1D-CNN | **HPO** | {_f(cnn_hpo['test_mse'])} | "
                     f"{_f(cnn_hpo['test_mae'])} | {_f(cnn_hpo['test_rmse'])} | "
                     f"{cnn_hpo['best_config']} |")

    lines.append("\n## Search spaces\n")
    lines.append("| Model | Search space | # configs | Selection metric (val) |")
    lines.append("|---|---|---|---|")
    lines.append("| GraphMixer | lr {1e-3, 3e-4, 1e-4} × hidden {64,128,256} × mixer_layers {1,2} | 18 | AP |")
    lines.append("| LSTM | lr {1e-3, 3e-4, 1e-4} × hidden {32,64,128} × num_layers {1,2} | 18 | MSE |")
    lines.append("| Hybrid GRU | lr {1e-3, 3e-4, 1e-4} × hidden {32,64,128} × λ_count {0.5,1,2} | 27 | AP |")
    lines.append("| Hybrid 1D-CNN | lr {1e-3, 3e-4, 1e-4} × hidden {32,64,128} × kernel {3,5} | 18 | AP |")

    lines.append("\n## Detailed logs\n")
    lines.append("Full per-config validation metrics: "
                 "`grid_graphmixer.csv`, `grid_lstm.csv`, `grid_hybrid_gru.csv`, "
                 "`grid_hybrid_cnn.csv` (all in `results/`).")

    out = os.path.join(RES, "ablation_comparison.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"written: {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
