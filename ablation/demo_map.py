# -*- coding: utf-8 -*-
"""
Three-panel flow map for one 30-minute window: reality, prediction, error.

Stage 2 of the demo. Reads the CSV that demo_score_bin.py writes, so the
picture can be re-tweaked without retraining.

The middle panel shows the model's top-K pairs, where K = the number of trips
that actually happened. Equal number of lines left and right, so the two panels
are visually comparable, and the hit rate is precision@K. A fixed threshold is
the wrong choice here: the model is calibrated for the 1:5 evaluation protocol
(16.7 % positives), while a full grid is 0.35 % positives, so at 0.5 it would
draw ~3,400 lines and nothing would be readable.

    python ablation/demo_map.py --bin 1332
"""
import os
import argparse
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(os.path.dirname(HERE), "prepared Data")
RES = os.path.join(HERE, "results")

NAVY, TEAL, TERRA, MUTED = "#13334C", "#028090", "#B85042", "#64748B"
MINT, AMBER, BG, GREEN = "#02C39A", "#F59E0B", "#F1F5F9", "#10B981"
BORDER = "#CBD5E1"

T0 = dt.date(2024, 5, 16)
DE_WD = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
EN_WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def bin_label(b, lang="de"):
    day = T0 + dt.timedelta(days=b // 48)
    hh, mm = (b % 48) // 2, (b % 48) % 2 * 30
    wd = (DE_WD if lang == "de" else EN_WD)[day.weekday()]
    fmt = "%d.%m.%Y" if lang == "de" else "%Y-%m-%d"
    return f"{wd} {day.strftime(fmt)}, {hh:02d}:{mm:02d}"


def project(lat, lon, lat0, lon0):
    """Local equirectangular projection in km -> equal aspect is honest."""
    x = (lon - lon0) * np.cos(np.radians(lat0)) * 111.32
    y = (lat - lat0) * 110.57
    return x, y


def draw_edges(ax, edges, X, Y, color, wscale, alpha=0.75):
    """edges: iterable of (u, i, weight). Curved so u->i and i->u stay apart."""
    for u, i, w in edges:
        ax.add_patch(FancyArrowPatch(
            (X[u], Y[u]), (X[i], Y[i]),
            connectionstyle="arc3,rad=0.14",
            arrowstyle="-", linewidth=min(3.2, 0.7 + wscale * w),
            color=color, alpha=alpha, zorder=2))


def panel(ax, title, edges, color, X, Y, in_box, stats, wscale=0.55):
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=12.5, color=NAVY, fontweight="bold", pad=9)
    # context: every station in the frame
    ax.scatter(X[in_box], Y[in_box], s=9, color=BORDER, zorder=1, linewidths=0)
    # stations touched by these edges
    touched = sorted({n for e in edges for n in e[:2]})
    if touched:
        ax.scatter(X[touched], Y[touched], s=26, color=NAVY, zorder=3,
                   edgecolor="white", linewidth=0.7)
    draw_edges(ax, edges, X, Y, color, wscale)
    ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=9.5,
            color=MUTED, va="bottom", ha="left", linespacing=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(BORDER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=int, default=1332)
    ap.add_argument("--lang", choices=["de", "en"], default="de")
    args = ap.parse_args()
    b, lang = args.bin, args.lang

    g = pd.read_csv(os.path.join(RES, f"demo_bin{b}_scored.csv"))
    static = np.load(os.path.join(PREP, "node_static.npy"))
    lat, lon = static[:, 1], static[:, 2]

    truth = g[g.label == 1]
    K = len(truth)
    top = g.nlargest(K, "score")
    tp_mask = top.label == 1
    n_tp = int(tp_mask.sum())

    t_edges = list(zip(truth.u, truth.i, truth["count"]))
    p_edges = list(zip(top.u, top.i, top.pred_count.clip(upper=4)))

    # frame: stations involved either way, plus padding
    inv = sorted(set(truth.u) | set(truth.i) | set(top.u) | set(top.i))
    lat0, lon0 = lat[inv].mean(), lon[inv].mean()
    X, Y = project(lat, lon, lat0, lon0)
    pad = 0.35
    x0, x1 = X[inv].min() - pad, X[inv].max() + pad
    y0, y1 = Y[inv].min() - pad, Y[inv].max() + pad
    in_box = np.where((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1))[0]

    L = dict(
        de=dict(sup="Realität und Vorhersage im selben 30-Minuten-Fenster",
                t1="Realität", t2="Vorhersage (Top-%d)" % K, t3="Abgleich",
                s1="%d Verbindungen\n%d Fahrten" % (K, int(truth["count"].sum())),
                s2="%d höchstbewertete Paare\ndavon %d korrekt" % (K, n_tp),
                s3="%d getroffen (Precision@%d = %.2f)\n%d falsch, %d verpasst"
                   % (n_tp, K, n_tp / K, K - n_tp, K - n_tp),
                foot="Jede Linie = ein Stationspaar · Dicke ∝ Fahrtenzahl · "
                     "Kartenausschnitt Jersey City / Hoboken, ca. %.1f × %.1f km"),
        en=dict(sup="Reality and prediction for the same 30-minute window",
                t1="Reality", t2="Prediction (top %d)" % K, t3="Comparison",
                s1="%d connections\n%d trips" % (K, int(truth["count"].sum())),
                s2="%d highest-scoring pairs\n%d of them correct" % (K, n_tp),
                s3="%d hit (precision@%d = %.2f)\n%d wrong, %d missed"
                   % (n_tp, K, n_tp / K, K - n_tp, K - n_tp),
                foot="Each line = one station pair · width ∝ trip count · "
                     "Jersey City / Hoboken, about %.1f × %.1f km"),
    )[lang]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.9), facecolor=BG)
    fig.suptitle(f"{L['sup']}   —   {bin_label(b, lang)}",
                 fontsize=15, color=NAVY, fontweight="bold", y=0.975)

    panel(axes[0], L["t1"], t_edges, TEAL, X, Y, in_box, L["s1"])
    panel(axes[1], L["t2"], p_edges, MINT, X, Y, in_box, L["s2"])

    # panel 3: hits muted, misses and false alarms in colour
    ax = axes[2]
    hits = list(zip(top.u[tp_mask], top.i[tp_mask], np.ones(n_tp)))
    fps = list(zip(top.u[~tp_mask], top.i[~tp_mask], np.ones(K - n_tp)))
    tset = set(zip(top.u, top.i))
    fns = [(u, i, 1.0) for u, i in zip(truth.u, truth.i) if (u, i) not in tset]
    panel(ax, L["t3"], [], TEAL, X, Y, in_box, L["s3"])
    draw_edges(ax, hits, X, Y, MUTED, 0.0, alpha=0.30)
    draw_edges(ax, fns, X, Y, AMBER, 0.0, alpha=0.85)
    draw_edges(ax, fps, X, Y, TERRA, 0.0, alpha=0.70)
    touched = sorted({n for e in hits + fps + fns for n in e[:2]})
    ax.scatter(X[touched], Y[touched], s=26, color=NAVY, zorder=3,
               edgecolor="white", linewidth=0.7)
    for lab, col in ((("getroffen" if lang == "de" else "hit"), MUTED),
                     (("verpasst" if lang == "de" else "missed"), AMBER),
                     (("Fehlalarm" if lang == "de" else "false alarm"), TERRA)):
        ax.plot([], [], color=col, lw=2.4, label=lab)
    ax.legend(loc="upper left", fontsize=9.5, frameon=True, facecolor="white",
              edgecolor=BORDER, borderpad=0.5, framealpha=0.92)

    for a in axes:
        a.set_xlim(x0, x1); a.set_ylim(y0, y1); a.set_aspect("equal")

    fig.text(0.5, 0.022, L["foot"] % (x1 - x0, y1 - y0), ha="center",
             fontsize=10, color=MUTED, style="italic")
    plt.tight_layout(rect=[0, 0.045, 1, 0.945])
    suffix = "" if lang == "de" else "_en"
    out = os.path.join(RES, f"demo_map_bin{b}{suffix}.png")
    plt.savefig(out, dpi=165, facecolor=BG, bbox_inches="tight")
    print("wrote", out)
    print(f"  K={K}  hits={n_tp}  precision@K={n_tp/K:.3f}")


if __name__ == "__main__":
    main()
