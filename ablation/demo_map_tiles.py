# -*- coding: utf-8 -*-
"""
Three-panel flow map on a street basemap.

Tile variant of demo_map.py, which stays as it is. Both read the same scored
CSV, so this costs no extra compute. Separate file because demo_map.py uses a
local flat projection (lon x cos(lat)) that would drift against the tiles;
everything here is Web Mercator.

    python ablation/demo_map_tiles.py --bin 1332 --provider osm
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

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from basemap import basemap, lonlat_to_mercator

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(os.path.dirname(HERE), "prepared Data")
RES = os.path.join(HERE, "results")

NAVY, TEAL, TERRA, MUTED = "#13334C", "#00707F", "#B02A1F", "#64748B"
MINT, AMBER, BG = "#00A88A", "#E08A00", "#F1F5F9"
BORDER = "#CBD5E1"
# Grey at 35 % opacity is invisible on a street basemap, so true positives get
# their own colour here (it works fine on white in demo_map.py).
HIT_BLUE = "#1B6CA8"

T0 = dt.date(2024, 5, 16)
DE_WD = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
EN_WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def bin_label(b, lang="de"):
    day = T0 + dt.timedelta(days=b // 48)
    hh, mm = (b % 48) // 2, (b % 48) % 2 * 30
    wd = (DE_WD if lang == "de" else EN_WD)[day.weekday()]
    fmt = "%d.%m.%Y" if lang == "de" else "%Y-%m-%d"
    return f"{wd} {day.strftime(fmt)}, {hh:02d}:{mm:02d}"


def draw_edges(ax, edges, X, Y, color, wscale, alpha=0.85):
    for u, i, w in edges:
        ax.add_patch(FancyArrowPatch(
            (X[u], Y[u]), (X[i], Y[i]), connectionstyle="arc3,rad=0.14",
            arrowstyle="-", linewidth=min(3.4, 0.9 + wscale * w),
            color=color, alpha=alpha, zorder=4))


def panel(ax, title, edges, color, X, Y, in_box, stats, img, ext, veil):
    ax.imshow(np.asarray(img), extent=ext, origin="upper", zorder=0,
              interpolation="bilinear")
    # a white veil keeps the streets readable while letting the flows dominate
    ax.add_patch(plt.Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2],
                               facecolor="white", alpha=veil, zorder=1, lw=0))
    ax.set_title(title, fontsize=12.5, color=NAVY, fontweight="bold", pad=9)
    ax.scatter(X[in_box], Y[in_box], s=9, color="#94A3B8", zorder=2,
               linewidths=0, alpha=0.7)
    touched = sorted({n for e in edges for n in e[:2]})
    if touched:
        ax.scatter(X[touched], Y[touched], s=30, color=NAVY, zorder=5,
                   edgecolor="white", linewidth=0.9)
    draw_edges(ax, edges, X, Y, color, 0.55)
    ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=9.5,
            color=NAVY, va="bottom", ha="left", linespacing=1.5, zorder=6,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=BORDER, alpha=0.85))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(BORDER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=int, default=1332)
    ap.add_argument("--lang", choices=["de", "en"], default="de")
    ap.add_argument("--provider", default="gray",
                    choices=["gray", "osm", "osm_de"])
    ap.add_argument("--veil", type=float, default=0.35,
                    help="0 = raw map, 1 = white. Higher makes flows stand out.")
    args = ap.parse_args()
    b, lang = args.bin, args.lang

    g = pd.read_csv(os.path.join(RES, f"demo_bin{b}_scored.csv"))
    static = np.load(os.path.join(PREP, "node_static.npy"))
    lat, lon = static[:, 1], static[:, 2]
    X, Y = lonlat_to_mercator(lon, lat)

    truth = g[g.label == 1]
    K = len(truth)
    top = g.nlargest(K, "score")
    tp_mask = top.label == 1
    n_tp = int(tp_mask.sum())
    t_edges = list(zip(truth.u, truth.i, truth["count"]))
    p_edges = list(zip(top.u, top.i, top.pred_count.clip(upper=4)))

    inv = sorted(set(truth.u) | set(truth.i) | set(top.u) | set(top.i))
    pad = 400.0                                   # metres
    x0, x1 = X[inv].min() - pad, X[inv].max() + pad
    y0, y1 = Y[inv].min() - pad, Y[inv].max() + pad
    in_box = np.where((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1))[0]

    # tiles need the box back in degrees, with a little margin
    m = 0.004
    img, ext, attribution = basemap(lon[inv].min() - m, lat[inv].min() - m,
                                    lon[inv].max() + m, lat[inv].max() + m,
                                    provider=args.provider)

    L = dict(
        de=dict(sup="Realität und Vorhersage im selben 30-Minuten-Fenster",
                t1="Realität", t2="Vorhersage (Top-%d)" % K, t3="Abgleich",
                s1="%d Verbindungen\n%d Fahrten" % (K, int(truth["count"].sum())),
                s2="%d höchstbewertete Paare\ndavon %d korrekt" % (K, n_tp),
                s3="%d richtig positiv (Precision@%d = %.2f)\n"
                   "%d falsch positiv, %d falsch negativ"
                   % (n_tp, K, n_tp / K, K - n_tp, K - n_tp),
                foot="Jede Linie = ein Stationspaar · Dicke ∝ Fahrtenzahl · "
                     "Jersey City / Hoboken, ca. %.1f × %.1f km · %s",
                # standard classification terms rather than "hit / missed /
                # false alarm"
                hit="richtig positiv", miss="falsch negativ", fp="falsch positiv"),
        en=dict(sup="Reality and prediction for the same 30-minute window",
                t1="Reality", t2="Prediction (top %d)" % K, t3="Comparison",
                s1="%d connections\n%d trips" % (K, int(truth["count"].sum())),
                s2="%d highest-scoring pairs\n%d of them correct" % (K, n_tp),
                s3="%d true positive (precision@%d = %.2f)\n"
                   "%d false positive, %d false negative"
                   % (n_tp, K, n_tp / K, K - n_tp, K - n_tp),
                foot="Each line = one station pair · width ∝ trip count · "
                     "Jersey City / Hoboken, about %.1f × %.1f km · %s",
                hit="true positive", miss="false negative", fp="false positive"),
    )[lang]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.9), facecolor=BG)
    fig.suptitle(f"{L['sup']}   —   {bin_label(b, lang)}",
                 fontsize=15, color=NAVY, fontweight="bold", y=0.975)

    panel(axes[0], L["t1"], t_edges, TEAL, X, Y, in_box, L["s1"], img, ext, args.veil)
    panel(axes[1], L["t2"], p_edges, MINT, X, Y, in_box, L["s2"], img, ext, args.veil)

    ax = axes[2]
    hits = list(zip(top.u[tp_mask], top.i[tp_mask], np.ones(n_tp)))
    fps = list(zip(top.u[~tp_mask], top.i[~tp_mask], np.ones(K - n_tp)))
    tset = set(zip(top.u, top.i))
    fns = [(u, i, 1.0) for u, i in zip(truth.u, truth.i) if (u, i) not in tset]
    panel(ax, L["t3"], [], TEAL, X, Y, in_box, L["s3"], img, ext, args.veil)
    draw_edges(ax, hits, X, Y, HIT_BLUE, 0.9, alpha=0.75)
    draw_edges(ax, fns, X, Y, AMBER, 0.0, alpha=0.9)
    draw_edges(ax, fps, X, Y, TERRA, 0.0, alpha=0.8)
    touched = sorted({n for e in hits + fps + fns for n in e[:2]})
    ax.scatter(X[touched], Y[touched], s=30, color=NAVY, zorder=5,
               edgecolor="white", linewidth=0.9)
    for lab, col in ((L["hit"], HIT_BLUE), (L["miss"], AMBER), (L["fp"], TERRA)):
        ax.plot([], [], color=col, lw=2.6, label=lab)
    ax.legend(loc="upper left", fontsize=9.5, frameon=True, facecolor="white",
              edgecolor=BORDER, borderpad=0.5, framealpha=0.92).set_zorder(7)

    for a in axes:
        a.set_xlim(x0, x1); a.set_ylim(y0, y1); a.set_aspect("equal")

    km_x, km_y = (x1 - x0) / 1000, (y1 - y0) / 1000
    fig.text(0.5, 0.022, L["foot"] % (km_x, km_y, attribution), ha="center",
             fontsize=9.5, color=MUTED, style="italic")
    plt.tight_layout(rect=[0, 0.045, 1, 0.945])
    suffix = "" if lang == "de" else "_en"
    out = os.path.join(RES, f"demo_map_tiles_{args.provider}_bin{b}{suffix}.png")
    plt.savefig(out, dpi=165, facecolor=BG, bbox_inches="tight")
    print("wrote", out)
    print(f"  K={K}  hits={n_tp}  precision@K={n_tp/K:.3f}")


if __name__ == "__main__":
    main()
