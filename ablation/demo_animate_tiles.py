# -*- coding: utf-8 -*-
"""
Animated flow map over one day, on a real street basemap.

Map-tile variant of demo_animate.py, which stays untouched. Same scored CSV,
same 48 frames -- the difference is that the flows sit on actual streets.

The basemap is fetched once and reused for every frame; without that the run
would issue ~2,000 tile requests instead of ~50.

    python ablation/demo_animate_tiles.py --day 27 --provider osm
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
from matplotlib.animation import FuncAnimation, PillowWriter

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from basemap import basemap, lonlat_to_mercator

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(os.path.dirname(HERE), "prepared Data")
RES = os.path.join(HERE, "results")

NAVY, TEAL, MUTED = "#13334C", "#00707F", "#64748B"
MINT, BG, BORDER = "#00A88A", "#F1F5F9", "#CBD5E1"

T0 = dt.date(2024, 5, 16)
DE_WD = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
EN_WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

TXT = dict(
    de=dict(real="Realität", pred="Vorhersage (Top-K)",
            prof="Verbindungen je 30-Minuten-Fenster",
            conn="%d Verbindungen", hit="%d/%d getroffen  (%.2f)",
            foot="Linien = Stationspaare, keine einzelnen Fahrräder · Dicke ∝ Fahrtenzahl · %s"),
    en=dict(real="Reality", pred="Prediction (top K)",
            prof="Connections per 30-minute window",
            conn="%d connections", hit="%d/%d hit  (%.2f)",
            foot="Lines = station pairs, not individual bikes · width ∝ trip count · %s"),
)


def edges_patches(ax, edges, X, Y, color, wscale, alpha):
    for u, i, w in edges:
        ax.add_patch(FancyArrowPatch(
            (X[u], Y[u]), (X[i], Y[i]), connectionstyle="arc3,rad=0.14",
            arrowstyle="-", linewidth=min(3.4, 0.9 + wscale * w),
            color=color, alpha=alpha, zorder=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=27)
    ap.add_argument("--lang", choices=["de", "en"], default="de")
    ap.add_argument("--provider", default="osm", choices=["gray", "osm", "osm_de"])
    ap.add_argument("--veil", type=float, default=0.45)
    ap.add_argument("--fps", type=float, default=3.0)
    args = ap.parse_args()
    day, lang, L = args.day, args.lang, TXT[args.lang]

    d = pd.read_csv(os.path.join(RES, f"demo_day{day}_scored.csv"))
    static = np.load(os.path.join(PREP, "node_static.npy"))
    lat, lon = static[:, 1], static[:, 2]
    X, Y = lonlat_to_mercator(lon, lat)

    bins = sorted(d.bin_idx.unique())
    date = T0 + dt.timedelta(days=day)
    wd = (DE_WD if lang == "de" else EN_WD)[date.weekday()]
    datestr = date.strftime("%d.%m.%Y" if lang == "de" else "%Y-%m-%d")

    frames = []
    for b in bins:
        s = d[d.bin_idx == b]
        real = s[s.label == 1]
        top = s[s.in_topk == 1]
        hits = int(((s.in_topk == 1) & (s.label == 1)).sum())
        frames.append(dict(
            b=b, real=list(zip(real.u, real.i, real["count"])),
            pred=list(zip(top.u, top.i, top.pred_count.clip(upper=4))),
            n_real=len(real), K=len(top), hits=hits))

    # Same framing rule as the plain version: cut on a quantile of the edge
    # endpoints so a handful of cross-Hudson trips do not stretch the frame.
    ends = np.concatenate([d.u.values, d.i.values])
    qlo, qhi, pad = 0.002, 0.998, 400.0
    x0 = np.quantile(X[ends], qlo) - pad; x1 = np.quantile(X[ends], qhi) + pad
    y0 = np.quantile(Y[ends], qlo) - pad; y1 = np.quantile(Y[ends], qhi) + pad
    in_box = np.where((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1))[0]
    inside = set(in_box.tolist())
    n_tot = int((d.label == 1).sum())
    n_vis = int(d[(d.label == 1) & d.u.isin(inside) & d.i.isin(inside)].shape[0])
    n_drop = n_tot - n_vis

    keep = [n for n in range(len(lon)) if n in inside]
    m = 0.004
    img, ext, attribution = basemap(lon[keep].min() - m, lat[keep].min() - m,
                                    lon[keep].max() + m, lat[keep].max() + m,
                                    provider=args.provider)
    img = np.asarray(img)
    n_real_all = np.array([f["n_real"] for f in frames])

    fig = plt.figure(figsize=(12.4, 6.35), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[5.3, 0.9], hspace=0.23,
                          wspace=0.05, left=0.02, right=0.98, top=0.895,
                          bottom=0.105)
    axL = fig.add_subplot(gs[0, 0]); axR = fig.add_subplot(gs[0, 1])
    axP = fig.add_subplot(gs[1, :])
    sup = fig.suptitle("", fontsize=15, color=NAVY, fontweight="bold", y=0.965)
    foot = L["foot"] % attribution
    if n_drop:
        foot += (f" · {n_drop} von {n_tot} außerhalb des Ausschnitts" if lang == "de"
                 else f" · {n_drop} of {n_tot} outside the frame")
    fig.text(0.5, 0.012, foot, ha="center", va="bottom", fontsize=9,
             color=MUTED, style="italic")

    def draw(k):
        f = frames[k]
        for ax, title, edges, col, stat in (
                (axL, L["real"], f["real"], TEAL, L["conn"] % f["n_real"]),
                (axR, L["pred"], f["pred"], MINT,
                 L["hit"] % (f["hits"], f["K"], f["hits"] / max(1, f["K"])))):
            ax.clear()
            ax.imshow(img, extent=ext, origin="upper", zorder=0,
                      interpolation="bilinear")
            ax.add_patch(plt.Rectangle((ext[0], ext[2]), ext[1] - ext[0],
                                       ext[3] - ext[2], facecolor="white",
                                       alpha=args.veil, zorder=1, lw=0))
            ax.set_title(title, fontsize=12.5, color=NAVY, fontweight="bold", pad=8)
            ax.scatter(X[in_box], Y[in_box], s=8, color="#94A3B8", zorder=2,
                       linewidths=0, alpha=0.7)
            touched = sorted({n for e in edges for n in e[:2]})
            if touched:
                ax.scatter(X[touched], Y[touched], s=26, color=NAVY, zorder=5,
                           edgecolor="white", linewidth=0.7)
            edges_patches(ax, edges, X, Y, col, 0.55, 0.85)
            ax.text(0.02, 0.02, stat, transform=ax.transAxes, fontsize=10.5,
                    color=NAVY, va="bottom", ha="left", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=BORDER, alpha=0.85))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(BORDER)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")

        axP.clear(); axP.set_facecolor("white")
        cols = [TEAL if j == k else BORDER for j in range(len(frames))]
        axP.axvline(k, color=TEAL, lw=1.4, alpha=0.45, zorder=0)
        axP.bar(range(len(frames)), n_real_all, color=cols, width=0.82, zorder=2)
        axP.set_xlim(-0.7, len(frames) - 0.3)
        axP.set_xticks(range(0, len(frames), 4))
        axP.set_xticklabels([f"{(bins[j] % 48)//2:02d}" for j in range(0, len(frames), 4)],
                            fontsize=9, color=MUTED)
        axP.set_yticks([])
        axP.set_title(L["prof"], fontsize=9.5, color=MUTED, loc="left", pad=4)
        for sp in ("top", "right", "left"):
            axP.spines[sp].set_visible(False)
        axP.spines["bottom"].set_color(BORDER)

        b = f["b"]
        sup.set_text(f"{wd} {datestr},  {(b % 48)//2:02d}:{(b % 48) % 2 * 30:02d}")

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / args.fps)
    suffix = "" if lang == "de" else "_en"
    out = os.path.join(RES, f"demo_day{day}_tiles_{args.provider}{suffix}.gif")
    anim.save(out, writer=PillowWriter(fps=args.fps))
    print(f"wrote {out}  ({len(frames)} frames, "
          f"{os.path.getsize(out)/1e6:.1f} MB, {len(frames)/args.fps:.1f} s)")


if __name__ == "__main__":
    main()
