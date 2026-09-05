# -*- coding: utf-8 -*-
"""
Animated flow map over one day: reality next to prediction, 48 frames.

Reads the compact CSV from demo_score_day.py. The map frame is fixed across
all frames so the eye can track stations; the strip at the bottom shows where
in the day we are and how the hit rate moves with it.

The model does not predict individual bikes -- it predicts, per station pair
and per 30-minute window, whether a trip happens and how many. So the lines are
flows, not vehicles.

    python ablation/demo_animate.py --day 27
    python ablation/demo_animate.py --day 27 --lang en
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

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(os.path.dirname(HERE), "prepared Data")
RES = os.path.join(HERE, "results")

NAVY, TEAL, TERRA, MUTED = "#13334C", "#028090", "#B85042", "#64748B"
MINT, AMBER, BG = "#02C39A", "#F59E0B", "#F1F5F9"
BORDER = "#CBD5E1"

T0 = dt.date(2024, 5, 16)
DE_WD = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
EN_WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

TXT = dict(
    de=dict(real="Realität", pred="Vorhersage (Top-K)",
            prof="Verbindungen je 30-Minuten-Fenster",
            conn="%d Verbindungen", hit="%d/%d getroffen  (%.2f)",
            foot="Linien = Stationspaare, keine einzelnen Fahrräder · "
                 "Dicke ∝ Fahrtenzahl · K = Zahl der echten Verbindungen"),
    en=dict(real="Reality", pred="Prediction (top K)",
            prof="Connections per 30-minute window",
            conn="%d connections", hit="%d/%d hit  (%.2f)",
            foot="Lines = station pairs, not individual bikes · "
                 "width ∝ trip count · K = number of real connections"),
)


def project(lat, lon, lat0, lon0):
    return ((lon - lon0) * np.cos(np.radians(lat0)) * 111.32,
            (lat - lat0) * 110.57)


def edges_patches(ax, edges, X, Y, color, wscale, alpha):
    for u, i, w in edges:
        ax.add_patch(FancyArrowPatch(
            (X[u], Y[u]), (X[i], Y[i]), connectionstyle="arc3,rad=0.14",
            arrowstyle="-", linewidth=min(3.2, 0.7 + wscale * w),
            color=color, alpha=alpha, zorder=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=27)
    ap.add_argument("--lang", choices=["de", "en"], default="de")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--no_footnote", action="store_true",
                    help="drop the static caveat lines (put them on the slide "
                         "instead); makes the maps bigger")
    ap.add_argument("--out_suffix", default="")
    args = ap.parse_args()
    day, lang, L = args.day, args.lang, TXT[args.lang]

    d = pd.read_csv(os.path.join(RES, f"demo_day{day}_scored.csv"))
    static = np.load(os.path.join(PREP, "node_static.npy"))
    lat, lon = static[:, 1], static[:, 2]

    bins = sorted(d.bin_idx.unique())
    date = T0 + dt.timedelta(days=day)
    wd = (DE_WD if lang == "de" else EN_WD)[date.weekday()]
    datestr = date.strftime("%d.%m.%Y" if lang == "de" else "%Y-%m-%d")

    # per-bin edge lists
    frames = []
    for b in bins:
        s = d[d.bin_idx == b]
        real = s[s.label == 1]
        top = s[s.in_topk == 1]
        hits = int(((s.in_topk == 1) & (s.label == 1)).sum())
        frames.append(dict(
            b=b,
            real=list(zip(real.u, real.i, real["count"])),
            pred=list(zip(top.u, top.i, top.pred_count.clip(upper=4))),
            n_real=len(real), K=len(top), hits=hits))

    # One fixed frame for the whole day. A handful of real trips run across the
    # Hudson into Manhattan (3 of 3307 on day 27) and would stretch the frame to
    # 9 km, leaving most of it empty. Cut on a quantile of the edge endpoints
    # instead and report how many edges that drops.
    ends = np.concatenate([d.u.values, d.i.values])
    lat0, lon0 = lat[ends].mean(), lon[ends].mean()
    X, Y = project(lat, lon, lat0, lon0)
    qlo, qhi = 0.002, 0.998
    pad = 0.35
    x0 = np.quantile(X[ends], qlo) - pad; x1 = np.quantile(X[ends], qhi) + pad
    y0 = np.quantile(Y[ends], qlo) - pad; y1 = np.quantile(Y[ends], qhi) + pad
    in_box = np.where((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1))[0]
    inside = set(in_box.tolist())

    n_real_tot = int((d.label == 1).sum())
    n_real_vis = int(d[(d.label == 1) & d.u.isin(inside) & d.i.isin(inside)].shape[0])
    n_drop = n_real_tot - n_real_vis
    print(f"  frame {x1-x0:.1f} x {y1-y0:.1f} km | "
          f"{n_drop} of {n_real_tot} real edges outside the frame")

    n_real_all = np.array([f["n_real"] for f in frames])

    # The two map panels are near-square (equal aspect), so widening the figure
    # only adds whitespace instead of enlarging them. Shrink it vertically
    # instead: with --no_footnote the static caveats move onto the slide and
    # the maps get a bigger share of the frame.
    h = 6.35 if args.no_footnote else 6.9
    bot = 0.055 if args.no_footnote else 0.135
    fig = plt.figure(figsize=(12.4, h), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[5.3, 0.9],
                          hspace=0.23, wspace=0.05,
                          left=0.02, right=0.98, top=0.895, bottom=bot)
    axL = fig.add_subplot(gs[0, 0]); axR = fig.add_subplot(gs[0, 1])
    axP = fig.add_subplot(gs[1, :])
    sup = fig.suptitle("", fontsize=15, color=NAVY, fontweight="bold", y=0.965)
    foot = L["foot"]
    if n_drop:
        foot += (("\n%d der %d Verbindungen (%.1f %%) liegen außerhalb des "
                  "Ausschnitts und werden nicht gezeigt"
                  % (n_drop, n_real_tot, 100 * n_drop / n_real_tot))
                 if lang == "de" else
                 ("\n%d of %d connections (%.1f%%) fall outside the frame "
                  "and are not shown"
                  % (n_drop, n_real_tot, 100 * n_drop / n_real_tot)))
    if not args.no_footnote:
        fig.text(0.5, 0.012, foot, ha="center", va="bottom", fontsize=9.5,
                 color=MUTED, style="italic", linespacing=1.5)

    def draw(k):
        f = frames[k]
        for ax, title, edges, col, stat in (
                (axL, L["real"], f["real"], TEAL, L["conn"] % f["n_real"]),
                (axR, L["pred"], f["pred"], MINT,
                 L["hit"] % (f["hits"], f["K"], f["hits"] / max(1, f["K"])))):
            ax.clear()
            ax.set_facecolor("white")
            ax.set_title(title, fontsize=12.5, color=NAVY, fontweight="bold", pad=8)
            ax.scatter(X[in_box], Y[in_box], s=8, color=BORDER, zorder=1, linewidths=0)
            touched = sorted({n for e in edges for n in e[:2]})
            if touched:
                ax.scatter(X[touched], Y[touched], s=22, color=NAVY, zorder=3,
                           edgecolor="white", linewidth=0.6)
            edges_patches(ax, edges, X, Y, col, 0.55, 0.75)
            ax.text(0.02, 0.02, stat, transform=ax.transAxes, fontsize=10.5,
                    color=MUTED, va="bottom", ha="left")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(BORDER)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")

        # daily profile strip
        axP.clear(); axP.set_facecolor("white")
        cols = [TEAL if j == k else BORDER for j in range(len(frames))]
        # the bar alone is invisible at night, so mark the position as well
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
    suffix = ("" if lang == "de" else "_en") + args.out_suffix
    out = os.path.join(RES, f"demo_day{day}{suffix}.gif")
    anim.save(out, writer=PillowWriter(fps=args.fps))
    size = os.path.getsize(out) / 1e6
    print(f"wrote {out}  ({len(frames)} frames, {size:.1f} MB, "
          f"{len(frames)/args.fps:.1f} s)")


if __name__ == "__main__":
    main()
