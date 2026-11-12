#!/usr/bin/env python3
"""Figures from evaluation JSON files (snn.eval): accuracy versus time since stimulus onset, and the
accuracy/latency trade-off at the output margins.

  python -m snn.plot results/eval/<run>_test.json [more.json ...] --out results/figures/<name>
Each file contributes its configs as series; --label renames files (comma list, same order).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed categorical order (validated palette, light surface); color follows the series, never its rank
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9); ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--label", default=None)
    ap.add_argument("--only", default=None, help="per file, the configs to plot: 'async,layered1;layered10;...' (';' separates files)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    labels = a.label.split(",") if a.label else [f.stem for f in a.files]
    only = [s.split(",") if s else None for s in a.only.split(";")] if a.only else [None] * len(a.files)
    series = []                                                     # (name, eval dict)
    for f, lab, keep in zip(a.files, labels, only):
        d = json.load(open(f))
        for key, c in d["configs"].items():
            if keep is not None and key not in keep: continue
            series.append((f"{lab} / {key}" if len(a.files) > 1 else key, c, d))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=SURFACE)
    style(ax1, "time since stimulus onset (ms)", "accuracy")
    style(ax2, "decision latency, median (ms)", "accuracy at the margin")
    for k, (name, c, d) in enumerate(series):
        col = SERIES[k % len(SERIES)]
        g, acc = c["accuracy_vs_time"]["grid_ms"], c["accuracy_vs_time"]["accuracy"]
        if c["mode"] == "layered":                                   # the output only changes at bin ends: step curve
            ax1.step(g, acc, where="post", color=col, linewidth=2, label=name)
        else:
            ax1.plot(g, acc, color=col, linewidth=2, label=name)
        if len(series) <= 2:                                          # direct labels only when they cannot collide
            ax1.annotate(name, (g[-1], acc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK2, va="center")
        pts = [(m["latency_ms_median"], m["accuracy"], mk) for mk, m in c["margin_decisions"].items() if m["latency_ms_median"] is not None]
        if pts:
            ax2.plot([p[0] for p in pts], [p[1] for p in pts], color=col, linewidth=2, marker="o", markersize=4, label=name)
            x, y, mk = max(pts, key=lambda p: p[1])                      # label the best operating point (margin value) only
            ax2.annotate(f"m={mk}", (x, y), xytext=(3, 4), textcoords="offset points", fontsize=6, color=MUTED)
    ax1.set_ylim(0, 1.02); ax2.set_ylim(0, 1.02)
    ax1.legend(fontsize=7, frameon=False, loc="lower right"); ax2.legend(fontsize=7, frameon=False, loc="lower right")
    if a.title: fig.suptitle(a.title, color=INK, fontsize=10)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".pdf")); fig.savefig(a.out.with_suffix(".png"), dpi=160)
    print("saved", a.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
