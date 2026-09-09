#!/usr/bin/env python3
"""Figures 2 and 3 of the paper, from the frozen CSVs and rank dumps.

  F2  per-family filtered MRR, GAT (mean ± std over 3 seeds) vs ULTRA-3g, families ordered
      from our strongest to weakest; hatching = the family is in the model's pretraining corpus.
  F3  FBNELL rank histogram: ours (GAT, one panel per seed collapsed into one: seed 0 shown,
      seeds 1-2 as thin outlines) and, when a dump is available, ULTRA-3g (right panel).

Usage:  uv run --with matplotlib python scripts/figures.py [--ultra_ranks PATH]
Outputs: paper_short/figures/F2_families.{pdf,png}, F3_fbnell_ranks.{pdf,png}
"""
import argparse
import collections
import csv
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PS = os.path.join(ROOT, "results")
RES = os.path.join(ROOT, "checkpoints")
OUT = os.path.join(ROOT, "figures")

BLUE, ORANGE = "#2a78d6", "#eb6834"      # categorical slots 1 and 2 (validated pair)
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "hatch.linewidth": 0.5,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

FAM = {"Freebase": "Freebase", "WordNet": "WordNet", "NELL": "NELL", "Wikidata": "Wikidata",
       "INDIGO-HM": "Hetionet-\nderived", "Metafam": "Metafam", "Freebase+NELL": "FBNELL"}


def load():
    rows = [r for r in csv.DictReader(open(os.path.join(PS, "broad_eval_matrix.csv"))) if r["in_scope"] == "1"]
    for r in rows:
        r["mrr"] = float(r["mrr"]); r["ultra"] = float(r["mrr_ultra3g_local"])
    return rows


def fig2(rows):
    fams = sorted({r["family"] for r in rows})
    data = []
    for f in fams:
        sp = sorted({r["split"] for r in rows if r["family"] == f})
        per_seed = [st.mean(r["mrr"] for r in rows if r["backbone"] == "gat" and r["seed"] == s and r["split"] in sp) for s in "012"]
        ultra = st.mean(next(r for r in rows if r["split"] == x)["ultra"] for x in sp)
        m = next(r for r in rows if r["family"] == f)
        data.append(dict(fam=f, n=len(sp), gat=st.mean(per_seed), sd=st.stdev(per_seed), ultra=ultra,
                         seen_us=m["regime_reifm"] != "new-KG", seen_ultra=m["regime_ultra3g"] != "new-KG"))
    data.sort(key=lambda d: -d["gat"])
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    w = 0.38
    for i, d in enumerate(data):
        ax.bar(i - w / 2, d["gat"], w, color=BLUE, edgecolor="white", linewidth=0.8,
               hatch="////" if d["seen_us"] else None, yerr=d["sd"], error_kw=dict(ecolor=INK, elinewidth=0.8, capsize=2))
        ax.bar(i + w / 2, d["ultra"], w, color=ORANGE, edgecolor="white", linewidth=0.8,
               hatch="////" if d["seen_ultra"] else None)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([f"{FAM[d['fam']]}\n({d['n']})" for d in data])
    ax.set_ylabel("filtered MRR")
    ax.set_ylim(0, 0.62)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
    handles = [Patch(facecolor=BLUE, label="GAT, ours (mean ± std, 3 seeds)"),
               Patch(facecolor=ORANGE, label="ULTRA-3g (local re-run)"),
               Patch(facecolor="white", edgecolor=MUTED, hatch="////", label="family in the model's pretraining corpus")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=3,
              handlelength=1.6, columnspacing=1.2, borderaxespad=0.0)
    i_fb = [d["fam"] for d in data].index("Freebase+NELL")
    ax.annotate("isolated entities,\n§5.3", (i_fb, data[i_fb]["ultra"] + 0.02), ha="center", va="bottom", fontsize=7, color=MUTED)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "F2_families.pdf")); fig.savefig(os.path.join(OUT, "F2_families.png"), dpi=200)
    plt.close(fig)


def load_ranks(path):
    if path.endswith(".npz"):          # ULTRA dump (scripts/ultra_dump_ranks.py): rank_tail + rank_head
        import numpy as np
        z = np.load(path)
        return [int(x) for x in np.concatenate([z["rank_tail"], z["rank_head"]])]
    d = json.load(open(path))
    rk = d["ranks"]
    if isinstance(rk, dict):
        rk = rk.get("FBNELL:FBNELL_v1", list(rk.values())[0])
    if rk and isinstance(rk[0], dict):
        rk = [x["rank"] for x in rk]
    return [int(x) for x in rk]


def hist(ax, ranks, color, title, kmax=30):
    c = collections.Counter(min(r, kmax + 1) for r in ranks)
    xs = list(range(1, kmax + 2))
    ys = [c.get(x, 0) for x in xs]
    ax.bar(xs, ys, width=0.8, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks([1, 5, 6, 10, 15, 20, 25, kmax + 1])
    ax.set_xticklabels(["1", "5", "6", "10", "15", "20", "25", f">{kmax}"])
    ax.set_xlim(0.3, kmax + 1.8)
    ax.set_title(title, loc="left")
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
    n = len(ranks); mrr = st.mean(1 / r for r in ranks)
    ax.set_ylim(0, max(ys) * 1.28)
    ax.text(0.98, 0.97, f"n = {n:,}   MRR {mrr:.3f}   Hits@1 {sum(r == 1 for r in ranks) / n:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    return c


def fig3(ultra_path=None):
    ours = [load_ranks(os.path.join(RES, f"gat_seed{s}", "s2_ext_FBNELL_FBNELL_v1_ranks.json")) for s in "012"]
    panels = 2 if ultra_path else 1
    fig, axes = plt.subplots(1, panels, figsize=(6.4 if panels == 2 else 4.2, 2.4), sharey=True, squeeze=False)
    ax = axes[0, 0]
    c = hist(ax, ours[0], BLUE, "(a) GAT (ours), seed 0")
    # seeds 1 and 2 as thin outlines
    for s in (1, 2):
        cc = collections.Counter(min(r, 31) for r in ours[s])
        ax.step([x - 0.5 for x in range(1, 33)], [cc.get(x, 0) for x in range(1, 33)], where="post",
                color=INK, linewidth=0.6, alpha=0.6)
    ax.axvspan(0.5, 5.5, color="#f2f2f2", zorder=0)
    ax.text(3, max(c.values()) * 0.72, "ranks\n1–5:\nthe five\nisolated\nentities", ha="center", va="center", fontsize=6.5, color=MUTED)
    ax.annotate(f"{c.get(5, 0)} queries\nat rank 5", (5, c.get(5, 0)), (2.6, max(c.values()) * 0.28),
                fontsize=6.5, color=MUTED, ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_ylabel("queries")
    ax.set_xlabel("filtered rank of the true answer")
    if ultra_path:
        ur = load_ranks(ultra_path)
        hist(axes[0, 1], ur, ORANGE, "(b) ULTRA-3g, same queries")
        axes[0, 1].set_xlabel("filtered rank of the true answer")
        iso_idx = [i for i, r in enumerate(ours[0]) if r == 5]          # the six rank-5 queries of panel (a)
        iso_r = sorted(ur[i] for i in iso_idx)
        axes[0, 1].text(0.98, 0.84, "the six rank-5 queries of (a):\nranks " + ", ".join(str(r) for r in iso_r),
                        transform=axes[0, 1].transAxes, ha="right", va="top", fontsize=6.5, color=MUTED)
    fig.savefig(os.path.join(OUT, "F3_fbnell_ranks.pdf")); fig.savefig(os.path.join(OUT, "F3_fbnell_ranks.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ultra_ranks", default=None, help="ULTRA-3g FBNELL rank dump (JSON) for panel (b)")
    a = ap.parse_args()
    rows = load()
    fig2(rows)
    fig3(a.ultra_ranks)
    print("written", sorted(os.listdir(OUT)))
