"""AMP 数据集 EDA：长度分布 / 氨基酸组成 / 理化性质对比。
用法: python -I src/eda.py <amp_train.fa> <nonamp_train.fa> <figdir>"""
import sys
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# dataviz 参考调色板(浅色模式), 双系列蓝/橙已通过校验
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_AMP, C_NON = "#2a78d6", "#eb6834"
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 1.0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False,
})


def read_fasta(path):
    seqs, buf, has = [], [], False
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if has:
                seqs.append("".join(buf).upper())
            has, buf = True, []
        else:
            buf.append(line)
    if has:
        seqs.append("".join(buf).upper())
    return seqs


def style_ax(ax, grid_axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def fig1_length(amp, non, outdir):
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=200)
    sns.kdeplot(x=[len(s) for s in non], color=C_NON, label="Non-AMP",
                fill=True, alpha=0.10, linewidth=2, clip=(0, 120), ax=ax)
    sns.kdeplot(x=[len(s) for s in amp], color=C_AMP, label="AMP",
                fill=True, alpha=0.10, linewidth=2, clip=(0, 120), ax=ax)
    style_ax(ax)
    ax.set_xlabel("Sequence length (aa)")
    ax.set_ylabel("Density")
    ax.set_title("Length distribution (train set, x-axis clipped at 120 aa)", loc="left", fontsize=11)
    leg = ax.legend(loc="upper right", handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.savefig(Path(outdir) / "fig1_length.png", bbox_inches="tight")
    plt.close(fig)


def fig2_aac(amp, non, outdir):
    c_amp = Counter("".join(amp))
    c_non = Counter("".join(non))
    tot_amp, tot_non = sum(c_amp.values()), sum(c_non.values())
    aas = sorted(STANDARD_AA, key=lambda a: c_amp.get(a, 0), reverse=True)
    fig, ax = plt.subplots(figsize=(6.2, 6.8), dpi=200)
    y = np.arange(len(aas))[::-1]
    for i, aa in enumerate(aas):
        fa, fn = c_amp[aa] / tot_amp * 100, c_non[aa] / tot_non * 100
        ax.plot([fa, fn], [y[i], y[i]], color=AXIS, linewidth=1.0, zorder=1)
        ax.scatter([fa], [y[i]], color=C_AMP, s=56, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter([fn], [y[i]], color=C_NON, s=56, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_yticks(y, list(aas))
    ax.set_xlim(0, max(max(c_amp.values()) / tot_amp, max(c_non.values()) / tot_non) * 100 * 1.15)
    style_ax(ax, "x")
    ax.set_xlabel("Frequency (%)")
    ax.set_title("Amino acid composition (train set, sorted by AMP frequency)", loc="left", fontsize=11)
    handles = [
        mlines.Line2D([], [], marker="o", linestyle="", color=C_AMP, markeredgecolor=SURFACE,
                      markeredgewidth=2, markersize=7, label="AMP"),
        mlines.Line2D([], [], marker="o", linestyle="", color=C_NON, markeredgecolor=SURFACE,
                      markeredgewidth=2, markersize=7, label="Non-AMP"),
    ]
    leg = ax.legend(handles=handles, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.savefig(Path(outdir) / "fig2_aac.png", bbox_inches="tight")
    plt.close(fig)


def physchem(seqs):
    charge, pi, gravy = [], [], []
    for s in seqs:
        try:
            pa = ProteinAnalysis(s)
            charge.append(pa.charge_at_pH(7.0))
            pi.append(pa.isoelectric_point())
            gravy.append(pa.gravy())
        except Exception:
            pass
    return charge, pi, gravy


def fig3_physchem(amp, non, outdir):
    p_amp = physchem(amp)
    p_non = physchem(non)
    labels = ["Net charge at pH 7", "Isoelectric point (pI)", "GRAVY (hydrophobicity)"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), dpi=200)
    for ax, (a, b, lab) in zip(axes, zip(p_amp, p_non, labels)):
        bp = ax.boxplot([a, b], widths=0.34, patch_artist=True,
                        boxprops=dict(linewidth=2), whiskerprops=dict(linewidth=2),
                        capprops=dict(linewidth=2), medianprops=dict(color=INK, linewidth=2),
                        flierprops=dict(marker="o", markersize=3, markeredgecolor=AXIS, markerfacecolor="none"))
        for patch, color in zip(bp["boxes"], [C_AMP, C_NON]):
            patch.set_facecolor(color)
            patch.set_alpha(0.10)
            patch.set_edgecolor(color)
        ax.set_xticklabels(["AMP", "Non-AMP"])
        ax.set_title(lab, loc="left", fontsize=11)
        style_ax(ax)
    handles = [
        mlines.Line2D([], [], marker="s", linestyle="", color=C_AMP, markersize=8, label="AMP"),
        mlines.Line2D([], [], marker="s", linestyle="", color=C_NON, markersize=8, label="Non-AMP"),
    ]
    leg = axes[1].legend(handles=handles, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig3_physchem.png", bbox_inches="tight")
    plt.close(fig)
    return p_amp, p_non


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("amp")
    ap.add_argument("nonamp")
    ap.add_argument("outdir")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    amp = read_fasta(args.amp)
    non = read_fasta(args.nonamp)
    print(f"AMP={len(amp)} 条, Non-AMP={len(non)} 条")

    fig1_length(amp, non, outdir)
    fig2_aac(amp, non, outdir)
    p_amp, p_non = fig3_physchem(amp, non, outdir)

    for name, (c, pi, g) in [("AMP", p_amp), ("Non-AMP", p_non)]:
        print(f"{name}: 电荷@pH7 mean={np.mean(c):+.2f} median={np.median(c):+.2f} | "
              f"pI median={np.median(pi):.1f} | GRAVY mean={np.mean(g):+.3f}")

    c_amp = Counter("".join(amp))
    c_non = Counter("".join(non))
    tot_amp, tot_non = sum(c_amp.values()), sum(c_non.values())
    ratios = {a: (c_amp[a] / tot_amp) / (c_non[a] / tot_non) for a in STANDARD_AA}
    ranked = sorted(ratios, key=ratios.get, reverse=True)
    print("AMP 富集 top5: " + ", ".join(f"{a}:{ratios[a]:.2f}x" for a in ranked[:5]))
    print("AMP 贫化 top5: " + ", ".join(f"{a}:{ratios[a]:.2f}x" for a in ranked[-5:]))
    print(f"图表已保存到 {outdir}")


if __name__ == "__main__":
    main()
