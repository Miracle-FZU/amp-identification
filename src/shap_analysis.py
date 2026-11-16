"""XGBoost + SHAP 特征重要性分析。
用法: python -I src/shap_analysis.py <amp_train.fa> <nonamp_train.fa> <outdir>"""
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from baseline import read_fasta
from features import featurize, feature_names

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C1 = "#2a78d6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 1.0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False,
})


def human_name(f):
    """AAC_K -> K; DPC_KK -> dipep KK; CK2_KK -> KK gap2 (图表用英文标签, 防缺字形)"""
    if f.startswith("AAC_"):
        return f[4:]
    if f.startswith("DPC_"):
        return f"dipep {f[4:]}"
    if f.startswith("CK"):
        k = f[2:].split("_")[0]
        return f"{f.split('_')[1]} gap{k}"
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("amp")
    ap.add_argument("nonamp")
    ap.add_argument("outdir")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    amp, non = read_fasta(args.amp), read_fasta(args.nonamp)
    X = featurize(amp + non)
    y = np.array([1] * len(amp) + [0] * len(non))
    names = feature_names()
    print(f"拟合 XGBoost (n={len(y)}, d={X.shape[1]}) ...")
    model = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8,
                          tree_method="hist", n_jobs=4, random_state=42)
    model.fit(X, y)

    rng = np.random.default_rng(42)
    pos_idx = rng.choice(np.where(y == 1)[0], 250, replace=False)
    neg_idx = rng.choice(np.where(y == 0)[0], 250, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])

    print("计算 SHAP ...")
    explainer = shap.TreeExplainer(model)
    sv = explainer(X[idx])
    mean_abs = np.abs(sv.values).mean(0)
    order = np.argsort(mean_abs)[::-1][:20]

    # ---- 图1: top20 |SHAP| 条形图 ----
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=200)
    vals = mean_abs[order][::-1]
    labs = [human_name(names[i]) for i in order][::-1]
    ypos = np.arange(len(labs))
    ax.barh(ypos, vals, height=0.62, color=C1)
    ax.set_yticks(ypos, labs)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Top-20 features by SHAP importance (XGBoost)", loc="left", fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=1.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(outdir / "figures" / "fig4_shap_top20.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 图2: beeswarm (SHAP 惯例配色) ----
    shap.plots.beeswarm(sv[:, order], max_display=20, show=False,
                        color_bar_label="Feature value (low → high)")
    fig = plt.gcf()
    fig.set_facecolor(SURFACE)
    ax = plt.gca()
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=MUTED)
    ax.set_xlabel("SHAP value (impact on AMP probability)", color=INK2)
    fig.savefig(outdir / "figures" / "fig5_shap_beeswarm.png", bbox_inches="tight")
    plt.close(fig)

    print("\nTop-20 特征 (mean|SHAP| 与方向):")
    for i in order:
        mean_v = sv.values[:, i].mean()
        direction = "↑促AMP" if mean_v > 0 else "↓抑AMP"
        print(f"  {names[i]:16s} ({human_name(names[i]):14s}) mean|SHAP|={mean_abs[i]:.4f}  {direction}")
    print(f"图表已保存到 {outdir / 'figures'}")


if __name__ == "__main__":
    main()
