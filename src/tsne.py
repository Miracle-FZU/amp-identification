"""t-SNE 嵌入可视化: 手工特征空间 vs 训练好的 CNN-BiLSTM-Att 嵌入空间。
用法: python -I src/tsne.py <amp_train.fa> <nonamp_train.fa> <outdir>"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from baseline import read_fasta
from features import featurize
from dnn import AA_IDX, MAXLEN, CNNBiLSTMAtt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C1, C2 = "#2a78d6", "#eb6834"


class EmbModel(CNNBiLSTMAtt):
    """导出注意力池化后的嵌入向量"""

    def embed(self, x, lens):
        h = F.relu(self.conv(self.emb(x).transpose(1, 2))).transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(h, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        w = torch.softmax(self.att(out).squeeze(2), dim=1)
        return (out * w.unsqueeze(2)).sum(1)


def encode(seqs):
    x = np.zeros((len(seqs), MAXLEN), dtype=np.int64)
    lens = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        l = min(len(s), MAXLEN)
        lens[i] = l
        for j in range(l):
            x[i, j] = AA_IDX.get(s[j], 20)
    return x, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("amp")
    ap.add_argument("nonamp")
    ap.add_argument("outdir")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    amp, non = read_fasta(args.amp), read_fasta(args.nonamp)
    seqs = amp + non
    y = np.array([1] * len(amp) + [0] * len(non))

    rng = np.random.default_rng(42)
    idx = np.concatenate([rng.choice(np.where(y == 1)[0], 600, replace=False),
                          rng.choice(np.where(y == 0)[0], 600, replace=False)])
    ys = y[idx]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 1.0,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "legend.frameon": False,
    })

    print("计算手工特征 t-SNE ...")
    X = featurize(seqs)[idx]
    Xs = StandardScaler().fit_transform(X)
    Xp = PCA(n_components=50, random_state=42).fit_transform(Xs)
    sil_feat = silhouette_score(Xs, ys, sample_size=400, random_state=42)
    t1 = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(Xp)
    print(f"手工特征空间 silhouette={sil_feat:.4f}")

    print("计算 DL 嵌入 t-SNE ...")
    model = EmbModel()
    model.load_state_dict(torch.load(outdir / "models" / "CNN-BiLSTM-Att_fold0.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()
    x, l = encode([seqs[i] for i in idx])
    with torch.no_grad():
        emb = model.embed(torch.from_numpy(x), torch.from_numpy(l)).numpy()
    sil_emb = silhouette_score(emb, ys, sample_size=400, random_state=42)
    t2 = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(emb)
    print(f"DL 嵌入空间 silhouette={sil_emb:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)
    for ax, (t, title) in zip(axes, [
        (t1, f"Handcrafted features (silhouette={sil_feat:.3f})"),
        (t2, f"CNN-BiLSTM-Att embeddings (silhouette={sil_emb:.3f})"),
    ]):
        for mask, color, label in [(ys == 1, C1, "AMP"), (ys == 0, C2, "Non-AMP")]:
            ax.scatter(t[mask, 0], t[mask, 1], s=18, color=color, edgecolor=SURFACE,
                       linewidth=1.2, label=label, alpha=0.85)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_axisbelow(True)
        ax.grid(color=GRID, linewidth=1.0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0, labelleft=False, labelbottom=False)
        leg = ax.legend(loc="upper right", markerscale=1.6)
        for t in leg.get_texts():
            t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "fig6_tsne.png", bbox_inches="tight")
    print(f"已保存 {outdir / 'figures' / 'fig6_tsne.png'}")


if __name__ == "__main__":
    main()
