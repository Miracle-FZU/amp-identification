"""CNN-BiLSTM-Att 注意力热图: 对若干 AMP/非AMP 示例, 展示模型关注哪些残基位置。
用法: python -I src/attention_heatmap.py <amp_train.fa> <nonamp_train.fa> <outdir>"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from baseline import read_fasta
from dnn import AA_IDX, MAXLEN, CNNBiLSTMAtt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


class AttModel(CNNBiLSTMAtt):
    def attention(self, x, lens):
        h = F.relu(self.conv(self.emb(x).transpose(1, 2))).transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(h, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return torch.softmax(self.att(out).squeeze(2), dim=1)  # B,L


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
    rng = np.random.default_rng(7)

    def pick(seqs, k):
        pool = [s for s in seqs if 12 <= len(s) <= 40]
        chosen = [pool[i] for i in rng.choice(len(pool), k, replace=False)]
        return chosen

    model = AttModel()
    model.load_state_dict(torch.load(outdir / "models" / "CNN-BiLSTM-Att_fold0.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()

    cand = [(s, "AMP") for s in pick(amp, 12)] + [(s, "Non-AMP") for s in pick(non, 12)]
    x, l = encode([s for s, _ in cand])
    with torch.no_grad():
        logits = model(torch.from_numpy(x), torch.from_numpy(l))
    probs = torch.sigmoid(logits).numpy()
    good_amp = [cand[i] for i in range(12) if probs[i] > 0.9][:4]
    good_non = [cand[i] for i in range(12, 24) if probs[i] < 0.1][:4]
    examples = good_amp + good_non  # 只保留高置信且正确的示例

    x, l = encode([s for s, _ in examples])
    with torch.no_grad():
        w = model.attention(torch.from_numpy(x), torch.from_numpy(l)).numpy()
        logits = model(torch.from_numpy(x), torch.from_numpy(l))
        probs = torch.sigmoid(logits).numpy()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 1.0,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })

    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#104281"])
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 5.2), dpi=200,
                             gridspec_kw={"hspace": 0.55})
    vmax = max(w[i, :l[i]].max() for i in range(len(examples)))

    print("注意力峰值位置:")
    for i, ((seq, label), ax) in enumerate(zip(examples, axes)):
        ll = l[i]
        ww = w[i, :ll] / w[i, :ll].sum()  # 在真实位置上重新归一化
        ax.imshow(ww[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(ll), list(seq), fontsize=7)
        ax.set_yticks([])
        ax.set_ylabel(f"{label}-{i + 1}\nP(AMP)={probs[i]:.2f}", rotation=0, ha="right", va="center",
                      fontsize=8, color=INK2)
        top = int(np.argmax(ww))
        print(f"  {label}-{i+1}: 峰值在位置 {top+1} ({seq[top]}), 权重 {ww[top]:.3f}, P(AMP)={probs[i]:.3f}")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)

    axes[0].set_title("Attention heatmap over residue positions (CNN-BiLSTM-Att)", loc="left", fontsize=11)
    cbar = fig.colorbar(axes[0].images[0], ax=axes, shrink=0.6, pad=0.02, label="attention weight")
    cbar.ax.yaxis.label.set_color(INK2)
    cbar.ax.tick_params(length=0, colors=MUTED)
    fig.savefig(outdir / "figures" / "fig7_attention.png", bbox_inches="tight")
    print(f"已保存 {outdir / 'figures' / 'fig7_attention.png'}")


if __name__ == "__main__":
    main()
