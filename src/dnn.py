"""深度学习模型: CNN / BiLSTM / CNN+BiLSTM+Attention, 5 折 CV(与基线同种子 42)。
用法: python -I src/dnn.py <amp_train.fa> <nonamp_train.fa> <outdir> [--epochs 20] [--device cuda]"""
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from baseline import read_fasta

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AAS)}
MAXLEN = 64
SEED = 42


def encode(seqs):
    x = np.zeros((len(seqs), MAXLEN), dtype=np.int64)
    lens = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        l = min(len(s), MAXLEN)
        lens[i] = l
        for j in range(l):
            x[i, j] = AA_IDX.get(s[j], 20)
    return x, lens


class CNN(nn.Module):
    def __init__(self, embed=64, filters=128, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(21, embed, padding_idx=20)
        self.convs = nn.ModuleList([nn.Conv1d(embed, filters, k, padding=k // 2) for k in (3, 5, 7)])
        self.fc = nn.Sequential(nn.Linear(filters * 3, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, 1))

    def forward(self, x, lens):
        h = self.emb(x).transpose(1, 2)  # B,E,L
        mask = torch.arange(h.size(2), device=x.device)[None, :] < lens[:, None]
        feats = []
        for c in self.convs:
            o = F.relu(c(h)).masked_fill(~mask[:, None, :], -1e9)
            feats.append(o.max(dim=2).values)
        return self.fc(torch.cat(feats, 1)).squeeze(1)


class BiLSTM(nn.Module):
    def __init__(self, embed=64, hidden=64, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(21, embed, padding_idx=20)
        self.lstm = nn.LSTM(embed, hidden, num_layers=2, bidirectional=True,
                            batch_first=True, dropout=drop)
        self.fc = nn.Sequential(nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, 1))

    def forward(self, x, lens):
        h = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(h, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        fwd, bwd = hn[-2], hn[-1]
        return self.fc(torch.cat([fwd, bwd], 1)).squeeze(1)


class CNNBiLSTMAtt(nn.Module):
    def __init__(self, embed=64, filters=128, hidden=64, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(21, embed, padding_idx=20)
        self.conv = nn.Conv1d(embed, filters, 5, padding=2)
        self.lstm = nn.LSTM(filters, hidden, bidirectional=True, batch_first=True)
        self.att = nn.Linear(hidden * 2, 1)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x, lens):
        h = F.relu(self.conv(self.emb(x).transpose(1, 2))).transpose(1, 2)  # B,L,F
        packed = nn.utils.rnn.pack_padded_sequence(h, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # B,L,2H
        w = torch.softmax(self.att(out).squeeze(2), dim=1)
        c = (out * w.unsqueeze(2)).sum(1)
        return self.fc(c).squeeze(1)


def train_fold(model, train_loader, val_loader, device, epochs, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    best = {"auc": 0.0, "state": None, "y": None, "p": None}
    for _ in range(epochs):
        model.train()
        for xb, lb, yb in train_loader:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb, lb), yb)
            loss.backward()
            opt.step()
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, lb, yb in val_loader:
                xb, lb = xb.to(device), lb.to(device)
                ys.append(yb.numpy())
                ps.append(torch.sigmoid(model(xb, lb)).cpu().numpy())
        yv, pv = np.concatenate(ys), np.concatenate(ps)
        auc = roc_auc_score(yv, pv)
        if auc > best["auc"]:
            best = {"auc": auc, "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "y": yv, "p": pv}
    return best


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, p),
        "f1": f1_score(y, pred),
        "matthews_corrcoef": matthews_corrcoef(y, pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("amp")
    ap.add_argument("nonamp")
    ap.add_argument("outdir")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} (cuda={torch.cuda.is_available()})")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    amp, non = read_fasta(args.amp), read_fasta(args.nonamp)
    over = sum(1 for s in amp + non if len(s) > MAXLEN)
    print(f"样本={len(amp) + len(non)}, MAXLEN={MAXLEN}, 截断 {over} 条({over / (len(amp) + len(non)):.2%})")
    X, L = encode(amp + non)
    y = np.array([1] * len(amp) + [0] * len(non), dtype=np.float32)
    Xt = torch.from_numpy(X)
    Lt = torch.from_numpy(L)
    yt = torch.from_numpy(y)

    models = {
        "CNN": lambda: CNN(),
        "BiLSTM": lambda: BiLSTM(),
        "CNN-BiLSTM-Att": lambda: CNNBiLSTMAtt(),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, make in models.items():
        t0 = time.time()
        fold_res = []
        for fi, (tr, va) in enumerate(cv.split(X, y)):
            model = make().to(device)
            tr_loader = DataLoader(TensorDataset(Xt[tr], Lt[tr], yt[tr]), batch_size=64, shuffle=True)
            va_loader = DataLoader(TensorDataset(Xt[va], Lt[va], yt[va]), batch_size=256)
            best = train_fold(model, tr_loader, va_loader, device, args.epochs)
            m = metrics(best["y"], best["p"])
            fold_res.append(m)
            torch.save(best["state"], outdir / "models" / f"{name}_fold{fi}.pt")
            print(f"  [{name}] fold{fi}: acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f} "
                  f"f1={m['f1']:.4f} mcc={m['matthews_corrcoef']:.4f} ({time.time() - t0:.0f}s)")
        row = {"model": name}
        for k in fold_res[0]:
            vals = [r[k] for r in fold_res]
            row[f"{k}_mean"] = float(np.mean(vals))
            row[f"{k}_std"] = float(np.std(vals))
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"[{name}] 均值: acc={row['accuracy_mean']:.4f} auc={row['roc_auc_mean']:.4f} "
              f"f1={row['f1_mean']:.4f} mcc={row['matthews_corrcoef_mean']:.4f} 总耗时 {row['seconds']}s\n")

    import pandas as pd
    pd.DataFrame(rows).to_csv(outdir / "dnn_metrics.csv", index=False)
    print(f"结果已保存: {outdir / 'dnn_metrics.csv'}")


if __name__ == "__main__":
    main()
