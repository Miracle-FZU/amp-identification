"""ESM-2 预训练蛋白语言模型微调, 5 折 CV(与基线同种子 42)。
用法: HF_ENDPOINT=https://hf-mirror.com python -I src/esm.py <amp_train.fa> <nonamp_train.fa> <outdir> [--model facebook/esm2_t12_35M_UR50D] [--epochs 4] [--batch 32]"""
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, EsmForSequenceClassification
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from baseline import read_fasta

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 42


def tokenize(seqs, tok, maxlen):
    enc = tok(list(seqs), padding="max_length", truncation=True, max_length=maxlen, return_tensors="np")
    return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)


def train_fold(model, tr_loader, va_loader, device, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    best = {"auc": 0.0, "state": None, "y": None, "p": None}
    for _ in range(epochs):
        model.train()
        for ids, am, yb in tr_loader:
            ids, am, yb = ids.to(device), am.to(device), yb.to(device)
            opt.zero_grad()
            out = model(ids, attention_mask=am)
            loss = lossf(out.logits, yb.long())
            loss.backward()
            opt.step()
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for ids, am, yb in va_loader:
                ids, am = ids.to(device), am.to(device)
                logits = model(ids, attention_mask=am).logits
                ys.append(yb.numpy())
                ps.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
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
    ap.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--maxlen", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"加载模型 {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = EsmForSequenceClassification.from_pretrained(args.model, num_labels=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}, 参数量={n_params / 1e6:.1f}M")

    amp, non = read_fasta(args.amp), read_fasta(args.nonamp)
    seqs = amp + non
    y = np.array([1] * len(amp) + [0] * len(non), dtype=np.float32)
    ids, att = tokenize(seqs, tok, args.maxlen)
    ids_t = torch.from_numpy(ids)
    att_t = torch.from_numpy(att)
    y_t = torch.from_numpy(y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    t0 = time.time()
    fold_res = []
    for fi, (tr, va) in enumerate(cv.split(ids, y)):
        model.to(device)
        tr_loader = DataLoader(TensorDataset(ids_t[tr], att_t[tr], y_t[tr]), batch_size=args.batch, shuffle=True)
        va_loader = DataLoader(TensorDataset(ids_t[va], att_t[va], y_t[va]), batch_size=args.batch * 4)
        best = train_fold(model, tr_loader, va_loader, device, args.epochs, args.lr)
        m = metrics(best["y"], best["p"])
        fold_res.append(m)
        torch.save(best["state"], outdir / "models" / f"ESM2-35M_fold{fi}.pt")
        print(f"  fold{fi}: acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f} f1={m['f1']:.4f} "
              f"mcc={m['matthews_corrcoef']:.4f} ({time.time() - t0:.0f}s)")
        model.cpu()

    row = {"model": args.model.split("/")[-1]}
    for k in fold_res[0]:
        vals = [r[k] for r in fold_res]
        row[f"{k}_mean"] = float(np.mean(vals))
        row[f"{k}_std"] = float(np.std(vals))
    row["seconds"] = round(time.time() - t0, 1)
    print(f"[ESM-2] acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} "
          f"auc={row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f} "
          f"f1={row['f1_mean']:.4f} mcc={row['matthews_corrcoef_mean']:.4f} 总耗时 {row['seconds']}s")

    import pandas as pd
    pd.DataFrame([row]).to_csv(outdir / "esm_metrics.csv", index=False)
    print(f"结果已保存: {outdir / 'esm_metrics.csv'}")


if __name__ == "__main__":
    main()
