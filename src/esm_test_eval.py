"""ESM-2 测试集最终评估: 完整训练集上微调 → 官方测试集评估, 保存 final_ESM2.pt。
用法: HF_ENDPOINT=https://hf-mirror.com python -I src/esm_test_eval.py <amp_train> <nonamp_train> <amp_test> <nonamp_test> <outdir>"""
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, EsmForSequenceClassification
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, recall_score

sys.path.insert(0, str(Path(__file__).parent))
from baseline import read_fasta
from esm import tokenize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("amp_train")
    ap.add_argument("nonamp_train")
    ap.add_argument("amp_test")
    ap.add_argument("nonamp_test")
    ap.add_argument("outdir")
    ap.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--maxlen", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"device={device}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = EsmForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)

    tr_amp, tr_non = read_fasta(args.amp_train), read_fasta(args.nonamp_train)
    te_amp, te_non = read_fasta(args.amp_test), read_fasta(args.nonamp_test)
    tr_ids, tr_att = tokenize(tr_amp + tr_non, tok, args.maxlen)
    te_ids, te_att = tokenize(te_amp + te_non, tok, args.maxlen)
    ytr = torch.from_numpy(np.array([1] * len(tr_amp) + [0] * len(tr_non), dtype=np.float32))
    yte = np.array([1] * len(te_amp) + [0] * len(te_non))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(tr_ids), torch.from_numpy(tr_att), ytr),
                        batch_size=args.batch, shuffle=True)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for ids, am, yb in loader:
            ids, am, yb = ids.to(device), am.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(ids, attention_mask=am).logits, yb.long())
            loss.backward()
            opt.step()
        print(f"epoch {ep + 1}/{args.epochs} ({time.time() - t0:.0f}s)")

    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(te_ids), args.batch * 4):
            ids = torch.from_numpy(te_ids[i:i + args.batch * 4]).to(device)
            am = torch.from_numpy(te_att[i:i + args.batch * 4]).to(device)
            ps.append(torch.softmax(model(ids, attention_mask=am).logits, dim=1)[:, 1].cpu().numpy())
    p = np.concatenate(ps)
    pred = (p >= 0.5).astype(int)
    print(f"[ESM-2 测试集] acc={accuracy_score(yte, pred):.4f} auc={roc_auc_score(yte, p):.4f} "
          f"f1={f1_score(yte, pred):.4f} mcc={matthews_corrcoef(yte, pred):.4f} "
          f"sen={recall_score(yte, pred):.4f} spe={recall_score(1 - yte, 1 - pred):.4f}")
    torch.save(model.state_dict(), outdir / "models" / "final_ESM2.pt")
    print(f"已保存 {outdir / 'models' / 'final_ESM2.pt'}")


if __name__ == "__main__":
    main()
