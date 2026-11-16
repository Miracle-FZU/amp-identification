"""ESM-2 泛化评估: 非平衡压力测试 + 8 库外部验证 (加载 final_ESM2.pt, GPU 推理)。
用法: HF_ENDPOINT=https://hf-mirror.com python -I src/esm_extra_eval.py <outdir>"""
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForSequenceClassification
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, accuracy_score, precision_recall_curve)

sys.path.insert(0, str(Path(__file__).parent))
from baseline import read_fasta
from esm import tokenize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
MAXLEN = 64


def predict(model, tok, seqs, device, batch=128):
    ps = []
    for i in range(0, len(seqs), batch):
        ids, att = tokenize(seqs[i:i + batch], tok, MAXLEN)
        with torch.no_grad():
            logits = model(torch.from_numpy(ids).to(device),
                           attention_mask=torch.from_numpy(att).to(device)).logits
        ps.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    base = Path(__file__).parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, 加载模型 ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    model.load_state_dict(torch.load(outdir / "models" / "final_ESM2.pt", map_location=device))
    model.eval()

    # ---- 非平衡压力测试 ----
    amp = read_fasta(base / "data" / "raw" / "AMPlify_AMP_test_common.fa")
    non = read_fasta(base / "data" / "raw" / "AMPlify_non_AMP_test_imbalanced.fa")
    y = np.array([1] * len(amp) + [0] * len(non))
    t0 = time.time()
    p = predict(model, tok, amp + non, device)
    pr, rc, ths = precision_recall_curve(y, p)
    f1s = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    best = int(np.argmax(f1s))
    best_th = ths[best] if best < len(ths) else 0.5
    pred_b = (p >= best_th).astype(int)
    pred_05 = (p >= 0.5).astype(int)
    print(f"\n[ESM-2 非平衡 1:{len(non) // len(amp)}] ({time.time() - t0:.0f}s)")
    print(f"  auc={roc_auc_score(y, p):.4f} pr_auc={average_precision_score(y, p):.4f}")
    print(f"  阈值0.5: f1={f1_score(y, pred_05):.4f} p={precision_score(y, pred_05):.4f} r={recall_score(y, pred_05):.4f}")
    print(f"  最优阈值{best_th:.3f}: f1={f1_score(y, pred_b):.4f} p={precision_score(y, pred_b):.4f} r={recall_score(y, pred_b):.4f}")

    # ---- 外部 8 库 ----
    ext = base / "data" / "external"
    pairs = {}
    for f in sorted(ext.glob("*.fasta")):
        stem = f.stem.replace("_AMP", "").replace("_nonAMP", "")
        pairs.setdefault(stem, {})["amp" if "_AMP" in f.stem else "non"] = f
    print(f"\n{'数据集':10s} {'n':>5s} | ESM-2 acc/auc")
    for stem in sorted(pairs, key=lambda s: s.upper()):
        a = read_fasta(pairs[stem]["amp"])
        b = read_fasta(pairs[stem]["non"])
        yy = np.array([1] * len(a) + [0] * len(b))
        if len(set(yy)) < 2:
            continue
        pp = predict(model, tok, a + b, device)
        predd = (pp >= 0.5).astype(int)
        print(f"{stem:10s} {len(yy):5d} | {accuracy_score(yy, predd):.3f}/{roc_auc_score(yy, pp):.3f} "
              f"f1={f1_score(yy, predd):.3f}")


if __name__ == "__main__":
    main()
