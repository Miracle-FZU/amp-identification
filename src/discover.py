"""应用案例: 用 ESM-2 扫描牛蛙蛋白组证据集, 发现候选抗菌肽片段。
用法: HF_ENDPOINT=https://hf-mirror.com python -I src/discover.py <proteins.fa> <outdir>"""
import sys
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from transformers import AutoTokenizer, EsmForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent))
from baseline import read_fasta
from esm import tokenize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ESM_NAME = "facebook/esm2_t12_35M_UR50D"
WINDOWS = [20, 30, 40]
STEP = 10
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def digest(seqs, windows=WINDOWS, step=STEP):
    frags = []
    for si, s in enumerate(seqs):
        if len(s) < min(windows):
            continue
        for L in windows:
            for i in range(0, len(s) - L + 1, step):
                frags.append((f"{si}:{i + 1}-{i + L}", s[i:i + L]))
    return frags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proteins")
    ap.add_argument("outdir")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    base = Path(__file__).parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seqs = read_fasta(args.proteins)
    print(f"蛋白数={len(seqs)}")
    seqs = [s for s in seqs if set(s) <= STANDARD_AA]
    frags = digest(seqs)
    print(f"候选片段={len(frags)}")

    print("ESM-2 扫描 ...")
    tok = AutoTokenizer.from_pretrained(ESM_NAME, local_files_only=True)
    model = EsmForSequenceClassification.from_pretrained(ESM_NAME, num_labels=2, local_files_only=True)
    model.load_state_dict(torch.load(base / "reports" / "models" / "final_ESM2.pt", map_location=device))
    model = model.to(device)
    model.eval()
    names, frag_seq = zip(*frags)
    ps = []
    with torch.no_grad():
        for i in range(0, len(frag_seq), 128):
            ids, att = tokenize(list(frag_seq[i:i + 128]), tok, 64)
            logits = model(torch.from_numpy(ids).to(device),
                           attention_mask=torch.from_numpy(att).to(device)).logits
            ps.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    p = np.concatenate(ps)

    # 筛选: ESM 高置信 + 阳离子性
    keep = []
    for name, seq, prob in zip(names, frag_seq, p):
        if prob < 0.9:
            continue
        pa = ProteinAnalysis(seq)
        if pa.charge_at_pH(7.0) < 2.0:
            continue
        keep.append((name, seq, prob))
    print(f"通过筛选 (P≥0.9 且电荷≥+2) = {len(keep)}")

    # 同一蛋白只保留最高分片段
    best = {}
    for name, seq, prob in keep:
        pid = name.split(":")[0]
        if pid not in best or prob > best[pid][2]:
            best[pid] = (name, seq, prob)
    ranked = sorted(best.values(), key=lambda x: -x[2])
    print(f"去冗余后候选蛋白={len(ranked)}")

    # top20 用 XGBoost 复核
    top = ranked[:20]
    if top:
        from features import featurize
        xgb = joblib.load(base / "models" / "xgb_full.joblib")
        px = xgb.predict_proba(featurize([t[1] for t in top]))[:, 1]
    else:
        px = []

    rows = []
    print(f"\n{'候选片段':<22s} {'长度':>4s} {'电荷':>6s} {'ESM-2':>7s} {'XGB':>6s}")
    for (name, seq, prob), pxv in zip(top, px):
        pa = ProteinAnalysis(seq)
        rows.append({"fragment": name, "sequence": seq, "length": len(seq),
                     "charge": round(pa.charge_at_pH(7.0), 2),
                     "esm_prob": round(float(prob), 4), "xgb_prob": round(float(pxv), 4)})
        print(f"{name:<22s} {len(seq):4d} {pa.charge_at_pH(7.0):+6.2f} {prob:7.1%} {pxv:6.1%}")
    pd.DataFrame(rows).to_csv(outdir / "discovery_candidates.csv", index=False)
    print(f"\n已保存 {outdir / 'discovery_candidates.csv'}")


if __name__ == "__main__":
    main()
