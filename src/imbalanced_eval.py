"""非平衡压力测试: 在 AMPlify 非平衡测试集 (835 AMP vs 25689 非AMP) 上评估代表性模型。
用已保存模型: models/xgb_full.joblib + models/final_*.pt (需先跑 save_models.py 与 test_eval.py)。
用法: python -I src/imbalanced_eval.py <nonamp_test_imbalanced.fa> <outdir>"""
import sys
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score)

sys.path.insert(0, str(Path(__file__).parent))
from baseline import read_fasta
from features import featurize
from dnn import MAXLEN, CNNBiLSTMAtt, encode

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def dl_predict(weight_path, seqs, batch=1024):
    net = CNNBiLSTMAtt()
    net.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    net.eval()
    ps = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        x, l = encode(chunk)
        with torch.no_grad():
            p = torch.sigmoid(net(torch.from_numpy(x), torch.from_numpy(l))).numpy()
        ps.append(p)
    return np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nonamp_test_imbalanced")
    ap.add_argument("outdir")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    base = Path(__file__).parent.parent

    amp = read_fasta(base / "data" / "raw" / "AMPlify_AMP_test_common.fa")
    non = read_fasta(args.nonamp_test_imbalanced)
    print(f"AMP={len(amp)}, Non-AMP={len(non)}, 正负比=1:{len(non) / len(amp):.0f}")

    y = np.array([1] * len(amp) + [0] * len(non))
    seqs = amp + non

    t0 = time.time()
    X = featurize(seqs)
    xgb = joblib.load(base / "models" / "xgb_full.joblib")
    p_xgb = xgb.predict_proba(X)[:, 1]
    print(f"XGBoost 推理完成 ({time.time() - t0:.0f}s)")

    from sklearn.metrics import precision_recall_curve

    def report(name, yv, pv):
        pred = (pv >= 0.5).astype(int)
        pr, rc, ths = precision_recall_curve(yv, pv)
        f1s = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
        best = int(np.argmax(f1s))
        best_th = ths[best] if best < len(ths) else 0.5
        pred_b = (pv >= best_th).astype(int)
        print(f"[{name}] auc={roc_auc_score(yv, pv):.4f} pr_auc={average_precision_score(yv, pv):.4f}")
        print(f"  阈值0.5: f1={f1_score(yv, pred):.4f} precision={precision_score(yv, pred):.4f} "
              f"recall={recall_score(yv, pred):.4f}")
        print(f"  最优阈值{best_th:.3f}: f1={f1_score(yv, pred_b):.4f} "
              f"precision={precision_score(yv, pred_b):.4f} recall={recall_score(yv, pred_b):.4f}")

    for name, yv, pv in [("XGBoost", y, p_xgb)]:
        report(name, yv, pv)

    wpath = base / "models" / "final_CNN-BiLSTM-Att.pt"
    if wpath.exists():
        t0 = time.time()
        p_dl = dl_predict(wpath, seqs)
        print(f"CNN-BiLSTM-Att 推理完成 ({time.time() - t0:.0f}s)")
        report("CNN-BiLSTM-Att", y, p_dl)
    else:
        print(f"跳过 CNN-BiLSTM-Att: {wpath} 不存在 (先跑 test_eval.py)")


if __name__ == "__main__":
    main()
