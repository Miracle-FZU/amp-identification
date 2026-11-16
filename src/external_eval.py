"""跨数据库外部验证: 在 8 个独立外部测试集上评估已保存的 XGBoost 与 CNN-BiLSTM-Att。
用法: python -I src/external_eval.py <outdir>"""
import sys
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).parent))
from baseline import read_fasta
from features import featurize
from dnn import CNNBiLSTMAtt, encode

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def dl_predict(net, seqs, batch=1024):
    ps = []
    for i in range(0, len(seqs), batch):
        x, l = encode(seqs[i:i + batch])
        with torch.no_grad():
            p = torch.sigmoid(net(torch.from_numpy(x), torch.from_numpy(l))).numpy()
        ps.append(p)
    return np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    base = Path(__file__).parent.parent
    ext = base / "data" / "external"

    pairs = {}
    for f in sorted(ext.glob("*.fasta")):
        stem = f.stem.replace("_AMP", "").replace("_nonAMP", "")
        pairs.setdefault(stem, {})["amp" if "_AMP" in f.stem else "non"] = f

    xgb = joblib.load(base / "models" / "xgb_full.joblib")
    net = CNNBiLSTMAtt()
    net.load_state_dict(torch.load(base / "models" / "final_CNN-BiLSTM-Att.pt",
                                   map_location="cpu", weights_only=True))
    net.eval()

    rows = []
    print(f"{'数据集':12s} {'n_AMP':>6s} {'n_non':>6s} | XGB acc/auc/f1/mcc | DL acc/auc/f1/mcc")
    for stem in sorted(pairs, key=lambda s: s.upper()):
        amp = read_fasta(pairs[stem]["amp"])
        non = read_fasta(pairs[stem]["non"])
        seqs = amp + non
        y = np.array([1] * len(amp) + [0] * len(non))
        if len(seqs) == 0 or len(set(y)) < 2:
            print(f"{stem:12s} 数据异常, 跳过")
            continue
        X = featurize(seqs)
        p_x = xgb.predict_proba(X)[:, 1]
        p_d = dl_predict(net, seqs)

        def m(p):
            pred = (p >= 0.5).astype(int)
            return (f"{accuracy_score(y, pred):.3f}/{roc_auc_score(y, p):.3f}/"
                    f"{f1_score(y, pred):.3f}/{matthews_corrcoef(y, pred):.3f}")

        rows.append({"dataset": stem, "n_AMP": len(amp), "n_nonAMP": len(non),
                     "xgb_acc": accuracy_score(y, (p_x >= .5).astype(int)),
                     "xgb_auc": roc_auc_score(y, p_x),
                     "dl_acc": accuracy_score(y, (p_d >= .5).astype(int)),
                     "dl_auc": roc_auc_score(y, p_d)})
        print(f"{stem:12s} {len(amp):6d} {len(non):6d} | XGB {m(p_x)} | DL {m(p_d)}")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "external_metrics.csv", index=False)
    print(f"\n结果已保存: {outdir / 'external_metrics.csv'}")


if __name__ == "__main__":
    main()
