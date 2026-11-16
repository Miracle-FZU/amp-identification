"""传统 ML 基线: RF / SVM / XGBoost, 5 折分层交叉验证(仅训练集, 测试集留到最终评估)。
用法: python -I src/baseline.py <amp_train.fa> <nonamp_train.fa> <outdir>"""
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import featurize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCORING = ["accuracy", "roc_auc", "f1", "matthews_corrcoef"]


def read_fasta(path):
    seqs, buf, has = [], [], False
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if has:
                seqs.append("".join(buf).upper())
            has, buf = True, []
        else:
            buf.append(line)
    if has:
        seqs.append("".join(buf).upper())
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("amp")
    ap.add_argument("nonamp")
    ap.add_argument("outdir")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"sklearn {sklearn.__version__}, xgboost {xgboost.__version__}")
    amp, non = read_fasta(args.amp), read_fasta(args.nonamp)
    X = featurize(amp + non)
    y = np.array([1] * len(amp) + [0] * len(non))
    print(f"样本={len(y)} (正{len(amp)}/负{len(non)}), 特征维度={X.shape[1]}")
    print(f"特征组分: AAC 20 + DPC 400 + CKSAAP(k=1..3) 1200 + 理化 6\n")

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=1),
        "SVM(RBF)": Pipeline([
            ("scale", StandardScaler()),
            ("svc", CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=42), cv=3)),
        ]),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8,
                                 tree_method="hist", n_jobs=1, random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        t0 = time.time()
        res = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=4)
        dt = time.time() - t0
        row = {"model": name}
        for m in SCORING:
            row[f"{m}_mean"] = res[f"test_{m}"].mean()
            row[f"{m}_std"] = res[f"test_{m}"].std()
        row["seconds"] = round(dt, 1)
        rows.append(row)
        print(f"[{name}] {dt:.0f}s")
        for m in SCORING:
            print(f"  {m:22s} = {res[f'test_{m}'].mean():.4f} ± {res[f'test_{m}'].std():.4f}")
        print()

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "baseline_metrics.csv", index=False)
    print(f"结果已保存: {outdir / 'baseline_metrics.csv'}")


if __name__ == "__main__":
    main()
