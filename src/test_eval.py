"""测试集最终评估: 各模型在完整训练集上训练 → 官方测试集 (835 AMP + 835 非AMP) 评估。
用法: python -I src/test_eval.py <amp_train> <nonamp_train> <amp_test> <nonamp_test> <outdir> [--device auto] [--epochs 20]"""
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             matthews_corrcoef, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from baseline import read_fasta
from features import featurize
from dnn import AA_IDX, MAXLEN, CNN, BiLSTM, CNNBiLSTMAtt, encode

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 42


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, p),
        "f1": f1_score(y, pred),
        "matthews_corrcoef": matthews_corrcoef(y, pred),
        "sensitivity": recall_score(y, pred),
        "specificity": recall_score(1 - y, 1 - pred),
    }


def fit_dl(net, X, L, y, device, epochs, batch=64):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(L), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        net.train()
        for xb, lb, yb in loader:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(net(xb, lb), yb)
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.from_numpy(X).to(device), torch.from_numpy(L).to(device))).cpu().numpy()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("amp_train")
    ap.add_argument("nonamp_train")
    ap.add_argument("amp_test")
    ap.add_argument("nonamp_test")
    ap.add_argument("outdir")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"device={device}")

    tr_amp, tr_non = read_fasta(args.amp_train), read_fasta(args.nonamp_train)
    te_amp, te_non = read_fasta(args.amp_test), read_fasta(args.nonamp_test)
    Xtr = featurize(tr_amp + tr_non)
    ytr = np.array([1] * len(tr_amp) + [0] * len(tr_non))
    Xte = featurize(te_amp + te_non)
    yte = np.array([1] * len(te_amp) + [0] * len(te_non))
    print(f"train={len(ytr)}, test={len(yte)}, d={Xtr.shape[1]}")

    rows = []

    def report(name, y, p, t0):
        m = metrics(y, p)
        rows.append({"model": name, **{f"{k}": v for k, v in m.items()},
                     "seconds": round(time.time() - t0, 1)})
        print(f"[{name}] acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f} f1={m['f1']:.4f} "
              f"mcc={m['matthews_corrcoef']:.4f} sen={m['sensitivity']:.4f} spe={m['specificity']:.4f} "
              f"({time.time() - t0:.0f}s)")

    # ---- 传统 ML ----
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=4)
    rf.fit(Xtr, ytr)
    report("RandomForest", yte, rf.predict_proba(Xte)[:, 1], t0)

    t0 = time.time()
    svm = Pipeline([("scale", StandardScaler()),
                    ("svc", CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=42), cv=3))])
    svm.fit(Xtr, ytr)
    report("SVM(RBF)", yte, svm.predict_proba(Xte)[:, 1], t0)

    t0 = time.time()
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8,
                        tree_method="hist", n_jobs=4, random_state=42)
    xgb.fit(Xtr, ytr)
    report("XGBoost", yte, xgb.predict_proba(Xte)[:, 1], t0)

    # ---- 深度学习 (GPU) ----
    x_tr, l_tr = encode(tr_amp + tr_non)
    x_te, l_te = encode(te_amp + te_non)
    ytr_f, yte_f = ytr.astype(np.float32), yte.astype(np.float32)
    for name, make in [("CNN", CNN), ("BiLSTM", BiLSTM), ("CNN-BiLSTM-Att", CNNBiLSTMAtt)]:
        t0 = time.time()
        net = make().to(device)
        p_tr = fit_dl(net, x_tr, l_tr, ytr_f, device, args.epochs)
        torch.save(net.state_dict(), outdir / "models" / f"final_{name}.pt")
        report(name, yte, fit_dl(net, x_te, l_te, yte_f, device, 0), t0)
        # 上面 fit_dl(epochs=0) 直接复用训练好的 net 做推理

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "test_metrics.csv", index=False)
    print(f"\n结果已保存: {outdir / 'test_metrics.csv'}")


if __name__ == "__main__":
    main()
