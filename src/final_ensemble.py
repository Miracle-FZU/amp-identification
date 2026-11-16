"""三代集成评估 + 测试集 7 模型 ROC 对比图。
用法: HF_ENDPOINT=https://hf-mirror.com python -I src/final_ensemble.py <amp_train> <nonamp_train> <amp_test> <nonamp_test> <outdir>"""
import sys
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_curve, accuracy_score, roc_auc_score, f1_score,
                             matthews_corrcoef, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import AutoTokenizer, EsmForSequenceClassification
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from baseline import read_fasta
from features import featurize
from dnn import CNN, BiLSTM, CNNBiLSTMAtt, encode
from esm import tokenize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ESM_NAME = "facebook/esm2_t12_35M_UR50D"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
KEY_COLORS = {"XGBoost": "#2a78d6", "CNN-BiLSTM-Att": "#eb6834", "ESM-2": "#1baf7a"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 1.0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False,
})


def dl_probs(net_cls, weight_path, seqs, batch=512):
    net = net_cls()
    net.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    net.eval()
    ps = []
    for i in range(0, len(seqs), batch):
        x, l = encode(seqs[i:i + batch])
        with torch.no_grad():
            p = torch.sigmoid(net(torch.from_numpy(x), torch.from_numpy(l))).numpy()
        ps.append(p)
    return np.concatenate(ps)


def esm_probs(model, tok, seqs, batch=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ps = []
    for i in range(0, len(seqs), batch):
        ids, att = tokenize(seqs[i:i + batch], tok, 64)
        with torch.no_grad():
            logits = model(torch.from_numpy(ids).to(device),
                           attention_mask=torch.from_numpy(att).to(device)).logits
        ps.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    return np.concatenate(ps)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"acc": accuracy_score(y, pred), "auc": roc_auc_score(y, p),
            "f1": f1_score(y, pred), "mcc": matthews_corrcoef(y, pred),
            "sen": recall_score(y, pred), "spe": recall_score(1 - y, 1 - pred)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("amp_train")
    ap.add_argument("nonamp_train")
    ap.add_argument("amp_test")
    ap.add_argument("nonamp_test")
    ap.add_argument("outdir")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    base = Path(__file__).parent.parent

    tr_amp, tr_non = read_fasta(args.amp_train), read_fasta(args.nonamp_train)
    te_amp, te_non = read_fasta(args.amp_test), read_fasta(args.nonamp_test)
    tr_seq, te_seq = tr_amp + tr_non, te_amp + te_non
    ytr = np.array([1] * len(tr_amp) + [0] * len(tr_non))
    yte = np.array([1] * len(te_amp) + [0] * len(te_non))
    print(f"train={len(ytr)}, test={len(yte)}")

    preds = {}
    t0 = time.time()
    Xtr, Xte = featurize(tr_seq), featurize(te_seq)
    for name, model in [
        ("RF", RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=4)),
        ("SVM(RBF)", Pipeline([("scale", StandardScaler()),
                               ("svc", CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=42), cv=3))])),
        ("XGBoost", XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.8,
                                  tree_method="hist", n_jobs=4, random_state=42)),
    ]:
        model.fit(Xtr, ytr)
        preds[name] = model.predict_proba(Xte)[:, 1]
        print(f"[{name}] 完成 ({time.time() - t0:.0f}s)")
    for name, cls, w in [("CNN", CNN, base / "models" / "final_CNN.pt"),
                         ("BiLSTM", BiLSTM, base / "models" / "final_BiLSTM.pt"),
                         ("CNN-BiLSTM-Att", CNNBiLSTMAtt, base / "models" / "final_CNN-BiLSTM-Att.pt")]:
        preds[name] = dl_probs(cls, w, te_seq)
        print(f"[{name}] 完成 ({time.time() - t0:.0f}s)")
    tok = AutoTokenizer.from_pretrained(ESM_NAME, local_files_only=True)
    esm = EsmForSequenceClassification.from_pretrained(ESM_NAME, num_labels=2, local_files_only=True)
    esm.load_state_dict(torch.load(base / "reports" / "models" / "final_ESM2.pt", map_location="cpu"))
    esm = esm.to("cuda" if torch.cuda.is_available() else "cpu")
    preds["ESM-2"] = esm_probs(esm, tok, te_seq)
    print(f"[ESM-2] 完成 ({time.time() - t0:.0f}s)")

    # 保存逐样本预测
    df = pd.DataFrame({"y": yte, **preds})
    df.to_csv(outdir / "test_preds.csv", index=False)

    # 三代集成 (均值概率)
    ens = (preds["XGBoost"] + preds["CNN-BiLSTM-Att"] + preds["ESM-2"]) / 3
    rows = []
    for name in list(preds) + ["三代集成(均值)"]:
        p = preds[name] if name != "三代集成(均值)" else ens
        m = metrics(yte, p)
        rows.append({"model": name, **{f"{k}": v for k, v in m.items()}})
        print(f"[{name}] acc={m['acc']:.4f} auc={m['auc']:.4f} f1={m['f1']:.4f} mcc={m['mcc']:.4f}")
    pd.DataFrame(rows).to_csv(outdir / "ensemble_metrics.csv", index=False)

    # ROC 对比图
    fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=200)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1.0, zorder=1)
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(yte, p)
        a = roc_auc_score(yte, p)
        if name in KEY_COLORS:
            ax.plot(fpr, tpr, color=KEY_COLORS[name], linewidth=2, zorder=3,
                    label=f"{name} (AUC={a:.3f})")
        else:
            ax.plot(fpr, tpr, color=MUTED, linewidth=1.4, alpha=0.75, zorder=2,
                    label=f"{name} (AUC={a:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on official test set", loc="left", fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(color=GRID, linewidth=1.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    leg = ax.legend(loc="lower right", fontsize=9, ncol=2)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "fig8_roc.png", bbox_inches="tight")
    print(f"已保存 {outdir / 'figures' / 'fig8_roc.png'} 与 {outdir / 'ensemble_metrics.csv'}")


if __name__ == "__main__":
    main()
