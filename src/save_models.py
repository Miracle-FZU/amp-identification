"""拟合并保存 XGBoost 全量模型 (供 Gradio 演示加载)。用法: python -I src/save_models.py"""
import sys
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from baseline import read_fasta
from features import featurize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if __name__ == "__main__":
    base = Path(__file__).parent.parent
    amp = read_fasta(base / "data" / "raw" / "AMPlify_AMP_train_common.fa")
    non = read_fasta(base / "data" / "raw" / "AMPlify_non_AMP_train_balanced.fa")
    X = featurize(amp + non)
    y = np.array([1] * len(amp) + [0] * len(non))
    model = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8,
                          tree_method="hist", n_jobs=4, random_state=42)
    print("拟合 XGBoost 全量模型 ...")
    model.fit(X, y)
    (base / "models").mkdir(exist_ok=True)
    joblib.dump(model, base / "models" / "xgb_full.joblib")
    print(f"已保存 {base / 'models' / 'xgb_full.joblib'}")
