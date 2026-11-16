# 抗菌肽识别系统（AMP Identification）

生物信息学课程期末项目：基于多策略机器学习的抗菌肽（Antimicrobial Peptide, AMP）二分类识别。
输入一段氨基酸序列，判断其是否具有抗菌活性。

## 数据

AMPlify 平衡数据集（Li et al., BMC Genomics 2022）：
- 训练集：3338 AMP + 3338 非AMP
- 测试集：835 + 835
- 另有非平衡测试集（835 vs 25689）与 8 个外部数据库测试集用于泛化评估

## 方法对比

| 代际 | 方法 | 测试集 Acc / AUC |
|---|---|---|
| 第一代 | 特征工程(1626维) + RF/SVM/XGBoost | 最高 90.90% / 97.07% (XGB) |
| 第二代 | CNN / BiLSTM / CNN+BiLSTM+Attention | 最高 90.72% / 96.78% |
| 第三代 | ESM-2 蛋白语言模型微调 | 92.75% / 97.83% |
| **集成** | 三代概率集成(均值) | **93.35% / 98.27%** (最佳) |

## 目录结构

```
bioinfo-amp/
├── app.py                  # Gradio 演示(定制UI, 三代模型同框): python app.py → http://127.0.0.1:7860
├── src/
│   ├── inspect_data.py     # 数据质检
│   ├── eda.py              # EDA 三图 (长度/氨基酸组成/理化性质)
│   ├── features.py         # 特征工程: AAC/DPC/CKSAAP/理化
│   ├── baseline.py         # RF/SVM/XGBoost 5折CV
│   ├── dnn.py              # CNN/BiLSTM/CNN-BiLSTM-Att 5折CV
│   ├── esm.py              # ESM-2 微调
│   ├── test_eval.py        # 测试集最终评估
│   ├── imbalanced_eval.py  # 非平衡压力测试+阈值优化
│   ├── external_eval.py    # 跨数据库外部验证
│   ├── shap_analysis.py    # SHAP 特征重要性
│   ├── tsne.py             # t-SNE 嵌入可视化
│   ├── attention_heatmap.py# 注意力热图
│   ├── save_models.py      # 保存演示用模型
│   └── download_external.py# 下载外部测试集(jsDelivr)
├── data/raw/               # 原始数据 (来自 AMPlify 仓库)
├── data/external/          # 外部测试集 (sAMPPred-GAT)
├── reports/figures/        # 全部图表
├── reports/                # 指标 CSV + 报告
└── models/                 # 训练好的模型权重
```

## 复现步骤

```bash
# 1. 数据: 克隆 AMPlify 仓库取 data/ (或直接使用 data/raw/)
# 2. 下载外部测试集
python -I src/download_external.py
# 3. EDA
python -I src/eda.py data/raw/AMPlify_AMP_train_common.fa data/raw/AMPlify_non_AMP_train_balanced.fa reports/figures
# 4. 传统 ML 基线 (5折CV)
python -I src/baseline.py data/raw/AMPlify_AMP_train_common.fa data/raw/AMPlify_non_AMP_train_balanced.fa reports
# 5. 深度学习 (需 CUDA torch)
python -I src/dnn.py data/raw/AMPlify_AMP_train_common.fa data/raw/AMPlify_non_AMP_train_balanced.fa reports --device cuda
# 6. ESM-2 微调 (国内网络需 hf-mirror)
HF_ENDPOINT=https://hf-mirror.com python -I src/esm.py data/raw/AMPlify_AMP_train_common.fa data/raw/AMPlify_non_AMP_train_balanced.fa reports --device cuda
# 7. 测试集最终评估
python -I src/test_eval.py data/raw/AMPlify_AMP_train_common.fa data/raw/AMPlify_non_AMP_train_balanced.fa data/raw/AMPlify_AMP_test_common.fa data/raw/AMPlify_non_AMP_test_balanced.fa reports --device cuda
# 8. 可解释性/可视化
python -I src/shap_analysis.py data/raw/AMPlify_AMP_train_common.fa data/raw/AMPlify_non_AMP_train_balanced.fa reports
python -I src/tsne.py data/raw/AMPlify_AMP_train_common.fa data/raw/AMPlify_non_AMP_train_balanced.fa reports
python -I src/attention_heatmap.py data/raw/AMPlify_AMP_train_common.fa data/raw/AMPlify_non_AMP_train_balanced.fa reports
# 8b. 三代集成评估 + ROC 对比图 (fig8)
HF_ENDPOINT=https://hf-mirror.com python -I src/final_ensemble.py data/raw/AMPlify_AMP_train_common.fa data/raw/AMPlify_non_AMP_train_balanced.fa data/raw/AMPlify_AMP_test_common.fa data/raw/AMPlify_non_AMP_test_balanced.fa reports
# 8c. 应用案例: 牛蛙蛋白组扫描
HF_ENDPOINT=https://hf-mirror.com python -I src/discover.py AMPlify_repo/auxiliary/amp_discovery_pipeline/amp-evidence-proteins.fa reports
# 9. 演示
python -I src/save_models.py && python app.py
```

## 环境

- Python 3.13, PyTorch 2.11+cu128（**Windows CUDA 版需用 `--index-url https://download.pytorch.org/whl/cu128`，PyPI/国内镜像仅有 CPU 版**）
- 其余依赖：`pip install biopython xgboost transformers gradio shap seaborn python-docx`
- 国内 HuggingFace 下载：`HF_ENDPOINT=https://hf-mirror.com`（首次下载模型用；演示 app.py 已设为离线加载缓存）
- 演示 app.py 依赖 `reports/models/final_ESM2.pt`（ESM-2 最终模型，较大，提交仓库时可排除）

## 报告

完整报告见 `reports/报告初稿.md`（含全部图表、结果表格与参考文献）。
