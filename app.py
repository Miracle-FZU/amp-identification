"""抗菌肽识别系统 — Gradio 演示 (三代模型同框: XGBoost / CNN-BiLSTM-Att / ESM-2)
运行: python app.py → 浏览器打开 http://127.0.0.1:7860"""
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import gradio as gr
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from transformers import AutoTokenizer, EsmForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import featurize
from dnn import AA_IDX, MAXLEN, CNNBiLSTMAtt

BASE = Path(__file__).parent
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
ESM_NAME = "facebook/esm2_t12_35M_UR50D"

# ---- 视觉规范 (与报告图表同一套配色体系) ----
C_AMP, C_NON = "#2a78d6", "#eb6834"
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
HAIR, TRACK = "rgba(11,11,11,.10)", "#e1e0d9"

CSS = f"""
body {{ background:{PAGE}; }}
.gradio-container {{ max-width:1020px !important; margin:0 auto !important; }}
.result-card {{ background:{SURFACE}; border:1px solid {HAIR}; border-radius:14px; padding:22px 26px; }}
.verdict {{ display:flex; align-items:center; gap:9px; font-size:18px; font-weight:600; color:{INK}; }}
.verdict-dot {{ width:11px; height:11px; border-radius:50%; flex:none; }}
.verdict.pos .verdict-dot {{ background:{C_AMP}; box-shadow:0 0 0 4px rgba(42,120,214,.15); }}
.verdict.neg .verdict-dot {{ background:{C_NON}; box-shadow:0 0 0 4px rgba(235,104,52,.15); }}
.verdict-sub {{ font-size:13px; color:{INK2}; margin:5px 0 18px 20px; }}
.model-row {{ display:flex; align-items:center; gap:12px; margin:11px 0; }}
.model-name {{ width:225px; font-size:13px; color:{INK2}; text-align:right; flex:none; }}
.model-tag {{ display:inline-block; font-size:11px; color:{MUTED}; border:1px solid {HAIR};
               border-radius:999px; padding:0 8px; margin-left:6px; }}
.bar-track {{ flex:1; height:10px; background:{TRACK}; border-radius:5px; overflow:hidden; }}
.bar-fill {{ height:100%; border-radius:5px; transition:width .3s ease; }}
.model-val {{ width:54px; font-size:13px; font-weight:600; color:{INK}; font-variant-numeric:tabular-nums; text-align:right; }}
.stats {{ display:flex; gap:10px; margin-top:22px; }}
.stat {{ flex:1; background:{PAGE}; border:1px solid {HAIR}; border-radius:10px; padding:10px 14px; }}
.stat .k {{ font-size:12px; color:{MUTED}; }}
.stat .v {{ font-size:16px; font-weight:600; color:{INK}; margin-top:2px; font-variant-numeric:tabular-nums; }}
.footer {{ text-align:center; color:{MUTED}; font-size:12px; margin-top:20px; line-height:1.7; }}
"""

HEADER = f"""
<div style="text-align:center; margin:8px 0 22px;">
  <h1 style="font-size:26px; font-weight:700; color:{INK}; margin:0;">🧬 抗菌肽（AMP）识别系统</h1>
  <p style="font-size:14px; color:{INK2}; margin:8px 0 14px;">
    输入一段氨基酸序列，三代机器学习模型联合判断其是否具有抗菌活性
  </p>
  <div style="display:flex; gap:8px; justify-content:center; flex-wrap:wrap;">
    <span style="font-size:12px; color:{INK2}; background:{SURFACE}; border:1px solid {HAIR};
                 border-radius:999px; padding:4px 12px;">数据 · AMPlify (BMC Genomics 2022)</span>
    <span style="font-size:12px; color:{INK2}; background:{SURFACE}; border:1px solid {HAIR};
                 border-radius:999px; padding:4px 12px;">XGBoost · 特征工程</span>
    <span style="font-size:12px; color:{INK2}; background:{SURFACE}; border:1px solid {HAIR};
                 border-radius:999px; padding:4px 12px;">CNN-BiLSTM-Att · 深度学习</span>
    <span style="font-size:12px; color:{INK2}; background:{SURFACE}; border:1px solid {HAIR};
                 border-radius:999px; padding:4px 12px;">ESM-2 · 蛋白语言模型</span>
  </div>
</div>
"""

FOOTER = f"""
<div class="footer">
  生物信息学课程期末项目 · 训练数据 AMPlify 平衡数据集 (Li et al., BMC Genomics 2022)<br>
  测试集 AUC: XGBoost 97.07% · CNN-BiLSTM-Att 96.78% · ESM-2 97.83%（详见项目报告）
</div>
"""

PLACEHOLDER = f"""
<div class="result-card" style="text-align:center; padding:56px 26px;">
  <div style="font-size:32px; margin-bottom:10px;">🔬</div>
  <div style="font-size:15px; color:{INK2};">在左侧输入肽序列并点击「预测」，<br>三代模型将给出各自的抗菌肽概率</div>
</div>
"""

EXAMPLES = [
    ["GLLDTFKNLALNAAKSAGVSVLNSLSCKLSKTC", "训练集 AMP 示例"],
    ["GLFDVIKKVASVIGGL", "经典抗菌肽风格序列"],
    ["AHRFAAEDFGALDTA", "训练集非AMP示例"],
    ["MKWVTFISLLFLFSSAYS", "常见非AMP (血清白蛋白信号肽)"],
]


def load_models():
    xgb = joblib.load(BASE / "models" / "xgb_full.joblib")
    net = CNNBiLSTMAtt()
    net.load_state_dict(torch.load(BASE / "models" / "CNN-BiLSTM-Att_fold0.pt",
                                   map_location="cpu", weights_only=True))
    net.eval()
    tok = AutoTokenizer.from_pretrained(ESM_NAME, local_files_only=True)
    esm = EsmForSequenceClassification.from_pretrained(ESM_NAME, num_labels=2, local_files_only=True)
    esm.load_state_dict(torch.load(BASE / "reports" / "models" / "final_ESM2.pt",
                                   map_location="cpu", weights_only=True))
    esm.eval()
    return xgb, net, (tok, esm)


XGB, NET, (TOK, ESM) = load_models()


def encode(seq):
    l = min(len(seq), MAXLEN)
    x = np.zeros((1, MAXLEN), dtype=np.int64)
    lens = np.array([l], dtype=np.int64)
    for j in range(l):
        x[0, j] = AA_IDX.get(seq[j], 20)
    return x, lens


def model_row(name, tag, p):
    return f"""
    <div class="model-row">
      <div class="model-name">{name}<span class="model-tag">{tag}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:{p * 100:.1f}%;"></div></div>
      <div class="model-val">{p:.1%}</div>
    </div>"""


def predict(seq_in):
    seq = seq_in.upper().replace(" ", "").replace("\n", "").replace("\r", "")
    if not seq:
        raise gr.Error("请输入氨基酸序列")
    bad = set(seq) - STANDARD_AA
    if bad:
        raise gr.Error(f"含非标准氨基酸字符: {''.join(sorted(bad))}（仅支持 20 种标准氨基酸）")
    if len(seq) < 2:
        raise gr.Error("序列太短（至少 2 个残基）")

    p_xgb = float(XGB.predict_proba(featurize([seq]))[0, 1])
    x, l = encode(seq)
    with torch.no_grad():
        p_dl = float(torch.sigmoid(NET(torch.from_numpy(x), torch.from_numpy(l))).item())
        enc = TOK([seq], return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        logits = ESM(**enc).logits
        p_esm = float(torch.softmax(logits, dim=1)[0, 1].item())

    mean_p = (p_xgb + p_dl + p_esm) / 3
    is_amp = mean_p >= 0.5
    verdict_cls = "pos" if is_amp else "neg"
    verdict_txt = "抗菌肽（AMP）" if is_amp else "非抗菌肽（Non-AMP）"

    pa = ProteinAnalysis(seq)
    charge, pi, gravy = pa.charge_at_pH(7.0), pa.isoelectric_point(), pa.gravy()

    html = f"""
    <div class="result-card">
      <div class="verdict {verdict_cls}">
        <span class="verdict-dot"></span>判定：{verdict_txt}
      </div>
      <div class="verdict-sub">三模型平均概率 {mean_p:.1%}</div>
      {model_row("XGBoost", "特征工程", p_xgb)}
      {model_row("CNN-BiLSTM-Att", "深度学习", p_dl)}
      {model_row("ESM-2", "蛋白语言模型", p_esm)}
      <div class="stats">
        <div class="stat"><div class="k">序列长度</div><div class="v">{len(seq)} aa</div></div>
        <div class="stat"><div class="k">净电荷 (pH 7)</div><div class="v">{charge:+.2f}</div></div>
        <div class="stat"><div class="k">等电点 pI</div><div class="v">{pi:.1f}</div></div>
        <div class="stat"><div class="k">疏水性 GRAVY</div><div class="v">{gravy:+.3f}</div></div>
      </div>
    </div>"""
    return html


with gr.Blocks(
    title="抗菌肽识别系统",
    css=CSS,
    theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate",
                         font=["Segoe UI", "Microsoft YaHei", "sans-serif"]),
) as demo:
    gr.HTML(HEADER)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            seq = gr.Textbox(
                label="氨基酸序列（20 种标准氨基酸）",
                lines=4,
                placeholder="粘贴或输入肽序列，例如 GLLDTFKNLALNAAKSAGVSVLNSLSCKLSKTC",
            )
            with gr.Row():
                btn = gr.Button("预测", variant="primary", size="lg")
                clear = gr.Button("清空", size="lg")
        with gr.Column(scale=7):
            out = gr.HTML(value=PLACEHOLDER)
    gr.Examples(examples=[e[0] for e in EXAMPLES], inputs=seq,
                label="示例（点击填入，含 AMP 与非AMP）")
    gr.HTML(FOOTER)
    btn.click(predict, inputs=seq, outputs=out)
    seq.submit(predict, inputs=seq, outputs=out)
    clear.click(lambda: "", outputs=seq)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
