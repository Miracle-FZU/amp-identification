"""序列特征工程: AAC / DPC / CKSAAP / 理化性质(电荷、pI、GRAVY 等)。"""
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AAS)}


def aac(seq):
    """氨基酸组成 (20 维)"""
    v = np.zeros(20)
    for a in seq:
        if a in AA_IDX:
            v[AA_IDX[a]] += 1
    return v / max(len(seq), 1)


def dpc(seq):
    """二肽组成 (400 维)"""
    v = np.zeros(400)
    n = max(len(seq) - 1, 1)
    for i in range(len(seq) - 1):
        k = AA_IDX.get(seq[i], -1) * 20 + AA_IDX.get(seq[i + 1], -1)
        if k >= 0:
            v[k] += 1
    return v / n


def cksaap(seq, k_list=(1, 2, 3)):
    """k 间隔残基对组成, k=1,2,3 (各 400 维, 共 1200 维)"""
    parts = []
    L = len(seq)
    for k in k_list:
        v = np.zeros(400)
        n = max(L - k - 1, 1)
        for i in range(L - k - 1):
            a, b = AA_IDX.get(seq[i], -1), AA_IDX.get(seq[i + k + 1], -1)
            if a >= 0 and b >= 0:
                v[a * 20 + b] += 1
        parts.append(v / n)
    return np.concatenate(parts)


def physchem(seq):
    """理化性质 (6 维): 长度/电荷@pH7/pI/GRAVY/芳香性/不稳定指数"""
    try:
        pa = ProteinAnalysis(seq)
        return np.array([
            len(seq),
            pa.charge_at_pH(7.0),
            pa.isoelectric_point(),
            pa.gravy(),
            pa.aromaticity(),
            pa.instability_index() if len(seq) >= 5 else 0.0,
        ])
    except Exception:
        return np.zeros(6)


def featurize(seqs):
    return np.hstack([
        np.array([aac(s) for s in seqs]),
        np.array([dpc(s) for s in seqs]),
        np.array([cksaap(s) for s in seqs]),
        np.array([physchem(s) for s in seqs]),
    ])


def feature_names():
    names = [f"AAC_{a}" for a in AAS]
    names += [f"DPC_{a}{b}" for a in AAS for b in AAS]
    names += [f"CK{k}_{a}{b}" for k in (1, 2, 3) for a in AAS for b in AAS]
    names += ["length", "charge7", "pI", "gravy", "aromaticity", "instability"]
    return names
