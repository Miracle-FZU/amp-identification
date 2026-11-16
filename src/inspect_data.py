"""检查 FASTA 数据集的基本统计信息。用法: python -I src/inspect_data.py <file.fa> [more.fa ...]"""
import argparse
from collections import Counter

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path):
    seqs = []
    with open(path, encoding="utf-8", errors="replace") as f:
        buf = []
        has_header = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if has_header:
                    seqs.append("".join(buf).upper())
                has_header = True
                buf = []
            else:
                buf.append(line)
        if has_header:
            seqs.append("".join(buf).upper())
    return seqs


def stats(name, seqs):
    lens = sorted(len(s) for s in seqs)
    chars = Counter(c for s in seqs for c in s)
    total = sum(chars.values())
    nonstd = {c: n for c, n in chars.items() if c not in STANDARD_AA}
    dup = len(seqs) - len(set(seqs))
    top = sorted(chars, key=chars.get, reverse=True)[:8]
    print(f"[{name}]")
    print(f"  序列数={len(seqs)}, 长度 min/中位/max = {lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")
    print(f"  非标准字符={nonstd if nonstd else '无'}, 完全重复序列={dup}")
    print(f"  氨基酸占比 top8: " + ", ".join(f"{aa}:{chars[aa]/total:.3f}" for aa in top))
    print(f"  示例: {seqs[0][:60]}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    for f in args.files:
        stats(f, read_fasta(f))
