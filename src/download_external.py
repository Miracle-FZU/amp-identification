"""下载 sAMPPred-GAT 独立测试数据集 (8 个数据库的 AMP/nonAMP 对) 到 data/external/。
经 jsDelivr CDN 中转 (raw.githubusercontent.com 被墙)。
用法: python -I src/download_external.py"""
import pathlib
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://cdn.jsdelivr.net/gh/HongWuL/sAMPPred-GAT@main/datasets/independent%20test%20datasets/{d}/{name}"
DIRS = ["DBAASP", "XUAMP", "other%20six"]
FILES = {
    "DBAASP": ["DBAASP_AMP.fasta", "DBAASP_nonAMP.fasta"],
    "XUAMP": ["XU_AMP.fasta", "XU_nonAMP.fasta"],
    "other%20six": ["APD3_AMP.fasta", "APD3_nonAMP.fasta", "CAMP_AMP.fasta", "CAMP_nonAMP.fasta",
                    "DRAMP_AMP.fasta", "DRAMP_nonAMP.fasta", "LAMP_AMP.fasta", "LAMP_nonAMP.fasta",
                    "YADAMP_AMP.fasta", "YADAMP_nonAMP.fasta", "dbAMP_AMP.fasta", "dbAMP_nonAMP.fasta"],
}
OUT = pathlib.Path(__file__).parent.parent / "data" / "external"
OUT.mkdir(parents=True, exist_ok=True)

for d, names in FILES.items():
    for name in names:
        out = OUT / name
        if out.exists() and out.stat().st_size > 0:
            print("skip", name)
            continue
        try:
            urllib.request.urlretrieve(BASE.format(d=d, name=name), out)
            print("saved", name, out.stat().st_size)
        except Exception as e:
            print("FAIL", name, e)
