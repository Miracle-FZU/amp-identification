"""Markdown 报告转 docx (支持标题/段落/列表/表格/加粗)。
用法: python -I src/md_to_docx.py <input.md> <output.docx>"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt


def add_runs(par, text):
    for p in re.split(r"(\*\*.+?\*\*)", text):
        if not p:
            continue
        if p.startswith("**") and p.endswith("**"):
            par.add_run(p[2:-2]).bold = True
        else:
            par.add_run(p)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    style.font.size = Pt(11)

    lines = Path(src).read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.strip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip().strip("|"))
                i += 1
            rows = [[c.strip() for c in b.split("|")] for b in block]
            if rows and all(re.fullmatch(r":?-{2,}:?", c) for c in rows[0]):
                rows = rows[1:]
            ncol = max(len(r) for r in rows) if rows else 0
            table = doc.add_table(rows=len(rows), cols=ncol)
            table.style = "Table Grid"
            for ri, r in enumerate(rows):
                for ci in range(ncol):
                    cell = table.cell(ri, ci)
                    cell.text = ""
                    par = cell.paragraphs[0]
                    add_runs(par, r[ci] if ci < len(r) else "")
                    for run in par.runs:
                        run.font.size = Pt(9)
                        if ri == 0:
                            run.bold = True
            doc.add_paragraph()
            continue
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=0)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=1)
        elif line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=2)
        elif line.startswith("- ") or line.startswith("* "):
            doc.add_paragraph(line[2:].strip(), style="List Bullet")
        elif re.match(r"^\d+\. ", line):
            doc.add_paragraph(re.sub(r"^\d+\. ", "", line).strip(), style="List Number")
        elif line.strip():
            par = doc.add_paragraph()
            add_runs(par, line)
        i += 1
    doc.save(dst)
    print("saved", dst)


if __name__ == "__main__":
    main()
