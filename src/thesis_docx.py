"""论文.md → 福州大学本科毕业论文格式 docx。
用法: python -I src/thesis_docx.py <论文.md> <输出.docx>"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 字号: 小2=18 3号=16 小三=15 4号=14 小4=12 5号=10.5 小5=9 一号=26
HEI = "黑体"
SONG = "宋体"
TNR = "Times New Roman"
ARIAL_BLACK = "Arial Black"


def set_run(run, latin, east, size, bold=False):
    run.font.name = latin
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east)


def para(doc, text="", latin=SONG, east=SONG, size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         line=20, before=0, after=0, first_indent=None):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing = Pt(line)
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if first_indent is not None:
        pf.first_line_indent = Pt(first_indent)
    if text:
        set_run(p.add_run(text), latin, east, size, bold)
    return p


def add_page_field(section, fmt="decimal", start=1):
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for tag, attrs, text in [("w:fldChar", {"w:fldCharType": "begin"}, None),
                             ("w:instrText", None, "PAGE"),
                             ("w:fldChar", {"w:fldCharType": "end"}, None)]:
        el = OxmlElement(tag)
        if attrs:
            for k, v in attrs.items():
                el.set(qn(k), v)
        if text:
            el.text = text
        run = p.add_run()
        run._r.append(el)
        set_run(run, TNR, SONG, 9)
    sectPr = section._sectPr
    pgNum = OxmlElement("w:pgNumType")
    pgNum.set(qn("w:fmt"), fmt)
    pgNum.set(qn("w:start"), str(start))
    anchor = sectPr.find(qn("w:cols"))
    if anchor is None:
        anchor = sectPr.find(qn("w:docGrid"))
    if anchor is not None:
        sectPr.insert(list(sectPr).index(anchor), pgNum)
    else:
        sectPr.append(pgNum)


def add_toc_field(doc):
    p = doc.add_paragraph()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), r'TOC \o "1-2" \h \z \u')
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = "（打开文档后 Ctrl+A 全选再按 F9 更新目录域）"
    r.append(t)
    fld.append(r)
    p._p.append(fld)


def cover(doc, title):
    for _ in range(2):
        para(doc, "", line=20)
    para(doc, "福州大学", HEI, HEI, 26, align=WD_ALIGN_PARAGRAPH.CENTER, line=36)
    para(doc, "本科毕业论文", HEI, HEI, 22, align=WD_ALIGN_PARAGRAPH.CENTER, line=32, after=36)
    para(doc, title, HEI, HEI, 18, align=WD_ALIGN_PARAGRAPH.CENTER, line=30, after=48)
    info = [("学    院", "＿＿＿＿＿＿＿＿"), ("专    业", "软件工程"),
            ("年    级", "2024级"), ("学    号", "102400216"),
            ("学生姓名", "黄思哲"), ("指导教师", "＿＿＿＿＿＿"),
            ("完成日期", "2026年11月")]
    for k, v in info:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = Pt(28)
        set_run(p.add_run(f"{k}："), SONG, SONG, 14)
        set_run(p.add_run(v), SONG, SONG, 14)


def page_break(doc):
    from docx.enum.text import WD_BREAK
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def zh_abstract(doc, title, body_lines):
    para(doc, title, HEI, HEI, 18, align=WD_ALIGN_PARAGRAPH.CENTER, line=30, after=12)
    para(doc, "摘    要", HEI, HEI, 14, align=WD_ALIGN_PARAGRAPH.CENTER, line=24, after=6)
    for line in body_lines:
        if line.startswith("**关键词**"):
            p = para(doc, "", SONG, SONG, 12, line=20, before=6)
            set_run(p.add_run("关键词："), HEI, HEI, 12)
            set_run(p.add_run(line.replace("**关键词**：", "")), SONG, SONG, 12)
        else:
            para(doc, line, SONG, SONG, 12, line=20, first_indent=24)


def en_abstract(doc, title, body_lines):
    para(doc, title, ARIAL_BLACK, ARIAL_BLACK, 15, align=WD_ALIGN_PARAGRAPH.CENTER, line=24, after=8)
    para(doc, "Abstract", ARIAL_BLACK, ARIAL_BLACK, 14, align=WD_ALIGN_PARAGRAPH.CENTER, line=20, after=4)
    for line in body_lines:
        if line.startswith("**Key words**"):
            p = para(doc, "", TNR, SONG, 12, line=16, before=6)
            set_run(p.add_run("Key words: "), ARIAL_BLACK, ARIAL_BLACK, 12)
            set_run(p.add_run(line.replace("**Key words**", "").lstrip(": ")), TNR, SONG, 12)
        else:
            para(doc, line, TNR, SONG, 12, line=16)


def parse_table(lines, doc):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    if rows and all(re.fullmatch(r":?-{2,}:?", c) for c in rows[0]):
        rows = rows[1:]
    ncol = max(len(r) for r in rows) if rows else 0
    table = doc.add_table(rows=len(rows), cols=ncol)
    table.style = "Table Grid"
    for ri, r in enumerate(rows):
        for ci in range(ncol):
            cell = table.cell(ri, ci)
            cell.text = ""
            cp = cell.paragraphs[0]
            cp.paragraph_format.line_spacing = Pt(14)
            cell_text = r[ci] if ci < len(r) else ""
            set_run(cp.add_run(cell_text), SONG, SONG, 10.5, bold=(ri == 0))
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, "", size=5, line=10)


def build(src, dst):
    lines = Path(src).read_text(encoding="utf-8").splitlines()
    # 提取前置部分
    title = lines[0].lstrip("# ").strip()
    sec = {}
    cur = None
    for i, l in enumerate(lines):
        if l.startswith("## "):
            cur = l[3:].strip()
            sec[cur] = []
        elif cur and (cur in ("摘要", "Abstract")) and l.strip():
            sec[cur].append(l.strip())
        elif cur and l.startswith("# ") and cur not in ("摘要", "Abstract"):
            cur = l[2:].strip()
    body_start = None
    for i, l in enumerate(lines):
        if l.strip() == "## 目录":
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("# "):
                    body_start = j
                    break
            break
    body_lines = lines[body_start:]

    doc = Document()
    # 页面: A4, 天头/订口25mm+, 地脚/切口20mm+
    s1 = doc.sections[0]
    s1.page_width, s1.page_height = Cm(21), Cm(29.7)
    s1.top_margin, s1.bottom_margin = Cm(2.8), Cm(2.5)
    s1.left_margin, s1.right_margin = Cm(3.0), Cm(2.5)

    # 默认样式
    style = doc.styles["Normal"]
    style.font.name = TNR
    style.element.rPr.rFonts.set(qn("w:eastAsia"), SONG)
    style.font.size = Pt(12)

    # 前置部分 (罗马页码)
    add_page_field(s1, fmt="lowerRoman", start=1)
    cover(doc, title)
    page_break(doc)
    zh_abstract(doc, title, sec.get("摘要", []))
    page_break(doc)
    en_abstract(doc, title, sec.get("Abstract", []))
    page_break(doc)
    para(doc, "目    录", HEI, HEI, 16, align=WD_ALIGN_PARAGRAPH.CENTER, line=24, after=12)
    add_toc_field(doc)

    # 正文 (阿拉伯页码, 从1重新编)
    s2 = doc.add_section(WD_SECTION.NEW_PAGE)
    s2.page_width, s2.page_height = Cm(21), Cm(29.7)
    s2.top_margin, s2.bottom_margin = Cm(2.8), Cm(2.5)
    s2.left_margin, s2.right_margin = Cm(3.0), Cm(2.5)
    add_page_field(s2, fmt="decimal", start=1)

    i = 0
    pending_table = []
    while i < len(body_lines):
        line = body_lines[i].rstrip()
        if line.strip().startswith("|"):
            pending_table = [line.strip().strip("|")]
            i += 1
            while i < len(body_lines) and body_lines[i].strip().startswith("|"):
                pending_table.append(body_lines[i].strip().strip("|"))
                i += 1
            parse_table(pending_table, doc)
            continue
        if line.startswith("# "):
            para(doc, line[2:].strip(), HEI, HEI, 18, align=WD_ALIGN_PARAGRAPH.CENTER, line=30, after=30)
        elif line.startswith("## "):
            para(doc, line[3:].strip(), HEI, HEI, 16, line=24, after=18)
        elif line.startswith("### "):
            para(doc, line[4:].strip(), HEI, HEI, 15, line=20, after=12)
        elif line.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]*)\)", line)
            if m:
                cap, img = m.group(1), Path(__file__).parent.parent / "reports" / m.group(2)
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.line_spacing = Pt(14)
                if img.exists():
                    p.add_run().add_picture(str(img), width=Cm(14))
                para(doc, cap, SONG, SONG, 10.5, align=WD_ALIGN_PARAGRAPH.CENTER, line=14, after=8)
        elif line.startswith("| "):
            pass
        elif re.match(r"^表 \d", line.strip()):
            para(doc, line.strip(), SONG, SONG, 10.5, align=WD_ALIGN_PARAGRAPH.CENTER, line=14, before=6, after=4)
        elif line.startswith("- ") or line.startswith("* "):
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = Pt(20)
            set_run(p.add_run(line[2:].strip()), SONG, SONG, 12)
        elif line.strip():
            if line.strip() in ("参考文献", "谢辞", "结论", "附录A 开源发布", "附录B 复现说明", "附录C AI 协作开发记录"):
                para(doc, line.strip(), HEI, HEI, 18, align=WD_ALIGN_PARAGRAPH.CENTER, line=30, after=30)
            elif re.match(r"^\[(\d+)\]", line.strip()):
                p = para(doc, line.strip(), SONG, SONG, 10.5, line=17, before=3)
            else:
                para(doc, line.strip(), SONG, SONG, 12, line=20, first_indent=24)
        i += 1

    doc.save(dst)
    print("saved", dst)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
