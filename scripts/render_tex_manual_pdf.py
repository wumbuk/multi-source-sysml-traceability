from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "trace_dev_manual.tex"
TARGET = ROOT / "需求模型追溯功能开发手册.pdf"

PAGE_W, PAGE_H = 1240, 1754  # A4 at about 150 dpi
MARGIN_L, MARGIN_R = 86, 86
MARGIN_T, MARGIN_B = 88, 88
HEADER_Y, FOOTER_Y = 42, PAGE_H - 54
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

FONT_REG = "C:/Windows/Fonts/msyh.ttc"
FONT_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
FONT_MONO = "C:/Windows/Fonts/msyh.ttc"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


F_TITLE = font(FONT_BOLD, 52)
F_SUBTITLE = font(FONT_REG, 30)
F_H1 = font(FONT_BOLD, 35)
F_H2 = font(FONT_BOLD, 28)
F_H3 = font(FONT_BOLD, 24)
F_BODY = font(FONT_REG, 22)
F_BODY_BOLD = font(FONT_BOLD, 22)
F_SMALL = font(FONT_REG, 19)
F_CODE = font(FONT_MONO, 19)
F_FOOTER = font(FONT_REG, 16)


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> int:
    if not text:
        return 0
    return int(draw.textbbox((0, 0), text, font=fnt)[2])


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return [""]
    lines: list[str] = []
    current = ""
    for ch in text:
        candidate = current + ch
        if text_width(draw, candidate, fnt) <= width:
            current = candidate
            continue
        if current:
            lines.append(current.rstrip())
        current = ch.lstrip()
    if current:
        lines.append(current.rstrip())
    return lines


def clean_inline(text: str) -> str:
    replacements = {
        r"\%": "%",
        r"\_": "_",
        r"\&": "&",
        r"\#": "#",
        r"\$": "$",
        r"\{": "{",
        r"\}": "}",
        "--": "—",
        r"$\geq$": "≥",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    patterns = [
        (r"\\code\{([^{}]*)\}", r"\1"),
        (r"\\textbf\{([^{}]*)\}", r"\1"),
        (r"\\texttt\{\\detokenize\{([^{}]*)\}\}", r"\1"),
        (r"\\texttt\{([^{}]*)\}", r"\1"),
        (r"\\emph\{([^{}]*)\}", r"\1"),
    ]
    changed = True
    while changed:
        changed = False
        for pat, repl in patterns:
            new_text = re.sub(pat, repl, text)
            changed = changed or (new_text != text)
            text = new_text

    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", text)
    text = text.replace("{", "").replace("}", "")
    return text.strip()


def tex_to_lines(tex: str) -> list[tuple[str, str]]:
    start_marker = r"\section{手册目标与适用边界}"
    if start_marker in tex:
        tex = start_marker + "\n" + tex.split(start_marker, 1)[1]
    tex = tex.split(r"\end{document}", 1)[0]

    lines: list[tuple[str, str]] = []
    in_verbatim = False
    skip_envs = {
        "document",
        "titlepage",
        "longtable",
        "itemize",
        "enumerate",
    }

    for raw in tex.splitlines():
        line = raw.strip()
        if not line:
            lines.append(("blank", ""))
            continue

        if line.startswith(r"\begin{verbatim}"):
            in_verbatim = True
            lines.append(("blank", ""))
            continue
        if line.startswith(r"\end{verbatim}"):
            in_verbatim = False
            lines.append(("blank", ""))
            continue
        if in_verbatim:
            lines.append(("code", raw.rstrip()))
            continue

        if re.match(r"\\(toprule|midrule|bottomrule)", line):
            continue
        if re.match(r"\\(begin|end)\{(" + "|".join(skip_envs) + r")\}", line):
            lines.append(("blank", ""))
            continue
        if line.startswith(r"\newpage") or line.startswith(r"\tableofcontents"):
            continue

        section_match = re.match(r"\\section\{([^{}]+)\}", line)
        subsection_match = re.match(r"\\subsection\{([^{}]+)\}", line)
        subsub_match = re.match(r"\\subsubsection\{([^{}]+)\}", line)
        if section_match:
            lines.append(("h1", clean_inline(section_match.group(1))))
            continue
        if subsection_match:
            lines.append(("h2", clean_inline(subsection_match.group(1))))
            continue
        if subsub_match:
            lines.append(("h3", clean_inline(subsub_match.group(1))))
            continue

        if line.startswith(r"\item"):
            line = "- " + line[len(r"\item") :].strip()
        line = line.replace(r"\\", "")
        line = line.replace("&", "  |  ")
        line = clean_inline(line)
        if not line:
            continue
        if line.startswith("- "):
            lines.append(("bullet", line[2:].strip()))
        elif "  |  " in line:
            lines.append(("table", line))
        else:
            lines.append(("p", line))
    return collapse_blanks(lines)


def collapse_blanks(lines: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    prev_blank = False
    for typ, text in lines:
        if typ == "blank":
            if not prev_blank:
                out.append((typ, text))
            prev_blank = True
        else:
            out.append((typ, text))
            prev_blank = False
    return out


class Renderer:
    def __init__(self) -> None:
        self.pages: list[Image.Image] = []
        self.page_no = 0
        self.image: Image.Image
        self.draw: ImageDraw.ImageDraw
        self.y = MARGIN_T
        self.new_page(cover=False)

    def new_page(self, cover: bool = False) -> None:
        self.page_no += 1
        self.image = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        self.draw = ImageDraw.Draw(self.image)
        if not cover:
            self.draw.text((MARGIN_L, HEADER_Y), "需求-模型追溯模块开发手册", font=F_FOOTER, fill="#555555")
            page_text = f"{self.page_no}"
            self.draw.text((PAGE_W - MARGIN_R - text_width(self.draw, page_text, F_FOOTER), HEADER_Y), page_text, font=F_FOOTER, fill="#555555")
            self.draw.line((MARGIN_L, 72, PAGE_W - MARGIN_R, 72), fill="#D9D9D9", width=1)
        self.y = MARGIN_T
        self.pages.append(self.image)

    def ensure(self, needed: int) -> None:
        if self.y + needed > PAGE_H - MARGIN_B:
            self.new_page()

    def paragraph(self, text: str, fnt=F_BODY, fill="#222222", indent: int = 0, spacing: int = 12) -> None:
        lines = wrap_text(self.draw, text, fnt, CONTENT_W - indent)
        line_h = int(fnt.size * 1.55)
        self.ensure(line_h * len(lines) + spacing)
        for line in lines:
            self.draw.text((MARGIN_L + indent, self.y), line, font=fnt, fill=fill)
            self.y += line_h
        self.y += spacing

    def bullet(self, text: str) -> None:
        bullet_indent = 30
        lines = wrap_text(self.draw, text, F_BODY, CONTENT_W - 44)
        line_h = int(F_BODY.size * 1.52)
        self.ensure(line_h * len(lines) + 9)
        self.draw.text((MARGIN_L, self.y), "•", font=F_BODY_BOLD, fill="#222222")
        for i, line in enumerate(lines):
            self.draw.text((MARGIN_L + bullet_indent, self.y), line, font=F_BODY, fill="#222222")
            self.y += line_h
        self.y += 8

    def code(self, text: str) -> None:
        lines = wrap_text(self.draw, text, F_CODE, CONTENT_W - 28)
        line_h = int(F_CODE.size * 1.45)
        box_h = line_h * len(lines) + 16
        self.ensure(box_h + 10)
        x0, y0 = MARGIN_L, self.y
        x1, y1 = PAGE_W - MARGIN_R, self.y + box_h
        self.draw.rounded_rectangle((x0, y0, x1, y1), radius=6, fill="#F5F7FA", outline="#D6DEE8")
        y_text = y0 + 8
        for line in lines:
            self.draw.text((x0 + 14, y_text), line, font=F_CODE, fill="#1F2933")
            y_text += line_h
        self.y = y1 + 10

    def heading(self, level: int, text: str) -> None:
        if level == 1:
            fnt, top, bottom, color = F_H1, 24, 14, "#0F172A"
        elif level == 2:
            fnt, top, bottom, color = F_H2, 18, 10, "#1F2937"
        else:
            fnt, top, bottom, color = F_H3, 14, 8, "#1F2937"
        lines = wrap_text(self.draw, text, fnt, CONTENT_W)
        line_h = int(fnt.size * 1.4)
        self.ensure(top + line_h * len(lines) + bottom + 8)
        self.y += top
        for line in lines:
            self.draw.text((MARGIN_L, self.y), line, font=fnt, fill=color)
            self.y += line_h
        if level == 1:
            self.draw.line((MARGIN_L, self.y + 2, PAGE_W - MARGIN_R, self.y + 2), fill="#CBD5E1", width=2)
            self.y += 10
        self.y += bottom

    def table_line(self, text: str) -> None:
        self.paragraph(text, fnt=F_SMALL, fill="#263238", indent=0, spacing=8)

    def cover(self) -> None:
        self.pages.pop()
        self.page_no = 1
        self.image = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.pages.append(self.image)
        self.draw.rectangle((0, 0, PAGE_W, 18), fill="#1F4E79")
        title = "需求-模型追溯模块开发手册"
        subtitle = "面向 Codex 等编码智能体的实现规范"
        note = "基于当前项目中的 XMI 信息提取源码与追溯关系实验主流程编写"
        date = "生成日期：2026 年 7 月 3 日"
        y = 480
        for line, fnt, fill, gap in [
            (title, F_TITLE, "#111827", 44),
            (subtitle, F_SUBTITLE, "#374151", 36),
            (note, F_BODY, "#4B5563", 26),
            (date, F_BODY, "#4B5563", 0),
        ]:
            x = (PAGE_W - text_width(self.draw, line, fnt)) // 2
            self.draw.text((x, y), line, font=fnt, fill=fill)
            y += int(fnt.size * 1.45) + gap
        self.draw.line((MARGIN_L, PAGE_H - 220, PAGE_W - MARGIN_R, PAGE_H - 220), fill="#CBD5E1", width=2)
        self.draw.text((MARGIN_L, PAGE_H - 180), "交付物：PDF 开发手册 + trace_system 环境 requirements.txt", font=F_BODY, fill="#374151")
        self.new_page()


def main() -> None:
    tex = SOURCE.read_text(encoding="utf-8")
    lines = tex_to_lines(tex)
    renderer = Renderer()
    renderer.cover()

    for typ, text in lines:
        if typ == "blank":
            renderer.y += 10
        elif typ == "h1":
            renderer.heading(1, text)
        elif typ == "h2":
            renderer.heading(2, text)
        elif typ == "h3":
            renderer.heading(3, text)
        elif typ == "bullet":
            renderer.bullet(text)
        elif typ == "code":
            renderer.code(text)
        elif typ == "table":
            renderer.table_line(text)
        else:
            renderer.paragraph(text)

    renderer.pages[0].save(
        TARGET,
        "PDF",
        resolution=150.0,
        save_all=True,
        append_images=renderer.pages[1:],
        quality=95,
    )
    print(TARGET)
    print(f"pages={len(renderer.pages)}")


if __name__ == "__main__":
    main()
