"""Render the beginner team guides as simple, readable PDF handouts."""

from __future__ import annotations

import re
import tempfile
import unicodedata
from pathlib import Path

from fpdf import FPDF
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent / "team_guides"
OUTPUT = ROOT / "pdf"
GUIDES = (
    "00_overall_project_and_code_flow.md",
    "01_algorithm_and_research.md",
    "02_cpu_fp64_implementation.md",
    "03_fp32_and_cuda_variants.md",
    "04_validation_results_and_delivery.md",
)


def ascii_text(value: str) -> str:
    replacements = {
        "→": "->",
        "–": "-",
        "—": "-",
        "…": "...",
        "≈": "approx",
        "≤": "<=",
        "≥": ">=",
        "∞": "infinity",
        "×": "x",
        "•": "*",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = value.replace("\\[", "").replace("\\]", "")
    value = value.replace("\\(", "").replace("\\)", "")
    value = value.replace("\\|", "|")
    value = re.sub(r"\\operatorname\{([^}]*)\}", r"\1", value)
    value = re.sub(r"\\text\{([^}]*)\}", r"\1", value)
    value = value.replace("\\approx", "approx").replace("\\le", "<=")
    value = value.replace("\\cdot", "*").replace("\\sqrt", "sqrt")
    value = value.replace("\\infty", "infinity")
    value = value.replace("\\", "")
    value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
    value = value.replace("`", "")
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


class GuidePdf(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 5, "SPAI-GMRES-IR team learning guide", 0, 1, "R")
        self.ln(1)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 5, f"Page {self.page_no()}", 0, 0, "C")

    def write_heading(self, text: str, level: int) -> None:
        sizes = {1: 18, 2: 14, 3: 11}
        self.ln(3 if level == 1 else 2)
        self.set_font("Helvetica", "B", sizes.get(level, 10))
        self.set_text_color(20, 48, 84)
        self.multi_cell(0, 7 if level == 1 else 5, ascii_text(text))
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def write_body(self, text: str, indent: float = 0, code: bool = False) -> None:
        if not text.strip():
            self.ln(2)
            return
        font = "Courier" if code else "Helvetica"
        size = 8.5 if code else 10
        self.set_font(font, "", size)
        self.set_x(self.l_margin + indent)
        width = self.w - self.r_margin - self.l_margin - indent
        self.multi_cell(width, 4.8 if code else 5.2, ascii_text(text))
        self.ln(0.6)

    def write_equation(self, expression: str) -> None:
        """Typeset a Markdown display-math block as an image.

        FPDF's built-in fonts do not understand LaTeX commands.  Matplotlib's
        math renderer gives the handouts real fractions, roots, subscripts,
        superscripts, Greek letters, and norm symbols instead of stripped text.
        """
        expression = " ".join(expression.split())
        expression = expression.replace(r"\lVert", r"\Vert").replace(r"\rVert", r"\Vert")
        expression = expression.replace(r"\operatorname", r"\mathrm")
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                image_path = Path(handle.name)
            figure = plt.figure(figsize=(7.2, 0.7), dpi=180)
            figure.patch.set_alpha(0)
            figure.text(0.5, 0.5, f"${expression}$", ha="center", va="center", fontsize=13)
            figure.savefig(image_path, transparent=True, bbox_inches="tight", pad_inches=0.08)
            plt.close(figure)
            if self.get_y() > self.h - 38:
                self.add_page()
            self.image(str(image_path), x=self.l_margin + 5, w=self.w - self.l_margin - self.r_margin - 10)
            self.ln(2)
            image_path.unlink(missing_ok=True)
        except Exception:
            # A readable fallback is better than aborting every PDF because one
            # unusual LaTeX command is unsupported by Matplotlib mathtext.
            plt.close("all")
            self.write_body(expression, indent=5, code=True)


def render_markdown(path: Path) -> None:
    pdf = GuidePdf(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(16, 14, 16)
    pdf.add_page()

    in_code = False
    in_math = False
    paragraph: list[str] = []
    math_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            pdf.write_body(" ".join(part.strip() for part in paragraph))
            paragraph.clear()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.strip() == r"\[":
            flush_paragraph()
            in_math = True
            math_lines.clear()
            continue
        if line.strip() == r"\]" and in_math:
            pdf.write_equation(" ".join(math_lines))
            in_math = False
            math_lines.clear()
            continue
        if in_math:
            math_lines.append(line.strip())
            continue
        if line.startswith("```"):
            flush_paragraph()
            in_code = not in_code
            if not in_code:
                pdf.ln(1)
            continue
        if in_code:
            pdf.write_body(line or " ", indent=5, code=True)
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            flush_paragraph()
            pdf.write_heading(heading.group(2), len(heading.group(1)))
            continue
        if line.startswith("> "):
            flush_paragraph()
            pdf.set_fill_color(238, 244, 252)
            pdf.set_font("Helvetica", "I", 9.5)
            pdf.multi_cell(0, 5.2, ascii_text(line[2:]), fill=True)
            pdf.ln(1)
            continue
        if line.startswith("|"):
            flush_paragraph()
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(re.fullmatch(r"[: -]+", cell) for cell in cells):
                continue
            pdf.write_body(" - ".join(cells), indent=2)
            continue
        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        numbered = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
        if bullet or numbered:
            flush_paragraph()
            content = bullet.group(1) if bullet else f"{numbered.group(1)}. {numbered.group(2)}"
            pdf.write_body("- " + content, indent=4)
            continue
        if not line.strip():
            flush_paragraph()
            continue
        paragraph.append(line)

    flush_paragraph()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / f"{path.stem}.pdf"
    pdf.output(str(target))


def main() -> None:
    for guide in GUIDES:
        render_markdown(ROOT / guide)


if __name__ == "__main__":
    main()
