from pathlib import Path
import textwrap

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def build_book_pdf(out_path: Path, title: str, pages: list[str]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(out_path), pagesize=letter)
    W, H = letter

    # Cover
    c.setFont("Helvetica-Bold", 28)
    c.drawString(1.0 * inch, H - 1.5 * inch, title.strip() or "My Story")
    c.setFont("Helvetica", 14)
    c.drawString(1.0 * inch, H - 2.0 * inch, "A story told out loud")
    c.showPage()

    # Story pages
    for i, text in enumerate(pages, start=1):
        c.setFont("Helvetica-Bold", 18)
        c.drawString(1.0 * inch, H - 1.0 * inch, f"Page {i}")

        # text box
        c.setFont("Helvetica", 14)
        wrapped = textwrap.wrap((text or "").strip(), width=70)
        y = H - 1.5 * inch
        for line in wrapped[:12]:
            c.drawString(1.0 * inch, y, line)
            y -= 0.28 * inch
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas


def _wrap_text(text: str, max_width: float, font: str, font_size: int):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        test = f"{current} {word}"
        if pdfmetrics.stringWidth(test, font, font_size) <= max_width:
            current = test
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def build_book_pdf(out_path: Path, title: str, pages: list[str]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=letter)
    W, H = letter
    margin = 54
    line_height = 18

    for idx, page_text in enumerate(pages, start=1):
        if idx == 1:
            c.setFont("Helvetica-Bold", 20)
            c.drawString(margin, H - margin, title)
            y = H - margin - 32
        else:
            y = H - margin

        c.setFont("Helvetica", 14)
        lines = _wrap_text(page_text, W - 2 * margin, "Helvetica", 14)
        for line in lines:
            if y < margin:
                c.showPage()
                c.setFont("Helvetica", 14)
                y = H - margin
            c.drawString(margin, y, line)
            y -= line_height

        c.showPage()

    c.save()
    return out_path
