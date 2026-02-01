from pathlib import Path
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
