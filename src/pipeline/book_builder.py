from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

OUT_BOOKS = Path("out") / "books"


def _wrap_text(text: str, max_width: float, font: str, font_size: int) -> list[str]:
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


def make_book_pdf(title: str, subtitle: str, pages: list[str]) -> Path:
    OUT_BOOKS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_BOOKS / f"story_{ts}.pdf"

    c = canvas.Canvas(str(out_path), pagesize=letter)
    W, H = letter
    margin = 54
    line_height = 18

    cover_title = title.strip() or "My Story"
    cover_subtitle = subtitle.strip() or "A story told out loud"
    c.setFont("Helvetica-Bold", 28)
    c.drawString(1.0 * inch, H - 1.5 * inch, cover_title)
    c.setFont("Helvetica", 14)
    c.drawString(1.0 * inch, H - 2.0 * inch, cover_subtitle)
    c.showPage()

    for idx, page_text in enumerate(pages, start=1):
        y = H - margin
        c.setFont("Helvetica", 14)
        lines = _wrap_text((page_text or "").strip(), W - 2 * margin, "Helvetica", 14)
        if not lines:
            lines = [""]
        for line in lines:
            if y < margin:
                c.showPage()
                c.setFont("Helvetica", 14)
                y = H - margin
            c.drawString(margin, y, line)
            y -= line_height

        page_number = str(idx + 1)
        c.setFont("Helvetica", 10)
        number_width = pdfmetrics.stringWidth(page_number, "Helvetica", 10)
        c.drawString((W - number_width) / 2, margin / 2, page_number)

        c.showPage()

    c.save()
    return out_path
