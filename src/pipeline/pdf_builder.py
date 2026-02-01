from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas


def _wrap_text(text: str, max_width: float, font: str, font_size: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
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


def render_story_pdf(
    title: str,
    subtitle: str,
    pages: list[str],
    out_pdf: Path,
) -> Path:
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    canvas_pdf = canvas.Canvas(str(out_pdf), pagesize=letter)
    page_width, page_height = letter
    margin = 72

    cover_title = (title or "My Story").strip()
    cover_subtitle = (subtitle or "").strip()

    canvas_pdf.setFont("Helvetica-Bold", 30)
    canvas_pdf.drawCentredString(page_width / 2, page_height - 2.5 * margin, cover_title)
    if cover_subtitle:
        canvas_pdf.setFont("Helvetica", 15)
        canvas_pdf.drawCentredString(
            page_width / 2,
            page_height - 3.25 * margin,
            cover_subtitle,
        )
    canvas_pdf.showPage()

    body_font = "Helvetica"
    body_size = 13
    line_height = 18

    for index, page_text in enumerate(pages, start=1):
        y = page_height - margin
        canvas_pdf.setFont(body_font, body_size)
        lines = _wrap_text((page_text or "").strip(), page_width - 2 * margin, body_font, body_size)
        for line in lines:
            if y < margin:
                break
            canvas_pdf.drawString(margin, y, line)
            y -= line_height

        page_number = str(index)
        canvas_pdf.setFont("Helvetica", 10)
        number_width = pdfmetrics.stringWidth(page_number, "Helvetica", 10)
        canvas_pdf.drawString(page_width - margin - number_width, margin / 2, page_number)
        canvas_pdf.showPage()

    canvas_pdf.save()
    return out_pdf
