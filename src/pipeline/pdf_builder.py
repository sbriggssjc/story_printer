from __future__ import annotations

from datetime import datetime
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


def _extract_paragraphs(pages: list[str]) -> list[str]:
    paragraphs: list[str] = []
    for page_text in pages:
        if not page_text:
            continue
        for paragraph in page_text.split("\n\n"):
            cleaned = paragraph.strip()
            if cleaned:
                paragraphs.append(cleaned)
    return paragraphs


def _draw_cover_page(
    canvas_pdf: canvas.Canvas,
    page_width: float,
    page_height: float,
    margin: float,
    title: str,
    subtitle: str,
    narrator: str | None,
) -> None:
    cover_title = (title or "My Story").strip()
    cover_subtitle = (subtitle or "").strip()

    canvas_pdf.setFont("Helvetica-Bold", 30)
    canvas_pdf.drawCentredString(page_width / 2, page_height - 2.5 * margin, cover_title)

    offset = 3.25 * margin
    if cover_subtitle:
        canvas_pdf.setFont("Helvetica", 15)
        canvas_pdf.drawCentredString(
            page_width / 2,
            page_height - offset,
            cover_subtitle,
        )
        offset += 0.6 * margin

    if narrator:
        canvas_pdf.setFont("Helvetica-Oblique", 13)
        canvas_pdf.drawCentredString(
            page_width / 2,
            page_height - offset,
            f"Told by {narrator}",
        )
        offset += 0.6 * margin

    today = datetime.now().strftime("%B %d, %Y")
    canvas_pdf.setFont("Helvetica", 11)
    canvas_pdf.drawCentredString(page_width / 2, margin * 1.25, today)
    canvas_pdf.showPage()


def _draw_end_page(
    canvas_pdf: canvas.Canvas,
    page_width: float,
    page_height: float,
) -> None:
    canvas_pdf.setFont("Helvetica-Bold", 28)
    canvas_pdf.drawCentredString(page_width / 2, page_height / 2, "The End")
    canvas_pdf.showPage()


def render_story_pdf(
    title: str,
    subtitle: str,
    pages: list[str],
    out_pdf: Path,
    narrator: str | None = None,
) -> Path:
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    canvas_pdf = canvas.Canvas(str(out_pdf), pagesize=letter)
    page_width, page_height = letter
    margin = 72
    footer_height = margin * 0.6

    _draw_cover_page(
        canvas_pdf=canvas_pdf,
        page_width=page_width,
        page_height=page_height,
        margin=margin,
        title=title,
        subtitle=subtitle,
        narrator=narrator,
    )

    body_font = "Helvetica"
    body_size = 13
    line_height = body_size * 1.4
    paragraph_spacing = line_height * 0.6
    max_width = page_width - 2 * margin
    bottom_limit = margin + footer_height

    paragraphs = _extract_paragraphs(pages)
    if not paragraphs:
        paragraphs = [""]

    page_number = 1
    y = page_height - margin
    canvas_pdf.setFont(body_font, body_size)

    def finish_story_page() -> None:
        nonlocal page_number
        page_label = str(page_number)
        canvas_pdf.setFont("Helvetica", 10)
        number_width = pdfmetrics.stringWidth(page_label, "Helvetica", 10)
        canvas_pdf.drawString(page_width - margin - number_width, margin / 2, page_label)
        canvas_pdf.showPage()
        page_number += 1
        canvas_pdf.setFont(body_font, body_size)

    for index, paragraph in enumerate(paragraphs):
        lines = _wrap_text(paragraph, max_width, body_font, body_size)
        for line in lines:
            if y - line_height < bottom_limit:
                finish_story_page()
                y = page_height - margin
            canvas_pdf.drawString(margin, y, line)
            y -= line_height

        if index < len(paragraphs) - 1:
            if y - paragraph_spacing < bottom_limit:
                finish_story_page()
                y = page_height - margin
            else:
                y -= paragraph_spacing

    finish_story_page()
    _draw_end_page(canvas_pdf, page_width, page_height)

    canvas_pdf.save()
    return out_pdf
