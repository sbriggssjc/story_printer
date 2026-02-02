from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from src.pipeline.story_builder import StoryPage


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


def _wrap_paragraphs(text: str, max_width: float, font: str, font_size: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [""]

    lines: list[str] = []
    for index, paragraph in enumerate(paragraphs):
        lines.extend(_wrap_text(paragraph, max_width, font, font_size))
        if index < len(paragraphs) - 1:
            lines.append("")
    return lines


def _coerce_story_pages(pages: list[str] | list[StoryPage]) -> list[StoryPage]:
    if not pages:
        return [StoryPage(text="", illustration_prompt="")]
    if isinstance(pages[0], StoryPage):
        return list(pages)
    return [StoryPage(text=page, illustration_prompt="") for page in pages]


def _draw_cover_page(
    canvas_pdf: canvas.Canvas,
    page_width: float,
    page_height: float,
    margin: float,
    title: str,
    subtitle: str,
    narrator: str | None,
    cover_image_path: str | None,
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

    if cover_image_path and Path(cover_image_path).exists():
        img = ImageReader(cover_image_path)
        box_w = page_width - 2 * margin
        top_y = page_height - offset - (0.5 * margin)
        bottom_y = margin * 2
        if top_y > bottom_y:
            box_h = top_y - bottom_y
            box_x = margin
            box_y = bottom_y
            img_w, img_h = img.getSize()
            scale = min(box_w / img_w, box_h / img_h)
            draw_w = img_w * scale
            draw_h = img_h * scale
            offset_x = box_x + (box_w - draw_w) / 2
            offset_y = box_y + (box_h - draw_h) / 2
            canvas_pdf.drawImage(img, offset_x, offset_y, draw_w, draw_h, preserveAspectRatio=True)

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
    pages: list[str] | list[StoryPage],
    out_pdf: Path,
    narrator: str | None = None,
    cover_image_path: str | None = None,
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
        cover_image_path=cover_image_path,
    )

    body_font = "Helvetica"
    body_size = 13
    max_width = page_width - 2 * margin
    bottom_limit = margin + footer_height

    page_number = 1

    def finish_story_page() -> None:
        nonlocal page_number
        page_label = str(page_number)
        canvas_pdf.setFont("Helvetica", 10)
        number_width = pdfmetrics.stringWidth(page_label, "Helvetica", 10)
        canvas_pdf.drawString(page_width - margin - number_width, margin / 2, page_label)
        canvas_pdf.showPage()
        page_number += 1

    def draw_illustration_box(page: StoryPage, box_x: float, box_y: float, box_w: float, box_h: float) -> None:
        image_path = getattr(page, "image_path", None) or getattr(page, "illustration_path", None)
        if image_path and Path(image_path).exists():
            image = ImageReader(image_path)
            img_w, img_h = image.getSize()
            scale = min(box_w / img_w, box_h / img_h)
            draw_w = img_w * scale
            draw_h = img_h * scale
            offset_x = box_x + (box_w - draw_w) / 2
            offset_y = box_y + (box_h - draw_h) / 2
            canvas_pdf.drawImage(image, offset_x, offset_y, draw_w, draw_h, preserveAspectRatio=True)
            return

        canvas_pdf.setStrokeColor(colors.lightgrey)
        canvas_pdf.setFillColor(colors.whitesmoke)
        canvas_pdf.rect(box_x, box_y, box_w, box_h, stroke=1, fill=1)
        canvas_pdf.setFillColor(colors.black)
        prompt = page.illustration_prompt.strip() or "Illustration: A cozy children's storybook scene."
        prompt_text = f"Illustration: {prompt}"
        prompt_font = "Helvetica-Oblique"
        prompt_size = 10
        padding = 10
        prompt_lines = _wrap_text(prompt_text, box_w - 2 * padding, prompt_font, prompt_size)
        y_cursor = box_y + box_h - padding - prompt_size
        canvas_pdf.setFont(prompt_font, prompt_size)
        for line in prompt_lines:
            if y_cursor < box_y + padding:
                break
            canvas_pdf.drawString(box_x + padding, y_cursor, line)
            y_cursor -= prompt_size * 1.3

    def draw_story_text(page: StoryPage, text_top: float) -> None:
        max_font = body_size
        min_font = 9
        font_size = max_font
        while font_size >= min_font:
            line_height = font_size * 1.4
            paragraph_spacing = line_height * 0.45
            lines = _wrap_paragraphs(page.text, max_width, body_font, font_size)
            total_height = 0.0
            for line in lines:
                total_height += paragraph_spacing if line == "" else line_height
            if text_top - total_height >= bottom_limit:
                break
            font_size -= 1

        canvas_pdf.setFont(body_font, font_size)
        line_height = font_size * 1.4
        paragraph_spacing = line_height * 0.45
        lines = _wrap_paragraphs(page.text, max_width, body_font, font_size)
        y_cursor = text_top
        for line in lines:
            if line == "":
                y_cursor -= paragraph_spacing
                continue
            canvas_pdf.drawString(margin, y_cursor, line)
            y_cursor -= line_height

    story_pages = _coerce_story_pages(pages)
    for page in story_pages:
        illustration_height = page_height * 0.4
        illustration_top = page_height - margin
        illustration_bottom = illustration_top - illustration_height
        draw_illustration_box(page, margin, illustration_bottom, max_width, illustration_height)

        text_top = illustration_bottom - (margin * 0.25)
        draw_story_text(page, text_top)
        finish_story_page()

    _draw_end_page(canvas_pdf, page_width, page_height)

    canvas_pdf.save()
    return out_pdf
