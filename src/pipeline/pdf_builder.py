from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

OUT_BOOKS = Path("out") / "books"


@dataclass
class PageContent:
    text: str
    image_path: Path | None = None
    caption: str | None = None


def _coerce_page(page: object) -> PageContent:
    if isinstance(page, PageContent):
        return page
    if isinstance(page, str):
        return PageContent(text=page)
    if isinstance(page, dict):
        text = str(page.get("text") or "")
        image = page.get("image") or page.get("image_path")
        image_path = Path(image) if image else None
        caption = page.get("caption")
        return PageContent(text=text, image_path=image_path, caption=caption)

    text = getattr(page, "text", "")
    image = getattr(page, "image_path", None) or getattr(page, "image", None)
    caption = getattr(page, "caption", None)
    image_path = Path(image) if image else None
    return PageContent(text=str(text or ""), image_path=image_path, caption=caption)


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


def _iter_pages(pages: Iterable[object]) -> list[PageContent]:
    return [_coerce_page(page) for page in pages]


def make_book_pdf(
    title: str,
    subtitle: str | None,
    pages: list[object],
    out_dir: Path | None = None,
    filename_prefix: str = "story",
) -> Path:
    out_dir = out_dir or OUT_BOOKS
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{filename_prefix}_{ts}.pdf"
    return build_book_pdf(
        out_path=out_path,
        title=title,
        subtitle=subtitle,
        pages=pages,
    )


def build_book_pdf(
    out_path: Path,
    title: str,
    pages: list[object],
    subtitle: str | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(out_path), pagesize=letter)
    W, H = letter
    margin = 54
    line_height = 18

    cover_title = title.strip() or "My Story"
    cover_subtitle = (subtitle or "A story told out loud").strip()
    c.setFont("Helvetica-Bold", 28)
    c.drawString(1.0 * inch, H - 1.5 * inch, cover_title)
    c.setFont("Helvetica", 14)
    if cover_subtitle:
        c.drawString(1.0 * inch, H - 2.0 * inch, cover_subtitle)
    c.showPage()

    rendered_pages = _iter_pages(pages)
    for idx, page in enumerate(rendered_pages, start=1):
        y = H - margin

        if page.image_path and page.image_path.exists():
            try:
                image_reader = ImageReader(str(page.image_path))
                img_w, img_h = image_reader.getSize()
                max_w = W - 2 * margin
                max_h = H * 0.35
                scale = min(max_w / img_w, max_h / img_h, 1.0)
                draw_w = img_w * scale
                draw_h = img_h * scale
                x = (W - draw_w) / 2
                y -= draw_h
                c.drawImage(
                    image_reader,
                    x,
                    y,
                    width=draw_w,
                    height=draw_h,
                    preserveAspectRatio=True,
                    anchor="n",
                )
                y -= 12
                if page.caption:
                    c.setFont("Helvetica-Oblique", 10)
                    caption_lines = _wrap_text(
                        str(page.caption),
                        W - 2 * margin,
                        "Helvetica-Oblique",
                        10,
                    )
                    for line in caption_lines:
                        if y < margin:
                            c.showPage()
                            y = H - margin
                        c.drawString(margin, y, line)
                        y -= 12
                    y -= 6
            except Exception:
                pass

        c.setFont("Helvetica", 14)
        lines = _wrap_text((page.text or "").strip(), W - 2 * margin, "Helvetica", 14)
        if not lines:
            lines = [""]
        for line in lines:
            if y < margin:
                c.showPage()
                c.setFont("Helvetica", 14)
                y = H - margin
            c.drawString(margin, y, line)
            y -= line_height

        page_number = str(idx)
        c.setFont("Helvetica", 10)
        number_width = pdfmetrics.stringWidth(page_number, "Helvetica", 10)
        c.drawString((W - number_width) / 2, margin / 2, page_number)

        c.showPage()

    c.save()
    return out_path
