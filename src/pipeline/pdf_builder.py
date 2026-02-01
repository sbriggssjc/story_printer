from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from .storyboarder import Storyboard


def _wrap_text(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
    """
    Simple word-wrap using ReportLab font metrics.
    Returns a list of lines that fit within max_width.
    """
    if not text:
        return [""]

    words = text.replace("\r", " ").split()
    lines: List[str] = []
    cur: List[str] = []

    for w in words:
        trial = (" ".join(cur + [w])).strip()
        if pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
                cur = [w]
            else:
                # single long word: hard cut
                lines.append(w)
                cur = []

    if cur:
        lines.append(" ".join(cur))

    return lines or [""]


def _draw_centered(c: canvas.Canvas, text: str, y: float, font_name: str, font_size: int, W: float):
    c.setFont(font_name, font_size)
    tw = pdfmetrics.stringWidth(text, font_name, font_size)
    c.drawString(max(0, (W - tw) / 2.0), y, text)


def _safe_get_image(image_paths_by_page: Dict[int, List[Path]], page_num: int, idx: int) -> Optional[Path]:
    try:
        arr = image_paths_by_page.get(page_num) or image_paths_by_page.get(int(page_num)) or []
        if 0 <= idx < len(arr):
            p = arr[idx]
            if p and Path(p).exists():
                return Path(p)
    except Exception:
        pass
    return None


def _draw_image_or_placeholder(
    c: canvas.Canvas,
    img_path: Optional[Path],
    x: float,
    y: float,
    w: float,
    h: float,
    label: str = "Image missing",
):
    if img_path and img_path.exists():
        try:
            # ImageReader avoids some edge cases with certain formats
            c.drawImage(
                ImageReader(str(img_path)),
                x,
                y,
                width=w,
                height=h,
                preserveAspectRatio=True,
                anchor="c",
                mask="auto",
            )
            return
        except Exception:
            # fall through to placeholder
            pass

    # Placeholder box
    c.rect(x, y, w, h, stroke=1, fill=0)
    c.setFont("Helvetica-Oblique", 12)
    c.drawString(x + 10, y + h / 2.0, label)


def _collect_story_text(story: Storyboard) -> str:
    # Best effort: transcript/text if present, otherwise join captions
    for attr in ("transcript", "text", "raw_text"):
        if hasattr(story, attr):
            v = getattr(story, attr)
            if isinstance(v, str) and v.strip():
                return v.strip()

    caps: List[str] = []
    for p in getattr(story, "pages", []) or []:
        for panel in getattr(p, "panels", []) or []:
            cap = getattr(panel, "caption", "") or ""
            if cap.strip():
                caps.append(cap.strip())

    return " ".join(caps).strip()


def build_pdf(story: Storyboard, image_paths_by_page: Dict[int, List[Path]], out_pdf: Path) -> Path:
    """
    Renders:
      1) Cover page
      2) Story pages with up to 3 panels per page (image + wrapped caption)
      3) If no panels exist, prints full story text across pages
    """
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(out_pdf), pagesize=letter)
    W, H = letter

    title = (getattr(story, "title", None) or "My Story").strip()

    # -----------------------
    # Cover page
    # -----------------------
    _draw_centered(c, title, H * 0.62, "Helvetica-Bold", 28, W)
    _draw_centered(c, "A Story Book", H * 0.55, "Helvetica", 16, W)
    _draw_centered(c, datetime.now().strftime("%b %d, %Y"), H * 0.48, "Helvetica-Oblique", 12, W)
    c.showPage()

    pages = getattr(story, "pages", []) or []

    # Do we actually have any panels?
    has_any_panels = any((getattr(p, "panels", None) or []) for p in pages)

    # -----------------------
    # Fallback: text-only book
    # -----------------------
    if not has_any_panels:
        body = _collect_story_text(story) or "Once upon a time..."
        margin = 60
        max_w = W - 2 * margin
        y = H - 80

        c.setFont("Helvetica", 13)
        lines = _wrap_text(body, "Helvetica", 13, max_w)

        page_num = 1
        for line in lines:
            if y < 80:
                # page number
                c.setFont("Helvetica-Oblique", 10)
                c.drawString(margin, 40, f"Page {page_num}")
                c.showPage()
                page_num += 1
                c.setFont("Helvetica", 13)
                y = H - 80

            c.drawString(margin, y, line)
            y -= 18

        c.setFont("Helvetica-Oblique", 10)
        c.drawString(margin, 40, f"Page {page_num}")
        c.save()
        return out_pdf

    # -----------------------
    # Panel pages
    # -----------------------
    margin_x = 50
    margin_bottom = 60
    content_top = H - 60
    content_h = content_top - margin_bottom
    panel_h = content_h / 3.0

    img_w = W - 2 * margin_x
    img_h = panel_h * 0.70
    cap_area_h = panel_h * 0.25

    story_page_counter = 1  # excludes cover
    for p in pages:
        panels = getattr(p, "panels", []) or []

        # Always produce a page for each story page object, even if panels are missing
        for idx in range(3):
            y_top = content_top - idx * panel_h
            img_y = y_top - img_h
            cap_y = img_y - cap_area_h + 12

            panel_caption = ""
            if idx < len(panels):
                panel_caption = (getattr(panels[idx], "caption", "") or "").strip()

            img_path = _safe_get_image(image_paths_by_page, getattr(p, "page", story_page_counter), idx)

            _draw_image_or_placeholder(
                c,
                img_path,
                margin_x,
                img_y,
                img_w,
                img_h,
                label="(no illustration generated)",
            )

            # Caption (wrapped)
            c.setFont("Helvetica", 12)
            max_caption_w = img_w
            lines = _wrap_text(panel_caption, "Helvetica", 12, max_caption_w) if panel_caption else [""]
            y_line = cap_y
            for line in lines[:4]:  # prevent runaway captions
                c.drawString(margin_x, y_line, line)
                y_line -= 14

        # page number
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(margin_x, 40, f"Page {story_page_counter}")
        story_page_counter += 1

        c.showPage()

    c.save()
    return out_pdf
