from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas


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


def build_book_pdf(
    out_path: Path,
    title: str,
    pages: list[str],
    subtitle: str | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(out_path), pagesize=letter)
    W, H = letter
    margin = 54
    line_height = 18

    cover_title = title.strip() or "My Story"
    c.setFont("Helvetica-Bold", 28)
    c.drawString(1.0 * inch, H - 1.5 * inch, cover_title)
    c.setFont("Helvetica", 14)
    cover_subtitle = (subtitle or "A story told out loud").strip()
    if cover_subtitle:
        c.drawString(1.0 * inch, H - 2.0 * inch, cover_subtitle)
    c.showPage()

    for idx, page_text in enumerate(pages, start=1):
        c.setFont("Helvetica-Bold", 18)
        c.drawString(margin, H - margin, f"Page {idx}")
        y = H - margin - 24

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

        c.showPage()

    c.save()
    return out_path
