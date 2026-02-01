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

        c.showPage()

    c.save()
