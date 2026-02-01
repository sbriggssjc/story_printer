from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path
from .storyboarder import Storyboard

def build_pdf(story: Storyboard, image_paths_by_page, out_pdf: Path) -> Path:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_pdf), pagesize=letter)
    W, H = letter

    for page in story.pages:
        if page.page == 1:
            c.setFont('Helvetica-Bold', 20)
            c.drawString(50, H - 50, story.title)

        top = H - (90 if page.page == 1 else 60)
        panel_h = (top - 60) / 3.0
        margin_x = 50
        img_w = W - 2 * margin_x
        img_h = panel_h * 0.72
        cap_h = panel_h * 0.20

        for idx, panel in enumerate(page.panels):
            y_top = top - idx * panel_h
            img_y = y_top - img_h
            cap_y = img_y - cap_h

            img_path = image_paths_by_page[page.page][idx]
            c.drawImage(str(img_path), margin_x, img_y, width=img_w, height=img_h, preserveAspectRatio=True, anchor='c')

            c.setFont('Helvetica', 12)
            c.drawString(margin_x, cap_y + 8, panel.caption)

        c.showPage()

    c.save()
    return out_pdf
