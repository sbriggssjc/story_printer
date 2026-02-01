from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.pipeline.pdf_builder import render_story_pdf


def make_book_pdf(
    title: str,
    subtitle: str,
    pages: list[str],
    out_dir: Path = Path("out/books"),
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"story_{timestamp}.pdf"
    return render_story_pdf(title=title, subtitle=subtitle, pages=pages, out_pdf=out_path)
