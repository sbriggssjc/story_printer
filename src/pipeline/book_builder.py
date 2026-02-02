from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.pipeline.pdf_builder import render_story_pdf
from src.pipeline.story_builder import StoryPage


def make_book_pdf(
    title: str,
    subtitle: str,
    pages: list[str] | list[StoryPage],
    narrator: str | None = None,
    cover_illustration_path: str | None = None,
    out_dir: Path = Path("out/books"),
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"story_{timestamp}.pdf"
    return render_story_pdf(
        title=title,
        subtitle=subtitle,
        pages=pages,
        narrator=narrator,
        cover_illustration_path=cover_illustration_path,
        out_pdf=out_path,
    )
