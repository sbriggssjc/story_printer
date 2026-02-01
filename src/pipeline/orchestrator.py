from datetime import datetime
from pathlib import Path

from src.pipeline.book_builder import build_book_pdf
from src.pipeline.story_builder import build_storybook

OUT_BOOKS = Path("out") / "books"


def run_once(transcript: str, pages: list[str] | None = None) -> Path:
    book = build_storybook(transcript, pages=pages)

    OUT_BOOKS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_pdf = OUT_BOOKS / f"story_{ts}.pdf"

    build_book_pdf(
        out_path=out_pdf,
        title=book.title,
        subtitle=book.subtitle,
        pages=book.pages,
    )
    return out_pdf
