from pathlib import Path

from src.pipeline.book_builder import make_book_pdf
from src.pipeline.story_builder import build_storybook


def run_once(transcript: str, pages: list[str] | None = None) -> Path:
    book = build_storybook(transcript, pages=pages)

    return make_book_pdf(
        title=book.title,
        subtitle=book.subtitle,
        pages=book.pages,
    )
