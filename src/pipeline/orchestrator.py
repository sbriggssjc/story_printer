import os
from pathlib import Path

from src.pipeline.book_builder import make_book_pdf
from src.pipeline.story_enhancer import enhance_to_storybook
from src.pipeline.story_builder import build_storybook


def run_once(transcript: str, pages: list[str] | None = None) -> Path:
    if pages:
        book = build_storybook(transcript, pages=pages)
    else:
        target_pages = int(os.getenv("STORY_TARGET_PAGES", "2"))
        book = enhance_to_storybook(transcript, target_pages=target_pages)

    return make_book_pdf(
        title=book.title,
        subtitle=book.subtitle,
        pages=book.pages,
        narrator=book.narrator,
    )
