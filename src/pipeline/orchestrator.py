import os
from pathlib import Path

from src.pipeline.book_builder import make_book_pdf
from src.pipeline.story_enhancer import enhance_to_storybook
from src.pipeline.story_builder import build_storybook

# ADD THIS IMPORT
from src.pipeline.audio_transcriber import transcribe_audio

__all__ = ["run_once", "run_once_from_audio"]

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
        cover_image_path=getattr(book, "cover_image_path", None),
    )


def run_once_from_audio(audio_path: str, pages: list[str] | None = None) -> Path:
    """
    Build a storybook PDF from an audio recording:
    audio -> transcript -> story enhancement/build -> PDF
    """
    audio_path = Path(audio_path)
    transcript = transcribe_audio(str(audio_path))
    return run_once(transcript, pages=pages)
