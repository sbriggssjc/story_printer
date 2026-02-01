from pathlib import Path
from datetime import datetime
from src.pipeline.storyboarder import make_stub_storyboard
from src.pipeline.book_builder import build_book_pdf

OUT = Path('out')
BOOKS = OUT / 'books'

def run_once(transcript: str, pages: list[str] | None = None) -> Path:
    story = make_stub_storyboard(transcript)

    if pages is None:
        pages = [transcript]

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_pdf = BOOKS / f'story_{ts}.pdf'
    build_book_pdf(out_pdf, story.title, pages)
    return out_pdf
