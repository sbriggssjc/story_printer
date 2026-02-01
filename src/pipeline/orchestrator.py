from pathlib import Path

from src.pipeline.book_builder import make_book_pdf
from src.pipeline.storyboarder import make_stub_storyboard
from src.pipeline.book_builder import build_book_pdf
from src.pipeline.story_builder import build_storybook


def run_once(transcript: str, pages: list[str] | None = None) -> Path:
    book = build_storybook(transcript, pages=pages)

    return make_book_pdf(
        title=story_title,
        subtitle="",
        pages=pages,
    )


def guess_title_from_transcript(t: str) -> str:
    low = t.lower()
    if "pizza" in low:
        return "The Pizza Crust Mystery"
    if "monster" in low:
        return "The Surprise Monster"
    return "My Awesome Story"


def pages_from_transcript(t: str, max_pages: int = 6) -> list[str]:
    import re

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    if not sents:
        return ["(No story text captured.)"]

    chunks = []
    cur = []
    for s in sents:
        cur.append(s)
        if len(cur) >= 2:
            chunks.append(" ".join(cur))
            cur = []
        if len(chunks) >= max_pages:
            break
    if cur and len(chunks) < max_pages:
        chunks.append(" ".join(cur))

    return chunks[:max_pages]
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
