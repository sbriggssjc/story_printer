from datetime import datetime
from pathlib import Path

from src.pipeline.book_builder import build_book_pdf
from src.pipeline.storyboarder import make_stub_storyboard

OUT_BOOKS = Path("out") / "books"


def run_once(transcript: str, pages: list[str] | None = None) -> Path:
    transcript = (transcript or "").strip()

    story = None
    if pages is None:
        if transcript:
            pages = pages_from_transcript(transcript, max_pages=6)
        else:
            story = make_stub_storyboard(transcript)
            pages = [
                panel.caption
                for page in story.pages
                for panel in page.panels
                if panel.caption
            ]
            if not pages:
                pages = ["(No story text captured.)"]

    if transcript:
        story_title = guess_title_from_transcript(transcript)
    else:
        if story is None:
            story = make_stub_storyboard(transcript)
        story_title = story.title

    OUT_BOOKS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_pdf = OUT_BOOKS / f"story_{ts}.pdf"

    build_book_pdf(
        out_path=out_pdf,
        title=story_title,
        pages=pages,
    )
    return out_pdf


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
