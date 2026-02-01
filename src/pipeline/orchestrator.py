from pathlib import Path
from datetime import datetime

from src.pipeline.book_builder import build_book_pdf

OUT_BOOKS = Path("out") / "books"

def run_once(transcript: str) -> Path:
    transcript = (transcript or "").strip()

    if transcript:
        story_title = guess_title_from_transcript(transcript)
        pages = pages_from_transcript(transcript, max_pages=6)
    else:
        story_title = "The Amazing Story"
        pages = [
            "Once upon a time, a kid had a big idea.",
            "Then something surprising popped up!",
            "They decided to be brave and try.",
            "A silly problem got in the way.",
            "But teamwork made it easy.",
            "And that’s how the adventure ended happily.",
        ]

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

def pages_from_transcript(t: str, max_pages: int = 6):
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
