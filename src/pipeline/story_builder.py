from __future__ import annotations

from dataclasses import dataclass

from src.pipeline.storyboarder import make_stub_storyboard


@dataclass
class StoryBook:
    title: str
    subtitle: str
    pages: list[str]


def build_storybook(transcript: str, pages: list[str] | None = None) -> StoryBook:
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

    return StoryBook(
        title=story_title,
        subtitle="A story told out loud",
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
