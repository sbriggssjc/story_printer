from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

from src.pipeline.storyboarder import make_stub_storyboard


@dataclass
class StoryBook:
    title: str
    subtitle: str | None
    pages: list[object]


def guess_title_from_transcript(t: str) -> str:
    low = t.lower()
    if "pizza" in low:
        return "The Pizza Crust Mystery"
    if "monster" in low:
        return "The Surprise Monster"
    return "My Awesome Story"


def _clean_transcript(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _split_transcript(transcript: str, max_chars: int = 1100) -> list[str]:
    text = _clean_transcript(transcript)
    if not text:
        return ["(No story text captured.)"]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    buf: List[str] = []
    cur = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        add_len = len(sentence) + (1 if buf else 0)
        if cur + add_len > max_chars and buf:
            chunks.append(" ".join(buf))
            buf = [sentence]
            cur = len(sentence)
        else:
            buf.append(sentence)
            cur += add_len
    if buf:
        chunks.append(" ".join(buf))
    return chunks or ["(No story text captured.)"]


def _openai_story(transcript: str) -> Optional[StoryBook]:
    """
    Optional: if OPENAI_API_KEY is set, generate a stronger story.
    This is written to be drop-in, but won't break offline runs.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    # Lazy import so offline users don't need the dependency immediately
    try:
        from openai import OpenAI
    except Exception:
        return None

    client = OpenAI(api_key=api_key)

    t = _clean_transcript(transcript)

    system = (
        "You write short children's picture books. "
        "Output JSON only."
    )

    prompt = f"""
Turn this spoken transcript into a fun children's picture book.

Rules:
- Keep Claire as the main character if she appears.
- Make it warm and funny, ages 5–9.
- 8 to 12 pages.
- Each page: 1–3 short paragraphs, simple words.
- Add a clear beginning, middle, end, and a gentle lesson.
- Output JSON with keys: title, subtitle, pages (array of strings).

Transcript:
{t}
""".strip()

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    data = resp.output_parsed
    if not isinstance(data, dict):
        return None

    title = str(data.get("title") or "My Story").strip()
    subtitle = str(data.get("subtitle") or "A story told out loud").strip()
    pages = data.get("pages") or []
    if not isinstance(pages, list) or not pages:
        return None

    pages = [str(p).strip() for p in pages if str(p).strip()]
    if not pages:
        return None

    return StoryBook(title=title, subtitle=subtitle, pages=pages)


def build_storybook(transcript: str, pages: list[str] | None = None) -> StoryBook:
    transcript = (transcript or "").strip()

    if pages is not None:
        if transcript:
            title = guess_title_from_transcript(transcript)
        else:
            story = make_stub_storyboard(transcript)
            title = story.title
        return StoryBook(
            title=title,
            subtitle="A story told out loud",
            pages=pages or ["(No story text captured.)"],
        )

    if not transcript:
        story = make_stub_storyboard(transcript)
        pages = [
            panel.caption
            for page in story.pages
            for panel in page.panels
            if panel.caption
        ]
        if not pages:
            pages = ["(No story text captured.)"]
        return StoryBook(
            title=story.title,
            subtitle="A story told out loud",
            pages=pages,
        )

    book = _openai_story(transcript)
    if book:
        return book
    return StoryBook(
        title=guess_title_from_transcript(transcript),
        subtitle="A story told out loud",
        pages=_split_transcript(transcript),
    )
