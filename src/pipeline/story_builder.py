from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

from src.pipeline.storyboarder import make_stub_storyboard


@dataclass
class StoryBook:
    title: str
    subtitle: str
    pages: list[str]


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


def _fallback_story(transcript: str) -> StoryBook:
    """
    No-API fallback: expands transcript into a simple multi-page kids story.
    Deterministic and safe to run offline.
    """
    t = _clean_transcript(transcript)

    # Naive title inference
    title = "My Story"
    if "pizza" in t.lower():
        title = "The Pizza Crust Mystery"
    elif "monster" in t.lower():
        title = "The Friendly Monster Surprise"

    subtitle = "A story told out loud"

    # Expand into a simple narrative (lightly)
    base = (
        f"Once upon a time, there was a kid named Claire.\n\n"
        f"One day, something small felt like a big deal: {t}\n\n"
        f"At first, Claire felt nervous. What if Dad found out?\n\n"
        f"But then Claire remembered something important: telling the truth is braver than hiding.\n\n"
        f"So Claire took a deep breath and decided to fix it.\n\n"
        f"In the end, the problem got smaller, the lesson got bigger, and everyone felt better.\n\n"
        f"The end."
    )

    # Paginate: ~500 chars/page is a decent “picture book” feel
    chunks: List[str] = []
    buf: List[str] = []
    cur = 0
    for para in base.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        add_len = len(para) + (2 if buf else 0)
        if cur + add_len > 520 and buf:
            chunks.append("\n\n".join(buf))
            buf = [para]
            cur = len(para)
        else:
            buf.append(para)
            cur += add_len
    if buf:
        chunks.append("\n\n".join(buf))

    return StoryBook(title=title, subtitle=subtitle, pages=chunks)


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
    return _fallback_story(transcript)
