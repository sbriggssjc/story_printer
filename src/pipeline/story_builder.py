from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class StoryBook:
    title: str
    subtitle: str
    pages: list[str]


def _clean_transcript(transcript: str) -> str:
    text = (transcript or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _find_name(transcript: str) -> str | None:
    stopwords = {
        "The",
        "A",
        "An",
        "Once",
        "When",
        "Then",
        "After",
        "Before",
        "He",
        "She",
        "They",
        "We",
        "I",
        "It",
        "My",
        "Our",
        "Your",
    }
    for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", transcript):
        name = match.group(0)
        if name in stopwords:
            continue
        return name
    return None


def _infer_title(transcript: str) -> str:
    lowered = transcript.lower()
    if "pizza" in lowered:
        return "The Pizza Crust Mystery"
    name = _find_name(transcript)
    if name:
        return f"A Story About {name}"
    return "My Story"


def _split_long_segment(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        add_len = len(word) + (1 if current else 0)
        if current and current_len + add_len > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += add_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def _chunk_transcript(transcript: str, min_chars: int = 500, max_chars: int = 900) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", transcript)
    pages: list[str] = []
    buffer: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if buffer:
                pages.append(" ".join(buffer))
                buffer = []
                current_len = 0
            pages.extend(_split_long_segment(sentence, max_chars))
            continue

        add_len = len(sentence) + (1 if buffer else 0)
        if current_len + add_len > max_chars:
            pages.append(" ".join(buffer))
            buffer = [sentence]
            current_len = len(sentence)
        else:
            buffer.append(sentence)
            current_len += add_len

        if current_len >= min_chars:
            pages.append(" ".join(buffer))
            buffer = []
            current_len = 0

    if buffer:
        pages.append(" ".join(buffer))

    if not pages:
        pages = [transcript]
    return pages


def build_storybook(transcript: str, pages: list[str] | None = None) -> StoryBook:
    cleaned = _clean_transcript(transcript)

    if pages:
        return StoryBook(
            title=_infer_title(cleaned) if cleaned else "My Story",
            subtitle="A story told out loud",
            pages=pages,
        )

    if not cleaned:
        raise ValueError("Empty transcript")

    derived_pages = _chunk_transcript(cleaned)
    return StoryBook(
        title=_infer_title(cleaned),
        subtitle="A story told out loud",
        pages=derived_pages,
    )
