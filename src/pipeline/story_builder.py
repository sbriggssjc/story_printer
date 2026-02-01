from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class StoryBook:
    title: str
    subtitle: str
    pages: list[str]
    narrator: str | None = None


_STOPWORDS = {
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
    "And",
    "But",
    "Or",
    "In",
    "On",
    "At",
    "With",
    "From",
    "To",
}


def _clean_transcript(transcript: str) -> str:
    text = (transcript or "").strip()
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    paragraphs = []
    for part in text.split("\n\n"):
        collapsed = re.sub(r"\s+", " ", part.replace("\n", " ")).strip()
        if collapsed:
            paragraphs.append(collapsed)
    return "\n\n".join(paragraphs).strip()


def _find_name(transcript: str) -> str | None:
    for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", transcript):
        name = match.group(0)
        if name in _STOPWORDS:
            continue
        return name
    return None


def _infer_title(transcript: str) -> str:
    lowered = transcript.lower()
    if "pizza" in lowered:
        return "The Pizza Crust Mystery"
    if "dragon" in lowered:
        return "The Dragon's Tale"
    if "monster" in lowered:
        return "The Monster's Surprise"
    if "castle" in lowered:
        return "The Castle Adventure"
    name = _find_name(transcript)
    if name:
        return f"{name}'s Storybook Adventure"
    tokens = re.findall(r"[A-Za-z']+", transcript)
    for token in tokens:
        if token.capitalize() not in _STOPWORDS and len(token) > 3:
            return f"The {token.capitalize()} Adventure"
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


def _split_into_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _chunk_transcript(transcript: str, min_chars: int = 450, max_chars: int = 900) -> list[str]:
    raw_paragraphs = re.split(r"\n\s*\n", transcript)
    chunks: list[str] = []

    for raw_paragraph in raw_paragraphs:
        paragraph = raw_paragraph.strip()
        if not paragraph:
            continue
        sentences = _split_into_sentences(paragraph)
        buffer: list[str] = []
        current_len = 0

        for sentence in sentences:
            if len(sentence) > max_chars:
                if buffer:
                    chunks.append(" ".join(buffer))
                    buffer = []
                    current_len = 0
                chunks.extend(_split_long_segment(sentence, max_chars))
                continue

            add_len = len(sentence) + (1 if buffer else 0)
            if current_len + add_len > max_chars:
                chunks.append(" ".join(buffer))
                buffer = [sentence]
                current_len = len(sentence)
            else:
                buffer.append(sentence)
                current_len += add_len

            if current_len >= min_chars:
                chunks.append(" ".join(buffer))
                buffer = []
                current_len = 0

        if buffer:
            chunks.append(" ".join(buffer))

    if not chunks:
        chunks = [transcript]
    return chunks


def build_storybook(transcript: str, pages: list[str] | None = None) -> StoryBook:
    cleaned = _clean_transcript(transcript)
    narrator = _find_name(cleaned) if cleaned else None

    if pages:
        return StoryBook(
            title=_infer_title(cleaned) if cleaned else "My Story",
            subtitle="A story told out loud",
            pages=pages,
            narrator=narrator,
        )

    if not cleaned:
        raise ValueError("Empty transcript")

    derived_pages = _chunk_transcript(cleaned)
    return StoryBook(
        title=_infer_title(cleaned),
        subtitle="A story told out loud",
        pages=derived_pages,
        narrator=narrator,
    )
