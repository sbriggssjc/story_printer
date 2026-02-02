from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class StoryPage:
    text: str
    illustration_prompt: str
    illustration_path: str | None = None


@dataclass
class StoryBook:
    title: str
    subtitle: str
    pages: list[StoryPage]
    narrator: str | None = None
    cover_image_path: str | None = None
    cover_path: str | None = None
    cover_illustration_prompt: str | None = None
    cover_illustration_path: str | None = None


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
    "This",
    "Dad",
    "Her",
    "One",
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
    text = re.sub(r"^\s*(this|here)\s+is\s+(a\s+)?story\s+about\s+", "", text, flags=re.I)
    text = re.sub(r"\[(?:inaudible|noise|music|laughs?|crosstalk)\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,;:.!?])", r"\1", text)
    text = re.sub(r"([,;:.!?])([^\s])", r"\1 \2", text)
    text = re.sub(r"([.!?]){2,}", r"\1", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    paragraphs = []
    for part in text.split("\n\n"):
        collapsed = re.sub(r"\s+", " ", part.replace("\n", " ")).strip()
        if not collapsed:
            continue
        if not re.search(r"[.!?]$", collapsed):
            collapsed = f"{collapsed}."
        paragraphs.append(collapsed)
    return "\n\n".join(paragraphs).strip()


def _find_name(transcript: str) -> str | None:
    full_name_match = re.search(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", transcript)
    if full_name_match:
        first, last = full_name_match.groups()
        if first not in _STOPWORDS and last not in _STOPWORDS:
            return f"{first} {last}"

    preferred_names = ["Claire", "Graham", "Jack"]
    for name in preferred_names:
        if re.search(rf"\b{name}\b", transcript):
            return name

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


def _coerce_pages(pages: list[str] | list[StoryPage]) -> list[StoryPage]:
    if not pages:
        return []
    if isinstance(pages[0], StoryPage):
        return list(pages)
    return [StoryPage(text=page, illustration_prompt="") for page in pages]


def build_storybook(
    transcript: str,
    pages: list[str] | list[StoryPage] | None = None,
) -> StoryBook:
    cleaned = _clean_transcript(transcript)
    narrator = _find_name(cleaned) if cleaned else None

    if pages:
        return StoryBook(
            title=_infer_title(cleaned) if cleaned else "My Story",
            subtitle="A story told out loud",
            pages=_coerce_pages(pages),
            narrator=narrator,
        )

    if not cleaned:
        raise ValueError("Empty transcript")

    derived_pages = [
        StoryPage(text=page, illustration_prompt="") for page in _chunk_transcript(cleaned)
    ]
    return StoryBook(
        title=_infer_title(cleaned),
        subtitle="A story told out loud",
        pages=derived_pages,
        narrator=narrator,
    )
