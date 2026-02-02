from __future__ import annotations

import re
from dataclasses import dataclass
from collections import Counter


@dataclass
class StoryPage:
    text: str
    illustration_prompt: str
    # Back-compat: older code used illustration_path; newer code sometimes uses image_path.
    illustration_path: str | None = None
    image_path: str | None = None

    def get_image_path(self) -> str | None:
        # Prefer image_path if provided, else illustration_path
        return self.image_path or self.illustration_path

    def __post_init__(self) -> None:
        if self.image_path is None and self.illustration_path:
            self.image_path = self.illustration_path
        if self.illustration_path is None and self.image_path:
            self.illustration_path = self.image_path


@dataclass
class StoryBook:
    title: str
    subtitle: str
    pages: list[StoryPage]
    narrator: str | None = None
    # Optional cover image (for PDF cover art)
    cover_image_path: str | None = None
    cover_prompt: str | None = None


_STOPWORDS = {
    "the", "a", "an", "once", "when", "then", "after", "before",
    "he", "she", "they", "we", "i", "it", "my", "our", "your",
    "and", "but", "or", "in", "on", "at", "with", "from", "to",
    "of", "for", "as", "is", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "there", "here",
}


def _clean_transcript(transcript: str) -> str:
    text = (transcript or "").strip()
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t ]+", " ", text)
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
    """
    Heuristic: pick the first likely proper name that isn't a stopword.
    Works for many kids/parents, not story-specific.
    """
    for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", transcript or ""):
        name = match.group(0)
        if name.lower() in _STOPWORDS:
            continue
        return name
    return None


def _infer_title(transcript: str) -> str:
    """
    Generic title inference:
    - If we see a strong noun like monster/dragon/pizza, use it.
    - Else use first discovered name.
    - Else pick a top keyword.
    """
    t = (transcript or "").strip()
    if not t:
        return "My Story"

    lowered = t.lower()
    for keyword, title in [
        ("pizza", "The Pizza Mystery"),
        ("dragon", "The Dragon Adventure"),
        ("monster", "The Monster Surprise"),
        ("castle", "The Castle Adventure"),
        ("robot", "The Robot Mix-Up"),
        ("unicorn", "The Unicorn Surprise"),
        ("dinosaur", "The Dinosaur Day"),
    ]:
        if keyword in lowered:
            return title

    name = _find_name(t)
    if name:
        return f"{name}'s Story"

    words = re.findall(r"[A-Za-z']+", lowered)
    words = [w for w in words if w not in _STOPWORDS and len(w) > 3]
    if words:
        top = Counter(words).most_common(1)[0][0]
        return f"The {top.capitalize()} Adventure"

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

    derived_pages = [StoryPage(text=page, illustration_prompt="") for page in _chunk_transcript(cleaned)]
    return StoryBook(
        title=_infer_title(cleaned),
        subtitle="A story told out loud",
        pages=derived_pages,
        narrator=narrator,
    )
