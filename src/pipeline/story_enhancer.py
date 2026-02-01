"""
Environment variables:
- STORY_ENHANCE_MODE: set to "openai" to use OpenAI for story enhancement.
- STORY_IMAGE_MODE: set to "openai" to generate images with OpenAI.
- OPENAI_API_KEY: required for OpenAI modes.
- STORY_TARGET_WORDS_TOTAL: total word count target (default: 750).
- STORY_STYLE: story tone/style (default: "whimsical, funny, heartwarming").
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import request

from src.pipeline.story_builder import StoryBook, StoryPage, _clean_transcript, _find_name, _infer_title

_DEFAULT_TARGET_WORDS = int(os.getenv("STORY_TARGET_WORDS_TOTAL", "750"))
_DEFAULT_STYLE = os.getenv("STORY_STYLE", "whimsical, funny, heartwarming")


def enhance_to_storybook(transcript: str, *, target_pages: int = 2) -> StoryBook:
    cleaned = _clean_transcript(transcript)
    narrator = _find_name(cleaned) if cleaned else None
    title = _infer_title(cleaned) if cleaned else "My Story"
    subtitle = "A story told out loud"

    if _use_openai_story_mode():
        openai_story = _openai_storybook(cleaned, title, narrator, target_pages)
        if openai_story:
            _maybe_generate_images(openai_story)
            return openai_story

    local_story = _local_storybook(cleaned, title, narrator, target_pages)
    _maybe_generate_images(local_story)
    return local_story


def _use_openai_story_mode() -> bool:
    return os.getenv("STORY_ENHANCE_MODE", "").lower() == "openai" and bool(os.getenv("OPENAI_API_KEY"))


def _use_openai_image_mode() -> bool:
    return os.getenv("STORY_IMAGE_MODE", "").lower() == "openai" and bool(os.getenv("OPENAI_API_KEY"))


def _get_openai_client():
    if not os.getenv("OPENAI_API_KEY"):
        return None
    if importlib.util.find_spec("openai") is None:
        return None
    openai_module = importlib.import_module("openai")
    if not hasattr(openai_module, "OpenAI"):
        return None
    return openai_module.OpenAI()


def _openai_storybook(
    cleaned: str,
    title: str,
    narrator: str | None,
    target_pages: int,
) -> StoryBook | None:
    client = _get_openai_client()
    if not client:
        return None

    words_total = _DEFAULT_TARGET_WORDS
    system_prompt = (
        "You are a children's story editor. "
        "Return ONLY strict JSON with a title, subtitle, narrator (optional), "
        "and exactly two pages. Each page must include text and illustration_prompt. "
        "Keep the story kid-safe, creative, and coherent."
    )
    user_prompt = (
        "TRANSCRIPT:\n"
        f"{cleaned or 'Empty transcript.'}\n\n"
        "OUTPUT JSON schema:\n"
        "{"
        '"title": string, '
        '"subtitle": string, '
        '"narrator": string|null, '
        '"pages": ['
        '{"text": string, "illustration_prompt": string}'
        "]"
        "}\n\n"
        f"Write exactly {target_pages} pages. "
        f"Target total length {words_total} words across both pages. "
        f"Style: {_DEFAULT_STYLE}."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
    except Exception:
        return None

    content = response.choices[0].message.content if response.choices else ""
    if not content:
        return None

    data = _parse_json(content)
    if not isinstance(data, dict):
        return None

    pages = _normalize_pages(data.get("pages"), cleaned, target_pages)
    if not pages:
        return None

    return StoryBook(
        title=(data.get("title") or title).strip() or title,
        subtitle=(data.get("subtitle") or "A story told out loud").strip() or "A story told out loud",
        pages=pages,
        narrator=(data.get("narrator") or narrator),
    )


def _parse_json(content: str) -> Any:
    trimmed = content.strip()
    if trimmed.startswith("```"):
        trimmed = re.sub(r"^```[a-zA-Z]*", "", trimmed).strip()
        trimmed = re.sub(r"```$", "", trimmed).strip()
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        return None


def _normalize_pages(raw_pages: Any, cleaned: str, target_pages: int) -> list[StoryPage]:
    pages: list[StoryPage] = []
    if isinstance(raw_pages, list):
        for page in raw_pages:
            if not isinstance(page, dict):
                continue
            text = (page.get("text") or "").strip()
            prompt = (page.get("illustration_prompt") or "").strip()
            if text:
                pages.append(StoryPage(text=text, illustration_prompt=prompt or "Whimsical storybook illustration."))

    if len(pages) >= target_pages:
        return pages[:target_pages]

    if len(pages) == 1:
        split_pages = _split_story_text(pages[0].text, target_pages)
        if split_pages:
            first_prompt = pages[0].illustration_prompt
            pages = [
                StoryPage(text=split_pages[0], illustration_prompt=first_prompt),
                StoryPage(text=split_pages[1], illustration_prompt="A joyful resolution scene."),
            ]

    while len(pages) < target_pages:
        filler_text = cleaned or "A warm, gentle moment fills the page."
        pages.append(StoryPage(text=filler_text, illustration_prompt="A friendly scene in a cozy setting."))

    return pages[:target_pages]


def _split_story_text(text: str, target_pages: int) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) < target_pages:
        midpoint = max(1, len(text) // 2)
        return [text[:midpoint].strip(), text[midpoint:].strip()]

    midpoint = len(sentences) // 2
    return [" ".join(sentences[:midpoint]), " ".join(sentences[midpoint:])]


def _local_storybook(
    cleaned: str,
    title: str,
    narrator: str | None,
    target_pages: int,
) -> StoryBook:
    topic = _infer_topic(cleaned)
    name = narrator or "A young storyteller"
    total_words = _DEFAULT_TARGET_WORDS
    page1_target = int(total_words * 0.55)
    page2_target = total_words - page1_target

    page1 = (
        f"On a bright afternoon, {name} felt a tiny spark of curiosity about {topic}. "
        f"The day started with a giggle and a whisper of adventure, "
        f"and soon a playful idea bubbled up: {cleaned or 'a surprising little secret'}. "
        "At first it seemed like a quick, harmless choice, but the world around them began to feel a little "
        "different—like the air was holding its breath. "
        "Friends noticed, a gentle worry flickered, and a silly misunderstanding grew into a big, bouncy problem. "
        "Even the smallest decision can turn into a story when imagination runs wild."
    )
    page1 = _ensure_word_count(page1, page1_target)

    page2 = (
        "With a deep breath and a brave heart, the truth tumbled out. "
        "Everyone paused, then smiled, because honesty made the room feel lighter. "
        f"Together they fixed the mix-up, turning {topic} into a shared joke and a brand-new game. "
        "They learned that apologies are like warm blankets: cozy, kind, and strong. "
        "By the end, laughter danced through the room, and the adventure settled into a happy ending "
        "with a gentle promise to choose kindness next time."
    )
    page2 = _ensure_word_count(page2, page2_target)

    pages = [
        StoryPage(
            text=page1,
            illustration_prompt=(
                f"A whimsical storybook scene of {name} discovering {topic}, "
                "with bright colors, cozy surroundings, and a playful hint of mischief."
            ),
        ),
        StoryPage(
            text=page2,
            illustration_prompt=(
                f"A warm, joyful illustration of friends making amends around {topic}, "
                "smiling and laughing in a cozy setting with gentle, friendly colors."
            ),
        ),
    ]

    if target_pages != 2:
        pages = pages[:target_pages]

    return StoryBook(
        title=title,
        subtitle="A story told out loud",
        pages=pages,
        narrator=narrator,
    )


def _infer_topic(cleaned: str) -> str:
    if not cleaned:
        return "a small mystery"
    lowered = cleaned.lower()
    for keyword in ("pizza", "dragon", "monster", "castle", "forest", "robot"):
        if keyword in lowered:
            return f"the {keyword}"
    words = re.findall(r"[A-Za-z']+", cleaned)
    stopwords = {"the", "and", "then", "with", "from", "that", "this", "when", "they", "she", "he"}
    for word in words:
        if word.lower() in stopwords:
            continue
        if len(word) > 3:
            return f"the {word.lower()}"
    return "a small surprise"


def _ensure_word_count(text: str, target_words: int) -> str:
    sentences = [
        "The sunlight seemed to smile along.",
        "Tiny footsteps pattered with excitement.",
        "A curious breeze whispered secrets.",
        "Every friend had a helpful idea.",
        "Kindness made the moment shine.",
        "The room felt warmer with each laugh.",
        "Imagination added sparkle to the day.",
        "A gentle giggle bubbled up again.",
    ]
    words = text.split()
    index = 0
    while len(words) < target_words and index < len(sentences):
        text = f"{text} {sentences[index]}"
        words = text.split()
        index += 1
    while len(words) < target_words:
        text = f"{text} Kind hearts made the adventure better."
        words = text.split()
    return text


def _maybe_generate_images(story: StoryBook) -> None:
    if not _use_openai_image_mode():
        return
    client = _get_openai_client()
    if not client:
        return

    out_dir = Path("out/books/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for index, page in enumerate(story.pages, start=1):
        prompt = page.illustration_prompt or "A cozy children's storybook illustration."
        try:
            response = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size="1024x1024",
            )
        except Exception:
            continue

        data = response.data[0] if response.data else None
        if not data:
            continue

        image_path = out_dir / f"story_{timestamp}_p{index}.png"
        if getattr(data, "b64_json", None):
            image_bytes = base64.b64decode(data.b64_json)
            image_path.write_bytes(image_bytes)
            page.illustration_path = str(image_path)
            continue

        if getattr(data, "url", None):
            try:
                request.urlretrieve(data.url, image_path)
            except Exception:
                continue
            page.illustration_path = str(image_path)
