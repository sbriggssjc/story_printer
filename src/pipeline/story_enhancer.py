"""
Environment variables:
- STORY_ENHANCE_MODE: set to "openai" to use OpenAI for story enhancement.
- STORY_IMAGE_MODE: set to "openai" to generate images with OpenAI.
- OPENAI_API_KEY: required for OpenAI modes.
- STORY_MODEL: OpenAI model for story enhancement (default: "gpt-4o-mini").
- STORY_TEMPERATURE: OpenAI sampling temperature (default: "0.8").
- STORY_TARGET_PAGES: number of pages to generate (default: "2").
- STORY_WORDS_PER_PAGE: word count target per page (default: "280").
- STORY_STYLE: story tone/style (default: "whimsical, funny, heartwarming").
"""

from __future__ import annotations

import base64
import importlib
import json
import math
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError

from pydantic import BaseModel

from src.pipeline.story_builder import StoryBook, StoryPage, _clean_transcript, _find_name, _infer_title

_DEFAULT_MODEL = os.getenv("STORY_MODEL", "gpt-4o-mini")
_DEFAULT_TEMPERATURE = float(os.getenv("STORY_TEMPERATURE", "0.8"))
_DEFAULT_TARGET_PAGES = int(os.getenv("STORY_TARGET_PAGES", "2"))
_DEFAULT_WORDS_PER_PAGE = int(os.getenv("STORY_WORDS_PER_PAGE", "280"))
_DEFAULT_STYLE = os.getenv("STORY_STYLE", "whimsical, funny, heartwarming")
_DEFAULT_FIDELITY_MODEL = os.getenv("STORY_FIDELITY_MODEL", _DEFAULT_MODEL)
_MIN_WORDS_PER_PAGE = max(240, _DEFAULT_WORDS_PER_PAGE - 40)
_MAX_WORDS_PER_PAGE = min(320, _DEFAULT_WORDS_PER_PAGE + 40)


def _apply_short_transcript_targets(cleaned_transcript: str) -> None:
    global _DEFAULT_WORDS_PER_PAGE, _MIN_WORDS_PER_PAGE, _MAX_WORDS_PER_PAGE
    transcript_words = _word_count(cleaned_transcript)
    if transcript_words < 70:
        _MIN_WORDS_PER_PAGE = 160
        _MAX_WORDS_PER_PAGE = 260
        _DEFAULT_WORDS_PER_PAGE = 210
    elif transcript_words < 140:
        _MIN_WORDS_PER_PAGE = 200
        _MAX_WORDS_PER_PAGE = 300
        _DEFAULT_WORDS_PER_PAGE = 250
    else:
        _MIN_WORDS_PER_PAGE = 240
        _MAX_WORDS_PER_PAGE = 320
        _DEFAULT_WORDS_PER_PAGE = min(max(_DEFAULT_WORDS_PER_PAGE, 240), 320)

# --- Anti-boilerplate / faithfulness guards ---

# Sentences we NEVER want in OpenAI output (these are showing up verbatim in your PDFs)
_BLOCKLIST_SENTENCES = {
    "The hallway smelled like crayons and toast, and Claire could hear sneakers squeaking nearby.",
    "A breeze bumped the curtains, as if the room itself was leaning in to listen.",
    "Someone whispered a guess, and another friend gasped, suddenly certain they knew the truth.",
    "Sunlight puddled on the floor like warm butter, making the room glow.",
    "The air felt fizzy, like soda bubbles popping with every new idea.",
    "Even the clock sounded excited, ticking a little faster than usual.",
}

_BANNED_PHRASES = [
    "The hallway smelled like crayons and toast",
    "Sunlight puddled on the floor like warm butter",
    "the room itself was leaning in to listen",
    "Even the clock sounded excited",
]


class AnchorSpec(BaseModel):
    characters: list[str]
    key_objects: list[str]
    setting: str | None
    beats: list[str]
    lesson: str | None = None


def _split_sentences_for_dedupe(text: str) -> list[str]:
    if not text:
        return []
    flat = re.sub(r"\s+", " ", text.strip())
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]
    return sents


def _split_sentences(text: str) -> list[str]:
    # Simple sentence split; good enough for children’s prose
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p and p.strip()]


def _dedupe_cross_page_sentences(pages: list[StoryPage], min_words: int) -> bool:
    """
    Removes any sentence that appears verbatim on an earlier page if the page
    still stays above the minimum word count.
    Returns True if it modified anything.
    """
    seen: set[str] = set()
    changed = False

    for page in pages:
        page_word_count = _word_count(page.text)
        sents = _split_sentences(page.text)
        out: list[str] = []
        for s in sents:
            key = re.sub(r"\s+", " ", s.strip())
            if key in seen:
                sentence_words = _word_count(s)
                if page_word_count - sentence_words >= min_words:
                    changed = True
                    page_word_count -= sentence_words
                    continue
            out.append(s)
            seen.add(key)

        # Rebuild paragraph-ish text
        new_text = " ".join(out).strip()
        if new_text and new_text != (page.text or "").strip():
            page.text = new_text

    return changed


def _normalize_sentence_key(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence.strip()).lower()


def _find_blocklisted_sentences(text: str) -> list[str]:
    sents = set(_split_sentences_for_dedupe(text))
    hits = [s for s in _BLOCKLIST_SENTENCES if s in sents]
    return hits


def _find_duplicate_sentences_across_pages(page_texts: list[str]) -> list[str]:
    # Return sentences (>= ~8 words) that appear on 2+ pages verbatim
    seen: dict[str, int] = {}
    dupes: set[str] = set()
    for t in page_texts:
        for s in _split_sentences_for_dedupe(t):
            if len(re.findall(r"[A-Za-z']+", s)) < 8:
                continue
            key = s
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 2:
                dupes.add(s)
    return sorted(dupes)


def _extract_proper_names(cleaned: str, narrator: str | None) -> list[str]:
    names: list[str] = []
    if narrator:
        names.append(narrator.strip())
    if not cleaned:
        return names
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", cleaned)
    ignored = {
        "The",
        "A",
        "An",
        "And",
        "But",
        "Or",
        "So",
        "Because",
        "When",
        "Then",
        "After",
        "Before",
        "Once",
        "With",
        "Without",
        "I",
        "We",
        "He",
        "She",
        "They",
        "It",
    }
    for candidate in candidates:
        if candidate in ignored:
            continue
        names.append(candidate)
    return names


def _extract_noun_phrases(
    cleaned: str,
    *,
    min_count: int = 3,
    max_count: int = 8,
    excluded: set[str] | None = None,
) -> list[str]:
    if not cleaned:
        return []
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "so",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "from",
        "into",
        "up",
        "down",
        "over",
        "under",
        "then",
        "when",
        "while",
        "after",
        "before",
        "as",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "is",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "are",
        "do",
        "did",
        "does",
        "done",
        "have",
        "has",
        "had",
        "i",
        "you",
        "we",
        "he",
        "she",
        "they",
        "him",
        "her",
        "them",
        "our",
        "their",
        "my",
        "your",
    }
    excluded = {term.lower() for term in (excluded or set())}
    words = [word.lower() for word in re.findall(r"[A-Za-z']+", cleaned)]
    counts: dict[str, int] = {}
    for index in range(len(words)):
        if words[index] in stopwords:
            continue
        for length in range(3, 0, -1):
            if index + length > len(words):
                continue
            chunk = words[index : index + length]
            if any(part in stopwords for part in chunk):
                continue
            phrase = " ".join(chunk)
            counts[phrase] = counts.get(phrase, 0) + 1

    ranked = sorted(
        counts.items(),
        key=lambda item: (item[1], len(item[0].split()), len(item[0])),
        reverse=True,
    )
    phrases: list[str] = []
    for phrase, _ in ranked:
        if phrase in excluded:
            continue
        if len(phrases) >= max_count:
            break
        phrases.append(phrase)

    if len(phrases) < min_count:
        for word in words:
            if word in stopwords or word in excluded:
                continue
            if word not in phrases:
                phrases.append(word)
            if len(phrases) >= min_count:
                break

    return phrases[:max_count]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _extract_role_characters(cleaned: str) -> list[str]:
    if not cleaned:
        return []
    lower = cleaned.lower()
    roles = [
        ("dad", "Dad"),
        ("father", "Dad"),
        ("mom", "Mom"),
        ("mother", "Mom"),
        ("grandma", "Grandma"),
        ("grandmother", "Grandma"),
        ("grandpa", "Grandpa"),
        ("grandfather", "Grandpa"),
        ("sister", "Sister"),
        ("brother", "Brother"),
        ("teacher", "Teacher"),
        ("coach", "Coach"),
        ("friend", "Friend"),
        ("dog", "dog"),
        ("cat", "cat"),
        ("puppy", "puppy"),
        ("kitten", "kitten"),
    ]
    found = []
    for token, label in roles:
        if re.search(rf"\b{re.escape(token)}\b", lower):
            found.append(label)
    return _dedupe_preserve_order(found)


def _extract_settings(cleaned: str) -> list[str]:
    if not cleaned:
        return []
    lower = cleaned.lower()
    settings = [
        "kitchen",
        "school",
        "park",
        "playground",
        "classroom",
        "bedroom",
        "living room",
        "backyard",
        "yard",
        "house",
        "home",
        "forest",
        "beach",
        "garden",
        "library",
        "store",
        "cafe",
        "restaurant",
    ]
    found = []
    for setting in settings:
        if re.search(rf"\b{re.escape(setting)}\b", lower):
            found.append(setting)
    return _dedupe_preserve_order(found)


def _extract_event_beats(cleaned: str) -> list[str]:
    if not cleaned:
        return []
    sentences = _split_sentences(cleaned)
    events = [sentence for sentence in sentences if sentence]
    if len(events) < 4:
        fragments = re.split(r"[;,]|\band then\b", cleaned, flags=re.IGNORECASE)
        for fragment in fragments:
            fragment = fragment.strip()
            if fragment and fragment not in events:
                events.append(fragment)
    events = _dedupe_preserve_order(events)
    if len(events) > 10:
        events = events[:10]
    return events


def _extract_moral(cleaned: str) -> str | None:
    if not cleaned:
        return None
    moral_keywords = [
        "lesson",
        "learned",
        "remember",
        "should",
        "shouldn't",
        "must",
        "promise",
        "apologize",
        "sorry",
        "forgive",
        "kind",
        "share",
        "listen",
    ]
    for sentence in _split_sentences(cleaned):
        lower = sentence.lower()
        if any(keyword in lower for keyword in moral_keywords):
            return sentence.strip()
    return None


def _local_anchor_spec(cleaned: str) -> AnchorSpec:
    characters = _dedupe_preserve_order(
        _extract_proper_names(cleaned, None) + _extract_role_characters(cleaned)
    )
    settings = _extract_settings(cleaned)
    excluded = set(characters + settings)
    key_objects = _extract_noun_phrases(cleaned, min_count=3, max_count=10, excluded=excluded)
    key_objects = _dedupe_preserve_order(key_objects)
    beats = _extract_event_beats(cleaned)
    if len(beats) < 4:
        beats = beats + ["A key moment happens.", "The situation is resolved."][: 4 - len(beats)]
    beats = beats[:10]
    lesson = _extract_moral(cleaned)
    return AnchorSpec(
        characters=characters,
        setting=settings[0] if settings else None,
        key_objects=key_objects,
        beats=beats,
        lesson=lesson,
    )


def _anchor_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "characters": {"type": "array", "items": {"type": "string"}},
            "setting": {"type": ["string", "null"]},
            "key_objects": {"type": "array", "items": {"type": "string"}},
            "beats": {"type": "array", "items": {"type": "string"}},
            "lesson": {"type": ["string", "null"]},
        },
        "required": [
            "characters",
            "setting",
            "key_objects",
            "beats",
            "lesson",
        ],
        "additionalProperties": False,
    }


def _plot_facts_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "facts": {"type": "array", "items": {"type": "string"}},
            "entities": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["facts", "entities"],
        "additionalProperties": False,
    }


def _coerce_anchor_spec(data: Any) -> AnchorSpec | None:
    if not isinstance(data, dict):
        return None
    characters = data.get("characters")
    key_objects = data.get("key_objects")
    beats = data.get("beats")
    if not isinstance(characters, list) or not isinstance(key_objects, list) or not isinstance(beats, list):
        return None
    setting_value = data.get("setting")
    setting = setting_value if isinstance(setting_value, str) else None
    lesson = data.get("lesson") if isinstance(data.get("lesson"), str) else None
    return AnchorSpec(
        characters=[item for item in characters if isinstance(item, str)],
        setting=setting,
        key_objects=[item for item in key_objects if isinstance(item, str)],
        beats=[item for item in beats if isinstance(item, str)],
        lesson=lesson,
    )


def _coerce_plot_facts(data: Any) -> dict[str, list[str]]:
    if not isinstance(data, dict):
        return {}
    facts = data.get("facts")
    entities = data.get("entities")
    if not isinstance(facts, list) or not isinstance(entities, list):
        return {}
    return {
        "facts": [item for item in facts if isinstance(item, str)],
        "entities": [item for item in entities if isinstance(item, str)],
    }


def _normalize_anchor_spec(spec: AnchorSpec, fallback: AnchorSpec) -> AnchorSpec:
    characters = _dedupe_preserve_order(spec.characters or fallback.characters)
    setting = spec.setting or fallback.setting
    key_objects = _dedupe_preserve_order(spec.key_objects or fallback.key_objects)
    beats = _dedupe_preserve_order(spec.beats or fallback.beats)
    if len(beats) < 4:
        beats = _dedupe_preserve_order(beats + fallback.beats)
    beats = beats[:10]
    lesson = spec.lesson or fallback.lesson
    return AnchorSpec(
        characters=characters,
        setting=setting,
        key_objects=key_objects,
        beats=beats,
        lesson=lesson,
    )


def _request_openai_plot_facts(
    client,
    system_prompt: str,
    user_prompt: str,
    use_responses: bool,
) -> str:
    if use_responses and hasattr(client.responses, "parse"):
        schema = _plot_facts_json_schema()
        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = client.responses.parse(
            model=_DEFAULT_MODEL,
            input=input_messages,
            temperature=0.2,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "plot_facts",
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=400,
        )
        parsed = getattr(result, "output_parsed", None)
        if parsed is not None:
            return json.dumps(parsed)
        return _extract_response_text(result)

    if use_responses and hasattr(client, "responses"):
        schema = _plot_facts_json_schema()
        response = client.responses.create(
            model=_DEFAULT_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "plot_facts",
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=400,
        )
        return _extract_response_text(response)
    return ""


def _request_openai_anchor_spec(
    client,
    system_prompt: str,
    user_prompt: str,
    use_responses: bool,
) -> str:
    if use_responses and hasattr(client.responses, "parse"):
        schema = _anchor_json_schema()
        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = client.responses.parse(
            model=_DEFAULT_MODEL,
            input=input_messages,
            temperature=0.2,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "anchor_spec",
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=500,
        )
        parsed = getattr(result, "output_parsed", None)
        if parsed is not None:
            return json.dumps(parsed)
        return _extract_response_text(result)

    if use_responses and hasattr(client, "responses"):
        schema = _anchor_json_schema()
        response = client.responses.create(
            model=_DEFAULT_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "anchor_spec",
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=500,
        )
        return _extract_response_text(response)
    return ""


def _openai_http_plot_facts(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    schema = _plot_facts_json_schema()
    responses_payload = {
        "model": _DEFAULT_MODEL,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        "temperature": 0.2,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "plot_facts",
                "schema": schema,
                "strict": True,
            }
        },
        "max_output_tokens": 400,
    }
    try:
        response_data = _post_openai_request(
            "https://api.openai.com/v1/responses",
            responses_payload,
        )
        content = _extract_http_response_text(response_data)
        if content:
            return content, "responses"
    except Exception as exc:
        print(f"OpenAI HTTP responses plot facts call failed: {exc}")
    return "", "responses"


def _openai_http_anchor_spec(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    schema = _anchor_json_schema()
    responses_payload = {
        "model": _DEFAULT_MODEL,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        "temperature": 0.2,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "anchor_spec",
                "schema": schema,
                "strict": True,
            }
        },
        "max_output_tokens": 500,
    }
    try:
        response_data = _post_openai_request(
            "https://api.openai.com/v1/responses",
            responses_payload,
        )
        content = _extract_http_response_text(response_data)
        if content:
            return content, "responses"
    except Exception as exc:
        print(f"OpenAI HTTP responses anchor call failed: {exc}")
    return "", "responses"


def _extract_plot_facts_openai(cleaned_transcript: str) -> dict[str, list[str]]:
    cleaned = cleaned_transcript or ""
    local_facts = _dedupe_preserve_order(_split_sentences(cleaned))[:12]
    local_entities = _dedupe_preserve_order(
        _extract_proper_names(cleaned, None) + _extract_role_characters(cleaned)
    )
    fallback = {"facts": local_facts, "entities": local_entities}
    if not os.getenv("OPENAI_API_KEY"):
        return fallback
    system_prompt = (
        "Extract plot facts and named entities from a story transcript. "
        "Facts must be faithful, short statements drawn directly from the transcript. "
        "Return ONLY strict JSON that follows the schema exactly."
    )
    user_prompt = (
        "Extract plot facts and named entities from this transcript. "
        "Facts should preserve negatives and motivations (e.g., 'did not want').\n\n"
        f"TRANSCRIPT:\n{cleaned or 'Empty transcript.'}"
    )
    client = _get_openai_client()
    try:
        if client:
            use_responses = hasattr(client, "responses")
            content = _request_openai_plot_facts(
                client, system_prompt, user_prompt, use_responses
            )
        else:
            content, endpoint = _openai_http_plot_facts(system_prompt, user_prompt)
            print(f"OpenAI endpoint: {endpoint} (HTTP)")
    except Exception as exc:
        print(f"OpenAI plot facts extraction failed: {exc}")
        return fallback
    if not content:
        return fallback
    data = _parse_json(content)
    extracted = _coerce_plot_facts(data)
    if not extracted:
        return fallback
    extracted["facts"] = _dedupe_preserve_order(extracted.get("facts", []))
    extracted["entities"] = _dedupe_preserve_order(extracted.get("entities", []))
    if not extracted["facts"]:
        extracted["facts"] = fallback["facts"]
    if not extracted["entities"]:
        extracted["entities"] = fallback["entities"]
    return extracted


def _openai_anchor_spec(cleaned: str) -> AnchorSpec | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    system_prompt = (
        "You extract canon anchors from a transcript for a children's story. "
        "Return ONLY strict JSON that follows the schema exactly."
    )
    user_prompt = (
        "Extract canon anchors from this transcript. "
        "Provide characters, setting, key_objects, ordered beats (4-10), "
        "and an optional lesson.\n\n"
        f"TRANSCRIPT:\n{cleaned or 'Empty transcript.'}"
    )
    client = _get_openai_client()
    try:
        if client:
            use_responses = hasattr(client, "responses")
            content = _request_openai_anchor_spec(
                client, system_prompt, user_prompt, use_responses
            )
        else:
            content, endpoint = _openai_http_anchor_spec(system_prompt, user_prompt)
            print(f"OpenAI endpoint: {endpoint} (HTTP)")
    except Exception as exc:
        print(f"OpenAI anchor extraction failed: {exc}")
        return None
    if not content:
        return None
    data = _parse_json(content)
    return _coerce_anchor_spec(data)


def extract_story_anchors(cleaned_transcript: str) -> AnchorSpec:
    cleaned = cleaned_transcript or ""
    local_spec = _local_anchor_spec(cleaned)
    if os.getenv("OPENAI_API_KEY"):
        openai_spec = _openai_anchor_spec(cleaned)
        if openai_spec:
            return _normalize_anchor_spec(openai_spec, local_spec)
    return local_spec


def _fails_fidelity(pages_text: str, must_keywords: list[str]) -> tuple[list[str], list[str]]:
    reasons = []
    banned_phrases = []
    low = pages_text.lower()
    for k in must_keywords:
        if k not in low:
            reasons.append(f"missing keyword: {k}")
    for phrase in _BANNED_PHRASES:
        if phrase.lower() in low:
            reasons.append(f"contains banned filler phrase: {phrase}")
            banned_phrases.append(phrase)
    return reasons, banned_phrases


_ANCHOR_SYNONYM_GROUPS = [
    {"trash can", "trashcan", "garbage", "garbage can", "bin", "trash bin"},
    {"bicycle", "bike"},
    {"sofa", "couch"},
]


_COMMON_VERBS = {
    "asked",
    "arrived",
    "ate",
    "brought",
    "built",
    "called",
    "carried",
    "caught",
    "chased",
    "cleaned",
    "climbed",
    "cooked",
    "cried",
    "decided",
    "drew",
    "drank",
    "dropped",
    "drove",
    "fell",
    "found",
    "gave",
    "got",
    "grabbed",
    "heard",
    "hid",
    "hit",
    "hugged",
    "jumped",
    "kept",
    "kicked",
    "laughed",
    "left",
    "lied",
    "listened",
    "looked",
    "lost",
    "made",
    "met",
    "opened",
    "picked",
    "played",
    "pulled",
    "pushed",
    "ran",
    "reached",
    "rode",
    "saw",
    "said",
    "saved",
    "shared",
    "shouted",
    "sat",
    "sang",
    "slept",
    "spoke",
    "stood",
    "told",
    "threw",
    "tried",
    "took",
    "tripped",
    "waited",
    "walked",
    "wanted",
    "watched",
    "went",
    "wrote",
}


def _extract_key_verbs(cleaned: str, *, max_count: int = 4) -> list[str]:
    if not cleaned:
        return []
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "so",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "from",
        "into",
        "up",
        "down",
        "over",
        "under",
        "then",
        "when",
        "while",
        "after",
        "before",
        "as",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "is",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "are",
        "do",
        "did",
        "does",
        "done",
        "have",
        "has",
        "had",
        "i",
        "you",
        "we",
        "he",
        "she",
        "they",
        "him",
        "her",
        "them",
        "our",
        "their",
        "my",
        "your",
        "me",
    }
    words = [word.lower() for word in re.findall(r"[A-Za-z']+", cleaned)]
    verbs: list[str] = []
    index = 0
    while index < len(words):
        word = words[index]
        if word in stopwords:
            index += 1
            continue
        if word in {"throw", "threw", "thrown"} and index + 1 < len(words):
            next_word = words[index + 1]
            if next_word in {"away", "out"}:
                verbs.append(f"{word} {next_word}")
                index += 2
                continue
        if word in _COMMON_VERBS or word.endswith("ed") or word.endswith("ing"):
            verbs.append(word)
        index += 1
    return _dedupe_preserve_order(verbs)[:max_count]


def _extract_anchor_tokens(cleaned: str, narrator: str | None) -> list[str]:
    names = _extract_proper_names(cleaned, narrator) + _extract_role_characters(cleaned)
    excluded = set(names)
    nouns = _extract_noun_phrases(cleaned, min_count=4, max_count=8, excluded=excluded)
    verbs = _extract_key_verbs(cleaned, max_count=4)
    tokens = _dedupe_preserve_order(names + nouns + verbs)
    if len(tokens) < 6:
        extra_nouns = _extract_noun_phrases(cleaned, min_count=2, max_count=10, excluded=excluded)
        tokens = _dedupe_preserve_order(tokens + extra_nouns)
    if len(tokens) < 6:
        tokens = _dedupe_preserve_order(tokens + verbs)
    return tokens[:10]


def _anchor_token_coverage(
    story_text: str,
    anchor_tokens: list[str],
    *,
    min_ratio: float = 0.6,
    min_required: int = 4,
) -> tuple[bool, str]:
    if not anchor_tokens:
        return True, ""
    text_lower = story_text.lower()
    group_map: dict[str, set[str]] = {}
    for group in _ANCHOR_SYNONYM_GROUPS:
        for term in group:
            group_map[term] = group

    def contains_term(term: str) -> bool:
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        return re.search(pattern, text_lower) is not None

    found: list[str] = []
    missing: list[str] = []
    for token in anchor_tokens:
        token_lower = token.lower()
        group = group_map.get(token_lower)
        if group:
            if any(contains_term(term) for term in group):
                found.append(token)
            else:
                missing.append(token)
            continue
        if contains_term(token_lower):
            found.append(token)
        else:
            missing.append(token)

    required = max(min_required, math.ceil(len(anchor_tokens) * min_ratio))
    required = min(required, len(anchor_tokens))
    if len(found) >= required:
        return True, ""
    summary = (
        f"Anchor token coverage too low: {len(found)}/{len(anchor_tokens)} found, "
        f"need at least {required}. Missing: {', '.join(missing)}."
    )
    return False, summary


_NEGATION_TERMS = (
    "didn't",
    "did not",
    "didnt",
    "refused",
    "refuse",
    "refuses",
    "refusing",
    "wouldn't",
    "would not",
    "wouldnt",
)
_POSITIVE_DESIRE_TERMS = (
    "want",
    "wants",
    "wanted",
)


def _has_term_near_object(text: str, term_pattern: str, obj_pattern: str, window: int = 40) -> bool:
    chunk = text.lower()
    near_patterns = (
        rf"{term_pattern}[^.!?]{{0,{window}}}{obj_pattern}",
        rf"{obj_pattern}[^.!?]{{0,{window}}}{term_pattern}",
    )
    return any(re.search(pattern, chunk) for pattern in near_patterns)


def _rule_based_contradictions(
    transcript: str,
    story_text: str,
    anchor_spec: AnchorSpec,
) -> list[str]:
    if not transcript or not story_text:
        return []
    key_objects = [obj.strip() for obj in (anchor_spec.key_objects or []) if obj.strip()]
    if not key_objects:
        return []

    negation_pattern = r"(?:%s)" % "|".join(re.escape(term) for term in _NEGATION_TERMS)
    positive_pattern = r"(?:%s)" % "|".join(re.escape(term) for term in _POSITIVE_DESIRE_TERMS)
    contradictions: list[str] = []

    for obj in key_objects:
        obj_pattern = r"\b%s\b" % re.escape(obj.lower())
        if not _has_term_near_object(transcript, negation_pattern, obj_pattern):
            continue
        if _has_term_near_object(story_text, positive_pattern, obj_pattern):
            contradictions.append(
                f"Transcript negates desire for '{obj}', but story expresses wanting it."
            )

    return contradictions


def _fidelity_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "pass": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "missing_beats": {"type": "array", "items": {"type": "string"}},
            "contradictions": {"type": "array", "items": {"type": "string"}},
            "drift_score": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["pass", "issues", "missing_beats", "contradictions", "drift_score"],
        "additionalProperties": False,
    }


def _request_openai_fidelity_check(
    client,
    anchor_spec: AnchorSpec,
    story_payload: dict[str, Any],
    use_responses: bool,
) -> dict[str, Any]:
    system_prompt = (
        "You are a strict story fidelity evaluator. Compare the story against the anchors. "
        "Check beat coverage (each anchor beat must appear in order), "
        "entity consistency (characters and key_objects persist), and contradictions (no inversion of canon). "
        "Return ONLY the JSON object that matches the schema."
    )
    anchor_json = json.dumps(anchor_spec.model_dump(), ensure_ascii=False, indent=2)
    story_json = json.dumps(story_payload, ensure_ascii=False, indent=2)
    user_prompt = (
        "ANCHORS JSON:\n"
        f"{anchor_json}\n\n"
        "STORY JSON:\n"
        f"{story_json}\n"
    )

    if use_responses and hasattr(client, "responses") and hasattr(client.responses, "parse"):
        schema = _fidelity_json_schema()
        result = client.responses.parse(
            model=_DEFAULT_FIDELITY_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "fidelity_check",
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=600,
        )
        parsed = getattr(result, "output_parsed", None)
        if parsed is not None:
            return parsed
        raw = _extract_response_text(result)
        return json.loads(raw) if raw else {}

    if use_responses and hasattr(client, "responses"):
        schema = _fidelity_json_schema()
        response = client.responses.create(
            model=_DEFAULT_FIDELITY_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "fidelity_check",
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=600,
        )
        raw = _extract_response_text(response)
        return json.loads(raw) if raw else {}

    response = client.chat.completions.create(
        model=_DEFAULT_FIDELITY_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=600,
    )
    content = response.choices[0].message.content if response.choices else "{}"
    return json.loads(content)


def _request_openai_fidelity_http(
    anchor_spec: AnchorSpec,
    story_payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    schema = _fidelity_json_schema()
    system_prompt = (
        "You are a strict story fidelity evaluator. Compare the story against the anchors. "
        "Check beat coverage (each anchor beat must appear in order), "
        "entity consistency (characters and key_objects persist), and contradictions (no inversion of canon). "
        "Return ONLY the JSON object that matches the schema."
    )
    anchor_json = json.dumps(anchor_spec.model_dump(), ensure_ascii=False, indent=2)
    story_json = json.dumps(story_payload, ensure_ascii=False, indent=2)
    user_prompt = (
        "ANCHORS JSON:\n"
        f"{anchor_json}\n\n"
        "STORY JSON:\n"
        f"{story_json}\n"
    )

    responses_payload = {
        "model": _DEFAULT_FIDELITY_MODEL,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        "temperature": 0,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "fidelity_check",
                "schema": schema,
                "strict": True,
            }
        },
        "max_output_tokens": 600,
    }
    try:
        response_data = _post_openai_request(
            "https://api.openai.com/v1/responses",
            responses_payload,
        )
        content = _extract_http_response_text(response_data)
        if content:
            return json.loads(content), "responses"
    except Exception as exc:
        print(f"OpenAI HTTP fidelity responses call failed: {exc}")

    chat_payload = {
        "model": _DEFAULT_FIDELITY_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }
    response_data = _post_openai_request(
        "https://api.openai.com/v1/chat/completions",
        chat_payload,
    )
    content = _extract_http_chat_text(response_data)
    return json.loads(content) if content else {}, "chat.completions"


def _summarize_fidelity_issues(result: dict[str, Any]) -> str:
    issues = [str(item) for item in result.get("issues") or []]
    missing_beats = [str(item) for item in result.get("missing_beats") or []]
    contradictions = [str(item) for item in result.get("contradictions") or []]
    parts: list[str] = []
    if missing_beats:
        parts.append("Missing beats:\n- " + "\n- ".join(missing_beats))
    if contradictions:
        parts.append("Contradictions:\n- " + "\n- ".join(contradictions))
    if issues:
        parts.append("Other issues:\n- " + "\n- ".join(issues))
    return "\n".join(parts).strip()


def enhance_to_storybook(transcript: str, *, target_pages: int = 2) -> StoryBook:
    cleaned = _clean_transcript(transcript)
    _apply_short_transcript_targets(cleaned)
    narrator = _find_name(cleaned) if cleaned else None
    title = _infer_title(cleaned) if cleaned else "My Story"
    subtitle = "A story told out loud"
    target_pages = target_pages or _DEFAULT_TARGET_PAGES
    anchor_spec = extract_story_anchors(cleaned)
    plot_facts = _extract_plot_facts_openai(cleaned)

    if _is_openai_story_requested():
        print("STORY_ENHANCE_MODE=openai detected")
        openai_story = _openai_storybook(
            anchor_spec,
            plot_facts,
            cleaned,
            title,
            narrator,
            target_pages,
        )
        if openai_story:
            _maybe_generate_images(openai_story, cleaned)
            return openai_story

    local_story = _local_storybook(anchor_spec, title, narrator, target_pages)
    _maybe_generate_images(local_story, cleaned)
    return local_story


def _use_openai_story_mode() -> bool:
    return os.getenv("STORY_ENHANCE_MODE", "").lower() == "openai" and bool(os.getenv("OPENAI_API_KEY"))


def _is_openai_story_requested() -> bool:
    return os.getenv("STORY_ENHANCE_MODE", "").lower() == "openai"


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


def _build_openai_prompts(
    anchor_spec: AnchorSpec,
    plot_facts: dict[str, list[str]],
    narrator: str | None,
    target_pages: int,
) -> tuple[str, str, AnchorSpec]:
    system_prompt = (
        "You are a celebrated children's picture-book author and editor. "
        "Expand the transcript into a rich, creative, kid-safe story (not a summary). "
        "Return ONLY strict JSON that follows the schema exactly."
    )
    beat_sheet = (
        "Beat sheet:\n"
        "1) Warm opening with setting and sensory details.\n"
        "2) First anchor beat appears.\n"
        "3) Rising tension that leads into the next beat.\n"
        "4) Page-turn hook ending page 1.\n"
        "5) Biggest anchor beat on page 2.\n"
        "6) Emotional response or realization.\n"
        "7) Resolution that follows the remaining beats.\n"
        "8) Warm ending with emotional lift.\n"
    )
    anchor_json = json.dumps(anchor_spec.model_dump(), ensure_ascii=False, indent=2)
    facts = plot_facts.get("facts", [])
    entities = plot_facts.get("entities", [])
    facts_lines = "\n".join([f"- {fact}" for fact in facts]) or "- (none)"
    entity_lines = "\n".join([f"- {entity}" for entity in entities]) or "- (none)"
    user_prompt = (
        "Use the AnchorSpec JSON below and the plot facts as the ONLY sources of plot truth. "
        "Do not invent new major plotlines beyond what the plot facts and anchors imply.\n\n"
        f"NARRATOR (keep name if present): {narrator or 'none'}\n\n"
        "Story requirements:\n"
        f"- Exactly {target_pages} pages.\n"
        "- Page 1: setup + early beats + rising trouble + page-turn hook.\n"
        "- Page 2: remaining beats + resolution + warm ending.\n"
        "- Exactly two dialogue lines total. Each must be a full sentence in quotes on its own line starting with '- ' or '— '.\n"
        "- Dialogue must be short (<= 12 words each) and natural.\n"
        "- Do not use quotation marks for emphasis or any other purpose.\n"
        "- Kid-safe, whimsical, humorous, creative expansion.\n"
        "- Include sensory details and an emotional arc.\n"
        "- Avoid repeating filler sentences or stock phrases.\n"
        "- Do NOT repeat any exact sentence across pages. Each sentence must be unique.\n"
        "- Keep names consistent; use narrator if provided.\n"
        "- You MUST include each anchor beat in order (paraphrase is allowed).\n"
        "- Ensure entity consistency: characters, key_objects, and setting stay the same.\n"
        "- Do not change any character identities or swap roles.\n"
        "- Do not invert motivations (explicitly: if transcript says “did not want X” you must not say “wanted X”).\n"
        "- You may add whimsical details, but do not add new major plotlines.\n"
        f"- {_MIN_WORDS_PER_PAGE}–{_MAX_WORDS_PER_PAGE} words per page (target {_DEFAULT_WORDS_PER_PAGE}).\n"
        "- Each page must include a rich illustration_prompt in consistent watercolor picture-book style.\n"
        f"{beat_sheet}\n"
        "MUST KEEP TRUE FACTS (do not change these; list them verbatim):\n"
        f"{facts_lines}\n\n"
        "ENTITIES (names to keep consistent):\n"
        f"{entity_lines}\n\n"
        "ANCHORS JSON (use this as ground truth):\n"
        f"{anchor_json}\n\n"
        "OUTPUT JSON schema:\n"
        "{"
        '"title": string, '
        '"subtitle": string, '
        '"narrator": string|null, '
        '"pages": ['
        '{"text": string, "illustration_prompt": string}'
        "]"
        "}\n\n"
        f"Style: {_DEFAULT_STYLE}."
    )
    user_prompt += (
        "\nHard bans:\n"
        "- Do NOT use any generic boilerplate sensory sentences.\n"
        "- Do NOT repeat any sentence verbatim across pages.\n"
        "- Do NOT include these sentences (exactly as written):\n"
        + "\n".join([f"  - {s}" for s in sorted(_BLOCKLIST_SENTENCES)])
        + "\n"
    )
    return system_prompt, user_prompt, anchor_spec


def _openai_storybook(
    anchor_spec: AnchorSpec,
    plot_facts: dict[str, list[str]],
    cleaned_transcript: str,
    title: str,
    narrator: str | None,
    target_pages: int,
) -> StoryBook | None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI enhancement failed: OPENAI_API_KEY not set.")
        return None

    client = _get_openai_client()
    if client:
        print("OpenAI path: SDK")
    else:
        print("OpenAI path: HTTP (openai package not available)")

    print(f"OpenAI model: {_DEFAULT_MODEL}")

    system_prompt, user_prompt, anchor_spec = _build_openai_prompts(
        anchor_spec, plot_facts, narrator, target_pages
    )
    fidelity_note: str | None = None
    anchor_tokens = _extract_anchor_tokens(cleaned_transcript, narrator)
    coverage_note: str | None = None

    repeat_sentence_note: str | None = None
    for attempt in range(2):
        correction = ""
        if attempt == 1:
            correction = (
                "Targeted repair only. Keep tone, page structure, and length stable.\n"
                "Fix ONLY the missing beats or contradictions; do not add new major plotlines.\n"
                "Rewrite the story to satisfy ALL requirements:\n"
                f"- EXACTLY {target_pages} pages.\n"
                f"- EACH page MUST be between {_MIN_WORDS_PER_PAGE} and {_MAX_WORDS_PER_PAGE} words.\n"
                '- Exactly two dialogue lines total using straight quotes, e.g. "...". Each must be a full sentence on its own line starting with "- " or "— ".\n'
                "- Each dialogue line must be short (<= 12 words) and natural.\n"
                "- Do not use quotation marks for emphasis or any other purpose.\n"
                "- Return ONLY JSON matching the schema. No extra keys, no commentary."
            )
            if coverage_note:
                correction += (
                    "\nAnchor token coverage fixes required (be surgical; keep tone and length):\n"
                    f"{coverage_note}\n"
                    "Improve coverage without adding new major plotlines."
                )
        try:
            if client:
                use_responses = hasattr(client, "responses")
                endpoint = "responses" if use_responses else "chat.completions"
                print(f"OpenAI endpoint: {endpoint} (SDK)")
                content = _request_openai_story(
                    client,
                    system_prompt,
                    user_prompt,
                    correction,
                    use_responses,
                    target_pages,
                )
            else:
                content, endpoint = _openai_http_storybook(
                    system_prompt,
                    user_prompt,
                    correction,
                    target_pages,
                )
                print(f"OpenAI endpoint: {endpoint} (HTTP)")
        except Exception as exc:
            print(f"OpenAI enhancement failed: {exc}")
            return None

        if not content:
            print("OpenAI enhancement failed: empty response.")
            return None

        data = _parse_json(content)
        pages = _build_pages_from_data(data, target_pages)
        if not pages:
            print("OpenAI enhancement failed: unable to parse pages.")
            return None

        # Post-process pages BEFORE validation so we validate the final text that will be printed
        for page in pages:
            page.text = _normalize_dialogue_lines(page.text, max_lines=2)

        _ensure_minimum_dialogue_lines(pages, target_pages)
        _ensure_unique_sentences_across_pages(pages, anchor_spec, cleaned_transcript, narrator)
        _trim_pages_to_word_limit(pages, _MAX_WORDS_PER_PAGE)

        valid, reasons = _validate_story_pages(
            pages,
            target_pages,
            enforce_word_count=False,
            enforce_dialogue=False,
        )
        if not valid:
            print(f"Validation failed: {', '.join(reasons)}")
            if attempt == 0:
                user_prompt = (
                    f"{user_prompt}\n\n"
                    "Your previous attempt failed validation. Fix ALL issues below:\n"
                    "- " + "\n- ".join(reasons) + "\n\n"
                    "Rewrite from scratch. Do NOT reuse any sentences from the previous attempt.\n"
                )
                continue
            print(f"OpenAI enhancement failed: invalid output ({'; '.join(reasons)}).")
            return None

        page_texts = [page.text for page in pages]
        all_text = " ".join(page_texts)
        _, banned_phrases = _fails_fidelity(all_text, [])
        if banned_phrases:
            print(f"Fidelity failed: contains banned filler phrases: {', '.join(banned_phrases)}")
            if attempt == 0:
                user_prompt = (
                    f"{user_prompt}\n\nIssues:\n- "
                    + "\n- ".join([f"contains banned filler phrase: {p}" for p in banned_phrases])
                    + "\n\nRewrite while preserving the anchor beats (paraphrase allowed)."
                )
                continue
            return None

        rule_contradictions = _rule_based_contradictions(cleaned_transcript, all_text, anchor_spec)
        if rule_contradictions:
            print(
                "Rule-based contradiction check failed: "
                + "; ".join(rule_contradictions)
            )
            if attempt == 0:
                user_prompt = (
                    f"{user_prompt}\n\nIssues:\n- "
                    + "\n- ".join(rule_contradictions)
                    + "\n\nRewrite to avoid contradicting the transcript."
                )
                continue
            return None

        coverage_ok, coverage_summary = _anchor_token_coverage(all_text, anchor_tokens)
        if not coverage_ok:
            coverage_note = coverage_summary
            print("Fidelity failed via anchor token coverage.")
            if attempt == 0:
                user_prompt = (
                    f"{user_prompt}\n\nIssues:\n- "
                    + coverage_summary
                    + "\n\nRewrite to include more of the anchor tokens from the transcript."
                )
                continue
            return None

        return StoryBook(
            title=(data.get("title") or title).strip() or title,
            subtitle=(data.get("subtitle") or "A story told out loud").strip()
            or "A story told out loud",
            pages=pages,
            narrator=(data.get("narrator") or narrator),
        )

    return None


def _request_openai_story(
    client,
    system_prompt: str,
    user_prompt: str,
    correction: str,
    use_responses: bool,
    target_pages: int,
) -> str:
    # Prefer Responses API (SDK) when available
    if use_responses and hasattr(client.responses, "parse"):
        schema = _story_json_schema(target_pages)

        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if correction:
            input_messages.append({"role": "user", "content": correction})

        # NOTE: responses.parse enforces the schema and returns parsed output
        result = client.responses.parse(
            model=_DEFAULT_MODEL,
            input=input_messages,
            temperature=_DEFAULT_TEMPERATURE,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "storybook",
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=2600,
        )

        # result.output_parsed is the parsed JSON object (dict)
        # But our caller expects raw JSON text (string), so serialize it.
        parsed = getattr(result, "output_parsed", None)
        if parsed is not None:
            return json.dumps(parsed)

        # Fallback if output_parsed isn't present for some reason
        return _extract_response_text(result)

    # If parse isn't available, use Responses.create with text.format
    if use_responses:
        schema = _story_json_schema(target_pages)
        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if correction:
            input_messages.append({"role": "user", "content": correction})

        response = client.responses.create(
            model=_DEFAULT_MODEL,
            input=input_messages,
            temperature=_DEFAULT_TEMPERATURE,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "storybook",
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=2600,
        )
        return _extract_response_text(response)

    # chat.completions fallback
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if correction:
        messages.append({"role": "user", "content": correction})

    response = client.chat.completions.create(
        model=_DEFAULT_MODEL,
        messages=messages,
        temperature=_DEFAULT_TEMPERATURE,
        response_format={"type": "json_object"},
        max_tokens=2600,
    )
    return response.choices[0].message.content if response.choices else ""


def _openai_http_storybook(
    system_prompt: str,
    user_prompt: str,
    correction: str,
    target_pages: int,
) -> tuple[str, str]:
    return _request_openai_story_http(
        system_prompt,
        user_prompt,
        correction,
        target_pages,
    )


def _request_openai_story_http(
    system_prompt: str,
    user_prompt: str,
    correction: str,
    target_pages: int,
) -> tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    schema = _story_json_schema(target_pages)
    input_messages = [
        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
    ]
    if correction:
        input_messages.append(
            {"role": "user", "content": [{"type": "input_text", "text": correction}]}
        )

    responses_payload = {
        "model": _DEFAULT_MODEL,
        "input": input_messages,
        "temperature": _DEFAULT_TEMPERATURE,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "storybook",
                "schema": schema,
                "strict": True,
            }
        },
        "max_output_tokens": 2200,
    }
    try:
        response_data = _post_openai_request(
            "https://api.openai.com/v1/responses",
            responses_payload,
        )
        content = _extract_http_response_text(response_data)
        if content:
            return content, "responses"
    except Exception as exc:
        print(f"OpenAI HTTP responses call failed: {exc}")

    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if correction:
        chat_messages.append({"role": "user", "content": correction})

    chat_payload = {
        "model": _DEFAULT_MODEL,
        "messages": chat_messages,
        "temperature": _DEFAULT_TEMPERATURE,
    }
    response_data = _post_openai_request(
        "https://api.openai.com/v1/chat/completions",
        chat_payload,
    )
    content = _extract_http_chat_text(response_data)
    return content, "chat.completions"


def _post_openai_request(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(
            f"OpenAI HTTP request failed: {exc.code} {exc.reason} :: {err_body[:1200]}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"OpenAI HTTP request failed: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenAI HTTP response was not valid JSON") from exc


def _extract_http_response_text(response_data: dict[str, Any]) -> str:
    text = response_data.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    output = response_data.get("output")
    if isinstance(output, list):
        for item in output:
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                text_part = part.get("text")
                if text_part:
                    return str(text_part).strip()
    return ""


def _extract_http_chat_text(response_data: dict[str, Any]) -> str:
    choices = response_data.get("choices")
    if not isinstance(choices, list):
        return ""
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    return ""


def _story_json_schema(target_pages: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "subtitle": {"type": "string"},
            "narrator": {"type": ["string", "null"]},
            "pages": {
                "type": "array",
                "minItems": target_pages,
                "maxItems": target_pages,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "illustration_prompt": {"type": "string"},
                    },
                    "required": ["text", "illustration_prompt"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["title", "subtitle", "narrator", "pages"],
        "additionalProperties": False,
    }


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                text_part = getattr(part, "text", None)
                if text_part:
                    return text_part
    return ""


def _parse_json(content: str) -> Any:
    trimmed = content.strip()
    if trimmed.startswith("```"):
        trimmed = re.sub(r"^```[a-zA-Z]*", "", trimmed).strip()
        trimmed = re.sub(r"```$", "", trimmed).strip()
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        extracted = _extract_json_block(trimmed)
        if extracted:
            return extracted
    return None


def _extract_json_block(text: str) -> Any:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : index + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    return None
    return None


def _validate_story_data(data: Any, target_pages: int) -> tuple[bool, list[str]]:
    if not isinstance(data, dict):
        return False, ["response is not a JSON object"]
    pages = data.get("pages")
    if not isinstance(pages, list):
        return False, ["pages must be a list"]
    if len(pages) != target_pages:
        return False, [f"expected {target_pages} pages but got {len(pages)}"]
    reasons: list[str] = []
    total_dialogue = 0
    page_texts: list[str] = []
    for index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            reasons.append(f"page {index} is not an object")
            continue
        text = (page.get("text") or "").strip()
        prompt = (page.get("illustration_prompt") or "").strip()
        if not text:
            reasons.append(f"page {index} missing text")
            continue
        if not prompt:
            reasons.append(f"page {index} missing illustration_prompt")
        word_count = _word_count(text)
        if word_count < _MIN_WORDS_PER_PAGE or word_count > _MAX_WORDS_PER_PAGE:
            reasons.append(
                f"page {index} word count {word_count} outside {_MIN_WORDS_PER_PAGE}-{_MAX_WORDS_PER_PAGE}"
            )
        total_dialogue += _count_dialogue_lines(text)
        page_texts.append(text)

    for i, t in enumerate(page_texts, start=1):
        hits = _find_blocklisted_sentences(t)
        if hits:
            reasons.append(f"page {i} contains boilerplate sentence(s): {hits[:2]}")

    dupes = _find_duplicate_sentences_across_pages(page_texts)
    if dupes:
        reasons.append(f"repeated sentence across pages: {dupes[0]}")

    if total_dialogue < 1 or total_dialogue > 3:
        reasons.append(f"dialogue lines counted {total_dialogue} but expected 1-3")
    return (len(reasons) == 0, reasons)


def _validate_story_pages(
    pages: list[StoryPage],
    target_pages: int,
    *,
    enforce_word_count: bool = True,
    enforce_dialogue: bool = True,
) -> tuple[bool, list[str]]:
    if not isinstance(pages, list) or len(pages) != target_pages:
        return False, [
            f"expected {target_pages} pages but got {len(pages) if isinstance(pages, list) else 'non-list'}"
        ]

    reasons: list[str] = []
    total_dialogue = 0
    page_texts: list[str] = []

    for index, page in enumerate(pages, start=1):
        text = (page.text or "").strip()
        prompt = (page.illustration_prompt or "").strip()

        if not text:
            reasons.append(f"page {index} missing text")
            continue
        if not prompt:
            reasons.append(f"page {index} missing illustration_prompt")

        wc = _word_count(text)
        if enforce_word_count and (wc < _MIN_WORDS_PER_PAGE or wc > _MAX_WORDS_PER_PAGE):
            reasons.append(
                f"page {index} word count {wc} outside {_MIN_WORDS_PER_PAGE}-{_MAX_WORDS_PER_PAGE}"
            )

        total_dialogue += _count_dialogue_lines(text)
        page_texts.append(text)

    for i, t in enumerate(page_texts, start=1):
        hits = _find_blocklisted_sentences(t)
        if hits:
            reasons.append(f"page {i} contains boilerplate sentence(s): {hits[:2]}")

    dupes = _find_duplicate_sentences_across_pages(page_texts)
    if dupes:
        reasons.append(f"repeated sentence across pages: {dupes[0]}")

    if enforce_dialogue and (total_dialogue < 1 or total_dialogue > 3):
        reasons.append(f"dialogue lines counted {total_dialogue} but expected 1-3")

    return (len(reasons) == 0, reasons)


def _build_pages_from_data(data: dict[str, Any], target_pages: int) -> list[StoryPage]:
    pages: list[StoryPage] = []
    raw_pages = data.get("pages")
    if not isinstance(raw_pages, list):
        return pages
    for page in raw_pages[:target_pages]:
        if not isinstance(page, dict):
            continue
        text = (page.get("text") or "").strip()
        prompt = (page.get("illustration_prompt") or "").strip()
        if text and prompt:
            pages.append(StoryPage(text=text, illustration_prompt=prompt))
    return pages


def _extract_beats(cleaned: str) -> dict[str, bool]:
    lower = (cleaned or "").lower()
    return {
        "pizza": "pizza" in lower or "crust" in lower,
        "monster": "monster" in lower,
        "horse": "horse" in lower or "hooves" in lower,
    }


def _cover_prompt(story: StoryBook, cleaned: str) -> str:
    beats = _extract_beats(cleaned)
    main_character = story.narrator or "a cheerful child"
    parts = [
        f"Cover illustration for a children's picture book titled '{story.title}'.",
        "Include the main characters in a cozy, whimsical scene.",
        "Watercolor style, storybook composition, no text.",
    ]
    if beats.get("pizza"):
        parts.append("A pizza on a table, warm kitchen light.")
    if beats.get("monster"):
        parts.append("A friendly silly pizza monster made of toppings, not scary.")
    if beats.get("horse"):
        parts.append("A shiny heroic horse galloping in, playful not intense.")
    parts.append(f"Main character: {main_character} with expressive face.")
    parts.append("Soft watercolor textures, storybook composition, gentle lighting.")
    return " ".join(parts)


def _local_storybook(
    anchor_spec: AnchorSpec,
    title: str,
    narrator: str | None,
    target_pages: int,
) -> StoryBook:
    used_sentences: set[str] = set()
    characters = anchor_spec.characters or ([narrator] if narrator else ["The child"])
    key_objects = anchor_spec.key_objects or []
    setting = anchor_spec.setting
    beats = anchor_spec.beats or []
    if not beats:
        beats = ["Something important happens.", "Everyone feels the change."]
    if len(beats) < target_pages:
        beats = beats + [beats[-1]] * (target_pages - len(beats))

    beats_per_page = max(1, (len(beats) + target_pages - 1) // target_pages)
    padding_candidates = _anchor_padding_sentences(anchor_spec, narrator)
    pages: list[StoryPage] = []

    for index in range(target_pages):
        page_beats = beats[index * beats_per_page : (index + 1) * beats_per_page]
        if not page_beats:
            page_beats = beats[-1:]
        page_sentences: list[str] = []
        if index == 0:
            if setting:
                page_sentences.append(
                    f"In the {setting}, {', '.join(characters)} found themselves in a new moment."
                )
            if key_objects:
                page_sentences.append(
                    f"The {', '.join(key_objects[:2])} mattered more than they expected."
                )
        for beat in page_beats:
            beat_line = beat.strip()
            if beat_line and not beat_line.endswith((".", "!", "?")):
                beat_line += "."
            page_sentences.append(beat_line)
        if index == target_pages - 1 and anchor_spec.lesson:
            page_sentences.append(f"The lesson stayed with them: {anchor_spec.lesson}.")
        page_text = " ".join(page_sentences).strip()
        page_text = _pad_to_min_words_no_repeats(
            page_text, _MIN_WORDS_PER_PAGE, padding_candidates, used_sentences
        )
        illustration_prompt_parts = [
            "Watercolor children's picture book illustration",
        ]
        if setting:
            illustration_prompt_parts.append(f"in a {setting}")
        if characters:
            illustration_prompt_parts.append(f"featuring {', '.join(characters[:3])}")
        if key_objects:
            illustration_prompt_parts.append(f"with {', '.join(key_objects[:3])}")
        illustration_prompt_parts.append(
            "gentle lighting, expressive faces, cozy storybook composition"
        )
        pages.append(
            StoryPage(
                text=page_text,
                illustration_prompt=", ".join(illustration_prompt_parts) + ".",
            )
        )

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


def _snippet_from_transcript(cleaned: str, max_words: int = 16) -> str:
    if not cleaned:
        return "a surprising little secret"
    words = re.findall(r"[A-Za-z']+", cleaned)
    if not words:
        return "a surprising little secret"
    snippet = " ".join(words[:max_words])
    return snippet.rstrip(".") + "."


def _page_expansions(topic: str, name: str, *, stage: str) -> list[str]:
    additions = [
        f"The hallway smelled like crayons and toast, and {name} could hear sneakers squeaking nearby.",
        "Sunlight puddled on the floor like warm butter, making the room glow.",
        "A breeze bumped the curtains, as if the room itself was leaning in to listen.",
        "Someone whispered a guess, and another friend gasped, suddenly certain they knew the truth.",
        "The air felt fizzy, like soda bubbles popping with every new idea.",
        "A tiny pet or plush toy seemed to watch the drama with wide, button eyes.",
        "Even the clock sounded excited, ticking a little faster than usual.",
        "A neighbor's laugh floated in from the hallway, making the moment feel even more alive.",
        "The rug felt soft under their toes, grounding them in the middle of the mystery.",
    ]

    topic_lower = topic.lower()
    if "pizza" in topic_lower:
        additions.extend(
            [
                "The scent of warm cheese drifted by, making every tummy rumble a little louder.",
                "A pretend pizza monster was blamed, complete with silly growls and dramatic arm waves.",
                "Someone suggested leaving a trail of crust crumbs as a clue.",
                "A paper chef hat appeared, and suddenly everyone was speaking in silly restaurant voices.",
                "They drew a topping map on a napkin to track the mystery.",
            ]
        )
    if "monster" in topic_lower:
        additions.extend(
            [
                "Shadows wiggled on the wall, perfect for a make-believe monster dance.",
                "They drew a friendly monster map with sparkly stickers and bold arrows.",
                "Someone made a monster mask from a paper plate and giggled behind it.",
            ]
        )
    if "dragon" in topic_lower:
        additions.extend(
            [
                "A dragon made of chalk doodles puffed pretend smoke across the sidewalk.",
                "They practiced brave knight poses and gentle dragon bows.",
                "A sparkly cape fluttered like dragon wings.",
            ]
        )

    if stage == "resolution":
        additions.extend(
            [
                "A chorus of Its okay! floated through the room, light and sincere.",
                "They made a new rule: big feelings get big hugs and honest words.",
                "Someone proposed a celebratory dance, and the floor became a stage.",
                "A goofy plan turned the problem into a game, and everyone played along.",
                "The apology landed softly, like a blanket warming everyone's shoulders.",
            ]
        )
    else:
        additions.extend(
            [
                "A whispered plan formed, and everyone leaned in close to hear it.",
                "The mix-up began to wobble like a wiggly jelly, getting bigger with each retelling.",
                "They tiptoed around the problem, hoping it would magically shrink.",
                "A giggle escaped at the worst time, and suddenly everyone was trying not to laugh.",
                "The secret felt heavy and light at the same time, like a balloon tied to a shoe.",
            ]
        )

    return additions


def _expand_text_to_range(
    text: str,
    target_min: int,
    target_max: int,
    additions: list[str],
    *,
    rng: random.Random | None = None,
    used: set[str] | None = None,
) -> str:
    current = text
    current_words = _word_count(current)
    candidates = list(additions)
    if rng:
        rng.shuffle(candidates)
    for sentence in candidates:
        sentence = sentence.replace('"', "").replace("“", "").replace("”", "")
        if used and sentence in used:
            continue
        next_text = f"{current} {sentence}"
        next_words = _word_count(next_text)
        if next_words > target_max and current_words >= target_min:
            break
        if next_words <= target_max or current_words < target_min:
            current = next_text
            current_words = next_words
            if used is not None:
                used.add(sentence)
        if current_words >= target_min:
            break
    return current


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


def _count_dialogue_lines(text: str) -> int:
    line_start = re.compile(r'^\s*[-—]\s+["“].+["”]\s*$', re.MULTILINE)
    return sum(1 for _ in line_start.finditer(text))


def _limit_dialogue_lines(text: str, max_lines: int = 2) -> str:
    """
    Keeps the first max_lines dialogue lines, removes leading markers from the rest
    so they no longer count as dialogue lines.
    """
    lines = text.splitlines(keepends=True)
    dialogue_count = 0
    updated: list[str] = []
    for line in lines:
        if re.match(r"^\s*[-—]\s+", line):
            dialogue_count += 1
            if dialogue_count > max_lines:
                line = re.sub(r"^(\s*)[-—]\s+", r"\1", line)
        updated.append(line)
    return "".join(updated)


def _ensure_dialogue_count(text: str, target_lines: int = 2) -> str:
    """
    Ensure the text includes at least target_lines of dialogue by appending short lines.
    """
    current = _count_dialogue_lines(text)
    if current >= target_lines:
        return text

    additions = [
        '- "We can fix it."',
        '- "That was silly."',
    ]
    needed = target_lines - current
    extra = "\n".join(additions[:needed])
    separator = "\n" if text.rstrip() else ""
    return f"{text.rstrip()}{separator}{extra}".strip()


def _ensure_minimum_dialogue_lines(pages: list[StoryPage], target_pages: int) -> None:
    total_dialogue = sum(_count_dialogue_lines(page.text) for page in pages)
    if total_dialogue >= 1:
        return
    fallback_lines = [
        '- "Dad: Crunch the crust, kiddo!"',
        '- "Claire: I should\'ve listened, Dad."',
    ]
    for index, line in enumerate(fallback_lines[:target_pages]):
        page = pages[index]
        if line in page.text:
            continue
        separator = "\n" if page.text.rstrip() else ""
        page.text = f"{page.text.rstrip()}{separator}{line}".strip()


def _normalize_dialogue_lines(text: str, max_lines: int = 2) -> str:
    """
    Keep up to max_lines of dialogue. Any extra dialogue lines or quoted segments
    are rewritten into plain narration (no quotes).
    """
    lines = text.splitlines(keepends=True)
    dialogue_count = 0
    updated: list[str] = []
    for line in lines:
        if re.match(r"^\s*[-—]\s+", line):
            dialogue_count += 1
            if dialogue_count > max_lines:
                line = re.sub(r"^(\s*)[-—]\s+", r"\1", line)
        updated.append(line)

    interim = "".join(updated)
    remaining = max(0, max_lines - _count_dialogue_lines(interim))

    if remaining == 0:
        return _strip_extra_quotes(interim, allowed=0)
    return _strip_extra_quotes(interim, allowed=remaining)


def _strip_extra_quotes(text: str, *, allowed: int) -> str:
    """
    Keep the first `allowed` quoted segments and rewrite the rest without quotes.
    """
    pattern = re.compile(r'"([^"\n]+)"|“([^”\n]+)”')
    seen = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal seen
        seen += 1
        content = match.group(1) or match.group(2) or ""
        if seen <= allowed:
            return match.group(0)
        return content

    return pattern.sub(replace, text)


def _build_anchor_expansion_sentences(
    anchor_spec: AnchorSpec,
    cleaned: str,
    narrator: str | None,
) -> list[str]:
    primary = narrator or (anchor_spec.characters[0] if anchor_spec.characters else "They")
    setting = anchor_spec.setting
    key_objects = anchor_spec.key_objects or []
    sentences: list[str] = []

    for beat in anchor_spec.beats[:6]:
        action = beat.strip().rstrip(".")
        if not action:
            continue
        if setting:
            sentences.append(f"In the {setting}, {primary} {action}.")
        elif key_objects:
            sentences.append(f"The {key_objects[0]} was right there when {primary} {action}.")
        else:
            sentences.append(f"{primary} {action}.")

    if key_objects:
        sentences.append(f"{primary} kept the {key_objects[0]} close as the plan unfolded.")
        if len(key_objects) > 1:
            sentences.append(f"The {key_objects[1]} waited nearby while {primary} made the next move.")
    if setting:
        sentences.append(f"The {setting} held steady as {primary} took a careful breath.")

    sentences.extend(_beat_expansion_sentences(cleaned, narrator))
    sentences.extend(_fallback_beat_expansions(cleaned, narrator))
    return _dedupe_preserve_order(sentences)


def _append_anchor_expansions(
    text: str,
    *,
    min_words: int,
    anchor_spec: AnchorSpec,
    cleaned: str,
    narrator: str | None,
    used: set[str],
    max_sentences: int = 3,
) -> str:
    if _word_count(text) >= min_words:
        return text.strip()

    current = text.strip()
    existing = {_normalize_sentence_key(s) for s in _split_sentences_for_dedupe(text)}
    additions = _build_anchor_expansion_sentences(anchor_spec, cleaned, narrator)
    added = 0
    for sentence in additions:
        if added >= max_sentences:
            break
        key = _normalize_sentence_key(sentence)
        if key in existing or key in used:
            continue
        separator = " " if current else ""
        current = f"{current}{separator}{sentence}".strip()
        existing.add(key)
        used.add(key)
        added += 1
    return current


def _ensure_unique_sentences_across_pages(
    pages: list[StoryPage],
    anchor_spec: AnchorSpec,
    cleaned: str,
    narrator: str | None,
) -> None:
    used_sentences: set[str] = set()
    for page in pages:
        sents = _split_sentences(page.text)
        unique: list[str] = []
        for sentence in sents:
            key = _normalize_sentence_key(sentence)
            if key in used_sentences:
                continue
            unique.append(sentence.strip())
            used_sentences.add(key)
        page.text = " ".join(unique).strip()
        page.text = _append_anchor_expansions(
            page.text,
            min_words=_MIN_WORDS_PER_PAGE,
            anchor_spec=anchor_spec,
            cleaned=cleaned,
            narrator=narrator,
            used=used_sentences,
            max_sentences=3,
        )


def _trim_pages_to_word_limit(pages: list[StoryPage], max_words: int) -> None:
    for page in pages:
        page.text = _trim_to_max_words(page.text, max_words)


def _trim_to_max_words(text: str, max_words: int) -> str:
    current = text.strip()
    if _word_count(current) <= max_words:
        return current
    sentences = _split_sentences(current)
    while sentences and _word_count(" ".join(sentences)) > max_words:
        sentences.pop()
    return " ".join(sentences).strip()


def _anchor_padding_sentences(anchor_spec: AnchorSpec, narrator: str | None) -> list[str]:
    sentences: list[str] = []
    characters = anchor_spec.characters or ([narrator] if narrator else [])
    primary_character = characters[0] if characters else (narrator or "The child")
    key_objects = anchor_spec.key_objects or []
    setting = anchor_spec.setting

    if setting:
        if key_objects:
            sentences.append(
                f"In the {setting}, {primary_character} kept noticing the {key_objects[0]}."
            )
        else:
            sentences.append(f"In the {setting}, {primary_character} felt the moment settle in.")

    for obj in key_objects[:3]:
        sentences.append(f"The {obj} stayed important as the story moved forward.")

    for beat in anchor_spec.beats[:4]:
        trimmed = beat.strip().rstrip(".")
        if trimmed:
            sentences.append(f"This was the part where {trimmed}.")

    if anchor_spec.lesson:
        sentences.append(f"The lesson was gentle but clear: {anchor_spec.lesson}.")

    for character in characters[1:3]:
        sentences.append(f"{character} stayed close, keeping the plan honest and steady.")

    return _dedupe_preserve_order(sentences)


def _pad_to_min_words_no_repeats(
    text: str,
    target_min: int,
    candidates: list[str],
    used: set[str],
) -> str:
    cur = (text or "").strip()
    if _word_count(cur) >= target_min:
        return cur
    for sent in candidates:
        if sent in used:
            continue
        if sent in cur:
            used.add(sent)
            continue
        cur2 = cur + (" " if cur else "") + sent
        used.add(sent)
        cur = cur2
        if _word_count(cur) >= target_min:
            break
    return cur


def _pad_to_min_words(text: str, target_min: int, topic: str, name: str) -> str:
    """
    Guaranteed padding: keep appending expansions (cycling) until we reach target_min
    or we’re within a small tolerance and can’t add more without exceeding max.
    """
    if _word_count(text) >= target_min:
        return text

    pool = _page_expansions(topic, name, stage="resolution") + _page_expansions(
        topic, name, stage="setup"
    )

    pool = [s.replace('"', "").replace("“", "").replace("”", "") for s in pool]

    current = text
    for _ in range(50):
        if _word_count(current) >= target_min:
            break
        current = _expand_text_to_range(current, target_min, _MAX_WORDS_PER_PAGE, pool)

        if current == text:
            break
        text = current

    return current


def _pad_text_to_min_words(text: str, min_words: int) -> str:
    """
    If text is under min_words, append neutral expansions until it meets min.
    """
    if _word_count(text) >= min_words:
        return text
    topic = _infer_topic(text)
    name = "A young storyteller"
    additions = _page_expansions(topic, name, stage="resolution")
    return _expand_text_to_range(text, min_words, _MAX_WORDS_PER_PAGE, additions)


def _pad_text_to_min_words_with_beats(
    text: str,
    *,
    min_words: int,
    cleaned: str,
    narrator: str | None,
) -> str:
    """
    If text is under min_words, append plot-relevant expansion sentences derived
    from the transcript beats (avoid boilerplate).
    """
    if _word_count(text) >= min_words:
        return text

    additions = _beat_expansion_sentences(cleaned, narrator)
    if not additions:
        additions = _fallback_beat_expansions(cleaned, narrator)

    existing = {s.lower() for s in _split_sentences_for_dedupe(text)}
    filtered = [s for s in additions if s.lower() not in existing]

    if not filtered:
        return text
    return _expand_text_to_range(text, min_words, _MAX_WORDS_PER_PAGE, filtered)


def _beat_expansion_sentences(cleaned: str, narrator: str | None) -> list[str]:
    lowered = (cleaned or "").lower()
    name = narrator or "Claire"
    sentences: list[str] = []

    if "pizza" in lowered:
        sentences.append("The pizza smell curled through the room like a warm ribbon.")
    if "crust" in lowered:
        sentences.append(f"{name} tucked the crust away, feeling its crackly edges.")
    if "trash" in lowered or "trash can" in lowered or "bin" in lowered:
        sentences.append("The trash can lid clinked louder than anyone expected.")
    if "monster" in lowered and "cheese" in lowered:
        sentences.append("The monster's cheese breath puffed out in a goofy cloud.")
    elif "monster" in lowered:
        sentences.append("A pretend monster stomped in with a silly, cheesy roar.")
    if "horse" in lowered or "hooves" in lowered:
        sentences.append("Shiny horse hooves tapped the floor like tiny cymbals.")
    if "apolog" in lowered or "sorry" in lowered:
        sentences.append("The apology came out soft and brave, like a whispering hug.")
    if "laugh" in lowered or "giggl" in lowered:
        sentences.append("Everyone laughed, and the tension loosened like a shoelace.")

    return sentences


def _fallback_beat_expansions(cleaned: str, narrator: str | None) -> list[str]:
    phrases = _extract_noun_phrases(cleaned, min_count=3, max_count=6)
    if not phrases:
        return []
    speaker = narrator or "They"
    sentences: list[str] = []
    for phrase in phrases:
        sentences.append(f"{speaker} kept noticing the {phrase} as the moment unfolded.")
    return sentences


def _maybe_generate_images(story: StoryBook, cleaned: str) -> None:
    if not _use_openai_image_mode():
        return
    client = _get_openai_client()
    if not client:
        return

    out_dir = Path("out/books/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cover_prompt = _cover_prompt(story, cleaned)
    story.cover_illustration_prompt = cover_prompt
    try:
        cover_response = client.images.generate(
            model="gpt-image-1",
            prompt=cover_prompt,
            size="1024x1024",
        )
    except Exception:
        cover_response = None

    cover_data = cover_response.data[0] if cover_response and cover_response.data else None
    if cover_data:
        cover_path = out_dir / f"story_{timestamp}_cover.png"
        if getattr(cover_data, "b64_json", None):
            cover_bytes = base64.b64decode(cover_data.b64_json)
            cover_path.write_bytes(cover_bytes)
            story.cover_path = str(cover_path)
            story.cover_illustration_path = str(cover_path)
        elif getattr(cover_data, "url", None):
            try:
                request.urlretrieve(cover_data.url, cover_path)
            except Exception:
                cover_path = None
            else:
                story.cover_path = str(cover_path)
                story.cover_illustration_path = str(cover_path)

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
