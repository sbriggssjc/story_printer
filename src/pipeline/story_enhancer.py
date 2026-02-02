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
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError

from src.pipeline.story_builder import StoryBook, StoryPage, _clean_transcript, _find_name, _infer_title

_DEFAULT_MODEL = os.getenv("STORY_MODEL", "gpt-4o-mini")
_DEFAULT_TEMPERATURE = float(os.getenv("STORY_TEMPERATURE", "0.8"))
_DEFAULT_TARGET_PAGES = int(os.getenv("STORY_TARGET_PAGES", "2"))
_DEFAULT_WORDS_PER_PAGE = int(os.getenv("STORY_WORDS_PER_PAGE", "280"))
_DEFAULT_STYLE = os.getenv("STORY_STYLE", "whimsical, funny, heartwarming")
_MIN_WORDS_PER_PAGE = max(240, _DEFAULT_WORDS_PER_PAGE - 40)
_MAX_WORDS_PER_PAGE = min(320, _DEFAULT_WORDS_PER_PAGE + 40)


def _apply_short_transcript_targets(cleaned_transcript: str) -> None:
    global _DEFAULT_WORDS_PER_PAGE, _MIN_WORDS_PER_PAGE, _MAX_WORDS_PER_PAGE
    if _word_count(cleaned_transcript) < 40:
        _DEFAULT_WORDS_PER_PAGE = 200
        _MIN_WORDS_PER_PAGE = 180
        _MAX_WORDS_PER_PAGE = 240

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


def _dedupe_cross_page_sentences(pages: list[StoryPage]) -> bool:
    """
    Removes any sentence that appears verbatim on an earlier page.
    Returns True if it modified anything.
    """
    seen: set[str] = set()
    changed = False

    for page in pages:
        sents = _split_sentences(page.text)
        out: list[str] = []
        for s in sents:
            key = re.sub(r"\s+", " ", s.strip())
            if key in seen:
                changed = True
                continue
            out.append(s)
            seen.add(key)

        # Rebuild paragraph-ish text
        new_text = " ".join(out).strip()
        if new_text and new_text != (page.text or "").strip():
            page.text = new_text

    return changed


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


def _infer_required_terms(cleaned_transcript: str) -> list[str]:
    """Lightweight anchor terms derived from the transcript."""
    lower = (cleaned_transcript or "").lower()
    required: list[str] = []
    # Hard anchors for your current use case; safe + simple
    if "pizza" in lower:
        required.extend(["pizza", "crust"])
    if "monster" in lower:
        required.append("monster")
    # If transcript implies apology/laughter
    if "sorry" in lower or "apolog" in lower:
        required.append("sorry")
    if "laugh" in lower or "giggl" in lower:
        required.append("laug")
    return sorted(set(required))


def _missing_required_terms(page_texts: list[str], required_terms: list[str]) -> list[str]:
    blob = " ".join(page_texts).lower()
    missing = []
    for term in required_terms:
        if term not in blob:
            missing.append(term)
    return missing


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


def _build_anchor_list(cleaned: str, narrator: str | None) -> list[str]:
    anchors: list[str] = []
    seen: set[str] = set()

    for name in _extract_proper_names(cleaned, narrator):
        normalized = name.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        anchors.append(normalized)
        seen.add(key)

    apology_terms = []
    lowered = (cleaned or "").lower()
    if "sorry" in lowered:
        apology_terms.append("sorry")
    if "apolog" in lowered:
        apology_terms.append("apologize")
    if "fix" in lowered:
        apology_terms.append("fix")

    for term in apology_terms:
        if term not in seen:
            anchors.append(term)
            seen.add(term)

    phrase_exclusions = set(seen)
    phrases = _extract_noun_phrases(
        cleaned, min_count=3, max_count=8, excluded=phrase_exclusions
    )
    for phrase in phrases:
        key = phrase.lower()
        if key in seen:
            continue
        anchors.append(phrase)
        seen.add(key)

    return anchors


def _build_anchor_beats(cleaned: str) -> dict[str, bool]:
    lower = (cleaned or "").lower()
    return {
        "pizza": "pizza" in lower,
        "crust": "crust" in lower,
        "monster": "monster" in lower,
        "horse": "horse" in lower,
        "dad": "dad" in lower or "father" in lower,
    }


def _anchor_ok(text: str, beats: dict[str, bool]) -> tuple[bool, list[str]]:
    t = (text or "").lower()
    missing = []
    if beats.get("pizza") and "pizza" not in t:
        missing.append("pizza")
    if beats.get("crust") and "crust" not in t:
        missing.append("crust")
    if beats.get("monster") and "monster" not in t:
        missing.append("monster")
    if beats.get("horse") and "horse" not in t:
        missing.append("horse")
    if beats.get("dad") and ("dad" not in t and "father" not in t):
        missing.append("dad")
    return (len(missing) == 0, missing)


def _extract_fidelity_keywords(cleaned: str) -> list[str]:
    # small, pragmatic: keep the “story spine”
    lowered = (cleaned or "").lower()
    must = []
    for k in ["pizza", "crust", "monster", "apolog", "laugh"]:
        if k in lowered:
            must.append(k)
    # if transcript is tiny, at least demand 2 anchors
    return must[:4]


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


def enhance_to_storybook(transcript: str, *, target_pages: int = 2) -> StoryBook:
    cleaned = _clean_transcript(transcript)
    _apply_short_transcript_targets(cleaned)
    narrator = _find_name(cleaned) if cleaned else None
    title = _infer_title(cleaned) if cleaned else "My Story"
    subtitle = "A story told out loud"
    target_pages = target_pages or _DEFAULT_TARGET_PAGES

    if _is_openai_story_requested():
        print("STORY_ENHANCE_MODE=openai detected")
        openai_story = _openai_storybook(cleaned, title, narrator, target_pages)
        if openai_story:
            _maybe_generate_images(openai_story, cleaned)
            return openai_story

    local_story = _local_storybook(cleaned, title, narrator, target_pages)
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
    cleaned: str, narrator: str | None, target_pages: int
) -> tuple[str, str, list[str]]:
    system_prompt = (
        "You are a celebrated children's picture-book author and editor. "
        "Expand the transcript into a rich, creative, kid-safe story (not a summary). "
        "Return ONLY strict JSON that follows the schema exactly."
    )
    beat_sheet = (
        "Beat sheet:\n"
        "1) Warm opening with setting and sensory details.\n"
        "2) Mischief/choice from the transcript idea.\n"
        "3) Rising trouble and playful tension.\n"
        "4) Page-turn hook ending page 1.\n"
        "5) Big moment on page 2.\n"
        "6) Sincere apology.\n"
        "7) Funny resolution.\n"
        "8) Warm ending with emotional lift.\n"
    )
    anchors = _build_anchor_list(cleaned, narrator)
    anchor_line = ", ".join(anchors) if anchors else "none"
    user_prompt = (
        "TRANSCRIPT (core plot MUST be preserved; you may rephrase sentences, but do not invent new major events):\n"
        f"{cleaned or 'Empty transcript.'}\n\n"
        "Non-negotiable story anchors (must appear clearly):\n"
        "- Claire hides the pizza crust.\n"
        "- A pizza monster shows up (real or pretend is fine).\n"
        "- Claire apologizes.\n"
        "- Everyone laughs / warm resolution.\n\n"
        f"NARRATOR (keep name if present): {narrator or 'none'}\n\n"
        "Story requirements:\n"
        f"- Exactly {target_pages} pages.\n"
        "- Page 1: setup + mischief + rising trouble + page-turn hook.\n"
        "- Page 2: big moment + apology + funny resolution + warm ending.\n"
        "- Exactly two dialogue lines total. Each must be a full sentence in quotes on its own line starting with '- ' or '— '.\n"
        "- Dialogue must be short (<= 12 words each) and natural.\n"
        "- Do not use quotation marks for emphasis or any other purpose.\n"
        "- Kid-safe, whimsical, humorous, creative expansion.\n"
        "- Include sensory details and an emotional arc.\n"
        "- Avoid repeating filler sentences or stock phrases.\n"
        "- Do NOT repeat any exact sentence across pages. Each sentence must be unique.\n"
        "- Keep names consistent; use narrator if provided.\n"
        f"- Your story MUST include all anchors exactly once or more: {anchor_line}.\n"
        f"- {_MIN_WORDS_PER_PAGE}–{_MAX_WORDS_PER_PAGE} words per page (target {_DEFAULT_WORDS_PER_PAGE}).\n"
        "- Each page must include a rich illustration_prompt in consistent watercolor picture-book style.\n"
        f"{beat_sheet}\n"
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
    required_terms = _infer_required_terms(cleaned)
    if required_terms:
        user_prompt += (
            "\nFaithfulness requirements (must include these exact terms somewhere in the story):\n"
            + "- " + ", ".join(required_terms) + "\n"
        )
    user_prompt += (
        "\nHard bans:\n"
        "- Do NOT use any generic boilerplate sensory sentences.\n"
        "- Do NOT repeat any sentence verbatim across pages.\n"
        "- Do NOT include these sentences (exactly as written):\n"
        + "\n".join([f"  - {s}" for s in sorted(_BLOCKLIST_SENTENCES)])
        + "\n"
    )
    return system_prompt, user_prompt, anchors


def _openai_storybook(
    cleaned: str,
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

    system_prompt, user_prompt, anchors = _build_openai_prompts(
        cleaned, narrator, target_pages
    )
    missing_anchor_note: list[str] = []

    for attempt in range(2):
        correction = ""
        if attempt == 1:
            correction = (
                "Rewrite the story to satisfy ALL requirements:\n"
                f"- EXACTLY {target_pages} pages.\n"
                f"- EACH page MUST be between {_MIN_WORDS_PER_PAGE} and {_MAX_WORDS_PER_PAGE} words.\n"
                '- Exactly two dialogue lines total using straight quotes, e.g. "...". Each must be a full sentence on its own line starting with "- " or "— ".\n'
                "- Each dialogue line must be short (<= 12 words) and natural.\n"
                "- Do not use quotation marks for emphasis or any other purpose.\n"
                "- Return ONLY JSON matching the schema. No extra keys, no commentary."
            )
            if missing_anchor_note:
                correction += (
                    "\nMissing anchors from previous attempt:\n- "
                    + "\n- ".join(missing_anchor_note)
                    + "\nEnsure every anchor appears at least once."
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
        used_sentences: set[str] = set()
        beats = _extract_beats(cleaned)
        padding_candidates = _transcript_padding_sentences(beats, narrator or "Claire")
        for page in pages:
            page.text = _pad_to_min_words_no_repeats(
                page.text,
                _MIN_WORDS_PER_PAGE,
                padding_candidates,
                used_sentences,
            )
            page.text = _normalize_dialogue_lines(page.text, max_lines=2)

        total_dialogue = sum(_count_dialogue_lines(page.text) for page in pages)
        if total_dialogue < 2:
            pages[-1].text = _ensure_dialogue_count(pages[-1].text, target_lines=2)

        # Remove any verbatim repeated sentences across pages, then re-pad to hit min words.
        if _dedupe_cross_page_sentences(pages):
            for page in pages:
                page.text = _pad_to_min_words_no_repeats(
                    page.text,
                    _MIN_WORDS_PER_PAGE,
                    padding_candidates,
                    used_sentences,
                )

        os.environ["__STORY_CLEANED_TRANSCRIPT__"] = cleaned or ""
        valid, reasons = _validate_story_pages(pages, target_pages)
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

        all_text = " ".join(
            [
                (p.get("text") or "")
                for p in (data.get("pages") or [])
                if isinstance(p, dict)
            ]
        )
        must = _extract_fidelity_keywords(cleaned)
        fidelity_reasons, banned_phrases = _fails_fidelity(all_text, must)
        if fidelity_reasons:
            print(f"Fidelity failed: {', '.join(fidelity_reasons)}")
            if attempt == 0:
                retry_instruction = ""
                if banned_phrases:
                    retry_instruction = (
                        "\nDo not use any of these phrases: "
                        + "; ".join(banned_phrases)
                        + ". Replace them with fresh, specific detail from the transcript."
                    )
                user_prompt = (
                    f"{user_prompt}\n\nIssues:\n- "
                    + "\n- ".join(fidelity_reasons)
                    + "\n\nRewrite while keeping the transcript’s core events EXACT."
                    + retry_instruction
                )
                continue
            return None

        page_texts = [page.text for page in pages]
        beat_map = _build_anchor_beats(cleaned)
        ok, missing_beats = _anchor_ok(" ".join(page_texts), beat_map)
        if not ok:
            print(f"Anchor validation failed: {', '.join(missing_beats)}")
            if attempt == 0:
                missing_anchor_note = missing_beats
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

    required_terms = _infer_required_terms(os.getenv("__STORY_CLEANED_TRANSCRIPT__", "") or "")
    missing = _missing_required_terms(page_texts, required_terms) if required_terms else []
    if missing:
        reasons.append(f"missing required transcript term(s): {missing}")

    if total_dialogue != 2:
        reasons.append(f"dialogue lines counted {total_dialogue} but expected 2")
    return (len(reasons) == 0, reasons)


def _validate_story_pages(pages: list[StoryPage], target_pages: int) -> tuple[bool, list[str]]:
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
        if wc < _MIN_WORDS_PER_PAGE or wc > _MAX_WORDS_PER_PAGE:
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

    required_terms = _infer_required_terms(os.getenv("__STORY_CLEANED_TRANSCRIPT__", "") or "")
    missing = _missing_required_terms(page_texts, required_terms) if required_terms else []
    if missing:
        reasons.append(f"missing required transcript term(s): {missing}")

    if total_dialogue != 2:
        reasons.append(f"dialogue lines counted {total_dialogue} but expected 2")

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


def _extract_story_beats(cleaned: str) -> dict[str, str]:
    lower = (cleaned or "").lower()
    claire_name = "Claire" if "claire" in lower else "Claire"
    dad_name = "Dad" if any(token in lower for token in ("dad", "father")) else "Dad"
    object_name = "pizza crust" if "crust" in lower or "pizza" in lower else "pizza crust"

    deception_phrases = ("hide", "hid", "threw away", "throw away", "lied", "lie")
    deception_action = "hid it under her napkin"
    if any(phrase in lower for phrase in deception_phrases):
        if "threw away" in lower or "throw away" in lower:
            deception_action = "threw it away when no one was looking"
        elif "lied" in lower or "lie" in lower:
            deception_action = "tucked it away and told a small lie"
        else:
            deception_action = "hid it under her napkin"

    lesson_line = "I wish I would have listened."
    if "should've listened" in lower or "should have listened" in lower:
        lesson_line = "I should have listened."
    if "wish i listened" in lower or "wish i'd listened" in lower:
        lesson_line = "I wish I listened."

    return {
        "claire_name": claire_name,
        "dad_name": dad_name,
        "object_name": object_name,
        "deception_action": deception_action,
        "lesson_line": lesson_line,
    }


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
        "Watercolor children's picture book cover illustration, cozy and whimsical, no text.",
        f"Theme: {story.title}.",
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


    t = (cleaned or "").lower()
    return {
        "pizza": "pizza" in t,
        "crust": "crust" in t,
        "trash": "trash" in t or "threw" in t or "thrown" in t,
        "monster": "monster" in t,
        "horse": "horse" in t,
        "dad": "dad" in t or "father" in t,
        "lied": "lied" in t or "lie" in t,
        "apology": "sorry" in t or "wish i would have listened" in t or "apolog" in t,
    }


def _local_page_expansions(claire_name: str, dad_name: str, *, stage: str) -> list[str]:
    if stage == "resolution":
        return [
            f"{dad_name}'s horse snorted softly and stamped the ground, eager and brave.",
            f"{claire_name} felt the knot in her chest loosen as the danger passed.",
            "The kitchen smelled warm and buttery, and the plates clinked gently on the table.",
            "A neighbor waved through the window, smiling at the excitement.",
            "The wind carried away the last rumble of the monster.",
            f"{dad_name} brushed {claire_name}'s hair back and listened closely.",
            "The house felt safe again, like a blanket tucked around their shoulders.",
            "They laughed in relief, the kind of laughter that turns fear into a story.",
        ]
    return [
        "The afternoon light painted soft squares on the floor.",
        f"{claire_name} shifted her feet, unsure, while the room grew still.",
        "The pizza smelled cheesy and warm, but the air felt prickly with worry.",
        f"{dad_name} looked up with a gentle question in his eyes.",
        "A curtain fluttered, and a shadow slid across the wall.",
        "The monster's footsteps thudded like drums outside.",
        "The table vibrated with each stomp, making cups tremble.",
        f"{claire_name}'s heart beat fast, like a small bird in her chest.",
    ]


def _local_storybook(
    cleaned: str,
    title: str,
    narrator: str | None,
    target_pages: int,
) -> StoryBook:
    beats = _extract_story_beats(cleaned)
    claire_name = beats["claire_name"]
    dad_name = beats["dad_name"]
    object_name = beats["object_name"]
    seed = abs(hash(cleaned or "storybook")) % (2**32)
    rng = random.Random(seed)
    used_sentences: set[str] = set()

    page1 = (
        f"{claire_name} and {dad_name} shared a cozy pizza lunch at home. "
        f"There was just one {object_name} left, and {claire_name} wanted it so badly her fingers curled. "
        f"Instead of asking, she {beats['deception_action']}, and when {dad_name} asked about the missing crust, "
        f"{claire_name} gave a tiny fib that felt like a pebble in her pocket. "
        "The room went quiet for a moment, as if the house was holding its breath. "
        "Outside the window, a shadow stretched long and a growl rumbled, soft at first and then louder. "
        f"A pizza monster stomped into the yard, sniffing the air for {object_name}s, "
        "and it peered in with hungry, googly eyes. "
        f"{claire_name} froze, her stomach fluttering as the monster crept closer."
    )
    page1 = _expand_text_to_range(
        page1,
        _MIN_WORDS_PER_PAGE,
        _MAX_WORDS_PER_PAGE,
        _local_page_expansions(claire_name, dad_name, stage="setup"),
        rng=rng,
        used=used_sentences,
    )

    page2 = (
        "The pizza monster lunged, and the table rattled as it swiped the air. "
        f"Just then, {dad_name} burst through the gate riding a shiny horse, its mane glittering like coins. "
        f"With one brave swing, {dad_name} sent the monster tumbling away and chased it from the yard. "
        f"{claire_name}'s eyes filled with tears. She whispered, \"{beats['lesson_line']}\" "
        f"and she told {dad_name} the truth about the {object_name}. "
        f"{dad_name} knelt beside her, hugged her tight, and said mistakes could be mended with honesty. "
        f"Together they shared fresh pizza, and {claire_name} promised to speak up next time. "
        "The sun set warm and golden, and the house felt safe and peaceful again."
    )
    page2 = _expand_text_to_range(
        page2,
        _MIN_WORDS_PER_PAGE,
        _MAX_WORDS_PER_PAGE,
        _local_page_expansions(claire_name, dad_name, stage="resolution"),
        rng=rng,
        used=used_sentences,
    )

    pages = [
        StoryPage(
            text=page1,
            illustration_prompt=(
                f"A cozy kitchen storybook scene with {claire_name} and {dad_name} at a pizza lunch, "
                f"a single {object_name} on the plate, and a silly pizza monster looming outside the window, "
                "bright colors, gentle watercolor textures, warm afternoon light, and playful tension."
            ),
        ),
        StoryPage(
            text=page2,
            illustration_prompt=(
                f"A triumphant, heartwarming illustration of {dad_name} on a shiny horse chasing away a pizza monster, "
                f"with {claire_name} relieved and hugging {dad_name} afterward, cozy home setting, "
                "gentle watercolor textures, warm golden light, and a peaceful ending."
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


def _transcript_padding_sentences(beats: dict[str, bool], name: str) -> list[str]:
    s: list[str] = []
    if beats.get("pizza"):
        s.append(
            "The kitchen smelled like warm pizza, and Claire’s tummy rumbled in a very dramatic way."
        )
    if beats.get("crust"):
        s.append("The crust felt crackly and brave, like it was daring her to take one tiny bite.")
    if beats.get("trash"):
        s.append("The trash can lid went thump, and the secret suddenly felt louder than it should have.")
    if beats.get("lied"):
        s.append("A little lie can start small, but it can grow legs and tap-dance around your heart.")
    if beats.get("monster"):
        s.append("Then came the pizza monster—cheesy, stompy, and much sillier than scary if you looked closely.")
    if beats.get("horse") and beats.get("dad"):
        s.append(
            "Her dad burst in like a storybook hero, riding a shiny horse that looked like it had been polished with sunshine."
        )
    if beats.get("apology"):
        s.append("Claire’s voice got quiet, and her honesty finally felt like a breath of fresh air.")
    s.append(
        "Claire tried to act normal, but her cheeks felt warm, like they were telling the truth without permission."
    )
    s.append(
        "For one tiny moment, everything paused—then the story rolled forward like a pizza dough being stretched."
    )
    return s


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
        elif getattr(cover_data, "url", None):
            try:
                request.urlretrieve(cover_data.url, cover_path)
            except Exception:
                cover_path = None
            else:
                story.cover_path = str(cover_path)

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
