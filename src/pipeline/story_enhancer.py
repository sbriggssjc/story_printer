"""
Environment variables:
- STORY_ENHANCE_MODE: set to "openai" to use OpenAI for story enhancement.
- STORY_IMAGE_MODE: set to "openai" to generate images with OpenAI.
- OPENAI_API_KEY: required for OpenAI modes.
- STORY_MODEL: OpenAI model for story enhancement (default: "gpt-4o-mini").
- STORY_TEMPERATURE: OpenAI sampling temperature (default: "0.7").
- STORY_TARGET_PAGES: number of pages to generate (default: "2").
- STORY_WORDS_PER_PAGE: word count target per page (default: "260").
- STORY_STYLE: story tone/style (default: "whimsical, funny, heartwarming").

Optional robustness knobs:
- STORY_DIALOGUE_MIN: minimum dialogue lines across the whole story (default: 1)
- STORY_DIALOGUE_MAX: maximum dialogue lines across the whole story (default: 3)
- STORY_ANCHOR_MAX_TERMS: number of extracted anchor terms (default: 8)
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import request

from src.pipeline.story_builder import StoryBook, StoryPage, _clean_transcript, _find_name, _infer_title

_DEFAULT_MODEL = os.getenv("STORY_MODEL", "gpt-4o-mini")
_DEFAULT_TEMPERATURE = float(os.getenv("STORY_TEMPERATURE", "0.7"))
_DEFAULT_TARGET_PAGES = int(os.getenv("STORY_TARGET_PAGES", "2"))
_DEFAULT_WORDS_PER_PAGE = int(os.getenv("STORY_WORDS_PER_PAGE", "260"))
_DEFAULT_STYLE = os.getenv("STORY_STYLE", "whimsical, funny, heartwarming")

# Keep ranges forgiving; enforce via postprocessing.
_MIN_WORDS_PER_PAGE = max(200, _DEFAULT_WORDS_PER_PAGE - 60)
_MAX_WORDS_PER_PAGE = min(340, _DEFAULT_WORDS_PER_PAGE + 80)

_DIALOGUE_MIN = int(os.getenv("STORY_DIALOGUE_MIN", "1"))
_DIALOGUE_MAX = int(os.getenv("STORY_DIALOGUE_MAX", "3"))

_ANCHOR_MAX_TERMS = int(os.getenv("STORY_ANCHOR_MAX_TERMS", "8"))
_ANCHOR_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "so",
    "then",
    "when",
    "one",
    "day",
    "about",
    "story",
    "comes",
    "came",
    "along",
    "tries",
    "tried",
    "told",
    "said",
    "says",
    "its",
    "it's",
}

_STOPWORDS = {
    "the", "and", "then", "with", "from", "that", "this", "when", "they", "she", "he",
    "a", "an", "to", "of", "in", "on", "at", "for", "as", "is", "was", "were", "be",
    "it", "its", "their", "his", "her", "we", "i", "you", "my", "our", "your",
    "mom", "mother", "mommy", "mum", "momma", "dad", "father", "daddy", "papa", "pa",
    "parent", "parents",
}

_CREATURE_KEYWORDS = {
    "monster",
    "dragon",
    "dinosaur",
    "creature",
    "beast",
    "ogre",
    "troll",
    "goblin",
    "alien",
    "ghost",
    "vampire",
    "werewolf",
    "unicorn",
}

# Phrases that commonly appear as generic filler; we REMOVE them (do not fail).
_BOILERPLATE_SENTENCES = {
    "Sunlight puddled on the floor like warm butter, making the room glow.",
    "The hallway smelled like crayons and toast, and {name} could hear sneakers squeaking nearby.",
    "Someone whispered a guess, and another friend gasped, suddenly certain they knew the truth.",
    "The air felt fizzy, like soda bubbles popping with every new idea.",
    "Even the clock sounded excited, ticking a little faster than usual.",
}


def _normalize_anchor_terms(terms: list[str] | None, *, max_terms: int = 8) -> list[str]:
    if not terms:
        return []
    cleaned: list[str] = []
    for t in terms:
        if not t:
            continue
        tok = re.sub(r"[^a-z0-9\s'-]+", "", str(t).strip().lower())
        tok = re.sub(r"\s+", " ", tok).strip()
        if not tok or tok in _ANCHOR_STOPWORDS:
            continue
        if len(tok) < 4 and tok not in {"dad", "mom", "kid"}:
            continue
        cleaned.append(tok)
    out: list[str] = []
    seen = set()
    for t in cleaned:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def enhance_to_storybook(transcript: str, *, target_pages: int = 2) -> StoryBook:
    cleaned = _clean_transcript(transcript)
    narrator = _find_name(cleaned) if cleaned else None
    title = _infer_title(cleaned) if cleaned else "My Story"
    subtitle = "A story told out loud"
    target_pages = target_pages or _DEFAULT_TARGET_PAGES

    if _is_openai_story_requested():
        print("STORY_ENHANCE_MODE=openai detected")
        openai_story = _openai_storybook(cleaned, title, narrator, target_pages)
        if openai_story:
            _maybe_generate_images(openai_story)
            return openai_story

    local_story = _local_storybook(cleaned, title, narrator, target_pages)
    _maybe_generate_images(local_story)
    return local_story


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


def _build_cover_prompt(title: str, narrator: str | None, cleaned: str) -> str:
    # Generic: pull a couple of important nouns and proper nouns without theme-locking
    proper = re.findall(r"\b[A-Z][a-z]{2,}\b", cleaned or "")
    proper = [p for p in proper if p.lower() not in {"the", "and", "then", "after", "before"}]
    main_name = narrator or (proper[0] if proper else "a child")

    keywords = re.findall(r"[A-Za-z']+", (cleaned or "").lower())
    stop = {
        "the", "and", "then", "after", "before", "with", "from", "that", "this", "when", "they",
        "she", "he", "her", "his", "dad", "mom", "a", "an", "to", "in", "on", "at", "of",
    }
    nouns = [w for w in keywords if w not in stop and len(w) > 3]
    key_bits = ", ".join(dict.fromkeys(nouns[:6])) if nouns else "a whimsical adventure"

    return (
        "Children's picture book cover illustration, watercolor style, soft texture, warm lighting, "
        "no text, no letters, no watermark. "
        f"Main character: {main_name}. "
        f"Theme elements: {key_bits}. "
        "Centered composition, friendly expressive faces, cozy whimsical mood."
    )


def _build_openai_prompts(cleaned: str, narrator: str | None, target_pages: int) -> tuple[str, str]:
    system_prompt = (
        "You are a celebrated children's picture-book author and editor. "
        "Expand the transcript into a rich, creative, kid-safe story (not a summary). "
        "IMPORTANT: stay faithful to the transcript's core plot beats and characters. "
        "Return ONLY strict JSON that follows the schema exactly."
    )

    # We keep constraints, but we repair post-hoc too.
    user_prompt = (
        "TRANSCRIPT (idea source; keep the SAME core plot and characters; do NOT invent a different story):\n"
        f"{cleaned or 'Empty transcript.'}\n\n"
        f"NARRATOR (use name if present): {narrator or 'none'}\n\n"
        "Story requirements:\n"
        f"- Exactly {target_pages} pages.\n"
        "- Page 1: setup + mischief/choice + rising trouble + page-turn hook.\n"
        "- Page 2: big moment + apology/lesson + funny resolution + warm ending.\n"
        f"- Dialogue: {_DIALOGUE_MIN}–{_DIALOGUE_MAX} short quoted lines TOTAL across the entire story.\n"
        f"- About {_MIN_WORDS_PER_PAGE}–{_MAX_WORDS_PER_PAGE} words per page.\n"
        "- Kid-safe, whimsical, humorous, vivid sensory details.\n"
        "- Avoid repeated stock phrases or generic filler.\n"
        "- Each page must include a rich illustration_prompt in consistent watercolor picture-book style.\n\n"
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
    return system_prompt, user_prompt


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
        print("OpenAI enhancement skipped: openai package not available.")
        return None

    print(f"OpenAI model: {_DEFAULT_MODEL}")

    system_prompt, user_prompt = _build_openai_prompts(cleaned, narrator, target_pages)
    anchor_spec = _extract_anchor_spec(cleaned, max_terms=_ANCHOR_MAX_TERMS)
    required_terms = _anchor_terms_for_expansion(anchor_spec, max_terms=_ANCHOR_MAX_TERMS)

    # We will try up to 3 attempts, but we also *repair*.
    for attempt in range(3):
        correction = ""
        if attempt >= 1:
            correction = (
                "Revise the previous JSON to better satisfy word-count per page, dialogue-count, "
                "avoid repeated sentences, and stay faithful to the transcript plot."
            )

        try:
            use_responses = hasattr(client, "responses")
            endpoint = "responses" if use_responses else "chat.completions"
            print(f"OpenAI endpoint: {endpoint} (SDK)")

            content = _request_openai_story(
                client=client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                correction=correction,
                use_responses=use_responses,
                target_pages=target_pages,
            )
        except Exception as exc:
            print(f"OpenAI enhancement failed: {exc}")
            return None

        if not content:
            print("OpenAI enhancement failed: empty response.")
            return None

        data = _parse_json(content)
        if not isinstance(data, dict):
            print("Validation failed: response is not a JSON object")
            continue

        pages = _build_pages_from_data(data, target_pages)
        if len(pages) != target_pages:
            print("Validation failed: unable to parse pages")
            continue

        # REPAIR PHASE (critical for robustness)
        pages = _repair_pages(
            pages=pages,
            narrator=narrator,
            required_terms=required_terms,
            target_min=_MIN_WORDS_PER_PAGE,
            target_max=_MAX_WORDS_PER_PAGE,
        )

        # Validate after repair
        ok, reasons = _validate_story_pages(
            pages=pages,
            anchor_spec=anchor_spec,
            target_min=_MIN_WORDS_PER_PAGE,
            target_max=_MAX_WORDS_PER_PAGE,
        )
        if not ok:
            print(f"Validation failed: {', '.join(reasons)}")
            # tighten prompt on subsequent tries by appending issues
            user_prompt = f"{user_prompt}\n\nIMPORTANT FIXES NEEDED:\n- " + "\n- ".join(reasons)
            continue

        return StoryBook(
            title=(data.get("title") or title).strip() or title,
            subtitle=(data.get("subtitle") or "A story told out loud").strip() or "A story told out loud",
            pages=pages,
            narrator=(data.get("narrator") or narrator),
            cover_prompt=_build_cover_prompt(title, narrator, cleaned),
            cover_image_path=None,
        )

    print("OpenAI enhancement failed: could not produce a valid story after retries.")
    return None


def _request_openai_story(
    client,
    system_prompt: str,
    user_prompt: str,
    correction: str,
    use_responses: bool,
    target_pages: int,
) -> str:
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
        )
        return _extract_response_text(response)

    # fallback path (rare if responses exists)
    response = client.chat.completions.create(
        model=_DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            *([{"role": "user", "content": correction}] if correction else []),
        ],
        temperature=_DEFAULT_TEMPERATURE,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content if response.choices else ""


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
        # For strict JSON schema enforcement in OpenAI Responses,
        # required must include every property key (even if nullable).
        "required": ["title", "subtitle", "narrator", "pages"],
        "additionalProperties": False,
    }


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                text_part = getattr(part, "text", None)
                if text_part:
                    return str(text_part).strip()
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


@dataclass(frozen=True)
class AnchorSpec:
    names: list[str]
    keywords: list[str]
    creature_keywords: list[str]
    transcript_word_count: int


# -----------------------------
# Robust fidelity / anchor terms
# -----------------------------
def _extract_anchor_spec(transcript: str, max_terms: int = 8) -> AnchorSpec:
    """
    Transcript-agnostic term extraction for fidelity checking.
    Extracts proper names + high-signal keywords for loose matching in the story.
    """
    t = (transcript or "").strip()
    if not t:
        return AnchorSpec(names=[], keywords=[], creature_keywords=[], transcript_word_count=0)

    # Keep names as anchors when possible, but avoid generic family labels.
    raw_names = re.findall(r"\b[A-Z][a-z]{2,}\b", t)
    names: list[str] = []
    for name in raw_names:
        lowered = name.lower()
        if lowered in _STOPWORDS:
            continue
        if lowered not in names:
            names.append(lowered)

    words_all = re.findall(r"[A-Za-z']+", t.lower())
    transcript_word_count = len(words_all)
    words = [w for w in words_all if w not in _STOPWORDS and len(w) > 3]
    freq = [w for w, _ in Counter(words).most_common(max_terms * 2)]

    keywords: list[str] = []
    for w in freq:
        if w in names:
            continue
        if w not in keywords:
            keywords.append(w)
        if len(keywords) >= max_terms:
            break
    keywords = _normalize_anchor_terms(keywords, max_terms=max_terms)

    creature_hits: list[str] = []
    for creature in sorted(_CREATURE_KEYWORDS):
        if re.search(rf"\\b{re.escape(creature)}s?\\b", t.lower()):
            creature_hits.append(creature)

    return AnchorSpec(
        names=names,
        keywords=keywords,
        creature_keywords=creature_hits,
        transcript_word_count=transcript_word_count,
    )


def _anchor_terms_for_expansion(anchor_spec: AnchorSpec, max_terms: int) -> list[str]:
    anchors: list[str] = []
    for name in anchor_spec.names:
        if name not in anchors:
            anchors.append(name)
        if len(anchors) >= max_terms:
            return anchors
    for keyword in anchor_spec.keywords:
        if keyword not in anchors:
            anchors.append(keyword)
        if len(anchors) >= max_terms:
            break
    return anchors


def _count_term_hits(text: str, terms: list[str]) -> int:
    lowered = (text or "").lower()
    hits = 0
    for term in terms:
        if not term:
            continue
        if re.search(rf"\\b{re.escape(term.lower())}s?\\b", lowered):
            hits += 1
    return hits


# -----------------------------
# Repair / validation utilities
# -----------------------------
def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text or ""))


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def _count_dialogue_lines(text: str) -> int:
    quoted = re.findall(r'"[^"]+"', text or "")
    curly = re.findall(r"“[^”]+”", text or "")
    return len(quoted) + len(curly)


def _limit_dialogue(text: str, max_lines: int) -> str:
    """
    If the model adds lots of quoted lines, convert extras to unquoted text.
    """
    if max_lines <= 0:
        return re.sub(r'["“”]', "", text or "")

    s = text or ""
    # Normalize curly quotes to straight for easier handling, then restore not needed
    s = s.replace("“", '"').replace("”", '"')

    matches = list(re.finditer(r'"([^"]+)"', s))
    if len(matches) <= max_lines:
        return text or ""

    # Replace extras: remove surrounding quotes but keep content
    out_s = s
    extras = matches[max_lines:]
    for m in reversed(extras):
        inner = m.group(1)
        out_s = out_s[: m.start()] + inner + out_s[m.end() :]
    return out_s


def _ensure_min_dialogue(text: str, narrator: str | None) -> str:
    """
    If no dialogue exists, inject one short quote line near the end.
    """
    if _count_dialogue_lines(text) >= _DIALOGUE_MIN:
        return text
    name = narrator or _find_first_name(text) or "Someone"
    # Add a single short line. Keep it generic.
    addition = f' "{name} whispered, \\"I\\\'m sorry.\\""'
    # Avoid adding to a blank text
    if text.strip().endswith((".", "!", "?")):
        return text.strip() + addition
    return text.strip() + "." + addition


def _find_first_name(text: str) -> str | None:
    for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", text or ""):
        cand = match.group(0)
        if cand.lower() in _STOPWORDS:
            continue
        return cand
    return None


def _remove_boilerplate(text: str, narrator: str | None) -> str:
    name = narrator or "A child"
    sentences = _split_sentences(text)
    cleaned = []
    for s in sentences:
        if s in _BOILERPLATE_SENTENCES:
            continue
        # also remove formatted boilerplate template
        if s == f"The hallway smelled like crayons and toast, and {name} could hear sneakers squeaking nearby.":
            continue
        cleaned.append(s)
    return " ".join(cleaned).strip()


def _dedupe_across_pages(pages: list[StoryPage]) -> list[StoryPage]:
    seen = set()
    out: list[StoryPage] = []
    for p in pages:
        sentences = _split_sentences(p.text)
        kept = []
        for s in sentences:
            key = s.strip()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            kept.append(key)
        p.text = " ".join(kept).strip()
        out.append(p)
    return out


def _expand_text_to_range(text: str, target_min: int, target_max: int, additions: list[str]) -> str:
    """
    Expand with varied, transcript-keyword-aware additions until we hit target_min (without exceeding target_max).
    """
    current = text.strip()
    used = set()
    current_words = _word_count(current)
    for sentence in additions:
        if sentence in used:
            continue
        next_text = f"{current} {sentence}".strip()
        next_words = _word_count(next_text)
        if next_words > target_max and current_words >= target_min:
            break
        if next_words <= target_max or current_words < target_min:
            current = next_text
            current_words = next_words
            used.add(sentence)
        if current_words >= target_min:
            break
    return current


def _keyword_expansions(required_terms: list[str], stage: str, rng: random.Random) -> list[str]:
    """
    Generate non-stock, keyword-aware expansion sentences so we don't get the same filler every story.
    """
    terms = [t for t in required_terms if t and t.lower() not in _STOPWORDS]
    rng.shuffle(terms)
    terms = terms[:4]

    sensory = [
        "The air felt busy with tiny sounds—shuffling feet, a soft laugh, and something far away clinking.",
        "Somewhere nearby, something smelled delicious, and it made the moment feel even bigger.",
        "The room seemed to hold its breath, like it was waiting to see what would happen next.",
        "A silly idea wobbled into everyone’s thoughts, and suddenly it felt hard not to grin.",
    ]
    action = []
    for t in terms:
        action.append(f"Everything seemed connected to {t}, as if {t} was the clue hiding in plain sight.")
        action.append(f"Someone pointed at {t} and said it was probably the reason for the whole mix-up.")
        action.append(f"The thought of {t} made everyone react at once—gasping, giggling, and whispering.")

    if stage == "setup":
        extra = [
            "It felt exciting and risky at the same time—like balancing a secret on the tip of a spoon.",
            "The first little choice didn’t seem huge… until it started rolling like a snowball.",
            "A tiny mistake can be loud inside your heart, even when nobody else knows yet.",
        ]
    else:
        extra = [
            "The truth finally felt lighter once it was said out loud.",
            "An honest apology changed the whole mood, like turning on a warm lamp.",
            "Once everyone worked together, the scary part turned into the funny part.",
        ]

    pool = sensory + action + extra
    rng.shuffle(pool)
    return pool


def _repair_pages(
    pages: list[StoryPage],
    narrator: str | None,
    required_terms: list[str],
    target_min: int,
    target_max: int,
) -> list[StoryPage]:
    """
    Make the story pass constraints by repairing common failure modes:
    - boilerplate removal
    - repeated sentences across pages
    - dialogue normalization (min/max)
    - word count normalization (expand if short)
    """
    rng = random.Random(abs(hash("||".join(p.text for p in pages))) % (2**32))

    # 1) remove boilerplate
    for p in pages:
        p.text = _remove_boilerplate(p.text, narrator)

    # 2) dedupe across pages
    pages = _dedupe_across_pages(pages)

    # 3) dialogue normalization (limit first, then ensure minimum)
    for p in pages:
        p.text = _limit_dialogue(p.text, _DIALOGUE_MAX)

    # Ensure at least minimum exists somewhere (inject on last page)
    total_dialogue = sum(_count_dialogue_lines(p.text) for p in pages)
    if total_dialogue < _DIALOGUE_MIN and pages:
        pages[-1].text = _ensure_min_dialogue(pages[-1].text, narrator)

    # If still too many (edge-case), strip quotes from last page
    total_dialogue = sum(_count_dialogue_lines(p.text) for p in pages)
    if total_dialogue > _DIALOGUE_MAX and pages:
        pages[-1].text = re.sub(r'["“”]', "", pages[-1].text)

    # 4) word count normalization (expand if short)
    for idx, p in enumerate(pages):
        wc = _word_count(p.text)
        if wc < target_min:
            stage = "setup" if idx == 0 else "resolution"
            additions = _keyword_expansions(required_terms, stage=stage, rng=rng)
            p.text = _expand_text_to_range(p.text, target_min, target_max, additions)

        # If still short (very short transcript), we allow slightly under-min rather than failing hard
        # but we will try to hit min as best effort.

    return pages


def _validate_story_pages(
    pages: list[StoryPage],
    anchor_spec: AnchorSpec,
    target_min: int,
    target_max: int,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    # Word count checks (soft-ish): require within range OR close if transcript is tiny.
    for i, p in enumerate(pages, start=1):
        wc = _word_count(p.text)
        if wc < target_min:
            reasons.append(f"page {i} word count {wc} below {target_min}")
        if wc > target_max:
            reasons.append(f"page {i} word count {wc} above {target_max}")

    # Dialogue check (global)
    total_dialogue = sum(_count_dialogue_lines(p.text) for p in pages)
    if total_dialogue < _DIALOGUE_MIN or total_dialogue > _DIALOGUE_MAX:
        reasons.append(f"dialogue lines counted {total_dialogue} outside {_DIALOGUE_MIN}-{_DIALOGUE_MAX}")

    # Anchor/fidelity check (global)
    combined = " ".join(p.text for p in pages)
    if anchor_spec.names:
        name_hits = _count_term_hits(combined, anchor_spec.names)
        if name_hits < 1:
            reasons.append("anchor/fidelity missing proper-name match")

    if anchor_spec.keywords:
        required_keyword_hits = min(2, len(anchor_spec.keywords))
        keyword_hits = _count_term_hits(combined, anchor_spec.keywords)
        if keyword_hits < required_keyword_hits:
            reasons.append(
                "anchor/fidelity too low: keyword hits "
                f"{keyword_hits} < {required_keyword_hits} (terms={', '.join(anchor_spec.keywords[:6])})"
            )

    if anchor_spec.creature_keywords:
        creature_hits = _count_term_hits(combined, anchor_spec.creature_keywords)
        if creature_hits < 1:
            reasons.append(
                "anchor/fidelity missing creature keyword match "
                f"(terms={', '.join(anchor_spec.creature_keywords[:4])})"
            )

    return (len(reasons) == 0, reasons)


# -----------------------------
# Local fallback (kept simple)
# -----------------------------
def _infer_topic(cleaned: str) -> str:
    if not cleaned:
        return "a small mystery"
    lowered = cleaned.lower()
    for keyword in ("pizza", "dragon", "monster", "castle", "forest", "robot", "unicorn", "dinosaur"):
        if keyword in lowered:
            return f"the {keyword}"
    words = re.findall(r"[A-Za-z']+", cleaned)
    for word in words:
        wl = word.lower()
        if wl in _STOPWORDS:
            continue
        if len(wl) > 3:
            return f"the {wl}"
    return "a small surprise"


def _snippet_from_transcript(cleaned: str, max_words: int = 18) -> str:
    if not cleaned:
        return "a surprising little secret"
    words = re.findall(r"[A-Za-z']+", cleaned)
    if not words:
        return "a surprising little secret"
    snippet = " ".join(words[:max_words])
    return snippet.rstrip(".") + "."


def _local_storybook(cleaned: str, title: str, narrator: str | None, target_pages: int) -> StoryBook:
    topic = _infer_topic(cleaned)
    name = narrator or "A young storyteller"
    snippet = _snippet_from_transcript(cleaned)

    seed = abs(hash(cleaned or topic)) % (2**32)
    rng = random.Random(seed)

    p1 = (
        f"One day, {name} had a tricky little idea about {topic}. "
        f"It started like a tiny secret: {snippet} "
        f"{name} wondered if it would matter… but the moment wobbled bigger and bigger, like a snowball rolling downhill. "
        "Soon, everyone noticed something felt off, and the worry started to tap like a drum. "
        'Someone asked, "What’s going on?" and the air suddenly felt full of questions.'
    )
    p2 = (
        f"At last, {name} took a deep breath and told the truth. "
        f'"I’m sorry," said {name}, and that brave sentence changed everything. '
        f"Together, they faced {topic} in a way that turned the scary part into the funny part. "
        "A warm apology, a clever plan, and a few giggles later, everyone felt closer than before. "
        "By the end, the lesson felt simple: honest words make hearts lighter."
    )

    anchor_spec = _extract_anchor_spec(cleaned, max_terms=_ANCHOR_MAX_TERMS)
    required_terms = _anchor_terms_for_expansion(anchor_spec, max_terms=_ANCHOR_MAX_TERMS)
    pages = [
        StoryPage(
            text=_expand_text_to_range(
                p1,
                _MIN_WORDS_PER_PAGE,
                _MAX_WORDS_PER_PAGE,
                _keyword_expansions(required_terms, "setup", rng),
            ),
            illustration_prompt=(
                f"Watercolor children's book illustration of {name} and {topic} beginning, "
                "cozy setting, playful mischief, warm light, gentle textures."
            ),
        ),
        StoryPage(
            text=_expand_text_to_range(
                p2,
                _MIN_WORDS_PER_PAGE,
                _MAX_WORDS_PER_PAGE,
                _keyword_expansions(required_terms, "resolution", rng),
            ),
            illustration_prompt=(
                f"Watercolor children's book illustration of {name} resolving {topic}, "
                "warm apology moment, funny relief, cozy happy ending, soft light."
            ),
        ),
    ]
    if target_pages != 2:
        pages = pages[:target_pages]

    story = StoryBook(
        title=title,
        subtitle="A story told out loud",
        pages=pages,
        narrator=narrator,
        cover_prompt=_build_cover_prompt(title, narrator, cleaned),
        cover_image_path=None,
    )
    return story


# -----------------------------
# Image generation + cover art
# -----------------------------
def _maybe_generate_images(story: StoryBook) -> None:
    if not _use_openai_image_mode():
        return
    client = _get_openai_client()
    if not client:
        return

    out_dir = Path("out/books/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Cover art (no text rendered; leave space for title)
    try:
        cover_prompt = (
            "Watercolor children's picture book cover illustration. "
            f"Title: {story.title!r}. "
            f"Main characters: {story.narrator or 'the child and a parent'}. "
            "Scene inspired by the story (but do not include any words or letters). "
            "Leave a clean, empty area near the top for the title to be placed later. "
            "Warm, whimsical, cinematic lighting, cozy picture-book style."
        )
        cover_resp = client.images.generate(
            model="gpt-image-1",
            prompt=cover_prompt,
            size="1024x1024",
        )
        cover_data = cover_resp.data[0] if getattr(cover_resp, "data", None) else None
        if cover_data and getattr(cover_data, "b64_json", None):
            cover_path = out_dir / f"cover_{timestamp}.png"
            cover_path.write_bytes(base64.b64decode(cover_data.b64_json))
            story.cover_image_path = str(cover_path)
    except Exception:
        # Cover art is optional; don't fail the run if it errors.
        pass

    # 2) Page images
    for index, page in enumerate(story.pages, start=1):
        prompt = (page.illustration_prompt or "").strip()
        if not prompt:
            continue

        try:
            resp = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size="1024x1024",
            )
            data = resp.data[0] if resp.data else None
            if not data:
                continue

            image_path = out_dir / f"story_{timestamp}_p{index}.png"
            if getattr(data, "b64_json", None):
                image_path.write_bytes(base64.b64decode(data.b64_json))
            elif getattr(data, "url", None):
                request.urlretrieve(data.url, image_path)
            else:
                continue

            saved_path = str(image_path)
            page.illustration_path = saved_path
            setattr(page, "image_path", saved_path)  # back-compat
        except Exception:
            continue
