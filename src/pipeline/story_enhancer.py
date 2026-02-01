"""
Environment variables:
- STORY_ENHANCE_MODE: set to "openai" to use OpenAI for story enhancement.
- STORY_IMAGE_MODE: set to "openai" to generate images with OpenAI.
- OPENAI_API_KEY: required for OpenAI modes.
- STORY_MODEL: OpenAI model for story enhancement (default: "gpt-4o-mini").
- STORY_TEMPERATURE: OpenAI sampling temperature (default: "0.8").
- STORY_TARGET_PAGES: number of pages to generate (default: "2").
- STORY_WORDS_PER_PAGE: word count target per page (default: "260").
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

_DEFAULT_MODEL = os.getenv("STORY_MODEL", "gpt-4o-mini")
_DEFAULT_TEMPERATURE = float(os.getenv("STORY_TEMPERATURE", "0.8"))
_DEFAULT_TARGET_PAGES = int(os.getenv("STORY_TARGET_PAGES", "2"))
_DEFAULT_WORDS_PER_PAGE = int(os.getenv("STORY_WORDS_PER_PAGE", "260"))
_DEFAULT_STYLE = os.getenv("STORY_STYLE", "whimsical, funny, heartwarming")
_MIN_WORDS_PER_PAGE = max(200, _DEFAULT_WORDS_PER_PAGE - 60)
_MAX_WORDS_PER_PAGE = min(320, _DEFAULT_WORDS_PER_PAGE + 60)


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


def _openai_storybook(
    cleaned: str,
    title: str,
    narrator: str | None,
    target_pages: int,
) -> StoryBook | None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI enhancement skipped: OPENAI_API_KEY not set.")
        return None
    client = _get_openai_client()
    if not client:
        print("OpenAI enhancement skipped: openai package not available.")
        return None

    print("OpenAI package available: yes")
    use_responses = hasattr(client, "responses")
    endpoint = "responses" if use_responses else "chat.completions"
    print(f"OpenAI endpoint: {endpoint}")
    print(f"OpenAI model: {_DEFAULT_MODEL}")

    system_prompt = (
        "You are a children's picture-book author and editor. "
        "Expand the transcript into a vivid, whimsical two-page story. "
        "Return ONLY strict JSON that follows the schema exactly."
    )
    user_prompt = (
        "TRANSCRIPT:\n"
        f"{cleaned or 'Empty transcript.'}\n\n"
        f"NARRATOR: {narrator or 'none'}\n\n"
        "Story requirements:\n"
        "- Two pages only.\n"
        "- Page 1: setup + mischief + rising trouble + page-turn hook.\n"
        "- Page 2: big moment + apology + funny resolution + warm ending.\n"
        "- 1–3 short lines of dialogue across the story.\n"
        "- Kid-safe, whimsical, funny, creative, sensory details.\n"
        "- Keep names consistent; use narrator if provided.\n"
        f"- About {_MIN_WORDS_PER_PAGE}–{_MAX_WORDS_PER_PAGE} words per page.\n"
        "- Each page must include a strong illustration_prompt in consistent picture-book style.\n\n"
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
        f"Style: {_DEFAULT_STYLE}."
    )

    for attempt in range(2):
        correction = ""
        if attempt == 1:
            correction = (
                "Fix the previous response to match the JSON schema exactly and meet all requirements."
            )
        try:
            content = _request_openai_story(
                client,
                system_prompt,
                user_prompt,
                correction,
                use_responses,
                target_pages,
            )
        except Exception as exc:
            print(f"OpenAI enhancement failed: {exc}")
            return None

        if not content:
            print("OpenAI enhancement failed: empty response.")
            return None

        data = _parse_json(content)
        valid, reasons = _validate_story_data(data, target_pages)
        if not valid:
            if attempt == 0:
                user_prompt = f"{user_prompt}\n\nPrevious response:\n{content}\n\nIssues:\n- " + "\n- ".join(reasons)
                continue
            print(f"OpenAI enhancement failed: invalid output ({'; '.join(reasons)}).")
            return None

        pages = _build_pages_from_data(data, target_pages)
        if not pages:
            print("OpenAI enhancement failed: unable to parse pages.")
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
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "storybook",
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        return _extract_response_text(response)

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
        "required": ["title", "subtitle", "pages"],
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


def _local_storybook(
    cleaned: str,
    title: str,
    narrator: str | None,
    target_pages: int,
) -> StoryBook:
    topic = _infer_topic(cleaned)
    name = narrator or "A young storyteller"
    snippet = _snippet_from_transcript(cleaned)

    page1 = (
        f"On a bright afternoon, {name} felt a fizz of curiosity about {topic}. "
        f"The day began with a giggle and a whisper of adventure, and soon a playful idea popped up: {snippet}. "
        "At first it seemed small and sneaky, like hiding a secret in a pocket, but the air started to buzz. "
        f"\"Should we tell?\" someone asked, and {name} just grinned, watching the trouble wobble bigger. "
        "A silly misunderstanding bounced from friend to friend, and even the pets looked suspicious. "
        "By the time the sun slid behind a cloud, the mix-up had become a swirl of whispers and wide eyes."
    )
    page1 = _expand_text_to_range(
        page1,
        _MIN_WORDS_PER_PAGE,
        _MAX_WORDS_PER_PAGE,
        _page_expansions(topic, name, stage="setup"),
    )

    page2 = (
        "With a deep breath and a brave heart, the truth finally tumbled out. "
        f"\"I did it,\" {name} said, and the room froze for one tiny second before melting into relieved smiles. "
        f"Together they turned the problem into a plan, transforming {topic} into a funny, shared solution. "
        "There was an apology that sounded like a warm hug and a giggle that sounded like bells. "
        "Everyone joined in to fix the mix-up, and the story swung toward a cozy, happy ending. "
        "By bedtime, the adventure felt like a secret handshake—silly, sweet, and sure to be remembered."
    )
    page2 = _expand_text_to_range(
        page2,
        _MIN_WORDS_PER_PAGE,
        _MAX_WORDS_PER_PAGE,
        _page_expansions(topic, name, stage="resolution"),
    )

    pages = [
        StoryPage(
            text=page1,
            illustration_prompt=(
                f"A whimsical storybook scene of {name} discovering {topic}, "
                "with bright colors, cozy surroundings, gentle watercolor textures, and a playful hint of mischief."
            ),
        ),
        StoryPage(
            text=page2,
            illustration_prompt=(
                f"A warm, joyful illustration of friends making amends around {topic}, "
                "smiling and laughing in a cozy setting with gentle, friendly colors and soft light."
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
        "A breeze bumped the curtains, as if the room itself was leaning in to listen.",
        "Someone whispered a guess, and another friend gasped, suddenly certain they knew the truth.",
        "The air felt fizzy, like soda bubbles popping with every new idea.",
        "A tiny pet or plush toy seemed to watch the drama with wide, button eyes.",
        "Even the clock sounded excited, ticking a little faster than usual.",
    ]

    topic_lower = topic.lower()
    if "pizza" in topic_lower:
        additions.extend(
            [
                "The scent of warm cheese drifted by, making every tummy rumble a little louder.",
                "A pretend pizza monster was blamed, complete with silly growls and dramatic arm waves.",
                "Someone suggested leaving a trail of crust crumbs as a clue.",
            ]
        )
    if "monster" in topic_lower:
        additions.extend(
            [
                "Shadows wiggled on the wall, perfect for a make-believe monster dance.",
                "They drew a friendly monster map with sparkly stickers and bold arrows.",
            ]
        )
    if "dragon" in topic_lower:
        additions.extend(
            [
                "A dragon made of chalk doodles puffed pretend smoke across the sidewalk.",
                "They practiced brave knight poses and gentle dragon bows.",
            ]
        )

    if stage == "resolution":
        additions.extend(
            [
                "A chorus of \"It's okay!\" floated through the room, light and sincere.",
                "They made a new rule: big feelings get big hugs and honest words.",
                "Someone proposed a celebratory dance, and the floor became a stage.",
            ]
        )
    else:
        additions.extend(
            [
                "A whispered plan formed, and everyone leaned in close to hear it.",
                "The mix-up began to wobble like a wiggly jelly, getting bigger with each retelling.",
                "They tiptoed around the problem, hoping it would magically shrink.",
            ]
        )

    return additions


def _expand_text_to_range(
    text: str,
    target_min: int,
    target_max: int,
    additions: list[str],
) -> str:
    used: set[str] = set()
    current = text
    current_words = _word_count(current)
    for sentence in additions:
        if sentence in used:
            continue
        next_text = f"{current} {sentence}"
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


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


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
