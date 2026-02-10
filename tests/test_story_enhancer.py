"""Tests for src/pipeline/story_enhancer.py — local fallback, validation, repair logic."""

import os
import pytest

from src.pipeline.story_builder import StoryPage, StoryBook
from src.pipeline.story_enhancer import (
    _clean_transcript_for_story,
    _truncate_after_the_end,
    _ensure_watercolor_prompt,
    _extract_proper_names,
    _build_anchor_list,
    _anchor_coverage,
    _anchor_required_count,
    _word_count,
    _count_dialogue_lines,
    _limit_dialogue,
    _ensure_min_dialogue,
    _remove_boilerplate,
    _dedupe_across_pages,
    _strip_generic_filler,
    _validate_story_pages,
    _local_storybook,
    _parse_json,
    _build_pages_from_data,
    enhance_to_storybook,
    _normalize_anchor_terms,
    _WATERCOLOR_PROMPT_PREFIX,
)


# ---------------------------------------------------------------------------
# Transcript cleaning
# ---------------------------------------------------------------------------
class TestCleanTranscriptForStory:
    def test_strips_story_prefix(self):
        result = _clean_transcript_for_story("This is a story about a cat")
        assert not result.lower().startswith("this is a story about")
        assert "cat" in result

    def test_strips_story_suffix(self):
        result = _clean_transcript_for_story("The cat ran away and it's the end of the story")
        assert "end of the story" not in result.lower()

    def test_empty(self):
        assert _clean_transcript_for_story("") == ""
        assert _clean_transcript_for_story(None) == ""


class TestTruncateAfterTheEnd:
    def test_truncates(self):
        result = _truncate_after_the_end("The cat slept. The end. And more stuff.")
        assert "more stuff" not in result
        assert "The end." in result

    def test_no_end(self):
        text = "The cat slept peacefully."
        assert _truncate_after_the_end(text) == text

    def test_empty(self):
        assert _truncate_after_the_end("") == ""


# ---------------------------------------------------------------------------
# Watercolor prompt
# ---------------------------------------------------------------------------
class TestEnsureWatercolorPrompt:
    def test_adds_prefix(self):
        result = _ensure_watercolor_prompt("A cat sitting on a chair.")
        assert result.startswith(_WATERCOLOR_PROMPT_PREFIX)
        assert "cat sitting" in result

    def test_already_has_prefix(self):
        full = _WATERCOLOR_PROMPT_PREFIX + "A dog."
        assert _ensure_watercolor_prompt(full) == full

    def test_empty(self):
        result = _ensure_watercolor_prompt("")
        assert result.startswith("Watercolor")


# ---------------------------------------------------------------------------
# Name / anchor extraction
# ---------------------------------------------------------------------------
class TestExtractProperNames:
    def test_finds_names(self):
        result = _extract_proper_names("Claire went to find Maxwell", None)
        assert "Claire" in result
        assert "Maxwell" in result

    def test_narrator_first(self):
        result = _extract_proper_names("Maxwell went home", "Claire")
        assert result[0] == "Claire"

    def test_skips_stopwords(self):
        result = _extract_proper_names("Then after that", None)
        assert len(result) == 0


class TestBuildAnchorList:
    def test_includes_names(self):
        anchors = _build_anchor_list("Claire found a pizza monster", "Claire")
        lower = [a.lower() for a in anchors]
        assert "claire" in lower

    def test_empty_transcript(self):
        anchors = _build_anchor_list("", None)
        assert isinstance(anchors, list)


class TestNormalizeAnchorTerms:
    def test_deduplicates(self):
        result = _normalize_anchor_terms(["pizza", "Pizza", "PIZZA"])
        assert len(result) == 1

    def test_removes_stopwords(self):
        result = _normalize_anchor_terms(["the", "and", "pizza"])
        assert "pizza" in result
        assert "the" not in result

    def test_empty(self):
        assert _normalize_anchor_terms([]) == []
        assert _normalize_anchor_terms(None) == []


# ---------------------------------------------------------------------------
# Anchor coverage
# ---------------------------------------------------------------------------
class TestAnchorCoverage:
    def test_all_present(self):
        ok, summary, missing = _anchor_coverage(["Claire found pizza"], ["claire", "pizza"])
        assert ok is True
        assert len(missing) == 0

    def test_missing_anchors(self):
        ok, summary, missing = _anchor_coverage(
            ["A simple story"],
            ["claire", "pizza", "monster", "dragon", "castle"],
        )
        assert ok is False
        assert len(missing) > 0

    def test_no_anchors(self):
        ok, summary, missing = _anchor_coverage(["Any text"], [])
        assert ok is True


class TestAnchorRequiredCount:
    def test_zero(self):
        assert _anchor_required_count(0) == 0

    def test_small(self):
        assert _anchor_required_count(3) >= 2

    def test_caps_at_four(self):
        assert _anchor_required_count(100) <= 4


# ---------------------------------------------------------------------------
# Word count and dialogue
# ---------------------------------------------------------------------------
class TestWordCount:
    def test_basic(self):
        assert _word_count("Hello world") == 2

    def test_empty(self):
        assert _word_count("") == 0

    def test_with_punctuation(self):
        assert _word_count("Hello, world! How's it going?") == 5


class TestCountDialogueLines:
    def test_straight_quotes(self):
        assert _count_dialogue_lines('She said "hello" and "goodbye"') == 2

    def test_curly_quotes(self):
        assert _count_dialogue_lines('She said \u201chello\u201d') == 1

    def test_no_quotes(self):
        assert _count_dialogue_lines("No quotes here.") == 0


class TestLimitDialogue:
    def test_under_limit_unchanged(self):
        text = 'She said "hello" and left.'
        assert _limit_dialogue(text, 3) == text

    def test_over_limit_removes_extras(self):
        text = 'She said "one" then "two" then "three" then "four"'
        result = _limit_dialogue(text, 2)
        # Should still have "one" and "two" quoted, but "three"/"four" unquoted
        assert result.count('"') <= 4  # at most 2 quoted pairs


class TestEnsureMinDialogue:
    def test_adds_dialogue_when_missing(self):
        text = "The cat sat on the mat."
        result = _ensure_min_dialogue(text, "Claire")
        assert '"' in result or '\u201c' in result

    def test_keeps_existing(self):
        text = 'The cat said "meow" to the dog.'
        result = _ensure_min_dialogue(text, "Claire")
        assert "meow" in result


# ---------------------------------------------------------------------------
# Repair utilities
# ---------------------------------------------------------------------------
class TestRemoveBoilerplate:
    def test_removes_known_boilerplate(self):
        bp = "Sunlight puddled on the floor like warm butter, making the room glow."
        text = f"The cat ran. {bp} The dog barked."
        result = _remove_boilerplate(text, None)
        assert bp not in result
        assert "cat ran" in result

    def test_keeps_normal_text(self):
        text = "A perfectly normal sentence. Another one."
        assert _remove_boilerplate(text, None) == text


class TestDedupeAcrossPages:
    def test_removes_duplicates(self):
        pages = [
            StoryPage(text="The cat ran. The dog barked.", illustration_prompt=""),
            StoryPage(text="The dog barked. The bird sang.", illustration_prompt=""),
        ]
        result = _dedupe_across_pages(pages)
        # "The dog barked." should only appear once (on page 1)
        all_text = " ".join(p.text for p in result)
        assert all_text.count("The dog barked.") == 1


class TestStripGenericFiller:
    def test_removes_filler(self):
        text = "The cat ran. I heard tiny sounds all around me. The dog barked."
        result = _strip_generic_filler(text)
        assert "tiny sounds" not in result
        assert "cat ran" in result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidateStoryPages:
    def test_valid_pages(self):
        # Need to include dialogue (1-3 lines total) to pass validation
        text1 = " ".join(["word"] * 195) + ' She said "hello" to everyone.'
        text2 = " ".join(["word"] * 198)
        pages = [
            StoryPage(text=text1, illustration_prompt="A scene"),
            StoryPage(text=text2, illustration_prompt="Another scene"),
        ]
        ok, reasons = _validate_story_pages(pages, target_min=170, target_max=340)
        assert ok is True, f"Validation failed: {reasons}"

    def test_too_short(self):
        pages = [
            StoryPage(text="Too short.", illustration_prompt=""),
        ]
        ok, reasons = _validate_story_pages(pages, target_min=170, target_max=340)
        assert ok is False
        assert any("below" in r for r in reasons)

    def test_too_long(self):
        pages = [
            StoryPage(text=" ".join(["word"] * 400), illustration_prompt=""),
        ]
        ok, reasons = _validate_story_pages(pages, target_min=170, target_max=340)
        assert ok is False
        assert any("above" in r for r in reasons)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
class TestParseJson:
    def test_valid_json(self):
        result = _parse_json('{"title": "Test"}')
        assert result == {"title": "Test"}

    def test_wrapped_in_backticks(self):
        result = _parse_json('```json\n{"title": "Test"}\n```')
        assert result == {"title": "Test"}

    def test_invalid_json(self):
        result = _parse_json("not json at all")
        assert result is None

    def test_json_with_surrounding_text(self):
        result = _parse_json('Here is the story: {"title": "Hello"} end')
        assert result == {"title": "Hello"}


class TestBuildPagesFromData:
    def test_builds_pages(self):
        data = {
            "pages": [
                {"text": "Page one.", "illustration_prompt": "A cat."},
                {"text": "Page two.", "illustration_prompt": "A dog."},
            ]
        }
        pages = _build_pages_from_data(data, target_pages=2)
        assert len(pages) == 2
        assert pages[0].text == "Page one."

    def test_skips_empty_pages(self):
        data = {
            "pages": [
                {"text": "", "illustration_prompt": "A cat."},
                {"text": "Page two.", "illustration_prompt": "A dog."},
            ]
        }
        pages = _build_pages_from_data(data, target_pages=2)
        assert len(pages) == 1  # first page skipped (empty text)

    def test_no_pages_key(self):
        pages = _build_pages_from_data({}, target_pages=2)
        assert pages == []


# ---------------------------------------------------------------------------
# Local storybook fallback (end-to-end, no API)
# ---------------------------------------------------------------------------
class TestLocalStorybook:
    def test_produces_valid_book(self):
        book = _local_storybook(
            cleaned="Claire hid the pizza crust and a pizza monster came",
            title="The Pizza Mystery",
            narrator="Claire",
            display_name="Claire",
            target_pages=2,
        )
        assert isinstance(book, StoryBook)
        assert book.title == "The Pizza Mystery"
        assert len(book.pages) == 2
        for page in book.pages:
            assert len(page.text) > 0
            assert page.illustration_prompt.startswith("Watercolor")

    def test_single_page(self):
        book = _local_storybook(
            cleaned="A dragon appeared",
            title="Dragon Day",
            narrator=None,
            display_name=None,
            target_pages=1,
        )
        assert len(book.pages) == 1


class TestEnhanceToStorybookLocal:
    def test_without_openai(self):
        """With no OPENAI_API_KEY, should fall back to local mode."""
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        old_mode = os.environ.pop("STORY_ENHANCE_MODE", None)
        try:
            book = enhance_to_storybook("Jake found a robot in the attic")
            assert isinstance(book, StoryBook)
            assert len(book.pages) >= 1
            assert book.title
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key
            if old_mode is not None:
                os.environ["STORY_ENHANCE_MODE"] = old_mode
