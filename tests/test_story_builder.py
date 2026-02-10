"""Tests for src/pipeline/story_builder.py — transcript cleaning, chunking, title inference."""

import pytest
from src.pipeline.story_builder import (
    StoryPage,
    StoryBook,
    _clean_transcript,
    _find_name,
    _infer_title,
    _chunk_transcript,
    _split_into_sentences,
    build_storybook,
)


# ---------------------------------------------------------------------------
# _clean_transcript
# ---------------------------------------------------------------------------
class TestCleanTranscript:
    def test_empty(self):
        assert _clean_transcript("") == ""
        assert _clean_transcript(None) == ""

    def test_removes_noise_tags(self):
        result = _clean_transcript("Hello [inaudible] world")
        assert "[inaudible]" not in result
        assert "Hello" in result

    def test_collapses_whitespace(self):
        result = _clean_transcript("Hello    world   today")
        assert "    " not in result

    def test_fixes_punctuation_spacing(self):
        result = _clean_transcript("Hello ,world")
        assert result.startswith("Hello, world") or ", world" in result

    def test_adds_trailing_period(self):
        result = _clean_transcript("This is a sentence without ending punctuation")
        assert result.endswith(".")

    def test_preserves_existing_punctuation(self):
        result = _clean_transcript("Is this a question?")
        assert result.endswith("?")

    def test_deduplicates_punctuation(self):
        result = _clean_transcript("Wow!!!")
        assert "!!!" not in result


# ---------------------------------------------------------------------------
# _find_name
# ---------------------------------------------------------------------------
class TestFindName:
    def test_finds_capitalized_name(self):
        assert _find_name("One day Claire went to the store") == "Claire"

    def test_skips_stopwords(self):
        # "Once" and "Then" are stopwords (capitalized at sentence start)
        assert _find_name("Once upon a time") is None

    def test_returns_none_for_empty(self):
        assert _find_name("") is None
        assert _find_name(None) is None

    def test_finds_first_name(self):
        result = _find_name("Jake and Emma went outside")
        assert result == "Jake"


# ---------------------------------------------------------------------------
# _infer_title
# ---------------------------------------------------------------------------
class TestInferTitle:
    def test_empty_gives_default(self):
        assert _infer_title("") == "My Story"

    def test_keyword_pizza(self):
        assert "Pizza" in _infer_title("I ate some pizza today")

    def test_keyword_dragon(self):
        assert "Dragon" in _infer_title("There was a dragon in the cave")

    def test_keyword_monster(self):
        assert "Monster" in _infer_title("A big monster appeared")

    def test_falls_back_to_name(self):
        result = _infer_title("Maxwell went to the park and played")
        assert "Maxwell" in result

    def test_falls_back_to_keyword(self):
        result = _infer_title("there was a wonderful adventure in the magical forest")
        assert result != "My Story"  # should pick something


# ---------------------------------------------------------------------------
# _split_into_sentences
# ---------------------------------------------------------------------------
class TestSplitSentences:
    def test_basic_split(self):
        result = _split_into_sentences("Hello. World. Goodbye.")
        assert len(result) == 3

    def test_empty(self):
        assert _split_into_sentences("") == []

    def test_preserves_content(self):
        result = _split_into_sentences("First sentence. Second sentence.")
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence."


# ---------------------------------------------------------------------------
# _chunk_transcript
# ---------------------------------------------------------------------------
class TestChunkTranscript:
    def test_short_text_single_chunk(self):
        result = _chunk_transcript("This is short.")
        assert len(result) >= 1
        assert "This is short." in result[0]

    def test_long_text_multiple_chunks(self):
        long_text = "This is a sentence. " * 60  # ~1200 chars
        result = _chunk_transcript(long_text)
        assert len(result) >= 2

    def test_respects_max_chars(self):
        long_text = "This is a test sentence. " * 60
        result = _chunk_transcript(long_text, max_chars=900)
        for chunk in result:
            # Allow some slack (sentence boundaries)
            assert len(chunk) <= 1200


# ---------------------------------------------------------------------------
# StoryPage dataclass
# ---------------------------------------------------------------------------
class TestStoryPage:
    def test_basic_creation(self):
        page = StoryPage(text="Hello", illustration_prompt="A scene")
        assert page.text == "Hello"
        assert page.illustration_prompt == "A scene"

    def test_image_path_sync(self):
        page = StoryPage(text="Hi", illustration_prompt="x", image_path="/tmp/test.png")
        assert page.get_image_path() == "/tmp/test.png"
        assert page.illustration_path == "/tmp/test.png"

    def test_illustration_path_sync(self):
        page = StoryPage(text="Hi", illustration_prompt="x", illustration_path="/tmp/test.png")
        assert page.get_image_path() == "/tmp/test.png"
        assert page.image_path == "/tmp/test.png"


# ---------------------------------------------------------------------------
# build_storybook
# ---------------------------------------------------------------------------
class TestBuildStorybook:
    def test_basic_build(self):
        book = build_storybook("Claire went to the park and found a dragon.")
        assert isinstance(book, StoryBook)
        assert book.title
        assert len(book.pages) >= 1
        assert book.pages[0].text

    def test_with_pages(self):
        book = build_storybook("Anything", pages=["Page one text", "Page two text"])
        assert len(book.pages) == 2
        assert book.pages[0].text == "Page one text"

    def test_empty_transcript_raises(self):
        with pytest.raises(ValueError, match="Empty transcript"):
            build_storybook("")

    def test_narrator_extracted(self):
        book = build_storybook("Maxwell found a secret cave behind the waterfall.")
        assert book.narrator == "Maxwell"
