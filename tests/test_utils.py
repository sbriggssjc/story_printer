"""Tests for utility modules: text_limits, constraints, book_builder path logic."""

import tempfile
from pathlib import Path

from src.pipeline.text_limits import clamp_words
from src.pipeline.constraints import MAX_SECONDS, MAX_WORDS_2P
from src.pipeline.book_builder import make_book_pdf, _PROJECT_ROOT


class TestClampWords:
    def test_under_limit(self):
        assert clamp_words("Hello world", 10) == "Hello world"

    def test_at_limit(self):
        text = "one two three"
        assert clamp_words(text, 3) == text

    def test_over_limit(self):
        text = "one two three four five"
        result = clamp_words(text, 3)
        assert result == "one two three ..."

    def test_empty(self):
        assert clamp_words("", 10) == ""
        assert clamp_words(None, 10) == ""


class TestConstraints:
    def test_max_seconds_reasonable(self):
        assert 30 <= MAX_SECONDS <= 300  # between 30s and 5 min

    def test_max_words_reasonable(self):
        assert 200 <= MAX_WORDS_2P <= 1000


class TestBookBuilderPath:
    def test_project_root_is_absolute(self):
        assert _PROJECT_ROOT.is_absolute()

    def test_default_out_dir(self):
        expected = _PROJECT_ROOT / "out" / "books"
        # The default param in make_book_pdf should be this path
        import inspect
        sig = inspect.signature(make_book_pdf)
        default_out_dir = sig.parameters["out_dir"].default
        assert default_out_dir == expected

    def test_make_book_pdf_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            result = make_book_pdf(
                title="Test",
                subtitle="A test",
                pages=["Hello world."],
                out_dir=out_dir,
            )
            assert result.exists()
            assert result.suffix == ".pdf"
            assert result.parent == out_dir
