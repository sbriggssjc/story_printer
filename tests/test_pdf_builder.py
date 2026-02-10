"""Tests for src/pipeline/pdf_builder.py — PDF generation without crashing."""

import tempfile
from pathlib import Path

from src.pipeline.pdf_builder import render_story_pdf, _wrap_text, _wrap_paragraphs
from src.pipeline.story_builder import StoryPage


class TestWrapText:
    def test_empty(self):
        result = _wrap_text("", 500, "Helvetica", 12)
        assert result == [""]

    def test_short_line(self):
        result = _wrap_text("Hello world", 500, "Helvetica", 12)
        assert len(result) == 1
        assert result[0] == "Hello world"

    def test_long_line_wraps(self):
        long_text = "word " * 50
        result = _wrap_text(long_text.strip(), 200, "Helvetica", 12)
        assert len(result) > 1


class TestWrapParagraphs:
    def test_single_paragraph(self):
        result = _wrap_paragraphs("Hello world", 500, "Helvetica", 12)
        assert len(result) >= 1

    def test_multiple_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph."
        result = _wrap_paragraphs(text, 500, "Helvetica", 12)
        assert "" in result  # blank line separator

    def test_empty(self):
        result = _wrap_paragraphs("", 500, "Helvetica", 12)
        assert result == [""]


class TestRenderStoryPdf:
    def test_renders_with_string_pages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_pdf = Path(tmpdir) / "test_story.pdf"
            result = render_story_pdf(
                title="Test Story",
                subtitle="A test",
                pages=["This is page one.", "This is page two."],
                out_pdf=out_pdf,
            )
            assert result.exists()
            assert result.stat().st_size > 0
            # Basic PDF validation: starts with %PDF
            with open(result, "rb") as f:
                header = f.read(5)
            assert header == b"%PDF-"

    def test_renders_with_story_pages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_pdf = Path(tmpdir) / "test_story2.pdf"
            pages = [
                StoryPage(text="Page one text here.", illustration_prompt="A cat."),
                StoryPage(text="Page two text here.", illustration_prompt="A dog."),
            ]
            result = render_story_pdf(
                title="My Adventure",
                subtitle="Told out loud",
                pages=pages,
                out_pdf=out_pdf,
                narrator="Claire",
            )
            assert result.exists()
            assert result.stat().st_size > 500  # non-trivial PDF

    def test_renders_empty_pages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_pdf = Path(tmpdir) / "test_empty.pdf"
            result = render_story_pdf(
                title="Empty",
                subtitle="Nothing",
                pages=[],
                out_pdf=out_pdf,
            )
            assert result.exists()

    def test_renders_with_cover_and_narrator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_pdf = Path(tmpdir) / "test_cover.pdf"
            result = render_story_pdf(
                title="Dragon Day",
                subtitle="A wild ride",
                pages=["Once a dragon flew overhead."],
                out_pdf=out_pdf,
                narrator="Emma",
                cover_image_path=None,
            )
            assert result.exists()

    def test_long_text_shrinks_font(self):
        """A page with lots of text should still render (font shrinks to fit)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_pdf = Path(tmpdir) / "test_long.pdf"
            long_text = "This is a fairly long sentence for testing. " * 30
            result = render_story_pdf(
                title="Long Story",
                subtitle="Wordy",
                pages=[long_text],
                out_pdf=out_pdf,
            )
            assert result.exists()
            assert result.stat().st_size > 0

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_pdf = Path(tmpdir) / "sub" / "dir" / "story.pdf"
            result = render_story_pdf(
                title="Nested",
                subtitle="Deep",
                pages=["Hello."],
                out_pdf=out_pdf,
            )
            assert result.exists()
