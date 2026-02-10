"""End-to-end test: transcript → PDF, no API keys needed (local fallback)."""

import os
import tempfile
from pathlib import Path

from src.pipeline.orchestrator import run_once
from src.pipeline.story_builder import StoryBook


class TestEndToEnd:
    def test_transcript_to_pdf_local_mode(self):
        """Full pipeline: transcript → story enhancement (local) → PDF."""
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        old_mode = os.environ.pop("STORY_ENHANCE_MODE", None)
        old_image = os.environ.pop("STORY_IMAGE_MODE", None)
        try:
            transcript = "Claire hid the pizza crust under her bed and then a pizza monster came to find it!"
            pdf_path = run_once(transcript)

            assert pdf_path.exists(), f"PDF not created at {pdf_path}"
            assert pdf_path.suffix == ".pdf"
            assert pdf_path.stat().st_size > 500, "PDF is too small"

            # Verify it's a real PDF
            with open(pdf_path, "rb") as f:
                header = f.read(5)
            assert header == b"%PDF-", "Output is not a valid PDF"
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key
            if old_mode is not None:
                os.environ["STORY_ENHANCE_MODE"] = old_mode
            if old_image is not None:
                os.environ["STORY_IMAGE_MODE"] = old_image

    def test_different_transcripts_produce_different_titles(self):
        """Different stories should get different titles."""
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        old_mode = os.environ.pop("STORY_ENHANCE_MODE", None)
        try:
            from src.pipeline.story_enhancer import enhance_to_storybook

            book1 = enhance_to_storybook("A big dragon flew over the castle")
            book2 = enhance_to_storybook("The robot fixed the broken spaceship")

            # They shouldn't both be "My Story"
            assert not (book1.title == "My Story" and book2.title == "My Story")
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key
            if old_mode is not None:
                os.environ["STORY_ENHANCE_MODE"] = old_mode

    def test_with_explicit_pages(self):
        """Test the pre-built pages path."""
        pdf_path = run_once(
            transcript="A quick story",
            pages=["This is page one of the story.", "This is page two of the story."],
        )
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0
