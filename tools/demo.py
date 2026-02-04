#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.pipeline.orchestrator import run_once_from_audio


def find_latest_audio() -> Path:
    audio_dir = Path("out") / "audio"
    candidates = sorted(audio_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No .wav files found in ./out/audio. Provide --audio.")
    return candidates[0]


def open_pdf(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
        return
    subprocess.run(["xdg-open", str(path)], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Story Printer demo.")
    parser.add_argument("--audio", type=Path, help="Path to a .wav file")
    parser.add_argument("--no-open", action="store_true", help="Do not open the generated PDF")
    parser.add_argument("--no-images", action="store_true", help="Disable image generation")
    args = parser.parse_args()

    audio_path = args.audio or find_latest_audio()

    os.environ["STORY_ENHANCE_MODE"] = "openai"
    os.environ["STORY_IMAGE_MODE"] = "none" if args.no_images else "openai"
    os.environ["STORY_TARGET_PAGES"] = "2"
    os.environ["STORY_WORDS_PER_PAGE"] = "260"
    os.environ["STORY_VOICE_MODE"] = "kid"
    os.environ["STORY_FIDELITY_MODE"] = "fun"

    out_path = run_once_from_audio(str(audio_path))
    print(out_path)

    if not args.no_open:
        open_pdf(Path(out_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
