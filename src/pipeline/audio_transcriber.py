from __future__ import annotations

import importlib
import os
from pathlib import Path


def transcribe_audio(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if importlib.util.find_spec("openai") is None:
        raise RuntimeError("openai package not installed")

    from openai import OpenAI

    client = OpenAI()

    with p.open("rb") as f:
        r = client.audio.transcriptions.create(
            model=os.getenv("STORY_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"),
            file=f,
        )

    return getattr(r, "text", None) or (r.get("text") if isinstance(r, dict) else "") or ""
