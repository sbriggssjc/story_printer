from __future__ import annotations

import importlib
import os
from pathlib import Path


def transcribe_audio(path: str) -> str:
    p = Path(os.path.expandvars(os.path.expanduser(path))).resolve()
    if not p.exists():
        cwd = Path.cwd().resolve()
        wavs = list(cwd.glob("*.wav"))
        msg = (
            f"Audio file not found.\n"
            f"  provided: {path}\n"
            f"  resolved: {p}\n"
            f"  cwd:      {cwd}\n"
            f"  wavs in cwd: {[w.name for w in wavs]}\n"
        )
        raise FileNotFoundError(msg)

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
