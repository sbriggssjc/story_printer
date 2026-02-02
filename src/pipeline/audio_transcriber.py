from __future__ import annotations

import importlib.util as importlib_util
import logging
import os
from pathlib import Path

# Import the util submodule explicitly so static analyzers and packagers detect it.


def transcribe_audio(path: str) -> str:
    logger = logging.getLogger(__name__)
    raw_path = Path(os.path.expandvars(os.path.expanduser(path)))
    p = raw_path.resolve()
    if not p.exists():
        project_root = Path(__file__).resolve().parents[2]
        candidates: list[Path] = []
        if not raw_path.is_absolute():
            candidates.append((project_root / raw_path).resolve())
        candidates.append(Path.cwd().resolve() / "out" / "audio" / raw_path.name)

        for candidate in candidates:
            if candidate.exists():
                p = candidate
                break

    if not p.exists() and raw_path.parent == Path("."):
        out_audio_dir = Path.cwd().resolve() / "out" / "audio"
        exact_match = out_audio_dir / raw_path.name
        if exact_match.exists():
            p = exact_match
        else:
            wavs = sorted(
                out_audio_dir.glob("*.wav"),
                key=lambda wav: wav.stat().st_mtime,
                reverse=True,
            )
            if wavs:
                p = wavs[0]
                logger.info(
                    "Audio file not found; using most recent .wav from out/audio: %s",
                    p,
                )

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

    if importlib_util.find_spec("openai") is None:
        raise RuntimeError("openai package not installed")

    from openai import OpenAI

    client = OpenAI()

    with p.open("rb") as f:
        r = client.audio.transcriptions.create(
            model=os.getenv("STORY_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"),
            file=f,
        )

    return getattr(r, "text", None) or (r.get("text") if isinstance(r, dict) else "") or ""
