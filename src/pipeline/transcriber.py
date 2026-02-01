from __future__ import annotations

from pathlib import Path

def transcribe_audio(wav_path: Path) -> str:
    """
    Transcribe a WAV file to text.
    Uses faster-whisper (local, offline). Designed to run on desktop now, Jetson later.
    """
    wav_path = Path(wav_path)

    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError(
            "Missing dependency faster-whisper. Install it with: pip install faster-whisper\n"
            f"Original error: {repr(e)}"
        )

    # CPU mode for Windows desktop (fast enough for short clips).
    # On Jetson later, we'll switch device='cuda' if available.
    model = WhisperModel("base", device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        str(wav_path),
        language="en",
        vad_filter=True,          # trims silence nicely
        beam_size=2
    )

    text_parts = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            text_parts.append(t)

    transcript = " ".join(text_parts).strip()
    return transcript
