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

    # Auto-detect CUDA (Jetson Orin Nano has GPU); fall back to CPU.
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if has_cuda:
        model = WhisperModel("base", device="cuda", compute_type="float16")
    else:
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
