from __future__ import annotations

from pathlib import Path

_model = None


def _get_model():
    """Load the Whisper model once and cache it for all future calls."""
    global _model
    if _model is not None:
        return _model

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
        print("  Whisper: using CUDA (GPU)")
        _model = WhisperModel("base", device="cuda", compute_type="float16")
    else:
        print("  Whisper: using CPU")
        _model = WhisperModel("base", device="cpu", compute_type="int8")

    return _model


def transcribe_audio(wav_path: Path) -> str:
    """
    Transcribe a WAV file to text.
    Uses faster-whisper (local, offline). Model is loaded once and cached.
    """
    wav_path = Path(wav_path)
    model = _get_model()

    segments, info = model.transcribe(
        str(wav_path),
        language="en",
        vad_filter=True,
        beam_size=2
    )

    text_parts = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            text_parts.append(t)

    transcript = " ".join(text_parts).strip()
    return transcript
