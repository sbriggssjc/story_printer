"""
Story Printer — Jetson Orin Nano entry point.

Hardware: Jetson Orin Nano, Rode II mic, mini monitor, USB printer, Anker power brick.
Usage:   python -m src.app_jetson
         Press SPACE to start recording, SPACE again to stop and generate + print.
         Recording auto-stops after MAX_SECONDS if you forget.
         Press Q or ESC to quit.

Environment:
    INPUT_DEVICE_NAME  — substring to match mic (default: "rode", also tries "default")
    INPUT_DEVICE_INDEX — explicit device index (overrides name search)
    PRINTER_NAME       — CUPS printer name (default: system default)
    AUTO_PRINT         — set to "1" to auto-print after generation (default: "1")
    STORY_ENHANCE_MODE — "openai" for GPT story expansion (default: "openai")
    STORY_IMAGE_MODE   — "openai" for DALL-E illustrations (default: "openai")
    OPENAI_API_KEY     — required for OpenAI features
    MAX_OLD_RECORDINGS — number of old WAV files to keep (default: "20")
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

# Load .env before any pipeline imports so env vars are available at module load time
from dotenv import load_dotenv
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

from src.pipeline.orchestrator import run_once
from src.pipeline.transcriber import transcribe_audio
from src.io.audio_windows import Recorder, find_input_device, get_default_input_info
from src.pipeline.constraints import MAX_SECONDS


# ---------------------------------------------------------------------------
# Non-blocking keyboard input (Linux)
# ---------------------------------------------------------------------------
def _key_available(timeout: float = 0.0) -> str | None:
    """Check for a keypress with timeout. Returns the character or None."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _wait_for_key() -> str:
    """Block until a key is pressed. Returns the character."""
    while True:
        ch = _key_available(timeout=0.5)
        if ch is not None:
            return ch


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------
def _detect_mic() -> int | None:
    """Find the Rode II mic (or fall back to any input device)."""
    explicit = os.getenv("INPUT_DEVICE_INDEX")
    if explicit is not None:
        return int(explicit)

    name_hint = os.getenv("INPUT_DEVICE_NAME", "rode")
    idx = find_input_device(name_hint)
    if idx is not None:
        return idx

    for fallback in ["usb", "mic", "audio"]:
        idx = find_input_device(fallback)
        if idx is not None:
            return idx

    return None


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def _print_pdf(pdf_path: Path) -> bool:
    """Print a PDF using CUPS (lp command). Returns True on success."""
    printer = os.getenv("PRINTER_NAME")
    cmd = ["lp"]
    if printer:
        cmd += ["-d", printer]
    cmd.append(str(pdf_path))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True
        print(f"  Print command failed: {result.stderr.strip()}")
        return False
    except FileNotFoundError:
        print("  'lp' not found. Install CUPS: sudo apt install cups")
        return False
    except Exception as e:
        print(f"  Print error: {e}")
        return False


# ---------------------------------------------------------------------------
# Disk cleanup
# ---------------------------------------------------------------------------
def _cleanup_old_files(directory: Path, pattern: str, keep: int) -> None:
    """Remove oldest files matching pattern, keeping only the newest `keep`."""
    if not directory.is_dir():
        return
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for old_file in files[keep:]:
        try:
            old_file.unlink()
        except Exception:
            pass


def _cleanup_old_recordings() -> None:
    keep = int(os.getenv("MAX_OLD_RECORDINGS", "20"))
    audio_dir = _PROJECT_ROOT / "out" / "audio"
    _cleanup_old_files(audio_dir, "*.wav", keep)

    images_dir = _PROJECT_ROOT / "out" / "books" / "images"
    _cleanup_old_files(images_dir, "*.png", keep * 3)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def _status(msg: str) -> None:
    """Print a large status line visible on the mini monitor."""
    print(f"\n{'=' * 44}")
    print(f"  {msg}")
    print(f"{'=' * 44}\n", flush=True)


def _safe_stop(rec: Recorder) -> None:
    try:
        s = getattr(rec, "_stream", None)
        if s is not None:
            s.stop()
            s.close()
            rec._stream = None
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Recording with auto-stop
# ---------------------------------------------------------------------------
def _record_with_autostop(rec: Recorder) -> tuple[Path, dict] | None:
    """
    Record audio. Auto-stops after MAX_SECONDS or when SPACE is pressed.
    Returns (wav_path, stats) or None on failure.
    """
    try:
        meta = rec.start()
    except Exception as e:
        print(f"  Could not start recording: {e}")
        print("  Check mic connection. Press SPACE to retry.")
        return None

    _status("RECORDING... Tell your story!")
    print(f"  (auto-stops after {MAX_SECONDS}s, or press SPACE to stop)")

    start_time = time.monotonic()
    while True:
        elapsed = time.monotonic() - start_time
        remaining = MAX_SECONDS - elapsed

        if remaining <= 0:
            print("\n  Time's up! Saving your story...")
            break

        # Check for keypress every 0.5s
        ch = _key_available(timeout=0.5)
        if ch == ' ':
            break
        if ch in ('\x1b', 'q', 'Q', '\x03'):
            _safe_stop(rec)
            return None

        # Show a countdown every 10 seconds
        secs = int(elapsed)
        if secs > 0 and secs % 10 == 0 and elapsed - secs < 0.5:
            print(f"  {secs}s recorded... ({int(remaining)}s remaining)", flush=True)

    _status("SAVING AUDIO...")
    try:
        wav_path, stats = rec.stop_and_save(max_seconds=MAX_SECONDS)
        print(f"  Audio saved: {wav_path.name}")
        print(f"  Peak: {stats['peak']:.4f}  RMS: {stats['rms']:.4f}")
        return wav_path, stats
    except Exception as e:
        print(f"  Audio save failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    _status("STORY PRINTER")
    print("  Press SPACE to record a story.")
    print("  Press SPACE again to stop and make the book.")
    print("  Press Q or ESC to quit.\n")

    # Set defaults for the invention convention demo
    os.environ.setdefault("STORY_ENHANCE_MODE", "openai")
    os.environ.setdefault("STORY_IMAGE_MODE", "openai")
    os.environ.setdefault("STORY_TARGET_PAGES", "2")
    os.environ.setdefault("STORY_VOICE_MODE", "kid")
    os.environ.setdefault("STORY_FIDELITY_MODE", "fun")

    auto_print = os.getenv("AUTO_PRINT", "1") == "1"

    # Detect microphone
    dev_index = _detect_mic()
    if dev_index is not None:
        print(f"  Microphone: device index {dev_index}")
    else:
        print("  Microphone: system default")

    try:
        info = get_default_input_info()
        print(f"  Default input: {info.get('name', 'unknown')}")
    except Exception:
        pass

    # Pre-warm the Whisper model on startup so first recording is fast
    print("  Loading speech recognition model...", flush=True)
    try:
        from src.pipeline.transcriber import _get_model
        _get_model()
        print("  Model ready.")
    except Exception as e:
        print(f"  Model pre-load failed (will retry later): {e}")

    rec = Recorder(device=dev_index, samplerate=None, channels=1)

    # Clean up old files from previous sessions
    _cleanup_old_recordings()

    print("\n  Ready! Press SPACE to start recording...\n")

    try:
        while True:
            ch = _wait_for_key()

            # Quit keys
            if ch in ('\x1b', 'q', 'Q', '\x03'):
                _status("GOODBYE!")
                return 0

            # SPACE starts the full pipeline
            if ch != ' ':
                continue

            # --- RECORD ---
            result = _record_with_autostop(rec)
            if result is None:
                print("  Press SPACE to try again.\n")
                continue
            wav_path, stats = result

            # --- TRANSCRIBE ---
            _status("LISTENING TO YOUR STORY...")
            try:
                transcript = transcribe_audio(wav_path)
            except Exception as e:
                print(f"  Transcription failed: {e}")
                print("  Press SPACE to try again.\n")
                continue

            if not transcript or not transcript.strip():
                print("  No speech detected. Try speaking louder next time!")
                print("  Press SPACE to try again.\n")
                continue

            print(f"  Heard: \"{transcript[:120]}{'...' if len(transcript) > 120 else ''}\"")

            # --- GENERATE STORY + PDF ---
            _status("MAKING YOUR STORYBOOK...")
            print("  (writing story + drawing pictures, this takes a minute)")
            try:
                out_pdf = run_once(transcript=transcript)
                print(f"  Book created: {out_pdf.name}")
            except Exception as e:
                print(f"  Book creation failed: {e}")
                print("  Press SPACE to try again.\n")
                continue

            # --- PRINT ---
            if auto_print:
                _status("PRINTING YOUR STORY!")
                if _print_pdf(out_pdf):
                    _status("ALL DONE! Your story is printing!")
                else:
                    _status("DONE! (Printing failed -- check printer)")
                    print(f"  PDF is at: {out_pdf}")
            else:
                _status("ALL DONE!")
                print(f"  PDF is at: {out_pdf}")

            # Cleanup after each run
            _cleanup_old_recordings()

            print("\n  Press SPACE to make another story, or Q to quit.\n")

    except KeyboardInterrupt:
        _status("GOODBYE!")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
