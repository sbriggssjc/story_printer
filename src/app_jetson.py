"""
Story Printer — Jetson Orin Nano entry point.

Hardware: Jetson Orin Nano, Rode II mic, mini monitor, USB printer, Anker power brick.
Usage:   python -m src.app_jetson
         Press SPACE to start recording, SPACE again to stop and generate + print.
         Press Q or ESC to quit.

Environment:
    INPUT_DEVICE_NAME  — substring to match mic (default: "rode", also tries "default")
    INPUT_DEVICE_INDEX — explicit device index (overrides name search)
    PRINTER_NAME       — CUPS printer name (default: system default)
    AUTO_PRINT         — set to "1" to auto-print after generation (default: "1")
    STORY_ENHANCE_MODE — "openai" for GPT story expansion (default: "openai")
    STORY_IMAGE_MODE   — "openai" for DALL-E illustrations (default: "openai")
    OPENAI_API_KEY     — required for OpenAI features
"""

from __future__ import annotations

import os
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


def _get_key() -> str:
    """Read a single keypress from stdin (Linux). Returns the character."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def _detect_mic() -> int | None:
    """Find the Rode II mic (or fall back to any input device)."""
    # Explicit index from env overrides everything
    explicit = os.getenv("INPUT_DEVICE_INDEX")
    if explicit is not None:
        return int(explicit)

    # Search by name hint
    name_hint = os.getenv("INPUT_DEVICE_NAME", "rode")
    idx = find_input_device(name_hint)
    if idx is not None:
        return idx

    # Try common fallbacks
    for fallback in ["usb", "mic", "audio"]:
        idx = find_input_device(fallback)
        if idx is not None:
            return idx

    # Let sounddevice pick the system default
    return None


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


def _status(msg: str) -> None:
    """Print a large status line visible on the mini monitor."""
    print(f"\n{'=' * 44}")
    print(f"  {msg}")
    print(f"{'=' * 44}\n")


def _safe_stop(rec: Recorder) -> None:
    try:
        s = getattr(rec, "_stream", None)
        if s is not None:
            s.stop()
            s.close()
            rec._stream = None
    except Exception:
        pass


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

    rec = Recorder(device=dev_index, samplerate=None, channels=1)
    recording = False

    print("\n  Ready! Press SPACE to start recording...\n")

    try:
        while True:
            ch = _get_key()

            # Quit keys
            if ch in ('\x1b', 'q', 'Q', '\x03'):
                if recording:
                    _safe_stop(rec)
                _status("GOODBYE!")
                return 0

            # SPACE toggles recording
            if ch == ' ':
                if not recording:
                    # START recording
                    try:
                        meta = rec.start()
                        recording = True
                        _status("RECORDING... Tell your story!")
                        print(f"  (max {MAX_SECONDS}s, press SPACE to stop)")
                    except Exception as e:
                        print(f"  Could not start recording: {e}")
                        print("  Check mic connection. Press SPACE to retry.")
                    continue

                # STOP recording and run the pipeline
                recording = False
                _status("SAVING AUDIO...")
                try:
                    wav_path, stats = rec.stop_and_save(max_seconds=MAX_SECONDS)
                    print(f"  Audio saved: {wav_path}")
                    print(f"  Peak: {stats['peak']:.4f}  RMS: {stats['rms']:.4f}")
                except Exception as e:
                    print(f"  Audio save failed: {e}")
                    print("  Press SPACE to try again.")
                    continue

                # Transcribe
                _status("LISTENING TO YOUR STORY...")
                try:
                    transcript = transcribe_audio(wav_path)
                except Exception as e:
                    print(f"  Transcription failed: {e}")
                    print("  Press SPACE to try again.")
                    continue

                if not transcript or not transcript.strip():
                    print("  No speech detected. Try speaking louder.")
                    print("  Press SPACE to try again.")
                    continue

                print(f"  Heard: \"{transcript[:100]}{'...' if len(transcript) > 100 else ''}\"")

                # Generate story + PDF
                _status("MAKING YOUR STORYBOOK...")
                try:
                    out_pdf = run_once(transcript=transcript)
                    print(f"  Book created: {out_pdf}")
                except Exception as e:
                    print(f"  Book creation failed: {e}")
                    print("  Press SPACE to try again.")
                    continue

                # Auto-print
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

                print("\n  Press SPACE to make another story, or Q to quit.\n")

    except KeyboardInterrupt:
        if recording:
            _safe_stop(rec)
        _status("GOODBYE!")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
