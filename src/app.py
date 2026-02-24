import os
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so "from src..." imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import msvcrt

from src.pipeline.orchestrator import run_once
from src.pipeline.transcriber import transcribe_audio
from src.io.audio_windows import Recorder, get_default_input_info, find_input_device
from src.pipeline.constraints import MAX_SECONDS

# Read device index from env, or auto-detect by name, or fall back to None (system default)
_explicit = os.getenv("INPUT_DEVICE_INDEX")
if _explicit is not None:
    INPUT_DEVICE_INDEX = int(_explicit)
else:
    _hint = os.getenv("INPUT_DEVICE_NAME")
    INPUT_DEVICE_INDEX = find_input_device(_hint) if _hint else None

BANNER = r'''
=========================================
   STORY PRINTER (DESKTOP SIMULATOR)
-----------------------------------------
SPACE  : start/stop recording and make book
ESC    : quit
CTRL+C : quit
=========================================
'''

def safe_stop_stream(rec: Recorder):
    try:
        s = getattr(rec, "_stream", None)
        if s is not None:
            s.stop()
            s.close()
            rec._stream = None
    except Exception:
        pass

def main():
    print("RUNNING FILE:", __file__)
    print(BANNER)

    try:
        info = get_default_input_info()
        print("Default input device info:", info)
    except Exception as e:
        print("Default input device: (unable to query)", repr(e))

    if INPUT_DEVICE_INDEX is not None:
        print(f"Using mic device index: {INPUT_DEVICE_INDEX}")
    else:
        print("Using system default mic (set INPUT_DEVICE_INDEX or INPUT_DEVICE_NAME in .env)")

    rec = Recorder(device=INPUT_DEVICE_INDEX, samplerate=None, channels=1)
    recording = False
    print("Ready. Press SPACE to start recording...")

    try:
        while True:
            # Non-blocking keyboard polling so Ctrl+C works reliably
            if msvcrt.kbhit():
                ch = msvcrt.getwch()  # wide-char; returns a string like ' ' or '\x1b'

                if ch == "\x03":
                    raise KeyboardInterrupt

                # ESC
                if ch == '\x1b':
                    if recording:
                        safe_stop_stream(rec)
                    print("Exiting.")
                    return

                # SPACE toggles record
                if ch == ' ':
                    if not recording:
                        try:
                            meta = rec.start()
                            recording = True
                            msg = (
                                "Recording on device=" + str(meta.get("device")) +
                                " sr=" + str(meta.get("samplerate")) +
                                "Hz ch=" + str(meta.get("channels")) +
                                "... Press SPACE to stop (cap " + str(MAX_SECONDS) + "s)."
                            )
                            print(msg)
                        except Exception as e:
                            print("Could not start recording:", repr(e))
                            print("Tip: set INPUT_DEVICE_INDEX in .env to a different input device index.")
                        continue

                    # STOP recording + save
                    recording = False
                    try:
                        wav_path, stats = rec.stop_and_save(max_seconds=MAX_SECONDS)
                        print("Saved audio:", str(wav_path.resolve()))
                        print("   audio stats:", stats)
                    except Exception as e:
                        print("Audio save failed:", repr(e))
                        print("Press SPACE to try recording again.")
                        continue

                    try:
                        transcript = transcribe_audio(wav_path)
                    except Exception as e:
                        print("Transcription failed:", repr(e))
                        print("Press SPACE to try recording again.")
                        continue

                    if not transcript:
                        transcript = "(No transcript captured.)"
                    try:
                        out_pdf = run_once(transcript=transcript)
                        print("Book created:", str(out_pdf.resolve()))
                    except Exception as e:
                        print("Book creation failed:", repr(e))

                    print("Ready. Press SPACE to start recording...")

            time.sleep(0.02)  # yield so KeyboardInterrupt is handled

    except KeyboardInterrupt:
        if recording:
            safe_stop_stream(rec)
        print("\nExiting.")
        return

if __name__ == "__main__":
    main()
