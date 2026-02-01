import time
import msvcrt
import sys

from src.pipeline.orchestrator import run_once
from src.pipeline.transcriber import transcribe_audio
from src.io.audio_windows import Recorder, get_default_input_info
from src.pipeline.constraints import MAX_SECONDS

# Use your working Yeti X device index
INPUT_DEVICE_INDEX = 29

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

    rec = Recorder(device=INPUT_DEVICE_INDEX, samplerate=None, channels=1)
    recording = False
    print("Ready. Press SPACE to start recording...")

    try:
        while True:
            # Non-blocking keyboard polling so Ctrl+C works reliably
            if msvcrt.kbhit():
                ch = msvcrt.getwch()  # wide-char; returns a string like ' ' or '\x1b'

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
                                "🎙️ Recording on device=" + str(meta.get("device")) +
                                " sr=" + str(meta.get("samplerate")) +
                                "Hz ch=" + str(meta.get("channels")) +
                                "... Press SPACE to stop (cap " + str(MAX_SECONDS) + "s)."
                            )
                            print(msg)
                        except Exception as e:
                            print("❌ Could not start recording:", repr(e))
                            print("Tip: set INPUT_DEVICE_INDEX to a different input device index.")
                        continue

                    # STOP recording + save
                    recording = False
                    try:
                        wav_path, stats = rec.stop_and_save(max_seconds=MAX_SECONDS)
                        print("✅ Saved audio:", str(wav_path.resolve()))
                        print("   audio stats:", stats)
                    except Exception as e:
                        print("❌ Audio save failed:", repr(e))
                        print("Press SPACE to try recording again.")
                        continue

                    try:
                        transcript = transcribe_audio(wav_path)
                    except Exception as e:
                        print("❌ Transcription failed:", repr(e))
                        print("Press SPACE to try recording again.")
                        continue

                    if not transcript:
                        transcript = "(No transcript captured.)"
                    try:
                        out_pdf = run_once(transcript=transcript)
                        print("📘 Book created:", str(out_pdf.resolve()))
                    except Exception as e:
                        print("❌ Book creation failed:", repr(e))

                    print("Ready. Press SPACE to start recording...")

            time.sleep(0.02)  # yield so KeyboardInterrupt is handled

    except KeyboardInterrupt:
        if recording:
            safe_stop_stream(rec)
        print("\nExiting.")
        return

if __name__ == "__main__":
    main()
