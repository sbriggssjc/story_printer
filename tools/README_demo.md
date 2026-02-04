# Story Printer Demo Runner

## PowerShell (Windows)

Run from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\demo.ps1
```

Custom output naming and print:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\demo.ps1 -Name "Claire" -Title "Pizza Monster" -Print
```

Pass a specific audio file:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\demo.ps1 -AudioPath .\out\audio\my_clip.wav
```

Defaults:
- If `-AudioPath` is omitted, the most recent `./out/audio/*.wav` file is used.
- `-Open` defaults to enabled (open the generated PDF).
- `-NoImages` disables image generation.
- `-Name` and `-Title` default to `Claire` and `Story`.

## Python (cross-platform)

```bash
python tools/demo.py
```

Custom output naming and print (Windows only):

```bash
python tools/demo.py --name "Claire" --title "Pizza Monster" --print
```

Pass a specific audio file:

```bash
python tools/demo.py --audio ./out/audio/my_clip.wav
```

Defaults:
- If `--audio` is omitted, the most recent `./out/audio/*.wav` file is used.
- The PDF is opened automatically unless `--no-open` is set.
- `--no-images` disables image generation.
- `--name` and `--title` default to `Claire` and `Story`.

## Golden path settings

Both scripts set the following defaults before running:

- `STORY_ENHANCE_MODE="openai"`
- `STORY_IMAGE_MODE="openai"` (or `"none"` when images are disabled)
- `STORY_TARGET_PAGES="2"`
- `STORY_WORDS_PER_PAGE="260"`
- `STORY_VOICE_MODE="kid"`
- `STORY_FIDELITY_MODE="fun"`
