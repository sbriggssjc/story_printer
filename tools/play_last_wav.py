import sys
from pathlib import Path
import sounddevice as sd
import soundfile as sf

def newest_wav():
    p = Path('out') / 'audio'
    if not p.exists():
        return None
    wavs = list(p.glob('*.wav'))
    if not wavs:
        return None
    wavs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return wavs[0]

def main():
    wav = newest_wav()
    if wav is None:
        print('No wav files found in out/audio')
        return
    print('Playing:', wav)
    data, sr = sf.read(str(wav), dtype='float32')
    print('sr=', sr, 'shape=', getattr(data, 'shape', None))
    sd.play(data, sr)
    sd.wait()
    print('Done.')

if __name__ == '__main__':
    main()

