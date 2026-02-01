from pathlib import Path
from datetime import datetime
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf

OUT_AUDIO = Path('out') / 'audio'

class Recorder:
    def __init__(self, device=None, samplerate=None, channels=1):
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self._q = queue.Queue()
        self._stream = None
        self._frames = []
        self._printed = 0

    def _callback(self, indata, frames, time, status):
        if status:
            pass
        chunk = indata.copy()
        self._q.put(chunk)
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        self._printed += 1
        if self._printed % 10 == 0:
            print(f'   level(rms)={rms:.6f}', end='\\r')

    def start(self):
        self._frames = []
        self._printed = 0

        if self.device is None:
            info = sd.query_devices(kind='input')
            dev_index = info['index']
            dev_sr = int(info['default_samplerate'])
            max_in = int(info.get('max_input_channels', 1))
        else:
            dinfo = sd.query_devices(self.device)
            dev_index = self.device
            dev_sr = int(dinfo['default_samplerate'])
            max_in = int(dinfo.get('max_input_channels', 1))

        sr = int(self.samplerate) if self.samplerate else dev_sr
        ch = min(int(self.channels), max_in) if max_in > 0 else 1

        self._stream = sd.InputStream(
            device=dev_index,
            samplerate=sr,
            channels=ch,
            dtype='float32',
            callback=self._callback
        )
        self._stream.start()

        # IMPORTANT: remember actual device used
        self.device = dev_index
        self.samplerate = sr
        self.channels = ch
        return {'device': dev_index, 'samplerate': sr, 'channels': ch}

    def stop_and_save(self, max_seconds: int = 75):
        total_samples_limit = int(self.samplerate * max_seconds)
        collected = 0
        while not self._q.empty() and collected < total_samples_limit:
            chunk = self._q.get_nowait()
            self._frames.append(chunk)
            collected += len(chunk)

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        print(' ' * 50, end='\\r')

        if not self._frames:
            raise RuntimeError('No audio captured. Check mic permissions / input device.')

        audio = np.concatenate(self._frames, axis=0)
        audio = audio[:total_samples_limit]

        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0

        OUT_AUDIO.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_wav = OUT_AUDIO / f'recording_{ts}.wav'
        sf.write(str(out_wav), audio, self.samplerate)

        stats = {'peak': peak, 'rms': rms, 'samplerate': self.samplerate, 'channels': self.channels, 'device': self.device}
        return out_wav, stats

def get_default_input_info():
    return sd.query_devices(kind='input')

