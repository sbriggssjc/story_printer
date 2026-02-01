import time
import numpy as np
import sounddevice as sd

SECONDS = 2.0

def probe(device_index):
    d = sd.query_devices(device_index)
    if d.get('max_input_channels', 0) <= 0:
        return None

    sr = int(d.get('default_samplerate', 44100))
    ch = 1
    name = d.get('name')
    print('[' + str(device_index) + '] ' + str(name) + ' (sr=' + str(sr) + ') -> recording ' + str(SECONDS) + 's...')
    try:
        audio = sd.rec(int(SECONDS * sr), samplerate=sr, channels=ch, device=device_index, dtype='float32')
        sd.wait()
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
        return (peak, rms)
    except Exception as e:
        return ('ERR', repr(e))

def main():
    devs = sd.query_devices()
    results = []
    for i, d in enumerate(devs):
        if d.get('max_input_channels', 0) <= 0:
            continue
        r = probe(i)
        if r is None:
            continue
        results.append((i, d.get('name'), r))
        time.sleep(0.1)

    print('')
    print('=== RESULTS (sorted by PEAK desc) ===')
    clean = []
    for i, name, r in results:
        if isinstance(r, tuple) and len(r) == 2 and r[0] != 'ERR':
            clean.append((i, name, r[0], r[1]))
        else:
            print(str(i) + ' ' + str(name) + ' ' + str(r))

    clean.sort(key=lambda x: x[2], reverse=True)
    for i, name, peak, rms in clean:
        print(str(i) + ': peak=' + format(peak, '.6f') + ' rms=' + format(rms, '.6f') + '  name=' + str(name))

    if len(clean) > 0:
        print('')
        print('Best guess INPUT_DEVICE_INDEX = ' + str(clean[0][0]))

if __name__ == '__main__':
    main()

