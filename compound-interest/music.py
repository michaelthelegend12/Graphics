#!/usr/bin/env python3
"""Original friendly ukulele-and-marimba track for the compound-interest explainer.

numpy only. Writes audio.wav (20 s, 44.1 kHz stereo). Sound effects are timed from timeline.py,
so they land on the same frames as the picture.
"""
import sys
import wave

import numpy as np

import timeline as TL

SR = 44100
N = int(TL.DUR * SR)
rng = np.random.default_rng(3)
L = np.zeros(N)
Rt = np.zeros(N)

BPM = 96
BEAT = 60.0 / BPM
BAR = 4 * BEAT                      # 2.5 s -> 8 bars = 20 s


def mf(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def add(sig, t0, gain=1.0, pan=0.0):
    i = int(t0 * SR)
    if i < 0 or i >= N:
        return
    n = min(len(sig), N - i)
    a = (pan + 1) * np.pi / 4
    L[i:i + n] += sig[:n] * gain * np.cos(a)
    Rt[i:i + n] += sig[:n] * gain * np.sin(a)


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def pluck(f, dur=1.1):
    t = tt(dur)
    s = sum(a * np.sin(2 * np.pi * f * k * t + ph) for k, a, ph in ((1, 1, 0), (2, .42, .3), (3, .2, .7), (4, .09, 1.1)))
    s = s * np.exp(-t * 6.0) * np.minimum(1, t / 0.003)
    s += rng.standard_normal(len(t)) * np.exp(-t * 90) * 0.10
    return s


def marimba(f, dur=0.9):
    t = tt(dur)
    s = np.sin(2 * np.pi * f * t) + 0.30 * np.sin(2 * np.pi * 4 * f * t) * np.exp(-t * 18)
    return s * np.exp(-t * 5.0) * np.minimum(1, t / 0.004)


def bell(f, dur=1.4):
    t = tt(dur)
    s = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 6) + 0.2 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t * 12)
    return s * np.exp(-t * 3.6) * np.minimum(1, t / 0.003)


def bass(f, dur=0.8):
    t = tt(dur)
    s = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)
    return s * np.exp(-t * 3.2) * np.minimum(1, t / 0.01)


def shaker(dur=0.09):
    t = tt(dur)
    n = np.diff(rng.standard_normal(len(t) + 1))
    return n * np.exp(-t * 45)


def thud(f0, dur=0.25):
    t = tt(dur)
    f = f0 * (1 + 1.2 * np.exp(-t * 30))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 16)


def glide(f0, f1, dur):
    t = tt(dur)
    f = np.linspace(f0, f1, len(t))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / dur) ** 0.7


CHORDS = [[60, 64, 67], [57, 60, 64], [53, 57, 60], [55, 59, 62], [60, 64, 67], [57, 60, 64], [53, 57, 60], [60, 64, 67]]
ROOTS = [48, 45, 41, 43, 48, 45, 41, 48]
MEL = {
    1: [(1, 69), (2, 72), (3, 76)],
    2: [(0, 77), (1.5, 76), (2, 72), (3, 69)],
    3: [(0, 74), (1, 71), (2, 74), (3, 79)],
    4: [(0, 76), (1.5, 72), (2, 76), (3, 79)],
    5: [(0, 81), (1, 79), (2, 76), (3, 72)],
    6: [(0, 77), (1, 81), (2, 79), (3, 77)],
    7: [(0, 84), (1.5, 79), (2.5, 76)],
}
PENTA = [79, 81, 84, 86, 88, 91, 93, 96, 98, 100]


def main(out='audio.wav'):
    # --- band
    for b in range(8):
        t0 = b * BAR
        ch = CHORDS[b]
        pat = [0, 1, 2, 1, 0, 1, 2, 1]
        for k, ix in enumerate(pat):
            add(pluck(mf(ch[ix])), t0 + k * BEAT / 2, 0.15 if k % 2 == 0 else 0.11, pan=-0.25 if k % 2 == 0 else 0.25)
        add(bass(mf(ROOTS[b])), t0, 0.32)
        add(bass(mf(ROOTS[b])), t0 + 2 * BEAT, 0.26)
        if b >= 1:
            for k in range(8):
                if k % 2 == 1:
                    add(shaker(), t0 + k * BEAT / 2, 0.05, pan=0.3)
        for beat, m in MEL.get(b, []):
            dur_gain = 0.2 if not (b == 7 and beat == 0) else 0.0
            add(marimba(mf(m)), t0 + beat * BEAT, dur_gain, pan=0.1)
    add(bell(mf(84), 2.4), 7 * BAR, 0.16)          # closing chime
    add(bell(mf(88), 2.4), 7 * BAR + 0.02, 0.10)

    # --- sound effects on the picture's timeline
    for p, g in ((1 / 2.75, 0.5), (2 / 2.75, 0.3), (2.5 / 2.75, 0.16), (1.0, 0.09)):
        add(thud(150 - 40 * (1 - g)), TL.HERO_DROP + p, g)
    add(glide(280, 620, 0.18), TL.TIP_T, 0.16)
    add(thud(170), TL.TIP_T + TL.TIP_D, 0.3)
    for i, ta in enumerate(TL.coin_times()):
        add(bell(mf(PENTA[i % len(PENTA)]), 1.0), ta + 0.3, 0.16, pan=-0.2)
        add(thud(260, 0.12), ta + 0.3, 0.10)
    for Y in range(1, 11):
        add(glide(620, 980, 0.10), TL.tick_time(Y), 0.10, pan=0.3)
    add(glide(260, 900, 0.40), TL.CANOPY_T, 0.18)
    for j, m in enumerate((79, 84, 88, 91)):
        add(bell(mf(m), 1.4), TL.CANOPY_T + 0.30 + 0.07 * j, 0.14, pan=(j - 1.5) * 0.3)
    for j, m in enumerate((84, 88, 91, 96, 100)):
        add(bell(mf(m), 1.6), TL.END_T + 0.1 + 0.09 * j, 0.15, pan=(j - 2) * 0.25)

    # --- warmth: two soft echoes
    for d, g in ((0.19, 0.22), (0.34, 0.12)):
        k = int(d * SR)
        l0, r0 = L.copy(), Rt.copy()
        L[k:] += r0[:-k] * g
        Rt[k:] += l0[:-k] * g

    peak = max(np.abs(L).max(), np.abs(Rt).max())
    L[:] *= 0.85 / peak
    Rt[:] *= 0.85 / peak
    fade = np.ones(N)
    fi = int(0.05 * SR)
    fade[:fi] = np.linspace(0, 1, fi)
    fo = int(1.6 * SR)
    fade[-fo:] = np.linspace(1, 0, fo) ** 1.5
    st = np.stack([L * fade, Rt * fade], axis=1)
    pcm = (np.clip(st, -1, 1) * 32767).astype('<i2')
    with wave.open(out, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'audio.wav')
