#!/usr/bin/env python3
"""Original chill travel track for flight-globe, synthesised from scratch.

15 s, 84 bpm, F maj7 - E m7 - D m7 - C maj7 - F maj7. Warm pad, soft electric-piano
arpeggio, sub bass and light brushed drums, with reverb and a slow fade out.
Needs numpy and scipy.   python3 music.py [out.wav]
"""
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfilt

SR, DUR, BPM = 44100, 15.0, 84
BEAT = 60.0 / BPM
N = int(SR * DUR)
rng = np.random.RandomState(3)


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def lp(x, f, order=2):
    return sosfilt(butter(order, f, "low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, "high", fs=SR, output="sos"), x)


def put(buf, sig, start):
    i = int(start * SR)
    if i >= len(buf):
        return
    seg = sig[: len(buf) - i]
    buf[i:i + len(seg)] += seg


CHORDS = [  # (bass root, chord tones)
    (41, [53, 57, 60, 64]),   # Fmaj7
    (40, [52, 55, 59, 62]),   # Em7
    (38, [50, 53, 57, 60]),   # Dm7
    (36, [48, 52, 55, 59]),   # Cmaj7
    (41, [53, 57, 60, 64]),   # Fmaj7, rings out
]
BAR = 4 * BEAT

pad = np.zeros(N)
keys = np.zeros(N)
bass = np.zeros(N)
drums = np.zeros(N)


def pad_note(m, length):
    t = np.arange(int(length * SR)) / SR
    s = np.zeros_like(t)
    for det in (-0.07, 0.0, 0.07):                          # three slightly detuned voices
        f = hz(m) * 2 ** (det / 12)
        for k, a in ((1, 1.0), (2, 0.35), (3, 0.15)):
            s += a * np.sin(2 * np.pi * f * k * t + rng.rand() * 6.28)
    env = np.minimum(t / 0.9, 1) * np.minimum((length - t) / 1.4, 1)
    return s * env


def ep_note(m, length=1.6):
    t = np.arange(int(length * SR)) / SR
    f = hz(m)
    s = (np.sin(2 * np.pi * f * t) * np.exp(-t * 2.6)
         + 0.35 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 5.5)
         + 0.12 * np.sin(2 * np.pi * 4.02 * f * t) * np.exp(-t * 14))
    return s * np.minimum(t / 0.004, 1)


for bi, (root, tones) in enumerate(CHORDS):
    t0 = bi * BAR
    ln = BAR + 1.2 if bi < len(CHORDS) - 1 else DUR - t0 + 1.0
    for m in tones:
        put(pad, pad_note(m, ln), t0)
    # bass: root on beat 1, again on the "and" of 2
    put(bass, np.sin(2 * np.pi * hz(root) * np.arange(int(1.6 * BEAT * SR)) / SR)
        * np.exp(-np.arange(int(1.6 * BEAT * SR)) / SR * 1.6), t0)
    put(bass, np.sin(2 * np.pi * hz(root) * np.arange(int(1.2 * BEAT * SR)) / SR)
        * np.exp(-np.arange(int(1.2 * BEAT * SR)) / SR * 2.2) * 0.7, t0 + 2.5 * BEAT)
    # electric piano arpeggio, swung eighths
    order = [0, 2, 3, 2, 1, 3, 2, 1]
    for k, idx in enumerate(order):
        swing = 0.09 * BEAT if k % 2 else 0.0
        vel = 0.9 if k % 4 == 0 else 0.62
        put(keys, ep_note(tones[idx] + 12) * vel, t0 + k * BEAT / 2 + swing)
    # a soft melodic touch on the last beat of every second bar
    if bi % 2 == 1:
        put(keys, ep_note(tones[3] + 19, 2.2) * 0.5, t0 + 3 * BEAT)

# drums: soft kick on 1 and 3, brushed snare on 2 and 4, quiet swung hats
for bi in range(len(CHORDS)):
    t0 = bi * BAR
    for b in (0, 2):
        t = np.arange(int(0.35 * SR)) / SR
        k = np.sin(2 * np.pi * (46 + 70 * np.exp(-t * 28)) * t) * np.exp(-t * 11)
        put(drums, k * 0.9, t0 + b * BEAT)
    for b in (1, 3):
        n = lp(hp(rng.randn(int(0.3 * SR)), 900), 5200)
        t = np.arange(len(n)) / SR
        put(drums, n * np.exp(-t * 16) * 0.22, t0 + b * BEAT)
    for k in range(8):
        n = hp(rng.randn(int(0.08 * SR)), 6500)
        t = np.arange(len(n)) / SR
        swing = 0.09 * BEAT if k % 2 else 0.0
        put(drums, n * np.exp(-t * 55) * (0.10 if k % 2 else 0.065), t0 + k * BEAT / 2 + swing)
drums[: int(0.0 * SR)] = 0

pad = lp(pad, 1400)
keys = lp(keys, 4200)
bass = lp(bass, 220)
drums = lp(drums, 9000)

# stereo spread + reverb
def reverb(x, secs=2.2, seed=0):
    r = np.random.RandomState(seed)
    t = np.arange(int(secs * SR)) / SR
    ir = lp(r.randn(len(t)), 4500) * np.exp(-t * 2.4)
    return fftconvolve(x, ir)[: len(x)] / np.sqrt(np.sum(ir ** 2))


def delay(x, secs, fb=0.35, n=4):
    out = x.copy()
    for k in range(1, n + 1):
        d = int(secs * SR * k)
        out[d:] += x[: len(x) - d] * fb ** k
    return out


keys_d = delay(lp(keys, 2800), BEAT * 0.75, 0.38)
send = pad * 0.6 + keys_d * 0.8
wet_l, wet_r = reverb(send, seed=1), reverb(send, seed=2)

left = pad * 0.30 + keys * 0.55 + keys_d * 0.16 + bass * 0.55 + drums * 0.55 + wet_l * 0.22
right = pad * 0.30 + keys * 0.45 + keys_d * 0.22 + bass * 0.55 + drums * 0.55 + wet_r * 0.22
# slight pad width
pad_w = np.roll(pad, 380)
left += pad_w * 0.08
right += np.roll(pad, -380) * 0.08

t = np.arange(N) / SR
fade = np.minimum(t / 0.6, 1) * np.minimum((DUR - t) / 2.6, 1) ** 1.5
stereo = np.stack([left, right], axis=1) * fade[:, None]
stereo = np.tanh(stereo * 1.2) / np.tanh(1.2)               # gentle tape-style saturation
stereo = hp(stereo.T, 28).T
stereo *= 0.89 / np.max(np.abs(stereo))
wavfile.write(sys.argv[1] if len(sys.argv) > 1 else "flight-globe.wav", SR, (stereo * 32767).astype(np.int16))
print("wrote audio")
