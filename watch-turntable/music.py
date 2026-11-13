#!/usr/bin/env python3
"""Soft ambient bed for the watch turntable: 10 s, 44.1 kHz stereo.

A warm detuned pad (Dmaj9-ish), a breath of air, a quiet whoosh and a glassy chime when
the text slides in (7.1 s), all through a small synthetic room reverb.

    python3 music.py audio.wav
"""
import sys
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

SR, DUR = 44100, 10.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(11)


def lp(x, hz, order=2):
    return sosfilt(butter(order, hz, 'low', fs=SR, output='sos'), x)


def bp(x, lo, hi):
    return sosfilt(butter(2, [lo, hi], 'band', fs=SR, output='sos'), x)


# ---- pad: each voice = three slightly detuned sines, slow tremolo, slow stereo drift
notes = [(73.42, 0.50), (146.83, 0.34), (220.00, 0.28), (329.63, 0.20), (369.99, 0.17), (554.37, 0.07)]
L = np.zeros(N); R = np.zeros(N)
for i, (f, a) in enumerate(notes):
    for d, pan in ((-0.0017, 0.0), (0.0, 0.5), (0.0019, 1.0)):
        ph = rng.uniform(0, 2 * np.pi)
        tr = 1 + 0.18 * np.sin(2 * np.pi * (0.07 + 0.03 * i) * t + ph)
        s = np.sin(2 * np.pi * f * (1 + d) * t + ph) * a * tr / 3
        # slow drift of each detuned layer between channels
        w = 0.5 + 0.4 * np.sin(2 * np.pi * 0.05 * t + ph + pan * 3)
        L += s * w
        R += s * (1 - w)
# gentle harmonics via soft saturation, then warm low-pass
L, R = lp(np.tanh(1.6 * L), 1800), lp(np.tanh(1.6 * R), 1800)

# ---- air: low-passed noise breathing slowly
for ch in (0, 1):
    n = bp(rng.normal(0, 1, N), 300, 2200)
    n *= 0.012 * (0.6 + 0.4 * np.sin(2 * np.pi * 0.11 * t + ch * 1.7))
    if ch == 0: L += n
    else: R += n

# ---- whoosh when the text starts sliding in (7.1 s)
ts = 7.1
g = np.exp(-((t - (ts + 0.45)) / 0.38) ** 2)
for ch in (0, 1):
    x = bp(rng.normal(0, 1, N), 700, 4200) * g * 0.035     # independent noise per channel = wide
    if ch == 0: L += x
    else: R += x

# ---- chime at the reveal: three glassy partials with long decay, panned a little apart
def chime(f, start, level, pan, decay):
    e = np.where(t >= start, np.exp(-(t - start) / decay), 0.0)
    e *= np.minimum(1, np.maximum(0, (t - start) / 0.02))
    s = np.sin(2 * np.pi * f * t) * e * level
    s += np.sin(2 * np.pi * f * 2.01 * t) * e * level * 0.12
    return s * (1 - pan), s * pan

for f, lv, pan, dc in ((739.99, 0.060, 0.40, 1.9), (880.00, 0.050, 0.55, 1.6), (1108.73, 0.035, 0.65, 1.3)):
    cl, cr = chime(f, ts + 0.12, lv, pan, dc)
    L += cl; R += cr

# ---- envelope: soft fade in, a slow swell toward the reveal, long fade out
env = np.minimum(1, t / 1.6) ** 2 * (0.82 + 0.18 * np.sin(np.pi * np.clip((t - 1) / 7.5, 0, 1)))
env *= np.minimum(1, (DUR - t) / 1.8) ** 2
L *= env; R *= env

# ---- room reverb (synthetic stereo impulse response, ~2 s tail)
ir_n = int(SR * 2.2)
tt = np.arange(ir_n) / SR
decay = np.exp(-tt / 0.55)
irL = lp(rng.normal(0, 1, ir_n), 3500) * decay
irR = lp(rng.normal(0, 1, ir_n), 3500) * decay
irL /= np.sqrt(np.sum(irL ** 2)); irR /= np.sqrt(np.sum(irR ** 2))
wetL = fftconvolve(L, irL)[:N]; wetR = fftconvolve(R, irR)[:N]
L = 0.68 * L + 0.55 * wetL
R = 0.68 * R + 0.55 * wetR

out = np.stack([L, R], 1)
out *= np.minimum(1, (DUR - t) / 0.4)[:, None]          # make sure it ends at silence
out *= 0.42 / np.max(np.abs(out))                       # about -7.5 dBFS peak, soft
wavfile.write(sys.argv[1] if len(sys.argv) > 1 else 'audio.wav', SR, (out * 32767).astype(np.int16))
