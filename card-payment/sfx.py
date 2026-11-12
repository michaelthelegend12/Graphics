"""Synthesizes the subtle sound effects for the card payment explainer.

    python3 sfx.py            -> sfx.wav (48 kHz stereo, 15 s)

Every sound is generated from sine waves and filtered noise, so there are no
sample files and no licensing to worry about. Event times match the timeline
in index.html (the T object).
"""
import wave
import numpy as np
from scipy.signal import butter, sosfilt

SR, DUR = 48000, 15.0
N = int(SR * DUR)
L = np.zeros(N)
R = np.zeros(N)
rng = np.random.default_rng(3)


def env(n, a, d, curve=4.0):
    """attack (s) then exponential-ish decay over the rest of n samples"""
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(a, 1e-4))
    rel = np.clip((t - a) / max(d, 1e-4), 0, 1)
    return e * (1 - rel) ** curve


def place(sig, at, gain=1.0, pan=0.0):
    i = int(at * SR)
    j = min(N, i + len(sig))
    sig = sig[: j - i] * gain
    L[i:j] += sig * np.sqrt((1 - pan) / 2) * 1.414
    R[i:j] += sig * np.sqrt((1 + pan) / 2) * 1.414


def tone(freq, length, a=0.004, d=None, harm=(1, 0.25, 0.08)):
    n = int(length * SR)
    t = np.arange(n) / SR
    s = sum(h * np.sin(2 * np.pi * freq * (k + 1) * t) for k, h in enumerate(harm))
    return s * env(n, a, d or length, 3.0)


def noise(length):
    return rng.standard_normal(int(length * SR))


def band(sig, lo, hi):
    return sosfilt(butter(2, [lo, hi], btype="band", fs=SR, output="sos"), sig)


def lowp(sig, f):
    return sosfilt(butter(2, f, btype="low", fs=SR, output="sos"), sig)


def whoosh(length, lo=300, hi=2400, peak=0.55):
    """filtered noise swell whose band sweeps upward, peaking at `peak` (0..1)"""
    n = int(length * SR)
    src = noise(length)
    out = np.zeros(n)
    chunks = 24
    for c in range(chunks):
        a, b = c * n // chunks, (c + 1) * n // chunks
        f = lo * (hi / lo) ** (c / chunks)
        out[a:b] = band(src[max(0, a - 2000):b], f * 0.7, min(f * 1.6, SR / 2 - 100))[-(b - a):]
    t = np.linspace(0, 1, n)
    shape = np.where(t < peak, (t / peak) ** 2, ((1 - t) / (1 - peak)) ** 2)
    return out * shape


# --- ambient bed: very quiet low pad so the clip never sits in dead silence
t = np.arange(N) / SR
pad = (np.sin(2 * np.pi * 110 * t) + 0.6 * np.sin(2 * np.pi * 164.8 * t) + 0.35 * np.sin(2 * np.pi * 220.4 * t))
pad *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.18 * t)
pad *= np.clip(t / 1.5, 0, 1) * np.clip((DUR - t) / 1.2, 0, 1)
pad = lowp(pad, 500)
L += pad * 0.018
R += pad * 0.018

# --- opening: soft swell as the camera pushes in
place(whoosh(1.4, 200, 1400, 0.6), 0.05, 0.05)

# --- card taps the terminal (T.tap = 2.3): tick + the classic contactless beep
place(lowp(noise(0.02), 3000) * env(int(0.02 * SR), 0.001, 0.02), 2.28, 0.25, -0.1)
place(tone(1568, 0.16, 0.006, harm=(1, 0.15)), 2.31, 0.20, -0.1)

# --- processing ticks
for k, at in enumerate([2.75, 2.95, 3.15]):
    place(tone(2600, 0.03, 0.001, harm=(1,)), at, 0.05, -0.15)

# --- request leaves for the bank (T.out0 3.4 -> T.out1 6.0)
place(whoosh(2.6, 250, 2200, 0.45), 3.3, 0.09, 0.0)
place(tone(880, 0.12, 0.003), 3.42, 0.08, -0.2)
# arrives: soft low knock
place(tone(110, 0.35, 0.003, harm=(1, 0.4, 0.1)), 5.95, 0.22, 0.25)

# --- bank scan shimmer
for i, at in enumerate([6.25, 7.4]):
    n = int(1.4 * SR)
    sh = band(noise(1.4), 4000, 9000) * np.sin(np.linspace(0, np.pi, n)) ** 2
    place(sh, at, 0.025, 0.3)

# --- three checks tick (T.chk) rising C6, E6, G6
for at, f in zip([7.0, 7.7, 8.4], [1046.5, 1318.5, 1568.0]):
    place(tone(f, 0.28, 0.003, harm=(1, 0.3, 0.05)), at, 0.14, 0.3)

# --- response heads back (T.back0 9.4 -> T.back1 11.6)
place(whoosh(2.3, 2200, 300, 0.5), 9.3, 0.08, 0.0)

# --- approved chime (T.ok 11.6): bell-ish E6 + B6, then a sparkle
place(tone(1318.5, 1.2, 0.004, 1.2, harm=(1, 0.35, 0.12, 0.05)), 11.6, 0.17, -0.1)
place(tone(1975.5, 1.1, 0.004, 1.1, harm=(1, 0.3, 0.08)), 11.72, 0.13, -0.1)
for i in range(7):
    at = 11.75 + i * 0.07 + rng.random() * 0.03
    place(tone(3000 + rng.random() * 2500, 0.12, 0.001, harm=(1,)), at, 0.035, rng.uniform(-0.6, 0.6))
# receipt printer: short buzzy zip
n = int(1.1 * SR)
zp = band(noise(1.1), 1500, 5000) * (0.6 + 0.4 * np.sign(np.sin(2 * np.pi * 38 * np.arange(n) / SR)))
zp *= np.clip(np.arange(n) / (0.05 * SR), 0, 1) * np.clip((n - np.arange(n)) / (0.15 * SR), 0, 1)
place(zp, 12.0, 0.03, -0.25)

# --- pull out to the wide shot
place(whoosh(1.8, 1600, 200, 0.4), 12.9, 0.06, 0.0)

# --- master: gentle limiter + normalize to about -3 dBFS peak
mix = np.stack([L, R], 1)
mix = np.tanh(mix * 1.2) / 1.2
mix *= 10 ** (-3 / 20) / np.max(np.abs(mix))
fade = np.clip((N - np.arange(N)) / (0.4 * SR), 0, 1)[:, None]
mix *= fade
pcm = (mix * 32767).astype(np.int16)
with wave.open("sfx.wav", "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote sfx.wav")
