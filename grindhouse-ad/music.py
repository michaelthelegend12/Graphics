"""Grindhouse ad music: 128 BPM house-ish loop, 15 s (32 beats), synthesised with numpy + scipy.
Run: python3 music.py  -> music.wav (stereo 44.1 kHz)"""
import numpy as np, wave
from scipy import signal

SR = 44100
BPM = 128
B = 60 / BPM
DUR = 32 * B  # exactly 15.0 s
N = int(SR * DUR)
rng = np.random.default_rng(5)
buf = np.zeros((N, 2))   # drums + fx (not ducked)
mus = np.zeros((N, 2))   # bass + chords + arp (sidechained to the kick)

def add(dst, t0, x, g=1.0, pan=0.0):
    i = int(t0 * SR)
    if i >= N: return
    j = min(N, i + len(x))
    seg = x[: j - i]
    dst[i:j, 0] += seg * g * (1 - max(0, pan))
    dst[i:j, 1] += seg * g * (1 + min(0, pan))

def tt(n): return np.arange(n) / SR
def sos_f(kind, fc, x, order=4):
    return signal.sosfilt(signal.butter(order, fc, kind, fs=SR, output="sos"), x)
def noise(n): return rng.standard_normal(n)

def kick(big=False):
    n = int(SR * (0.9 if big else 0.45)); t = tt(n)
    f = 42 + 130 * np.exp(-t / 0.032)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (0.5 if big else 0.22))
    y += noise(n) * np.exp(-t / 0.004) * 0.25
    return y

def clap():
    n = int(SR * 0.3); t = tt(n)
    e = sum(np.exp(-np.clip(t - d, 0, None) / 0.07) * (t >= d) for d in (0, 0.011, 0.022))
    return sos_f("bandpass", [1100, 7500], noise(n)) * e * 0.9

def hat(open_=False):
    n = int(SR * (0.2 if open_ else 0.06)); t = tt(n)
    return sos_f("highpass", 7000, noise(n)) * np.exp(-t / (0.09 if open_ else 0.018))

def crash():
    n = int(SR * 1.4); t = tt(n)
    return sos_f("highpass", 4500, noise(n)) * np.exp(-t / 0.5) * 0.6

def whoosh(dur=0.3, up=True):
    n = int(SR * dur); t = tt(n) / dur
    e = (t ** 2 if up else (1 - t) ** 2) * np.minimum(1, (1 - t) * 8 + 0.0)
    return sos_f("highpass", 1800, noise(n)) * e * 0.6

def saw(f, n, detune=0.0):
    t = tt(n)
    return signal.sawtooth(2 * np.pi * (f * (1 + detune)) * t)

def midi(m): return 440 * 2 ** ((m - 69) / 12)

# ---- drums / fx
for b in range(32):
    t0 = b * B
    add(buf, t0, kick(big=(b == 0 or b == 22 or b == 28)), 0.95)
    if b % 4 in (1, 3) and b >= 2: add(buf, t0, clap(), 0.55)
    add(buf, t0 + B / 2, hat(), 0.28)
    if b >= 18:
        add(buf, t0 + B / 4, hat(), 0.12); add(buf, t0 + 3 * B / 4, hat(), 0.12)
# snare roll into the price reveal (beats 20 -> 22)
for k in range(8): add(buf, 20 * B + k * B / 4, clap(), 0.25 + 0.05 * k)
for k in range(8): add(buf, 21 * B + k * B / 8, clap(), 0.5 + 0.06 * k)
# scene-cut whooshes + crashes
for cb in (2, 6, 12, 18, 22, 28):
    add(buf, cb * B - 0.22, whoosh(0.22, True), 0.5)
    add(buf, cb * B, whoosh(0.35, False), 0.35)
for cb in (0, 22, 28): add(buf, cb * B, crash(), 0.55)
# riser (beats 20 -> 22)
n = int(2 * B * SR); t = tt(n); r = t / t[-1]
sweep = np.sin(2 * np.pi * np.cumsum(300 * (6 ** r)) / SR) * r ** 2 * 0.25
add(buf, 20 * B, sweep + sos_f("highpass", 2500, noise(n)) * r ** 2 * 0.35, 0.8)
# coin ping on the price pop
n = int(0.7 * SR); t = tt(n)
ping = (np.sin(2 * np.pi * 1318 * t) + 0.6 * np.sin(2 * np.pi * 1760 * t) + 0.3 * np.sin(2 * np.pi * 2637 * t)) * np.exp(-t / 0.18)
add(buf, 22 * B + 0.12, ping, 0.22)
# extra impact when the CTA slams in and on the final hit
add(buf, 31 * B, kick(True), 0.9); add(buf, 31 * B, crash(), 0.7)

# ---- bass / stabs / arp (Am F C G x2)
roots = [45, 41, 48, 43] * 2           # A2 F2 C3 G2
chords = [[57, 60, 64], [53, 57, 60], [60, 64, 67], [55, 59, 62]] * 2
bass_pat = [0, None, 0, 12, None, 0, 12, 7]
for bar in range(8):
    for s, off in enumerate(bass_pat):
        if off is None: continue
        t0 = bar * 4 * B + s * B / 2
        n = int(B / 2 * 0.95 * SR); f = midi(roots[bar] + off - 12 + 12)
        y = sos_f("lowpass", 650, saw(f, n)) * np.exp(-tt(n) / 0.22)
        y += np.sin(2 * np.pi * f * tt(n)) * 0.6
        add(mus, t0, y, 0.42)
    # offbeat stabs
    for s in (2, 3, 6):
        t0 = bar * 4 * B + s * B / 2
        n = int(0.22 * SR); y = np.zeros(n)
        for m in chords[bar]:
            for d, p in ((-0.006, -0.5), (0.006, 0.5)):
                y += saw(midi(m + 12), n, d)
        y = sos_f("lowpass", 3200, y) * np.exp(-tt(n) / 0.13)
        add(mus, t0, y, 0.05)
    # 16th arp (from beat 6, off during the price riser quiet bit is fine)
    if bar * 4 >= 6 or bar >= 2:
        notes = chords[bar] + [chords[bar][0] + 12]
        for s in range(16):
            t0 = bar * 4 * B + s * B / 4
            if t0 < 6 * B: continue
            m = notes[[0, 1, 2, 3, 2, 1, 2, 3][s % 8]] + 12
            n = int(0.13 * SR)
            y = sos_f("lowpass", 4200, saw(midi(m), n)) * np.exp(-tt(n) / 0.05)
            add(mus, t0, y, 0.07, pan=0.35 if s % 2 else -0.35)
# final held chord
n = int(B * 1.0 * SR); y = np.zeros(n)
for m in (57, 60, 64, 69):
    for d in (-0.007, 0.007): y += saw(midi(m), n, d)
add(mus, 31 * B, sos_f("lowpass", 3800, y) * np.exp(-tt(n) / 0.5), 0.07)

# sidechain duck keyed to the beat
tg = tt(N); duck = 1 - 0.72 * np.exp(-(tg % B) / 0.11)
mix = buf + mus * duck[:, None]
mix = np.tanh(1.25 * mix) / np.tanh(1.25)
mix *= 0.89 / np.max(np.abs(mix))
fade = np.clip((DUR - tg) / 0.25, 0, 1); mix *= fade[:, None]
pcm = (mix * 32767).astype("<i2")
with wave.open("music.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print("music.wav", DUR, "s")
