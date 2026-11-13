"""Avocado toast music + sound effects: 128 BPM upbeat pop loop, 15 s (32 beats), numpy + scipy.
Sound effects are placed from timeline.py so they land on the picture's moments.
Run: python3 music.py  -> music.wav (stereo 44.1 kHz)"""
import numpy as np, wave
from scipy import signal
from timeline import *

SR = 44100
DUR = TOTAL_BEATS * B
N = int(SR * DUR)
rng = np.random.default_rng(7)
drums = np.zeros((N, 2)); mus = np.zeros((N, 2)); fx = np.zeros((N, 2))

def add(dst, t0, x, g=1.0, pan=0.0):
    i = int(t0 * SR)
    if i >= N or i < 0: return
    j = min(N, i + len(x)); seg = x[: j - i]
    dst[i:j, 0] += seg * g * (1 - max(0, pan)); dst[i:j, 1] += seg * g * (1 + min(0, pan))

def tt(n): return np.arange(n) / SR
def filt(kind, fc, x, order=4): return signal.sosfilt(signal.butter(order, fc, kind, fs=SR, output="sos"), x)
def noise(n): return rng.standard_normal(n)
def mtof(m): return 440 * 2 ** ((m - 69) / 12)

def kick():
    n = int(SR * 0.4); t = tt(n)
    f = 45 + 120 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18) + noise(n) * np.exp(-t / 0.004) * 0.2

def clap():
    n = int(SR * 0.25); t = tt(n)
    e = sum(np.exp(-np.clip(t - d, 0, None) / 0.06) * (t >= d) for d in (0, 0.011, 0.022))
    return filt("bandpass", [1200, 7000], noise(n)) * e * 0.8

def hat(o=False):
    n = int(SR * (0.18 if o else 0.05)); t = tt(n)
    return filt("highpass", 7500, noise(n)) * np.exp(-t / (0.06 if o else 0.012)) * 0.35

def pluck(m, dur=0.25, bright=1.0):
    n = int(SR * dur); t = tt(n); f = mtof(m)
    y = sum(np.sin(2 * np.pi * f * k * t) * a for k, a in ((1, 1), (2, 0.5 * bright), (3, 0.25 * bright), (4, 0.12 * bright)))
    return y * np.exp(-t / (dur * 0.35)) * (1 - np.exp(-t / 0.002))

def bass(m, dur=0.22):
    n = int(SR * dur); t = tt(n); f = mtof(m)
    y = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t)
    return np.tanh(y * 1.3) * np.exp(-t / (dur * 0.6)) * (1 - np.exp(-t / 0.003))

def chord_stab(ms, dur=0.2):
    return sum(pluck(m, dur, 0.6) for m in ms) / len(ms)

def whoosh(dur=0.45, up=True):
    n = int(SR * dur); t = tt(n); u = t / dur
    fc = 400 + 5000 * (u if up else 1 - u)
    x = noise(n); out = np.zeros(n); blk = 512
    for i in range(0, n, blk):
        f = min(fc[i], 18000)
        out[i:i + blk] = signal.sosfilt(signal.butter(2, [max(200, f * 0.6), f * 1.2], "bandpass", fs=SR, output="sos"), x[i:i + blk])
    return out * np.sin(np.pi * u) ** 1.5 * 0.5

def blip(f0, f1, dur=0.18, g=1.0):
    n = int(SR * dur); t = tt(n)
    f = f0 * (f1 / f0) ** (t / dur)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (dur * 0.4)) * g

def bell(f=2093, dur=1.0):
    n = int(SR * dur); t = tt(n)
    return sum(np.sin(2 * np.pi * f * r * t) * a * np.exp(-t / (dur * d)) for r, a, d in ((1, 1, 0.5), (2.76, 0.4, 0.25), (5.4, 0.2, 0.12)))

def tick(dur=0.03):
    n = int(SR * dur); t = tt(n)
    return filt("bandpass", [2500, 9000], noise(n)) * np.exp(-t / 0.008)

bt = lambda beat: beat * B

# ------------------------------------------------ music: C - G - Am - F, repeated (8 bars)
prog_roots = [48, 43, 45, 41]   # C2 G1 A1 F1 (bass roots, MIDI)
chords = [[60, 64, 67], [59, 62, 67], [60, 64, 69], [60, 65, 69]]
arp = [[72, 76, 79, 84], [71, 74, 79, 83], [72, 76, 81, 84], [72, 77, 81, 84]]
for bar in range(8):
    ci = bar % 4; t0 = bar * 4
    for b in range(4):
        add(drums, bt(t0 + b), kick(), 0.9)
        if b in (1, 3): add(drums, bt(t0 + b), clap(), 0.55)
        add(drums, bt(t0 + b + 0.5), hat(), 0.5)
        if b == 3: add(drums, bt(t0 + b + 0.5), hat(True), 0.4)
    for e in range(8):  # eighth-note bounce bass
        add(mus, bt(t0 + e * 0.5), bass(prog_roots[ci] + (12 if e % 4 == 3 else 0)), 0.4)
    for b in (0.5, 1.5, 2.5, 3.0):  # offbeat chord stabs
        add(mus, bt(t0 + b), chord_stab(chords[ci]), 0.35, 0.2)
    pat = [0, 1, 2, 3, 2, 1, 2, 3]  # marimba-ish sixteenth arp, in every bar from bar 2 (bar 1 starts with just stabs)
    if bar >= 1:
        for k in range(16):
            n_ = arp[ci][(k * 3 + (k // 4)) % 4] if k % 2 == 0 else arp[ci][pat[k % 8] % 4] + 0
            if k % 2 == 0 or k % 4 == 3:
                add(mus, bt(t0 + k * 0.25), pluck(n_, 0.2, 0.8), 0.22, -0.25 if k % 4 else 0.25)

# sidechain-ish duck of music on kicks
env = np.ones(N)
for bar in range(8):
    for b in range(4):
        i = int(bt(bar * 4 + b) * SR); m = min(N, i + int(0.18 * SR))
        env[i:m] = np.minimum(env[i:m], 0.45 + 0.55 * np.linspace(0, 1, m - i) ** 1.5)
mus *= env[:, None]

# ------------------------------------------------ sound effects on the picture's beats
def pop(beat, f=520, g=0.45): add(fx, bt(beat), blip(f, f * 2.2, 0.14, g))
def slide(beat, up=True, g=0.5, d=0.45): add(fx, bt(beat), whoosh(d, up), g)
def thud(beat, g=0.7): add(fx, bt(beat), blip(220, 55, 0.3, g))

# intro
slide(T_PLATE_IN, True, 0.45, 0.5); thud(T_PLATE_IN + 1.0, 0.5)
thud(T_TITLE1 + 0.2, 0.9); thud(T_TITLE2 + 0.2, 0.9); pop(T_PILL + 0.2, 700)
slide(T_TITLE_OUT, False, 0.4, 0.4)
# each step number lands with a bounce boing + whoosh wipe
for s_ in S + [OUT]:
    add(fx, bt(s_ - 0.25), whoosh(0.5, True), 0.5)
    add(fx, bt(s_), blip(300, 900, 0.22, 0.7)); add(fx, bt(s_ + 0.25), blip(450, 1200, 0.15, 0.4))
    add(fx, bt(s_ + 0.9), blip(800, 1500, 0.1, 0.25))  # chip pop
# step 1
slide(T_BREAD_IN, False, 0.5); thud(T_BREAD_IN + 0.8, 0.6)
for k in range(8): add(fx, bt(T_TOAST_A + k * 0.2), tick(), 0.5)
add(fx, bt(T_DING), bell(2093, 1.0), 0.4); add(fx, bt(T_DING), bell(3136, 0.8), 0.2)
slide(T_PLATE_OUT1, True, 0.5, 0.5)
# step 2
slide(T_AVO_IN, False, 0.5); thud(T_AVO_IN + 0.8, 0.5)
slide(T_BOWL_IN, True, 0.45); thud(T_BOWL_IN + 0.8, 0.5)
add(fx, bt(T_PIT), blip(150, 900, 0.25, 0.8)); slide(T_PIT + 0.1, True, 0.5, 0.6)
slide(T_FLIP, True, 0.4, 0.5)
add(fx, bt(T_PLOP), blip(400, 90, 0.3, 0.9)); add(fx, bt(T_PLOP), bell(1568, 0.6), 0.2)
slide(T_FORK_IN, False, 0.35, 0.3)
for k in range(8): add(fx, bt(T_MASH_A + k * 0.5 - 0.5), blip(250, 120, 0.1, 0.55))
slide(T_BOWL_UP, True, 0.4, 0.6)
# step 3
slide(T_PLATE_IN2, True, 0.5, 0.6); thud(T_PLATE_IN2 + 1.1, 0.5)
slide(T_TILT, True, 0.3, 0.3)
for tl in T_DOLLOP: add(fx, bt(tl + 0.4), blip(350, 110, 0.18, 0.8))
for k in range(4): add(fx, bt(T_SPREAD_A + k * 0.55), whoosh(0.4, k % 2 == 0), 0.35)
add(fx, bt(T_SHINE + 0.1), bell(2637, 0.7), 0.3)
slide(T_SPREAD_B + 0.2, True, 0.4, 0.6)
# step 4
slide(T_EGG, False, 0.5); thud(T_EGG + 0.5, 0.8); thud(T_EGG + 0.95, 0.4)
for k in range(18): add(fx, bt(T_CHILI + k * 0.055 + 0.3), tick(), 0.45)
for k in range(16): add(fx, bt(T_PEPPER + k * 0.055 + 0.3), tick(), 0.45)
slide(T_LEAF, True, 0.35, 0.4); pop(T_LEAF + 0.5, 600); pop(T_LEAF + 0.75, 760)
slide(T_LEMON, True, 0.45, 0.45); pop(T_LEMON + 0.6, 880, 0.5)
# outro: confetti burst and a little arpeggio flourish
add(fx, bt(T_CONFETTI), blip(120, 60, 0.4, 1.0)); add(fx, bt(T_CONFETTI), whoosh(0.7, True), 0.6)
for k, m in enumerate([84, 88, 91, 96, 91, 96]):
    add(fx, bt(OUT + 0.2 + k * 0.25), pluck(m, 0.5, 1.0), 0.3, 0.3 if k % 2 else -0.3)
    add(fx, bt(OUT + 0.2 + k * 0.25), bell(mtof(m), 0.5), 0.05)
for k in range(10): add(fx, bt(T_CONFETTI + 0.3 + k * 0.12), tick(), 0.3)

mix = drums * 0.8 + mus * 0.9 + fx * 1.0
# gentle fade in/out and soft limiter
fade = np.minimum(1, np.arange(N) / (0.01 * SR)) * np.minimum(1, (N - np.arange(N)) / (0.5 * SR))
mix *= fade[:, None]
mix = np.tanh(mix * 1.1)
mix = mix / np.abs(mix).max() * 0.8
pcm = (mix * 32767).astype(np.int16)
with wave.open("music.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print("music.wav", round(N / SR, 3), "s")
