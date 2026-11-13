"""Synthesises the techy sound fx for the CPU cycle explainer -> audio.wav (stereo, 44.1 kHz).
Reads timeline.py so every blip lines up with the picture. Needs numpy and scipy."""
import numpy as np
from scipy import signal
from scipy.io import wavfile

import timeline as TL

SR = 44100
N = int(TL.DUR * SR)
rng = np.random.default_rng(11)
L = np.zeros(N)
R = np.zeros(N)

PAN = {"addr": -0.7, "data": -0.6, "ir2dec": -0.15, "dec2alu": 0.15, "dec2reg": 0.2,
       "reg2alu_a": 0.35, "reg2alu_b": 0.45, "alu2res": 0.4, "wb": 0.7}


def tt(d):
    return np.arange(int(d * SR)) / SR


def put(sig, t0, pan=0.0, g=1.0):
    i = int(t0 * SR)
    if i < 0 or i >= N:
        return
    n = min(len(sig), N - i)
    a = (pan + 1) * np.pi / 4
    L[i:i + n] += sig[:n] * g * np.cos(a)
    R[i:i + n] += sig[:n] * g * np.sin(a)


def lp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "high", fs=SR, output="sos"), x)


def bp(x, lo, hi):
    return signal.sosfilt(signal.butter(2, [lo, hi], "band", fs=SR, output="sos"), x)


def blip(f, d=0.06, wave="sq", g=1.0):
    t = tt(d)
    ph = 2 * np.pi * f * t
    s = np.sign(np.sin(ph)) * 0.6 + 0.4 * np.sin(ph) if wave == "sq" else np.sin(ph)
    env = np.minimum(1, t / 0.003) * np.exp(-t / (d * 0.35))
    return s * env * g


def sweep(f0, f1, d, wave="sin", mod=0.0):
    t = tt(d)
    f = f0 * (f1 / f0) ** (t / d)
    ph = 2 * np.pi * np.cumsum(f) / SR
    if wave == "saw":
        s = 2 * ((ph / (2 * np.pi)) % 1) - 1
    elif wave == "tri":
        s = 2 * np.abs(2 * ((ph / (2 * np.pi)) % 1) - 1) - 1
    else:
        s = np.sin(ph) + 0.3 * np.sin(2 * ph)
    if mod:
        s *= 0.65 + 0.35 * np.sign(np.sin(2 * np.pi * mod * t))
    env = np.minimum(1, t / 0.012) * np.minimum(1, (d - t) / 0.04)
    return s * env


def click(g=1.0, f=3500):
    t = tt(0.006)
    return hp(rng.standard_normal(len(t)), 1500) * np.exp(-t / 0.0015) * g


def noise_riser(d, f0, f1):
    n = int(d * SR)
    x = rng.standard_normal(n)
    out = np.zeros(n)
    for k, fc in enumerate(np.geomspace(f0, f1, 8)):  # crossfade bands -> rising colour
        c = k / 7.0
        w = np.clip(1 - np.abs(np.linspace(0, 1, n) - c) * 5.0, 0, 1)
        out += bp(x, fc * 0.7, min(fc * 1.4, SR / 2 - 100)) * w
    return out * np.linspace(0, 1, n) ** 2


def thump(f0=95, f1=38, d=0.5, g=1.0):
    t = tt(d)
    f = f0 * (f1 / f0) ** (t / d)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (d * 0.3)) * g


# ---- ambience: electrical hum + a little shimmer
t_all = np.arange(N) / SR
bed = 0.05 * (np.sin(2 * np.pi * 55 * t_all) + 0.5 * np.sin(2 * np.pi * 110 * t_all) + 0.25 * np.sin(2 * np.pi * 165 * t_all))
shim = bp(rng.standard_normal(N), 3000, 7000) * 0.012 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.37 * t_all))
bed = (bed + shim) * np.minimum(1, t_all / 1.0)
L += bed
R += bed

# ---- boot-up: rising sweep + thump, then the clock starts ticking
put(lp(sweep(70, 1500, 1.1, "saw"), 1800) * np.linspace(0, 1, int(1.1 * SR)) ** 2, 0.05, 0.0, 0.22)
put(thump(95, 38, 0.6), 0.0, 0.0, 0.55)
put(noise_riser(1.0, 400, 5000), 0.1, 0.0, 0.10)
for i in range(30):
    put(click(), 0.2 + i / 34, -0.4, 0.07)  # title typing
for i, c in enumerate(TL.TITLE_2):
    put(click(), 13.0 + i / 30, -0.4, 0.07)
for k in range(8):
    put(click(), 12.7 + k * 0.03, -0.4, 0.05)  # backspacing
t = 1.2
while t < TL.LOOP_T:
    put(click(f=5000), t, 0.0, 0.05 if int((t - 1.2) / 0.125) % 4 else 0.09)
    t += 0.125

# ---- stage changes: riser into a two-note ping
for i, (name, t0, t1) in enumerate(TL.STAGES):
    put(lp(noise_riser(0.35, 500, 6000), 7000), t0 - 0.35, 0.0, 0.30)
    f = [523, 659, 784][i] * 2
    put(blip(f, 0.14, "sq"), t0, 0.0, 0.30)
    put(blip(f * 1.5, 0.16, "sq"), t0 + 0.08, 0.0, 0.28)
    put(thump(120, 60, 0.3), t0, 0.0, 0.25)

# ---- packets: data zip while travelling, double blip on arrival
for pid, tr, lab, t0, t1 in TL.PACKETS:
    d = t1 - t0
    base = 420 if lab.startswith(("ADD", "0x")) else 700
    put(sweep(base, base * 3.2, d, "tri", mod=26), t0, PAN[tr], 0.16)
    put(blip(2093, 0.05, "sq"), t1, PAN[tr], 0.24)
    put(blip(2637, 0.07, "sq"), t1 + 0.05, PAN[tr], 0.22)

# ---- blocks waking up
for name, ivs in TL.BLOCKS.items():
    for t0, t1 in ivs:
        if name in ("dec", "alu"):
            continue
        f = {"pc": 196, "ram": 165, "ir": 247, "reg": 220, "res": 294}[name]
        put(blip(f, 0.14, "sq"), t0, 0.0, 0.18)
        put(blip(f * 2, 0.08, "sq"), t0 + 0.03, 0.0, 0.10)
# memory read: short burst of digital static
put(bp(rng.standard_normal(int(0.2 * SR)), 1500, 6000) * np.hanning(int(0.2 * SR)), 2.25, -0.6, 0.18)

# ---- decoder chatter
pent = [1320, 1568, 1760, 2093, 2349, 2637, 3136]
t = 5.7
while t < 7.6:
    put(blip(float(rng.choice(pent)), 0.035, "sq"), t, 0.0, 0.16)
    t += float(rng.choice([0.04, 0.055, 0.07]))

# ---- ALU: rising whirr then a bell when the answer lands
d = 10.5 - 9.1
whirr = lp(sweep(90, 440, d, "saw", mod=34), 1500)
vib = 1 + 0.15 * np.sin(2 * np.pi * 9 * tt(d))
put(whirr * vib, 9.1, 0.3, 0.30)
put(sweep(900, 2400, d, "sin", mod=17) * 0.4, 9.1, 0.35, 0.12)
tb = tt(0.8)
bell = (np.sin(2 * np.pi * 1760 * tb) + 0.5 * np.sin(2 * np.pi * 1760 * 2.76 * tb) + 0.3 * np.sin(2 * np.pi * 1760 * 5.4 * tb)) * np.exp(-tb / 0.18)
put(bell, 10.5, 0.4, 0.22)

# ---- write-back confirm
for k, f in enumerate((880, 1320, 1760)):
    put(blip(f, 0.1, "sq"), 12.2 + k * 0.07, 0.6, 0.26)

# ---- console typing
for tt0, s in TL.CONSOLE:
    for i in range(len(s)):
        put(click(), tt0 + i / 45, 0.75, 0.045)

# ---- outro: the loop spins up faster and faster, then a final hit
put(noise_riser(1.9, 300, 7000), 12.6, 0.0, 0.34)
put(lp(sweep(220, 3200, 1.9, "saw"), 5000), 12.6, 0.0, 0.10)
for k, ts in enumerate(TL.STEPS):
    put(blip(880 + (k % 3) * 330, 0.05, "sq"), ts, [-0.4, 0.0, 0.4][k % 3], 0.24)
put(thump(75, 28, 0.9), 14.55, 0.0, 0.8)
put(bp(rng.standard_normal(int(0.5 * SR)), 800, 7000) * np.exp(-tt(0.5) / 0.1), 14.55, 0.0, 0.30)
put(blip(2093, 0.4, "sine", ), 14.55, 0.0, 0.12)

# ---- space + master
def echo(x, delay, fb):
    d = int(delay * SR)
    y = x.copy()
    y[d:] += fb * lp(x, 4500)[:-d]
    return y

L2 = echo(L, 0.17, 0.28)
R2 = echo(R, 0.23, 0.28)
out = np.stack([L2, R2], 1)
out = signal.sosfilt(signal.butter(2, 30, "high", fs=SR, output="sos"), out, axis=0)
out = np.tanh(out * 1.25)
fade = np.clip((TL.DUR - np.arange(N) / SR) / 0.3, 0, 1)
out *= fade[:, None]
out *= 0.89 / np.abs(out).max()
wavfile.write("audio.wav", SR, (out * 32767).astype("<i2"))
print("peak", np.abs(out).max(), "rms dB", 20 * np.log10(np.sqrt((out ** 2).mean())))
