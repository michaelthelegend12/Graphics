#!/usr/bin/env python3
"""Ambient sound bed for the hurricane explainer (30 s, 48 kHz stereo).

Calm sea swell and steam hiss at the start, then wind that builds to a howl, a deep
rumble, distant thunder and a low drone, with soft swooshes on each new stage.
Needs numpy and scipy.   python3 sound.py [out.wav]
"""
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

SR, DUR = 48000, 30.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(42)


def filt(x, kind, f, order=2):
    sos = butter(order, f, btype=kind, fs=SR, output="sos")
    return sosfilt(sos, x)


def white():
    return rng.standard_normal(N)


def ramp(pts):
    xs, ys = zip(*pts)
    return np.interp(t, xs, ys)


def slow_random(rate, seed, depth):
    """Smooth random modulator around 1.0."""
    r = np.random.default_rng(seed).standard_normal(N)
    r = filt(r, "low", rate, 2)
    r = r / (np.std(r) + 1e-9)
    return np.clip(1.0 + depth * r, 0.05, 3.0)


def stereo(fn):
    return np.stack([fn(), fn()])


# --- sea swell: filtered noise with slow surges
def sea():
    n = filt(white(), "band", [90, 1400])
    swell = 0.5 + 0.5 * np.sin(2 * np.pi * t / 7.3 + rng.uniform(0, 6)) ** 2
    return n * swell * slow_random(0.4, int(rng.integers(1e6)), 0.25)


# --- wind: three bands whose balance shifts brighter as the storm grows
def wind():
    lo = filt(white(), "band", [120, 420])
    mid = filt(white(), "band", [420, 1300])
    hi = filt(white(), "band", [1300, 3600])
    gust = slow_random(0.35, int(rng.integers(1e6)), 0.35)
    lvl = ramp([(0, 0.0), (3.0, 0.03), (8, 0.14), (16, 0.26), (24, 0.36), (30, 0.42)])
    bright = ramp([(0, 0.0), (10, 0.3), (20, 0.7), (30, 1.0)])
    return (lo * 1.0 + mid * (0.6 + 0.5 * bright) + hi * (0.15 + 0.7 * bright)) * lvl * gust


# --- howl: narrow resonances that drift slowly
def howl():
    out = np.zeros(N)
    for f0, s in ((520, 1), (810, 2), (1130, 3)):
        n = filt(white(), "band", [f0 * 0.97, f0 * 1.03], 3)
        out += n * slow_random(0.3, s + 100, 0.5)
    return out * ramp([(0, 0), (15, 0), (20, 0.05), (30, 0.16)])


def rumble():
    n = np.cumsum(white())
    n = filt(n - np.mean(n), "high", 18)
    n = filt(n, "low", 90, 3)
    n = n / (np.std(n) + 1e-9)
    return n * ramp([(0, 0), (6, 0.0), (12, 0.10), (20, 0.30), (30, 0.45)]) * slow_random(0.5, 77, 0.2)


def steam():
    n = filt(white(), "band", [4200, 9500])
    return n * ramp([(3.5, 0), (5, 0.015), (10, 0.015), (12, 0)]) * slow_random(1.2, 31, 0.4)


def thunder_roll(at, level, length=3.6):
    n0 = int(at * SR)
    m = int(length * SR)
    tt = np.arange(m) / SR
    body = filt(rng.standard_normal(m), "low", 160, 3)
    body *= np.minimum(tt / 0.06, 1) * np.exp(-tt / 1.3)
    body *= 1 + 0.5 * np.sin(2 * np.pi * 7 * tt + rng.uniform(0, 6)) * np.exp(-tt / 1.5)
    crack = filt(rng.standard_normal(m), "band", [500, 2500]) * np.exp(-tt / 0.18) * 0.25
    clicks = np.zeros(m)
    for k in range(5):
        c = int(rng.uniform(0.05, 1.2) * SR)
        clicks[c:c + 200] += rng.standard_normal(200) * np.exp(-np.arange(200) / 40)
    clicks = filt(clicks, "band", [300, 3000]) * 0.4 * np.exp(-tt / 0.9)
    sig = (body + crack + clicks) * level
    out = np.zeros(N)
    end = min(n0 + m, N)
    out[n0:end] = sig[:end - n0]
    return out


def swoosh(at, level, dur=1.3):
    n0, m = int(at * SR), int(dur * SR)
    tt = np.arange(m) / SR
    n = rng.standard_normal(m)
    out = np.zeros(m)
    # rising band-pass sweep built from a few overlapping bands
    for f in (300, 600, 1200, 2400):
        band = filt(n, "band", [f * 0.6, f * 1.4])
        env = np.exp(-((tt - dur * (0.35 + 0.1 * np.log2(f / 300))) / 0.28) ** 2)
        out += band * env
    thump = np.sin(2 * np.pi * 58 * tt) * np.exp(-tt / 0.35) * 2.2
    full = np.zeros(N)
    end = min(n0 + m, N)
    full[n0:end] = ((out * 0.5 + thump) * level)[:end - n0]
    return full


def drone():
    out = np.zeros(N)
    for f, a, ph in ((55.0, 1.0, 0), (82.4, 0.6, 1), (110.0, 0.5, 2), (130.8, 0.35, 3), (164.8, 0.25, 4)):
        det = 1 + 0.0015 * np.sin(2 * np.pi * (0.05 + 0.02 * ph) * t + ph)
        out += a * np.sin(2 * np.pi * f * det * t + ph)
    swell = ramp([(0, 0.0), (1.5, 0.35), (10, 0.5), (17, 0.8), (24, 1.0), (30, 0.8)])
    shimmer = np.sin(2 * np.pi * 440 * t) * 0.03 + np.sin(2 * np.pi * 659.3 * t) * 0.02
    return (out + shimmer * ramp([(0, 0), (16, 0), (24, 1), (30, 1)])) * swell * 0.045


bed = np.stack([sea(), sea()]) * ramp([(0, 0.0), (1.5, 0.34), (12, 0.32), (22, 0.16), (30, 0.12)])
bed += np.stack([wind(), wind()])
bed += np.stack([howl(), howl()])
bed += np.stack([rumble(), rumble()]) * 0.9
bed += np.stack([steam(), steam()])
d = drone()
bed += np.stack([d, d])
for at, lvl, pan in ((13.1, 0.55, -0.4), (19.0, 0.60, 0.4), (25.4, 0.65, -0.2), (28.0, 0.5, 0.5)):
    th = thunder_roll(at, lvl)
    bed += np.stack([th * (1 - pan) * 0.7, th * (1 + pan) * 0.7])
for at in (3.9, 10.9, 16.9, 23.4):
    sw = swoosh(at, 0.18)
    bed += np.stack([sw, sw])

bed *= (np.minimum(t / 0.9, 1.0) * np.minimum((DUR - t) / 1.4, 1.0))[None, :]
bed = np.tanh(bed * 1.1) / 1.1
peak = np.max(np.abs(bed))
bed *= 0.70 / peak
rms = 20 * np.log10(np.sqrt(np.mean(bed ** 2)))
print(f"peak 0.70, rms {rms:.1f} dBFS")

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "hurricane.wav")
wavfile.write(out, SR, (bed.T * 32767).astype(np.int16))
