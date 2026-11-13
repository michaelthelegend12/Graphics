import numpy as np
from scipy.signal import fftconvolve
from scipy.io import wavfile

SR, DUR, T_IMP = 48000, 12.0, 8.0
n = int(SR * DUR)
t = np.arange(n) / SR
rng = np.random.default_rng(3)

def swept_noise(seed, f0, f1, t0, t1, width=1.1, blk=4096):
    """band-passed noise whose centre frequency sweeps exponentially f0->f1 over t0..t1."""
    r = np.random.default_rng(seed)
    x = r.standard_normal(n)
    out = np.zeros(n)
    win = np.hanning(blk)
    hop = blk // 2
    freqs = np.fft.rfftfreq(blk, 1 / SR)[1:]
    for s in range(0, n - blk, hop):
        tc = (s + blk / 2) / SR
        u = np.clip((tc - t0) / (t1 - t0), 0, 1)
        fc = f0 * (f1 / f0) ** u
        g = np.zeros(len(freqs) + 1)
        g[1:] = np.exp(-0.5 * (np.log2(freqs / fc) / width) ** 2)
        out[s:s + blk] += np.fft.irfft(np.fft.rfft(x[s:s + blk] * win) * g, blk) * win
    return out

# ---- riser (soft): swept noise + rising tonal pad ----
t0, t1 = 0.4, T_IMP
u = np.clip((t - t0) / (t1 - t0), 0, 1)
env = u ** 2.2 * (t >= t0)
env *= np.clip((T_IMP - t) / 0.06, 0, 1)            # tiny cut just before the hit
nz_l = swept_noise(1, 250, 7000, t0, t1)
nz_r = swept_noise(2, 250, 7000, t0, t1)
f = 110 * 2 ** (2.3 * u)
ph = 2 * np.pi * np.cumsum(f) / SR
pad = (np.sin(ph) + 0.5 * np.sin(2 * ph * 1.003) + 0.25 * np.sin(3 * ph * 0.997)) / 1.75
pad2 = (np.sin(ph * 1.004) + 0.5 * np.sin(2 * ph * 0.998)) / 1.5
fade_in = np.clip((t - t0) / 1.5, 0, 1)
riser_l = env * (0.55 * nz_l / np.std(nz_l) * 0.22 + 0.16 * pad * fade_in)
riser_r = env * (0.55 * nz_r / np.std(nz_r) * 0.22 + 0.16 * pad2 * fade_in)

# ---- impact: sub thump + noise burst + crack, with reverb tail ----
tt = np.clip(t - T_IMP, 0, None)
on = (t >= T_IMP).astype(float)
fsub = 38 + 90 * np.exp(-tt / 0.09)
sub = np.sin(2 * np.pi * np.cumsum(fsub) / SR) * np.exp(-tt / 1.3) * on
boom = np.sin(2 * np.pi * np.cumsum(70 + 160 * np.exp(-tt / 0.05)) / SR) * np.exp(-tt / 0.35) * on
burst = swept_noise(5, 4000, 300, T_IMP, T_IMP + 1.2, width=1.6) * np.exp(-tt / 0.55) * on
burst /= np.max(np.abs(burst)) + 1e-9
crack = rng.standard_normal(n) * np.exp(-tt / 0.012) * on
imp_l = 0.9 * sub + 0.45 * boom + 0.35 * burst + 0.18 * crack
imp_r = 0.9 * sub + 0.45 * boom + 0.35 * np.roll(burst, 37) + 0.18 * np.roll(crack, 21)

# reverb on impact only
irn = int(SR * 2.6)
ti = np.arange(irn) / SR
ir_l = rng.standard_normal(irn) * np.exp(-ti / 0.7)
ir_r = rng.standard_normal(irn) * np.exp(-ti / 0.7)
# darken the reverb
k = np.exp(-np.arange(64) / 12.0); k /= k.sum()
ir_l = np.convolve(ir_l, k, "same"); ir_r = np.convolve(ir_r, k, "same")
ir_l /= np.sqrt(np.sum(ir_l ** 2)); ir_r /= np.sqrt(np.sum(ir_r ** 2))
wet_l = fftconvolve(imp_l, ir_l)[:n]
wet_r = fftconvolve(imp_r, ir_r)[:n]
L = riser_l + imp_l + 0.28 * wet_l
R = riser_r + imp_r + 0.28 * wet_r

# gentle glue + normalise to about -1 dBFS, fade out the last 0.4 s
st = np.stack([L, R], 1)
st = np.tanh(st * 1.1)
st /= np.max(np.abs(st))
st *= 0.89
st *= np.clip((DUR - t) / 0.4, 0, 1)[:, None]
wavfile.write("audio.wav", SR, (st * 32767).astype(np.int16))
print("peak", np.abs(st).max(), "rms riser(4-7s)", np.sqrt(np.mean(st[4*SR:7*SR] ** 2)))
