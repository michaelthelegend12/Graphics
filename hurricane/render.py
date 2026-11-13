#!/usr/bin/env python3
"""How a hurricane forms: a 30 s documentary-style cutaway explainer.

1920x1080, 30 fps. A pseudo-3D slab of ocean and atmosphere is cut in half so the
viewer sees the cross-section: warm water, rising moist air, towering storm clouds,
the eye opening up and the winds circling it. Everything is drawn procedurally.

Needs numpy, opencv, Pillow and ffmpeg (plus sound.py for the ambient track).
    python3 render.py [out.mp4]            full render (2 worker processes, then joined)
    python3 render.py --preview 3 9 15     write PNG stills of those times to ./preview
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)

W, H, FPS, DUR = 1920, 1080, 30, 30.0
S0 = 560.0          # pixels per world unit (before the slow push-in)
RO = 1.15           # radius of the ocean disc
DEPTH = 0.20        # depth of the ocean slab shown
ZTROP = 0.95        # top of the cut plane (tropopause)
ANV_R = 0.88        # final radius of the anvil
HERE = os.path.dirname(os.path.abspath(__file__))
FD = "/usr/share/fonts/opentype/inter/"
F_XB, F_B, F_M = FD + "Inter-ExtraBold.otf", FD + "Inter-Bold.otf", FD + "Inter-Medium.otf"


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return min(max(x, a), b)


def sstep(a, b, x):
    t = clamp((x - a) / (b - a))
    return t * t * (3 - 2 * t)


def sstep_np(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def rgb(r, g, b):
    """Colour as a float32 BGR vector (the frame is BGR)."""
    return np.array([b, g, r], np.float32)


def band_noise(h, w, kmin, kmax, beta, seed, stretch=(1.0, 1.0)):
    """Tileable band-limited fractal noise in 0..1 (FFT filtered white noise)."""
    rng = np.random.default_rng(seed)
    F = np.fft.rfft2(rng.standard_normal((h, w)).astype(np.float32))
    fy = np.fft.fftfreq(h)[:, None] / stretch[0]
    fx = np.fft.rfftfreq(w)[None, :] / stretch[1]
    k = np.sqrt(fx * fx + fy * fy)
    k[0, 0] = 1e9
    amp = k ** (-beta / 2.0)
    amp *= np.clip((k - kmin * 0.7) / (kmin * 0.6), 0, 1)
    amp *= np.clip((kmax * 1.3 - k) / (kmax * 0.6), 0, 1)
    n = np.fft.irfft2(F * amp, s=(h, w)).astype(np.float32)
    lo, hi = np.percentile(n, [1, 99])
    return np.clip((n - lo) / (hi - lo), 0, 1).astype(np.float32)


NA = band_noise(1100, 2300, 1 / 200, 1 / 36, 2.6, 1)           # big billows
NB = band_noise(1100, 2300, 1 / 60, 1 / 5, 2.0, 2)             # fine detail / water
PN = band_noise(512, 1024, 1 / 70, 1 / 5, 2.2, 3, (1.0, 3.0))  # polar streaks (tiles in u)
PR = band_noise(512, 1024, 1 / 80, 1 / 8, 2.3, 6, (1.0, 1.0))  # softer polar billows for cloud tops
NC = band_noise(1100, 2300, 1 / 110, 1 / 14, 2.3, 4)           # mid-size cloud detail
NW = band_noise(1100, 2300, 1 / 120, 1 / 12, 2.5, 7)           # smooth water
GRAIN = [(np.random.default_rng(10 + i).standard_normal((H, W)) * 2.2 + 128).clip(0, 255).astype(np.uint8)
         for i in range(4)]


def blend_patch(frame, patch, alpha, x0, y0):
    """frame = frame*(1-a) + patch*a on a clipped rectangle. patch: HxWx3 or (3,)"""
    h, w = alpha.shape
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    a = alpha[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, None]
    p = patch if patch.ndim == 1 else patch[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    roi = frame[Y0:Y1, X0:X1]
    roi[:] = np.clip(roi.astype(np.float32) * (1 - a) + p * a + 0.5, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# camera and timeline
# ----------------------------------------------------------------------------
class Cam:
    """Oblique view from above-front: world x right, y away from the viewer, z up."""

    def __init__(self, t):
        u = sstep(0, 30, t)
        phi = math.radians(lerp(27.0, 21.5, u))
        self.sin, self.cos = math.sin(phi), math.cos(phi)
        self.S = S0 * lerp(1.0, 1.05, u)
        self.cx = 960.0
        self.cy0 = lerp(770.0, 800.0, u)

    def P(self, x, y, z):
        return self.cx + np.asarray(x) * self.S, self.cy0 - (np.asarray(z) * self.cos + np.asarray(y) * self.sin) * self.S


def prm(t):
    return dict(
        warm=sstep(0.6, 3.8, t),       # sea heat map fades in
        g=sstep(8.6, 17.0, t),         # cloud tower growth
        anv=sstep(12.8, 20.5, t),      # anvil spreads
        e=sstep(16.8, 22.6, t),        # eye opens
        spin=sstep(15.0, 21.0, t),     # circulation organises
        wind=sstep(14.0, 27.0, t),     # whitecaps, strong wind
    )


def omega(t):
    return 0.12 + 0.95 * sstep_np(15.0, 21.0, t) + 0.2 * sstep_np(22.0, 27.0, t)


_RT = np.linspace(0, 30, 6001)
_RV = np.cumsum(omega(_RT)) * (30 / 6000.0)


def rot_at(t):
    return float(np.interp(t, _RT, _RV))


def tower_top(g):
    return 0.14 + 0.74 * g


def ri_mature(z):
    return 0.07 + 0.19 * np.minimum(z / 0.85, 1.3) ** 1.25


def wall_rc(z):
    zn = min(z / 0.85, 1.3)
    return 0.07 + 0.19 * zn ** 1.25 + (0.20 - 0.07 * min(zn, 1.0)) / 2


# cut-plane cloud puffs: (x, half-width, start height, final height, start time)
BLOBS = [(-1.02, .10, .08, .19, 4.6), (-0.84, .09, .07, .14, 5.3), (-0.66, .12, .10, .33, 4.9),
         (-0.47, .09, .08, .20, 5.7), (-0.36, .08, .08, .26, 5.9),
         (0.36, .08, .08, .27, 5.5), (0.50, .10, .09, .30, 5.0), (0.69, .12, .11, .36, 4.8),
         (0.87, .09, .08, .17, 5.4), (1.03, .10, .08, .19, 6.1)]
# puffs behind the cut plane: (x, y, half-width, start height, final height, start time)
rngp = np.random.default_rng(5)
PUFFS = []
for i in range(16):
    ang = rngp.uniform(0.15, math.pi - 0.15)
    rad = rngp.uniform(0.45, 1.02)
    PUFFS.append((rad * math.cos(ang), rad * math.sin(ang), rngp.uniform(0.09, 0.15), rngp.uniform(0.05, 0.09),
                  rngp.uniform(0.12, 0.28), rngp.uniform(4.6, 7.2), i % 3))


# ----------------------------------------------------------------------------
# cloud shading shared by the cut face and the puff sprites
# ----------------------------------------------------------------------------
C_DARK, C_MID, C_LIGHT = rgb(66, 84, 122), rgb(166, 183, 210), rgb(252, 253, 255)


def shade_cloud(fn, nd, vert):
    """fn: noisy field in world units (>0 is cloud), nd: detail noise 0..1, vert: 0..1 height."""
    fn = fn.astype(np.float32)
    nd = nd.astype(np.float32)
    alpha = sstep_np(-0.010, 0.014, fn).astype(np.float32)
    body = np.clip(fn * 12.0, 0, 1).astype(np.float32)
    hf = cv2.GaussianBlur(body, (0, 0), 4.0)
    gx = cv2.Sobel(hf, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(hf, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    dn = cv2.GaussianBlur(nd, (0, 0), 2.4)
    dx = cv2.Sobel(dn, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    dy = cv2.Sobel(dn, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    lit = 0.58 + 15.0 * (0.6 * gx + 0.8 * gy) + 2.6 * (0.6 * dx + 0.8 * dy) * body + (nd - 0.5) * 0.22 * body
    lit = np.clip(lit, 0, 1)[..., None]
    a = np.clip(lit * 2, 0, 1)
    b = np.clip(lit * 2 - 1, 0, 1)
    col = C_DARK + (C_MID - C_DARK) * a + (C_LIGHT - C_MID) * b
    col *= (0.82 + 0.18 * vert)[..., None] if np.ndim(vert) == 2 else (0.82 + 0.18 * vert)
    ab = cv2.GaussianBlur(alpha, (0, 0), 3.0)
    rim = np.clip((alpha - ab) * 2.4, 0, 1)[..., None]
    col += rim * (255 - col) * 0.35
    return col.astype(np.float32), alpha


def make_sprite(w, h, seed):
    rng = np.random.default_rng(seed)
    u = (np.arange(w, dtype=np.float32)[None, :] - w / 2) / (w / 2)
    v = (h - np.arange(h, dtype=np.float32)[:, None]) / h
    n1 = band_noise(h, w, 1 / 60, 1 / 8, 2.4, seed)
    n2 = band_noise(h, w, 1 / 40, 1 / 9, 2.2, seed + 50)
    d = np.sqrt((u / 0.92) ** 2 + ((v - 0.42) / 0.52) ** 2)
    base = (1 - d) * 0.35 + (n1 - 0.5) * 0.28
    base = np.where(v > 0.96, -1.0, base)
    fn = base * 0.4
    col, alpha = shade_cloud(fn.astype(np.float32), n2.astype(np.float32), v)
    alpha = alpha * np.clip(v / 0.18, 0, 1).astype(np.float32)
    return col, alpha


SPRITES = [make_sprite(300, 190, 20 + i) for i in range(3)]


# ----------------------------------------------------------------------------
# text
# ----------------------------------------------------------------------------
_fc = {}


def font(path, size):
    k = (path, size)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(path, size)
    return _fc[k]


def tracked(draw, xy, text, fnt, fill, tracking=0.0):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += draw.textlength(ch, font=fnt) + tracking
    return x


def tracked_len(draw, text, fnt, tracking):
    return sum(draw.textlength(ch, font=fnt) + tracking for ch in text)


def to_patch(img):
    a = np.asarray(img).astype(np.float32)
    return a[..., 2::-1].copy(), a[..., 3] / 255.0


def text_block(items, pad=8, shadow=True, bar=None):
    """items: list of (text, fontpath, size, rgb, tracking, gap_after). Returns (bgr, alpha, w, h)."""
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    widths, heights = [], []
    for text, fp, sz, col, tr, gap in items:
        f = font(fp, sz)
        widths.append(tracked_len(tmp, text, f, tr))
        heights.append(sz * 1.2)
    w = int(max(widths)) + pad * 2 + (14 if bar else 0)
    h = int(sum(heights) + sum(i[5] for i in items)) + pad * 2
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    y = pad
    x0 = pad + (14 if bar else 0)
    if shadow:
        sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        yy = pad
        for (text, fp, sz, col, tr, gap), hh in zip(items, heights):
            tracked(sd, (x0 + 2, yy + 2), text, font(fp, sz), (0, 0, 0, 170), tr)
            yy += hh + gap
        from PIL import ImageFilter
        sh = sh.filter(ImageFilter.GaussianBlur(2.2))
        img = Image.alpha_composite(img, sh)
    d = ImageDraw.Draw(img)
    if bar:
        d.rectangle([pad, pad + 2, pad + 4, h - pad - 2], fill=bar)
    for (text, fp, sz, col, tr, gap), hh in zip(items, heights):
        tracked(d, (x0, y), text, font(fp, sz), col + (255,), tr)
        y += hh + gap
    bgr, al = to_patch(img)
    return bgr, al, w, h


YEL = (255, 204, 0)
WHITE = (246, 248, 252)
SOFT = (196, 208, 224)

# labels: stage windows, side, y, text, world anchor
LABELS = [
    dict(t0=4.7, t1=10.6, side="L", y=690, title="WARM OCEAN", lines=["26.5 °C (80 °F) or more", "to a depth of ~50 m"],
         anchor=(-1.08, 0.0, -0.06), chip=True),
    dict(t0=5.5, t1=10.6, side="R", y=440, title="EVAPORATION", lines=["Warm, moist air rises"],
         anchor=(0.86, 0.0, 0.26)),
    dict(t0=11.2, t1=16.6, side="L", y=600, title="LOW PRESSURE", lines=["Air rushes in", "along the surface"],
         anchor=(-0.95, 0.0, 0.07)),
    dict(t0=12.0, t1=16.6, side="R", y=400, title="CONDENSATION", lines=["Vapor turns to cloud,", "releasing heat"],
         anchor=(0.27, 0.0, 0.50)),
    dict(t0=17.2, t1=23.0, side="L", y=470, title="EARTH'S SPIN", lines=["Twists the winds", "counter-clockwise", "(Northern Hemisphere)"],
         anchor=(-0.506, 0.215, 0.30)),
    dict(t0=19.6, t1=23.0, side="R", y=400, title="THE EYE FORMS", lines=["Dry air sinks and", "clears the center"],
         anchor=(0.03, 0.0, 0.50)),
    dict(t0=23.5, t1=29.3, side="R", y=410, title="OUTFLOW", lines=["Air spreads out", "about 15 km up"],
         anchor=(0.78, 0.0, 0.86)),
    dict(t0=24.0, t1=29.3, side="R", y=560, title="THE EYE", lines=["Calm and clear,", "20 to 40 miles wide"],
         anchor=(0.03, 0.0, 0.45)),
    dict(t0=24.5, t1=29.3, side="R", y=700, title="RAINBANDS", lines=["Spiral arms of storms"],
         anchor=(0.95, 0.0, 0.14)),
    dict(t0=23.8, t1=29.3, side="L", y=470, title="EYEWALL", lines=["Strongest winds,", "heaviest rain"],
         anchor=(-0.24, 0.0, 0.46)),
    dict(t0=24.4, t1=29.3, side="L", y=700, title="INFLOW", lines=["Warm, moist air", "spirals in"],
         anchor=(-0.9, 0.0, 0.07)),
]
for L in LABELS:
    items = [(L["title"], F_XB, 25, YEL, 1.5, 4)] + [(s, F_M, 21, WHITE, 0.2, 1) for s in L["lines"]]
    L["bgr"], L["a"], L["w"], L["h"] = text_block(items)
    if L.get("chip"):
        pass

CAPTIONS = [
    (3.9, 10.9, "01", "WARM OCEAN", "Seas of 26.5 °C (80 °F) or more heat the air above, and moist air starts to rise."),
    (10.9, 16.9, "02", "STORM CLOUDS TOWER", "Rising vapor condenses, releasing heat that powers the storm. Air rushes in below."),
    (16.9, 23.4, "03", "SPIN AND SINKING AIR", "Earth's rotation twists the winds into a spiral, and sinking air opens the eye."),
    (23.4, 29.4, "04", "A MATURE HURRICANE", "Winds of 74 mph (119 km/h) or more circle the eye, strongest in the eyewall."),
]
CAP_IMG = []
for (a, b, num, title, desc) in CAPTIONS:
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    nf = font(F_XB, 64)
    items_t = [(title, F_XB, 40, WHITE, 2.0, 6), (desc, F_M, 26, SOFT, 0.2, 0)]
    bgr, al, w, h = text_block(items_t, pad=6)
    img = Image.new("RGBA", (w + 110, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((4, 6), num, font=nf, fill=YEL + (255,))
    d.rectangle([96, 10, 100, h - 10], fill=YEL + (255,))
    base = np.asarray(img).astype(np.float32)
    big = np.zeros((h, w + 110, 3), np.float32)
    ab = np.zeros((h, w + 110), np.float32)
    big[..., :] = base[..., 2::-1]
    ab = base[..., 3] / 255.0
    big[:, 110:] = np.where(al[..., None] > 0.01, bgr, big[:, 110:])
    ab[:, 110:] = np.maximum(ab[:, 110:], al)
    CAP_IMG.append((big, ab, w + 110, h))

TAG = text_block([("HOW A HURRICANE FORMS", F_B, 21, WHITE, 3.2, 0)], pad=4)
TITLE = text_block([("HOW A HURRICANE", F_XB, 92, WHITE, 3.0, 0), ("FORMS", F_XB, 92, WHITE, 3.0, 0)], pad=10)
SUBT = text_block([("From warm ocean to the eye of the storm", F_M, 32, SOFT, 1.2, 0)], pad=6)
INSET_CAP = text_block([("VIEW FROM ABOVE", F_B, 18, WHITE, 2.6, 0)], pad=3)
SCALE_TXT = {k: text_block([(k, F_M, 18, SOFT, 0.5, 0)], pad=2, shadow=False) for k in
             ["15 km", "10 km", "5 km", "0", "50 m", "100 m", "TROPOPAUSE"]}
CHIP = text_block([("28 °C", F_B, 18, WHITE, 0.3, 0)], pad=2, shadow=False), text_block([("5 °C", F_B, 18, WHITE, 0.3, 0)], pad=2, shadow=False)

# ----------------------------------------------------------------------------
# static background and vignette
# ----------------------------------------------------------------------------
yy_, xx_ = np.mgrid[0:H, 0:W].astype(np.float32)
_bg = np.zeros((H, W, 3), np.float32)
_t = (yy_ / H)[..., None]
_bg += rgb(9, 15, 28) * (1 - _t) + rgb(5, 9, 17) * _t
_glow = np.exp(-(((xx_ - 960) / 760) ** 2 + ((yy_ - 560) / 470) ** 2))[..., None]
_bg += _glow * rgb(16, 36, 54) * 1.0
BG = np.clip(_bg, 0, 255).astype(np.uint8)
_v = 1.0 - 0.55 * np.clip((np.sqrt(((xx_ - 960) / 1100) ** 2 + ((yy_ - 540) / 640) ** 2) - 0.55) / 0.75, 0, 1) ** 1.6
VIG = np.repeat(np.clip(_v * 255, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
del yy_, xx_, _bg, _t, _glow, _v

# ----------------------------------------------------------------------------
# ocean
# ----------------------------------------------------------------------------
STOPS_D = np.array([0.0, 0.18, 0.40, 0.58, 0.80, 1.0])
STOPS_WARM = np.array([rgb(232, 72, 44), rgb(246, 132, 50), rgb(250, 198, 78), rgb(52, 168, 172), rgb(24, 92, 152), rgb(10, 36, 92)])
STOPS_COOL = np.array([rgb(34, 104, 150), rgb(28, 90, 138), rgb(22, 72, 118), rgb(16, 56, 100), rgb(12, 42, 82), rgb(8, 30, 66)])
SEA_NEAR, SEA_FAR = rgb(10, 52, 86), rgb(24, 92, 124)
HEAT = rgb(255, 128, 52)


def lut(stops, d):
    return np.stack([np.interp(d, STOPS_D, stops[:, c]) for c in range(3)], axis=-1).astype(np.float32)


def draw_shadow(frame, cam):
    S = cam.S
    h_px = DEPTH * cam.cos * S
    cy = cam.cy0 + h_px + 22
    rx, ry = RO * S * 1.04, 36
    x0, x1 = int(cam.cx - rx * 1.2), int(cam.cx + rx * 1.2)
    y0, y1 = int(cy - ry * 2.5), int(cy + ry * 2.5)
    xs = (np.arange(x0, x1, dtype=np.float32)[None, :] - cam.cx) / rx
    ys = (np.arange(y0, y1, dtype=np.float32)[:, None] - cy) / ry
    m = np.exp(-(xs * xs + ys * ys) * 1.6) * 0.75
    blend_patch(frame, np.zeros(3, np.float32), m.astype(np.float32), x0, y0)


def draw_sea_surface(frame, cam, t, P):
    S, sn, cx, cy0 = cam.S, cam.sin, cam.cx, cam.cy0
    x0, x1 = int(cx - RO * S) - 2, int(cx + RO * S) + 2
    y0, y1 = int(cy0 - RO * sn * S) - 2, int(cy0) + 1
    cols = np.arange(x0, x1, dtype=np.float32)[None, :]
    rows = np.arange(y0, y1, dtype=np.float32)[:, None]
    xx = (cols - cx) / S
    yy = (cy0 - rows) / (sn * S)
    rr = np.sqrt(xx * xx + yy * yy)
    inside = (np.clip((RO - rr) * S * 0.9, 0, 1) * (yy >= 0)).astype(np.float32)
    ix = ((xx * S * 0.9 + t * 16.0).astype(np.int32)) % NW.shape[1]
    iy = ((yy * S * 0.9 + t * 7.0).astype(np.int32)) % NW.shape[0]
    n = NW[iy, ix]
    n2 = NB[iy, ix]
    far = np.clip(yy / RO, 0, 1)[..., None] * np.ones_like(n)[..., None]
    base = SEA_NEAR * (1 - far) + SEA_FAR * far
    hm = np.exp(-(rr / 0.78) ** 2) * (1.0 - 0.4 * sstep(15, 24, t))
    hm = np.clip(hm * (0.80 + 0.40 * n) * P["warm"], 0, 0.78)[..., None]
    col = base + (HEAT - base) * hm
    col = col * (0.86 + 0.30 * n[..., None]) * (0.95 + 0.10 * n2[..., None])
    spec = (np.clip((n2 - 0.78) * 4, 0, 1) * 0.30 * (1 - 0.6 * far[..., 0]))[..., None]
    col = col + spec * 120
    wc = (np.clip((n - 0.62) * 4.5, 0, 1) * np.clip((n2 - 0.35) * 2, 0, 1) * P["wind"] * 0.5)[..., None]
    col = col + wc * (235 - col)
    blend_patch(frame, col.astype(np.float32), inside, x0, y0)
    cv2.ellipse(frame, (int(round(cx)), int(round(cy0))), (int(RO * S), int(RO * sn * S)), 0, 180, 360,
                (120, 160, 175), 1, cv2.LINE_AA)


def draw_sea_cut(frame, cam, t, P):
    S, cs, cx, cy0 = cam.S, cam.cos, cam.cx, cam.cy0
    x0, x1 = int(cx - RO * S), int(cx + RO * S)
    h_px = int(round(DEPTH * cs * S))
    d = (np.arange(h_px, dtype=np.float32) + 0.5) / h_px
    warm = P["warm"]
    col = lut(STOPS_COOL, d) * (1 - warm) + lut(STOPS_WARM, d) * warm
    xs = np.arange(x1 - x0, dtype=np.float32)[None, :]
    shim = 1.0 + 0.045 * np.sin(xs * 0.021 + t * 1.6 + d[:, None] * 11) + 0.03 * np.sin(xs * 0.057 - t * 2.1 + d[:, None] * 5)
    patch = col[:, None, :] * shim[..., None]
    nsl = NB[300:300 + h_px, 100:100 + (x1 - x0)]
    patch = patch * (0.94 + 0.12 * nsl[..., None])
    y0 = int(cy0)
    blend_patch(frame, patch.astype(np.float32), np.ones((h_px, x1 - x0), np.float32), x0, y0)
    cv2.rectangle(frame, (x0, y0), (x1, y0 + h_px), (235, 215, 190), 2, cv2.LINE_AA)
    cv2.line(frame, (x0, y0), (x1, y0), (255, 240, 225), 2, cv2.LINE_AA)
    return x0, x1, y0, h_px


# ----------------------------------------------------------------------------
# clouds: puffs behind the plane, eye wall, anvil roof, rainband deck, cut-face clouds
# ----------------------------------------------------------------------------
def draw_puffs(frame, cam, t, P):
    g = P["g"]
    items = []
    for (x, y, hw, h0, hf, t0, k) in PUFFS:
        s = sstep(t0, t0 + 2.4, t)
        if s < 0.02:
            continue
        hh = (h0 + (hf - h0) * sstep(9.0, 17.0, t)) * s
        items.append((y, x, hw * (0.45 + 0.55 * s) * (1 + 0.25 * g), hh, k))
    items.sort(reverse=True)
    for (y, x, hw, hh, k) in items:
        col, al = SPRITES[k]
        wpx = max(int(hw * 2 * cam.S * 1.25), 4)
        hpx = max(int(hh * cam.cos * cam.S * 1.9), 4)
        c2 = cv2.resize(col, (wpx, hpx), interpolation=cv2.INTER_AREA)
        a2 = cv2.resize(al, (wpx, hpx), interpolation=cv2.INTER_AREA)
        px, py = cam.P(x, y, 0.0)
        blend_patch(frame, c2, a2 * 0.92, int(px - wpx / 2), int(py - hpx * 0.92))


def eye_ri(P):
    return P["e"]


def draw_eye_wall(frame, cam, t, P, rot):
    e, g = P["e"], P["g"]
    if e < 0.05 or g < 0.05:
        return
    S, sn, cs, cx, cy0 = cam.S, cam.sin, cam.cos, cam.cx, cam.cy0
    zt = tower_top(g)
    N = 120
    zs = np.linspace(0.0, zt, N).astype(np.float32)
    ri = (e * ri_mature(zs)).astype(np.float32)
    rmax = float(ri.max())
    x0, x1 = int(cx - rmax * S) - 6, int(cx + rmax * S) + 7
    ytop = int(cy0 - (zt * cs + rmax * sn) * S) - 6
    ybot = int(cy0) + 2
    zmap = np.zeros((ybot - ytop, x1 - x0), np.uint8)
    for i in range(N):
        c = (int(round(cx - x0)), int(round(cy0 - zs[i] * cs * S - ytop)))
        ax = (max(int(round(ri[i] * S)), 1), max(int(round(ri[i] * sn * S)), 1))
        cv2.ellipse(zmap, c, ax, 0, 180, 360, i + 1, 5)
    ys, xs = np.nonzero(zmap)
    if len(ys) == 0:
        return
    idx = zmap[ys, xs].astype(np.int32) - 1
    zi = zs[idx]
    rpx = np.maximum(ri[idx] * S, 1.0)
    u = np.clip((xs + x0 - cx) / rpx, -1, 1)
    th = np.arccos(u)
    shade = np.sin(th) ** 0.55
    tx = (xs + x0 + rot * rpx * np.sin(th) * 0.9 + 6000).astype(np.int32) % NC.shape[1]
    ty = (ys + ytop + 400).astype(np.int32) % NC.shape[0]
    tex = NC[ty, tx]
    hz = (zi / zt)[:, None]
    band = 0.5 + 0.5 * np.sin(zi * 55 + tex * 5)
    dark, light = rgb(30, 44, 76), rgb(178, 198, 226)
    c = dark + (light - dark) * (0.10 + 0.85 * hz) * (0.35 + 0.8 * shade[:, None])
    c = c * (0.58 + 0.60 * tex[:, None] + 0.18 * band[:, None])
    c = c + np.array([0, 10, 26], np.float32) * (1 - hz) * shade[:, None]
    frame[ys + ytop, xs + x0] = np.clip(c, 0, 255).astype(np.uint8)


def plane_layer(frame, cam, t, z, r0, r1, rot, kind, P, strength=1.0):
    """Horizontal cloud layer (back half only) drawn by inverse-mapping screen pixels onto the plane."""
    S, sn, cs, cx, cy0 = cam.S, cam.sin, cam.cos, cam.cx, cam.cy0
    yline = cy0 - z * cs * S
    x0, x1 = int(cx - r1 * S), int(cx + r1 * S) + 1
    y0, y1 = max(int(yline - r1 * sn * S) - 1, 0), int(yline) + 2
    cols = np.arange(x0, x1, dtype=np.float32)[None, :]
    rows = np.arange(y0, y1, dtype=np.float32)[:, None]
    xx = (cols - cx) / S
    yy = (yline - rows) / (sn * S)
    r = np.sqrt(xx * xx + yy * yy)
    th = np.arctan2(yy, xx)
    ph = th - rot + (1.15 if kind == "roof" else 3.0) * np.log(r + 0.1)
    u = ((ph / (2 * math.pi)) % 1.0 * PR.shape[1]).astype(np.int32) % PR.shape[1]
    v = ((r * (230 if kind == "roof" else 200)).astype(np.int32)) % PR.shape[0]
    tex = PR[v, u]
    org = P["spin"]
    hf = cv2.GaussianBlur(tex, (0, 0), 2.2)
    gx = cv2.Sobel(hf, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(hf, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    bump = 10.0 * (0.6 * gx + 0.8 * gy)
    if kind == "roof":
        ri_a = P["e"] * float(ri_mature(z))
        arms = 0.5 + 0.5 * np.cos(3 * (th - rot + 1.8 * np.log(r + 0.1)))
        arms = 0.5 + (arms - 0.5) * (0.25 + 0.75 * org)
        edge = sstep_np(ri_a - 0.006, ri_a + 0.012, r) * sstep_np(r1, r1 - 0.20 - 0.10 * (1 - P["anv"]), r + (tex - 0.5) * 0.28)
        alpha = edge * (yy >= 0) * (0.97 - 0.28 * sstep_np(0.45, 1.0, r / r1))
        br = 0.52 + 0.34 * arms + 0.34 * (tex - 0.5) + bump
        br = br * (0.70 + 0.30 * sstep_np(ri_a, ri_a + 0.16, r))
        lo, hi = rgb(96, 116, 152), rgb(252, 253, 255)
    else:
        arms = 0.5 + 0.5 * np.cos(2 * ph)
        arms = 0.5 + (arms - 0.5) * org
        v_ = arms * 0.9 + (tex - 0.5) * 0.9
        alpha = sstep_np(0.46, 0.78, v_) * sstep_np(r0, r0 + 0.10, r) * sstep_np(r1, r1 - 0.20, r + (tex - 0.5) * 0.2)
        alpha = alpha * (yy >= 0) * 0.8 * strength
        br = 0.50 + 0.55 * (tex - 0.5) + bump
        lo, hi = rgb(92, 112, 146), rgb(232, 238, 248)
    col = lo + (hi - lo) * np.clip(br, 0, 1)[..., None]
    blend_patch(frame, col.astype(np.float32), alpha.astype(np.float32), x0, y0)


def draw_cut_clouds(frame, cam, t, P):
    S, cs, cx, cy0 = cam.S, cam.cos, cam.cx, cam.cy0
    g, e, anv = P["g"], P["e"], P["anv"]
    x0, x1 = int(cx - RO * S), int(cx + RO * S)
    y1 = int(cy0)
    y0 = int(cy0 - ZTROP * cs * S)
    Wr, Hr = x1 - x0, y1 - y0
    cols = np.arange(x0, x1, dtype=np.float32)
    rows = np.arange(y0, y1, dtype=np.float32)
    X = ((cols - cx) / S)[None, :]
    Z = ((cy0 - rows) / (cs * S))[:, None]
    R = np.abs(X)
    field = np.full((Hr, Wr), -1.0, np.float32)
    zt = tower_top(g)
    if g > 0.002:
        zn = np.clip(Z / 0.85, 0, 1.3)
        ri_m = 0.07 + 0.19 * zn ** 1.25
        th = 0.20 - 0.07 * np.minimum(zn, 1.0)
        w = 0.55 + 0.45 * g
        ro = (ri_m + th) * w
        ri = e * ri_m
        fw = np.minimum(ro - R, R - ri + (1.0 - sstep(0.0, 0.06, e)))
        cap = (zt - Z) - 0.14 * (R / 0.4) ** 2 * (1.0 - 0.5 * e)
        field = np.maximum(field, np.minimum(fw, cap * 0.9))
        if anv > 0.01:
            za = zt - 0.02
            zn_t = min(zt / 0.85, 1.3)
            ro_t = ((0.07 + 0.19 * zn_t ** 1.25) + (0.20 - 0.07 * min(zn_t, 1.0))) * w
            rmax = ro_t + (ANV_R - ro_t) * anv
            ha = 0.05 * (1 - 0.55 * np.clip((R - 0.3) / 0.7, 0, 1))
            ri_a = e * (0.07 + 0.19 * min(za / 0.85, 1.3) ** 1.25)
            fa = np.minimum(ha - np.abs(Z - za), np.minimum((rmax - R) * 0.35, R - ri_a + (1.0 - sstep(0.0, 0.06, e))))
            field = np.maximum(field, fa)
    for (bx, bw, h0, hf, t0) in BLOBS:
        s = sstep(t0, t0 + 2.2, t)
        if s <= 0.01:
            continue
        hh = (h0 + (hf - h0) * sstep(9.0, 17.0, t)) * s
        ww = bw * (0.45 + 0.55 * s) * (1 + 0.25 * g)
        j0 = max(int((bx - ww * 1.15) * S + cx) - x0, 0)
        j1 = min(int((bx + ww * 1.15) * S + cx) - x0 + 1, Wr)
        i0 = max(int(cy0 - hh * 1.12 * cs * S) - y0, 0)
        if j1 <= j0 or i0 >= Hr:
            continue
        Xs, Zs = X[:, j0:j1], Z[i0:]
        d = np.sqrt(((Xs - bx) / ww) ** 2 + ((Zs - 0.5 * hh) / (0.58 * hh)) ** 2)
        fb = (1.0 - d) * min(ww, 0.58 * hh)
        sl = field[i0:, j0:j1]
        np.maximum(sl, fb, out=sl)
    ox1, oy1 = int(t * 9) % (NA.shape[1] - Wr), int(t * 4) % (NA.shape[0] - Hr)
    ox2, oy2 = (int(-t * 14) + 700) % (NC.shape[1] - Wr), (int(t * 6) + 300) % (NC.shape[0] - Hr)
    n1 = NA[oy1:oy1 + Hr, ox1:ox1 + Wr]
    n2 = NC[oy2:oy2 + Hr, ox2:ox2 + Wr]
    fn = field + (0.62 * n1 + 0.38 * n2 - 0.5) * 0.17
    col, alpha = shade_cloud(fn, n2, np.clip(Z / 0.9, 0, 1) * np.ones((1, Wr), np.float32))
    blend_patch(frame, col, alpha, x0, y0)


# ----------------------------------------------------------------------------
# arrows
# ----------------------------------------------------------------------------
def catmull(pts, n=60):
    pts = np.asarray(pts, np.float64)
    p = np.vstack([pts[0], pts, pts[-1]])
    out = []
    m = len(pts) - 1
    per = max(n // m, 2)
    for i in range(m):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        for s in np.linspace(0, 1, per, endpoint=(i == m - 1)):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s * s + (-p0 + 3 * p1 - 3 * p2 + p3) * s ** 3))
    return np.array(out)


def flow(img, pts, prog, color, width, t, speed=75.0, dash=30, gap=16):
    """Dashed, flowing arrow along a screen-space polyline revealed up to `prog` (0..1)."""
    if prog <= 0.01 or len(pts) < 2:
        return
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1]
    Ld = L * prog
    per = dash + gap
    s = -((t * speed) % per)
    dark = (12, 14, 22)
    pieces = []
    while s < Ld:
        a, b = max(s, 0.0), min(s + dash, Ld)
        if b - a > 2:
            ss = np.linspace(a, b, max(int((b - a) / 6), 2))
            pieces.append(np.stack([np.interp(ss, cum, pts[:, 0]), np.interp(ss, cum, pts[:, 1])], 1).astype(np.int32))
        s += per
    for pc in pieces:
        cv2.polylines(img, [pc], False, dark, int(width + 3), cv2.LINE_AA)
    for pc in pieces:
        cv2.polylines(img, [pc], False, color, int(width), cv2.LINE_AA)
    # arrow head at the tip
    tip = np.array([np.interp(Ld, cum, pts[:, 0]), np.interp(Ld, cum, pts[:, 1])])
    back = np.array([np.interp(max(Ld - 14, 0), cum, pts[:, 0]), np.interp(max(Ld - 14, 0), cum, pts[:, 1])])
    d = tip - back
    nd = np.hypot(*d)
    if nd < 1e-3:
        return
    d /= nd
    nrm = np.array([-d[1], d[0]])
    hs = width * 2.0
    tri = np.array([tip + d * hs * 0.9, tip - d * hs * 0.8 + nrm * hs * 0.95, tip - d * hs * 0.8 - nrm * hs * 0.95])
    cv2.fillConvexPoly(img, tri.astype(np.int32), dark, cv2.LINE_AA)
    tri2 = np.array([tip + d * hs * 0.55, tip - d * hs * 0.55 + nrm * hs * 0.62, tip - d * hs * 0.55 - nrm * hs * 0.62])
    cv2.fillConvexPoly(img, tri2.astype(np.int32), color, cv2.LINE_AA)


def project(cam, pts3):
    x, y = cam.P(pts3[:, 0], pts3[:, 1], pts3[:, 2])
    return np.stack([x, y], 1)


def ring3(R, z, th0=14, th1=166, n=60, wob=0.0, ph=0.0):
    th = np.radians(np.linspace(th0, th1, n))
    r = R * (1 + wob * np.sin(th * 3 + ph))
    return np.stack([r * np.cos(th), r * np.sin(th), np.full(n, z)], 1)


def bgr(r, g, b):
    return (b, g, r)


def build_arrows(cam, t, P):
    """Returns list of dict(pts, prog, color, width, alpha)."""
    A = []

    def add(pts3, t0, dur, t_end, color, width=9, fade_len=0.7):
        prog = sstep(t0, t0 + dur, t)
        fade = 1.0 - sstep(t_end, t_end + fade_len, t)
        if prog > 0.01 and fade > 0.01:
            A.append(dict(pts=project(cam, pts3), prog=prog, color=color, width=width, alpha=fade))

    g = P["g"]
    zt = tower_top(g)
    ORG, CREAM, PINK, ICE, GOLD, SKY = (bgr(255, 146, 64), bgr(255, 238, 160), bgr(255, 196, 214),
                                       bgr(150, 232, 255), bgr(255, 204, 40), bgr(255, 226, 130))
    # 1. evaporation: warm, moist air rising from the sea
    for i, x in enumerate([-0.88, -0.58, -0.28, 0.28, 0.58, 0.88]):
        z = np.linspace(0.02, 0.30, 12)
        add(np.stack([x + 0.012 * np.sin(z * 14 + i), np.zeros(12), z], 1), 4.9 + i * 0.13, 1.3, 10.9, ORG, 8)
    # 2. circulation on the cut plane: inflow, updraft, outflow (both sides)
    for sgn in (1, -1):
        r = np.linspace(1.08, 0.36, 14)
        add(np.stack([sgn * r, np.zeros(14), np.full(14, 0.07)], 1), 11.0, 1.8, 99, CREAM, 8)
        if g > 0.25:
            zs = np.linspace(0.12, max(zt - 0.06, 0.14), 14)
            rc = np.array([wall_rc(z) * (0.55 + 0.45 * g) for z in zs])
            add(np.stack([sgn * rc, np.zeros(14), zs], 1), 12.8, 2.4, 22.6, ORG, 9)
        if P["anv"] > 0.15:
            r0 = wall_rc(zt) * (0.55 + 0.45 * g) + 0.06
            rr = np.linspace(r0, r0 + (ANV_R + 0.05 - r0) * clamp(P["anv"] * 1.1), 12)
            add(np.stack([sgn * rr, np.zeros(12), np.full(12, zt - 0.02)], 1), 14.6, 2.2, 99, PINK, 8)
    # 3. surface winds spiral in counter-clockwise (back half only)
    tw = sstep(15.5, 21.0, t)
    for k, a0 in enumerate([22, 68, 116]):
        s_ = np.linspace(0, 1, 26)
        r = 1.06 - 0.62 * s_
        th = np.radians(a0 + 52 * tw * s_ ** 1.1)
        add(np.stack([r * np.cos(th), r * np.sin(th), np.full(26, 0.07)], 1), 17.0 + k * 0.25, 2.0, 99, CREAM, 7)
    # 4. sinking air in the eye
    zz = np.linspace(0.80, 0.16, 12)
    add(np.stack([np.zeros(12), np.zeros(12), zz], 1), 19.8, 2.0, 99, SKY, 10)
    # 5. winds circling the eyewall
    for k, (R, z) in enumerate([(0.55, 0.30), (0.50, 0.62)]):
        add(ring3(R, z, 12, 168, 50, 0.02, k), 18.4 + k * 0.5, 2.2, 99, GOLD, 9)
    return A


def draw_arrows(frame, arrows, t):
    groups = {}
    for a in arrows:
        groups.setdefault(round(a["alpha"] * 10), []).append(a)
    for key, lst in sorted(groups.items(), reverse=True):
        al = key / 10.0
        if al <= 0:
            continue
        if key >= 10:
            for a in lst:
                flow(frame, a["pts"], a["prog"], a["color"], a["width"], t)
        else:
            ov = frame.copy()
            for a in lst:
                flow(ov, a["pts"], a["prog"], a["color"], a["width"], t)
            frame[:] = cv2.addWeighted(ov, al, frame, 1 - al, 0)


def draw_vapor(frame, cam, t, P):
    vis = sstep(4.4, 5.6, t) * (1 - sstep(10.8, 13.5, t))
    if vis < 0.02:
        return
    ov = frame.copy()
    rng = np.random.default_rng(11)
    for i in range(46):
        ang = rng.uniform(0.0, math.pi)
        rad = rng.uniform(0.05, 1.0)
        x0, y0 = rad * math.cos(ang), rad * math.sin(ang)
        per, ph = rng.uniform(3.0, 4.6), rng.uniform(0, 10)
        u = ((t + ph) % per) / per
        z = 0.03 + 0.30 * u
        sw = 0.025 * math.sin(u * 9 + i)
        px, py = cam.P(x0 + sw, y0, z)
        a = math.sin(u * math.pi)
        rad_px = int(3 + 5 * a)
        cv2.circle(ov, (int(px), int(py)), rad_px, (236, 238, 242), -1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(ov, 0.38 * vis, frame, 1 - 0.38 * vis, 0)


# ----------------------------------------------------------------------------
# satellite inset
# ----------------------------------------------------------------------------
INS_R = 96
_iy, _ix = np.mgrid[-INS_R:INS_R, -INS_R:INS_R].astype(np.float32)
INS_RAD = np.sqrt(_ix ** 2 + _iy ** 2) / INS_R
INS_TH = np.arctan2(-_iy, _ix)
INS_MASK = np.clip((1 - INS_RAD) * INS_R * 0.8, 0, 1).astype(np.float32)


def draw_inset(frame, t, P, rot, alpha):
    if alpha <= 0.01:
        return
    r, th = INS_RAD, INS_TH
    ph = th - rot * 0.9 + 3.0 * np.log(r + 0.12)
    arms = 0.5 + 0.5 * np.cos(3 * ph)
    u = ((ph / (2 * math.pi)) % 1.0 * PN.shape[1]).astype(np.int32) % PN.shape[1]
    v = (r * 300).astype(np.int32) % PN.shape[0]
    tex = PN[v, u]
    org = P["spin"]
    env = sstep_np(1.0, 0.30, r)
    dens = env * (0.18 + 0.82 * (0.55 * tex + 0.45 * (0.5 + (arms - 0.5) * (0.3 + 0.7 * org)))) * sstep(8.6, 14.0, t)
    re = 0.05 + 0.07 * P["e"]
    dens = dens * sstep_np(re, re + 0.07, r)
    ring = np.exp(-((r - (re + 0.11)) / 0.075) ** 2) * (0.25 + 0.75 * P["e"]) * sstep(10.0, 16.0, t)
    dens = np.clip(dens * 1.25 + ring * 0.55, 0, 1)[..., None]
    ocean = rgb(16, 52, 92) * (0.85 + 0.3 * NB[:2 * INS_R, :2 * INS_R, None])
    col = ocean * (1 - dens) + rgb(238, 243, 250) * dens
    icx, icy = 1760, 150
    x0, y0 = icx - INS_R, icy - INS_R
    blend_patch(frame, col.astype(np.float32), INS_MASK * alpha, x0, y0)
    cv2.circle(frame, (icx, icy), INS_R, (235, 235, 235), 2, cv2.LINE_AA)
    bgr_, al, w, h = INSET_CAP
    blend_patch(frame, bgr_, al * alpha, int(icx - w / 2), icy + INS_R + 8)


# ----------------------------------------------------------------------------
# frame
# ----------------------------------------------------------------------------
def draw_labels(frame, cam, t):
    for L in LABELS:
        fi = sstep(L["t0"], L["t0"] + 0.5, t)
        fo = 1 - sstep(L["t1"] - 0.5, L["t1"], t)
        a = fi * fo
        if a < 0.01:
            continue
        w, h = L["w"], L["h"]
        x = 40 if L["side"] == "L" else 1664
        slide = (1 - fi) * (-14 if L["side"] == "L" else 14)
        y = L["y"]
        blend_patch(frame, L["bgr"], L["a"] * a, int(x + slide), y)
        ax, ay = cam.P(*L["anchor"])
        sx = x + w + 4 if L["side"] == "L" else x - 6
        sy = y + 22
        lp = sstep(L["t0"] + 0.15, L["t0"] + 0.85, t)
        ex, ey = sx + (ax - sx) * lp, sy + (ay - sy) * lp
        ov = frame.copy()
        cv2.line(ov, (int(sx), int(sy)), (int(ex), int(ey)), (10, 12, 18), 4, cv2.LINE_AA)
        cv2.line(ov, (int(sx), int(sy)), (int(ex), int(ey)), (0, 204, 255), 2, cv2.LINE_AA)
        if lp > 0.98:
            cv2.circle(ov, (int(ax), int(ay)), 8, (10, 12, 18), -1, cv2.LINE_AA)
            cv2.circle(ov, (int(ax), int(ay)), 5, (0, 204, 255), -1, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(ov, 0.9 * a, frame, 1 - 0.9 * a, 0)
        if L.get("chip"):
            cw, ch = 210, 12
            cx0, cy0_ = int(x + slide) + 22, y + h + 4
            d = np.linspace(0, 1, cw, dtype=np.float32)
            colr = lut(STOPS_WARM, d)[None, :, :].repeat(ch, 0)
            blend_patch(frame, colr, np.full((ch, cw), a, np.float32), cx0, cy0_)
            for (bgr_, al_, ww, hh), xx in zip(CHIP, (cx0, cx0 + cw - 40)):
                blend_patch(frame, bgr_, al_ * a, xx, cy0_ + ch + 2)


def draw_scales(frame, cam, t, vis):
    if vis < 0.01:
        return
    S, cs, cx, cy0 = cam.S, cam.cos, cam.cx, cam.cy0
    xl = int(cx - RO * S) + 4
    xr = int(cx + RO * S) - 4
    ytop = int(cy0 - ZTROP * cs * S)
    ov = frame.copy()
    cv2.line(ov, (xl, ytop), (xl, int(cy0)), (150, 172, 200), 1, cv2.LINE_AA)
    cv2.line(ov, (xr, ytop), (xr, int(cy0)), (150, 172, 200), 1, cv2.LINE_AA)
    for z, key in [(0.95, "15 km"), (0.633, "10 km"), (0.317, "5 km")]:
        y = int(cy0 - z * cs * S)
        cv2.line(ov, (xl, y), (xl + 12, y), (190, 205, 225), 2, cv2.LINE_AA)
        bgr_, al, w, h = SCALE_TXT[key]
        blend_patch(ov, bgr_, al * 0.85, xl + 16, y - h // 2)
    for xx in range(xl + 80, xr - 12, 26):
        cv2.line(ov, (xx, ytop), (xx + 12, ytop), (170, 190, 215), 1, cv2.LINE_AA)
    bgr_, al, w, h = SCALE_TXT["TROPOPAUSE"]
    blend_patch(ov, bgr_, al * 0.8, xr - 12 - w, ytop - h - 2)
    hpx = DEPTH * cs * S
    for d, key in [(0.45, "50 m"), (0.95, "100 m")]:
        yy = int(cy0 + d * hpx)
        cv2.line(ov, (xr + 4, yy), (xr + 16, yy), (235, 235, 235), 2, cv2.LINE_AA)
        bgr_, al, w, h = SCALE_TXT[key]
        blend_patch(ov, bgr_, al * 0.95, xr + 18, yy - h // 2)
    frame[:] = cv2.addWeighted(ov, vis, frame, 1 - vis, 0)


def render(t, idx=0):
    cam = Cam(t)
    P = prm(t)
    rot = rot_at(t)
    frame = BG.copy()
    fade_in = sstep(0.0, 0.6, t) * (1 - sstep(29.35, 30.0, t))

    draw_shadow(frame, cam)
    draw_sea_surface(frame, cam, t, P)
    draw_puffs(frame, cam, t, P)
    if P["spin"] > 0.001 or P["g"] > 0.05:
        deck_a = sstep(9.0, 15.0, t)
        if deck_a > 0.01:
            plane_layer(frame, cam, t, 0.10, 0.30, 1.12, rot, "deck", P, deck_a)
    draw_eye_wall(frame, cam, t, P, rot)
    if P["anv"] > 0.02:
        zt = tower_top(P["g"])
        zn_t = min(zt / 0.85, 1.3)
        ro_t = ((0.07 + 0.19 * zn_t ** 1.25) + (0.20 - 0.07 * min(zn_t, 1.0))) * (0.55 + 0.45 * P["g"])
        rmax = ro_t + (ANV_R - ro_t) * P["anv"]
        plane_layer(frame, cam, t, zt - 0.02, 0.0, rmax, rot, "roof", P)
    x0, x1, y0, hpx = draw_sea_cut(frame, cam, t, P)
    draw_cut_clouds(frame, cam, t, P)
    draw_scales(frame, cam, t, sstep(3.6, 4.8, t) * (1 - sstep(29.0, 29.6, t)))
    draw_vapor(frame, cam, t, P)
    draw_arrows(frame, build_arrows(cam, t, P), t)
    draw_inset(frame, t, P, rot, sstep(15.0, 16.2, t) * (1 - sstep(29.0, 29.5, t)))
    draw_labels(frame, cam, t)

    # captions
    for (a, b, *_), (bgr, al, w, h) in zip(CAPTIONS, CAP_IMG):
        fi, fo = sstep(a, a + 0.55, t), 1 - sstep(b - 0.45, b, t)
        if fi * fo > 0.01:
            blend_patch(frame, bgr, al * fi * fo, 96, int(944 + (1 - fi) * 16))
    # progress ticks
    for i, (a, b, *_) in enumerate(CAPTIONS if t > 3.9 else []):
        x = 1920 - 96 - (4 - i) * 84
        cv2.rectangle(frame, (x, 1012), (x + 70, 1017), (70, 78, 92), -1)
        p = clamp((t - a) / (b - a))
        if p > 0:
            cv2.rectangle(frame, (x, 1012), (x + int(70 * p), 1017), (0, 204, 255), -1)
    # title card then the small tag
    ti = sstep(0.4, 1.1, t) * (1 - sstep(3.3, 4.0, t))
    if ti > 0.01:
        cv2.rectangle(frame, (100, 112), (100 + int(170 * sstep(0.4, 1.1, t)), 124), (0, 204, 255), -1)
        for k, (bgr, al, w, h) in enumerate([TITLE]):
            blend_patch(frame, bgr, al * ti, 92, 150 + int((1 - ti) * 12))
        blend_patch(frame, SUBT[0], SUBT[1] * ti * sstep(1.2, 1.9, t), 98, 150 + TITLE[3] + 2)
    tg = sstep(4.0, 4.7, t) * (1 - sstep(29.0, 29.5, t))
    if tg > 0.01:
        cv2.rectangle(frame, (64, 60), (64 + 44, 66), (0, 204, 255), -1)
        blend_patch(frame, TAG[0], TAG[1] * tg, 60, 74)

    # grade, vignette, grain, fade
    frame = cv2.multiply(frame, VIG, scale=1 / 255.0)
    frame = cv2.addWeighted(frame, 1.0, cv2.merge([GRAIN[idx % 4]] * 3), 0.55, -0.55 * 128)
    if fade_in < 0.999:
        frame = (frame.astype(np.float32) * fade_in).astype(np.uint8)
    return frame


# ----------------------------------------------------------------------------
# encoding
# ----------------------------------------------------------------------------
ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p",
       "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
       "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-g", "60"]


def render_segment(a, b, out):
    tmp = out + ".part.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-"] + ENC + [tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(a, b):
        p.stdin.write(render(i / FPS, i).tobytes())
        if (i - a) % 30 == 0:
            print(f"  frame {i}/{b}", flush=True)
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit("ffmpeg failed")
    os.replace(tmp, out)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--preview":
        os.makedirs(os.path.join(HERE, "preview"), exist_ok=True)
        for s in args[1:]:
            t = float(s)
            cv2.imwrite(os.path.join(HERE, "preview", f"t{t:05.2f}.png"), render(t, int(t * 30)))
        return
    if args and args[0] == "--seg":
        render_segment(int(args[1]), int(args[2]), args[3])
        return
    out = args[0] if args else os.path.join(HERE, "hurricane.mp4")
    total = int(DUR * FPS)
    work = os.path.join(os.path.dirname(os.path.abspath(out)), "_seg")
    os.makedirs(work, exist_ok=True)
    parts = [(0, total // 2), (total // 2, total)]
    procs = []
    for k, (a, b) in enumerate(parts):
        f = os.path.join(work, f"seg{k}.mp4")
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "--seg", str(a), str(b), f]))
    if any(p.wait() != 0 for p in procs):
        raise SystemExit("segment render failed")
    lst = os.path.join(work, "list.txt")
    with open(lst, "w") as fh:
        for k in range(len(parts)):
            fh.write(f"file '{os.path.join(work, f'seg{k}.mp4')}'\n")
    silent = os.path.join(work, "silent.mp4")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent])
    wav = os.path.join(HERE, "hurricane.wav")
    if os.path.exists(wav):
        tmp = out + ".part.mp4"
        subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-map", "0:v", "-map", "1:a",
                               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest",
                               "-movflags", "+faststart", tmp])
        os.replace(tmp, out)
    else:
        os.replace(silent, out)


if __name__ == "__main__":
    main()
