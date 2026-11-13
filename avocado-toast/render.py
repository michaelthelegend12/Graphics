#!/usr/bin/env python3
"""Avocado toast in 4 steps: 15 s vertical (1080x1920, 30 fps) flat top-down motion graphic.
Run music.py first (writes music.wav), then:  python3 render.py
Needs Python, numpy, Pillow and ffmpeg. Everything is drawn in code, no image files.
  python3 render.py --preview 60,150   writes PNGs of those frames to /tmp/avocado-preview instead."""
import os, sys, math, random, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool
from timeline import *

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1080, 1920
NFRAMES = int(round(TOTAL_BEATS * B * FPS))  # 450
SS = 3
FONT = "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"
INK = (29, 43, 38)
GREEN = (44, 91, 42)
WHITE = (255, 255, 255)
BG = [(255, 233, 176), (255, 158, 128), (248, 195, 216), (255, 217, 102), (159, 216, 242), (255, 233, 176)]
STAGE_Y = 1030

# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def prog(t, s, d): return clamp((t - s) / d)
def lerp(a, b, u): return a + (b - a) * u
def outCubic(u): return 1 - (1 - u) ** 3
def inCubic(u): return u ** 3
def inOut(u): return u * u * (3 - 2 * u)
def outBack(u, s=1.70158): u -= 1; return 1 + (s + 1) * u ** 3 + s * u ** 2
def inBack(u, s=1.70158): return (s + 1) * u ** 3 - s * u ** 2
def outElastic(u):
    if u <= 0: return 0.0
    if u >= 1: return 1.0
    return 2 ** (-10 * u) * math.sin((u * 10 - 0.75) * 2 * math.pi / 3) + 1
def outBounce(u):
    n, d = 7.5625, 2.75
    if u < 1 / d: return n * u * u
    if u < 2 / d: u -= 1.5 / d; return n * u * u + 0.75
    if u < 2.5 / d: u -= 2.25 / d; return n * u * u + 0.9375
    u -= 2.625 / d; return n * u * u + 0.984375
def frac(x): return x - math.floor(x)
def mix(c1, c2, u): return tuple(int(round(lerp(a, b, u))) for a, b in zip(c1, c2))
def tri(x): return abs(frac(x) * 4 - 2) - 1

# ---------------------------------------------------------------- sprite drawing
class Pen:
    def __init__(s, w, h):
        s.w, s.h = w, h
        s.im = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
        s.d = ImageDraw.Draw(s.im)
    def circ(s, cx, cy, r, col): s.d.ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS], fill=col)
    def ell(s, x0, y0, x1, y1, col): s.d.ellipse([x0 * SS, y0 * SS, x1 * SS, y1 * SS], fill=col)
    def rr(s, x0, y0, x1, y1, r, col): s.d.rounded_rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], radius=r * SS, fill=col)
    def poly(s, pts, col): s.d.polygon([(x * SS, y * SS) for x, y in pts], fill=col)
    def line(s, pts, col, wd): s.d.line([(x * SS, y * SS) for x, y in pts], fill=col, width=int(wd * SS), joint="curve")
    def pie(s, box, a0, a1, col): s.d.pieslice([v * SS for v in box], a0, a1, fill=col)
    def done(s): return s.im.resize((s.w, s.h), Image.LANCZOS)

def A(col, a=255): return tuple(col) + (a,)

def bread_shape(p, d, col):
    p.ell(30 + d, 40 + d, 290 - d, 300 - d, col)
    p.ell(230 + d, 40 + d, 490 - d, 300 - d, col)
    p.rr(45 + d, 170 + d, 475 - d, 490 - d, max(6, 70 - d), col)

def inside_bread(x, y, d):
    r = 130 - d
    if (x - 160) ** 2 + (y - 170) ** 2 < r * r or (x - 360) ** 2 + (y - 170) ** 2 < r * r: return True
    return 45 + d < x < 475 - d and 170 + d < y < 490 - d

def make_bread(toast):
    p = Pen(520, 520)
    cols = ((156, 90, 34), (207, 138, 54), (232, 169, 74)) if toast else ((212, 154, 85), (244, 223, 174), (250, 235, 196))
    pore = (186, 116, 44) if toast else (230, 202, 146)
    bread_shape(p, 0, A(cols[0])); bread_shape(p, 26, A(cols[1])); bread_shape(p, 58, A(cols[2]))
    rnd = random.Random(3)
    for _ in range(90):
        x, y = rnd.uniform(90, 430), rnd.uniform(70, 440)
        if inside_bread(x, y, 76):
            r = rnd.uniform(3, 8)
            p.ell(x - r, y - r * 0.7, x + r, y + r * 0.7, A(pore))
    if toast:  # golden toasted shine marks
        for x, y, l in ((150, 150, 70), (330, 140, 60)):
            p.line([(x, y), (x + l, y + 12)], A((244, 190, 98)), 9)
    return p.done()

def make_avocado(pit):
    p = Pen(400, 540)
    def pear(inset, col):
        n = 70
        for i in range(n + 1):
            t = i / n; s = t * t * (3 - 2 * t)
            p.circ(200, lerp(135, 335, t), lerp(92, 172, s) - inset, col)
    pear(0, A((44, 91, 42))); pear(12, A((79, 138, 52))); pear(30, A((168, 204, 85))); pear(64, A((220, 233, 140)))
    if pit:
        p.circ(200, 355, 84, A((122, 74, 40))); p.circ(178, 333, 27, A((160, 106, 62))); p.circ(166, 322, 8, A((205, 160, 120)))
    else:
        p.circ(200, 357, 86, A((201, 221, 127))); p.circ(200, 361, 72, A((190, 212, 115)))
    return p.done()

def make_pit():
    p = Pen(180, 180)
    p.circ(90, 90, 84, A((122, 74, 40))); p.circ(68, 66, 27, A((160, 106, 62))); p.circ(56, 55, 8, A((205, 160, 120)))
    return p.done()

def make_plate():
    p = Pen(840, 840)
    p.circ(420, 420, 400, A(WHITE)); p.circ(420, 420, 338, A((232, 228, 220))); p.circ(420, 420, 328, A((247, 245, 240)))
    return p.done()

def make_bowl():
    p = Pen(560, 560)
    p.circ(280, 280, 262, A(WHITE)); p.circ(280, 280, 246, A((226, 232, 236))); p.circ(280, 284, 222, A((221, 228, 232)))
    p.circ(280, 290, 208, A((238, 242, 244)))
    return p.done()

def blob(p, cx, cy, r, jit, seed, col, n=36):
    rnd = random.Random(seed); ph = [rnd.uniform(0, 6.28) for _ in range(4)]
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        rr_ = r * (1 + jit * (math.sin(3 * a + ph[0]) * 0.5 + math.sin(5 * a + ph[1]) * 0.35 + math.sin(7 * a + ph[2]) * 0.25))
        pts.append((cx + rr_ * math.cos(a), cy + rr_ * math.sin(a)))
    p.poly(pts, col)

def make_mash(smooth):
    p = Pen(440, 440); rnd = random.Random(11 if smooth else 12)
    if smooth:
        blob(p, 220, 220, 205, 0.04, 2, A((169, 205, 86)))
        for i in range(7):
            a = i * 0.9
            p.line([(220 + 60 * math.cos(a + k * 0.35) * (0.5 + k * 0.12), 220 + 60 * math.sin(a + k * 0.35) * (0.5 + k * 0.12)) for k in range(14)], A((186, 217, 104)), 10)
        for _ in range(26):
            x, y = 220 + rnd.uniform(-150, 150), 220 + rnd.uniform(-150, 150)
            if math.hypot(x - 220, y - 220) < 165: p.circ(x, y, rnd.uniform(3, 7), A((140, 180, 62)))
    else:
        blob(p, 220, 220, 205, 0.09, 1, A((150, 190, 72)))
        for k in range(18):
            a = rnd.uniform(0, 6.28); r = rnd.uniform(0, 140)
            blob(p, 220 + r * math.cos(a), 220 + r * math.sin(a), rnd.uniform(26, 50), 0.25, k, A(rnd.choice([(196, 221, 120), (127, 169, 56), (178, 208, 96), (214, 232, 150)])), 14)
        for _ in range(14):
            x, y = 220 + rnd.uniform(-150, 150), 220 + rnd.uniform(-150, 150)
            if math.hypot(x - 220, y - 220) < 170: p.circ(x, y, rnd.uniform(4, 8), A((92, 130, 40)))
    return p.done()

def make_fork():
    p = Pen(130, 640)
    def shapes(g, col):
        for dx in (-33, -11, 11, 33): p.rr(65 + dx - 8 - g, 10 - g, 65 + dx + 8 + g, 200 + g, 8 + g, col)
        p.rr(65 - 46 - g, 150 - g, 65 + 46 + g, 262 + g, 34 + g, col)
        p.poly([(65 - 15 - g, 250), (65 + 15 + g, 250), (65 + 19 + g, 330), (65 - 19 - g, 330)], col)
        p.rr(65 - 18 - g, 300 - g, 65 + 18 + g, 630 + g, 18 + g, col)
    shapes(5, A((150, 164, 172))); shapes(0, A((222, 229, 233)))
    p.rr(65 - 8, 340, 65 + 8, 600, 8, A((246, 249, 250)))
    return p.done()

def make_knife():
    p = Pen(110, 640)
    def blade(g, col): p.rr(55 - 34 - g, 10 - g, 55 + 34 + g, 380 + g, 34 + g, col)
    blade(5, A((150, 164, 172))); blade(0, A((222, 229, 233)))
    p.rr(55 - 24 - 4, 350 - 4, 55 + 24 + 4, 630 + 4, 24 + 4, A((90, 56, 30))); p.rr(55 - 24, 350, 55 + 24, 630, 24, A((140, 92, 52)))
    p.rr(55 - 34, 346, 55 + 34, 376, 12, A((190, 202, 209)))
    p.rr(55 - 6, 40, 55 + 6, 250, 6, A((246, 249, 250)))
    return p.done()

def make_egg():
    p = Pen(340, 340)
    for cx, cy, r in ((170, 170, 124), (100, 138, 72), (225, 125, 76), (222, 222, 82), (110, 224, 72), (140, 90, 60)):
        p.circ(cx, cy, r, A((240, 237, 228)))
    for cx, cy, r in ((170, 168, 118), (100, 136, 68), (225, 123, 72), (222, 220, 78), (110, 222, 68), (140, 88, 56)):
        p.circ(cx, cy, r, A(WHITE))
    p.circ(176, 172, 64, A((255, 170, 20))); p.circ(176, 172, 58, A((255, 190, 40)))
    p.ell(150, 140, 186, 160, A((255, 224, 130))); p.circ(200, 200, 6, A((255, 224, 130)))
    return p.done()

def make_leaf():
    p = Pen(130, 200)
    L, R = [], []
    for i in range(41):
        t = i / 40; w = (math.sin(math.pi * t) ** 0.8) * 52
        y = 12 + t * 176
        L.append((65 - w, y)); R.append((65 + w, y))
    p.poly(L + R[::-1], A((43, 140, 66)))
    p.poly([(65, 14)] + R + [(65, 188)], A((62, 168, 88)))
    p.line([(65, 20), (65, 196)], A((28, 100, 48)), 5)
    for k in range(1, 5):
        y = 20 + k * 36; p.line([(65, y + 10), (65 - 34, y - 10)], A((28, 100, 48)), 3); p.line([(65, y + 10), (65 + 34, y - 10)], A((28, 100, 48)), 3)
    return p.done()

def make_lemon():
    p = Pen(300, 170)
    p.pie((10, 10, 290, 290), 180, 360, A((244, 206, 46)))
    p.pie((30, 30, 270, 270), 180, 360, A((255, 245, 170)))
    for ang in (200, 225, 250, 270, 290, 315, 340):
        a = math.radians(ang); p.line([(150, 150), (150 + 122 * math.cos(a), 150 + 122 * math.sin(a))], A((255, 252, 220)), 6)
    p.circ(150, 150, 10, A((255, 252, 220)))
    return p.done()

def make_dollop():
    p = Pen(170, 140)
    blob(p, 85, 70, 58, 0.18, 5, A((150, 192, 67)), 24); blob(p, 78, 62, 36, 0.22, 6, A((181, 218, 106)), 20)
    return p.done()

def make_spread():
    p = Pen(520, 520); rnd = random.Random(21)
    bread_shape(p, 60, A((158, 203, 82)))
    sp = p.done()
    # texture: wavy lighter ridges and darker specks, masked to the shape
    t = Pen(520, 520)
    for k in range(12):
        y = 110 + k * 30
        t.line([(80 + i * 12, y + 7 * math.sin(i * 0.6 + k)) for i in range(34)], A((181, 222, 110)), 7)
    for _ in range(40):
        x, y = rnd.uniform(90, 430), rnd.uniform(100, 430)
        t.circ(x, y, rnd.uniform(2.5, 5), A((130, 176, 60)))
    tex = t.done()
    out = Image.composite(tex, sp, sp.getchannel("A").point(lambda v: 255 if v > 250 else 0)) if False else sp.copy()
    mask = sp.getchannel("A")
    tex_rgb = tex.convert("RGBA")
    texa = Image.fromarray((np.asarray(tex_rgb.getchannel("A")).astype(np.float32) * np.asarray(mask).astype(np.float32) / 255).astype(np.uint8))
    tex_rgb.putalpha(texa)
    out.alpha_composite(tex_rgb)
    return out

def make_flakes():
    out = []
    for i in range(6):
        p = Pen(30, 30); rnd = random.Random(i)
        p.rr(5, 11, 5 + 14 + rnd.uniform(0, 6), 11 + 7, 3, A(rnd.choice([(226, 48, 30), (200, 32, 26), (240, 90, 40)])))
        im = p.done().convert("RGBa").rotate(i * 33, resample=Image.BICUBIC).convert("RGBA")
        out.append(im)
    return out

def make_pepper():
    out = []
    for i in range(4):
        p = Pen(16, 16); r = 3 + i * 0.8
        p.circ(8, 8, r, A((38, 34, 34)))
        out.append(p.done())
    return out

def make_star(col):
    p = Pen(120, 120); r = 56; k = 0.2
    pts = [(0, -r), (k * r, -k * r), (r, 0), (k * r, k * r), (0, r), (-k * r, k * r), (-r, 0), (-k * r, -k * r)]
    p.poly([(60 + x, 60 + y) for x, y in pts], A(col))
    return p.done()

def make_badge():
    p = Pen(300, 300)
    p.circ(150, 150, 140, A(WHITE)); p.circ(150, 150, 124, A(INK))
    return p.done()

def make_glow():
    p = Pen(200, 200); p.circ(100, 100, 100, (255, 255, 255, 46)); return p.done()

_pill_cache = {}
def pill(w, h, col):
    k = (w, h, col)
    if k not in _pill_cache:
        p = Pen(w, h); p.rr(0, 0, w, h, h / 2, A(col)); _pill_cache[k] = p.done()
    return _pill_cache[k]

# text is drawn at 2x and placed at 0.5 scale so pops stay sharp
_fc, _tc = {}, {}
def font(sz):
    if sz not in _fc: _fc[sz] = ImageFont.truetype(FONT, sz)
    return _fc[sz]

def text_img(s, sz, col=INK, stroke=0, scol=WHITE, spacing=0):
    key = (s, sz, col, stroke, scol, spacing)
    if key in _tc: return _tc[key]
    f = font(sz * 2); sp = spacing * 2
    widths = [f.getlength(c) for c in s]
    tw = int(f.getlength(s) if not sp else sum(widths) + sp * (len(s) - 1))
    asc, desc = f.getmetrics(); pad = stroke * 2 + 16
    im = Image.new("RGBA", (tw + 2 * pad, asc + desc + 2 * pad), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    if sp:
        x = pad
        for c, wd in zip(s, widths):
            d.text((x, pad), c, font=f, fill=col, stroke_width=stroke * 2, stroke_fill=scol); x += wd + sp
    else:
        d.text((pad, pad), s, font=f, fill=col, stroke_width=stroke * 2, stroke_fill=scol)
    _tc[key] = im
    return im

_cc = {}
def chip_img(s, sz=46, col=WHITE, tcol=INK):
    key = (s, sz, col, tcol)
    if key in _cc: return _cc[key]
    f = font(sz * 2); tw = int(f.getlength(s)); asc, desc = f.getmetrics()
    w, h = tw + 130, (asc + desc) + 60
    p = Pen(w, h); p.rr(0, 0, w, h, h / 2, A(col)); im = p.done()
    ImageDraw.Draw(im).text((w // 2, h // 2 + 2), s, font=f, fill=tcol, anchor="mm")
    _cc[key] = im
    return im

def build():
    sp = {"plate": make_plate(), "bread0": make_bread(False), "bread1": make_bread(True),
          "avo": make_avocado(True), "avo0": make_avocado(False), "pit": make_pit(), "bowl": make_bowl(),
          "mash0": make_mash(False), "mash1": make_mash(True), "fork": make_fork(), "knife": make_knife(),
          "egg": make_egg(), "leaf": make_leaf(), "lemon": make_lemon(), "dollop": make_dollop(),
          "spread": make_spread(), "flakes": make_flakes(), "pepper": make_pepper(),
          "star_w": make_star((255, 255, 255)), "star_y": make_star((255, 214, 64)), "badge": make_badge(), "glow": make_glow()}
    a0, a1 = np.asarray(sp["bread0"]).astype(np.float32), np.asarray(sp["bread1"]).astype(np.float32)
    yy, xx = np.mgrid[0:520, 0:520]
    sp["bread_arr"] = (a0, a1, np.clip(np.hypot(xx - 260, yy - 265) / 300, 0, 1).astype(np.float32))
    sp["spread_arr"] = (np.asarray(sp["spread"]).astype(np.float32), yy.astype(np.float32), xx.astype(np.float32))
    return sp

SP = build()

def bread_img(p):
    if p <= 0: return SP["bread0"]
    if p >= 1: return SP["bread1"]
    a0, a1, g = SP["bread_arr"]
    m = np.clip((p * 1.55 - (1 - g)) * 5, 0, 1)[..., None]
    out = a0 + (a1 - a0) * m; out[..., 3] = a0[..., 3]
    return Image.fromarray(out.astype(np.uint8), "RGBA")

def spread_img(P, ph):
    base, yy, xx = SP["spread_arr"]
    edge = lerp(88, 450, P) + 8 * np.sin(xx / 18 + ph)
    out = base.copy(); out[..., 3] = base[..., 3] * np.clip((edge - yy) / 3, 0, 1)
    return Image.fromarray(out.astype(np.uint8), "RGBA")

# ---------------------------------------------------------------- compositing
def lut(f): return [int(i * f) for i in range(256)]

def place(cv, im, x, y, sx=1.0, sy=None, rot=0.0, alpha=1.0, shadow=-1.0, sh=0.16):
    """Draw sprite centred at (x, y). rot in degrees (counter-clockwise). shadow>=0 adds a flat offset shadow."""
    if sy is None: sy = sx
    if sx <= 0.01 or sy <= 0.01 or alpha <= 0.01: return
    w, h = im.size
    nw, nh = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    t = im if (nw, nh) == (w, h) else im.resize((nw, nh), Image.BILINEAR)
    if rot: t = t.convert("RGBa").rotate(rot, resample=Image.BICUBIC, expand=True).convert("RGBA")
    px, py = int(round(x - t.width / 2)), int(round(y - t.height / 2))
    if px > W or py > H or px + t.width < 0 or py + t.height < 0: return
    a = t.getchannel("A")
    if alpha < 1: a = a.point(lut(alpha))
    if shadow >= 0:
        k = (1 + shadow) * max(sx, sy)
        cv.paste((40, 30, 50), (px + int(9 * k), py + int(15 * k)), a.point(lut(sh)))
    cv.paste(t.convert("RGB"), (px, py), a)

def ptext(cv, s, sz, x, y, sc=1.0, col=INK, rot=0.0, alpha=1.0, stroke=0, scol=WHITE, spacing=0, sx=None):
    im = text_img(s, sz, col, stroke, scol, spacing)
    place(cv, im, x, y, 0.5 * sc * (1 if sx is None else sx), 0.5 * sc, rot, alpha)

def pchip(cv, s, x, y, sc=1.0, sz=46, col=WHITE, tcol=INK, alpha=1.0):
    place(cv, chip_img(s, sz, col, tcol), x, y, 0.5 * sc, alpha=alpha, shadow=0.2, sh=0.12)

def rot_vec(v, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return (v[0] * c + v[1] * s, -v[0] * s + v[1] * c)

def place_anchor(cv, im, anchor, target, rot, sc, **kw):
    """Place sprite so that the point at `anchor` (offset from sprite centre) lands on `target`."""
    ox, oy = rot_vec((anchor[0] * sc, anchor[1] * sc), rot)
    place(cv, im, target[0] - ox, target[1] - oy, sc, rot=rot, **kw)

def sparkle(cv, t, t0, gx, gy, n=6, rad=150, size=0.7, dur=0.8, seed=0):
    u = prog(t, t0, dur)
    if u <= 0 or u >= 1: return
    for j in range(n):
        ang = 2 * math.pi * j / n + seed
        r = rad * outCubic(u) * (0.75 + 0.25 * ((j * 37) % 3) / 2)
        s = math.sin(math.pi * u) * size * (0.7 + 0.3 * (j % 2))
        place(cv, SP["star_y" if j % 2 else "star_w"], gx + r * math.cos(ang), gy + r * math.sin(ang), s, rot=90 * u + j * 20)

def ring(cv, t, t0, gx, gy, r1, bgc, col=WHITE, wd=12, dur=0.7):
    u = prog(t, t0, dur)
    if u <= 0 or u >= 1: return
    r = r1 * outCubic(u)
    c = mix(col, bgc, inCubic(u))
    d = ImageDraw.Draw(cv)
    d.ellipse([gx - r, gy - r, gx + r, gy + r], outline=c, width=max(2, int(wd * (1 - u))))

# ---------------------------------------------------------------- scene state
def bg_index(t):
    k = 0
    for i, b in enumerate(S + [OUT]):
        if t >= b: k = i + 1
    return k

def bg_color(t): return BG[bg_index(t)]

def draw_bg(t):
    k = bg_index(t)
    base = BG[k]
    wipe = None
    if k >= 1:
        tb = (S + [OUT])[k - 1]
        u = prog(t, tb, 1.1)
        if u < 1: base = BG[k - 1]; wipe = (BG[k], 2300 * outCubic(u))
    cv = Image.new("RGB", (W, H), base)
    if wipe:
        col, r = wipe
        m = Image.new("L", (W, H // 1), 0)
        big = Image.new("L", (W * 2, H * 2), 0)
        ImageDraw.Draw(big).ellipse([(540 - r) * 2, (300 - r) * 2, (540 + r) * 2, (300 + r) * 2], fill=255)
        cv.paste(col, (0, 0), big.resize((W, H), Image.BILINEAR))
    return cv

BOK = [(120, 300, 0.7, 0.9), (940, 540, 1.1, 0.6), (200, 1500, 1.5, 0.5), (900, 1720, 0.9, 1.0), (500, 1800, 0.6, 1.3), (990, 1000, 0.6, 0.8), (70, 1050, 1.0, 0.7)]

def draw_bokeh(cv, t):
    sec = t * B
    for x, y, s, sp in BOK:
        yy = (y - sp * 60 * sec) % (H + 400) - 200
        place(cv, SP["glow"], x + 14 * math.sin(sec * 0.9 + x), yy, s)

def plate_state(t):
    if t < S[0]:
        u = prog(t, T_PLATE_IN, 1.3)
        return 540, lerp(2400, 1150, outBack(u, 1.2)), 0.95
    if t < S[2]:
        u = prog(t, S[0], 1.0)
        y = lerp(1150, STAGE_Y, outCubic(u)); sc = lerp(0.95, 1.0, outCubic(u))
        x = lerp(540, 1560, inBack(prog(t, T_PLATE_OUT1, 1.0), 1.2))
        return x, y, sc
    if t < OUT:
        u = prog(t, T_PLATE_IN2, 1.25)
        return lerp(-480, 540, outBack(u, 1.1)), STAGE_Y, 1.0
    u = prog(t, OUT, 0.9)
    pulse = 0.025 * math.exp(-6 * frac(t)) if t > OUT + 0.9 else 0
    return 540, lerp(STAGE_Y, 1090, outCubic(u)), lerp(1.0, 1.1, outBack(u, 2.0)) + pulse

BREAD_LOCAL = (0, -8)
EGG_LOCAL = (-8, 6)
rnd_ = random.Random(8)
def _scatter(n, seed, ry=180):
    r = random.Random(seed); pts = []
    while len(pts) < n:
        x, y = r.uniform(-175, 175), r.uniform(-ry, ry)
        if (x / 175) ** 2 + (y / ry) ** 2 < 1 and y > -150 + abs(x) * 0.0: pts.append((x, y - 8))
    return pts
CHILI = [(p, random.Random(i).uniform(0, 0.85), i % 6) for i, p in enumerate(_scatter(36, 31))]
PEPPER = [(p, random.Random(100 + i).uniform(0, 0.8), i % 4) for i, p in enumerate(_scatter(30, 41))]
DOLLOP_LOCAL = [(-100, -55), (95, -40), (0, 90)]
LEAF_LOCAL = [((-150, -135), 40, 0.0), ((-100, -178), -12, 0.25)]
LEMON_LOCAL = (236, 252)

def bowl_state(t):
    """x, y, scale, rot, mash scale"""
    if t < T_BOWL_IN: return None
    u = prog(t, T_BOWL_IN, 0.9)
    x, y, s = lerp(-460, 540, outBack(u, 1.1)), 1290, 0.85
    uo = inOut(prog(t, T_BOWL_UP, 1.1))
    x, y, s = lerp(x, 810, uo), lerp(y, 745, uo), lerp(s, 0.62, uo)
    rot = 0
    if t >= S[2]:
        rot = lerp(0, -28, outCubic(prog(t, T_TILT, 0.4))) * (1 - inOut(prog(t, T_DOLLOP[2] + 0.4, 0.5)))
        x = lerp(x, 1600, inBack(prog(t, T_SPREAD_B + 0.1, 0.9), 1.2))
        if x > 1500: return None
    ms = 0.9 * outBack(prog(t, T_PLOP, 0.5), 2.2)
    ms *= 1 - 0.5 * prog(t, T_DOLLOP[0], T_DOLLOP[2] + 0.4 - T_DOLLOP[0])
    return x, y, s, rot, ms

# ---------------------------------------------------------------- scene drawing
def draw_header(cv, t):
    # intro title
    if t < S[0] + 0.1:
        ue = inBack(prog(t, T_TITLE_OUT, 0.7), 1.2); dy = -520 * ue
        def slam(t0): u = prog(t, t0, 0.55); return lerp(1.55, 1.0, outBack(u, 2.0)), clamp(u * 5)
        s, a = slam(T_TITLE1)
        ptext(cv, "AVOCADO", 150, 540, 350 + dy, s, GREEN, rot=0, alpha=a)
        s, a = slam(T_TITLE2)
        ptext(cv, "TOAST", 235, 540, 545 + dy, s, INK, rot=0, alpha=a)
        u = prog(t, T_PILL, 0.6)
        pchip(cv, "in 4 steps", 540, 735 + dy, outBack(u, 2.4), 62, INK, WHITE)
    # step badge + caption
    caps = ["TOAST IT", "MASH IT", "SPREAD IT", "TOP IT"]
    for k in range(4):
        ts, te = S[k], S[k] + 6
        if not (ts <= t < te): continue
        ub = prog(t, ts, 0.95); exit_ = 1 - prog(t, te - 0.4, 0.4) ** 2
        pulse = 1 + 0.06 * math.exp(-6 * frac(t))
        sc = outElastic(ub) * exit_ * pulse
        rot = 16 * (1 - ub) ** 2
        place(cv, SP["badge"], 540, 290, sc * 0.72, rot=rot, shadow=0.4)
        ptext(cv, str(k + 1), 130, 540, 290, sc * 1.0, WHITE, rot=rot)
        uc = prog(t, ts + 0.25, 0.7)
        cs = outBack(uc, 2.0) * exit_
        ptext(cv, caps[k], 112, 540, 470 - 40 * (1 - outCubic(uc)), cs, INK, rot=-4 * (1 - uc))
        chips = ["1 slice of sourdough", "1 ripe avocado", "thick and creamy", "egg  •  chili  •  pepper  •  lemon"]
        uchip = prog(t, ts + 0.9, 0.6)
        pchip(cv, chips[k], 540, 1565, outBack(uchip, 2.2) * exit_, 44)
    # outro
    if t >= OUT:
        u = prog(t, OUT + 0.1, 1.0); sc = outElastic(u)
        ptext(cv, "YUM!", 260, 540, 330, sc * (1 + 0.04 * math.exp(-6 * frac(t))), INK, rot=-3 * (1 - u))
        u2 = prog(t, OUT + 0.7, 0.6)
        ptext(cv, "avocado toast in 4 steps", 52, 540, 505, outBack(u2, 2.0), GREEN)
        u3 = prog(t, T_CTA, 0.6)
        pchip(cv, "SAVE FOR LATER", 540, 1700, outBack(u3, 2.4), 54, INK, WHITE)

def draw_dots(cv, t):
    if t < S[0] + 0.2: return
    out = 1 - prog(t, OUT, 0.5)
    if out <= 0: return
    cur = [prog(t, S[i] - 0.25, 0.5) * (1 - (prog(t, S[i + 1] - 0.25, 0.5) if i < 3 else prog(t, OUT - 0.25, 0.5))) for i in range(4)]
    done = [1 if t >= (S[i + 1] if i < 3 else OUT) - 0.25 else 0 for i in range(4)]
    ws = [lerp(26, 92, inOut(c)) * out for c in cur]
    total = sum(ws) + 18 * 3
    x = 540 - total / 2
    for i in range(4):
        w = max(2, int(ws[i])); col = INK if (done[i] or cur[i] > 0.05) else (255, 255, 255)
        place(cv, pill(max(w, 26), 26, col), x + w / 2, 1665, 1.0 if w >= 26 else w / 26)
        x += ws[i] + 18

def draw_stage(cv, t):
    px, py, ps = plate_state(t)
    bgc = bg_color(t)
    if -520 < px < 1600:
        place(cv, SP["plate"], px, py, ps, shadow=0.9, sh=0.17)
        def L(lx, ly): return px + lx * ps, py + ly * ps
        # ---- step 1: bread slides in, toasts
        if t >= T_BREAD_IN:
            u = prog(t, T_BREAD_IN, 0.9)
            gx, gy = L(*BREAD_LOCAL)
            x = lerp(1560, gx, outBack(u, 1.2)) if u < 1 else gx
            rot = lerp(22, 0, outCubic(u))
            tp = prog(t, T_TOAST_A, T_TOAST_B - T_TOAST_A)
            if T_TOAST_A <= t < T_DING: x += 3.5 * math.sin(2 * math.pi * t * 8)
            ud = prog(t, T_DING, 0.7); jump = math.sin(math.pi * ud) * 0.2 * (1 - ud * 0.5) if 0 < ud < 1 else 0
            for j in range(3):  # heat rings
                if T_TOAST_A <= t < T_DING + 0.5:
                    ring(cv, t, T_TOAST_A + 0.15 + j * 0.45, gx, gy, 330 * ps, bgc, (255, 196, 110), 14, 0.9)
            place(cv, bread_img(tp), x, gy, ps * (1 + jump), rot=rot, shadow=0.25 + jump * 5, sh=0.16)
            sparkle(cv, t, T_DING, gx, gy, 7, 230 * ps, 0.8, 0.9)
            ud2 = prog(t, T_DING, 1.3)
            if 0 < ud2 < 1:
                ptext(cv, "DING!", 92, gx + 215 * ps, gy - 250 * ps, outBack(prog(t, T_DING, 0.35), 3.0) * (1 - inCubic(prog(t, T_DING + 0.7, 0.6))), (255, 90, 54), rot=12, stroke=7, scol=WHITE)
        # ---- step 3: spread
        if t >= T_DOLLOP[0] and t >= S[2]:
            P = prog(t, T_SPREAD_A, T_SPREAD_B - T_SPREAD_A)
            if P > 0:
                bx, by = L(*BREAD_LOCAL)
                place(cv, spread_img(P, P * 9), bx, by, ps)
        # dollops
        if S[2] <= t < OUT:
            bs = bowl_state(t)
            for j, tl in enumerate(T_DOLLOP):
                u = prog(t, tl, 0.4)
                if u <= 0: continue
                lx, ly = DOLLOP_LOCAL[j]
                tx, ty = L(lx, ly + BREAD_LOCAL[1] + 8)
                fade = 1 - inOut(prog(t, T_SPREAD_A + 0.1 + j * 0.15, 0.7))
                if fade <= 0: continue
                if u < 1 and bs:
                    sx0, sy0 = bs[0] - 60, bs[1] + 30
                    x = lerp(sx0, tx, u); y = lerp(sy0, ty, u) - 170 * math.sin(math.pi * u)
                    place(cv, SP["dollop"], x, y, lerp(0.55, 0.9, u), rot=360 * u, shadow=0.8 * math.sin(math.pi * u))
                else:
                    lt = t - tl - 0.4
                    sq = math.exp(-7 * lt) * math.cos(14 * lt) * 0.2
                    place(cv, SP["dollop"], tx, ty, (0.9 + 0.35 * (1 - fade)) * ps, (0.9 + 0.35 * (1 - fade)) * ps, alpha=fade, shadow=0.1) if sq == 0 else \
                        place(cv, SP["dollop"], tx, ty, (0.9 + 0.35 * (1 - fade)) * ps * (1 + sq), (0.9 + 0.35 * (1 - fade)) * ps * (1 - sq), alpha=fade, shadow=0.1)
        # ---- step 4: egg, flakes, pepper, leaves, lemon
        if t >= T_EGG:
            gx, gy = L(*EGG_LOCAL)
            u1 = prog(t, T_EGG, 0.5); u2 = prog(t, T_EGG + 0.5, 0.45)
            x = lerp(1560, gx, outCubic(u1))
            s = lerp(1.5, 1.0, outBounce(u2)) * ps
            place(cv, SP["egg"], x, gy, s * 0.95, rot=lerp(30, 0, outCubic(u1)), shadow=0.25 + 1.5 * (1 - outBounce(u2)))
            sparkle(cv, t, T_EGG + 0.95, gx, gy, 6, 210 * ps, 0.7, 0.8, 0.3)
        for pts, t0, sprites, sc0, pre in ((CHILI, T_CHILI, SP["flakes"], 1.7, "f"), (PEPPER, T_PEPPER, SP["pepper"], 1.7, "p")):
            for (lx, ly), d, v in pts:
                u = prog(t, t0 + d, 0.35)
                if u <= 0: continue
                fx, fy = L(lx, ly)
                place(cv, sprites[v], fx + 40 * (1 - u), fy - 300 * (1 - u) ** 1.4, lerp(2.4, 1.0, inCubic(u)) * sc0 * ps / 1.7 * 1.0 if False else lerp(2.4, 1.0, inCubic(u)) * sc0, alpha=clamp(u * 3))
        for (ll, rot, dly) in LEAF_LOCAL:
            u = prog(t, T_LEAF + dly, 0.6)
            if u > 0:
                gx, gy = L(*ll)
                place(cv, SP["leaf"], lerp(-420, gx, outBack(u, 1.5)), lerp(gy - 80, gy, outCubic(u)), 0.85 * ps, rot=rot + 40 * (1 - u), shadow=0.3)
        u = prog(t, T_LEMON, 0.7)
        if u > 0:
            gx, gy = L(*LEMON_LOCAL)
            place(cv, SP["lemon"], lerp(1560, gx, outBack(u, 1.4)), lerp(gy + 500, gy, outBack(u, 1.4)), 0.9 * ps, rot=-24 + 30 * (1 - u), shadow=0.3)
            sparkle(cv, t, T_LEMON + 0.5, gx, gy, 5, 150, 0.6, 0.7, 0.8)
        # shine sweep over the finished spread
        us = prog(t, T_SHINE, 0.9)
        if 0 < us < 1 and t < T_EGG:
            bx, by = L(*BREAD_LOCAL)
            sx = lerp(-260, 260, us)
            place(cv, SP["star_w"], bx + sx * ps, by - 40 * ps, 1.3 * math.sin(math.pi * us) * ps, rot=45 * us)
            sparkle(cv, t, T_SHINE + 0.2, bx, by, 6, 200 * ps, 0.7, 0.8, 0.5)
    # ---- step 2: avocado, bowl, pit, fork
    if S[1] <= t < S[2] + 4.6:
        bs = bowl_state(t)
        if bs:
            bx, by, bsc, brot, ms = bs
            ph = (t - T_MASH_A) * 2; pr = (1 - frac(ph)) ** 3 if T_MASH_A - 0.5 <= t < T_FORK_OUT + 0.2 else 0
            plop = math.exp(-8 * (t - T_PLOP)) * math.cos(18 * (t - T_PLOP)) * 0.07 if t > T_PLOP else 0
            place(cv, SP["bowl"], bx, by, bsc * (1 + plop), bsc * (1 - plop), rot=brot, shadow=0.6)
            if ms > 0.02:
                c = prog(t, T_MASH_A, T_MASH_B - T_MASH_A)
                msc = bsc * ms * (1 + 0.03 * pr + plop)
                place(cv, SP["mash0"], bx, by, msc, rot=brot)
                place(cv, SP["mash1"], bx, by, msc, rot=brot, alpha=c)
            sparkle(cv, t, T_PLOP, bx, by, 6, 270 * bsc, 0.7, 0.8, 0.2)
        #AVO
        ua = prog(t, T_AVO_IN, 0.9)
        if ua > 0 and t < T_FLIP + 0.75:
            tt = t - T_PIT
            pop = 1 + (0.1 * math.exp(-9 * tt) * math.cos(16 * tt) if tt > 0 else 0)
            x, y, rot, s, al = lerp(1560, 540, outBack(ua, 1.2)), 840, lerp(40, 0, outCubic(ua)), 0.85, 1.0
            uf = prog(t, T_FLIP, 0.7)
            if uf > 0:
                e = inOut(uf); x, y, rot, s = 540, lerp(840, 1290, e), lerp(0, -170, e), lerp(0.85, 0.45, e); al = 1 - prog(uf, 0.8, 0.2)
            place(cv, SP["avo"] if t < T_PIT else SP["avo0"], x, y, s * pop, s * pop, rot=rot, alpha=al, shadow=0.35 + 0.5 * math.sin(math.pi * uf))
        # pit flies off
        up = prog(t, T_PIT, 1.1)
        if 0 < up < 1:
            x = 540 + 900 * up; y = 840 + 90 - 760 * up + 520 * up * up
            place(cv, SP["pit"], x, y, lerp(0.85, 1.5, up), rot=900 * up, shadow=1.2 * math.sin(math.pi * up) + 0.3)
        ring(cv, t, T_PIT, 540, 930, 300, bg_color(t), WHITE, 16, 0.6)
        if bs:
            bx, by, bsc, brot, ms = bs
            ph = (t - T_MASH_A) * 2; pr = (1 - frac(ph)) ** 3 if T_MASH_A - 0.5 <= t < T_FORK_OUT + 0.2 else 0
            # fork
            fin = outCubic(prog(t, T_FORK_IN, 0.5)); fout = inCubic(prog(t, T_FORK_OUT, 0.5))
            if fin > 0 and fout < 1 and t < S[2]:
                a = 2 * math.pi * (t - T_FORK_IN) * 1.0
                away = (1 - fin) + fout
                tx = bx + 55 * math.cos(a) + 650 * away; ty = by + 38 * math.sin(a) - 420 * away
                place_anchor(cv, SP["fork"], (0, -170), (tx, ty), 156 + 9 * math.sin(a), 0.95 * (1 - 0.07 * pr), shadow=0.5 * (1 - pr) + 0.2)
    # ---- step 3: knife
    if S[2] + 1.4 <= t < T_SPREAD_B + 1.0:
        px, py, ps = plate_state(t)
        P = prog(t, T_SPREAD_A, T_SPREAD_B - T_SPREAD_A)
        kx = 260 + 150 * tri(P * 3.5); ky = lerp(88, 450, P) + 10
        bx, by = px + BREAD_LOCAL[0] * ps, py + BREAD_LOCAL[1] * ps
        tip = (bx + (kx - 260) * ps, by + (ky - 260) * ps)
        uin = outCubic(prog(t, T_SPREAD_A - 0.55, 0.55)); uout = inCubic(prog(t, T_SPREAD_B + 0.1, 0.7))
        away = (1 - uin) + uout
        place_anchor(cv, SP["knife"], (0, -310), (tip[0] + 650 * away, tip[1] + 330 * away), 33 + 7 * math.sin(P * 40), 0.95, shadow=0.6)
    # ---- outro confetti
    if t >= T_CONFETTI:
        sec = (t - T_CONFETTI) * B
        for i, (vx, vy, col, w_, rotv) in enumerate(CONF):
            x = 540 + vx * sec; y = 1000 + vy * sec + 1500 * sec * sec
            if y > H + 60: continue
            place(cv, CONF_SPR[i % len(CONF_SPR)][col], x, y, 1.0, rot=rotv * sec * 360)

CONF_COLS = [(255, 90, 70), (255, 200, 60), (70, 190, 140), (60, 150, 230), (255, 140, 200), (255, 255, 255)]
def _conf():
    r = random.Random(77); out = []
    for _ in range(90):
        a = r.uniform(-math.pi * 0.95, -math.pi * 0.05); v = r.uniform(500, 1500)
        out.append((v * math.cos(a), v * math.sin(a), r.randrange(len(CONF_COLS)), r.uniform(0, 1), r.uniform(-2, 2)))
    return out
CONF = _conf()
def _confspr():
    res = []
    for shape in range(2):
        d = {}
        for ci, col in enumerate(CONF_COLS):
            p = Pen(40, 40)
            if shape == 0: p.rr(10, 4, 30, 36, 3, A(col))
            else: p.circ(20, 20, 11, A(col))
            d[ci] = p.done()
        res.append(d)
    return res
CONF_SPR = _confspr()

def frame(i):
    t = i / FPS / B
    cv = draw_bg(t)
    draw_bokeh(cv, t)
    draw_header(cv, t)
    draw_stage(cv, t)
    draw_dots(cv, t)
    return cv

# ---------------------------------------------------------------- rendering
ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS)]

def render_chunk(args):
    name, i0, i1 = args
    tmp = os.path.join(HERE, f"_{name}.part.mp4"); final = os.path.join(HERE, f"_{name}.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"] + ENC + [tmp],
                         stdin=subprocess.PIPE)
    for i in range(i0, i1): p.stdin.write(frame(i).tobytes())
    p.stdin.close()
    if p.wait() != 0: sys.exit("ffmpeg failed on " + name)
    os.replace(tmp, final)
    return final

def main():
    bounds = [0, 56, 141, 225, 309, 394, NFRAMES]
    chunks = [(f"c{k}", bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]
    with Pool(2) as pool: files = pool.map(render_chunk, chunks, chunksize=1)
    lst = os.path.join(HERE, "_list.txt")
    open(lst, "w").write("".join(f"file '{f}'\n" for f in files))
    silent = os.path.join(HERE, "_silent.mp4")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent])
    wav = os.path.join(HERE, "music.wav"); out = os.path.join(HERE, "avocado-toast.mp4")
    tmp = os.path.join(HERE, "_out.part.mp4")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                           "-c:a", "aac", "-b:a", "192k", "-t", str(TOTAL_BEATS * B), "-movflags", "+faststart", tmp])
    os.replace(tmp, out)
    for f in files + [lst, silent]: os.remove(f)
    print("wrote", out)

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--preview":
        os.makedirs("/tmp/avocado-preview", exist_ok=True)
        for n in sys.argv[2].split(","):
            frame(int(n)).save(f"/tmp/avocado-preview/f{int(n):03d}.png")
    else:
        main()
