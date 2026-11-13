#!/usr/bin/env python3
"""Compound interest explainer: 20 s, 1920x1080, 30 fps, flat 2D green and cream.

Python, Pillow and ffmpeg only.

  python3 music.py                        # writes audio.wav
  python3 render.py video out.mp4 audio.wav
  python3 render.py frame 12.5 out.png    # preview one frame
"""
import math
import os
import random
import subprocess
import sys
from multiprocessing import Pool

from PIL import Image, ImageDraw, ImageFont

import timeline as TL

W, H, S = 1920, 1080, 2          # output size and supersampling factor
FD = '/usr/share/fonts/truetype/google-fonts/'

# palette: green and cream
BG = (247, 241, 222)
CARD = (255, 251, 236)
CARD_EDGE = (231, 223, 188)
G_DARK = (31, 92, 58)
G_MID = (62, 142, 90)
G_SOFT = (143, 193, 120)
G_PALE = (221, 235, 193)
OUT = (36, 104, 66)
FACE = (255, 241, 184)
RING = (236, 212, 118)
SIDE = (76, 160, 104)
SIDE_D = (52, 124, 80)
C_DARK = (72, 158, 96)
C_MID = (104, 184, 112)
C_LIGHT = (164, 217, 142)
CREAM_TXT = (255, 248, 226)
MUTED = (96, 142, 108)

CX = 520                 # tree / stack centre x
R = 100                  # coin radius
THICK = 24               # coin thickness
RYF = 0.42               # flat coin ellipse ratio
BASE = 860               # y of bottom ellipse centre of slot 0
STAND = 906 - R          # standing coin centre y

# ---------------------------------------------------------------- helpers


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def mix(c1, c2, t):
    t = clamp(t)
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c1, c2))


def eo3(p):
    p = clamp(p)
    return 1 - (1 - p) ** 3


def eio(p):
    p = clamp(p)
    return p * p * (3 - 2 * p)


def eob(p, c1=1.70158):
    p = clamp(p)
    c3 = c1 + 1
    return 1 + c3 * (p - 1) ** 3 + c1 * (p - 1) ** 2


def ebounce(p):
    p = clamp(p)
    n1, d1 = 7.5625, 2.75
    if p < 1 / d1:
        return n1 * p * p
    if p < 2 / d1:
        p -= 1.5 / d1
        return n1 * p * p + 0.75
    if p < 2.5 / d1:
        p -= 2.25 / d1
        return n1 * p * p + 0.9375
    p -= 2.625 / d1
    return n1 * p * p + 0.984375


_fonts = {}


def font(name, size):
    k = (name, size)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FD + 'Poppins-%s.ttf' % name, int(round(size * S)))
    return _fonts[k]


class G:
    """Tiny drawing wrapper that works in 1920x1080 coordinates on a 2x canvas."""

    def __init__(self, img):
        self.img = img
        self.d = ImageDraw.Draw(img)

    def ell(self, cx, cy, rx, ry, fill=None, out=None, w=0):
        self.d.ellipse([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S],
                       fill=fill, outline=out, width=int(w * S) if out else 0)

    def rect(self, x0, y0, x1, y1, fill):
        self.d.rectangle([x0 * S, y0 * S, x1 * S, y1 * S], fill=fill)

    def rrect(self, x0, y0, x1, y1, r, fill=None, out=None, w=0):
        self.d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=r * S,
                                 fill=fill, outline=out, width=int(w * S) if out else 0)

    def line(self, pts, fill, w, joint=None):
        self.d.line([(x * S, y * S) for x, y in pts], fill=fill, width=int(w * S), joint=joint)

    def poly(self, pts, fill):
        self.d.polygon([(x * S, y * S) for x, y in pts], fill=fill)

    def text(self, x, y, s, f, fill, anchor='ls'):
        self.d.text((x * S, y * S), s, font=f, fill=fill, anchor=anchor)

    def dot(self, x, y, r, fill, out=None, w=0):
        self.ell(x, y, r, r, fill, out, w)


def tlen(s, f):
    return f.getlength(s) / S


def pill(g, cx, cy, s, f, fg, bg, padx=26, h=52, out=None):
    w = tlen(s, f) + 2 * padx
    g.rrect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, h / 2, fill=bg, out=out, w=4 if out else 0)
    g.text(cx, cy + 1, s, f, fg, 'mm')
    return w


def star(g, cx, cy, r, fill):
    pts = []
    for i in range(8):
        a = math.pi / 4 * i - math.pi / 2
        rr = r if i % 2 == 0 else r * 0.32
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    g.poly(pts, fill)


def dashed(g, p0, p1, dash, gap, w, col):
    (x0, y0), (x1, y1) = p0, p1
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 1:
        return
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    s = 0.0
    while s < L:
        e = min(L, s + dash)
        g.line([(x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)], col, w)
        s += dash + gap


# ---------------------------------------------------------------- coin art

_face = {}


def face_tex(blink):
    if blink in _face:
        return _face[blink]
    T = int(2 * R * S)
    T2 = T * 2
    c = T2 / 2
    u = R * S * 2
    im = Image.new('RGBA', (T2, T2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FD + 'Poppins-Bold.ttf', int(1.6 * u))
    d.text((c, c + 0.06 * u), '$', font=f, fill=(246, 224, 138, 255), anchor='mm')
    dk = G_DARK + (255,)
    for sx in (-1, 1):
        ex, ey = c + sx * 0.30 * u, c - 0.10 * u
        if blink:
            d.arc([ex - 0.10 * u, ey - 0.06 * u, ex + 0.10 * u, ey + 0.14 * u], 200, 340, fill=dk, width=int(0.05 * u))
        else:
            d.ellipse([ex - 0.075 * u, ey - 0.115 * u, ex + 0.075 * u, ey + 0.115 * u], fill=dk)
            d.ellipse([ex - 0.035 * u, ey - 0.085 * u, ex + 0.015 * u, ey - 0.035 * u], fill=(255, 255, 255, 255))
        d.ellipse([c + sx * 0.52 * u - 0.1 * u, c + 0.14 * u - 0.07 * u,
                   c + sx * 0.52 * u + 0.1 * u, c + 0.14 * u + 0.07 * u], fill=(176, 222, 150, 255))
    d.arc([c - 0.24 * u, c + 0.0 * u, c + 0.24 * u, c + 0.40 * u], 20, 160, fill=dk, width=int(0.05 * u))
    im = im.resize((T, T), Image.LANCZOS)
    _face[blink] = im
    return im


def coin(g, cx, ytop, ry, th, blink=None):
    """A coin whose top face is an ellipse centred (cx, ytop); th = visible side band."""
    if th > 0.5:
        g.ell(cx, ytop + th, R, ry, fill=SIDE, out=OUT, w=5)
        g.rect(cx - R, ytop, cx + R, ytop + th, SIDE)
        for i in range(-4, 5):
            dx = R * i / 5 * 0.97
            yy = ytop + ry * math.sqrt(max(0.0, 1 - (dx / R) ** 2))
            g.line([(cx + dx, yy), (cx + dx, yy + th)], SIDE_D, 3)
        g.line([(cx - R, ytop), (cx - R, ytop + th)], OUT, 5)
        g.line([(cx + R, ytop), (cx + R, ytop + th)], OUT, 5)
    g.ell(cx, ytop, R, ry, fill=FACE, out=OUT, w=5)
    g.ell(cx, ytop, R * .80, ry * .80, out=RING, w=3)
    if blink is not None:
        tex = face_tex(blink)
        h = max(2, int(2 * ry * S))
        tex = tex.resize((tex.width, h), Image.BICUBIC)
        g.img.paste(tex, (int((cx - R) * S), int((ytop - ry) * S)), tex)


BLINKS = [(2.0, 2.16), (6.2, 6.36), (10.1, 10.26), (14.0, 14.16), (18.2, 18.36)]


def blinking(t):
    return any(a <= t < b for a, b in BLINKS)


# ---------------------------------------------------------------- static background

_bg = None


def background():
    global _bg
    if _bg is None:
        img = Image.new('RGB', (W * S, H * S), BG)
        g = G(img)
        g.dot(CX, 520, 340, (241, 241, 215))
        # ground mound
        g.ell(CX, 905, 390, 52, fill=(206, 226, 168), out=(146, 188, 118), w=5)
        for tx in (230, 300, 740, 812):
            for k, a in enumerate((-0.45, 0, 0.45)):
                g.line([(tx, 893 + (4 if k == 1 else 0)), (tx + math.sin(a) * 22, 893 - 24 + abs(a) * 8)], (118, 168, 96), 6)
        _bg = img
    return _bg.copy()


_bokeh = None


def bokeh(g, t):
    global _bokeh
    if _bokeh is None:
        rng = random.Random(7)
        _bokeh = [(rng.uniform(50, 990), rng.uniform(0, 1100), rng.uniform(14, 34), rng.uniform(10, 24)) for _ in range(14)]
    for x, ph, v, r in _bokeh:
        y = 1040 - ((t * v + ph) % 1120)
        if y < 300:
            continue
        g.dot(x + math.sin(t * 0.6 + ph) * 12, y, r, mix(BG, (226, 238, 198), 0.85))


# ---------------------------------------------------------------- scene pieces


def title(g, t):
    a = eo3(t / 0.9)
    off = (1 - a) * 50
    g.text(80 - off, 122, 'Compound interest', font('Bold', 70), mix(BG, G_DARK, a))
    a2 = eo3((t - 0.25) / 0.9)
    g.text(84 - (1 - a2) * 50, 170, '$1,000  •  10% a year  •  10 years', font('Medium', 30), mix(BG, MUTED, a2))


def pills(g, t):
    a = eo3((t - 1.0) / 0.6)
    if a <= 0:
        return
    y = TL.year_at(t)
    yr = int(y + 1e-9)
    bal = TL.balance(y)
    dy = (1 - a) * 30
    f = font('Bold', 30)
    g.rrect(80, 215 + dy, 250, 271 + dy, 28, fill=mix(BG, G_PALE, a))
    g.text(165, 244 + dy, 'Year %d' % yr, f, mix(BG, G_DARK, a), 'mm')
    g.rrect(266, 215 + dy, 470, 271 + dy, 28, fill=mix(BG, G_DARK, a))
    g.text(368, 244 + dy, '${:,.0f}'.format(bal), f, mix(BG, CREAM_TXT, a), 'mm')


def caption(g, t):
    for a, b, s in TL.CAPTIONS:
        if a <= t < b:
            fi = eo3((t - a) / 0.35)
            fo = 1 - eio((t - (b - 0.3)) / 0.3)
            al = min(fi, fo)
            if al <= 0.01:
                return
            f = font('Medium', 40)
            w = tlen(s, f) + 90
            cy = 992 + (1 - fi) * 16
            g.rrect(960 - w / 2, cy - 38, 960 + w / 2, cy + 38, 38, fill=mix(BG, G_DARK, al))
            g.text(960, cy + 2, s, f, mix(BG, CREAM_TXT, al), 'mm')
            return


# ---- chart

PX0, PX1, PY0, PY1, VMAX = 1130, 1790, 800, 330, 3000


def chart(g, t):
    a = eo3((t - 1.2) / 1.0)
    if a <= 0:
        return
    ox = (1 - a) * 900
    X = lambda yr: PX0 + yr / 10 * (PX1 - PX0) + ox
    Y = lambda v: PY0 - v / VMAX * (PY0 - PY1)
    g.rrect(1030 + ox + 6, 216, 1860 + ox + 6, 886, 38, fill=(236, 229, 200))   # flat shadow
    g.rrect(1030 + ox, 210, 1860 + ox, 880, 38, fill=CARD, out=CARD_EDGE, w=4)
    g.text(1070 + ox, 266, 'Your $1,000 over 10 years', font('Medium', 32), G_DARK)

    yc = TL.year_at(t)
    pts = []
    n = max(2, int(yc / 0.04) + 1)
    for i in range(n + 1):
        yr = yc * i / n
        pts.append((X(yr), Y(TL.balance(yr))))
    if yc > 0.02:
        g.poly([(X(0), PY0)] + pts + [(X(yc), PY0)], (232, 244, 208))

    for v, lab in ((1000, '$1k'), (2000, '$2k'), (3000, '$3k')):
        dashed(g, (PX0 + ox, Y(v)), (PX1 + ox, Y(v)), 10, 10, 2, (222, 226, 188))
        g.text(PX0 - 24 + ox, Y(v), lab, font('Medium', 24), MUTED, 'rm')
    g.line([(PX0 + ox, PY0), (PX1 + ox, PY0)], (186, 178, 140), 4)
    for yr in range(0, 11, 2):
        g.text(X(yr), PY0 + 34, str(yr), font('Medium', 24), MUTED, 'mm')
    g.text((PX0 + PX1) / 2 + ox, PY0 + 62, 'years', font('Medium', 24), MUTED, 'mm')

    # legend
    la = eo3((t - 3.2) / 0.6)
    if la > 0:
        lx, ly = 1150 + ox, 358
        col = lambda c: mix(CARD, c, la)
        g.line([(lx, ly), (lx + 46, ly)], col(G_DARK), 10)
        g.text(lx + 64, ly + 2, 'Compound interest', font('Bold', 26), col(G_DARK), 'lm')
        dashed(g, (lx, ly + 44), (lx + 46, ly + 44), 12, 8, 6, col(G_SOFT))
        g.text(lx + 64, ly + 46, 'Simple interest', font('Medium', 26), col(MUTED), 'lm')

    if yc <= 0.0:
        return
    # simple interest (dashed straight line)
    dashed(g, (X(0), Y(1000)), (X(yc), Y(TL.simple(yc))), 18, 14, 6, G_SOFT)
    # compound curve
    g.line(pts, G_DARK, 10, joint='curve')
    g.dot(pts[0][0], pts[0][1], 5, G_DARK)
    tx, ty = pts[-1]
    g.dot(X(yc), Y(TL.simple(yc)), 9, G_SOFT, CARD, 3)
    g.dot(tx, ty, 15, CREAM_TXT, G_DARK, 7)
    f = font('Bold', 28)
    s = '${:,.0f}'.format(TL.balance(yc))
    w = tlen(s, f) + 40
    right = tx + 34
    g.rrect(right - w, ty - 26 - 46, right, ty - 26, 23, fill=G_DARK)
    g.text(right - w / 2, ty - 26 - 22, s, f, CREAM_TXT, 'mm')

    # the gap, once year 10 lands
    ea = eob((t - TL.END_T - 0.1) / 0.5)
    if t > TL.END_T + 0.1:
        bx = X(10) + 28
        g.line([(bx - 10, Y(2000)), (bx, Y(2000)), (bx, Y(TL.balance(10))), (bx - 10, Y(TL.balance(10)))], G_DARK, 5, joint='curve')
        dashed(g, (bx, Y(2000) + 8), (bx, 585), 8, 8, 4, G_SOFT)
        f2 = font('Bold', 28)
        s2 = '+$594 from interest on interest'
        # compact pill sitting under the lines
        s2 = '+$594 extra'
        w2 = (tlen(s2, f2) + 44) * min(1, ea)
        g.rrect(bx + 12 - w2, 585, bx + 12, 635, 25, fill=G_MID)
        if ea > 0.7:
            g.text(bx + 12 - w2 / 2, 610, s2, f2, CREAM_TXT, 'mm')


# ---- stack and tree

CAN_BACK = [(-0.95, 0.20, 0.72), (0.95, 0.20, 0.72), (0.0, -0.60, 0.95), (-0.55, -0.15, 0.85), (0.55, -0.15, 0.85), (0.0, 0.25, 0.80)]
CAN_MID = [(-0.92, 0.14, 0.60), (0.90, 0.14, 0.60), (0.0, -0.63, 0.80), (-0.52, -0.22, 0.72), (0.50, -0.22, 0.72), (0.0, 0.20, 0.62)]
CAN_HI = [(-0.55, -0.62, 0.30), (0.20, -1.00, 0.22), (-1.00, -0.05, 0.20), (0.62, -0.32, 0.16)]
FRUIT = [(-0.75, 0.30), (0.72, 0.38), (0.05, 0.55), (-0.35, -0.50), (0.45, -0.58), (-1.02, 0.05), (1.0, 0.08), (0.0, -0.12), (-0.30, 0.08), (0.36, 0.10)]
K = 135


def canopy_scale(y):
    if y < TL.CANOPY_Y:
        return 0.0
    p = (y - TL.CANOPY_Y) / 1.2
    a = 1.1 ** TL.CANOPY_Y
    grow = max(0.0, (1.1 ** y - a) / (1.1 ** 10 - a))
    return eob(p) * 0.6 + 0.4 * grow


def hero_state(t):
    """Returns (ytop, ry, th) for the hero coin."""
    if t < TL.HERO_DROP:
        return None
    p = (t - TL.HERO_DROP) / 1.0
    y = lerp(-250, STAND, ebounce(p))
    u = eio((t - TL.TIP_T) / TL.TIP_D)
    ctimes = TL.coin_times()
    slot = sum(eob((t - ta) / 0.5) for ta in ctimes if t >= ta)
    flat = BASE - THICK * (1 + slot)
    ytop = lerp(y, flat, u) - 60 * math.sin(math.pi * u)
    return ytop, R * (1 - u * (1 - RYF)), THICK * u


def stack(g, t):
    hs = hero_state(t)
    if hs is None:
        return None
    ytop_h, ry_h, th_h = hs
    ctimes = TL.coin_times()
    # shadow
    sw = R * 0.95 * (clamp(1 - (STAND - ytop_h) / 1100, 0.3, 1) if t < 1.3 else 1)
    g.ell(CX, 896, sw, 17, fill=(196, 218, 158))
    coins = []
    for k, ta in enumerate(ctimes, start=1):
        if t < ta:
            continue
        slot = sum(eob((t - tb) / 0.5) for tb in ctimes[k:] if t >= tb)
        xoff = -(1 - eo3((t - ta) / 0.45)) * 640
        coins.append((slot, CX + xoff, BASE - THICK * (1 + slot)))
    coins.sort(key=lambda c: c[0])
    for slot, x, yt in coins:
        coin(g, x, yt, R * RYF, THICK)
    coin(g, CX, ytop_h, ry_h, th_h, blinking(t))
    return ytop_h


def canopy(g, t, ytop):
    y = TL.year_at(t)
    sc = canopy_scale(y)
    if sc <= 0.01 or ytop is None:
        return 0.0
    k = K * sc
    cy = ytop - 0.62 * k
    sw = math.sin(t * 1.7) * 5 * sc

    def pos(dx, dy):
        return CX + (dx + 0) * k + sw * (1 - dy) * 0.6, cy + dy * k

    for dx, dy, r in CAN_BACK:
        x, yy = pos(dx, dy)
        g.dot(x, yy, r * k + 6, OUT)
    for dx, dy, r in CAN_BACK:
        x, yy = pos(dx, dy)
        g.dot(x, yy, r * k, C_DARK)
    for dx, dy, r in CAN_MID:
        x, yy = pos(dx - 0.03, dy - 0.05)
        g.dot(x, yy, r * k, C_MID)
    for dx, dy, r in CAN_HI:
        x, yy = pos(dx, dy)
        g.dot(x, yy, r * k, C_LIGHT)
    nfruit = 0
    for i, (dx, dy) in enumerate(FRUIT):
        ta = 4.9 + 0.55 * i
        p = eob((y - ta) / 0.35)
        if p <= 0:
            continue
        x, yy = pos(dx, dy)
        rr = 17 * min(1.0, sc * 1.1) * p
        g.dot(x, yy, rr, FACE, OUT, 3)
        g.dot(x - rr * 0.25, yy - rr * 0.25, max(1, rr * 0.28), (255, 255, 255))
        nfruit += 1
    return sc


def effects(g, t, sc):
    # +10% bubbles and sparkles at each yearly tick
    ctimes = TL.coin_times()
    bx = CX + 220 + 150 * sc
    for Y in range(1, 11):
        a = t - TL.tick_time(Y)
        if 0 <= a < 1.2:
            rise = 80 * eo3(a / 1.2)
            al = 1 - eio((a - 0.7) / 0.5)
            pa = eob(a / 0.25)
            f = font('Bold', 34)
            w = (tlen('+10%', f) + 44) * min(1, pa)
            cy = 740 - rise
            g.rrect(bx - w / 2, cy - 28, bx + w / 2, cy + 28, 28, fill=mix(BG, G_MID, al))
            if pa > 0.7:
                g.text(bx, cy + 1, '+10%', f, mix(BG, CREAM_TXT, al), 'mm')
    for ta in ctimes:
        a = t - ta - 0.25
        if 0 <= a < 0.5:
            s = math.sin(math.pi * a / 0.5)
            for sx, sy, rr in ((CX + R + 26, 842, 16), (CX - R - 30, 826, 12), (CX + R - 10, 790, 10)):
                star(g, sx, sy - a * 30, rr * s, G_SOFT)
    # canopy pop
    a = t - TL.CANOPY_T
    if 0 <= a < 0.6:
        s = math.sin(math.pi * a / 0.6)
        for ang in range(0, 360, 45):
            rr = 90 + 110 * a / 0.6
            star(g, CX + rr * math.cos(math.radians(ang)) * 1.3, 650 + rr * math.sin(math.radians(ang)) * 0.8, 14 * s, G_SOFT)
    # twinkles around the finished tree
    if t > TL.END_T + 0.4:
        for i in range(7):
            ph = (t * 1.3 + i * 0.37) % 1.0
            s = math.sin(math.pi * ph)
            ang = i * 1.9 + 0.5
            x = CX + math.cos(ang) * (230 + 25 * (i % 3))
            yy = 470 + math.sin(ang) * 170
            star(g, x, yy, (10 + 5 * (i % 2)) * s, mix(BG, (255, 255, 255), 1) if i % 2 else G_SOFT)


# ---------------------------------------------------------------- frame


def render(t):
    img = background()
    g = G(img)
    bokeh(g, t)
    title(g, t)
    pills(g, t)
    chart(g, t)
    ytop = stack(g, t)
    sc = canopy(g, t, ytop)
    effects(g, t, sc)
    caption(g, t)
    out = img.reduce(S)
    if t < 0.3:                                    # soft fade in from cream
        out = Image.blend(Image.new('RGB', (W, H), BG), out, eio(t / 0.3))
    return out


# ---------------------------------------------------------------- video


def render_chunk(args):
    path, i0, i1 = args
    tmp = path + '.part.mp4'
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H),
           '-r', str(TL.FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
           '-pix_fmt', 'yuv420p', '-r', str(TL.FPS), tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(i0, i1):
        p.stdin.write(render(i / TL.FPS).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit('ffmpeg failed for ' + path)
    os.rename(tmp, path)
    return path


def make_video(out, wav=None, workdir='/tmp/ci_parts', chunk=60, procs=2):
    os.makedirs(workdir, exist_ok=True)
    total = int(TL.DUR * TL.FPS)
    jobs = []
    for n, i0 in enumerate(range(0, total, chunk)):
        path = os.path.join(workdir, 'part%02d.mp4' % n)
        if not os.path.exists(path):              # reuse finished parts
            jobs.append((path, i0, min(total, i0 + chunk)))
    with Pool(procs) as pool:
        for done in pool.imap_unordered(render_chunk, jobs):
            print('done', done, flush=True)
    parts = sorted(f for f in os.listdir(workdir) if f.startswith('part') and f.endswith('.mp4') and '.part' not in f[4:])
    lst = os.path.join(workdir, 'list.txt')
    with open(lst, 'w') as f:
        for p in parts:
            f.write("file '%s'\n" % os.path.join(workdir, p))
    silent = os.path.join(workdir, 'silent.mp4')
    subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', silent])
    tmp = out + '.part.mp4'
    if wav:
        subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-i', silent, '-i', wav, '-map', '0:v', '-map', '1:a',
                               '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', tmp])
    else:
        subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-i', silent, '-c', 'copy', '-movflags', '+faststart', tmp])
    os.rename(tmp, out)


if __name__ == '__main__':
    if sys.argv[1] == 'frame':
        render(float(sys.argv[2])).save(sys.argv[3])
    elif sys.argv[1] == 'video':
        make_video(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
