#!/usr/bin/env python3
"""Hand-drawn whiteboard explainer: how vaccines train your immune system.

Everything is drawn as pen strokes (single-line hand lettering, wobbly outlines,
marker hatching) that reveal along their length, like someone sketching live.
Needs only numpy, opencv-python and ffmpeg.

    python3 render.py              -> vaccine-whiteboard.mp4 (1280x720, 30 fps, 30 s)
    python3 render.py preview 2.9 13 ...   -> PNG stills at those times
"""
import math, os, subprocess, sys
import numpy as np
import cv2
from multiprocessing import Pool

W, H, FPS = 1280, 720, 30


def C(h):  # '#RRGGBB' -> BGR tuple
    h = h.lstrip('#')
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


INK = C('#23262B'); RED = C('#D93B3B'); BLUE = C('#2D6CDF'); GREEN = C('#24A05A')
ORANGE = C('#F08A1C'); YELLOW = C('#F2B705')

# ----------------------------------------------------------------- geometry
def spline(P, closed=False, n=8, over=0):
    P = np.asarray(P, float); N = len(P); out = []
    for i in (range(N) if closed else range(N - 1)):
        p0 = P[(i - 1) % N] if (closed or i > 0) else P[0]
        p1 = P[i]; p2 = P[(i + 1) % N]
        p3 = P[(i + 2) % N] if (closed or i + 2 < N) else P[-1]
        for k in range(n):
            t = k / n
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(out[0] if closed else P[-1])
    out = np.array(out)
    if closed and over:
        out = np.vstack([out, out[1:1 + over]])
    return out


def resample(P, step=5.0):
    P = np.asarray(P, np.float32)
    if len(P) < 2:
        P = np.vstack([P, P])
    d = np.hypot(*np.diff(P, axis=0).T)
    L = float(d.sum())
    if L < 1e-3:
        return np.vstack([P[0], P[0] + [0.01, 0]]).astype(np.float32)
    cum = np.concatenate([[0], np.cumsum(d)])
    s = np.linspace(0, L, max(2, int(L / step) + 1))
    return np.stack([np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])], 1).astype(np.float32)


def blob(cx, cy, R, seed, n=72, amp=(0.05, 0.03)):
    rng = np.random.RandomState(seed); ph = rng.rand(2) * 6.28
    th = np.linspace(0, 2 * math.pi, n, endpoint=False) + rng.rand() * 6.28
    r = R * (1 + amp[0] * np.sin(3 * th + ph[0]) + amp[1] * np.sin(5 * th + ph[1]))
    return np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1)


def closed(P, over=4):
    return np.vstack([P, P[:over + 1] + [1.5, -1.5]])


def circ(cx, cy, r, n=14):
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return closed(np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1), 2)


# ----------------------------------------------------------------- hand lettering
def ell(w, n=9, ry=0.5, cy=0.5):
    return [(w / 2 + w / 2 * math.cos(a), cy + ry * math.sin(a))
            for a in [-math.pi / 2 + 2 * math.pi * k / n for k in range(n)]]

_B1 = [(0, 0), (0.5, 0.01), (0.72, 0.2), (0.55, 0.45), (0, 0.5)]
_P1 = [(0, 0), (0.52, 0.01), (0.74, 0.22), (0.55, 0.48), (0, 0.5)]
G = {
 'A': (.9, [('p', [(0, 1), (.45, 0), (.9, 1)]), ('p', [(.17, .66), (.73, .66)])]),
 'B': (.8, [('p', [(0, 1), (0, 0)]), ('s', _B1), ('s', [(0, .5), (.6, .52), (.8, .75), (.6, .98), (0, 1)])]),
 'C': (.85, [('s', [(.8, .17), (.5, 0), (.17, .18), (.05, .5), (.17, .82), (.5, 1), (.8, .83)])]),
 'D': (.85, [('p', [(0, 0), (0, 1)]), ('s', [(0, 0), (.5, .02), (.82, .3), (.82, .7), (.5, .98), (0, 1)])]),
 'E': (.75, [('p', [(.75, 0), (0, 0), (0, 1), (.75, 1)]), ('p', [(0, .5), (.55, .5)])]),
 'F': (.7, [('p', [(.75, 0), (0, 0), (0, 1)]), ('p', [(0, .5), (.55, .5)])]),
 'G': (.85, [('s', [(.8, .17), (.5, 0), (.17, .18), (.05, .5), (.17, .82), (.5, 1), (.82, .85), (.82, .55)]),
             ('p', [(.82, .55), (.5, .55)])]),
 'H': (.8, [('p', [(0, 0), (0, 1)]), ('p', [(.8, 0), (.8, 1)]), ('p', [(0, .5), (.8, .5)])]),
 'I': (.3, [('p', [(0, 0), (.3, 0)]), ('p', [(.15, 0), (.15, 1)]), ('p', [(0, 1), (.3, 1)])]),
 'J': (.65, [('p', [(.35, 0), (.65, 0)]), ('s', [(.5, 0), (.5, .75), (.38, .97), (.15, .96), (.02, .78)])]),
 'K': (.8, [('p', [(0, 0), (0, 1)]), ('p', [(.75, 0), (0, .58)]), ('p', [(.25, .4), (.8, 1)])]),
 'L': (.7, [('p', [(0, 0), (0, 1), (.7, 1)])]),
 'M': (.95, [('p', [(0, 1), (0, 0), (.47, .62), (.94, 0), (.94, 1)])]),
 'N': (.8, [('p', [(0, 1), (0, 0), (.8, 1), (.8, 0)])]),
 'O': (.85, [('c', ell(.85))]),
 'P': (.75, [('p', [(0, 1), (0, 0)]), ('s', _P1)]),
 'Q': (.85, [('c', ell(.85)), ('p', [(.5, .72), (.88, 1.08)])]),
 'R': (.8, [('p', [(0, 1), (0, 0)]), ('s', _P1), ('p', [(.35, .5), (.8, 1)])]),
 'S': (.78, [('s', [(.75, .15), (.45, 0), (.12, .15), (.1, .38), (.4, .5), (.72, .62), (.72, .85), (.4, 1), (.05, .85)])]),
 'T': (.8, [('p', [(0, 0), (.8, 0)]), ('p', [(.4, 0), (.4, 1)])]),
 'U': (.8, [('s', [(0, 0), (0, .7), (.15, .95), (.4, 1), (.65, .95), (.8, .7), (.8, 0)])]),
 'V': (.84, [('p', [(0, 0), (.42, 1), (.84, 0)])]),
 'W': (1.0, [('p', [(0, 0), (.25, 1), (.5, .3), (.75, 1), (1, 0)])]),
 'X': (.8, [('p', [(0, 0), (.8, 1)]), ('p', [(.8, 0), (0, 1)])]),
 'Y': (.84, [('p', [(0, 0), (.42, .5), (.84, 0)]), ('p', [(.42, .5), (.42, 1)])]),
 'Z': (.8, [('p', [(0, 0), (.8, 0), (0, 1), (.8, 1)])]),
 '1': (.35, [('p', [(0, .22), (.3, 0), (.3, 1)])]),
 '2': (.7, [('s', [(.05, .2), (.3, 0), (.6, .1), (.65, .35), (0, 1)]), ('p', [(0, 1), (.7, 1)])]),
 '3': (.7, [('s', [(.05, .1), (.35, 0), (.6, .15), (.55, .4), (.25, .5)]),
            ('s', [(.25, .5), (.62, .6), (.65, .85), (.35, 1), (0, .9)])]),
 '4': (.75, [('p', [(.55, 1), (.55, 0), (0, .7), (.75, .7)])]),
 '.': (.12, [('p', [(.05, .92), (.05, 1.0)])]),
 ',': (.12, [('p', [(.07, .9), (0, 1.12)])]),
 '!': (.12, [('p', [(.05, 0), (.05, .68)]), ('p', [(.05, .92), (.05, 1.0)])]),
 '?': (.6, [('s', [(0, .2), (.25, 0), (.55, .12), (.5, .38), (.28, .55), (.28, .7)]), ('p', [(.28, .92), (.28, 1.0)])]),
 '=': (.8, [('p', [(0, .36), (.8, .36)]), ('p', [(0, .66), (.8, .66)])]),
 '-': (.5, [('p', [(0, .55), (.5, .55)])]),
}


def text_strokes(s, x, y, cap, align, rng):
    adv = lambda ch: 0.5 * cap if ch == ' ' else (G[ch][0] + 0.24) * cap
    total = sum(adv(c) for c in s) - 0.24 * cap
    cur = x - total / 2 if align == 'c' else x
    out = []
    for ch in s:
        if ch == ' ':
            cur += adv(ch); continue
        w, strs = G[ch]
        rot = rng.normal(0, 0.04); sc = 1 + rng.normal(0, 0.03); dy = rng.normal(0, 0.025) * cap
        cs, sn = math.cos(rot), math.sin(rot)
        for kind, pts in strs:
            pts = np.array(pts, float)
            if kind == 's': pts = spline(pts, n=6)
            elif kind == 'c': pts = spline(pts, closed=True, n=5, over=2)
            loc = (pts - [w / 2, 0.5]) * cap * sc
            loc[:, 0] -= 0.07 * loc[:, 1]                         # marker lean
            xr = loc[:, 0] * cs - loc[:, 1] * sn; yr = loc[:, 0] * sn + loc[:, 1] * cs
            out.append(np.stack([xr + cur + w / 2 * cap, yr + y + dy], 1))
        cur += adv(ch)
    return out, total


# ----------------------------------------------------------------- scene elements
def ease(t, a, b):
    if t <= a: return 0.0
    if t >= b: return 1.0
    return 0.5 - 0.5 * math.cos(math.pi * (t - a) / (b - a))


class El:
    """A set of pen strokes that draw on (and optionally un-draw) over time."""
    def __init__(s, strokes, color, w, t0, dur, ease_=True, group=None, t_out=None, out_dur=.3, seed=0):
        s.strokes = [resample(p) for p in strokes]
        s.cums, s.starts, tot = [], [], 0.0
        for p in s.strokes:
            c = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))]).astype(np.float32)
            s.cums.append(c); s.starts.append(tot); tot += float(c[-1])
        s.total = max(tot, 1e-3)
        rng = np.random.RandomState(seed)
        s.var = []
        for v in range(3):                                       # 3 'boil' variants, cycled at 10 fps
            ph = rng.rand(4) * 6.28
            s.var.append([p + np.stack([1.0 * np.sin(c * .045 + ph[0]) + .5 * np.sin(c * .13 + ph[1]),
                                        1.0 * np.sin(c * .05 + ph[2]) + .5 * np.sin(c * .11 + ph[3])], 1)
                          for p, c in zip(s.strokes, s.cums)])
        s.color, s.w, s.t0, s.dur, s.ease = color, int(round(w)), t0, dur, ease_
        s.group, s.t_out, s.out_dur = group, t_out, out_dur

    def prog(s, t):
        if t < s.t0: return 0.0
        p = min(1.0, (t - s.t0) / s.dur)
        if s.ease: p = 0.5 - 0.5 * math.cos(math.pi * p)
        if s.t_out is not None and t > s.t_out:
            q = (t - s.t_out) / s.out_dur
            return 0.0 if q >= 1 else p * (1 - q)
        return p

    def draw(s, cv, t, off, vi, tips):
        p = s.prog(t)
        if p <= 0: return
        L = p * s.total
        last = None
        for i, P in enumerate(s.var[vi]):
            st = s.starts[i]
            if st >= L and i > 0: break
            c = s.cums[i]; rem = L - st
            if rem >= c[-1]:
                Q = P
            else:
                k = max(1, int(np.searchsorted(c, rem)))
                f = (rem - c[k - 1]) / max(float(c[k] - c[k - 1]), 1e-6)
                Q = np.vstack([P[:k], P[k - 1] + (P[k] - P[k - 1]) * f])
            pts = ((Q + off) * 16).astype(np.int32)
            cv2.polylines(cv, [pts.reshape(-1, 1, 2)], False, s.color, s.w, cv2.LINE_AA, 4)
            last = Q[-1] + off
        undrawing = s.t_out is not None and t > s.t_out
        if last is not None and not undrawing:
            if t < s.t0 + s.dur: tips[0].append((s.t0, last, s.color))
            elif t - (s.t0 + s.dur) < .35: tips[1].append((s.t0 + s.dur, last, s.color))


class Fill:
    """Diagonal marker hatching inside a polygon; sweeps in like colouring."""
    def __init__(s, polys, color, t0, dur, group=None, t_out=None, out_dur=.3, alpha=.7, seed=0, gap=10):
        allp = np.vstack(polys)
        s.x0, s.y0 = (np.floor(allp.min(0)) - 6).astype(int)
        x1, y1 = (np.ceil(allp.max(0)) + 6).astype(int)
        w, h = int(x1 - s.x0), int(y1 - s.y0)
        pm = np.zeros((h, w), np.uint8)
        for P in polys:
            cv2.fillPoly(pm, [np.round(P - [s.x0, s.y0]).astype(np.int32).reshape(-1, 1, 2)], 255)
        hm = np.zeros((h, w), np.uint8)
        rng = np.random.RandomState(seed); x = -h
        while x < w:
            j = rng.uniform(-1.5, 1.5)
            cv2.line(hm, (int(x), 0), (int(x + h + j * 3), h), 255, 3, cv2.LINE_AA)
            x += gap + rng.uniform(-1.5, 1.5)
        s.mask = (cv2.bitwise_and(pm, hm).astype(np.float32) / 255.0) * alpha
        yy, xx = np.mgrid[0:h, 0:w]
        s.D = ((xx + yy) / float(w + h)).astype(np.float32)
        s.w, s.h = w, h
        s.color = np.array(color, np.float32)
        s.t0, s.dur, s.group, s.t_out, s.out_dur = t0, dur, group, t_out, out_dur

    def draw(s, cv, t, off, vi, tips):
        if t < s.t0: return
        p = min(1.0, (t - s.t0) / s.dur)
        if s.t_out is not None and t > s.t_out:
            q = (t - s.t_out) / s.out_dur
            if q >= 1: return
            p *= (1 - q)
        if p <= 0: return
        m = s.mask * (s.D <= p)
        X0 = int(round(s.x0 + off[0])); Y0 = int(round(s.y0 + off[1]))
        cx0, cy0 = max(X0, 0), max(Y0, 0); cx1, cy1 = min(X0 + s.w, W), min(Y0 + s.h, H)
        if cx1 <= cx0 or cy1 <= cy0: return
        sub = m[cy0 - Y0:cy1 - Y0, cx0 - X0:cx1 - X0, None]
        reg = cv[cy0:cy1, cx0:cx1].astype(np.float32)
        cv[cy0:cy1, cx0:cx1] = (reg * (1 - sub) + s.color * sub).astype(np.uint8)


class Scene:
    def __init__(s, dur):
        s.dur, s.els, s.groups, s.seed = dur, [], {}, 100

    def sd(s):
        s.seed += 1; return s.seed

    def line(s, strokes, color, t0, dur, w=5, **kw):
        return s.els.append(El(strokes, color, w, t0, dur, seed=s.sd(), **kw))

    def fill(s, polys, color, t0, dur, **kw):
        return s.els.append(Fill(polys, color, t0, dur, seed=s.sd(), **kw))

    def txt(s, string, x, y, cap, color, t0, cps=12, align='c', w=None, **kw):
        rng = np.random.RandomState(s.sd())
        strokes, width = text_strokes(string, x, y, cap, align, rng)
        n = max(1, len([c for c in string if c != ' ']))
        s.els.append(El(strokes, color, w or max(3.2, min(7.5, cap * .115)), t0, max(.25, n / cps),
                        ease_=False, seed=s.sd(), **kw))
        return width

    def off(s, g, t):
        return s.groups[g](t) if g in s.groups else (0.0, 0.0)


# ----------------------------------------------------------------- drawing helpers (characters & props)
def wavy(x0, x1, y, amp=5, ph=0.0):
    x = np.linspace(x0, x1, 40)
    return np.stack([x, y + amp * np.sin(x * .05 + ph) + .3 * amp * np.sin(x * .21)], 1)


def arrow(p0, p1, bend=.12, head=15):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    v = p1 - p0; L = np.hypot(*v); nrm = np.array([-v[1], v[0]]) / max(L, 1e-6)
    mid = (p0 + p1) / 2 + nrm * bend * L
    shaft = spline([p0, mid, p1], n=10)
    tg = shaft[-1] - shaft[-4]; a = math.atan2(tg[1], tg[0])
    hd = [np.array([p1, p1 - head * np.array([math.cos(a + s * .5), math.sin(a + s * .5)])]) for s in (1, -1)]
    return [shaft] + hd


def sparkle(x, y, s):
    return [np.array([[x, y - s], [x, y + s]]), np.array([[x - s, y], [x + s, y]]),
            np.array([[x - .55 * s, y - .55 * s], [x + .55 * s, y + .55 * s]]),
            np.array([[x + .55 * s, y - .55 * s], [x - .55 * s, y + .55 * s]])]


def burst(x, y, r1, r2, n=9, seed=3):
    rng = np.random.RandomState(seed); out = []
    for k in range(n):
        a = 2 * math.pi * k / n + rng.uniform(-.15, .15); r = rng.uniform(r2 * .75, r2)
        out.append(np.array([[x + r1 * math.cos(a), y + r1 * math.sin(a)], [x + r * math.cos(a), y + r * math.sin(a)]]))
    return out


def rrect(x0, y0, x1, y1, rad=18):
    pts = []
    for cx, cy, a0 in [(x1 - rad, y0 + rad, -90), (x1 - rad, y1 - rad, 0), (x0 + rad, y1 - rad, 90), (x0 + rad, y0 + rad, 180)]:
        for a in np.linspace(a0, a0 + 90, 6):
            pts.append((cx + rad * math.cos(math.radians(a)), cy + rad * math.sin(math.radians(a))))
    pts.append(pts[0])
    return np.array(pts)


def cloud(cx, cy, a, b):
    th = np.linspace(0, 2 * math.pi, 120)
    r = 1 + .11 * np.abs(np.sin(4 * th))
    return closed(np.stack([cx + a * np.cos(th) * r, cy + b * np.sin(th) * r], 1), 3)


def antibody(x, y, s):
    arm = lambda sx: np.array([[0, 0], [sx * .62 * s, -.85 * s]]) + [x, y]
    tick = lambda sx: np.array([[sx * .62 * s - sx * .05 * s - .2 * s, -.85 * s - .12 * s],
                                [sx * .62 * s + .2 * s - sx * .05 * s, -.85 * s + .12 * s]]) + [x, y]
    return [np.array([[0, .95 * s], [0, 0]]) + [x, y], arm(-1), arm(1),
            np.array([[-.62 * s - .2 * s, -.85 * s + .06 * s], [-.62 * s + .2 * s, -.85 * s - .06 * s]]) + [x, y],
            np.array([[.62 * s - .2 * s, -.85 * s - .06 * s], [.62 * s + .2 * s, -.85 * s + .06 * s]]) + [x, y]]


def syringe(x0, y):
    L = lambda *p: np.array(p, float)
    barrel = rrect(x0 + 74, y - 30, x0 + 282, y + 30, 10)
    parts = [L((x0, y - 28), (x0, y + 28)), L((x0, y), (x0 + 62, y)),
             L((x0 + 62, y - 20), (x0 + 76, y - 20), (x0 + 76, y + 20), (x0 + 62, y + 20), (x0 + 62, y - 20)),
             barrel, L((x0 + 90, y - 46), (x0 + 90, y + 46)),
             L((x0 + 282, y - 13), (x0 + 304, y - 8), (x0 + 304, y + 8), (x0 + 282, y + 13)),
             L((x0 + 304, y), (x0 + 388, y))]
    for i, k in enumerate(range(5)):
        xx = x0 + 120 + k * 32; parts.append(L((xx, y + 30), (xx, y + (14 if k % 2 == 0 else 20))))
    liquid = np.array([(x0 + 100, y - 24), (x0 + 276, y - 24), (x0 + 276, y + 24), (x0 + 100, y + 24)], float)
    return parts, liquid, x0 + 388


def cell_face(kind, cx, cy, R):
    fx, fy = cx, cy + .18 * R
    P = lambda pts: np.array(pts, float) * R + [fx, fy]
    dot = lambda x, y, r=.065: circ(fx + x * R, fy + y * R, r * R, 10)
    ring = lambda x, y, r=.11: circ(fx + x * R, fy + y * R, r * R, 14)
    sp = lambda pts: spline(P(pts), n=7)
    if kind == 'calm':
        return [dot(-.3, -.02), dot(.3, -.02), sp([(-.2, .22), (0, .32), (.2, .22)])]
    if kind == 'confused':
        return [dot(-.3, -.02), dot(.3, -.02), P([(-.45, -.28), (-.18, -.34)]), P([(.18, -.2), (.45, -.14)]),
                sp([(-.25, .28), (-.1, .22), (.05, .32), (.2, .24), (.28, .3)])]
    if kind == 'scared':
        return [ring(-.3, -.02), ring(.3, -.02), dot(-.3, 0, .03), dot(.3, 0, .03),
                P([(-.46, -.38), (-.18, -.5)]), P([(.18, -.5), (.46, -.38)]), circ(fx, fy + .32 * R, .09 * R, 12),
                sp([(.62, -.42), (.56, -.28), (.62, -.2), (.68, -.28), (.62, -.42)])]
    if kind == 'happy':
        return [sp([(-.42, -.02), (-.3, -.14), (-.18, -.02)]), sp([(.18, -.02), (.3, -.14), (.42, -.02)]),
                sp([(-.35, .12), (-.2, .38), (0, .45), (.2, .38), (.35, .12)])]
    if kind == 'determined':
        return [dot(-.3, -.02), dot(.3, -.02), P([(-.46, -.38), (-.16, -.2)]), P([(.16, -.2), (.46, -.38)]),
                sp([(-.2, .3), (0, .35), (.22, .25)])]
    if kind == 'curious':
        return [ring(-.3, -.02), ring(.3, -.02), dot(-.26, 0, .04), dot(.34, 0, .04),
                sp([(-.45, -.3), (-.32, -.4), (-.18, -.34)]), sp([(.18, -.34), (.32, -.4), (.45, -.3)]),
                circ(fx + .02 * R, fy + .33 * R, .06 * R, 10)]


def virus_face(kind, cx, cy, r):
    P = lambda pts: np.array(pts, float) * r + [cx, cy]
    dot = lambda x, y, rr=.07: circ(cx + x * r, cy + y * r, rr * r, 10)
    if kind == 'angry':
        return [dot(-.3, -.1), dot(.3, -.1), P([(-.5, -.36), (-.14, -.2)]), P([(.14, -.2), (.5, -.36)]),
                P([(-.4, .3), (-.27, .18), (-.13, .34), (0, .18), (.13, .34), (.27, .18), (.4, .3)])]
    if kind == 'sleepy':
        return [spline(P([(-.46, -.06), (-.32, .04), (-.18, -.06)]), n=7), spline(P([(.18, -.06), (.32, .04), (.46, -.06)]), n=7),
                spline(P([(-.2, .27), (0, .38), (.2, .27)]), n=7)]
    if kind == 'defeated':
        x = lambda ex: [P([(ex - .1, -.2), (ex + .1, 0)]), P([(ex + .1, -.2), (ex - .1, 0)])]
        return x(-.3) + x(.3) + [spline(P([(-.25, .4), (-.1, .28), (.1, .4), (.25, .28)]), n=7)]


def bandage(cx, cy, r):
    a = math.radians(-22); c, s_ = math.cos(a), math.sin(a)
    rot = lambda x, y: (cx + (x * c - y * s_) * r, cy - .4 * r + (x * s_ + y * c) * r)
    rect = np.array([rot(-.3, -.11), rot(.3, -.11), rot(.3, .11), rot(-.3, .11), rot(-.3, -.11)])
    return [rect, np.array([rot(-.08, -.04), rot(-.08, -.03)]), np.array([rot(.08, .04), rot(.08, .05)])]


def virus_parts(cx, cy, r, seed, n=11, sl=.34):
    P = blob(cx, cy, r, seed, n=60, amp=(.03, .02))
    strokes = [closed(P, 3)]
    for k in range(n):
        a = 2 * math.pi * k / n + .2; ca, sa = math.cos(a), math.sin(a)
        strokes.append(np.array([[cx + r * .97 * ca, cy + r * .97 * sa], [cx + r * (1 + sl) * ca, cy + r * (1 + sl) * sa]]))
        strokes.append(circ(cx + r * (1 + sl + .1) * ca, cy + r * (1 + sl + .1) * sa, r * .1, 10))
    return P, strokes


def cell_parts(cx, cy, R, seed, pose):
    P = blob(cx, cy, R, seed)
    nu = blob(cx + .38 * R, cy - .5 * R, .2 * R, seed + 1, n=24, amp=(.05, .02))
    strokes = [closed(P), closed(nu, 3)]
    hands = {'down': ((-1.45, .7), (1.45, .7)), 'up': ((-1.25, -.85), (1.25, -.85)),
             'wave': ((-1.45, .7), (1.3, -.8))}[pose]
    for sgn, (hx, hy) in zip((-1, 1), hands):
        sh = np.array([cx + sgn * .93 * R, cy + .15 * R]); hand = np.array([cx + hx * R, cy + hy * R])
        v = hand - sh; nrm = np.array([-v[1], v[0]]) / max(np.hypot(*v), 1e-6)
        mid = sh + v * .5 + nrm * .14 * np.hypot(*v) * (-sgn)
        strokes.append(spline([sh, mid, hand], n=8))
        strokes.append(circ(hand[0], hand[1], .17 * R, 12))
    return P, nu, strokes


# scene-level compound helpers
def add_cell(sc, cx, cy, R, seed, pose, t0, dur=.9, group=None):
    P, nu, strokes = cell_parts(cx, cy, R, seed, pose)
    sc.fill([P], BLUE, t0 + .5, .8, group=group, alpha=.55)
    sc.fill([nu], BLUE, t0 + .6, .5, group=group, alpha=.95, gap=7)
    sc.line(strokes, BLUE, t0, dur, group=group)


def add_virus(sc, cx, cy, r, seed, color, t0, dur=.9, group=None, n=11, sl=.34, w=5):
    P, strokes = virus_parts(cx, cy, r, seed, n, sl)
    sc.fill([P], color, t0 + .45, .7, group=group, alpha=.6)
    sc.line(strokes, color, t0, dur, group=group, w=w)


def face(sc, kind, cx, cy, R, t0, dur=.4, group=None, t_out=None, virus=False, w=4):
    f = virus_face(kind, cx, cy, R) if virus else cell_face(kind, cx, cy, R)
    sc.line(f, INK, t0, dur, w=w, group=group, t_out=t_out, out_dur=.2)


def header(sc, s, accent, t0=.3):
    width = sc.txt(s, 62, 72, 44, INK, t0, cps=15, align='l')
    n = len([c for c in s if c != ' '])
    sc.line([wavy(60, 66 + width, 112, 4), wavy(70, 56 + width, 122, 3, 1.5)], accent, t0 + n / 15.0, .5, w=5)


# ----------------------------------------------------------------- the five scenes
def scene0():
    sc = Scene(3.2)
    sc.txt("HOW VACCINES TRAIN", 640, 215, 62, INK, .2, cps=17)
    sc.txt("YOUR IMMUNE SYSTEM", 640, 330, 62, INK, 1.3, cps=17)
    sc.line([wavy(150, 1130, 392, 6), wavy(210, 1070, 407, 5, 2)], RED, 2.35, .65, w=6)
    add_cell(sc, 640, 580, 52, 5, 'up', 1.6, dur=.8)
    sc.line(cell_face('happy', 640, 580, 52), INK, 2.3, .35, w=3)
    for (x, y, s_) in [(520, 520, 16), (770, 530, 14), (800, 610, 10)]:
        sc.line(sparkle(x, y, s_), ORANGE, 2.5 + (x % 3) * .1, .3, w=4)
    return sc


def scene1():
    sc = Scene(5.7)
    cx, cy, R = 330, 410, 95
    sc.groups['virus'] = lambda t: (-385 * ease(t, 2.9, 3.6), 3 * math.sin(5 * t))
    sc.groups['cell'] = lambda t: (7 * math.sin(50 * (t - 3.6)) * math.exp(-(t - 3.6) * 4) if t > 3.6 else 0.0, 0.0)
    header(sc, "1. THE INVADER", RED)
    add_cell(sc, cx, cy, R, 11, 'down', .6, group='cell')
    face(sc, 'confused', cx, cy, R, 1.35, group='cell', t_out=3.55)
    sc.txt("IMMUNE CELL", cx, 550, 30, BLUE, 1.6, cps=14)
    sc.txt("??", cx, 262, 44, INK, 2.2, cps=6, group='cell', t_out=3.5)
    add_virus(sc, 960, 410, 78, 21, RED, 1.2, group='virus')
    face(sc, 'angry', 960, 410, 78, 2.0, group='virus', virus=True)
    sc.txt("VIRUS", 960, 262, 32, RED, 2.2, cps=10, t_out=2.85)
    sc.line(arrow((960, 285), (960, 312), .05, 10), RED, 2.6, .25, w=5, t_out=2.85)
    sc.line(burst(448, 395, 14, 46), ORANGE, 3.55, .25, w=5, t_out=4.3)
    face(sc, 'scared', cx, cy, R, 3.65, dur=.4, group='cell')
    sc.txt("UNTRAINED = SLOW RESPONSE", 640, 640, 34, INK, 3.8, cps=16)
    return sc


def scene2():
    sc = Scene(6.8)
    cx, cy, R = 1000, 410, 95
    sc.groups['wv'] = lambda t: (250 * ease(t, 4.1, 4.8), 2 * math.sin(6 * t))
    header(sc, "2. THE VACCINE", GREEN)
    add_cell(sc, cx, cy, R, 31, 'down', .5)
    face(sc, 'calm', cx, cy, R, 1.2, t_out=4.4)
    parts, liquid, tipx = syringe(60, 330)
    sc.fill([liquid], GREEN, 1.9, .5, alpha=.8)
    sc.line(parts, INK, 1.0, 1.1, w=5)
    sc.txt("VACCINE", 230, 430, 30, GREEN, 2.2, cps=10)
    add_virus(sc, 545, 330, 58, 41, GREEN, 2.5, dur=.7, group='wv', n=9, sl=.26)
    sc.line(virus_face('sleepy', 545, 330, 58), INK, 3.05, .3, w=4, group='wv')
    sc.line(bandage(545, 330, 58), ORANGE, 3.2, .3, w=4, group='wv')
    sc.txt("HARMLESS COPY", 545, 472, 30, GREEN, 3.2, cps=15, t_out=4.05)
    sc.line(arrow((545, 450), (545, 418), .05, 10), GREEN, 3.85, .2, w=5, t_out=4.05)
    face(sc, 'curious', cx, cy, R, 4.35, dur=.35)
    sc.fill([blob(1135, 255, 30, 9, n=40, amp=(0, 0))], YELLOW, 4.8, .4, alpha=.95, gap=6)
    bulb = [circ(1135, 255, 30, 20), np.array([[1123, 288], [1147, 288]]), np.array([[1125, 296], [1145, 296]]),
            np.array([[1128, 304], [1142, 304]]), np.array([[1128, 258], [1131, 248], [1135, 258], [1139, 248], [1142, 258]])]
    for a in np.linspace(-160, -20, 6):
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        bulb.append(np.array([[1135 + 42 * ca, 255 + 42 * sa], [1135 + 62 * ca, 255 + 62 * sa]]))
    sc.line(bulb, ORANGE, 4.8, .6, w=4)
    sc.txt("SAFE PRACTICE, NO ILLNESS", 640, 640, 34, INK, 4.9, cps=16.5)
    return sc


def scene3():
    sc = Scene(6.6)
    cx, cy, R = 300, 430, 95
    header(sc, "3. THE TRAINING", ORANGE)
    add_cell(sc, cx, cy, R, 51, 'up', .5)
    face(sc, 'happy', cx, cy, R, 1.3, dur=.35)
    sc.line(arrow((415, 440), (650, 430), -.08, 16), INK, 1.5, .45, w=5)
    pos = [(720, 390), (820, 340), (920, 405), (1030, 345), (1120, 410), (770, 510), (900, 525), (1030, 505)]
    for i, (x, y) in enumerate(pos):
        sc.groups[f'ab{i}'] = lambda t, i=i: (0.0, 5 * math.sin(2 * math.pi * t / 1.6 + i))
        sc.line(antibody(x, y, 36), ORANGE, 1.7 + .17 * i, .22, w=6, group=f'ab{i}')
    sc.txt("ANTIBODIES", 925, 245, 32, ORANGE, 3.0, cps=11)
    sc.line(arrow((925, 268), (925, 296), .05, 11), ORANGE, 3.8, .2, w=5)
    sc.line([circ(338, 302, 9, 10)], INK, 3.2, .15, w=4)
    sc.line([circ(362, 276, 13, 12)], INK, 3.3, .15, w=4)
    sc.line([cloud(450, 205, 135, 52)], INK, 3.4, .7, w=5)
    mv = 28
    P, strokes = virus_parts(450, 205, mv, 61, n=10, sl=.38)
    sc.fill([P], RED, 4.1, .4, alpha=.75, gap=7)
    sc.line(strokes, RED, 4.0, .5, w=3)
    sc.line(virus_face('angry', 450, 205, mv), INK, 4.4, .2, w=2)
    sc.txt("MEMORY CELLS SAVE ITS FACE", 640, 640, 34, INK, 4.5, cps=15)
    return sc


def scene4():
    sc = Scene(7.7)
    cx, cy, R = 280, 420, 95
    vx, vy, vr = 1000, 420, 78
    sc.groups['virus'] = lambda t: ((6 * math.sin(55 * (t - 4.5)) * math.exp(-(t - 4.5) * 3) if t > 4.5 else 0.0), 0.0)
    base = [(470, 345), (505, 410), (470, 465), (545, 370), (555, 440), (520, 505)]
    tgt = [(vx + 140 * math.cos(math.radians(a)), vy + 140 * math.sin(math.radians(a))) for a in (150, 175, 200, 225, 125, 250)]
    for i in range(6):
        dx, dy = tgt[i][0] - base[i][0], tgt[i][1] - base[i][1]
        a0 = 3.8 + .06 * i
        sc.groups[f'a{i}'] = lambda t, dx=dx, dy=dy, a0=a0, i=i: (
            dx * ease(t, a0, a0 + .6), dy * ease(t, a0, a0 + .6) + (3 * math.sin(14 * t + i) if t > a0 + .6 else 0))
    header(sc, "4. THE REAL THING", BLUE)
    add_cell(sc, cx, cy, R, 71, 'down', .6)
    face(sc, 'determined', cx, cy, R, 1.35, dur=.35)
    add_virus(sc, vx, vy, vr, 81, RED, 1.2, group='virus')
    face(sc, 'angry', vx, vy, vr, 2.0, group='virus', virus=True, t_out=4.55)
    sc.txt("SAME VIRUS!", vx, 268, 32, RED, 2.1, cps=12, t_out=4.1)
    sc.line(arrow((vx, 290), (vx, 312), .05, 10), RED, 2.9, .2, w=5, t_out=4.1)
    sc.line([rrect(345, 195, 700, 270, 20), np.array([[400, 270], [346, 335], [445, 270]])], BLUE, 2.4, .5, w=5)
    sc.txt("I KNOW YOU!", 522, 233, 32, BLUE, 2.9, cps=13)
    for i, (x, y) in enumerate(base):
        sc.line(antibody(x, y, 32), ORANGE, 3.1 + .1 * i, .18, w=6, group=f'a{i}')
    face(sc, 'defeated', vx, vy, vr, 4.65, dur=.35, group='virus', virus=True)
    sc.line([np.array([[vx, vy - 108], [vx, vy - 195]]),
             spline([(vx, vy - 195), (vx + 28, vy - 202), (vx + 60, vy - 194), (vx + 58, vy - 160),
                     (vx + 28, vy - 168), (vx, vy - 158)], n=6)], INK, 5.0, .5, w=5, group='virus')
    for (x, y, s_) in [(130, 330, 18), (430, 300, 14), (150, 520, 12)]:
        sc.line(sparkle(x, y, s_), ORANGE, 5.3 + (x % 5) * .02, .3, w=4)
    sc.txt("PRACTICE NOW = PROTECTED LATER", 640, 622, 34, INK, 5.3, cps=18)
    sc.line([wavy(150, 1130, 662, 3), wavy(220, 1060, 673, 3, 2)], GREEN, 7.0, .45, w=5)
    return sc


SCENES = [scene0(), scene1(), scene2(), scene3(), scene4()]
STARTS = []
_a = 0.0
for _s in SCENES:
    STARTS.append(round(_a, 3)); _a += _s.dur
TOTAL = round(_a, 3)
WIPE = 0.5

# ----------------------------------------------------------------- board, pen, eraser, compositing
def make_bg():
    rng = np.random.RandomState(7)
    bg = np.full((H, W, 3), (244, 247, 248), np.uint8)
    ghost = np.zeros((H, W), np.uint8)
    for _ in range(16):
        x, y = rng.randint(60, W - 60), rng.randint(60, H - 60); a = rng.uniform(0, 6.28); L = rng.randint(60, 260)
        cv2.line(ghost, (x, y), (int(x + L * math.cos(a)), int(y + L * math.sin(a))), 255, rng.randint(10, 26), cv2.LINE_AA)
    ghost = cv2.GaussianBlur(ghost, (0, 0), 14).astype(np.float32)[..., None] / 255 * .12
    bg = (bg * (1 - ghost) + np.array((205, 208, 210), np.float32) * ghost).astype(np.uint8)
    bg = np.clip(bg.astype(np.float32) + rng.normal(0, 1.0, (H, W, 1)), 0, 255).astype(np.uint8)
    cv2.rectangle(bg, (0, 0), (W - 1, H - 1), (206, 203, 199), 30)
    cv2.rectangle(bg, (30, 30), (W - 31, H - 31), (170, 166, 160), 2)
    cv2.rectangle(bg, (0, 0), (W - 1, H - 1), (232, 230, 227), 2)
    return bg


BG = make_bg()
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def draw_marker(cv, tx, ty, col, t):
    a = math.radians(-58 + 3 * math.sin(t * 9))
    u = np.array([math.cos(a), math.sin(a)]); n = np.array([-u[1], u[0]]); tip = np.array([tx, ty], float)

    def quad(s0, s1, h0, h1):
        return np.array([tip + u * s0 + n * h0, tip + u * s1 + n * h1, tip + u * s1 - n * h1, tip + u * s0 - n * h0])

    def fp(q, c):
        cv2.fillConvexPoly(cv, (q * 16).astype(np.int32), c, cv2.LINE_AA, 4)
    fp(quad(0, 16, 2.5, 8), col)
    fp(quad(16, 160, 12, 12), (238, 238, 238))
    cv2.polylines(cv, [(quad(16, 160, 12, 12) * 16).astype(np.int32)], True, (90, 90, 90), 2, cv2.LINE_AA, 4)
    fp(quad(70, 110, 12.3, 12.3), col)
    fp(quad(160, 178, 10, 10), col)


def draw_eraser(cv, x, y):
    a = math.radians(-8); u = np.array([math.cos(a), math.sin(a)]); n = np.array([-u[1], u[0]]); c = np.array([x, y], float)
    quad = lambda s0, s1, h0, h1: np.array([c + u * s0 + n * h0, c + u * s1 + n * h0 * 0 + n * h1, c + u * s1 - n * h1, c + u * s0 - n * h0])
    cv2.fillConvexPoly(cv, (quad(-70, 70, 38, 38) * 16).astype(np.int32), (222, 222, 222), cv2.LINE_AA, 4)
    cv2.fillConvexPoly(cv, (quad(-70, 70, -8, -38)[::1] * 16).astype(np.int32), (200, 120, 60), cv2.LINE_AA, 4)
    cv2.polylines(cv, [(quad(-70, 70, 38, 38) * 16).astype(np.int32)], True, (90, 90, 90), 2, cv2.LINE_AA, 4)


def render_scene(sc, t, vi):
    cv = BG.copy()
    tips = ([], [])
    for e in sc.els:
        e.draw(cv, t, sc.off(e.group, t), vi, tips)
    pick = max(tips[0], key=lambda z: z[0]) if tips[0] else (max(tips[1], key=lambda z: z[0]) if tips[1] else None)
    if pick is not None:
        draw_marker(cv, pick[1][0], pick[1][1], pick[2], t)
    return cv


def frame(i):
    t = i / FPS; vi = (i // 3) % 3
    for k in range(1, len(SCENES)):
        b = STARTS[k]
        if b - WIPE / 2 <= t < b + WIPE / 2:
            old = render_scene(SCENES[k - 1], t - STARTS[k - 1], vi)
            new = render_scene(SCENES[k], t - b, vi)
            q = ease(t, b - WIPE / 2, b + WIPE / 2)
            E = -120 + (W + 240) * q
            edge = E + (YY - H / 2) * .28
            m = np.clip((edge - XX) / 28 + .5, 0, 1)[..., None]
            out = (old * (1 - m) + new * m).astype(np.uint8)
            yb = H / 2 + 230 * math.sin(2 * math.pi * 1.5 * q)
            draw_eraser(out, E + (yb - H / 2) * .28, yb)
            return out.tobytes()
    k = 0
    for j in range(len(SCENES)):
        if t >= STARTS[j] + (WIPE / 2 if j else 0):
            k = j
    return render_scene(SCENES[k], t - STARTS[k], vi).tobytes()


def preview(times):
    for t in times:
        i = int(round(t * FPS))
        buf = np.frombuffer(frame(i), np.uint8).reshape(H, W, 3)
        cv2.imwrite(f'./prev_{t:05.2f}.png', buf)
        print('wrote', t)


def render(out):
    os.makedirs('./seg', exist_ok=True)
    segs = []
    bounds = [int(round(s * FPS)) for s in STARTS] + [int(round(TOTAL * FPS))]
    with Pool(2) as pool:
        for k in range(len(SCENES)):
            tmp = f'./seg/s{k}.tmp.mp4'; fin = f'./seg/s{k}.mp4'
            p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                                  '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast',
                                  '-pix_fmt', 'yuv420p', tmp], stdin=subprocess.PIPE)
            for fr in pool.imap(frame, range(bounds[k], bounds[k + 1]), chunksize=3):
                p.stdin.write(fr)
            p.stdin.close()
            if p.wait() != 0:
                raise SystemExit('ffmpeg failed')
            os.rename(tmp, fin); segs.append(fin)
            print('scene', k, 'done', flush=True)
    lst = './seg/list.txt'
    open(lst, 'w').write(''.join(f"file '{s}'\n" for s in segs))
    subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy',
                           '-movflags', '+faststart', out])


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'preview':
        preview([float(x) for x in sys.argv[2:]])
    else:
        render(sys.argv[1] if len(sys.argv) > 1 else 'vaccine-whiteboard.mp4')
