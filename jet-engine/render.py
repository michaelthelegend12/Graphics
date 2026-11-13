#!/usr/bin/env python3
"""Jet engine explainer: blueprint cutaway, 20 s, 1080p30.
usage: render.py still <t> <out.png>        one test frame
       render.py chunk <i> <n> <out.mp4>    render slice i of n (run n in parallel)
then: ffmpeg concat the chunks with -c copy, and mux hum.wav (from hum.py) with -c:v copy
"""
import sys, math, subprocess, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter

W, H, FPS, DUR = 1920, 1080, 30, 20
NF = FPS * DUR
SS = 2
W2, H2 = W * SS, H * SS
CY = 570
BY0, BY1 = 320, 820
BH = BY1 - BY0
BH2 = BH * SS

WHITE = (240, 248, 255)
LBLUE = (150, 200, 255)
DIM = (105, 155, 220)
DARK = (6, 34, 82)
HOT = (255, 232, 176)

FD = '/usr/share/fonts/opentype/inter/'
F_TITLE = ImageFont.truetype(FD + 'InterDisplay-Bold.otf', 46)
F_WORD = ImageFont.truetype(FD + 'InterDisplay-Bold.otf', 84)
F_CAP = ImageFont.truetype(FD + 'Inter-Medium.otf', 28)
F_CHIP = ImageFont.truetype(FD + 'Inter-SemiBold.otf', 27)
F_M18 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 18)
F_M20 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 20)
F_M22 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 22)
F_MB18 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 18)
F_MB22 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 22)

STARTS = [2.0, 6.5, 11.0, 15.5]
ENDS = [6.5, 11.0, 15.5, 20.0]
STAGES = [
    dict(name='SUCK', num='01', x0=200, x1=490, comp='INTAKE + FAN',
         cap='Spinning fan blades pull in a river of air'),
    dict(name='SQUEEZE', num='02', x0=490, x1=870, comp='COMPRESSOR',
         cap='Compressor stages crush it to ~30× pressure'),
    dict(name='BANG', num='03', x0=870, x1=1190, comp='COMBUSTOR',
         cap='Fuel ignites — hot gas expands violently'),
    dict(name='BLOW', num='04', x0=1190, x1=1740, comp='TURBINE + NOZZLE',
         cap='Gas spins the turbine, then blasts out: thrust'),
]


# ---------------------------------------------------------------- helpers
def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def stage_w(i, t):
    s, e = STARTS[i], ENDS[i]
    a = smooth((t - s) / 0.35)
    if i < 3:
        a *= 1 - smooth((t - (e - 0.25)) / 0.35)
    return a


def smooth_interp(pts, sigma):
    px = [p[0] for p in pts]
    py = [p[1] for p in pts]
    y = np.interp(XS, px, py)
    k = np.exp(-0.5 * (np.arange(-3 * sigma, 3 * sigma + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(y, (3 * sigma, 3 * sigma), mode='edge'), k, mode='valid')


XS = np.arange(0, W + 1, 1.0)
# casing inner radius (gas path outer wall) and hub radius
R = smooth_interp([(0, 165), (200, 165), (300, 160), (440, 160), (520, 150), (820, 92),
                   (900, 100), (960, 150), (1140, 150), (1220, 105), (1400, 118),
                   (1480, 125), (1740, 78), (1920, 112)], 9)
HUB = smooth_interp([(0, 0), (230, 0), (300, 34), (400, 50), (520, 62), (820, 72), (900, 70),
                     (960, 60), (1140, 60), (1220, 64), (1400, 58), (1520, 44), (1640, 0),
                     (1920, 0)], 6)


def thick(x):
    x = np.asarray(x, float)
    lip = np.sqrt(np.clip(1 - ((215 - x) / 35.0) ** 2, 0, 1))
    lip = np.where(x >= 215, 1.0, lip)
    tail = np.interp(x, [0, 1600, 1740], [1, 1, 0.45])
    return 30 * lip * tail


def Y2(y):
    return (y - BY0) * SS


def ttext(d, xy, s, font, fill, tr=0.0):
    x, y = xy
    for ch in s:
        d.text((x, y), ch, font=font, fill=fill)
        x += font.getlength(ch) + tr
    return x


def twidth(s, font, tr=0.0):
    return sum(font.getlength(c) + tr for c in s)


def dashed(d, p0, p1, fill, width, dash=10, gap=7):
    (x0, y0), (x1, y1) = p0, p1
    L = math.hypot(x1 - x0, y1 - y0)
    if L == 0:
        return
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    s = 0.0
    while s < L:
        e = min(L, s + dash)
        d.line([(x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)], fill=fill, width=width)
        s += dash + gap


def lerp(a, b, k):
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))


# ---------------------------------------------------------------- static art
def build_bg():
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.clip(np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.78)) ** 2), 0, 1)
    c0 = np.array((17, 74, 146), float)
    c1 = np.array((7, 38, 94), float)
    arr = (c0[None, None, :] * (1 - d[..., None]) + c1[None, None, :] * d[..., None]).astype(np.uint8)
    base = Image.fromarray(arr).resize((W2, H2), Image.BILINEAR).convert('RGBA')
    ov = Image.new('RGBA', (W2, H2), (0, 0, 0, 0))
    g = ImageDraw.Draw(ov)
    for x in range(0, W2, 60):
        major = (x % 300 == 0)
        g.line([(x, 0), (x, H2)], fill=(255, 255, 255, 46 if major else 17), width=3 if major else 2)
    for y in range(0, H2, 60):
        major = (y % 300 == 0)
        g.line([(0, y), (W2, y)], fill=(255, 255, 255, 46 if major else 17), width=3 if major else 2)
    ins = 36 * SS
    g.rectangle([ins, ins, W2 - ins, H2 - ins], outline=(235, 245, 255, 110), width=3)
    for (cx, cy) in [(ins, ins), (W2 - ins, ins), (ins, H2 - ins), (W2 - ins, H2 - ins)]:
        g.line([(cx - 26 * SS, cy), (cx + 26 * SS, cy)], fill=(235, 245, 255, 190), width=3)
        g.line([(cx, cy - 26 * SS), (cx, cy + 26 * SS)], fill=(235, 245, 255, 190), width=3)
    base.alpha_composite(ov)
    return base


def build_engine():
    eng = Image.new('RGBA', (W2, BH2), (0, 0, 0, 0))
    d = ImageDraw.Draw(eng)
    xs = np.arange(180, 1741, 2)
    th = thick(xs)
    Ri = np.array([R[int(x)] for x in xs])

    def P(x, yy):
        return (x * SS, Y2(yy))

    cas_mask = Image.new('L', (W2, BH2), 0)
    cm = ImageDraw.Draw(cas_mask)
    outlines = []
    for s in (-1, 1):
        inner = [P(x, CY + s * r) for x, r in zip(xs, Ri)]
        outer = [P(x, CY + s * (r + t)) for x, r, t in zip(xs, Ri, th)]
        poly = outer + inner[::-1]
        d.polygon(poly, fill=(7, 38, 88, 255))
        cm.polygon(poly, fill=255)
        outlines.append((inner, outer))
    # hatch
    hatch = Image.new('RGBA', (W2, BH2), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hatch)
    for c in range(-BH2, W2, 26):
        hd.line([(c, 0), (c + BH2, BH2)], fill=(170, 212, 255, 120), width=2)
    hatch.putalpha(ImageChops.multiply(hatch.getchannel('A'), cas_mask))
    eng.alpha_composite(hatch)
    d = ImageDraw.Draw(eng)
    for inner, outer in outlines:
        d.line(inner, fill=WHITE + (255,), width=6, joint='curve')
        d.line(outer, fill=WHITE + (255,), width=6, joint='curve')
        d.line([inner[-1], outer[-1]], fill=WHITE + (255,), width=6)
    # flanges
    for fx in (520, 880, 1190):
        for s in (-1, 1):
            r = R[fx]
            t = float(thick(fx))
            y_a, y_b = CY + s * (r - 2), CY + s * (r + t + 12)
            d.rectangle([P(fx - 6, min(y_a, y_b)), P(fx + 6, max(y_a, y_b))],
                        fill=(7, 38, 88, 255), outline=WHITE + (255,), width=4)
    # hub
    hx = np.arange(230, 1641, 2)
    hh = np.array([HUB[int(x)] for x in hx])
    hub_poly = [P(x, CY - h) for x, h in zip(hx, hh)] + [P(x, CY + h) for x, h in zip(hx[::-1], hh[::-1])]
    d.polygon(hub_poly, fill=DARK + (255,))
    d.line([P(x, CY - h) for x, h in zip(hx, hh)] + [P(1640, CY)] + [P(x, CY + h) for x, h in zip(hx[::-1], hh[::-1])] + [P(230, CY)],
           fill=(205, 228, 255, 255), width=5, joint='curve')
    # shaft
    for s in (-1, 1):
        d.line([P(300, CY + s * 7), P(1330, CY + s * 7)], fill=(120, 175, 240, 235), width=4)
    # stators
    for xs_ in (575, 635, 695, 755, 815, 1285, 1355):
        r, h = R[xs_], HUB[xs_]
        for s in (-1, 1):
            d.line([P(xs_ - 7, CY + s * (r - 1)), P(xs_ + 7, CY + s * (h + 1))],
                   fill=(150, 205, 255, 240), width=6)
    # combustor liner + injectors
    for s in (-1, 1):
        for x in range(930, 1170, 22):
            r = R[x]
            d.line([P(x, CY + s * (r - 26)), P(x + 13, CY + s * (r - 26))], fill=(190, 225, 255, 230), width=4)
            h = HUB[x]
            d.line([P(x, CY + s * (h + 18)), P(x + 13, CY + s * (h + 18))], fill=(190, 225, 255, 160), width=4)
        gy = (R[960] + HUB[960]) / 2
        r0 = R[945]
        t0 = float(thick(945))
        d.line([P(945, CY + s * (r0 + t0)), P(945, CY + s * gy)], fill=WHITE + (255,), width=6)
        d.polygon([P(940, CY + s * (gy - 9)), P(940, CY + s * (gy + 9)), P(974, CY + s * gy)],
                  fill=(240, 248, 255, 255))
    # centre line (dash-dot)
    x = 100
    while x < 1830:
        d.line([P(x, CY), P(x + 44, CY)], fill=(200, 228, 255, 140), width=3)
        d.ellipse([P(x + 52, CY - 1.5), P(x + 55, CY + 1.5)], fill=(200, 228, 255, 140))
        x += 64
    return eng


def build_flowmask():
    m = Image.new('L', (W2, BH2), 0)
    d = ImageDraw.Draw(m)
    xs = np.arange(180, 1900, 3)
    for s in (-1, 1):
        top = [(x * SS, Y2(CY + s * R[int(x)])) for x in xs]
        bot = [(x * SS, Y2(CY + s * HUB[int(x)])) for x in xs]
        d.polygon(top + bot[::-1], fill=255)
    return m


BG = build_bg()
BG1 = BG.convert('RGB').reduce(SS)
BG_BAND = BG.crop((0, BY0 * SS, W2, BY1 * SS))
ENG = build_engine()
FLOWMASK = build_flowmask()

# ---------------------------------------------------------------- particles
N = 1100
SPX = [0, 200, 440, 520, 840, 900, 980, 1160, 1220, 1480, 1560, 1740, 1920]
SPV = [230, 240, 230, 205, 170, 200, 290, 330, 380, 430, 520, 720, 900]


def speed(x):
    return np.interp(x, SPX, SPV)


def make_state():
    rng = np.random.default_rng(7)
    S = dict(rng=rng,
             x=rng.uniform(40, 1900, N),
             v=rng.uniform(0.12, 0.88, N),
             side=rng.choice([-1.0, 1.0], N),
             ph=rng.uniform(0, 6.283, N))
    for _ in range(300):
        step(S)
    return S


def step(S, dt=1.0 / FPS):
    S['x'] += speed(S['x']) * dt
    gone = S['x'] > 1900
    n = int(gone.sum())
    if n:
        rng = S['rng']
        S['x'][gone] = 40 + rng.uniform(0, 30, n)
        S['v'][gone] = rng.uniform(0.12, 0.88, n)
        S['side'][gone] = rng.choice([-1.0, 1.0], n)
        S['ph'][gone] = rng.uniform(0, 6.283, n)


ROTORS = [360, 545, 605, 665, 725, 785, 1250, 1320]


def draw_blades(d, t):
    for xr in ROTORS:
        r, h = R[xr], HUB[xr]
        span = r - h
        pitch = max(span / 3.4, 11.0)
        hb = 0.62 * pitch
        wb = 9 if xr == 360 else 7
        sl = 12 if xr == 360 else 9
        off = (t * 150.0) % pitch
        n = int(span / pitch) + 2
        for s in (-1, 1):
            for k in range(-1, n):
                p = k * pitch + off
                if p < 0 or p > span:
                    continue
                edge = min(p, span - p) / (0.18 * span)
                a = int(215 * min(1.0, edge))
                if a < 12:
                    continue
                yy = CY + s * (h + (p if s < 0 else span - p))
                # moving up on both halves
                xc = xr
                poly = [((xc - wb / 2 + sl / 2) * SS, Y2(yy - hb / 2)),
                        ((xc + wb / 2 + sl / 2) * SS, Y2(yy - hb / 2)),
                        ((xc + wb / 2 - sl / 2) * SS, Y2(yy + hb / 2)),
                        ((xc - wb / 2 - sl / 2) * SS, Y2(yy + hb / 2))]
                d.polygon(poly, fill=(215, 236, 255, a))


def draw_dynamic(t, S, xfront):
    w = [stage_w(i, t) for i in range(4)]
    layer = Image.new('RGBA', (W2, BH2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    pulse = 0.5 + 0.5 * math.sin(2 * math.pi * 5.0 * t)

    # --- highlighted flow-path regions
    for i, st in enumerate(STAGES):
        if w[i] < 0.01:
            continue
        a = (0.16 + (0.10 * pulse if i == 2 else 0.0)) * w[i]
        col = (140, 205, 255, int(255 * a))
        xs = np.arange(st['x0'], st['x1'] + 1, 6)
        for s in (-1, 1):
            top = [(x * SS, Y2(CY + s * R[int(x)])) for x in xs]
            bot = [(x * SS, Y2(CY + s * HUB[int(x)])) for x in xs]
            d.polygon(top + bot[::-1], fill=col)
        # boundaries
        yb0 = CY - (np.max(R[st['x0']:st['x1']]) + 30) - 28
        for xb in (st['x0'], st['x1']):
            dashed(d, (xb * SS, Y2(yb0)), (xb * SS, Y2(805)), (240, 248, 255, int(150 * w[i])), 3, 16, 12)

    # --- blades
    draw_blades(d, t)
    # shaft dashes
    for s in (-1, 1):
        x = 300 - (t * 140) % 32
        while x < 1330:
            xa, xb = max(x, 300), min(x + 16, 1330)
            if xb > xa:
                d.line([(xa * SS, Y2(CY + s * 7)), (xb * SS, Y2(CY + s * 7))], fill=(215, 236, 255, 235), width=4)
            x += 32

    # --- particles
    x, v, side, ph = S['x'], S['v'], S['side'], S['ph']
    xi = np.clip(x.astype(int), 0, W)
    r, h = R[xi], HUB[xi]
    turb = np.interp(x, [0, 880, 960, 1190, 1260, 1500, 1920], [0, 0, 1, 1, 0.5, 0.3, 0.2])
    amp = turb * np.minimum(8.0, 0.1 * (r - h))
    y = CY + side * (h + v * (r - h)) + amp * np.sin(ph + 11 * t + x * 0.02)
    sp = speed(x)
    Tm = np.interp(x, [0, 880, 1000, 1500, 1920], [0, 0, 1, 1, 0.75])
    Cm = np.interp(x, [0, 440, 840, 900, 1000], [0, 0, 1, 1, 0.0])
    cold = np.array((150, 215, 255), float)
    wht = np.array((245, 250, 255), float)
    hot = np.array(HOT, float)
    col = cold[None, :] * (1 - 0.65 * Cm[:, None]) + wht[None, :] * (0.65 * Cm[:, None])
    col = col * (1 - Tm[:, None]) + hot[None, :] * Tm[:, None]
    boost = np.zeros(N)
    for i, st in enumerate(STAGES):
        boost += w[i] * ((x >= st['x0']) & (x < st['x1']))
    ag = smooth(t / 0.8)
    a_dot = ag * (150 + 105 * np.clip(boost, 0, 1))
    rad = (2.6 + 1.1 * np.clip(boost, 0, 1)) * SS
    trail = sp * 0.045 * SS
    X2 = x * SS
    Yb = (y - BY0) * SS

    glow = Image.new('RGBA', (W2 // 4, BH2 // 4), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)

    pl = Image.new('RGBA', (W2, BH2), (0, 0, 0, 0))
    pd = ImageDraw.Draw(pl)
    for i in range(N):
        if X2[i] < 0 or X2[i] > W2:
            continue
        c = (int(col[i, 0]), int(col[i, 1]), int(col[i, 2]))
        pd.line([(X2[i] - trail[i], Yb[i]), (X2[i], Yb[i])], fill=c + (int(a_dot[i] * 0.55),), width=int(rad[i] * 1.2))
        pd.ellipse([X2[i] - rad[i], Yb[i] - rad[i], X2[i] + rad[i], Yb[i] + rad[i]], fill=c + (int(a_dot[i]),))
        if Tm[i] > 0.5 and (i % 3 == 0):
            gx, gy = X2[i] / 4, Yb[i] / 4
            gd.ellipse([gx - 4, gy - 4, gx + 4, gy + 4], fill=(255, 225, 160, int(70 * ag)))

    # --- flame / fuel spray / shock rings (masked to flow path)
    fx = Image.new('RGBA', (W2, BH2), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fx)
    fi = 0.55 * smooth((t - 1.2) / 1.0) + 0.45 * w[2]
    if fi > 0.01:
        for s in (-1, 1):
            x0f = 990
            yc = CY + s * (R[x0f] + HUB[x0f]) / 2
            n1 = 0.5 + 0.5 * math.sin(t * 17 + s * 1.3) * math.sin(t * 7.3 + 0.4)
            n2 = 0.5 + 0.5 * math.sin(t * 13 + s * 2.1)
            L = (80 + 85 * fi) * (0.82 + 0.18 * n1)
            T = (14 + 16 * fi) * (0.88 + 0.12 * n2)
            yj = 3 * math.sin(t * 9 + s)
            gd.ellipse([(x0f + L * 0.5) / 2 - L * 0.5 / 2 - 14, Y2(yc + yj) / 4 - (T + 14) / 2,
                        (x0f + L * 0.5) / 2 + L * 0.5 / 2 + 14, Y2(yc + yj) / 4 + (T + 14) / 2],
                       fill=(255, 215, 140, int(150 * fi)))
            layers = [(1.0, (255, 205, 140, int(80 * fi))), (0.78, (255, 224, 168, int(125 * fi))),
                      (0.56, (255, 240, 206, int(185 * fi))), (0.34, (255, 252, 238, int(245 * fi)))]
            for f, c in layers:
                fd.ellipse([x0f * SS, Y2(yc + yj - T * f), (x0f + L * f) * SS, Y2(yc + yj + T * f)], fill=c)
            # fuel spray
            for j in range(16):
                p = (t * 2.4 + j / 16.0) % 1.0
                sx = 974 + 40 * p
                sy = yc + (j % 5 - 2) * p * 9
                a = int(230 * (1 - p) * min(1.0, fi * 1.5))
                fd.ellipse([(sx - 2.5) * SS, Y2(sy - 2.5), (sx + 2.5) * SS, Y2(sy + 2.5)], fill=(250, 252, 255, a))
    if w[2] > 0.01:
        for k in range(3):
            p = ((t - STARTS[2]) * 0.9 + k / 3.0) % 1.0
            rr = (20 + 150 * p) * SS
            a = int(210 * (1 - p) * w[2])
            for s in (-1, 1):
                yc = CY + s * (R[1010] + HUB[1010]) / 2
                cx, cy = 1010 * SS, Y2(yc)
                fd.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(255, 245, 220, a), width=3 * SS)
    fx.putalpha(ImageChops.multiply(fx.getchannel('A'), FLOWMASK))

    glow = glow.filter(ImageFilter.GaussianBlur(4)).resize((W2, BH2), Image.BICUBIC)
    layer.alpha_composite(glow)
    layer.alpha_composite(pl)
    layer.alpha_composite(fx)

    # reveal wipe
    if xfront < W:
        layer.paste((0, 0, 0, 0), (int(xfront * SS), 0, W2, BH2))
    return layer


# ---------------------------------------------------------------- frame
def render_frame(t, S):
    xfront = 40 + (1900 - 40) * (1 - (1 - min(1.0, max(0.0, (t - 0.3) / 1.7))) ** 3)
    revealed = xfront >= 1900
    band = BG_BAND.copy()
    eng = ENG
    if not revealed:
        eng = ENG.copy()
        eng.paste((0, 0, 0, 0), (int(xfront * SS), 0, W2, BH2))
    band.alpha_composite(eng)
    band.alpha_composite(draw_dynamic(t, S, xfront))
    if not revealed and t > 0.3:
        sc = Image.new('RGBA', (W2, BH2), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sc)
        xx = xfront * SS
        sd.rectangle([xx - 18, 0, xx + 18, BH2], fill=(200, 230, 255, 40))
        sd.line([(xx, 0), (xx, BH2)], fill=(255, 255, 255, 230), width=4)
        band.alpha_composite(sc)
    full = BG1.copy()
    full.paste(band.convert('RGB').reduce(SS), (0, BY0))
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    # title block
    ta = smooth(t / 0.9)
    ttext(d, (120, 50), 'HOW A JET ENGINE WORKS', F_TITLE, WHITE + (int(255 * ta),), 4)
    ttext(d, (122, 114), 'SUCK  ·  SQUEEZE  ·  BANG  ·  BLOW', F_M20, LBLUE + (int(230 * ta),), 2)
    s1 = 'FIG. 01 — TURBOJET, SECTION A–A'
    s2 = 'NOT TO SCALE'
    d.text((1800 - F_M18.getlength(s1) - 2 * len(s1), 62), s1, font=F_M18, fill=LBLUE + (int(220 * ta),))
    ttext(d, (1800 - twidth(s2, F_M18, 2), 88), s2, F_M18, DIM + (int(220 * ta),), 2)

    w = [stage_w(i, t) for i in range(4)]
    # dimension line + component names
    for i, st in enumerate(STAGES):
        k = w[i]
        reach = smooth((t - 1.6) / 0.6)
        c = lerp(DIM, WHITE, k) + (int(255 * reach),)
        yd = 805
        d.line([(st['x0'] + 6, yd), (st['x1'] - 6, yd)], fill=c, width=2)
        d.line([(st['x0'] + 6, yd - 7), (st['x0'] + 6, yd + 7)], fill=c, width=2)
        d.line([(st['x1'] - 6, yd - 7), (st['x1'] - 6, yd + 7)], fill=c, width=2)
        tw = twidth(st['comp'], F_MB18 if k > 0.5 else F_M18, 2)
        ttext(d, ((st['x0'] + st['x1']) / 2 - tw / 2, 816), st['comp'], F_MB18 if k > 0.5 else F_M18, c, 2)

    # callouts
    for i, st in enumerate(STAGES):
        a = w[i]
        if a < 0.01:
            continue
        A = int(255 * a)
        wd = max(twidth(st['name'], F_WORD, 5), F_CAP.getlength(st['cap']))
        ax = min(max(st['x0'], 120), 1800 - wd)
        ttext(d, (ax, 166), 'STAGE ' + st['num'], F_M20, LBLUE + (A,), 4)
        ttext(d, (ax, 188), st['name'], F_WORD, WHITE + (A,), 5)
        d.text((ax, 292), st['cap'], font=F_CAP, fill=(214, 234, 255, A))
        yb = CY - (np.max(R[st['x0']:st['x1']]) + 30) - 28
        c = WHITE + (A,)
        d.line([(st['x0'], 336), (st['x0'], yb)], fill=c, width=2)
        d.line([(st['x0'], yb), (st['x1'], yb)], fill=c, width=2)
        d.line([(st['x1'], yb), (st['x1'], yb + 14)], fill=c, width=2)

    # thrust arrow
    a = w[3]
    if a > 0.01:
        A = int(255 * a)
        yt = 884
        run = smooth((t - STARTS[3] - 0.6) / 0.8)
        xa = 1560 - 560 * run
        d.line([(1560, yt), (xa + 22, yt)], fill=WHITE + (A,), width=6)
        d.polygon([(xa, yt), (xa + 30, yt - 16), (xa + 30, yt + 16)], fill=WHITE + (A,))
        ttext(d, (1580, yt - 12), 'THRUST', F_MB22, WHITE + (A,), 4)

    # stage chips
    cw, gap, x0c, yc0, chh = 380, 27, 160, 950, 56
    for i, st in enumerate(STAGES):
        x = x0c + i * (cw + gap)
        act = w[i]
        reach = smooth((t - STARTS[i]) / 0.35)
        base_a = smooth((t - 1.4) / 0.6)
        fill_a = int(255 * act + 40 * (reach - act))
        d.rounded_rectangle([x, yc0, x + cw, yc0 + chh], radius=10,
                            fill=(240, 248, 255, fill_a),
                            outline=lerp(DIM, WHITE, reach) + (int(255 * base_a),), width=2)
        tc = lerp(LBLUE, DARK, act) + (int(255 * base_a),)
        ttext(d, (x + 22, yc0 + 17), st['num'], F_MB18, tc, 1)
        ttext(d, (x + 62, yc0 + 11), st['name'], F_CHIP, tc, 4)
        prog = min(1.0, max(0.0, (t - STARTS[i]) / (ENDS[i] - STARTS[i])))
        if act > 0.01:
            d.rectangle([x + 14, yc0 + chh - 8, x + 14 + (cw - 28) * prog, yc0 + chh - 5], fill=(10, 70, 150, int(255 * act)))

    out = full.convert('RGBA')
    out.alpha_composite(ov)
    out = out.convert('RGB')
    g = smooth(t / 0.35) * (1 - smooth((t - 19.55) / 0.45))
    if g < 0.999:
        out = Image.blend(Image.new('RGB', (W, H), (4, 24, 60)), out, g)
    return out


def advance(S, frames):
    for _ in range(frames):
        step(S)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'still':
        t = float(sys.argv[2])
        S = make_state()
        advance(S, int(round(t * FPS)))
        render_frame(t, S).save(sys.argv[3])
    elif mode == 'chunk':
        ci, n, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        f0, f1 = ci * NF // n, (ci + 1) * NF // n
        S = make_state()
        advance(S, f0)
        tmp = out[:-4] + '.tmp.mp4'
        p = subprocess.Popen(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo',
                              '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                              '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
                              '-pix_fmt', 'yuv420p', tmp], stdin=subprocess.PIPE)
        for f in range(f0, f1):
            p.stdin.write(render_frame(f / FPS, S).tobytes())
            step(S)
        p.stdin.close()
        if p.wait() == 0:
            os.rename(tmp, out)
        else:
            sys.exit('ffmpeg failed')
