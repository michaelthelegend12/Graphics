"""15 s 1080p neon-green-on-black explainer: how a CPU runs one instruction (fetch, decode, execute).

    python3 sound.py                      # makes audio.wav
    python3 render.py cpu-cycle.mp4 audio.wav
    python3 render.py preview 2.0 6.5     # writes /tmp/prev_<t>.png for checking single frames

Needs numpy, OpenCV, Pillow and ffmpeg.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import timeline as TL

W, H = 1920, 1080
FPS, DUR = TL.FPS, TL.DUR
NEON = (57, 255, 20)
WHITE_G = (200, 255, 215)
DIM = (0, 70, 28)
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

_fonts = {}


def font(size, bold=True):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FB if bold else FR, size)
    return _fonts[k]


# ---------------------------------------------------------------- drawing helpers (sub-pixel AA)
def col(c, k):
    return tuple(int(max(0, min(255, v * k))) for v in c)


def _p(p):
    return (int(round(p[0] * 16)), int(round(p[1] * 16)))


def _a(pts):
    return np.round(np.asarray(pts, float) * 16).astype(np.int32)


def line(img, a, b, c, th):
    cv2.line(img, _p(a), _p(b), c, th, cv2.LINE_AA, 4)


def poly(img, pts, c, th, closed=True):
    cv2.polylines(img, [_a(pts)], closed, c, th, cv2.LINE_AA, 4)


def fill(img, pts, c):
    cv2.fillPoly(img, [_a(pts)], c, cv2.LINE_AA, 4)


def circ(img, ctr, r, c, th=-1):
    cv2.circle(img, _p(ctr), int(round(r * 16)), c, th, cv2.LINE_AA, 4)


def rr_pts(x0, y0, x1, y1, r, n=6):
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def tc(a):
    """text colour on a lit element: neon when dim, near-white when fully lit so bloom can't wash it out"""
    m = max(0.0, (a - 0.4) / 0.6)
    return col(tuple(NEON[i] + (WHITE_G[i] - NEON[i]) * m for i in range(3)), a)


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- geometry
def chamfer(pts, r=26):
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        p, v, n = (np.array(q, float) for q in (pts[i - 1], pts[i], pts[i + 1]))
        d1, d2 = np.linalg.norm(p - v), np.linalg.norm(n - v)
        rr = min(r, d1 / 2, d2 / 2)
        out.append(tuple(v + (p - v) / d1 * rr))
        out.append(tuple(v + (n - v) / d2 * rr))
    out.append(pts[-1])
    return out


class Path:
    def __init__(self, pts, r=26):
        self.pts = np.array(chamfer(pts, r), float)
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.cum = np.concatenate([[0], np.cumsum(seg)])
        self.L = float(self.cum[-1])
        n = max(2, int(self.L / 6))
        self.s = np.linspace(0, self.L, n + 1)
        self.xy = np.stack([np.interp(self.s, self.cum, self.pts[:, 0]),
                            np.interp(self.s, self.cum, self.pts[:, 1])], 1)
        self.mid = (self.s[:-1] + self.s[1:]) / 2

    def at(self, d):
        d = min(max(d, 0.0), self.L)
        return (float(np.interp(d, self.cum, self.pts[:, 0])), float(np.interp(d, self.cum, self.pts[:, 1])))


TRACES = {
    "addr": [(500, 280), (350, 280), (350, 490), (300, 490)],
    "data": [(300, 530), (390, 530), (390, 800), (500, 800)],
    "ir2dec": [(720, 800), (920, 800), (920, 640)],
    "dec2alu": [(1020, 540), (1184, 540)],
    "dec2reg": [(920, 440), (920, 280), (1150, 280)],
    "reg2alu_a": [(1230, 330), (1230, 490)],
    "reg2alu_b": [(1320, 330), (1320, 490)],
    "alu2res": [(1275, 630), (1275, 750)],
    "wb": [(1400, 800), (1440, 800), (1440, 280), (1400, 280)],
}
PATHS = {k: Path(v) for k, v in TRACES.items()}

BL = {
    "pc": (500, 230, 720, 330), "ir": (500, 750, 720, 850), "dec": (820, 440, 1020, 640),
    "reg": (1150, 230, 1400, 330), "res": (1150, 750, 1400, 850), "ram": (50, 400, 300, 680),
}
ALU_PTS = [(1170, 490), (1250, 490), (1275, 520), (1300, 490), (1380, 490), (1340, 630), (1210, 630)]
SHAPES = {k: rr_pts(*v, 10) for k, v in BL.items()}
SHAPES["alu"] = ALU_PTS
LABELS = {"pc": "PROGRAM COUNTER", "ir": "INSTRUCTION REG", "dec": "DECODER", "reg": "REGISTERS", "res": "RESULT"}
RAM_ROWS = [("0x0038", "LDR R1,#5"), ("0x003C", "LDR R2,#3"), ("0x0040", "ADD R1,R2"), ("0x0044", "STR R1,[X]"),
            ("0x0048", "SUB R3,R1"), ("0x004C", "JMP 0x0038"), ("0x0050", "NOP")]
RAM_Y0, RAM_DY = 438, 36
PILLS = [("01  FETCH", 600), ("02  DECODE", 960), ("03  EXECUTE", 1320)]
PILL_Y0, PILL_Y1 = 972, 1018


def draw_text(img, items):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for x, y, s, size, bold, c, anc in items:
        d.text((x, y), s, font=font(size, bold), fill=c, anchor=anc)
    return np.array(im)


# ---------------------------------------------------------------- static background
def build_base():
    img = np.zeros((H, W, 3), np.uint8)
    img[::60, ::60] = (0, 30, 11)
    T = []
    # chip package + die
    pkg = rr_pts(420, 150, 1500, 930, 38)
    fill(img, pkg, (2, 12, 7))
    poly(img, pkg, (0, 100, 40), 3)
    die = rr_pts(450, 180, 1470, 900, 14)
    fill(img, die, (1, 7, 4))
    poly(img, die, (0, 62, 25), 2)
    fill(img, [(436, 166), (462, 166), (436, 192)], (0, 70, 28))  # pin-1 notch
    pin = (0, 85, 34)
    for x in range(470, 1460, 40):
        line(img, (x, 150), (x, 124), pin, 3)
        line(img, (x, 930), (x, 956), pin, 3)
    for y in range(200, 890, 40):
        line(img, (420, y), (394, y), pin, 3)
        line(img, (1500, y), (1526, y), pin, 3)
    # little stub traces off the right/bottom pins
    rng = np.random.default_rng(7)
    for y in range(200, 890, 40):
        n = int(rng.integers(0, 3))
        if n == 0:
            continue
        x = 1526
        poly(img, [(x, y), (x + 20, y), (x + 20 + 14 * n, y + (14 * n if y < 540 else -14 * n))], (0, 48, 19), 2, False)
    for x in range(470, 1460, 40):
        if rng.random() < 0.4:
            poly(img, [(x, 956), (x, 966)], (0, 48, 19), 2, False)
    # capacitor-like specks inside the die
    for i in range(18):
        x = 540 + i * 50
        for y in (200, 876):
            cv2.rectangle(img, (x, y - 5), (x + 10, y + 5), (0, 42, 17), -1)
    T.append((470, 204, "CPU CORE", 14, True, (0, 90, 36), "lm"))
    # traces + pads
    for k, pa in PATHS.items():
        poly(img, pa.pts, DIM, 3, False)
        for e in (pa.pts[0], pa.pts[-1]):
            circ(img, e, 6, DIM, 2)
    # blocks
    for k, pts in SHAPES.items():
        fill(img, pts, (2, 16, 8))
        poly(img, pts, (0, 80, 32), 2)
    for k, s in LABELS.items():
        x0, y0 = BL[k][0], BL[k][1]
        T.append((x0 + 12, y0 + 17, s, 16, True, (0, 120, 48), "lm"))
    T.append((1275, 612, "ALU", 20, True, (0, 120, 48), "mm"))
    T.append((50, 376, "MEMORY (RAM)", 18, True, (0, 120, 48), "lm"))
    for i, (a, ins) in enumerate(RAM_ROWS):
        y = RAM_Y0 + i * RAM_DY
        T.append((62, y, a, 16, False, (0, 105, 42), "lm"))
        T.append((142, y, ins, 16, False, (0, 105, 42), "lm"))
    # decoder matrix
    for i in range(6):
        x = 850 + i * 28
        line(img, (x, 490), (x, 596), (0, 44, 18), 1)
    for j in range(5):
        y = 500 + j * 24
        line(img, (846, y), (994, y), (0, 44, 18), 1)
    # stage strip
    for i, (s, cx) in enumerate(PILLS):
        pts = rr_pts(cx - 140, PILL_Y0, cx + 140, PILL_Y1, 23)
        fill(img, pts, (2, 14, 7))
        poly(img, pts, (0, 70, 28), 2)
        T.append((cx, (PILL_Y0 + PILL_Y1) // 2, s, 26, True, (0, 110, 44), "mm"))
        if i < 2:
            ax = cx + 150
            line(img, (ax, 995), (ax + 50, 995), (0, 70, 28), 2)
            poly(img, [(ax + 40, 987), (ax + 52, 995), (ax + 40, 1003)], (0, 70, 28), 2, False)
    # side panels
    for (x0, y0, x1, y1) in ((1600, 190, 1890, 520), (1600, 545, 1890, 655), (1600, 700, 1890, 900)):
        pts = rr_pts(x0, y0, x1, y1, 10)
        fill(img, pts, (1, 8, 4))
        poly(img, pts, (0, 62, 25), 2)
    T.append((1612, 210, "// TRACE LOG", 17, True, (0, 120, 48), "lm"))
    T.append((1612, 565, "CLK  3.0 GHz", 16, True, (0, 120, 48), "lm"))
    T.append((1612, 720, "THROUGHPUT", 17, True, (0, 120, 48), "lm"))
    line(img, (0, 1076), (W, 1076), (0, 36, 14), 2)
    return draw_text(img, T)


BASE = build_base()
EVENTS = {}
for pid, tr, lab, t0, t1 in TL.PACKETS:
    EVENTS.setdefault(tr, []).append((t0, t1))


def block_env(name, t):
    a, seen = 0.0, False
    for t0, t1 in TL.BLOCKS[name]:
        if t >= t0:
            seen = True
            v = min(1.0, (t - t0) / 0.12) if t <= t1 else 0.15 + 0.85 * math.exp(-(t - t1) / 0.45)
            a = max(a, v)
    return max(a, 0.14) if seen else 0.0


def stage_env(i, t):
    _, t0, t1 = TL.STAGES[i]
    if t < t0:
        return 0.0
    if t <= t1:
        return min(1.0, (t - t0) / 0.12)
    return 0.32


# ---------------------------------------------------------------- one frame
def type_count(t, t0, rate):
    return max(0, int((t - t0) * rate))


def title_text(t):
    a, b = TL.TITLE_1, TL.TITLE_2
    if t < 0.2:
        return ""
    if t < 1.2:
        return a[:type_count(t, 0.2, 34)]
    if t < 12.7:
        return a
    if t < 13.0:
        return a[:max(0, len(a) - type_count(t, 12.7, 100))]
    return b[:type_count(t, 13.0, 30)]


def frame(t):
    G = np.zeros((H, W, 3), np.uint8)
    T = []
    PT = []  # dark packet labels, drawn after the bloom so they stay crisp
    # boot scan sweeps across the chip
    if 0.05 < t < 1.25:
        xs = 400 + (t - 0.05) / 1.15 * 1150
        x0, x1 = int(max(0, xs - 90)), int(min(W, xs + 90))
        if x1 > x0:
            w = 1 - np.abs(np.arange(x0, x1) - xs) / 90.0
            g = BASE[120:960, x0:x1, 1].astype(np.float32) * 4.5 * w[None, :]
            ratio = np.array(NEON, np.float32) / 255.0
            G[120:960, x0:x1] = np.clip(g[:, :, None] * ratio[None, None, :], 0, 255).astype(np.uint8)

    # lit traces
    for name, pa in PATHS.items():
        ev = EVENTS.get(name, [])
        if not ev or t < min(e[0] for e in ev):
            continue
        b = np.zeros(len(pa.mid))
        for t0, t1 in ev:
            tp = t0 + (t1 - t0) * pa.mid / pa.L
            b = np.maximum(b, np.where(t >= tp, 0.3 + 0.7 * np.exp(-(t - tp) / 0.9), 0.0))
        for i in np.nonzero(b > 0.03)[0]:
            line(G, pa.xy[i], pa.xy[i + 1], col(NEON, b[i]), 3)
        t_first = min(e[0] for e in ev)
        circ(G, pa.pts[0], 7, col(NEON, 0.3 + 0.7 * math.exp(-(t - t_first) / 0.6)))
        for t0, t1 in ev:
            if t >= t1:
                circ(G, pa.pts[-1], 7, col(NEON, 0.3 + 0.7 * math.exp(-(t - t1) / 0.6)))
                break

    # stage pulse on the die outline
    for name, t0, t1 in TL.STAGES:
        if t >= t0:
            k = 0.8 * math.exp(-(t - t0) / 0.45)
            if k > 0.03:
                poly(G, rr_pts(450, 180, 1470, 900, 14), col(NEON, k), 2)

    # blocks
    A = {n: block_env(n, t) for n in TL.BLOCKS}
    for name, a in A.items():
        if a < 0.03:
            continue
        pts = SHAPES[name]
        fill(G, pts, (0, int(22 * a), int(7 * a)))
        poly(G, pts, col(NEON, a), 2)
        if name in LABELS:
            T.append((BL[name][0] + 12, BL[name][1] + 17, LABELS[name], 16, True, tc(a), "lm"))
    # memory rows
    a = A["ram"]
    if a > 0.03:
        hi = min(2, int((t - 2.2) / 0.1)) if t < 2.5 else 2
        k = 0.6 if t < 2.5 else a
        y = RAM_Y0 + hi * RAM_DY
        fill(G, rr_pts(58, y - 17, 292, y + 17, 4), (0, int(26 * k), int(8 * k)))
        T.append((62, y, RAM_ROWS[hi][0], 16, True, tc(k), "lm"))
        T.append((142, y, RAM_ROWS[hi][1], 16, True, tc(k), "lm"))
    T.append((50, 376, "MEMORY (RAM)", 18, True, tc(A["ram"]), "lm")) if A["ram"] > 0.03 else None
    T.append((1275, 612, "ALU", 20, True, tc(A["alu"]), "mm")) if A["alu"] > 0.03 else None
    # PC / IR / registers / result values
    pc_txt = "0x0044" if t >= 13.0 else "0x0040"
    k = max(A["pc"], 0.3) if t >= 1.0 else 0.0
    if t >= 13.0:
        k = min(1.0, k + 0.7 * math.exp(-(t - 13.0) / 0.3))
    if k:
        T.append((610, 298, pc_txt, 30, True, tc(k), "mm"))
    if t >= 1.0:
        if t >= 3.9:
            T.append((610, 812, "ADD R1,R2", 26, True, tc(max(A["ir"], 0.5)), "mm"))
        else:
            T.append((610, 812, "--------", 26, True, col(NEON, 0.2), "mm"))
    if t >= 0.9:
        k = max(A["reg"], 0.35)
        r1 = "8" if t >= 12.2 else "5"
        k1 = k if t < 12.2 else min(1.0, k + 0.6 * math.exp(-(t - 12.2) / 0.4))
        T.append((1215, 298, "R1 = " + r1, 26, True, tc(k1), "mm"))
        T.append((1335, 298, "R2 = 3", 26, True, tc(k), "mm"))
        T.append((1275, 805, "8" if t >= 11.0 else "--", 40, True, tc(max(A["res"], 0.3) if t >= 11.0 else 0.2), "mm"))
    # decoder internals
    a = A["dec"]
    if a > 0.03:
        active = 5.7 <= t <= 7.6
        for i in range(6):
            x = 850 + i * 28
            kk = a * (0.25 + 0.75 * max(0.0, math.sin(t * 16 - i * 1.1))) if active else a * 0.3
            line(G, (x, 490), (x, 596), col(NEON, kk), 1)
        for j in range(5):
            y = 500 + j * 24
            kk = a * (0.25 + 0.75 * max(0.0, math.sin(t * 13 + j * 1.3))) if active else a * 0.3
            line(G, (846, y), (994, y), col(NEON, kk), 1)
        if active:
            fr = int(t * 14)
            for i in range(6):
                for j in range(5):
                    if (i * 7 + j * 13 + fr * 5 + (i * j * fr)) % 4 == 0:
                        circ(G, (850 + i * 28, 500 + j * 24), 4, col(WHITE_G, a))
        if t >= 6.7:
            T.append((920, 620, "OP:ADD SRC:R1,R2", 16, True, tc(max(a, 0.5)), "mm"))
    # ALU internals
    a = A["alu"]
    if a > 0.03:
        if t < 9.1:
            sym, sz = "+", 40
        elif t < 10.5:
            sym, sz = "5 + 3", 32
        else:
            sym, sz = "= 8", 32
        T.append((1275, 568, sym, sz, True, tc(max(a, 0.5)), "mm"))
        if 9.1 <= t <= 10.5:
            yy = 545 + 70 * (0.5 + 0.5 * math.sin((t - 9.1) * 14))
            line(G, (1215, yy), (1335, yy), col(WHITE_G, 0.8), 2)

    # packets
    for pid, tr, lab, t0, t1 in TL.PACKETS:
        if not (t0 <= t <= t1 + 0.15):
            continue
        pa = PATHS[tr]
        if t <= t1:
            d = pa.L * (t - t0) / (t1 - t0)
            k = min(1.0, (t - t0) / 0.1)
        else:
            d = pa.L
            k = 1 - (t - t1) / 0.15
        pos = pa.at(d)
        for s0 in np.arange(max(0, d - 70), d, 5):  # hot comet tail
            f = 1 - (d - s0) / 70
            line(G, pa.at(s0), pa.at(min(d, s0 + 5)), col(WHITE_G, 0.9 * f * k), 5)
        w = font(21).getlength(lab) + 28
        sc = 0.4 + 0.6 * k
        hw, hh = w / 2 * sc, 17 * sc
        pts = rr_pts(pos[0] - hw, pos[1] - hh, pos[0] + hw, pos[1] + hh, hh)
        fill(G, pts, col((140, 255, 160), k))
        poly(G, pts, col((235, 255, 240), k), 2)
        if k > 0.7:
            PT.append((pos[0], pos[1] + 1, lab, 21, True, (0, 28, 6), "mm"))

    # clock wave
    if t > 0.9:
        xs = np.arange(1612, 1879, 2)
        ph = ((xs - t * 170) / 40.0) % 1
        ys = np.where(ph < 0.5, 593, 633)
        poly(G, np.stack([xs, ys], 1), col(NEON, 0.75), 2, False)

    # console
    for i, (tt, s) in enumerate(TL.CONSOLE):
        if t < tt:
            continue
        n = type_count(t, tt, 45)
        txt = s[:n]
        age = t - tt
        k = 1.0 if age < 0.8 else 0.55
        last = i == max(j for j, (q, _) in enumerate(TL.CONSOLE) if t >= q)
        if last and n < len(s) + 6 and int(t * 4) % 2 == 0:
            txt += "_"
        T.append((1612, 252 + i * 30, txt, 19, True, col(NEON, k), "lm"))

    # throughput counter
    if t >= 12.9:
        v = 3.2e9 * smooth((t - 12.9) / 1.1)
        T.append((1612, 785, f"{int(v):,}", 30, True, col(NEON, 1.0), "lm"))
        T.append((1612, 825, "INSTRUCTIONS / SEC", 16, True, col(NEON, 0.6), "lm"))
        cv2.rectangle(G, (1612, 856), (1612 + int(264 * smooth((t - 12.9) / 1.1)), 872), col(NEON, 0.8), -1)
    else:
        T.append((1612, 785, "---", 30, True, col(NEON, 0.25), "lm"))

    # stage strip
    steps = [s for s in TL.STEPS if s <= t]
    for i, (s, cx) in enumerate(PILLS):
        if t >= TL.LOOP_T:
            k = steps[-1] if steps else TL.LOOP_T
            idx = (len(steps) - 1) % 3
            a = (1.0 if idx == i else 0.4) * (1.0 if idx != i else 0.55 + 0.45 * math.exp(-(t - k) / 0.12))
            a = max(a, 0.35)
        else:
            a = stage_env(i, t)
        if a < 0.03:
            continue
        pts = rr_pts(cx - 140, PILL_Y0, cx + 140, PILL_Y1, 23)
        fill(G, pts, (0, int(24 * a), int(8 * a)))
        poly(G, pts, col(NEON, a), 2)
        T.append((cx, (PILL_Y0 + PILL_Y1) // 2, s, 26, True, tc(a), "mm"))
        if i < 2 and a > 0.1:
            ax = cx + 150
            line(G, (ax, 995), (ax + 50, 995), col(NEON, a * 0.8), 2)
            poly(G, [(ax + 40, 987), (ax + 52, 995), (ax + 40, 1003)], col(NEON, a * 0.8), 2, False)
    if t >= TL.LOOP_T:  # return arrow EXECUTE -> FETCH
        k = smooth((t - TL.LOOP_T) / 0.4)
        pts = [(1320, 1018), (1320, 1052), (600, 1052), (600, 1024)]
        poly(G, pts, col(NEON, 0.8 * k), 2, False)
        poly(G, [(592, 1034), (600, 1020), (608, 1034)], col(NEON, 0.8 * k), 2, False)
        tot = 34 + 720 + 28
        d = ((t - TL.LOOP_T) * 700) % tot
        # dot rides the return path
        seg = [(1320, 1018, 1320, 1052, 34), (1320, 1052, 600, 1052, 720), (600, 1052, 600, 1024, 28)]
        for x0, y0, x1, y1, ln in seg:
            if d <= ln:
                circ(G, (x0 + (x1 - x0) * d / ln, y0 + (y1 - y0) * d / ln), 6, col(WHITE_G, k))
                break
            d -= ln

    # progress line
    line(G, (0, 1076), (W * t / DUR, 1076), col(NEON, 0.7), 2)

    # title
    ttl = title_text(t)
    if ttl or t < 0.3:
        typing = (0.2 <= t < 1.2) or (12.7 <= t < 14.7)
        cur = "_" if (typing or int(t * 3) % 2 == 0) else ""
        T.append((60, 62, ttl + cur, 36, True, col(NEON, 1.0), "lm"))

    G = draw_text(G, T)

    fade = 0.25 + 0.75 * smooth(t / 0.8)
    base = cv2.convertScaleAbs(BASE, alpha=fade) if fade < 1 else BASE
    out = cv2.add(base, G)
    small = cv2.resize(G, (480, 270), interpolation=cv2.INTER_AREA)
    b1 = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3.5), (W, H), interpolation=cv2.INTER_LINEAR)
    tiny = cv2.resize(small, (120, 68), interpolation=cv2.INTER_AREA)
    b2 = cv2.resize(cv2.GaussianBlur(tiny, (0, 0), 3.0), (W, H), interpolation=cv2.INTER_LINEAR)
    out = cv2.add(out, cv2.addWeighted(b1, 1.2, b2, 1.6, 0))
    if PT:
        out = draw_text(out, PT)
    # end fade
    if t > DUR - 0.35:
        out = cv2.convertScaleAbs(out, alpha=max(0.0, (DUR - t) / 0.35))
    return out


# ---------------------------------------------------------------- main
def render(out_path, audio=None):
    tmp = "/tmp/cpu-cycle-video.part.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    for i in range(n):
        p.stdin.write(frame(i / FPS).tobytes())
        if i % 60 == 0:
            print(f"frame {i}/{n}", flush=True)
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed")
    if audio:
        tmp2 = "/tmp/cpu-cycle-final.part.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-i", audio, "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", tmp2], check=True)
        os.replace(tmp2, out_path)
    else:
        os.replace(tmp, out_path)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "preview":
        for s in a[1:]:
            Image.fromarray(frame(float(s))).save(f"/tmp/prev_{s}.png")
    else:
        render(a[0] if a else "cpu-cycle.mp4", a[1] if len(a) > 1 else None)
