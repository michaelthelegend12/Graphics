#!/usr/bin/env python3
"""Smartwatch turntable: 10 s, 1920x1080, 30 fps.

A smartwatch built only from shapes (rounded-box case, glass, crown, button, a swept
band loop), rendered with a tiny software 3D engine (numpy + OpenCV), studio-style
reflections, a soft floor reflection and shadow, then 'Pulse. Always on.' slides in.

    python3 render.py video out.mp4        # silent video, two scenes rendered in parallel
    python3 render.py still 4.5 out.png    # one frame at t=4.5 s (for checking)
"""
import sys, os, math, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)

W, H, FPS, DUR = 1920, 1080, 30, 10
NF = FPS * DUR
SS = 2                      # supersampling for the watch
RW, RH = 960, 1080          # region around the watch that is rendered
FONTS = '/usr/share/fonts/opentype/inter/'

# ----------------------------------------------------------------------------- helpers
def smooth(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)

def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

# ----------------------------------------------------------------------------- geometry
YC = 1.085          # height of the case centre above the floor
ZPIV = -0.30        # the watch spins about this z (centre of the whole assembly)
CASE_W, CASE_H, CASE_R, CASE_D = 1.0, 1.2, 0.28, 0.15   # D is half depth
BAND_T, BAND_W = 0.07, 0.80
LOOP_AY, LOOP_AZ = 1.05, 0.55
GLASS_Z = 0.15

# layers: 0 band, 1 case, 2 crown+button
# materials: 0 case metal, 1 band, 2 crown metal, 3 coral, 4 ceramic back, 5 sensor glass


def rrect(w, h, r, n=12):
    pts, nrm = [], []
    for (cx, cy, a0) in [(w/2-r, h/2-r, 0), (-(w/2-r), h/2-r, 90),
                         (-(w/2-r), -(h/2-r), 180), (w/2-r, -(h/2-r), 270)]:
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
            nrm.append((math.cos(a), math.sin(a)))
    return np.array(pts), np.array(nrm)


class Mesh:
    def __init__(self):
        self.v, self.n, self.m, self.t, self.l = [], [], [], [], []

    def add(self, verts, normal, mat, layer, tone=1.0):
        self.v.append(np.asarray(verts, float))
        self.n.append(unit(normal))
        self.m.append(mat); self.l.append(layer); self.t.append(tone)

    def finish(self):
        K = max(len(x) for x in self.v)
        N = len(self.v)
        PV = np.zeros((N, K, 3)); PK = np.zeros(N, int)
        for i, x in enumerate(self.v):
            PK[i] = len(x)
            PV[i, :len(x)] = x
            PV[i, len(x):] = x[-1]
        CL = np.array([x.mean(axis=0) for x in self.v])
        self.PV, self.PK, self.CL = PV, PK, CL
        self.NRM = np.array(self.n); self.MAT = np.array(self.m)
        self.LAY = np.array(self.l); self.TONE = np.array(self.t)
        return self


def cyl_x(M, x0, x1, y0, z0, r, n, mat, layer, tones=(1, 1), cap=True):
    for k in range(n):
        a0, a1 = 2*math.pi*k/n, 2*math.pi*(k+1)/n
        am = (a0 + a1) / 2
        v = [(x0, y0+r*math.cos(a0), z0+r*math.sin(a0)), (x1, y0+r*math.cos(a0), z0+r*math.sin(a0)),
             (x1, y0+r*math.cos(a1), z0+r*math.sin(a1)), (x0, y0+r*math.cos(a1), z0+r*math.sin(a1))]
        M.add(v, (0, math.cos(am), math.sin(am)), mat, layer, tones[k % 2])
    if cap:
        v = [(x1, y0+r*math.cos(2*math.pi*k/n), z0+r*math.sin(2*math.pi*k/n)) for k in range(n)]
        M.add(v, (1, 0, 0), mat, layer)


def cyl_z(M, z0, z1, r, n, mat, layer, tone=1.0, cap=True, rot=0.0, yc=YC):
    """cylinder along z (z0 < z1), centred on the case axis; cap at z0 (back)."""
    for k in range(n):
        a0, a1 = 2*math.pi*k/n, 2*math.pi*(k+1)/n
        am = (a0 + a1) / 2
        v = [(r*math.cos(a0), yc + r*math.sin(a0), z0), (r*math.cos(a0), yc + r*math.sin(a0), z1),
             (r*math.cos(a1), yc + r*math.sin(a1), z1), (r*math.cos(a1), yc + r*math.sin(a1), z0)]
        M.add(v, (math.cos(am), math.sin(am), 0), mat, layer, tone)
    if cap:
        v = [(r*math.cos(2*math.pi*k/n), yc + r*math.sin(2*math.pi*k/n), z0) for k in range(n)]
        M.add(v, (0, 0, -1), mat, layer, tone)


def build_mesh():
    M = Mesh()
    o, on = rrect(CASE_W, CASE_H, CASE_R)
    i_, _ = rrect(CASE_W - 0.08, CASE_H - 0.08, CASE_R - 0.04)
    N = len(o)
    zb, zf = -CASE_D, CASE_D - 0.025
    for a in range(N):
        b = (a + 1) % N
        nm = on[a] + on[b]
        # side wall
        v = [(o[a][0], YC+o[a][1], zb), (o[b][0], YC+o[b][1], zb),
             (o[b][0], YC+o[b][1], zf), (o[a][0], YC+o[a][1], zf)]
        M.add(v, (nm[0], nm[1], 0), 0, 1)
        # front chamfer
        v = [(o[a][0], YC+o[a][1], zf), (o[b][0], YC+o[b][1], zf),
             (i_[b][0], YC+i_[b][1], GLASS_Z), (i_[a][0], YC+i_[a][1], GLASS_Z)]
        nn = unit(nm)
        M.add(v, (nn[0]*0.53, nn[1]*0.53, 0.85), 0, 1, 1.05)
    # back plate and sensor bump
    M.add([(p[0], YC+p[1], zb) for p in o], (0, 0, -1), 4, 1)
    cyl_z(M, zb - 0.035, zb, 0.235, 32, 4, 1, 1.0, cap=False)
    cyl_z(M, zb - 0.035, zb, 0.235, 32, 4, 1, 1.0, cap=False)
    M.add([(0.235*math.cos(2*math.pi*k/32), YC+0.235*math.sin(2*math.pi*k/32), zb-0.035) for k in range(32)],
          (0, 0, -1), 4, 1, 1.4)
    M.add([(0.14*math.cos(2*math.pi*k/28), YC+0.14*math.sin(2*math.pi*k/28), zb-0.038) for k in range(28)],
          (0, 0, -1), 5, 1)
    # crown (cylinder along +x), coral ring, knurled cap
    cy = YC + 0.20
    cyl_x(M, 0.49, 0.525, cy, 0.0, 0.085, 28, 2, 2, cap=False)
    cyl_x(M, 0.525, 0.54, cy, 0.0, 0.095, 28, 3, 2, cap=False)
    cyl_x(M, 0.54, 0.595, cy, 0.0, 0.088, 36, 2, 2, tones=(1.0, 0.72), cap=True)
    # side button (box)
    bx0, bx1, by0, by1, bz = 0.49, 0.535, YC - 0.27, YC - 0.07, 0.045
    M.add([(bx1, by0, -bz), (bx1, by1, -bz), (bx1, by1, bz), (bx1, by0, bz)], (1, 0, 0), 2, 2)
    M.add([(bx0, by1, -bz), (bx1, by1, -bz), (bx1, by1, bz), (bx0, by1, bz)], (0, 1, 0), 2, 2)
    M.add([(bx0, by0, -bz), (bx1, by0, -bz), (bx1, by0, bz), (bx0, by0, bz)], (0, -1, 0), 2, 2)
    M.add([(bx0, by0, bz), (bx1, by0, bz), (bx1, by1, bz), (bx0, by1, bz)], (0, 0, 1), 2, 2)
    M.add([(bx0, by0, -bz), (bx1, by0, -bz), (bx1, by1, -bz), (bx0, by1, -bz)], (0, 0, -1), 2, 2)

    # band: swept strip around an ellipse in the y-z plane
    zc = -0.02 - LOOP_AZ
    ayc = LOOP_AY
    yb = YC          # loop centre height
    phi0 = math.asin(0.52 / ayc)
    nseg = 150
    phis = np.linspace(phi0, 2*math.pi - phi0, nseg + 1)
    cen = np.stack([yb + ayc*np.sin(phis), zc + LOOP_AZ*np.cos(phis)], 1)       # (y, z)
    nor = unit(np.stack([np.sin(phis)/ayc, np.cos(phis)/LOOP_AZ], 1))
    hw = BAND_W / 2
    def pt(i, side, sgn):  # side: +1 outer, -1 inner; sgn: -1 left x, +1 right x
        c = cen[i] + nor[i] * side * BAND_T / 2
        return (sgn * hw, c[0], c[1])
    for i in range(nseg):
        j = i + 1
        nm = nor[i] + nor[j]
        M.add([pt(i, 1, -1), pt(i, 1, 1), pt(j, 1, 1), pt(j, 1, -1)], (0, nm[0], nm[1]), 1, 0)
        M.add([pt(i, -1, -1), pt(i, -1, 1), pt(j, -1, 1), pt(j, -1, -1)], (0, -nm[0], -nm[1]), 1, 0, 0.9)
        M.add([pt(i, 1, 1), pt(i, -1, 1), pt(j, -1, 1), pt(j, 1, 1)], (1, 0, 0), 1, 0, 1.1)
        M.add([pt(i, 1, -1), pt(i, -1, -1), pt(j, -1, -1), pt(j, 1, -1)], (-1, 0, 0), 1, 0, 1.1)
    return M.finish()


MESH = build_mesh()
GLASS_LOCAL = np.array([[-0.46, YC+0.56, GLASS_Z], [0.46, YC+0.56, GLASS_Z],
                        [0.46, YC-0.56, GLASS_Z], [-0.46, YC-0.56, GLASS_Z]])

# ----------------------------------------------------------------------------- lighting
L1 = unit((-0.55, 0.75, 0.55)); L2 = unit((0.8, 0.25, 0.5)); L3 = unit((0.25, 0.5, -0.9))
SOFT = [(unit((-0.55, 0.55, 0.62)), 1.0, 16), (unit((0.85, 0.3, -0.45)), 1.2, 70),
        (unit((0.0, 1.0, 0.15)), 0.9, 8), (unit((-0.9, 0.2, -0.35)), 0.7, 55),
        (unit((0.1, 0.25, -1.0)), 0.6, 30), (unit((0.2, 0.15, 1.0)), 0.5, 26)]
FLAG = [(unit((0.65, 0.05, 0.75)), 0.55, 6), (unit((-0.8, -0.1, 0.1)), 0.35, 8)]


def env(r):
    base = 0.28 + 0.60 * smooth(-0.35, 0.65, r[:, 1])
    lobe = np.zeros(len(r)); flag = np.zeros(len(r))
    for d, k, p in SOFT:
        lobe += k * np.maximum(0, r @ d) ** p
    for d, k, p in FLAG:
        flag += k * np.maximum(0, r @ d) ** p
    return base, lobe, np.clip(flag, 0, 0.8)


TINT = np.array([0.50, 0.52, 0.57])
CORAL = np.array([1.0, 0.34, 0.28])


def shade(C, n, cam_s, mat, tone):
    v = unit(cam_s - C)
    nv = np.sum(n * v, 1)
    r = 2 * nv[:, None] * n - v
    base, lobe, flag = env(r)
    d1 = np.maximum(0, n @ L1); d2 = np.maximum(0, n @ L2); d3 = np.maximum(0, n @ L3)
    col = np.zeros((len(C), 3))
    # metal (case + crown)
    m = (mat == 0) | (mat == 2)
    cm = TINT * (0.10 + base * (1 - flag))[:, None] + (0.35 + 0.55 * TINT) * lobe[:, None] + TINT * 0.10 * d1[:, None]
    col[m] = cm[m]
    # band (matte rubber)
    m = mat == 1
    A = np.array([0.085, 0.088, 0.10])
    cb = A * (0.9 + 2.7 * d1 + 0.9 * d2 + 0.6 * d3)[:, None] \
        + (0.16 * lobe + 0.04 * base + 0.10 * (1 - np.abs(nv)) ** 3)[:, None] * np.array([1, 1, 1.05])
    col[m] = cb[m]
    # coral accent
    m = mat == 3
    cc = CORAL * (0.40 + 0.75 * d1 + 0.25 * d2)[:, None] + 0.5 * lobe[:, None]
    col[m] = cc[m]
    # ceramic back
    m = mat == 4
    ce = np.array([0.045, 0.045, 0.055]) + (0.25 * lobe + 0.08 * base)[:, None]
    col[m] = ce[m]
    # sensor glass
    m = mat == 5
    cs = np.array([0.015, 0.03, 0.07]) + (0.7 * lobe + 0.10 * base)[:, None]
    col[m] = cs[m]
    col *= tone[:, None]
    return np.clip(col, 0, 1)


# ----------------------------------------------------------------------------- display texture
GW, GH = 920, 1120
FT = lambda name, size: ImageFont.truetype(FONTS + name, size)


def rr_mask(w, h, r, inset=0):
    m = Image.new('L', (w * 2, h * 2), 0)
    ImageDraw.Draw(m).rounded_rectangle([inset*2, inset*2, w*2 - 1 - inset*2, h*2 - 1 - inset*2], radius=r*2, fill=255)
    return np.array(m.resize((w, h), Image.LANCZOS))


def build_static():
    img = Image.new('RGB', (GW, GH), (0, 0, 0))
    d = ImageDraw.Draw(img)
    # bezel ring (slightly lighter than the display)
    bez = Image.new('RGB', (GW, GH), (16, 16, 19))
    disp = rr_mask(GW, GH, 204, inset=36)
    img = Image.composite(Image.new('RGB', (GW, GH), (0, 0, 0)), bez, Image.fromarray(disp))
    d = ImageDraw.Draw(img)
    f_date = FT('Inter-Medium.otf', 46)
    f_time = FT('Inter-SemiBold.otf', 270)
    f_num = FT('Inter-SemiBold.otf', 150)
    f_bpm = FT('Inter-SemiBold.otf', 40)
    d.text((GW / 2, 170), 'FRI 13', font=f_date, fill=(150, 150, 158), anchor='mm')
    d.text((GW / 2, 345), '10:09', font=f_time, fill=(246, 246, 248), anchor='mm')
    d.text((505, 925), '72', font=f_num, fill=(246, 246, 248), anchor='lm')
    d.text((735, 950), 'BPM', font=f_bpm, fill=(255, 92, 78), anchor='lm')
    return np.array(img)


STATIC = build_static()
TEXMASK = rr_mask(GW, GH, 240)


def ecg(ph):
    g = lambda c, w: np.exp(-((ph - c) / w) ** 2)
    return 0.10 * g(0.18, 0.035) - 0.14 * g(0.36, 0.010) + 1.0 * g(0.39, 0.009) - 0.28 * g(0.42, 0.011) + 0.20 * g(0.64, 0.05)


def make_texture(t):
    img = STATIC.copy()
    # scrolling ECG (RGB layer), head on the right
    x = np.arange(90, 841)
    tau = t - (840 - x) / 750.0 * 2.0
    y = 640 - 175 * ecg((tau * 1.2) % 1.0)
    fade = ((x - 90) / 750.0) ** 1.3
    layer = np.zeros((GH, GW, 3), np.float32)
    for i in range(len(x) - 1):
        c = np.array([78, 92, 255]) * (0.15 + 0.85 * fade[i])      # BGR coral
        cv2.line(layer, (int(x[i]), int(y[i])), (int(x[i+1]), int(y[i+1])), c.tolist(), 8, cv2.LINE_AA)
    cv2.circle(layer, (840, int(y[-1])), 13, (150, 170, 255), -1, cv2.LINE_AA)
    band = layer[440:840]
    glow = cv2.GaussianBlur(band, (0, 0), 16)
    layer[440:840] = np.clip(band + glow * 1.5, 0, 255)
    # heart (beats with the R spike at the head)
    ph = (t * 1.2) % 1.0
    s = 1 + 0.20 * math.exp(-((ph - 0.39) / 0.07) ** 2) + 0.06 * math.exp(-((ph - 0.62) / 0.08) ** 2)
    a = np.linspace(0, 2 * math.pi, 80)
    hx = 16 * np.sin(a) ** 3
    hy = -(13 * np.cos(a) - 5 * np.cos(2*a) - 2 * np.cos(3*a) - np.cos(4*a))
    sc = 5.4 * s
    pts = np.stack([410 + hx * sc, 925 + hy * sc + 6], 1)
    hl = np.zeros((GH, GW, 3), np.float32)
    cv2.fillPoly(hl, [np.round(pts * 16).astype(np.int32)], (78, 92, 255), cv2.LINE_AA, 4)
    hl = hl + cv2.GaussianBlur(hl, (0, 0), 10) * 0.6
    out = np.clip(img.astype(np.float32)[..., ::-1] + layer + hl, 0, 255)   # RGB->BGR
    return out


# ----------------------------------------------------------------------------- camera
CAM = np.array([0.0, 1.55, 9.5])
TARGET = np.array([0.0, 1.05, 0.0])
FOCAL = 3100.0
PY0 = 450.0
FWD = unit(TARGET - CAM)
RIGHT = unit(np.cross(FWD, [0, 1, 0]))
UPV = np.cross(RIGHT, FWD)


def project(P, cx):
    d = P - CAM
    xc, yc, zc = d @ RIGHT, d @ UPV, d @ FWD
    return np.stack([cx + FOCAL * xc / zc, PY0 - FOCAL * yc / zc], -1)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def world(P, R):
    return (P - np.array([0, 0, ZPIV])) @ R.T


# ----------------------------------------------------------------------------- one render pass
def render_pass(alpha, cx_roi, t, mirror):
    """Returns premultiplied BGRA (RH, RW, 4) uint8 of the watch (or its mirror image)."""
    R = rot_y(alpha)
    M = MESH
    Pw = world(M.PV, R)
    Cw = world(M.CL, R)
    Nw = M.NRM @ R.T
    cam_s = CAM * np.array([1, -1, 1]) if mirror else CAM
    vis = np.sum(Nw * (cam_s - Cw), 1) > 0
    # painter ordering by layer then distance
    cam_loc = (cam_s @ R) + np.array([0, 0, ZPIV])      # camera in watch-local coordinates
    case_r = 1.0 if cam_loc[2] > -0.1 else 0.0
    band_r = 1.0 - case_r
    rank = np.where(M.LAY == 0, band_r, np.where(M.LAY == 1, case_r, case_r + (0.5 if cam_loc[0] > 0.45 else -0.5)))
    dist = np.linalg.norm(Cw - cam_s, axis=1)
    key = rank * 1000.0 - dist
    order = np.argsort(key)
    order = order[vis[order]]
    col = shade(Cw, Nw, cam_s, M.MAT, M.TONE)
    Pm = Pw * np.array([1, -1, 1]) if mirror else Pw
    uv = project(Pm, cx_roi)
    P2 = np.round(uv * SS * 16).astype(np.int32)
    canvas = np.zeros((RH * SS, RW * SS, 4), np.uint8)
    bgr = (col[:, ::-1] * 255).astype(np.uint8)
    for i in order:
        k = M.PK[i]
        c = bgr[i]
        cv2.fillConvexPoly(canvas, P2[i, :k], (int(c[0]), int(c[1]), int(c[2]), 255), cv2.LINE_8, 4)
    # glass (drawn last, it is the front-most part of the case)
    gn = np.array([0, 0, 1.0]) @ R.T
    gc = world(GLASS_LOCAL.mean(0, keepdims=True), R)[0]
    if np.dot(gn, cam_s - gc) > 0:
        gw = world(GLASS_LOCAL, R)
        gp = project(gw * np.array([1, -1, 1]) if mirror else gw, cx_roi) * SS
        tex = make_texture(t)
        # glossy reflection on the glass
        gy, gx = np.mgrid[0:8, 0:6]
        fu, fv = gx / 5.0, gy / 7.0
        pos = (gw[0][None, None] * (1 - fu[..., None]) * (1 - fv[..., None]) + gw[1][None, None] * fu[..., None] * (1 - fv[..., None])
               + gw[3][None, None] * (1 - fu[..., None]) * fv[..., None] + gw[2][None, None] * fu[..., None] * fv[..., None])
        pos = pos.reshape(-1, 3)
        v = unit(cam_s - pos)
        nvg = v @ gn
        rg = 2 * nvg[:, None] * gn - v
        b, lo, fl = env(rg)
        gl = (0.07 + 0.9 * (1 - nvg) ** 3) * (0.35 * b + 1.1 * lo) * 1.7
        gl = gl.reshape(8, 6).astype(np.float32)
        gl = cv2.resize(gl, (GW, GH), interpolation=cv2.INTER_LINEAR)
        xs = np.linspace(-1, 1, GW)[None, :]; ys = np.linspace(-1, 1, GH)[:, None]
        sheen = 0.55 + 0.9 * (0.5 + 0.5 * np.cos(2.4 * (xs * 0.8 + ys * 0.6) + 1.3 * alpha))
        tex = tex + (gl * sheen)[..., None] * 255.0
        tex = np.clip(tex, 0, 255)
        m = TEXMASK.astype(np.float32) / 255.0
        pre = np.dstack([tex * m[..., None], m * 255.0]).astype(np.uint8)
        x0 = int(max(0, math.floor(gp[:, 0].min()) - 2)); x1 = int(min(RW * SS, math.ceil(gp[:, 0].max()) + 2))
        y0 = int(max(0, math.floor(gp[:, 1].min()) - 2)); y1 = int(min(RH * SS, math.ceil(gp[:, 1].max()) + 2))
        if x1 > x0 and y1 > y0:
            src = np.float32([[0, 0], [GW, 0], [GW, GH], [0, GH]])
            Hm = cv2.getPerspectiveTransform(src, np.float32(gp - [x0, y0]))
            wp = cv2.warpPerspective(pre, Hm, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
            roi = canvas[y0:y1, x0:x1].astype(np.uint16)
            wa = wp[..., 3:4].astype(np.uint16)
            canvas[y0:y1, x0:x1] = np.minimum(wp.astype(np.uint16) + roi * (255 - wa) // 255, 255).astype(np.uint8)
    return cv2.resize(canvas, (RW, RH), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------- scene
def make_bg():
    rows = np.arange(H, dtype=np.float32)
    lum = 255 - 15 * smooth(520, 1010, rows)
    bg = np.repeat(lum[:, None], W, 1)
    xs = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    bg = bg - 4.0 * (xs ** 2)                       # very soft vignette
    rng = np.random.default_rng(7)
    bg = bg + rng.normal(0, 0.7, bg.shape).astype(np.float32)   # dither against banding
    return np.repeat(bg[..., None], 3, 2).astype(np.float32)


BG = make_bg()


def make_text():
    f1, f2 = FT('Inter-SemiBold.otf', 92), FT('Inter-Light.otf', 92)
    a, b = 'Pulse.', ' Always on.'
    wa, wb = f1.getlength(a), f2.getlength(b)
    tw, th = int(wa + wb) + 40, 170
    im = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((10, th / 2), a, font=f1, fill=(22, 22, 25, 255), anchor='lm')
    d.text((10 + wa, th / 2), b, font=f2, fill=(105, 105, 112, 255), anchor='lm')
    arr = np.array(im).astype(np.float32)
    pre = np.dstack([arr[..., 2::-1] * arr[..., 3:4] / 255.0, arr[..., 3]])   # BGR premult + A
    return pre


TEXT = make_text()
TEXT_X, TEXT_Y = 1000, 362      # final top-left of the text layer


def state(t):
    u = t / DUR
    p = 0.5 * u + 0.5 * float(smooth(0, 1, u))
    alpha = math.radians(314 - 300 * p)
    sh = float(smooth(0, 1, (t - 6.5) / 2.1))
    cx = 960 - 400 * sh
    return alpha, cx


def frame(t):
    alpha, cx = state(t)
    x0 = int(round(cx - RW / 2))
    cx_roi = cx - x0
    out = BG.copy()
    # contact shadow on the floor (soft ellipses on y=0, spinning with the watch)
    R = rot_y(alpha)
    def floor_poly(cx_l, cz_l, ax, az, n=48):
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        loc = np.stack([cx_l + ax * np.cos(th), np.zeros(n), cz_l + az * np.sin(th)], 1)
        w = world(loc, R)
        return project(w, cx_roi)
    sh = np.zeros((RH, RW), np.float32)
    cv2.fillPoly(sh, [np.round(floor_poly(0, -0.30, 0.62, 0.95) * 16).astype(np.int32)], 0.55, cv2.LINE_AA, 4)
    sh = cv2.GaussianBlur(sh, (0, 0), 30)
    tight = np.zeros((RH, RW), np.float32)
    cv2.fillPoly(tight, [np.round(floor_poly(0, -0.57, 0.46, 0.13) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
    tight = cv2.GaussianBlur(tight, (0, 0), 9)
    dark = np.clip(0.30 * sh + 0.42 * tight, 0, 0.7)
    xa, xb = max(0, x0), min(W, x0 + RW)
    reg = out[:, xa:xb]
    reg *= (1 - dark[:, xa - x0:xb - x0])[..., None]
    # reflection
    refl = render_pass(alpha, cx_roi, t, True).astype(np.float32)
    row0 = float(project(world(np.array([[0, 0, ZPIV]]), R), cx_roi)[0, 1])
    rows = np.arange(RH, dtype=np.float32)
    k = 0.34 * np.clip(1 - (rows - row0) / 520.0, 0, 1) ** 1.5
    k[rows < row0] = 0
    wb_ = np.clip((rows - row0) / 260.0, 0, 1)[:, None, None] * 0.85
    refl = refl * (1 - wb_) + cv2.GaussianBlur(refl, (0, 0), 6) * wb_
    a = refl[..., 3:4] / 255.0
    reg = out[:, xa:xb]
    kk = k[:, None, None]
    sl = slice(xa - x0, xb - x0)
    reg[:] = reg * (1 - a[:, sl] * kk) + refl[:, sl, :3] * kk
    # watch
    wp = render_pass(alpha, cx_roi, t, False).astype(np.float32)
    a = wp[..., 3:4] / 255.0
    reg[:] = reg * (1 - a[:, sl]) + wp[:, sl, :3]
    # text
    te = float(smooth(0, 1, (t - 7.1) / 1.2))
    if te > 0:
        e = 1 - (1 - te) ** 3
        off = int(round((1 - e) * 170))
        th_, tw_ = TEXT.shape[:2]
        tx, ty = TEXT_X + off, TEXT_Y
        x1 = min(W, tx + tw_)
        tr = TEXT[:, :x1 - tx]
        ta = tr[..., 3:4] / 255.0 * e
        seg = out[ty:ty + th_, tx:x1]
        out[ty:ty + th_, tx:x1] = seg * (1 - ta) + tr[..., :3] * e
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- output
def encode_scene(f0, f1, path):
    tmp = path + '.tmp.mp4'
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                          '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast',
                          '-crf', '17', '-pix_fmt', 'yuv420p', '-g', '30', tmp], stdin=subprocess.PIPE)
    for i in range(f0, f1):
        p.stdin.write(frame(i / FPS).tobytes())
    p.stdin.close()
    if p.wait() == 0:
        os.replace(tmp, path)
    else:
        raise SystemExit('ffmpeg failed')


def _job(args):
    encode_scene(*args)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'still':
        t = float(sys.argv[2])
        cv2.imwrite(sys.argv[3], frame(t))
    elif cmd == 'video':
        import multiprocessing as mp
        out = sys.argv[2]
        base = os.path.splitext(out)[0]
        half = NF // 2
        jobs = [(0, half, base + '.s0.mp4'), (half, NF, base + '.s1.mp4')]
        with mp.Pool(2) as pool:
            pool.map(_job, jobs)
        lst = base + '.txt'
        with open(lst, 'w') as f:
            for j in jobs:
                f.write(f"file '{os.path.abspath(j[2])}'\n")
        subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst,
                               '-c', 'copy', out])
        for j in jobs:
            os.remove(j[2])
        os.remove(lst)
