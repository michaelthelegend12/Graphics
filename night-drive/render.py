#!/usr/bin/env python3
"""Night Drive: 10 second seamless-loop synthwave title card (1920x1080, 30 fps).

Everything is a function of t/10 with integer cycle counts, so frame 300 would
equal frame 0 and the 10 s synth pad is built circularly, so the loop has no seam.

Needs: numpy, OpenCV, Pillow, ffmpeg.
Usage:  python3 render.py            -> night-drive.mp4
        python3 render.py --preview  -> a few PNG stills in ./preview
"""
import os
import subprocess
import sys
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, DUR = 1920, 1080, 30, 10
NF = FPS * DUR
HZN = 650                      # horizon row
CX = W / 2
SUN_CY, SUN_R = 520, 300
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = "/usr/share/fonts/truetype/google-fonts/"

rng = np.random.default_rng(7)


def stops(v, pts):
    """Piecewise-linear colour ramp. pts = [(pos, (r,g,b)), ...]"""
    xs = [p[0] for p in pts]
    return np.stack([np.interp(v, xs, [p[1][c] for p in pts]) for c in range(3)], -1)


# ------------------------------------------------------------------ background
yy = np.arange(H, dtype=np.float32)
sky = stops(yy, [(0, (5, 0, 18)), (260, (24, 3, 60)), (470, (78, 10, 104)),
                 (600, (190, 40, 130)), (HZN, (255, 90, 150))]).astype(np.float32)
base = np.repeat(sky[:, None, :], W, 1)

floor_col = stops(yy, [(HZN, (80, 12, 105)), (HZN + 70, (26, 3, 56)), (H, (8, 0, 22))]).astype(np.float32)
floor_base = np.repeat(floor_col[HZN:, None, :], W, 1)
# soft horizon glow on the floor
fy = np.arange(H - HZN, dtype=np.float32)
floor_base += (np.exp(-(fy / 40.0) ** 2)[:, None, None] * np.array([150, 40, 130], np.float32))

# sun (static colour, animated scanline cuts)
S0, S1 = SUN_CY - SUN_R, SUN_CY + SUN_R
X0, X1 = int(CX - SUN_R), int(CX + SUN_R)
sy, sx = np.mgrid[S0:S1, X0:X1].astype(np.float32)
dist = np.hypot(sx - CX, sy - SUN_CY)
sun_mask = np.clip(SUN_R - dist, 0, 1)[..., None]
sv = (sy[:, 0] - S0) / (2 * SUN_R)
sun_col = stops(sv, [(0, (255, 244, 120)), (0.32, (255, 178, 64)), (0.58, (255, 62, 132)),
                     (1.0, (170, 22, 176))]).astype(np.float32)
sun_col = np.repeat(sun_col[:, None, :], X1 - X0, 1)
thick = np.clip((sv - 0.42) / 0.58, 0, 1) * 0.72          # gaps widen toward the bottom
BAND_P = 44.0                                               # px per scanline cycle
BAND_CYCLES = 3                                             # whole cycles per loop -> seamless

# sun glow
G0, G1 = 0, HZN + 40
gl = np.zeros((H, W), np.float32)
gl[S0:S1, X0:X1] = sun_mask[..., 0]
glow_big = cv2.GaussianBlur(gl, (0, 0), 90)[G0:G1]
glow_small = cv2.GaussianBlur(gl, (0, 0), 28)[G0:G1]
glow_rgb = (glow_big[..., None] * np.array([255, 50, 150], np.float32) * 0.9 +
            glow_small[..., None] * np.array([255, 120, 90], np.float32) * 0.45)

# stars
stars = []
for _ in range(170):
    x = int(rng.integers(4, W - 4))
    y = int(rng.integers(4, HZN - 120))
    stars.append((x, y, float(rng.uniform(0.35, 1.0)), float(rng.random()), int(rng.integers(1, 4)),
                  rng.choice([0, 1, 2], p=[0.7, 0.2, 0.1])))
STAR_COL = [np.array(c, np.float32) for c in [(255, 255, 255), (170, 220, 255), (255, 190, 235)]]
K3 = np.array([[.25, .5, .25], [.5, 1, .5], [.25, .5, .25]], np.float32)[..., None]

# ------------------------------------------------------------------ chrome title
def build_text():
    f_big = ImageFont.truetype(FONT_DIR + "Poppins-Bold.ttf", 205)
    text, track = "NIGHT DRIVE", 16
    widths = [f_big.getlength(c) for c in text]
    tw = int(sum(widths) + track * (len(text) - 1))
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    x = (W - tw) / 2
    ty = 205
    for c, w in zip(text, widths):
        d.text((x, ty), c, font=f_big, fill=255, stroke_width=4)
        x += w + track
    cy = ty + 130
    sh = 0.2
    m = m.transform((W, H), Image.AFFINE, (1, sh, -sh * cy, 0, 1, 0), resample=Image.BICUBIC)
    M = np.asarray(m, np.float32) / 255.0

    ys = np.where(M.max(1) > 0.5)[0]
    xs = np.where(M.max(0) > 0.5)[0]
    y0, y1, x0, x1 = ys[0], ys[-1], xs[0], xs[-1]
    v = (np.arange(H, dtype=np.float32) - y0) / (y1 - y0)
    chrome = stops(v, [(0, (255, 255, 255)), (0.28, (190, 238, 255)), (0.46, (70, 160, 255)),
                       (0.495, (18, 30, 120)), (0.505, (12, 6, 56)), (0.54, (96, 30, 160)),
                       (0.66, (255, 70, 200)), (0.86, (255, 176, 110)), (1.0, (255, 240, 190))]).astype(np.float32)
    col = np.repeat(chrome[:, None, :], W, 1)

    # bevel lighting from top-left
    B = cv2.GaussianBlur(M, (0, 0), 3.2)
    gx = cv2.Sobel(B, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(B, cv2.CV_32F, 0, 1, ksize=3)
    shade = np.clip(-(gx + gy) * 2.2, -1, 1)[..., None]
    col = col * (1 + 0.38 * np.minimum(shade, 0)) + 105 * np.maximum(shade, 0)

    # dark rim + neon glow
    D = cv2.dilate(M, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    rim_col = np.array([14, 0, 44], np.float32)
    glow = (cv2.GaussianBlur(D, (0, 0), 22)[..., None] * np.array([255, 40, 190], np.float32) * 0.85 +
            cv2.GaussianBlur(D, (0, 0), 70)[..., None] * np.array([120, 0, 220], np.float32) * 0.7)

    # subtitle
    f_small = ImageFont.truetype(FONT_DIR + "Poppins-Medium.ttf", 44)
    sub, strack = "P L A Y L I S T", 6
    sw_ = int(f_small.getlength(sub) + strack * (len(sub) - 1))
    sm = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(sm)
    xx = (W - sw_) / 2
    for c in sub:
        sd.text((xx, HZN + 36), c, font=f_small, fill=255)
        xx += f_small.getlength(c) + strack
    SM = np.asarray(sm, np.float32) / 255.0
    sub_col = np.array([140, 245, 255], np.float32)
    glow += cv2.GaussianBlur(SM, (0, 0), 9)[..., None] * np.array([0, 200, 255], np.float32) * 1.3
    return dict(M=M, col=col, D=D, rim=rim_col, glow=glow, SM=SM, sub_col=sub_col,
                box=(y0, y1 + 1, x0, x1 + 1))


T = build_text()
ty0, ty1, tx0, tx1 = T["box"]
PAD_Y, PAD_X = 260, 260
by0, by1 = max(ty0 - PAD_Y, 0), min(ty1 + PAD_Y, H)
bx0, bx1 = max(tx0 - PAD_X, 0), min(tx1 + PAD_X, W)

# static "post" layer: vignette + faint CRT rows
ry, rx = np.mgrid[0:H, 0:W].astype(np.float32)
r = np.hypot((rx - CX) / (W / 2), (ry - H / 2) / (H / 2)) / 1.25
post = (1 - 0.5 * np.clip(r, 0, 1.2) ** 2.3)
post = post * (1 - 0.05 * (ry.astype(np.int32) % 3 == 0))
post = post[..., None].astype(np.float32)
del ry, rx, r

noise = rng.normal(0, 1.4, (H + 64, W + 64, 1)).astype(np.float32)

# floor grid geometry
FH = H - HZN
frow = (np.arange(FH, dtype=np.float32) + 0.5)[:, None]        # distance below horizon
fcol = (np.arange(W, dtype=np.float32) - CX)[None, :]
C_, S_, K_ = 460.0, 0.15, 12                                    # K whole cells scroll per loop
G_ = 0.30
spacing_h = S_ * frow ** 2 / C_
spacing_v = frow * G_
fade = np.clip((spacing_h - 2) / 7, 0, 1) * np.clip((spacing_v - 3) / 10, 0, 1)
half_w = 1.1 + 0.0032 * frow
vv = fcol / (frow * G_)
fv = np.abs(vv - np.round(vv)) * spacing_v
LINE_CORE = np.array([255, 190, 255], np.float32)
LINE_GLOW = np.array([255, 30, 200], np.float32)


def smoothstep(a, b, x):
    k = np.clip((x - a) / (b - a), 0, 1)
    return k * k * (3 - 2 * k)


def render_frame(i, k=None):
    t = (i % NF) / FPS
    ph = t / DUR                                   # 0..1 loop phase
    f = base.copy()

    # twinkling stars
    for (x, y, b0, p0, kk, ci) in stars:
        tw = 0.5 + 0.5 * np.sin(2 * np.pi * (kk * ph + p0))
        f[y - 1:y + 2, x - 1:x + 2] += K3 * STAR_COL[ci] * (b0 * (0.25 + 0.75 * tw))

    # sun glow (breathes once per loop) then the sun with moving scanline cuts
    f[G0:G1] += glow_rgb * (1.0 + 0.12 * np.sin(2 * np.pi * ph))
    p = BAND_P
    cyc = ((sy[:, 0] - BAND_CYCLES * p * ph) / p) % 1.0
    gap = np.clip(cyc * p, 0, 1) - np.clip((cyc - thick) * p, 0, 1)
    alpha = sun_mask * (1 - gap)[:, None, None]
    reg = f[S0:S1, X0:X1]
    f[S0:S1, X0:X1] = reg * (1 - alpha) + sun_col * alpha

    # grid floor
    u = (C_ / frow) / S_ + K_ * ph
    hd = np.abs(u - np.round(u)) * S_ * frow ** 2 / C_
    d = np.minimum(hd, fv)
    core = np.clip(1 - d / half_w, 0, 1) * fade
    gl_ = np.exp(-(d / (half_w * 3.2)) ** 2) * fade * 0.38
    fl = floor_base + core[..., None] * LINE_CORE * 0.95 + (gl_ + core * 0.5)[..., None] * LINE_GLOW
    f[HZN:] = fl

    # chrome title
    cy0, cy1, cx0, cx1 = by0, by1, bx0, bx1
    sub = f[cy0:cy1, cx0:cx1]
    M = T["M"][cy0:cy1, cx0:cx1, None]
    D = T["D"][cy0:cy1, cx0:cx1, None]
    sub = sub + T["glow"][cy0:cy1, cx0:cx1] * (1.0 + 0.1 * np.sin(2 * np.pi * ph + 1.0))
    sub = sub * (1 - D) + T["rim"] * D
    col = T["col"][cy0:cy1, cx0:cx1].copy()
    # shine: two diagonal sweeps per loop, fully off-screen at the loop point
    hh, ww = col.shape[:2]
    gy_, gx_ = np.mgrid[0:hh, 0:ww].astype(np.float32)
    s = gx_ + 0.45 * gy_
    smax = ww + 0.45 * hh
    shine = np.zeros_like(s)
    for t0 in (1.2, 6.2):
        q = (t - t0) / 2.2
        if 0 <= q <= 1:
            pos = -200 + (smax + 400) * smoothstep(0, 1, q)
            shine += 0.95 * np.exp(-((s - pos) / 34.0) ** 2) + 0.35 * np.exp(-((s - pos) / 120.0) ** 2)
    col = col + shine[..., None] * np.array([255, 255, 255], np.float32) * 0.9
    sub = sub * (1 - M) + col * M
    SMc = T["SM"][cy0:cy1, cx0:cx1, None]
    f[cy0:cy1, cx0:cx1] = sub
    # subtitle is below the box: composite it separately
    SM = T["SM"][:, :, None]
    row0 = HZN + 10
    f[row0:row0 + 120] = f[row0:row0 + 120] * (1 - SM[row0:row0 + 120]) + T["sub_col"] * 255 / 255 * SM[row0:row0 + 120] * 1.0
    del SMc

    f *= post
    ox, oy = (i * 37) % 64, (i * 53) % 64
    f += noise[oy:oy + H, ox:ox + W]
    return np.clip(f, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ audio
SR = 44100


def make_pad(path):
    """10 s periodic synth pad: Am7 > Fmaj7 > Cmaj7 > G, detuned saws, circular reverb."""
    n = SR * DUR
    t = np.arange(n, dtype=np.float64) / SR
    chords = [([57, 60, 64, 67], 45), ([57, 60, 64, 65], 41), ([55, 59, 60, 64], 36), ([55, 59, 62, 67], 43)]
    L = DUR / len(chords)

    def mtof(m):
        return 440.0 * 2 ** ((m - 69) / 12)

    def snap(f):                         # whole multiples of 0.1 Hz -> exact period of 10 s
        return round(f * DUR) / DUR

    out = np.zeros((2, n))
    for ci, (notes, bass) in enumerate(chords):
        d = np.abs(((t - (ci + 0.5) * L) + DUR / 2) % DUR - DUR / 2)       # circular distance
        win = np.where(d < L, np.cos(np.pi * d / (2 * L)) ** 2, 0.0)
        # a bit of slow bloom inside each chord
        voice = np.zeros((2, n))
        for m in notes:
            for ch, dets in ((0, (-7, 0, 5)), (1, (-4, 0, 8))):
                for cents in dets:
                    f = snap(mtof(m) * 2 ** (cents / 1200))
                    for h in range(1, 11):
                        voice[ch] += np.sin(2 * np.pi * f * h * t + 0.7 * h + ch) / h * np.exp(-h / 3.2)
        sub = np.sin(2 * np.pi * snap(mtof(bass)) * t) * 1.6
        out += (voice * 0.045 + sub * 0.22) * win
    out *= (1 + 0.08 * np.sin(2 * np.pi * 0.2 * t))
    # circular reverb
    irn = int(SR * 2.2)
    ir = rng.normal(0, 1, (2, irn)) * np.exp(-np.arange(irn) / (SR * 0.6))
    ir /= np.abs(ir).sum(1, keepdims=True)
    wet = np.fft.irfft(np.fft.rfft(out, axis=1) * np.fft.rfft(ir, n, axis=1), n, axis=1)
    mix = out * 0.7 + wet * 2.2
    mix = mix / np.abs(mix).max() * 0.7
    pcm = (mix.T * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    if "--preview" in sys.argv:
        os.makedirs(os.path.join(HERE, "preview"), exist_ok=True)
        for tt in (0.0, 2.2, 3.6, 7.3):
            fr = render_frame(int(tt * FPS))
            Image.fromarray(fr).save(os.path.join(HERE, "preview", f"t{tt:04.1f}.png"))
        a, b = render_frame(0).astype(int), render_frame(NF).astype(int)
        print("seam: mean|f(0)-f(300)| excluding noise =", np.abs(a - b).mean())
        sys.exit()

    wav = "/tmp/night-drive-pad.wav"
    make_pad(wav)
    out = os.path.join(HERE, "night-drive.mp4")
    tmp = os.path.join(HERE, "tmp-night-drive.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-t", str(DUR), "-movflags", "+faststart", tmp]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(NF):
        pr.stdin.write(render_frame(i).tobytes())
    pr.stdin.close()
    if pr.wait() == 0:
        os.replace(tmp, out)
        print("wrote", out)
    else:
        sys.exit("ffmpeg failed")
