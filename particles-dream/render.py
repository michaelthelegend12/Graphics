"""DREAM particle animation. Usage:
  particles.py png <t1> <t2> ...          -> writes frame PNGs to ./frames_test
  particles.py video <f0> <f1> <out.mp4>  -> renders frames [f0,f1) to a single mp4
"""
import sys, os, subprocess, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
CX, CY = W / 2, H / 2
N_MAIN, N_DUST = 20000, 3000
N = N_MAIN + N_DUST
T_FORM_END, T_EXPLODE, T_END = 5.0, 8.0, 12.0

# ---------- target points from the word ----------
def make_targets(rng):
    font = ImageFont.truetype("/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf", 330)
    word, track = "DREAM", 34
    widths = [font.getlength(c) for c in word]
    total = sum(widths) + track * (len(word) - 1)
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    x = (W - total) / 2
    bb = font.getbbox("D")
    y = CY - (bb[1] + bb[3]) / 2
    for c, w in zip(word, widths):
        d.text((x, y), c, font=font, fill=255)
        x += w + track
    m = np.array(img) > 127
    er = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    edge = m & ~er
    fill_ys, fill_xs = np.nonzero(m)
    edge_ys, edge_xs = np.nonzero(edge)
    n_edge = int(N_MAIN * 0.35)
    ie = rng.integers(0, len(edge_xs), n_edge)
    i_f = rng.integers(0, len(fill_xs), N_MAIN - n_edge)
    px = np.concatenate([edge_xs[ie], fill_xs[i_f]]).astype(np.float64)
    py = np.concatenate([edge_ys[ie], fill_ys[i_f]]).astype(np.float64)
    px += rng.uniform(-0.5, 0.5, N_MAIN); py += rng.uniform(-0.5, 0.5, N_MAIN)
    # dust: loose halo around the word
    idd = rng.integers(0, len(fill_xs), N_DUST)
    dx = fill_xs[idd] + rng.normal(0, 38, N_DUST)
    dy = fill_ys[idd] + rng.normal(0, 38, N_DUST)
    return np.concatenate([px, dx]), np.concatenate([py, dy])

rng = np.random.default_rng(7)
TX, TY = make_targets(rng)
RX, RY = TX - CX, TY - CY
RT = np.hypot(RX, RY)
AT = np.arctan2(RY, RX)

# swirl start state
R0 = 40 + 880 * rng.random(N) ** 1.35
ARM = rng.integers(0, 3, N) * (2 * math.pi / 3)
A0 = 2.4 * np.log(R0 / 30.0) + ARM + rng.normal(0, 0.22, N)
SPIN = (A0 - AT) + 2 * math.pi * rng.integers(1, 3, N)
OMEGA = 2.6 * (200.0 / (R0 + 200.0)) + rng.uniform(-0.1, 0.1, N)
DELAY = rng.uniform(0.0, 1.5, N)
DUR = rng.uniform(3.0, 3.5, N)
# swirl "thickness" wobble
WOB_A = rng.uniform(10, 60, N); WOB_F = rng.uniform(0.4, 1.3, N); WOB_P = rng.uniform(0, 6.28, N)
# brightness
BASE = rng.uniform(0.45, 1.5, N) ** 1.4
BASE[N_MAIN:] *= 0.45
TW_F = rng.uniform(0.4, 1.4, N); TW_P = rng.uniform(0, 6.28, N)
# shimmer
SH_P = rng.uniform(0, 6.28, (N, 2)); SH_F = rng.uniform(0.6, 1.6, (N, 2))
# explosion
EDIR = AT + rng.normal(0, 0.45, N)
ESPD = 350 + 2300 * rng.random(N) ** 1.6
ETAN = rng.normal(0, 260, N)
EK = rng.uniform(1.2, 2.0, N)
EFADE = rng.uniform(2.6, 4.0, N)
# fade-in at start
FADE_IN = rng.uniform(0.0, 1.2, N)

NOISE = [rng.uniform(-0.5, 0.5, (H, W, 3)).astype(np.float32) for _ in range(3)]
GOLD = np.array([0.95, 0.52, 0.10], np.float32)  # R,G,B exposure weights (warm gold)

def sstep(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (x * 6 - 15) + 10)

def state(t):
    """returns positions (x,y) and intensity weights for time t."""
    e = sstep((t - DELAY) / DUR)
    ang = AT + (1 - e) * (SPIN + OMEGA * t)
    r = RT + (R0 - RT) * (1 - e)
    wob = (1 - e) * WOB_A * np.sin(WOB_F * t * 2 + WOB_P)
    r = r + wob
    x = CX + r * np.cos(ang)
    y = CY + r * np.sin(ang)
    alpha = np.clip((t - FADE_IN) / 0.8, 0, 1)
    tw = 0.8 + 0.2 * np.sin(2 * math.pi * TW_F * t + TW_P)
    w = BASE * alpha * tw
    # hold-phase shimmer and pre-explosion tension
    build = np.clip((t - 7.0) / 1.0, 0, 1) ** 2  # 0..1 over 7.0..8.0
    amp = 0.9 + 6.0 * build
    hold = np.clip((t - 4.2) / 1.0, 0, 1) * (t < T_EXPLODE)
    x = x + hold * amp * np.sin(2 * math.pi * SH_F[:, 0] * t * (1 + 2 * build) + SH_P[:, 0])
    y = y + hold * amp * np.sin(2 * math.pi * SH_F[:, 1] * t * (1 + 2 * build) + SH_P[:, 1])
    w = w * (1 + 0.5 * build)
    extra_flash = 0.0
    if t >= T_EXPLODE:
        tau = t - T_EXPLODE
        f = (1 - np.exp(-EK * tau)) / EK
        x = x + np.cos(EDIR) * ESPD * f - np.sin(EDIR) * ETAN * f
        y = y + np.sin(EDIR) * ESPD * f + np.cos(EDIR) * ETAN * f
        fade = np.clip(1 - (tau / EFADE) ** 1.2, 0, 1)
        w = w * fade * (1 + 1.2 * np.exp(-tau / 0.15))
    return x, y, w

def splat(x, y, w):
    buf = np.zeros(H * W, np.float32)
    ok = (x > 1) & (x < W - 2) & (y > 1) & (y < H - 2) & (w > 1e-4)
    x, y, w = x[ok], y[ok], w[ok]
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    fx = x - x0; fy = y - y0
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                       (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        idx = (y0 + dy) * W + (x0 + dx)
        buf += np.bincount(idx, weights=(w * wt), minlength=H * W).astype(np.float32)
    return buf.reshape(H, W)

def render(t, fi=0):
    x, y, w = state(t)
    core = splat(x, y, w)
    core = cv2.GaussianBlur(core, (0, 0), 0.9)
    # glow at quarter res
    lo = cv2.resize(core, (W // 4, H // 4), interpolation=cv2.INTER_AREA) * 16.0
    g1 = cv2.GaussianBlur(lo, (0, 0), 1.6)
    g2 = cv2.GaussianBlur(lo, (0, 0), 5.5)
    g3 = cv2.GaussianBlur(lo, (0, 0), 16.0)
    G = 0.55 * g1 + 0.5 * g2 + 0.45 * g3
    # impact flash + shockwave ring
    if t >= T_EXPLODE:
        tau = t - T_EXPLODE
        yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32)
        rr = np.hypot(xx - CX / 4, yy - CY / 4)
        G += 1.4 * np.exp(-tau / 0.12) * np.exp(-(rr / 90.0) ** 2)
        rad = (2600 * (1 - np.exp(-2.4 * tau)) / 2.4) / 4
        G += 0.5 * np.exp(-tau / 0.45) * np.exp(-((rr - rad) / 7.0) ** 2)
    Gf = cv2.resize(G, (W, H), interpolation=cv2.INTER_LINEAR)
    I = 2.6 * core + 1.0 * Gf
    out = 1.0 - np.exp(-I[..., None] * GOLD)
    out = out ** 0.95
    img = out * 255.0 + NOISE[fi % 3]
    return np.clip(img, 0, 255).astype(np.uint8)

def main():
    mode = sys.argv[1]
    if mode == "png":
        os.makedirs("frames_test", exist_ok=True)
        for ts in sys.argv[2:]:
            t = float(ts)
            Image.fromarray(render(t)).save(f"frames_test/t{t:05.2f}.png")
    else:
        f0, f1, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        tmp = out + ".part.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "17",
               "-preset", "veryfast", "-pix_fmt", "yuv420p", "-threads", "1", "-movflags", "+faststart", tmp]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in range(f0, f1):
            p.stdin.write(render(f / FPS, f).tobytes())
        p.stdin.close()
        if p.wait() == 0:
            os.rename(tmp, out)
        else:
            sys.exit("ffmpeg failed")

if __name__ == "__main__":
    main()
