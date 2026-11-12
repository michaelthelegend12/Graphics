"""Smartphone users worldwide, 2010-2025: self-drawing line graph with a following camera.

Usage:
  python3 render.py preview OUT_DIR T1 T2 ...   -> PNG frames at those times (seconds)
  python3 render.py render OUT.mp4              -> full 25 s, 1920x1080, 30 fps
"""
import os
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from scipy.interpolate import PchipInterpolator

W, H, FPS, DUR = 1920, 1080, 30, 25.0
NF = int(round(FPS * DUR))

# ---- palette: near-black + one neon accent ---------------------------------
BG = "#06070b"
PANEL = "#0d1118"
ACC = "#00e5ff"
WHITE = "#f2f5fa"
GREY = "#8b94a7"
DIM = "#4a5266"
FONT = "Inter"

# ---- data (billions of users). 2014-2025 Statista; 2010-2013 rounded estimates
YEARS = np.arange(2010, 2026)
USERS = np.array([0.45, 0.60, 0.75, 0.88, 1.01, 1.22, 1.43, 1.66, 1.94, 2.27,
                  2.67, 3.10, 3.62, 4.25, 4.88, 5.28])
curve = PchipInterpolator(YEARS, USERS)
XD = np.linspace(2010, 2025, 3001)
YD = curve(XD)


def cross_year(level):
    i = int(np.argmax(YD >= level))
    return float(XD[i])


# ---- timeline --------------------------------------------------------------
T_DRAW0, T_DRAW1 = 1.2, 19.2     # line draws
T_PULL1 = 22.2                   # camera back to full view by here


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def draw_progress(t):
    u = np.clip((t - T_DRAW0) / (T_DRAW1 - T_DRAW0), 0, 1)
    return 0.8 * u + 0.2 * smooth(u)


def x_tip_at(t):
    return 2010 + 15 * draw_progress(t)


def time_at_x(x):
    """invert x_tip_at on a fine grid"""
    ts = np.linspace(0, DUR, 20001)
    xs = x_tip_at(ts)
    return float(ts[int(np.argmax(xs >= x - 1e-9))])


# milestones: (level, x, big text, caption, t_show)
MILES = []
MILES.append(dict(x=2010.0, y=0.45, year="2010", big="450 million", cap="smartphone users to start",
                  short="0.45B", t=0.6))
for lvl, big, cap in [(1, "1 billion", "first billion users"),
                      (2, "2 billion", "about 1 in 4 people"),
                      (3, "3 billion", "growth through the pandemic"),
                      (4, "4 billion", "roughly half of humanity"),
                      (5, "5 billion", "about 6 in 10 people")]:
    x = cross_year(lvl)
    MILES.append(dict(x=x, y=float(curve(x)), year=str(int(round(x))), big=big, cap=cap,
                      short=f"{lvl}B", t=time_at_x(x)))

# ---- camera path (precomputed, smoothed) -----------------------------------
FULL = dict(cx=(2009.2 + 2025.9) / 2, cy=(-0.55 + 6.0) / 2, w=2025.9 - 2009.2, h=6.0 + 0.55)


def gauss_smooth(a, sigma):
    r = int(sigma * 3)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    p = np.pad(a, r, mode="edge")
    return np.convolve(p, k, mode="valid")


def build_camera():
    ts = np.arange(NF) / FPS
    xt = x_tip_at(ts)
    yt = curve(np.clip(xt, 2010, 2025))
    u = np.clip((ts - T_DRAW0) / (T_DRAW1 - T_DRAW0), 0, 1)
    w = 3.4 + 4.2 * u ** 1.15
    h = 1.25 + 0.52 * yt
    cx = xt - 0.16 * w
    cy = yt - 0.18 * h
    cx, cy = gauss_smooth(cx, 10), gauss_smooth(cy, 10)
    lw, lh = gauss_smooth(np.log(w), 14), gauss_smooth(np.log(h), 14)
    b = smooth((ts - T_DRAW1) / (T_PULL1 - T_DRAW1))
    b = b[:, None]
    follow = np.stack([cx, cy, lw, lh], 1)
    full = np.array([[FULL["cx"], FULL["cy"], np.log(FULL["w"]), np.log(FULL["h"])]])
    cam = follow * (1 - b) + full * b
    return cam


CAM = build_camera()

# axes rectangle (pixels, y-down) for the data window
AX_L, AX_R, AX_T, AX_B = 115, 1824, 190, 940


def ease_out_back(x):
    x = np.clip(x, 0, 1)
    c1, c3 = 1.4, 2.4
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


class Painter:
    def __init__(self):
        plt.rcParams["font.family"] = FONT
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
        self.ax = self.fig.add_axes([AX_L / W, 1 - AX_B / H, (AX_R - AX_L) / W, (AX_B - AX_T) / H])
        self.ov = self.fig.add_axes([0, 0, 1, 1])
        self.ov.set_xlim(0, W)
        self.ov.set_ylim(H, 0)
        self.ov.axis("off")
        self.ov.patch.set_alpha(0)
        self.ov.set_zorder(5)

    def to_px(self, x, y):
        d = self.ax.transData.transform((x, y))
        return d[0], H - d[1]

    def frame(self, i):
        t = i / FPS
        ax, ov = self.ax, self.ov
        ax.cla()
        ov.cla()
        ov.set_xlim(0, W)
        ov.set_ylim(H, 0)
        ov.axis("off")
        ov.patch.set_alpha(0)

        cx, cy, lw, lh = CAM[i]
        w, h = np.exp(lw), np.exp(lh)
        x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_visible(False)

        # grid + ticks
        step = 0.5 if h < 3.2 else 1.0
        yt = np.arange(np.ceil(y0 / step) * step, y1, step)
        ax.set_yticks(yt)
        ax.set_yticklabels([("0" if v == 0 else f"{v:g}B") for v in yt])
        ax.set_xticks(YEARS)
        ax.set_xticklabels([str(y) for y in YEARS])
        ax.tick_params(axis="both", length=0, labelsize=17, colors=DIM, pad=12)
        ax.grid(axis="y", color="#ffffff", alpha=0.07, lw=1.2)
        ax.grid(axis="x", color="#ffffff", alpha=0.045, lw=1.2)
        ax.set_axisbelow(True)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.axhline(0, color="#ffffff", alpha=0.22, lw=1.6)

        # intro fade for everything
        fin = smooth(t / 0.9)

        # line up to the tip
        xt = float(x_tip_at(t))
        xt = min(xt, 2025.0)
        m = XD <= xt
        xs = np.append(XD[m], xt)
        ys = np.append(YD[m], float(curve(xt)))
        ax.fill_between(xs, 0, ys, color=ACC, alpha=0.09 * fin, lw=0)
        for lwid, a in [(26, 0.05), (15, 0.09), (8, 0.16)]:
            ax.plot(xs, ys, color=ACC, alpha=a * fin, lw=lwid, solid_capstyle="round", zorder=3)
        ax.plot(xs, ys, color=ACC, alpha=fin, lw=4.2, solid_capstyle="round", zorder=4)

        # tip glow
        tx, ty = self.to_px(xt, ys[-1])
        for r, a in [(34, 0.07), (22, 0.13), (13, 0.25)]:
            ov.add_patch(Circle((tx, ty), r, color=ACC, alpha=a * fin, lw=0))
        ov.add_patch(Circle((tx, ty), 7.5, color="#ffffff", alpha=fin, lw=0))

        # milestones
        for mm in MILES:
            age = t - mm["t"]
            if age < 0:
                continue
            px, py = self.to_px(mm["x"], mm["y"])
            if px < -50 or px > W + 50:
                continue
            clip_ok = (AX_L - 5 < px < AX_R + 5) and (AX_T < py < AX_B)
            # ring on the point
            ring_a = min(age / 0.25, 1)
            if clip_ok:
                ov.add_patch(Circle((px, py), 11, fc=BG, ec=ACC, lw=3, alpha=ring_a))
            if age < 0.55:  # expanding pulse
                p = age / 0.55
                ov.add_patch(Circle((px, py), 11 + 46 * p, fc="none", ec=ACC, lw=2.5, alpha=0.6 * (1 - p)))
            # card
            life = 2.5
            if age < life + 0.5:
                pop = ease_out_back(age / 0.5)
                a = min(age / 0.22, 1) * (1 - smooth((age - life) / 0.5))
                cw, ch = 372, 132
                cxr = px - 46
                left = cxr - cw
                top = py - 78 - ch + 20 * (1 - pop)
                top = max(top, 204)
                left = max(left, AX_L + 10)
                ov.plot([px - 8, left + cw - 20], [py - 8, top + ch], color=ACC, lw=2, alpha=0.7 * a)
                ov.add_patch(FancyBboxPatch((left, top), cw, ch, boxstyle="round,pad=0,rounding_size=18",
                                            fc=PANEL, ec=ACC, lw=1.8, alpha=0.96 * a))
                ov.text(left + 26, top + 30, mm["year"], color=ACC, fontsize=15, fontweight="semibold",
                        va="center", alpha=a)
                ov.text(left + 26, top + 70, mm["big"], color=WHITE, fontsize=31, fontweight="bold",
                        va="center", alpha=a)
                ov.text(left + 26, top + 106, mm["cap"], color=GREY, fontsize=14.5, va="center", alpha=a)
            elif clip_ok and mm["short"] != "0.45B":
                ov.text(px + 16, py + 30, mm["short"], color=WHITE, fontsize=17, fontweight="semibold",
                        va="center", alpha=0.7 * smooth((age - life - 0.2) / 0.5))

        # HUD, top left
        ov.text(96, 82, "SMARTPHONE USERS WORLDWIDE", color=GREY, fontsize=17, fontweight="semibold",
                va="center", alpha=fin)
        ov.text(94, 142, "2010 → 2025", color=WHITE, fontsize=44, fontweight="bold", va="center", alpha=fin)
        # HUD, top right: live counter
        val = float(curve(min(max(xt, 2010), 2025)))
        yr = int(min(np.floor(xt + 1e-6), 2025))
        ov.text(W - 96, 78, str(yr), color=GREY, fontsize=22, fontweight="semibold", va="center", ha="right",
                alpha=fin)
        ov.text(W - 200, 138, f"{val:.2f}", color=ACC, fontsize=58, fontweight="bold", va="center", ha="right",
                alpha=fin)
        ov.text(W - 96, 145, "billion", color=WHITE, fontsize=20, fontweight="medium", va="center", ha="right",
                alpha=fin * 0.9)

        # end stat
        e = smooth((t - 21.0) / 0.8)
        if e > 0:
            ov.text(AX_L + 60, 330, "+4.8 billion", color=ACC, fontsize=64, fontweight="bold",
                    va="center", alpha=e)
            ov.text(AX_L + 62, 400, "new smartphone users in 15 years", color=WHITE, fontsize=24,
                    va="center", alpha=e * 0.92)

        # source
        s = smooth((t - 22.3) / 0.8)
        if s > 0:
            ov.text(W / 2, 1038,
                    "Source: Statista, smartphone users worldwide (2014–2025). 2010–2013 are rounded estimates. "
                    "Population shares approximate.",
                    color=GREY, fontsize=15.5, va="center", ha="center", alpha=s)

        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]


_P = None


def painter():
    global _P
    if _P is None:
        _P = Painter()
    return _P


def render_chunk(args):
    idx, a, b, out = args
    tmp = out + ".part.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-r", str(FPS), tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    pa = painter()
    for i in range(a, b):
        p.stdin.write(pa.frame(i).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for chunk {idx}")
    os.rename(tmp, out)
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "preview":
        out = sys.argv[2]
        os.makedirs(out, exist_ok=True)
        from PIL import Image
        pa = painter()
        for ts in sys.argv[3:]:
            i = min(int(round(float(ts) * FPS)), NF - 1)
            Image.fromarray(pa.frame(i)).save(f"{out}/f_{float(ts):05.2f}.png")
    else:
        import multiprocessing as mp
        final = sys.argv[2]
        d = os.path.dirname(os.path.abspath(final))
        work = os.path.join(d, "chunks")
        os.makedirs(work, exist_ok=True)
        n = 10
        per = NF // n
        jobs = [(k, k * per, NF if k == n - 1 else (k + 1) * per, os.path.join(work, f"c{k:02d}.mp4"))
                for k in range(n)]
        with mp.Pool(2) as pool:
            outs = pool.map(render_chunk, jobs, chunksize=1)
        lst = os.path.join(work, "list.txt")
        with open(lst, "w") as f:
            for o in outs:
                f.write(f"file '{o}'\n")
        tmpf = final + ".part.mp4"
        subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                               "-c", "copy", "-movflags", "+faststart", tmpf])
        os.rename(tmpf, final)
        print("done", final)
