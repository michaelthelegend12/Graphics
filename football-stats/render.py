#!/usr/bin/env python3
"""Football (soccer) broadcast stats card pack.

Three 5 second 1080p cards (player name, photo placeholder, three stat bars
that fill up, team colour accents) in a Premier League TV graphics style.
All clubs and players are made up.

Needs: Python, numpy, Pillow, ffmpeg and the Inter font (fonts-inter).
Usage:  python3 render.py          render the 3 cards and the joined pack
        python3 render.py --test   save a few preview frames to /tmp/fs_test
"""
import os
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

W, H, FPS, DUR = 1920, 1080, 30, 5.0
S = 2                      # supersampling for the card layer
CW, CH = 1560, 620         # card size (1x)
CX, CY = 180, 230          # card position on screen
SK = 60                    # slant of the panel / wipe edges
BW = 46                    # wipe band width
FONT_DIR = "/usr/share/fonts/opentype/inter/"
OUT = os.path.dirname(os.path.abspath(__file__))

CARDS = [
    dict(slug="card-1-delaney", first="MARCUS", last="DELANEY", pos="FORWARD",
         team="NORTHGATE FC", crest="NFC", num="9", color=(255, 59, 78),
         stats=[("GOALS", 18, 25, "int"),
                ("SHOT ACCURACY", 62, 100, "pct"),
                ("BIG CHANCES SCORED", 14, 20, "int")]),
    dict(slug="card-2-arriaga", first="JOSE", last="ARRIAGA", pos="MIDFIELDER",
         team="HARBOUR UNITED", crest="HBU", num="8", color=(0, 178, 255),
         stats=[("ASSISTS", 11, 15, "int"),
                ("PASS COMPLETION", 91, 100, "pct"),
                ("KEY PASSES", 64, 80, "int")]),
    dict(slug="card-3-brandt", first="TOMAS", last="BRANDT", pos="GOALKEEPER",
         team="ROYAL MEADOW", crest="RMD", num="1", color=(255, 199, 44),
         stats=[("CLEAN SHEETS", 14, 20, "int"),
                ("SAVE PERCENTAGE", 78, 100, "pct"),
                ("PENALTIES SAVED", 4, 6, "int")]),
]

# ---------------------------------------------------------------- helpers


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def e_out3(x):
    return 1 - (1 - clamp(x)) ** 3


def e_out4(x):
    return 1 - (1 - clamp(x)) ** 4


def e_io3(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def e_back(x):
    x = clamp(x)
    c1, c3 = 1.4, 2.4
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


@lru_cache(None)
def font(name, size):
    return ImageFont.truetype(FONT_DIR + name, int(size * S))


def text_w(f, txt, tr=0):
    return sum(f.getlength(ch) + tr * S for ch in txt) - tr * S


def draw_tracked(d, x, y, txt, f, fill, tr=0):
    """Left-baseline text with letter spacing, x/y in 2x pixels."""
    for ch in txt:
        d.text((x, y), ch, font=f, fill=fill, anchor="ls")
        x += f.getlength(ch) + tr * S


def poly(pts, size, fill=255, mode="L"):
    im = Image.new(mode, size, 0)
    ImageDraw.Draw(im).polygon([(x * S, y * S) for x, y in pts], fill=fill)
    return im


def scale_alpha(im, a):
    if a >= 1:
        return im
    im.putalpha(im.getchannel("A").point(lambda v: int(v * a)))
    return im


def paste_text(lay, txt, x, y, f, fill, tr=0, p=1.0, dx=0, right=False):
    """Draw animated text. x,y = baseline-left (or baseline-right) in 1x."""
    if p <= 0:
        return
    wpx = int(text_w(f, txt, tr)) + 30 * S
    base = int(f.size * 1.1)
    tmp = Image.new("RGBA", (wpx, int(f.size * 1.5)), (0, 0, 0, 0))
    draw_tracked(ImageDraw.Draw(tmp), 10 * S, base, txt, f, fill, tr)
    scale_alpha(tmp, p)
    x0 = x * S + dx * S - 10 * S
    if right:
        x0 -= text_w(f, txt, tr)
    lay.alpha_composite(tmp, (int(x0), int(y * S - base)))


def lum(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


# ---------------------------------------------------------------- statics


def make_bg(color):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ty = (yy / H)[..., None]
    img = np.array([46, 8, 74], np.float32) * (1 - ty) + np.array([10, 1, 22], np.float32) * ty
    g = np.clip(1 - np.sqrt(((xx - 960) / 1100) ** 2 + ((yy - 540) / 650) ** 2), 0, 1) ** 2
    img += g[..., None] * np.array([90, 20, 120], np.float32) * 0.55
    g2 = np.clip(1 - np.sqrt(((xx - 250) / 700) ** 2 + ((yy - 950) / 400) ** 2), 0, 1) ** 2
    img += g2[..., None] * np.array(color, np.float32) * 0.18
    v = np.clip(((xx - 960) / 1100) ** 2 + ((yy - 540) / 700) ** 2, 0, 1.4)
    img *= (1 - 0.35 * v)[..., None]
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGBA")
    ov = Image.new("RGBA", (W * 2, H * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.ellipse((2 * (960 - 260), 2 * (540 - 260), 2 * (960 + 260), 2 * (540 + 260)),
              outline=(255, 255, 255, 14), width=8)
    d.line((1920, 0, 1920, 2 * H), fill=(255, 255, 255, 12), width=8)
    for x0, w, a in [(-100, 220, 10), (380, 120, 8), (1200, 300, 9), (1650, 140, 10)]:
        d.polygon([(2 * (x0 + 400), 0), (2 * (x0 + 400 + w), 0),
                   (2 * (x0 + w - 300), 2 * H), (2 * (x0 - 300), 2 * H)],
                  fill=(255, 255, 255, a))
    ov = ov.resize((W, H), Image.LANCZOS)
    return Image.alpha_composite(im, ov)


def build_static(c):
    col = c["color"]
    w, h = CW * S, CH * S
    size = (w, h)
    xs = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    t = np.clip(0.6 * xs + 0.4 * ys, 0, 1)[..., None]
    arr = np.array([66, 12, 102], np.float32) * (1 - t) + np.array([28, 3, 50], np.float32) * t
    base = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")

    # faint diagonal stripes on the info area
    ov = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for x0 in range(640, CW + 200, 90):
        d.polygon([((x0 + SK) * S, 0), ((x0 + SK + 34) * S, 0), ((x0 + 34) * S, h), (x0 * S, h)],
                  fill=(255, 255, 255, 6))
    base = Image.alpha_composite(base, ov)

    # photo panel
    pw = 590 * S
    pa = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    top = np.array([74, 24, 112], np.float32)
    bot = np.array(col, np.float32) * 0.5 + np.array([20, 4, 36], np.float32) * 0.5
    parr = np.broadcast_to(top * (1 - pa) + bot * pa, (h, pw, 3))
    panel = Image.fromarray(parr.astype(np.uint8)).convert("RGBA")
    ov = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.text((400 * S, 400 * S), c["num"], font=font("InterDisplay-Black.otf", 560),
           fill=col + (75,), anchor="mm")
    # corner brackets
    fx0, fy0, fx1, fy1, L = 40, 40, 480, 580, 34
    for (px, py, sx, sy) in [(fx0, fy0, 1, 1), (fx1, fy0, -1, 1), (fx0, fy1, 1, -1), (fx1, fy1, -1, -1)]:
        d.line([(px * S, (py + sy * L) * S), (px * S, py * S), ((px + sx * L) * S, py * S)],
               fill=(255, 255, 255, 110), width=5 * S // 2 + 2)
    f = font("Inter-Bold.otf", 18)
    lab = "PLAYER PHOTO"
    draw_tracked(d, 260 * S - text_w(f, lab, 4) / 2, 84 * S, lab, f, (255, 255, 255, 120), 4)
    panel = Image.alpha_composite(panel, ov)
    panel_mask = poly([(0, 0), (560, 0), (500, CH), (0, CH)], size)
    full_panel_mask = panel_mask
    pm = panel_mask.crop((0, 0, pw, h))
    base.paste(panel, (0, 0), pm)

    # team colour band + top stripe
    band = poly([(560, 0), (584, 0), (524, CH), (500, CH)], size)
    base.paste(col + (255,), mask=band)
    stripe = poly([(0, 0), (CW, 0), (CW - 5, 8), (0, 8)], size)
    base.paste(col + (255,), mask=stripe)

    # footer
    ov = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.polygon([(620 * S, 573 * S), (1500 * S, 573 * S), (1500 * S, 575 * S), (620 * S, 575 * S)],
              fill=(255, 255, 255, 40))
    f = font("Inter-Bold.otf", 19)
    draw_tracked(d, 620 * S, 604 * S, "2026/27 SEASON", f, (255, 255, 255, 120), 3)
    t2 = "MATCHDAY STATS"
    draw_tracked(d, 1500 * S - text_w(f, t2, 3), 604 * S, t2, f, (255, 255, 255, 120), 3)
    base = Image.alpha_composite(base, ov)

    card_mask = poly([(0, 0), (CW, 0), (CW - 40, CH), (0, CH)], size)
    base.putalpha(card_mask)

    # silhouette (placeholder for the player photo)
    sil = Image.new("RGBA", (pw, h), (0, 0, 0, 0))
    for off, colr in [(14, col + (210,)), (0, (20, 4, 38, 240))]:
        d = ImageDraw.Draw(sil)
        ox = (off - 40) * S
        d.ellipse(((265 - 88) * S + ox, (240 - 88) * S, (265 + 88) * S + ox, (240 + 88) * S), fill=colr)
        d.rectangle(((235) * S + ox, 300 * S, 295 * S + ox, 380 * S), fill=colr)
        d.ellipse(((265 - 215) * S + ox, 360 * S, (265 + 215) * S + ox, 820 * S), fill=colr)

    # drop shadow
    pad = 120
    sh = Image.new("L", ((CW + 2 * pad), (CH + 2 * pad)), 0)
    ImageDraw.Draw(sh).polygon([(pad, pad + 26), (pad + CW, pad + 26), (pad + CW - 40, pad + CH + 26), (pad, pad + CH + 26)], fill=170)
    sh = sh.filter(ImageFilter.GaussianBlur(34))
    shadow = Image.new("RGBA", sh.size, (0, 0, 0, 0))
    shadow.putalpha(sh)

    return dict(bg=make_bg(col), card=base, card_mask=card_mask, panel_mask=pm,
                sil=sil, shadow=shadow, pad=pad)


# ---------------------------------------------------------------- frame


def draw_row(lay, i, stat, rs, t, col):
    label, val, vmax, kind = stat
    a = e_out3((t - rs) / 0.4)
    if a <= 0:
        return
    fp = e_out4((t - rs - 0.25) / 1.55)
    row = Image.new("RGBA", (920 * S, 84 * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(row)
    draw_tracked(d, 20 * S, 30 * S, label, font("Inter-SemiBold.otf", 26), (255, 255, 255, 215), 3)
    cur = val * fp
    txt = f"{cur:.0f}%" if kind == "pct" else f"{cur:.0f}"
    fv = font("InterDisplay-ExtraBold.otf", 54)
    d.text((900 * S, 38 * S), txt, font=fv, fill=(255, 255, 255, 255), anchor="rs")
    # track
    x0, x1, y0, y1 = 20, 900, 52, 70
    d.polygon([((x0 + 7) * S, y0 * S), ((x1 + 7) * S, y0 * S), (x1 * S, y1 * S), (x0 * S, y1 * S)],
              fill=(255, 255, 255, 34))
    # fill
    fw = (x1 - x0) * (val / vmax) * fp
    if fw > 1:
        d.polygon([((x0 + 7) * S, y0 * S), ((x0 + fw + 7) * S, y0 * S), ((x0 + fw) * S, y1 * S), (x0 * S, y1 * S)],
                  fill=col + (255,))
        hi = tuple(int(c + (255 - c) * 0.4) for c in col) + (255,)
        d.polygon([((x0 + 7) * S, y0 * S), ((x0 + fw + 7) * S, y0 * S), ((x0 + fw + 5.5) * S, (y0 + 5) * S), ((x0 + 5.5) * S, (y0 + 5) * S)],
                  fill=hi)
        d.polygon([((x0 + fw - 5 + 7) * S, y0 * S), ((x0 + fw + 9) * S, y0 * S), ((x0 + fw + 2) * S, y1 * S), ((x0 + fw - 5) * S, y1 * S)],
                  fill=(255, 255, 255, 255))
    # quarter ticks
    for q in (0.25, 0.5, 0.75):
        tx = x0 + (x1 - x0) * q
        d.polygon([((tx + 7) * S, y0 * S), ((tx + 8.5) * S, y0 * S), ((tx + 1.5) * S, y1 * S), (tx * S, y1 * S)],
                  fill=(20, 4, 38, 120))
    scale_alpha(row, a)
    lay.alpha_composite(row, (int((600 + (a - 1) * 50) * S), int((306 + 88 * i) * S)))


def render_frame(c, st, t):
    col = c["color"]
    lay = st["card"].copy()
    ink = (20, 4, 38, 255) if lum(col) > 150 else (255, 255, 255, 255)

    # silhouette rise
    e = e_out3((t - 0.35) / 0.65)
    if e > 0:
        sh = Image.new("RGBA", st["sil"].size, (0, 0, 0, 0))
        sh.paste(st["sil"], (0, int((1 - e) * 90 * S)))
        al = ImageChops.multiply(sh.getchannel("A"), st["panel_mask"]).point(lambda v: int(v * e))
        sh.putalpha(al)
        lay.alpha_composite(sh, (0, 0))

    # tag + names
    p = e_out3((t - 0.45) / 0.4)
    paste_text(lay, "SEASON STATS", 620, 44, font("Inter-Bold.otf", 18), col + (255,), 5, p, (p - 1) * 50)
    paste_text(lay, c["first"], 620, 92, font("Inter-Bold.otf", 42), (255, 255, 255, 235), 6, p, (p - 1) * 50)
    p = e_out3((t - 0.55) / 0.45)
    size = 132
    while text_w(font("InterDisplay-Black.otf", size), c["last"], -2) > 730 * S and size > 60:
        size -= 4
    paste_text(lay, c["last"], 614, 214, font("InterDisplay-Black.otf", size),
               (255, 255, 255, 255), -2, p, (p - 1) * 70)

    # crest placeholder
    p = e_back((t - 0.7) / 0.4)
    if p > 0.02:
        cx, cy, r = 1440, 128, 64 * p
        d = ImageDraw.Draw(lay)
        d.ellipse(((cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S), fill=(26, 4, 46, 255),
                  outline=col + (255,), width=int(7 * S * p))
        if p > 0.85:
            f = font("InterDisplay-Black.otf", 40)
            d.text((cx * S, (cy + 1) * S), c["crest"], font=f, fill=(255, 255, 255, 255), anchor="mm")

    # position chip + team
    p = e_out3((t - 0.9) / 0.35)
    if p > 0:
        chip = Image.new("RGBA", (760 * S, 56 * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(chip)
        f = font("Inter-Bold.otf", 22)
        cw_ = text_w(f, c["pos"], 3) / S + 44
        d.polygon([(10 * S, 4 * S), ((10 + cw_ + 8) * S, 4 * S), ((10 + cw_) * S, 44 * S), (2 * S, 44 * S)],
                  fill=col + (255,))
        draw_tracked(d, 30 * S, 33 * S, c["pos"], f, ink, 3)
        draw_tracked(d, (10 + cw_ + 28) * S, 33 * S, c["team"], font("Inter-Bold.otf", 24), (255, 255, 255, 225), 3)
        scale_alpha(chip, p)
        lay.alpha_composite(chip, (int((610 + (p - 1) * 50) * S), int(224 * S)))

    # stat rows
    for i, stat in enumerate(c["stats"]):
        draw_row(lay, i, stat, 1.2 + 0.2 * i, t, col)

    # shine sweep
    ts = (t - 3.45) / 0.6
    if 0 < ts < 1:
        xs = -200 + (CW + 500) * e_io3(ts)
        sm = Image.new("L", lay.size, 0)
        d = ImageDraw.Draw(sm)
        for wdt, al in [(150, 18), (90, 28), (40, 40)]:
            d.polygon([((xs + SK + wdt / 2) * S, 0), ((xs + SK - wdt / 2) * S, 0),
                       ((xs - wdt / 2) * S, CH * S), ((xs + wdt / 2) * S, CH * S)], fill=al)
        sm = ImageChops.multiply(sm, st["card_mask"]).point(lambda v: v * 255 // 255)
        lay.paste((255, 255, 255, 255), mask=sm)

    # wipe in / out (slanted edge with a team colour band)
    pin = e_io3(t / 0.6)
    pout = e_io3((t - 4.4) / 0.55)
    size = lay.size
    wmask = st["card_mask"]
    band_m = None
    white_m = None
    if pin < 1:
        xe = -BW + pin * (CW + 2 * BW)
        m = poly([(-20, 0), (xe + SK, 0), (xe, CH), (-20, CH)], size)
        wmask = ImageChops.multiply(wmask, m)
        band_m = poly([(xe - BW + SK, 0), (xe + SK, 0), (xe, CH), (xe - BW, CH)], size)
        white_m = poly([(xe + SK, 0), (xe + SK + 10, 0), (xe + 10, CH), (xe, CH)], size)
    if pout > 0:
        xo = -BW + pout * (CW + 2 * BW)
        m = poly([(xo + SK, 0), (CW + 200, 0), (CW + 200, CH), (xo, CH)], size)
        wmask = ImageChops.multiply(wmask, m)
        band_m = poly([(xo + SK, 0), (xo + SK + BW, 0), (xo + BW, CH), (xo, CH)], size)
        white_m = poly([(xo + SK - 10, 0), (xo + SK, 0), (xo, CH), (xo - 10, CH)], size)
    lay.putalpha(ImageChops.multiply(lay.getchannel("A"), wmask))
    if band_m is not None:
        cm = st["card_mask"]
        lay.paste(col + (255,), mask=ImageChops.multiply(band_m, cm))
        lay.paste((255, 255, 255, 255), mask=ImageChops.multiply(white_m, cm))

    small = lay.resize((CW, CH), Image.BOX)
    frame = st["bg"].copy()
    # shadow
    sa = clamp((t - 0.4) / 0.25) * (1 - clamp((t - 4.3) / 0.15))
    if sa > 0:
        sh = scale_alpha(st["shadow"].copy(), sa)
        frame.alpha_composite(sh, (CX - st["pad"], CY - st["pad"]))
    frame.alpha_composite(small, (CX, CY))
    return frame.convert("RGB")


# ---------------------------------------------------------------- output


def render_card(c):
    st = build_static(c)
    final = os.path.join(OUT, c["slug"] + ".mp4")
    tmp = os.path.join(OUT, c["slug"] + ".tmp.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-framerate", str(FPS), "-i", "-", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-movflags", "+faststart", tmp]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n in range(int(DUR * FPS)):
        pr.stdin.write(render_frame(c, st, n / FPS).tobytes())
    pr.stdin.close()
    if pr.wait() != 0:
        raise SystemExit("ffmpeg failed for " + c["slug"])
    os.replace(tmp, final)
    return final


def main():
    if "--test" in sys.argv:
        os.makedirs("/tmp/fs_test", exist_ok=True)
        for k, c in enumerate(CARDS):
            st = build_static(c)
            for t in (0.3, 0.9, 1.8, 3.0, 3.7, 4.7):
                render_frame(c, st, t).save(f"/tmp/fs_test/c{k + 1}_{t:.1f}.png")
        return
    with Pool(2) as pool:
        files = pool.map(render_card, CARDS)
    lst = os.path.join(OUT, "list.txt")
    with open(lst, "w") as f:
        for p in files:
            f.write(f"file '{p}'\n")
    tmp = os.path.join(OUT, "football-stats-pack.tmp.mp4")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                           "-i", lst, "-c", "copy", tmp])
    os.replace(tmp, os.path.join(OUT, "football-stats-pack.mp4"))
    os.remove(lst)


if __name__ == "__main__":
    main()
