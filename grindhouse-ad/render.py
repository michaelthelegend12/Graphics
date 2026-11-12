"""Grindhouse: 15 s vertical (1080x1920, 30 fps) ad for a made-up coffee subscription.
Every product shot (bag, cup, beans, box, flame, loop) is built from plain shapes.
Run: python3 music.py && python3 render.py  -> grindhouse-ad.mp4
Preview frames: python3 render.py --preview 0.1 1.5 3.5 ..."""
import math, random, subprocess, sys, os
from functools import lru_cache
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageChops
from multiprocessing import Pool

W, H, FPS = 1080, 1920, 30
BPM = 128; B = 60 / BPM
N = 450
FONT = "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"
DARK = (27, 15, 10); CREAM = (245, 230, 207); ORANGE = (255, 90, 31); YELLOW = (255, 201, 60)
S = 2  # sprite supersampling

# ---------------------------------------------------------------- helpers
_f = {}
def font(sz):
    sz = max(4, int(round(sz)))
    if sz not in _f: _f[sz] = ImageFont.truetype(FONT, sz)
    return _f[sz]
def tw(s, size): return font(size).getlength(s)
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def prog(t, t0, d): return clamp((t - t0) / d)
def lerp(a, b, k): return a + (b - a) * k
def out_back(x, s=1.8): x -= 1; return 1 + (s + 1) * x ** 3 + s * x ** 2
def out_expo(x): return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)
def out_cubic(x): return 1 - (1 - x) ** 3
def out_bounce(x):
    n1, d1 = 7.5625, 2.75
    if x < 1 / d1: return n1 * x * x
    if x < 2 / d1: x -= 1.5 / d1; return n1 * x * x + .75
    if x < 2.5 / d1: x -= 2.25 / d1; return n1 * x * x + .9375
    x -= 2.625 / d1; return n1 * x * x + .984375
def pop(lt, t0, d=0.22, frm=0.0):
    if lt < t0: return 0.0
    return frm + (1 - frm) * out_back(prog(lt, t0, d))
def slam(lt, t0, d=0.13, big=1.5):
    if lt < t0: return None
    return 1 + (big - 1) * (1 - out_expo(prog(lt, t0, d)))
def shk(lt, t0, mag=10, dur=0.25):
    k = lt - t0
    if k < 0 or k > dur: return 0.0, 0.0
    a = mag * (1 - k / dur)
    return a * math.sin(k * 130), a * math.cos(k * 110)

# ---------------------------------------------------------------- sprite drawing (shapes only)
class P:
    def __init__(s, w, h): s.im = Image.new("RGBA", (int(w * S), int(h * S)), (0, 0, 0, 0)); s.d = ImageDraw.Draw(s.im)
    def sc(s, v): return [x * S for x in v]
    def rect(s, box, r=0, fill=None, outline=None, width=1, corners=None):
        s.d.rounded_rectangle(s.sc(box), radius=r * S, fill=fill, outline=outline, width=int(width * S), corners=corners)
    def ell(s, box, fill=None, outline=None, width=1): s.d.ellipse(s.sc(box), fill=fill, outline=outline, width=int(width * S))
    def poly(s, pts, fill): s.d.polygon([(x * S, y * S) for x, y in pts], fill=fill)
    def line(s, pts, fill, width): s.d.line([(x * S, y * S) for x, y in pts], fill=fill, width=int(width * S), joint="curve")
    def arc(s, box, a0, a1, fill, width): s.d.arc(s.sc(box), a0, a1, fill=fill, width=int(width * S))
    def text(s, xy, t, size, fill): s.d.text((xy[0] * S, xy[1] * S), t, font=font(size * S), fill=fill, anchor="mm")

def shade(im, shapes):
    base = im.getchannel("A")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
    for pts, col in shapes: od.polygon([(x * S, y * S) for x, y in pts], fill=col)
    out = Image.alpha_composite(im, ov)
    out.putalpha(ImageChops.multiply(out.getchannel("A"), base))
    return out

def chaikin(pts, it=3):
    for _ in range(it):
        new = []
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            new += [(.75 * a[0] + .25 * b[0], .75 * a[1] + .25 * b[1]), (.25 * a[0] + .75 * b[0], .25 * a[1] + .75 * b[1])]
        pts = new
    return pts

BEAN = (118, 66, 36); BEAN_D = (52, 28, 16); BEAN_HI = (168, 104, 60)
def build_bean():
    w, h = 110, 150; p = P(w, h)
    p.ell((4, 4, w - 4, h - 4), fill=BEAN_D); p.ell((9, 9, w - 9, h - 9), fill=BEAN)
    p.ell((24, 22, 50, 70), fill=BEAN_HI)
    p.line([(w / 2 + 9 * math.sin(i / 24 * 2 * math.pi), 18 + i / 24 * (h - 36)) for i in range(25)], BEAN_D, 8)
    return p.im

def build_bag():
    w, h = 600, 900; p = P(w, h)
    BAG = (38, 24, 17)
    p.rect((40, 90, 560, 870), 44, BAG)
    p.rect((40, 36, 560, 132), 18, (62, 40, 28))
    for x in range(62, 540, 24): p.line([(x, 50), (x, 120)], (22, 13, 9), 3)
    p.ell((268, 160, 332, 224), fill=(74, 50, 38)); p.ell((284, 176, 316, 208), fill=(20, 12, 8))
    p.rect((92, 262, 508, 716), 30, CREAM)
    p.rect((92, 262, 508, 346), 30, ORANGE, corners=(True, True, False, False))
    p.text((300, 304), "FRESH ROASTED", 34, CREAM)
    p.ell((228, 372, 372, 516), fill=DARK); p.text((300, 446), "G", 112, CREAM)
    p.text((300, 570), "GRINDHOUSE", 50, DARK)
    p.text((300, 626), "DARK ROAST", 30, ORANGE)
    p.text((300, 676), "WHOLE BEAN  ·  12 OZ", 22, (120, 84, 60))
    for i in range(3): p.rect((150, 740 + i * 22, 450, 750 + i * 22), 5, ORANGE if i == 1 else (62, 40, 28))
    im = shade(p.im, [([(40, 90), (118, 90), (118, 870), (40, 870)], (0, 0, 0, 70)),
                      ([(478, 90), (560, 90), (560, 870), (478, 870)], (255, 255, 255, 26)),
                      ([(150, 100), (176, 100), (176, 250), (150, 250)], (255, 255, 255, 22))])
    return im

def build_cup(sad=False):
    w, h = 640, 640; p = P(w, h)
    hw = lambda y: 190 - (y - 210) * (55 / 300)
    p.ell((50, 528, 590, 624), fill=(196, 142, 24)); p.ell((50, 508, 590, 604), fill=YELLOW)
    p.ell((190, 524, 450, 584), fill=(232, 172, 40))
    p.ell((428, 250, 584, 414), outline=DARK, width=28)
    pts = [(320 - hw(y), y) for y in range(210, 521, 10)] + [(320 + hw(y), y) for y in range(520, 209, -10)]
    p.poly(pts, DARK); p.ell((320 - hw(520), 484, 320 + hw(520), 556), fill=DARK)
    p.ell((130, 170, 510, 250), fill=(74, 48, 36)); p.ell((148, 182, 492, 238), fill=(104, 60, 34))
    hp = P(100, 90)
    hp.ell((4, 4, 58, 58), fill=CREAM); hp.ell((42, 4, 96, 58), fill=CREAM); hp.poly([(8, 40), (92, 40), (50, 88)], CREAM)
    heart = hp.im.resize((130 * S, 30 * S), Image.LANCZOS); p.im.paste(heart, (255 * S, 195 * S), heart)
    if not sad:
        p.ell((252, 330, 388, 466), fill=ORANGE); p.text((320, 398), "G", 100, CREAM)
    else:
        for x in (268, 372):
            p.ell((x - 24, 356, x + 24, 404), fill=CREAM); p.ell((x - 10, 376, x + 10, 398), fill=DARK)
        p.line([(236, 340), (292, 322)], CREAM, 9); p.line([(348, 322), (404, 340)], CREAM, 9)
        p.arc((274, 428, 366, 496), 200, 340, CREAM, 10)
    return shade(p.im, [([(320 - hw(y) + 22, y) for y in range(230, 500, 10)] + [(320 - hw(y) + 46, y) for y in range(490, 229, -10)], (255, 255, 255, 34))])

def build_box():
    p = P(520, 440)
    p.poly([(40, 150), (400, 150), (400, 410), (40, 410)], (206, 150, 92))
    p.poly([(40, 150), (400, 150), (484, 84), (124, 84)], (232, 182, 122))
    p.poly([(400, 150), (484, 84), (484, 344), (400, 410)], (168, 116, 66))
    p.poly([(196, 150), (262, 150), (346, 84), (280, 84)], CREAM); p.poly([(196, 150), (262, 150), (262, 280), (196, 280)], CREAM)
    p.rect((80, 310, 180, 372), 8, CREAM); p.text((130, 341), "G", 44, DARK)
    return p.im

def build_flame():
    p = P(300, 410)
    outer = [(150, 8), (205, 110), (258, 200), (262, 292), (212, 372), (150, 398), (88, 372), (38, 292), (42, 200), (98, 110)]
    o = chaikin(outer); p.poly(o, (255, 64, 20))
    inner = [(150 + (x - 150) * .56, 330 + (y - 330) * .56) for x, y in chaikin(outer, 1)]
    p.poly(chaikin(inner, 2), YELLOW)
    return p.im

def build_loop():
    p = P(400, 400); c = 200; r = 130; col = DARK
    for a0, a1 in ((20, 150), (200, 330)):
        p.arc((c - r, c - r, c + r, c + r), a0, a1, col, 38)
        th = math.radians(a1); ex, ey = c + r * math.cos(th), c + r * math.sin(th)
        tx, ty = -math.sin(th), math.cos(th); nx, ny = math.cos(th), math.sin(th)
        p.poly([(ex + tx * 70, ey + ty * 70), (ex + nx * 62 - tx * 10, ey + ny * 62 - ty * 10), (ex - nx * 62 - tx * 10, ey - ny * 62 - ty * 10)], col)
    return p.im

def build_shadow():
    p = P(600, 80); p.ell((0, 0, 600, 80), fill=(0, 0, 0, 70)); return p.im

@lru_cache(maxsize=None)
def sticker(lines, w, h, sizes, bg, fg, shape="rect", r=None):
    p = P(w, h)
    if shape == "rect": p.rect((0, 0, w, h), r if r is not None else h / 2, fill=bg)
    else: p.ell((0, 0, w, h), fill=bg)
    total = sum(sizes) * 1.12; y = h / 2 - total / 2
    for s, sz in zip(lines, sizes):
        p.text((w / 2, y + sz * .58), s, sz, fg); y += sz * 1.12
    return p.im

SPR = {}
def build_all():
    SPR.update(bean=build_bean(), bag=build_bag(), cup=build_cup(), cupsad=build_cup(True), box=build_box(),
              flame=build_flame(), loop=build_loop(), shadow=build_shadow())

def blit(img, spr, cx, cy, scale=1.0, rot=0.0, alpha=1.0):
    if scale <= 0.02: return
    k = scale / S
    s = spr.resize((max(1, int(spr.width * k)), max(1, int(spr.height * k))), Image.BILINEAR)
    if rot: s = s.rotate(rot, resample=Image.BICUBIC, expand=True)
    if alpha < 1:
        s = s.copy(); s.putalpha(s.getchannel("A").point(lambda v: int(v * alpha)))
    img.paste(s, (int(cx - s.width / 2), int(cy - s.height / 2)), s)

def text(img, s, cx, cy, size, fill, rot=0, shadow=None, scale=1.0, maxw=None):
    if maxw: size = min(size, size * maxw / tw(s, size))
    size *= scale
    if size < 5: return
    f = font(size)
    if rot == 0:
        d = ImageDraw.Draw(img)
        if shadow: d.text((cx + shadow[0], cy + shadow[1]), s, font=f, fill=shadow[2], anchor="mm")
        d.text((cx, cy), s, font=f, fill=fill, anchor="mm"); return
    l, t_, r, b = f.getbbox(s, anchor="mm"); pad = 30 + (abs(shadow[0]) + abs(shadow[1]) if shadow else 0)
    lay = Image.new("RGBA", (int(r - l + 2 * pad), int(b - t_ + 2 * pad)), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
    c0 = (lay.width / 2, lay.height / 2)
    if shadow: ld.text((c0[0] + shadow[0], c0[1] + shadow[1]), s, font=f, fill=shadow[2], anchor="mm")
    ld.text(c0, s, font=f, fill=fill, anchor="mm")
    lay = lay.rotate(rot, resample=Image.BICUBIC, expand=True)
    img.paste(lay, (int(cx - lay.width / 2), int(cy - lay.height / 2)), lay)

def letters(img, s, cx, cy, size, fill, t0, lt, stag=0.05, shadow=None, maxw=None):
    if maxw: size = min(size, size * maxw / tw(s, size))
    f = font(size); total = f.getlength(s)
    for i, ch in enumerate(s):
        sc = slam(lt, t0 + i * stag, 0.14, 2.2)
        if sc is None: continue
        x0 = cx - total / 2 + f.getlength(s[:i]); w = f.getlength(ch)
        text(img, ch, x0 + w / 2, cy, size, fill, shadow=shadow, scale=sc)

# ---------------------------------------------------------------- numpy backgrounds
def solid(c):
    a = np.empty((H, W, 3), np.uint8); a[:] = c; return a
def rays(c1, c2, cx, cy, rot, n=18):
    a = solid(c1); R = 2800
    for i in range(n):
        a0 = rot + i * 2 * math.pi / n; a1 = a0 + math.pi / n
        pts = np.array([[cx, cy], [cx + R * math.cos(a0), cy + R * math.sin(a0)], [cx + R * math.cos(a1), cy + R * math.sin(a1)]], np.int32)
        cv2.fillPoly(a, [pts], c2, lineType=cv2.LINE_AA)
    return a
def burst(a, cx, cy, R, n, color, rot, inner=0.84):
    pts = []
    for i in range(n * 2):
        r = R if i % 2 == 0 else R * inner; ang = rot + i * math.pi / n
        pts.append([cx + r * math.cos(ang), cy + r * math.sin(ang)])
    cv2.fillPoly(a, [np.array(pts, np.int32)], color, lineType=cv2.LINE_AA)
def sparkle(a, cx, cy, r, color):
    if r < 2: return
    pts = []
    for i in range(8):
        rr = r if i % 2 == 0 else r * .22; ang = i * math.pi / 4 - math.pi / 2
        pts.append([cx + rr * math.cos(ang), cy + rr * math.sin(ang)])
    cv2.fillPoly(a, [np.array(pts, np.int32)], color, lineType=cv2.LINE_AA)
def make_glow():
    yy, xx = np.mgrid[0:H, 0:W]; d = np.sqrt((xx - 540) ** 2 + (yy - 900) ** 2) / 1000.0
    k = np.clip(1 - d, 0, 1) ** 2 * .55
    base = np.array(DARK, float); g = np.array(ORANGE, float)
    return (base + (g - base) * k[..., None]).astype(np.uint8)
GLOW = None

def steam_layer(t, w=360, h=420):
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for k in range(3):
        x0 = 90 + k * 90; ph = k * 1.9
        pts = []
        for j in range(41):
            u = j / 40; y = h - 20 - u * (h - 40)
            x = x0 + 24 * math.sin(u * 8 - t * 5 + ph) * (.35 + u)
            pts.append((x, y, u))
        for (xa, ya, ua), (xb, yb, ub) in zip(pts, pts[1:]):
            col = (255, 255, 255, int(235 * (1 - ua) ** .9))
            d.line([(xa * S, ya * S), (xb * S, yb * S)], fill=col, width=18 * S)
            d.ellipse([(xb - 9) * S, (yb - 9) * S, (xb + 9) * S, (yb + 9) * S], fill=col)
    return im.resize((w, h), Image.LANCZOS)

# ---------------------------------------------------------------- scenes
rnd = random.Random(3)
RAIN = [(rnd.uniform(0, W), rnd.uniform(0, H + 400), rnd.uniform(1300, 2300), rnd.uniform(0, 360), rnd.uniform(-360, 360), rnd.uniform(.7, 1.5)) for _ in range(34)]
BURST = [(rnd.uniform(-260, 260), rnd.uniform(-1500, -800), rnd.uniform(-500, 500), rnd.uniform(-540, 540), rnd.uniform(.5, 1.0), rnd.choice((-1, 1))) for _ in range(16)]

def sc_hook(lt):
    img = Image.fromarray(rays(ORANGE, (255, 118, 54), 540, 760, lt * .9, 16))
    ox, oy = shk(lt, 0, 12, .3)
    s = slam(lt, 0.0, .12, 1.4)
    if s: text(img, "SAD", 540 + ox, 430 + oy, 380, CREAM, shadow=(14, 14, DARK), scale=s, maxw=800)
    s = slam(lt, 0.17, .12, 1.5)
    if s:
        ox, oy = shk(lt, .17, 12, .3)
        text(img, "COFFEE?", 540 + ox, 700 + oy, 260, CREAM, shadow=(12, 12, DARK), scale=s, maxw=940)
    s = pop(lt, 0.5, .2, .2)
    if s: blit(img, sticker(("NOT ANYMORE.",), 800, 150, (74,), YELLOW, DARK), 540, 930, s, -6)
    blit(img, SPR["shadow"], 540, 1560, 1.0)
    blit(img, SPR["cupsad"], 540 + 8 * math.sin(lt * 7), 1260, 1.05, 3 * math.sin(lt * 6))
    ty = 1380 + ((lt * 1.8) % 1) * 170; d = ImageDraw.Draw(img)
    d.ellipse((488, ty, 510, ty + 30), fill=(130, 205, 255)); 
    return img

def sc_logo(lt):
    img = Image.fromarray(GLOW.copy())
    def beans(front):
        for i, (x, y0, v, r0, rv, sc) in enumerate(RAIN):
            if (i % 3 == 0) != front: continue
            y = (y0 + lt * v) % (H + 400) - 200
            blit(img, SPR["bean"], x, y, sc * (1.0 if front else 1), r0 + rv * lt)
    beans(False)
    letters(img, "GRIND", 540, 800, 300, CREAM, .05, lt, shadow=(12, 12, ORANGE), maxw=940)
    letters(img, "HOUSE", 540, 1090, 300, YELLOW, .30, lt, shadow=(12, 12, ORANGE), maxw=940)
    s = pop(lt, .85, .2)
    if s: text(img, "COFFEE SUBSCRIPTION", 540, 1320, 56, CREAM, scale=s, maxw=900)
    beans(True)
    return img

def sc_bag(lt):
    bg = rays(YELLOW, (255, 222, 112), 540, 1100, lt * .7, 20)
    img = Image.fromarray(bg)
    x = prog(lt, 0, .7); y = lerp(-800, 1110, out_bounce(x))
    rot = -10 * (1 - out_cubic(prog(lt, 0, .8))) + 1.5 * math.sin(lt * 3)
    sh = clamp((y + 800) / 1900)
    blit(img, SPR["shadow"], 540, 1650, 1.2 * sh, alpha=sh)
    blit(img, SPR["bag"], 540, y, 1.15, rot)
    for bx, by0, vx, vy, sc, sg in BURST:
        k = lt - .26
        if k < 0: continue
        px = 540 + bx + vx * k; py = 1350 + by0 * .25 + vy * 0.0 + (by0 * k * 1.1) + 0.5 * 2600 * k * k
        blit(img, SPR["bean"], px, py, sc, sg * 500 * k)
    ox, oy = shk(lt, .3, 10)
    s = slam(lt, .3, .13, 1.5)
    if s: text(img, "FRESH", 540 + ox, 215 + oy, 290, DARK, shadow=(10, 10, CREAM), scale=s, maxw=880)
    s = slam(lt, .5, .13, 1.5)
    if s: text(img, "ROASTED", 540 + ox, 440 + oy, 200, ORANGE, shadow=(10, 10, DARK), scale=s, maxw=900)
    s = pop(lt, 1.1, .25)
    if s: blit(img, sticker(("100%", "ARABICA"), 290, 290, (72, 40), ORANGE, CREAM, "circle"), 880, 760, s, 12)
    return img

def sc_cup(lt):
    bg = solid(CREAM); pr = out_back(prog(lt, 0, .35))
    cv2.circle(bg, (540, 1060), int(560 * pr), ORANGE, -1, cv2.LINE_AA)
    ring = (lt * 1.1) % 1; col = tuple(int(lerp(c, k, ring)) for c, k in zip(YELLOW, CREAM))
    cv2.circle(bg, (540, 1060), int(590 + 230 * ring), col, 8, cv2.LINE_AA)
    for (sx, sy, ph) in ((190, 700, 0), (900, 820, .5), (840, 560, 1.0)):
        sparkle(bg, sx, sy, 46 * abs(math.sin(lt * 5 + ph * 3)) + 10, YELLOW if sy > 600 else ORANGE)
    img = Image.fromarray(bg)
    e = out_back(prog(lt, 0, .5)); cx = lerp(-600, 540, e); rot = -25 * (1 - e)
    blit(img, SPR["shadow"], cx, 1520, 1.35, alpha=clamp(e))
    blit(img, SPR["cup"], cx, 1130, 1.5, rot)
    if lt > .5: blit(img, steam_layer(lt), 540, 735, 1.5)
    words = [("BOLD.", DARK, 0), ("SMOOTH.", ORANGE, 2 * B), ("YOURS.", DARK, 4 * B)]
    for wd, c, t0 in reversed(words):
        if lt >= t0:
            s = slam(lt, t0, .13, 1.5); ox, oy = shk(lt, t0, 10)
            text(img, wd, 540 + ox, 330 + oy, 300, c, shadow=(10, 10, (255, 255, 255)), scale=s, maxw=900); break
    return img

MONT = [("ROAST.", DARK, CREAM, ORANGE, (40, 24, 18), "flame"), ("SHIP.", YELLOW, DARK, CREAM, (255, 214, 100), "box"),
        ("SIP.", CREAM, DARK, ORANGE, (236, 214, 180), "cup"), ("REPEAT.", ORANGE, CREAM, DARK, (255, 118, 54), "loop")]
def sc_montage(lt):
    i = min(3, int(lt / B)); wd, c1, tc, sh, c2, ic = MONT[i]; k = lt - i * B
    img = Image.fromarray(rays(c1, c2, 540, 800, k * 1.5 + i, 14))
    z = 1 + .1 * k
    s = pop(k, 0, .2, .3)
    if ic == "cup":
        blit(img, SPR["cup"], 540, 760, 1.0 * s * z); blit(img, steam_layer(lt), 540, 400, 1.0 * s)
    elif ic == "flame": blit(img, SPR[ic], 540, 760, 1.5 * s * z, 3 * math.sin(lt * 30))
    elif ic == "loop": blit(img, SPR[ic], 540, 760, 1.9 * s * z, k * 160)
    else: blit(img, SPR[ic], 540, 760, 1.6 * s * z, -4)
    sc = slam(k, .04, .1, 1.35)
    ox, oy = shk(k, 0, 8, .2)
    text(img, wd, 540 + ox, 1320 + oy, 330, tc, shadow=(12, 12, sh), scale=sc * z if sc else 0, maxw=900)
    return img

def sc_price(lt):
    bg = GLOW.copy()
    pr = out_back(prog(lt, .02, .3))
    burst(bg, 540, 880, 640 * pr, 22, ORANGE, -lt * .35, .86)
    burst(bg, 540, 880, 560 * pr, 20, YELLOW, lt * .5, .86)
    for (sx, sy, ph) in ((130, 380, 0), (960, 1180, 1.2), (160, 1230, 2.1), (940, 400, .6)):
        sparkle(bg, sx, sy, 44 * abs(math.sin(lt * 5 + ph)) + 6, YELLOW)
    img = Image.fromarray(bg)
    s = pop(lt, 0, .2)
    if s: text(img, "ONLY", 540, 330, 130, CREAM, shadow=(8, 8, ORANGE), scale=s)
    n = int(14 * out_cubic(prog(lt, .15, .4)))
    s = pop(lt, .15, .2, .4); ox, oy = shk(lt, .55, 12)
    if s: text(img, f"${n}", 540 + ox, 850 + oy, 400, DARK, scale=s, maxw=680)
    s = pop(lt, .5, .22)
    if s: text(img, "/month", 540, 1090, 150, DARK, scale=s)
    s = pop(lt, 1.0, .22)
    if s: blit(img, sticker(("FREE SHIPPING",), 780, 120, (60,), CREAM, DARK), 540, 1500, s)
    s = pop(lt, 1.35, .22)
    if s: blit(img, sticker(("CANCEL ANYTIME",), 780, 120, (60,), CREAM, DARK), 540, 1650, s)
    return img

def sc_cta(lt):
    img = Image.fromarray(rays(ORANGE, (255, 118, 54), 540, 900, lt * .6, 18))
    s = slam(lt, .0, .13, 1.4); ox, oy = shk(lt, 0, 10)
    if s: text(img, "START YOUR", 540 + ox, 330 + oy, 170, CREAM, shadow=(8, 8, DARK), scale=s, maxw=900)
    s = slam(lt, .15, .13, 1.5)
    if s:
        ox, oy = shk(lt, .15, 12)
        text(img, "GRIND.", 540 + ox, 600 + oy, 340, DARK, shadow=(12, 12, YELLOW), scale=s, maxw=900)
    s = pop(lt, .35, .25)
    blit(img, SPR["shadow"], 540, 1400, .95 * s)
    blit(img, SPR["bag"], 340, 1075, .6 * s, -9)
    blit(img, SPR["cup"], 760, 1160, .66 * s, 8)
    if lt > .5: blit(img, steam_layer(lt), 750, 930, .6)
    s = pop(lt, .6, .25)
    if s:
        p = 1 + .04 * math.sin(lt * 12) if lt > .9 else 1
        blit(img, sticker(("SUBSCRIBE NOW",), 820, 170, (72,), DARK, DARK), 540, 1584, s * p * 1.0)
        blit(img, sticker(("SUBSCRIBE NOW",), 820, 170, (72,), YELLOW, DARK), 540, 1568, s * p * 1.0)
    return img

SCENES = [(0, sc_hook), (2, sc_logo), (6, sc_bag), (12, sc_cup), (18, sc_montage), (22, sc_price), (28, sc_cta)]
CUTS = [(0, .35), (2 * B, .6), (6 * B, .6), (12 * B, .6), (18 * B, .6), (19 * B, .25), (20 * B, .25), (21 * B, .25), (22 * B, .7), (28 * B, .6)]

def frame(i):
    t = i / FPS; bt = t / B; k = 0
    for j, (b0, _) in enumerate(SCENES):
        if bt >= b0 - 1e-9: k = j
    lt = t - SCENES[k][0] * B
    arr = np.asarray(SCENES[k][1](lt))
    z = 1.0; fl = 0.0
    for tc, amp in CUTS:
        dt = t - tc
        if 0 <= dt < .16:
            q = 1 - dt / .16; z = max(z, 1 + .06 * amp * q * q); fl = max(fl, .5 * amp * max(0, 1 - dt / .09))
    if z > 1.0005:
        M = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H / 2]])
        arr = cv2.warpAffine(arr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if fl > .01: arr = cv2.addWeighted(arr, 1 - fl, np.full_like(arr, 255), fl, 0)
    return np.ascontiguousarray(arr).tobytes()

def init():
    global GLOW
    GLOW = make_glow(); build_all()

if __name__ == "__main__":
    init()
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        os.makedirs("prev", exist_ok=True)
        for a in sys.argv[2:]:
            i = int(round(float(a) * FPS)); Image.frombytes("RGB", (W, H), frame(i)).save(f"prev/f_{a}.png")
        sys.exit()
    out = "grindhouse-ad.mp4"; tmp = "tmp_render.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", "music.wav", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-af", "volume=-2.9dB", "-c:a", "aac", "-b:a", "192k", "-t", "15", "-movflags", "+faststart", tmp]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(2, initializer=lambda: None) as pool:
        for fb in pool.imap(frame, range(N), chunksize=6): pr.stdin.write(fb)
    pr.stdin.close()
    if pr.wait() == 0: os.replace(tmp, out); print("wrote", out)
    else: sys.exit("ffmpeg failed")
