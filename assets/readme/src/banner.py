"""CAS-Canglong README banner, drawn as pixel art on a 480 x 144 grid and written as SVG at 4x (light and dark).

    python assets/readme/src/banner.py assets/readme        # -> banner.svg, banner_dark.svg (+ PNG previews with -p)

Every pixel is placed by this script: the azure dragon coiled round a Pacific-centred Earth (the logo in logo.py),
the dithered sky that fades into the page instead of ending at a frame, waves, a Niño 3.4 ensemble plume and a seal.
Needs numpy and Pillow. Fonts (SIL OFL) go in ./fonts next to this file:
    Jersey10-Regular.ttf, Silkscreen-Regular.ttf   github.com/google/fonts  (ofl/jersey10, ofl/silkscreen)
    fusion-pixel-12px-proportional-zh_hans.ttf    github.com/TakWolf/fusion-pixel-font (release zip)
land_1deg.txt is Natural Earth 1:110m land (public domain) on a 1-degree grid.
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H, SCALE = 480, 144, 4
BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16
DITHER = np.tile(BAYER, (H // 4 + 1, W // 4 + 1))[:H, :W]
YY, XX = np.mgrid[0:H, 0:W]

FONT = {
    "title": ImageFont.truetype(str(HERE / "fonts" / "Jersey10-Regular.ttf"), 30),
    "small": ImageFont.truetype(str(HERE / "fonts" / "Silkscreen-Regular.ttf"), 8),
    "text": ImageFont.truetype(str(HERE / "fonts" / "fusion-pixel-12px-proportional-zh_hans.ttf"), 12),
}

# Sprite colours are shared by both themes; the scene around them changes.
SPRITE = dict(
    outline="#10294d", body=("#2a64a8", "#3b88d8", "#69b2f2", "#b3e0ff"),
    belly=("#c99f4c", "#e8c977", "#f8e8b2"), fin=("#b8322d", "#e8553b", "#ff9258"),
    horn=("#c8963f", "#f1cf7e", "#fff0c2"), whisker="#e2a63e", eye="#ffe14d", mouth="#7c1f30", tooth="#ffffff",
    ocean=("#1d4f93", "#2867b4", "#3a85cf", "#5aa6e6"), land=("#2f7342", "#45964f", "#6cbd62", "#a5dd7c"),
    ice=("#c4dcef", "#eef7ff"), warm=("#d6402f", "#f2703f", "#fba35a", "#f7cf7e"), seal="#cf3528", seal_ink="#fff4e4",
)
THEMES = {
    "light": dict(
        sky=("#c6e2fa", "#d5eafc", "#e3f1fd", "#eff8fe"), star=("#ffffff", "#ffffff"),
        sea=("#a9dcfb", "#73bdf0", "#4b9ee0", "#357fc6", "#2765a8"), foam="#ffffff", glint="#d8f1ff",
        title=("#2b78c6", "#1a5ea7", "#143f73"), title_shadow="#a6d0f3", ink2="#3d5370",
        muted="#6c7f98", atmos="#bfe6ff",
        card="#fbfdff", card_bar="#e2effb", card_edge="#a8cbee", card_shadow="#b3d3f0", grid="#9fb6cc",
        plume=("#ffd9c7", "#ffae8c", "#f37a5a"), mean="#d3312a", cool=("#cfe2f7", "#9cc4ee", "#5e9ad8"), cool_mean="#2a68b8", obs="#20344f", nina="#eaf3fd", nino="#fff1ea",
        chips=[("#d6eaff", "#99c3ee", "#1b5ea6"), ("#d9f4de", "#93d3a0", "#21733a"), ("#ffecc9", "#f0c179", "#9a5500")],
        link="#1b6fc2", outline="#10294d",
    ),
    "dark": dict(
        sky=("#0b1730", "#0e1d3b", "#122546", "#162d53"), star=("#9fb8dc", "#eaf3ff"),
        sea=("#3d7fc4", "#27609f", "#1c4b84", "#163c6c", "#112f57"), foam="#cfeaff", glint="#5f9fdc",
        title=("#ffffff", "#d9ecff", "#9fcaf4"), title_shadow="#1d4b85", ink2="#c3d1e4",
        muted="#8a9cb5", atmos="#2d5f9a",
        card="#0c1a31", card_bar="#132848", card_edge="#2a548a", card_shadow="#070f1d", grid="#35557f",
        plume=("#4a2a35", "#8e4140", "#e0674d"), mean="#ff7a5c", cool=("#1c3558", "#2d5c92", "#5d9fe4"), cool_mean="#7cc0ff", obs="#e7eef8", nina="#0f2140", nino="#24203a",
        chips=[("#15345a", "#2d6299", "#a3d3ff"), ("#143b2a", "#2c6f48", "#9be3ad"), ("#3a2c12", "#7d5c1f", "#ffcd7d")],
        link="#7cc0ff", outline="#081428",
    ),
}


def rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def shift(m, dx, dy):
    """Mask moved by (dx, dy) pixels, empty where it slides in."""
    out = np.zeros_like(m)
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = m[ys, xs]
    return out


def grow(m):
    """Mask plus its 4-neighbours; grow(m) & ~m is a one-pixel outline."""
    return m | shift(m, 1, 0) | shift(m, -1, 0) | shift(m, 0, 1) | shift(m, 0, -1)


class Canvas:
    def __init__(self):
        self.c = np.zeros((H, W, 4), np.uint8)

    def fill(self, mask, color):
        self.c[mask] = (*rgb(color), 255)

    def px(self, x, y, color):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < W and 0 <= y < H:
            self.c[y, x] = (*rgb(color), 255)

    def over(self, other):
        m = other.c[..., 3] > 0
        self.c[m] = other.c[m]

    @property
    def mask(self):
        return self.c[..., 3] > 0


def supersample(draw_fn, k=4, thresh=0.5):
    """Mask from shapes drawn by draw_fn(ImageDraw, k) on a k-times canvas, kept where coverage >= thresh."""
    im = Image.new("L", (W * k, H * k), 0)
    draw_fn(ImageDraw.Draw(im), k)
    a = np.asarray(im, float).reshape(H, k, W, k).mean(axis=(1, 3)) / 255
    return a >= thresh


def text_mask(s, font, x, y, anchor="la"):
    im = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(im)
    d.fontmode = "1"
    d.text((x, y), s, font=font, fill=1, anchor=anchor)
    return np.asarray(im, bool)


def text_width(s, font):
    return int(round(font.getlength(s)))


def line_pixels(points):
    """Pixel-perfect 1-px line through points: 8-connected, no L-shaped doubles."""
    pts = np.asarray(points, float)
    dense = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = int(max(abs(b - a)) * 3) + 1
        for t in np.linspace(0, 1, n, endpoint=False):
            dense.append(a + (b - a) * t)
    dense.append(pts[-1])
    out = []
    for p in dense:
        q = (int(math.floor(p[0] + 0.5)), int(math.floor(p[1] + 0.5)))
        if not out or q != out[-1]:
            out.append(q)
    clean = [out[0]]
    for i in range(1, len(out) - 1):
        a, b, c = clean[-1], out[i], out[i + 1]
        if abs(a[0] - c[0]) == 1 and abs(a[1] - c[1]) == 1:       # b is the corner of an L: drop it
            continue
        clean.append(b)
    if len(out) > 1:
        clean.append(out[-1])
    return clean


# ---------------------------------------------------------------- geometry
def catmull(ctrl, n):
    """n points evenly spaced along a Catmull-Rom spline through ctrl."""
    P = np.asarray(ctrl, float)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, 60, endpoint=False):
            out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    out = np.array(out + [P[-2]])
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(out, axis=0).T))]
    s = np.linspace(0, seg[-1], n)
    return np.c_[np.interp(s, seg, out[:, 0]), np.interp(s, seg, out[:, 1])]


# The head faces right on a raised S-neck; coordinates are pixels, x forward and y down from the neck joint.
SKULL = [(-3, -7.5), (2, -10), (7, -12.4), (11, -13.2), (14.5, -11.6), (19, -10.2), (25, -9.4), (29, -10.2),
         (31.5, -12.6), (34.5, -12.4), (37, -9.6), (37.6, -6.4), (35.6, -3.6), (28, -2.6), (20, -1.6), (15.5, -0.8),
         (14, 1.5), (12, 5), (6, 7.6), (-3, 7.6)]
JAW = [(15.5, -0.2), (21, 1), (27, 2), (32.6, 2.6), (32.2, 5.2), (27, 6.8), (18, 7.4), (10, 8), (3, 8.2), (-3, 7.6),
       (6, 3), (12, 1)]
MOUTH = [(15.5, -0.9), (35.4, -3.4), (32.4, 2.7), (21, 0.9)]
ANTLERS = [([(1.5, -10.5), (-4, -13.5), (-11, -15.5)], 1.6, 0),                 # far antler, in shade
           ([(-6, -14.2), (-8, -18.5)], 1, 0),
           ([(6.5, -11.8), (1, -16.5), (-6, -20.5), (-12.5, -22)], 2, 1),     # near antler with one tine
           ([(-2, -18.2), (-1, -22), (1.5, -25.5)], 1.2, 1)]
MANE = [((-3, -6.5), (-13, -8)), ((0, -8.5), (-11, -13.5)), ((-4, 0), (-13, -1.5)), ((-3, 5), (-11, 8))]
WHISKERS = [[(36.8, -4.6), (41, -1), (42, 4), (38, 8.5), (32, 10), (27, 12.5)],
            [(33.5, -12), (36, -16.5), (34, -21), (28, -23), (22, -21.5)]]

# layout of the mark, in banner pixels
CX, CY, GLOBE_R, K = 54, 75, 24, 1.0             # globe centre and radius; logo.py units -> pixels for the coil


def spine():
    """The dragon's centre line: logo.py's coil round the Earth, with the neck raised so the head faces right."""
    R = 38
    ang = np.radians([66, 100, 140, 180, 220, 262, 300, 328])
    ctrl = [(27, -45), (17, -43.5)] + [(R * np.cos(a), -R * np.sin(a)) for a in ang]
    pts = catmull(np.asarray(ctrl) * K + (CX, CY), 1200)
    tan = np.gradient(pts, axis=0)
    tan /= np.hypot(*tan.T)[:, None]
    nor = np.c_[-tan[:, 1], tan[:, 0]]
    flip = np.sign(np.einsum("ij,ij->i", nor, pts - (CX, CY)))
    nor *= flip[:, None]                                     # outward, away from the globe
    u = np.linspace(0, 1, len(pts))
    half = K * (0.5 + 4.4 * (1 - u ** 1.35) + 1.6 * np.exp(-u / 0.05))
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    return pts, tan, nor, u, half, s


# ---------------------------------------------------------------- scene
def backdrop(t):
    """Sky that fades into the page through ordered dither, so the banner has no edge at all."""
    cv = Canvas()
    R0, F = 44, 26
    qx = np.abs(XX + 0.5 - W / 2) - (W / 2 - R0)
    qy = np.abs(YY + 0.5 - H / 2) - (H / 2 - R0)
    sdf = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - R0
    alpha = np.clip(-sdf / F, 0, 1) ** 0.9
    vis = alpha > DITHER
    level = np.clip(YY / 118, 0, 1) * (len(t["sky"]) - 1) + (DITHER - 0.5) * 0.9
    idx = np.clip(np.round(level), 0, len(t["sky"]) - 1).astype(int)
    for i, col in enumerate(t["sky"]):
        cv.fill(vis & (idx == i), col)
    return cv, alpha


def sky_details(cv, t, alpha, theme):
    rng = np.random.default_rng(11)
    if theme == "dark":
        for _ in range(150):
            x, y = rng.integers(0, W), rng.integers(0, 120)
            if alpha[y, x] > 0.3:
                cv.px(x, y, t["star"][0] if rng.random() < 0.75 else t["star"][1])
        for x, y, big in [(222, 7, 1), (124, 112, 0), (474, 70, 0), (16, 22, 1), (330, 112, 0), (120, 4, 0)]:
            sparkle(cv, x, y, t["star"][1], big)
    else:
        for x, y, big in [(222, 7, 1), (16, 22, 1), (124, 112, 0), (474, 70, 0)]:
            sparkle(cv, x, y, "#ffffff", big)
        for x, y in [(301, 109), (311, 105)]:                # two gulls over the buoy
            sprite(cv, x, y, [".o.o.", "o.o.o"], {"o": t["muted"]})


def sparkle(cv, x, y, col, big=False):
    arm = 2 if big else 1
    for i in range(-arm, arm + 1):
        cv.px(x + i, y, col)
        cv.px(x, y + i, col)


def sprite(cv, x, y, rows, pal):
    for j, row in enumerate(rows):
        for i, ch in enumerate(row):
            if ch in pal:
                cv.px(x + i, y + j, pal[ch])


def ocean(cv, t):
    top = 124
    x = np.arange(W)
    crest = top + np.round(1.1 * np.sin(2 * np.pi * x / 40 + 0.6) + 0.6 * np.sin(2 * np.pi * x / 17 + 2)).astype(int)
    fade = np.minimum(np.clip(np.minimum(XX + 0.5, W - XX - 0.5) / 56, 0, 1), np.clip((H - YY - 0.5) / 7, 0, 1))
    vis = (fade ** 0.8 > DITHER) & (YY >= crest[None, :])
    depth = YY - crest[None, :]
    swell = np.round(0.9 * np.sin(2 * np.pi * XX / 23 + YY * 0.55)).astype(int)
    level = np.select([depth < 2, depth + swell < 6, depth + swell < 11, depth + swell < 16], [0, 1, 2, 3], 4)
    for i, col in enumerate(t["sea"]):
        cv.fill(vis & (level == i), col)
    # foam where the crest is highest, glints on the swell
    for xx in range(W):
        y = crest[xx]
        if y == crest.min() and fade[y, xx] > 0.35:
            cv.px(xx, y, t["foam"])
            if 0 < xx < W - 1 and crest[xx - 1] > y and fade[y, xx] > 0.6:
                cv.px(xx - 1, y + 1, t["foam"])
    rng = np.random.default_rng(5)
    for _ in range(60):
        gx, gy = int(rng.integers(0, W - 4)), int(rng.integers(top + 4, H - 3))
        if fade[gy, gx] > 0.75:
            for j in range(int(rng.integers(2, 5))):
                if level[gy, gx + j] >= 1:
                    cv.px(gx + j, gy, t["glint"])
    return crest


def buoy(cv, x, crest):
    """A TAO-style moored buoy, the kind that watches ENSO from the equatorial Pacific."""
    y = crest[x] - 1
    rows = ["..o..", "..o..", ".ooo.", ".oYo.", "ooooo", "oRRRo", "oWWWo", "oRRRo"]
    pal = {"o": SPRITE["outline"], "Y": "#ffd34d", "R": "#e8553b", "W": "#ffffff"}
    sprite(cv, x - 2, y - len(rows) + 2, rows, pal)
    cv.px(x, y - len(rows) + 1, "#ffd34d")


def satellite(cv, x, y):
    rows = ["bbb.....bbb", "bBb.ooo.bBb", "bbbooWoobbb", "bBb.ooo.bBb", "bbb.....bbb"]
    sprite(cv, x, y, rows, {"b": "#2a5fa5", "B": "#7fb8ef", "o": SPRITE["outline"], "W": "#ffd34d"})


# ---------------------------------------------------------------- the mark: Earth and dragon
def globe(t):
    cv = Canvas()
    land_bits = [ln.strip() for ln in (HERE / "land_1deg.txt").read_text().splitlines() if ln and ln[0] != "#"]
    land = np.unpackbits(np.array([bytearray.fromhex(r) for r in land_bits]), axis=1)[:, :360].astype(bool)
    lon0, lat0 = math.radians(-138), math.radians(14)
    L = np.array([-0.55, -0.62, 0.56])
    L /= np.linalg.norm(L)
    r = GLOBE_R
    for y in range(CY - r - 1, CY + r + 2):
        for x in range(CX - r - 1, CX + r + 2):
            sub = [(x + 0.5 + a - CX) / r for a in (-1 / 3, 0, 1 / 3)]
            suby = [(y + 0.5 + b - CY) / r for b in (-1 / 3, 0, 1 / 3)]
            land_n, cov, lat_c, lon_c = 0, 0, 0, 0
            for px in sub:
                for py in suby:
                    rho = math.hypot(px, py)
                    if rho >= 1:
                        continue
                    cov += 1
                    c = math.asin(rho)
                    yy = -py
                    lat = math.asin(math.cos(c) * math.sin(lat0) + (yy * math.sin(c) * math.cos(lat0) / rho if rho else 0))
                    lon = lon0 + math.atan2(px * math.sin(c), rho * math.cos(lat0) * math.cos(c) - yy * math.sin(lat0) * math.sin(c))
                    la, lo = math.degrees(lat), (math.degrees(lon) + 180) % 360 - 180
                    i, j = min(179, int(90 - la)), int(lo + 180) % 360
                    land_n += land[i, j]
                    lat_c += la
                    lon_c += lo if abs(lo - math.degrees(lon0)) < 180 else lo
            if cov < 5:
                continue
            nx, ny = (x + 0.5 - CX) / r, (y + 0.5 - CY) / r
            nz = math.sqrt(max(0, 1 - nx * nx - ny * ny))
            light = nx * L[0] + ny * L[1] + nz * L[2] + (DITHER[y, x] - 0.5) * 0.18
            tone = 0 if light < 0.12 else 1 if light < 0.5 else 2 if light < 0.84 else 3
            la, lo = lat_c / cov, ((lon_c / cov) + 360) % 360
            if abs(la) > 68 or (la < -62):
                col = SPRITE["ice"][0 if tone < 2 else 1]
            elif land_n * 2 >= cov:
                col = SPRITE["land"][tone]
            elif 192 < lo < 282 and abs(la) < 1.5 + 9 * (lo - 192) / 90:   # El Nino: a warm tongue
                heat = (1 - abs(la) / (1.5 + 9 * (lo - 192) / 90)) * min(1, (lo - 192) / 45) + (DITHER[y, x] - 0.5) * 0.3
                if heat < 0.18:
                    col = SPRITE["ocean"][tone]
                else:
                    k = 0 if heat > 0.7 else 1 if heat > 0.48 else 2 if heat > 0.3 else 3
                    col = SPRITE["warm"][min(3, k + (tone == 0))]
            else:
                col = SPRITE["ocean"][tone]
            cv.px(x, y, col)
    disc = cv.mask
    # atmosphere: a dithered glow ring that thins outward
    d = np.hypot(XX + 0.5 - CX, YY + 0.5 - CY) - r
    glow = (~disc) & (d < 5) & (np.clip(1 - d / 5, 0, 1) ** 1.5 > DITHER)
    cv.fill(glow, t["atmos"])
    edge = grow(disc) & ~disc
    cv.fill(edge & ((XX - CX) + (YY - CY) > -4), SPRITE["outline"])
    hl = [(CX - 11, CY - 15), (CX - 10, CY - 15), (CX - 12, CY - 14), (CX - 11, CY - 14), (CX - 12, CY - 13)]
    for x, y in hl:
        cv.px(x, y, "#ffffff")
    return cv


def dragon(t):
    pts, tan, nor, u, half, s = spine()
    out = Canvas()
    # fins first, so the body overlaps their roots
    fins = Canvas()
    fin_mask = np.zeros((H, W), bool)
    for ss in np.arange(5, s[-1] * 0.86, 7.2):
        i = int(np.searchsorted(s, ss))
        uu = u[i]
        h = 2.2 + 5.2 * (1 - uu) ** 1.3
        base = pts[i] + nor[i] * half[i] * 0.55
        a = base - tan[i] * h * 0.55
        b = base + tan[i] * h * 0.55
        tip = pts[i] + nor[i] * (half[i] + h) + tan[i] * h * 0.8
        tri = supersample(lambda d, k, a=a, b=b, tip=tip: d.polygon([tuple(a * k), tuple(tip * k), tuple(b * k)], fill=255),
                          thresh=0.42)
        fin_mask |= tri
    # tail flame
    i = len(pts) - 1
    for rot, ln in [(-0.5, 9), (0.05, 11), (0.6, 8)]:
        c, sn = math.cos(rot), math.sin(rot)
        d = np.array([tan[i][0] * c - tan[i][1] * sn, tan[i][0] * sn + tan[i][1] * c])
        p0 = pts[i] - tan[i] * 1.5
        perp = np.array([-d[1], d[0]])
        poly = [p0 + perp * 2.2, p0 + d * ln, p0 - perp * 2.2]
        fin_mask |= supersample(lambda dr, k, poly=poly: dr.polygon([tuple(p * k) for p in poly], fill=255), thresh=0.4)
    fins.fill(fin_mask, SPRITE["fin"][1])
    fins.fill(fin_mask & ~shift(fin_mask, 1, 1), SPRITE["fin"][2])
    fins.fill(fin_mask & ~shift(fin_mask, -1, -1), SPRITE["fin"][0])
    fins.fill(grow(fin_mask) & ~fin_mask, SPRITE["outline"])
    out.over(fins)

    # legs reach in toward the Earth, three golden claws each
    legs = Canvas()
    for uu, reach in [(0.22, 6.5), (0.73, 5.5)]:
        i = int(uu * (len(pts) - 1))
        n, tg, hf = nor[i], tan[i], half[i]
        joints = [pts[i] - n * hf * 0.2 + tg * 1.5, pts[i] - n * (hf + 2.2) + tg * 3.2,
                  pts[i] - n * (hf + reach) - tg * 0.5]
        limb = supersample(lambda d, k, j=joints: d.line([tuple(p * k) for p in j], fill=255, width=int(3.4 * k),
                                                         joint="curve"), thresh=0.45)
        legs.fill(limb, SPRITE["body"][1])
        legs.fill(limb & ~shift(limb, 1, 1), SPRITE["body"][2])
        legs.fill(limb & ~shift(limb, -1, -1), SPRITE["body"][0])
        lm = limb.copy()
        wrist = joints[-1]
        claws = np.zeros((H, W), bool)
        for rot in (-0.75, 0, 0.75):
            cr, sr = math.cos(rot), math.sin(rot)
            d = -np.array([n[0] * cr - n[1] * sr, n[0] * sr + n[1] * cr])
            for x, y in line_pixels([wrist + d * 1.2, wrist + d * 3.6]):
                if 0 <= x < W and 0 <= y < H and not lm[y, x]:
                    claws[y, x] = True
        legs.fill(claws, SPRITE["horn"][1])
        allm = lm | claws
        legs.fill(grow(allm) & ~allm & ~legs.mask, SPRITE["outline"])
    out.over(legs)

    # body: nearest spine sample per pixel, shaded as a lit tube with a cream belly facing the globe
    body = Canvas()
    x0, x1 = int(pts[:, 0].min() - 9), int(pts[:, 0].max() + 9)
    y0, y1 = int(pts[:, 1].min() - 9), int(pts[:, 1].max() + 9)
    gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    P = np.c_[gx.ravel(), gy.ravel()]
    best_d = np.full(len(P), 1e9)
    best_i = np.zeros(len(P), int)
    for j in range(0, len(pts), 60):
        dd = np.hypot(P[:, None, 0] - pts[None, j:j + 60, 0], P[:, None, 1] - pts[None, j:j + 60, 1])
        k = dd.argmin(1)
        v = dd[np.arange(len(P)), k]
        better = v < best_d
        best_d[better] = v[better]
        best_i[better] = j + k[better]
    inside = best_d <= half[best_i] + 0.15
    L = np.array([-0.5, -0.78, 0.62])
    L /= np.linalg.norm(L)
    vx = (P[:, 0] - pts[best_i, 0]) / half[best_i]
    vy = (P[:, 1] - pts[best_i, 1]) / half[best_i]
    vz = np.sqrt(np.clip(1 - vx ** 2 - vy ** 2, 0, 1))
    light = vx * L[0] + vy * L[1] + vz * L[2]
    side = vx * nor[best_i, 0] + vy * nor[best_i, 1]
    belly = (side < -0.32) & (half[best_i] > 1.6)
    seg = (s[best_i] % 3.4) < 1.0
    for idx in np.nonzero(inside)[0]:
        X, Y = int(P[idx, 0]), int(P[idx, 1])
        if not (0 <= X < W and 0 <= Y < H):
            continue
        li = light[idx] + (DITHER[Y, X] - 0.5) * 0.1
        if belly[idx]:
            tone = 0 if (seg[idx] or li < 0.05) else 1 if li < 0.55 else 2
            body.px(X, Y, SPRITE["belly"][tone])
        else:
            tone = 0 if li < 0.18 else 1 if li < 0.6 else 2 if li < 0.88 else 3
            # scales: a staggered grid of darker ticks on the back
            row = int((side[idx] + 1) * half[best_i[idx]] / 2.3)
            a = s[best_i[idx]] / 3.4 + 0.5 * (row % 2)
            if half[best_i[idx]] > 2.2 and (a % 1) < 0.3 and tone > 0:
                tone -= 1
            body.px(X, Y, SPRITE["body"][tone])
    bm = body.mask
    body.fill(grow(bm) & ~bm, SPRITE["outline"])
    out.over(body)

    # head, on top of the neck joint, tilted with it
    angle = math.atan2(-tan[0][1], -tan[0][0])
    c, sn = math.cos(angle), math.sin(angle)
    to_px = lambda q: np.asarray(q, float).reshape(-1, 2) @ np.array([[c, sn], [-sn, c]]) + pts[0]
    poly = lambda q, th=0.45: supersample(lambda d, k: d.polygon([tuple(p * k) for p in to_px(q)], fill=255), thresh=th)

    def stroke(points, width, th=0.4):
        pp = catmull(to_px(points), 80) if len(points) > 2 else to_px(points)
        return supersample(lambda d, k: d.line([tuple(p * k) for p in pp], fill=255, width=int(width * k),
                                               joint="curve"), thresh=th)

    def shaded(cv, m, ramp, outline=True):
        cv.fill(m, ramp[1])
        cv.fill(m & ~shift(m, 1, 1), ramp[2])
        cv.fill(m & ~shift(m, -1, -1), ramp[0])
        if outline:
            cv.fill(grow(m) & ~m, SPRITE["outline"])

    head = Canvas()
    mane = np.zeros((H, W), bool)
    for base, tip in MANE:
        base, tip = np.array(base, float), np.array(tip, float)
        d = (tip - base) / np.hypot(*(tip - base))
        perp = np.array([-d[1], d[0]])
        mane |= poly([base + perp * 2.4, tip, base - perp * 2.4], 0.4)
    shaded(head, mane, SPRITE["fin"])
    for layer in (0, 1):
        horns = np.zeros((H, W), bool)
        for points, w, near in ANTLERS:
            if near == layer:
                horns |= stroke(points, w + 0.6)
        shaded(head, horns, SPRITE["horn"] if layer else ("#9c6f2a", SPRITE["horn"][0], SPRITE["horn"][1]))

    skull, jaw, mouth = poly(SKULL), poly(JAW), poly(MOUTH, 0.5)
    jaw &= ~skull
    mouth &= ~(skull | jaw)
    face = Canvas()
    face.fill(skull, SPRITE["body"][1])
    face.fill(skull & ~shift(skull, 1, 1), SPRITE["body"][2])
    face.fill(skull & ~shift(skull, 0, 1) & ~shift(skull, 1, 1), SPRITE["body"][3])
    face.fill(skull & ~shift(skull, 0, -1), SPRITE["body"][0])
    face.fill(jaw, SPRITE["belly"][1])
    face.fill(jaw & ~shift(jaw, 1, 1), SPRITE["belly"][2])
    face.fill(jaw & ~shift(jaw, 0, -1), SPRITE["belly"][0])
    face.fill(mouth, SPRITE["mouth"])
    whole = skull | jaw | mouth
    face.fill(grow(whole) & ~whole, SPRITE["outline"])
    # brow ridge, eye, nostril, cheek scales and fangs, placed in head coordinates
    put = lambda q, col: face.px(*to_px([q])[0], col)
    for q in [(7.5, -10), (8.5, -10.6), (9.5, -11), (10.5, -11), (11.5, -11), (12.5, -10.6), (13.5, -10)]:
        put(q, SPRITE["body"][3])
    for q in [(8.5, -9), (9.5, -9.4), (10.5, -9.4), (11.5, -9.4), (12.5, -9.2), (13.5, -8.6)]:
        put(q, SPRITE["outline"])
    for q in [(9.5, -8), (10.5, -8), (11.5, -8), (12.5, -8), (9.5, -7), (10.5, -7), (11.5, -7)]:
        put(q, SPRITE["eye"])
    for q in [(11.5, -7), (12.5, -7), (12.5, -8.2)]:
        put(q, SPRITE["outline"])
    put((9.5, -8), "#ffffff")
    for q in [(32.5, -9.6), (33.5, -9.6), (33.5, -8.6)]:
        put(q, SPRITE["outline"])
    for q in [(3.5, -3), (5.5, -1), (3.5, 1), (6.5, -4.5), (1.5, -1.5)]:
        put(q, SPRITE["body"][0])
    for q in [(34.5, -2.4), (34.5, -1.4), (29.5, -2), (25.5, -1.4), (21.5, -0.6), (30.5, 1.6), (30.5, 0.6), (25.5, 1)]:
        put(q, SPRITE["tooth"])
    head.over(face)
    for points in WHISKERS:
        wm = np.zeros((H, W), bool)
        for x, y in line_pixels(catmull(to_px(points), 120)):
            if 0 <= x < W and 0 <= y < H:
                wm[y, x] = True
        head.fill(wm & ~whole, SPRITE["whisker"])
    out.over(head)
    return out


# ---------------------------------------------------------------- the words
def title_block(cv, t, x, y):
    f = FONT["title"]
    s = "CAS-Canglong"
    bb = ImageDraw.Draw(Image.new("1", (1, 1))).textbbox((0, 0), s, font=f)
    m = text_mask(s, f, x - bb[0], y - bb[1])
    cv.fill(shift(m, 2, 2) & ~m, t["title_shadow"])
    top, bottom = y, y + (bb[3] - bb[1])
    band = (YY - top) / max(1, bottom - top)
    cv.fill(m & (band < 0.34), t["title"][0])
    cv.fill(m & (band >= 0.34) & (band < 0.66), t["title"][1])
    cv.fill(m & (band >= 0.66), t["title"][2])
    return x + (bb[2] - bb[0]), bottom


def seal(cv, x, y):
    """A red seal stamp: 中科 / 苍龙."""
    m = np.zeros((H, W), bool)
    m[y:y + 30, x:x + 30] = True
    for cx, cy in [(x, y), (x + 29, y), (x, y + 29), (x + 29, y + 29)]:
        m[cy, cx] = False
    rng = np.random.default_rng(3)                        # a little wear on the stamp's edge
    for _ in range(16):
        side = rng.integers(0, 4)
        p = int(rng.integers(2, 28))
        cx, cy = [(x + p, y), (x + p, y + 29), (x, y + p), (x + 29, y + p)][side]
        m[cy, cx] = False
    cv.fill(m, SPRITE["seal"])
    f = FONT["text"]
    ink = np.zeros((H, W), bool)
    for ch, gx, gy in [("中", 2, 2), ("科", 16, 2), ("苍", 2, 16), ("龙", 16, 16)]:
        bb = ImageDraw.Draw(Image.new("1", (1, 1))).textbbox((0, 0), ch, font=f)
        ink |= text_mask(ch, f, x + gx - bb[0] + (12 - (bb[2] - bb[0])) // 2, y + gy - bb[1] + (12 - (bb[3] - bb[1])) // 2)
    cv.fill(ink & m, SPRITE["seal_ink"])


def chip(cv, x, y, label, cols):
    fill, edge, ink = cols
    w = text_width(label, FONT["small"]) + 9
    h = 11
    m = np.zeros((H, W), bool)
    m[y:y + h, x:x + w] = True
    for cx, cy in [(x, y), (x + w - 1, y), (x, y + h - 1), (x + w - 1, y + h - 1)]:
        m[cy, cx] = False
    cv.fill(m, fill)
    cv.fill(m & ~shift(m, 0, -1), edge)                  # bevel: bottom row darker
    cv.fill(m & ~shift(m, -1, 0) & (YY > y), edge)
    cv.fill(text_mask(label, FONT["small"], x + 5, y + 8, anchor="ls") & m, ink)
    return x + w


def plume_card(cv, t, x, y, w, h):
    m = np.zeros((H, W), bool)
    m[y:y + h, x:x + w] = True
    for cx, cy in [(x, y), (x + w - 1, y), (x, y + h - 1), (x + w - 1, y + h - 1)]:
        m[cy, cx] = False
    cv.fill(shift(m, 3, 3) & ~m, t["card_shadow"])
    cv.fill(m, t["card"])
    bar = m & (YY < y + 11)
    cv.fill(bar, t["card_bar"])
    edge = grow(m) & ~m
    cv.fill(edge & ~shift(m, 3, 3) | (edge & (YY < y + 1)) | (edge & (XX < x)), t["card_edge"])
    for i, col in enumerate(["#ff6b5e", "#ffc23f", "#43c463"]):
        for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            cv.px(x + 5 + i * 5 + dx, y + 4 + dy, col)
    cv.fill(text_mask("NIÑO 3.4 OUTLOOK", FONT["small"], x + 22, y + 8, anchor="ls"), t["muted"])
    # plot
    px0, px1, py0, py1 = x + 8, x + w - 8, y + 18, y + h - 12
    zero = py0 + int((py1 - py0) * 0.55)
    cv.fill((YY >= py0) & (YY < zero) & (XX >= px0) & (XX <= px1), t["nino"])
    cv.fill((YY > zero) & (YY <= py1) & (XX >= px0) & (XX <= px1), t["nina"])
    for xx in range(px0, px1 + 1, 3):
        cv.px(xx, zero, t["grid"])
    months = np.arange(-6, 17)
    xs = px0 + (months + 6) / 22 * (px1 - px0)
    scale = (zero - py0) / 4.0
    obs = np.array([0.9, 1.3, 1.8, 2.3, 2.7, 2.9, 3.0])
    rng = np.random.default_rng(7)
    tt = np.arange(0, 17)
    # stylised on the README outlook: El Nino peaking at lead 3, neutral by lead 8-9, La Nina deepest near lead 14
    mean = np.interp(tt, [0, 1, 2, 3, 5, 7, 8.5, 11, 14, 16], [3.0, 3.06, 3.12, 3.24, 2.6, 1.0, 0.0, -1.4, -2.17, -1.8])
    density = np.zeros((H, W), int)
    for _ in range(32):
        walk = np.cumsum(rng.normal(0, 1, len(tt)))
        walk -= walk[0]
        member = mean + walk * (0.1 + 0.035 * tt) / np.sqrt(1 + tt * 0.25)
        member[0] = obs[-1]
        ys = zero - member * scale
        seen = set()
        for X, Y in line_pixels(np.c_[xs[6:], ys]):
            if (X, Y) not in seen and py0 <= Y <= py1:
                density[Y, X] += 1
                seen.add((X, Y))
    for lvl, warm, cool in zip([1, 3, 7], t["plume"], t["cool"]):   # warm above zero, cool below
        cv.fill((density >= lvl) & (YY <= zero), warm)
        cv.fill((density >= lvl) & (YY > zero), cool)
    for X, Y in line_pixels(np.c_[xs[6:], zero - mean * scale]):
        col = t["mean"] if Y < zero else t["cool_mean"]
        cv.px(X, Y, col)
        cv.px(X, Y + 1, col)
    for X, Y in line_pixels(np.c_[xs[:7], zero - obs * scale]):
        cv.px(X, Y, t["obs"])
        cv.px(X, Y + 1, t["obs"])
    for X, Y in [(xs[6], zero - obs[-1] * scale)]:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1, 2):
                cv.px(X + dx, Y + dy, t["obs"])
    # axis captions
    cv.fill(text_mask("NOW", FONT["small"], int(xs[6]) - 7, py1 + 9, anchor="ls"), t["muted"])
    cv.fill(text_mask("16 MO", FONT["small"], px1 + 1, py1 + 9, anchor="rs"), t["muted"])


def banner(theme):
    t = THEMES[theme]
    SPRITE["outline"] = t["outline"]                     # sprites keep their colours; only the outline follows the sky
    cv, alpha = backdrop(t)
    sky_details(cv, t, alpha, theme)
    crest = ocean(cv, t)
    buoy(cv, 296, crest)
    satellite(cv, 318, 2)
    cv.over(globe(t))
    cv.over(dragon(t))

    tx = 132
    right, bottom = title_block(cv, t, tx, 13)
    seal(cv, right + 8, 9)
    for i, s in enumerate(["AI forecasts of global sea-surface", "temperature and ENSO, from one",
                           "month to 16 months ahead."]):
        cv.fill(text_mask(s, FONT["text"], tx, 44 + i * 13), t["ink2"])
    x = tx
    for label, cols in zip(["0.25° SST", "16-MO ENSO", "32 MEMBERS"], t["chips"]):
        x = chip(cv, x, 89, label, cols) + 4
    fill, edge, ink = t["chips"][2]
    sprite(cv, tx, 104, ["oooooo.", "oWWWWoo", "oWbbWWo", "oWWWWWo", "oWbbbWo", "oWWWWWo", "oWbbbWo", "oWWWWWo",
                         "ooooooo"], {"o": ink, "W": fill, "b": edge})
    x = tx + 11
    for part, col in [("Advanced Science", ink), (", 2026", t["ink2"])]:
        cv.fill(text_mask(part, FONT["text"], x, 101), col)
        x += text_width(part, FONT["text"])

    plume_card(cv, t, 342, 12, 128, 86)
    s = "live at canglong.earth"
    lw = text_width(s, FONT["text"])
    x0 = 470 - lw
    cv.fill(text_mask(s, FONT["text"], x0, 101), t["link"])
    sprite(cv, x0 - 8, 104, ["o...", "oo..", "ooo.", "oooo", "ooo.", "oo..", "o..."], {"o": t["link"]})
    return cv.c


def to_svg(c, label):
    """One <path> per colour, each pixel run as a 1-high rectangle; crisp at any size."""
    runs = {}
    for y in range(H):
        row = c[y]
        x = 0
        while x < W:
            if row[x, 3] == 0:
                x += 1
                continue
            col = "#%02x%02x%02x" % tuple(row[x, :3])
            x0 = x
            while x < W and row[x, 3] and "#%02x%02x%02x" % tuple(row[x, :3]) == col:
                x += 1
            runs.setdefault(col, []).append(f"M{x0} {y}h{x - x0}v1h{x0 - x}z")
    body = "\n".join(f'<path fill="{col}" d="{"".join(r)}"/>' for col, r in runs.items())
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W * SCALE}" height="{H * SCALE}" '
            f'shape-rendering="crispEdges" role="img" aria-label="{label}">\n{body}\n</svg>\n')


if __name__ == "__main__":
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    label = ("CAS-Canglong (中科苍龙): AI forecasts of global sea-surface temperature and ENSO, from one month to 16 "
             "months ahead")
    for theme in THEMES:
        c = banner(theme)
        name = "banner" if theme == "light" else "banner_dark"
        (dest / f"{name}.svg").write_text(to_svg(c, label), encoding="utf-8", newline="\n")
        if "-p" in sys.argv:
            Image.fromarray(c, "RGBA").save(dest / f"{name}_1x.png")
    print("ok")
