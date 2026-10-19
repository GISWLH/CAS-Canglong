"""CAS-Canglong brand artwork: logo mark and README banner, light and dark, as self-contained SVG.

    python assets/readme/src/brand.py assets/readme      # -> banner{,_dark}.svg, logo{,_dark}.svg

The banner draws the Azure Dragon (苍龙七宿) on a star chart: the dragon's spine runs through the real stars of the
seven eastern mansions, with Antares (心宿二) as its heart. Text is converted to outlines with HarfBuzz, so the SVGs need
no web fonts. Needs numpy, fonttools and uharfbuzz, plus the OFL fonts listed in typeset.py.
"""
import math
import re
import sys
from pathlib import Path

import numpy as np

import stars
from typeset import text_path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
THEMES = {
    "light": dict(bg="#F7F6F1", border="#E6E3D9", ink="#0F1F1C", ink2="#4A5A56", muted="#8A9691", line="#1D5A5E",
                  star="#1D5A5E", faint="#C9D4CF", grid="#E4E6DE", warm="#E0603C", halo="#F7F6F1"),
    "dark": dict(bg="#0A1210", border="#1B2723", ink="#EAF2EC", ink2="#A9B8B1", muted="#6F817A", line="#B9E3C6",
                 star="#E2F4E7", faint="#2A3B36", grid="#16231F", warm="#FF7B54", halo="#0A1210"),
}


def f(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def smooth_path(pts, closed=False):
    """Catmull-Rom through pts -> cubic Bezier path data."""
    pts = np.asarray(pts, float)
    n = len(pts)
    if closed:
        P = np.vstack([pts[-1], pts, pts[0], pts[1]])
    else:
        P = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    d = [f"M{f(P[1][0])},{f(P[1][1])}"]
    for i in range(1, n + (1 if closed else 0)):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        c1, c2 = p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6
        d.append(f"C{f(c1[0])},{f(c1[1])} {f(c2[0])},{f(c2[1])} {f(p2[0])},{f(p2[1])}")
    return " ".join(d) + (" Z" if closed else "")


def sample_spline(ctrl, n=400):
    """Dense Catmull-Rom samples along control points, re-parameterised by arc length."""
    P = np.asarray(ctrl, float)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, 40, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    out = np.array(out)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(out, axis=0).T))]
    s = np.linspace(0, seg[-1], n)
    return np.c_[np.interp(s, seg, out[:, 0]), np.interp(s, seg, out[:, 1])], seg[-1]


def frames(pts):
    tan = np.gradient(pts, axis=0)
    tan /= np.hypot(*tan.T)[:, None]
    nor = np.c_[-tan[:, 1], tan[:, 0]]
    return tan, nor


def xf(points, origin, angle, scale=1.0):
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]]) * scale
    return np.asarray(points, float) @ R.T + origin


def path_xf(d, origin, angle, scale=1.0):
    """Transform absolute-coordinate path data (M/C/L/Q/Z only)."""
    out = []
    toks = re.findall(r"[MCLQZ]|-?\d*\.?\d+", d)
    cmd = None
    buf = []
    for t in toks + ["M"]:
        if t in "MCLQZ":
            if cmd:
                pts = np.array(buf, float).reshape(-1, 2) if buf else np.zeros((0, 2))
                pts = xf(pts, origin, angle, scale) if len(pts) else pts
                out.append(cmd + " ".join(f"{f(x)},{f(y)}" for x, y in pts))
            cmd, buf = t, []
        else:
            buf.append(t)
    return " ".join(out[:-1] if out and out[-1] == "M" else out)


def _coords(d):
    return np.array([float(v) for v in re.findall(r"-?\d*\.?\d+", d)]).reshape(-1, 2)


# --------------------------------------------------------------------------- dragon head (local frame, facing +x)
HEAD_LINES = [
    # skull, brow and snout ridge, ending in an upturned nose
    "M-2,-11 C8,-16 17,-21 28,-19 C36,-18 42,-13 51,-12 C57,-11.5 61,-13.5 62,-18",
    # nose tip down to the upper lip and back to the mouth corner
    "M62,-18 C66,-13 65,-7 59,-5.5 C51,-4 41,-3.5 32,-1",
    # lower lip forward to the chin, chin back along the jaw to the throat
    "M32,-1 C40,0.5 48,1.5 55,1.5 C53,6 46,8 38,9 C26,11 10,12.5 -2,11",
    # brow arch
    "M14,-15.5 C20,-22.5 29,-22.5 35,-16.5",
    # antler (main beam sweeping back) and its tines
    "M9,-17 C0,-26 -14,-31 -34,-31",
    "M-8,-28.5 C-10,-35 -9,-41 -4,-46",
    "M-22,-31 C-26,-36 -27,-41 -25,-46",
    # far antler
    "M17,-19.5 C10,-31 -2,-39 -20,-42",
    "M2,-35 C2,-41 5,-45 10,-48",
    # whisker streaming back from the nose
    "M58,-8 C68,0 64,14 50,18 C38,21 30,28 28,40",
    # beard tuft under the jaw
    "M22,11 C21,18 16,23 9,27",
    # mane strands from the nape
    "M-4,-11 C-14,-19 -26,-20 -38,-15",
    "M-6,-4 C-17,-9 -28,-7 -36,-1",
]
HEAD_FILL = ("M-2,-11 C8,-16 17,-21 28,-19 C36,-18 42,-13 51,-12 C57,-11.5 61,-13.5 62,-18 C66,-13 65,-7 59,-5.5 "
             "C51,-4 41,-3.5 32,-1 C40,0.5 48,1.5 55,1.5 C53,6 46,8 38,9 C26,11 10,12.5 -2,11 Z")
EYE = "M20,-11.5 Q25.5,-15 31,-11.5 Q25.5,-8.5 20,-11.5 Z"
EYE_DOT = (26.5, -11.6)


def dragon(t, S, C):
    """Line-art dragon whose spine runs through the Azure Dragon stars. Returns SVG elements."""
    def P(x, y):
        return np.array([C[0] + S * x, C[1] - S * y])

    spine_sky = [(24.6, 11.2), (19.5, 15.0), (13.5, 15.6), (8.0, 11.5), (3.4, 5.0), (-1.5, -1.0), (-7.9, -4.7),
                 (-11.5, -9.8), (-11.8, -16.5), (-12.6, -21.3), (-15.4, -23.6), (-19.6, -24.3), (-22.6, -21.6),
                 (-21.9, -18.7), (-19.9, -17.2)]
    ctrl = [P(*stars._rot(*p)) for p in spine_sky]
    pts, length = sample_spline(ctrl, 600)
    tan, nor = frames(pts)
    u = np.linspace(0, 1, len(pts))
    w = S * (1.22 + 0.25 * np.sin(np.clip(u / 0.3, 0, 1) * math.pi / 2) - 1.32 * u ** 1.6)
    w = np.maximum(w, 0.6)
    # dorsal side = the side the head's crown faces at the neck
    head_dir = -tan[0]
    angle = math.atan2(head_dir[1], head_dir[0])
    crown = xf([(0, -1)], (0, 0), angle)[0]
    sgn = 1 if np.dot(nor[0], crown) > 0 else -1
    dors = pts + sgn * nor * w[:, None]
    vent = pts - sgn * nor * w[:, None]
    belly = pts - sgn * nor * (w * 0.32)[:, None]
    el = []
    stroke = f'stroke="{t["line"]}" fill="none" stroke-linecap="round" stroke-linejoin="round"'
    # ribbon edges closing in a fine tail point
    tail = np.vstack([dors[::12], pts[-1:] + tan[-1:] * 7, vent[::-12]])
    el.append(f'<path d="{smooth_path(tail)} Z" fill="{t["bg"]}" stroke="none"/>')
    el.append(f'<path d="{smooth_path(tail)}" {stroke} stroke-width="2.4"/>')
    el.append(f'<path d="{smooth_path(belly[4:-60:10])}" {stroke} stroke-width="1.2" opacity=".55"/>')
    # belly scutes between the ventral edge and the belly line
    for i in range(10, len(pts) - 70, 9):
        a, b = vent[i], belly[i]
        el.append(f'<path d="M{f(a[0])},{f(a[1])} L{f(b[0])},{f(b[1])}" {stroke} stroke-width="1.1" opacity=".45"/>')
    # dorsal fin: small swept-back spikes
    for i in range(16, len(pts) - 50, 17):
        h = w[i] * 0.62
        a, b = dors[i - 5], dors[i + 3]
        tip = dors[i] + sgn * nor[i] * h + tan[i] * h * 0.9
        el.append(f'<path d="M{f(a[0])},{f(a[1])} L{f(tip[0])},{f(tip[1])} L{f(b[0])},{f(b[1])}" {stroke} '
                  'stroke-width="1.7"/>')
    # tail fin: two flicks at the tip
    tp, tt, tn = pts[-1], tan[-1], nor[-1]
    for side, ln in ((1, 1.0), (0, 1.25), (-1, 0.9)):
        q1 = tp + tt * 8 * ln + side * tn * 4
        q2 = tp + tt * 20 * ln + side * tn * 11 - sgn * tn * 6
        el.append(f'<path d="M{f(tp[0])},{f(tp[1])} Q{f(q1[0])},{f(q1[1])} {f(q2[0])},{f(q2[1])}" {stroke} '
                  'stroke-width="1.7"/>')
    # legs: outlined, tapered limbs with three claws, reaching back toward the tail
    for i in (128, 380):
        o, ang = vent[i] + sgn * nor[i] * 1.5, math.atan2(tan[i][1], tan[i][0])
        side = -sgn                              # local +y points away from the belly
        cl, cw = np.array([(0, 0), (5, 9), (4, 18), (-2, 25), (-10, 28)], float), np.array([5, 4.2, 3.4, 2.7, 2.2])
        lp, _ = sample_spline(cl, 60)
        ln = frames(lp)[1]
        ww = np.interp(np.linspace(0, 1, len(lp)), np.linspace(0, 1, len(cw)), cw)
        outline = np.vstack([lp + ln * ww[:, None], (lp - ln * ww[:, None])[::-1]])
        q = xf(outline * [1, side], o, ang)
        el.append(f'<path d="{smooth_path(q[::3])} Z" fill="{t["bg"]}" stroke="{t["line"]}" stroke-width="1.8" '
                  'stroke-linejoin="round"/>')
        # three claws, then two tufts at the elbow
        for d in ("M-10,28 C-15,26 -18,27 -21,30", "M-10,28 C-14,30 -16,33 -17,37", "M-10,28 C-10,32 -11,35 -13,38",
                  "M4,17 C9,18 13,21 15,26", "M5,12 C10,12 14,14 17,18"):
            q = xf(_coords(d) * [1, side], o, ang)
            el.append(f'<path d="M{f(q[0][0])},{f(q[0][1])} C' + " ".join(f"{f(x)},{f(y)}" for x, y in q[1:]) +
                      f'" {stroke} stroke-width="1.7"/>')
    # head
    origin, hs = pts[0], 1.3
    el.append(f'<path d="{path_xf(HEAD_FILL, origin, angle, hs)}" fill="{t["bg"]}"/>')
    for d in HEAD_LINES:
        el.append(f'<path d="{path_xf(d, origin, angle, hs)}" {stroke} stroke-width="2.4"/>')
    el.append(f'<path d="{path_xf(EYE, origin, angle, hs)}" {stroke} stroke-width="1.8"/>')
    e = xf([EYE_DOT], origin, angle, hs)[0]
    el.append(f'<circle cx="{f(e[0])}" cy="{f(e[1])}" r="1.9" fill="{t["line"]}"/>')
    return el


def constellation(t, S, C):
    def P(name):
        x, y, m = stars.project(name)
        return C[0] + S * x, C[1] - S * y, m

    el = []
    # faint celestial graticule
    for dec in range(-50, 21, 10):
        pts = [stars.project_radec(ra, dec) for ra in np.linspace(12.6, 19.2, 60)]
        el.append(f'<path d="{smooth_path([(C[0] + S * x, C[1] - S * y) for x, y in pts])}" stroke="{t["grid"]}" '
                  'stroke-width="1" fill="none"/>')
    for ra in np.arange(13, 19.01, 1):
        pts = [stars.project_radec(ra, dec) for dec in np.linspace(-52, 22, 30)]
        el.append(f'<path d="{smooth_path([(C[0] + S * x, C[1] - S * y) for x, y in pts])}" stroke="{t["grid"]}" '
                  'stroke-width="1" fill="none"/>')
    # ecliptic
    ecl = [stars.project_ecliptic(lam) for lam in np.linspace(188, 285, 60)]
    el.append(f'<path d="{smooth_path([(C[0] + S * x, C[1] - S * y) for x, y in ecl])}" stroke="{t["faint"]}" '
              'stroke-width="1" stroke-dasharray="2 5" stroke-linecap="round" fill="none"/>')
    # asterism stick figures for the mansions off the dragon's body
    for key in ("氐", "房", "箕"):
        pts = [P(n)[:2] for n in stars.MANSIONS[key]]
        d = "M" + " L".join(f"{f(x)},{f(y)}" for x, y in pts)
        el.append(f'<path d="{d}" stroke="{t["line"]}" stroke-width="1.2" opacity=".5" fill="none" '
                  'stroke-linejoin="round"/>')
    return el


def star_dots(t, S, C):
    el = [f'<radialGradient id="glow"><stop offset="0" stop-color="{t["warm"]}" stop-opacity=".45"/>'
          f'<stop offset="1" stop-color="{t["warm"]}" stop-opacity="0"/></radialGradient>']
    for name in stars.STARS:
        x, y, m = stars.project(name)
        x, y = C[0] + S * x, C[1] - S * y
        r = 1.3 + max(0.0, 4.9 - m) * 0.95
        if name == "alpSco":
            el.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="22" fill="url(#glow)"/>')
            el.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="9.5" fill="none" stroke="{t["warm"]}" stroke-width="1"/>')
            el.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="{f(r + 2.2)}" fill="{t["halo"]}"/>')
            el.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="{f(r)}" fill="{t["warm"]}"/>')
            continue
        el.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="{f(r + 2.2)}" fill="{t["halo"]}"/>')
        el.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="{f(r)}" fill="{t["star"]}"/>')
    return el


LABELS = {"角": ("alpVir", -2, 30), "亢": ("phiVir", -20, -6), "氐": ("iotLib", 8, 22), "房": ("betSco", -24, -4),
          "心": ("alpSco", -10, -22), "尾": ("theSco", -6, 30), "箕": ("delSgr", -22, -10)}


def mansion_labels(t, S, C):
    el = []
    for ch, (name, dx, dy) in LABELS.items():
        x, y, _ = stars.project(name)
        d, _ = text_path(ch, "NotoSerifSC.ttf", 18, C[0] + S * x + dx, C[1] - S * y + dy, wght=500, anchor="middle")
        el.append(f'<path d="{d}" fill="{t["warm"] if ch == "心" else t["muted"]}"/>')
    return el


# --------------------------------------------------------------------------- logo mark
LOGO_HEAD = ("M-4,-6.8 C3,-8.6 8,-11.2 14,-10.4 C19.5,-9.8 24,-7.2 30,-6.7 C32.8,-6.5 34.4,-8.4 34.2,-10.8 "
             "C37.4,-8.4 38,-4.2 34.4,-2.7 C29,-1.6 23.4,-0.9 18.6,0.3 C23.4,1.1 27.6,1.9 30.6,3.3 "
             "C27.4,6.4 22,7.5 16,7.7 C8.4,8 2.4,7.5 -4,6.8 Z")
LOGO_LINES = [("M7,-9.6 C0,-17 -10,-20.5 -22,-20.5", 2.6), ("M-5,-18.8 C-6,-23.5 -4.5,-27.5 -0.5,-30.5", 2.2),
              ("M-15,-20.4 C-18,-24 -19,-28 -17.5,-32", 2.2),
              ("M13,-10.6 C9,-19 2,-26 -8,-30", 2.2),
              ("M33.4,-3.4 C40,2.6 37,10.6 28.6,12.2 C22,13.4 17.6,16.6 15,21.6", 1.8)]


def mark_parts(t, uid):
    """Dragon coiled round the Earth: a C for Canglong, the warm wave for the sea surface."""
    R = 38
    el = [f'<circle cx="0" cy="0" r="20.5" fill="none" stroke="{t["line"]}" stroke-width="4"/>',
          f'<clipPath id="g{uid}"><circle cx="0" cy="0" r="18.5"/></clipPath>',
          f'<path clip-path="url(#g{uid})" d="M-30,3 C-24,-4 -18,-4 -12,2 C-6,8 0,8 6,2 C12,-4 18,-4 24,2" '
          f'fill="none" stroke="{t["warm"]}" stroke-width="4" stroke-linecap="round"/>']
    ang = np.radians([66, 100, 140, 180, 220, 262, 300, 328])
    ctrl = [(25, -R - 3)] + [(R * np.cos(a), -R * np.sin(a)) for a in ang]
    pts, _ = sample_spline(ctrl, 300)
    tan, nor = frames(pts)
    u = np.linspace(0, 1, len(pts))
    half = 0.45 + 4.6 * (1 - u ** 1.4) + 1.7 * np.exp(-u / 0.05)
    mid = len(pts) // 2
    outer = 1 if np.dot(nor[mid], pts[mid]) > 0 else -1
    body = np.vstack([(pts + nor * half[:, None])[::5], pts[-1:] + tan[-1:] * 1.5, (pts - nor * half[:, None])[::-5]])
    el.append(f'<path d="{smooth_path(body)} Z" fill="{t["line"]}"/>')
    allpts = [body]
    # dorsal spikes along the outer edge of the coil
    for uu in np.arange(0.07, 0.6, 0.075):
        i = int(uu * (len(pts) - 1))
        h = 3.6 * (1 - uu * 1.1)
        n = nor[i] * outer
        base = pts[i] + n * half[i] * 0.6
        a, b = base - tan[i] * h * 0.75, base + tan[i] * h * 0.75
        tip = pts[i] + n * (half[i] + h) + tan[i] * h * 0.9
        el.append(f'<path d="M{f(a[0])},{f(a[1])} L{f(tip[0])},{f(tip[1])} L{f(b[0])},{f(b[1])} Z" '
                  f'fill="{t["line"]}" stroke-linejoin="round"/>')
        allpts.append(tip[None])
    angle = math.atan2(-tan[0][1], -tan[0][0])
    head = path_xf(LOGO_HEAD, pts[0], angle)
    el.append(f'<path d="{head}" fill="{t["line"]}"/>')
    allpts.append(_coords(head))
    e = xf([(13.6, -4.4)], pts[0], angle)[0]
    el.append(f'<circle cx="{f(e[0])}" cy="{f(e[1])}" r="1.6" fill="{t["bg"]}"/>')
    for d, w in LOGO_LINES:
        d = path_xf(d, pts[0], angle)
        el.append(f'<path d="{d}" fill="none" stroke="{t["line"]}" stroke-width="{w}" stroke-linecap="round"/>')
        allpts.append(_coords(d))
    P = np.vstack(allpts)
    return el, P.min(0) - 3, P.max(0) + 3


def mark(t, size=120, x=0, y=0, standalone=False):
    el, lo, hi = mark_parts(t, size)
    side = max(hi - lo)
    c = (lo + hi) / 2
    vb = (c[0] - side / 2, c[1] - side / 2, side, side)
    if standalone:
        return el, vb
    return (f'<g transform="translate({f(x)},{f(y)}) scale({f(size / side)}) translate({f(-vb[0])},{f(-vb[1])})">'
            + "\n".join(el) + "</g>")


def logo_svg(theme):
    t = THEMES[theme]
    el, vb = mark(t, standalone=True)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{" ".join(f(v) for v in vb)}" width="240" height="240" '
            'role="img" aria-label="CAS-Canglong logo">\n' + "\n".join(el) + "\n</svg>\n")


# --------------------------------------------------------------------------- banner
def chip(t, x, y, label):
    d, w = text_path(label, "PlexMono-Medium.ttf", 13.5, x + 17, y + 22.5, tracking=0.05)
    return (f'<rect x="{f(x)}" y="{y}" width="{f(w + 34)}" height="34" rx="17" fill="none" stroke="{t["faint"]}" '
            f'stroke-width="1.25"/>\n<path d="{d}" fill="{t["ink2"]}"/>', w + 34)


def banner_svg(theme, W=1600, H=480):
    t = THEMES[theme]
    S, C = 10.0, (1160, 214)
    el = [f'<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="28" fill="{t["bg"]}" stroke="{t["border"]}"/>']
    el.append(f'<clipPath id="card"><rect width="{W}" height="{H}" rx="28"/></clipPath>')
    el.append(f'<linearGradient id="fade" x1="0" x2="1"><stop offset="0" stop-color="#fff" stop-opacity="0"/>'
              f'<stop offset=".22" stop-color="#fff"/><stop offset="1" stop-color="#fff"/></linearGradient>'
              f'<mask id="m"><rect x="700" width="{W - 700}" height="{H}" fill="url(#fade)"/></mask>')
    el.append('<g clip-path="url(#card)"><g mask="url(#m)">' + "\n".join(constellation(t, S, C)) + "</g></g>")
    el += dragon(t, S, C)
    el += star_dots(t, S, C)
    el += mansion_labels(t, S, C)
    # left: lockup, tagline, chips
    el.append(mark(t, 104, 80, 74))
    d, _ = text_path("CAS-Canglong", "Inter.ttf", 70, 204, 146, wght=640, tracking=-0.028)
    el.append(f'<path d="{d}" fill="{t["ink"]}"/>')
    d, _ = text_path("中科苍龙", "NotoSerifSC.ttf", 25, 207, 189, wght=500, tracking=0.5)
    el.append(f'<path d="{d}" fill="{t["ink2"]}"/>')
    d, _ = text_path("AZURE DRAGON", "PlexMono-Regular.ttf", 13, 364, 185, tracking=0.2)
    el.append(f'<path d="{d}" fill="{t["muted"]}"/>')
    for i, line in enumerate(["AI forecasts of global sea-surface temperature",
                              "and ENSO, from one month to more than a year ahead."]):
        d, _ = text_path(line, "Inter.ttf", 23, 84, 278 + i * 34, wght=400, tracking=-0.005)
        el.append(f'<path d="{d}" fill="{t["ink2"]}"/>')
    x = 84
    for label in ("SST + ENSO · 16 MONTHS", "ATMOSPHERE · 6 WEEKS", "ADVANCED SCIENCE 2026"):
        s, w = chip(t, x, 362, label)
        el.append(s)
        x += w + 10
    d, _ = text_path("苍龙七宿 · THE AZURE DRAGON OF THE EAST · J2000", "NotoSerifSC.ttf", 13, W - 48, H - 30,
                     wght=500, tracking=0.12, anchor="end")
    el.append(f'<path d="{d}" fill="{t["muted"]}"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
            f'aria-label="CAS-Canglong — AI forecasts of global sea-surface temperature and ENSO">\n'
            + "\n".join(el) + "\n</svg>\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        sfx = "" if theme == "light" else "_dark"
        (OUT / f"banner{sfx}.svg").write_text(banner_svg(theme))
        (OUT / f"logo{sfx}.svg").write_text(logo_svg(theme))
    print("ok")
