"""CAS-Canglong logo: a dragon coiled round the Earth (a C for Canglong), the warm wave for the sea surface.

    python assets/readme/src/logo.py assets/readme      # -> logo.svg, logo_dark.svg, src/logo_shapes.json

logo_shapes.json holds the same geometry for banner.mjs, which redraws the mark by hand. Needs numpy.
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

THEMES = {
    "light": dict(bg="#F7F6F1", line="#1D5A5E", warm="#E0603C"),
    "dark": dict(bg="#0A1210", line="#B9E3C6", warm="#FF7B54"),
}
HEAD = ("M-4,-6.8 C3,-8.6 8,-11.2 14,-10.4 C19.5,-9.8 24,-7.2 30,-6.7 C32.8,-6.5 34.4,-8.4 34.2,-10.8 "
        "C37.4,-8.4 38,-4.2 34.4,-2.7 C29,-1.6 23.4,-0.9 18.6,0.3 C23.4,1.1 27.6,1.9 30.6,3.3 "
        "C27.4,6.4 22,7.5 16,7.7 C8.4,8 2.4,7.5 -4,6.8 Z")
LINES = [("M7,-9.6 C0,-17 -10,-20.5 -22,-20.5", 2.6), ("M-5,-18.8 C-6,-23.5 -4.5,-27.5 -0.5,-30.5", 2.2),
         ("M-15,-20.4 C-18,-24 -19,-28 -17.5,-32", 2.2), ("M13,-10.6 C9,-19 2,-26 -8,-30", 2.2),
         ("M33.4,-3.4 C40,2.6 37,10.6 28.6,12.2 C22,13.4 17.6,16.6 15,21.6", 1.8)]


def f(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def smooth_path(pts, closed=False):
    """Catmull-Rom through pts -> cubic Bezier path data."""
    pts = np.asarray(pts, float)
    P = np.vstack([pts[-1], pts, pts[0], pts[1]]) if closed else np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    d = [f"M{f(P[1][0])},{f(P[1][1])}"]
    for i in range(1, len(pts) + (1 if closed else 0)):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        c1, c2 = p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6
        d.append(f"C{f(c1[0])},{f(c1[1])} {f(c2[0])},{f(c2[1])} {f(p2[0])},{f(p2[1])}")
    return " ".join(d) + (" Z" if closed else "")


def sample_spline(ctrl, n=300):
    """Dense Catmull-Rom samples through the control points, evenly spaced by arc length."""
    P = np.asarray(ctrl, float)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, 40, endpoint=False):
            out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    out = np.array(out + [P[-2]])
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(out, axis=0).T))]
    s = np.linspace(0, seg[-1], n)
    return np.c_[np.interp(s, seg, out[:, 0]), np.interp(s, seg, out[:, 1])]


def xf(points, origin, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(points, float) @ np.array([[c, -s], [s, c]]).T + origin


def path_xf(d, origin, angle):
    """Rotate and move absolute-coordinate path data (M/C/L/Z only)."""
    out = []
    for cmd, nums in re.findall(r"([MCLZ])([^MCLZ]*)", d):
        pts = np.array(re.findall(r"-?\d*\.?\d+", nums), float).reshape(-1, 2)
        out.append(cmd + " ".join(f"{f(x)},{f(y)}" for x, y in (xf(pts, origin, angle) if len(pts) else [])))
    return " ".join(out)


def shapes():
    """The mark as a list of {role, d | circle, width}; roles: globe, wave, body, spike, head, eye, line, and spine
    (the body's centre line, which only the hand-drawn banner uses)."""
    R = 38
    out = [dict(role="globe", cx=0, cy=0, r=20.5, width=4),
           dict(role="wave", d="M-18,3 C-12,-4 -6,-4 0,2 C6,8 12,8 18,2", width=4)]
    ang = np.radians([66, 100, 140, 180, 220, 262, 300, 328])
    pts = sample_spline([(25, -R - 3)] + [(R * np.cos(a), -R * np.sin(a)) for a in ang])
    tan = np.gradient(pts, axis=0)
    tan /= np.hypot(*tan.T)[:, None]
    nor = np.c_[-tan[:, 1], tan[:, 0]]
    u = np.linspace(0, 1, len(pts))
    half = 0.45 + 4.6 * (1 - u ** 1.4) + 1.7 * np.exp(-u / 0.05)
    outer = 1 if np.dot(nor[len(pts) // 2], pts[len(pts) // 2]) > 0 else -1
    ring = np.vstack([(pts + nor * half[:, None])[::5], pts[-1:] + tan[-1:] * 1.5, (pts - nor * half[:, None])[::-5]])
    out.append(dict(role="body", d=smooth_path(ring) + " Z"))
    out.append(dict(role="spine", d=smooth_path(np.vstack([pts[::12], pts[-1:]]))))
    for uu in np.arange(0.07, 0.6, 0.075):                      # dorsal spikes along the outer edge
        i = int(uu * (len(pts) - 1))
        h, n = 3.6 * (1 - uu * 1.1), nor[i] * outer
        base = pts[i] + n * half[i] * 0.6
        a, b, tip = base - tan[i] * h * 0.75, base + tan[i] * h * 0.75, pts[i] + n * (half[i] + h) + tan[i] * h * 0.9
        out.append(dict(role="spike", d=f"M{f(a[0])},{f(a[1])} L{f(tip[0])},{f(tip[1])} L{f(b[0])},{f(b[1])} Z"))
    angle = math.atan2(-tan[0][1], -tan[0][0])
    out.append(dict(role="head", d=path_xf(HEAD, pts[0], angle)))
    e = xf([(13.6, -4.4)], pts[0], angle)[0]
    out.append(dict(role="eye", cx=round(float(e[0]), 2), cy=round(float(e[1]), 2), r=1.6))
    out += [dict(role="line", d=path_xf(d, pts[0], angle), width=w) for d, w in LINES]
    coords = np.vstack([np.array(re.findall(r"-?\d*\.?\d+", s["d"]), float).reshape(-1, 2) for s in out if "d" in s])
    lo, hi = coords.min(0) - 3, coords.max(0) + 3
    side = float(max(hi - lo))
    c = (lo + hi) / 2
    return out, [round(float(c[0] - side / 2), 2), round(float(c[1] - side / 2), 2), round(side, 2)]


def logo_svg(theme, items, box):
    t = THEMES[theme]
    el = []
    for s in items:
        if s["role"] == "globe":
            el.append(f'<circle cx="0" cy="0" r="{s["r"]}" fill="none" stroke="{t["line"]}" stroke-width="{s["width"]}"/>')
        elif s["role"] == "wave":
            el.append(f'<path d="{s["d"]}" fill="none" stroke="{t["warm"]}" stroke-width="{s["width"]}" '
                      'stroke-linecap="round"/>')
        elif s["role"] == "spine":
            continue
        elif s["role"] == "eye":
            el.append(f'<circle cx="{s["cx"]}" cy="{s["cy"]}" r="{s["r"]}" fill="{t["bg"]}"/>')
        elif s["role"] == "line":
            el.append(f'<path d="{s["d"]}" fill="none" stroke="{t["line"]}" stroke-width="{s["width"]}" '
                      'stroke-linecap="round"/>')
        else:
            el.append(f'<path d="{s["d"]}" fill="{t["line"]}" stroke-linejoin="round"/>')
    x, y, side = box
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{f(x)} {f(y)} {f(side)} {f(side)}" width="240" '
            'height="240" role="img" aria-label="CAS-Canglong logo">\n' + "\n".join(el) + "\n</svg>\n")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    items, box = shapes()
    for theme in THEMES:
        (out / f"logo{'' if theme == 'light' else '_dark'}.svg").write_text(logo_svg(theme, items, box))
    Path(__file__).with_name("logo_shapes.json").write_text(json.dumps({"box": box, "shapes": items}, indent=1) + "\n")
    print("ok")
