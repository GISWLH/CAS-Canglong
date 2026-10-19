"""Shape text with HarfBuzz and return it as SVG path data (so the artwork needs no web fonts).

Fonts (SIL Open Font License) go in ./fonts, from https://github.com/google/fonts/tree/main/ofl:
inter/Inter[opsz,wght].ttf -> Inter.ttf, notoserifsc/NotoSerifSC[wght].ttf -> NotoSerifSC.ttf,
ibmplexmono/IBMPlexMono-{Medium,Regular}.ttf -> PlexMono-{Medium,Regular}.ttf.
"""
from functools import lru_cache
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

FONTS = str(Path(__file__).with_name("fonts")) + "/"


@lru_cache(None)
def _font(name, wght):
    face = hb.Face(hb.Blob.from_file_path(FONTS + name))
    font = hb.Font(face)
    if wght is not None:
        axes = {a.tag for a in face.axis_infos}
        font.set_variations({"wght": wght, **({"opsz": 32} if "opsz" in axes else {})})
    return font, face.upem


def text_path(text, font, size, x, y, wght=None, tracking=0.0, anchor="start"):
    """Return (d, width). tracking is in em; anchor in start|middle|end."""
    f, upem = _font(font, wght)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(f, buf, {"kern": True, "liga": True})
    s = size / upem
    adv = [p.x_advance * s + tracking * size for p in buf.glyph_positions]
    width = sum(adv) - tracking * size
    x0 = x - {"start": 0, "middle": width / 2, "end": width}[anchor]
    pen = SVGPathPen(None, ntos=lambda v: f"{v:.2f}".rstrip("0").rstrip("."))
    cx = x0
    for info, pos, a in zip(buf.glyph_infos, buf.glyph_positions, adv):
        tp = TransformPen(pen, (s, 0, 0, -s, cx + pos.x_offset * s, y - pos.y_offset * s))
        f.draw_glyph_with_pen(info.codepoint, tp)
        cx += a
    return pen.getCommands(), width
