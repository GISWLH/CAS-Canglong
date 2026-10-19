// CAS-Canglong README banner, drawn as a whiteboard sketch (light and dark).
//
//   npm i roughjs@4.6.6 opentype.js@1.3.4
//   python assets/readme/src/logo.py assets/readme && node assets/readme/src/banner.mjs assets/readme
//
// Shapes are roughened with rough.js; text is set in Kalam and Long Cang (SIL OFL, from github.com/google/fonts:
// ofl/kalam/Kalam-{Bold,Regular,Light}.ttf, ofl/longcang/LongCang-Regular.ttf) and converted to outlines, so the
// SVG needs no web fonts. Fonts go in ./fonts next to this file.
import { createRequire } from "node:module";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const rough = require("roughjs");
const opentype = require("opentype.js");
const here = dirname(fileURLToPath(import.meta.url));
const font = (name) => opentype.loadSync(join(here, "fonts", name));
const FONTS = { bold: font("Kalam-Bold.ttf"), regular: font("Kalam-Regular.ttf"), light: font("Kalam-Light.ttf"),
  cn: font("LongCang-Regular.ttf") };
const logo = JSON.parse(readFileSync(join(here, "logo_shapes.json"), "utf8"));

const THEMES = {
  light: { bg: "#fcfcfa", frame: "#ced4da", ink: "#1e1e1e", ink2: "#495057", muted: "#868e96",
    blue: "#1971c2", blueFill: "#a5d8ff", green: "#2f9e44", greenFill: "#b2f2bb", orange: "#f08c00",
    yellowFill: "#ffe066", red: "#e03131", redFill: "#ffc9c9" },
  dark: { bg: "#14171b", frame: "#3b4148", ink: "#e9ecef", ink2: "#ced4da", muted: "#909aa4",
    blue: "#74c0fc", blueFill: "#2b5c87", green: "#69db7c", greenFill: "#2c6a3b", orange: "#ffa94d",
    yellowFill: "#7d6514", red: "#ff8787", redFill: "#7a2e2e" },
};
const W = 1600, H = 480;

function banner(t) {
  const gen = rough.generator();
  let seed = 7;
  const out = [];
  const base = { roughness: 1.1, bowing: 1, hachureAngle: -41, hachureGap: 6, fillWeight: 1.4 };
  // draw a rough.js drawable; `g` wraps it in a transform group
  const put = (drawable, g) => {
    const paths = gen.toPaths(drawable).map((p) =>
      `<path d="${p.d}" stroke="${p.stroke}" stroke-width="${p.strokeWidth}" fill="${p.fill || "none"}" ` +
      'stroke-linecap="round" stroke-linejoin="round"/>');
    out.push(g ? `<g transform="${g}">${paths.join("")}</g>` : paths.join("\n"));
  };
  const o = (opts) => ({ ...base, seed: seed++, ...opts });
  const text = (s, x, y, size, f, fill, anchor = "start") => {
    const w = f.getAdvanceWidth(s, size);
    const x0 = anchor === "middle" ? x - w / 2 : anchor === "end" ? x - w : x;
    out.push(`<path d="${f.getPath(s, x0, y, size).toPathData(2)}" fill="${fill}"/>`);
    return w;
  };
  const arrow = (pts, color, width = 2.2) => {
    put(gen.curve(pts, o({ stroke: color, strokeWidth: width, roughness: 1.3 })));
    const [x2, y2] = pts.at(-1), [x1, y1] = pts.at(-2);
    const a = Math.atan2(y2 - y1, x2 - x1);
    for (const s of [-1, 1]) {
      const b = a + Math.PI + s * 0.5;
      put(gen.line(x2, y2, x2 + 15 * Math.cos(b), y2 + 15 * Math.sin(b), o({ stroke: color, strokeWidth: width })));
    }
  };

  out.push(`<rect width="${W}" height="${H}" rx="26" fill="${t.bg}"/>`);
  put(gen.rectangle(16, 16, W - 32, H - 32, o({ stroke: t.frame, strokeWidth: 1.6, roughness: 1.4 })));

  // ---- hand-drawn logo: dragon coiled round the Earth, its body one tapering marker stroke
  const [bx, by, side] = logo.box, k = 124 / side;
  const g = `translate(${74 - bx * k},${44 - by * k}) scale(${k})`;
  for (const s of logo.shapes) {
    if (s.role === "globe")
      put(gen.circle(s.cx, s.cy, 2 * s.r, o({ stroke: t.blue, strokeWidth: 2.2 / k, fill: t.blueFill, fillStyle: "hachure",
        hachureGap: 4.2, fillWeight: 0.9, roughness: 0.8 })), g);
    else if (s.role === "wave") put(gen.path(s.d, o({ stroke: t.red, strokeWidth: 3 / k, roughness: 0.6 })), g);
    else if (s.role === "spine") {
      const pts = s.d.match(/-?\d*\.?\d+/g).map(Number);
      const xy = []; for (let i = 0; i < pts.length; i += 2) xy.push([pts[i], pts[i + 1]]);
      const knots = [xy[0], ...xy.slice(1).filter((_, i) => i % 3 === 2)];    // end points of each cubic
      for (const [frac, w] of [[1, 3], [0.62, 5], [0.3, 7]])
        put(gen.curve(knots.slice(0, Math.max(2, Math.round(knots.length * frac))),
          o({ stroke: t.green, strokeWidth: w / k, roughness: 0.5, disableMultiStroke: true })), g);
    } else if (s.role === "head")
      put(gen.path(s.d, o({ stroke: t.ink, strokeWidth: 1.8 / k, fill: t.greenFill, fillStyle: "hachure", hachureGap: 3.2,
        fillWeight: 0.9, roughness: 0.5 })), g);
    else if (s.role === "eye") out.push(`<circle transform="${g}" cx="${s.cx}" cy="${s.cy}" r="1.5" fill="${t.ink}"/>`);
    else if (s.role === "line") put(gen.path(s.d, o({ stroke: t.ink, strokeWidth: 1.8 / k, roughness: 0.5 })), g);
  }

  // ---- title, name, tagline
  text("CAS-Canglong", 214, 140, 84, FONTS.bold, t.ink);
  put(gen.curve([[220, 166], [380, 158], [540, 162], [636, 152]], o({ stroke: t.blue, strokeWidth: 4, roughness: 1.6 })));
  const cn = text("中科苍龙", 222, 226, 46, FONTS.cn, t.red);
  text("· the Azure Dragon", 222 + cn + 14, 222, 23, FONTS.light, t.muted);
  text("AI forecasts of global sea-surface temperature", 84, 300, 27, FONTS.regular, t.ink2);
  text("and ENSO, from one month to more than a year ahead.", 84, 336, 27, FONTS.regular, t.ink2);
  let x = 84;
  for (const [label, stroke, fill] of [["SST + ENSO · 16 months", t.blue, t.blueFill],
    ["Atmosphere · 6 weeks", t.green, t.greenFill], ["Advanced Science 2026", t.orange, t.yellowFill]]) {
    const w = FONTS.regular.getAdvanceWidth(label, 19) + 32;
    put(gen.rectangle(x, 370, w, 42, o({ stroke, strokeWidth: 1.8, fill, fillStyle: "hachure", hachureGap: 5,
      fillWeight: 1.1, roughness: 1.2 })));
    text(label, x + w / 2, 398, 19, FONTS.regular, t.ink, "middle");
    x += w + 14;
  }

  // ---- right: ERA5 globe -> 3-D Earth transformer -> SST + ENSO outlook
  const gx = 900, gy = 220, r = 72;
  put(gen.circle(gx, gy, 2 * r, o({ stroke: t.blue, strokeWidth: 2.6, fill: t.blueFill, fillStyle: "hachure",
    hachureGap: 7 })));
  put(gen.ellipse(gx, gy, 62, 2 * r, o({ stroke: t.blue, strokeWidth: 1.5, roughness: 0.9 })));
  for (const dy of [-40, 0, 40]) {
    const hw = Math.sqrt(r * r - dy * dy) - 4;
    put(gen.line(gx - hw, gy + dy, gx + hw, gy + dy, o({ stroke: t.blue, strokeWidth: 1.5, roughness: 0.9 })));
  }
  text("ocean + atmosphere", gx, 336, 23, FONTS.regular, t.ink, "middle");
  text("16 months of reanalysis", gx, 363, 19, FONTS.light, t.muted, "middle");
  arrow([[986, 218], [1020, 206], [1054, 216]], t.ink);

  // cube of 3-D patches
  const cx0 = 1074, cy0 = 186, s = 92, dx = 34, dy = -28;
  put(gen.polygon([[cx0, cy0], [cx0 + dx, cy0 + dy], [cx0 + s + dx, cy0 + dy], [cx0 + s, cy0]],
    o({ stroke: t.ink, strokeWidth: 2.2, fill: t.yellowFill, fillStyle: "hachure", hachureAngle: 41, hachureGap: 6 })));
  put(gen.polygon([[cx0 + s, cy0], [cx0 + s + dx, cy0 + dy], [cx0 + s + dx, cy0 + s + dy], [cx0 + s, cy0 + s]],
    o({ stroke: t.ink, strokeWidth: 2.2, fill: t.yellowFill, fillStyle: "hachure", hachureAngle: 90, hachureGap: 6 })));
  put(gen.rectangle(cx0, cy0, s, s, o({ stroke: t.ink, strokeWidth: 2.4, fill: t.yellowFill, fillStyle: "hachure",
    hachureGap: 6 })));
  for (const f of [1 / 3, 2 / 3]) {
    put(gen.line(cx0 + s * f, cy0, cx0 + s * f, cy0 + s, o({ stroke: t.ink, strokeWidth: 1.2, roughness: 0.8 })));
    put(gen.line(cx0, cy0 + s * f, cx0 + s, cy0 + s * f, o({ stroke: t.ink, strokeWidth: 1.2, roughness: 0.8 })));
    put(gen.line(cx0 + s * f + dx * 0.02, cy0, cx0 + s * f + dx, cy0 + dy, o({ stroke: t.ink, strokeWidth: 1.2,
      roughness: 0.8 })));
    put(gen.line(cx0 + s + dx * f, cy0 + dy * f, cx0 + s + dx * f, cy0 + s + dy * f, o({ stroke: t.ink,
      strokeWidth: 1.2, roughness: 0.8 })));
  }
  text("3-D Earth transformer", cx0 + (s + dx) / 2, 336, 23, FONTS.regular, t.ink, "middle");
  text("0.8 s on one GPU", cx0 + (s + dx) / 2, 363, 19, FONTS.light, t.muted, "middle");
  arrow([[1214, 218], [1248, 206], [1282, 216]], t.ink);
  text("one forward pass", 1206, 112, 19, FONTS.light, t.blue, "middle");
  put(gen.curve([[1228, 122], [1250, 150], [1246, 192]], o({ stroke: t.blue, strokeWidth: 1.5, roughness: 0.8 })));

  // ENSO outlook chart
  const ax = 1306, ay = 286, aw = 214, ah = 156;
  put(gen.line(ax, ay - ah, ax, ay, o({ stroke: t.ink, strokeWidth: 2.2 })));
  put(gen.line(ax, ay, ax + aw, ay, o({ stroke: t.ink, strokeWidth: 2.2 })));
  put(gen.line(ax + 6, 228, ax + aw - 6, 228, o({ stroke: t.muted, strokeWidth: 1.4, strokeLineDash: [6, 7],
    roughness: 0.6 })));
  const curve = [[ax + 10, 220], [ax + 42, 174], [ax + 76, 154], [ax + 112, 184], [ax + 150, 246], [ax + 184, 260],
    [ax + 208, 242]];
  put(gen.curve(curve, o({ stroke: t.red, strokeWidth: 3.2, roughness: 1.2 })));
  for (const [px, py] of [curve[2], curve[5]])
    put(gen.circle(px, py, 10, o({ stroke: t.red, strokeWidth: 1.6, fill: t.red, fillStyle: "solid", roughness: 0.6 })));
  text("Niño 3.4", ax + 12, ay - ah + 14, 18, FONTS.light, t.muted);
  text("SST + ENSO outlook", ax + aw / 2, 336, 23, FONTS.regular, t.ink, "middle");
  text("1–16 months ahead", ax + aw / 2, 363, 19, FONTS.light, t.muted, "middle");

  // a note in the corner pointing to the live service
  const lw = text("live at canglong.earth", 1528, 428, 22, FONTS.regular, t.blue, "end");
  put(gen.line(1528 - lw, 438, 1528, 436, o({ stroke: t.blue, strokeWidth: 2, roughness: 1.4 })));
  arrow([[1528 - lw - 46, 410], [1528 - lw - 26, 424], [1528 - lw - 8, 422]], t.blue, 1.8);

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" ` +
    'aria-label="CAS-Canglong: AI forecasts of global sea-surface temperature and ENSO, from one month to more than a ' +
    `year ahead">\n${out.join("\n")}\n</svg>\n`;
}

const dest = process.argv[2] || ".";
for (const [name, t] of Object.entries(THEMES))
  writeFileSync(join(dest, name === "light" ? "banner.svg" : "banner_dark.svg"), banner(t));
console.log("ok");
