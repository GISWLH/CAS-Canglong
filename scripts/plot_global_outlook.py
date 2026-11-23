"""Plot the global SST anomaly outlook of CAS-Canglong 0.25° Ensemble in IPCC journal style (no GPU, no ERA5 needed).

    pip install huggingface_hub xarray netCDF4 scipy matplotlib cartopy
    python scripts/plot_global_outlook.py                       # -> assets/readme/global_outlook{,_dark}.png
    python scripts/plot_global_outlook.py --leads 1 3 6 9 12 15 --theme light

Six Pacific-centred Robinson panels (a–f) across the 16-month forecast with discrete IPCC RdBu_r contours,
quasi-uniform hexagonal stippling for robust ensemble signal (|mean| >= sigma), dashed graticules, and an
outlined triangular-ended discrete colourbar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import ndimage  # noqa: E402

SPACE = "LonghaoWang/CAS-Canglong"
THEMES = {
    "light": dict(
        surface="#ffffff",
        ink="#1a1a1a",
        ink2="#3f3f3f",
        muted="#5c5c5c",
        land="#ededeb",
        coast="#1a1a1a",
        rim="#1a1a1a",
        grid="#9a9a9a",
        stipple="#1a1a1a",
        nino="#1a1a1a",
        legend_bg="#ffffff",
        legend_ec="#c8c8c8",
    ),
    "dark": dict(
        surface="#0d1117",
        ink="#e6edf3",
        ink2="#b1bac4",
        muted="#8b949e",
        land="#1c232d",
        coast="#c9d1d9",
        rim="#6e7681",
        grid="#484f58",
        stipple="#141414",
        nino="#141414",
        legend_dot="#c9d1d9",
        legend_bg="#161b22",
        legend_ec="#30363d",
    ),
}
NINO34 = dict(lon=190, lat=-5, width=50, height=10)
LEVELS = np.round(np.arange(-2.0, 2.01, 0.25), 2)


def fetch(name: str, local_dir: Path | None = None) -> Path:
    if local_dir is not None:
        candidate = local_dir / name
        if candidate.exists():
            return candidate
    return Path(hf_hub_download(SPACE, name, repo_type="space"))


def discrete_cmap(cmap_name: str, levels: np.ndarray, extend: str = "both") -> tuple[ListedColormap, BoundaryNorm]:
    """IPCC-style discrete colormap with distinct extend triangle extremes."""
    n_int = len(levels) - 1
    n_extra = {"neither": 0, "min": 1, "max": 1, "both": 2}[extend]
    base = mpl.colormaps[cmap_name]
    colors = list(base(np.linspace(0.02, 0.98, n_int + n_extra)))
    lo = 1 if extend in ("min", "both") else 0
    out = ListedColormap(colors[lo : lo + n_int], name=f"{cmap_name}_discrete")
    extremes = {"bad": "none"}
    if extend in ("min", "both"):
        extremes["under"] = colors[0]
    if extend in ("max", "both"):
        extremes["over"] = colors[-1]
    out = out.with_extremes(**extremes)
    norm = BoundaryNorm(levels, n_int, extend="neither")
    return out, norm


def smooth_field(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, step: float = 0.25,
                 sigma: float = 0.7, reach: float = 1.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(field)
    dist, idx = ndimage.distance_transform_edt(~valid, return_indices=True)
    filled = ndimage.gaussian_filter(field[tuple(idx)], sigma, mode=("nearest", "wrap"))
    pad = 3
    filled = np.concatenate([filled[:, -pad:], filled, filled[:, :pad]], axis=1)
    dist = np.concatenate([dist[:, -pad:], dist, dist[:, :pad]], axis=1)
    lat_hi = np.arange(lat[0], lat[-1] + step / 2, step)
    lon_hi = np.arange(lon[0], lon[0] + 360 + step / 2, step)
    rows = np.interp(lat_hi, lat, np.arange(lat.size))
    cols = (lon_hi - lon[0]) / (lon[1] - lon[0]) + pad
    grid = np.meshgrid(rows, cols, indexing="ij")
    out = ndimage.map_coordinates(filled, grid, order=3, mode="nearest")
    out[ndimage.map_coordinates(dist, grid, order=1, mode="nearest") > reach] = np.nan
    return out, lat_hi, lon_hi


def add_hex_stippling(ax, lon: np.ndarray, lat: np.ndarray, mask: np.ndarray, *,
                      base_stride: int = 5, size: float = 1.6, color: str = "#1a1a1a",
                      alpha: float = 0.72, zorder: float = 4.0) -> None:
    """Quasi-uniform staggered hexagonal stippling compensated for cos(lat) convergence."""
    pts_lon, pts_lat = [], []
    n_lat, n_lon = mask.shape
    row_indices = list(range(2, n_lat - 2, base_stride))
    for k, ri in enumerate(row_indices):
        la = float(lat[ri])
        if abs(la) > 78.0:
            continue
        lon_stride = max(base_stride, int(round(base_stride / max(np.cos(np.deg2rad(la)), 0.22))))
        start = (lon_stride // 2) if (k % 2 == 1) else 0
        cols = np.arange(start, n_lon, lon_stride)
        hit = cols[mask[ri, cols]]
        if hit.size:
            pts_lon.append(lon[hit])
            pts_lat.append(np.full(hit.size, la))
    if pts_lon:
        ax.scatter(
            np.concatenate(pts_lon), np.concatenate(pts_lat),
            s=size, c=color, marker="o", linewidths=0, alpha=alpha,
            transform=ccrs.PlateCarree(), zorder=zorder, rasterized=False,
        )


def clean_tick(v: float, _=None) -> str:
    if abs(v) < 1e-12:
        return "0"
    s = f"{v:.1f}" if not np.isclose(v, round(v)) else f"{int(round(v))}"
    return s.replace("-", "\u2212")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--leads", type=int, nargs=6, default=[1, 3, 6, 9, 12, 15], help="six forecast months, 1..16")
    p.add_argument("--theme", choices=["light", "dark", "both"], default="both")
    p.add_argument("--out", type=Path, default=Path("assets/readme/global_outlook.png"))
    p.add_argument("--local-dir", type=Path, default=None, help="optional local Space root containing data/")
    a = p.parse_args()

    run = json.loads(fetch("data/run.json", a.local_dir).read_text(encoding="utf-8"))
    ds = xr.open_dataset(fetch(f"data/{run['sst_file']}", a.local_dir))
    months = pd.to_datetime(ds.time.values)
    nino = ds.nino34_9120.values.astype(float)
    lat, lon = ds.lat.values.astype(float), ds.lon.values.astype(float)
    init = pd.Timestamp(run["initialization"] + "-01")

    panels = {}
    for k in a.leads:
        ssta = ds.ssta_9120.isel(time=k - 1).values.astype(float)
        sigma = ds.sigma.isel(time=k - 1).values.astype(float)
        ssta_hi, lat_hi, lon_hi = smooth_field(ssta, lat, lon)
        robust = np.isfinite(ssta) & np.isfinite(sigma) & (sigma > 1e-6) & (np.abs(ssta) >= sigma)
        panels[k] = (ssta_hi, lat_hi, lon_hi, robust)

    for theme in (["light", "dark"] if a.theme == "both" else [a.theme]):
        out = a.out if theme == "light" else a.out.with_name(f"{a.out.stem}_dark{a.out.suffix}")
        draw(panels, lat, lon, months, nino, init, theme, out)
        print(out)


def draw(panels: dict, lat: np.ndarray, lon: np.ndarray, months: pd.DatetimeIndex,
         nino: np.ndarray, init: pd.Timestamp, theme: str, out: Path) -> None:
    t = THEMES[theme]
    cmap, norm = discrete_cmap("RdBu_r", LEVELS, extend="both")

    rc = {
        "font.family": ["Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "text.color": t["ink"],
        "axes.edgecolor": t["ink"],
        "axes.linewidth": 0.6,
        "xtick.color": t["ink"],
        "ytick.color": t["ink"],
        "axes.unicode_minus": True,
    }
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=(15.4, 7.6), dpi=180, facecolor=t["surface"])
        crs = ccrs.Robinson(central_longitude=200)
        w, h, gx, gy, x0, y0 = 0.308, 0.365, 0.018, 0.072, 0.020, 0.160
        cs_ref = None
        letters = "abcdef"

        for i, (k, (data, lat_hi, lon_hi, robust)) in enumerate(panels.items()):
            r, c = divmod(i, 3)
            ax = fig.add_axes([x0 + c * (w + gx), y0 + (1 - r) * (h + gy), w, h], projection=crs)
            ax.set_global()
            ax.set_facecolor(t["land"])

            cs = ax.contourf(
                lon_hi, lat_hi, data, levels=LEVELS, cmap=cmap, norm=norm,
                extend="both", transform=ccrs.PlateCarree(), zorder=1,
            )
            cs_ref = cs

            gl = ax.gridlines(
                color=t["grid"], linewidth=0.35, linestyle=(0, (3, 3)),
                xlocs=range(-180, 181, 60), ylocs=range(-60, 61, 30), zorder=2,
            )
            gl.draw_labels = False

            ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=t["land"], edgecolor="none", zorder=2.5)
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=t["coast"], linewidth=0.42, zorder=3)

            add_hex_stippling(ax, lon, lat, robust, base_stride=5, size=1.45, color=t["stipple"], alpha=0.68, zorder=4)

            ax.add_patch(Rectangle(
                (NINO34["lon"], NINO34["lat"]), NINO34["width"], NINO34["height"],
                fill=False, edgecolor=t["nino"], linewidth=0.8, linestyle=(0, (2.5, 1.5)),
                transform=ccrs.PlateCarree(), zorder=5,
            ))

            ax.spines["geo"].set_edgecolor(t["rim"])
            ax.spines["geo"].set_linewidth(0.65)

            box = ax.get_position()
            y_lbl = box.y1 + 0.012
            fig.text(box.x0 + 0.002, y_lbl, letters[i], fontsize=13.5, fontweight="bold", color=t["ink"], va="bottom")
            fig.text(box.x0 + 0.020, y_lbl + 0.001, f"{months[k - 1]:%b %Y}  (lead {k})",
                     fontsize=11.0, color=t["ink"], va="bottom")
            nino_txt = f"{nino[k - 1]:+.2f}".replace("-", "\u2212")
            fig.text(box.x1 - 0.002, y_lbl + 0.001, f"Niño 3.4: {nino_txt} °C",
                     fontsize=9.8, color=t["ink2"], va="bottom", ha="right")

        # Centered outlined IPCC discrete colourbar with triangular ends
        cax = fig.add_axes([0.22, 0.092, 0.44, 0.024])
        cb = fig.colorbar(
            cs_ref, cax=cax, orientation="horizontal", extend="both",
            extendfrac=0.042, drawedges=False,
        )
        ticks = LEVELS[::2]  # -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0
        cb.set_ticks(ticks)
        cb.ax.minorticks_off()
        cb.ax.xaxis.set_major_formatter(FuncFormatter(clean_tick))
        cb.outline.set_visible(True)
        cb.outline.set_linewidth(0.65)
        cb.outline.set_edgecolor(t["ink"])
        cb.ax.tick_params(labelsize=9.5, length=3.2, width=0.65, pad=2.5, direction="out", colors=t["ink"])
        for patch in getattr(cb, "_extend_patches", []):
            patch.set_edgecolor(t["ink"])
            patch.set_linewidth(0.65)
        cb.set_label("SST anomaly (°C, vs ERA5 1991–2020 climatology)", fontsize=10.2, color=t["ink"], labelpad=4)

        # Compact IPCC-style legend next to the colourbar
        lax = fig.add_axes([0.685, 0.068, 0.295, 0.068])
        lax.axis("off")
        leg_ink = t.get("legend_dot", t["stipple"])
        handles = [
            Line2D([], [], ls="none", marker="o", ms=3.4, color=leg_ink, alpha=0.88,
                   label="Robust signal (|ensemble mean| ≥ σ)"),
            Line2D([], [], color=leg_ink, lw=0.95, ls=(0, (2.5, 1.5)),
                   label="Niño 3.4 region (5°S–5°N, 170°–120°W)"),
        ]
        lax.legend(
            handles=handles, loc="center left", frameon=True, fancybox=False,
            facecolor=t["legend_bg"], edgecolor=t["legend_ec"], framealpha=0.95,
            fontsize=8.8, labelcolor=t["ink2"], handlelength=1.8, handletextpad=0.6,
            borderpad=0.5, labelspacing=0.35,
        )

        # Clean journal footer (exact journal-significance.png style)
        fig.text(
            0.020, 0.020,
            f"CAS-Canglong 0.25° Ensemble (32 members), initialized with ERA5 through {init:%b %Y} "
            f"({months[0]:%b %Y}–{months[-1]:%b %Y}). Stippling denotes robust ensemble signal (|mean| ≥ σ); "
            "research forecast, not an official product.",
            fontsize=8.3, color=t["muted"], ha="left", va="center",
        )
        fig.text(0.980, 0.020, "canglong.earth", fontsize=8.5, color=t["muted"], ha="right", va="center")

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor=t["surface"], pil_kwargs={"optimize": True})
        plt.close(fig)


if __name__ == "__main__":
    main()
