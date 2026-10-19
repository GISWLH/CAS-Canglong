"""Plot the global SST outlook of the latest CAS-Canglong run as six maps (no GPU, no ERA5 needed).

    pip install huggingface_hub xarray netCDF4 scipy matplotlib cartopy
    python scripts/plot_global_outlook.py                       # -> assets/readme/global_outlook{,_dark}.png
    python scripts/plot_global_outlook.py --leads 1 2 3 4 5 6 --theme light

Every third forecast month from lead 1 to lead 16 on a Pacific-centred Robinson projection (cut at 20°E, the
Atlantic-Indian boundary), with isotherms every 5 °C and the 28 °C isotherm that bounds the warm pool. The native 1°
SST is Gaussian-smoothed (σ = 1.5 grid cells) and resampled bicubically to 0.25° for display only; cells more than 1.5°
from a model ocean point (sea ice, inland) stay blank.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402

SPACE = "LonghaoWang/CAS-Canglong"
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", land="#dcdbd4",
                  coast="#a9a79f", rim="#c9c7bf", iso="#3d2a20", warm_pool="#ffffff",
                  ramp=["#eef3f6", "#f7e6cc", "#f3c08a", "#e9874b", "#cc5229", "#93311a", "#4f170b"]),
    "dark": dict(surface="#0d1117", ink="#e6edf3", ink2="#9da7b3", muted="#6e7781", land="#161b22",
                 coast="#484f58", rim="#30363d", iso="#0d1117", warm_pool="#0d1117",
                 ramp=["#1d2a33", "#4a2f24", "#83391f", "#c2532a", "#ec7f45", "#f8b77c", "#fde6c4"]),
}
NINO34 = dict(lon=190, lat=-5, width=50, height=10)          # 170°W-120°W, 5°S-5°N


def fetch(name: str) -> Path:
    return Path(hf_hub_download(SPACE, name, repo_type="space"))


def smooth(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, step: float = 0.25, sigma: float = 1.5,
           reach: float = 1.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian-smooth a 1° ocean field and resample it bicubically to `step` degrees, keeping the model's ocean."""
    valid = np.isfinite(field)
    dist, idx = ndimage.distance_transform_edt(~valid, return_indices=True)
    filled = ndimage.gaussian_filter(field[tuple(idx)], sigma, mode=("nearest", "wrap"))
    pad = 3
    filled = np.concatenate([filled[:, -pad:], filled, filled[:, :pad]], axis=1)
    dist = np.concatenate([dist[:, -pad:], dist, dist[:, :pad]], axis=1)
    lat_hi = np.arange(lat[0], lat[-1] + step / 2, step)
    lon_hi = np.arange(lon[0], lon[0] + 360 + step / 2, step)
    rows = (lat_hi - lat[0]) / (lat[1] - lat[0])
    cols = (lon_hi - lon[0]) / (lon[1] - lon[0]) + pad
    grid = np.meshgrid(rows, cols, indexing="ij")
    out = ndimage.map_coordinates(filled, grid, order=3, mode="nearest")
    out[ndimage.map_coordinates(dist, grid, order=1, mode="nearest") > reach] = np.nan
    return out, lat_hi, lon_hi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--leads", type=int, nargs=6, default=[1, 4, 7, 10, 13, 16], help="six forecast months, 1..16")
    p.add_argument("--theme", choices=["light", "dark", "both"], default="both")
    p.add_argument("--out", type=Path, default=Path("assets/readme/global_outlook.png"),
                   help="light-theme path; the dark theme gets a _dark suffix")
    a = p.parse_args()
    run = json.loads(fetch("data/run.json").read_text())
    ds = xr.open_dataset(fetch(f"data/{run['sst_file']}"))
    months = pd.to_datetime(ds.time.values)
    nino = ds.nino34.values.astype(float)
    lat, lon = ds.lat.values.astype(float), ds.lon.values.astype(float)
    fields = {k: smooth(ds.sst.isel(time=k - 1).values - 273.15, lat, lon) for k in a.leads}
    init = pd.Timestamp(run["initialization"] + "-01")
    for theme in (["light", "dark"] if a.theme == "both" else [a.theme]):
        out = a.out if theme == "light" else a.out.with_name(f"{a.out.stem}_dark{a.out.suffix}")
        draw(fields, months, nino, init, theme, out)
        print(out)


def draw(fields: dict, months: pd.DatetimeIndex, nino: np.ndarray, init: pd.Timestamp, theme: str, out: Path) -> None:
    t = THEMES[theme]
    cmap = LinearSegmentedColormap.from_list(f"sst_{theme}", t["ramp"], N=512)
    cmap.set_bad(t["land"])
    norm = Normalize(-2, 32)
    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 10, "text.color": t["ink"]})
    fig = plt.figure(figsize=(16, 7.6), dpi=130, facecolor=t["surface"])
    fig.text(0.03, 0.935, "Global sea-surface temperature outlook", fontsize=20, fontweight="bold", va="center")
    fig.text(0.03, 0.885, f"CAS-Canglong · ERA5 input through {init:%B %Y} · every third month of the 16-month forecast",
             fontsize=12, color=t["ink2"], va="center")

    crs = ccrs.Robinson(central_longitude=200)
    w, h, gx, gy, x0, y0 = 0.305, 0.335, 0.0175, 0.06, 0.03, 0.075
    mesh = None
    for i, (k, (data, lat_hi, lon_hi)) in enumerate(fields.items()):
        r, c = divmod(i, 3)
        ax = fig.add_axes([x0 + c * (w + gx), y0 + (1 - r) * (h + gy), w, h], projection=crs)
        ax.set_global()
        ax.set_facecolor(t["land"])
        ax.spines["geo"].set_edgecolor(t["rim"])
        ax.spines["geo"].set_linewidth(0.6)
        mesh = ax.pcolormesh(lon_hi, lat_hi, data, cmap=cmap, norm=norm, shading="nearest",
                             transform=ccrs.PlateCarree(), rasterized=True, zorder=1)
        ax.contour(lon_hi, lat_hi, data, levels=np.arange(0, 26, 5), colors=t["iso"], linewidths=0.4, alpha=0.3,
                   transform=ccrs.PlateCarree(), zorder=1.4)
        ax.contour(lon_hi, lat_hi, data, levels=[28], colors=t["warm_pool"], linewidths=1.1,
                   transform=ccrs.PlateCarree(), zorder=1.5)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=t["land"], edgecolor="none", zorder=2)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=t["coast"], linewidth=0.3, zorder=3)
        ax.add_patch(Rectangle((NINO34["lon"], NINO34["lat"]), NINO34["width"], NINO34["height"], fill=False,
                               edgecolor=t["warm_pool"], linewidth=0.8, linestyle=(0, (2.5, 1.5)),
                               transform=ccrs.PlateCarree(), zorder=4))
        box = ax.get_position()
        month = fig.text(box.x0 + 0.004, box.y1 + 0.016, f"{months[k - 1]:%b %Y}", fontsize=14, fontweight="bold",
                         va="bottom")
        right = month.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).x1
        fig.text(right + 0.008, box.y1 + 0.016, f"lead {k}", fontsize=12, color=t["muted"], va="bottom")
        fig.text(box.x1 - 0.004, box.y1 + 0.016, f"Niño 3.4 index  {nino[k - 1]:+.2f} °C", fontsize=12, color=t["ink2"],
                 va="bottom", ha="right")

    cax = fig.add_axes([0.775, 0.927, 0.195, 0.02])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both", ticks=np.arange(0, 31, 5),
                      extendfrac=0.05)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=10, length=0, pad=4, colors=t["ink2"])
    fig.text(0.765, 0.937, "SST (°C)", fontsize=11, color=t["ink2"], va="center", ha="right")
    fig.text(0.03, 0.035, "Research forecast, not an official product. Native 1° output, smoothed (σ = 1.5°) and resampled to "
             "0.25° for display. Isotherms every 5 °C; bold line: 28 °C; dashed box: Niño 3.4 region.",
             fontsize=10, color=t["muted"], va="center")
    fig.text(0.97, 0.035, "canglong.earth", fontsize=10, color=t["muted"], va="center", ha="right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=t["surface"], pil_kwargs={"optimize": True})
    plt.close(fig)


if __name__ == "__main__":
    main()
