"""Download the latest CAS-Canglong SST forecast from the Hugging Face Space and plot it (no GPU, no ERA5 needed).

    pip install huggingface_hub xarray netCDF4 matplotlib cartopy
    python scripts/plot_latest_forecast.py                      # -> assets/readme/latest_forecast.png
    python scripts/plot_latest_forecast.py --lead 4 --out jan.png

Left: SST anomaly for one forecast month (native 1° grid, vs ERA5 1959-2016). Right: monthly Niño 3.4 from
CAS-Canglong with the IRI ENSO plume means published at the same time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap  # noqa: E402

SPACE = "LonghaoWang/CAS-Canglong"
SURFACE, INK, INK2, MUTED, LAND, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#e1e0d9"
CAS, DYN, STAT = "#2a78d6", "#eb6834", "#1baf7a"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#0d366b", "#2a78d6", "#86b6ef", "#f0efec", "#f3a5a1", "#e34948", "#8c1c1c"])
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})


def fetch(name: str) -> Path:
    return Path(hf_hub_download(SPACE, name, repo_type="space"))


def season_center(season: str) -> pd.Timestamp:
    year, code = season.split("-")
    order = "JFMAMJJASOND"
    if len(code) == 3 and code.isupper():
        i = next(i for i in range(12) if "".join(order[(i + k) % 12] for k in range(3)) == code)
        return pd.Timestamp(int(year), i + 1, 1) + pd.DateOffset(months=1)
    return pd.Timestamp(f"{year}-{code}-01")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lead", type=int, default=3, help="forecast month to map, 1..16 (default 3)")
    p.add_argument("--out", type=Path, default=Path("assets/readme/latest_forecast.png"))
    a = p.parse_args()
    run = json.loads(fetch("data/run.json").read_text())
    ds = xr.open_dataset(fetch(f"data/{run['sst_file']}"))
    iri = json.loads(fetch("data/iri_plume.json").read_text())
    months = pd.to_datetime(ds.time.values)
    nino = ds.nino34.values.astype(float)
    k = a.lead - 1

    fig = plt.figure(figsize=(13, 4.9), dpi=160, facecolor=SURFACE)
    fig.text(0.02, 0.95, f"CAS-Canglong forecast · ERA5 input through {run['initialization']}", fontsize=14,
             fontweight="bold", va="center")
    fig.text(0.02, 0.895, f"Niño 3.4 {months[0]:%b %Y}: {nino[0]:+.2f} °C · {months[-1]:%b %Y}: {nino[-1]:+.2f} °C · "
             "research forecast, not an official product", fontsize=10, color=INK2, va="center")

    ax = fig.add_axes([0.0, 0.11, 0.58, 0.68], projection=ccrs.Robinson(central_longitude=180))
    ax.set_facecolor(LAND)
    field = ds.ssta.isel(time=k).values
    lon = np.append(ds.lon.values, ds.lon.values[0] + 360)
    mesh = ax.pcolormesh(lon, ds.lat.values, np.concatenate([field, field[:, :1]], axis=1), cmap=DIVERGING,
                         norm=BoundaryNorm(np.arange(-3, 3.01, 0.25), DIVERGING.N, extend="both"),
                         transform=ccrs.PlateCarree(), shading="nearest", rasterized=True)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND, edgecolor="none", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=INK2, linewidth=0.3, zorder=3)
    ax.spines["geo"].set_edgecolor("#c3c2b7")
    ax.set_global()
    ax.set_title(f"SST anomaly · {months[k]:%b %Y} ({a.lead}-month lead)", fontsize=11, loc="left", color=INK, pad=8)
    cax = fig.add_axes([0.13, 0.07, 0.32, 0.025])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    cb.set_label("°C vs ERA5 1959–2016 monthly climatology", fontsize=8.5, color=INK2)

    bx = fig.add_axes([0.63, 0.17, 0.33, 0.64])
    bx.set_facecolor(SURFACE)
    for m in iri["models"]:
        bx.plot([season_center(s) for s in m["seasons"]], m["data"], color="#d9d8d2", lw=0.8, zorder=1)
    for avg, color in (("dynamical", DYN), ("statistical", STAT)):
        rec = next(x for x in iri["averages"] if x["model"] == avg)
        bx.plot([season_center(s) for s in rec["seasons"]], rec["data"], color=color, lw=1.8, zorder=3,
                label=f"IRI {avg} mean (3-month)")
    bx.plot(months, nino, color=CAS, lw=2.2, marker="o", ms=4, mec=SURFACE, mew=1, zorder=4, label="CAS-Canglong (monthly)")
    obs = [(season_center(o["season"]), o["data"]) for o in iri["observed"]]
    bx.scatter([o[0] for o in obs], [o[1] for o in obs], marker="D", s=22, color=INK, zorder=5, label="observed (IRI)")
    for y in (0.5, -0.5):
        bx.axhline(y, color=MUTED, lw=0.8, ls=":", zorder=0)
    bx.axhline(0, color="#c3c2b7", lw=0.8, zorder=0)
    bx.annotate("CAS-Canglong", (months[-1], nino[-1]), xytext=(6, 0), textcoords="offset points", color=INK,
                fontsize=9, va="center")
    bx.set_xlim(right=months[-1] + pd.DateOffset(months=4))
    bx.set_title("Niño 3.4 SST anomaly (°C)", fontsize=11, loc="left", color=INK)
    bx.grid(axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    bx.tick_params(labelsize=8.5, length=0)
    bx.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK2)
    fig.text(0.63, 0.04, f"IRI plume published {iri['published']} (redrawn) · CAS index vs ERA5 1959–2016",
             fontsize=7.5, color=MUTED)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=SURFACE)
    print(a.out)


if __name__ == "__main__":
    main()
