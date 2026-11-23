"""Download the latest CAS-Canglong 0.25° Ensemble SST forecast from the Hugging Face Space and plot it (no GPU, no ERA5 needed).

    pip install huggingface_hub xarray netCDF4 scipy matplotlib cartopy
    python scripts/plot_latest_forecast.py                      # -> assets/readme/latest_forecast.png
    python scripts/plot_latest_forecast.py --lead 3 --out dec.png

Left: smoothed global SST anomaly map for one forecast month (vs ERA5 1991–2020). Right: 16-month monthly Niño 3.4
ensemble plume (mean, 50% IQR, and 90% CI) from CAS-Canglong 0.25° Ensemble with the IRI ENSO plume; dashed connectors
radiate from the single latest ERA5 observation point to every forecast trajectory.
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
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402

SPACE = "LonghaoWang/CAS-Canglong"
SURFACE, INK, INK2, MUTED, LAND, GRID = "#fcfcfb", "#141413", "#4e4d49", "#85837c", "#e4e3dc", "#e6e5de"
COAST, RIM = "#8f8d85", "#c8c6be"
CAS, DYN, STAT, PLUME = "#b80000", "#eb6834", "#1baf7a", "#cfcdc4"
DIVERGING = LinearSegmentedColormap.from_list(
    "ssta_clean",
    ["#0a2c5a", "#1d5ca3", "#5898dc", "#b5d4f4", "#f5f4f0", "#f8c5bd", "#e56a61", "#c22428", "#6d0d14"],
    N=256,
)
plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})


def fetch(name: str, local_dir: Path | None = None) -> Path:
    if local_dir is not None:
        candidate = local_dir / name
        if candidate.exists():
            return candidate
    return Path(hf_hub_download(SPACE, name, repo_type="space"))


def season_center(season: str) -> pd.Timestamp:
    year, code = season.split("-")
    order = "JFMAMJJASOND"
    if len(code) == 3 and code.isupper():
        i = next(i for i in range(12) if "".join(order[(i + k) % 12] for k in range(3)) == code)
        return pd.Timestamp(int(year), i + 1, 1) + pd.DateOffset(months=1)
    return pd.Timestamp(f"{year}-{code}-01")


def smooth_field(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, step: float = 0.25,
                 sigma: float = 0.55, reach: float = 1.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lead", type=int, default=3, help="forecast month to map, 1..16 (default 3)")
    p.add_argument("--out", type=Path, default=Path("assets/readme/latest_forecast.png"))
    p.add_argument("--local-dir", type=Path, default=None, help="optional local Space root containing data/")
    a = p.parse_args()

    run = json.loads(fetch("data/run.json", a.local_dir).read_text(encoding="utf-8"))
    ds = xr.open_dataset(fetch(f"data/{run['sst_file']}", a.local_dir))
    iri = json.loads(fetch("data/iri_plume.json", a.local_dir).read_text(encoding="utf-8"))

    months = pd.to_datetime(ds.time.values)
    nino = ds.nino34_9120.values.astype(float)
    hist_t = pd.to_datetime(ds.hist_time.values)
    hist_v = ds.nino34_history_9120.values.astype(float)
    t_init, v_init = pd.Timestamp(hist_t[-1]), float(hist_v[-1])
    k = a.lead - 1
    peak_i = int(np.argmax(nino))
    trough_i = int(np.argmin(nino))

    fig = plt.figure(figsize=(14.4, 5.45), dpi=180, facecolor=SURFACE)
    fig.text(0.022, 0.942, "CAS-Canglong 0.25° Ensemble · Global SST & Niño 3.4 Outlook",
             fontsize=14.8, fontweight="bold", color=INK, va="center")
    fig.text(
        0.022, 0.888,
        f"ERA5 input through {t_init:%b %Y} (obs {v_init:+.2f} °C) · 32-member 16-month forecast "
        f"({months[0]:%b %Y} – {months[-1]:%b %Y}) · Peak {months[peak_i]:%b %Y} ({nino[peak_i]:+.2f} °C)",
        fontsize=10.1, color=INK2, va="center",
    )

    # --- Left panel: smoothed Pacific-centred Robinson SST anomaly map ---
    ax = fig.add_axes([0.015, 0.178, 0.505, 0.625], projection=ccrs.Robinson(central_longitude=200))
    ax.set_facecolor(LAND)
    field_hi, lat_hi, lon_hi = smooth_field(
        ds.ssta_9120.isel(time=k).values.astype(float), ds.lat.values.astype(float), ds.lon.values.astype(float)
    )
    levels = np.arange(-3.0, 3.01, 0.25)
    mesh = ax.pcolormesh(
        lon_hi, lat_hi, field_hi, cmap=DIVERGING,
        norm=BoundaryNorm(levels, DIVERGING.N, extend="both"),
        transform=ccrs.PlateCarree(), shading="nearest", rasterized=True, zorder=1,
    )
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND, edgecolor="none", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=COAST, linewidth=0.35, zorder=3)
    ax.add_patch(Rectangle(
        (190, -5), 50, 10, fill=False, edgecolor="#1c1b18", linewidth=1.0,
        linestyle=(0, (3.0, 2.0)), transform=ccrs.PlateCarree(), zorder=4,
    ))
    ax.text(
        215, -13.5, "Niño 3.4", transform=ccrs.PlateCarree(), ha="center", va="top",
        fontsize=8.2, fontweight="bold", color="#262522", zorder=5,
        path_effects=[pe.withStroke(linewidth=2.4, foreground=SURFACE, alpha=0.85)],
    )
    ax.spines["geo"].set_edgecolor(RIM)
    ax.spines["geo"].set_linewidth(0.7)
    ax.set_global()

    map_box = ax.get_position()
    fig.text(map_box.x0 + 0.008, 0.824, f"SST anomaly · {months[k]:%b %Y} ({a.lead}-month lead)",
             fontsize=11.2, fontweight="bold", color=INK, va="bottom")

    cax = fig.add_axes([map_box.x0 + 0.075, 0.114, map_box.width - 0.15, 0.023])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both", ticks=[-3, -2, -1, 0, 1, 2, 3], extendfrac=0.04)
    cb.outline.set_visible(False)
    cb.ax.set_xticklabels(["−3", "−2", "−1", "0", "+1", "+2", "+3"])
    cb.ax.tick_params(labelsize=8.6, length=0, pad=3, colors=INK2)
    cb.set_label("SST anomaly (°C, vs ERA5 1991–2020 monthly climatology)", fontsize=8.6, color=INK2, labelpad=3)

    # --- Right panel: 16-month Niño 3.4 ensemble plume with dashed connectors from single obs point ---
    bx = fig.add_axes([0.578, 0.168, 0.400, 0.635])
    bx.set_facecolor(SURFACE)
    fig.text(0.578, 0.824, "Niño 3.4 SST anomaly · 16-month ensemble plume (°C)",
             fontsize=11.2, fontweight="bold", color=INK, va="bottom")

    # Neutral zone tint and El Niño / La Niña thresholds
    bx.axhspan(-0.5, 0.5, color="#f2f1eb", alpha=0.85, zorder=0)
    for y in (0.5, -0.5):
        bx.axhline(y, color="#a7a59d", lw=0.8, ls=":", zorder=0.4)
    bx.axhline(0.0, color="#c8c6bd", lw=0.75, zorder=0.4)
    bx.axvline(t_init, color="#cbc9c0", lw=0.95, ls=":", zorder=0.5)

    # 1) All individual IRI plume models connected by dashed lines from the single observation point
    for m in iri["models"]:
        mx = [season_center(s) for s in m["seasons"]]
        my = m["data"]
        if mx and my:
            bx.plot([t_init, mx[0]], [v_init, my[0]], color=PLUME, lw=0.75, ls="--", alpha=0.72, zorder=1)
            bx.plot(mx, my, color=PLUME, lw=0.85, alpha=0.88, zorder=1.1)

    # 2) Ensemble 90% CI & 50% IQR bands emerging cleanly from the single observation point
    xs_ens = [t_init, *months]
    if "nino34_p05_9120" in ds and "nino34_p95_9120" in ds:
        bx.fill_between(
            xs_ens, [v_init, *ds.nino34_p05_9120.values], [v_init, *ds.nino34_p95_9120.values],
            color="#d9383a", alpha=0.10, lw=0, zorder=1.6,
        )
    if "nino34_p25_9120" in ds and "nino34_p75_9120" in ds:
        bx.fill_between(
            xs_ens, [v_init, *ds.nino34_p25_9120.values], [v_init, *ds.nino34_p75_9120.values],
            color="#d9383a", alpha=0.22, lw=0, zorder=1.9,
        )

    # 3) IRI dynamical & statistical means with dashed connectors from the single observation point
    for avg, color in (("dynamical", DYN), ("statistical", STAT)):
        rec = next(x for x in iri["averages"] if x["model"] == avg)
        rx = [season_center(s) for s in rec["seasons"]]
        ry = rec["data"]
        bx.plot([t_init, rx[0]], [v_init, ry[0]], color=color, lw=1.6, ls="--", zorder=3.0)
        bx.plot(rx, ry, color=color, lw=2.0, marker="o", ms=3.4, mec=SURFACE, mew=0.65, zorder=3.2)

    # 4) CAS-Canglong 0.25° Ensemble mean with dashed connector from observation to lead 1
    bx.plot([t_init, months[0]], [v_init, nino[0]], color=CAS, lw=2.1, ls="--", zorder=4.0)
    bx.plot(months, nino, color=CAS, lw=2.5, marker="o", ms=4.3, mec=SURFACE, mew=0.85, zorder=4.3)

    # 5) Trailing 6 months of ERA5 observed ending at the single observation anchor diamond
    hist_n = 6
    bx.plot(hist_t[-hist_n:], hist_v[-hist_n:], color=INK, lw=2.0, zorder=5.0)
    bx.scatter([t_init], [v_init], marker="D", s=46, color=INK, edgecolor=SURFACE, linewidth=1.2, zorder=6.0)

    # Clean, non-overlapping annotations
    bx.set_ylim(-4.95, 4.75)
    bx.set_xlim(hist_t[-hist_n] - pd.Timedelta(days=12), months[-1] + pd.Timedelta(days=18))
    bx.text(t_init - pd.Timedelta(days=8), 0.62, "El Niño (+0.5°C)", fontsize=7.6, color="#9a5c46", ha="right", va="bottom")
    bx.text(t_init - pd.Timedelta(days=8), -0.64, "La Niña (−0.5°C)", fontsize=7.6, color="#497099", ha="right", va="top")
    bx.text(t_init - pd.Timedelta(days=10), 4.48, "Observed", fontsize=8.0, color=MUTED, ha="right", va="top")
    bx.text(t_init + pd.Timedelta(days=10), 4.48, "Forecast (16 mo)", fontsize=8.0, color=MUTED, ha="left", va="top")

    bx.annotate(
        f"Obs {v_init:+.2f}°C",
        xy=(t_init, v_init), xytext=(t_init - pd.Timedelta(days=78), v_init + 0.72),
        fontsize=8.0, fontweight="bold", color=INK, ha="center",
        arrowprops=dict(arrowstyle="-", color="#9f9d95", lw=0.75, shrinkA=2, shrinkB=5),
        zorder=7,
    )
    bx.annotate(
        f"Peak {months[peak_i]:%b %Y}\n{nino[peak_i]:+.2f}°C",
        xy=(months[peak_i], nino[peak_i]), xytext=(months[peak_i] - pd.Timedelta(days=26), 1.38),
        fontsize=7.9, fontweight="bold", color=CAS, ha="center", va="top",
        arrowprops=dict(arrowstyle="-", color=CAS, lw=0.8, alpha=0.75, shrinkA=2, shrinkB=6),
        zorder=7,
    )
    bx.annotate(
        f"{months[trough_i]:%b %Y}: {nino[trough_i]:+.2f}°C",
        xy=(months[trough_i], nino[trough_i]), xytext=(months[trough_i] - pd.Timedelta(days=145), nino[trough_i] - 0.92),
        fontsize=8.0, fontweight="bold", color=CAS, ha="center",
        arrowprops=dict(arrowstyle="-", color=CAS, lw=0.8, alpha=0.75, shrinkA=2, shrinkB=5),
        zorder=7,
    )

    bx.set_yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    bx.set_yticklabels(["−4", "−3", "−2", "−1", "0", "+1", "+2", "+3", "+4"])
    bx.grid(axis="y", color=GRID, lw=0.65, zorder=0)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.spines["left"].set_color("#d2d0c7")
    bx.spines["bottom"].set_color("#d2d0c7")
    bx.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    bx.tick_params(labelsize=8.4, length=0, pad=4)

    handles = [
        Line2D([0], [0], color=INK, lw=2.0, marker="D", ms=5.0, mec=SURFACE, mew=0.8, label="ERA5 observed"),
        Line2D([0], [0], color=CAS, lw=2.5, marker="o", ms=4.5, mec=SURFACE, mew=0.8, label="CAS-Canglong 0.25° Ensemble"),
        Patch(facecolor="#d9383a", alpha=0.26, edgecolor="none", label="50% IQR / 90% CI (32 members)"),
        Line2D([0], [0], color=DYN, lw=2.0, marker="o", ms=3.8, mec=SURFACE, mew=0.7, label="IRI dynamical mean (3-mo)"),
        Line2D([0], [0], color=STAT, lw=2.0, marker="o", ms=3.8, mec=SURFACE, mew=0.7, label="IRI statistical mean (3-mo)"),
        Line2D([0], [0], color=PLUME, lw=1.0, label="IRI individual models"),
    ]
    bx.legend(
        handles=handles, loc="upper right", frameon=True, facecolor=SURFACE,
        framealpha=0.94, edgecolor="#e8e7e0", fontsize=7.8, labelcolor=INK2,
        handlelength=1.9, borderpad=0.55, labelspacing=0.38,
    )

    fig.text(0.022, 0.028, "Research forecast, not an official product · CAS-Canglong 0.25° Ensemble (32-member mean)",
             fontsize=8.0, color=MUTED, va="center")
    fig.text(0.978, 0.028, f"IRI plume published {iri['published']} (redrawn) · dashed lines connect latest ERA5 obs to each forecast",
             fontsize=8.0, color=MUTED, va="center", ha="right")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=SURFACE, pil_kwargs={"optimize": True})
    plt.close(fig)
    print(a.out)


if __name__ == "__main__":
    main()
