"""Render the Hugging Face Space assets from one CAS-Canglong forecast run.

    python hf_space/scripts/build_assets.py --sst canglong_sst_init202609.nc --iri iri_latest.json \
        --manifest showcase_manifest.json --out hf_space/build

Writes `data/` (NetCDF, Nino3.4 CSV, provenance JSON) and `assets/maps/*.png` (Robinson maps) into --out.
The six-week atmospheric run is fetched from canglong.earth through WeatherPre (`wp.forecast("canglong", ...)`)
unless --atmos points to a saved copy.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, MUTED, LAND, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#0d366b", "#2a78d6", "#86b6ef", "#f0efec", "#f3a5a1", "#e34948", "#8c1c1c"])
WARM = LinearSegmentedColormap.from_list("warm", ["#fdf3e7", "#f7c99b", "#eb6834", "#b8421b", "#5e1e0d"])
BLUE = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 10,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})


def month_label(t) -> str:
    return pd.Timestamp(t).strftime("%b %Y")


def robinson_map(field, lat, lon, path: Path, *, cmap, levels, extend, cbar_label, title, subtitle, footnote, land=True):
    fig = plt.figure(figsize=(10, 5.6), dpi=150, facecolor=SURFACE)
    ax = fig.add_axes([0.02, 0.17, 0.96, 0.70], projection=ccrs.Robinson(central_longitude=180))
    ax.set_facecolor(LAND if land else SURFACE)   # SST: cells the model treats as land but Natural Earth does not
    norm = BoundaryNorm(levels, cmap.N, extend=extend)
    lon_c = np.append(lon, lon[0] + 360.0)
    data_c = np.concatenate([field, field[:, :1]], axis=1)
    mesh = ax.pcolormesh(lon_c, lat, data_c, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="nearest",
                         rasterized=True)
    if land:
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND, edgecolor="none", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=INK2, linewidth=0.35, zorder=3)
    ax.spines["geo"].set_edgecolor(GRID)
    ax.spines["geo"].set_linewidth(0.6)
    ax.set_global()
    fig.text(0.03, 0.945, title, fontsize=14, fontweight="bold", color=INK, va="center")
    fig.text(0.03, 0.895, subtitle, fontsize=10, color=INK2, va="center")
    cax = fig.add_axes([0.25, 0.105, 0.50, 0.028])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend=extend)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, length=0, color=MUTED)
    cb.set_label(cbar_label, fontsize=9, color=INK2)
    fig.text(0.03, 0.03, footnote, fontsize=7.5, color=MUTED, va="center")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def build_sst(ds: xr.Dataset, out: Path, init: str) -> list[dict]:
    lat, lon = ds.lat.values, ds.lon.values
    lat2, lon2 = ds.lat025.values, ds.lon025.values
    foot = f"CAS-Canglong · input: ERA5 monthly means through {init} · research forecast, not an official product"
    ssta_levels = np.arange(-3.0, 3.01, 0.25)
    sst_levels = np.arange(-2.0, 32.01, 2.0)
    frames = [("t00", f"{month_label(ds.obs_time.values[-1])} (observed, ERA5)", ds.observed_ssta.values,
               ds.observed_sst_025.values - 273.15, "Observed ERA5 monthly mean — the last input month")]
    for i, t in enumerate(ds.time.values):
        frames.append((f"t{i + 1:02d}", f"{month_label(t)} (lead {i + 1})", ds.ssta.isel(time=i).values,
                       ds.sst_025.isel(time=i).values - 273.15, f"Forecast for {month_label(t)}, {i + 1}-month lead"))
    index = []
    for key, label, ssta, sst, sub in frames:
        robinson_map(ssta, lat, lon, out / f"assets/maps/ssta_{key}.png", cmap=DIVERGING, levels=ssta_levels,
                     extend="both", cbar_label="SST anomaly (°C) vs ERA5 1959–2016 monthly climatology",
                     title=f"SST anomaly · {label}", subtitle=sub + " · native 1° model grid", footnote=foot)
        sst = np.maximum(sst, -1.8)
        robinson_map(sst, lat2, lon2, out / f"assets/maps/sst_{key}.png", cmap=WARM, levels=sst_levels, extend="both",
                     cbar_label="Sea surface temperature (°C)", title=f"Sea surface temperature · {label}",
                     subtitle=sub + " · 0.25°, coastal belt blended with ERA5 for display (as on canglong.earth)",
                     footnote=foot + " · values below the −1.8 °C freezing point shown as −1.8 °C")
        index.append({"key": key, "label": label})
        print("map", key, flush=True)
    return index


ATMOS = {
    "t2m": ("2 m temperature", "°C", lambda x: x - 273.15, DIVERGING, np.arange(-40, 41, 5), "both"),
    "tp": ("Precipitation", "mm/day", lambda x: x, BLUE, np.array([0, 0.5, 1, 2, 4, 6, 8, 10, 15, 20]), "max"),
    "msl": ("Mean sea-level pressure", "hPa", lambda x: x, DIVERGING, np.arange(983, 1044, 4), "both"),
    "z500": ("500 hPa geopotential height", "m", lambda x: x, BLUE.reversed(), np.arange(4900, 5951, 50), "both"),
}


def build_atmos(u: xr.Dataset, out: Path) -> list[dict]:
    u = u.sortby("latitude")
    lat, lon = u.latitude.values, u.longitude.values
    init = pd.Timestamp(u.init_time.values).strftime("%Y-%m-%d")
    foot = (f"CAS-Canglong atmosphere · start {init} · weekly means from canglong.earth via WeatherPre · "
            "not drift-corrected; research forecast")
    weeks = []
    for w in u.week.values:
        start = pd.Timestamp(u.valid_start.sel(week=w).values)
        end = pd.Timestamp(u.valid_end.sel(week=w).values) - pd.Timedelta(days=1)
        label = f"week {int(w)}: {start:%d %b}–{end:%d %b %Y}"
        for v, (name, units, conv, cmap, levels, extend) in ATMOS.items():
            if v not in u: continue
            robinson_map(conv(u[v].sel(week=w).values), lat, lon, out / f"assets/maps/atm_{v}_w{int(w)}.png", cmap=cmap,
                         levels=levels, extend=extend, cbar_label=f"{name} ({units})", title=f"{name} · {label}",
                         subtitle="Weekly mean, 1° grid", footnote=foot, land=False)
        weeks.append({"week": int(w), "label": label})
        print("atmos week", int(w), flush=True)
    return weeks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sst", type=Path, required=True)
    p.add_argument("--iri", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--atmos", type=Path)
    p.add_argument("--out", type=Path, default=Path("hf_space/build"))
    a = p.parse_args()
    data = a.out / "data"
    data.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(a.sst)
    init = pd.Timestamp(ds.obs_time.values[-1]).strftime("%Y-%m")
    shutil.copy2(a.sst, data / a.sst.name)
    shutil.copy2(a.iri, data / "iri_plume.json")

    if a.atmos:
        u = xr.open_dataset(a.atmos)
    else:
        import weatherpre as wp
        u = wp.forecast("canglong", ["t2m", "tp", "msl", "z500"], "6w")
    atmos_name = f"canglong_atmos_start{pd.Timestamp(u.init_time.values):%Y%m%d}.nc"
    u.to_netcdf(data / atmos_name)

    nino = pd.DataFrame({"valid_month": [pd.Timestamp(t).strftime("%Y-%m") for t in ds.time.values],
                         "lead_month": np.arange(1, ds.sizes["time"] + 1),
                         "nino34_degC": ds.nino34.values.round(3), "nino34_1deg_degC": ds.nino34_1deg.values.round(3)})
    nino.to_csv(data / "nino34.csv", index=False)
    run = json.loads(a.manifest.read_text())
    meta = {"sst_file": a.sst.name, "atmos_file": atmos_name, "initialization": init,
            "atmos_start": pd.Timestamp(u.init_time.values).strftime("%Y-%m-%d"), "atmos_source": u.attrs.get("source"),
            "atmos_release": u.attrs.get("release"), "run_time_utc": run.get("run_time_utc"),
            "checkpoint_sha256": run.get("checkpoint_sha256"), "input_cache_sha256": run.get("input_cache_sha256"),
            "check_vs_operational": run.get("check_vs_operational"),
            "frames": build_sst(ds, a.out, init), "weeks": build_atmos(u, a.out)}
    (data / "run.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("frames", "weeks")}, indent=1))


if __name__ == "__main__":
    main()
