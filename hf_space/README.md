---
title: CAS-Canglong SST & ENSO Forecast
emoji: 🐉
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "6.30.0"
python_version: "3.12"
app_file: app.py
pinned: true
license: mit
short_description: Live 16-month global SST and ENSO forecast from CAS-Canglong
tags:
  - weather
  - climate
  - enso
  - sea-surface-temperature
  - s2s
  - forecasting
  - earth-science
---

# 🐉 CAS-Canglong · global SST & ENSO forecast

The current operational run of **CAS-Canglong**, a 3-D Earth-specific transformer for sub-seasonal to seasonal prediction
of global sea-surface temperature ([*Advanced Science* 2026](https://doi.org/10.1002/advs.78271) ·
[code](https://github.com/GISWLH/CAS-Canglong) · live portal [canglong.earth](https://canglong.earth)).

* **Ocean & ENSO** — 16 monthly SST and SST-anomaly maps (1° model grid and 0.25° refined field) and the Niño 3.4
  outlook against the IRI ENSO plume.
* **Regional outlook** — Niño regions, the Indian Ocean Dipole, the tropical North Atlantic, the Kuroshio, the
  Mediterranean and more, month by month.
* **Atmosphere · 6 weeks (experimental)** — weekly 2 m temperature, precipitation, sea-level pressure and 500 hPa height
  from the CAS-Canglong atmospheric run, read with [WeatherPre](https://github.com/GISWLH/WeatherPre).
* **Data** — the NetCDF and CSV files behind every map.

The fields are identical to those on canglong.earth. They come from re-running the frozen checkpoints on a GPU
workstation; the Space only displays them. Assets are rebuilt with
[`hf_space/scripts/build_assets.py`](https://github.com/GISWLH/CAS-Canglong/tree/main/hf_space).
This is a research forecast, not an official warning product; see the *About* tab for limitations.
