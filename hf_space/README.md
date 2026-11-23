---
title: CAS-Canglong 0.25° V2.6 · SST & ENSO Forecast
emoji: 🐉
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "6.30.0"
python_version: "3.12"
app_file: app.py
pinned: true
license: mit
short_description: 16-month 0.25° global SST & 32-member AI ensemble outlook
models:
  - LonghaoWang/CAS-Canglong-SST
tags:
  - weather
  - climate
  - enso
  - sea-surface-temperature
  - s2s
  - forecasting
  - earth-science
  - ensemble
---

# 🐉 CAS-Canglong 0.25° V2.6 · 16-Month Global SST & ENSO Forecast

Live showcase of **CAS-Canglong 0.25° V2.6** — a native 0.25° 16-month Dual-Branch Pareto coupled ocean–atmosphere model
with a flow-dependent **32-member AI ensemble** ([single-file `.tar.gz` weights](https://huggingface.co/LonghaoWang/CAS-Canglong-SST) ·
[paper, *Advanced Science* 2026](https://doi.org/10.1002/advs.78271) · [GitHub](https://github.com/GISWLH/CAS-Canglong) ·
[WeatherAI port](https://github.com/GISWLH/WeatherAI/blob/main/docs/models/canglong.md) ·
live 3-D portal [canglong.earth](https://canglong.earth)).

* **Ocean & ENSO (16m)** — 16 monthly maps of SST, SST anomaly, top-quintile exceedance probability, and 32-member AI
  ensemble spread (`σ`), plus the 16-month Niño 3.4 AI ensemble plume (mean, 50% IQR, 90% CI) against the IRI ENSO plume.
* **Regional outlook** — Niño 3.4/3/4/1+2, Indian Ocean Dipole (DMI), Tropical North Atlantic, Kuroshio, South China Sea,
  Mediterranean, and Global ocean with the trailing 12 observed ERA5 months.
* **Skill & hindcasts** — strictly out-of-sample verification (`train ≤ 2009-08`, `val ≤ 2016-04`, untouched `2016-05..2026-09`)
  across all 8 paper regions vs CMA, ECMWF, UKMO, BoM, NMME, and ERA5 Persistence, plus lead 1–16 Niño 3.4 hindcasts.
* **Atmosphere · 6 weeks (experimental)** — weekly 2 m temperature, precipitation, sea-level pressure, and 500 hPa height
  from the CAS-Canglong atmospheric run via [WeatherPre](https://github.com/GISWLH/WeatherPre).
* **Model & data** — NetCDF, CSVs, and copy-paste Python snippets to load the Space dataset or run the 273 MB `.tar.gz`
  checkpoint locally.

This is a research forecast, not an official or operational warning product.
