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
short_description: 12-month global SST & ENSO outlook with probabilities
models:
  - LonghaoWang/CAS-Canglong-SST
datasets:
  - LonghaoWang/SST-Hindcast-2016-2022-1deg-lead1-12
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

The latest run of the **2026-10-25 release of the CAS-Canglong 1° SST model**
([weights and code](https://huggingface.co/LonghaoWang/CAS-Canglong-SST) ·
[2016–2022 hindcast](https://huggingface.co/datasets/LonghaoWang/SST-Hindcast-2016-2022-1deg-lead1-12) ·
[paper, *Advanced Science* 2026](https://doi.org/10.1002/advs.78271) · [code](https://github.com/GISWLH/CAS-Canglong) ·
live portal [canglong.earth](https://canglong.earth)).

* **Ocean & ENSO** — 12 monthly maps of SST, SST anomaly, the chance of a top-quintile month and the forecast spread, and
  the Niño 3.4 outlook against the IRI ENSO plume.
* **Regional outlook** — Niño regions, the Indian Ocean Dipole, the tropical North Atlantic, the Kuroshio, the
  Mediterranean and more, with the last 12 observed months.
* **Skill & hindcasts** — a held-out test on 2023–2026 targets and the 2016–2022 Niño 3.4 hindcast, lead by lead.
* **Atmosphere · 6 weeks (experimental)** — weekly 2 m temperature, precipitation, sea-level pressure and 500 hPa height
  from the CAS-Canglong atmospheric run, read with [WeatherPre](https://github.com/GISWLH/WeatherPre).
* **Model & data** — the NetCDF and CSV files behind every map, and code to load the model and the hindcast.

The Space only displays precomputed fields; they come from the frozen model run on a GPU workstation and are rebuilt with
[`hf_space/scripts/build_assets.py`](https://github.com/GISWLH/CAS-Canglong/tree/main/hf_space). This is a research
forecast, not an official warning product; see the *About* tab for its limits, in particular for ENSO.
