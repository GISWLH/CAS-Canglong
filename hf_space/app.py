"""CAS-Canglong forecast showcase (Hugging Face Space).

Everything shown is read from `data/` and `assets/`, written by `scripts/build_assets.py` from one forecast release.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr

ROOT = Path(__file__).resolve().parent
DATA, MAPS = ROOT / "data", ROOT / "assets" / "maps"
RUN = json.loads((DATA / "run.json").read_text())
SST = xr.open_dataset(DATA / RUN["sst_file"]).load()
IRI = json.loads((DATA / "iri_plume.json").read_text())
HINDCAST = pd.read_csv(DATA / "hindcast_nino34.csv", parse_dates=["target_month"])
HINDCAST_SKILL = pd.read_csv(DATA / "hindcast_nino34_skill.csv")
HELDOUT = pd.read_csv(DATA / "round6_heldout_metrics.csv")
MODEL_REPO, HINDCAST_REPO = RUN["model_repo"], RUN["hindcast_dataset"]

SURFACE, GRIDLINE, BASELINE, INK, INK2, MUTED = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#0b0b0b", "#52514e", "#898781"
CAS_BLUE, DYN_ORANGE, STAT_AQUA, CONTEXT = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"

FRAMES = RUN["frames"]                       # index 0 = observed last input month, 1..12 = forecast leads
WEEKS = RUN["weeks"]
MONTHS = pd.to_datetime(SST.time.values)
OBS_MONTH = pd.Timestamp(SST.obs_time.values[-1])
HIST = pd.to_datetime(SST.hist_time.values)  # last 12 observed months, ending with OBS_MONTH
NINO = SST.nino34_9120.values.astype(float)
NINO_HIST = SST.nino34_history_9120.values.astype(float)

# lat/lon boxes on the 0..360 grid; DMI is west minus east
REGIONS = {
    "Niño 3.4 (5°S–5°N, 170°W–120°W)": [(-5, 5, 190, 240)],
    "Niño 3 (5°S–5°N, 150°W–90°W)": [(-5, 5, 210, 270)],
    "Niño 4 (5°S–5°N, 160°E–150°W)": [(-5, 5, 160, 210)],
    "Niño 1+2 (10°S–0°, 90°W–80°W)": [(-10, 0, 270, 280)],
    "Indian Ocean Dipole (DMI, west − east)": "dmi",
    "Indian Ocean basin (20°S–20°N, 40°E–110°E)": [(-20, 20, 40, 110)],
    "Tropical North Atlantic (5.5°N–23.5°N, 57.5°W–15°W)": [(5.5, 23.5, 302.5, 345)],
    "Northeast Pacific (35°N–50°N, 150°W–125°W)": [(35, 50, 210, 235)],
    "Kuroshio Extension (30°N–40°N, 140°E–170°E)": [(30, 40, 140, 170)],
    "South China Sea (0°–25°N, 105°E–120°E)": [(0, 25, 105, 120)],
    "Mediterranean (30°N–46°N, 0°–36°E)": [(30, 46, 0, 36)],
    "Global ocean (60°S–60°N)": [(-60, 60, 0, 360)],
}
FIELDS = {"SST anomaly": "ssta", "SST": "sst", "Chance of a top-quintile month": "pupper", "Forecast uncertainty (σ)": "sigma"}
HELDOUT_TRACKS = {
    "12 common start months, Jul 2023–Jun 2024 (8 models)": "all_available_models",
    "25 common start months, Jul 2023–Jul 2025 (7 models)": "longer_common_period",
    "34 fixed target months, Dec 2023–Sep 2026 (own models only)": "own_fixed_target_months",
}
HELDOUT_REGIONS = {"Global": "Global", "Niño 3.4": "Nino3.4", "Tropical Indo-Pacific (20°S–20°N)": "WP",
                   "North Pacific (20°N–70°N)": "PDO", "Arctic (70°N–90°N)": "AO", "South Indian (60°S–40°S)": "SIO",
                   "South Pacific (60°S–40°S)": "SPO", "South Atlantic (60°S–40°S)": "SAO"}
HELDOUT_LABELS = {"Cutoff-2022 model": "CAS-Canglong method, refit with data to 2022",
                  "Seasonal baseline": "frozen seasonal model (before calibration)",
                  "Original baseline": "original head", "Surface baseline": "surface head", "Joint baseline": "joint head"}


def box_mean(field: xr.DataArray, boxes) -> np.ndarray:
    lat, lon = field["lat"], field["lon"]
    mask = xr.zeros_like(lat * lon, dtype=bool)
    for la0, la1, lo0, lo1 in boxes:
        mask = mask | ((lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1))
    w = np.cos(np.deg2rad(lat)) * mask
    return (field.where(mask) * w).sum(("lat", "lon")).values / w.where(field.notnull()).sum(("lat", "lon")).values


def region_series(name: str) -> np.ndarray:
    """12 observed months (ending with the last input month) followed by the 12 forecast months, 1991-2020 base."""
    fields = xr.concat([SST.history_ssta_9120.rename(hist_time="t"), SST.ssta_9120.rename(time="t")], "t")
    spec = REGIONS[name]
    if spec == "dmi":
        return box_mean(fields, [(-10, 10, 50, 70)]) - box_mean(fields, [(-10, 0, 90, 110)])
    return box_mean(fields, spec)


def _layout(fig: go.Figure, ytitle: str, height: int = 430) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=56, r=120, t=24, b=48), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK2), hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", y=-0.16, x=0, font=dict(size=12, color=INK2)),
        hoverlabel=dict(bgcolor=SURFACE, font=dict(family=FONT, color=INK)))
    fig.update_xaxes(showgrid=False, linecolor=BASELINE, tickfont=dict(color=MUTED), ticks="outside", tickcolor=BASELINE)
    fig.update_yaxes(title=ytitle, gridcolor=GRIDLINE, zeroline=True, zerolinecolor=BASELINE, zerolinewidth=1,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK2))
    return fig


def _center(season: str) -> pd.Timestamp | None:
    """'2026-NDJ' -> 2026-12 (middle month; the year is the first month's); '2026-Aug' -> 2026-08."""
    year, code = season.split("-")
    order = "JFMAMJJASOND"
    if len(code) == 3 and code[0].isupper() and code[1:].isupper():
        for i in range(12):
            if "".join(order[(i + k) % 12] for k in range(3)) == code:
                return pd.Timestamp(int(year), i + 1, 1) + pd.DateOffset(months=1)
        return None
    return pd.Timestamp(f"{year}-{code}-01")


def plume_figure() -> go.Figure:
    fig = go.Figure()
    for k, m in enumerate(IRI["models"]):
        x = [_center(s) for s in m["seasons"]]
        fig.add_trace(go.Scatter(x=x, y=m["data"], mode="lines", line=dict(color=CONTEXT, width=1),
                                 name="IRI plume models", legendgroup="iri", showlegend=k == 0, opacity=0.9,
                                 hovertemplate=f"{m['model']} ({m['type'].lower()}): %{{y:+.2f}} °C<extra></extra>"))
    colors = {"dynamical": DYN_ORANGE, "statistical": STAT_AQUA}
    for avg in IRI["averages"]:
        if avg["model"] not in colors: continue
        x = [_center(s) for s in avg["seasons"]]
        name = f"IRI {avg['model']} mean"
        fig.add_trace(go.Scatter(x=x, y=avg["data"], mode="lines+markers", name=name,
                                 line=dict(color=colors[avg["model"]], width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                                 hovertemplate=f"{name}: %{{y:+.2f}} °C<extra></extra>"))
    fig.add_trace(go.Scatter(x=HIST, y=NINO_HIST, mode="lines", name="ERA5 observed (monthly)",
                             line=dict(color=INK, width=1.5), hovertemplate="ERA5 %{x|%b %Y}: %{y:+.2f} °C<extra></extra>"))
    run = pd.Series(NINO).rolling(3, center=True).mean().values[1:-1]   # forecast-only seasons, OND .. JAS
    xs = MONTHS[1:-1]
    fig.add_trace(go.Scatter(x=xs, y=run, mode="lines+markers", name="CAS-Canglong (3-month mean)",
                             line=dict(color=CAS_BLUE, width=3), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                             hovertemplate="CAS-Canglong: %{y:+.2f} °C<extra></extra>"))
    fig.add_annotation(x=xs[-1], y=run[-1], text="CAS-Canglong", xanchor="left", xshift=8, showarrow=False,
                       font=dict(color=INK, size=12))
    obs = [(_center(o["season"]), o["data"], o["label"]) for o in IRI["observed"]]
    fig.add_trace(go.Scatter(x=[o[0] for o in obs], y=[o[1] for o in obs], mode="markers", name="Observed (IRI)",
                             marker=dict(color=INK, size=9, symbol="diamond", line=dict(color=SURFACE, width=2)),
                             text=[o[2] for o in obs], hovertemplate="%{text}: %{y:+.2f} °C<extra></extra>"))
    for y, label in ((0.5, "El Niño +0.5 °C"), (-0.5, "La Niña −0.5 °C")):
        fig.add_hline(y=y, line=dict(color=MUTED, width=1, dash="dot"))
        fig.add_annotation(x=0, xref="paper", y=y, text=label, xanchor="left", yanchor="bottom", xshift=4,
                           showarrow=False, font=dict(color=MUTED, size=11))
    fig.update_xaxes(dtick="M2", tickformat="%b<br>%Y", tickangle=0)
    return _layout(fig, "Niño 3.4 SST anomaly (°C, 1991–2020 base)", 460)


def region_figure(name: str) -> go.Figure:
    y = region_series(name)
    x = [*HIST, *MONTHS]
    n = len(HIST)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[:n], y=y[:n], mode="lines+markers", name="ERA5 observed",
                             line=dict(color=INK, width=1.5), marker=dict(size=6),
                             hovertemplate="observed %{x|%b %Y}: %{y:+.2f} °C<extra></extra>"))
    fig.add_trace(go.Scatter(x=x[n - 1:n + 1], y=y[n - 1:n + 1], mode="lines", line=dict(color=CAS_BLUE, width=2, dash="dot"),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x[n:], y=y[n:], mode="lines+markers", name="CAS-Canglong forecast",
                             line=dict(color=CAS_BLUE, width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{x|%b %Y}: %{y:+.2f} °C<extra></extra>"))
    if name.startswith("Niño"):
        for v in (0.5, -0.5):
            fig.add_hline(y=v, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_xaxes(dtick="M2", tickformat="%b<br>%Y", tickangle=0)
    return _layout(fig, "SST anomaly (°C, 1991–2020 base)", 400)


def region_table(name: str) -> pd.DataFrame:
    y = region_series(name)[len(HIST) - 1:]
    labels = [f"{OBS_MONTH:%b %Y} (observed)"] + [f"{t:%b %Y}" for t in MONTHS]
    return pd.DataFrame({"Month": labels, "Lead (months)": ["0"] + [str(i) for i in range(1, len(MONTHS) + 1)],
                         "SST anomaly (°C)": [f"{v:+.2f}" for v in y]})


def map_path(field: str, index: int) -> str:
    code, i = FIELDS[field], int(index)
    if code in ("pupper", "sigma"):
        i = max(i, 1)                           # probability and spread exist for forecast months only
    return str(MAPS / f"{code}_{FRAMES[i]['key']}.png")


def frame_label(field: str, index: int) -> str:
    i = int(index)
    if FIELDS[field] in ("pupper", "sigma") and i == 0:
        return f"**{FRAMES[1]['label']}** — probability and spread maps start at lead 1"
    return f"**{FRAMES[i]['label']}**"


def atmos_path(var: str, week: int) -> str:
    code = {"2 m temperature": "t2m", "Precipitation": "tp", "Mean sea-level pressure": "msl",
            "500 hPa geopotential height": "z500"}[var]
    return str(MAPS / f"atm_{code}_w{int(week)}.png")


def week_label(week: int) -> str:
    return f"**{WEEKS[int(week) - 1]['label']}**"


def heldout_figure(track: str, region: str, metric: str) -> go.Figure:
    col = "acc_spatial" if metric.startswith("Spatial ACC") else "rmse_K"
    d = HELDOUT[(HELDOUT.track == HELDOUT_TRACKS[track]) & (HELDOUT.region == HELDOUT_REGIONS[region])]
    fig = go.Figure()
    greys = iter(["#898781", "#a9a79f", "#c3c2b7", "#d9d8d2"])
    nwp = iter([DYN_ORANGE, "#b8421b", "#f3a5a1"])
    for model, g in sorted(d.groupby("model"), key=lambda kv: (kv[0] != "Cutoff-2022 model", "baseline" not in kv[0], kv[0])):
        g = g.sort_values("lead")
        main = model == "Cutoff-2022 model"
        color = CAS_BLUE if main else next(greys) if "baseline" in model else next(nwp)
        fig.add_trace(go.Scatter(x=g.lead, y=g[col], mode="lines+markers", name=HELDOUT_LABELS.get(model, f"{model} (raw archive)"),
                                 line=dict(color=color, width=3 if main else 1.6, dash=None if main or "baseline" in model else "dash"),
                                 marker=dict(size=7 if main else 5),
                                 hovertemplate=f"{model} · lead %{{x}}: %{{y:.3f}}<extra></extra>"))
    fig.update_xaxes(title="Lead (months)", dtick=1)
    fig = _layout(fig, "Spatial ACC" if col == "acc_spatial" else "RMSE (°C)", 480)
    fig.update_layout(hovermode="closest", margin=dict(r=40, b=110), legend=dict(y=-0.2))
    return fig


def heldout_table(track: str, region: str) -> pd.DataFrame:
    d = HELDOUT[(HELDOUT.track == HELDOUT_TRACKS[track]) & (HELDOUT.region == HELDOUT_REGIONS[region])]
    t = d.groupby("model").agg(acc=("acc_spatial", "mean"), rmse=("rmse_K", "mean"), months=("n_months", "first"),
                               first=("first_valid", "min"), last=("last_valid", "max")).sort_values("acc", ascending=False)
    return pd.DataFrame({"Model": [HELDOUT_LABELS.get(m, f"{m} (raw archive)") for m in t.index],
                         "Mean spatial ACC (leads 1–12)": t.acc.round(3).values,
                         "Mean RMSE °C (leads 1–12)": t.rmse.round(3).values, "Samples per lead": t.months.values,
                         "Target months": [f"{a} – {b}" for a, b in zip(t["first"], t["last"])]})


def hindcast_figure(lead: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=HINDCAST.target_month, y=HINDCAST.obs, mode="lines", name="ERA5 observed",
                             line=dict(color=INK, width=2), hovertemplate="observed %{x|%b %Y}: %{y:+.2f} °C<extra></extra>"))
    fig.add_trace(go.Scatter(x=HINDCAST.target_month, y=HINDCAST[f"lead{int(lead)}"], mode="lines",
                             name=f"CAS-Canglong hindcast, lead {int(lead)}", line=dict(color=CAS_BLUE, width=2),
                             hovertemplate="hindcast %{x|%b %Y}: %{y:+.2f} °C<extra></extra>"))
    for v in (0.5, -0.5):
        fig.add_hline(y=v, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_xaxes(dtick="M6", tickformat="%b<br>%Y", tickangle=0)
    return _layout(fig, "Niño 3.4 anomaly (°C, 1959–2009 base)", 400)


def hindcast_caption(lead: int) -> str:
    r = HINDCAST_SKILL[HINDCAST_SKILL.lead == int(lead)].iloc[0]
    return (f"Lead {int(lead)}: correlation **{r['corr']:.2f}**, RMSE **{r['rmse_K']:.2f} °C** over {int(r['n'])} target months "
            "(2016–2022). These targets were used to calibrate the released model, so the scores are in-sample.")


above = [t for t, v in zip(MONTHS, NINO) if v >= 0.5]
glob = region_series("Global ocean (60°S–60°N)")
KPIS = [
    (f"Niño 3.4 · {OBS_MONTH:%b %Y}", f"{NINO_HIST[-1]:+.2f} °C", "observed (ERA5), last input month"),
    (f"Niño 3.4 · {MONTHS[0]:%b %Y}", f"{NINO[0]:+.2f} °C", "forecast, 1-month lead"),
    (f"Niño 3.4 · {MONTHS[3]:%b %Y}", f"{NINO[3]:+.2f} °C", "forecast, 4-month lead"),
    ("El Niño (≥ +0.5 °C)", f"{above[0]:%b %Y} – {above[-1]:%b %Y}" if above else "–", f"{len(above)} of {len(NINO)} forecast months"),
    ("Global ocean 60°S–60°N", f"{glob[len(HIST)]:+.2f} °C", f"area-mean anomaly, {MONTHS[0]:%b %Y}"),
]
kpi_html = "".join(f'<div class="kpi"><div class="kpi-label">{a}</div><div class="kpi-value">{b}</div>'
                   f'<div class="kpi-note">{c}</div></div>' for a, b, c in KPIS)

HEADER = f"""
<div class="hero">
  <div class="hero-kicker">CAS-Canglong · research forecast · model release 2026-10-25</div>
  <h1>🐉 Global sea-surface temperature &amp; ENSO outlook</h1>
  <p>Input: ERA5 monthly means through <b>{OBS_MONTH:%B %Y}</b> → mean, spread and calibrated quintile probabilities for
  <b>{MONTHS[0]:%b %Y} – {MONTHS[-1]:%b %Y}</b> (12 months) · six-week atmosphere from <b>{RUN['atmos_start']}</b></p>
  <div class="chips">
    <a href="https://huggingface.co/{MODEL_REPO}" target="_blank">🧠 Model weights</a>
    <a href="https://huggingface.co/datasets/{HINDCAST_REPO}" target="_blank">🗂️ Hindcast 2016–2022</a>
    <a href="https://doi.org/10.1002/advs.78271" target="_blank">📄 Paper · Advanced Science</a>
    <a href="https://github.com/GISWLH/CAS-Canglong" target="_blank">⭐ GitHub</a>
    <a href="https://canglong.earth" target="_blank">🌍 canglong.earth</a>
    <a href="https://github.com/GISWLH/WeatherPre" target="_blank">📊 WeatherPre</a>
  </div>
</div>
<div class="kpis">{kpi_html}</div>
"""

HELDOUT_NOTE = """
**What was held out.** These curves come from a copy of the released method that was re-fitted with targets up to
December 2022 only (calibration settings chosen on 2021–2022) and then scored on 2023-01 to 2026-09 targets it never saw.
The four base heads were trained on targets up to August 2009 and validated up to April 2016. The published weights themselves were calibrated with
targets through September 2026, so they have no held-out score of their own. The method design was shaped by earlier
looks at the post-2022 period, so this is a retrospective out-of-time check, not a pristine blind test.
Spatial ACC is the area-weighted anomaly correlation of each map (not the grid-point temporal correlation). The three
dynamical systems are raw ensemble-mean archives with no drift or bias correction, which inflates their RMSE. 12–34
monthly samples per lead are serially correlated; no significance test was made. **Niño 3.4 is the weak spot:** there the
refit's ACC is below that of its own uncalibrated seasonal model at every lead.
"""

ABOUT = f"""
### What this is
CAS-Canglong learns the coupled ocean–atmosphere system from ERA5 to predict global sea-surface temperature (SST)
([Wang et al., *Advanced Science* 2026](https://doi.org/10.1002/advs.78271); preprint
[arXiv:2409.05369](https://arxiv.org/abs/2409.05369)). This Space shows the **2026-10-25 release of the 1° monthly SST
model** ([weights and code on Hugging Face](https://huggingface.co/{MODEL_REPO})):

* **Inputs** — the last 120 months of eight ERA5 monthly fields on a 1° grid: SST, mean sea-level pressure, surface latent
  heat flux, 10 m u and v wind, 500 and 850 hPa geopotential, net surface solar radiation.
* **Core** — four frozen heads built on a 1° CAS-Canglong network: the original network/EOF blend, a slow-background
  branch, a joint ocean–atmosphere branch and a seasonal-state branch whose response depends on the calendar month.
* **Calibration** — a per-grid-cell ridge correction (α = 0.05, 19 coefficients per lead and cell, using the four head
  forecasts, the observed SST anomaly of the last 1, 3 and 12 months, annual/semi-annual harmonics, a time trend and a
  few seasonal interactions),
  mixed 10 % with 90 % of the seasonal model; a CRPS-fitted Gaussian spread; and quintile probabilities calibrated in the
  spirit of the probability-based correction of [Guan et al., *Nature* 2026](https://www.nature.com/articles/s41586-026-11126-x).
* **Output** — 12 monthly leads on 40,701 ocean cells (nominal 1°): mean SST, Gaussian σ and quantiles, and the
  probability of each fifth of the 1959–2009 climate. Leads 13–16 are not produced.
* **Atmosphere (6 weeks, experimental)** — CAS-Canglong V3.5 weekly means published on canglong.earth, read through
  [WeatherPre](https://github.com/GISWLH/WeatherPre) (`wp.forecast("canglong", "t2m", "6w")`).

### How to read the maps
* **SST anomaly** maps and all regional and Niño 3.4 series here use the WMO 1991–2020 normal, computed from the same
  ERA5 record as the inputs, so they can be compared with the [IRI ENSO plume]({IRI['source']}) (published
  {IRI['published']}, redrawn). The model's own reference is 1959-01–2009-08; on that base Niño 3.4 is 0.04–0.23 °C higher,
  depending on the month. Both are in the NetCDF.
* **Chance of a top-quintile month** is the calibrated probability that the monthly SST falls in the warmest fifth of the
  1959–2009 climate for that calendar month (20 % = climatological odds). Because the ocean has warmed since then, high
  values are widespread.
* **Forecast uncertainty (σ)** is a per-cell Gaussian spread. The probabilities are marginal per grid cell, not a coherent
  ensemble, so they cannot be summed into regional or Niño 3.4 uncertainties.

### Limits — please read
* A research forecast, not an official or operational warning product.
* The released weights were calibrated with targets through 2026-09, and the calibration settings were selected on
  year-blocked diagnostics of the same years. Every historical score of these exact weights, including the published
  2016–2022 hindcast, is therefore in-sample. The *Skill & hindcasts* tab shows the held-out check of a refit copy.
* **ENSO:** the forecast damps the 2026 El Niño quickly. Niño 3.4 drops from {NINO_HIST[-1]:+.2f} °C observed in
  {OBS_MONTH:%B} to {NINO[0]:+.2f} °C in {MONTHS[0]:%B}, well below the IRI dynamical models. The damping is already in
  the four frozen base heads (lead 1: +1.5 to +1.8 °C on the model base), and Niño 3.4 is the region where the held-out
  check found this method weakest. Read the Niño 3.4 outlook with that in mind. canglong.earth still serves the previous
  16-month Public-Swin run, whose Niño 3.4 stays at +3.2 °C or more through December 2026 (on its 1959–2016 base).
* The six-week atmospheric run is **not drift-corrected**; high-latitude 2 m temperature drifts warm with lead time
  (about +5 °C in week 1 to +14 °C in week 6 poleward of 60°N in earlier runs). Read those fields as model output; anomaly
  maps are not shown.

### Provenance of this run
* Forecast file: `{RUN['sst_file']}`, derived from the release NetCDF (SHA-256 `{(RUN['source_forecast_sha256'] or '')[:16]}…`)
  without changing any value. Checks: SST − anomaly reproduces the stored climatology to
  {RUN['checks']['forecast_sst_minus_ssta_vs_climatology_max_abs_K']:.1e} K, and the observed {OBS_MONTH:%B} anomaly
  equals the model's own input context (max |Δ| = {RUN['checks']['observed_ssta_vs_model_context_max_abs_K']} K).
* Atmosphere: {RUN['atmos_source']}.

### Cite
```bibtex
@article{{wang2026canglong,
  title   = {{Improved Skills for Predicting Sub-Seasonal to Seasonal Global Sea Surface Temperature}},
  author  = {{Wang, Longhao and Zhang, Xuanze and Leung, L. Ruby and Chiew, Francis H. S. and
             AghaKouchak, Amir and Ying, Kairan and Zhang, Yongqiang}},
  journal = {{Advanced Science}},
  year    = {{2026}},
  pages   = {{e78271}},
  doi     = {{10.1002/advs.78271}}
}}
```
"""

CSS = """
.hero {padding: 8px 4px 4px;}
.hero h1 {font-size: 1.9rem; margin: 2px 0 6px; line-height: 1.2;}
.hero p {margin: 0 0 10px; color: var(--body-text-color-subdued);}
.hero-kicker {text-transform: uppercase; letter-spacing: .08em; font-size: .78rem; color: var(--body-text-color-subdued);}
.chips {display: flex; flex-wrap: wrap; gap: 8px;}
.hero .chips a {text-decoration: none !important; padding: 5px 12px; border-radius: 999px; font-size: .9rem;
  border: 1px solid var(--border-color-primary); color: var(--body-text-color) !important; background: var(--block-background-fill);}
.hero .chips a:hover {border-color: var(--color-accent);}
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: 14px 0 4px;}
.kpi {padding: 12px 14px; border-radius: 12px; border: 1px solid var(--border-color-primary); background: var(--block-background-fill);}
.kpi-label {font-size: .8rem; color: var(--body-text-color-subdued);}
.kpi-value {font-size: 1.55rem; font-weight: 650; margin: 2px 0;}
.kpi-note {font-size: .78rem; color: var(--body-text-color-subdued);}
"""

with gr.Blocks(title="CAS-Canglong · SST & ENSO forecast") as demo:
    gr.HTML(HEADER)
    with gr.Tabs():
        with gr.Tab("🌊 Ocean & ENSO"):
            with gr.Row(equal_height=True):
                field = gr.Radio(list(FIELDS), value="SST anomaly", label="Field", scale=2)
                lead = gr.Slider(0, len(FRAMES) - 1, value=1, step=1, label="Month (0 = last observed input month)", scale=3)
            caption = gr.Markdown(frame_label("SST anomaly", 1))
            image = gr.Image(map_path("SST anomaly", 1), show_label=False, interactive=False, buttons=["download", "fullscreen"])
            gr.Markdown("#### Niño 3.4 outlook vs the IRI ENSO plume")
            gr.Plot(plume_figure(), show_label=False)
            gr.Markdown("Monthly CAS-Canglong Niño 3.4 (°C, 1991–2020 base): " + ", ".join(
                f"{t:%b %Y} {v:+.2f}" for t, v in zip(MONTHS, NINO)) + ". The forecast damps the observed El Niño "
                "much faster than the IRI dynamical models; see *About* and *Skill & hindcasts*.")
            lead.change(lambda f, i: (map_path(f, i), frame_label(f, i)), [field, lead], [image, caption])
            field.change(lambda f, i: (map_path(f, i), frame_label(f, i)), [field, lead], [image, caption])
        with gr.Tab("📍 Regional outlook"):
            region = gr.Dropdown(list(REGIONS), value=list(REGIONS)[0], label="Region")
            rplot = gr.Plot(region_figure(list(REGIONS)[0]), show_label=False)
            gr.Markdown("Box means of the native 1° anomaly field (cos-latitude weighted, model ocean cells only), "
                        "relative to ERA5 1991–2020, with the last 12 observed months for context.")
            rtable = gr.Dataframe(region_table(list(REGIONS)[0]), interactive=False)
            region.change(lambda r: (region_figure(r), region_table(r)), region, [rplot, rtable])
        with gr.Tab("📈 Skill & hindcasts"):
            gr.Markdown("#### Held-out test on 2023–2026")
            with gr.Row():
                htrack = gr.Dropdown(list(HELDOUT_TRACKS), value=list(HELDOUT_TRACKS)[0], label="Comparison", scale=3)
                hregion = gr.Dropdown(list(HELDOUT_REGIONS), value="Global", label="Region", scale=2)
                hmetric = gr.Radio(["Spatial ACC (higher is better)", "RMSE (lower is better)"],
                                   value="Spatial ACC (higher is better)", label="Metric", scale=2)
            hplot = gr.Plot(heldout_figure(list(HELDOUT_TRACKS)[0], "Global", "Spatial ACC"), show_label=False)
            htable = gr.Dataframe(heldout_table(list(HELDOUT_TRACKS)[0], "Global"), interactive=False)
            gr.Markdown(HELDOUT_NOTE)
            for c in (htrack, hregion, hmetric):
                c.change(lambda t, r, m: (heldout_figure(t, r, m), heldout_table(t, r)), [htrack, hregion, hmetric], [hplot, htable])
            gr.Markdown(f"#### Hindcast 2016–2022 · [dataset](https://huggingface.co/datasets/{HINDCAST_REPO})")
            hlead = gr.Slider(1, 12, value=3, step=1, label="Lead (months)")
            hcplot = gr.Plot(hindcast_figure(3), show_label=False)
            hccap = gr.Markdown(hindcast_caption(3))
            hlead.change(lambda k: (hindcast_figure(k), hindcast_caption(k)), hlead, [hcplot, hccap])
            gr.Dataframe(HINDCAST_SKILL.rename(columns={"lead": "Lead", "n": "Months", "corr": "Correlation", "rmse_K": "RMSE (°C)"}),
                         interactive=False, label="Niño 3.4 hindcast scores by lead (in-sample)")
        with gr.Tab("🌦️ Atmosphere · 6 weeks (experimental)"):
            gr.Markdown("Weekly-mean fields of the CAS-Canglong V3.5 atmospheric run published on canglong.earth. "
                        "**Not drift-corrected** — high-latitude temperatures drift warm with lead time (see *About*).")
            with gr.Row():
                avar = gr.Dropdown(["2 m temperature", "Precipitation", "Mean sea-level pressure", "500 hPa geopotential height"],
                                   value="2 m temperature", label="Variable")
                week = gr.Slider(1, len(WEEKS), value=1, step=1, label="Week")
            wcap = gr.Markdown(week_label(1))
            aimg = gr.Image(atmos_path("2 m temperature", 1), show_label=False, interactive=False, buttons=["download", "fullscreen"])
            week.change(lambda v, w: (atmos_path(v, w), week_label(w)), [avar, week], [aimg, wcap])
            avar.change(lambda v, w: atmos_path(v, w), [avar, week], aimg)
        with gr.Tab("📦 Model & data"):
            gr.Markdown(f"* **Model** — frozen weights, calibration states, inference code and a CPU example: "
                        f"[{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})\n"
                        f"* **Hindcast** — observed and lead 1–12 SST for every target month of 2016–2022: "
                        f"[{HINDCAST_REPO}](https://huggingface.co/datasets/{HINDCAST_REPO})\n"
                        "* **This run** — the files behind every map and chart on this page:")
            gr.File([str(DATA / RUN["sst_file"]), str(DATA / "nino34.csv"), str(DATA / RUN["atmos_file"]),
                     str(DATA / "hindcast_nino34.csv"), str(DATA / "round6_heldout_metrics.csv"), str(DATA / "run.json")],
                    file_count="multiple", label="Downloads", interactive=False)
            gr.Code(f"""import xarray as xr
from huggingface_hub import hf_hub_download, snapshot_download

# this run: mean, anomaly (both bases), sigma, 10/90 % quantiles, quintile probabilities, Nino 3.4
ds = xr.open_dataset(hf_hub_download("LonghaoWang/CAS-Canglong", "data/{RUN['sst_file']}", repo_type="space"))
print(ds.nino34_9120.to_series().round(2))

# the model: verify the frozen package and replay the October 2026 forecast on a CPU
pkg = snapshot_download("{MODEL_REPO}", allow_patterns=["frozen_model_20261025_v1/*"])
# cd <pkg>/frozen_model_20261025_v1 && python -B run_frozen.py --verify-only
# python -B run_frozen.py --prepared-input examples/input_only_202610.npz --output /tmp/forecast_202610.nc

# the 2016-2022 hindcast (obs + lead1..lead12 per target month)
hc = xr.open_dataset(hf_hub_download("{HINDCAST_REPO}",
                     "hindcast_2016_2022_v1/sst_hindcast_201601_202212_lead01_12.nc", repo_type="dataset"))""",
                    language="python", label="Load in Python")
        with gr.Tab("ℹ️ About"):
            gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="stone",
                                     font=["system-ui", "-apple-system", "Segoe UI", "Helvetica", "Arial", "sans-serif"]), css=CSS)
