"""CAS-Canglong forecast showcase (Hugging Face Space).

Everything shown is read from `data/` and `assets/`, written by `scripts/build_assets.py` from one forecast run.
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

SURFACE, GRIDLINE, BASELINE, INK, INK2, MUTED = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#0b0b0b", "#52514e", "#898781"
CAS_BLUE, DYN_ORANGE, STAT_AQUA, CONTEXT = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"

FRAMES = RUN["frames"]                       # index 0 = observed initial month, 1..16 = forecast leads
WEEKS = RUN["weeks"]
MONTHS = pd.to_datetime(SST.time.values)
OBS_MONTH = pd.Timestamp(SST.obs_time.values[-1])
NINO = SST.nino34.values.astype(float)

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


def box_mean(field: xr.DataArray, boxes) -> np.ndarray:
    lat, lon = field["lat"], field["lon"]
    mask = xr.zeros_like(lat * lon, dtype=bool)
    for la0, la1, lo0, lo1 in boxes:
        mask = mask | ((lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1))
    w = np.cos(np.deg2rad(lat)) * mask
    return (field.where(mask) * w).sum(("lat", "lon")).values / w.where(field.notnull()).sum(("lat", "lon")).values


def region_series(name: str) -> np.ndarray:
    fields = xr.concat([SST.observed_ssta.expand_dims(time=[np.datetime64(OBS_MONTH, "ns")]), SST.ssta], "time")
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
    run = pd.Series(NINO).rolling(3, center=True).mean().values[1:-1]
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
    return _layout(fig, "Niño 3.4 SST anomaly (°C)", 460)


def region_figure(name: str) -> go.Figure:
    y = region_series(name)
    x = [OBS_MONTH, *MONTHS]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[:2], y=y[:2], mode="lines", line=dict(color=CAS_BLUE, width=2, dash="dot"),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x[1:], y=y[1:], mode="lines+markers", name="CAS-Canglong forecast",
                             line=dict(color=CAS_BLUE, width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{x|%b %Y}: %{y:+.2f} °C<extra></extra>"))
    fig.add_trace(go.Scatter(x=x[:1], y=y[:1], mode="markers", name=f"ERA5 observed {OBS_MONTH:%b %Y}",
                             marker=dict(color=INK, size=10, symbol="diamond", line=dict(color=SURFACE, width=2)),
                             hovertemplate="observed %{x|%b %Y}: %{y:+.2f} °C<extra></extra>"))
    if name.startswith("Niño"):
        for v in (0.5, -0.5):
            fig.add_hline(y=v, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_xaxes(dtick="M2", tickformat="%b<br>%Y", tickangle=0)
    return _layout(fig, "SST anomaly (°C)", 400)


def region_table(name: str) -> pd.DataFrame:
    y = region_series(name)
    labels = [f"{OBS_MONTH:%b %Y} (observed)"] + [f"{t:%b %Y}" for t in MONTHS]
    return pd.DataFrame({"Month": labels, "Lead (months)": ["0"] + [str(i) for i in range(1, len(MONTHS) + 1)],
                         "SST anomaly (°C)": [f"{v:+.2f}" for v in y]})


def map_path(field: str, index: int) -> str:
    key = FRAMES[int(index)]["key"]
    return str(MAPS / f"{'ssta' if field.startswith('SST anomaly') else 'sst'}_{key}.png")


def atmos_path(var: str, week: int) -> str:
    code = {"2 m temperature": "t2m", "Precipitation": "tp", "Mean sea-level pressure": "msl",
            "500 hPa geopotential height": "z500"}[var]
    return str(MAPS / f"atm_{code}_w{int(week)}.png")


def frame_label(index: int) -> str:
    return f"**{FRAMES[int(index)]['label']}**"


def week_label(week: int) -> str:
    return f"**{WEEKS[int(week) - 1]['label']}**"


above = [t for t, v in zip(MONTHS, NINO) if v > 0.5]
peak_i = int(np.argmax(NINO))
glob = region_series("Global ocean (60°S–60°N)")
KPIS = [
    ("Niño 3.4 · " + f"{MONTHS[0]:%b %Y}", f"{NINO[0]:+.2f} °C", "first forecast month"),
    ("Niño 3.4 · " + f"{MONTHS[3]:%b %Y}", f"{NINO[3]:+.2f} °C", "boreal winter, 4-month lead"),
    ("El Niño (> +0.5 °C)", f"{above[0]:%b %Y} – {above[-1]:%b %Y}" if above else "–", f"{len(above)} of {len(NINO)} forecast months"),
    ("Global ocean 60°S–60°N", f"{glob[1]:+.2f} °C", f"area-mean anomaly, {MONTHS[0]:%b %Y}"),
]
kpi_html = "".join(f'<div class="kpi"><div class="kpi-label">{a}</div><div class="kpi-value">{b}</div>'
                   f'<div class="kpi-note">{c}</div></div>' for a, b, c in KPIS)

HEADER = f"""
<div class="hero">
  <div class="hero-kicker">CAS-Canglong · live research forecast</div>
  <h1>🐉 Global sea-surface temperature &amp; ENSO outlook</h1>
  <p>Input: ERA5 monthly means through <b>{OBS_MONTH:%B %Y}</b> → forecasts for <b>{MONTHS[0]:%b %Y} – {MONTHS[-1]:%b %Y}</b>
  (16 months) · six-week atmosphere from <b>{RUN['atmos_start']}</b></p>
  <div class="chips">
    <a href="https://doi.org/10.1002/advs.78271" target="_blank">📄 Paper · Advanced Science</a>
    <a href="https://github.com/GISWLH/CAS-Canglong" target="_blank">⭐ GitHub</a>
    <a href="https://canglong.earth" target="_blank">🌍 canglong.earth</a>
    <a href="https://github.com/GISWLH/WeatherAI" target="_blank">🧩 WeatherAI (PyTorch port)</a>
    <a href="https://github.com/GISWLH/WeatherPre" target="_blank">📊 WeatherPre</a>
  </div>
</div>
<div class="kpis">{kpi_html}</div>
"""

ABOUT = f"""
### What this is
CAS-Canglong is a 3-D Earth-specific transformer for sub-seasonal to seasonal prediction of global sea-surface
temperature (SST) ([Wang et al., *Advanced Science* 2026](https://doi.org/10.1002/advs.78271);
preprint [arXiv:2409.05369](https://arxiv.org/abs/2409.05369)). This Space shows the **current operational run**, the same
forecast that [canglong.earth](https://canglong.earth) serves:

* **Ocean (16 months)** — a one-pass Swin-transformer SST model on a 1° grid reads 16 months of eight ERA5 monthly
  fields (SST, MSLP, latent heat flux, 10 m winds, 500/850 hPa geopotential, net solar radiation) and predicts the next
  16 monthly SST fields in one forward pass; a learned refiner adds 0.25° detail without changing the 1° block means.
* **Atmosphere (6 weeks, experimental)** — CAS-Canglong V3.5 weekly means published on canglong.earth, read through
  [WeatherPre](https://github.com/GISWLH/WeatherPre) (`wp.forecast("canglong", "t2m", "6w")`).

### How to read the maps
* **SST anomaly** maps are the native 1° model output minus the ERA5 1959–2016 monthly climatology (the model's training
  period, the same reference as its Niño 3.4 index). On the 1991–2020 base that IRI uses, Niño 3.4 values would be
  about 0.1 °C lower (0.04–0.18 °C, depending on the month).
* **SST** maps are the 0.25° field. Within about 2° of land it is blended with the latest ERA5 SST for display (8 grid
  rings replaced, 6 rings faded), exactly as on canglong.earth. That coastal belt is a visual repair, not model skill.
* The **ENSO plume** compares CAS-Canglong's 3-month running mean with the
  [IRI ENSO plume]({IRI['source']}) published {IRI['published']}. The plume is redrawn from IRI's own figure.

### Limits — please read
* A research forecast, not an official or operational warning product. No probability is implied: this is one
  deterministic run.
* Checkpoint acceptance for the operational ocean model used agreement with the historical NWP multi-model mean and
  with the current IRI plume, so this run is **not an independent test of skill**. Published hindcast skill is in the paper.
* The six-week atmospheric run is **not drift-corrected**. Against the ERA5 1990–2017 weekly climatology, its
  area-mean 2 m temperature poleward of 60°N is about +5 °C in week 1 and +14 °C in week 6, while 60°S–60°N stays within
  ±1 °C. Treat high-latitude fields as model climate, not as anomalies; anomaly maps are therefore not shown.

### Provenance of this run
* Ocean run: {RUN['run_time_utc'][:16].replace('T', ' ')} UTC, re-run from the frozen checkpoints in an isolated
  directory; identical to the operational product (max |ΔSST| = {RUN['check_vs_operational']['max_abs_sst_diff_K']} K,
  max |ΔNiño 3.4| = {RUN['check_vs_operational']['max_abs_nino34_diff_K']} K).
* Checkpoint SHA-256: 1° `{RUN['checkpoint_sha256']['lowres_checkpoint'][:16]}…`, 0.25° refiner
  `{RUN['checkpoint_sha256']['highres_checkpoint'][:16]}…`; input cache `{RUN['input_cache_sha256'][:16]}…`.
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
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin: 14px 0 4px;}
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
                field = gr.Radio(["SST anomaly (1°)", "SST (0.25°)"], value="SST anomaly (1°)", label="Field", scale=1)
                lead = gr.Slider(0, len(FRAMES) - 1, value=1, step=1, label="Month (0 = last observed input month)", scale=3)
            caption = gr.Markdown(frame_label(1))
            image = gr.Image(map_path("SST anomaly", 1), show_label=False, interactive=False, buttons=["download", "fullscreen"])
            gr.Markdown("#### Niño 3.4 outlook vs the IRI ENSO plume")
            gr.Plot(plume_figure(), show_label=False)
            gr.Markdown("Monthly CAS-Canglong Niño 3.4 (°C): " + ", ".join(
                f"{t:%b %Y} {v:+.2f}" for t, v in zip(MONTHS, NINO)) + ".")
            lead.change(lambda f, i: (map_path(f, i), frame_label(i)), [field, lead], [image, caption])
            field.change(lambda f, i: map_path(f, i), [field, lead], image)
        with gr.Tab("📍 Regional outlook"):
            region = gr.Dropdown(list(REGIONS), value=list(REGIONS)[0], label="Region")
            rplot = gr.Plot(region_figure(list(REGIONS)[0]), show_label=False)
            gr.Markdown(f"Box means of the native 1° anomaly field (cos-latitude weighted, ocean cells only). The headline "
                        f"Niño 3.4 ({NINO[0]:+.2f} °C for {MONTHS[0]:%b %Y}) is the operational index from the 0.25° field.")
            rtable = gr.Dataframe(region_table(list(REGIONS)[0]), interactive=False)
            region.change(lambda r: (region_figure(r), region_table(r)), region, [rplot, rtable])
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
        with gr.Tab("📦 Data"):
            gr.Markdown("NetCDF and CSV files behind every map and chart on this page.")
            gr.File([str(DATA / RUN["sst_file"]), str(DATA / RUN["atmos_file"]), str(DATA / "nino34.csv"), str(DATA / "run.json")],
                    file_count="multiple", label="Downloads", interactive=False)
            gr.Code(f"""import xarray as xr
from huggingface_hub import hf_hub_download
path = hf_hub_download("LonghaoWang/CAS-Canglong", "data/{RUN['sst_file']}", repo_type="space")
ds = xr.open_dataset(path)          # ssta / sst (1°), sst_025 (0.25°), nino34, observed fields
print(ds.nino34.to_series())""", language="python", label="Load in Python")
        with gr.Tab("ℹ️ About"):
            gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="stone",
                                     font=["system-ui", "-apple-system", "Segoe UI", "Helvetica", "Arial", "sans-serif"]), css=CSS)
