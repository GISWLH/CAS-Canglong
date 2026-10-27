<!-- readme-visual-refresh:banner -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/readme/banner_dark.svg">
  <img src="assets/readme/banner.svg" width="100%" alt="CAS-Canglong (中科苍龙): AI forecasts of global sea-surface temperature and ENSO, from one month to more than a year ahead. Whiteboard sketch: ocean and atmosphere data go into a 3-D Earth transformer, which draws the SST and ENSO outlook.">
</picture>
<!-- /readme-visual-refresh:banner -->

<div align="center">

[![Paper](https://img.shields.io/badge/Advanced%20Science-2026-1D4ED8.svg)](https://doi.org/10.1002/advs.78271)
[![arXiv](https://img.shields.io/badge/arXiv-2409.05369-B31B1B.svg)](https://arxiv.org/abs/2409.05369)
[![Live forecast](https://img.shields.io/badge/%F0%9F%A4%97%20Space-live%20forecast-FFD21E.svg)](https://huggingface.co/spaces/LonghaoWang/CAS-Canglong)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-CAS--Canglong--SST-FFD21E.svg)](https://huggingface.co/LonghaoWang/CAS-Canglong-SST)
[![Hindcast](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-hindcast%202016--2022-FFD21E.svg)](https://huggingface.co/datasets/LonghaoWang/SST-Hindcast-2016-2022-1deg-lead1-12)
[![canglong.earth](https://img.shields.io/badge/portal-canglong.earth-0EA5E9.svg)](https://canglong.earth)
[![Weights](https://img.shields.io/badge/paper%20weights-figshare-556B2F.svg)](https://doi.org/10.6084/m9.figshare.26779969)
[![PyTorch port](https://img.shields.io/badge/PyTorch-WeatherAI-EE4C2C.svg?logo=pytorch&logoColor=white)](https://github.com/GISWLH/WeatherAI/blob/main/docs/models/canglong.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-14B8A6.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/GISWLH/CAS-Canglong?style=social)](https://github.com/GISWLH/CAS-Canglong/stargazers)

**English** · [简体中文](README_zh.md)

</div>

CAS-Canglong is a 3-D Earth-specific transformer that learns the coupled ocean–atmosphere system from ERA5 and predicts
monthly global sea-surface temperature (SST). In the *Advanced Science* paper it improves one-to-three-month SST skill by
**55.6 ± 14.1 % to 63.4 ± 15.5 %** over physics-based seasonal models in six ocean regions that dominate S2S variability
over land. The project now releases a **12-month probabilistic SST and ENSO model** (release 2026-10-25,
[weights on Hugging Face](https://huggingface.co/LonghaoWang/CAS-Canglong-SST)), shown live in the
[Hugging Face Space](https://huggingface.co/spaces/LonghaoWang/CAS-Canglong), alongside the **16-month operational
outlook** and the **six-week atmospheric outlook** on [canglong.earth](https://canglong.earth).

## 📢 News

- **2026-10-27** · 🧠 [12-month SST model released](https://huggingface.co/LonghaoWang/CAS-Canglong-SST): frozen
  weights, calibration states and inference code of the 2026-10-25 release, with a held-out test on 2023–2026 targets.
  The [Space](https://huggingface.co/spaces/LonghaoWang/CAS-Canglong) now shows this model: mean, spread and quintile
  probabilities for October 2026 – September 2027.
- **2026-10-26** · 🗂️ [Hindcast dataset](https://huggingface.co/datasets/LonghaoWang/SST-Hindcast-2016-2022-1deg-lead1-12):
  observed SST and leads 1–12 for every target month of 2016–2022 (CC BY 4.0).
- **2026-10** · 🤗 [Live forecast Space](https://huggingface.co/spaces/LonghaoWang/CAS-Canglong) launched with the
  16-month operational run: SST maps, the Niño 3.4 outlook against the IRI plume, regional outlooks and NetCDF downloads.
- **2026-10** · The six-week atmospheric run can be read with [WeatherPre](https://github.com/GISWLH/WeatherPre):
  `wp.forecast("canglong", "t2m", "6w")`.
- **2026-10** · The figshare checkpoint runs in [WeatherAI](https://github.com/GISWLH/WeatherAI)'s native PyTorch port. On
  the released data its output is identical to this repository's notebook code (max |Δ| = 0.0, RTX 4090).
- **2026-10-15** · Paper published in *Advanced Science*:
  [doi:10.1002/advs.78271](https://doi.org/10.1002/advs.78271) (open access, CC BY 4.0).
- **2024-11** · Weights and normalised training data released on
  [figshare](https://doi.org/10.6084/m9.figshare.26779969). **2024-09** · Preprint
  [arXiv:2409.05369](https://arxiv.org/abs/2409.05369).

## 🌊 Latest forecast

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/readme/global_outlook_dark.png">
  <img src="assets/readme/global_outlook.png" width="100%" alt="Global SST outlook from the 2026-10-25 CAS-Canglong model: six Robinson maps for October and December 2026 and February, April, June and August 2027, with the 28 °C isotherm and the Niño 3.4 index of each month">
</picture>

Initialised with ERA5 monthly means through **September 2026**, when Niño 3.4 stood at **+2.99 °C** (ERA5, 1991–2020
base), the 2026-10-25 model predicts **+1.81 °C** for October 2026, +1.25 °C for January 2027, a return to neutral by
April 2027 and −0.4 to −0.7 °C by July–September 2027. That decay is much faster than in the
[IRI plume](https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/) of 19 October (dynamical-model mean
+3.64 °C for OND) or in the previous operational run still shown on canglong.earth (+3.75 °C for October, 1959–2016
base). The damping is already present in the model's four frozen base heads, and Niño 3.4 is the region where the
held-out test below found the method weakest, so read the ENSO outlook with care. The maps step through the 12 forecast
months two at a time; fields are the native 1° output, smoothed for display only.

![CAS-Canglong forecast: SST anomaly map and Niño 3.4 outlook](assets/readme/latest_forecast.png)

The [Space](https://huggingface.co/spaces/LonghaoWang/CAS-Canglong) adds the spread and the chance of a top-quintile month
for every month, regional outlooks and the data files. This is a research forecast, not an official one. Regenerate both
figures without a GPU: `python scripts/plot_global_outlook.py` and `python scripts/plot_latest_forecast.py`.

### How good is the 2026-10-25 model?

The released weights were calibrated with targets through September 2026, so all their historical scores, including the
[2016–2022 hindcast](https://huggingface.co/datasets/LonghaoWang/SST-Hindcast-2016-2022-1deg-lead1-12), are in-sample.
For an out-of-time check, the same method was refitted with targets up to 2022 only and scored on 2023-01 – 2026-09
targets (mean of leads 1–12, global ocean, 12 common issue months July 2023 – June 2024):

| Model | Spatial ACC | RMSE (°C) |
|---|---|---|
| CAS-Canglong method, refit to 2022 | **0.735** | **0.667** |
| frozen seasonal model, before calibration | 0.700 | 0.755 |
| COLA-RSMAS-CESM1 · CCSM4 · CanSIPS-IC4 (raw ensemble-mean archives) | 0.331 · 0.229 · 0.216 | 1.53 · 1.51 · 1.23 |

The calibration helps most in the Arctic, North Pacific and South Atlantic; it is worse than the uncalibrated model for
Niño 3.4 at every lead. The method design had seen post-2022 results, the dynamical archives are not drift-corrected, and
the samples are short, so this is a retrospective check rather than a blind test. Protocol, audits and all scores are in
the [model repository](https://huggingface.co/LonghaoWang/CAS-Canglong-SST/tree/main/evaluation/round6_temporal_holdout_20261026).

## ✨ Highlights

- **Ocean memory, learned end to end.** Sixteen months of eight ERA5 fields (SST, sea-level pressure, latent heat flux,
  10 m winds, 500/850 hPa geopotential, net solar radiation) on the 0.25° grid are embedded as 3-D patches and processed by
  Earth-specific windowed self-attention.
- **Fast.** The 43.9 M-parameter paper model runs a full 721 × 1440 forecast in 0.8 s on an RTX 4090, or in about a
  minute on four CPU cores. Training took about 7 days on 4 × A100 80 GB.
- **Open and checkable.** Weights and data are on figshare; the notebook code is in this repository; an independent
  PyTorch re-implementation in WeatherAI loads the released checkpoint and gives identical output.
- **Operational.** Monthly 16-month SST and ENSO outlooks plus weekly atmospheric outlooks, and a released 12-month model
  with probabilities and a public hindcast, with every product published as data rather than only as images.

## 🧭 Forecast products

CAS-Canglong provides global ocean and atmospheric outlooks for subseasonal-to-seasonal applications.

| Product | Forecast range | Grid | Access |
|---|---|---|---|
| **SST and ENSO model, release 2026-10-25** | monthly SST mean, spread and quintile probabilities, 1–12 months ahead | 1° | [Space](https://huggingface.co/spaces/LonghaoWang/CAS-Canglong) · [weights](https://huggingface.co/LonghaoWang/CAS-Canglong-SST) · [hindcast](https://huggingface.co/datasets/LonghaoWang/SST-Hindcast-2016-2022-1deg-lead1-12) |
| **Operational SST and ENSO outlook** | monthly SST and Niño 3.4, 1–16 months ahead | 1° + learned 0.25° refiner | [canglong.earth](https://canglong.earth/sst/) |
| **Atmospheric outlook** | weekly means of 36 variables, weeks 1–6 | 0.25°, published at 1° | [canglong.earth](https://canglong.earth/viewer.html) and [WeatherPre](https://github.com/GISWLH/WeatherPre), route `canglong` |

For reproducible research, this repository provides the [paper's code](notebook/) for monthly SST prediction at 0.25°
with 1–3-month lead times, together with public weights on [figshare](https://doi.org/10.6084/m9.figshare.26779969) and a
[WeatherAI implementation](https://github.com/GISWLH/WeatherAI/blob/main/docs/models/canglong.md). The 12-month model of
2026-10-25 is released on [Hugging Face](https://huggingface.co/LonghaoWang/CAS-Canglong-SST) with its inference code
and a CPU example; the weights of the 16-month operational run on canglong.earth have not been released.

The 2026-10-25 model reads 120 months of eight ERA5 fields on a 1° grid. Four frozen heads built on a 1° CAS-Canglong
network (network/EOF blend, slow background, joint ocean–atmosphere and seasonal state) form a seasonal model; a
per-cell ridge correction is blended in at 10 %, and a Gaussian spread and calibrated quintile probabilities are added.
The 16-month operational outlook is generated in a single forward pass of a Swin model; its checkpoint selection used
agreement with the historical NWP multi-model mean and the IRI ENSO plume, so neither live product is an independent
skill test, and published skill refers to the configuration evaluated in the paper. Atmospheric forecasts are not
drift-corrected (high latitudes warm with lead time); read these fields as model output, not as calibrated anomalies.

## 🚀 Quick start

**1 · Latest forecast as data (no GPU, no ERA5)**

```python
import xarray as xr
from huggingface_hub import hf_hub_download

path = hf_hub_download("LonghaoWang/CAS-Canglong", "data/canglong_sst_round5_init202609.nc", repo_type="space")
ds = xr.open_dataset(path)            # sst, ssta / ssta_9120, sigma, sst_q10 / sst_q90, quintile probabilities (1°)
print(ds.nino34_9120.to_series().round(2))   # Niño 3.4 on the 1991-2020 base
```

`data/run.json` in the Space names the current file. `scripts/plot_latest_forecast.py` downloads and plots it.

**2 · Run the 12-month model (CPU example, no ERA5 needed)**

```python
from huggingface_hub import snapshot_download
pkg = snapshot_download("LonghaoWang/CAS-Canglong-SST", allow_patterns=["frozen_model_20261025_v1/*"])
```

```bash
cd <pkg>/frozen_model_20261025_v1        # Python 3.11, torch 2.6, xarray, netCDF4, scipy: see requirements.txt
python -B run_frozen.py --verify-only     # every file against FROZEN_MANIFEST.json
python -B run_frozen.py --prepared-input examples/input_only_202610.npz --output /tmp/forecast_202610.nc
```

The example replays the October 2026 forecast from the shipped head forecasts. Full inference from raw ERA5 monthly input
(`--input`, `--as-of-month`, `--gpu-uuid`) needs one BF16-capable NVIDIA GPU; the
[model card](https://huggingface.co/LonghaoWang/CAS-Canglong-SST) describes the input contract.

**3 · The 2016–2022 hindcast**

```python
hc = xr.open_dataset(hf_hub_download("LonghaoWang/SST-Hindcast-2016-2022-1deg-lead1-12",
                     "hindcast_2016_2022_v1/sst_hindcast_201601_202212_lead01_12.nc", repo_type="dataset"))
hc.lead3.sel(time="2020-12")               # obs and lead1 ... lead12, all verifying in the same target month
```

**4 · Run the paper model with the official weights**

```bash
pip install git+https://github.com/GISWLH/WeatherAI.git
# canglong3_0005_600ep_base.tar, normalized_data.pt and ocean_mask.npy from https://doi.org/10.6084/m9.figshare.26779969
```

```python
import torch
from weatherai.models.canglong import load_official

model = load_official("canglong3_0005_600ep_base.tar").eval()   # trusted file: torch.save of the full module
data = torch.load("normalized_data.pt", mmap=True)               # (160, 8, 721, 1440) normalised monthly ERA5
x = data[141:157].permute(1, 0, 2, 3)[None].float()              # 16 input months -> (1, 8, 16, 721, 1440)
with torch.no_grad():
    sst = model(x)[0, 0] * 11.627 + 286.79                       # next 3 months of SST [K], (3, 721, 1440)
```

The original training and inference notebooks are in [`notebook/`](notebook): [model.ipynb](notebook/model.ipynb) and
[Infer.ipynb](notebook/Infer.ipynb). They were run with Python 3.8, torch 2.1.0+cu118, timm 1.0.11 and numpy 1.24.4.

**5 · Six-week atmospheric outlook**

```python
import weatherpre as wp                                           # pip install "weatherpre @ git+https://github.com/GISWLH/WeatherPre"
u = wp.forecast("canglong", ["t2m", "tp", "z500", "msl"], "6w")  # weekly means, latest canglong.earth run
```

## 📁 Repository

| Path | Content |
|---|---|
| [`notebook/`](notebook) | model definition and training ([model.ipynb](notebook/model.ipynb)), inference ([Infer.ipynb](notebook/Infer.ipynb)), paper figures ([Figure.ipynb](notebook/Figure.ipynb)), experiments ([cang.ipynb](notebook/cang.ipynb)) |
| [`utils/`](utils) | 3-D patch embedding, recovery, Earth-specific position index, shifted-window masks, padding and cropping; plotting helpers |
| [`data/`](data) | evaluation tables behind the paper figures (per-model skill by region and lead, IRI plume comparison) and two sample GeoTIFFs |
| [`scripts/`](scripts) | `plot_global_outlook.py` and `plot_latest_forecast.py`: download and plot the latest forecast |
| [`hf_space/`](hf_space) | source of the Hugging Face Space: app, asset builder and deploy script |

## 📚 Papers

- Wang, L., Zhang, X., Leung, L. R., Chiew, F. H. S., AghaKouchak, A., Ying, K., & Zhang, Y. (2026). Improved skills for
  predicting sub-seasonal to seasonal global sea surface temperature. *Advanced Science*, e78271.
  [doi:10.1002/advs.78271](https://doi.org/10.1002/advs.78271)
- Wang, L., Zhang, X., Leung, L. R., Chiew, F. H. S., AghaKouchak, A., Ying, K., & Zhang, Y. (2024). CAS-Canglong: A
  skillful 3D Transformer model for sub-seasonal to seasonal global sea surface temperature prediction. *arXiv*
  [2409.05369](https://arxiv.org/abs/2409.05369).
- Wang, L., Zhang, Y., & Zhang, X. (2024). Skilful prediction of mid-term sea surface temperature using 3D
  self-attention-based neural network. *EGU General Assembly 2024*.
  [doi:10.5194/egusphere-egu24-9446](https://doi.org/10.5194/egusphere-egu24-9446)

The full publication list is on [giswlh.github.io](https://giswlh.github.io/#publications).

## 📖 Citation

```bibtex
@article{wang2026canglong,
  title   = {Improved Skills for Predicting Sub-Seasonal to Seasonal Global Sea Surface Temperature},
  author  = {Wang, Longhao and Zhang, Xuanze and Leung, L. Ruby and Chiew, Francis H. S. and
             AghaKouchak, Amir and Ying, Kairan and Zhang, Yongqiang},
  journal = {Advanced Science},
  year    = {2026},
  pages   = {e78271},
  doi     = {10.1002/advs.78271}
}
```

## 🌐 Related

- [canglong.earth](https://canglong.earth) — live portal: ocean and ENSO globe, six-week atmosphere viewer.
- [WeatherAI](https://github.com/GISWLH/WeatherAI) — readable PyTorch ports of 20 weather and climate AI models, CAS-Canglong included.
- [WeatherPre](https://github.com/GISWLH/WeatherPre) — one-line forecasts and scoring across 29 AI and NWP models.

## License

Code: [MIT](LICENSE). The Pangu-style building blocks follow [WeatherLearn](https://github.com/lizhuoq/WeatherLearn) (MIT).
ERA5 is © ECMWF / Copernicus Climate Change Service. Forecasts are research products, not official warnings.

[![Star History Chart](https://api.star-history.com/svg?repos=GISWLH/CAS-Canglong&type=Date)](https://star-history.com/#GISWLH/CAS-Canglong&Date)

## 🌍 See it live on canglong.earth

<a href="https://canglong.earth"><img src="assets/readme/canglong_earth.webp" width="100%" alt="canglong.earth: interactive globes for the CAS-Canglong ocean, ENSO and atmospheric forecasts"></a>

[**canglong.earth**](https://canglong.earth) is the visualization service for CAS-Canglong. Every run is published there
as interactive 3-D globes that you can rotate, zoom and play through month by month or week by week.

| View | What it shows | Open |
|---|---|---|
| **Ocean & ENSO** | SST for each of the 16 forecast months on the 721 × 1440 grid, 10 m wind vectors, and Niño 3.4 against the IRI plume | [canglong.earth/sst](https://canglong.earth/sst/) |
| **Atmosphere** | weekly means of 36 surface and upper-air variables, weeks 1–6 | [canglong.earth/viewer.html](https://canglong.earth/viewer.html) |
| **Portal** | both forecast models and an overview of the system | [canglong.earth](https://canglong.earth) |

<div align="center">

**[Open canglong.earth →](https://canglong.earth)**

</div>

<!-- readme-visual-refresh:artwork-source -->
<sub>[Editable artwork source · logo, whiteboard banner and portal showcase](assets/readme/src/)</sub>
<!-- /readme-visual-refresh:artwork-source -->
