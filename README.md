<!-- readme-visual-refresh:banner -->
![CAS-Canglong — hand-drawn project banner](assets/readme/banner.svg)
<!-- /readme-visual-refresh:banner -->

<div align="center">

# 🐉 CAS-Canglong

**AI forecasts of global sea-surface temperature and ENSO, from one month to more than a year ahead**

[![Paper](https://img.shields.io/badge/Advanced%20Science-2026-1D4ED8.svg)](https://doi.org/10.1002/advs.78271)
[![arXiv](https://img.shields.io/badge/arXiv-2409.05369-B31B1B.svg)](https://arxiv.org/abs/2409.05369)
[![Live forecast](https://img.shields.io/badge/%F0%9F%A4%97%20Space-live%20forecast-FFD21E.svg)](https://huggingface.co/spaces/LonghaoWang/CAS-Canglong)
[![canglong.earth](https://img.shields.io/badge/portal-canglong.earth-0EA5E9.svg)](https://canglong.earth)
[![Weights](https://img.shields.io/badge/weights-figshare-556B2F.svg)](https://doi.org/10.6084/m9.figshare.26779969)
[![PyTorch port](https://img.shields.io/badge/PyTorch-WeatherAI-EE4C2C.svg?logo=pytorch&logoColor=white)](https://github.com/GISWLH/WeatherAI/blob/main/docs/models/canglong.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-14B8A6.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/GISWLH/CAS-Canglong?style=social)](https://github.com/GISWLH/CAS-Canglong/stargazers)

**English** · [简体中文](README_zh.md)

</div>

CAS-Canglong is a 3-D Earth-specific transformer that learns the coupled ocean–atmosphere system from ERA5 and predicts
monthly global sea-surface temperature (SST). In the *Advanced Science* paper it improves one-to-three-month SST skill by
**55.6 ± 14.1 % to 63.4 ± 15.5 %** over physics-based seasonal models in six ocean regions that dominate S2S variability
over land. The same model family now runs every month: a **16-month SST and ENSO outlook** and a **six-week atmospheric
outlook**, published on [canglong.earth](https://canglong.earth) and the
[Hugging Face Space](https://huggingface.co/spaces/LonghaoWang/CAS-Canglong).

## 📢 News

- **2026-10** · 🤗 [Live forecast Space](https://huggingface.co/spaces/LonghaoWang/CAS-Canglong): 16 monthly SST maps,
  the Niño 3.4 outlook against the IRI plume, regional outlooks and NetCDF downloads.
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

![CAS-Canglong forecast: SST anomaly map and Niño 3.4 outlook](assets/readme/latest_forecast.png)

Initialised with ERA5 monthly means through **September 2026**, the run predicts a Niño 3.4 anomaly of **+3.75 °C** in
October 2026. The anomaly decays to +2.9 °C in January 2027 and stays above +0.5 °C until October 2027. This is one
deterministic research run, not an official forecast; the [Space](https://huggingface.co/spaces/LonghaoWang/CAS-Canglong)
explains how to read it and its limits. Regenerate the figure without a GPU: `python scripts/plot_latest_forecast.py`.

## ✨ Highlights

- **Ocean memory, learned end to end.** Sixteen months of eight ERA5 fields (SST, sea-level pressure, latent heat flux,
  10 m winds, 500/850 hPa geopotential, net solar radiation) on the 0.25° grid are embedded as 3-D patches and processed by
  Earth-specific windowed self-attention.
- **Fast.** The 43.9 M-parameter paper model runs a full 721 × 1440 forecast in 0.8 s on an RTX 4090, or in about a
  minute on four CPU cores. Training took about 7 days on 4 × A100 80 GB.
- **Open and checkable.** Weights and data are on figshare; the notebook code is in this repository; an independent
  PyTorch re-implementation in WeatherAI loads the released checkpoint and gives identical output.
- **Operational.** Monthly 16-month SST and ENSO outlooks plus weekly atmospheric outlooks, with every product published
  as data rather than only as images.

## 🧭 Model family

| Model | Predicts | Grid | Where to get it |
|---|---|---|---|
| **CAS-Canglong** (paper) | monthly SST, 1–3 months ahead | 0.25° | this repo (`notebook/`), weights on [figshare](https://doi.org/10.6084/m9.figshare.26779969), [WeatherAI port](https://github.com/GISWLH/WeatherAI/blob/main/docs/models/canglong.md) |
| **CAS-Canglong operational SST** | monthly SST and Niño 3.4, 1–16 months ahead | 1° + learned 0.25° refiner | forecasts on [canglong.earth](https://canglong.earth/sst/) and the [Space](https://huggingface.co/spaces/LonghaoWang/CAS-Canglong); weights not released |
| **CAS-Canglong atmosphere V3.5** | weekly means of 36 variables, weeks 1–6 | 0.25°, published at 1° | forecasts on [canglong.earth](https://canglong.earth/viewer.html), [WeatherPre](https://github.com/GISWLH/WeatherPre) route `canglong`; weights not released |

The operational SST model is a one-pass Swin transformer that predicts all 16 months in a single forward pass. Its
checkpoint selection used agreement with the historical NWP multi-model mean and the current IRI ENSO plume, so its live
runs are not an independent skill test; published skill refers to the paper model. The atmospheric run is not
drift-corrected (high latitudes warm with lead time); read its fields as model output, not as calibrated anomalies.

## 🚀 Quick start

**1 · Latest forecast as data (no GPU, no ERA5)**

```python
import xarray as xr
from huggingface_hub import hf_hub_download

path = hf_hub_download("LonghaoWang/CAS-Canglong", "data/canglong_sst_init202609.nc", repo_type="space")
ds = xr.open_dataset(path)            # ssta / sst (1°), sst_025 (0.25°), nino34, observed initial month
print(ds.nino34.to_series().round(2))
```

`data/run.json` in the Space names the current file. `scripts/plot_latest_forecast.py` downloads and plots it.

**2 · Run the paper model with the official weights**

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

**3 · Six-week atmospheric outlook**

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
| [`scripts/`](scripts) | `plot_latest_forecast.py`: download and plot the latest forecast |
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

<!-- readme-visual-refresh:artwork-source -->
[Editable banner artwork · Excalidraw](assets/readme/banner.excalidraw)
<!-- /readme-visual-refresh:artwork-source -->
