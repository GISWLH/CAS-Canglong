# CAS-Canglong: Improved AI model for predicting global sea surface temperature

<img src="https://imagecollection.oss-cn-beijing.aliyuncs.com/office/20241101224146.png" style="zoom: 5%;" />

## Overview

[![DOI](https://zenodo.org/badge/DOI/10.6084/m9.figshare.24614958.svg)](doi.org/10.6084/m9.figshare.26779969)

``CAS-Canglong`` is a sub-seasonal prediction model 

The code is public and reproducible and easily accessible in this repository

Pre-trained models are provided, get them at the figshare link above!

Please consider cite this ref:

> Wang, L., Zhang, X., Leung, L. R., Chiew, F. H., AghaKouchak, A., Ying, K., & Zhang, Y. (2024). CAS-Canglong: A skillful 3D Transformer model for sub-seasonal to seasonal global sea surface temperature prediction. arXiv preprint arXiv:2409.05369.  

## Python Dependencies

Model trained on 4 × A100 80G GPUs

Model inference does not require a GPU

* torch: 2.1.0+cu118
* timm: 1.0.11
* numpy: 1.24.4

## Infer

Please see the  [Infer.ipynb](notebook\Infer.ipynb) notebook

```
import torch
DEVICE = torch.device("cuda:0")
the_model = torch.load('canglong3_0005_600ep_base.tar') #Get in the figshare
the_model.to(DEVICE)
rmse_list = []
r2_list = []
lead_2 = []
with torch.no_grad():
    the_model.eval()
    for step, (upper_air, target_surface) in enumerate(test_loader):
        upper_air, target_surface = upper_air.to(DEVICE), target_surface.to(DEVICE)
        output = the_model(upper_air.cuda())
        with torch.no_grad():
            sst1 = output[0, 0, 0, :, :].cpu().detach().numpy() * std_all.numpy()[0, 0, 0, 0]
            sst2 = target_surface[0, 0, :, :].cpu().detach().numpy() * std_all.numpy()[0, 0, 0, 0]

            sst1[ocean_mask] = None
            sst2[ocean_mask] = None
```

```
lat = np.linspace(90, -90, 721)
lon = np.linspace(0, 359.75, 1440)
sst_dataarray = xr.DataArray(
    sst1,
    coords=[("lat", lat), ("lon", lon)],
    name="sst"
)
fig = plt.figure()
proj = ccrs.Robinson() #ccrs.Robinson()ccrs.Mollweide()Mollweide()
ax = fig.add_subplot(111, projection=proj)
levels = np.linspace(-30, 30, num=19)
plot.one_map_flat(sst_dataarray, ax, levels=levels, cmap="RdBu_r", mask_ocean=False, add_coastlines=True, add_land=False, plotfunc="pcolormesh")
```

<img src="https://imagecollection.oss-cn-beijing.aliyuncs.com/office/20241101222715.png" style="zoom:50%;" />

## Training

Please see the [model.ipynb](notebook\model.ipynb) 

A100 80GB GPU recommended

## Website

CAS-canglong releases predicted global SSTs on its website

http://112.126.70.230/ 

![](https://imagecollection.oss-cn-beijing.aliyuncs.com/office/20241029132454.png)
