# Daily 1-KM resolution PM2.5 for Contiguous US from 2005 to 2021

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567)
![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8-blue)

> **High-resolution daily PM₂․₅ maps (1 km) for the contiguous United States (2005-08-25 ➜ 2021-12-31),  
> generated with a Bi-LSTM-Attention model described in _Wang et al., Remote Sensing 17(1):126, 2025_.**

<p align="center">
  <img src="docs/_static/pred_map_wildfire_revised2.png" alt="Example map" width="85%">
</p>

---

## Table of Contents
1. [Dataset overview](#dataset-overview)  
2. [Quick start](#quick-start)  (under construction) 
3. [Install & environment](#install--environment)   
4. [Data sources](#data-sources)  
5. [Citation](#citation)  
6. [License](#license)  

---

## Dataset overview
* **Spatial coverage:** CONUS (lower 48 + DC)  
* **Spatial resolution:** 0.01° (~1 km) MODIS sinusoidal grid  
* **Temporal coverage:** 2005-08-25 → 2021-12-31 (2 392 days)  

---
## Quick start&nbsp;<img src="https://img.shields.io/badge/status-under%20construction-orange" alt="under construction" height="20">
```bash
# 1. Clone & install
git clone https://github.com/air-lab/pm25-1km.git
cd pm25-1km
mamba env create -f environment.yml
conda activate pm25-1km

# 2. Download a 3-day sample (~25 MB)
python scripts/download_sample.py --dest data/sample

# 3. Run inference for 2020-07-05
python scripts/infer.py --date 2020-07-05 --input data/sample \
                        --checkpoint checkpoints/bi_lstm_attention.pth \
                        --output outputs/pm25_2020-07-05.nc

# 4. Visualise
python notebooks/plot_example.py outputs/pm25_2020-07-05.nc
```

## Install & environment
### Prerequisites
* **Conda ≥ 4.12** (or the drop-in replacement **Mamba** for faster solves).  
  <small>Miniconda download: <https://docs.conda.io/en/latest/miniconda.html></small>
* **Python 3.8** (pinned inside the env file).
* **Git** to clone this repository.
* **GPU (optional)**  
  *Inference works on CPU; training the full 17-year model requires an NVIDIA card with ≥ 10 GB VRAM and CUDA 11.4.*

## Create the Conda environment
```bash
# with Conda
conda env create -f environment.yml

# …or with Mamba (recommended: ~5× faster)
mamba env create -f environment.yml

conda activate pm25-1km        # activate the new env
```

---

## Data Sources

| Data Source                | Variable                                                                                    | Description                                                                                                                                                                             | Temporal Resolution | Spatial Resolution  |
|----------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|---------------------|
| MODIS MCD19A2.061          | AOD 047 and AOD 055                                                                         | MODIS Terra and Aqua combined Multi-angle Implementation of Atmospheric Correction (MAIAC)  Land Aerosol Optical Depth (AOD) gridded Level 2 product produced daily at 1 km resolution. | Daily               | 1-km                |
| MODIS MCD43A4_006_NDVI     | NDVI                                                                                        | NDVI generated from the MODIS/006/MCD43A4 surface reflectance composites.                                                                                                               | 16-Day              | 0.5-km              |
| Daymet                     | Day Length, Pricipitation, Shortwave Radiation,  Max and Min Temperature, and Vapor Pressue | Daily Surface Weather and Climatological Summaries                                                                                                                                      | Daily               | 1-km                |
| gridMet                    | Wind direction and velocity                                                                 | A dataset of daily high-spatial resolution (~4-km, 1/24th degree) surface meteorological data covering  the contiguous US from 1979-yesterday.                                          | Daily               | 1/24th degree ~4-km |
| GMTED                      | DEM                                                                                         | The Global Multi-resolution Terrain Elevation Data 2010 (GMTED2010).                                                                                                                    | Static (2010)       | 1-km                |
| NOAA Hazard Mapping System | Wildfire Smoke Density                                                                      | NOAA's Hazard Mapping System (HMS) Smoke Product                                                                                                                                        | Daily               | 1-km                |
---
## Citation

If you use this repository or the derived 1-km PM₂․₅ dataset in your work, please cite:

```bibtex
@article{wang2025high,
  title   = {High-Resolution Estimation of Daily PM2.5 Levels in the Contiguous US Using Bi-LSTM with Attention},
  author  = {Wang, Zhongying and Crooks, James L. and Regan, Elizabeth Anne and Karimzadeh, Morteza},
  journal = {Remote Sensing},
  volume  = {17},
  number  = {1},
  pages   = {126},
  year    = {2025},
  publisher = {MDPI},
  doi     = {10.3390/rs17010126}
}
```
---

## License

Distributed under the **Apache License 2.0**.  
