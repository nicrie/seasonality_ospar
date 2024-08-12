
# Code Repository
Welcome to the code repository for our paper currently under revision.

## How to Reproduce the Results
To promote transparency, open science, and reproducibility, we have provided all necessary scripts for processing and analyzing the data, as well as for generating the main figures used in our study. 

## 1. Set Up the Python Environment
The analysis for this work was conducted using Python 3.10. Ensure you have the required software and dependencies installed. Refer to the `environment.yml` file for a list of necessary packages. You can install these dependencies using either `conda` or `mamba`:

```sh
conda env create -f environment.yml
conda activate ospar_seasonality
```

## 2. Obtain the Data
Please note: Due to the large size of some source data files, not all original data is provided within this repository. In such cases, we have made the post-processed data files available either within this repository or through an associated data repository hosted on [figshare](https://doi.org/10.6084/m9.figshare.26155156).

### Beach Litter Surveys
Source data is available in this repository at [`data/beach_litter/ospar/single/`](`data/beach_litter/ospar/single/`). Alternatively, OSPAR commission surveys can be downloaded as `.csv` files from [here](https://beachlitter.ospar.org/survey/export). 

### EFAS River Discharge
River discharge data for the European Flood Awareness System (EFAS) is provided by the Copernicus Climate Data Store (CDS) (https://doi.org/10.24381/cds.9f696a7a). We use version 5 of the dataset, covering 2010-2020 (approximately 873 GB). We provide a script for preprocessing the monthly files and a preprocessed data file on [figshare](https://doi.org/10.6084/m9.figshare.26155156).

### Riverine Plastic Emissions
Source data is available at [`data/physical/river/strokal2023/`](data/physical/river/strokal2023/). The original data for riverine plastic emissions can be obtained from Strokal et al. ([2023](https://doi.org/10.1038/s41467-023-40501-9)).


### Fishing Intensity
Global Fishing Watch AIS-based fishing effort data is available at 10th-degree resolution, version 2, covering 2012-2020 [here](https://globalfishingwatch.org/data-download/datasets/public-fishing-effort), accessible after free registration (Kroodsma et al. [2018](https://science.sciencemag.org/content/359/6378/904)). The dataset is aproximately 13 GB. We provide preprocessing scripts and preprocessed files.

### Mariculture Production Areas
Source data is available at [`data/economic/aquaculture/mariculture/all_marine_aquaculture_farms_sources_final.csv`](data/economic/aquaculture/mariculture/all_marine_aquaculture_farms_sources_final.csv). The original data can be obtained from Clawson et al. ([2022](https://doi.org/10.1016/j.aquaculture.2022.738066)).


### Ocean Wave Height
A preprocessed data file is available at [`data/physical/wave_height/wave_climatology.nc`](data/physical/wave_height/wave_climatology.nc). Monthly mean ocean wave height data can be obtained from the [Global Ocean Waves Reanalysis](https://doi.org/10.48670/moi-00022) dataset (2013-2020) provided by the E.U. Copernicus Marine Service Information (CMEMS). The dataset is approximately 1.5 GB. Download it by running:

```sh
python download_cmems_wave_height.py
```

> **_NOTE:_**  You need to register with CMEMS and set up your API key before running the script. Please check with CMEMS official instructions [here](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_WAV_001_032/description).


### EUMOFA Fish Exports
Preprocessed source data of fish exports is provided at [`/data/economic/exports/exports_nea_countries.nc`](/data/economic/exports/exports_nea_countries.nc). Monthly exports for fish and aquaculture products can be sourced from the European Market Observatory for Fisheries and Aquaculture Products ([EUMOFA](https://eumofa.eu/data)). The dataset is approximately 1.7 GB. Place individual files into `/data/economic/exports/eumofa/`.


### FAO Fish Production
Preprocessed data is available at [`data/economic/production/fao_fish_production.nc`](data/economic/production/fao_fish_production.nc). Source data can be obtained from the Food and Agriculture Organization (FAO) for [fisheries](https://www.fao.org/fishery/en/collection/capture?lang=en) and [aquaculture](https://www.fao.org/fishery/en/collection/aquaculture?lang=en). Download and extract the CSV raw data into `data/economic/production/Capture_2023.1.1/` and `/data/economic/production/Aquaculture_2023.1.1/`.


## 3. Pre-process the Data

#### Beach Litter Surveys

Preprocess OSPAR beach litter survey data:

```sh
python preprocess_ospar.py
```

The results will be stored at `data/beach_litter/ospar/preprocessed.zarr`.

### EFAS River Discharge

Preprocess EFAS river discharge:

```sh
python preprocess_river_efas_v5.py
```

> **_NOTE:_**  Source data not available within this repository.


### Riverine Plastic Emissions

Preprocess riverine plastic emissions:

```sh
python preprocess_river_strokal.py
```

> **_NOTE:_**  For seasonal scaling, donload `data/physical/river/efas_v5_discharge.nc` from the associated [figshare repository](https://doi.org/10.6084/m9.figshare.26155156).


### Fishing Intensity

Preprocess fishing intensity data:

```sh
python preprocess_fishing_capture.py

```

> **_NOTE:_**  Source data not available within this repository.

### Ocean Wave Height

Preprocess ocean wave height data:

```sh
python preprocess_wave_height.py
```

> **_NOTE:_**  Source data not available within this repository. However, the output is available within this repository.


### EUMOFA Fish Exports

Preprocess fish exports data:

```sh
python preprocess_exports.py
```

> **_NOTE:_**  Source data not available within this repository. However, the output is available within this repository.


### FAO Fish Production

Preprocess FAO fish production data:

```sh
python create_species_conversion_dict_fao_eumofa.py
python preprocess_fao.py
```

> **_NOTE:_**  Source data from FAO not available within this repository. However, the output is available within this repository.

### Mariculture Production Areas

Preprocess mariculture farm locations by Clawson et al:

```sh
python preprocess_mariculture.py
python preprocess_mariculture_seaosnality.py
```

## 4. Analyze the Data

#### Seasonality of Beach Litter

Perform Log-Gaussian Cox Process regression on beach litter data for each season:

```sh
python run_ospar_gpr.py
```
Perform a similar GP regression for the fraction of items related to land, aquaculture, and fishing-related processes. Set the `VARIABLE` in the following script accordingly:

```sh
python run_ospar_gpr_fraction.py
```

> **_NOTE:_**  This analysis requires sufficient memory (RAM) to run. Consider using HPC facilities. Results are available from the associated [figshare repository](https://doi.org/10.6084/m9.figshare.26155156).

Once you have the results (posterior probability distributions), either by running the scripts or by downloading the results from figshare, you can post-process the results to compute relevant statistical quantities such as effect size and confidence. Use the following script, setting the `VARIABLE` accordingly:

```sh
python postprocess_models.py
```

Perform PCA-clustering of the models, specifying the model with `VARIABLE`:

```sh
python perform_clustering_pca.py
```

Finally, compute the seasonal potential index for different beach litter sources:

```sh
python postprocess_litter_sources.py
```

> **_NOTE:_** Postprocessing beach litter sources requires all intermediate data results. The final postprocessed results are available in the associated figshare repository and contain all results in the zarr file `litter_sources.zarr`. This file is needed to produce figure 5.



## 5. Create the Figures

Generate the main figures (Figure 1 - 5) by running:

```
python figure_01.py
python figure_02.py
python figure_03.py
python figure_04.py
python figure_05.py
```
