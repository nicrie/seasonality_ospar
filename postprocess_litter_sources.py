# %%
import pandas as pd
import xarray as xr
from datatree import DataTree

# Define the coordinates for the 9 regions based on the provided image
# These coordinates are approximate and should be adjusted as necessary
from utils.definitions import nea_ocean_basins


# %%
def monthly2seasonal(da, dim):
    """Convert any DataArray with monthly time dimension to seasonal sum.

    Seasonal sum is computed as follows:
    DJF: 12, 1, 2
    MAM: 3, 4, 5
    JJA: 6, 7, 8
    SON: 9, 10, 11

    New output dim is "month" with coordinates [1, 4, 7, 10]

    """
    # Define the seasons
    seasons = {
        1: [12, 1, 2],  # DJF
        4: [3, 4, 5],  # MAM
        7: [6, 7, 8],  # JJA
        10: [9, 10, 11],  # SON
    }
    # Create a dictionary with the sum of each season
    seasonal_sum = {}
    for season, months in seasons.items():
        seasonal_sum[season] = da.sel({dim: months}).sum(dim)
    # Concatenate the dictionary into a new DataArray
    seasonal_sum = xr.concat(
        seasonal_sum.values(), dim=pd.Index(seasons.keys(), name="season")
    )
    return seasonal_sum


# %%
# Load data
# =============================================================================
# Beach litter clusters
beach_litter = xr.open_dataset(
    "data/clustering/pca/absolute/Plastic/2001/pca_clustering.nc"
)
beach_litter = beach_litter.scores.sel(mode=[1, 2]).median("n")
beach_litter = beach_litter.assign_coords(season=[1, 4, 7, 10])
beach_litter = beach_litter.rename({"mode": "cluster"})

# River discharge
river_discharge_v5 = xr.open_dataset("data/physical/river/efas_v5_discharge.nc")["mean"]
river_discharge_v5 = river_discharge_v5.rename({"latitude": "lat", "longitude": "lon"})

river_discharge_v4 = xr.open_dataarray("data/physical/river/efas_river_discharge.nc")
river_discharge_v4 = river_discharge_v4.assign_coords(
    x=river_discharge_v4.x, y=river_discharge_v4.y
)

total_discharge_v4 = river_discharge_v4.sum("month")
cv_river_v4 = river_discharge_v4.std("month") / river_discharge_v4.mean("month")
cv_river_v4.name = "coefficient_of_variation"

total_discharge_v5 = river_discharge_v5.sum("month")
cv_river_v5 = river_discharge_v5.std("month") / river_discharge_v5.mean("month")
cv_river_v5.name = "coefficient_of_variation"

# Plastic emissions
plastic = xr.open_dataarray("data/physical/river/strokal2023_efas_v5_weights.nc")
cv_plastic = plastic.std("month") / plastic.mean("month")


# Fishing (World Fishing Watch) - Wild Capture
fish_wild = xr.open_dataarray("data/economic/fishing/wild_capture_fishing_intensity.nc")
fish_wild = fish_wild.resample(time="1ME").sum()

# Compute fish stats for meta regions
fish_wild_region = fish_wild.where(nea_ocean_basins.mask_3D(fish_wild)).sum(
    ("lat", "lon")
)

fish_wild_region = fish_wild_region.assign_coords(
    year=fish_wild_region.time.dt.year, month=fish_wild_region.time.dt.month
)
fish_wild_region = fish_wild_region.set_index(time=["year", "month"]).unstack()

fish_region = fish_wild_region.mean("year")
fish_region_std = fish_wild_region.std("year")
cv_fish_region = fish_region.std("month") / fish_region.mean("month")
cv_fish_region.name = "coefficient_of_variation"


fish_wild = fish_wild.groupby("time.month").mean()
cv_fish_wild = fish_wild.std("month") / fish_wild.mean("month")


# Aquaculture
mariculture = xr.open_dataset("data/economic/aquaculture/mariculture_seasonality.nc")
mariculture = mariculture.to_array("forcing")
cv_mariculture = mariculture.std("month") / mariculture.mean("month")


# %%
# Streamline datasets
# -----------------------------------------------------------------------------
river = xr.Dataset(
    {
        "climatology": river_discharge_v4,
        "size": river_discharge_v4.mean("month"),
        "cv": cv_river_v4,
    }
)
river_v5 = xr.Dataset(
    {
        "climatology": river_discharge_v5,
        "size": river_discharge_v5.mean("month"),
        "cv": cv_river_v5,
    }
)
plastic = xr.Dataset(
    {"climatology": plastic, "size": plastic.mean("month"), "cv": cv_plastic}
)
fishing = xr.Dataset(
    {"climatology": fish_wild, "size": fish_wild.sum("month"), "cv": cv_fish_wild},
    attrs=dict(
        description="Wild capture fishing intensity in average hours per month per 0.5ºx0.5º"
    ),
)
fishing_region = xr.Dataset(
    {"climatology": fish_region, "size": fish_region.sum("month"), "cv": cv_fish_region}
)
mari = xr.Dataset(
    {
        "climatology": mariculture,
        "size": mariculture.mean("month"),
        "cv": cv_mariculture,
    }
)


# %%
# Compute seasonal similarity between beach litter clusters and sources
# =============================================================================
def seasonal_potential(r, cv):
    return r * cv


# River discharge
river_seasonal = monthly2seasonal(river.climatology.roll(month=1), "month")
r_river = xr.corr(beach_litter, river_seasonal, dim="season")
r_river.name = "pearson_correlation"
r_river.attrs["description"] = (
    "Correlation coefficient between beach litter clusters and river discharge"
)
river["correlation"] = r_river
river["seasonal_potential"] = seasonal_potential(river.correlation, river.cv)

# River discharge v5
river_v5_seasonal = monthly2seasonal(river_v5.climatology.roll(month=1), "month")
r_river_v5 = xr.corr(beach_litter, river_v5_seasonal, dim="season")
r_river_v5.name = "pearson_correlation"
r_river_v5.attrs["description"] = (
    "Correlation coefficient between beach litter clusters and river_v5 discharge"
)
river_v5["correlation"] = r_river_v5
river_v5["seasonal_potential"] = seasonal_potential(river_v5.correlation, river_v5.cv)

# Riverine plastic emissions
plastic_seasonal = monthly2seasonal(plastic.climatology.roll(month=1), "month")
r_plastic = xr.corr(beach_litter, plastic_seasonal, dim="season")
r_plastic.name = "pearson_correlation"
r_plastic.attrs["description"] = (
    "Correlation coefficient between beach litter clusters and plastic emissions"
)
plastic["correlation"] = r_plastic
plastic["seasonal_potential"] = seasonal_potential(plastic.correlation, plastic.cv)

# Fishing (capture)
fish_seasonal = monthly2seasonal(fishing.climatology.roll(month=1), "month")
r_fish = xr.corr(beach_litter, fish_seasonal, dim="season")
r_fish.name = "pearson_correlation"
r_fish.attrs["description"] = (
    "Correlation coefficient between beach litter clusters and wild capture fishing intensity"
)
fishing["correlation"] = r_fish
fishing["seasonal_potential"] = seasonal_potential(fishing.correlation, fishing.cv)

# Fishing (capture, regional)
fish_region_seasonal = monthly2seasonal(
    fishing_region.climatology.roll(month=1), "month"
)
r_fish_region = xr.corr(beach_litter, fish_region_seasonal, dim="season")
r_fish_region.name = "pearson_correlation"
r_fish_region.attrs["description"] = (
    "Correlation coefficient between beach litter clusters and regional wild capture fishing intensity"
)
fishing_region["correlation"] = r_fish_region
fishing_region["seasonal_potential"] = seasonal_potential(
    fishing_region.correlation, fishing_region.cv
)


# Aquaculture (wave height, farm density)
mari_seasonal = monthly2seasonal(mari.climatology.roll(month=1), "month")
r_mari = xr.corr(beach_litter, mari_seasonal, dim="season")
r_mari.name = "pearson_correlation"
r_mari.attrs["description"] = (
    "Correlation coefficient between beach litter clusters and wave height"
)
mari["correlation"] = r_mari
mari["seasonal_potential"] = seasonal_potential(mari.correlation, mari.cv)


# %%
# Save some storage by removing the climatology
river = river.drop_vars("climatology")
river_v5 = river_v5.drop_vars("climatology")
plastic = plastic.drop_vars("climatology")
fishing = fishing.drop_vars("climatology")
fishing_region = fishing_region.drop_vars("climatology")
mari = mari.drop_vars("climatology")

# %%
sources = DataTree.from_dict(
    {
        "river/discharge": river,
        "river/discharge_v5": river_v5,
        "river/plastic": plastic,
        "fishing": fishing,
        "fishing/region": fishing_region,
        "mariculture": mari,
    }
)
sources.name = "litter_sources"

# %%
# Store the results
# =============================================================================
sources.to_zarr("data/litter_sources.zarr")
