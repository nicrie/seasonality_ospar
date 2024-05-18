# %%
import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from scipy.interpolate import NearestNDInterpolator
from tqdm import tqdm

from utils.definitions import nea_mask

# %%
# Load data
# =============================================================================
# Farm density
mariculture = xr.open_dataarray("data/economic/aquaculture/aquaculture_farm_density.nc")
mariculture = mariculture.coarsen(lon=20, lat=20, boundary="trim").mean()


# Oceanic wave energy (extreme weather, oceanic)
wave_energy = xr.open_dataset("data/wave_height/wave_climatology.nc")
wave_energy = wave_energy["wave_energy"]

wave_energy = xr.Dataset(
    {"monthly_mean": wave_energy, "annual_mean": wave_energy.mean("month")}
)

# River discharge (extreme weather; continental)
river_discharge = xr.open_dataset("data/river/efas_v5_discharge.nc")
river_discharge = river_discharge.rename({"latitude": "lat", "longitude": "lon"})
river_discharge = river_discharge.sortby("lat")
river_discharge = river_discharge.rename({"mean": "monthly_mean"})
river_discharge["annual_mean"] = river_discharge["monthly_mean"].mean("month").fillna(0)

# Exports of fish and seafood from NEA countries
exports = xr.open_dataset("data/economic/exports/exports_nea_countries.nc")
production = xr.open_dataset("data/economic/production/fao_fish_production.nc")

# production_commodity_group = production["average_live_weight"].sel(production_type="aqua").groupby("commodity_group").sum()
production_weights = production["average_live_weight"].sel(production_type="aqua")
production_weights = production_weights.where(production_weights > 500)


def convert_species_groups(da):
    group_mapping = {
        "Bivalve molluscs": ["Bivalves and other molluscs and aquatic invertebrates"],
        "General marine fish": [
            "Cephalopods",
            "Flatfish",
            "Groundfish",
            "Other marine fish",
            "Small pelagics",
        ],
        "Bluefin tuna": ["Tuna and tuna-like species"],
        "Salmonidae fish": ["Salmonids"],
    }
    new_groups = []
    for c in da.commodity_group.values:
        if c in group_mapping["Bivalve molluscs"]:
            new_groups.append("Bivalve molluscs")
        elif c in group_mapping["General marine fish"]:
            new_groups.append("General marine fish")
        elif c in group_mapping["Bluefin tuna"]:
            new_groups.append("Bluefin tuna")
        elif c in group_mapping["Salmonidae fish"]:
            new_groups.append("Salmonidae fish")
        else:
            new_groups.append("Other")
    new_groups = xr.IndexVariable("species_group", new_groups)
    return da.rename({"commodity_group": "species_group"}).assign_coords(
        species_group=("species", new_groups)
    )


production_weights = convert_species_groups(production_weights)
exports_group = convert_species_groups(exports["seasonal_variations"])


# Normalize the seasonal variations of the exports
# such that the sum of the seasonal variations over all months is 1
def min_max_normalize(da, dim):
    return (da - da.min(dim)) / (da.max(dim) - da.min(dim))


exports_normalized = min_max_normalize(exports_group, "month")
exports_normalized = exports_normalized / exports_normalized.sum("month")

has_valid_seasonality = exports_normalized.notnull().all("month")
# %%
# Weigh the seasonal export cycle of each country/species by the overall aquaculture production

# Normalize the production weights by the sum of the production weights of the same commodity group
# Consider weights only for species/countries that have seasonality
production_weights = production_weights.where(has_valid_seasonality)

ref = production_weights.groupby("species_group").sum()
species2group = {
    sp: com
    for sp, com in zip(
        production_weights.species.values, production_weights.species_group.values
    )
}
for species in production_weights.species.values:
    norm_value = ref.loc[{"species_group": species2group[species]}]
    production_weights.loc[{"species": species}] = (
        production_weights.loc[{"species": species}] / norm_value
    )


# Group the exports by species group
exports_scaled = exports_normalized * production_weights
exports_species_group = exports_scaled.groupby("species_group").sum()


# %%

# Load the world countries dataset
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# Create a mask
country_mask = regionmask.mask_geopandas(
    world, lon_or_obj=mariculture.lon, lat=mariculture.lat
)


# Fill the NaN values with the nearest neighbour

indices = np.where(np.isfinite(country_mask))
interp = NearestNDInterpolator(np.transpose(indices), country_mask.data[indices])
country_mask[...] = interp(*np.indices(country_mask.shape))

country_mask = country_mask.where(nea_mask.mask(country_mask).notnull())

region2country = {
    21: "Norway",
    43: "France",
    110: "Sweden",
    121: "Germany",
    128: "Luxmbourg",
    129: "Belgium",
    130: "Netherlands",
    131: "Portugal",
    132: "Spain",
    133: "Ireland",
    142: "Denmark",
    143: "United Kingdom",
    162: "Marocco",
}

# Apply the mask to your data
mariculture_ext = []
for region, country in region2country.items():
    mariculture_ext.append(mariculture.where(country_mask == region, drop=False))

mariculture_ext = xr.concat(
    mariculture_ext, pd.Index(region2country.values(), name="country")
)

# %%
mariculture_exports = (mariculture_ext * exports_species_group).sum("country")


# %%


MONTHS = wave_energy.month
coastline = (
    mariculture.isel(species_group=0, drop=True)
    .where(mariculture.sum("species_group") > 0, drop=True)
    .stack(point=["lat", "lon"])
    .dropna("point")
    .point
)


# %%
def get_weights_from_coastline(reference_data, coastline):
    weights = []
    for lat, lon in tqdm(coastline.point.values):
        local_region = reference_data.sel(
            lat=slice(lat - 0.25, lat + 0.25),
            lon=slice(lon - 0.25, lon + 0.25),
            drop=True,
        )
        local_region = local_region.fillna(0)
        monthly_weights = (
            local_region["monthly_mean"]
            .weighted(local_region["annual_mean"])
            .mean(dim=["lat", "lon"])
        )
        monthly_weight_normed = monthly_weights / monthly_weights.mean("month")
        weights.append(monthly_weight_normed)
    weights = xr.concat(
        weights,
        dim=xr.IndexVariable(
            "point",
            pd.MultiIndex.from_tuples(coastline.point.values, names=["lat", "lon"]),
        ),
    )
    return weights.set_index(point=["lat", "lon"])


# %%
river_weights = get_weights_from_coastline(river_discharge, coastline)
wave_weights = get_weights_from_coastline(wave_energy, coastline)

seasonal_weights = xr.Dataset({"river": river_weights, "wave": wave_weights})
seasonal_weights = seasonal_weights.unstack()

seasonal_weights["river"] = seasonal_weights["river"] * mariculture
seasonal_weights["wave"] = seasonal_weights["wave"] * mariculture

seasonal_weights["exports"] = mariculture_exports
seasonal_weights = seasonal_weights.drop(["step", "surface"])
# %%
seasonal_weights.to_netcdf("data/economic/aquaculture/mariculture_seasonality.nc")

# %%

# %%
