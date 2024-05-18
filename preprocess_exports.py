# %%
import os

import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from datatree import open_datatree
from statsmodels.tsa.seasonal import seasonal_decompose

# %%
# Load data
# =============================================================================
ospar = open_datatree("data/ospar/preprocessed.zarr", engine="zarr")
countries_nea = np.unique(ospar["preprocessed"].country)

path = "data/economic/exports/eumofa/"

# Get all .csv files in the directory
files = [f for f in os.listdir(path) if f.endswith(".csv")]

# Load all files into a single DataFrame using dask
trade_EU = dd.read_csv(path + "*_EU_*.csv", sep=";")
trade_non_EU = dd.read_csv(path + "*_non-EU_*.csv", sep=";")

trade_EU = trade_EU = trade_EU.rename(
    columns={
        "flow_typ": "flow_type",
        "partner_contry": "partner_country",
        " preservation": "preservation",
    }
)
trade_non_EU = trade_non_EU.rename(
    columns={
        "reporting country": "country",
        "partner country": "partner_country",
        "commodity group": "commodity_group",
        "flow type": "flow_type",
        "main commercial species": "main_commercial_species",
        "value(Eur)": "value(EUR)",
        "volume(Kg)": "volume(kg)",
    }
)
trade_EU = trade_EU.drop(columns=["intra_extra_EU"])


trade = dd.concat([trade_EU, trade_non_EU], axis=0)
trade = trade.drop(columns=["presentation", "preservation", "value(EUR)", "weight"])

exports = trade[trade["flow_type"] == "Export"].drop(columns=["flow_type"])

exports_nea = exports[exports["country"].isin(countries_nea)]

exports_nea = exports_nea.groupby(
    ["year", "month", "country", "commodity_group", "main_commercial_species"]
).sum()["volume(kg)"]
# %%
with ProgressBar():
    exports_nea = exports_nea.compute()

# %%
exports_nea_temp = exports_nea.reset_index("commodity_group")
exports_da = exports_nea_temp.to_xarray()["volume(kg)"]

# add commodity_group as an secondary coordinates to the main_commercial_species index
# get the 3rd and 4th index levels
index_commodity_group = exports_nea.index.get_level_values(3).values
index_main_commercial_species = exports_nea.index.get_level_values(4).values
# map each main_commercial_species to its corresponding commodity_group
mapping_species_to_group = dict(
    zip(index_main_commercial_species, index_commodity_group)
)
coords_group = [
    mapping_species_to_group[species]
    for species in exports_da.indexes["main_commercial_species"]
]
exports_da = exports_da.assign_coords(
    commodity_group=("main_commercial_species", coords_group)
)
exports_da = exports_da.rename({"main_commercial_species": "species"})
exports_da = exports_da.sortby("year").sortby("species").sortby("country")
exports_da.name = "exports"
exports_da.attrs["units"] = "tonnes"
exports_da = exports_da / 1000  # convert to tonnes


# %%
# convert the dimensions "year" and "month" to one single dimension "time" with a datetime index
exports_da = exports_da.stack(time=("year", "month"))
exports_da["time"] = pd.date_range(
    start="2009-01-01", periods=exports_da.sizes["time"], freq="M"
)
exports_da = exports_da.sel(time=slice(None, "2023"))

# %%
# Remove long term trends from data


def _np_seasonal_decompose(data):
    result = seasonal_decompose(data, model="additive", period=12)
    return (result.observed, result.seasonal, result.trend)


observed, seasonal, trend = xr.apply_ufunc(
    _np_seasonal_decompose,
    exports_da.where(exports_da > 1, 1).fillna(1),
    input_core_dims=[["time"]],
    output_core_dims=[["time"], ["time"], ["time"]],
    vectorize=True,
    dask="parallelized",
)

# %%
# Create monthly climatologies
# =============================================================================
mean_exports_per_year = trend.groupby("time.year").mean()
annual_weights = mean_exports_per_year / mean_exports_per_year.sum("year")
mean_exports_per_year.name = "exports"
mean_exports_per_year = mean_exports_per_year.assign_attrs(
    {
        "units": "tonnes",
        "description": "Long-term average exports of fish and seafood from NEA countries.",
    }
)

residuals = observed - trend
# convert "time" coordinate (which is datetime) of residuals to two separate coordinates "year" and "month"
residuals = residuals.assign_coords(
    year=residuals.time.dt.year, month=residuals.time.dt.month
)
residuals = residuals.set_index(time=["year", "month"]).unstack()


weighted_exports = (residuals * annual_weights).sum("year")
weighted_exports = weighted_exports.assign_attrs(
    {
        "units": "tonnes",
        "description": "Seasonal variations of exports of fish and seafood from NEA countries. First, long-term trends of exports have been removed. Then monthy climatologies were computed weighting the month of each year by the average (long-term) exports of that year.",
    }
)
weighted_exports.name = "exports"

# %%

ds = xr.Dataset(
    {"annual_mean": mean_exports_per_year, "seasonal_variations": weighted_exports}
)
ds = ds.assign_attrs(
    {
        "description": "Exports of fish and seafood from NEA countries. The dataset contains two variables: 'annual_mean' and 'seasonal_variations'. The 'annual_mean' variable represents the long-term average exports of fish and seafood from NEA countries. The 'seasonal_variations' variable represents the seasonal variations of exports of fish and seafood from NEA countries. First, long-term trends of exports have been removed. Then monthy climatologies were computed weighting the month of each year by the average (long-term) exports of that year.",
        "units": "tonnes",
    }
)
ds.to_netcdf("data/economic/exports/exports_nea_countries.nc")
# %%
