# %%
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.crs import PlateCarree
from sklearn.metrics.pairwise import haversine_distances

from utils.definitions import nea_mask

# %%
# Load data
# =============================================================================
# Important! Some important production countries are missing in the data.
# Shellfish: Germany, Portugal, Sweden and Croatia. Except for Germany, the shellfish production remains limited in those MS.
# Finfish: Italy, Croatia, France, Portugal and Germany.

# Define the path to the shapefile
fname_shellfish_new = "EMODnet_HA_Aquaculture_Shellfish_20220225_Download.gdb"
fname_finfish_new = "EMODnet_HA_Aquaculture_Marine_Finfish_20210913_Download.gdb"

fname_shellfish = "EMODnet_HA_Aquaculture_Shellfish_20220225.gdb"
fname_finfish = "EMODnet_HA_Aquaculture_Marine_Finfish_20210913.gdb"
root = "data/economic/aquaculture/{:s}/"
path_shellfish = root.format("shellfish") + fname_shellfish
path_shellfish_new = root.format("shellfish_new") + fname_shellfish_new
path_finfish = root.format("marine_finfish") + fname_finfish

# Read the shapefile using geopandas
finfish_raw = gpd.read_file(path_finfish, layer=1)
shellfish_raw = gpd.read_file(path_shellfish, layer=0)

# There's a mismatch between the column names in the two datasets; we fix this
finfish_raw = finfish_raw.rename(columns={"COAST_DIST": "COAST_DIST_M"})

# Select the following columns which are of interest to us
columns_finfish = [
    "COUNTRY",
    "FARM_TYPE",
    "PRODUCTION_METHOD",
    "POINT_INFO",
    "COAST_DIST_M",
    "POSITION_COASTLINE",
    "PRODUCTS_DETAIL",
    "geometry",
]
columns_shellfish = columns_finfish + ["POINT_DEFINITION"]

finfish = finfish_raw[columns_finfish]
shellfish = shellfish_raw[columns_shellfish]

# %%
# What countries are represented in the data?
print("Countries (shellfish) : ", shellfish.COUNTRY.unique())
print("Countries (finfish) : ", finfish.COUNTRY.unique())

# %%
# Filter the data to only include the countries of interest
# -----------------------------------------------------------------------------
shellfish_countries = [
    "United Kingdom",
    "Ireland",
    "Denmark",
    "Spain",
    "Norway",
    "France",
    "Netherlands",
]

finfish_countries = [
    "Denmark",
    "Ireland",
    "United Kingdom",
    "Norway",
    "Spain",
]
finfish = finfish[finfish["COUNTRY"].isin(finfish_countries)]
shellfish = shellfish[shellfish["COUNTRY"].isin(shellfish_countries)]


# %%
# Add columns for latitude and longitude
# -----------------------------------------------------------------------------
finfish["lat"] = finfish["geometry"].apply(lambda point: point.y)
finfish["lon"] = finfish["geometry"].apply(lambda point: point.x)

shellfish["lat"] = shellfish["geometry"].apply(lambda point: point.y)
shellfish["lon"] = shellfish["geometry"].apply(lambda point: point.x)

# %%
# Add a column for the product type
# -----------------------------------------------------------------------------
shellfish["PRODUCT"] = "shellfish"
finfish["PRODUCT"] = "finfish"

# %%
# Manually add aquaculture farms in Portugal (Rocha et al. 2022)
# =============================================================================
# The data is not available in the EMODnet database, so we add it manually
# The data is taken from Rocha et al. 2022
# We assign the data to shellfish because the majority of the production is shellfish accroding to Rocha et al. 2022
portugal = pd.read_csv("data/economic/aquaculture/aquaculture_portugal.csv")
shellfish = pd.concat([shellfish, portugal], axis=0)
aquaculture = pd.concat([shellfish, finfish])
# %%
is_species = ["Oysters", "Mussels", "Mussels-Oysters"]


fig = plt.figure(figsize=(7.2, 3.5), dpi=500)
ax1 = fig.add_subplot(1, 2, 1, projection=PlateCarree())
ax2 = fig.add_subplot(1, 2, 2, projection=PlateCarree())
for a in [ax1, ax2]:
    a.coastlines(lw=0.3)
    a.set_extent([-15, 15, 35, 65])
aquaculture.plot.scatter(x="lon", y="lat", alpha=0.1, ax=ax1)
aquaculture.loc[aquaculture.FARM_TYPE.isin(is_species)].plot.scatter(
    x="lon", y="lat", alpha=0.1, ax=ax2
)

# %%
aquaculture.to_csv("data/economic/aquaculture/aquaculture_shellfish_finfish.csv")

# Consider only the aquaculture at sea
aquaculture = aquaculture.loc[aquaculture["POSITION_COASTLINE"] == "At sea"]
shellfish = shellfish.loc[shellfish["POSITION_COASTLINE"] == "At sea"]
finfish = finfish.loc[finfish["POSITION_COASTLINE"] == "At sea"]

# Consider only following production methods:
# Beds, Rafts, Longlines, Trestles or trays
valid_methods = [
    "Beds",
    "Rafts",
    "Long Lines",
    "Trestles or trays",
    "Sea cages",
    "Long Lines, Rafts",
    "Beds, Rafts, Trestles or trays",
    "Rafts, Other",
    "Beds, Trestles or trays",
    "Other",
    "n.a.",
]
shellfish = shellfish.loc[shellfish["PRODUCTION_METHOD"].isin(valid_methods)]
finfish = finfish.loc[finfish["PRODUCTION_METHOD"].isin(valid_methods)]

# %%
# Aquaculture locations are either represented as the center of a production area
# or as the location of a farm. We need to make these two comparable.
# -----------------------------------------------------------------------------
is_area = aquaculture["POINT_DEFINITION"] == "Production area"
is_farm = aquaculture["POINT_DEFINITION"] == "Farm"

n_areas, n_farms = is_area.sum(), is_farm.sum()
print(f"Number of areas: {n_areas}, number of farms: {n_farms}")

areas = aquaculture[is_area]
farms = aquaculture[is_farm]

rad_areas = np.deg2rad(areas[["lat", "lon"]].values)
rad_farms = np.deg2rad(farms[["lat", "lon"]].values)


# Compute the distance matrix using haversine formula
dist_matrix_areas = haversine_distances(rad_areas)
dist_matrix_farms = haversine_distances(rad_farms)
# multiply by Earth radius to get kilometers
EARTH_RADIUS = 6371
dist_matrix_areas = dist_matrix_areas * EARTH_RADIUS
dist_matrix_farms = dist_matrix_farms * EARTH_RADIUS

# Set the diagonal to NaN to avoid self-comparisons
np.fill_diagonal(dist_matrix_areas, np.nan)
np.fill_diagonal(dist_matrix_farms, np.nan)

# Set the lower triangle to NaN to avoid double counting
dist_matrix_areas[np.tril_indices(dist_matrix_areas.shape[0])] = np.nan
dist_matrix_farms[np.tril_indices(dist_matrix_farms.shape[0])] = np.nan

# %%
# We plot the pdf of pairwise distances for production areas and compare it with farms
# We expect that the pdfs are different for small distances, since the minimal distance
# between two farms will be much smaller than the minimal distance between two production areas
# By identifying this point where both pdfs differ, we can set a threshold below which we
# consider two points to be the same (basicially upscaling farms to production areas)
# -----------------------------------------------------------------------------

# From the figure it looks like there's a strong discrepenacy between the two distributions
# for distances below 3 km. We set the threshold to 3 km (approx. 0.027 degrees)
min_dist_thrs = 3  # km
min_dist_thrs_degree = min_dist_thrs / (2 * np.pi * EARTH_RADIUS / 360)
min_dist_thrs_degree = np.round(min_dist_thrs_degree, 3)
print(f"Threshold for merging farms and areas: {min_dist_thrs_degree} degrees")


max_dist_thres = 40
a = np.where(dist_matrix_areas < max_dist_thres, dist_matrix_areas, np.nan)
b = np.where(dist_matrix_farms < max_dist_thres, dist_matrix_farms, np.nan)
ax = plt.axes()
ax.hist(
    a.ravel(),
    alpha=0.5,
    density=True,
    bins=np.linspace(0, max_dist_thres, 50),
    label="areas",
)
ax.hist(
    b.ravel(),
    alpha=0.5,
    density=True,
    bins=np.linspace(0, max_dist_thres, 50),
    label="Farm",
)
ax.axvline(min_dist_thrs, color="black", linestyle="--")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Density")

ax.legend()


# %%


xbins = np.arange(-14.9, 20.1, min_dist_thrs_degree)
ybins = np.arange(34.9, 70.1, min_dist_thrs_degree)

xcenters = (xbins[1:] + xbins[:-1]) / 2
ycenters = (ybins[1:] + ybins[:-1]) / 2

n_farms_shellfish, _, _ = np.histogram2d(
    shellfish["lon"],
    shellfish["lat"],
    bins=[xbins, ybins],
)
n_farms_finfish, _, _ = np.histogram2d(
    finfish["lon"],
    finfish["lat"],
    bins=[xbins, ybins],
)

n_farms_shellfish = n_farms_shellfish.T
n_farms_finfish = n_farms_finfish.T

da_shellfish = xr.DataArray(
    n_farms_shellfish,
    dims=["lat", "lon"],
    coords={"lat": ycenters, "lon": xcenters},
    name="number_of_shellfish_farms",
    attrs={"description": "Aquaculture shellfish farms", "units": "number/4km^2"},
)
da_finfish = xr.DataArray(
    n_farms_finfish,
    dims=["lat", "lon"],
    coords={"lat": ycenters, "lon": xcenters},
    name="number_of_finfish_farms",
    attrs={"description": "Aquaculture finfish farms", "units": "number/4km^2"},
)
ds = xr.Dataset({"shellfish": da_shellfish, "finfish": da_finfish})

# Apply the thresholding, per bin max 1 count
ds = (ds / ds).fillna(0)

# The bin size is 0.018 degrees, which is approximately 2 km, so we have counts per 4 km^2
# However, due to the curvature of the earth, the area of a bin will be smaller towards the poles
# so we need to apply a correction factor to account for this.
coslat_weights = np.cos(np.deg2rad(ds.lat))
ds = ds * coslat_weights

# Get the total number of farms
ds["total"] = ds["shellfish"] + ds["finfish"]

# Focus on NEA region
is_nea = nea_mask.mask(ds).notnull()
ds = ds.where(is_nea)

# %%
# Store data
# =============================================================================
ds.to_netcdf("data/aquaculture/aquaculture_farm_density.nc")

# %%


slon, slat = 10, 10
fin = ds["finfish"].coarsen(lon=slon, lat=slat, boundary="trim").sum()
shell = ds["shellfish"].coarsen(lon=slon, lat=slat, boundary="trim").sum()
aqua = (
    ds.to_array("production")
    .sum("production")
    .coarsen(lon=slon, lat=slat, boundary="trim")
    .sum()
)
aqua.name = "aqua"

fin_dens = (
    fin.where(fin > 0, drop=True)
    .stack(loc=[...])
    .dropna("loc")
    .to_dataframe()["finfish"]
    .reset_index()
)
shell_dens = (
    shell.where(shell > 0, drop=True)
    .stack(loc=[...])
    .dropna("loc")
    .to_dataframe()["shellfish"]
    .reset_index()
)
aqua_dens = (
    aqua.where(aqua > 0, drop=True)
    .stack(loc=[...])
    .dropna("loc")
    .to_dataframe()["aqua"]
    .reset_index()
)


fig = plt.figure(figsize=(7.2, 3.5), dpi=500)
ax = [fig.add_subplot(1, 3, i, projection=ccrs.PlateCarree()) for i in range(1, 4)]
shell_dens.plot.scatter(
    x="lon", y="lat", s=shell_dens.shellfish * 1, alpha=0.5, ec=".3", ax=ax[0]
)
fin_dens.plot.scatter(
    x="lon", y="lat", s=fin_dens.finfish * 1, alpha=0.5, ec=".3", ax=ax[1]
)
aqua_dens.plot.scatter(
    x="lon", y="lat", s=aqua_dens.aqua * 1, alpha=0.5, ec=".3", ax=ax[2]
)
for a in ax:
    a.coastlines(lw=0.3)
    a.set_extent([-15, 15, 35, 65])
# %%
