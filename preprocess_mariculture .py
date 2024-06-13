# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from sklearn.metrics.pairwise import haversine_distances

from utils.definitions import nea_mask

# %%
# Load data
# =============================================================================
path = "data/economic/aquaculture/mariculture/all_marine_aquaculture_farms_sources_final.csv"
mari = pd.read_csv(path)
is_nea = mari.X.between(-15, 15) & mari.Y.between(35, 65)
mari = mari.loc[is_nea]

mari = mari.replace("Unfed or algae fed bivalve molluscs", "Bivalve molluscs")


# %%
# Filter the data to only include the countries of interest
# -----------------------------------------------------------------------------
countries = [
    "Spain",
    "France",
    "Netherlands",
    "Portugal",
    "Sweden",
    "Germany",
    "Denmark",
    "Ireland",
    "United Kingdom",
    "Norway",
    "Jersey",
]

mari = mari[mari["country"].isin(countries)]
# %%
sns.scatterplot(
    data=mari[mari.species_group == "Bivalve molluscs"],
    x="X",
    y="Y",
    alpha=0.05,
    size="tonnes_per_farm",
    legend=False,
)
# %%


rad_france = np.deg2rad(mari.loc[mari.country == "France"][["Y", "X"]].values)
rad_rest = np.deg2rad(mari.loc[mari.country != "France"][["Y", "X"]].values)

# Compute the distance matrix using haversine formula
dist_matrix_france = haversine_distances(rad_france)
dist_matrix_rest = haversine_distances(rad_rest)
# multiply by Earth radius to get kilometers
EARTH_RADIUS = 6371
dist_matrix_france = dist_matrix_france * EARTH_RADIUS
dist_matrix_rest = dist_matrix_rest * EARTH_RADIUS

# Set the diagonal to NaN to avoid self-comparisons
np.fill_diagonal(dist_matrix_france, np.nan)
np.fill_diagonal(dist_matrix_rest, np.nan)

# Set the lower triangle to NaN to avoid double counting
dist_matrix_france[np.tril_indices(dist_matrix_france.shape[0])] = np.nan
dist_matrix_rest[np.tril_indices(dist_matrix_rest.shape[0])] = np.nan

# %%
# for distances below 3 km. We set the threshold to 3 km (approx. 0.027 degrees)
min_dist_thrs = 3  # km
min_dist_thrs_degree = min_dist_thrs / (2 * np.pi * EARTH_RADIUS / 360)
min_dist_thrs_degree = np.round(min_dist_thrs_degree, 3)
print(f"Threshold for merging farms and areas: {min_dist_thrs_degree} degrees")


max_dist_thres = 10
a = np.where(dist_matrix_rest < max_dist_thres, dist_matrix_rest, np.nan)
b = np.where(dist_matrix_france < max_dist_thres, dist_matrix_france, np.nan)
ax = plt.axes()
ax.hist(
    a.ravel(),
    alpha=0.5,
    density=True,
    bins=np.linspace(0, max_dist_thres, 10),
    label="Rest of Europe",
)
ax.hist(
    b.ravel(),
    alpha=0.5,
    density=True,
    bins=np.linspace(0, max_dist_thres, 10),
    label="France",
)
ax.axvline(min_dist_thrs, color="black", linestyle="--")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Density")

ax.legend()

# %%


def points2density(data, species, min_dist_thrs_degree):
    data = data.loc[data.species_group == species]
    lons = data.X
    lats = data.Y
    xbins = np.arange(-14.9, 20.1, min_dist_thrs_degree)
    ybins = np.arange(34.9, 70.1, min_dist_thrs_degree)

    xcenters = (xbins[1:] + xbins[:-1]) / 2
    ycenters = (ybins[1:] + ybins[:-1]) / 2
    n, _, _ = np.histogram2d(lons, lats, bins=[xbins, ybins])
    return xr.DataArray(
        n.T,
        dims=["lat", "lon"],
        coords={"lat": ycenters, "lon": xcenters},
    )


species = mari.species_group.unique()
da = []
for sp in species:
    da.append(points2density(mari, sp, min_dist_thrs_degree))
da = xr.concat(da, pd.Index(species, name="species_group"))

# Apply the thresholding, per bin max 1 count
da = (da / da).fillna(0)

# The bin size is 0.027 degrees, which is approximately 3 km, so we have counts per 9 km^2
# However, due to the curvature of the earth, the area of a bin will be smaller towards the poles
# so we need to apply a correction factor to account for this.
coslat_weights = np.cos(np.deg2rad(da.lat))
da = da * coslat_weights

# Focus on NEA region
is_nea = nea_mask.mask(da).notnull()
da = da.where(is_nea, drop=True)

# Assign meta data
da.name = "mariculture_production_area_density"
da = da.assign_attrs(
    {"description": "Density of mariculture production areas", "units": "farms/9km^2"}
)

# %%
da.coarsen(lon=50, lat=50, boundary="trim").sum().plot(
    col="species_group", col_wrap=2, cmap="viridis", levels=[0, 1, 5, 10, 15, 20, 25]
)
# %%
da.coarsen(lon=10, lat=10, boundary="trim").sum().sum("species_group").plot(
    cmap="viridis"
)
# %%
# Store data
# =============================================================================
da.to_netcdf("data/economic/aquaculture/aquaculture_farm_density.nc")

# %%
