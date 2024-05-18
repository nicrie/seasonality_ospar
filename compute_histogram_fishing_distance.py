# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask as rm
import seaborn as sns
import xarray as xr
from scipy.stats import cumfreq
from sklearn.metrics.pairwise import haversine_distances

from archive.utils.parcels import get_shore_nodes
from utils.definitions import nea_mask

# %%
# Aquaculture
# -----------------------------------------------------------------------------
aquaculture = pd.read_csv("data/aquaculture/aquaculture_shellfish_finfish.csv")

# Consider only aquaculture at sea, ignore land-based farms
aquaculture = aquaculture[aquaculture["POSITION_COASTLINE"] == "At sea"]


# %%
# Coastline
# -----------------------------------------------------------------------------
# Land sea mask
landmask = rm.defined_regions.natural_earth.land_10

lons = np.arange(-20, 15.1, 0.25)
lats = np.arange(35, 65.1, 0.25)
is_land = landmask.mask(lons, lats).notnull().astype(int)

shore_nodes = get_shore_nodes(is_land)
shore_nodes = xr.DataArray(
    shore_nodes,
    dims=("lat", "lon"),
    coords={"lat": is_land.lat.values, "lon": is_land.lon.values},
)

is_coastline = shore_nodes.where(shore_nodes == 1)
is_coastline = is_coastline.where(nea_mask.mask(is_coastline).notnull())

coastline = is_coastline.stack(lonlat=("lon", "lat")).dropna("lonlat")


# %%
# Fishing intensity
# -----------------------------------------------------------------------------
wild_capture = xr.open_dataarray("data/fishing/wild_capture_fishing_intensity.nc")

wild_capture = wild_capture.sum("time")

# %%

capture_in_nea = nea_mask.mask(wild_capture).notnull()
wild_capture_nea = wild_capture.where(capture_in_nea)
wild_capture_nea = wild_capture_nea.stack(x=[...])
wild_capture_nea = wild_capture_nea.where(wild_capture_nea != 0.0, drop=True)
wild_capture_nea = wild_capture_nea.dropna("x")


# %%
# Compute distance to coast
# =============================================================================


def _np_dist_to_coast(lat1, lon1, lat2, lon2):
    coords1 = np.vstack([lat1, lon1]).T
    coords2 = np.vstack([lat2, lon2]).T
    coords1 = np.radians(coords1)
    coords2 = np.radians(coords2)
    return haversine_distances(coords1, coords2).min(axis=1) * 6371


def dist_to_coast(lat, lon, lat_coast, lon_coast):
    return xr.apply_ufunc(
        _np_dist_to_coast,
        lat,
        lon,
        input_core_dims=[["x"], ["x"]],
        output_core_dims=[["x"]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float32],
        kwargs={"lat2": lat_coast.values, "lon2": lon_coast.values},
    )


dist_wild_capture = dist_to_coast(
    wild_capture_nea.lat, wild_capture_nea.lon, coastline.lat, coastline.lon
)

# %%
# Histogram plot
# =============================================================================


fig, ax = plt.subplots(2, 1, figsize=(7.2, 5), gridspec_kw={"hspace": 0.3})

bins = np.arange(0, 500, 20)
ax[0].hist(
    dist_wild_capture,
    weights=wild_capture_nea.values,
    bins=bins,
    cumulative=True,
    density=True,
    alpha=0.5,
    label="Wild capture",
)

bins = np.arange(0, 4.2, 0.25)
ax[1].hist(
    aquaculture.COAST_DIST_M * 1e-3,
    # weights=aqua_nea.values,
    bins=bins,
    cumulative=True,
    density=True,
    alpha=0.5,
    label="Aquaculture",
)
for a in ax:
    a.set_xlabel("Distance to coast [km]")
    a.set_ylabel("Accumulated probability [%]")

ax[0].set_title("A | Wild capture", loc="left")
ax[1].set_title("B | Aquaculture", loc="left")
sns.despine(fig)
plt.tight_layout()
plt.savefig("figs/figure_supp_fishing_distance_coastline.png", dpi=300)
plt.show()
# %%


# Calculate the cumulative frequency of the data
a = cumfreq(
    dist_wild_capture, numbins=len(dist_wild_capture), weights=wild_capture_nea.values
)

# Get the cumulative probability for 4 km
cum_prob_4km = (
    a.cumcount[
        np.digitize(
            4,
            a.lowerlimit + np.linspace(0, a.binsize * a.cumcount.size, a.cumcount.size),
        )
        - 1
    ]
    / a.cumcount[-1]
)

print("Cumulative probability for 4 km:", cum_prob_4km)
# %%
