# %%
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics.pairwise import haversine_distances
from tqdm import tqdm


def compute_distance_matrix(data, coastline, threshold=120):
    """Compute distance matrix between data and coastline.

    Parameters
    ----------
    data : pd.DataFrame
        Data locations.
    coastline : pd.DataFrame
        Coastline locations.
    threshold : float
        Maximum distance between farm and coastline in km.
        data further away are not considered.

    Returns
    -------
    pd.DataFrame
        Distance matrix between data and coastline.

    """
    latlons_data = data[["lat", "lon"]].values
    latlons_coastline = coastline[["lat", "lon"]].values

    rads_data = np.radians(latlons_data)
    rads_coastline = np.radians(latlons_coastline)

    multiindex = list(zip(coastline["lat"], coastline["lon"]))

    dist = haversine_distances(rads_data, rads_coastline) * 6371
    dist = pd.DataFrame(dist, index=data.index, columns=multiindex)

    # Only keep data within certain distance of coastline
    min_distance = dist.min(axis=1)
    return dist[min_distance <= threshold]


def fetch_river_emissions(river_data, dist, coastal_reference, threshold=120):
    """Fetch river emission.

    Parameters
    ----------
    dist : pd.DataFrame
        Distance matrix between data and coastline.
    coastal_reference : xr.DataArray
        Reference grid of coastline.

    Returns
    -------
    xr.DataArray
        Data reassigned to coastal grid.

    """

    prior = xr.zeros_like(coastal_reference)

    # iterate through rows
    for (lat, lon), series in tqdm(dist.items(), total=dist.shape[1]):
        if series.min() <= threshold:
            idx = series.idxmin()
            prior.loc[{"lon": lon, "lat": lat}] = river_data.loc[idx, "ma_mpw_dif"]

    return prior


# %%
# Load data
# =============================================================================
# Plastic emission by Strokal et al. 2023))
strokal = pd.read_excel(
    "data/physical/river/strokal2023/main_model_outputs.ods",
    sheet_name="MARINA-Plastics_results",
    usecols=[5, 6, 16],
)
strokal = strokal.rename(columns={"lon_outlet": "lon", "lat_outlet": "lat"})
# %%
is_europe = (
    (strokal["lat"] > 36)
    & (strokal["lat"] < 65)
    & (strokal["lon"] > -13)
    & (strokal["lon"] < 13)
)
is_mediterranean = (
    (strokal["lat"] > 35)
    & (strokal["lat"] < 43)
    & (strokal["lon"] > -4.8)
    & (strokal["lon"] < 16)
) | (
    (strokal["lat"] > 40)
    & (strokal["lat"] < 48)
    & (strokal["lon"] > 2)
    & (strokal["lon"] < 13)
)
plastic_emission = strokal.loc[is_europe & ~is_mediterranean]
plastic_emission.plot(
    x="lon", y="lat", kind="scatter", s=plastic_emission.ma_mpw_dif * 1e-2, alpha=0.5
)
plastic_emission.set_index(["lat", "lon"], inplace=True)

# %%
# EFAS river discharge
# =============================================================================
print("Loading EFAS river emission data...")

rd = xr.open_dataset("data/physical/river/efas_v5_discharge.nc")
rd = rd.rename({"latitude": "lat", "longitude": "lon"})

rd["annual_mean"] = rd["mean"].mean("month").fillna(0)
rd["scaling"] = rd["mean"] / rd["annual_mean"]

# %%
# Iterate through rows of plastic_emissions
weights = pd.DataFrame(index=plastic_emission.index, columns=rd.month.values)
for i, row in enumerate(plastic_emission.itertuples()):
    lat, lon = row.Index[0], row.Index[1]
    local_region = rd.sel(
        lat=slice(lat + 0.25, lat - 0.25), lon=slice(lon - 0.25, lon + 0.25), drop=True
    )
    if local_region["annual_mean"].size == 0:
        continue
    monthly_weights = (
        local_region["mean"]
        .weighted(local_region["annual_mean"])
        .mean(dim=["lat", "lon"])
    )
    monthly_weights = monthly_weights / monthly_weights.mean("month")
    weights.iloc[i] = monthly_weights.values * row.ma_mpw_dif

# %%
# Store data
# =============================================================================
weights = weights.to_xarray().to_dataarray("month")
weights.name = "macro_plastic_emission"
weights.to_netcdf("data/physical/river/strokal2023_efas_v5_weights.nc")

# %%
