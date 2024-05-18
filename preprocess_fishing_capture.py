# %%
import glob
import os

import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


def get_roi(df):
    """Get region of interest."""
    df = df.loc[df["lat"] < 65]
    df = df.loc[df["lat"] > 35]
    df = df.loc[df["lon"] > -20]
    df = df.loc[df["lon"] < 15]
    return df


# %%
# Define grid
# -----------------------------------------------------------------------------
step_size_lon = 0.25
step_size_lat = 0.25

lons = np.arange(-20, 15.2, step_size_lon)
lats = np.arange(35, 65.2, step_size_lat)

xedges = np.append(lons - step_size_lon / 2, lons[-1] + step_size_lon / 2)
yedges = np.append(lats - step_size_lat / 2, lats[-1] + step_size_lat / 2)


# %%
# Load data
# -----------------------------------------------------------------------------
path = "/media/nrieger/data/projects/basura/data/socio/fishing/*/"
all_files = glob.glob(os.path.join(path, "*.csv"))
YEARS = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]


fishing = []
for year in YEARS:
    print("Loading year ", year)
    files = [f for f in all_files if year in f]
    for f in tqdm(files):
        dtypes = {
            "date": str,
            "cell_ll_lat": np.float32,
            "cell_ll_lon": np.float32,
            "mmsi": np.int32,
            "hours": np.float32,
            "fishing_hours": np.float32,
        }
        df_one_day = dd.read_csv(f, dtype=dtypes)
        df_one_day.columns = ["time", "lat", "lon", "mmsi", "hours", "fishing_hours"]
        # Focus on North East Atlantic
        df_one_day = get_roi(df_one_day).compute()

        df_one_day.where(df_one_day["fishing_hours"] > 0, inplace=True)
        df_one_day = df_one_day.dropna()

        # Sum fishing hours per grid cell
        H, _, _ = np.histogram2d(
            df_one_day["lon"].values,
            df_one_day["lat"].values,
            bins=[xedges, yedges],
            weights=df_one_day["fishing_hours"],
        )
        H = xr.DataArray(
            H.T[None, ...],
            dims=["time", "lat", "lon"],
            coords={
                "time": [pd.to_datetime(df_one_day["time"].iloc[0])],
                "lat": lats,
                "lon": lons,
            },
        )
        fishing.append(H)

wild_capture = xr.concat(fishing, dim="time")
wild_capture = wild_capture.sortby("time")

# Save data
wild_capture.to_netcdf("data/economic/fishing/wild_capture_fishing_intensity.nc")
