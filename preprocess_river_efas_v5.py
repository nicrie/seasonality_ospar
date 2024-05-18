# %%
import os

import dask
import flox  # noqa
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm import tqdm


def preprocess(da):
    """Preprocess the river data."""
    return da.resample(time="1M").mean()
    # Group by month and calculate the mean


YEARS = range(2010, 2021)
MONTHS = range(1, 13)
root = "/media/nrieger/data/efas/v5/netcdf/"

file = "efas_{:}_{:02d}_river_discharge.nc"
# %%

for year in YEARS:
    for month in tqdm(MONTHS):
        print(f"Processing {year}-{month}...", end=" ", flush=True)
        if (year == 2018) & (month == 2):
            # This file seems corrupted by CDS
            print("Skip.", flush=True)
            continue
        target_file = file.format(year, month)
        path_to_file = os.path.join(root, target_file)
        save_to = os.path.join(root, "monthly", target_file)
        if os.path.exists(save_to):
            print("Skip.", flush=True)
            continue

        chunks_pre = {"time": -1, "longitude": 1000, "latitude": 500}
        with xr.open_dataset(path_to_file, chunks=chunks_pre) as ds:
            discharge = ds.resample(time="1M").mean()
            with ProgressBar():
                discharge = discharge.compute()

            discharge.to_netcdf(save_to)
        print("Done.", flush=True)


# %%

path_to_monthly = os.path.join(root, "monthly", "*.nc")
with xr.open_mfdataset(path_to_monthly) as ds:
    pass

with ProgressBar():
    save_to = os.path.join(root, "monthly", "complete", "efas_v5_river_discharge.nc")
    ds.to_netcdf(save_to)

# %%
chunks = {"time": -1, "longitude": 400, "latitude": 200}
with xr.open_mfdataset(save_to, chunks=chunks) as ds:
    discharge_mean = ds.groupby("time.month").mean()
    discharge_std = ds.groupby("time.month").std()


# %%


with ProgressBar():
    dmean, dstd = dask.compute(discharge_mean, discharge_std)

dmean = dmean.dis06
dstd = dstd.dis06

dmean.name = "discharge"
dstd.name = "discharge"

dmean.attrs["units"] = "m3/s"
dstd.attrs["units"] = "m3/s"
discharge = xr.Dataset(
    {"mean": dmean, "std": dstd},
    attrs={"description": "River discharge", "source": "EFAS v5"},
)

# %%
discharge.to_netcdf("data/physical/river/efas_v5_discharge.nc")
