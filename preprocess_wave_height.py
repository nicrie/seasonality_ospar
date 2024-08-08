# %%
import xarray as xr
from dask.diagnostics import ProgressBar

path = "data/physical/wave_height/"
file = "cmems_mod_glo_wav_my_0.2deg_PT3H-i_VHM0_15.00W-15.00E_35.00N-65.00N_2010-01-01-2021-01-01.nc"
chunks = {"time": -1, "latitude": 20, "longitude": 20}
wave = xr.open_mfdataset(path + file, chunks=chunks)
wave = wave["VHM0"]
wave

wave_clim = wave.groupby("time.month").mean()
with ProgressBar():
    wave_clim = wave_clim.compute()

wave_clim = wave_clim.rename({"longitude": "lon", "latitude": "lat"})

wave_stats = xr.Dataset({"wave_height": wave_clim, "wave_energy": wave_clim**2})

wave_stats.to_netcdf("data/physical/wave_height/wave_climatology.nc")

# %%
