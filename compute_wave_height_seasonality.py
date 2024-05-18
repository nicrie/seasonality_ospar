# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

wave_clim = xr.open_dataarray("data/wave_height/wave_climatology.nc")

is_land = wave_clim.isnull().all("month")

max_month = wave_clim.argmax("month", skipna=False).where(~is_land)
max_diff = wave_clim.max("month") - wave_clim.min("month")

# %%
fig, ax = plt.subplots(
    ncols=2, figsize=(7.2, 2.8), subplot_kw={"projection": ccrs.PlateCarree()}, dpi=300
)

cmap = plt.get_cmap("twilight")
norm = colors.BoundaryNorm(
    np.arange(-0.5, 12, 1), cmap.N
)  # Set boundaries for each color


max_diff.plot(
    ax=ax[0],
    transform=ccrs.PlateCarree(),
    cbar_kwargs={"ticks": np.arange(0, 3.1, 0.5), "label": ""},
    vmin=0,
    vmax=3,
)
max_month.plot(
    ax=ax[1],
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    norm=norm,
    cbar_kwargs={"ticks": np.arange(0, 13, 1), "label": ""},
)

# change colorbar labels from 0..11 to Jan...Dec
cbar = ax[1].collections[0].colorbar
cbar.set_ticks(np.arange(0, 12, 1))
cbar.set_ticklabels(
    [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
)

for a in ax:
    a.add_feature(cfeature.LAND.with_scale("50m"), color=".3", zorder=1)

ax[0].set_title("A | Annual difference in wave height [in m]", loc="left")
ax[1].set_title("B | Annual peak month", loc="left")
plt.tight_layout()
plt.savefig("figs/figure_supp_wave_height.png")
plt.show()


# %%
