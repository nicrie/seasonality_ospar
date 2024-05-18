# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.io import shapereader
from matplotlib.gridspec import GridSpec

import utils.styles

utils.styles.set_theme()

# %%
# Load data
# =============================================================================
ds = xr.open_dataset("data/aquaculture/aquaculture_farm_density.nc")
# %%
# ds has resolution of about 0.018 degrees (2km), we coarsen it to ~ 20 km
# so we have counts / 200 km^2
n_bins_coarsen = 15
unit_area = (2 * n_bins_coarsen) ** 2
print("Coarsening data to {} km2".format(unit_area))
counts = ds.coarsen(lon=n_bins_coarsen, lat=n_bins_coarsen, boundary="trim").sum()
counts = np.log10(counts + 1)

# %%
# get country borders
resolution = "10m"
category = "cultural"
name = "admin_0_countries"
shpfilename = shapereader.natural_earth(resolution, category, name)

# read the shapefile using geopandas
df = geopandas.read_file(shpfilename)


# List of missing countries
countries = [
    ["Germany", "Portugal", "Sweden"],
    ["Germany", "Portugal", "France"],
]

kwargs = dict(
    transform=ccrs.PlateCarree(),
    cmap="inferno",
    zorder=3,
    cbar_kwargs=dict(
        pad=0.0,
        extend="neither",
        orientation="horizontal",
        label=f"Production area density [log$_{{10}}$ #/{unit_area} km$^2$]",
    ),
)

fig = plt.figure(figsize=(7.2, 6), dpi=300)
gs = GridSpec(
    1,
    2,
    figure=fig,
    width_ratios=[1, 1],
    wspace=0.05,
    hspace=0.05,
)
ax = [
    fig.add_subplot(gs[0, 0], projection=ccrs.TransverseMercator()),
    fig.add_subplot(gs[0, 1], projection=ccrs.TransverseMercator()),
]
cax = [
    ax[0].inset_axes([0.0, 0.0, 1, 0.03]),
    ax[1].inset_axes([0.0, 0.0, 1, 0.03]),
]

for a in ax:
    a.coastlines("50m", lw=0.2, color="w", zorder=10, alpha=0.5)
    a.set_extent([-12, 12, 34, 65], crs=ccrs.PlateCarree())
    a.add_feature(
        cfeature.BORDERS, linestyle="-", color="w", lw=0.2, zorder=4, alpha=0.5
    )
    a.add_feature(cfeature.LAND, facecolor="k", zorder=1)
    a.add_feature(cfeature.OCEAN, facecolor="k", zorder=1)


counts.shellfish.plot(ax=ax[0], cbar_ax=cax[0], vmin=0, vmax=1.2, **kwargs)
counts.finfish.plot(ax=ax[1], cbar_ax=cax[1], vmin=0, vmax=1.2, **kwargs)

for i, a in enumerate(ax):
    for country in countries[i]:
        # read the borders of the country in this loop
        poly = df.loc[df["ADMIN"] == country]["geometry"].values[0]
        # plot the country on a map
        a.add_geometries(
            poly,
            crs=ccrs.PlateCarree(),
            facecolor=".3",
            edgecolor="none",
            zorder=5,
            alpha=0.5,
        )
ax[0].set_title("A | Shellfish", loc="left")
ax[1].set_title("B | Finfish", loc="left")
fig.subplots_adjust(left=0.05, right=0.95, top=0.99, bottom=0.01)

plt.savefig("figs/aquaculture_farm_density.png", dpi=300)

# %%
