# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask as rm
import xarray as xr
from matplotlib.gridspec import GridSpec

import utils.styles

utils.styles.set_theme()

lsm = rm.defined_regions.natural_earth_v5_0_0.land_10


seasons_dict = {
    "DJF": [12, 1, 2],
    "JFM": [1, 2, 3],
    "FMA": [2, 3, 4],
    "MAM": [3, 4, 5],
    "AMJ": [4, 5, 6],
    "MJJ": [5, 6, 7],
    "JJA": [6, 7, 8],
    "JAS": [7, 8, 9],
    "ASO": [8, 9, 10],
    "SON": [9, 10, 11],
    "OND": [10, 11, 12],
    "NDJ": [11, 12, 1],
}


# Compute seasonality as coefficient of variation (CV)
def compute_cv(da, dim):
    return da.std(dim=dim) / da.mean(dim=dim)


def compute_cv2(da, dim):
    return (da.std(dim=dim) / da.mean(dim=dim)) ** 2


def compute_robust_cv2(da, dim):
    return (abs(da - da.median(dim)).median(dim) / da.median(dim=dim)) ** 2


def seasonal_average(da):
    da = da.copy()
    bla = []
    for months in seasons_dict.values():
        bla.append(da.sel(month=months).mean("month"))
    da = xr.concat(bla, dim="month")
    da = da.assign_coords({"month": ("month", list(seasons_dict.keys()))})
    return da


# %%
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="lightgrey")
ax.set_extent([-15, 15, 30, 65])
ax.gridlines(draw_labels=True)

# Define meta regions
# =============================================================================
# North-East Atlantic
NEA = np.array([[-12, 36], [-1, 36], [5, 43], [13, 43], [13, 65], [-12, 65], [-12, 36]])

regions = rm.Regions(
    [NEA],
    names=["North-East Atlantic"],
    abbrevs=["NEA"],
    name="European regions",
)
regions.plot(ax=ax, resolution="50m")

# %%

# %%
usecols = [
    "X",
    "Y",
    "i_mid_jan",
    "i_mid_feb",
    "i_mid_mar",
    "i_mid_apr",
    "i_mid_may",
    "i_mid_jun",
    "i_mid_jul",
    "i_mid_aug",
    "i_mid_sep",
    "i_mid_oct",
    "i_mid_nov",
    "i_mid_dec",
]
lebreton = pd.read_csv(
    "data/river/lebreton2017/PlasticRiverInputs.csv", usecols=usecols
)
is_within_X = (lebreton.X > -15) & (lebreton.X < 15)
is_within_Y = (lebreton.Y > 30) & (lebreton.Y < 65)
is_valid = is_within_X & is_within_Y
lebreton = lebreton.loc[is_valid]


lb_river = lebreton.drop_duplicates(subset=["X", "Y"])
lb_river = lb_river.to_xarray()
lb_river = lb_river.assign_coords(
    {"lon": ("index", lb_river.X.data), "lat": ("index", lb_river.Y.data)}
)
lb_river = lb_river.drop_vars(["X", "Y"])
lb_river = lb_river.to_array("month", "influx")
lb_river = lb_river.assign_coords({"month": ("month", np.arange(1, 13))})
lb_river = lb_river.dropna("index")

lb_river.name = "emissions"
lb_river.attrs["units"] = "tonnes/month"
lb_river.attrs["long_name"] = "Riverine plastic emissions"
lb_river.attrs["source"] = "Lebreton et al. 2017"

mean_plastics = lb_river.mean("month")
cv_plastic = lb_river.std("month") / mean_plastics

plastic = xr.Dataset(
    {
        "emissions": lb_river,
        "annual_mean": mean_plastics,
        "cv": cv_plastic,
    }
)

plastic = plastic.where(plastic.annual_mean > 0, drop=True)

# %%
# EFAS river discharge
# -----------------------------------------------------------------------------
efas = xr.open_dataset("data/river/efas_v5_discharge.nc")
efas_mask = regions.mask(efas.longitude, efas.latitude)

efas = efas.where(efas_mask == 0, drop=True)


efas["annual_mean"] = efas["mean"].mean("month")
is_important_river = efas["annual_mean"] > 1e-2

efas = efas.where(is_important_river)
efas["cv"] = efas["mean"].std("month") / efas["annual_mean"]

efas_stacked = efas[["annual_mean", "cv"]].stack(loc=[...]).dropna("loc")

# %%


# %%
x1 = efas_stacked.annual_mean.values
y1 = efas_stacked.cv.values

x2 = plastic.annual_mean.values * 1e3  # in kg/month
y2 = plastic.cv.values


lognorm1 = mcolors.LogNorm(vmin=1e1, vmax=1e4)
lognorm2 = mcolors.LogNorm(vmin=1e-1, vmax=1e3)


fig = plt.figure(figsize=(7.2, 3))
gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 0.05, 0.5, 1, 0.05], wspace=0.0)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 3])
cax1 = fig.add_subplot(gs[0, 1])
cax2 = fig.add_subplot(gs[0, 4])

ax1.set_xscale("log")
ax2.set_xscale("log")

xbins1 = np.logspace(-2, 4, 40)
ybins2 = np.linspace(0, 3, 40)
hs1 = ax1.hist2d(x1, y1, bins=[xbins1, ybins2], cmap="Blues", norm=lognorm1)

xbins2 = np.logspace(-5, 6, 20)
ybins2 = np.linspace(0, 5, 20)
hs2 = ax2.hist2d(x2, y2, bins=[xbins2, ybins2], cmap="Blues", norm=lognorm2)

cbar1 = plt.colorbar(hs1[3], cax=cax1, orientation="vertical")
cbar2 = plt.colorbar(hs2[3], cax=cax2, orientation="vertical")

cbar1.set_label("Number of rivers")
cbar2.set_label("Number of rivers")

ax1.set_xlabel("Annual mean river discharge [m$^3$/s]")
ax1.set_ylabel("Coefficient of variation")
ax1.set_title("A | River discharge", loc="left")

ax2.set_xlabel("Annual mean plastic emission [kg/month]")
ax2.set_ylabel("Coefficient of variation")
ax2.set_title("B | Riverine plastic emissions", loc="left")

# Example: Rhine river
rhine_discharge = efas.sel(
    latitude=slice(51.83, 51.81), longitude=slice(4.05, 4.1)
).mean(("latitude", "longitude"))
rhine_plastic = plastic.sel(index=24316)

ax1.scatter(
    rhine_discharge["annual_mean"],
    rhine_discharge["cv"],
    s=30,
    ec="k",
    lw=0.5,
    color="C1",
)
ax2.scatter(
    rhine_plastic["annual_mean"] * 1e3,
    rhine_plastic["cv"],
    s=30,
    ec="k",
    lw=0.5,
    color="C1",
)
ax1.annotate(
    "Rhine river",
    xy=(rhine_discharge["annual_mean"], rhine_discharge["cv"] + 0.05),
    xytext=(1e1, 1.5),
    xycoords="data",
    color="C1",
    weight=800,
    # curved arrow
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc3,rad=-0.3",
        lw=0.5,
        color="C1",
    ),
)

plt.savefig("figs/rivers_discharge_vs_seasonality.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
