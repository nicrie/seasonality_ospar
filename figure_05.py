# %%
from string import ascii_uppercase as ABC

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import seaborn as sns
import xarray as xr
from datatree import open_datatree
from matplotlib import colormaps
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon

import utils.styles
from utils.definitions import nea_ocean_basins

utils.styles.set_theme()

# %%
# Load data
# =============================================================================
# Beach litter clusters
pca_result = xr.open_dataset("data/pca/pca_beaches.nc", engine="netcdf4")
components = pca_result.comps
pcs = pca_result.scores

expvar_p = pca_result["expvar_ratio_pos"].assign_coords(mode=["1+", "2+", "3+", "4+"])
expvar_n = pca_result["expvar_ratio_neg"].assign_coords(mode=["1-", "2-", "3-", "4-"])
expvar = xr.concat([expvar_p, expvar_n], dim="mode").rename({"mode": "cluster"})
expvar.name = "explained_variance_ratio"


def trans_effect_size(s):
    return s * 5e2


def trans_prob(c):
    return c


# Effect size
s = components.mean("n")
s = abs(s).where(s >= 0)

# Percentage of components above zero
c = (components > 0).mean("n")

litter_cl1 = pd.DataFrame(
    {
        "lon": components.lon,
        "lat": components.lat,
        "comps": (s.sel(mode=1)),
        "size": trans_effect_size(s.sel(mode=1)),
        "probability": trans_prob(c.sel(mode=1)),
    }
)
litter_cl2 = pd.DataFrame(
    {
        "lon": components.lon,
        "lat": components.lat,
        "comps": (s.sel(mode=2)),
        "size": trans_effect_size(s.sel(mode=2)),
        "probability": trans_prob(c.sel(mode=2)),
    }
)
litter_cl1 = litter_cl1.dropna(subset=["size"]).sort_values(
    by="probability", ascending=False
)
litter_cl2 = litter_cl2.dropna(subset=["size"]).sort_values(
    by="probability", ascending=False
)

# Litter sources
sources = open_datatree("data/litter_sources.zarr", engine="zarr")

# Rivers
river = sources["river/discharge"].to_dataset()
thrs_discharge = river["size"].quantile(0.05)
river = river.where(river["size"] > thrs_discharge)
river_stacked = river.stack(point=["y", "x"]).dropna("point")

river = river_stacked.seasonal_potential
river_cl1 = river.sel(cluster=1, drop=True)
river_cl2 = river.sel(cluster=2, drop=True)

# Riverine plastic emissions (Strokal et al., 2023)
plastic = sources["river/plastic"].to_dataset()
plastic_stacked = plastic.stack(point=["lat", "lon"]).dropna("point")
plastic_stacked["mean_emission"] = plastic_stacked["size"]


# Fishing (capture)
fishing_capture = sources["fishing/capture/region"].to_dataset()
# fishing_capture = fishing_capture.where(nea_mask.mask(fishing_capture).notnull())

fish_capture_cl1 = (
    fishing_capture[["size", "seasonal_potential"]]
    .sel(cluster=1, drop=True)
    .to_dataframe()
)
fish_capture_cl2 = (
    fishing_capture[["size", "seasonal_potential"]]
    .sel(cluster=2, drop=True)
    .to_dataframe()
)

fish_capture_cl1 = fish_capture_cl1.sort_values(
    by="seasonal_potential", ascending=True
).dropna()
fish_capture_cl2 = fish_capture_cl2.sort_values(
    by="seasonal_potential", ascending=True
).dropna()
fish_capture_cl1 = fish_capture_cl1.reset_index()
fish_capture_cl2 = fish_capture_cl2.reset_index()

# %%
# Aquaculture
mari = sources["mariculture"].sel(forcing="exports", species_group="all").to_dataset()
mari = mari.stack(point=["lat", "lon"]).dropna("point")

# Icons
icons = {
    "river": "figs/icons/sources/png/003-river.png",
    "wild": "figs/icons/sources/png/001-fishing-boat.png",
    "mari": "figs/icons/sources/png/002-fish.png",
}
images = {k: PIL.Image.open(fname) for k, fname in icons.items()}


# %%
# Create figure
# =============================================================================
def trans_wild(x):
    return x ** (1) * 5e-1


def trans_mari(x):
    return x ** (1) * 20


def trans_mari_bivalve(x):
    return x ** (1) * 5


def trans_discharge(x):
    return x ** (1.5) * 6e-6


def trans_plastic(x):
    return x ** (0.8) * 1e-2


cmap = {
    "river": "mako",
    "plastic": "mako",
    "mari": "mako",
    "wild": "mako",
    "wild_size": "inferno",
}
norm = {
    "river": mcolors.Normalize(vmin=0, vmax=0.6),
    "plastic": mcolors.Normalize(vmin=0.0, vmax=0.6),
    "mari": mcolors.Normalize(vmin=0, vmax=0.6),
    "wild": mcolors.Normalize(vmin=0, vmax=0.6),
    "wild_size": mcolors.Normalize(vmin=0, vmax=1e6),
}

proj = ccrs.TransverseMercator(central_longitude=0.0, central_latitude=50.0)
extent = [-17, 25, 34, 64]

X_LABELS = ["Beach Litter\nCluster 1", "Beach Litter\nCluster 2"]
Y_LABELS = ["River discharge", "Fishing", "Mariculture"]

fig = plt.figure(figsize=(7.2, 9.72), dpi=300)
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], wspace=0, hspace=0)
ax = [fig.add_subplot(gs[i, j], projection=proj) for i in range(3) for j in range(2)]
axes_first_col = [ax[0], ax[2], ax[4]]
axes_second_col = [ax[1], ax[3], ax[5]]
axes_first_row = [ax[0], ax[1]]
# Add titles
for a, lb in zip(axes_first_row, X_LABELS):
    a.text(
        0.5,
        0.98,
        lb,
        transform=a.transAxes,
        ha="center",
        va="top",
        color="w",
    )
for a, lb in zip(axes_first_col, Y_LABELS):
    a.text(
        0.02,
        0.5,
        lb,
        transform=a.transAxes,
        ha="left",
        va="center",
        rotation=90,
        color="w",
    )


# Set up axes
for i, (a, d) in enumerate(zip(ax, ABC)):
    a.set_extent(extent, crs=ccrs.PlateCarree())
    # set background black
    a.set_facecolor("black")
    a.add_feature(cfeature.LAND.with_scale("50m"), color="k", zorder=1)
    # a.add_feature(LAND.with_scale("50m"), color="k")
    # a.add_feature(RIVERS.with_scale("50m"), color=".4", lw=0.2)
    a.coastlines("50m", color=".5", lw=0.2)
    a.text(0.99, 0.98, f"({d})", transform=a.transAxes, ha="right", va="top", color="w")

# Add source icons
clr_bg = {
    "river": sns.color_palette("mako", as_cmap=True),
    "wild": sns.color_palette("mako", as_cmap=True),
    "mari": sns.color_palette("mako", as_cmap=True),
}
for a, (src, img) in zip(axes_first_col, icons.items()):
    iax = a.inset_axes([0.0, 0.79, 0.2, 0.2], transform=a.transAxes, zorder=55)
    iax.imshow(images[src])

    # set background color
    iax.patch.set_facecolor(clr_bg[src](0.85))
    iax.patch.set_alpha(1.0)
    # set border color
    iax.patch.set_edgecolor("white")
    iax.set_xticks([])
    iax.set_yticks([])


# Scatter plots
# -----------------------------------------------------------------------------
# Beach litter
for a in [ax[0], ax[2], ax[4]]:
    a.scatter(
        litter_cl1.lon,
        litter_cl1.lat,
        s=litter_cl1["size"].where(litter_cl1["probability"] > 0.60),
        color="None",
        ec="#ea60ff",
        lw=0.75,
        zorder=500,
        transform=ccrs.PlateCarree(),
    )
for a in [ax[1], ax[3], ax[5]]:
    a.scatter(
        litter_cl2.lon,
        litter_cl2.lat,
        s=litter_cl2["size"].where(litter_cl2["probability"] > 0.60),
        color="None",
        ec="#ea60ff",
        lw=0.75,
        zorder=500,
        transform=ccrs.PlateCarree(),
    )

# River discharge
s_river = trans_discharge(river_stacked["size"])
ax[0].scatter(
    river_cl1.lon,
    river_cl1.lat,
    s=s_river,
    c=river_cl1,
    norm=norm["river"],
    cmap="mako",
    transform=ccrs.PlateCarree(),
)
ax[1].scatter(
    river_cl2.lon,
    river_cl2.lat,
    s=s_river,
    c=river_cl2,
    norm=norm["river"],
    cmap="mako",
    transform=ccrs.PlateCarree(),
)

# Riverine macroplastic emission
s_plastic = trans_plastic(plastic_stacked["size"])
ax[0].scatter(
    plastic_stacked.lon,
    plastic_stacked.lat,
    s=s_plastic,
    c=plastic_stacked.seasonal_potential.sel(cluster=1),
    ec="yellow",
    lw=0.2,
    norm=norm["plastic"],
    cmap="mako",
    transform=ccrs.PlateCarree(),
    zorder=40,
)
ax[1].scatter(
    plastic_stacked.lon,
    plastic_stacked.lat,
    s=s_plastic,
    c=plastic_stacked.seasonal_potential.sel(cluster=2),
    ec="yellow",
    lw=0.2,
    norm=norm["plastic"],
    cmap="mako",
    transform=ccrs.PlateCarree(),
    zorder=40,
)


# Fishing  (capture)
for region in nea_ocean_basins:
    for a, clstr in zip([ax[2], ax[3]], [1, 2]):
        # Get size and seasonal potential for each region/cluster
        val_sp = (
            fishing_capture["seasonal_potential"]
            .sel(cluster=clstr, region=region.number)
            .item()
        )
        val_size = fishing_capture["size"].sel(region=region.number).item()

        # Convert values to colors
        fc = colormaps[cmap["wild"]](norm["wild"](val_sp))
        ec = colormaps[cmap["wild_size"]](norm["wild_size"](val_size))

        # Define the coordinates of the polygon vertices
        polygon_coords = list(
            zip(*nea_ocean_basins[[region.number]].polygons[0].exterior.xy)
        )
        # Create a Polygon object
        polygon = Polygon(
            polygon_coords,
            closed=True,
            lw=1.5,
            ls="-",
            facecolor=fc,
            edgecolor=ec,
            transform=ccrs.PlateCarree(),
            zorder=0,
        )

        # Add the polygon to the plot
        centroid = polygon.xy.mean(0)
        a.add_patch(polygon)
        # Add region number as label in center of polygon
        a.annotate(
            f"{region.number}",
            xy=centroid,
            xytext=centroid,
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            zorder=1000,
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.7, pad=0.5),
            transform=ccrs.PlateCarree(),
        )

# ax[2].scatter(
#     fish_capture_cl1.lon,
#     fish_capture_cl1.lat,
#     s=trans_wild(fish_capture_cl1["size"]),
#     c=fish_capture_cl1.seasonal_potential,
#     norm=norm["wild"],
#     ec="None",
#     lw=0.3,
#     cmap=cmap["wild"],
#     transform=ccrs.PlateCarree(),
# )
# ax[3].scatter(
#     fish_capture_cl2.lon,
#     fish_capture_cl2.lat,
#     s=trans_wild(fish_capture_cl2["size"]),
#     c=fish_capture_cl2.seasonal_potential,
#     norm=norm["wild"],
#     ec="None",
#     lw=0.3,
#     cmap=cmap["wild"],
#     transform=ccrs.PlateCarree(),
# )

# Aquaculture
ax[4].scatter(
    mari.lon,
    mari.lat,
    s=trans_mari(mari["size"]),
    c=mari["seasonal_potential"].sel(cluster=1),
    norm=norm["mari"],
    cmap=cmap["mari"],
    ec="none",
    lw=0.3,
    transform=ccrs.PlateCarree(),
    alpha=0.7,
    zorder=0,
)
ax[5].scatter(
    mari.lon,
    mari.lat,
    s=trans_mari(mari["size"]),
    # c="orange",
    c=mari["seasonal_potential"].sel(cluster=2),
    norm=norm["mari"],
    cmap=cmap["mari"],
    ec="none",
    lw=0.2,
    transform=ccrs.PlateCarree(),
    alpha=0.7,
    zorder=0,
)


# Add legends
def add_legend_background(ax, x, y, dx, dy):
    ax.add_patch(
        mpl.patches.Rectangle(
            (x, y),  # bottom left corner coordinates
            dx,  # width
            dy,  # height
            fc="black",
            ec="white",
            fill=True,
            alpha=1,
            transform=ax.transAxes,
            zorder=50,
        )
    )


def add_seasonal_potential_colorbar(
    ax, x, y, dx, dy, cmap, norm, ticks, title, ticklabels=None
):
    if ticklabels is None:
        ticklabels = ticks

    cax = ax.inset_axes([x, y, dx, dy], zorder=55)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
    )
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.set_label(title, color="white", size=6)  # colorbar title
    cax.tick_params(axis="x", colors="white", labelsize=6, width=0.5)
    cax.xaxis.set_label_position("top")  # Move cbar title to the top
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(0.5)
    # Adjust colorbar tick length
    cbar.ax.tick_params(length=2)


add_legend_background(ax[1], 0.6, 0.0, 0.4, 0.6)
add_legend_background(ax[3], 0.6, 0.0, 0.4, 0.4)
add_legend_background(ax[5], 0.6, 0.0, 0.4, 0.4)

add_seasonal_potential_colorbar(
    ax[1],
    0.65,
    0.5,
    0.3,
    0.03,
    cmap["river"],
    norm["river"],
    [0, 0.3, 0.6],
    "Seasonal potential",
)
add_seasonal_potential_colorbar(
    ax[3],
    0.65,
    0.3,
    0.3,
    0.03,
    cmap["wild"],
    norm["wild"],
    [0, 0.3, 0.6],
    "Seasonal potential",
)
add_seasonal_potential_colorbar(
    ax[3],
    0.65,
    0.1,
    0.3,
    0.03,
    cmap["wild_size"],
    norm["wild_size"],
    [0, 500000, 1000000],
    "Fishing intensity\n[$10^6$ hrs/yr]",
    ["0", "0.5", "1"],
)
add_seasonal_potential_colorbar(
    ax[5],
    0.65,
    0.3,
    0.3,
    0.03,
    cmap["mari"],
    norm["mari"],
    [0, 0.3, 0.6],
    "Seasonal potential",
)


# Add legend circles (discharge)
def add_legend_circles(
    ax, x, y, dx, dy, sizes, labels, trans_func, xlocs=None, yoffset=0, title="", ec="w"
):
    lax = ax.inset_axes([x, y, dx, dy], transform=ax.transAxes, zorder=55)
    # lax = ax.inset_axes([0.65, 0.01, 0.3, 0.2], transform=ax.transAxes, zorder=55)
    lax.set_xlim(-1, 1)
    lax.set_xlim(-1, 1)
    lax.set_ylim(-1, 1)
    lax.patch.set_alpha(0.0)
    lax.spines["top"].set_visible(False)
    lax.spines["bottom"].set_visible(False)
    lax.spines["left"].set_visible(False)
    lax.spines["right"].set_visible(False)
    lax.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False
    )
    lax.set_xticks([])
    lax.set_yticks([])

    sizes_trans = trans_func(sizes)
    radii = np.sqrt(sizes_trans / np.pi) / 30
    xlocs = [-0.6, -0.05, 0.6] if xlocs is None else xlocs
    for x, r, s, lb in zip(xlocs, radii, sizes_trans, labels):
        lax.scatter(
            x,
            -0.2 + r,
            s=s,
            edgecolors=ec,
            facecolors="none",
        )
        lax.text(x, -0.8, f"{lb}", color="white", fontsize=6, ha="center")
    lax.text(
        0, 0.45 + yoffset, title, color="white", fontsize=6, ha="center", va="bottom"
    )


# Add legend circles
# -> Plastic emissions
sizes = np.array([1e3, 1e4, 1e5])
labels = ["1", "10", "100"]
title = "Macroplastics [$t/yr$]"
add_legend_circles(
    ax[1], 0.65, 0.21, 0.3, 0.2, sizes, labels, trans_plastic, title=title, ec="yellow"
)
# -> River discharge
sizes = np.array([1e3, 5e3, 1e4])
labels = ["1", "5", "10"]
title = "Discharge [$10^{3}m^3/s$]"
add_legend_circles(
    ax[1], 0.65, 0.01, 0.3, 0.2, sizes, labels, trans_discharge, title=title
)
# -> Fishing (capture)
# sizes = np.array([30, 100, 300])
# labels = ["30", "100", "300"]
# title = "Fishing Intensity\n[$hrs/1º x 1º$]"
# add_legend_circles(ax[3], 0.65, 0.01, 0.3, 0.2, sizes, labels, trans_wild, title=title)

# -> Aquaculture (farm density)
sizes = np.array([1, 3, 9])
labels = ["1", "3", "9"]
title = "Production areas\n[$60km^{-1}$ coastline]"
add_legend_circles(ax[5], 0.65, 0.01, 0.3, 0.2, sizes, labels, trans_mari, title=title)


plt.savefig("figs/figure05.png", dpi=500, bbox_inches="tight")
plt.show()

# %%
# Figures showing alternative hypotheses for mariculture
# =============================================================================
# Aquaculture
mari_scoures = sources["mariculture"].to_dataset()


X_LABELS = ["Beach Litter\nCluster 1", "Beach Litter\nCluster 2"]
Y_LABELS = [
    "Mariculture | Exports",
    "Mariculture | Ocean wave energy",
    "Mariculture | River discharge",
]

# %%
fig = plt.figure(figsize=(7.2, 9.72), dpi=300)
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], wspace=0, hspace=0)
ax = [fig.add_subplot(gs[i, j], projection=proj) for i in range(3) for j in range(2)]
axes_first_col = [ax[0], ax[2], ax[4]]
axes_second_col = [ax[1], ax[3], ax[5]]
axes_first_row = [ax[0], ax[1]]
# Add titles
for a, lb in zip(axes_first_row, X_LABELS):
    a.text(
        0.5,
        0.98,
        lb,
        transform=a.transAxes,
        ha="center",
        va="top",
        color="w",
    )
for a, lb in zip(axes_first_col, Y_LABELS):
    a.text(
        0.02,
        0.5,
        lb,
        transform=a.transAxes,
        ha="left",
        va="center",
        rotation=90,
        color="w",
    )


# Set up axes
for i, (a, d) in enumerate(zip(ax, ABC)):
    a.set_extent(extent, crs=ccrs.PlateCarree())
    # set background black
    a.set_facecolor("black")
    a.add_feature(cfeature.LAND.with_scale("50m"), color="k", zorder=1)
    a.coastlines("50m", color=".5", lw=0.2)
    a.text(0.99, 0.98, f"({d})", transform=a.transAxes, ha="right", va="top", color="w")

# Add source icons
iax = ax[0].inset_axes([0.0, 0.79, 0.2, 0.2], transform=ax[0].transAxes, zorder=55)
iax.imshow(images["mari"])

# set background color
iax.patch.set_facecolor(clr_bg["mari"](0.85))
iax.patch.set_alpha(1.0)
# set border color
iax.patch.set_edgecolor("white")
iax.set_xticks([])
iax.set_yticks([])


# Scatter plots
# -----------------------------------------------------------------------------
# Beach litter
for a in [ax[0], ax[2], ax[4]]:
    a.scatter(
        litter_cl1.lon,
        litter_cl1.lat,
        s=litter_cl1["size"].where(litter_cl1["probability"] > 0.60),
        color="None",
        ec="#ea60ff",
        lw=0.75,
        zorder=500,
        transform=ccrs.PlateCarree(),
    )
for a in [ax[1], ax[3], ax[5]]:
    a.scatter(
        litter_cl2.lon,
        litter_cl2.lat,
        s=litter_cl2["size"].where(litter_cl2["probability"] > 0.60),
        color="None",
        ec="#ea60ff",
        lw=0.75,
        zorder=500,
        transform=ccrs.PlateCarree(),
    )

# Aquaculture
forcings = ["exports", "wave", "river"]
for a, fc in zip([ax[0], ax[2], ax[4]], forcings):
    mari_sub = mari_scoures.sel(forcing=fc, species_group="all").sel(cluster=1)
    mari_sub = mari_sub.stack(point=["lat", "lon"]).dropna("point")
    a.scatter(
        mari_sub.lon,
        mari_sub.lat,
        s=trans_mari(mari_sub["size"]),
        c=mari_sub["seasonal_potential"],
        norm=norm["mari"],
        cmap=cmap["mari"],
        ec="none",
        lw=0.3,
        transform=ccrs.PlateCarree(),
        alpha=0.7,
        zorder=0,
    )
for a, fc in zip([ax[1], ax[3], ax[5]], forcings):
    mari_sub = mari_scoures.sel(forcing=fc, species_group="all").sel(cluster=2)
    mari_sub = mari_sub.stack(point=["lat", "lon"]).dropna("point")
    a.scatter(
        mari_sub.lon,
        mari_sub.lat,
        s=trans_mari(mari_sub["size"]),
        c=mari_sub["seasonal_potential"],
        norm=norm["mari"],
        cmap=cmap["mari"],
        ec="none",
        lw=0.2,
        transform=ccrs.PlateCarree(),
        alpha=0.7,
        zorder=0,
    )


# Add legends
def add_legend_background(ax, x, y, dx, dy):
    ax.add_patch(
        mpl.patches.Rectangle(
            (x, y),  # bottom left corner coordinates
            dx,  # width
            dy,  # height
            fc="black",
            ec="white",
            fill=True,
            alpha=1,
            transform=ax.transAxes,
            zorder=50,
        )
    )


add_legend_background(ax[5], 0.6, 0.0, 0.4, 0.4)

add_seasonal_potential_colorbar(
    ax[5],
    0.65,
    0.3,
    0.3,
    0.03,
    cmap["mari"],
    norm["mari"],
    [0, 0.3, 0.6],
    "Seasonal potential",
)

# Add legend circles
# -> Aquaculture (farm density)
sizes = np.array([1, 3, 9])
labels = ["1", "3", "9"]
title = "Production areas\n[$60km^{-1}$ coastline]"
add_legend_circles(ax[5], 0.65, 0.01, 0.3, 0.2, sizes, labels, trans_mari, title=title)

plt.savefig("figs/figure_supp05_mariculture_all.png", dpi=500, bbox_inches="tight")
plt.show()
# %%


fig = plt.figure(figsize=(7.2, 9.72))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], wspace=0, hspace=0)
ax = [fig.add_subplot(gs[i, j], projection=proj) for i in range(3) for j in range(2)]
axes_first_col = [ax[0], ax[2], ax[4]]
axes_second_col = [ax[1], ax[3], ax[5]]
axes_first_row = [ax[0], ax[1]]
# Add titles
for a, lb in zip(axes_first_row, X_LABELS):
    a.text(
        0.5,
        0.98,
        lb,
        transform=a.transAxes,
        ha="center",
        va="top",
        color="w",
    )
for a, lb in zip(axes_first_col, Y_LABELS):
    a.text(
        0.02,
        0.5,
        lb,
        transform=a.transAxes,
        ha="left",
        va="center",
        rotation=90,
        color="w",
    )


# Set up axes
for i, (a, d) in enumerate(zip(ax, ABC)):
    a.set_extent(extent, crs=ccrs.PlateCarree())
    # set background black
    a.set_facecolor("black")
    a.add_feature(cfeature.LAND.with_scale("50m"), color="k", zorder=1)
    a.coastlines("50m", color=".5", lw=0.2)
    a.text(0.99, 0.98, f"({d})", transform=a.transAxes, ha="right", va="top", color="w")

# Add source icons
iax = ax[0].inset_axes([0.0, 0.79, 0.2, 0.2], transform=ax[0].transAxes, zorder=55)
iax.imshow(images["mari"])

# set background color
iax.patch.set_facecolor(clr_bg["mari"](0.85))
iax.patch.set_alpha(1.0)
# set border color
iax.patch.set_edgecolor("white")
iax.set_xticks([])
iax.set_yticks([])


# Scatter plots
# -----------------------------------------------------------------------------
# Beach litter
for a in [ax[0], ax[2], ax[4]]:
    a.scatter(
        litter_cl1.lon,
        litter_cl1.lat,
        s=litter_cl1["size"].where(litter_cl1["probability"] > 0.60),
        color="None",
        ec="#ea60ff",
        lw=0.75,
        zorder=500,
        transform=ccrs.PlateCarree(),
    )
for a in [ax[1], ax[3], ax[5]]:
    a.scatter(
        litter_cl2.lon,
        litter_cl2.lat,
        s=litter_cl2["size"].where(litter_cl2["probability"] > 0.60),
        color="None",
        ec="#ea60ff",
        lw=0.75,
        zorder=500,
        transform=ccrs.PlateCarree(),
    )

# Aquaculture
forcings = ["exports", "wave", "river"]
for a, fc in zip([ax[0], ax[2], ax[4]], forcings):
    mari_sub = mari_scoures.sel(forcing=fc, species_group="Bivalve molluscs").sel(
        cluster=1
    )
    mari_sub = mari_sub.stack(point=["lat", "lon"]).dropna("point")
    a.scatter(
        mari_sub.lon,
        mari_sub.lat,
        s=trans_mari(mari_sub["size"]),
        c=mari_sub["seasonal_potential"],
        norm=norm["mari"],
        cmap=cmap["mari"],
        ec="none",
        lw=0.3,
        transform=ccrs.PlateCarree(),
        alpha=0.7,
        zorder=0,
    )
for a, fc in zip([ax[1], ax[3], ax[5]], forcings):
    mari_sub = mari_scoures.sel(forcing=fc, species_group="Bivalve molluscs").sel(
        cluster=2
    )
    mari_sub = mari_sub.stack(point=["lat", "lon"]).dropna("point")
    a.scatter(
        mari_sub.lon,
        mari_sub.lat,
        s=trans_mari(mari_sub["size"]),
        c=mari_sub["seasonal_potential"],
        norm=norm["mari"],
        cmap=cmap["mari"],
        ec="none",
        lw=0.2,
        transform=ccrs.PlateCarree(),
        alpha=0.7,
        zorder=0,
    )


# Add legends
def add_legend_background(ax, x, y, dx, dy):
    ax.add_patch(
        mpl.patches.Rectangle(
            (x, y),  # bottom left corner coordinates
            dx,  # width
            dy,  # height
            fc="black",
            ec="white",
            fill=True,
            alpha=1,
            transform=ax.transAxes,
            zorder=50,
        )
    )


add_legend_background(ax[5], 0.6, 0.0, 0.4, 0.4)

add_seasonal_potential_colorbar(
    ax[5],
    0.65,
    0.3,
    0.3,
    0.03,
    cmap["mari"],
    norm["mari"],
    [0, 0.3, 0.6],
    "Seasonal potential",
)

# Add legend circles
# -> Aquaculture (farm density)
sizes = np.array([1, 3, 9])
labels = ["1", "3", "9"]
title = "Production areas\n[$60km^{-1}$ coastline]"
add_legend_circles(ax[5], 0.65, 0.01, 0.3, 0.2, sizes, labels, trans_mari, title=title)

plt.savefig("figs/figure_supp05_mariculture_bivalve.png", dpi=500, bbox_inches="tight")
plt.show()

# %%
