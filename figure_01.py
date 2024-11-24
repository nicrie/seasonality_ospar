# %%
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from cartopy.crs import NearsidePerspective, PlateCarree, TransverseMercator
from cartopy.feature import LAND, OCEAN, RIVERS
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

import utils.styles

utils.styles.set_theme()

# %%
# Load data
# =============================================================================
YEAR = 2001
QUANTITY = "absolute"
VARIABLE = "Plastic"
base_path = f"data/gpr/{QUANTITY}/{VARIABLE}/{YEAR}/"
results = xr.open_dataset(base_path + "effect_size_seasons.nc")

# %%
# Figure 1
# =============================================================================
effect_size = abs(results.effect_size_gp).max("combination")
eff_size = effect_size.where(effect_size > 0.5, 0.5)


def trans_effect_size(x):
    x_trans = (x - 0.5) / (1 - 0.5)
    x_trans = 10 + 200 * x_trans
    return np.where(x_trans < 5, 5, x_trans)


markersize = trans_effect_size(eff_size)
colors = results.max_hdi.max("combination")
is_above_95 = colors > 0.95

vmin = 0.5
vmax = 1
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)


proj = TransverseMercator(central_longitude=0.0, central_latitude=50.0)
extent = [-15, 13, 34, 64]

fig = plt.figure(figsize=(7.2, 9.0))
gs = GridSpec(
    2,
    2,
    figure=fig,
    hspace=0.0,
    wspace=0.01,
    width_ratios=[1, 1],
    height_ratios=[1, 1],
)
ax = fig.add_subplot(gs[:, :], projection=proj)


for i in np.argsort(colors).values:
    ec = "w"
    lw = 0.5
    if is_above_95[i]:
        ec = "C3"
        lw = 2

    ax.scatter(
        results.lon[i],
        results.lat[i],
        s=markersize[i],
        marker="o",
        c=colors[i],
        norm=norm,
        cmap="viridis",
        ec=ec,
        lw=lw,
        alpha=0.9,
        transform=PlateCarree(),
        zorder=10,
    )

for a in [ax]:
    a.add_feature(OCEAN.with_scale("50m"), facecolor=".1")
    a.add_feature(LAND.with_scale("50m"), facecolor=".25")
    a.add_feature(RIVERS.with_scale("50m"), color=".4", lw=0.5)
    a.set_extent(extent, crs=PlateCarree())


gl = ax.gridlines(
    lw=0.3,
    color=".5",
    alpha=1,
    linestyle="--",
    draw_labels=True,
    rotate_labels=False,
)
gl.xlocator = mticker.FixedLocator([-10, 0, 10])
gl.ylocator = mticker.FixedLocator([40, 50, 60])
gl.top_labels = False
gl.right_labels = False

ax.add_patch(
    mpl.patches.Rectangle(
        (0.6, 0.0),  # bottom left corner coordinates
        0.4,  # width
        0.4,  # height
        facecolor="black",
        alpha=0.9,
        transform=a.transAxes,
    )
)
# Colorbar for correlation
cax = a.inset_axes([0.65, 0.3, 0.3, 0.03])
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap="viridis"),
    cax=cax,
    orientation="horizontal",
)
cbar.set_ticks([0.5, 0.75, 1])
cbar.set_label("Confidence $\gamma$", color="white", size=9)  # colorbar title
cax.tick_params(axis="x", colors="white", labelsize=9, width=0.5)
cax.xaxis.set_label_position("top")  # Move cbar title to the top
cbar.outline.set_edgecolor("white")
cbar.outline.set_linewidth(0.5)

# Add legend circles (effect size)
lax1 = ax.inset_axes([0.65, 0.01, 0.3, 0.2], transform=ax.transAxes)
lax1.set_xlim(-1, 1)
lax1.set_xlim(-1, 1)
lax1.set_ylim(-1, 1)
lax1.patch.set_alpha(0.0)
lax1.spines["top"].set_visible(False)
lax1.spines["bottom"].set_visible(False)
lax1.spines["left"].set_visible(False)
lax1.spines["right"].set_visible(False)
lax1.tick_params(
    axis="both", which="both", bottom=False, top=False, left=False, right=False
)
lax1.set_xticks([])
lax1.set_yticks([])

legend_effect_size = np.array([0.5, 0.75, 1])  # Effect_size
circle_labels = [".50", ".75", "1."]
circle_sizes = trans_effect_size(legend_effect_size)  # Circle sizes
radii = np.sqrt(circle_sizes / np.pi) / 55
xlocs = [-0.6, -0.0, 0.6]
for x, r, s, lb in zip(xlocs, radii, circle_sizes, circle_labels):
    lax1.scatter(
        x,
        0.0 + r,
        s=s,
        edgecolors="w",
        facecolors="none",
    )
    lax1.text(x, -0.5, f"{lb}", color="white", fontsize=9, ha="center")
lax1.text(
    0,
    0.6,
    "Effect size of \nseasonal variations $\sigma$",
    color="white",
    fontsize=9,
    ha="center",
)

# Add globe with rectangle over Europe
proj_near_side = NearsidePerspective(
    central_longitude=-15.0, central_latitude=40.0, satellite_height=1e7
)
globe_ax = fig.add_axes([0.15, 0.7, 0.2, 0.2], projection=proj_near_side)
globe_ax.set_global()
globe_ax.add_feature(LAND, facecolor=".4")
globe_ax.add_feature(OCEAN, facecolor=".2")
rect = Rectangle(
    xy=(-15, 33),
    width=30,
    height=32,
    edgecolor="C1",
    facecolor="none",
    alpha=1,
    linewidth=1,
    transform=PlateCarree(),
)
globe_ax.add_patch(rect)
globe_ax.spines["geo"].set_edgecolor("w")
globe_ax.spines["geo"].set_linewidth(0.3)


# Add names of relevant seas
seas = ["Irish Sea", "Skagerrak", "North Sea", "Bay of Biscay"]
lons = [-4.2, 10, 2, -3]
lats = [53, 58, 56, 46]
txt_kws = dict(
    color="w", style="italic", fontsize=8, ha="center", transform=PlateCarree()
)
ax.text(3.5, 55, "North Sea", **txt_kws)
ax.text(-5, 45.5, "Bay of Biscay", **txt_kws)
ax.text(-5.1, 53, "IS", **txt_kws)
ax.text(9, 57.5, "SK", **txt_kws)


# Add reference circles of autocorrelation
# def km2deg(x):
#     return x * 360 / (2 * np.pi * 6371)


# r_km = np.array([16, 16, 12, 9])
# r_km *= 3
# r_km = r_km * np.cos(np.deg2rad(45))
# r_deg = km2deg(r_km)

# circle_centers = [(-12, 45), (-10, 45), (-8, 45), (-6, 45)]
# for center, radius in zip(circle_centers, r_deg):
#     ax.add_patch(
#         mpl.patches.Circle(
#             center,
#             radius,s
#             color="r",
#             transform=PlateCarree(),
#             zorder=100,
#         )
#     )

fig.savefig("figs/figure01.png", dpi=300, bbox_inches="tight")


# %%
# Figure 1 - Isolated approach (Supplementary)
# =============================================================================
effect_size = abs(results.effect_size_mann_whitney).max("combination")
eff_size = effect_size.where(effect_size > 0.5, 0.5)
markersize = trans_effect_size(eff_size)
markersize = np.where(effect_size.notnull(), markersize, np.nan)
markersize = np.where(results.pvals_kruskal_wallis.notnull(), markersize, np.nan)

colors = results.pvals_cr_mann_whitney.min("combination")
colors = 1 - colors
is_above_95 = colors > 0.95

vmin = 0.5
vmax = 1
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)


proj = TransverseMercator(central_longitude=0.0, central_latitude=50.0)

fig = plt.figure(figsize=(7.2, 9.0))
gs = GridSpec(
    2,
    2,
    figure=fig,
    hspace=0.0,
    wspace=0.01,
    width_ratios=[1, 1],
    height_ratios=[1, 1],
)
ax = fig.add_subplot(gs[:, :], projection=proj)


for i in np.argsort(colors).values:
    ec = "w"
    lw = 0.5
    marker = "o"
    s = markersize[i]
    c = colors[i]
    if is_above_95[i]:
        ec = "C3"
        lw = 2

    if np.isnan(s) or np.isnan(c):
        c = ".3"
        marker = "x"
        s = 10
        ec = "w"

    ax.scatter(
        results.lon[i],
        results.lat[i],
        s=s,
        marker=marker,
        c=c,
        norm=norm,
        cmap="viridis",
        ec=ec,
        lw=lw,
        alpha=0.9,
        transform=PlateCarree(),
    )

for a in [ax]:
    a.add_feature(OCEAN.with_scale("50m"), facecolor=".1")
    a.add_feature(LAND.with_scale("50m"), facecolor=".25")
    a.add_feature(RIVERS.with_scale("50m"), color=".4", lw=0.5)
    a.set_extent(extent, crs=PlateCarree())


gl = ax.gridlines(
    lw=0.3,
    color=".5",
    alpha=1,
    linestyle="--",
    draw_labels=True,
    rotate_labels=False,
)
gl.xlocator = mticker.FixedLocator([-10, 0, 10])
gl.ylocator = mticker.FixedLocator([40, 50, 60])
gl.top_labels = False
gl.right_labels = False

ax.add_patch(
    mpl.patches.Rectangle(
        (0.6, 0.0),  # bottom left corner coordinates
        0.4,  # width
        0.4,  # height
        facecolor="black",
        alpha=0.9,
        transform=ax.transAxes,
    )
)
# Colorbar for correlation
cax = ax.inset_axes([0.65, 0.3, 0.3, 0.03])
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap="viridis"),
    cax=cax,
    orientation="horizontal",
)
cbar.set_ticks([0.5, 0.75, 1])
cbar.set_label('"Confidence" (1 - $p$-value)', color="white", size=9)  # colorbar title
cax.tick_params(axis="x", colors="white", labelsize=9, width=0.5)
cax.xaxis.set_label_position("top")  # Move cbar title to the top
cbar.outline.set_edgecolor("white")
cbar.outline.set_linewidth(0.5)

# Add legend circles (effect size)
lax1 = ax.inset_axes([0.65, 0.01, 0.3, 0.2], transform=ax.transAxes)
lax1.set_xlim(-1, 1)
lax1.set_xlim(-1, 1)
lax1.set_ylim(-1, 1)
lax1.patch.set_alpha(0.0)
lax1.spines["top"].set_visible(False)
lax1.spines["bottom"].set_visible(False)
lax1.spines["left"].set_visible(False)
lax1.spines["right"].set_visible(False)
lax1.tick_params(
    axis="both", which="both", bottom=False, top=False, left=False, right=False
)
lax1.set_xticks([])
lax1.set_yticks([])

legend_effect_size = np.array([0.5, 0.75, 1])  # Effect_size
circle_labels = [".50", ".75", "1."]
circle_sizes = trans_effect_size(legend_effect_size)  # Circle sizes
radii = np.sqrt(circle_sizes / np.pi) / 55
xlocs = [-0.6, -0.0, 0.6]
for x, r, s, lb in zip(xlocs, radii, circle_sizes, circle_labels):
    lax1.scatter(
        x,
        0.0 + r,
        s=s,
        edgecolors="w",
        facecolors="none",
    )
    lax1.text(x, -0.5, f"{lb}", color="white", fontsize=9, ha="center")
lax1.text(
    0,
    0.6,
    "Effect size of \nseasonal variations $\sigma$",
    color="white",
    fontsize=9,
    ha="center",
)

# Add globe with rectangle over Europe
proj_near_side = NearsidePerspective(
    central_longitude=-15.0, central_latitude=40.0, satellite_height=1e7
)
globe_ax = fig.add_axes([0.15, 0.7, 0.2, 0.2], projection=proj_near_side)
globe_ax.set_global()
globe_ax.add_feature(LAND, facecolor=".4")
globe_ax.add_feature(OCEAN, facecolor=".2")
rect = Rectangle(
    xy=(-15, 33),
    width=30,
    height=32,
    edgecolor="C1",
    facecolor="none",
    alpha=1,
    linewidth=1,
    transform=PlateCarree(),
)
globe_ax.add_patch(rect)
globe_ax.spines["geo"].set_edgecolor("w")
globe_ax.spines["geo"].set_linewidth(0.3)


fig.savefig("figs/figure_supp01.png", dpi=300, bbox_inches="tight")

# %%

# %%
