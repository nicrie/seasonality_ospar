# %%
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from cartopy.crs import PlateCarree, TransverseMercator
from cartopy.feature import LAND, OCEAN, RIVERS
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

import utils.styles

utils.styles.set_theme()

# %%
# Load data
# =============================================================================
pca_result = xr.open_dataset("data/pca/pca_beaches.nc", engine="netcdf4")
components = pca_result.comps
pcs = pca_result.scores

expvar_p = pca_result["expvar_ratio_pos"].assign_coords(mode=["1+", "2+", "3+", "4+"])
expvar_n = pca_result["expvar_ratio_neg"].assign_coords(mode=["1-", "2-", "3-", "4-"])
expvar = xr.concat([expvar_p, expvar_n], dim="mode").rename({"mode": "cluster"})
expvar.name = "explained_variance_ratio"


def trans_effect_size(s):
    return s * 2e3


def trans_prob(c):
    return c


# Effect size
s = components.mean("n")
s = abs(s).where(s >= 0)

# Percentage of components above zero
c = (components > 0).mean("n")

cl1 = pd.DataFrame(
    {
        "lon": components.lon,
        "lat": components.lat,
        "comps": (s.sel(mode=1)),
        "size": trans_effect_size(s.sel(mode=1)),
        "probability": trans_prob(c.sel(mode=1)),
    }
)
cl2 = pd.DataFrame(
    {
        "lon": components.lon,
        "lat": components.lat,
        "comps": (s.sel(mode=2)),
        "size": trans_effect_size(s.sel(mode=2)),
        "probability": trans_prob(c.sel(mode=2)),
    }
)

# %%
# Figure - PCA eigenvectors and projections
# =============================================================================
seasons = pca_result.season
extent = [-15, 13, 34, 64]
proj = TransverseMercator(central_latitude=50)

norm = mcolors.Normalize(vmin=0.5, vmax=0.8)
cmap = sns.color_palette("inferno", as_cmap=True)
cmap_clrs = sns.color_palette("inferno", as_cmap=False, n_colors=4)
clr_highlight = cmap_clrs[3]

fig = plt.figure(figsize=(7.2, 4.6))
gs = GridSpec(
    1,
    3,
    figure=fig,
    hspace=0.02,
    wspace=0.0,
    width_ratios=[1, 1, 0.05],
)
ax1 = fig.add_subplot(gs[0, 0], projection=proj)
ax2 = fig.add_subplot(gs[0, 1], projection=proj)
cax = fig.add_subplot(gs[0, 2])

for ax in [ax1, ax2]:
    ax.add_feature(OCEAN, facecolor=".1")
    ax.add_feature(LAND, facecolor=".25")
    ax.add_feature(RIVERS, color=".4", lw=0.5)
    ax.set_extent(extent, crs=PlateCarree())

ax1.scatter(
    components.lon,
    components.lat,
    s=trans_effect_size(s.sel(mode=1)),
    c=c.sel(mode=1).values,
    norm=norm,
    cmap=cmap,
    transform=PlateCarree(),
    ec="w",
    lw=0.5,
    alpha=0.5,
)
ax2.scatter(
    components.lon,
    components.lat,
    s=trans_effect_size(s.sel(mode=2)),
    c=c.sel(mode=2).values,
    norm=norm,
    cmap=cmap,
    transform=PlateCarree(),
    ec="w",
    lw=0.5,
    alpha=0.5,
)

cbar3 = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=cax,
    label="Principal components above zero",
)
cbar3.set_ticks([0.5, 0.6, 0.7, 0.8])
cbar3.ax.set_yticklabels(["50%", "60%", "70%", "80%"])

xticks = np.arange(0.0, 3.5)
ax_pc1 = fig.add_axes([0.34, 0.2, 0.15, 0.08], facecolor=".1", frameon=False)
df_pcs1 = pcs.sel(mode=1, drop=True).to_dataframe().reset_index()
sns.barplot(
    df_pcs1,
    x="season",
    y="scores",
    hue="season",
    palette=[".5", clr_highlight, ".5", ".5"],
    zorder=1,
    ax=ax_pc1,
    err_kws={"color": "w"},
)
ax_pc1.set_title("PC$1^+$ scores", color="w", size=9)
ax_pc1.set_xticks(xticks)
ax_pc1.set_xticklabels(seasons, color="w", size=7)

ax_pc2 = fig.add_axes([0.73, 0.2, 0.15, 0.08], facecolor=".1", frameon=False)
df_pcs2 = pcs.sel(mode=2, drop=True).to_dataframe().reset_index()
sns.barplot(
    df_pcs2,
    x="season",
    y="scores",
    hue="season",
    palette=[clr_highlight, clr_highlight, ".5", ".5"],
    zorder=1,
    ax=ax_pc2,
    err_kws={"color": "w"},
)
ax_pc2.set_title("PC$2^+$ scores", color="w", size=9)
ax_pc2.set_xticks(xticks)
ax_pc2.set_xticklabels(seasons, color="w", size=7)

ax_pc1.set_yticks([])
ax_pc2.set_yticks([])

ax_pc1.set_xlabel("")
ax_pc2.set_xlabel("")

ax_pc1.tick_params(axis="x", length=0)
ax_pc2.tick_params(axis="x", length=0)


ax1.add_patch(
    Rectangle(
        xy=(0.55, 0),
        width=0.5,
        height=0.33,
        linewidth=1,
        edgecolor="k",
        facecolor="k",
        zorder=20,
        alpha=0.6,
        transform=ax1.transAxes,
    )
)
ax2.add_patch(
    Rectangle(
        xy=(0.55, 0),
        width=0.5,
        height=0.33,
        linewidth=1,
        edgecolor="k",
        facecolor="k",
        zorder=20,
        alpha=0.6,
        transform=ax2.transAxes,
    )
)

exp_var_ratio = pca_result.expvar_ratio_pos.quantile([0.5, 0.025, 0.975], "n") * 100
mid1, low1, up1 = exp_var_ratio.sel(mode=1)
mid2, low2, up2 = exp_var_ratio.sel(mode=2)

ax1.text(
    0.01,
    0.99,
    "A | Cluster $C1^+$",
    transform=ax1.transAxes,
    fontsize=10,
    fontweight="bold",
    color="w",
    va="top",
)
ax2.text(
    0.01,
    0.99,
    "B | Cluster $C2^+$",
    transform=ax2.transAxes,
    fontsize=10,
    fontweight="bold",
    color="w",
    va="top",
)

# Explained variance
# -----------------------------------------------------------------------------
ax_expvar1 = ax1.inset_axes(
    [0.01, 0.8, 0.4, 0.08], facecolor=".1", frameon=False, transform=ax1.transAxes
)
sns.barplot(
    x=[mid1.item()],
    y=["Explained Variance"],
    ax=ax_expvar1,
    color=clr_highlight,
    edgecolor="black",
    linewidth=1,
)
ax_expvar1.errorbar(
    x=mid1,
    y=["Explained Variance"],
    xerr=[[mid1 - low1], [up1 - mid1]],
    fmt="none",
    color=".9",
    capsize=2,
)

ax_expvar1.text(
    5,
    0,
    f"{mid1.item():.1f}%",
    color="w",
    ha="left",
    va="center",
    weight="bold",
)

ax_expvar1.text(
    0,
    0.5,
    "Explained variance",
    color="w",
    style="italic",
    size=8,
    transform=ax_expvar1.transData,
)

ax_expvar1.set_xlim([0, 50])
ax_expvar1.set_ylim([-0.5, 0.5])
ax_expvar1.set_xticks([])
ax_expvar1.set_yticks([])
ax_expvar1.set_xlabel("")
ax_expvar1.spines["left"].set_visible(False)
ax_expvar1.spines["right"].set_visible(False)
ax_expvar1.spines["top"].set_visible(False)
ax_expvar1.spines["bottom"].set_visible(False)

# -----------------------------------------------------------------------------
ax_expvar2 = ax2.inset_axes(
    [0.01, 0.8, 0.4, 0.08], facecolor=".1", frameon=False, transform=ax2.transAxes
)
sns.barplot(
    x=[mid2.item()],
    y=["Explained Variance"],
    ax=ax_expvar2,
    color=clr_highlight,
    edgecolor="black",
    linewidth=1,
)
ax_expvar2.errorbar(
    x=mid2,
    y=["Explained Variance"],
    xerr=[[mid2 - low2], [up2 - mid2]],
    fmt="none",
    color=".9",
    capsize=2,
)

ax_expvar2.text(
    1,
    0,
    f"{mid2.item():.1f}%",
    color="w",
    ha="left",
    va="center",
    weight="bold",
)

ax_expvar2.text(
    0,
    0.5,
    "Explained variance",
    color="w",
    style="italic",
    size=8,
    transform=ax_expvar2.transData,
)

ax_expvar2.set_xlim([0, 50])
ax_expvar2.set_ylim([-0.5, 0.5])
ax_expvar2.set_xticks([])
ax_expvar2.set_yticks([])
ax_expvar2.set_xlabel("")
ax_expvar2.spines["left"].set_visible(False)
ax_expvar2.spines["right"].set_visible(False)
ax_expvar2.spines["top"].set_visible(False)
ax_expvar2.spines["bottom"].set_visible(False)


plt.savefig("figs/figure03.png", dpi=300, bbox_inches="tight")

# %%
# Explained variance ratio
# =============================================================================
df = (
    expvar.sel(cluster=["1+", "2+", "1-", "2-", "3+", "3-"])
    .to_dataframe()
    .reset_index()
)
palette = sns.color_palette(["r", "r", ".8", ".8", ".8", ".8"], desat=0.8)
fig = plt.figure(figsize=(5.2, 3.5), dpi=500)
ax = fig.add_subplot(111)
sns.violinplot(
    ax=ax,
    data=df,
    x="cluster",
    hue="cluster",
    y="explained_variance_ratio",
    bw_adjust=1,
    cut=1,
    linewidth=1,
    palette=palette,
)
ax.set_yticks(np.arange(0, 0.55, 0.1))
ax.set_yticklabels([f"{i:.0%}" for i in np.arange(0, 0.55, 0.1)])
ax.set_ylabel("Explained variance (%)")
ax.set_xlabel("Cluster")
sns.despine(fig, trim=True, bottom=True, left=True)
# add horizontal grid line
ax.grid(axis="y", linestyle="--", alpha=0.5)
# make length of ticks shorter
ax.tick_params(axis="both", length=0)
plt.savefig("figs/figure_supp_cluster_expvar.png", dpi=300, bbox_inches="tight")

# %%
