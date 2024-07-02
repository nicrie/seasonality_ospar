# %%
from string import ascii_uppercase

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from cartopy.crs import PlateCarree, TransverseMercator
from cartopy.feature import LAND, OCEAN, RIVERS
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

import utils.styles

utils.styles.set_theme()


def trans_effect_size(s):
    return s * 2e3


def trans_prob(c):
    return c


def load_data(path):
    try:
        pca_result = xr.open_dataset(path + "pca_clustering.nc", engine="netcdf4")
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None
    pcs = pca_result.scores
    confidence = pca_result.confidence_pos
    effect_size = pca_result.effect_size_pos

    exp_var_ratio = pca_result.expvar_ratio_pos.quantile([0.5, 0.025, 0.975], "n") * 100

    return xr.Dataset(
        {"s": effect_size, "c": confidence, "pcs": pcs, "quantiles": exp_var_ratio},
    )


# %%
# Load data
# =============================================================================
YEAR = 2001
QUANTITY = "fraction"
VARIABLE = ["LAND", "FISH", "AQUA"]
NAMES = ["Land", "Fish", "Aquaculture"]

paths = {v: f"data/clustering/pca/{QUANTITY}/{v}/{YEAR}/" for v in VARIABLE}
ds = {v: load_data(p) for v, p in paths.items()}


# %%
# Figure - PCA eigenvectors and projections
# =============================================================================
seasons = ds["LAND"].season
extent = [-15, 13, 34, 64]
proj = TransverseMercator(central_latitude=50)

max_probability = 0.25
norm = mcolors.Normalize(vmin=0.0, vmax=max_probability)
cmap = sns.color_palette("inferno", as_cmap=True)
cmap_clrs = sns.color_palette("inferno", as_cmap=False, n_colors=4)
clr_highlight = cmap_clrs[3]

palettes = {
    "LAND": [clr_highlight, ".5", clr_highlight, ".5"],
    "FISH": [".5", ".5", ".5", clr_highlight],
    "AQUA": [clr_highlight, clr_highlight, ".5", ".5"],
}

fig = plt.figure(figsize=(7.2, 3.1))
gs = GridSpec(
    1,
    4,
    figure=fig,
    hspace=0.02,
    wspace=0.0,
    width_ratios=[1, 1, 1, 0.05],
)
ax = {v: fig.add_subplot(gs[0, i], projection=proj) for i, v in enumerate(VARIABLE)}
cax = fig.add_subplot(gs[0, 3])

for i, (v, a) in enumerate(ax.items()):
    # Map background
    a.set_extent(extent, crs=PlateCarree())
    a.add_feature(OCEAN, facecolor=".1")
    a.add_feature(LAND, facecolor=".25")
    a.add_feature(RIVERS, color=".4", lw=0.5)

    # Spatial distribution of clusters
    a.scatter(
        ds[v]["s"].lon,
        ds[v]["s"].lat,
        s=trans_effect_size(ds[v]["s"].sel(mode=1)),
        c=ds[v]["c"].sel(mode=1).values,
        norm=norm,
        cmap=cmap,
        transform=PlateCarree(),
        ec="w",
        lw=0.5,
        alpha=0.5,
    )

    # Cluster centroid in lower right corners
    xticks = np.arange(0.0, 3.5)
    ax_pc = a.inset_axes(
        [0.6, 0.08, 0.35, 0.15], fc="r", frameon=False, transform=a.transAxes
    )
    df_pcs = ds[v]["pcs"].sel(mode=1, drop=True).to_dataframe().reset_index()
    sns.barplot(
        df_pcs,
        x="season",
        y="pcs",
        hue=df_pcs["season"],
        palette=palettes[v],
        zorder=1,
        ax=ax_pc,
        err_kws={"color": "w"},
    )
    ax_pc.set_title("Pattern", color="w", size=7)
    ax_pc.set_xticks(xticks)
    ax_pc.set_xticklabels(seasons, color="w", size=5)
    ax_pc.set_yticks([])
    ax_pc.set_xlabel("")
    ax_pc.set_ylabel("")
    ax_pc.tick_params(axis="x", length=0)
    # Add a rectangle to highlight the cluster centroid
    a.add_patch(
        Rectangle(
            xy=(0.55, 0),
            width=0.45,
            height=0.33,
            linewidth=1,
            edgecolor="k",
            facecolor="k",
            zorder=20,
            alpha=0.3,
            transform=a.transAxes,
        )
    )

    # Title
    a.text(
        0.01,
        0.99,
        "{:} | {:}".format(ascii_uppercase[i], NAMES[i]),
        transform=a.transAxes,
        fontsize=10,
        fontweight="bold",
        color="w",
        va="top",
    )

    # Explained variance
    ax_expvar = a.inset_axes(
        [0.01, 0.8, 0.4, 0.08], facecolor=".1", frameon=False, transform=a.transAxes
    )
    mid, lower, upper = ds[v]["quantiles"].sel(mode=1)
    sns.barplot(
        x=[mid.item()],
        y=["Explained Variance"],
        ax=ax_expvar,
        color=clr_highlight,
        edgecolor="black",
        linewidth=1,
    )
    ax_expvar.errorbar(
        x=mid,
        y=["Explained Variance"],
        xerr=[[mid - lower], [upper - mid]],
        fmt="none",
        color=".9",
        capsize=2,
    )

    ax_expvar.text(
        5,
        0,
        f"{mid.item():.1f}%",
        color="w",
        ha="left",
        va="center",
        weight="bold",
    )
    ax_expvar.text(
        0,
        0.5,
        "Explained variance",
        color="w",
        style="italic",
        size=8,
        transform=ax_expvar.transData,
    )

    ax_expvar.set_xlim([0, 50])
    ax_expvar.set_ylim([-0.5, 0.5])
    ax_expvar.set_xticks([])
    ax_expvar.set_yticks([])
    ax_expvar.set_xlabel("")
    ax_expvar.spines[["left", "right", "top", "bottom"]].set_visible(False)


# Add vertical colorbar to the right border
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=cax,
    label="Confidence in Cluster Membership",
)
cticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
cbar.set_ticks(cticks)
cbar.ax.set_yticklabels([f"{t:.0%}" for t in cticks])

plt.savefig("figs/figure_supp03.png", dpi=300, bbox_inches="tight")

# %%
