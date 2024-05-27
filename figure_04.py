# %%
import datatree as dt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from matplotlib.gridspec import GridSpec

import utils.styles
from utils.statistics import weighted_percentile
from utils.styles import get_cyclic_palette

utils.styles.set_theme()

# %%
# Load data
# =============================================================================
COLORS = get_cyclic_palette(as_cmap=False, n_colors=4)
SEASONS = ["DJF", "MAM", "JJA", "SON"]
VARIABLE = "Plastic"
YEAR = 2001

base_path = f"data/gpr/{VARIABLE}/{YEAR}/"

# OSPAR data
# -----------------------------------------------------------------------------
ospar = dt.open_datatree("data/beach_litter/ospar/preprocessed.zarr", engine="zarr")
litter_o = ospar["preprocessed"].to_dataset()
litter_o = litter_o[VARIABLE]
litter_o = litter_o.sel(year=slice(YEAR, 2020)).dropna("beach_id", **{"how": "all"})

# GPR data
# -----------------------------------------------------------------------------
model = dt.open_datatree(base_path + "posterior_predictive.zarr", engine="zarr")
litter_m = model["posterior_predictive"][VARIABLE]

# Statistical evaluation
# -----------------------------------------------------------------------------
results = xr.open_dataset(base_path + "effect_size_seasons.nc")

# Clusters of seasonality (PCA-based)
pca_result = xr.open_dataset("data/pca/pca_beaches.nc", engine="netcdf4")
expvar_pos = pca_result.expvar_ratio_pos
components = pca_result.comps
pcs = pca_result.scores
p_above_zero = (components > 0).mean("n")
p_above_zero_cut = p_above_zero.where(p_above_zero > 0.5, 0.5)


# Transform probability to weights (-1...0...+1)
cluster_weights = 2 * (p_above_zero - 0.5)
# Extract weights for positive clusters
cluster_weights = cluster_weights.clip(0, 1)


# Effect size
s = components.mean("n")
cluster_weights = abs(s).where(s >= 0, 0)

# Percentage of components above zero
c = (components > 0).mean("n")


# %%
# Compute cluster centroids
# -----------------------------------------------------------------------------
quantiles = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75]

# ... using the OSPAR data
cluster1_centroid, ess1 = weighted_percentile(
    litter_o, cluster_weights.sel(mode=1), quantiles, dim=["beach_id", "year"]
)
cluster1_centroid_annual, ess1_annual = weighted_percentile(
    litter_o, cluster_weights.sel(mode=1), quantiles, dim=["beach_id", "year", "season"]
)

cluster2_centroid, ess2 = weighted_percentile(
    litter_o, cluster_weights.sel(mode=2), quantiles, dim=["beach_id", "year"]
)
cluster2_centroid_annual, ess2_annual = weighted_percentile(
    litter_o, cluster_weights.sel(mode=2), quantiles, dim=["beach_id", "year", "season"]
)

# ... using the GPR data
cluster1_centroid_m, ess1_m = weighted_percentile(
    litter_m, cluster_weights.sel(mode=1), quantiles, dim=["beach_id", "n"]
)
cluster1_centroid_annual_m, ess1_annual_m = weighted_percentile(
    litter_m, cluster_weights.sel(mode=1), quantiles, dim=["beach_id", "n", "season"]
)

cluster2_centroid_m, ess2_m = weighted_percentile(
    litter_m, cluster_weights.sel(mode=2), quantiles, dim=["beach_id", "n"]
)
cluster2_centroid_annual_m, ess2_annual_m = weighted_percentile(
    litter_m, cluster_weights.sel(mode=2), quantiles, dim=["beach_id", "n", "season"]
)


# Merge seasonal and annual data
# -----------------------------------------------------------------------------
# OSPAR data
cluster1_centroid_annual = cluster1_centroid_annual.expand_dims({"season": ["annual"]})
cluster2_centroid_annual = cluster2_centroid_annual.expand_dims({"season": ["annual"]})

ess1_annual = ess1_annual.expand_dims({"season": ["annual"]})
ess2_annual = ess2_annual.expand_dims({"season": ["annual"]})

cluster1_centroid = xr.concat(
    [cluster1_centroid, cluster1_centroid_annual], dim="season"
)
ess1 = xr.concat([ess1, ess1_annual], dim="season")
cluster2_centroid = xr.concat(
    [cluster2_centroid, cluster2_centroid_annual], dim="season"
)
ess2 = xr.concat([ess2, ess2_annual], dim="season")

# Model data
cluster1_centroid_annual_m = cluster1_centroid_annual_m.expand_dims(
    {"season": ["annual"]}
)
cluster2_centroid_annual_m = cluster2_centroid_annual_m.expand_dims(
    {"season": ["annual"]}
)
ess1_annual_m = ess1_annual_m.expand_dims({"season": ["annual"]})
ess2_annual_m = ess2_annual_m.expand_dims({"season": ["annual"]})

cluster1_centroid_m = xr.concat(
    [cluster1_centroid_m, cluster1_centroid_annual_m], dim="season"
)
cluster2_centroid_m = xr.concat(
    [cluster2_centroid_m, cluster2_centroid_annual_m], dim="season"
)
ess1_m = xr.concat([ess1_m, ess1_annual_m], dim="season")
ess2_m = xr.concat([ess2_m, ess2_annual_m], dim="season")


# %%
my_grey = ".3"
grey_style = {
    "axes.edgecolor": my_grey,
    "axes.labelcolor": my_grey,
    "text.color": my_grey,
    "xtick.color": my_grey,
    "ytick.color": my_grey,
    "grid.color": my_grey,
    "axes.titlecolor": my_grey,
}
cmap = sns.color_palette("inferno", as_cmap=True)
cmap_clrs = sns.color_palette("inferno", as_cmap=False, n_colors=4)
clr_highlight1 = cmap_clrs[2]
clr_highlight2 = cmap_clrs[3]


def create_cluster_plot(percentiles, perc_ref, ax, color="C0", lw=1):
    """Create a cluster plot with weighted percentiles.

    Args:
        percentiles (pd.DataFrame): DataFrame with percentiles
        ess (pd.DataFrame): DataFrame with effective sample size
        ax (matplotlib.axes.Axes): Axes to plot
        color (str, optional): Color of the shaded area. Defaults to "C0".
        lw (int, optional): Line width. Defaults to 1.
    """

    coords_quantile = percentiles.coords["quantile"]
    quantiles = coords_quantile.values
    combs = []
    for t1, t2 in zip(quantiles, quantiles[::-1]):
        if t1 < t2:
            combs.append((t1, t2))

    f = (1 - np.abs(np.array(quantiles) - 0.5)) ** 7
    factors = dict(zip(quantiles, f))

    xticks = np.arange(4)
    xx = np.vstack([xticks - 0.25, xticks + 0.25])
    # Add reference line (sample median)
    y_ref = perc_ref.sel(quantile=0.5, season=SEASONS).values
    yy = np.vstack([y_ref, y_ref])
    ax.plot(xx, yy, c="C0", lw=lw * 0.5, zorder=15, ls="-")

    # Add data for model percentiles
    for q in quantiles:
        y = percentiles.sel(quantile=q, season=SEASONS).values
        yy = np.vstack([y, y])
        ax.plot(xx, yy, c="w", lw=lw * factors[q], zorder=10)
        # Add number for 0.50 quantile
        if q == 0.50:
            for i, season in enumerate(SEASONS):
                ax.text(
                    i + 0.3,
                    percentiles.loc[season, q],
                    "{:.0f}".format(percentiles.loc[season, q]),
                    ha="left",
                    va="center",
                    color=color,
                    size=7,
                    zorder=100,
                    bbox=dict(facecolor="w", edgecolor="w", alpha=0.8, pad=0.5),
                )

    for i, season in enumerate(SEASONS):
        for j, cols in enumerate(combs):
            hdi = 100 * (cols[1] - cols[0])
            ax.fill_between(
                xx[:, i],
                percentiles.loc[season, cols[0]],
                percentiles.loc[season, cols[1]],
                color=color,
                alpha=factors[cols[0]],
                zorder=5,
                label="{:.0f} %".format(hdi),
            )


# plot with seaborn paper context; use with context manage
with plt.style.context(grey_style):
    with sns.plotting_context("paper"):
        fig = plt.figure(figsize=(7.2, 4))
        gs = GridSpec(1, 2, figure=fig, wspace=0.1)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])

        create_cluster_plot(
            cluster1_centroid_m, cluster1_centroid, ax1, clr_highlight1, lw=2
        )
        create_cluster_plot(
            cluster2_centroid_m, cluster2_centroid, ax2, clr_highlight1, lw=2
        )

        cluster1_centroid.sel(quantile=0.5)

        xticks = np.arange(5)

        # Add effective sample size below the x-axis
        ax1.text(
            -1,
            -200,
            "ESS",
            ha="center",
            va="center",
            color=".5",
            size=8,
            zorder=100,
        )
        for a, ess in zip([ax1, ax2], [ess1, ess2]):
            for x, s in zip(xticks, SEASONS + ["annual"]):
                a.text(
                    x,
                    -200,
                    "({:.0f})".format(ess.sel(season=s).item()),
                    ha="center",
                    va="center",
                    color=".5",
                    style="italic",
                    size=7,
                    zorder=100,
                )
        ax1.set_ylabel("Litter density [#/100m]")

        for ax in [ax1, ax2]:
            ax.set_xlim(-0.5, 5)
            ax.set_xticks(xticks[:4])
            ax.set_xticklabels(SEASONS, weight=700)
            ax.set_yticks([0, 500, 1000, 1500, 2000])
            ax.set_ylim(0, 2000)
            # show y grid lines
            ax.xaxis.set_tick_params(which="both", length=0)
            ax.yaxis.set_tick_params(which="both", length=0)
            ax.grid(axis="y", lw=0.5, color=".5", alpha=1, linestyle=":")
            ax.fill_between([-1, 5], 0, 20, color=clr_highlight2, zorder=5, alpha=0.7)

        c1_annual_median = cluster1_centroid_m.sel(quantile=0.50, season="annual")
        c2_annual_median = cluster2_centroid_m.sel(quantile=0.50, season="annual")
        ax1.axhline(
            c1_annual_median,
            color=".3",
            lw=0.8,
            ls=":",
            zorder=1,
        )
        ax2.axhline(
            c2_annual_median,
            color=".3",
            lw=0.8,
            ls=":",
            zorder=1,
        )

        ax1.text(
            5,
            c1_annual_median + 50,
            "{:.0f} (Annual)".format(c1_annual_median),
            ha="right",
            color=".3",
            size=8,
            zorder=100,
        )
        ax2.text(
            5,
            c2_annual_median + 50,
            "{:.0f} (Annual)".format(c2_annual_median),
            ha="right",
            color=".3",
            size=8,
            zorder=100,
        )
        ax2.text(
            5, 70, "20 (EU TV)", ha="right", color=clr_highlight2, size=8, weight=800
        )

        ax1.set_title(
            "A | Cluster $C1^+$",
            loc="left",
        )
        ax2.set_title(
            "B | Cluster $C2^+$",
            loc="left",
        )
        hdls, lbds = ax2.get_legend_handles_labels()
        ax2.legend(
            hdls[:3][::-1],
            lbds[:3][::-1],
            title="HDI",
            loc="upper right",
            frameon=False,
        )
        sns.despine(fig, trim=True, offset=5, bottom=True, left=True)
        ax2.set_yticklabels([])
        plt.savefig("figs/figure04.png", dpi=500, bbox_inches="tight")
        plt.show()

# %%
