# %%
import os

import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from matplotlib.gridspec import GridSpec

import utils.styles
from utils.styles import get_cyclic_palette

utils.styles.set_theme()

# %%
COLORS = get_cyclic_palette(as_cmap=False, n_colors=4)
SEASONS = ["DJF", "MAM", "JJA", "SON"]
VARIABLE = "absolute/Plastic"
YEAR = 2001

base_path = f"data/clustering/pca/{VARIABLE}/{YEAR}/"
fig_path = f"figs/pca/{VARIABLE}/{YEAR}/"

os.makedirs(fig_path, exist_ok=True)
pca_result = xr.open_dataset(base_path + "pca_clustering.nc", engine="netcdf4")
# %%
expvar_pos = pca_result["expvar_ratio_pos"]
expvar_neg = pca_result["expvar_ratio_neg"]

expvar_pos = expvar_pos.assign_coords(
    mode=[str(m) + "+" for m in expvar_pos.mode.values]
)
expvar_neg = expvar_neg.assign_coords(
    mode=[str(m) + "-" for m in expvar_neg.mode.values]
)

expvar = xr.concat([expvar_pos, expvar_neg], dim="mode")

idx_sorted = expvar.median("n").to_series().sort_values(ascending=False).index
expvar = expvar.sel(mode=idx_sorted).isel(mode=slice(None, 5))
expvar.name = "explained_variance"

q025 = expvar.quantile(0.025, "n")
q975 = expvar.quantile(0.975, "n")
n_significant = (q025 > q975.shift({"mode": -1})).sum("mode").item()
palette_colors = [".8"] * 5
for i in range(n_significant):
    palette_colors[i] = "r"
palette = sns.color_palette(palette_colors, desat=0.8)
# %%
fig = plt.figure(figsize=(7.2, 5))
gs = GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[0, 0])
sns.violinplot(
    data=expvar.to_dataframe().reset_index(),
    x="mode",
    y="explained_variance",
    hue="mode",
    legend=False,
    ax=ax,
    density_norm="width",
    bw_adjust=1,
    cut=1,
    linewidth=1,
    palette=palette,
)
ax.set_ylabel("Explained variance (%)")
ax.set_xlabel("Cluster")
ax.set_ylim(-0.01, 0.5)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_yticklabels([0, 10, 20, 30, 40, 50])
ax.grid(axis="y", linestyle="--", linewidth=1)
# add horizontal grid line
ax.grid(axis="y", linestyle="--", alpha=0.5)
# make length of ticks shorter
ax.tick_params(axis="both", length=0)
sns.despine(fig, trim=True, bottom=True, left=True)
fig.savefig(fig_path + "explained_variance.pdf", dpi=300)


# %%
