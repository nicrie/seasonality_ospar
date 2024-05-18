# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from datatree import open_datatree

import utils.styles
from utils.styles import get_cyclic_palette

utils.styles.set_theme()

# %%
ospar = open_datatree("data/ospar/preprocessed.zarr", engine="zarr")
ospar = ospar["preprocessed"].to_dataset()
plastics = ospar["Plastic"]
# %%
SEASONS = ["DJF", "MAM", "JJA", "SON"]
COLORS = get_cyclic_palette()
mu = plastics.mean("year")
var = plastics.var("year", ddof=1)
n_surveys = plastics.notnull().sum("year").fillna(0)

ds = xr.Dataset(
    {
        "mu": mu,
        "var": var,
        "n_surveys": n_surveys,
    }
)
ds = ds.where(ds.n_surveys >= 4, drop=True)

# %%
# Figure Overdisperion in Beach Litter
# =============================================================================
df = ds.to_dataframe().reset_index()
df.rename(columns={"season": "Season", "n_surveys": "Number of surveys"}, inplace=True)
# use seaborn darkgrid


def compute_var(mu, phi):
    return mu * (1 + mu / phi)


def compute_mu(var, phi):
    return -phi / 2 + np.sqrt(phi**2 / 4 + phi * var)


plt.figure(figsize=(7.2, 7.2), dpi=500)
plt.fill_between(
    [1e0, 1e10],
    [1e10, 1e10],
    [1e0, 1e10],
    color=".9",
    zorder=0,
)
plt.text(0.05, 0.95, "Overdispersion", transform=plt.gca().transAxes)
sns.scatterplot(
    data=df,
    x="mu",
    y="var",
    hue="Season",
    legend="brief",
    palette=COLORS,
    size="Number of surveys",
    alpha=0.75,
    zorder=50,
)
plt.xscale("log")
plt.yscale("log")
# plt.grid()
plt.xlim(1e0, 1e10)
plt.ylim(1e0, 1e10)
plt.xlabel("Sample mean $\mu$")
plt.ylabel("Unbiased sample variance $\sigma^2$")
plt.plot([1e0, 1e10], [1e0, 1e10], ls="--", color=".3", lw=0.5)
sns.despine()
# add text "Poisson distribution" along the diagonal
plt.text(
    1e8,
    6e7,
    "Theoretical Poisson distribution",
    fontsize=8,
    color=".3",
    rotation=45,
    va="center",
    ha="center",
)
x = np.logspace(0, 10)
phis = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
for phi in phis:
    plt.plot(x, compute_var(x, phi), ls="--", color=".3", lw=0.5)
    plt.text(
        compute_mu(var=1e8, phi=phi),
        2e8,
        f"{phi:.0E}",
        fontsize=6,
        rotation=60,
        color=".3",
        va="center",
        ha="center",
    )


plt.legend(loc="lower right", frameon=False)
# set the upper left triangle to be light grey

plt.savefig("figs/fig_beach_litter_overdispersion.png", dpi=300, bbox_inches="tight")

# %%
