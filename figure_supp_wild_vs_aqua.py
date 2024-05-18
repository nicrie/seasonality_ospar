# %%
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

import utils.styles

utils.styles.set_theme()

fish = xr.open_dataarray("data/fao_fisheries_aquaculture/fish_production_by_source.nc")
fish = fish.sel(year=slice(1950, 2020))
total = fish.sum("species").sum("source")
aqua = fish.sel(source="aqua").sum("species")
wild = fish.sel(source="wild").sum("species")

# %%


fig = plt.figure(figsize=(7.2, 3.5), dpi=300)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.fill_between(
    total.year, total / 1e6, aqua / 1e6, color=".5", label="Wild", alpha=0.7
)
ax1.fill_between(total.year, aqua / 1e6, color="C1", label="Aquaculture", alpha=0.7)

# relative
ax2.fill_between(
    total.year, 100 * total / total, 100 * aqua / total, color=".5", alpha=0.7
)
ax2.fill_between(total.year, 100 * aqua / total, color="C1", alpha=0.7)

# add horizontal lines at 2, 4, 6, 8, 10 million tonnes
for y in range(2, 11, 2):
    ax1.axhline(y, color=".3", lw=0.3)
    ax2.axhline(y * 10, color=".3", lw=0.3)

# shorten the length of y label ticks to zero
ax1.tick_params(axis="y", length=0)
ax2.tick_params(axis="y", length=0)

# add the value of aquaculture in the last year as a marker with number, use annotate
ax1.annotate(
    f"{aqua[-1].values / 1e6:.1f} Mt in 2020",
    (2020, aqua[-1] / 1e6),
    (1980, 3),
    # make a curved arrow
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color="k"),
    fontsize=8,
    fontweight="bold",
    color="k",
    ha="center",
)
ax2.annotate(
    f"{100 * aqua[-1] / total[-1].values:.1f}% in 2020",
    (2020, 100 * aqua[-1] / total[-1]),
    (1980, 30),
    # make a curved arrow
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color="k"),
    fontsize=8,
    fontweight="bold",
    color="k",
    ha="center",
)


# for both ax1 and ax2 place the label "Wild capture" and "Aquaculture" at the right end of the plot in white
ax1.text(
    2018,
    5,
    "Wild capture",
    color="w",
    ha="right",
    va="center",
    fontsize=8,
    fontweight="bold",
)
ax1.text(
    2018,
    0.6,
    "Aquaculture",
    color="k",
    ha="right",
    va="center",
    fontsize=8,
    fontweight="bold",
)
ax2.text(
    2018,
    65,
    "Wild capture",
    color="w",
    ha="right",
    va="center",
    fontsize=8,
    fontweight="bold",
)
ax2.text(
    2018,
    8,
    "Aquaculture",
    color="k",
    ha="right",
    va="center",
    fontsize=8,
    fontweight="bold",
)
ax1.set_title("A | Total production (million tonnes)", loc="left")
ax2.set_title("B | Relative production (%) ", loc="left")
ax1.set_ylabel("")
ax1.set_xlabel("")
ax1.set_ylim(0, 10)
ax2.set_ylim(0, 100)
ax1.set_xlim(1950, 2020)
ax2.set_xlim(1950, 2020)
fig.suptitle("Wild vs aquaculture fish production in the north-east Atlantic", y=0.98)
# ax1.legend(loc="upper left", ncols=2, frameon=False)
sns.despine(fig, trim=True, left=True)
plt.tight_layout()
plt.savefig("figs/figure_supp_wild_vs_aqua.png")


# %%
