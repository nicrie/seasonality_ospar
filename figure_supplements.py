# %%
import datatree as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from cartopy.crs import PlateCarree, TransverseMercator
from cartopy.feature import LAND, OCEAN, RIVERS
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import utils.styles
from utils.styles import get_cyclic_palette

utils.styles.set_theme()
# %%
COLORS = get_cyclic_palette(as_cmap=False, n_colors=4)
SEASONS = ["DJF", "MAM", "JJA", "SON"]
VARIABLE = "Plastic"
YEAR = 2001


base_path = f"data/gpr/{VARIABLE}/{YEAR}/"
ospar = dt.open_datatree("data/ospar/preprocessed.zarr", engine="zarr")
litter_o = ospar["preprocessed"].to_dataset()
litter_o = litter_o[VARIABLE]
litter_o = litter_o.sel(year=slice(YEAR, 2020)).dropna("beach_id", **{"how": "all"})

model = dt.open_datatree(base_path + "posterior_predictive.zarr", engine="zarr")
litter_m = model["posterior_predictive"][VARIABLE]
litter_m
results = xr.open_dataset(base_path + "effect_size_seasons.nc")

# %%
median = litter_o.median(("year", "season"))
mad = abs(litter_o - median).median(("year", "season"))
n_surveys = litter_o.notnull().sum(("year", "season"))
median.name = "median"
mad.name = "median_absolute_deviation"
ospar = xr.Dataset({"median": median, "mad": mad})

# Effect size (biserial correlation)
es = abs(results["effect_size_gp"]).max("combination")
# Coefficient of variation

# %%
# Perform weighted linear regression
# -----------------------------------------------------------------------------
# cv = ospar["mad"] / ospar["median"]
cv = litter_o.std(("season", "year")) / litter_o.mean(("season", "year"))
X = es.values[:, None]
y = cv.values
weights = n_surveys.values - 1
weights = weights / weights.max()
lreg = LinearRegression(fit_intercept=True)
lreg.fit(X, y, sample_weight=weights)

lreg.coef_
r2 = r2_score(y, lreg.predict(X), sample_weight=weights)
r = np.sqrt(r2)
print(f"R2: {r2:.2f}")
lreg.intercept_

# %%

proj = TransverseMercator(central_longitude=0.0, central_latitude=50.0)
extent = [-15, 13, 34, 64]


fig = plt.figure(figsize=(7.2, 3.0))
gs = fig.add_gridspec(1, 3, hspace=0.0, wspace=0.01)
ax1 = fig.add_subplot(gs[:, 0], projection=proj)
ax2 = fig.add_subplot(gs[:, 1], projection=proj)
ax3 = fig.add_subplot(gs[:, 2])

for a in [ax1, ax2]:
    a.add_feature(OCEAN.with_scale("50m"), facecolor=".1")
    a.add_feature(LAND.with_scale("50m"), facecolor=".25")
    a.add_feature(RIVERS.with_scale("50m"), color=".4", lw=0.5)
    a.set_extent(extent, crs=PlateCarree())

ax1.scatter(
    ospar.lon,
    ospar.lat,
    ospar["median"] / 10,
    ec="k",
    lw=0.3,
    color="C0",
    alpha=0.7,
    transform=PlateCarree(),
)
ax2.scatter(
    ospar.lon,
    ospar.lat,
    ospar["mad"] / 10,
    ec="k",
    lw=0.3,
    color="C1",
    alpha=0.7,
    transform=PlateCarree(),
)

# Relationship between seasonal effect size and coefficient of variation
ax3.scatter(X[:, 0], y, c=weights, cmap="Greys", ec="k", lw=0.3)
ax3.plot(X, lreg.predict(X), color="C2", linewidth=3, label="Weighted model")

# Add R2 to left top
ax3.text(
    0.05,
    0.95,
    f"$R^2$ = {r2:.2f}\n$r$ = {r:.2f}",
    color="black",
    fontsize=8,
    ha="left",
    va="top",
    transform=ax3.transAxes,
)

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 3)
sns.despine(ax=ax3, left=True, right=False, trim=True, offset=5)
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position("right")
ax3.set_xlabel("Effect size [unitless]")
ax3.set_ylabel("Coefficient of variation [unitless]")
ax3.set_title("C | Seasonal effect size vs. litter CV", loc="left")


# Add legend
for a in [ax1, ax2]:
    a.add_patch(
        mpl.patches.Rectangle(
            (0.6, 0.0),  # bottom left corner coordinates
            0.4,  # width
            0.25,  # height
            facecolor="black",
            alpha=0.9,
            transform=a.transAxes,
        )
    )
    # Add legend circles (effect size)
    lax1 = a.inset_axes([0.65, 0.01, 0.3, 0.2], transform=a.transAxes)
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

    legend_effect_size = np.array([100, 1000, 4000])  # Effect_size
    circle_labels = [str(s) for s in legend_effect_size]
    circle_sizes = legend_effect_size / 10  # Circle sizes
    radii = np.sqrt(circle_sizes / np.pi) / 20
    xlocs = [-0.2, -0.2, -0.2]
    for x, r, s, lb in zip(xlocs, radii, circle_sizes, circle_labels):
        lax1.scatter(
            x,
            -0.3 + r,
            s=s,
            edgecolors="w",
            facecolors="none",
        )
        lax1.text(
            0.5,
            1.7 * r - 0.3,
            f"{lb}",
            color="white",
            fontsize=6,
            ha="left",
            va="center",
        )
    lax1.text(
        0,
        -0.8,
        "# items/100m",
        color="white",
        fontsize=7,
        ha="center",
    )


ax1.set_title("A | Median pollution", loc="left")
ax2.set_title("B | Median absolute deviation", loc="left")
fig.savefig("figs/figure_supp_ospar_median_mad.png", dpi=300, bbox_inches="tight")
