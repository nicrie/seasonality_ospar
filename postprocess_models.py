# %%
from itertools import combinations

import arviz as az
import datatree as dt
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import seaborn as sns
import xarray as xr
from tqdm import tqdm

from utils.statistics import (
    benjamini_hochberg_correction,
    find_touching_hdi_size,
    kruskal_wallis_test,
    mann_whitney_test,
)
from utils.styles import get_cyclic_palette

# %%

COLORS = get_cyclic_palette(as_cmap=False, n_colors=4)
SEASONS = ["DJF", "MAM", "JJA", "SON"]
VARIABLE = "Plastic"
YEAR = 2001


base_path = f"data/gpr/{VARIABLE}/{YEAR}/"
fig_path = f"figs/gpr/evaluation/{VARIABLE}/{YEAR}/"
ospar = dt.open_datatree("data/beach_litter/ospar/preprocessed.zarr", engine="zarr")
litter_o = ospar["preprocessed"].to_dataset()
litter_o = litter_o[VARIABLE]
litter_o = litter_o.sel(year=slice(YEAR, 2020)).dropna("beach_id", **{"how": "all"})

model = dt.open_datatree(base_path + "posterior_predictive.zarr", engine="zarr")
litter_m = model["posterior_predictive"][VARIABLE]

idata = {}
for s in SEASONS:
    idata[s] = az.from_netcdf(base_path + f"idata_{s}.nc")


combs = list(combinations(SEASONS, 2))

# %%
# Observed data (isolated approach)
# =============================================================================

# Kruskal-Wallis test
# -----------------------------------------------------------------------------
pvals_kruskal_wallis = kruskal_wallis_test(litter_o, min_samples=5)
is_sig_kruskal_wallis = pvals_kruskal_wallis < 0.05


# Mann-Whitney U test (pairwise)
# -----------------------------------------------------------------------------
pvals_mann_whitney = []
effect_size_mann_whitney = []
for s1, s2 in combs:
    # Perform Mann-Whitney U test for each combination of beaches
    A = litter_o.sel(season=s1)
    B = litter_o.sel(season=s2)
    # Effect size and p value
    r, pval = mann_whitney_test(A, B, dim="year")
    pvals_mann_whitney.append(pval)
    effect_size_mann_whitney.append(r)

pvals_mann_whitney = xr.concat(pvals_mann_whitney, dim="combination")
effect_size_mann_whitney = xr.concat(effect_size_mann_whitney, dim="combination")

pvals_mann_whitney = pvals_mann_whitney.assign_coords(
    coords={"combination": [f"{s1}<{s2}" for s1, s2 in combs]},
)
effect_size_mann_whitney = effect_size_mann_whitney.assign_coords(
    coords={"combination": [f"{s1}<{s2}" for s1, s2 in combs]},
)


# Benjamini-Hochberg correction
# -----------------------------------------------------------------------------
pvals_cr_mann_whitney = benjamini_hochberg_correction(
    pvals_mann_whitney, dim="combination"
)


# %%

alpha = 0.05
is_sig_mann_whitney = pvals_cr_mann_whitney < alpha
is_sig_kruskal_wallis = pvals_kruskal_wallis < alpha

nb_kw = (pvals_kruskal_wallis < alpha).sum().item()
nb_mw = (pvals_mann_whitney < alpha).any("combination").sum().item()
nb_mw_cr = (
    (pvals_cr_mann_whitney.where(is_sig_kruskal_wallis) < alpha)
    .any("combination")
    .sum()
    .item()
)

print("Number beaches (Kruskal Wallis): ", nb_kw)
print("Number beaches (Mann-Whitney): ", nb_mw)
print("Number beaches (Mann-Whitney, corrected): ", nb_mw_cr)

# %%
result = xr.Dataset(
    {
        "pvals_kruskal_wallis": pvals_kruskal_wallis,
        "pvals_mann_whitney": pvals_mann_whitney,
        "pvals_cr_mann_whitney": pvals_cr_mann_whitney,
        "effect_size_mann_whitney": effect_size_mann_whitney,
    }
)

# %%

mu = model["posterior_predictive"]["mu"]
mu_hdi = pm.hdi(mu, hdi_prob=0.94, **{"input_core_dims": [["n"]]})["mu"]

effect_size = []
max_hdi = []
median_diff = []
relative_diff = []
for s1, s2 in tqdm(combs):
    # effect size
    A = litter_m.sel(season=s1)
    B = litter_m.sel(season=s2)
    f = (A < B).mean("n")
    rank_biserial_coef = 2 * f - 1
    effect_size.append(rank_biserial_coef)

    # Median difference
    mdiff = (B - A).median("n")
    median_diff.append(mdiff)

    # Relative difference
    diff_is_positive = np.sign(mdiff) > 0
    denominator = A.median("n").where(diff_is_positive, B.median("n"))
    rdiff = mdiff / denominator
    relative_diff.append(rdiff)

    # effect size of mean
    A = mu.sel(season=s1)
    B = mu.sel(season=s2)
    hdi = find_touching_hdi_size(A, B, dim="n")
    max_hdi.append(hdi)


effect_size = np.array(effect_size)
effect_size = xr.DataArray(
    effect_size,
    dims=("combination", "beach_id"),
    coords={
        "combination": list(["<".join([s1, s2]) for s1, s2 in combs]),
        "beach_id": mu.beach_id,
        "lon": ("beach_id", mu.lon.values),
        "lat": ("beach_id", mu.lat.values),
    },
    name="effect_size",
    attrs={"long_name": "Effect size (rank-biserial coefficient)"},
)


max_hdi = np.array(max_hdi)
max_hdi = xr.DataArray(
    max_hdi,
    dims=("combination", "beach_id"),
    coords={
        "combination": list(["<".join([s1, s2]) for s1, s2 in combs]),
        "beach_id": mu.beach_id,
        "lon": ("beach_id", mu.lon.values),
        "lat": ("beach_id", mu.lat.values),
    },
    name="max_hdi",
    attrs={
        "long_name": "HDI size for which the boundaries of the HDIs of the mean touch"
    },
)

median_diff = np.array(median_diff)
median_diff = xr.DataArray(
    median_diff,
    dims=("combination", "beach_id"),
    coords={
        "combination": list(["<".join([s1, s2]) for s1, s2 in combs]),
        "beach_id": mu.beach_id,
        "lon": ("beach_id", mu.lon.values),
        "lat": ("beach_id", mu.lat.values),
    },
    name="median_diff",
    attrs={"long_name": "Median difference"},
)

relative_diff = np.array(relative_diff)
relative_diff = xr.DataArray(
    relative_diff,
    dims=("combination", "beach_id"),
    coords={
        "combination": list(["<".join([s1, s2]) for s1, s2 in combs]),
        "beach_id": mu.beach_id,
        "lon": ("beach_id", mu.lon.values),
        "lat": ("beach_id", mu.lat.values),
    },
    name="relative_diff",
    attrs={
        "long_name": "Relative difference",
        "description": "median(B - A) / median(X). If median(B - A) >= 0, then X = A, else X = B",
    },
)

result["effect_size_gp"] = effect_size
result["max_hdi"] = max_hdi
result["median_diff"] = median_diff
result["relative_diff"] = relative_diff

result = result.transpose("beach_id", "combination")

# %%
# Ensure that we don't make inference for beaches with only one season surveyed
# -----------------------------------------------------------------------------
# NOTE: when using the StudentT process, this doesn't seems necessary anymore

# has_measurement = litter_o.notnull().any("year")
# is_not_single_season_survey = has_measurement.sum("season") > 1
# result[["effect_size_gp", "max_hdi", "median_diff", "relative_diff"]] = result[
#     ["effect_size_gp", "max_hdi", "median_diff", "relative_diff"]
# ].where(is_not_single_season_survey)


# %%
# Summary results
# =============================================================================
n_beaches_sig = result.max_hdi.max("combination") > 0.95
n_beaches_sig = n_beaches_sig.sum("beach_id").item()
print("Number of beaches with significant effect size: ", n_beaches_sig)

# %%
# Store results
# -----------------------------------------------------------------------------
result.to_netcdf(base_path + "effect_size_seasons.nc")

# %%

bla = litter_o.sel(beach_id=slice("NO001", "NO020"))
plt.scatter(bla.lon, bla.lat)
bla.beach_id

# %%

beach = "NO004"
beach = "DE002"
de002_o = litter_o.sel(beach_id=beach)
de002 = litter_m.sel(beach_id=beach)
de002_mu = model.posterior_predictive["mu"].sel(beach_id=beach)
plt.figure(figsize=(7.2, 5))
sns.kdeplot(de002[0], label="DJF", color="C0", fill=True)
sns.kdeplot(de002[1], label="MAM", color="C1", fill=True)
sns.kdeplot(de002[2], label="JJA", color="C2", fill=True)
sns.kdeplot(de002[3], label="SON", color="C3", fill=True)
sns.kdeplot(de002_mu[0], color="C0", ls="--")
sns.kdeplot(de002_mu[1], color="C1", ls="--")
sns.kdeplot(de002_mu[2], color="C2", ls="--")
sns.kdeplot(de002_mu[3], color="C3", ls="--")
plt.legend()
plt.xlim(-5, 1000)
# plt.ylim(0, 0.05)
plt.scatter(x=de002_o.sel(season="DJF").values, y=[0.001] * 20, color="C0")
plt.scatter(x=de002_o.sel(season="MAM").values, y=[0.002] * 20, color="C1")
plt.scatter(x=de002_o.sel(season="JJA").values, y=[0.003] * 20, color="C2")
plt.scatter(x=de002_o.sel(season="SON").values, y=[0.004] * 20, color="C3")
plt.show()
print("Mean (observed): ", de002_o.mean("year").round(0).values)
print("Mean (model): ", de002.mean("n").round(0).values)
print("Median (observed): ", de002_o.median("year").round(0).values)
print("Median (model): ", de002.median("n").round(0).values)
print("Model mu: ", de002_mu.median("n").round(0).values)
# %%
