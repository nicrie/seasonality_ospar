# %%
from itertools import combinations

import arviz as az
import datatree as dt
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import scipy.stats as st
import seaborn as sns
import statsmodels.stats.multitest as ssm
import xarray as xr
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from utils.styles import get_cyclic_palette


def _np_find_touching_hdi_size(samples1, samples2, initial_mass=0.95, tol=1e-5):
    """
    Find the HDI size for which the boundaries of HDIs of two distributions touch.
    """

    def overlap_loss(hdi_mass, dim="n"):
        hdi1 = pm.hdi(samples1, hdi_prob=hdi_mass)
        hdi2 = pm.hdi(samples2, hdi_prob=hdi_mass)

        # Loss function based on the distance between HDI boundaries
        if hdi2[1] > hdi1[1]:
            return np.abs((hdi1[1] - hdi2[0]))
        else:
            return np.abs((hdi2[1] - hdi1[0]))

    result = minimize_scalar(
        overlap_loss, bounds=(0, 1), method="bounded", options={"xatol": tol}
    )
    if result.success:
        return result.x  # Returns the HDI mass for which the boundaries touch
    else:
        raise ValueError("Optimization did not converge")


def find_touching_hdi_size(da1, da2, dim):
    return xr.apply_ufunc(
        _np_find_touching_hdi_size,
        da1,
        da2,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[]],
        vectorize=True,
    )


def check_interval_disjoint(A, B):
    # Check for overlap: A's lower <= B's upper AND B's lower <= A's upper
    is_disjoint = ~((A[:, 0] <= B[:, 1]) & (B[:, 0] <= A[:, 1]))
    return is_disjoint


def _np_get_percentage_overlap(A, B):
    assert A.shape == B.shape
    A = np.sort(A)
    B = np.sort(B)
    for i in range(A.size):
        overlaps = A[-(i + 1)] > B[i]
        if not overlaps:
            break
    return i / A.size


def get_percentage_overlap(A, B):
    return xr.apply_ufunc(
        _np_get_percentage_overlap,
        A,
        B,
        input_core_dims=[["n"], ["n"]],
        output_core_dims=[[]],
        vectorize=True,
    )


def _np_bh_correction(arr, alpha=0.05):
    # Perform Benjamini-Hochberg correction
    pvals = np.zeros_like(arr) * np.nan
    if np.isnan(arr).all():
        return pvals
    mask = ~np.isnan(arr)
    arr = arr[mask]
    pvals[mask] = ssm.multipletests(arr, alpha=alpha, method="fdr_bh")[1]
    return pvals


def benjamini_hochberg_correction(arr, dim, alpha=0.05):
    return xr.apply_ufunc(
        _np_bh_correction,
        arr,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        kwargs={"alpha": alpha},
    )


def _np_kruskal_wallis(X, min_samples=5):
    """Perform Kruskal-Wallis test on each group in X.

    Parameters
    ----------
    X : np.ndarray
        Array with shape (n_groups, n_samples_per_group)
    min_samples : int, optional
        Minimum number of samples required to perform the test, by default 5

    Returns
    -------
    float
        p-value of the Kruskal-Wallis test

    """
    n_samples_per_group = (~np.isnan(X)).sum(axis=1)
    has_enough_samples = n_samples_per_group >= min_samples
    # set columns with insufficient samples to NaN
    X = np.where(has_enough_samples, X.T, np.nan).T

    is_valid_group = (~np.isnan(X)).any(axis=1)
    n_valid_groups = is_valid_group.sum()

    if n_valid_groups < 2:
        return np.nan
    else:
        try:
            statistic, pval = st.kruskal(*X[is_valid_group], nan_policy="omit")
            return pval
        except ValueError:
            # If all values are the same, the test cannot be performed
            # Set p-value to 1
            return 1.0


def kruskal_wallis_test(da, min_samples=5):
    return xr.apply_ufunc(
        _np_kruskal_wallis,
        da,
        input_core_dims=[["season", "year"]],
        output_core_dims=[[]],
        exclude_dims=set(["season", "year"]),
        vectorize=True,
        kwargs={"min_samples": min_samples},
    )


def _np_mann_whitney_test(x, y):
    if np.isnan(x).all() or np.isnan(y).all():
        return np.nan, np.nan
    n1 = (~np.isnan(x)).sum()
    n2 = (~np.isnan(y)).sum()
    U1, pvalue = st.mannwhitneyu(x, y, nan_policy="omit")
    # Compute effect size (rank-biserial correlation coefficient)
    r = 2 * U1 / (n1 * n2) - 1
    return r, pvalue


def mann_whitney_test(da1, da2, dim):
    """Perform Mann-Whitney U test for each combination of beaches.

    Parameters
    ----------
    da1 : xr.DataArray
        Array with shape (n_beaches, n_years)
    da2 : xr.DataArray
        Array with shape (n_beaches, n_years)
    dim : str
        Dimension along which to perform the test

    Returns
    -------
    xr.DataArray
        Effect size of the Mann-Whitney U test
    xr.DataArray
        p-values of the Mann-Whitney U test


    """
    res = xr.apply_ufunc(
        _np_mann_whitney_test,
        da1,
        da2,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], []],
        vectorize=True,
    )
    return res


# %%

COLORS = get_cyclic_palette(as_cmap=False, n_colors=4)
SEASONS = ["DJF", "MAM", "JJA", "SON"]
VARIABLE = "Plastic"
YEAR = 2001


base_path = f"data/gpr/{VARIABLE}/{YEAR}/"
fig_path = f"figs/gpr/evaluation/{VARIABLE}/{YEAR}/"
ospar = dt.open_datatree("data/ospar/preprocessed.zarr", engine="zarr")
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