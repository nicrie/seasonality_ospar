# %%
import os

import datatree as dt
import xarray as xr
import xeofs as xe
from tqdm import tqdm


def compute_cluster_membership_confidence(components):
    """Compute the confidence of cluster membership based on principal components.

    The confidence is in the interval [0, 1].

    """
    fraction_above_0 = (components > 0).mean("n")
    # renomalize to [-1, 1]
    confidence = (fraction_above_0 - 0.5) / 0.5
    # ensure confidence is in [0, 1]
    return confidence.where(confidence > 0, 0)


def compute_cluster_effect_size(components):
    """Compute the effect size for a given cluster based on principal components.

    The effect size is in the interval [0, inf] (latent PCA space).

    """
    mean_comps = components.mean("n")
    return mean_comps.where(mean_comps > 0, 0)


# %%
# Load data
# =============================================================================
# VARIABLE can be one of the following:
# - "absolute/Plastic"
# - "fraction/LAND"
# - "fraction/FISH"
# - "fraction/AQUA"
VARIABLE = "fraction/AQUA"
SEASONS = ["DJF", "MAM", "JJA", "SON"]
YEAR = 2001

base_path = f"data/gpr/{VARIABLE}/{YEAR}/"
pca_path = f"data/clustering/pca/{VARIABLE}/{YEAR}/"

if not os.path.exists(pca_path):
    os.makedirs(pca_path)
# OSPAR data
# -----------------------------------------------------------------------------
ospar = dt.open_datatree("data/beach_litter/ospar/preprocessed.zarr", engine="zarr")
litter_o = ospar["preprocessed/" + VARIABLE]
litter_o = litter_o.sel(year=slice(YEAR, 2020)).dropna("beach_id", **{"how": "all"})

# GPR data
# -----------------------------------------------------------------------------
model = dt.open_datatree(base_path + "posterior_predictive.zarr", engine="zarr")
litter_m = model["posterior_predictive"][VARIABLE.split("/")[1]]

# Statistical evaluation
# -----------------------------------------------------------------------------
results = xr.open_dataset(base_path + "effect_size_seasons.nc")

# Pre-processing the data for PCA
# -----------------------------------------------------------------------------
# We want to weigh more the beaches with high confidence in seasonality
ref_effect_size = abs(results.effect_size_gp).max("combination")
confidence = results.max_hdi
weights = confidence.max("combination")

# %%
# Weighted PCA
# =============================================================================
n_modes = 4
scores = []
comps = []
expvar_ratio = []
clus_dat = litter_m / litter_m.sum("season")
clus_dat = clus_dat.isel(n=slice(0, 4000, 1))

# clus_dat = clus_dat.drop_sel(beach_id=beaches_skagerrak)
for n in tqdm(clus_dat.n.values):
    data = clus_dat.sel(n=n)
    pca = xe.models.EOF(n_modes=n_modes, standardize=False)
    pca.fit(data, dim="season", weights=weights)
    scores.append(pca.scores())
    comps.append(pca.components())
    expvar_ratio.append(pca.explained_variance_ratio())
scores = xr.concat(scores, dim="n")
comps = xr.concat(comps, dim="n")
expvar_ratio = xr.concat(expvar_ratio, dim="n")
scores = scores.assign_coords(n=range(clus_dat.n.size))
comps = comps.assign_coords(
    n=range(clus_dat.n.size),
    lon=("beach_id", clus_dat.lon.values),
    lat=("beach_id", clus_dat.lat.values),
)
expvar_ratio = expvar_ratio.assign_coords(n=range(clus_dat.n.size))

scores.attrs["solver_kwargs"] = ""
comps.attrs["solver_kwargs"] = ""
expvar_ratio.attrs["solver_kwargs"] = ""

# %%
# Separating positive and negative components
# -----------------------------------------------------------------------------
weights_positive = (comps.where(comps > 0, 0) ** 2).sum("beach_id")
weights_negative = (comps.where(comps < 0, 0) ** 2).sum("beach_id")

expvar_ratio_pos = weights_positive.mean("n") * expvar_ratio
expvar_ratio_neg = weights_negative.mean("n") * expvar_ratio

pca_result = xr.Dataset(
    {
        "scores": scores,
        "comps": comps,
        "expvar_ratio": expvar_ratio,
        "expvar_ratio_pos": expvar_ratio_pos,
        "expvar_ratio_neg": expvar_ratio_neg,
        "confidence_pos": compute_cluster_membership_confidence(comps),
        "confidence_neg": compute_cluster_membership_confidence(-comps),
        "effect_size_pos": compute_cluster_effect_size(comps),
        "effect_size_neg": compute_cluster_effect_size(-comps),
    },
)

# %%
# Store the results
# =============================================================================

pca_result.to_netcdf(pca_path + "pca_clustering.nc", engine="netcdf4")

# %%
