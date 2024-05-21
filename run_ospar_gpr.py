# %%
# Run on drago via
# sbatch -p generic -n 1 --cpus-per-task=4 --mem-per-cpu=20GB drago_gpr.sh
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import datatree as dt
import numpy as np
import numpyro
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from utils.kernel import Matern32Haversine
from utils.transformation import (
    estimate_lmbda,
    pt_yeojohnson_inv,
    yeojohnson,
    yeojohnson_inv,
)

numpyro.set_host_device_count(4)


# %%
print("Loading data...")

SEASON = ["DJF", "MAM", "JJA", "SON"]
RANDOM_SEED = 2803 + 2
VARIABLE = "Aquaculture"
YEAR = 2016


rng = np.random.default_rng(RANDOM_SEED)

ospar = dt.open_datatree("data/beach_litter/ospar/preprocessed.zarr", engine="zarr")
ospar = ospar["preprocessed"].to_dataset()
ospar = ospar.sel(year=slice(YEAR, 2020))
ospar = ospar[VARIABLE]
ospar = ospar.dropna("beach_id", **{"how": "all"})

# Transformation parameter for Yeo-Johnson transform
yeo_lmbda = estimate_lmbda(ospar.stack(x=[...]), dim="x").item()

# ospar = ospar.isel(beach_id=slice(None, None, 15))


# %%
# GP model
# =============================================================================
def fit_gp_model(season, fit=True):
    # Dimensions and coordinates
    beaches_all = ospar.beach_id.values
    beach_lonlats_all = np.vstack([ospar.lon, ospar.lat]).T

    data = ospar.sel(season=season).dropna("beach_id", **{"how": "all"})
    data = data.assign_coords(beach_idx=("beach_id", np.arange(data.beach_id.size)))
    data_stacked = data.stack(x=[...]).dropna("x")

    beach_lonlats = np.vstack([data.lon, data.lat]).T
    beach_idx = data_stacked.beach_idx
    beaches = data.beach_id.values

    coords = {
        "beach": beaches,
        "sample": beach_idx,
        "feature": ["lon", "lat"],
    }

    with pm.Model(coords=coords) as model:
        _X = pm.ConstantData("beach_X", beach_lonlats, dims=("beach", "feature"))
        _beach_idx = pm.ConstantData("beach_idx", beach_idx, dims="sample")
        y = pm.ConstantData("y", data_stacked.values, dims="sample")

        # Mean function
        mean_lower, mean_upper = ospar.mean("year").quantile([0.025, 0.975])
        mean_lower = mean_lower.item()
        mean_upper = mean_upper.item()
        params_mu_mu = pm.find_constrained_prior(
            distribution=pm.Normal,
            lower=yeojohnson(mean_lower, yeo_lmbda),
            upper=yeojohnson(mean_upper, yeo_lmbda),
            init_guess={"mu": 5, "sigma": 2},
        )
        # Covariance function
        params_rho1 = pm.find_constrained_prior(
            distribution=pm.InverseGamma,
            lower=10,
            upper=100,
            init_guess={"alpha": 3, "beta": 100},
        )
        params_rho2 = pm.find_constrained_prior(
            distribution=pm.InverseGamma,
            lower=100,
            upper=5000,
            init_guess={"alpha": 3, "beta": 2000},
        )
        params_eta = pm.find_constrained_prior(
            distribution=pm.Gamma,
            lower=0.5 / np.sqrt(2),
            upper=1.5 / np.sqrt(2),
            init_guess={"alpha": 10, "beta": 10},
        )

        # Mean function
        mu_mu = pm.Normal("mu_mu", **params_mu_mu)
        mean_func = pm.gp.mean.Constant(mu_mu)

        # Covariance function
        eta_1 = pm.Gamma("eta_1", **params_eta)
        eta_2 = pm.Gamma("eta_2", **params_eta)
        rho_1 = pm.InverseGamma("rho_1", **params_rho1)
        rho_2 = pm.InverseGamma("rho_2", **params_rho2)
        kernel_short = eta_1**2 * Matern32Haversine(2, ls=rho_1)
        kernel_long = eta_2**2 * Matern32Haversine(2, ls=rho_2)
        cov_func = kernel_short + kernel_long + pm.gp.cov.WhiteNoise(1e-4)

        # Latent GP for mean of NB distribution
        # try with alpha=6, otherwise prior to wild (?)
        # nu = pm.Gamma("nu", alpha=6, beta=1)
        # gp = pm.gp.TP(mean_func=mean_func, scale_func=cov_func, nu=nu)
        gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
        mu_trans = gp.prior("mu_trans", _X, dims="beach", jitter=1e-4)

        # Inverse transform the data
        # Theoretically, mu_yj_inv can be negavtive (min = -1), so we need to clip it to 0
        mu_neg = pm.Deterministic(
            "mu_neg", yeojohnson_inv(mu_trans[_beach_idx], yeo_lmbda), dims="sample"
        )
        mu = pm.Deterministic("mu", pt.where(mu_neg > 0, mu_neg, 0.01), dims="sample")
        # mu = pm.Deterministic("mu", mu_neg, dims="sample")

        # Phi parameter (~dispersion)
        params_phi = pm.find_constrained_prior(
            distribution=pm.InverseGamma,
            lower=2,
            upper=9,
            init_guess={"alpha": 2, "beta": 1},
        )
        phi = pm.InverseGamma("phi", **params_phi)

        observed = pm.NegativeBinomial(
            VARIABLE,
            mu=mu,
            alpha=phi,
            observed=y,
            dims="sample",
        )

    with model:
        idata = pm.sample(
            1000,
            tune=8000,
            init="jitter+adapt_diag_grad",
            random_seed=rng,
            chains=4,
            cores=4,
            target_accept=0.9,
            nuts_sampler="numpyro",
            # nuts_sampler_kwargs={"chain_method": "vectorized"},
        )

    # add the GP conditional to the model, given the new X values
    with model:
        model.add_coords({"beach_pred": beaches_all})
        mu_trans_pred = gp.conditional(
            "mu_trans_pred", beach_lonlats_all, jitter=1e-4, dims="beach_pred"
        )
        mu_neg_pred = pm.Deterministic(
            "mu_neg_pred",
            pt_yeojohnson_inv(mu_trans_pred, yeo_lmbda),
            dims="beach_pred",
        )
        mu_pred = pm.Deterministic(
            "mu_pred", pt.where(mu_neg_pred > 0, mu_neg_pred, 0.01), dims="beach_pred"
        )

        phi_pred = pm.Deterministic("phi_pred", phi)

        cond_observed = pm.NegativeBinomial(
            VARIABLE + "_pred",
            mu=mu_pred,
            alpha=phi_pred,
            dims=["beach_pred"],
        )

    with model:
        prior_p = pm.sample_prior_predictive(
            samples=2000,
            random_seed=rng,
            var_names=[
                "mu_mu",
                "eta_1",
                "eta_2",
                "rho_1",
                "rho_2",
                "phi",
                "mu_trans_pred",
            ],
        )
        idata.extend(prior_p)

    with model:
        # Sample from the GP conditional distributionç
        ppc = pm.sample_posterior_predictive(
            idata.posterior,
            var_names=["mu_trans_pred", "mu_pred", "phi_pred", VARIABLE + "_pred"],
            random_seed=rng,
        )
        idata.extend(ppc)

    return idata


# %%
# Run GP model for each season
# =============================================================================
print("Running GP model for each season")
print("================================\n")
idata = {}
for season in SEASON:
    print(f"{season}...")
    idata[season] = fit_gp_model(season, fit=True)

# %%
# Extract scaling parameter lambda of Yeo-Johnson transform
lmbda_yeo = xr.DataArray(yeo_lmbda, name="lambda")


# %%
# Posterior predictive
# =============================================================================
model_output = {}
for season in idata.keys():
    pp = idata[season].posterior_predictive
    pp = pp.stack(n=("chain", "draw"))
    pp = pp.drop_vars(["n", "chain", "draw"]).assign_coords({"n": range(pp.n.size)})
    pp = pp.rename(
        {
            "beach_pred": "beach_id",
            "mu_trans_pred": "mu_trans",
            "mu_pred": "mu",
            "phi_pred": "phi",
            VARIABLE + "_pred": VARIABLE,
        }
    )
    model_output[season] = pp

gpr_pp = xr.concat(list(model_output.values()), dim="season").assign_coords(
    season=list(model_output.keys())
)
gpr_pp = gpr_pp.assign_coords(
    {
        "lon": ("beach_id", ospar.lon.values),
        "lat": ("beach_id", ospar.lat.values),
    }
)

# Compute aggregated posterior predictive (uncertainty from mean and dispersion)
# -----------------------------------------------------------------------------
mu_trans = gpr_pp["mu_trans"]
mu = gpr_pp["mu"]
phi = gpr_pp["phi"]

# mu = yeojohnson_inv()
p = phi / (mu + phi)

rng = np.random.default_rng(42)

n_boots = 200
nb_n = phi.isel(n=slice(0, 500))
nb_p = p.isel(n=slice(0, 500))
nb_n = nb_n.broadcast_like(nb_p)

da_agg = rng.negative_binomial(
    nb_n.values.T[:, :, :, None],
    nb_p.values.T[:, :, :, None],
    (*nb_n.T.shape, n_boots),
)
da_agg = xr.DataArray(
    da_agg,
    dims=("beach_id", "n", "season", "bootstrap"),
    coords={
        "n": range(nb_n.n.size),
        "beach_id": gpr_pp.beach_id.values,
        "season": list(idata.keys()),
    },
)
da_agg = da_agg.stack(draw=("n", "bootstrap")).drop_vars(["draw", "n", "bootstrap"])
da_agg = da_agg.assign_coords(draw=range(da_agg.draw.size))
da_agg = da_agg.rename({"draw": "n"})
da_agg = da_agg.isel(n=slice(0, 2000))

da_agg = da_agg.sortby("beach_id")
da_agg = da_agg.assign_coords(
    {
        "lon": ("beach_id", ospar.lon.values),
        "lat": ("beach_id", ospar.lat.values),
    }
)

da_agg.name = VARIABLE
da_agg.attrs["description"] = "Posterior predictive distribution of litter abundance"
da_agg.attrs["units"] = "number of items per 100 m"

gpr_pp[VARIABLE + "2"] = da_agg


# %%
# Combine results to DataTree
# =============================================================================
gpr_pp = dt.DataTree.from_dict(
    {"posterior_predictive": gpr_pp, "lambda_yeojohnson": lmbda_yeo}
)
# %%
# Save GP model output
# =============================================================================
base_path = f"data/gpr/{VARIABLE}/{YEAR}/"
if not os.path.exists(base_path):
    os.makedirs(base_path)

for k, d in idata.items():
    d.to_netcdf(base_path + f"idata_{k}.nc")

gpr_pp.to_zarr(base_path + "posterior_predictive.zarr")
