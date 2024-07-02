# %%
# Run on drago via
# sbatch -p generic -n 1 --cpus-per-task=4 --mem-per-cpu=20GB drago_gpr.sh
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import datatree as dt
import numpy as np
import numpyro
import pymc as pm
import xarray as xr
from scipy.special import expit, logit

from utils.kernel import Matern32Chordal

numpyro.set_host_device_count(4)


# %%
print("Loading data...")

SEASON = ["DJF", "MAM", "JJA", "SON"]
RANDOM_SEED = 2803 + 2
QUANTITY = "fraction"
VARIABLE = "FISH"
YEAR = 2001


rng = np.random.default_rng(RANDOM_SEED)

ospar = dt.open_datatree("data/beach_litter/ospar/preprocessed.zarr", engine="zarr")
fraction = ospar["/".join(["preprocessed", QUANTITY, VARIABLE])]
fraction = fraction.sel(year=slice(YEAR, 2020))

is_valid = fraction.notnull()
fraction = (
    fraction.where(fraction >= 0.001, 0.001)
    .where(fraction <= 0.999, 0.999)
    .where(is_valid)
)

fraction = fraction.dropna("beach_id", **{"how": "all"})


# %%
# GP model
# =============================================================================
def fit_gp_model(season, fit=True):
    # Dimensions and coordinates
    beaches_all = fraction.beach_id.values
    beach_lonlats_all = np.vstack([fraction.lon, fraction.lat]).T

    data = fraction.sel(season=season).dropna("beach_id", **{"how": "all"})
    data = data.assign_coords(beach_idx=("beach_id", np.arange(data.beach_id.size)))
    data_stacked = data.stack(x=[...]).dropna("x")

    data_stacked = logit(data_stacked)

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
        mean_lower, mean_upper = fraction.mean("year").quantile([0.025, 0.975])
        mean_lower = max(mean_lower.item(), 0.001)
        mean_upper = min(mean_upper.item(), 0.999)
        params_mu_mu = pm.find_constrained_prior(
            distribution=pm.Normal,
            lower=logit(mean_lower),
            upper=logit(mean_upper),
            init_guess={"mu": logit(0.5), "sigma": 1},
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
            upper=3 / np.sqrt(2),
            init_guess={"alpha": 10, "beta": 10},
        )
        params_sigma = pm.find_constrained_prior(
            distribution=pm.InverseGamma,
            lower=0.1,
            upper=1,
            init_guess={"alpha": 2, "beta": 0.1},
        )

        # Mean function
        mu_mu = pm.Normal("mu_mu", **params_mu_mu)
        mean_func = pm.gp.mean.Constant(mu_mu)

        # Covariance function
        eta_1 = pm.Gamma("eta_1", **params_eta)
        eta_2 = pm.Gamma("eta_2", **params_eta)
        rho_1 = pm.InverseGamma("rho_1", **params_rho1)
        rho_2 = pm.InverseGamma("rho_2", **params_rho2)
        kernel_short = eta_1**2 * Matern32Chordal(2, ls=rho_1)
        kernel_long = eta_2**2 * Matern32Chordal(2, ls=rho_2)
        cov_func = kernel_short + kernel_long + pm.gp.cov.WhiteNoise(1e-4)

        # Latent GP
        gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
        mu_trans = gp.prior("mu_trans", _X, dims="beach", jitter=1e-4)

        # Broadcast beaches to samples
        mu = pm.Deterministic("mu", mu_trans[_beach_idx], dims="sample")

        sigma = pm.InverseGamma("sigma", **params_sigma)

        observed = pm.Normal(
            VARIABLE,
            mu=mu,
            sigma=sigma,
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
            "mu_pred", beach_lonlats_all, jitter=1e-4, dims="beach_pred"
        )

        sigma_pred = pm.Deterministic("sigma_pred", sigma)

        cond_observed = pm.Normal(
            VARIABLE + "_pred",
            mu=mu_trans_pred,
            sigma=sigma_pred,
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
                "sigma",
                "mu_pred",
            ],
        )
        idata.extend(prior_p)

    with model:
        # Sample from the GP conditional distributionç
        ppc = pm.sample_posterior_predictive(
            idata.posterior,
            var_names=["mu_pred", VARIABLE + "_pred"],
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
            "mu_pred": "mu",
            VARIABLE + "_pred": VARIABLE + "_trans",
        }
    )
    model_output[season] = pp

gpr_pp = xr.concat(list(model_output.values()), dim="season").assign_coords(
    season=list(model_output.keys())
)
gpr_pp = gpr_pp.assign_coords(
    {
        "lon": ("beach_id", fraction.lon.values),
        "lat": ("beach_id", fraction.lat.values),
    }
)

# Compute aggregated posterior predictive (uncertainty from mean and dispersion)
# -----------------------------------------------------------------------------
da_agg = expit(gpr_pp[VARIABLE + "_trans"])
da_agg.name = VARIABLE
da_agg.attrs["description"] = (
    f"Posterior predictive distribution of {VARIABLE} fraction."
)
da_agg.attrs["units"] = "fraction of items per 100 m"

gpr_pp[VARIABLE] = da_agg


# %%
# Combine results to DataTree
# =============================================================================
gpr_pp = dt.DataTree.from_dict({"posterior_predictive": gpr_pp})
# %%
# Save GP model output
# =============================================================================
base_path = f"data/gpr/{QUANTITY}/{VARIABLE}/{YEAR}/"
if not os.path.exists(base_path):
    os.makedirs(base_path)

for k, d in idata.items():
    d.to_netcdf(base_path + f"idata_{k}.nc")

gpr_pp.to_zarr(base_path + "posterior_predictive.zarr")
