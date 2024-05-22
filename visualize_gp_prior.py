# %%
# Run on drago via
# sbatch -p generic -n 1 --cpus-per-task=4 --mem-per-cpu=20GB drago_gpr.sh
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import datatree as dt
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pymc as pm

from utils.kernel import Matern32Haversine
from utils.transformation import estimate_lmbda, yeojohnson

numpyro.set_host_device_count(4)


# %%
# Load data
# =============================================================================
SEASON = ["DJF", "MAM", "JJA", "SON"]
RANDOM_SEED = 2803 + 1
VARIABLE = "Aquaculture"
YEAR = 2016

rng = np.random.default_rng(RANDOM_SEED)

ospar = dt.open_datatree("data/beach_litter/ospar/preprocessed.zarr", engine="zarr")
ospar = ospar["preprocessed"].to_dataset()
ospar = ospar.sel(year=slice(YEAR, 2020))
ospar = ospar[VARIABLE]
ospar = ospar.dropna("beach_id", **{"how": "all"})

# Transformation parameter for Yeo-Johnson transform
lmbda = estimate_lmbda(ospar.stack(x=[...]), dim="x").item()

# Dimensions and coordinates
beaches_all = ospar.beach_id.values
beach_lonlats_all = np.vstack([ospar.lon, ospar.lat]).T

data = ospar.dropna("beach_id", **{"how": "all"})
data = data.assign_coords(beach_idx=("beach_id", np.arange(data.beach_id.size)))
data_stacked = data.stack(x=[...]).dropna("x")

beach_lonlats = np.vstack([data.lon, data.lat]).T
beach_idx = data_stacked.beach_idx
beaches = data.beach_id.values


# %%
# Set up GP model
# =============================================================================
# Set up priors
mean_lower, mean_upper = ospar.mean("year").quantile([0.025, 0.975])
mean_lower = mean_lower.item()
mean_upper = mean_upper.item()
params_mu_mu = pm.find_constrained_prior(
    distribution=pm.Normal,
    lower=yeojohnson(mean_lower, lmbda),
    upper=yeojohnson(mean_upper, lmbda),
    init_guess={"mu": yeojohnson(10, lmbda), "sigma": yeojohnson(100, lmbda)},
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
    mass=0.95,
    init_guess={"alpha": 10, "beta": 10},
)

mu_mu = pm.Normal.dist(**params_mu_mu)
eta_1 = pm.Gamma.dist(**params_eta)
eta_2 = pm.Gamma.dist(**params_eta)
rho_1 = pm.InverseGamma.dist(**params_rho1)
rho_2 = pm.InverseGamma.dist(**params_rho2)

kernel_short = eta_1**2 * Matern32Haversine(2, ls=rho_1)
kernel_long = eta_2**2 * Matern32Haversine(2, ls=rho_2)
cov_func = kernel_short + kernel_long + pm.gp.cov.WhiteNoise(1e-4)


# %%
# Evaluate priors of GP model
# -----------------------------------------------------------------------------
mu_eval = pm.draw(mu_mu, 10)
K = cov_func(beach_lonlats).eval()
mvn = pm.MvNormal.dist(mu=mu_mu, cov=K)
# mvn = pm.MvNormal.dist(mu=len(beaches) * [5], cov=K)
samples = pm.draw(mvn, 50)
ax = plt.axes()
yeojohnson(ospar, lmbda).mean("year").sel(season="DJF").plot.scatter(
    x="beach_id", zorder=5, color="C1", ax=ax
)
ax.plot(np.arange(ospar.beach_id.size), samples.T, "C0", alpha=0.1)
ax.set_ylim(-2, 10)
plt.show()


# %%
