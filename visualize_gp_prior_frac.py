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
from scipy.special import expit, logit

from utils.kernel import Matern32Haversine

numpyro.set_host_device_count(4)


# %%
# Load data
# =============================================================================
SEASON = ["DJF", "MAM", "JJA", "SON"]
RANDOM_SEED = 2803 + 1
VARIABLE = "fraction/AQUA"
YEAR = 2001

rng = np.random.default_rng(RANDOM_SEED)

ospar = dt.open_datatree("data/beach_litter/ospar/preprocessed.zarr", engine="zarr")
ospar = ospar["preprocessed/" + VARIABLE]
ospar = ospar.sel(year=slice(YEAR, 2020))

is_valid = ospar.notnull()
ospar = ospar.where(ospar >= 0.001, 0.001).where(ospar <= 0.999, 0.999).where(is_valid)
ospar = ospar.dropna("beach_id", **{"how": "all"})

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
samples = pm.draw(mvn, 500)
ax = plt.axes()
x = np.arange(ospar.beach_id.size)
ax.plot(
    x,
    (ospar).sel(season="JJA").values,
    zorder=5,
    color="C1",
    marker=".",
    ls="",
    alpha=0.5,
)
ax.plot(x, expit(samples.T), "C0", alpha=0.1)
ax.set_ylim(0, 1)
plt.show()


# %%
