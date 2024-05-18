# %%
import arviz as az
import datatree as dt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

import utils.styles
from utils.styles import get_cyclic_palette
from utils.transformation import yeojohnson_inv

utils.styles.set_theme()

COLORS = get_cyclic_palette(as_cmap=False, n_colors=4)
SEASONS = ["DJF", "MAM", "JJA", "SON"]
VARIABLE = "Plastic"
YEAR = 2001

base_path = f"data/gpr/{VARIABLE}/{YEAR}/"
fig_path = f"figs/gpr/evaluation/{VARIABLE}/{YEAR}/"
# %%
ospar = dt.open_datatree("data/ospar/preprocessed.zarr", engine="zarr")
litter_o = ospar["preprocessed"].to_dataset()
litter_o = litter_o[VARIABLE]
litter_o = litter_o.sel(year=slice(YEAR, 2020)).dropna("beach_id", **{"how": "all"})

model = dt.open_datatree(base_path + "posterior_predictive.zarr", engine="zarr")
litter_m = model["posterior_predictive"][VARIABLE]

idata = {}
for s in SEASONS:
    idata[s] = az.from_netcdf(base_path + f"idata_{s}.nc")


# %%
# Model convergence // trace plots
# =============================================================================

summary = {}
for s in SEASONS:
    summary[s] = az.summary(
        idata[s],
        var_names=[
            "mu_mu",
            "eta_1",
            "eta_2",
            "rho_1",
            "rho_2",
            "phi",
            "mu_trans",
        ],
        round_to=2,
        fmt="xarray",
    )
summary = dt.DataTree.from_dict(summary)

summary.to_zarr(base_path + "evaluation_summary.zarr")


# Check that r_hat < 1.05 for all models/parameters
(summary.sel(metric="r_hat") < 1.05).all()

# %%

mu_o = litter_o.mean("year")
mu_m = litter_m.mean("n")

var_o = litter_o.var("year", ddof=1)
var_m = litter_m.var("n", ddof=1)
# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

for c, s in zip(COLORS, SEASONS):
    ax[0].scatter(
        mu_o.sel(season=s),
        mu_m.sel(season=s),
        s=20,
        ec="k",
        lw=0.2,
        marker="o",
        alpha=0.7,
        color=c,
        label=s,
    )
    ax[1].scatter(
        var_o.sel(season=s),
        var_m.sel(season=s),
        s=20,
        ec="k",
        lw=0.2,
        alpha=0.7,
        marker="o",
        color=c,
    )
for a in ax:
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("Observed")
    a.set_ylabel("Modeled")
ax[0].set_xlim(1e0, 1e5)
ax[0].set_ylim(1e0, 1e5)
ax[1].set_xlim(1e0, 1e10)
ax[1].set_ylim(1e0, 1e10)
ax[0].plot([1e0, 1e5], [1e0, 1e5], "k--", lw=0.5)
ax[1].plot([1e0, 1e10], [1e0, 1e10], "k--", lw=0.5)
ax[0].set_title("A | Expected value E[Y]", loc="left")
ax[1].set_title("B | Variance Var[Y]", loc="left")
ax[0].legend(loc="upper left", frameon=False)
sns.despine(fig)
fig.tight_layout()
fig.savefig(fig_path + "gpr_model_mean_variance.png", dpi=300)


# %%


# priors are the same for all season
lmbda = model["lambda_yeojohnson"]["lambda"]
prior = idata["DJF"].prior.squeeze()

fig = plt.figure(figsize=(7.2, 8))
gs = GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.1, height_ratios=[1, 0.2, 1, 1])
axes = {}
axes["mu_mu"] = fig.add_subplot(gs[0, 0])
axes["phi"] = fig.add_subplot(gs[0, 1])
axes["eta_1"] = fig.add_subplot(gs[2, 0])
axes["eta_2"] = fig.add_subplot(gs[2, 1])
axes["rho_1"] = fig.add_subplot(gs[3, 0])
axes["rho_2"] = fig.add_subplot(gs[3, 1])


axes["mu_mu"].set_title(r"a) GP mean | $\mu_{\mu}$", loc="left")
axes["eta_1"].set_title(r"c) GP covariance | $\eta_{short}$", loc="left")
axes["eta_2"].set_title(r"d) GP covariance | $\eta_{long}$", loc="left")
axes["rho_1"].set_title(r"e) GP covariance | $\rho_{short}$", loc="left")
axes["rho_2"].set_title(r"f) GP covariance | $\rho_{long}$", loc="left")
axes["phi"].set_title(r"g) Dispersion | $\phi$", loc="left")


# Priors
prior_mu_mu = yeojohnson_inv(prior.mu_mu, lmbda)
kwargs = {"clip": [0, None], "color": ".5", "fill": True}
kwargs1 = {"clip": [0, 1000], "color": ".5", "fill": True}
sns.kdeplot(data=prior_mu_mu, ax=axes["mu_mu"], label="Prior", **kwargs1)
sns.kdeplot(data=prior.phi, ax=axes["phi"], label="Prior", **kwargs)
sns.kdeplot(data=prior.eta_1, ax=axes["eta_1"], **kwargs)
sns.kdeplot(data=prior.eta_2, ax=axes["eta_2"], **kwargs)
sns.kdeplot(data=prior.rho_1, ax=axes["rho_1"], **kwargs)
sns.kdeplot(data=prior.rho_2, ax=axes["rho_2"], **kwargs)


for i, (season, id) in enumerate(idata.items()):
    post = az.extract(
        id,
        group="posterior",
        combined=True,
        var_names=["mu_mu", "phi", "eta_1", "eta_2", "rho_1", "rho_2"],
    )

    post_mu_mu = yeojohnson_inv(post.mu_mu, lmbda)

    kwargs = dict(label=season, color=COLORS[i], fill=False, alpha=0.8)
    sns.kdeplot(post_mu_mu, ax=axes["mu_mu"], **kwargs)
    sns.kdeplot(post.phi, ax=axes["phi"], **kwargs)
    sns.kdeplot(post.eta_1, ax=axes["eta_1"], **kwargs)
    sns.kdeplot(post.eta_2, ax=axes["eta_2"], **kwargs)
    sns.kdeplot(post.rho_1, ax=axes["rho_1"], **kwargs)
    sns.kdeplot(post.rho_2, ax=axes["rho_2"], **kwargs)

axes["mu_mu"].set_xlim(0, 1000)
axes["phi"].set_xlim(0, 8)
axes["eta_1"].set_xlim(0, 2)
axes["eta_2"].set_xlim(0, 2)
axes["rho_1"].set_xlim(0, 100)
axes["rho_2"].set_xlim(0, 4000)

axes["phi"].legend()

for ax in axes.values():
    ax.set_ylabel("")
    ax.set_yticks([])

sns.despine(fig=fig, trim=True, left=True)
plt.savefig(fig_path + "trace_plots.png", dpi=300, bbox_inches="tight")
# %%
