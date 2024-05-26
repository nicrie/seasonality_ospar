# %%
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import xarray as xr
from datatree import open_datatree
from sklearn.metrics import r2_score
from tqdm import tqdm

# %%
ospar = xr.open_dataset("data/beach_litter/ospar/preprocessed.zarr", engine="zarr")

# Litter sources
sources = open_datatree("data/litter_sources.zarr", engine="zarr")

pca_result = xr.open_dataset("data/pca/pca_beaches.nc", engine="netcdf4")
components = pca_result.comps.rename({"mode": "cluster"}).sel(cluster=[1, 2])
# Effect size
s = components.mean("n")
s = abs(s).where(s >= 0, 0)
s = s

# Probability
c = (components > 0).mean("n")
# %%

fishing = sources["fishing/capture"].to_dataset()[["size", "seasonal_potential"]]
river = sources["river/discharge_v5"].to_dataset()[["size", "seasonal_potential"]]
mari = sources["mariculture"].to_dataset()[["size", "seasonal_potential"]]

fishing = fishing.sortby("lat")
river = river.sortby("lat")
mari = mari.sortby("lat")

river = river.sel(lon=slice(-15, 15), lat=slice(35, 65))

mask_fishing = fishing.seasonal_potential.isel(cluster=0, drop=True).notnull()
mask_river = river.seasonal_potential.isel(cluster=0, drop=True).notnull()
mask_mari = mari.seasonal_potential.isel(cluster=0, drop=True).notnull()

fishing = fishing.where(mask_fishing, drop=True)
river = river.where(mask_river, drop=True)
mari = mari.where(mask_mari, drop=True)

# %%


def get_predictors(data, lon_b, lat_b):
    dims = ("lat", "lon")
    step = 1.5
    beach_lon_range = slice(lon_b - step, lon_b + step)
    beach_lat_range = slice(lat_b - step, lat_b + step)

    ok = 2 * step
    ref = 180
    alpha = ref / ok

    beach_data = data.sel(lon=beach_lon_range, lat=beach_lat_range)
    lon_weights = np.cos(np.deg2rad(alpha * (beach_data.lon - lon_b)))
    lat_weights = np.cos(np.deg2rad(alpha * (beach_data.lat - lat_b)))

    weights = lon_weights * lat_weights
    weights = weights.where(weights > 0, 0)
    weights = weights.where(beach_data["size"].notnull(), 0)

    weights = weights / weights.sum()
    weights = weights.fillna(0)

    size = beach_data["size"].weighted(weights).sum(dims)
    weights2 = beach_data["size"].fillna(0) * weights
    potential = beach_data.seasonal_potential.weighted(weights2).mean(dims)
    return size, potential


s_river = []
s_fish = []
s_mari = []
p_river = []
p_fish = []
p_mari = []

for lon, lat in tqdm(zip(pca_result.lon, pca_result.lat), total=len(pca_result.lon)):
    river_size, river_potential = get_predictors(river, lon, lat)
    fishing_size, fishing_potential = get_predictors(fishing, lon, lat)
    mari_size, mari_potential = get_predictors(mari, lon, lat)

    s_river.append(river_size)
    s_fish.append(fishing_size)
    s_mari.append(mari_size)

    p_river.append(river_potential)
    p_fish.append(fishing_potential)
    p_mari.append(mari_potential)
# %%
# Convert lists to DataArrays
s_river = xr.concat(s_river, dim="beach_id")
s_fish = xr.concat(s_fish, dim="beach_id")
s_mari = xr.concat(s_mari, dim="beach_id")
p_river = xr.concat(p_river, dim="beach_id")
p_fish = xr.concat(p_fish, dim="beach_id")
p_mari = xr.concat(p_mari, dim="beach_id")

s_river = s_river.assign_coords(beach_id=pca_result.beach_id)
s_fish = s_fish.assign_coords(beach_id=pca_result.beach_id)
s_mari = s_mari.assign_coords(beach_id=pca_result.beach_id)
p_river = p_river.assign_coords(beach_id=pca_result.beach_id)
p_fish = p_fish.assign_coords(beach_id=pca_result.beach_id)
p_mari = p_mari.assign_coords(beach_id=pca_result.beach_id)

size = xr.concat([s_river, s_fish, s_mari], dim="predictor")
potential = xr.concat([p_river, p_fish, p_mari], dim="predictor")
size = size.assign_coords(predictor=["river", "fishing", "mariculture"])
potential = potential.assign_coords(predictor=["river", "fishing", "mariculture"])

ds = xr.Dataset({"size": size, "potential": potential})
ds = ds.drop_vars(["quantile", "step", "surface", "production_type"])

# %%


# Fishing -> sqrt transform
# River -> log transform
# Mariculture -> no transform
predictors = ds.copy().fillna(0)
predictors["size"].loc[dict(predictor="fishing")] = np.sqrt(
    predictors["size"].loc[dict(predictor="fishing")]
)
predictors["size"].loc[dict(predictor="river")] = np.log(
    predictors["size"].loc[dict(predictor="river")] + 1
)


predictors = predictors / predictors.max("beach_id", skipna=True)
predictors = predictors.where(predictors > 0.001, 0.001)

predictors["y"] = c

# predictors = predictors.where(predictors["y"] > 0.5)

ok = predictors.copy()
ok = ok["y"].where(ok["y"] < 0.65, 1).where(ok["y"] >= 0.35, 0)

# %%

y = predictors["y"].sel(cluster=2).dropna("beach_id")
plt.scatter(y.lon, y.lat)
plt.show()
print(len(y))
valid_beaches = y.beach_id.values
x1 = predictors["size"].sel(
    beach_id=valid_beaches,
    predictor="mariculture",
    forcing="exports",
    species_group="Bivalve molluscs",
    # cluster=2,
)
x2 = predictors["potential"].sel(
    beach_id=valid_beaches,
    predictor="mariculture",
    forcing="exports",
    species_group="Bivalve molluscs",
    cluster=2,
)
plt.scatter(x1 * x2, y)
plt.axhline(0.5)
plt.axhline(0.4)
plt.axhline(0.6)
plt.xlim(0, 1)
plt.ylim(0, 1)

# %%


y = predictors["y"].sel(cluster=2).dropna("beach_id")
valid_beaches = y.beach_id.values
S = (
    predictors["size"]
    .sel(beach_id=valid_beaches, forcing="exports", species_group="Bivalve molluscs")
    .values
)
P = (
    predictors["potential"]
    .sel(
        beach_id=valid_beaches,
        forcing="exports",
        species_group="Bivalve molluscs",
        cluster=2,
    )
    .values
)
ONES = np.ones_like(S.shape[1])

with pm.Model() as model:
    # Priors for unknown model parameters
    # c_r = pm.HalfNormal("c_r", sigma=1)
    # c_f = pm.HalfNormal("c_f", sigma=1)
    # c_m = pm.HalfNormal("c_m", sigma=1)
    c_0 = pm.HalfNormal("c_0", sigma=1)
    c_1 = pm.HalfNormal("c_1", sigma=1)

    s_r = pm.Normal("s_r", mu=0, sigma=1)
    s_f = pm.Normal("s_f", mu=0, sigma=1)
    s_m = pm.Normal("s_m", mu=0, sigma=1)

    p_r = pm.Normal("p_r", mu=0, sigma=1)
    p_f = pm.Normal("p_f", mu=0, sigma=1)
    p_m = pm.Normal("p_m", mu=0, sigma=1)

    sigma = pm.Gamma("sigma", alpha=1, beta=1)

    # Expected value of the predictand
    # mu = pm.Deterministic(
    #     "mu",
    #     c_r * S[0] ** s_r * P[0] ** p_r
    #     + c_f * S[1] ** s_f * P[1] ** p_f
    #     + c_m * S[2] ** s_m * P[2] ** p_m
    #     + c_0 * ONES,
    # )
    mu = pm.Deterministic(
        "mu",
        c_1
        * S[0] ** s_r
        * P[0] ** p_r
        * S[1] ** s_f
        * P[1] ** p_f
        * S[2] ** s_m
        * P[2] ** p_m
        + c_0 * ONES,
    )

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y.values)

    # Posterior sampling
    trace = pm.sample(
        1000,
        tune=8000,
        return_inferencedata=True,
        target_accept=0.95,
        nuts_sampler="numpyro",
    )

# %%
# To check the results
var_list1 = ["c_r", "s_r", "p_r"]
var_list2 = ["c_f", "s_f", "p_f"]
var_list3 = ["c_m", "s_m", "p_m"]
var_list = var_list1 + var_list2 + var_list3 + ["sigma"]
pm.plot_trace(trace, var_names=var_list)
pm.summary(trace, var_names=["c_r", "c_f", "c_m"])


# %%
def model_predict(trace, s1, p1, s2, p2, s3, p3):
    pp = trace["posterior"].mean(("chain", "draw"))
    return (
        pp["c_r"].item() * s1 ** pp["s_r"].item() * p1 ** pp["p_r"].item()
        + pp["c_f"].item() * s2 ** pp["s_f"].item() * p2 ** pp["p_f"].item()
        + pp["c_m"].item() * s3 ** pp["s_m"].item() * p3 ** pp["p_m"].item()
        + pp["c_0"].item()
    )


def model_predict(trace, s1, p1, s2, p2, s3, p3):
    pp = trace["posterior"].mean(("chain", "draw"))
    return (
        pp["c_1"].item()
        * s1 ** pp["s_r"].item()
        * p1 ** pp["p_r"].item()
        * s2 ** pp["s_f"].item()
        * p2 ** pp["p_f"].item()
        * s3 ** pp["s_m"].item()
        * p3 ** pp["p_m"].item()
        + pp["c_0"].item()
    )


y_pred = model_predict(trace, S[0], P[0], S[1], P[1], S[2], P[2])
plt.scatter(y, y_pred)
plt.plot([0, 1], [0, 1], color="red")
plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
# Compute R2 score
r2 = r2_score(y, y_pred)
print("R2 score:", r2)
# %%
