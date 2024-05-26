# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
import seaborn as sns
import xarray as xr
from datatree import DataTree

# Define the coordinates for the 9 regions based on the provided image
# These coordinates are approximate and should be adjusted as necessary
from sklearn.cluster import HDBSCAN


# %%
def monthly2seasonal(da, dim):
    """Convert any DataArray with monthly time dimension to seasonal sum.

    Seasonal sum is computed as follows:
    DJF: 12, 1, 2
    MAM: 3, 4, 5
    JJA: 6, 7, 8
    SON: 9, 10, 11

    New output dim is "month" with coordinates [1, 4, 7, 10]

    """
    # Define the seasons
    seasons = {
        1: [12, 1, 2],  # DJF
        4: [3, 4, 5],  # MAM
        7: [6, 7, 8],  # JJA
        10: [9, 10, 11],  # SON
    }
    # Create a dictionary with the sum of each season
    seasonal_sum = {}
    for season, months in seasons.items():
        seasonal_sum[season] = da.sel({dim: months}).sum(dim)
    # Concatenate the dictionary into a new DataArray
    seasonal_sum = xr.concat(
        seasonal_sum.values(), dim=pd.Index(seasons.keys(), name="season")
    )
    return seasonal_sum


# %%
# Load data
# =============================================================================
# Beach litter clusters
beach_litter = xr.open_dataset("data/pca/pca_beaches.nc")
beach_litter = beach_litter.scores.sel(mode=[1, 2]).median("n")
beach_litter = beach_litter.assign_coords(season=[1, 4, 7, 10])
beach_litter = beach_litter.rename({"mode": "cluster"})


# First sales (fishing, aquaculture)
sales = xr.open_dataarray("data/economic/first_sales/first_sales_clean.nc")
sales = sales.fillna(0)
# Share aquaculture vs wild catch
fish_production_fao = xr.open_dataset("data/economic/production/fao_fish_production.nc")


sales_per_country = (
    sales.sum("month").set_index(location="country").groupby("location").sum()
)
sales_per_country = sales_per_country.assign_coords(
    location=sales_per_country.location.astype(str)
)
sales_per_country = sales_per_country.rename({"location": "country"})
fao_ref = fish_production_fao["average_live_weight"].sel(production_type="wild")
fao_ref = fao_ref.assign_coords(species=sales.species)

# Compute standard statistics
total_sales = sales.sum("month")
total_sales_cum = sales.cumsum("month")
sales_rel = sales / total_sales
cv_sales = sales.std("month") / sales.mean("month")
cv_sales.name = "coefficient_of_variation"

# River discharge
river_discharge_v5 = xr.open_dataset("data/physical/river/efas_v5_discharge.nc")["mean"]
river_discharge_v5 = river_discharge_v5.rename({"latitude": "lat", "longitude": "lon"})

river_discharge_v4 = xr.open_dataarray("data/physical/river/efas_river_discharge.nc")
river_discharge_v4 = river_discharge_v4.assign_coords(
    x=river_discharge_v4.x, y=river_discharge_v4.y
)

total_discharge_v4 = river_discharge_v4.sum("month")
cv_river_v4 = river_discharge_v4.std("month") / river_discharge_v4.mean("month")
cv_river_v4.name = "coefficient_of_variation"

total_discharge_v5 = river_discharge_v5.sum("month")
cv_river_v5 = river_discharge_v5.std("month") / river_discharge_v5.mean("month")
cv_river_v5.name = "coefficient_of_variation"

# Plastic emissions
plastic = xr.open_dataarray("data/physical/river/strokal2023_efas_v5_weights.nc")
cv_plastic = plastic.std("month") / plastic.mean("month")


# Fishing (World Fishing Watch) - Wild Capture
from utils.definitions import nea_ocean_basins

fish_wild = xr.open_dataarray("data/economic/fishing/wild_capture_fishing_intensity.nc")
fish_wild = fish_wild.resample(time="1ME").sum()

# Compute fish stats for meta regions
fish_wild_region = fish_wild.where(nea_ocean_basins.mask_3D(fish_wild)).sum(
    ("lat", "lon")
)

fish_wild_region = fish_wild_region.assign_coords(
    year=fish_wild_region.time.dt.year, month=fish_wild_region.time.dt.month
)
fish_wild_region = fish_wild_region.set_index(time=["year", "month"]).unstack()

fish_region = fish_wild_region.mean("year")
fish_region_std = fish_wild_region.std("year")
cv_fish_region = fish_region.std("month") / fish_region.mean("month")
cv_fish_region.name = "coefficient_of_variation"


fish_wild = fish_wild.groupby("time.month").mean()
cv_fish_wild = fish_wild.std("month") / fish_wild.mean("month")


# %%
# Aquaculture
mariculture = xr.open_dataset("data/economic/aquaculture/mariculture_seasonality.nc")
mariculture = mariculture.to_array("forcing")
cv_mariculture = mariculture.std("month") / mariculture.mean("month")


# %%
fishing_sales = xr.Dataset(
    {"climatology": sales, "size": sales.sum("month"), "cv": cv_sales}
)
river = xr.Dataset(
    {
        "climatology": river_discharge_v4,
        "size": river_discharge_v4.sum("month"),
        "cv": cv_river_v4,
    }
)
river_v5 = xr.Dataset(
    {
        "climatology": river_discharge_v5,
        "size": river_discharge_v5.sum("month"),
        "cv": cv_river_v5,
    }
)
plastic = xr.Dataset(
    {"climatology": plastic, "size": plastic.sum("month"), "cv": cv_plastic}
)
fishing = xr.Dataset(
    {"climatology": fish_wild, "size": fish_wild.sum("month"), "cv": cv_fish_wild},
    attrs=dict(
        description="Wild capture fishing intensity in average hours per month per 0.5ºx0.5º"
    ),
)
fishing_region = xr.Dataset(
    {"climatology": fish_region, "size": fish_region.sum("month"), "cv": cv_fish_region}
)
mari = xr.Dataset(
    {"climatology": mariculture, "size": mariculture.sum("month"), "cv": cv_mariculture}
)


# %%
# Compute seasonal similarity between beach litter clusters and sources
# =============================================================================
def seasonal_potential(r, cv):
    # While correlation is normed, the coefficient of variation
    # does not reflect the same scale for all variables;
    # Specifically, variations of fish exports doesn't have to follow the same relative variations
    # as the aquaculture production ==> normalize the CVs
    quntl = 0.95
    try:
        norm_vals = cv.quantile(quntl, ["lat", "lon"])
    except ValueError:
        try:
            norm_vals = cv.quantile(quntl, ["y", "x"])
        except ValueError:
            try:
                norm_vals = cv.quantile(quntl, ["location"])
            except ValueError:
                norm_vals = cv.quantile(quntl, ["region"])
    cv = cv / norm_vals
    return r * cv


# River discharge
river_seasonal = monthly2seasonal(river.climatology.roll(month=1), "month")
r_river = xr.corr(beach_litter, river_seasonal, dim="season")
r_river.name = "pearson_correlation"
r_river.attrs["description"] = (
    "Correlation coefficient between beach litter clusters and river discharge"
)
river["correlation"] = r_river
river["seasonal_potential"] = seasonal_potential(river.correlation, river.cv)

# River discharge v5
river_v5_seasonal = monthly2seasonal(river_v5.climatology.roll(month=1), "month")
r_river_v5 = xr.corr(beach_litter, river_v5_seasonal, dim="season")
r_river_v5.name = "pearson_correlation"
r_river_v5.attrs["description"] = (
    "Correlation coefficient between beach litter clusters and river_v5 discharge"
)
river_v5["correlation"] = r_river_v5
river_v5["seasonal_potential"] = seasonal_potential(river_v5.correlation, river_v5.cv)

# Riverine plastic emissions
plastic_seasonal = monthly2seasonal(plastic.climatology.roll(month=1), "month")
r_plastic = xr.corr(beach_litter, plastic_seasonal, dim="season")
r_plastic.name = "pearson_correlation"
r_plastic.attrs["description"] = (
    "Correlation coefficient between beach litter clusters and plastic emissions"
)
plastic["correlation"] = r_plastic
plastic["seasonal_potential"] = seasonal_potential(plastic.correlation, plastic.cv)

# Fishing (capture)
fish_seasonal = monthly2seasonal(fishing.climatology.roll(month=1), "month")
r_fish = xr.corr(beach_litter, fish_seasonal, dim="season")
r_fish.name = "pearson_correlation"
r_fish.attrs["description"] = (
    "Correlation coefficient between beach litter clusters and wild capture fishing intensity"
)
fishing["correlation"] = r_fish
fishing["seasonal_potential"] = seasonal_potential(fishing.correlation, fishing.cv)

# Fishing (capture, regional)
fish_region_seasonal = monthly2seasonal(
    fishing_region.climatology.roll(month=1), "month"
)
r_fish_region = xr.corr(beach_litter, fish_region_seasonal, dim="season")
r_fish_region.name = "pearson_correlation"
r_fish_region.attrs["description"] = (
    "Correlation coefficient between beach litter clusters and regional wild capture fishing intensity"
)
fishing_region["correlation"] = r_fish_region
fishing_region["seasonal_potential"] = seasonal_potential(
    fishing_region.correlation, fishing_region.cv
)


# Fishing (first sales)
sales_seasonal = monthly2seasonal(fishing_sales.climatology.roll(month=1), "month")
r_fishing = xr.corr(beach_litter, sales_seasonal, dim="season")
r_fishing.name = "pearson_correlation"
r_fishing.attrs["description"] = (
    "Correlation coefficient between beach litter clusters and fishing sales"
)
fishing_sales["correlation"] = r_fishing
fishing_sales["seasonal_potential"] = seasonal_potential(
    fishing_sales.correlation, fishing_sales.cv
)


# Aquaculture (wave height, farm density)
mari_seasonal = monthly2seasonal(mari.climatology.roll(month=1), "month")
r_mari = xr.corr(beach_litter, mari_seasonal, dim="season")
r_mari.name = "pearson_correlation"
r_mari.attrs["description"] = (
    "Correlation coefficient between beach litter clusters and wave height"
)
mari["correlation"] = r_mari
mari["seasonal_potential"] = seasonal_potential(mari.correlation, mari.cv)


# %%
# For FISHING (CAPTURE): check whether specific species match beach litter clusters
# =============================================================================
is_valid = fishing_sales.seasonal_potential > 0

sales_relevant = fishing_sales.climatology.where(is_valid)
data_sum = sales_relevant.sum("species")
fish_sales_cv = data_sum.std("month") / data_sum.mean("month")
fish_sales_cv.name = "coefficient_of_variation"
fish_sales_seas = monthly2seasonal(data_sum.roll(month=1), "month")
fish_sales_corr = xr.corr(beach_litter, fish_sales_seas, dim="season")
fish_sales_corr.name = "correlation"
fish_sales_seasonal_potential = fish_sales_cv * fish_sales_corr
fish_sales_seasonal_potential.name = "seasonal_potential"

fish_sales_relevant_species = xr.Dataset(
    {
        "climatology": sales_relevant,
        "size": sales_relevant.sum("month"),
        "cv": fish_sales_cv,
        "correlation": fish_sales_corr,
        "seasonal_potential": fish_sales_seasonal_potential,
    },
    attrs=dict(description="Fishing sales data with positive seasonal potential."),
)

# %%
# Save some storage by removing the climatology
river = river.drop_vars("climatology")
river_v5 = river_v5.drop_vars("climatology")
plastic = plastic.drop_vars("climatology")
fishing = fishing.drop_vars("climatology")
fishing_region = fishing_region.drop_vars("climatology")
fishing_sales = fishing_sales.drop_vars("climatology")
fish_sales_relevant_species = fish_sales_relevant_species.drop_vars("climatology")
mari = mari.drop_vars("climatology")

# %%
sources = DataTree.from_dict(
    {
        "river/discharge": river,
        "river/discharge_v5": river_v5,
        "river/plastic": plastic,
        "fishing/capture": fishing,
        "fishing/capture/region": fishing_region,
        "fishing/first_sales": fishing_sales,
        "fishing/first_sales/relevant_species": fish_sales_relevant_species,
        "mariculture": mari,
    }
)
sources.name = "litter_sources"

# %%
# Store the results
# =============================================================================
sources.to_zarr("data/litter_sources.zarr")

# %%
nc = 2
src = "aquaculture"
pdata = sources[src].sel(cluster=nc).sel(production="total").stack(point=("lat", "lon"))
s = pdata.farm_density * 5e1
c = pdata.seasonal_potential
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(lw=0.5, color=".3")
plt.scatter(
    pdata.lon,
    pdata.lat,
    s=s,
    c=c,
    vmin=0,
    vmax=1,
    cmap="Reds",
    ec="None",
    lw=0.3,
    alpha=0.3,
    transform=ccrs.PlateCarree(),
)


# %%


# %%
# Cluster of fishing data
# =============================================================================
threshold_potential = 0.5
min_cluster_size = 10

cluster_result = []

for cluster in [1, 2]:
    # Select data with positive seasonal potential
    is_valid = fishing.seasonal_potential > threshold_potential
    cl_data = fishing.sales.where(is_valid)

    # Compute CV and correlation for each cluster
    cl_data_cv = cl_data.sum("species").std("month") / cl_data.sum("species").mean(
        "month"
    )
    cl_data_cv.name = "coefficient_of_variation"
    cl_data_seas = monthly2seasonal(cl_data.sum("species").roll(month=1), "month")
    cl_data_corr = xr.corr(
        beach_litter.sel(cluster=cluster), cl_data_seas, dim="season"
    )
    cl_data_corr.name = "correlation"
    cl_data_seasonal_potential = cl_data_cv * cl_data_corr
    cl_data_seasonal_potential.name = "seasonal_potential"

    cl_data = cl_data.sum("month")

    # Select litter cluster of interest
    cl_data = cl_data.sel(cluster=cluster)

    total_volume_per_market = cl_data.sum("species")

    # Relative share of sold species
    cl_data = cl_data / total_volume_per_market
    cl_data = cl_data.dropna("location", **{"how": "all"})
    cl_data = cl_data.fillna(0)
    total_volume_per_market = total_volume_per_market.sel(location=cl_data.location)

    # Cluster the data
    cl = HDBSCAN(min_cluster_size=min_cluster_size, store_centers="medoid")
    cl.fit(cl_data.values)
    n_clusters = np.unique(cl.labels_).size - 1
    labels = xr.DataArray(
        cl.labels_,
        dims=["location"],
        coords=dict(location=cl_data.location),
        name="label",
        attrs=dict(
            description="HDBSCAN label for each location. -1 is noise.",
            min_cluster_size=min_cluster_size,
            threshold_potential=threshold_potential,
            n_clusters=n_clusters,
        ),
    )

    centroids = []
    for label in range(n_clusters):
        ok = cl_data.where(labels == label, drop=True)
        wghts = total_volume_per_market.where(labels == label, drop=True)
        # ok = ok.weighted(wghts).mean("location")
        ok = ok.mean("location")
        ok = ok / ok.sum("species")
        ok = ok.expand_dims({"fish_cluster": [label]})
        centroids.append(ok)
    centroids = xr.concat(centroids, dim="fish_cluster")
    centroids.name = "share_of_sales"
    centroids.attrs["description"] = (
        "Centroids of each fish cluster. Centroid computation is based on weighted samples taking into account the total sales per location."
    )

    medoids = xr.DataArray(
        cl.medoids_,
        dims=("fish_cluster", "species"),
        coords={
            "species": cl_data.species,
            "fish_cluster": pd.Index(range(n_clusters), name="fish_cluster"),
        },
        name="medoids",
    )
    ds = xr.Dataset(
        {
            "input_data": cl_data,
            "market_size": total_volume_per_market,
            "medoids": medoids,
            "centroids": centroids,
            "labels": labels,
        }
    )
    ds = ds.reindex_like(sales.location)
    ds.update({"lon": sales.lon, "lat": sales.lat, "country": sales.country})
    ds = ds.expand_dims({"cluster": [cluster]})
    ds["cv"] = cl_data_cv
    ds["correlation"] = cl_data_corr
    ds["seasonal_potential"] = cl_data_seasonal_potential
    cluster_result.append(ds)

cluster_result = xr.concat(cluster_result, "cluster")
cluster_result.attrs["description"] = "HDBSCAN cluster results for fishing data"

dt = DataTree(cluster_result, parent=sources["fishing"], name="clustering")

# Plot the results
# =============================================================================
ds = dt.sel(cluster=2).to_dataset()
df = ds[["market_size", "labels"]].to_dataframe()
# replace all labels that are -1 with np.nan
df["labels"] = df["labels"].replace(-1, np.nan)
df["size"] = df["market_size"] ** (0.5)
fig = plt.figure(figsize=(7.2, 9), dpi=300)
sns.scatterplot(
    data=df,
    x="lon",
    y="lat",
    size="size",
    sizes=(10, 1000),
    hue="labels",
    palette="tab20",
)

# print the species with the highest number for each cluster (centroids)
for cluster in ds.fish_cluster.values:
    centroid = ds.centroids.sel(fish_cluster=cluster).squeeze()
    species = centroid.idxmax("species")
    print(f"Cluster {cluster}: {species.values}")

# %%
