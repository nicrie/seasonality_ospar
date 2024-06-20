import os

import datatree as dt
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import trange

from utils.statistics import weighted_percentile

# %%
# Load data
# =============================================================================
SEASONS = ["DJF", "MAM", "JJA", "SON"]
VARIABLE = "absolute/Plastic"
YEAR = 2001

base_path = f"data/gpr/{VARIABLE}/{YEAR}/"
save_to = f"data/clustering/kmeans/{VARIABLE}/{YEAR}/"

if not os.path.exists(save_to):
    os.makedirs(save_to)

# Statistical evaluation
# -----------------------------------------------------------------------------
results = xr.open_dataset(base_path + "effect_size_seasons.nc")
confidence = results.max_hdi.max("combination")

# GPR data
# -----------------------------------------------------------------------------
model = dt.open_datatree(base_path + "posterior_predictive.zarr", engine="zarr")
litter_m = model["posterior_predictive"][VARIABLE.split("/")[1]]

clus_dat = litter_m / litter_m.sum("season")
clus_dat = clus_dat.isel(n=slice(0, 4000, 1))


# Example data
data = clus_dat.copy().transpose("beach_id", "season", "n")

# Compute silhouette scores for different numbers of clusters to find the optimal number
silhouette_scores = []

for n_clusters in trange(2, 50):
    kmeans = KMeans(n_clusters=n_clusters, n_init=100)
    kmeans.fit(data.mean("n"), sample_weight=confidence)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(data.mean("n"), labels))

# Plot the silhouette scores
plt.figure()
plt.plot(range(2, 50), silhouette_scores, marker=".")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.show()

# %%


# Parameters
n_clusters = 10
n_bootstrap_samples = 1000


# Store cluster assignments for each bootstrap sample
bootstrap_clusters = np.zeros((n_bootstrap_samples, len(data)), dtype=int)

# Initialize centroids storage
centroids_list = []

for i in trange(n_bootstrap_samples):
    # Generate a bootstrap sample
    boot_sample = data.isel(n=i)

    # Perform MiniBatchKMeans clustering with sample weights
    kmeans = KMeans(n_clusters=n_clusters, n_init=50)
    kmeans.fit(boot_sample, sample_weight=confidence)

    # Store the centroids
    centroids_list.append(kmeans.cluster_centers_)

    # Assign clusters to original data points based on nearest centroid
    labels = kmeans.labels_
    bootstrap_clusters[i] = labels


# Function to compute cost matrix
def compute_cost_matrix(centroids1, centroids2):
    cost_matrix = np.zeros((len(centroids1), len(centroids2)))
    for i, c1 in enumerate(centroids1):
        for j, c2 in enumerate(centroids2):
            cost_matrix[i, j] = np.linalg.norm(c1 - c2)
    return cost_matrix


# Align clusters using the Hungarian algorithm
aligned_clusters = np.zeros_like(bootstrap_clusters)

for i in trange(n_bootstrap_samples):
    if i == 0:
        aligned_clusters[i] = bootstrap_clusters[i]
    else:
        cost_matrix = compute_cost_matrix(centroids_list[0], centroids_list[i])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        mapping = dict(zip(col_ind, row_ind))
        aligned_clusters[i] = np.array(
            [mapping[label] for label in bootstrap_clusters[i]]
        )

# Calculate the probability of each sample belonging to each cluster
cluster_probabilities = np.zeros((len(data), n_clusters))

for i in trange(len(data)):
    for k in range(n_clusters):
        cluster_probabilities[i, k] = (
            np.sum(aligned_clusters[:, i] == k) / n_bootstrap_samples
        )


print("Number of cluster members:", np.round(cluster_probabilities.sum(0), 0))

# %%
probis = xr.DataArray(
    cluster_probabilities,
    dims=("beach_id", "cluster"),
    coords=dict(
        beach_id=clus_dat.coords["beach_id"],
        cluster=np.arange(1, n_clusters + 1, dtype=int),
    ),
)
weighted_cluster_mean = (litter_m.median("n") * probis).sum(
    "beach_id"
) / probis.sum()  # %%
weighted_cluster_mean.T.plot()

# %%
c = cluster_probabilities[:, 4 - 1]
plt.scatter(
    litter_m.lon,
    litter_m.lat,
    s=5 + c * 500,
    c=c,
    ec="k",
    cmap="Reds",
    vmin=0,
    vmax=0.8,
)
# %%

percis, ess = weighted_percentile(
    litter_m,
    probis.broadcast_like(litter_m.mean("n")),
    [0.25, 0.5, 0.75],
    dim=("beach_id", "n"),
)
# %%
