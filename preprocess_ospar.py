# %%
# Imports
# =============================================================================
import json
import os

import datatree as dt
import numpy as np
import pandas as pd
import scipy.stats as st
import xarray as xr


# %%
# Utilities #%%
# =============================================================================
def convert_coords(coord):
    try:
        # Extract orientation (N/S/E/W)
        orientation = coord[-1]
        coord = coord[:-1]
        # Replace special characters to allow splitting
        coord = coord.replace("°", "_").replace("'", "_").replace('"', "_")
        coord = coord.replace("′", "_").replace("″", "_").replace("”", "_")
        coord = coord.replace("’’", "_").replace(",", "_")
        coord = coord.split("_")
        # Extract sign for South/West orientation
        sign = 1
        if np.isin(orientation, ["S", "W"]) and (int(coord[0]) >= 0):
            sign *= -1

        # Extract degrees, minutes, seconds
        # Remove empty entries
        coord = [float(c) for c in coord if len(c) > 0]
        # NOTE: for some reason, some entries have the seconds reported as e.g. 5''0.49 instead of 5.49''
        # We therefore add the 3rd and 4th entry when necessary
        if len(coord) > 3:
            coord[2] += coord[3]
        coord = coord[:3]
        # Apply sign; if the degrees are 0, we add epsilon to the sign
        # to avoid multiplying by 0
        if coord[0] == 0:
            coord[0] += 1e-6
        coord[0] *= sign
        return coord
    except AttributeError:
        return [np.nan] * 3
    except ValueError:
        return [np.nan] * 3
    except IndexError:
        return [np.nan] * 3


def deg2dec(geocoords):
    def get_sign(deg):
        if deg < 0:
            return -1
        else:
            return 1

    try:
        deg, min, sec = geocoords
        sign = get_sign(deg)
        return sign * (abs(deg) + (min + (sec / 60)) / 60)
    except TypeError:
        return np.nan
    except ValueError:
        if len(geocoords) == 2:
            deg, min = geocoords
            sign = get_sign(deg)
            return sign * (abs(deg) + min / 60)
        else:
            raise ValueError("Invalid number of coordinates")


def assign_season(month):
    if month in [12, 1, 2]:
        return "DJF"
    elif month in [3, 4, 5]:
        return "MAM"
    elif month in [6, 7, 8]:
        return "JJA"
    elif month in [9, 10, 11]:
        return "SON"
    else:
        raise ValueError("not a valid month")


def _np_theil_sen_regression(y, x=None, n_samples_min=10):
    is_valid = ~np.isnan(y)
    y = y[is_valid]
    if x is not None:
        x = x[is_valid]
        # noramlize between 0 and 1
        xmin = 2001
        xmax = 2020
        x = (x - xmin) / (xmax - xmin)

    # Check if there are enough samples
    if y.shape[0] < n_samples_min:
        return np.nan, np.nan, np.nan, np.nan

    try:
        slope, intercept, low_slope, high_slope = st.theilslopes(
            y, x, 0.95, method="separate"
        )
    except TypeError:
        slope, intercept, low_slope, high_slope = np.nan, np.nan, np.nan, np.nan
    return slope, intercept, low_slope, high_slope


def theil_sen_regression(X, dim):
    slope, intercept, low_slope, high_slope = xr.apply_ufunc(
        _np_theil_sen_regression,
        X,
        X[dim].broadcast_like(X),
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], [], [], []],
        exclude_dims=set([dim]),
        vectorize=True,
    )
    is_significant = np.sign(low_slope) == np.sign(high_slope)
    return is_significant, intercept, slope, low_slope, high_slope


# %%
# Load OSPAR
# =============================================================================
base_path = "data/beach_litter/ospar/single"
filepaths = [f for f in os.listdir(base_path) if f.endswith(".csv")]
filepaths = [os.path.join(base_path, fp) for fp in filepaths]

# OSPAR data
ospar = pd.concat(map(pd.read_csv, filepaths))
ospar = ospar.set_index("Survey ID").sort_index()

categories_ospar = ospar.loc[
    :, "Plastic: Yokes [1]":"Survey: Remarks [999]"
].columns.values

# Convert survey date to datetime
ospar["date"] = pd.to_datetime(ospar["Survey date"], format="%d/%m/%Y")
ospar.drop("Survey date", axis=1, inplace=True)
ospar["year"] = ospar["date"].dt.year
ospar["month"] = ospar["date"].dt.month
ospar["season"] = ospar["month"].apply(lambda x: assign_season(x))

# Convert country name to match with other data sets
ospar["Country"] = ospar["Country"].str.replace(
    "Denmark (incl. the Faeroe Islands)", "Denmark", regex=False
)


# %%
# Convert lon and lat for each survey #%%
# -----------------------------------------------------------------------------
geocoords100 = ospar.loc[:, "100m Start N/S":"100m End E/W"]
geocoords100 = geocoords100.astype(str)
geocoords100 = geocoords100.map(convert_coords)
geocoords100 = geocoords100.map(deg2dec).round(6)

geocoords1000 = ospar.loc[:, "1km Start N/S":"1km End E/W"]
geocoords1000 = geocoords1000.astype(str)
geocoords1000 = geocoords1000.map(convert_coords)
geocoords1000 = geocoords1000.map(deg2dec).round(6)

lon100_cats = ["100m Start E/W", "100m End E/W"]
lat100_cats = ["100m Start N/S", "100m End N/S"]
lon100_dec = geocoords100.loc[:, lon100_cats].mean(axis=1)
lat100_dec = geocoords100.loc[:, lat100_cats].mean(axis=1)

lon1000_cats = ["1km Start E/W", "1km End E/W"]
lat1000_cats = ["1km Start N/S", "1km End N/S"]
lon1000_dec = geocoords1000.loc[:, lon1000_cats].mean(axis=1)
lat1000_dec = geocoords1000.loc[:, lat1000_cats].mean(axis=1)

# If 100m coordinates are missing, use 1km coordinates
ospar["lon"] = lon100_dec.fillna(lon1000_dec)
ospar["lat"] = lat100_dec.fillna(lat1000_dec)

# Save raw OSPAR data
n_entries_raw = len(ospar)
n_entries = n_entries_raw
print("OSPAR data set contains {} entries".format(n_entries_raw))


# %%
# Add geocoordinates of some beaches which are missing in OSPAR data set
additional_beach_latlons = [
    ["PT002", 38.70, -9.45],
    ["PT003", 38.68, -9.34],
    ["PT006", 37.16, -7.52],
]
additional_beach_latlons = pd.DataFrame(
    additional_beach_latlons, columns=["Beach ID", "lat", "lon"]
)

# Fill the gaps in the OSPAR data set
for idx, (beach_id, lat, lon) in additional_beach_latlons.iterrows():
    ospar.loc[ospar["Beach ID"] == beach_id, ["lon", "lat"]] = [lon, lat]


# %%
# Manually inspect survey remarks and remove surveys that are flagged
# -----------------------------------------------------------------------------
remarks = ospar["Survey: Remarks [999]"].dropna()
try:
    remarks.to_csv("data/beach_litter/ospar/remarks.csv", mode="x")
except FileExistsError:
    pass

# %%
# Remove surveys that were flagged based on remarks
# -----------------------------------------------------------------------------
remarks = pd.read_csv("data/beach_litter/ospar/remarks.csv")
remarks = remarks.set_index("Survey ID")["Treatment"]
remove_surveys = remarks.loc[remarks == "remove"].index
ospar = ospar.drop(remove_surveys)

remove_survey_beaches = remarks.loc[remarks == "remove beach"].index
drop_beaches = ospar.loc[remove_survey_beaches, "Beach ID"].unique()
ospar = ospar.loc[~ospar["Beach ID"].isin(drop_beaches)]

n_entries_removed = n_entries - len(ospar)
n_entries = len(ospar)
print("Removed {} entries based on remarks".format(n_entries_removed))
print("OSPAR data set contains {} entries".format(n_entries))


# %%
# Remove entries with missing coordinates
# -----------------------------------------------------------------------------
ospar = ospar.dropna(subset=["lon", "lat"])
n_entries_removed = n_entries - len(ospar)
n_entries = len(ospar)
print("Removed {} entries with missing coordinates".format(n_entries_removed))
print("OSPAR data set contains {} entries".format(n_entries))

# %%
# Remove invalid measurements
# -----------------------------------------------------------------------------
# Some surveys have measurements which deviate from the predetermined 100 metres
# Surveys that indicated the deviation in meters are kept. The rest are removed

# The beach PT018 was found to have measured 70m instead of 100m. Beach length was indicated as a sentence
search_str = "The beach lenght is 70 meters"
survey_id_pt018 = ospar.loc[ospar["Divert specify"] == search_str].index
ospar.loc[survey_id_pt018, "Divert specify"] = "70m"

# The beach SE005 was found to have measured 135m instead of 100m.
# However, the error was computed to be about 4%, so we keep this beach in the data set
survey_id_se005 = ospar.loc[ospar["Beach ID"] == "SE005"].index
ospar.loc[survey_id_se005, "Divert from the predetermined 100 metres?"] = "No"


# Convert beaches that deviated from the predetermined 100 metres to to 100m (if they provided the deviation in meters)
df = ospar.loc[ospar["Divert from the predetermined 100 metres?"] == "Yes"]
df_filtered = df[df["Divert specify"].str.len() <= 12]["Divert specify"]
length_cr_fac = 100.0 / df_filtered.str.extract(r"(\d+)").astype(float)
vals_cr = (
    ospar.loc[
        length_cr_fac.index,
        "Plastic: Yokes [1]":"Survey: Old_cloth_rope [210]",
    ]
    * length_cr_fac.values
)
vals_cr = vals_cr.round(0).astype(int)
ospar.loc[length_cr_fac.index, vals_cr.columns] = vals_cr
ospar.loc[length_cr_fac.index, "Divert from the predetermined 100 metres?"] = "No"

# Remove entries that deviated from the predetermined 100 metres & which could not be corrected
ospar = ospar[ospar["Divert from the predetermined 100 metres?"] == "No"]
n_entries_removed = n_entries - len(ospar)
n_entries = len(ospar)
print(
    "Removed {} entries which deviated from predetermined 100 metres".format(
        n_entries_removed
    )
)
print("OSPAR data set contains {} entries".format(n_entries))


# %%
# Take only continental Europe
# specifically, exclude islands - the Azores and Greenland
# -----------------------------------------------------------------------------
# Focus on continental Europe + UK
# ospar = ospar[ospar.lon > -15]
# ospar = ospar[ospar.lat < 65]


n_entries_removed = n_entries - len(ospar)
n_entries = len(ospar)
print(
    "Removed {} entries which are not in continental Europe".format(n_entries_removed)
)
print("OSPAR data set contains {} entries".format(n_entries))


da = ospar.groupby(["year", "month"]).size().to_xarray()

# %%
# Aggregate OSPAR categories to META (EUROPE) categories
# =============================================================================
ospar_meta = ospar.copy(deep=True)
ospar_meta.drop(categories_ospar, axis=1, inplace=True)

# Dictionary to convert from OSPAR to Científicos de la Basura (CB) categories
path_dictionary = "data/beach_litter/ospar/OSPAR_meta_litter_categories.xlsx"
ospar2meta = pd.read_excel(path_dictionary)

ospar2meta = ospar2meta.drop(["CB", "Comments"], axis=1, inplace=False)
ospar2meta = ospar2meta.set_index("OSPAR")

for cat in ospar2meta.columns:
    cats = ospar2meta.loc[ospar2meta[cat] == 1].index.values
    ospar_meta[cat] = ospar[cats].sum(axis=1)
# %%
# Aggregate OSPAR categories to CB categories
# =============================================================================
ospar_cb = ospar.copy(deep=True)
ospar_cb.drop(categories_ospar, axis=1, inplace=True)

# Dictionary to convert from OSPAR to Científicos de la Basura (CB) categories
path_dictionary = "data/beach_litter/ospar/OSPAR&CB_litter_categories.xlsx"
ospar2chile = pd.read_excel(path_dictionary, names=["OSPAR", "CB", "Comments"])
categories_cb = ospar2chile["CB"].dropna().unique()
# Do not use category "Other" which is inconsistent in CB
categories_cb = [cat for cat in categories_cb if cat not in ["Other"]]

for cat in categories_cb:
    cats = ospar2chile["OSPAR"][ospar2chile["CB"] == cat]
    ospar_cb[cat] = ospar[cats].sum(axis=1)


# %%
# Convert to xarray
# =============================================================================
da_ospar_meta = ospar_meta.reset_index()
da_ospar_meta = da_ospar_meta[
    [
        "Beach ID",
        "Beach name",
        "Country",
        "Region",
        "date",
        "year",
        "month",
        "season",
        "lon",
        "lat",
        "TA",
        "SUP",
        "SEA",
        "Cigarette butts",
        "Plastic",
        "Rubber",
        "Cloth",
        "Paper",
        "Wood",
        "Metal",
        "Glass",
        "Pottery",
        "San",
        "Med",
        "Faeces",
        "MICRO",
        "Aquaculture",
        "String",
        "Non-Aqua",
    ]
]
da_ospar_meta = da_ospar_meta.rename(
    {
        "Beach ID": "beach_id",
        "Beach name": "beach_name",
        "Country": "country",
        "Region": "region",
        "Cigarette butts": "CB",
    },
    axis=1,
)
da_ospar_meta = da_ospar_meta.set_index(["date", "beach_id"])
da_ospar_meta = da_ospar_meta.to_xarray()
da_ospar_meta["beach_name"] = da_ospar_meta["beach_name"].max("date").astype(str)
da_ospar_meta["country"] = da_ospar_meta["country"].max("date").astype(str)
da_ospar_meta["region"] = da_ospar_meta["region"].max("date").astype(str)
da_ospar_meta["year"] = da_ospar_meta["year"].max("beach_id").astype(int)
da_ospar_meta["month"] = da_ospar_meta["month"].max("beach_id").astype(int)
da_ospar_meta["season"] = da_ospar_meta["season"].max("beach_id").astype(str)
da_ospar_meta["lon"] = da_ospar_meta["lon"].mean("date").astype(float)
da_ospar_meta["lat"] = da_ospar_meta["lat"].mean("date").astype(float)
da_ospar_meta = da_ospar_meta.set_coords(
    ["beach_name", "country", "region", "year", "month", "season", "lon", "lat"]
)

da_ospar_meta["LOCAL"] = (
    da_ospar_meta[["Paper", "Metal", "Glass", "CB"]]
    .to_array()
    .sum("variable", skipna=False)
)

# %%
# How many measurement per beach, season and year?
# -----------------------------------------------------------------------------
n_surveys = ospar_meta.groupby(["year", "season", "Beach ID"]).size()
n, frequency = np.unique(n_surveys, return_counts=True)
print("Number of surveys per beach, season and year")
print(n)
print("Frequency of number of surveys")
print(frequency)

# %%
# Convert data to 1 survey per season, year and beach
# -----------------------------------------------------------------------------
ospar_df = da_ospar_meta.dropna("beach_id", **{"how": "all"})
ospar_df = ospar_df.to_dataframe().reset_index()
ospar_df = ospar_df.dropna(subset=["TA"])
idx, beach_ids = pd.factorize(ospar_df["beach_id"].values, sort=True)
ospar_df["beach_idx"] = idx
ospar_df = ospar_df[
    [
        "lon",
        "lat",
        "beach_idx",
        "year",
        "season",
        "TA",
        "Plastic",
        "SEA",
        "SUP",
        "CB",
        "LOCAL",
        "Aquaculture",
        "String",
        "Non-Aqua",
    ]
]
# NOTE: skipna=True is not yet implemented in pandas
# see https://github.com/pandas-dev/pandas/issues/15675
# For now, use workaround using fillna and replace (https://stackoverflow.com/a/62075919)
# ospar_df = ospar_df.groupby(["lon", "lat", "beach_idx", "year", "season"]).mean(skipna=True)
# ospar_df = ospar_df.groupby(["lon", "lat", "beach_idx", "year", "season"]).agg(
#     lambda g: g.max(skipna=True)
# )
ospar_df = ospar_df.groupby(["lon", "lat", "beach_idx", "year", "season"]).max()
da_ospar_meta2 = ospar_df.reset_index(["lon", "lat"]).to_xarray()
da_ospar_meta2 = da_ospar_meta2.assign_coords(
    {
        "lon": ("beach_idx", da_ospar_meta2.lon.max(("season", "year")).values),
        "lat": ("beach_idx", da_ospar_meta2.lat.max(("season", "year")).values),
        "beach_id": ("beach_idx", np.unique(beach_ids)),
    }
)
da_ospar_meta2 = da_ospar_meta2.set_index(beach_idx="beach_id")
da_ospar_meta2 = da_ospar_meta2.rename({"beach_idx": "beach_id"})
da_ospar_meta2 = da_ospar_meta2.sel(season=["DJF", "MAM", "JJA", "SON"])


# Add country coordinate
def convert_beach_id_to_country(ids):
    id2country = {
        "BE": "Belgium",
        "DE": "Germany",
        "DK": "Denmark",
        "ES": "Spain",
        "FO": "Faroe Islands",
        "FR": "France",
        "GR": "Greenland",
        "IM": "Isle of Man",
        "IR": "Ireland",
        "IS": "Iceland",
        "NL": "Netherlands",
        "NO": "Norway",
        "PT": "Portugal",
        "SE": "Sweden",
        "UK": "United Kingdom",
    }
    return [id2country[id[:2]] for id in ids]


beach_ids = da_ospar_meta2.beach_id
country_ids = convert_beach_id_to_country(beach_ids.values)
country_ids = xr.DataArray(country_ids, dims="beach_id", coords={"beach_id": beach_ids})

da_ospar_meta2 = da_ospar_meta2.assign_coords(country=country_ids)


# %%
# Detrend OSPAR data
# -----------------------------------------------------------------------------
X = da_ospar_meta2
X_median = X.median("year")
Xc = X - X_median
has_trend, intercept, slope, slope_l, slope_h = theil_sen_regression(Xc, dim="year")

# Only significant trends
years = da_ospar_meta2.year.astype(float)
years = (years - years.min()) / (years.max() - years.min())
trends = intercept + slope * years
trends = trends.where(has_trend, 0)
intercept = intercept.where(has_trend, 0)

da_ospar_meta2_detrend = Xc - trends + X_median


# %%
# Store data
# -----------------------------------------------------------------------------
categories = dict(
    ospar_meta=list(ospar2meta.columns),
    ospar_meta_special=["TA", "SUP", "SEA", "Cigarette butts"],
    ospar=list(categories_ospar),
    cb=list(categories_cb),
)
with open("utils/categories.json", "w", encoding="utf-8") as f:
    json.dump(categories, f, ensure_ascii=False, indent=4)


# Convert category mapping to xarray
da_ospar2meta = ospar2meta.drop(labels="Plastics (CB)", axis=1).to_xarray()
da_ospar2meta = da_ospar2meta.rename({"OSPAR": "category"})
description = "Mapping of individual to meta  OSPAR beach litter categories based on van Loon et al. (2023)"
da_ospar2meta = da_ospar2meta.assign_attrs({"description": description})

# Add some more meta data
description = "OSPAR beach litter data aggregated into meta categories based on van Loon et al. 2023. "
da_ospar_meta2 = da_ospar_meta2.assign_attrs(
    {
        "description": description,
        "units": "items/100 m beach line",
        "source": "OSPAR",
    }
)
da_ospar_meta2_detrend = da_ospar_meta2_detrend.assign_attrs(
    {
        "description": "Detrended OSPAR beach litter data",
        "units": "items/100 m beach line",
        "source": "OSPAR",
    }
)


ospar_datatree = dt.DataTree.from_dict(
    {
        "/category_mapping": da_ospar2meta,
        "/preprocessed/": da_ospar_meta2,
        "/preprocessed/detrended/": da_ospar_meta2_detrend,
    }
)
ospar_datatree.to_zarr("data/beach_litter/ospar/preprocessed.zarr", mode="w")
