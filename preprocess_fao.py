# %%
import os

import numpy as np
import pandas as pd
import xarray as xr
from datatree import open_datatree


def read_data(source):
    path = os.path.join(root, "{:}_2023.1.1/{:}_Quantity.csv")
    return pd.read_csv(path.format(source, source))


# Load data
# =============================================================================
root = "data/economic/production/"

# Conversion Table Species FAO <-> EUMOFA
path_table = os.path.join(root, "conversion_table_species_fao_to_eumofa.csv")
species_table_fao_eumofa = pd.read_csv(path_table, index_col=[0, 1])

# FAO species
code_species = pd.read_csv(root + "Capture_2023.1.1/CL_FI_SPECIES_GROUPS.csv")
# FAO country codes
code_countries = pd.read_csv(root + "Capture_2023.1.1/CL_FI_COUNTRY_GROUPS.csv")
code_countries.replace(
    {"Name_En": {"Netherlands (Kingdom of the)": "Netherlands"}}, inplace=True
)

# FAO Production
fish = read_data("Capture")
aqua = read_data("Aquaculture")

# Filter to only include countries for which beach litter data exists
ospar = open_datatree("data/beach_litter/ospar/preprocessed.zarr", engine="zarr")
countries_beach_litter = np.unique(ospar["preprocessed/absolute"].country).tolist()

code_countries = code_countries.loc[
    code_countries["Name_En"].isin(countries_beach_litter)
]

fish = fish.loc[fish["COUNTRY.UN_CODE"].isin(code_countries["UN_Code"])]
aqua = aqua.loc[aqua["COUNTRY.UN_CODE"].isin(code_countries["UN_Code"])]


# Filter to only include total live weight
fish = fish.loc[fish["MEASURE"] == "Q_tlw"]
aqua = aqua.loc[aqua["MEASURE"] == "Q_tlw"]

# Add FAO species names
fish["species_fao"] = fish["SPECIES.ALPHA_3_CODE"].map(
    code_species.set_index("3A_Code")["Name_En"]
)
aqua["species_fao"] = aqua["SPECIES.ALPHA_3_CODE"].map(
    code_species.set_index("3A_Code")["Name_En"]
)

# Add country names
fish["country"] = fish["COUNTRY.UN_CODE"].map(
    code_countries.set_index("UN_Code")["Name_En"]
)
aqua["country"] = aqua["COUNTRY.UN_CODE"].map(
    code_countries.set_index("UN_Code")["Name_En"]
)

# %%
# search for a species in fish
# keyword = "Tilapia"
# has_keyword = fish["species_fao"].str.contains(keyword, case=False).astype(bool)
# found = fish.loc[has_keyword].species_fao.dropna().unique()
# found

# %%

# Compute total live weight per species, country and year; convert to xarray
wild = fish.groupby(["species_fao", "country", "PERIOD"]).sum().VALUE.to_xarray()
aqua = aqua.groupby(["species_fao", "country", "PERIOD"]).sum().VALUE.to_xarray()
wild.name = "weight"
aqua.name = "weight"
aqua = aqua.rename({"PERIOD": "year"})
wild = wild.rename({"PERIOD": "year"})

# %%
# convert temp such that the "species_fao" are converted to the species in the eumofa dataset
# use the fao2eumofa dictionary


def convert_dataset_species(ds: xr.DataArray):
    """Convert the species in the dataset to the eumofa species.

    In practice, we want to sum up the values of all FAO species that belong to the same eumofa species.

    """
    convs = []
    for idx, mapping_fao in species_table_fao_eumofa.iterrows():
        species_eumofa, com_group = idx
        mapped_fao_species = mapping_fao[mapping_fao == 1].index.values
        is_valid_fao_species = ds.species_fao.isin(mapped_fao_species).values
        ds_new = ds.sel(species_fao=is_valid_fao_species).sum(dim="species_fao")

        ds_new = ds_new.expand_dims({"species": [species_eumofa]})
        ds_new = ds_new.assign_coords(commodity_group=("species", [com_group]))
        convs.append(ds_new)
    return xr.concat(convs, dim="species")


aqua = convert_dataset_species(aqua)
wild = convert_dataset_species(wild)

fish = xr.concat([aqua, wild], dim="production_type")
fish = fish.assign_coords(production_type=["aqua", "wild"])
fish.name = "live_weight"
fish.attrs["units"] = "tonnes"

# %%
# lead perprocessed OSPAR data
n_surveys = ospar.preprocessed["absolute/Plastic"].notnull().sum(("season"))
n_surveys = n_surveys.groupby("country").sum()

total_weights = fish.weighted(n_surveys).mean(("year"))
total_weights.name = "live_weight"
total_weights.attrs["units"] = "tonnes"
total_weights.attrs["description"] = (
    "Weighted mean live weight of fish caught in wild and aquaculture, using number of OSPAR surveys per year as weights."
)

share_production_type = total_weights / total_weights.sum("production_type")
share_production_type.name = "share_production_type"
share_production_type.attrs["description"] = (
    "Share of total production of fish caught in wild and aquaculture by production type."
    " Total absolute production is based on a OSPAR-weighted mean."
)

# Missing values represent no production for this species/country, neither wild nor aquaculture
share_production_type.attrs["missing_values"] = "no production for this species/country"


ds = xr.Dataset(
    {
        "live_weight": fish,
        "average_live_weight": total_weights,
        "relative_share": share_production_type,
    }
)


# %%
# Save the data
# -----------------------------------------------------------------------------
# Add commodity group as additional coordinate
ds.to_netcdf("data/economic/production/fao_fish_production.nc")


# %%
