# %%
# GOAL: Analyze the seasonality of first sale data of aquaculture products in Europe

from functools import partial

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry
import xarray as xr
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim, OpenCage
from tqdm import tqdm


def monthly2seasonal(da, dim):
    """Convert any DataArray with monthly time dimension to seasonal mean.

    Seasonal mena is computed as follows:
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
    # Create a dictionary with the mean of each season
    seasonal_mean = {}
    for season, months in seasons.items():
        seasonal_mean[season] = da.sel({dim: months}).mean(dim)
    # Concatenate the dictionary into a new DataArray
    seasonal_mean = xr.concat(
        seasonal_mean.values(), dim=pd.Index(seasons.keys(), name="season")
    )
    return seasonal_mean


# %%
base_path = "data/first_sales/"

columns = [
    "Year",
    "Month_of_Year",
    "Country",
    "Location",
    "Commodity_Group",
    "Main_commercial_species",
    "Volume(Kg)",
]
sales = []
YEARS = np.arange(2013, 2023)
for year in YEARS:
    df = pd.read_csv(
        base_path + f"{year}_first_sale_by_ERS_code.csv",
        sep=";",
        usecols=columns,
    )
    sales.append(df)

# Data from Germany is missing, so we add it manually
de = pd.read_csv(
    base_path + "DE_first_sale_by_ERS_code.csv",
    sep=";",
    usecols=columns,
)
sales.append(de)


df = pd.concat(sales)
df.columns = [
    "year",
    "month",
    "country",
    "location",
    "commodity_group",
    "species",
    "volume",
]

use_countries = [
    "France",
    "Denmark",
    "United Kingdom",
    "Sweden",
    "Germany",
    "Portugal",
    # "Iceland",
    "Belgium",
    "Netherlands",
    "Norway",
    "Spain",
    "Ireland",
]
df = df[df["country"].isin(use_countries)]

# %%
df2 = df.groupby(["year", "month", "location", "species"]).agg(
    {
        "country": "first",
        "commodity_group": "first",
        "volume": "sum",
    }
)
df2 = df2.reset_index()

df2["time"] = pd.to_datetime(df2[["year", "month"]].assign(day=1))


# %%
# Convert country names to ISO codes
# -----------------------------------------------------------------------------


# Create a dictionary to map country names to ISO codes
country_codes = {}
for country in use_countries:
    try:
        iso_code = pycountry.countries.search_fuzzy(country)[0].alpha_2
        country_codes[country] = iso_code
    except LookupError:
        # Handle countries that are not found in the pycountry database
        country_codes[country] = "Unknown"


# %%
# Geocode locations
# -----------------------------------------------------------------------------
def get_geolocation(locations, country_codes, service="OSM", api_key=None):
    """Get the geolocation of each location in the dataframe.

    The geolocation is obtained using the Nominatim geocoder from the geopy library.

    Parameters
    ----------
    locations : pd.DataFrame
        A dataframe with a column "location" containing the names of the locations.
    country_codes : dict
        A dictionary with country names as keys and ISO codes as values.
    service : str, optional
        The geocoding service to use. Can be "OSM" or "OpenCage". The default is "OSM".
    api_key : str, optional
        The API key for the OpenCage geocoder. The default is None.

    Returns
    -------
    pd.DataFrame
        A dataframe with the columns "location", "pygeo_loc", "lon", and "lat".

    """
    locations = locations.sort_values("location", ignore_index=True)

    params = {}
    tqdm.pandas()
    # Create a geolocator object
    match service:
        case "OSM":
            geolocator = Nominatim(user_agent="my_app")
            params.update(country_codes=list(country_codes.values()))
        case "OpenCage":
            if api_key is None:
                try:
                    with open("./utils/.api_key_geocache.txt", "r") as file:
                        api_key = file.read().strip()
                except FileNotFoundError:
                    raise FileNotFoundError(
                        "The API key for the OpenCage geocoder is missing."
                    )
            geolocator = OpenCage(
                api_key=api_key,
                user_agent="my_app",
            )
        case _:
            raise ValueError("Invalid value for the 'service' parameter.")

    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5, max_retries=5)

    locations["pygeo_loc"] = locations["location"].progress_apply(
        partial(geocode, **params)
    )
    locations["lon"] = locations["pygeo_loc"].apply(
        lambda loc: loc.longitude if loc else np.nan
    )
    locations["lat"] = locations["pygeo_loc"].apply(
        lambda loc: loc.latitude if loc else np.nan
    )

    locations.set_index("location", inplace=True)

    return locations


# %%

try:  # Load geolocation from file
    locations = pd.read_csv("data/first_sales/locations.csv", index_col="location")
except FileNotFoundError:
    # Get the geolocation of each location
    locations = pd.DataFrame(df2["location"].unique(), columns=["location"])
    locations = get_geolocation(locations, country_codes, service="OSM")
    # For locations that were not found, use the OpenCage geocoder
    missing_locs = locations.loc[locations.lon.isnull()].reset_index()
    missing_locs = get_geolocation(missing_locs, country_codes, service="OpenCage")
    locations.update(missing_locs)
    locations.to_csv("data/first_sales/locations.csv")

# NOTE: All data from Norway is allocated to location "Unspecified". We define
# the location of "Unspecified" as longitude 8.75 and latitude 59.8.

locations.loc["Unspecified", "lon"] = 8.75
locations.loc["Unspecified", "lat"] = 59.8

df2["lon"] = df2["location"].map(locations["lon"]).values
df2["lat"] = df2["location"].map(locations["lat"]).values

# %%
da = df2.set_index(["time", "location", "species"]).to_xarray()

coords_location = (
    df2[["location", "country", "lon", "lat"]]
    .drop_duplicates(subset="location")
    .set_index("location")
    .sort_index()
)
coords_species = (
    df2[["species", "commodity_group"]]
    .drop_duplicates(subset="species")
    .set_index("species")
    .sort_index()
)
da = da.assign_coords(
    lon=("location", coords_location["lon"]),
    lat=("location", coords_location["lat"]),
    country=("location", coords_location["country"]),
    commodity_group=("species", coords_species["commodity_group"]),
)
da = da["volume"]
da = da.assign_attrs(units="kg")


# %%
# Prepare data for analysis
# =============================================================================
# Europe only
is_in_europe = (da.lon > -15) & (da.lon < 35) & (da.lat > 35) & (da.lat < 65)

# Remove sales from the Mediterranean Sea
is_mediterranean = (
    ((da.lon > -5) & (da.lat < 41))
    | ((da.lon > 0) & (da.lat < 42))
    | ((da.lon > 2) & (da.lat < 45))
)
is_in_europe = is_in_europe & (~is_mediterranean)


sales = da.sel(location=is_in_europe)
sales = sales.fillna(0)
# Annual sales [kg/year] per location, species, year
sales_annual = sales.groupby("time.year").sum()
# Mean annual sales [kg/year] per location, species
sales_mean_annual = sales_annual.mean("year")
# Global sales [kg/year] per species
total_sales_species = sales_mean_annual.sum("location")

# Remove species with small overall sold volume
wghts_df = total_sales_species.to_dataframe().sort_values(by="volume", ascending=False)
wghts_df_rel_cum = wghts_df["volume"].cumsum() / wghts_df["volume"].sum()
n_species = (wghts_df_rel_cum <= 0.999).sum() + 1
important_species = wghts_df.iloc[:n_species].index.values

# sales = sales.sel(species=important_species)
print("Number of species:", sales.species.size)

# Compute monthly mean sales per location, species
sales = sales.groupby("time.month").mean()


# %%
# Store data
# =============================================================================
sales.to_netcdf("data/economic/first_sales/first_sales_clean.nc")


# %%
# Figures spatial patterns of sales
# =============================================================================
# Plot the spatial distribution of the mean annual sales per species


for sp in tqdm(sales.species.values):
    ok = sales.sel(species=sp)
    c = ok.std("month") / ok.mean("month")
    s = ok.sum("month")
    s = s / s.max() * 500
    ax = plt.axes(projection=ccrs.TransverseMercator())
    ax.coastlines(lw=0.5)
    ax.add_feature(cfeature.RIVERS.with_scale("50m"), color=".4", lw=0.3)
    ax.scatter(
        ok.lon,
        ok.lat,
        s=s,
        c=c,
        vmin=0.0,
        vmax=1.0,
        ec="k",
        lw=0.3,
        transform=ccrs.PlateCarree(),
    )
    plt.title(sp)
    plt.tight_layout()
    plt.savefig(f"figs/first_sales/{sp}.png", dpi=300)
    plt.close()
