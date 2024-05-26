import numpy as np
import regionmask as rm

# Mask for North-East Atlantic region
boundaries_nea = np.array([[-20, 35], [-6, 35], [15, 60], [15, 65], [-20, 65]])
nea_mask = rm.Regions([boundaries_nea], names=["North East Atlantic"], abbrevs=["NEA"])


# Sub-basins for the North-East Atlantic
regions = {
    "Skagerrak": [(9, 55), (13, 55), (13, 60), (6, 60)],
    "West Iberian Sea": [(-15, 37), (-8, 37), (-8, 44), (-7.5, 45), (-15, 45)],
    "Gulf of Cadiz": [(-10, 35), (-5, 35), (-5, 37), (-10, 37)],
    "West Irish Sea": [(-15, 52), (-9, 52), (-9, 55), (-15, 55)],
    "South Irish Sea": [(-12, 50), (-6, 50), (-6, 52), (-12, 52)],
}
region_abbrevs = {
    "Skagerrak": "SKG",
    "West Iberian Sea": "WIS",
    "Gulf of Cadiz": "GOC",
    "West Irish Sea": "WIR",
    "South Irish Sea": "SIR",
}

nea_ocean_basins = rm.defined_regions.natural_earth_v5_0_0.ocean_basins_50[
    [35, 41, 42, 70, 75, 76]
]
polygons = nea_ocean_basins.polygons
region_names = nea_ocean_basins.names
abbrevs = nea_ocean_basins.abbrevs


def add_region(name):
    polygons.append(regions[name])
    region_names.append(name)
    abbrevs.append(region_abbrevs[name])


add_region("Skagerrak")
add_region("West Iberian Sea")
add_region("Gulf of Cadiz")
add_region("West Irish Sea")
add_region("South Irish Sea")


nea_ocean_basins = rm.Regions(polygons, names=region_names, abbrevs=abbrevs)
