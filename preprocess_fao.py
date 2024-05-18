# %%
import os

import numpy as np
import pandas as pd
import xarray as xr
from datatree import open_datatree

root = "data/economic/production/"


def read_data(source):
    path = os.path.join(root, "{:}_2023.1.1/{:}_Quantity.csv")
    return pd.read_csv(path.format(source, source))


code_species = pd.read_csv(root + "Capture_2023.1.1/CL_FI_SPECIES_GROUPS.csv")
code_countries = pd.read_csv(root + "Capture_2023.1.1/CL_FI_COUNTRY_GROUPS.csv")
code_countries.replace(
    {"Name_En": {"Netherlands (Kingdom of the)": "Netherlands"}}, inplace=True
)

fish = read_data("Capture")
aqua = read_data("Aquaculture")

sources = open_datatree("data/litter_sources.zarr", engine="zarr")
species = np.unique(sources.fishing.first_sales.species)
ospar = open_datatree("data/ospar/preprocessed.zarr", engine="zarr")
countries = np.unique(ospar["preprocessed"].country).tolist()

code_countries = code_countries.loc[code_countries["Name_En"].isin(countries)]
code_countries

fish = fish.loc[fish["COUNTRY.UN_CODE"].isin(code_countries["UN_Code"])]
aqua = aqua.loc[aqua["COUNTRY.UN_CODE"].isin(code_countries["UN_Code"])]

fish = fish.loc[fish["MEASURE"] == "Q_tlw"]
aqua = aqua.loc[aqua["MEASURE"] == "Q_tlw"]

# Add species names
fish["species_fao"] = fish["SPECIES.ALPHA_3_CODE"].map(
    code_species.set_index("3A_Code")["Name_En"]
)
aqua["species_fao"] = aqua["SPECIES.ALPHA_3_CODE"].map(
    code_species.set_index("3A_Code")["Name_En"]
)

# Add countrie names
fish["country"] = fish["COUNTRY.UN_CODE"].map(
    code_countries.set_index("UN_Code")["Name_En"]
)
aqua["country"] = aqua["COUNTRY.UN_CODE"].map(
    code_countries.set_index("UN_Code")["Name_En"]
)


eumofas2fao = {
    "Abalone": ["Tuberculate abalone", "Abalones nei"],
    "Anchovy": ["European anchovy"],
    "Blue whiting": ["Blue whiting(=Poutassou)"],
    "Brill": ["Brill"],
    "Carp": [
        "Common carp",
        "Crucian carp",
        "Carpet shells nei",
        "Grooved carpet shell",
        "Japanese carpet shell",
        "Banded carpet shell",
        "Grass carp(=White amur)",
        "Silver carp",
        "Bighead carp",
        "Golden carpet shell",
    ],
    "Clam": [
        "Clams, etc. nei",
        "Solid surf clam",
        "Venus clams nei",
        "Northern quahog(=Hard clam)",
        "Donax clams",
        "Razor clams, knife clams nei",
        "Solen razor clams nei",
        "Surf clams nei",
        "Stimpson's surf clam",
        "Mactra surf clams nei",
        "European razor clam",
        "Subtruncate surf clam",
        "Manila clam",
        "Pod razor shell",
        "Truncate donax",
        "Common edible cockle",
        "Cockles nei",
        "Sand cockle",
        "Tuberculate cockle",
        "Olive green cockle",
        "Carpet shells nei",
        "Grooved carpet shell",
        "Japanese carpet shell",
        "Banded carpet shell",
        "Marine shells nei",
        "Pod razor shell",
        "Sword razor shell",
        "Granular top-shell",
        "Hungarian cap-shell",
        "Golden carpet shell",
        "Arched razor shell",
    ],
    "Cobia": ["Cobia"],
    "Cod": [
        "Atlantic cod",
        "Poor cod",
        "Tadpole codling",
        "Greenland cod",
        "Marbled rockcod",
        "Humped rockcod",
        "Striped-eyed rockcod",
        "Crocodile icefishes nei",
        "Pacific cod",
        "Polar cod",
        "Antarctic rockcods, noties nei",
        "Yellowbelly rockcod",
        "Black rockcod",
        "Moray cods nei",
        "Brazilian codling",
        "Crocodile shark",
        "Smalleye moray cod",
        "North Atlantic codling",
        "Patagonian rockcod",
        "Longtail Southern cod",
        "Red codling",
        "Striped rockcod",
    ],
    "Crab": [
        "Edible crab",
        "Spinous spider crab",
        "Marine crabs nei",
        "Stone king crab",
        "Portunus swimcrabs nei",
        "Green crab",
        # "Mediterranean shore crab",  # ignore mediterranean
        "Velvet swimcrab",
        "Right-handed hermit crabs nei",
        "Deep-sea red crab",
        "Swimming crabs, etc. nei",
        "Tanner crabs nei",
        "King crabs, stone crabs nei",
        "Chinese mitten crab",
        "Queen crab",
        "Red king crab",
        "Henslow’s swimming crab",
        "Shamefaced crab",
        "West african fiddler crab",
        "Atlantic rock crab",
        "Knobby swimcrab",
        "Blue crab",
        "Blue-leg swimcrab",
        "Maja spider crabs nei",
        "Common spider crab",
        "King crabs nei",
        "King crab",
        "Antarctic stone crab",
        "Red crab",
        "Southwest Atlantic red crab",
        "King crabs",
        "Red stone crab",
        "Globose king crab",
    ],
    "Cusk-eel": [
        "Pink cusk-eel",
        "Cusk-eels nei",
        "Pudgy cuskeel",
    ],
    "Cuttlefish": [
        "Common cuttlefish",
        "Cuttlefish, bobtail squids nei",
        "Elegant cuttlefish",
        "Pink cuttlefish",
        "Cuttlefishes nei",
        "African cuttlefish",
    ],
    "Dab": ["Common dab", "Amer. plaice(=Long rough dab)"],
    "Dogfish": [
        "Dogfishes and hounds nei",
        "Picked dogfish",
        "Dogfish sharks nei",
        "Birdbeak dogfish",
        "Arrowhead dogfish",
        "Portuguese dogfish",
        "Longnose velvet dogfish",
        "Knifetooth dogfish",
        "Black dogfish",
        "Dogfishes nei",
        "Dogfish sharks, etc. nei",
        "Rough longnose dogfish",
    ],
    "Eel": [
        "European eel",
        "Eelpout",
        "Sandeels(=Sandlances) nei",
        # "Pink cusk-eel",  # already in Cusk-eel
        # "Pudgy cuskeel",  # already in Cusk-eel
        # "Cusk-eels nei",  # already in Cusk-eel
        "Eelpouts",
        "Conger eels nei",
        "Conger eels, etc. nei",
        "Painted eel",
        "Small sandeel",
        # "Mediterranean sand eel",  # ignore mediterranean
        "Smooth sandeel",
        "Muddy arrowtooth eel",
        "Kaup's arrowtooth eel",
    ],
    "Flounder, European": ["European flounder"],
    "Flounder, other": [
        "Witch flounder",
        # "European flounder",  # alread in Flounder, European
        "Lefteye flounders nei",
        "Righteye flounders nei",
        "Winter flounder",
        "Spotted flounder",
        "Yellowtail flounder",
        "Summer flounder",
        "Wide-eyed flounder",
        "Antarctic armless flounder",
    ],
    "Freshwater catfish": [
        # "Wolffishes(=Catfishes) nei",  # marine
        "Wels(=Som) catfish",
        # "Crucifix sea catfish",
        # "Sea catfishes nei",
        # "Smoothmouth sea catfish",
    ],
    "Freshwater crayfish": [
        "Signal crayfish",
        "Euro-American crayfishes nei",
        "White-clawed crayfish",
        "Noble crayfish",
    ],
    "Grenadier": [
        "Roundnose grenadier",
        "Patagonian grenadier",
        "Grenadiers nei",
        "Roughhead grenadier",
        "Common Atlantic grenadier",
        "Blue grenadier",
        "Bigeye grenadier",
        "Roughsnout grenadier",
        "Softhead grenadier",
        "Western softhead grenadier",
        "Whitson's grenadier",
    ],
    "Gurnard": [
        "Gurnards, searobins nei",
        "Tub gurnard",
        "Red gurnard",
        "Grey gurnard",
        "Piper gurnard",
        "Streaked gurnard",
        "Flying gurnard",
        "Scorpionfishes, gurnards nei",
        "Gurnards nei",
        "Indo-Pacific gurnards",
        "Cape gurnard",
        "Longfin gurnard",
        "Spiny gurnard",
        "Large-scaled gurnard",
    ],
    "Haddock": ["Haddock"],
    "Hake": [
        "European hake",
        "White hake",
        "Hakes nei",
        "Cape hakes",
        "Argentine hake",
        "Red hake",
        "Senegalese hake",
        "Silver hake",
        "South Pacific hake",
        "North Pacific hake",
        "Southern hake",
        "Longfin hake",
        "Merluccid hakes nei",
        "Benguela hake",
        "Deep-water Cape hake",
        "Shallow-water Cape hake",
    ],
    "Halibut, Atlantic": ["Atlantic halibut"],
    "Halibut, Greenland": ["Greenland halibut"],
    "Herring": ["Atlantic herring", "Pacific herring", "Whitehead's round herring"],
    "Horse mackerel, Atlantic": ["Atlantic horse mackerel"],
    "Horse mackerel, other": [
        "Jack and horse mackerels nei",
        # "Atlantic horse mackerel",
        # "Mediterranean horse mackerel",  # ignore mediterranean
        "Cape horse mackerel",
        "Cunene horse mackerel",
    ],
    "Jellyfish": ["Common jellyfish", "Jellyfishes nei"],
    "John dory": ["John dory", "Silvery John dory"],
    "Ling": [
        "Grayling",
        "Ling",
        "Blue ling",
        "Tadpole codling",
        "Spanish ling",
        "Rocklings nei",
        "Lings nei",
        "Brazilian codling",
        "North Atlantic codling",
        "Threadfin rockling",
        # "Mediterranean bigeye rockling",  # ignore mediterranean
        "Bigeye rockling",
        "Shore rockling",
        "Three-bearded rockling",
        "Fivebeard rockling",
        "Red codling",
    ],
    "Lobster Homarus spp": ["Homarus lobsters nei"],
    "Lobster, Norway": ["Norway lobster"],
    "Mackerel": [
        # "Jack and horse mackerels nei",
        # "Atlantic horse mackerel",
        "Atlantic mackerel",
        "Atlantic chub mackerel",
        "Mackerel icefish",
        "Narrow-barred Spanish mackerel",
        "West African Spanish mackerel",
        # "Mediterranean horse mackerel",
        "Mackerels nei",
        "Pacific chub mackerel",
        "Atka mackerel",
        "Snake mackerels, escolars nei",
        "Chilean jack mackerel",
        "Pacific jack mackerel",
        # "Cape horse mackerel",
        # "Cunene horse mackerel",
        "Scomber mackerels nei",
        "Blue jack mackerel",
        "Mackerel sharks,porbeagles nei",
    ],
    "Megrim": ["Megrim", "Megrims nei", "Four-spot megrim"],
    "Miscellaneous small pelagics": [],  # we can safely assume 100 % marine
    "Molluscs and aquatic invertebrates, other": [
        "Marine molluscs nei",
        "Freshwater molluscs nei",
    ],
    "Monk": ["Monkfishes nei", "Angler(=Monk)"],
    "Mussel Mytilus spp": [
        "Blue mussel",
        "Sea mussels nei",
        "Mytilus mussels nei",
        "Mediterranean mussel",
        "Horse mussels nei",
        "European date mussel",
    ],
    "Octopus": [
        "Octopuses, etc. nei",
        "Common octopus",
        "Horned and musky octopuses",
        "Horned octopus",
        "Musky octopus",
        "White-spotted octopus",
        "Octopuses nei",
        "Antarctic octopuses",
    ],
    "Other cephalopods": ["Cephalopods nei"],
    "Other crustaceans": ["Marine crustaceans nei", "Freshwater crustaceans nei"],
    "Other flatfish": ["Flatfishes nei"],
    "Other freshwater fish": ["Freshwater fishes nei"],
    "Other groundfish": ["Groundfishes nei"],
    "Other marine fish": ["Marine fishes nei"],
    "Other salmonids": ["Salmonoids nei"],
    "Other sharks": [],  # we can safely assume 100 % marine
    "Other unspecified products": [],  # assume 100% marine
    "Oyster": [
        "European flat oyster",
        "Pacific cupped oyster",
        "Cupped oysters nei",
        "Flat oysters nei",
        "Rayed pearl oyster",
    ],
    "Picarel": [
        "Picarels nei",
        "Picarel",
        "Curled picarel",
        "Blotched picarel",
        "Bigeye picarel",
    ],
    "Pike": ["Northern pike", "Pike icefish"],
    "Pike-perch": ["Pike-perch"],
    "Plaice, European": ["European plaice"],
    "Plaice, other": ["Amer. plaice(=Long rough dab)"],
    "Pollack": ["Pollack"],
    "Pouting (=Bib)": ["Pouting(=Bib)"],
    "Ray": [
        "Rays and skates nei",
        "Thornback ray",
        "Spotted ray",
        "Blonde ray",
        "Small-eyed ray",
        "Undulate ray",
        "Starry ray",
        "Sailray",
        "Sandy ray",
        "Cuckoo ray",
        "Rays, stingrays, mantas nei",
        # "Mediterranean starry ray",  # ignore mediterranean
        "Bathyraja rays nei",
        "Shagreen ray",
        "Stingrays nei",
        "Common stingray",
        "Pelagic stingray",
        "Common eagle ray",
        "Torpedo rays",
        "Marbled electric ray",
        "Sharks, rays, skates, etc. nei",
        "Southern rays bream",
        "Raja rays nei",
        "Round ray",
        "Spinetail ray",
        "Brown ray",
        "Stingrays, butterfly rays nei",
        "Spiny butterfly ray",
        "Madeiran ray",
        "Rough ray",
        "Deep-water ray",
        "Roughtail stingray",
        "Eagle rays nei",
        "Electric rays nei",
    ],
    "Red mullet": ["Red mullet"],
    "Redfish": [
        "Atlantic redfishes nei",
        "Golden redfish",
        "Beaked redfish",
        "Scorpionfishes, redfishes nei",
        "Redfish",
        "Norway redfish",
        "Cape redfish",
    ],
    "Rock lobster and sea crawfish": [],  # assume 100 % marine
    "Saithe (=Coalfish)": ["Saithe(=Pollock)"],
    "Salmon": ["Salmonoids nei", "Atlantic salmon"],
    "Sardine": [
        "European pilchard(=Sardine)",
        "Sardinellas nei",
        "Round sardinella",
        "Madeiran sardinella",
        "Pacific sardine",
    ],
    "Scabbardfish": [
        "Silver scabbardfish",
        "Black scabbardfish",
        "Hairtails, scabbardfishes nei",
        "Intermediate scabbardfish",
    ],
    "Scallop": [
        "Great Atlantic scallop",
        "Scallops nei",
        "Queen scallop",
        "Great Mediterranean scallop",
        "Variegated scallop",
        "Iceland scallop",
        "Scalloped hammerhead",
        "Club scallop",
        "Patagonian scallop",
    ],
    "Sea cucumber": ["Sea cucumbers nei", "Royal cucumber"],
    "Sea urchin": [
        "European edible sea urchin",
        "Sea urchins, etc. nei",
        "Stony sea urchin",
        "Sea urchins nei",
        "Black sea urchin",
    ],
    "Seabass, European": ["European seabass"],
    "Seabass, other": [
        # "European seabass",
        "Groupers, seabasses nei",
        "Seabasses nei",
        "Spotted seabass",
    ],
    "Seabream, gilthead": [
        "Gilthead seabream",
    ],
    "Seabream, other": [
        "Blackspot seabream",
        "Axillary seabream",
        "Annular seabream",
        "White seabream",
        "Common two-banded seabream",
        "Zebra seabream",
        "Sharpsnout seabream",
        "Saddled seabream",
        "Bluespotted seabream",
        "Panga seabream",
        "Redbanded seabream",
        "Black seabream",
        "Porgies, seabreams nei",
    ],
    "Seaweed and other algae": [
        "Brown seaweeds",
        "Red seaweeds",
        "Green seaweeds",
        "Seaweeds nei",
        "Gelidium seaweeds",
    ],
    "Shrimp Crangon spp": [
        "Crangon shrimps nei",
    ],
    "Shrimp, coldwater": ["Northern prawn"],
    "Shrimp, deep-water rose": ["Deep-water rose shrimp"],
    "Shrimp, miscellaneous": [
        "Common shrimp",
        "Penaeus shrimps nei",
        "Aristeid shrimps nei",
        "Aesop shrimp",
        "Knife shrimp",
        "Pandalus shrimps nei",
        "Palaemonid shrimps nei",
        # "Crangon shrimps nei",
        "Pink glass shrimp",
        "Southern pink shrimp",
        "Speckled shrimp",
        "Giant red shrimp",
        "Blue and red shrimp",
        "Scarlet shrimp",
        "Striped soldier shrimp",
        # "Penaeid shrimps nei",
        "Metapenaeus shrimps nei",
        "Guinea shrimp",
        "Megalops shrimp",
        "Striped red shrimp",
        "Pandalid shrimps nei",
        "Armed nylon shrimp",
        "Golden shrimp",
        "Narwal shrimp",
        "Whip shrimp",
        "Palaemon shrimps nei",
        "Atlantic mud shrimp",
        "Jack-knife shrimp",
        "White glass shrimp",
        "Red snapping shrimp",
    ],
    "Shrimp, warmwater": ["Penaeid shrimps nei", "Banana prawn", "Giant river prawn"],
    "Smelt": [
        "European smelt",
        "Silversides(=Sand smelts) nei",
        "Big-scale sand smelt",
        "Sand smelt",
    ],
    "Sole, common": ["Common sole"],
    "Sole, other": [
        "Lemon sole",
        "Soles nei",
        # "Common sole",
        "Sand sole",
        "Solenette",
        "Senegalese sole",
        "Wedge sole",
        "Thickback soles nei",
        "Thickback sole",
        "Solen razor clams nei",
        "Black sole",
        "Southeast Atlantic soles nei",
        "Bean solen",
        "Deep water sole",
        "Ocellated wedge sole",
        "Whiskered sole",
        "Portuguese sole",
        "Guinean sole",
        "West coast sole",
        "Foureyed sole",
        "Klein's sole",
        "Cyclope sole",
        "Cadenat's sole",
        "Adriatic sole",
        "Senegalese tonguesole",
        "Guinean tonguesole",
    ],
    "Sprat (=Brisling)": ["European sprat"],
    "Squid": [
        "Inshore squids nei",
        "Common squids nei",
        "European squid",
        "Various squids nei",
        "Cuttlefish, bobtail squids nei",
        "Northern shortfin squid",
        "Patagonian squid",
        "Longfin squid",
        "Cape Hope squid",
        "European common squid",
        "Broadtail shortfin squid",
        "Argentine shortfin squid",
        "European flying squid",
        "Sevenstar flying squid",
        "Veined squid",
        "Neon flying squid",
        "Stout bobtail squid",
        "Dwarf bobtail squid",
        "Common bobtail squid",
        "Alloteuthis squids nei",
        "Midsize squid",
        "Ommastrephidae squids nei",
        "Flying squids nei",
        "Webbed flying squid",
        "Lesser flying squid",
        "Todarodes flying squids nei",
        "Angolan flying squid",
        "Orangeback flying squid",
        "Greater hooked squid",
        "Diamondback squid",
        "Shortfin squids nei",
    ],
    "Squillid": ["Squillids nei", "Spottail mantis squillid"],
    "Swordfish": ["Swordfish"],
    "Toothfish": ["Antarctic toothfish", "Patagonian toothfish"],
    "Trout": ["Sea trout", "Rainbow trout", "Trouts nei"],
    "Tuna, albacore": [
        "Albacore",
    ],
    "Tuna, bigeye": [
        "Bigeye tuna",
    ],
    "Tuna, bluefin": [
        "Pacific bluefin tuna",
        "Southern bluefin tuna",
        "Atlantic bluefin tuna",
    ],
    "Tuna, miscellaneous": [
        "Tuna-like fishes nei",
        "Tunas nei",
        "Dogtooth tuna",
        "Frigate and bullet tunas",
        "Frigate tuna",
        "Bullet tuna",
        "Blackfin tuna",
        "True tunas nei",
        "Longtail tuna",
    ],
    "Tuna, skipjack": [
        "Skipjack tuna",
    ],
    "Tuna, yellowfin": [
        "Yellowfin tuna",
    ],
    "Turbot": ["Turbot", "Turbots nei", "Spiny turbot"],
    "Weever": [
        "Greater weever",
        "Lesser weever",
        "Weeverfishes nei",
        "Weevers nei",
        "Spotted weever",
        "Starry weever",
    ],
    "Whiting": ["Whiting"],
}

# %%
# search for a species in fish
keyword = "seabream"
has_keyword = fish["species_fao"].str.contains(keyword, case=False).astype(bool)
found = fish.loc[has_keyword].species_fao.dropna().unique()
found


# %%

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
    for eumofas, fao in eumofas2fao.items():
        try:
            species = ds.species_fao.isin(fao)
            ds_new = ds.sel(species_fao=species).sum(dim="species_fao")
        except KeyError:
            print(f"Could not find {fao}")
            break
        ds_new = ds_new.expand_dims({"species": [eumofas]})
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


ospar = open_datatree("data/ospar/preprocessed.zarr", engine="zarr")
n_surveys = ospar.preprocessed["Plastic"].notnull().sum(("season"))
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
sales = xr.open_dataarray("data/economic/first_sales/first_sales_clean.nc")
ds = ds.assign_coords(commodity_group=sales.commodity_group)


ds.to_netcdf("data/economic/production/fao_fish_production.nc")


# %%
