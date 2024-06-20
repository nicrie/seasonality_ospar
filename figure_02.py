# %%
import datatree as dt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import PIL
import seaborn as sns
import xarray as xr
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase
from matplotlib.transforms import Bbox, TransformedBbox

import utils.styles
from utils.statistics import weighted_percentile
from utils.styles import get_cyclic_palette

utils.styles.set_theme()


# %%
COLORS = get_cyclic_palette(as_cmap=False, n_colors=4)
SEASONS = ["DJF", "MAM", "JJA", "SON"]
QUANTITY = "absolute"
VARIABLE = "Plastic"
YEAR = 2001


base_path = f"data/gpr/{QUANTITY}/{VARIABLE}/{YEAR}/"
ospar = dt.open_datatree("data/beach_litter/ospar/preprocessed.zarr", engine="zarr")
litter_o = ospar[f"preprocessed/{QUANTITY}/{VARIABLE}"]
litter_o = litter_o.sel(year=slice(YEAR, 2020)).dropna("beach_id", **{"how": "all"})

model = dt.open_datatree(base_path + "posterior_predictive.zarr", engine="zarr")
litter_m = model["posterior_predictive"][VARIABLE]
litter_m
results = xr.open_dataset(base_path + "effect_size_seasons.nc")

# %%
effect_size = results.effect_size_gp
confidence = results.max_hdi
median_diff = results.median_diff
relative_diff = results.relative_diff
is_pos = effect_size >= 0
is_neg = effect_size < 0
n_beaches = results.beach_id.size

# Weighted sample size (WSS)
weights_pos = confidence.where(is_pos)
weights_neg = confidence.where(is_neg)
wss_pos = weights_pos.sum("beach_id")
wss_neg = weights_neg.sum("beach_id")

# Weighted sample size in percentage of beaches
perc_beaches_pos = wss_pos / n_beaches
perc_beaches_neg = wss_neg / n_beaches

# Weighted effect size (WES)
wes_pos = (weights_pos * effect_size.where(is_pos)).sum("beach_id") / wss_pos
wes_neg = (weights_neg * effect_size.where(is_neg)).sum("beach_id") / wss_neg

# Weighed median difference
dims = ["beach_id"]
wmd_pos, ess_pos = weighted_percentile(
    median_diff, weights_pos, [0.25, 0.5, 0.75], dims
)
wmd_neg, ess_neg = weighted_percentile(
    -median_diff, weights_neg, [0.25, 0.5, 0.75], dims
)

# Weighted relative effect size (WRES)
wres_pos, ess_pos = weighted_percentile(
    relative_diff, weights_pos, [0.25, 0.5, 0.75], dims
)
wres_neg, ess_neg = weighted_percentile(
    -relative_diff, weights_neg, [0.25, 0.5, 0.75], dims
)


# %%

icons = {
    "DJF": "figs/icons/seasons/png/002-snowflake.png",
    "MAM": "figs/icons/seasons/png/003-flower.png",
    "JJA": "figs/icons/seasons/png/001-sun.png",
    "SON": "figs/icons/seasons/png/004-leaf-fall.png",
}

# Load images
images = {k: PIL.Image.open(fname) for k, fname in icons.items()}


def create_bidirectional_seasons_visualization(
    data_positive,
    data_negative,
    fig=None,
    ax=None,
    wfac=200,
    important_relations=None,
):
    if fig is None:
        fig, ax = plt.subplots(**{"figsize": (6, 6)})

    cmap_peak_season = sns.husl_palette(h=0.7, s=0.3, as_cmap=False, n_colors=4)
    node_colors = {
        "DJF": cmap_peak_season[0],
        "MAM": cmap_peak_season[1],
        "JJA": cmap_peak_season[2],
        "SON": cmap_peak_season[3],
    }

    # Image URLs for graph nodes
    # attribution: <div>Icons made by <a href="https://www.freepik.com" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div><div>Icons made by <a href="https://www.flaticon.com/authors/kmg-design" title="kmg design">kmg design</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div><div>Icons made by <a href="https://www.flaticon.com/authors/vector-stall" title="Vector Stall">Vector Stall</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div><div>Icons made by <a href="https://www.flaticon.com/authors/iconixar" title="iconixar">iconixar</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>
    icons = {
        "DJF": "figs/icons/seasons/png/002-snowflake.png",
        "MAM": "figs/icons/seasons/png/003-flower.png",
        "JJA": "figs/icons/seasons/png/001-sun.png",
        "SON": "figs/icons/seasons/png/004-leaf-fall.png",
    }

    # Load images
    images = {k: PIL.Image.open(fname) for k, fname in icons.items()}

    combinations = results.combination.values
    # Initialize the graph
    G = nx.MultiDiGraph()

    # Add nodes with the name of the seasons
    for season in node_colors.keys():
        G.add_node(season, image=images[season])

    # Add edges with weights
    for comb in combinations:
        season1, season2 = comb.split("<")
        season1 = season1.strip()
        season2 = season2.strip()
        wp = data_positive.sel(combination=comb).item()
        wn = data_negative.sel(combination=comb).item()
        G.add_edge(season2, season1, weight=wp)
        G.add_edge(season1, season2, weight=wn)

    # Define node positions in a circular layout
    pos = nx.circular_layout(G)

    # manually specify node position
    pos["DJF"] = np.array([-1, 0])
    pos["MAM"] = np.array([0, 1])
    pos["JJA"] = np.array([1, 0])
    pos["SON"] = np.array([0, -1])

    # Draw the nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=1000,
        edgecolors=None,
        node_color=[node_colors[n] for n in G.nodes],
    )

    highlight_color = sns.color_palette("colorblind")[3]

    def get_label_color(u, v, highlight_color):
        return highlight_color if (u, v) in important_relations else ".5"

    label_colors = [get_label_color(u, v, highlight_color) for u, v in G.edges()]

    # Draw the edges with varying thickness
    _ = nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrowstyle="-|>",
        arrowsize=15,
        # edge_color=".3",
        connectionstyle="arc3,rad=0.1",
        width=[G[u][v][0]["weight"] ** (2) / wfac for u, v in G.edges()],
        edge_color=label_colors,
        # edge_cmap=plt.cm.plasma,
        min_source_margin=20,
        min_target_margin=20,
    )

    # Draw the labels
    # nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edge_labels(
        G,
        pos,
        ax=ax,
        font_size=8,
        label_pos=0.65,
        font_color=highlight_color,
        font_weight="bold",
        edge_labels={
            (u, v): d["weight"]
            for u, v, d in G.edges(data=True)
            if (u, v) in important_relations
        },
        bbox=dict(boxstyle="round", fc="w", ec="w", alpha=0.75, pad=0.1),
        rotate=False,
    )
    nx.draw_networkx_edge_labels(
        G,
        pos,
        ax=ax,
        font_size=8,
        label_pos=0.65,
        font_color=".5",
        edge_labels={
            (u, v): d["weight"]
            for u, v, d in G.edges(data=True)
            if (u, v) not in important_relations
        },
        bbox=dict(boxstyle="round", fc="w", ec="w", alpha=0.75, pad=0.1),
        rotate=False,
    )

    # Transform from data coordinates (scaled between xlim and ylim) to display coordinates
    tr_figure = ax.transData.transform
    # Transform from display to figure coordinates
    tr_axes = fig.transFigure.inverted().transform

    # Select the size of the image (relative to the X axis)
    icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.04
    icon_center = icon_size / 2.0

    # Add the respective image to each node
    for n in G.nodes:
        xf, yf = tr_figure(pos[n])
        xa, ya = tr_axes((xf, yf))
        # get overlapped axes and plot icon
        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        a.imshow(G.nodes[n]["image"])
        a.axis("off")

    # Show the plot
    ax.axis("off")


# %%


important_relations_A = [
    ("DJF", "JJA"),
    ("DJF", "SON"),
    ("MAM", "JJA"),
    ("MAM", "SON"),
    ("MAM", "DJF"),
    ("SON", "JJA"),
]
important_relations_B = [
    ("DJF", "MAM"),
    ("DJF", "JJA"),
    ("DJF", "SON"),
    ("MAM", "JJA"),
    ("MAM", "SON"),
    ("SON", "JJA"),
]


fig, axes = plt.subplots(1, 2, **{"figsize": (7.2, 3.6)})
create_bidirectional_seasons_visualization(
    (perc_beaches_pos * 100).round(0).astype(int),
    (perc_beaches_neg * 100).round(0).astype(int),
    fig,
    axes[0],
    3e2,
    important_relations=important_relations_A,
)
dir_pos = (100 * wres_pos).sel(quantile=0.5).round(0).astype(int)
dir_neg = (100 * wres_neg).sel(quantile=0.5).round(0).astype(int)
create_bidirectional_seasons_visualization(
    dir_pos,
    dir_neg,
    fig,
    axes[1],
    wfac=2e3,
    important_relations=important_relations_B,
)
axes[0].set_title("A | How many beaches exhibit seasonal differences?", loc="left")
axes[1].set_title("B | What is the median seasonal variation?", loc="left")

axes[0].text(
    0.08,
    1,
    r"in % of 168 beaches",
    ha="left",
    va="top",
    style="italic",
    transform=axes[0].transAxes,
    fontsize=7,
)
axes[1].text(
    0.08,
    1,
    r"in %",
    ha="left",
    va="top",
    style="italic",
    transform=axes[1].transAxes,
    fontsize=7,
)

axes[0].annotate(
    "37 % of beaches exhibit \nmore litter in spring \nthan in summer",
    xy=(0.67, 0.79),
    xytext=(0.78, 0.93),
    xycoords="axes fraction",
    textcoords="axes fraction",
    ha="left",
    va="top",
    fontsize=7,
    color="k",
    arrowprops=dict(color=".5", arrowstyle="-", connectionstyle="arc3,rad=0.5"),
)
axes[1].annotate(
    "For those beaches, litter \nis 112 % more abundant \nin spring than in summer",
    xy=(0.67, 0.79),
    xytext=(0.78, 0.93),
    xycoords="axes fraction",
    textcoords="axes fraction",
    ha="left",
    va="top",
    fontsize=7,
    color="k",
    arrowprops=dict(color=".5", arrowstyle="-", connectionstyle="arc3,rad=0.5"),
)


# Add a legend to the lower right of the figure
class ImageHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # enlarge the image by these margins
        sx, sy = self.image_stretch

        # create a bounding box to house the image
        bb = Bbox.from_bounds(xdescent - sx, ydescent - sy, width + sx, height + sy)

        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)

        self.update_prop(image, orig_handle, legend)

        return [image]

    def set_image(self, image, image_stretch=(0, 0)):
        self.image_data = image
        self.image_stretch = image_stretch


s1 = axes[1].scatter(5, 5, zorder=0)
s2 = axes[1].scatter(5, 5, zorder=0)
s3 = axes[1].scatter(5, 5, zorder=0)
s4 = axes[1].scatter(5, 5, zorder=0)

for ax in axes:
    ax.set_xlim(-1.21, 1.21)
    ax.set_ylim(-1.21, 1.21)
# setup the handler instance for the scattered data
handles = {}
for season in SEASONS:
    handles[season] = ImageHandler()
    handles[season].set_image(images[season], image_stretch=(1, 1))

# add the legend for the scattered data, mapping the
# scattered points to the custom handler
axes[1].legend(
    [s1, s2, s3, s4],
    ["Winter", "Spring", "Summer", "Autumn"],
    handler_map={
        s1: handles["DJF"],
        s2: handles["MAM"],
        s3: handles["JJA"],
        s4: handles["SON"],
    },
    labelspacing=0.8,
    columnspacing=1.0,
    handleheight=1,
    handlelength=1,
    handletextpad=0.3,
    frameon=False,
    ncols=1,
    loc="center",
    bbox_to_anchor=(1, 0.15),
)


plt.savefig("figs/figure02.png", dpi=500, bbox_inches="tight", pad_inches=0.1)
plt.show()
# %%
