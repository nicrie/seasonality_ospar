import matplotlib.colors as mcolors
import seaborn as sns

SEASON = ["DJF", "MAM", "JJA", "SON"]
SEASON_LABELS = ["Winter", "Spring", "Summer", "Autumn"]

CATEGORY = ["TA", "SEA", "LOCAL"]
CATEGORY_LABELS = ["Total abundance", "SEA items", "LOCAL items"]

COLORS = {}

# add colors for seasons
clrs = sns.color_palette("twilight", as_cmap=False, n_colors=5)
clrs = [clrs[0], clrs[1], clrs[3], clrs[4]]
# Convert to RGB format
clrs_rgb = [mcolors.to_rgb(c) for c in clrs]
# Add alpha to the colors
palette = [(r, g, b, 0.6) for r, g, b in clrs_rgb]  # 0.6 is the alpha value
palette = dict(zip(SEASON_LABELS, palette))
COLORS.update(palette)


def get_cyclic_palette(as_cmap=False, n_colors=4):
    return sns.husl_palette(h=0.7, as_cmap=as_cmap, n_colors=n_colors, s=0.9)


def set_theme():
    dark_grey = ".3"
    light_grey = ".8"
    sns.set_theme(
        "paper",
        style="ticks",
        font="Lato",
        rc={
            "figure.figsize": [7.2, 9.7],
            "figure.edgecolor": dark_grey,
            # "font.size": 10,
            "text.color": dark_grey,
            "axes.edgecolor": dark_grey,
            "axes.labelcolor": dark_grey,
            "axes.titlecolor": dark_grey,
            "axes.titleweight": "semibold",
            "grid.color": light_grey,
            "xtick.color": dark_grey,
            "ytick.color": dark_grey,
            "text.usetex": False,
        },
        font_scale=0.9,
    )
