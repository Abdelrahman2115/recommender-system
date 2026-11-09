"""Shared chart style so every figure in the project looks consistent."""
import matplotlib.pyplot as plt

# One fixed colour per model - a model keeps its colour in every chart.
MODEL_COLORS = {
    "Random": "#a3a29c",        # baselines are grey: context, not the story
    "Popularity": "#77766f",
    "Global mean": "#a3a29c",
    "Content-Based": "#2a78d6",  # blue
    "SVD": "#eb6834",            # orange
    "Hybrid": "#1baf7a",         # aqua
}
MAIN = "#2a78d6"                 # single hue for one-series charts
TEXT = "#52514e"


def set_style():
    plt.rcParams.update({
        "figure.figsize": (8, 4.2), "figure.dpi": 100,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c9c8c2", "axes.labelcolor": TEXT, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.titlelocation": "left",
        "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": TEXT, "ytick.color": TEXT, "font.size": 10,
        "legend.frameon": False,
    })


def bar_labels(ax, fmt="{:.3f}", **kw):
    """Write the value on top of each bar (used sparingly, for small comparisons)."""
    for c in ax.containers:
        ax.bar_label(c, labels=[fmt.format(v) for v in c.datavalues], padding=2,
                     fontsize=8, color=TEXT, **kw)
