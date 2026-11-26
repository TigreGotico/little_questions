"""Shared matplotlib style for docs plots. Copy this file into docs/img/ of a repo.

    from plot_style import save

    def draw(fig, theme):
        ax = fig.subplots()
        ax.barh(names, values, color=theme["series"][0])
        ax.set_xlabel("PER (%)")

    save("docs/img/per-by-language", draw)   # writes .svg and -dark.svg

save() calls draw once for each theme. Embed both files with:

    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="img/per-by-language-dark.svg">
      <img alt="..." src="img/per-by-language.svg">
    </picture>

The categorical order is fixed: series 1 takes theme["series"][0], and so on.
Fold a ninth series into "Other". Colours come from the reference palette of the
dataviz skill; the light and dark steps are validated for colour-vision deficiency.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LIGHT = {
    "surface": "#ffffff", "text": "#191919", "muted": "#666666", "grid": "#e6e6e6",
    "series": ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"],
    "sequential": "Blues",
}
DARK = {
    "surface": "#191919", "text": "#efefef", "muted": "#9c9c9c", "grid": "#333333",
    "series": ["#338ced", "#e76624", "#00a888", "#d28e05", "#e24d8b", "#0aac23", "#9887ef", "#eb6263"],
    "sequential": "Blues_r",
}


def style(theme):
    return {
        "figure.facecolor": theme["surface"], "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "text.color": theme["text"], "axes.labelcolor": theme["text"], "axes.titlecolor": theme["text"],
        "xtick.color": theme["muted"], "ytick.color": theme["muted"], "axes.edgecolor": theme["muted"],
        "axes.grid": True, "axes.axisbelow": True, "grid.color": theme["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=theme["series"]),
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.linewidth": 2, "lines.markersize": 6,
        "legend.frameon": False, "svg.fonttype": "none", "svg.hashsalt": "docs",
    }


def save(stem, draw, width=7.0, height=3.6):
    """Write <stem>.svg and <stem>-dark.svg. draw(fig, theme) builds the plot."""
    for theme, suffix in ((LIGHT, ""), (DARK, "-dark")):
        with plt.rc_context(style(theme)):
            fig = plt.figure(figsize=(width, height), layout="constrained")
            draw(fig, theme)
            fig.savefig(f"{stem}{suffix}.svg", metadata={"Date": None})
            plt.close(fig)
