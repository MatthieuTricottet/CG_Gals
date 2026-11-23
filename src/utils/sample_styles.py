"""Visual encodings for the galaxy samples in the article."""

SAMPLE_STYLES = {
    "CG4": {"colour": "#B2182B", "marker": "o", "linestyle": "-"},
    "Control4B": {"colour": "#0072B2", "marker": "s", "linestyle": "--"},
    "Control4C": {"colour": "#E69F00", "marker": "^", "linestyle": "-."},
    "RG4": {"colour": "#009E73", "marker": "D", "linestyle": ":"},
}
SAMPLES = tuple(SAMPLE_STYLES)
MARKER_EDGE = "#222222"
LINE_COLOURS = {sample: style["colour"] for sample, style in SAMPLE_STYLES.items()}
# Use the darker amber for thin lines and error bars against white.
LINE_COLOURS["Control4C"] = "#C88A00"
# Fixed plotting order (and legend order) for the binned figures, from the
# compact groups to the most loosely defined control.
PLOT_ORDER = ("CG4", "Control4C", "Control4B", "RG4")
