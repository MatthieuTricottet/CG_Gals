"""Other galaxy properties versus stellar mass (descriptive figure, gary-r2 A4).

Four properties in the binning of Fig. 2, each shown separately for BGGs
(left column) and satellites (right column), as in Fig. 2:

  strong H-alpha emission fraction (EW <= -3 A, SDSS sign convention);
  AGN-like fraction among BPT-classified galaxies (Sect. 3.7 scheme);
  median D_n4000 (MPA-JHU galSpecIndx, Appendix A provenance);
  median log R_e,r (seeing-corrected Simard half-light radius, kpc).

Rows are produced for three scopes (all galaxies, BGGs, satellites); the
figure draws the rank-split scopes.  A Petrosian log R50 row set is computed
for the BGG coverage check but not drawn, because Simard sizes miss most
BGGs brighter than r = 14.

Purely descriptive: no model is fitted.  Intervals are 16--84% percentiles
of a bootstrap that resamples physical groups (``physical_group``), so two
CG4 selections sharing a host Lim group form one block.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import config as co
    from agn_environment import _classify as classify_bpt
    from descriptive_trends import (
        MORPHOLOGY_MASS_BINS,
        SAMPLE_LABELS,
        SAMPLE_STYLES,
        SAMPLES,
        _mass_bin_rows,
        draw_binned_series,
        finish_binned_figure,
    )
    from extended_data import ensure_galaxy_frame
    from extended_stats import safe_json
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from .agn_environment import _classify as classify_bpt
    from .descriptive_trends import (
        MORPHOLOGY_MASS_BINS,
        SAMPLE_LABELS,
        SAMPLE_STYLES,
        SAMPLES,
        _mass_bin_rows,
        draw_binned_series,
        finish_binned_figure,
    )
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import safe_json

STRONG_HALPHA_EW = -3.0
# (panel, value column, statistic)
PANELS = (
    ("strong_halpha", "strong_halpha", "fraction"),
    ("agn_like", "agn_like", "fraction"),
    ("dn4000", "Dn4000", "median"),
    ("log_rchl", "log_Rchl_r_kpc", "median"),
)
# computed for the BGG size-coverage check, not drawn
UNPLOTTED_PANELS = (("log_petro_r50", "log_petroR50_kpc", "median"),)
SCOPES = ("all", "bggs", "satellites")
# figure rows: (panel, title, y label, y limits); columns: (scope, label)
FIGURE_ROWS = (
    ("strong_halpha", "strong H$\\alpha$ fraction", "Fraction", (-0.03, 1.03)),
    ("agn_like", "AGN-like fraction", "Fraction", (-0.03, 1.03)),
    ("dn4000", r"median $D_n4000$", r"$D_n4000$", None),
    ("log_rchl", r"median $\log_{10}(R_{e,r}/{\rm kpc})$",
     r"$\log_{10}(R_{e,r}/{\rm kpc})$", None),
)
FIGURE_SCOPES = (("bggs", "BGGs"), ("satellites", "Satellites"))


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    work = classify_bpt(frame)
    halpha = pd.to_numeric(work.get("h_alpha_eqw", np.nan), errors="coerce")
    work["strong_halpha"] = np.where(
        np.isfinite(halpha), (halpha <= STRONG_HALPHA_EW).astype(float), np.nan
    )
    # _mass_bin_rows expects the catalogue column names used by Fig. 2
    work["rank_M"] = work["rank"]
    work["lgm"] = work["logMstar"]
    # _mass_bin_rows resamples the ``Group`` column: block on the physical
    # system so that CG4 selections sharing a host are one block
    work["Group"] = work["physical_group"] if "physical_group" in work else work["group_uid"]
    for column in ("Dn4000", "log_Rchl_r_kpc", "log_petroR50_kpc"):
        if column not in work:
            work[column] = np.nan
    return work


def compute_property_trends(frame: pd.DataFrame) -> pd.DataFrame:
    work = _prepare(frame)
    rows: list[dict] = []
    for scope in SCOPES:
        for sample_name in SAMPLES:
            part = work.loc[work["sample"] == sample_name]
            for panel, column, statistic in PANELS + UNPLOTTED_PANELS:
                rows.extend(
                    _mass_bin_rows(
                        part,
                        sample_name=sample_name,
                        panel=panel,
                        value_col=column,
                        statistic=statistic,
                        bins=MORPHOLOGY_MASS_BINS,
                        scope=scope,
                    )
                )
    return pd.DataFrame.from_records(rows)


def plot_property_trends(trends: pd.DataFrame, filename: str) -> None:
    """Four properties (rows) for BGGs (left) and satellites (right)."""

    # explicit default style: an earlier module may have set a seaborn style
    with plt.style.context("default"):
        fig, axes = plt.subplots(
            len(FIGURE_ROWS), len(FIGURE_SCOPES), figsize=(7.1, 8.0),
            sharex=True, sharey="row",
        )
        displayed = trends.loc[trends["displayed"]]
        letters = iter("abcdefgh")
        for row, (panel, title, ylabel, ylim) in enumerate(FIGURE_ROWS):
            for col, (scope, scope_label) in enumerate(FIGURE_SCOPES):
                ax = axes[row, col]
                draw_binned_series(
                    ax,
                    displayed.loc[
                        (displayed["panel"] == panel) & (displayed["scope"] == scope)
                    ],
                )
                ax.set_title(f"({next(letters)}) {scope_label}: {title}", fontsize=9)
                if ylim is not None:
                    ax.set_ylim(*ylim)
            axes[row, 0].set_ylabel(ylabel, fontsize=9)
        for ax in axes[:, -1]:
            ax.tick_params(axis="y", labelright=True)
        # the shared legend is read from panel (a); add an empty entry for any
        # sample that has no displayed point there
        _, present = axes[0, 0].get_legend_handles_labels()
        for sample_name in SAMPLES:
            if SAMPLE_LABELS[sample_name] not in present:
                style = SAMPLE_STYLES[sample_name]
                axes[0, 0].plot(
                    [], [], color=style["colour"], linestyle="none",
                    marker=style["marker"], markersize=4.3,
                    label=SAMPLE_LABELS[sample_name],
                )
        finish_binned_figure(fig, axes, filename, ylabel="")


def run_descriptive_properties(data, output_dir: str | None = None) -> dict:
    frame = ensure_galaxy_frame(data)
    if frame.empty:
        return {"status": "skipped", "reason": "no_galaxy_frame"}
    trends = compute_property_trends(frame)
    result = {
        "status": "ok",
        "bins": MORPHOLOGY_MASS_BINS.tolist(),
        "panels": [
            {"panel": panel, "column": column, "statistic": statistic}
            for panel, column, statistic in PANELS
        ],
        "unplotted_panels": [
            {"panel": panel, "column": column, "statistic": statistic}
            for panel, column, statistic in UNPLOTTED_PANELS
        ],
        "scopes": list(SCOPES),
        "blocking_unit": "physical_group",
        "strong_halpha_threshold_angstrom": STRONG_HALPHA_EW,
        "rows": trends.to_dict(orient="records"),
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "fig_property_trends.pdf")
        plot_property_trends(trends, path)
        result["figure"] = os.path.basename(path)
        trends.to_csv(os.path.join(co.OUTPUT_PATH, "descriptive_property_trends.csv"), index=False)
        result["rows_file"] = "descriptive_property_trends.csv"
    return safe_json(result)
