"""Other galaxy properties versus stellar mass (descriptive figure, gary-r2 A4).

Four panels in the style and binning of the morphology row of Fig. 2, for
all galaxies of the four samples:

  (a) strong H-alpha emission fraction (EW <= -3 A, SDSS sign convention);
  (b) AGN fraction among BPT-classified galaxies (Sect. 3.7 scheme);
  (c) median D_n4000 (MPA-JHU galSpecIndx, Appendix A provenance);
  (d) median log R_e,r (seeing-corrected Simard half-light radius, kpc).

Purely descriptive: no model is fitted.  Intervals are group-blocked
bootstrap 16--84% percentiles exactly as in Fig. 2.
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
        SAMPLES,
        _mass_bin_rows,
        draw_binned_series,
        finish_binned_figure,
    )
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import safe_json

STRONG_HALPHA_EW = -3.0
PANELS = (
    ("strong_halpha", "strong_halpha", "fraction",
     "(a) Strong H$\\alpha$ emission fraction (EW $\\leq -3$ \u00c5)", (-0.03, 1.03)),
    ("agn_like", "agn_like", "fraction",
     "(b) AGN fraction (BPT-classified)", (-0.03, 1.03)),
    ("dn4000", "Dn4000", "median", r"(c) Median $D_n4000$", None),
    ("log_rchl", "log_Rchl_r_kpc", "median",
     r"(d) Median $\log_{10}(R_{e,r}/{\rm kpc})$", None),
)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    work = classify_bpt(frame)
    halpha = pd.to_numeric(work.get("h_alpha_eqw", np.nan), errors="coerce")
    work["strong_halpha"] = np.where(
        np.isfinite(halpha), (halpha <= STRONG_HALPHA_EW).astype(float), np.nan
    )
    # _mass_bin_rows expects the catalogue column names used by Fig. 2
    work["rank_M"] = work["rank"]
    work["lgm"] = work["logMstar"]
    if "Group" not in work:
        work["Group"] = work["group_uid"]
    for column in ("Dn4000", "log_Rchl_r_kpc"):
        if column not in work:
            work[column] = np.nan
    return work


def compute_property_trends(frame: pd.DataFrame) -> pd.DataFrame:
    work = _prepare(frame)
    rows: list[dict] = []
    for sample_name in SAMPLES:
        part = work.loc[work["sample"] == sample_name]
        for panel, column, statistic, _title, _ylim in PANELS:
            rows.extend(
                _mass_bin_rows(
                    part,
                    sample_name=sample_name,
                    panel=panel,
                    value_col=column,
                    statistic=statistic,
                    bins=MORPHOLOGY_MASS_BINS,
                    scope="all",
                )
            )
    return pd.DataFrame.from_records(rows)


def plot_property_trends(trends: pd.DataFrame, filename: str) -> None:
    # explicit default style: an earlier module may have set a seaborn style
    with plt.style.context("default"):
        fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.3))
        displayed = trends.loc[trends["displayed"]]
        for ax, (panel, _column, _statistic, title, ylim) in zip(axes.flat, PANELS):
            draw_binned_series(ax, displayed.loc[displayed["panel"] == panel])
            ax.set_title(title, fontsize=9)
            if ylim is not None:
                ax.set_ylim(*ylim)
        axes[0, 0].set_ylabel("Fraction", fontsize=9)
        axes[1, 0].set_ylabel("Median", fontsize=9)
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
            for panel, column, statistic, _t, _y in PANELS
        ],
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
