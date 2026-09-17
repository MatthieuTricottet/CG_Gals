"""Descriptive stellar-mass trends requested for the revised manuscript.

This module does not fit inferential models.  It bins the existing catalogue
measurements, uses physical-group resampling for descriptive intervals, and
redraws two figures from already adopted classifications and residuals.
"""

from __future__ import annotations

import json
import os
import zlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import config as co
    import generate_report as report
    import sSFR
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from . import generate_report as report
    from . import sSFR


SAMPLES = ("CG4", "Control4B", "Control4C", "RG4")

# Fixed before examining any between-sample differences.  The 0.5-dex
# interior bins are retained, while the sparsely populated mass tails are
# merged so that the CG4 medians remain interpretable.
MORPHOLOGY_MASS_BINS = np.array([7.0, 9.5, 10.0, 10.5, 11.0, 12.5])
QUENCHED_MASS_BINS = np.array([7.0, 10.0, 10.5, 11.0, 12.5])
MIN_GALAXIES_TO_PLOT = 5
MIN_GROUPS_TO_PLOT = 3
N_BOOT = 2000
CI_QUANTILES = (0.16, 0.84)


SAMPLE_STYLES = {
    "CG4": {"colour": "#000000", "linestyle": "-", "marker": "o"},
    "Control4B": {"colour": "#0072B2", "linestyle": "--", "marker": "s"},
    "Control4C": {"colour": "#D55E00", "linestyle": "-.", "marker": "^"},
    "RG4": {"colour": "#009E73", "linestyle": ":", "marker": "D"},
}
SAMPLE_LABELS = {
    "CG4": r"CG$_4$",
    "Control4B": r"Control$_{4B}$",
    "Control4C": r"Control$_{4C}$",
    "RG4": r"RG$_4$",
}


def _seed(*parts: object) -> int:
    """Return a stable seed for one plotted sample/bin combination."""

    return zlib.crc32("|".join(map(str, parts)).encode("utf-8"))


def _group_blocked_interval(
    frame: pd.DataFrame,
    *,
    value_col: str,
    statistic: str,
    random_state: int,
    n_boot: int = N_BOOT,
) -> tuple[float, float, float, int, int]:
    """Estimate a median or binary fraction with a group-blocked interval."""

    work = frame[["Group", value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    groups = [
        part[value_col].to_numpy(dtype=float)
        for _, part in work.groupby("Group", observed=True)
        if len(part)
    ]
    values = np.concatenate(groups) if groups else np.array([], dtype=float)
    n_galaxies = int(values.size)
    n_groups = int(len(groups))
    if n_galaxies == 0:
        return np.nan, np.nan, np.nan, n_galaxies, n_groups

    if statistic == "median":
        estimate = float(np.median(values))
        summarise = np.median
    elif statistic == "fraction":
        estimate = float(np.mean(values))
        summarise = np.mean
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"Unknown statistic: {statistic}")

    rng = np.random.default_rng(random_state)
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        selected = rng.integers(0, n_groups, n_groups)
        resample = np.concatenate([groups[group] for group in selected])
        draws[index] = summarise(resample)
    low, high = np.quantile(draws, CI_QUANTILES)
    return estimate, float(low), float(high), n_galaxies, n_groups


def _mass_bin_rows(
    frame: pd.DataFrame,
    *,
    sample_name: str,
    panel: str,
    value_col: str,
    statistic: str,
    bins: np.ndarray,
    scope: str,
) -> list[dict]:
    """Return machine-readable rows for one panel and sample."""

    work = frame.copy()
    rank = pd.to_numeric(work["rank_M"], errors="coerce")
    if scope == "satellites":
        work = work.loc[rank.gt(1)].copy()
    elif scope == "bggs":
        work = work.loc[rank.eq(1)].copy()
    elif scope != "all":  # pragma: no cover - guarded by callers
        raise ValueError(f"Unknown scope: {scope}")

    work["lgm"] = pd.to_numeric(work["lgm"], errors="coerce")
    work["mass_bin"] = pd.cut(
        work["lgm"], bins=bins, right=False, include_lowest=True
    )
    rows = []
    categories = work["mass_bin"].cat.categories
    for bin_index, interval in enumerate(categories):
        current = work.loc[work["mass_bin"] == interval].copy()
        current[value_col] = pd.to_numeric(current[value_col], errors="coerce")
        contributors = current.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["lgm", value_col]
        )
        estimate, low, high, n_galaxies, n_groups = _group_blocked_interval(
            contributors,
            value_col=value_col,
            statistic=statistic,
            random_state=_seed(sample_name, panel, bin_index),
        )
        rows.append(
            {
                "panel": panel,
                "sample": sample_name,
                "scope": scope,
                "value_column": value_col,
                "statistic": statistic,
                "bin_left": float(interval.left),
                "bin_right": float(interval.right),
                "bin_centre": float((interval.left + interval.right) / 2),
                "mass_location": float(contributors["lgm"].median()),
                "n_galaxies": n_galaxies,
                "n_groups": n_groups,
                "estimate": estimate,
                "ci16": low,
                "ci84": high,
                "displayed": bool(
                    n_galaxies >= MIN_GALAXIES_TO_PLOT
                    and n_groups >= MIN_GROUPS_TO_PLOT
                ),
                "interval_method": (
                    f"{N_BOOT}-draw physical-group-blocked bootstrap, "
                    "16th--84th percentiles"
                ),
            }
        )
    return rows


def compute_mass_trends(sample: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute all four descriptive mass-trend panels."""

    rows: list[dict] = []
    specifications = (
        ("elliptical_vote", "p_E", "median", MORPHOLOGY_MASS_BINS, "all"),
        ("spiral_vote", "p_S", "median", MORPHOLOGY_MASS_BINS, "all"),
        (
            "quenched_satellites",
            "is_quenched",
            "fraction",
            QUENCHED_MASS_BINS,
            "satellites",
        ),
        (
            "quenched_bggs",
            "is_quenched",
            "fraction",
            QUENCHED_MASS_BINS,
            "bggs",
        ),
    )
    for sample_name in SAMPLES:
        frame = sample[sample_name + co.GASUFF].copy()
        frame["is_quenched"] = np.where(
            frame["sSFR_status"].isin(co.sSFR_status),
            (frame["sSFR_status"] == co.sSFR_status[0]).astype(float),
            np.nan,
        )
        for panel, value_col, statistic, bins, scope in specifications:
            rows.extend(
                _mass_bin_rows(
                    frame,
                    sample_name=sample_name,
                    panel=panel,
                    value_col=value_col,
                    statistic=statistic,
                    bins=bins,
                    scope=scope,
                )
            )
    return pd.DataFrame.from_records(rows)


def plot_mass_trends(trends: pd.DataFrame, filename: str) -> None:
    """Plot morphology and quenched fractions as one compact four-panel figure."""

    panel_order = (
        "elliptical_vote",
        "spiral_vote",
        "quenched_satellites",
        "quenched_bggs",
    )
    titles = (
        r"(a) Median $p_{\rm el,debiased}$",
        r"(b) Median $p_{\rm cs,debiased}$",
        "(c) Quenched fraction: satellites",
        "(d) Quenched fraction: BGGs",
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.3), sharey=True)
    for ax, panel, title in zip(axes.flat, panel_order, titles):
        subset = trends.loc[(trends["panel"] == panel) & trends["displayed"]]
        for sample_name in SAMPLES:
            current = subset.loc[subset["sample"] == sample_name].sort_values(
                "mass_location"
            )
            if current.empty:
                continue
            style = SAMPLE_STYLES[sample_name]
            estimate = current["estimate"].to_numpy(dtype=float)
            errors = np.vstack(
                [
                    estimate - current["ci16"].to_numpy(dtype=float),
                    current["ci84"].to_numpy(dtype=float) - estimate,
                ]
            )
            ax.errorbar(
                current["mass_location"],
                estimate,
                yerr=errors,
                color=style["colour"],
                linestyle="none",
                marker=style["marker"],
                markersize=4.3,
                markerfacecolor="white",
                markeredgewidth=1.0,
                capsize=2.0,
                label=SAMPLE_LABELS[sample_name],
            )
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[1, :]:
        ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.0, w_pad=1.0)
    fig.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _refresh_adopted_boundary_figure(sample: dict[str, pd.DataFrame]) -> None:
    """Redraw Fig. B.1 from the stored boundary without refitting the GMM."""

    with open(co.RESULTS_BUILD, encoding="utf-8") as stream:
        build = json.load(stream)
    boundary = report.decode_interp1d(build["sSFR_interp"])
    classified = sample["SDSS"].loc[
        sample["SDSS"]["sSFR_status"].isin(co.sSFR_status)
    ]
    sSFR.plot_classification(
        classified,
        sample["SDSS"],
        None,
        boundary,
        name="sSFR_classification",
    )


def _sfms_residual_summary(sample: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarise the already computed star-forming main-sequence residuals."""

    rows = []
    for sample_name in SAMPLES:
        values = pd.to_numeric(
            sample[sample_name + co.GASUFF]["MS_res"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "sample": sample_name,
                "n_star_forming": int(len(values)),
                "median": float(values.median()),
                "q10": float(values.quantile(0.10)),
                "q90": float(values.quantile(0.90)),
                "fraction_below_zero": float((values < 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def run(sample: dict[str, pd.DataFrame]) -> dict:
    """Generate descriptive figures and diagnostics for the manuscript."""

    os.makedirs(co.FIGURES_PATH, exist_ok=True)
    trends = compute_mass_trends(sample)
    trends_path = os.path.join(co.OUTPUT_PATH, "descriptive_mass_trends.csv")
    trends.to_csv(trends_path, index=False)
    plot_mass_trends(
        trends,
        os.path.join(co.FIGURES_PATH, "fig_mass_trends.pdf"),
    )

    _refresh_adopted_boundary_figure(sample)
    sSFR.plot_main_sequence_residuals(
        sample,
        figname="main_sequence_residuals",
    )
    residuals = _sfms_residual_summary(sample)
    residuals_path = os.path.join(co.OUTPUT_PATH, "sfms_residual_summary.csv")
    residuals.to_csv(residuals_path, index=False)

    return {
        "status": "ok",
        "figure": "fig_mass_trends.pdf",
        "morphology_mass_bins": MORPHOLOGY_MASS_BINS.tolist(),
        "quenched_mass_bins": QUENCHED_MASS_BINS.tolist(),
        "minimum_galaxies_displayed": MIN_GALAXIES_TO_PLOT,
        "minimum_groups_displayed": MIN_GROUPS_TO_PLOT,
        "bootstrap_draws": N_BOOT,
        "interval_quantiles": list(CI_QUANTILES),
        "x_coordinate": "median stellar mass of contributing galaxies",
        "connecting_lines": False,
        "counts_file": os.path.basename(trends_path),
        "sfms_residual_summary_file": os.path.basename(residuals_path),
        "sfms_residual_summary": residuals.to_dict(orient="records"),
    }
