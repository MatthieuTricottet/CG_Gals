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
    from utils import labels_utils as lu
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from . import generate_report as report
    from . import sSFR
    from .utils import labels_utils as lu


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
SAMPLE_LABELS = {name: lu.sample_tex_label(name) for name in SAMPLES}


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
    elif statistic in ("fraction", "mean"):
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
    # Top row (gary-r2 A1): the fraction classified elliptical (spiral) among
    # usable E/S classifications replaces the median debiased vote, whose
    # bimodality made the medians collapse toward 0/1; the mean debiased
    # votes over all galaxies with finite votes are kept as open markers.
    specifications = (
        ("elliptical_fraction", "is_elliptical", "fraction", MORPHOLOGY_MASS_BINS, "all"),
        ("spiral_fraction", "is_spiral", "fraction", MORPHOLOGY_MASS_BINS, "all"),
        ("elliptical_vote_mean", "p_E", "mean", MORPHOLOGY_MASS_BINS, "all"),
        ("spiral_vote_mean", "p_S", "mean", MORPHOLOGY_MASS_BINS, "all"),
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
        usable = frame["morphology"].isin(co.Morphologies[:2])
        frame["is_elliptical"] = np.where(
            usable, (frame["morphology"] == co.Morphologies[0]).astype(float), np.nan
        )
        frame["is_spiral"] = np.where(
            usable, (frame["morphology"] == co.Morphologies[1]).astype(float), np.nan
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


def draw_binned_series(
    ax,
    subset: pd.DataFrame,
    *,
    filled: bool = True,
    with_errors: bool = True,
    label: bool = True,
    x_offset: float = 0.0,
) -> None:
    """Draw one panel of per-sample binned estimates (shared Fig. 2 style)."""

    for sample_name in SAMPLES:
        current = subset.loc[subset["sample"] == sample_name].sort_values(
            "mass_location"
        )
        if current.empty:
            continue
        style = SAMPLE_STYLES[sample_name]
        estimate = current["estimate"].to_numpy(dtype=float)
        errors = None
        if with_errors:
            errors = np.vstack(
                [
                    estimate - current["ci16"].to_numpy(dtype=float),
                    current["ci84"].to_numpy(dtype=float) - estimate,
                ]
            )
        ax.errorbar(
            current["mass_location"] + x_offset,
            estimate,
            yerr=errors,
            color=style["colour"],
            linestyle="none",
            marker=style["marker"],
            markersize=4.3,
            markerfacecolor=style["colour"] if filled else "white",
            markeredgewidth=1.0,
            capsize=2.0 if with_errors else 0.0,
            alpha=1.0 if filled else 0.8,
            label=SAMPLE_LABELS[sample_name] if label else None,
        )


def finish_binned_figure(fig, axes, filename: str, ylabel: str = "Fraction") -> None:
    """Common axis cosmetics, legend, and save for the binned figures."""

    for ax in axes.flat:
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$", fontsize=9)
    if ylabel:
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel, fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
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


def plot_mass_trends(trends: pd.DataFrame, filename: str) -> None:
    """Plot morphology and quenched fractions as one compact four-panel figure.

    Top row: fraction classified elliptical / spiral among usable E/S
    classifications (filled markers, group-blocked 16--84% intervals) with
    the mean debiased vote fraction over all galaxies with finite votes as
    open markers.  Bottom row: quenched fractions among valid-sSFR
    satellites and BGGs.
    """

    panel_order = (
        ("elliptical_fraction", "elliptical_vote_mean"),
        ("spiral_fraction", "spiral_vote_mean"),
        ("quenched_satellites", None),
        ("quenched_bggs", None),
    )
    titles = (
        r"(a) Fraction classified elliptical",
        r"(b) Fraction classified spiral",
        "(c) Quenched fraction: satellites",
        "(d) Quenched fraction: BGGs",
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.3), sharey=True)
    displayed = trends.loc[trends["displayed"]]
    for ax, (panel, overlay), title in zip(axes.flat, panel_order, titles):
        draw_binned_series(ax, displayed.loc[displayed["panel"] == panel])
        if overlay is not None:
            draw_binned_series(
                ax,
                displayed.loc[displayed["panel"] == overlay],
                filled=False,
                with_errors=False,
                label=False,
                x_offset=0.03,
            )
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-0.03, 1.03)
    finish_binned_figure(fig, axes, filename)


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


QUENCHED_RESIDUAL_COLUMN = "QS_res"


def quenched_sequence_residuals(sample: dict[str, pd.DataFrame]) -> dict:
    """Residuals of quenched galaxies about a quenched-sequence fit (Fig. D.1b).

    Mirrors the star-forming main-sequence procedure: polynomial orders 1--4
    are fitted to the SDSS non-AGN reference *quenched* galaxies in the
    log M*--log sSFR plane, the lowest order with the smallest five-fold
    cross-validated RMS is adopted, residuals are attached to every quenched
    galaxy of the four group samples (column ``QS_res``, NaN otherwise), and
    the CG4-minus-control median offsets are given with 68% group-bootstrap
    intervals and sign-crossing p-values (same code path as the SFMS case).
    """

    reference = sample["SDSS"]
    quenched = reference.loc[reference["sSFR_status"] == co.sSFR_status[0], ["lgm", "sSFR"]]
    x = pd.to_numeric(quenched["lgm"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(quenched["sSFR"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    diagnostics = sSFR._polyfit_order_diagnostics(x, y)
    best_cv = min(item["cv_rms"] for item in diagnostics)
    # lowest order whose CV RMS is within 1e-6 dex of the minimum (ties -> lower order)
    order = min(item["order"] for item in diagnostics if item["cv_rms"] <= best_cv + 1e-6)
    model = sSFR.fit_ssfr_vs_lgm_poly(quenched.loc[finite], order=order)
    sSFR.add_MS_residuals(
        sample,
        model,
        suffix=co.GASUFF,
        sf_value=co.sSFR_status[0],
        out_col=QUENCHED_RESIDUAL_COLUMN,
        non_sf_value=np.nan,
    )
    offsets = {}
    cg4 = sample["CG4" + co.GASUFF]
    for name in SAMPLES[1:]:
        res = sSFR._group_blocked_bootstrap_median_difference(
            cg4, sample[name + co.GASUFF], value_col=QUENCHED_RESIDUAL_COLUMN,
            random_state=20260612,
        )
        offsets[name] = {
            "delta_median": res["delta"],
            "CI_16": res["CI_16"],
            "CI_84": res["CI_84"],
            "CI_95_low": res["CI_95_low"],
            "CI_95_high": res["CI_95_high"],
            "p_value": res["p_value"],
            "n_CG4_galaxies": res.get("n_galaxies_a"),
            "n_control_galaxies": res.get("n_galaxies_b"),
            "n_CG4_groups": res.get("n_groups_a"),
            "n_control_groups": res.get("n_groups_b"),
        }
    summary = []
    for name in SAMPLES:
        values = pd.to_numeric(
            sample[name + co.GASUFF][QUENCHED_RESIDUAL_COLUMN], errors="coerce"
        ).dropna()
        summary.append({
            "sample": name,
            "n_quenched": int(len(values)),
            "median": float(values.median()),
            "q16": float(values.quantile(0.16)),
            "q84": float(values.quantile(0.84)),
        })
    return {
        "fitted_on": "SDSS non-AGN reference, GMM quenched class",
        "n_fit": int(finite.sum()),
        "selected_order": int(order),
        "selected_cv_rms": float(best_cv),
        "order_diagnostics": diagnostics,
        "coefficients_highest_first": [float(c) for c in model.coeffs],
        "residual_column": QUENCHED_RESIDUAL_COLUMN,
        "delta_sign_convention": "median(CG4) - median(control)",
        "interval_16_84_level": 0.68,
        "offsets": offsets,
        "per_sample": summary,
        "caveat": ("MPA-JHU sSFRs of quenched galaxies are largely D4000-calibrated, "
                   "so residuals about the quenched sequence are weakly informative"),
    }


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
    quenched_sequence = quenched_sequence_residuals(sample)
    sSFR.plot_residual_ecdf_panels(
        sample,
        panels=(
            ("MS_res", "(a) Star-forming galaxies, about the SFMS"),
            (QUENCHED_RESIDUAL_COLUMN, "(b) Quenched galaxies, about the quenched sequence"),
        ),
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
        "quenched_sequence": quenched_sequence,
        "morphology_statistic": (
            "fraction classified elliptical/spiral among usable E/S classifications; "
            "mean debiased vote fractions over galaxies with finite votes as open markers"
        ),
    }
