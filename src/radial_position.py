"""Projected BGG-centric position and the satellite morphology contrast.

Compact-group satellites sit much closer to their BGG in projection than the
satellites of any control sample, and in every sample the GZ1 E-class
fraction falls with projected distance from the BGG.  This module asks how
much of the satellite contrast remains at a common projected position.

Two quantities are reported for each control, both from binomial GLMs on the
satellite rows of CG4 plus that control with the Table 4 covariates
(log M*, redshift, quartet luminosity, sigma_v; cluster-robust by physical
Lim group):

* the total contrast (no position term), and
* the contrast at a common projected position, adding log R_BGG (primary),
  or log(R_BGG/r_200) of the host halo, or a restricted cubic spline in
  log R_BGG, and refitting on the radial common support (control satellites
  no farther from their BGG than the most distant CG4 satellite).

Besides odds ratios, each model is standardised to the CG4 satellites: the
control relation evaluated at the masses, redshifts, quartet properties and
radii of the CG4 satellites predicts the E-class fraction those galaxies
would have in the control environment (``extended_stats.standardized_contrast``).
Projected radius is partly fixed by the compact-group selection itself, so
the position-adjusted contrast is a different quantity from the total one,
not a bound on it.
"""

from __future__ import annotations

import os

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from descriptive_trends import SAMPLE_LABELS, SAMPLE_STYLES, _group_blocked_interval
    from extended_data import ensure_galaxy_frame
    from extended_stats import (
        fit_logistic_model,
        overlap_coefficient,
        restricted_cubic_spline,
        safe_json,
        standardized_contrast,
    )
except ModuleNotFoundError:  # pragma: no cover
    from .descriptive_trends import SAMPLE_LABELS, SAMPLE_STYLES, _group_blocked_interval
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import (
        fit_logistic_model,
        overlap_coefficient,
        restricted_cubic_spline,
        safe_json,
        standardized_contrast,
    )

CONTROLS = ["Control4B", "Control4C", "RG4"]
SAMPLES = ["CG4", *CONTROLS]
BASE_COVARIATES = [
    "logMstar",
    "z_numeric",
    "log_group_luminosity",
    "velocity_dispersion",
]
# Fixed before looking at the between-sample fractions: roughly factor-of-two
# steps in projected radius, the outermost bin collecting the control tails.
RADIAL_BINS_KPC = np.array([0.0, 75.0, 150.0, 300.0, 600.0, 3000.0])
SPECIFICATIONS = {
    "none": [],
    "log_r": ["log_R_bgg"],
    "log_r_over_r200": ["log_R_over_r200"],
    "spline_log_r": ["log_R_bgg", "log_R_bgg_rcs"],
}
PRIMARY_SPECIFICATION = "log_r"
N_BOOT = 2000


def satellite_radial_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Satellite rows with projected BGG-centric radius columns."""

    satellites = frame.loc[frame["is_satellite"] == 1].copy()
    radius = pd.to_numeric(satellites["dist2BGG_kpc"], errors="coerce")
    r200 = pd.to_numeric(satellites.get("r_200_kpc"), errors="coerce")
    satellites["R_bgg_kpc"] = radius
    satellites["log_R_bgg"] = np.log10(radius.where(radius > 0))
    satellites["log_R_over_r200"] = np.log10((radius / r200).where((radius > 0) & (r200 > 0)))
    return satellites


def support_summary(satellites: pd.DataFrame) -> dict:
    """Radial distributions and their overlap with the CG4 satellites."""

    cg4 = satellites.loc[satellites["sample"] == "CG4", "R_bgg_kpc"].dropna()
    summary = {
        "cg4_max_kpc": float(cg4.max()),
        "quantiles_kpc": {},
        "control_fraction_within_cg4_max": {},
        "overlap_log_r": {},
    }
    for sample_name in SAMPLES:
        values = satellites.loc[satellites["sample"] == sample_name, "R_bgg_kpc"].dropna()
        summary["quantiles_kpc"][sample_name] = {
            str(q): float(values.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        }
        if sample_name != "CG4":
            summary["control_fraction_within_cg4_max"][sample_name] = float(
                (values <= cg4.max()).mean()
            )
            summary["overlap_log_r"][sample_name] = overlap_coefficient(
                np.log10(cg4), np.log10(values)
            )
    return summary


def _model_frame(satellites: pd.DataFrame, control: str) -> pd.DataFrame:
    work = satellites.loc[satellites["sample"].isin(["CG4", control])].copy()
    knots = np.nanquantile(work["log_R_bgg"], [0.1, 0.5, 0.9])
    work["log_R_bgg_rcs"] = restricted_cubic_spline(work["log_R_bgg"], knots)
    work.attrs["spline_knots_log_kpc"] = [float(k) for k in knots]
    return work


def morphology_models(satellites: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """E-versus-S satellite models with and without projected position."""

    results = {}
    for control in CONTROLS:
        work = _model_frame(satellites, control)
        cg4_max = work.loc[work["is_CG4"] == 1, "R_bgg_kpc"].max()
        entries = {"spline_knots_log_kpc": work.attrs["spline_knots_log_kpc"]}
        for name, extra in SPECIFICATIONS.items():
            predictors = ["is_CG4", *BASE_COVARIATES, *extra]
            entries[name] = standardized_contrast(
                work,
                "elliptical",
                predictors,
                continuous=[*BASE_COVARIATES, *extra],
                n_boot=n_boot,
            )
        support = work.loc[(work["is_CG4"] == 1) | (work["R_bgg_kpc"] <= cg4_max)].copy()
        entries["common_support_log_r"] = standardized_contrast(
            support,
            "elliptical",
            ["is_CG4", *BASE_COVARIATES, *SPECIFICATIONS[PRIMARY_SPECIFICATION]],
            continuous=[*BASE_COVARIATES, *SPECIFICATIONS[PRIMARY_SPECIFICATION]],
            n_boot=n_boot,
        )
        entries["common_support_none"] = standardized_contrast(
            support,
            "elliptical",
            ["is_CG4", *BASE_COVARIATES],
            continuous=BASE_COVARIATES,
            n_boot=n_boot,
        )
        entries["common_support_max_kpc"] = float(cg4_max)
        results[control] = entries
    return results


def quenched_models(satellites: pd.DataFrame) -> dict:
    """Quenched-versus-star-forming satellite odds ratios with and without position."""

    results = {}
    for control in CONTROLS:
        work = _model_frame(satellites, control)
        results[control] = {
            name: fit_logistic_model(
                work,
                "quenched",
                ["is_CG4", *BASE_COVARIATES, *extra],
                continuous=[*BASE_COVARIATES, *extra],
            )
            for name, extra in SPECIFICATIONS.items()
        }
    return results


def binned_fractions(satellites: pd.DataFrame) -> list[dict]:
    """Satellite E-class fraction in projected-radius bins, per sample."""

    rows = []
    work = satellites.dropna(subset=["elliptical", "R_bgg_kpc"]).copy()
    work["radial_bin"] = pd.cut(work["R_bgg_kpc"], RADIAL_BINS_KPC, right=True)
    for sample_name in SAMPLES:
        part = work.loc[work["sample"] == sample_name]
        for index, interval in enumerate(work["radial_bin"].cat.categories):
            current = part.loc[part["radial_bin"] == interval]
            if current.empty:
                continue
            estimate, low, high, n_gal, n_groups = _group_blocked_interval(
                current,
                value_col="elliptical",
                statistic="fraction",
                random_state=20260927 + 17 * index + SAMPLES.index(sample_name),
            )
            rows.append(
                {
                    "sample": sample_name,
                    "bin_left_kpc": float(interval.left),
                    "bin_right_kpc": float(interval.right),
                    "median_R_kpc": float(current["R_bgg_kpc"].median()),
                    "estimate": estimate,
                    "ci16": low,
                    "ci84": high,
                    "n_galaxies": n_gal,
                    "n_groups": n_groups,
                    "displayed": bool(n_gal >= 5 and n_groups >= 3),
                }
            )
    return rows


def plot_radial_figure(satellites: pd.DataFrame, rows: list[dict], path: str) -> str:
    """E-class fraction and radial distribution of satellites by sample."""

    table = pd.DataFrame(rows)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(3.5, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    offsets = dict(zip(SAMPLES, (0.985, 0.995, 1.005, 1.015)))
    for sample_name in SAMPLES:
        style = SAMPLE_STYLES[sample_name]
        part = table.loc[(table["sample"] == sample_name) & table["displayed"]]
        if not part.empty:
            estimate = part["estimate"].to_numpy(dtype=float)
            top.errorbar(
                part["median_R_kpc"] * offsets[sample_name],
                estimate,
                yerr=[estimate - part["ci16"], part["ci84"] - estimate],
                color=style["colour"],
                marker=style["marker"],
                markersize=4.3,
                linestyle="-",
                linewidth=0.8,
                capsize=2.0,
                label=SAMPLE_LABELS[sample_name],
            )
        radii = satellites.loc[satellites["sample"] == sample_name, "R_bgg_kpc"].dropna()
        bottom.hist(
            radii,
            bins=np.logspace(0.5, 3.5, 25),
            density=False,
            weights=np.full(len(radii), 1.0 / max(len(radii), 1)),
            histtype="step",
            color=style["colour"],
            linestyle=style["linestyle"],
            linewidth=1.3,
        )
    top.set_xscale("log")
    top.set_ylim(0.0, 1.0)
    top.set_ylabel(r"$N_{\rm E}/(N_{\rm E}+N_{\rm S})$, satellites", fontsize=9)
    top.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    bottom.set_xlabel(r"Projected distance to the BGG, $R_{\rm BGG}$ (kpc)", fontsize=9)
    bottom.set_ylabel("Fraction", fontsize=9)
    bottom.set_xlim(3.0, 3000.0)
    for axis in (top, bottom):
        axis.tick_params(labelsize=8, direction="in", which="both", top=True, right=True)
    fig.tight_layout(h_pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def run_radial_position_analysis(data, output_dir: str | None = None, n_boot: int = N_BOOT):
    """Run the projected-position analysis and draw its figure."""

    frame = ensure_galaxy_frame(data)
    satellites = satellite_radial_frame(frame)
    rows = binned_fractions(satellites)
    result = {
        "status": "ok",
        "population": "satellites",
        "covariates": BASE_COVARIATES,
        "radius_definition": (
            "projected distance to the quartet BGG in kpc (Planck 2015 "
            "angular-diameter distance); r_200 of the identified Lim host halo"
        ),
        "primary_specification": PRIMARY_SPECIFICATION,
        "radial_bins_kpc": RADIAL_BINS_KPC.tolist(),
        "support": support_summary(satellites),
        "satellite_E_fraction": {
            sample_name: {
                "n_classified": int(part["elliptical"].notna().sum()),
                "fraction": float(part["elliptical"].mean()),
            }
            for sample_name, part in satellites.groupby("sample")
        },
        "morphology": morphology_models(satellites, n_boot=n_boot),
        "quenched": quenched_models(satellites),
        "binned_fractions": rows,
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result["figure"] = plot_radial_figure(
            satellites, rows, os.path.join(output_dir, "fig_radial_position.pdf")
        )
    return safe_json(result)
