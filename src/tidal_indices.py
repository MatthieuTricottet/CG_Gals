"""Pairwise projected tidal-interaction indicators."""

from __future__ import annotations

import os

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15
from matplotlib.lines import Line2D

try:
    from extended_data import C_KMS, ensure_galaxy_frame
    from extended_stats import fit_logistic_model, safe_json, two_sample_summary
except ModuleNotFoundError:  # pragma: no cover
    from .extended_data import C_KMS, ensure_galaxy_frame
    from .extended_stats import fit_logistic_model, safe_json, two_sample_summary


def _angular_matrix(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    delta_ra = ra[:, None] - ra[None, :]
    delta_dec = dec[:, None] - dec[None, :]
    a = (
        np.sin(delta_dec / 2) ** 2
        + np.cos(dec[:, None]) * np.cos(dec[None, :]) * np.sin(delta_ra / 2) ** 2
    )
    return 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _derive(frame):
    output = frame.copy()
    for column in [
        "nearest_projected_distance",
        "nearest_velocity_difference",
        "nearest_stellar_mass_ratio",
        "tidal_index_sum",
    ]:
        output[column] = np.nan
    for _, group in output.groupby("group_uid", observed=True):
        clean = group[["RA", "Dec", "z_numeric", "logMstar"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if len(group) < 2 or clean[["RA", "Dec", "z_numeric"]].isna().any(axis=None):
            continue
        angular = _angular_matrix(clean["RA"].to_numpy(), clean["Dec"].to_numpy())
        z_group = float(np.nanmedian(clean["z_numeric"]))
        distance_kpc = angular * Planck15.angular_diameter_distance(z_group).to_value(
            "kpc"
        )
        np.fill_diagonal(distance_kpc, np.inf)
        velocity = (
            C_KMS
            * np.abs(
                clean["z_numeric"].to_numpy()[:, None]
                - clean["z_numeric"].to_numpy()[None, :]
            )
            / (1 + z_group)
        )
        np.fill_diagonal(velocity, np.inf)
        nearest = np.argmin(distance_kpc, axis=1)
        masses = np.power(10.0, clean["logMstar"].to_numpy())
        mass_ratio = masses[nearest] / masses
        with np.errstate(divide="ignore", invalid="ignore"):
            tidal = np.nansum(masses[None, :] / distance_kpc**3, axis=1)
        output.loc[group.index, "nearest_projected_distance"] = distance_kpc[
            np.arange(len(group)), nearest
        ]
        output.loc[group.index, "nearest_velocity_difference"] = velocity[
            np.arange(len(group)), nearest
        ]
        output.loc[group.index, "nearest_stellar_mass_ratio"] = mass_ratio
        output.loc[group.index, "tidal_index_sum"] = tidal
    output["log_tidal_index"] = np.log10(
        output["tidal_index_sum"].where(output["tidal_index_sum"] > 0)
    )
    return output


def _plot(work, path):
    outcomes = [
        ("quenched", "Quenched status", "Not quenched", "Quenched"),
        ("elliptical", "Elliptical/smooth morphology", "Not smooth", "Smooth"),
    ]
    outcomes = [item for item in outcomes if item[0] in work]
    columns = ["log_tidal_index", "is_CG4"] + [item[0] for item in outcomes]
    clean = work[columns].dropna(subset=["log_tidal_index", "is_CG4"])
    if len(clean) < 20 or not outcomes:
        return None
    fig, axes = plt.subplots(1, len(outcomes), figsize=(4.2 * len(outcomes), 3.8), sharex=True)
    if len(outcomes) == 1:
        axes = [axes]
    rng = np.random.default_rng(20260612)
    for ax, (column, title, false_label, true_label) in zip(axes, outcomes):
        panel = clean[["log_tidal_index", "is_CG4", column]].dropna()
        for value, marker in [(1, "o"), (0, "^")]:
            part = panel.loc[panel[column] == value]
            ax.scatter(
                part["log_tidal_index"],
                np.full(len(part), value) + rng.normal(0, 0.025, len(part)),
                s=10,
                alpha=0.35,
                marker=marker,
                c=np.where(part["is_CG4"] == 1, "#2864A6", "#777777"),
            )
        ax.set_title(title)
        ax.set_xlabel(r"$\log_{10}\sum_j(M_{\star,j}/R_{ij}^3)$ [$\mathrm{M_\odot\,kpc^{-3}}$]")
        ax.set_yticks([0, 1], [false_label, true_label])
    axes[0].set_ylabel("Outcome")
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2864A6", label="CG$_4$", markersize=6),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#777777", label="Controls", markersize=6),
        ],
        frameon=False,
        loc="upper center",
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def run_tidal_indices_analysis(data, output_dir: str | None = None):
    """Compute physical projected pairwise indicators from RA, Dec, and redshift."""

    frame = ensure_galaxy_frame(data)
    missing = [
        column
        for column in ["RA", "Dec", "z_numeric", "logMstar", "group_uid"]
        if column not in frame
    ]
    if missing:
        return {
            "status": "skipped",
            "reason": "missing_required_columns",
            "missing_columns": missing,
        }
    work = _derive(frame)
    complete = work["nearest_projected_distance"].notna()
    if complete.sum() < 20:
        return {
            "status": "skipped",
            "reason": "too_few_reconstructable_pairs",
            "n": int(complete.sum()),
        }

    summary = {}
    for column in [
        "nearest_projected_distance",
        "nearest_velocity_difference",
        "nearest_stellar_mass_ratio",
        "log_tidal_index",
    ]:
        summary[column] = two_sample_summary(
            work.loc[work["is_CG4"] == 1, column],
            work.loc[work["is_CG4"] == 0, column],
        )
    baseline_predictors = ["is_CG4", "logMstar", "is_satellite"]
    tidal_predictors = [*baseline_predictors, "log_tidal_index"]
    models = {}
    for outcome in ["quenched", "elliptical"]:
        baseline = fit_logistic_model(
            work, outcome, baseline_predictors, continuous=["logMstar"]
        )
        adjusted = fit_logistic_model(
            work, outcome, tidal_predictors, continuous=["logMstar", "log_tidal_index"]
        )
        reduction = None
        if baseline.get("status") == "ok" and adjusted.get("status") == "ok":
            base_coef = baseline.get("cg4_coefficient")
            adjusted_coef = adjusted.get("cg4_coefficient")
            if base_coef not in (None, 0):
                reduction = float(1 - abs(adjusted_coef) / abs(base_coef))
        models[outcome] = {
            "baseline": baseline,
            "with_tidal_index": adjusted,
            "absolute_cg4_coefficient_reduction_fraction": reduction,
        }
    result = {
        "status": "ok",
        "columns_used": ["RA", "Dec", "z", "logMstar", "Group"],
        "units": {
            "nearest_projected_distance": "kpc proper",
            "nearest_velocity_difference": "km/s rest-frame",
            "tidal_index_sum": "M_sun/kpc^3",
        },
        "n_galaxies_with_pairs": int(complete.sum()),
        "summary_by_sample": summary,
        "models": models,
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result["figure"] = _plot(
            work, os.path.join(output_dir, "fig_tidal_index_outcomes.pdf")
        )
    return safe_json(result)
