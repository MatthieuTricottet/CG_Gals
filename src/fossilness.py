"""Magnitude-gap and fossil-like assembly diagnostics."""

from __future__ import annotations

import os

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from extended_data import ensure_galaxy_frame
    from extended_stats import (
        fit_logistic_model,
        holm_correction,
        magnitude_gap,
        safe_json,
        two_sample_summary,
    )
except ModuleNotFoundError:  # pragma: no cover
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import (
        fit_logistic_model,
        holm_correction,
        magnitude_gap,
        safe_json,
        two_sample_summary,
    )


def _group_summary(frame):
    rows = []
    for group_uid, group in frame.groupby("group_uid", observed=True):
        magnitudes = (
            pd.to_numeric(group["M_r"], errors="coerce")
            .dropna()
            .sort_values()
            .to_numpy()
        )
        satellites = group.loc[group["is_satellite"] == 1]
        bgg = group.loc[group["is_bgg"] == 1]
        first = group.iloc[0]
        rows.append(
            {
                "group_uid": group_uid,
                "sample": first["sample"],
                "is_CG4": first["is_CG4"],
                "Delta_m12": magnitude_gap(magnitudes),
                "Delta_m14": (
                    float(magnitudes[3] - magnitudes[0])
                    if len(magnitudes) >= 4
                    else np.nan
                ),
                "quenched_satellite_fraction": satellites["quenched"].mean(),
                "elliptical_satellite_fraction": satellites["elliptical"].mean(),
                "bgg_elliptical": bgg["elliptical"].mean(),
                "crossing_time": pd.to_numeric(
                    pd.Series([first.get("t_cr", first.get("group_t_cr", np.nan))]),
                    errors="coerce",
                ).iloc[0],
                "velocity_dispersion": first.get("velocity_dispersion", np.nan),
                "group_luminosity": first.get("log_group_luminosity", np.nan),
                "virial_mass_to_light": first.get(
                    "M_virial_over_L", first.get("group_M_virial_over_L", np.nan)
                ),
                "bgg_luminosity_fraction": first.get(
                    "FracLumBGG", first.get("group_FracLumBGG", np.nan)
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot(groups, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, gap, label in zip(
        axes, ["Delta_m12", "Delta_m14"], [r"$\Delta m_{12}$", r"$\Delta m_{14}$"]
    ):
        for is_cg4, name, colour in [(1, "CG4", "#2864A6"), (0, "Controls", "#777777")]:
            values = groups.loc[groups["is_CG4"] == is_cg4, gap].dropna()
            if len(values):
                ax.hist(
                    values,
                    bins=18,
                    density=True,
                    histtype="step",
                    linewidth=2,
                    label=name,
                    color=colour,
                )
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def _plot_fraction(groups, path):
    clean = groups[["Delta_m12", "quenched_satellite_fraction", "is_CG4"]].dropna()
    if len(clean) < 10:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for is_cg4, name, colour in [(1, "CG4", "#2864A6"), (0, "Controls", "#777777")]:
        part = clean.loc[clean["is_CG4"] == is_cg4]
        ax.scatter(
            part["Delta_m12"],
            part["quenched_satellite_fraction"],
            s=18,
            alpha=0.55,
            label=name,
            color=colour,
        )
    ax.set_xlabel(r"$\Delta m_{12}$")
    ax.set_ylabel("Quenched satellite fraction")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def run_fossilness_analysis(data, output_dir: str | None = None):
    """Compare magnitude gaps and test whether they mediate CG4 outcomes."""

    frame = ensure_galaxy_frame(data)
    if "M_r" not in frame or "group_uid" not in frame:
        return {
            "status": "skipped",
            "reason": "missing_required_columns",
            "missing_columns": ["M_r", "group_uid"],
        }
    groups = _group_summary(frame)
    comparisons = {}
    for gap in ["Delta_m12", "Delta_m14"]:
        comparisons[gap] = two_sample_summary(
            groups.loc[groups["is_CG4"] == 1, gap],
            groups.loc[groups["is_CG4"] == 0, gap],
        )
    valid = [gap for gap, value in comparisons.items() if value.get("status") == "ok"]
    for gap, adjusted in zip(
        valid, holm_correction([comparisons[gap]["mannwhitney_p"] for gap in valid])
    ):
        comparisons[gap]["p_adj"] = adjusted

    correlations = {}
    correlation_p = []
    correlation_keys = []
    outcomes = [
        "quenched_satellite_fraction",
        "elliptical_satellite_fraction",
        "bgg_elliptical",
        "crossing_time",
        "velocity_dispersion",
        "group_luminosity",
        "virial_mass_to_light",
    ]
    for gap in ["Delta_m12", "Delta_m14"]:
        for outcome in outcomes:
            clean = groups[[gap, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
            key = f"{gap}_vs_{outcome}"
            if len(clean) < 8 or clean[outcome].nunique() < 2:
                correlations[key] = {
                    "status": "skipped",
                    "reason": "too_few_complete_groups",
                }
                continue
            rho, p_value = stats.spearmanr(clean[gap], clean[outcome])
            correlations[key] = {
                "status": "ok",
                "n": int(len(clean)),
                "spearman_rho": float(rho),
                "p": float(p_value),
            }
            correlation_p.append(p_value)
            correlation_keys.append(key)
    for key, adjusted in zip(correlation_keys, holm_correction(correlation_p)):
        correlations[key]["p_adj"] = adjusted

    gap_map = groups.set_index("group_uid")["Delta_m12"]
    model_frame = frame.copy()
    model_frame["Delta_m12"] = model_frame["group_uid"].map(gap_map)
    predictors = ["is_CG4", "Delta_m12", "logMstar", "is_satellite"]
    models = {
        outcome: fit_logistic_model(
            model_frame,
            outcome,
            predictors,
            continuous=["Delta_m12", "logMstar"],
        )
        for outcome in ["quenched", "elliptical"]
    }
    result = {
        "status": "ok",
        "n_groups": int(len(groups)),
        "gap_definition": "M_r,2 - M_r,1 and M_r,4 - M_r,1 after sorting brightest first",
        "sample_comparisons": comparisons,
        "correlations": correlations,
        "models_with_gap": models,
        "magnitude_gap_significant": any(
            comparisons[gap].get("p_adj", 1) < 0.05 for gap in valid
        ),
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result["figure"] = _plot(
            groups, os.path.join(output_dir, "fig_magnitude_gap_comparison.pdf")
        )
        result["quenched_fraction_figure"] = _plot_fraction(
            groups,
            os.path.join(output_dir, "fig_magnitude_gap_vs_quenched_fraction.pdf"),
        )
    return safe_json(result)
