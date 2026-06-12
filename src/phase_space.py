"""Projected phase-space diagnostics for satellite galaxies."""

from __future__ import annotations

import os

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

try:
    from extended_data import ensure_galaxy_frame
    from extended_stats import fit_logistic_model, holm_correction, safe_json
except ModuleNotFoundError:  # pragma: no cover
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import fit_logistic_model, holm_correction, safe_json


REGIONS = {
    "inner_low_velocity": lambda frame: (frame["R_norm"] <= 1) & (frame["V_norm"] <= 1),
    "inner_high_velocity": lambda frame: (frame["R_norm"] <= 1) & (frame["V_norm"] > 1),
    "outer_low_velocity": lambda frame: (frame["R_norm"] > 1) & (frame["V_norm"] <= 1),
    "outer_high_velocity": lambda frame: (frame["R_norm"] > 1) & (frame["V_norm"] > 1),
}


def _fraction(values):
    clean = values.dropna()
    return float(clean.mean()) if len(clean) else None


def _plot(panel, bin_results, path):
    if panel.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, is_cg4, label in zip(axes, [1, 0], ["CG4", "Controls"]):
        part = panel.loc[panel["is_CG4"] == is_cg4]
        colours = np.where(part["passive"].eq(1), "#A74752", "#2864A6")
        ax.scatter(
            part["R_norm"], part["V_norm"], c=colours, s=12, alpha=0.45, linewidth=0
        )
        ax.axvline(1, color="0.35", linestyle=":")
        ax.axhline(1, color="0.35", linestyle=":")
        ax.set_title(label)
        ax.set_xlabel(r"$R/\langle R_{ij}\rangle$")
        for region, x, y in [
            ("inner_low_velocity", 0.04, 0.06),
            ("inner_high_velocity", 0.04, 0.86),
            ("outer_low_velocity", 0.58, 0.06),
            ("outer_high_velocity", 0.58, 0.86),
        ]:
            fraction = bin_results[region]["cg4" if is_cg4 else "control"][
                "passive_fraction"
            ]
            text = "n/a" if fraction is None else f"$f_P={fraction:.2f}$"
            ax.text(x, y, text, transform=ax.transAxes, fontsize=9)
    axes[0].set_ylabel(r"$|\Delta v|/\sigma_v$")
    axes[0].set_xlim(left=0)
    axes[0].set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def run_phase_space_analysis(data, output_dir: str | None = None):
    """Compare CG4 and control satellites across projected phase space."""

    frame = ensure_galaxy_frame(data)
    required = ["R_norm", "V_norm", "is_satellite", "passive", "is_CG4"]
    missing = [column for column in required if column not in frame]
    if missing:
        return {
            "status": "skipped",
            "reason": "missing_required_columns",
            "missing_columns": missing,
        }
    panel = frame.loc[frame["is_satellite"] == 1].replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=["R_norm", "V_norm", "is_CG4"])
    if len(panel) < 30:
        return {
            "status": "skipped",
            "reason": "too_few_satellites",
            "n": int(len(panel)),
        }

    bin_results = {}
    p_values = []
    regions_with_tests = []
    for region, selector in REGIONS.items():
        part = panel.loc[selector(panel)]
        cg = part.loc[part["is_CG4"] == 1, "passive"].dropna()
        control = part.loc[part["is_CG4"] == 0, "passive"].dropna()
        comparison_p = None
        if len(cg) and len(control) and cg.nunique() + control.nunique() > 2:
            table = [
                [int(cg.sum()), int(len(cg) - cg.sum())],
                [int(control.sum()), int(len(control) - control.sum())],
            ]
            comparison_p = float(stats.fisher_exact(table).pvalue)
            p_values.append(comparison_p)
            regions_with_tests.append(region)
        bin_results[region] = {
            "cg4": {"n": int(len(cg)), "passive_fraction": _fraction(cg)},
            "control": {"n": int(len(control)), "passive_fraction": _fraction(control)},
            "fisher_p": comparison_p,
        }
    for region, adjusted in zip(regions_with_tests, holm_correction(p_values)):
        bin_results[region]["fisher_p_adj"] = adjusted

    panel = panel.copy()
    panel["R_x_V"] = panel["R_norm"] * panel["V_norm"]
    predictors = ["is_CG4", "R_norm", "V_norm", "R_x_V"]
    if "logMstar" in panel:
        predictors.append("logMstar")
    models = {
        "passive": fit_logistic_model(
            panel,
            "passive",
            predictors,
            continuous=[column for column in predictors if column != "is_CG4"],
        ),
        "elliptical": fit_logistic_model(
            panel,
            "elliptical",
            predictors,
            continuous=[column for column in predictors if column != "is_CG4"],
        ),
    }
    result = {
        "status": "ok",
        "n_satellites": int(len(panel)),
        "coordinates_used": {
            "radius": "dist2BGG_kpc / size_Group_Bary_kpc",
            "velocity": "abs(c * (z - z_group) / (1 + z_group)) / Vdisp",
        },
        "bin_thresholds": {"R_norm": 1.0, "V_norm": 1.0},
        "bin_results": bin_results,
        "logistic_models": models,
        "fixed_phase_space_cg4_significant": any(
            value.get("fisher_p_adj", 1) < 0.05 for value in bin_results.values()
        ),
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result["figure"] = _plot(
            panel,
            bin_results,
            os.path.join(output_dir, "fig_phase_space_satellites.pdf"),
        )
    return safe_json(result)
