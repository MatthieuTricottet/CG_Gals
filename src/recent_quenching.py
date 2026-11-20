"""SDSS fibre H-alpha emission diagnostics."""

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
    from extended_stats import fit_logistic_model, safe_json
except ModuleNotFoundError:  # pragma: no cover
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import fit_logistic_model, safe_json


HALPHA_ALIASES = ["h_alpha_eqw", "H_alpha_eqw", "halpha_eqw"]


def _find(frame, aliases):
    return next((column for column in aliases if column in frame), None)


def _plot_halpha(frame, column, path):
    clean = frame[["sample", column]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bins = np.linspace(-60, 5, 35)
    for is_cg4, label, colour in [
        (True, "CG4", "#2864A6"),
        (False, "Controls", "#777777"),
    ]:
        values = clean.loc[clean["sample"].eq("CG4") == is_cg4, column].clip(-60, 5)
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=label,
            color=colour,
        )
    ax.axvline(-3, color="0.4", linestyle=":", label="strong-emission threshold")
    ax.set_xlabel(r"H$\alpha$ equivalent width ($\mathrm{\AA}$; emission is negative)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def _morphology_adjusted_models(frame, halpha):
    """Strong-H-alpha odds ratios at fixed stellar mass and GZ1 E/S class.

    CG4 against each control separately, all ranks, rows with a finite EW,
    stellar mass and an E or S class; stellar mass standardised; binomial GLM
    clustered by physical group.  The ``mass_class`` model is the one quoted
    in Sect. 3 (originally computed outside the pipeline, reproduced by the
    audit-r3 referee script a4c_halpha.py); ``mass_class_satellite`` adds
    satellite status as a sensitivity.
    """

    work = frame.copy()
    values = pd.to_numeric(work[halpha], errors="coerce")
    work["strong_halpha"] = np.where(
        np.isfinite(values), (values <= -3).astype(float), np.nan
    )
    specifications = {
        "mass_class": ["is_CG4", "logMstar", "elliptical"],
        "mass_class_satellite": ["is_CG4", "logMstar", "elliptical", "is_satellite"],
    }
    models = {}
    for control_name in ["Control4B", "Control4C", "RG4"]:
        part = work.loc[work["sample"].isin(["CG4", control_name])]
        models[control_name] = {
            name: fit_logistic_model(
                part, "strong_halpha", predictors, continuous=["logMstar"]
            )
            for name, predictors in specifications.items()
        }
    return models


def run_recent_quenching_analysis(data, output_dir: str | None = None):
    """Compare strong H-alpha emission across the group samples."""

    frame = ensure_galaxy_frame(data)
    halpha = _find(frame, HALPHA_ALIASES)
    if halpha is None:
        return {"status": "skipped", "reason": "missing_halpha_equivalent_width"}

    values = (
        frame[["sample", halpha]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    )
    values["strong_halpha_emission"] = values[halpha] <= -3
    fractions = {}
    for sample_name, part in values.groupby("sample", observed=True):
        fractions[sample_name] = {
            "n_with_halpha": int(len(part)),
            "n_strong_halpha": int(part["strong_halpha_emission"].sum()),
            "median_halpha_eqw": float(part[halpha].median()),
            "strong_emission_fraction": float(part["strong_halpha_emission"].mean()),
        }
    comparisons = {}
    cg = values.loc[values["sample"] == "CG4"]
    for control_name in ["Control4B", "Control4C", "RG4"]:
        control = values.loc[values["sample"] == control_name]
        key = f"CG4_vs_{control_name}"
        if cg.empty or control.empty:
            comparisons[key] = {"status": "skipped", "reason": "no_complete_cases"}
            continue
        table = [
            [
                int(cg["strong_halpha_emission"].sum()),
                int((~cg["strong_halpha_emission"]).sum()),
            ],
            [
                int(control["strong_halpha_emission"].sum()),
                int((~control["strong_halpha_emission"]).sum()),
            ],
        ]
        p_value = float(stats.fisher_exact(table).pvalue)
        comparisons[key] = {
            "status": "ok",
            "strong_emission_fraction_difference": float(
                cg["strong_halpha_emission"].mean()
                - control["strong_halpha_emission"].mean()
            ),
            "fisher_p": p_value,
        }

    result = {
        "status": "ok",
        "columns_used": [halpha],
        "classification_thresholds": {
            "strong_halpha_emission": r"\mathrm{H}\alpha\,\mathrm{EW} \leq -3\,\text{\AA}",
            "sign_convention": "negative equivalent width denotes emission",
        },
        "fractions_by_sample": fractions,
        "comparisons": comparisons,
        "morphology_adjusted_models": _morphology_adjusted_models(frame, halpha),
        "any_low_p_comparison": any(
            value.get("fisher_p", 1) < 0.05 for value in comparisons.values()
        ),
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result["figure"] = _plot_halpha(
            values,
            halpha,
            os.path.join(output_dir, "fig_recent_quenching_diagnostics.pdf"),
        )
    return safe_json(result)
