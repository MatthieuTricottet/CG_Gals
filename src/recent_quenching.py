"""Recently quenched diagnostics, with a limited H-alpha fallback."""

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
    from extended_stats import holm_correction, safe_json
except ModuleNotFoundError:  # pragma: no cover
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import holm_correction, safe_json


DN4000_ALIASES = ["Dn4000", "dn4000", "d4000"]
HDELTA_ALIASES = ["H_delta_A", "h_delta_A", "lick_hd_a"]
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
    ax.set_xlabel(r"H$\alpha$ equivalent width (\AA; emission is negative)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def run_recent_quenching_analysis(data, output_dir: str | None = None):
    """Classify post-starburst candidates only when Dn4000 and H-delta exist."""

    frame = ensure_galaxy_frame(data)
    dn4000 = _find(frame, DN4000_ALIASES)
    hdelta = _find(frame, HDELTA_ALIASES)
    halpha = _find(frame, HALPHA_ALIASES)
    if halpha is None and (dn4000 is None or hdelta is None):
        return {
            "status": "skipped",
            "reason": "missing_spectral_age_diagnostics",
            "missing_columns": ["Dn4000/d4000", "H_delta_A", "h_alpha_eqw"],
        }

    if dn4000 and hdelta and halpha:
        work = frame[["sample", dn4000, hdelta, halpha]].copy()
        for column in [dn4000, hdelta, halpha]:
            work[column] = np.asarray(work[column], dtype=float)
        work["classification"] = "unclassified"
        work.loc[(work[halpha] > -3) & (work[hdelta] >= 4), "classification"] = (
            "recently_quenched"
        )
        work.loc[(work[halpha] <= -3), "classification"] = "starforming"
        work.loc[
            (work[dn4000] >= 1.6) & (work[halpha] > -3) & (work[hdelta] < 4),
            "classification",
        ] = "old_passive"
        fractions = {}
        for sample_name, part in work.groupby("sample", observed=True):
            denominator = int(part["classification"].ne("unclassified").sum())
            fractions[sample_name] = {
                category: (
                    float((part["classification"] == category).sum() / denominator)
                    if denominator
                    else None
                )
                for category in ["old_passive", "recently_quenched", "starforming"]
            }
        return safe_json(
            {
                "status": "ok",
                "columns_used": [dn4000, hdelta, halpha],
                "classification_thresholds": {
                    "weak_halpha_emission": f"{halpha} > -3 Angstrom",
                    "strong_hdelta_absorption": f"{hdelta} >= 4 Angstrom",
                    "old_continuum": f"{dn4000} >= 1.6",
                },
                "fractions_by_sample": fractions,
                "comparisons": {},
            }
        )

    values = (
        frame[["sample", halpha]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    )
    values["strong_halpha_emission"] = values[halpha] <= -3
    fractions = {}
    for sample_name, part in values.groupby("sample", observed=True):
        fractions[sample_name] = {
            "n_with_halpha": int(len(part)),
            "median_halpha_eqw": float(part[halpha].median()),
            "strong_emission_fraction": float(part["strong_halpha_emission"].mean()),
        }
    comparisons = {}
    cg = values.loc[values["sample"] == "CG4"]
    p_values = []
    keys = []
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
        p_values.append(p_value)
        keys.append(key)
    for key, adjusted in zip(keys, holm_correction(p_values)):
        comparisons[key]["p_adj"] = adjusted

    result = {
        "status": "limited",
        "reason": "Dn4000_and_Hdelta_unavailable",
        "columns_used": [halpha],
        "classification_thresholds": {
            "strong_halpha_emission": f"{halpha} <= -3 Angstrom",
            "sign_convention": "negative equivalent width denotes emission",
        },
        "fractions_by_sample": fractions,
        "comparisons": comparisons,
        "recent_quenching_classification_available": False,
        "any_significant_comparison": any(
            value.get("p_adj", 1) < 0.05 for value in comparisons.values()
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
