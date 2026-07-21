"""Primary per-control contrasts: CG4 vs each control sample separately.

The three control samples answer different questions and are therefore
never pooled here:

* Control4B - is the CG4 population special compared with the *luminous
  population of richer ordinary groups* (four brightest members)?
* Control4C - compared with *BGG-centred projected cores* of ordinary
  groups (BGG + three closest projected companions)?
* RG4       - compared with *true four-member ordinary groups*?

Each contrast fits the same family of adjusted logistic models as the
pooled (secondary) analysis, on the CG4 + one-control subset, with
cluster-robust standard errors by *physical* group (Lim group id for
controls and for CG4s via their host Lim group). Holm correction is applied
within each contrast across its outcome family.
"""

from __future__ import annotations

import os

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from extended_data import ensure_galaxy_frame
    from extended_stats import fit_logistic_model, holm_correction, safe_json
    from specialness_models import LABELS, MODEL_SPECS, _covariates
except ModuleNotFoundError:  # pragma: no cover
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import fit_logistic_model, holm_correction, safe_json
    from .specialness_models import LABELS, MODEL_SPECS, _covariates

CONTRAST_QUESTIONS = {
    "Control4B": "luminous population of richer ordinary groups",
    "Control4C": "BGG-centred projected cores of ordinary groups",
    "RG4": "true four-member ordinary groups",
}
PLOT_OUTCOMES = ["elliptical_all", "spiral_all", "quenched_all"]
PLOT_COLOURS = {"Control4B": "#2864A6", "Control4C": "#25876E", "RG4": "#A74752"}


def _plot(results, path):
    rows = []
    for outcome in PLOT_OUTCOMES:
        for control, contrast in results["contrasts"].items():
            model = contrast.get(outcome, {})
            if model.get("status") == "ok" and model.get("cg4_odds_ratio"):
                rows.append((outcome, control, model))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.6))
    y = np.arange(len(rows))
    for index, (outcome, control, model) in enumerate(rows):
        odds = model["cg4_odds_ratio"]
        low, high = model["cg4_ci95"]
        ax.errorbar(
            odds,
            y[index],
            xerr=[[odds - low], [high - odds]],
            fmt="o",
            color=PLOT_COLOURS.get(control, "#555555"),
            capsize=3,
        )
    ax.axvline(1, color="0.45", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(
        y,
        [
            f"{LABELS.get(outcome, outcome)} vs {control}"
            for outcome, control, _ in rows
        ],
    )
    ax.set_xlabel("CG4 odds ratio (95% confidence interval)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def run_primary_contrasts(data, output_dir: str | None = None, frame=None):
    """Fit the three separate CG4-vs-control model families.

    ``frame`` overrides the galaxy frame (referee sensitivity reruns on
    restricted subsets, e.g. the 55-arcsec crowding exclusion); the default
    ``ensure_galaxy_frame(data)`` is the published analysis.
    """

    if frame is None:
        frame = ensure_galaxy_frame(data)
    if frame.empty:
        return {"status": "skipped", "reason": "no_galaxy_samples"}
    covariates, continuous = _covariates(frame)
    results = {
        "status": "ok",
        "covariates_considered": covariates,
        "cluster_unit": "physical_group",
        "contrasts": {},
    }
    for control, question in CONTRAST_QUESTIONS.items():
        subset = frame.loc[frame["sample"].isin(["CG4", control])].copy()
        contrast = {
            "question": question,
            "n_galaxies": int(len(subset)),
            "n_physical_groups": int(subset["physical_group"].nunique()),
        }
        for name, (outcome, restriction) in MODEL_SPECS.items():
            panel = subset
            predictors = ["is_CG4", *covariates]
            if restriction is not None:
                panel = panel.loc[panel[restriction[0]] == restriction[1]].copy()
                predictors = [
                    column for column in predictors if column != restriction[0]
                ]
            contrast[name] = fit_logistic_model(
                panel,
                outcome,
                predictors,
                continuous=[column for column in continuous if column in predictors],
            )
        ok_names = [
            name for name in MODEL_SPECS if contrast[name].get("status") == "ok"
        ]
        adjusted = holm_correction([contrast[name].get("cg4_p") for name in ok_names])
        for name, p_adj in zip(ok_names, adjusted):
            contrast[name]["cg4_p_adj"] = p_adj
        contrast["significant_models"] = [
            name for name in ok_names if contrast[name].get("cg4_p_adj", 1) < 0.05
        ]
        results["contrasts"][control] = contrast
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results["figure"] = _plot(
            results, os.path.join(output_dir, "fig_primary_contrasts.pdf")
        )
    return safe_json(results)
