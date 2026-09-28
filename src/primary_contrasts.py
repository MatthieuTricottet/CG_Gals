"""Primary per-control contrasts: CG4 vs each control sample separately.

The three control samples answer different questions and are therefore
never pooled here:

* Control4B - is the CG4 population special compared with the *luminous
  population of eligible ordinary groups* (four brightest eligible members)?
* Control4C - compared with *BGG-centred projected cores* of ordinary
  groups (BGG + three closest projected companions)?
* RG4       - compared with *true four-member ordinary groups*?

Each comparison fits the same set of adjusted logistic models as the
pooled (secondary) analysis, on the CG4 + one-control subset, with
cluster-robust standard errors by *physical* group (Lim group id for
controls and for CG4s via their host Lim group). Elliptical/spiral and
quenched/star-forming are represented once each because they are exact
binary complements on their respective complete-case samples.
"""

from __future__ import annotations

import os

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import NullFormatter

try:
    from utils.labels_utils import sample_tex_label
except ModuleNotFoundError:  # pragma: no cover
    from .utils.labels_utils import sample_tex_label
try:
    from extended_data import ensure_galaxy_frame
    from extended_stats import fit_logistic_model, holm_correction, safe_json
    from specialness_models import LABELS, MODEL_SPECS, _covariates
except ModuleNotFoundError:  # pragma: no cover
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import fit_logistic_model, holm_correction, safe_json
    from .specialness_models import LABELS, MODEL_SPECS, _covariates

CONTRAST_QUESTIONS = {
    "Control4B": "luminous population of eligible ordinary groups",
    "Control4C": "BGG-centred projected cores of ordinary groups",
    "RG4": "true four-member ordinary groups",
}
PLOT_OUTCOMES = ["elliptical_all", "quenched_all"]
# same control palette as Fig. 2 (descriptive_trends.SAMPLE_STYLES)
PLOT_COLOURS = {"Control4B": "#0072B2", "Control4C": "#D55E00", "RG4": "#009E73"}
MORPHOLOGY_ADJUSTED_QUENCHING_KEY = "quenched_satellites_morphology_adjusted"
SAME_CASE_NO_MORPHOLOGY_KEY = (
    "quenched_satellites_same_complete_case_no_morphology"
)


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
    ax.set_xticks([0.25, 0.5, 1, 2, 4], labels=["0.25", "0.5", "1", "2", "4"])
    ax.set_xticks([], minor=True)  # no 4x10^-1-style minor labels
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(
        y,
        [
            f"{LABELS.get(outcome, outcome)} vs {sample_tex_label(control)}"
            for outcome, control, _ in rows
        ],
    )
    ax.set_xlabel(r"CG$_4$ odds ratio (95% confidence interval)")
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
        "morphology_adjusted_quenching": {
            "population": "satellites_only",
            "outcome": "quenched versus star-forming",
            "morphology_predictor": "binary GZ1 E versus S class",
            "missing_data": (
                "complete cases require measured sSFR class, binary GZ1 E/S "
                "morphology, all continuous covariates, and physical Lim host"
            ),
            "uncertainty": "cluster-robust by physical_group; Wald 95% CI",
            "standardized_absolute_effect": {
                "status": "not_computed",
                "reason": "no established standardization/marginalization framework in repository",
            },
        },
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

        # Direct quenching test requested for satellites: retain the principal
        # satellite-quenching covariates and add the existing conservative
        # binary GZ1 E/S indicator.  Uncertain and NoGZ rows are NaN in
        # ``elliptical`` and therefore excluded rather than promoted to a
        # third class or imputed.
        satellite_panel = subset.loc[subset["is_satellite"] == 1].copy()
        base_predictors = [
            "is_CG4", *[column for column in covariates if column != "is_satellite"]
        ]
        morphology_predictors = [*base_predictors, "elliptical"]
        required = [
            "quenched", *morphology_predictors, "physical_group"
        ]
        complete_case = satellite_panel.replace([np.inf, -np.inf], np.nan).dropna(
            subset=required
        )
        contrast[MORPHOLOGY_ADJUSTED_QUENCHING_KEY] = fit_logistic_model(
            complete_case,
            "quenched",
            morphology_predictors,
            continuous=[
                column for column in continuous if column in morphology_predictors
            ],
        )
        if control == "Control4B":
            same_case = fit_logistic_model(
                complete_case,
                "quenched",
                base_predictors,
                continuous=[
                    column for column in continuous if column in base_predictors
                ],
            )
            same_case["complete_case_defined_by"] = (
                MORPHOLOGY_ADJUSTED_QUENCHING_KEY
            )
            contrast[SAME_CASE_NO_MORPHOLOGY_KEY] = same_case
        results["contrasts"][control] = contrast
    # Holm bookkeeping across the three per-control tests of each model
    # family (gary-r2 D6/A8): stored next to the raw p-values; the controls
    # remain separate pre-specified contrasts and the raw p is primary.
    for name in MODEL_SPECS:
        controls = list(CONTRAST_QUESTIONS)
        raw = [results["contrasts"][c].get(name, {}).get("cg4_p") for c in controls]
        adjusted = holm_correction(raw)
        for control, value in zip(controls, adjusted):
            if results["contrasts"][control].get(name, {}).get("status") == "ok":
                results["contrasts"][control][name]["cg4_p_holm_across_controls"] = value
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results["figure"] = _plot(
            results, os.path.join(output_dir, "fig_primary_contrasts.pdf")
        )
    return safe_json(results)
