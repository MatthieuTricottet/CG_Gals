"""Mass-, rank-, and environment-adjusted compact-group models."""

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
except ModuleNotFoundError:  # pragma: no cover
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import fit_logistic_model, holm_correction, safe_json


MODEL_SPECS = {
    "passive_all": ("passive", None),
    "passive_satellites": ("passive", ("is_satellite", 1)),
    "starforming_satellites": ("starforming", ("is_satellite", 1)),
    "elliptical_all": ("elliptical", None),
    "elliptical_bgg": ("elliptical", ("is_bgg", 1)),
    "spiral_all": ("spiral", None),
}
LABELS = {
    "passive_all": "Passive, all",
    "passive_satellites": "Passive, satellites",
    "starforming_satellites": "Star-forming, satellites",
    "elliptical_all": "Elliptical, all",
    "elliptical_bgg": "Elliptical, BGG",
    "spiral_all": "Spiral, all",
}


def _covariates(frame):
    candidates = [
        ("logMstar", True),
        ("z_numeric", True),
        ("is_satellite", False),
        ("log_group_mass", True),
        ("log_group_luminosity", True),
        ("velocity_dispersion", True),
    ]
    selected = []
    continuous = []
    for column, is_continuous in candidates:
        if column not in frame or frame[column].notna().mean() < 0.65:
            continue
        selected.append(column)
        if is_continuous:
            continuous.append(column)
    return selected, continuous


def _plot(results, path):
    rows = []
    for key, result in results.items():
        if not isinstance(result, dict) or result.get("status") != "ok":
            continue
        if result.get("cg4_odds_ratio") is None:
            continue
        rows.append((LABELS.get(key, key), result))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(rows) + 1.6))
    y = np.arange(len(rows))
    odds = np.array([row[1]["cg4_odds_ratio"] for row in rows])
    low = np.array([row[1]["cg4_ci95"][0] for row in rows])
    high = np.array([row[1]["cg4_ci95"][1] for row in rows])
    colours = [
        "#A74752" if row[1].get("cg4_p_adj", 1) < 0.05 else "#555555" for row in rows
    ]
    for index, colour in enumerate(colours):
        ax.errorbar(
            odds[index],
            y[index],
            xerr=[[odds[index] - low[index]], [high[index] - odds[index]]],
            fmt="o",
            color=colour,
            capsize=3,
        )
    ax.axvline(1, color="0.45", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y, [row[0] for row in rows])
    ax.set_xlabel("CG4 odds ratio (95% confidence interval)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def fit_logistic_specialness_models(data, output_dir: str | None = None):
    """Fit the requested family of adjusted binary-outcome models."""

    frame = ensure_galaxy_frame(data)
    if frame.empty:
        return {"status": "skipped", "reason": "no_galaxy_samples"}
    covariates, continuous = _covariates(frame)
    results = {"status": "ok", "covariates_considered": covariates}
    for name, (outcome, subset) in MODEL_SPECS.items():
        panel = frame
        predictors = ["is_CG4", *covariates]
        if subset is not None:
            panel = panel.loc[panel[subset[0]] == subset[1]].copy()
            predictors = [column for column in predictors if column != subset[0]]
        results[name] = fit_logistic_model(
            panel,
            outcome,
            predictors,
            continuous=[column for column in continuous if column in predictors],
        )

    ok_names = [name for name in MODEL_SPECS if results[name].get("status") == "ok"]
    adjusted = holm_correction([results[name].get("cg4_p") for name in ok_names])
    for name, p_adj in zip(ok_names, adjusted):
        result = results[name]
        result["cg4_p_adj"] = p_adj
        coefficient = result.get("cg4_coefficient")
        if p_adj is None or p_adj >= 0.05:
            result["interpretation_flag"] = "not_significant"
        elif coefficient > 0:
            result["interpretation_flag"] = "positive"
        elif coefficient < 0:
            result["interpretation_flag"] = "negative"
        else:
            result["interpretation_flag"] = "null"
    results["significant_models"] = [
        name for name in ok_names if results[name].get("cg4_p_adj", 1) < 0.05
    ]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results["figure"] = _plot(
            results,
            os.path.join(output_dir, "fig_specialness_logistic_coefficients.pdf"),
        )
    return safe_json(results)
