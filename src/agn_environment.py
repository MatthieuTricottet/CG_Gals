"""AGN-like and BPT-class environment diagnostics."""

from __future__ import annotations

import os

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from extended_data import ensure_galaxy_frame
    from extended_stats import fit_logistic_model, safe_json
except ModuleNotFoundError:  # pragma: no cover
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import fit_logistic_model, safe_json


def _classify(frame):
    work = frame.copy()
    x = np.asarray(work.get("log_NII_Ha", np.nan), dtype=float)
    y = np.asarray(work.get("log_OIII_Hb", np.nan), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    category = np.full(len(work), "unclassified", dtype=object)
    with np.errstate(divide="ignore", invalid="ignore"):
        kauffmann = 0.61 / (x - 0.05) + 1.30
        kewley = 0.61 / (x - 0.47) + 1.19
    sf = valid & (x < 0.05) & (y < kauffmann)
    composite = valid & (x < 0.47) & (y >= kauffmann) & (y < kewley)
    agn_bpt = valid & ~sf & ~composite
    category[sf] = "starforming"
    category[composite] = "composite"
    category[agn_bpt] = "AGN_like"
    if "is_AGN" in work:
        flag = work["is_AGN"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        category[flag & (category == "unclassified")] = "AGN_like"
    work["bpt_class"] = category
    work["agn_like"] = np.where(
        category == "unclassified", np.nan, (category == "AGN_like").astype(float)
    )
    return work


def _fractions(work, grouping):
    output = {}
    for key, part in work.groupby(grouping, observed=True):
        classified = part["bpt_class"].ne("unclassified")
        denominator = int(classified.sum())
        name = "|".join(map(str, key)) if isinstance(key, tuple) else str(key)
        output[name] = {
            "n_classified": denominator,
            "agn_like_fraction": (
                float((part["bpt_class"] == "AGN_like").sum() / denominator)
                if denominator
                else None
            ),
            "composite_fraction": (
                float((part["bpt_class"] == "composite").sum() / denominator)
                if denominator
                else None
            ),
            "starforming_fraction": (
                float((part["bpt_class"] == "starforming").sum() / denominator)
                if denominator
                else None
            ),
        }
    return output


def _plot(fractions, path):
    samples = ["CG4", "Control4B", "Control4C", "RG4"]
    values = [fractions.get(sample, {}).get("agn_like_fraction") for sample in samples]
    if not any(value is not None for value in values):
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        samples,
        [0 if value is None else value for value in values],
        color=["#2864A6", "#777777", "#777777", "#777777"],
    )
    ax.set_ylabel("AGN-like fraction")
    ax.set_ylim(0, max(0.05, max(value or 0 for value in values) * 1.25))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def run_agn_environment_analysis(data, output_dir: str | None = None):
    """Classify BPT populations and fit mass/environment-adjusted AGN models."""

    frame = ensure_galaxy_frame(data)
    if "is_AGN" not in frame and not {"log_NII_Ha", "log_OIII_Hb"}.issubset(
        frame.columns
    ):
        return {"status": "skipped", "reason": "missing_bpt_and_agn_flag_columns"}
    work = _classify(frame)
    classified = work["agn_like"].notna().sum()
    if classified < 20:
        return {
            "status": "skipped",
            "reason": "too_few_classified_spectra",
            "n": int(classified),
        }

    predictors = ["is_CG4", "logMstar"]
    for column in ["is_satellite", "dist2BGG_kpc", "dominance"]:
        if column in work and work[column].notna().mean() >= 0.6:
            predictors.append(column)
    continuous = [
        column for column in predictors if column not in {"is_CG4", "is_satellite"}
    ]
    models = {
        "all": fit_logistic_model(work, "agn_like", predictors, continuous=continuous),
        "bgg": fit_logistic_model(
            work.loc[work["is_bgg"] == 1],
            "agn_like",
            [column for column in predictors if column != "is_satellite"],
            continuous=[column for column in continuous if column != "is_satellite"],
        ),
        "satellites": fit_logistic_model(
            work.loc[work["is_satellite"] == 1],
            "agn_like",
            [column for column in predictors if column != "is_satellite"],
            continuous=[column for column in continuous if column != "is_satellite"],
        ),
    }
    fractions = _fractions(work, "sample")
    result = {
        "status": (
            "ok" if {"log_NII_Ha", "log_OIII_Hb"}.issubset(work.columns) else "limited"
        ),
        "classification": {
            "scheme": "Kauffmann/Kewley [NII] BPT; existing is_AGN flag fills otherwise unclassified spectra",
            "classes": ["starforming", "composite", "AGN_like", "unclassified"],
            "seyfert_liner_split": "unavailable_without_SII_or_OI",
        },
        "fractions_by_sample": fractions,
        "bgg_satellite_split": _fractions(
            work.assign(rank_class=np.where(work["is_bgg"] == 1, "BGG", "satellite")),
            ["sample", "rank_class"],
        ),
        "mass_adjusted_models": models,
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result["figure"] = _plot(
            fractions, os.path.join(output_dir, "fig_agn_fraction_by_sample.pdf")
        )
    return safe_json(result)
