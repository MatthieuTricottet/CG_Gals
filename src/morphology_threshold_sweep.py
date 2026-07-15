"""Galaxy Zoo debiased-vote-fraction robustness for the morphology contrast.

Two complementary checks that the CG4 morphology signal is not an artefact
of the adopted p > 0.5 classification threshold:

* a threshold sweep (0.4 / 0.5 / 0.6 / 0.8) refitting the adjusted
  elliptical logistic model on the deduplicated pooled frame at each cut;
* a continuous model of the debiased elliptical vote fraction ``p_E``
  itself (cluster-robust OLS), which uses no threshold at all.

Standard errors are clustered by physical group throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from extended_data import dedup_control_pool, ensure_galaxy_frame
    from extended_stats import (
        fit_logistic_model,
        fit_ols_with_optional_cluster_se,
        safe_float,
        safe_json,
    )
    from specialness_models import _covariates
except ModuleNotFoundError:  # pragma: no cover
    from .extended_data import dedup_control_pool, ensure_galaxy_frame
    from .extended_stats import (
        fit_logistic_model,
        fit_ols_with_optional_cluster_se,
        safe_float,
        safe_json,
    )
    from .specialness_models import _covariates

THRESHOLDS = [0.4, 0.5, 0.6, 0.8]


def _classify_at(frame: pd.DataFrame, threshold: float) -> pd.Series:
    """Binary elliptical indicator at one vote-fraction threshold.

    A galaxy is elliptical when p_E exceeds the threshold and beats p_S
    (the tie rule only matters below 0.5, where both fractions can pass);
    spiral symmetrically; anything else is unclassified (NaN).
    """

    p_e = pd.to_numeric(frame["p_E"], errors="coerce")
    p_s = pd.to_numeric(frame["p_S"], errors="coerce")
    elliptical = (p_e > threshold) & (p_e > p_s)
    spiral = (p_s > threshold) & (p_s > p_e)
    return pd.Series(
        np.where(elliptical, 1.0, np.where(spiral, 0.0, np.nan)), index=frame.index
    )


def run_morphology_threshold_sweep(data, output_dir: str | None = None):
    """Sweep the Galaxy Zoo threshold and fit the continuous vote model."""

    frame = ensure_galaxy_frame(data)
    if frame.empty or "p_E" not in frame or "p_S" not in frame:
        return {"status": "skipped", "reason": "missing_vote_fractions"}
    frame = dedup_control_pool(frame)
    covariates, continuous = _covariates(frame)
    predictors = ["is_CG4", *covariates]

    results = {
        "status": "ok",
        "design": "pooled_dedup_cluster_physical_group",
        "thresholds": {},
    }
    for threshold in THRESHOLDS:
        work = frame.copy()
        work["elliptical_t"] = _classify_at(work, threshold)
        classified = work["elliptical_t"].notna()
        model = fit_logistic_model(
            work,
            "elliptical_t",
            predictors,
            continuous=[column for column in continuous if column in predictors],
        )
        entry = {
            "n_classified": int(classified.sum()),
            "fraction_unclassified": float(1 - classified.mean()),
            "model": model,
        }
        if model.get("status") == "ok":
            entry.update(
                {
                    "cg4_odds_ratio": model.get("cg4_odds_ratio"),
                    "cg4_ci95": model.get("cg4_ci95"),
                    "cg4_p": model.get("cg4_p"),
                }
            )
        results["thresholds"][f"{threshold:.1f}"] = entry

    # Continuous debiased-vote-fraction model: no threshold at all.
    work = frame.copy()
    work["p_E_numeric"] = pd.to_numeric(work["p_E"], errors="coerce")
    cluster_col = "physical_group" if "physical_group" in work else "group_uid"
    formula = "p_E_numeric ~ " + " + ".join(predictors)
    fitted = fit_ols_with_optional_cluster_se(
        formula, work, group_col=cluster_col, min_groups=8
    )
    if fitted is None:
        results["continuous"] = {"status": "skipped", "reason": "model_fit_failed"}
    else:
        conf = fitted.conf_int().loc["is_CG4"]
        results["continuous"] = {
            "status": "ok",
            "outcome": "p_E (debiased elliptical vote fraction)",
            "n": int(fitted.nobs),
            "cluster_col": cluster_col,
            "cg4_coefficient": safe_float(fitted.params["is_CG4"]),
            "cg4_ci95": [safe_float(conf[0]), safe_float(conf[1])],
            "cg4_p": safe_float(fitted.pvalues["is_CG4"]),
        }

    # Structural cross-check independent of Galaxy Zoo votes: a Sersic-index
    # early/late split (n_g > 2.5) from the Simard et al. (2011) fits, where
    # coverage allows. Differential Simard completeness is reported by the
    # size analysis; this model is read as a consistency check only.
    if "simard_ng" in frame:
        work = frame.copy()
        ng = pd.to_numeric(work["simard_ng"], errors="coerce")
        work["early_sersic"] = np.where(ng.notna(), (ng > 2.5).astype(float), np.nan)
        coverage = float(ng.notna().mean())
        model = fit_logistic_model(
            work,
            "early_sersic",
            predictors,
            continuous=[column for column in continuous if column in predictors],
        )
        results["sersic_early_late"] = {
            "definition": "early = Simard pure-Sersic n_g > 2.5",
            "coverage": coverage,
            "model": model,
            "cg4_odds_ratio": model.get("cg4_odds_ratio"),
            "cg4_ci95": model.get("cg4_ci95"),
            "cg4_p": model.get("cg4_p"),
        }
    else:
        results["sersic_early_late"] = {
            "status": "skipped",
            "reason": "missing_simard_ng",
        }

    ors = [
        entry.get("cg4_odds_ratio")
        for entry in results["thresholds"].values()
        if entry.get("cg4_odds_ratio")
    ]
    results["or_range"] = (
        [safe_float(min(ors)), safe_float(max(ors))] if ors else None
    )
    results["qualitatively_stable"] = bool(ors and (min(ors) > 1 or max(ors) < 1))
    return safe_json(results)
