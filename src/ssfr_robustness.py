"""Simple robustness checks for the sSFR classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

try:
    import config as co
    import generate_report as report
    from utils import graphics_utils as gu
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from . import generate_report as report
    from .utils import graphics_utils as gu


def _satellite_frame(sample: dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
    """Return satellites with finite sSFR values for one catalogue."""

    frame = sample[name + co.GASUFF].copy()
    if "rank_M" in frame:
        frame = frame.loc[pd.to_numeric(frame["rank_M"], errors="coerce") > 1].copy()
    ssfr_column = "sSFR_raw" if "sSFR_raw" in frame else "sSFR"
    frame["_ssfr_alt"] = pd.to_numeric(frame[ssfr_column], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["_ssfr_alt"])


def _summary(frame: pd.DataFrame, threshold: float) -> dict[str, object]:
    is_starforming = frame["_ssfr_alt"] >= threshold
    n_total = int(len(frame))
    n_starforming = int(is_starforming.sum())
    n_quenched = int(n_total - n_starforming)
    return {
        "n_total": n_total,
        "n_starforming": n_starforming,
        "n_quenched": n_quenched,
        "starforming_fraction": float(n_starforming / n_total) if n_total else None,
        "quenched_fraction": float(n_quenched / n_total) if n_total else None,
        "starforming_fraction_fmt": f"{100 * n_starforming / n_total:.1f}" if n_total else "NA",
        "quenched_fraction_fmt": f"{100 * n_quenched / n_total:.1f}" if n_total else "NA",
    }


def fixed_threshold_satellite_check(
    sample: dict[str, pd.DataFrame],
    threshold: float = -11.0,
) -> dict[str, object]:
    """Compare satellite star-forming fractions using a fixed sSFR threshold."""

    if "CG4" + co.GASUFF not in sample:
        return {"status": "skipped", "reason": "missing_CG4"}

    cg_frame = _satellite_frame(sample, "CG4")
    if cg_frame.empty:
        return {"status": "skipped", "reason": "no_CG4_satellites"}
    cg_summary = _summary(cg_frame, threshold)
    result = {
        "status": "ok",
        "threshold_log10_ssfr_per_year": float(threshold),
        "threshold_label": rf"$\log_{{10}}(\mathrm{{sSFR}}/\mathrm{{yr}}^{{-1}})={threshold:.1f}$",
        "cg4": cg_summary,
        "comparisons": {},
        "morphology_dependence": "Morphology classifications are independent of the sSFR threshold.",
    }

    for control in co.CONTROL:
        if control + co.GASUFF not in sample:
            continue
        control_frame = _satellite_frame(sample, control)
        control_summary = _summary(control_frame, threshold)
        table = [
            [
                cg_summary["n_starforming"],
                cg_summary["n_quenched"],
            ],
            [
                control_summary["n_starforming"],
                control_summary["n_quenched"],
            ],
        ]
        p_value = float(fisher_exact(table, alternative="two-sided").pvalue)
        delta = (
            cg_summary["starforming_fraction"] - control_summary["starforming_fraction"]
            if cg_summary["starforming_fraction"] is not None
            and control_summary["starforming_fraction"] is not None
            else None
        )
        # Use rounded display values for the delta so that the displayed
        # difference is arithmetically consistent with the two shown percentages.
        if delta is not None:
            cg_pct_rounded = round(100 * cg_summary["starforming_fraction"], 1)
            ctrl_pct_rounded = round(100 * control_summary["starforming_fraction"], 1)
            delta_pct_fmt = f"{cg_pct_rounded - ctrl_pct_rounded:.1f}"
        else:
            delta_pct_fmt = "NA"
        result["comparisons"][control] = {
            "control": control_summary,
            "delta_starforming_fraction": float(delta) if delta is not None else None,
            "delta_starforming_fraction_pct_fmt": delta_pct_fmt,
            "fisher_p": p_value,
            "fisher_p_fmt": gu.pvalue_latex(p_value, math_mode=False),
        }

    result["primary_comparison"] = result["comparisons"].get("RG4")
    if result["primary_comparison"]:
        p_value = result["primary_comparison"]["fisher_p"]
        result["summary"] = (
            "The fixed-threshold satellite comparison remains descriptive; the paper's "
            "main hierarchy is unchanged because morphology is unaffected and the "
            "star-formation contrast is threshold/model dependent."
        )
        result["primary_significance"] = (
            "significant" if p_value < co.P_LIMIT else "not_significant"
        )
    return result


def run(sample: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Append the fixed-threshold robustness check to results.json."""

    result = fixed_threshold_satellite_check(sample)
    report.append_json("sSFR_robustness", result)
    return result


def classifier_variants(sample: dict[str, pd.DataFrame], frame=None) -> dict[str, object]:
    """Does the sSFR classification method matter?

    The adopted quenched/star-forming boundary comes from a two-component
    bivariate Gaussian mixture fitted to the binned (log M*, log sSFR)
    distribution of the SDSS non-AGN reference by minimising the
    Kullback-Leibler divergence, with the low-sSFR component's mean
    constrained to -12 < log sSFR < -10 and its covariance determinant not
    exceeding that of the star-forming component; the boundary is the locus
    of equal weighted component densities.  Three alternatives are compared
    on the group galaxies: the same boundary applied to the MPA-JHU
    ``specsfr_tot_p50`` of the stored DR12 spectrum instead of the stored
    ``sfr_tot_p50 - lgm_tot_p50``; an unconstrained maximum-likelihood (EM)
    two-component mixture, fitted to the AGN-inclusive reference because on
    the AGN-filtered reference both EM components fall on the star-forming
    sequence; and a fixed cut at log sSFR = -11.
    """

    from sklearn.mixture import GaussianMixture

    try:
        from extended_data import ensure_galaxy_frame
        from extended_stats import fit_logistic_model
    except ModuleNotFoundError:  # pragma: no cover
        from .extended_data import ensure_galaxy_frame
        from .extended_stats import fit_logistic_model

    frame = ensure_galaxy_frame(sample) if frame is None else frame.copy()
    boundary = report.decode_interp1d(report._load_json(co.RESULTS_BUILD)["sSFR_interp"])

    def _reference(name):
        ref = sample[name].copy()
        ref = ref.loc[
            pd.to_numeric(ref["sSFR"], errors="coerce").between(*co.sSFR_VALID_RANGE)
            & pd.to_numeric(ref["lgm"], errors="coerce").between(*co.LGM_VALID_RANGE)
        ]
        return ref[["lgm", "sSFR"]].to_numpy(dtype=float)

    em_filtered = GaussianMixture(2, covariance_type="full", n_init=10, random_state=1).fit(
        _reference("SDSS")
    )
    em = GaussianMixture(2, covariance_type="full", n_init=10, random_state=1).fit(
        _reference("SDSS_withAGN")
    )
    star_forming = int(np.argmax(em.means_[:, 1]))

    provenance = pd.read_csv(
        co.DATA_PATH + "sdss_spectral_provenance_dr12.csv",
        usecols=["source_specobjid", "source_specsfr"],
    ).drop_duplicates("source_specobjid")
    frame["_sid"] = pd.to_numeric(frame["specobjid"], errors="coerce")
    frame = frame.merge(
        provenance.rename(columns={"source_specobjid": "_sid"}), on="_sid", how="left"
    )
    classified = frame["quenched"].notna()
    ssfr = pd.to_numeric(frame["sSFR"], errors="coerce")
    specsfr = pd.to_numeric(frame["source_specsfr"], errors="coerce").where(
        lambda v: v.between(*co.sSFR_VALID_RANGE)
    )
    mass = frame["logMstar"].to_numpy(dtype=float)
    variants = {"adopted": frame["quenched"]}
    variants["specsfr_p50"] = pd.Series(
        np.where(classified & specsfr.notna(), (specsfr <= boundary(mass)).astype(float), np.nan),
        index=frame.index,
    )
    em_quenched = np.full(len(frame), np.nan)
    rows = classified.to_numpy()
    posterior = em.predict_proba(np.column_stack([mass[rows], ssfr.to_numpy()[rows]]))
    em_quenched[rows] = (posterior[:, star_forming] < 0.5).astype(float)
    variants["em_gmm_agn_inclusive"] = pd.Series(em_quenched, index=frame.index)
    variants["fixed_minus11"] = pd.Series(
        np.where(classified, (ssfr <= -11.0).astype(float), np.nan), index=frame.index
    )

    covariates = ["logMstar", "z_numeric", "log_group_luminosity", "velocity_dispersion"]
    result = {
        "status": "ok",
        "em_non_agn_reference_component_means": em_filtered.means_.tolist(),
        "em_agn_inclusive_component_means": em.means_.tolist(),
        "stored_minus_specsfr_median_dex": float((ssfr - specsfr)[classified].median()),
        "variants": {},
    }
    satellites = frame["is_satellite"] == 1
    for name, labels in variants.items():
        work = frame.assign(quenched_variant=labels)
        both = labels.notna() & classified
        entry = {
            "agreement_with_adopted": float((labels[both] == frame.loc[both, "quenched"]).mean()),
            "n_compared": int(both.sum()),
            "satellite_quenched_fraction": {
                s: float(work.loc[satellites & (work["sample"] == s), "quenched_variant"].mean())
                for s in ("CG4", "Control4B", "Control4C", "RG4")
            },
            "models": {},
        }
        for control in ("Control4B", "Control4C", "RG4"):
            subset = work.loc[satellites & work["sample"].isin(["CG4", control])]
            entry["models"][control] = {
                "quenched": fit_logistic_model(
                    subset, "quenched_variant", ["is_CG4", *covariates], continuous=covariates
                ),
                "quenched_given_class": fit_logistic_model(
                    subset,
                    "quenched_variant",
                    ["is_CG4", "elliptical", *covariates],
                    continuous=covariates,
                ),
            }
        result["variants"][name] = entry
    return result
