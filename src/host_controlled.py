"""Host-controlled experiment: CG members vs other members of the same host.

For Embedded and Predominant CG4 groups — whose four members live inside a
richer Lim et al. (2017) parent group — this compares the CG member galaxies
with the *non-CG members of the same host group* (from ``PC_Gals.csv``),
controlling for stellar mass, luminosity rank in the host, and projected
host-centric radius.

Three estimators are fitted for each outcome:

1. ``conditional_logit`` (primary): conditional logistic regression
   stratified by host group (ConditionalLogit from statsmodels). The stratum
   intercept is conditioned out in the likelihood, so every host-level
   confounder shared by all members of that host (halo mass, richness,
   redshift, large-scale environment) is removed by design. Strata where
   all members have the same outcome (concordant strata) are automatically
   excluded; the number of such strata is reported.

2. ``fe_glm`` (robustness): binomial GLM with C(host_lim_group) fixed-effect
   dummies and the same galaxy-level covariates. Cluster-robust SEs by host.
   Quasi-separation is handled gracefully; if it occurs it is flagged.

3. ``pooled_clustered`` (labelled variant, retained for comparison): the
   original pooled binomial GLM with cluster-robust SEs by host group. Unlike
   the stratified estimators, this variant cannot remove host-level confounders.

The analysis runs behind the ``config.HOST_CONTROLLED_ANALYSIS`` toggle.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import Planck15

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.discrete.conditional_models import ConditionalLogit
    from statsmodels.tools.sm_exceptions import (
        PerfectSeparationError,
        PerfectSeparationWarning,
    )
    _SM_AVAILABLE = True
except ImportError:  # pragma: no cover
    sm = None
    smf = None
    ConditionalLogit = None
    PerfectSeparationError = Exception
    PerfectSeparationWarning = Warning
    _SM_AVAILABLE = False

try:
    import config as co
    from extended_stats import fit_logistic_model, restricted_cubic_spline, safe_json
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from .extended_stats import fit_logistic_model, restricted_cubic_spline, safe_json

CG_CLASSES = ["Embedded", "Predom"]
OUTCOMES = ["elliptical", "quenched"]


def _dist_kpc(dist2bgg, z_group):
    """Convert the catalogue's 3600x-radian projected separation to kpc."""

    theta_arcmin = ((np.asarray(dist2bgg, dtype=float) / 3600.0) * u.rad).to(
        u.arcmin
    ).value
    kpc_per_arcmin = (
        Planck15.kpc_proper_per_arcmin(np.asarray(z_group, dtype=float))
        .to(u.kpc / u.arcmin)
        .value
    )
    return theta_arcmin * kpc_per_arcmin


def build_host_frame(sample) -> pd.DataFrame:
    """Members of the Lim hosts of Embedded/Predominant CG4s, flagged."""

    pc = pd.read_csv(os.path.join(co.DATA_PATH, "PC_Gals.csv"))
    cg4_groups = pd.read_csv(os.path.join(co.DATA_PATH, "CG4_Groups.csv"))
    cg4_gals = sample["CG4_Gals"]

    classes = cg4_groups.set_index("Group")["Class"]
    analysis_groups = [
        group
        for group in cg4_gals["Group"].unique()
        if classes.get(group) in CG_CLASSES
    ]
    cg_member_objids = set(
        cg4_gals.loc[cg4_gals["Group"].isin(analysis_groups), "objid"]
    )
    hosts = (
        cg4_gals.loc[cg4_gals["Group"].isin(analysis_groups), ["objid"]]
        .merge(pc[["objid", "Group"]], on="objid")["Group"]
        .unique()
    )

    members = pc[pc["Group"].isin(hosts)].copy()
    members["is_CG_member"] = members["objid"].isin(cg_member_objids).astype(int)
    members["host_lim_group"] = members["Group"]
    # bookkeeping: hosts containing more than one CG4 (56 systems -> 54 hosts)
    cg_host = (
        cg4_gals.loc[cg4_gals["Group"].isin(analysis_groups), ["objid", "Group"]]
        .merge(pc[["objid", "Group"]].rename(columns={"Group": "host"}), on="objid")
        .drop_duplicates(["Group", "host"])
    )
    per_host = cg_host.groupby("host")["Group"].nunique()
    members.attrs["n_cg4_systems"] = int(len(analysis_groups))
    members.attrs["n_hosts_with_multiple_cg4"] = int((per_host > 1).sum())
    members.attrs["max_cg4_per_host"] = int(per_host.max()) if len(per_host) else 0
    members["logMstar"] = pd.to_numeric(members["lgm_tot_p50"], errors="coerce")
    members["rank_parent"] = pd.to_numeric(members["rank_M"], errors="coerce")
    z_group = pd.to_numeric(members["Yang_z_CMB_group"], errors="coerce").fillna(
        pd.to_numeric(members["z"], errors="coerce")
    )
    members["dist_host_kpc"] = _dist_kpc(members["dist2BGG"], z_group)

    # Morphology from the cached SDSS Galaxy Zoo debiased votes.
    sdss = sample.get("SDSS_withAGN")
    if sdss is not None and "p_E" in sdss:
        votes = sdss[["objid", "p_E", "p_S"]].drop_duplicates("objid")
        members = members.merge(votes, on="objid", how="left")
        p_e = pd.to_numeric(members["p_E"], errors="coerce")
        p_s = pd.to_numeric(members["p_S"], errors="coerce")
        members["elliptical"] = np.where(
            p_e > 0.5, 1.0, np.where(p_s > 0.5, 0.0, np.nan)
        )
    else:
        members["elliptical"] = np.nan

    # Quenched/star-forming from the paper's GMM decision boundary applied
    # to the members' measured sSFR (missing sSFR stays unclassified).
    boundary = _decision_boundary()
    ssfr = pd.to_numeric(members["sSFR"], errors="coerce")
    valid = (
        ssfr.between(*co.sSFR_VALID_RANGE)
        & members["logMstar"].between(*co.LGM_VALID_RANGE)
    )
    if boundary is not None:
        limit = boundary(members["logMstar"].to_numpy(dtype=float))
        members["quenched"] = np.where(
            valid, (ssfr.to_numpy() <= limit).astype(float), np.nan
        )
    else:
        members["quenched"] = np.nan
    return members


def _decision_boundary():
    """Load the GMM equal-posterior boundary saved by the build stage."""

    try:
        import generate_report as report
    except ModuleNotFoundError:  # pragma: no cover
        from . import generate_report as report
    build = report._load_json(co.RESULTS_BUILD)
    payload = build.get("sSFR_interp")
    if not isinstance(payload, dict) or payload.get("__type__") != "interp1d":
        return None
    return report.decode_interp1d(payload)


def _fit_conditional_logit(members, outcome, covariates):
    """Fit conditional logistic regression stratified by host_lim_group.

    Returns a dict with the is_CG_member coefficient, OR, CI, p-value, and
    diagnostics: n_strata_total, n_concordant_strata (all-0 or all-1 outcomes
    per host, which are automatically excluded from the conditional likelihood).
    """

    if ConditionalLogit is None:
        return {"status": "skipped", "reason": "statsmodels_unavailable"}

    work = members[["host_lim_group", "is_CG_member", outcome, *covariates]].copy()
    work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    for col in covariates:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna()
    if len(work) < 20 or work[outcome].nunique() < 2:
        return {"status": "skipped", "reason": "too_few_complete_cases", "n": int(len(work))}

    n_complete = int(len(work))
    groups = work["host_lim_group"]
    n_strata_total = int(groups.nunique())
    strata_outcomes = work.groupby("host_lim_group")[outcome].nunique()
    informative_hosts = strata_outcomes.loc[strata_outcomes > 1].index
    n_concordant = int((strata_outcomes == 1).sum())
    n_informative = n_strata_total - n_concordant

    if n_informative < 2:
        return {
            "status": "skipped",
            "reason": "no_informative_strata",
            "n_complete": n_complete,
            "n_strata_total": n_strata_total,
            "n_concordant_strata": n_concordant,
        }

    work = work.loc[work["host_lim_group"].isin(informative_hosts)].copy()
    groups = work["host_lim_group"]
    endog = work[outcome].to_numpy(dtype=float)
    exog_cols = ["is_CG_member", *covariates]
    # Standardize continuous covariates for numerical stability
    exog_data = work[exog_cols].copy()
    for col in covariates:
        std = exog_data[col].std(ddof=0)
        if std > 0:
            exog_data[col] = (exog_data[col] - exog_data[col].mean()) / std
    exog = exog_data.to_numpy(dtype=float)
    group_arr = groups.to_numpy()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ConditionalLogit(endog, exog, groups=group_arr)
            result = model.fit(method="bfgs", disp=False)
        coef = float(result.params[0])
        se = float(result.bse[0])
        pval = float(result.pvalues[0])
        ci = result.conf_int()
        ci_low = float(ci[0, 0])
        ci_high = float(ci[0, 1])
        return {
            "status": "ok",
            "estimator": "conditional_logit",
            "n": int(len(work)),
            "n_complete": n_complete,
            "n_dropped_concordant": int(n_complete - len(work)),
            "n_strata_total": n_strata_total,
            "n_concordant_strata": n_concordant,
            "n_informative_strata": n_informative,
            "is_CG_member_coef": coef,
            "is_CG_member_se": se,
            "is_CG_member_odds_ratio": float(np.exp(coef)),
            "is_CG_member_ci95": [float(np.exp(ci_low)), float(np.exp(ci_high))],
            "is_CG_member_p": pval,
            "note": "covariates standardized after excluding concordant strata",
        }
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}


def _fit_fe_glm(members, outcome, covariates):
    """Binomial GLM with host-group fixed-effect dummies and clustered SEs.

    Handles quasi-separation gracefully: if statsmodels raises a
    PerfectSeparationWarning or error, the flag is recorded and the result
    is returned with a caveat. Cluster-robust SEs are by host_lim_group.
    """

    if sm is None or smf is None:
        return {"status": "skipped", "reason": "statsmodels_unavailable"}

    work = members[["host_lim_group", "is_CG_member", outcome, *covariates]].copy()
    work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    for col in covariates:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna()
    if len(work) < 20 or work[outcome].nunique() < 2:
        return {"status": "skipped", "reason": "too_few_complete_cases", "n": int(len(work))}

    # Standardize continuous covariates
    for col in covariates:
        std = work[col].std(ddof=0)
        if std > 0:
            work[col] = (work[col] - work[col].mean()) / std

    formula = (
        f"{outcome} ~ is_CG_member + "
        + " + ".join(covariates)
        + " + C(host_lim_group)"
    )
    separation_flag = False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.glm(
                formula,
                data=work,
                family=sm.families.Binomial(),
                missing="drop",
            )
            try:
                fitted = model.fit(
                    cov_type="cluster",
                    cov_kwds={"groups": work["host_lim_group"]},
                    maxiter=500,
                )
            except Exception:
                fitted = model.fit(maxiter=500)
            separation_flag = any(
                issubclass(w.category, PerfectSeparationWarning) for w in caught
            )
    except PerfectSeparationError as exc:
        return {"status": "failed", "reason": str(exc), "separation_flag": True}
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}

    try:
        names = list(fitted.model.exog_names)
        if "is_CG_member" not in names:
            return {"status": "skipped", "reason": "no_is_CG_member_term"}
        idx = names.index("is_CG_member")
        coef = float(np.asarray(fitted.params)[idx])
        se = float(np.asarray(fitted.bse)[idx])
        pval = float(np.asarray(fitted.pvalues)[idx])
        ci = np.asarray(fitted.conf_int())
        ci_low = float(ci[idx, 0])
        ci_high = float(ci[idx, 1])
        n_host_dummies = sum(1 for n in names if n.startswith("C(host_lim_group)"))
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}

    return {
        "status": "ok",
        "estimator": "fe_glm",
        "n": int(fitted.nobs),
        "n_hosts": int(work["host_lim_group"].nunique()),
        "n_host_dummies": n_host_dummies,
        "covariance": getattr(fitted, "cov_type", None),
        "separation_flag": separation_flag,
        "is_CG_member_coef": coef,
        "is_CG_member_se": se,
        "is_CG_member_odds_ratio": float(np.exp(coef)),
        "is_CG_member_ci95": [float(np.exp(ci_low)), float(np.exp(ci_high))],
        "is_CG_member_p": pval,
        "note": "C(host_lim_group) fixed effects; cluster SEs by host; covariates standardized",
    }


RADIAL_BINS_KPC = [0.0, 100.0, 200.0, 400.0, 800.0, 6000.0]


def radial_overlap(members: pd.DataFrame) -> dict:
    """How far the host-centric radii of members and co-members overlap.

    Satellites only (the host BGG sits at R = 0).  Reports the radius
    quantiles of CG members and co-members, the area under the ROC curve
    of CG membership predicted by -R alone, the fraction of co-member
    satellites no farther out than the most distant CG member of their own
    host, and the E-class fractions of both populations in radial bins.
    """

    satellites = members.loc[pd.to_numeric(members["rank_parent"], errors="coerce") > 1].copy()
    satellites = satellites.dropna(subset=["dist_host_kpc"])
    is_member = satellites["is_CG_member"] == 1
    radius = satellites["dist_host_kpc"]
    # Mann-Whitney form of the AUC: P(R_member < R_comember)
    ranks = radius.rank()
    n1, n0 = int(is_member.sum()), int((~is_member).sum())
    auc = 1.0 - (ranks[is_member].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    host_max = satellites.loc[is_member].groupby("host_lim_group")["dist_host_kpc"].max()
    comembers = satellites.loc[~is_member]
    inside = comembers["dist_host_kpc"] <= comembers["host_lim_group"].map(host_max)
    classified = satellites.dropna(subset=["elliptical"]).copy()
    classified["radial_bin"] = pd.cut(classified["dist_host_kpc"], RADIAL_BINS_KPC)
    bins = []
    for interval, part in classified.groupby("radial_bin", observed=True):
        for flag, label in ((1, "member"), (0, "comember")):
            values = part.loc[part["is_CG_member"] == flag, "elliptical"]
            bins.append(
                {
                    "bin_left_kpc": float(interval.left),
                    "bin_right_kpc": float(interval.right),
                    "population": label,
                    "n": int(len(values)),
                    "fraction_E": float(values.mean()) if len(values) else None,
                }
            )
    return {
        "n_member_satellites": n1,
        "n_comember_satellites": n0,
        "member_quantiles_kpc": {str(q): float(radius[is_member].quantile(q)) for q in (0.05, 0.5, 0.95)},
        "comember_quantiles_kpc": {str(q): float(radius[~is_member].quantile(q)) for q in (0.05, 0.5, 0.95)},
        "auc_membership_from_radius": float(auc),
        "comember_fraction_inside_host_member_max": float(inside.mean()),
        "binned_E_fractions": bins,
    }


def radial_specifications(members: pd.DataFrame) -> dict:
    """Conditional-logit member odds ratios under several radius terms.

    E class (and quenched status) against CG membership within hosts:
    without a radius term, with the published linear radius, and for
    satellites with log radius, a restricted cubic spline in log radius, and
    on the radial common support (co-members no farther out than the 95th
    percentile of the members' radii).
    """

    work = members.copy()
    work["is_host_bgg"] = (pd.to_numeric(work["rank_parent"], errors="coerce") == 1).astype(float)
    radius = pd.to_numeric(work["dist_host_kpc"], errors="coerce")
    work["log_dist_host"] = np.log10(radius.where(radius > 0))
    satellites = work.loc[work["is_host_bgg"] == 0].copy()
    knots = np.nanquantile(satellites["log_dist_host"], [0.1, 0.5, 0.9])
    satellites["log_dist_host_rcs"] = restricted_cubic_spline(satellites["log_dist_host"], knots)
    member_p95 = satellites.loc[satellites["is_CG_member"] == 1, "dist_host_kpc"].quantile(0.95)
    support = satellites.loc[satellites["dist_host_kpc"] <= member_p95]
    specs = {
        "all_mass_rank": (work, ["logMstar", "rank_parent"]),
        "all_mass_rank_linear_r": (work, ["logMstar", "rank_parent", "dist_host_kpc"]),
        "satellites_mass_rank": (satellites, ["logMstar", "rank_parent"]),
        "satellites_mass_rank_log_r": (satellites, ["logMstar", "rank_parent", "log_dist_host"]),
        "satellites_mass_spline_log_r": (
            satellites, ["logMstar", "log_dist_host", "log_dist_host_rcs"]
        ),
        "satellites_common_support_mass": (support, ["logMstar"]),
        "satellites_common_support_mass_log_r": (support, ["logMstar", "log_dist_host"]),
    }
    results = {"common_support_max_kpc": float(member_p95)}
    for name, (frame, covariates) in specs.items():
        results[name] = _fit_conditional_logit(frame, "elliptical", covariates)
    results["quenched"] = {
        name: _fit_conditional_logit(frame, "quenched", covariates)
        for name, (frame, covariates) in specs.items()
    }
    return results


def run_host_controlled_analysis(sample, output_dir: str | None = None):
    """Fit the within-host CG-membership models."""

    if not getattr(co, "HOST_CONTROLLED_ANALYSIS", False):
        return {"status": "skipped", "reason": "disabled_by_config"}
    if "CG4_Gals" not in sample:
        return {"status": "skipped", "reason": "missing_CG4"}

    members = build_host_frame(sample)
    if members.empty:
        return {"status": "skipped", "reason": "no_host_members"}

    predictors = ["is_CG_member", "logMstar", "rank_parent", "dist_host_kpc"]
    covariates = ["logMstar", "rank_parent", "dist_host_kpc"]
    results = {
        "status": "ok",
        "design": (
            "within-host comparison: CG members vs non-CG members of the same "
            "Lim parent group (Embedded and Predominant CG4s only). "
            "Three estimators are fitted: conditional_logit (primary, stratified), "
            "fe_glm (fixed-effect dummies robustness), and pooled_clustered (labelled "
            "variant; cannot remove host-level confounders)."
        ),
        "cluster_unit": "host_lim_group",
        "n_hosts": int(members["host_lim_group"].nunique()),
        "n_cg4_systems": members.attrs.get("n_cg4_systems"),
        "n_hosts_with_multiple_cg4": members.attrs.get("n_hosts_with_multiple_cg4"),
        "max_cg4_per_host": members.attrs.get("max_cg4_per_host"),
        "n_members": int(len(members)),
        "n_cg_members": int(members["is_CG_member"].sum()),
        "covariates": covariates,
        "models": {},
    }
    for outcome in OUTCOMES:
        outcome_results = {}

        # (a) Primary: conditional logistic regression stratified by host
        outcome_results["conditional_logit"] = _fit_conditional_logit(
            members, outcome, covariates
        )

        # (b) Robustness: FE-GLM with host fixed-effect dummies
        outcome_results["fe_glm"] = _fit_fe_glm(members, outcome, covariates)

        # (c) Retained variant: original pooled binomial GLM with clustered SEs
        pooled = fit_logistic_model(
            members,
            outcome,
            predictors,
            continuous=covariates,
            cluster_col="host_lim_group",
        )
        if pooled.get("status") == "ok" and "is_CG_member" in pooled.get("terms", {}):
            term = pooled["terms"]["is_CG_member"]
            pooled["cg_member_odds_ratio"] = term["odds_ratio"]
            pooled["cg_member_ci95"] = term["ci95"]
            pooled["cg_member_p"] = term["p"]
        pooled["estimator"] = "pooled_clustered"
        pooled["note"] = (
            "Pooled binomial GLM with cluster-robust SEs by host; "
            "retained as a labelled variant for comparison only. "
            "Cannot remove host-level confounders."
        )
        outcome_results["pooled_clustered"] = pooled

        # Convenience aliases for the primary estimator (conditional logit);
        # fall back to pooled if conditional logit failed. status reflects
        # whether at least one estimator ran successfully.
        primary = outcome_results["conditional_logit"]
        if primary.get("status") == "ok":
            outcome_results["status"] = "ok"
            outcome_results["cg_member_odds_ratio"] = primary.get("is_CG_member_odds_ratio")
            outcome_results["cg_member_ci95"] = primary.get("is_CG_member_ci95")
            outcome_results["cg_member_p"] = primary.get("is_CG_member_p")
        elif pooled.get("status") == "ok":
            # Fallback to pooled if conditional logit fails
            outcome_results["status"] = "ok"
            outcome_results["cg_member_odds_ratio"] = pooled.get("cg_member_odds_ratio")
            outcome_results["cg_member_ci95"] = pooled.get("cg_member_ci95")
            outcome_results["cg_member_p"] = pooled.get("cg_member_p")
        else:
            outcome_results["status"] = "skipped"

        results["models"][outcome] = outcome_results

    results["radial_overlap"] = radial_overlap(members)
    results["radial_specifications"] = radial_specifications(members)
    return safe_json(results)
