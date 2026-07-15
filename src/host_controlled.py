"""Host-controlled experiment: CG members vs other members of the same host.

For Embedded and Predominant CG4 groups — whose four members live inside a
richer Lim et al. (2017) parent group — this compares the CG member galaxies
with the *non-CG members of the same host group* (from ``PC_Gals.csv``),
controlling for stellar mass, luminosity rank in the host, and projected
host-centric radius, with standard errors clustered by host group.

This is the sharpest available test of "compact subconfiguration" versus
"shared host environment": every comparison is within one host, so any
host-level confounder (halo mass, richness, redshift, large-scale
environment) cancels by construction.

The analysis is new relative to the pre-audit manuscript and runs behind
the ``config.HOST_CONTROLLED_ANALYSIS`` toggle.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import Planck15

try:
    import config as co
    from extended_stats import fit_logistic_model, holm_correction, safe_json
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from .extended_stats import fit_logistic_model, holm_correction, safe_json

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
    continuous = ["logMstar", "rank_parent", "dist_host_kpc"]
    results = {
        "status": "ok",
        "design": (
            "within-host comparison: CG members vs non-CG members of the same "
            "Lim parent group (Embedded and Predominant CG4s only)"
        ),
        "cluster_unit": "host_lim_group",
        "n_hosts": int(members["host_lim_group"].nunique()),
        "n_members": int(len(members)),
        "n_cg_members": int(members["is_CG_member"].sum()),
        "covariates": predictors[1:],
        "models": {},
    }
    for outcome in OUTCOMES:
        results["models"][outcome] = fit_logistic_model(
            members,
            outcome,
            predictors,
            continuous=continuous,
            cluster_col="host_lim_group",
        )
        model = results["models"][outcome]
        if model.get("status") == "ok" and "is_CG_member" in model.get("terms", {}):
            term = model["terms"]["is_CG_member"]
            model["cg_member_odds_ratio"] = term["odds_ratio"]
            model["cg_member_ci95"] = term["ci95"]
            model["cg_member_p"] = term["p"]
    ok_names = [
        name for name in OUTCOMES if results["models"][name].get("status") == "ok"
    ]
    adjusted = holm_correction(
        [results["models"][name].get("cg_member_p") for name in ok_names]
    )
    for name, p_adj in zip(ok_names, adjusted):
        results["models"][name]["cg_member_p_adj"] = p_adj
    return safe_json(results)
