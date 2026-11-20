"""T2 — within-host models with and without host-centric radius.

The Sect. 3.4 conditional logit conditions on projected host-centric
radius, so it asks whether CG members differ from other members of the
same host *beyond their central location*; the radius-free variant asks
whether they differ *including location*. Under the paper's local-density
interpretation the radius-conditioned coefficient is expected to be
strongly attenuated, so the two specifications are complementary, not
contradictory.

This script refits the Sect. 3.4 conditional logit and FE-GLM (elliptical;
quenched) with and without ``dist_host_kpc``, on the *identical*
complete-case sample (the complete-case mask always includes the radius
column, so dropping the term never changes the rows) and identical strata.
It reports a 2x2 grid {with radius, without radius} x {clogit, FE-GLM}
with OR, 95% CI, and p. Morphology and quenching answer different physical
questions and are not combined into a multiplicity family. The published
Sect. 3.4 numbers correspond to the with-radius cells and are cross-checked
against output/results.json.

Outputs: referee/values/T2.json, referee/T2_within_host_table.csv,
referee/T2_summary.md (hand-written).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gz1_recovery import apply_to_sample as apply_gz1_recovery  # noqa: E402

from host_controlled import (  # noqa: E402
    _fit_conditional_logit,
    _fit_fe_glm,
    build_host_frame,
)

OUT = ROOT / "referee"
OUTCOMES = ["elliptical", "quenched"]
FULL_COVARIATES = ["logMstar", "rank_parent", "dist_host_kpc"]
SPECS = {
    "with_radius": FULL_COVARIATES,
    "without_radius": ["logMstar", "rank_parent"],
}
ESTIMATORS = {
    "conditional_logit": _fit_conditional_logit,
    "fe_glm": _fit_fe_glm,
}


def radial_support(members: pd.DataFrame) -> dict:
    """Hosts where member and non-member satellites overlap in BGG radius.

    Satellites are eligible host members (M_r - M_r,BGG <= 3, host BGG
    excluded) with an E/S class, stellar mass, parent rank and Lim BGG-centric
    radius.  A host offers radial support when the radius ranges of its
    compact-group and other eligible satellites overlap.
    """

    from sample_construction import DMAG_MAX

    eligible = (members["M_r"] - members["M_BGG"]).le(DMAG_MAX)
    satellite = members["rank_parent"].ne(1)
    complete = members[
        ["elliptical", "logMstar", "rank_parent", "dist_host_kpc"]
    ].notna().all(axis=1)
    hosts = []
    for host, part in members[eligible & satellite & complete].groupby(
        "host_lim_group"
    ):
        cg = part.loc[part["is_CG_member"].eq(1), "dist_host_kpc"]
        other = part.loc[part["is_CG_member"].eq(0), "dist_host_kpc"]
        if len(cg) and len(other) and max(cg.min(), other.min()) <= min(
            cg.max(), other.max()
        ):
            hosts.append(int(host))
    return {
        "definition": (
            "eligible non-BGG host members with E/S class, stellar mass, rank "
            "and Lim BGG-centric radius; overlapping member/non-member ranges"
        ),
        "n_hosts": int(members["host_lim_group"].nunique()),
        "n_hosts_with_overlap": len(hosts),
        "hosts_with_overlap": hosts,
    }


def main() -> None:
    with open(ROOT / "data" / "processed_sample.pkl", "rb") as handle:
        sample = pickle.load(handle)
    sample = apply_gz1_recovery(sample)  # DR7 archival GZ1 votes
    members = build_host_frame(sample)

    values = {
        "note": (
            "complete cases fixed on the full covariate set incl. "
            "dist_host_kpc for both specifications"
        ),
        "grid": {},
    }
    rows = []
    for spec_name, covariates in SPECS.items():
        for est_name, fitter in ESTIMATORS.items():
            fits = {}
            for outcome in OUTCOMES:
                complete = members.dropna(
                    subset=["host_lim_group", "is_CG_member", outcome,
                            *FULL_COVARIATES]
                )
                fits[outcome] = fitter(complete, outcome, covariates)
            for outcome in OUTCOMES:
                fit = fits[outcome]
                entry = {
                    "status": fit.get("status"),
                    "n": fit.get("n"),
                    "odds_ratio": fit.get("is_CG_member_odds_ratio"),
                    "ci95": fit.get("is_CG_member_ci95"),
                    "p": fit.get("is_CG_member_p"),
                }
                values["grid"].setdefault(outcome, {}).setdefault(
                    spec_name, {})[est_name] = entry
                rows.append({"outcome": outcome, "spec": spec_name,
                             "estimator": est_name, **entry})

    # cross-check the with-radius cells against the shipped results.json
    with open(ROOT / "output" / "results.json") as handle:
        published = (json.load(handle)["extended_specialness"]
                     ["host_controlled"]["models"])
    checks = {}
    for outcome in OUTCOMES:
        for est_name in ESTIMATORS:
            mine = values["grid"][outcome]["with_radius"][est_name]
            ref = published[outcome][est_name]
            same = (
                ref.get("status") == "ok"
                and abs(mine["odds_ratio"] - ref["is_CG_member_odds_ratio"])
                < 1e-6
            )
            checks[f"{outcome}/{est_name}"] = bool(same)
    values["with_radius_matches_published"] = checks
    assert all(checks.values()), f"published cross-check failed: {checks}"

    values["radial_support"] = radial_support(members)

    (OUT / "values").mkdir(exist_ok=True)
    with open(OUT / "values" / "T2.json", "w") as handle:
        json.dump(values, handle, indent=1)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "T2_within_host_table.csv", index=False)
    print(table.to_string(index=False))
    print("\npublished cross-check:", checks)


if __name__ == "__main__":
    main()
