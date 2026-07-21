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
with OR, 95% CI, nominal p, and a Holm p computed under one consistent
convention: within each estimator x specification, Holm across the two
outcomes (the published family convention). The published Sect. 3.4
numbers correspond to the with-radius cells and are cross-checked against
output/results.json.

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

from extended_stats import holm_correction  # noqa: E402
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


def main() -> None:
    with open(ROOT / "data" / "processed_sample.pkl", "rb") as handle:
        sample = pickle.load(handle)
    members = build_host_frame(sample)

    values = {
        "note": (
            "complete cases fixed on the full covariate set incl. "
            "dist_host_kpc for both specifications; Holm across the two "
            "outcomes within each estimator x specification"
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
            holm = holm_correction(
                [fits[o].get("is_CG_member_p") if fits[o].get("status") == "ok"
                 else None for o in OUTCOMES]
            )
            for outcome, p_holm in zip(OUTCOMES, holm):
                fit = fits[outcome]
                if fit.get("status") == "ok":
                    fit["is_CG_member_p_holm"] = p_holm
                entry = {
                    "status": fit.get("status"),
                    "n": fit.get("n"),
                    "odds_ratio": fit.get("is_CG_member_odds_ratio"),
                    "ci95": fit.get("is_CG_member_ci95"),
                    "p": fit.get("is_CG_member_p"),
                    "p_holm": fit.get("is_CG_member_p_holm"),
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

    (OUT / "values").mkdir(exist_ok=True)
    with open(OUT / "values" / "T2.json", "w") as handle:
        json.dump(values, handle, indent=1)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "T2_within_host_table.csv", index=False)
    print(table.to_string(index=False))
    print("\npublished cross-check:", checks)


if __name__ == "__main__":
    main()
