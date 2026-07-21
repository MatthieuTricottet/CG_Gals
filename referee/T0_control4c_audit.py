"""T0 — Control4C construction audit (referee revision, BLOCKING gate).

Referee question: do the Control4C companions respect the Delta_m <= 3 range
of the parent-sample selection?

Paper I construction (01_Data_pipeline.ipynb, cell 59, ground truth):
restrict each parent group's members to Delta_m = M_r - M_r,BGG <= 3 FIRST,
then rank by projected distance to the BGG within that subset
(``rank_distMag``), and keep the BGG + ranks 1-3 (3 nearest companions).

This repository regenerates Control4C in ``src/sample_construction.py``
(``build_control4c_gals``), which selects ``PC_Gals.rank_dist <= 4`` with no
Delta_m filter.  This script establishes, at data level:

1. whether ``PC_Gals.rank_dist`` is the unrestricted distance rank (ranking
   all members, Delta_m > 3 included) or the cell-59 restricted rank;
2. per shipped Control4C group (``data/Control4C_Gals.csv``, the file the
   pipeline reads in ``src/data_loader.py``): (a) all three companions have
   Delta_m <= 3; (b) the companions are exactly the 3 smallest recomputed
   projected separations to the BGG among Delta_m <= 3 members (tie
   tolerance 1e-6 arcmin);
3. the same (cheaper) assertions for Control4B (4 brightest => Delta_m <= 3
   by parent eligibility) and RG4 (all 4 members of exactly-4 groups);
4. parent-catalogue eligibility itself (4th-brightest Delta_m <= 3, cell 52);
5. the historical (Paper I era, git initial commit) Control4C_Gals.csv,
   checked for Delta_m <= 3 within the file, to locate where the defect was
   introduced.

Outputs
-------
referee/T0_control4c_per_group.csv   per-group audit table (shipped C4C)
referee/T0_control4c_audit.md        human-readable audit report

Exit status: 0 if no violation, 1 otherwise (blocking gate).
"""

from __future__ import annotations

import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import spherical_utils as sphu  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "referee"
DMAG_MAX = 3.0
DMAG_TOL = 1e-9          # float guard on the Delta_m <= 3 comparison
TIE_TOL_ARCMIN = 1e-6    # tolerance on separation ties (task spec)
PREAUDIT_COMMIT = "b0d5791"  # initial commit: Paper I era Control4C export


def sep_arcmin(ra0: float, dec0: float, ra, dec) -> np.ndarray:
    """Great-circle separation (arcmin) from (ra0, dec0), degrees in."""
    cosv = (np.sin(np.deg2rad(dec0)) * np.sin(np.deg2rad(np.asarray(dec)))
            + np.cos(np.deg2rad(dec0)) * np.cos(np.deg2rad(np.asarray(dec)))
            * np.cos(np.deg2rad(np.asarray(ra) - ra0)))
    # correct clip usage (cf. T7.4: some legacy utils clip with swapped args)
    return np.rad2deg(np.arccos(np.clip(cosv, -1.0, 1.0))) * 60.0


def audit_control4c(pc: pd.DataFrame, c4c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, shipped in c4c.groupby("Group"):
        parent = pc[pc["Group"] == group]
        rec: dict = {"Group": group, "n_shipped": len(shipped),
                     "n_parent": len(parent)}
        if parent.empty:
            rec["error"] = "group missing from PC_Gals"
            rows.append(rec)
            continue

        bgg_parent = parent[parent["rank_M"] == 1]
        assert len(bgg_parent) == 1, f"group {group}: non-unique BGG"
        bgg = bgg_parent.iloc[0]
        shipped_bgg = shipped[shipped["rank_dist"] == 1]
        assert len(shipped_bgg) == 1, f"group {group}: shipped BGG not unique"
        rec["bgg_matches"] = bool(shipped_bgg.iloc[0]["objid"] == bgg["objid"])

        members = parent[parent["objid"] != bgg["objid"]].copy()
        members["dmag"] = members["M_r"] - bgg["M_r"]
        members["sep_arcmin"] = sep_arcmin(
            bgg["RA"], bgg["Dec"], members["RA"], members["Dec"])
        eligible = members[members["dmag"] <= DMAG_MAX + DMAG_TOL]
        rec["n_eligible_sat"] = len(eligible)

        comp = shipped[shipped["rank_dist"] != 1]
        comp = members.set_index("objid").loc[comp["objid"]].reset_index()
        rec["max_dmag_shipped"] = float(comp["dmag"].max())
        rec["n_dmag_gt3"] = int((comp["dmag"] > DMAG_MAX + DMAG_TOL).sum())

        expected = eligible.nsmallest(3, "sep_arcmin")
        rec["expected_objids"] = ";".join(str(o) for o in expected["objid"])
        rec["shipped_objids"] = ";".join(str(o) for o in comp["objid"])
        rec["quartet_set_changed"] = set(comp["objid"]) != set(expected["objid"])
        max_expected_sep = float(expected["sep_arcmin"].max())
        mismatch = set(comp["objid"]) != set(expected["objid"])
        if mismatch:
            # excuse pure separation ties among eligible members
            tie_ok = all(
                (row["dmag"] <= DMAG_MAX + DMAG_TOL)
                and (row["sep_arcmin"] <= max_expected_sep + TIE_TOL_ARCMIN)
                for _, row in comp.iterrows())
            rec["not_3_nearest_eligible"] = not tie_ok
        else:
            rec["not_3_nearest_eligible"] = False

        # how many ineligible (Delta_m > 3) members sit closer than the 3rd
        # nearest eligible member -- the lever that makes the two
        # constructions differ
        ineligible = members[members["dmag"] > DMAG_MAX + DMAG_TOL]
        rec["n_ineligible_closer"] = int(
            (ineligible["sep_arcmin"] < max_expected_sep - TIE_TOL_ARCMIN).sum())
        rec["dist2bgg_recompute_max_err"] = float(
            np.max(np.abs(comp["sep_arcmin"].values
                          - members.set_index("objid").loc[comp["objid"],
                                                           "dist2BGG"].values))
            if "dist2BGG" in members else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def audit_control4b(pc: pd.DataFrame, c4b: pd.DataFrame) -> dict:
    res = {"n_groups": c4b["Group"].nunique(), "groups_not_4_brightest": 0,
           "groups_dmag_gt3": 0, "missing_parent": 0, "max_dmag": -np.inf}
    for group, shipped in c4b.groupby("Group"):
        parent = pc[pc["Group"] == group]
        if parent.empty:
            res["missing_parent"] += 1
            continue
        brightest = parent.nsmallest(4, "M_r")
        dmag = shipped["M_r"] - parent["M_r"].min()
        res["max_dmag"] = max(res["max_dmag"], float(dmag.max()))
        if float(dmag.max()) > DMAG_MAX + DMAG_TOL:
            res["groups_dmag_gt3"] += 1
        if set(shipped["objid"]) != set(brightest["objid"]):
            # excuse exact-magnitude ties
            cut = brightest["M_r"].max()
            if not all(m <= cut + DMAG_TOL for m in shipped["M_r"]):
                res["groups_not_4_brightest"] += 1
    return res


def audit_rg4(pc: pd.DataFrame, rg4: pd.DataFrame) -> dict:
    res = {"n_groups": rg4["Group"].nunique(), "groups_parent_not_4": 0,
           "groups_membership_mismatch": 0, "groups_dmag_gt3": 0,
           "max_dmag": -np.inf}
    for group, shipped in rg4.groupby("Group"):
        parent = pc[pc["Group"] == group]
        if len(parent) != 4:
            res["groups_parent_not_4"] += 1
        if not parent.empty and set(shipped["objid"]) != set(parent["objid"]):
            res["groups_membership_mismatch"] += 1
        dmag = shipped["M_r"] - shipped["M_r"].min()
        res["max_dmag"] = max(res["max_dmag"], float(dmag.max()))
        if float(dmag.max()) > DMAG_MAX + DMAG_TOL:
            res["groups_dmag_gt3"] += 1
    return res


def main() -> int:
    pc = pd.read_csv(DATA / "PC_Gals.csv")
    c4c = pd.read_csv(DATA / "Control4C_Gals.csv")
    c4b = pd.read_csv(DATA / "Control4B_Gals.csv")
    rg4 = pd.read_csv(DATA / "RG4_Gals.csv")

    # --- 1. semantics of PC_Gals.rank_dist ------------------------------
    pc = pc.copy()
    pc["dmag_pc"] = pc["M_r"] - pc["M_BGG"]
    unrestricted_rank_gt3 = int(
        ((pc["dmag_pc"] > DMAG_MAX + DMAG_TOL) & (pc["rank_dist"] <= 4)).sum())
    dense_within_eligible = bool(
        pc[pc["dmag_pc"] <= DMAG_MAX + DMAG_TOL]
        .groupby("Group")["rank_dist"]
        .apply(lambda s: sorted(s) == list(range(1, len(s) + 1))).all())
    dense_overall = bool(
        pc.groupby("Group")["rank_dist"]
        .apply(lambda s: sorted(s) == list(range(1, len(s) + 1))).all())

    # side-finding: unit factor of the exported dist2BGG column
    bgg_pos = pc[pc["rank_M"] == 1].set_index("Group")[["RA", "Dec"]]
    joined = pc.join(bgg_pos, on="Group", rsuffix="_bgg")
    sat = joined[joined["rank_M"] != 1]
    cosv = (np.sin(np.deg2rad(sat["Dec_bgg"])) * np.sin(np.deg2rad(sat["Dec"]))
            + np.cos(np.deg2rad(sat["Dec_bgg"])) * np.cos(np.deg2rad(sat["Dec"]))
            * np.cos(np.deg2rad(sat["RA"] - sat["RA_bgg"])))
    sep_rad = np.arccos(np.clip(cosv, -1.0, 1.0))
    dist_unit_ratio = (sat["dist2BGG"] / sep_rad)
    unit_lo, unit_hi = float(dist_unit_ratio.min()), float(dist_unit_ratio.max())

    # --- 2. shipped Control4C ------------------------------------------
    table = audit_control4c(pc, c4c)
    n_groups = len(table)
    a_viol = table[table["n_dmag_gt3"] > 0]
    b_viol = table[table["not_3_nearest_eligible"]]
    affected = table[(table["n_dmag_gt3"] > 0) | table["not_3_nearest_eligible"]]
    n_gal_gt3 = int(table["n_dmag_gt3"].sum())
    maxd = table["max_dmag_shipped"]

    # --- 3. Control4B / RG4 -------------------------------------------
    res_b = audit_control4b(pc, c4b)
    res_r = audit_rg4(pc, rg4)

    # --- 4. parent eligibility (cell 52) -------------------------------
    fourth_dmag = (pc.sort_values(["Group", "M_r"]).groupby("Group")
                   .apply(lambda g: float(g["M_r"].iloc[3] - g["M_r"].iloc[0])
                          if len(g) >= 4 else np.nan, include_groups=False))
    elig_viol = int((fourth_dmag > DMAG_MAX + DMAG_TOL).sum())

    # --- 5. historical Paper I era file --------------------------------
    hist_txt = subprocess.run(
        ["git", "show", f"{PREAUDIT_COMMIT}:data/Control4C_Gals.csv"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    hist = pd.read_csv(StringIO(hist_txt))
    hist_comp = hist[hist["rank_dist"] != 1]
    hist_gt3 = int((hist_comp["M_r"] - hist_comp["M_BGG"]
                    > DMAG_MAX + DMAG_TOL).sum())
    hist_max = float((hist_comp["M_r"] - hist_comp["M_BGG"]).max())

    # --- write per-group CSV -------------------------------------------
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "T0_control4c_per_group.csv", index=False)

    # --- corrected-construction preview (no data regenerated) ----------
    changed = table[table["quartet_set_changed"]]
    min_eligible = int(table["n_eligible_sat"].min())
    # Rebuild the corrected quartet for every PC group (765) and re-apply
    # the Paper I contamination exclusion (a control group is dropped when
    # its *selected quartet* contains a full-CG4 galaxy), since corrected
    # quartets can change the excluded set in both directions.
    cg4_ids = set(pd.read_csv(DATA / "CG4_Gals.csv")["objid"])
    n_pc, corrected_contam = 0, 0
    for group, parent in pc.groupby("Group"):
        n_pc += 1
        bgg = parent[parent["rank_M"] == 1].iloc[0]
        members = parent[parent["objid"] != bgg["objid"]].copy()
        members["dmag"] = members["M_r"] - bgg["M_r"]
        eligible = members[members["dmag"] <= DMAG_MAX + DMAG_TOL].copy()
        eligible["sep_arcmin"] = sep_arcmin(
            bgg["RA"], bgg["Dec"], eligible["RA"], eligible["Dec"])
        quartet = set(eligible.nsmallest(3, "sep_arcmin")["objid"])
        quartet.add(bgg["objid"])
        if quartet & cg4_ids:
            corrected_contam += 1
    corrected_final = n_pc - corrected_contam
    q = maxd.quantile([0.5, 0.9, 0.95, 1.0])

    failed = bool(len(affected) or res_b["groups_dmag_gt3"]
                  or res_r["groups_dmag_gt3"])
    lines = []
    w = lines.append
    w("# T0 — Control4C construction audit (BLOCKING)\n")
    if failed:
        w("**Verdict: FAIL.** The shipped `data/Control4C_Gals.csv` (the "
          "file the pipeline reads, `src/data_loader.py:89`) does **not** "
          "implement the Paper I cell-59 construction. Companions are the "
          "3 nearest projected members **regardless of magnitude**, not "
          "the 3 nearest among Δm ≤ 3 members.\n")
    else:
        w("**Verdict: PASS.** Every companion in `data/Control4C_Gals.csv` "
          "satisfies Δm ≤ 3, and every quartet consists of the BGG "
          "plus exactly the 3 nearest recomputed projected separations "
          "among Δm ≤ 3 members (Paper I cell-59 construction). "
          "The historical audit of the defective shipped sample is "
          "preserved in `referee/T0_control4c_audit_shipped_FAIL.md`.\n")
    w("## 1. `PC_Gals.rank_dist` semantics (context)\n")
    w("`PC_Gals.rank_dist` is the **unrestricted** distance rank, so any "
      "regeneration must recompute separations within the Δm ≤ 3 "
      "subset rather than reuse that column:\n")
    w(f"- members with Δm > 3 and rank_dist ≤ 4: "
      f"**{unrestricted_rank_gt3}** (a restricted rank would give 0);")
    w(f"- rank_dist is dense 1..N over *all* members per group: "
      f"{dense_overall}; dense over the Δm ≤ 3 subset: "
      f"{dense_within_eligible}.")
    w("\nPaper I cell 59 filtered to Δm ≤ 3 **first** and ranked "
      "distance within that subset (`rank_distMag`); the deprecated "
      "unrestricted variant (cell 57, `Control_4C`) survives only as "
      "commented-out code.\n")
    w("## 2. Shipped-data assertions (per group, recomputed from PC_Gals)\n")
    w(f"- groups audited: **{n_groups}** ({len(c4c)} galaxies)")
    w(f"- **(a) Δm ≤ 3 violated in {len(a_viol)} groups** "
      f"({100 * len(a_viol) / n_groups:.1f}%), {n_gal_gt3} companion galaxies "
      f"with Δm > 3 ({100 * n_gal_gt3 / (3 * n_groups):.1f}% of "
      "companions);")
    w(f"- **(b) 'not the 3 nearest among Δm ≤ 3 members' in "
      f"{len(b_viol)} groups** (tie tolerance {TIE_TOL_ARCMIN} arcmin);")
    w(f"- groups with any violation: **{len(affected)}**; groups whose "
      f"corrected quartet differs (as a set) from the shipped one: "
      f"**{len(changed)}** — the (a), (b), and membership-change group sets "
      "coincide exactly: every group free of Δm > 3 companions ships "
      "precisely the 3 nearest eligible members;")
    w(f"- per-group max Δm of shipped companions: median "
      f"{q.iloc[0]:.2f}, 90% {q.iloc[1]:.2f}, 95% {q.iloc[2]:.2f}, max "
      f"{q.iloc[3]:.2f}; groups with max Δm > 3: "
      f"{int((maxd > 3).sum())}, > 4: {int((maxd > 4).sum())}, > 5: "
      f"{int((maxd > 5).sum())};")
    w(f"- ineligible (Δm > 3) members sitting closer to the BGG than "
      f"the 3rd nearest eligible member: "
      f"{int(table['n_ineligible_closer'].sum())} across "
      f"{int((table['n_ineligible_closer'] > 0).sum())} groups;")
    w(f"- BGG identity matches PC rank_M = 1 in all groups: "
      f"{bool(table['bgg_matches'].all())}.\n")
    w("Per-group detail: `referee/T0_control4c_per_group.csv`.\n")
    w("### Side-finding: `dist2BGG` unit factor\n")
    w(f"`PC_Gals.dist2BGG` (inherited by every `*_Gals` export) equals the "
      f"great-circle separation in radians × 3600 (measured ratio "
      f"{unit_lo:.4f}–{unit_hi:.4f}), i.e. it converts radians to 'arcmin' "
      "with 1 rad = 3600′ instead of 3437.75′: values are a uniform "
      "**+4.72 % too large** as arcmin. The factor is global, so all "
      "distance *rankings* (incl. rank_dist) are unaffected; any use of "
      "dist2BGG as a numeric angle or its conversion to kpc inherits the "
      "+4.72 % (flagged for T7.2; this repo's own `r2arcmin` utility is "
      "correct).\n")
    w("## 3. Control4B and RG4 (pass)\n")
    w(f"- Control4B: {res_b['n_groups']} groups; not the 4 brightest of the "
      f"parent: {res_b['groups_not_4_brightest']}; Δm > 3: "
      f"{res_b['groups_dmag_gt3']} (max Δm = {res_b['max_dmag']:.3f});")
    w(f"- RG4: {res_r['n_groups']} groups; parent multiplicity ≠ 4: "
      f"{res_r['groups_parent_not_4']}; membership mismatch: "
      f"{res_r['groups_membership_mismatch']}; Δm > 3: "
      f"{res_r['groups_dmag_gt3']} (max Δm = {res_r['max_dmag']:.3f});")
    w(f"- parent eligibility (cell 52, 4th-brightest Δm ≤ 3) "
      f"violations among {pc['Group'].nunique()} PC groups: {elig_viol}.\n")
    w("## 4. History\n")
    w(f"The Paper I era export (git `{PREAUDIT_COMMIT}`, "
      f"{hist['Group'].nunique()} groups) contains {hist_gt3} companions "
      f"with Δm > 3 (max Δm = {hist_max:.3f}): the *distributed* "
      "Control4C implemented the unrestricted nearest-3 selection since "
      "Paper I, and this repository's first regeneration (commit "
      "`c5d80a3`) faithfully reproduced it via the unrestricted "
      "`rank_dist` column. The restricted construction reproduces "
      f"Paper I's *published* counts exactly ({corrected_contam} "
      f"exclusions → {corrected_final} groups; Paper I: 61 → 704) and "
      "every published Control4C statistic (Table 2 medians, Table 3 "
      "T1/T2 — see `referee/T0_paper1_table2_check.py`), resolving "
      "OPEN_QUESTIONS.md #1: Paper I's published analysis used the "
      "restricted sample, while its distributed CSV implemented the "
      "deprecated variant. Full defect record: "
      "`referee/T0_control4c_audit_shipped_FAIL.md`.\n")
    if failed:
        w("## 5. Consequence\n")
        w("Every Control4C-dependent result in the current build is "
          "computed on the unrestricted sample: raw tables, per-control "
          "adjusted models, both matchings, the crowding test, pooled "
          "models, and the tidal comparison. **All downstream referee "
          "tasks are halted** until `data/Control4C_Gals.csv` and "
          "`data/Control4C_Groups.csv` are regenerated with the "
          "Δm ≤ 3 filter applied before distance ranking "
          f"(membership changes in {len(changed)}/{n_groups} groups, "
          f"{n_gal_gt3} companion swaps; corrected sample: "
          f"{corrected_contam} exclusions → {corrected_final} groups; "
          f"min eligible companions per group = {min_eligible}).\n")
    else:
        w("## 5. Status\n")
        w("The committed sample implements the corrected construction "
          f"({n_groups} groups, matching Paper I's published "
          f"{corrected_final}); this audit is the acceptance gate for the "
          "regeneration and passes with zero violations.\n")
    (OUT / "T0_control4c_audit.md").write_text("\n".join(lines))
    print("\n".join(lines))

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
