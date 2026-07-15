"""Independent verification of the external-audit findings A-F.

Each check reports whether the corresponding *defect* is PRESENT or ABSENT,
so the script can be run twice: before the refactor (all defects expected
PRESENT) and after (all expected ABSENT). Factual observations that are not
defects (e.g. the astrophysical overlap between control samples) are reported
as INFO.

Run:  python audit/verify_findings.py [--write-md]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUTPUT = os.path.join(BASE, "output")
SRC = os.path.join(BASE, "src")

RESULTS = []  # (finding_id, defect_present(bool|None), summary)


def record(finding, present, summary):
    RESULTS.append((finding, present, summary))
    status = {True: "DEFECT PRESENT", False: "defect absent", None: "INFO/UNKNOWN"}[present]
    print(f"[{finding:>4}] {status:>14} | {summary}")


def read_src(name):
    with open(os.path.join(SRC, name)) as f:
        return f.read()


def load_samples():
    frames = {}
    for name in ["CG4_Gals", "Control4B_Gals", "Control4C_Gals", "RG4_Gals", "PC_Gals"]:
        frames[name] = pd.read_csv(os.path.join(DATA, name + ".csv"))
    frames["CG4_Groups"] = pd.read_csv(os.path.join(DATA, "CG4_Groups.csv"))
    return frames


def nonsplit_cg4(frames):
    groups = frames["CG4_Groups"]
    keep = groups.loc[groups["Class"] != "Split", "Group"]
    gals = frames["CG4_Gals"]
    return gals[gals["Group"].isin(keep)].copy(), groups[groups["Class"] != "Split"].copy()


# ---------------------------------------------------------------- A. samples
def check_A1(frames):
    cg_gals, cg_groups = nonsplit_cg4(frames)
    cg_objids = set(cg_gals["objid"])

    c4b = frames["Control4B_Gals"]
    rg4 = frames["RG4_Gals"]
    c4c = frames["Control4C_Gals"]
    pc = frames["PC_Gals"]

    b_ok = (c4b["Group"].nunique() == 699 and len(c4b) == 2796
            and not set(c4b["objid"]) & cg_objids)
    r_ok = (rg4["Group"].nunique() == 56 and len(rg4) == 224
            and not set(rg4["objid"]) & cg_objids)
    record("A1-B", None, f"Control4B: {c4b['Group'].nunique()} groups, {len(c4b)} rows, "
           f"{len(set(c4b['objid']) & cg_objids)} CG4 objids -> "
           + ("matches Paper I" if b_ok else "MISMATCH"))
    record("A1-R", None, f"RG4: {rg4['Group'].nunique()} groups, {len(rg4)} rows, "
           f"{len(set(rg4['objid']) & cg_objids)} CG4 objids -> "
           + ("matches Paper I" if r_ok else "MISMATCH"))

    contaminated = c4c[c4c["objid"].isin(cg_objids)]
    contaminated_groups = sorted(contaminated["Group"].unique())
    cg_of_contam = sorted(
        cg_gals[cg_gals["objid"].isin(contaminated["objid"])]["Group"].unique()
    )
    classes = cg_groups.set_index("Group").loc[cg_of_contam, "Class"].tolist() if cg_of_contam else []
    present = len(contaminated) > 0
    record("A1-C", present,
           f"Control4C: {c4c['Group'].nunique()} groups, {len(c4c)} rows, "
           f"{len(contaminated)} CG4 galaxies in groups {contaminated_groups} "
           f"(CG4 groups {cg_of_contam}, classes {classes})")

    # Rebuild C4C from PC (BGG + 3 closest in projection = rank_dist 1..4)
    rebuilt = pc[pc["rank_dist"] <= 4]
    quartet_sizes = rebuilt.groupby("Group").size()
    contam_rebuilt = rebuilt[rebuilt["objid"].isin(cg_objids)]["Group"].nunique()
    clean_rebuilt = rebuilt["Group"].nunique() - contam_rebuilt
    record("A1-rebuild", None,
           f"Rebuilt C4C from PC: {rebuilt['Group'].nunique()} groups "
           f"(sizes {quartet_sizes.min()}-{quartet_sizes.max()}), "
           f"{contam_rebuilt} CG-contaminated -> {clean_rebuilt} clean "
           f"(Paper I says 61 -> 704)")

    # Composition drift between committed C4C and PC
    c4c_groups = set(c4c["Group"])
    pc_groups = set(pc["Group"])
    only_c4c = c4c_groups - pc_groups
    only_pc = pc_groups - c4c_groups
    contam_groups_rebuilt = set(rebuilt[rebuilt["objid"].isin(cg_objids)]["Group"])
    record("A1-drift", len(only_c4c) > 0,
           f"{len(only_c4c)} committed-C4C groups not in PC; {len(only_pc)} PC groups "
           f"absent from committed C4C ({len(only_pc & contam_groups_rebuilt)} of those "
           f"CG-contaminated)")
    rg4_in_c4c = len(set(frames["RG4_Gals"]["objid"]) & set(c4c["objid"]))
    record("A1-rg4c4c", None, f"{rg4_in_c4c}/224 RG4 galaxies appear in committed C4C "
           "(should be 224 for a faithful BGG+3-closest quartet of 4-member groups)")


def check_A2():
    path = os.path.join(DATA, "Control4C_Gals_old.csv")
    if not os.path.exists(path):
        record("A2", False, "Control4C_Gals_old.csv retired (absent from data/)")
        return
    old = pd.read_csv(path)
    # In a healthy file objid ~ 1.23e18 (SDSS DR16 photoObjID); specobjid differs.
    # In the malformed file the objid column holds specobjid values.
    ref = pd.read_csv(os.path.join(DATA, "Control4C_Gals.csv"))
    shifted = old["objid"].isin(set(ref["specobjid"])).mean()
    record("A2", shifted > 0.5,
           f"Control4C_Gals_old.csv present; {100*shifted:.1f}% of its 'objid' values "
           "are specobjid values (columns shifted left by one)")


def check_A3(frames):
    c4b, c4c, rg4 = (frames["Control4B_Gals"], frames["Control4C_Gals"],
                     frames["RG4_Gals"])
    bc = len(set(c4b["objid"]) & set(c4c["objid"]))
    rb = len(set(rg4["objid"]) & set(c4b["objid"]))
    rc = len(set(rg4["objid"]) & set(c4c["objid"]))
    pooled = pd.concat([c4b["objid"], c4c["objid"], rg4["objid"]])
    record("A3", None,
           f"Overlaps by objid: C4B&C4C={bc}, RG4&C4B={rb}/224, RG4&C4C={rc}; "
           f"pooled rows={len(pooled)}, unique={pooled.nunique()} "
           "(overlap itself is a result; the defect is pooling duplicates as "
           "independent - see B4)")


# ------------------------------------------------------- B. pseudoreplication
def check_B4():
    ext_data = read_src("extended_data.py")
    ext_stats = read_src("extended_stats.py")
    label_uid = ('group_uid' in ext_data
                 and re.search(r'label\s*\n?\s*\+\s*"?:"?', ext_data) is not None)
    default_uid = 'cluster_col: str | None = "group_uid"' in ext_stats
    record("B4", bool(label_uid or default_uid),
           f"label-prefixed group_uid built in extended_data.py: {label_uid}; "
           f"extended_stats.fit_logistic_model default cluster col is label-scoped "
           f"group_uid: {default_uid}")


# ----------------------------------------------------------------- C. matching
def check_C5(results):
    mc = (results or {}).get("extended_specialness", {}).get("matched_controls", {})
    if not mc:
        record("C5-json", None, "no matched_controls block in results.json")
        return
    comp = mc.get("matched_control_counts_by_sample", {})
    saved_matches_audit = (mc.get("n_cg4_matched") == 234
                           and comp.get("Control4B") == 164
                           and comp.get("Control4C") == 70
                           and "RG4" not in comp)
    record("C5-json", None,
           f"results.json matched_controls: n={mc.get('n_cg4_matched')}, "
           f"composition={comp} -> "
           + ("matches audited composition {164,70,0}" if saved_matches_audit
              else "differs from audited composition"))
    # Hard-constraint markers written by the rebuilt matching (Phase 4).
    dedup = mc.get("control_pool_deduplicated_by_objid")
    provenance = mc.get("provenance_table") or mc.get("provenance_file")
    self_excl = mc.get("cg4_objids_excluded_from_controls")
    record("C5-fix", not (dedup and self_excl),
           f"matching hard constraints recorded in results.json: "
           f"dedup_by_objid={dedup}, cg4_objids_excluded={self_excl}, "
           f"provenance={'yes' if provenance else 'no'}")


def check_C5_reimplementation():
    """Reproduce the matching and audit self-matches / duplicate controls."""

    import pickle
    pkl_candidates = [
        os.path.join(BASE, "baseline", "processed_sample_committed.pkl"),
        os.path.join(DATA, "processed_sample.pkl"),
    ]
    pkl_path = next((p for p in pkl_candidates if os.path.exists(p)), None)
    if pkl_path is None:
        record("C5-impl", None, "no processed_sample.pkl available; skipped")
        return
    sys.path.insert(0, SRC)
    os.environ.setdefault("MPLBACKEND", "Agg")
    with open(pkl_path, "rb") as f:
        sample = pickle.load(f)
    from extended_data import build_galaxy_frame
    import matched_controls as mcmod

    frame = build_galaxy_frame(sample)
    variables = mcmod._select_variables(frame)
    pairs, work, caliper = mcmod._greedy_match(frame, variables)
    t_idx = [p["treated_index"] for p in pairs]
    c_idx = [p["control_index"] for p in pairs]
    treated = frame.loc[t_idx]
    control = frame.loc[c_idx]
    comp = control["sample"].value_counts().to_dict()

    rg4_objids = set(frame.loc[frame["sample"] == "RG4", "objid"])
    cg4_objids = set(frame.loc[frame["sample"] == "CG4", "objid"])
    physically_rg4 = int(control["objid"].isin(rg4_objids).sum())
    dup_controls = int(control["objid"].duplicated().sum())
    self_pairs = int((treated["objid"].to_numpy() == control["objid"].to_numpy()).sum())
    control_is_cg4 = int(control["objid"].isin(cg4_objids).sum())

    present = self_pairs > 0 or dup_controls > 0 or control_is_cg4 > 0
    record("C5-impl", present,
           f"reimplemented matching: {len(pairs)} pairs, composition={comp}; "
           f"{physically_rg4} controls physically RG4, {dup_controls} duplicate "
           f"control objids, {self_pairs} self-pairs (treated objid == control "
           f"objid), {control_is_cg4} controls that are CG4 objids")


# --------------------------------------------------------------- D. inference
def check_D6(results):
    stats_code = read_src("extended_stats.py")
    bad_p = "2 * min(np.mean(boot <= 0), np.mean(boot >= 0))" in stats_code
    add_one = "(k + 1)" in stats_code or "+ 1) / (" in stats_code
    record("D6-code", bad_p and not add_one,
           f"bootstrap_difference uses sign-crossing p that can be exactly 0: {bad_p}; "
           f"add-one (k+1)/(B+1) rule present: {add_one}")

    tex_path = os.path.join(OUTPUT, "paper", "paper.tex")
    n_tiny = 0
    if os.path.exists(tex_path):
        with open(tex_path) as f:
            tex = f.read()
        n_tiny = len(re.findall(r"p\s*<\s*10\^\{-6\}", tex))
    record("D6-tex", n_tiny > 0, f"'p<10^{{-6}}' occurrences in output/paper/paper.tex: {n_tiny}")

    mc = (results or {}).get("extended_specialness", {}).get("matched_controls", {})
    zeros = [name for name, eff in (mc.get("effects") or {}).items()
             if isinstance(eff, dict) and eff.get("p") == 0.0]
    record("D6-json", len(zeros) > 0,
           f"matched effects with literal p==0 in results.json: {zeros or 'none'}")


# -------------------------------------------------------------------- E. sSFR
def check_E7(frames):
    counts = {}
    expected = {
        "CG4": (5, 62, 9, 186),
        "Control4B": (48, 698, 103, 2094),
        "Control4C": (55, 751, 110, 2253),
        "RG4": (3, 56, 2, 168),
    }
    cg_gals, _ = nonsplit_cg4(frames)
    pieces = {"CG4": cg_gals}
    for name in ["Control4B", "Control4C", "RG4"]:
        df = frames[name + "_Gals"]
        pieces[name] = df[df["Group"] != 3688]
    any_sentinel = False
    ok = True
    for name, df in pieces.items():
        ssfr = pd.to_numeric(df["sSFR"], errors="coerce")
        any_sentinel |= bool((ssfr <= -9000).any())
        bgg = df["rank_M"] == 1
        sat = df["rank_M"] > 1
        got = (int(ssfr[bgg].isna().sum()), int(bgg.sum()),
               int(ssfr[sat].isna().sum()), int(sat.sum()))
        counts[name] = got
        ok &= got == expected[name]
    record("E7-data", None,
           f"missing-sSFR (BGG miss/N | sat miss/N): "
           + "; ".join(f"{k} {v[0]}/{v[1]} | {v[2]}/{v[3]}" for k, v in counts.items())
           + f" -> {'matches audit' if ok else 'MISMATCH with audit'}; "
           f"raw files contain -9999 sentinels: {any_sentinel}")


def check_E7_code():
    ssfr_code = read_src("sSFR.py")
    config_code = read_src("config.py")
    loader_code = read_src("data_loader.py")
    common_code = open(os.path.join(BASE, "common.py")).read()
    has_flattens = "def flattens_quenched" in ssfr_code
    sentinel_class = re.search(r"sSFR_status\s*=\s*\['Quenched',\s*'Passive',\s*'Starforming'\]",
                               config_code) is not None
    floor_fabricates = "cat.loc[cat[sSFR]<co.sSFR_THRESHOLD, 'sSFR_status'] = co.sSFR_status[0]" in loader_code
    noop_loop = "np.where(col.isnull(),-9999,col)" in common_code.replace(" -9999", "-9999")
    record("E7-code", has_flattens or sentinel_class,
           f"flattens_quenched present: {has_flattens}; sentinel-based 3-class "
           f"config ['Quenched','Passive','Starforming']: {sentinel_class}; "
           f"sSFR_floor fabricates 'Quenched' from sentinel: {floor_fabricates}; "
           f"good_sfr no-op -9999 loop in common.py: {noop_loop}")

    template = open(os.path.join(BASE, "src", "paper_template",
                                 "paper_template.tex")).read()
    mentions = len(re.findall(r"-?9999", template))
    record("E7-tex", mentions > 0,
           f"template mentions the -9999 sentinel rule {mentions} time(s)")


# ------------------------------------------------------------------ F. hygiene
def check_F8():
    readme = open(os.path.join(BASE, "README.md")).read()
    wrong = "TRICOTTET-GAM-CG2" in readme
    record("F8", wrong, f"README describes 'TRICOTTET-GAM-CG2': {wrong}")


def check_F9(results):
    ti = (results or {}).get("extended_specialness", {}).get("tidal_indices", {})
    models = ti.get("models", {})
    ell = models.get("elliptical", {})
    base_or = (ell.get("baseline") or {}).get("cg4_odds_ratio")
    tidal_or = (ell.get("with_tidal_index") or {}).get("cg4_odds_ratio")
    record("F9-json", None,
           f"tidal attenuation of elliptical OR: baseline={base_or and round(base_or, 3)}, "
           f"with_tidal_index={tidal_or and round(tidal_or, 3)}")
    tex_path = os.path.join(OUTPUT, "paper", "paper.tex")
    bad_words = 0
    if os.path.exists(tex_path):
        tex = open(tex_path).read().lower()
        for m in re.finditer(r"tidal", tex):
            window = tex[max(0, m.start() - 400): m.end() + 400]
            bad_words += len(re.findall(r"\bexplain|\baccounts? for", window))
    record("F9-tex", bad_words > 0,
           f"'explains/accounts for' within 400 chars of 'tidal' in paper.tex: {bad_words}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-md", action="store_true")
    parser.add_argument("--skip-reimpl", action="store_true",
                        help="skip the slow matching reimplementation")
    args = parser.parse_args()

    frames = load_samples()
    try:
        with open(os.path.join(OUTPUT, "results.json")) as f:
            results = json.load(f)
    except Exception as exc:
        print(f"[warn] could not read output/results.json ({exc}); "
              "falling back to baseline copy")
        try:
            with open(os.path.join(BASE, "baseline", "output_committed",
                                   "results.json")) as f:
                results = json.load(f)
        except Exception:
            results = {}

    check_A1(frames)
    check_A2()
    check_A3(frames)
    check_B4()
    check_C5(results)
    if not args.skip_reimpl:
        check_C5_reimplementation()
    check_D6(results)
    check_E7(frames)
    check_E7_code()
    check_F8()
    check_F9(results)

    if args.write_md:
        lines = ["# Audit findings verification", "",
                 "| Finding | Status | Detail |", "|---|---|---|"]
        for finding, present, summary in RESULTS:
            status = {True: "DEFECT PRESENT", False: "defect absent",
                      None: "info"}[present]
            lines.append(f"| {finding} | {status} | {summary} |")
        with open(os.path.join(BASE, "audit", "FINDINGS.md"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\nWrote audit/FINDINGS.md")

    n_present = sum(1 for _, p, _ in RESULTS if p is True)
    print(f"\n{n_present} defect(s) present, "
          f"{sum(1 for _, p, _ in RESULTS if p is False)} absent, "
          f"{sum(1 for _, p, _ in RESULTS if p is None)} informational.")


if __name__ == "__main__":
    main()
