"""Consistency gate for the rendered manuscript.

Run after a full pipeline run:

    python audit/consistency_gate.py

Checks that
1. no stale vocabulary survives in the rendered paper: no `passive`, no
   `p<10^{-6}`, no unrendered Jinja markers, and `-9999` only inside the two
   methods sentences that legitimately document the MPA-JHU missing-value
   flag;
2. the galaxy/group counts quoted in the paper match the committed data
   files (recomputed independently here, not read from the JSON);
3. headline numbers in the rendered TeX trace back to `results.json` /
   `results_build.json` (spot-checked for the primary contrasts, the
   matched group-level effect, and the sample counts).

Exit code 0 = gate passed. Used by tests/test_consistency_gate.py.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(BASE, "output", "paper", "paper.tex")
RESULTS = os.path.join(BASE, "output", "results.json")
BUILD = os.path.join(BASE, "output", "results_build.json")

FAILURES: list[str] = []


def fail(message):
    FAILURES.append(message)
    print(f"[FAIL] {message}")


def ok(message):
    print(f"[ ok ] {message}")


def check_vocabulary(tex: str) -> None:
    lowered = tex.lower()
    n_passive = lowered.count("passive")
    if n_passive:
        fail(f"'passive' appears {n_passive} time(s) in the rendered paper")
    else:
        ok("no 'passive' in the rendered paper")

    n_tiny = len(re.findall(r"p[^a-zA-Z]{0,12}<\s*10\^\{-[5-9]\}", tex))
    if n_tiny or "10^{-6}" in tex:
        fail("a p<10^{-6}-style claim survives in the rendered paper")
    else:
        ok("no p<10^{-6}-style claims")

    # The whitelisted methods sentences that legitimately document the
    # MPA-JHU missing-value flag (author wording of 2026-07; update this
    # pattern in lockstep with any approved rephrasing).
    allowed_context = re.compile(
        r"-9999\\\)\s*(flags galaxies without a valid sSFR estimate"
        r"|in the SDSS have no valid sSFR estimate"
        r"|in the SDSS carry the MPA-JHU missing-value sentinel"
        r"|in the sSFR fields are sentinel values indicating that no valid"
        r"|as \\emph\{missing data\})"
        r"|values of \\\(-9999\\\) in the sSFR fields are sentinel values"
        r"|values of \\\(-9999\\\) are missing-value sentinels"
        r"|treat \\texttt\{specsfr\\_tot\\_p50\} \\\(= -9999\\\) as"
        r"|\\texttt\{sfr\\_tot\\_p50\} \\\(= -9999\\\), as \\emph"
        r"|including the \\\(-9999\\\) sentinel"
        # gary-r2 (2026-09) rephrasing of the classification and App. A sentences
        r"|\\\(-9999\\\) is the missing-value sentinel"
        r"|is the \\\(-9999\\\) sentinel"
    )
    for match in re.finditer(r"-9999", tex):
        window = tex[match.start() - 60 : match.end() + 80]
        if not allowed_context.search(window.replace("\n", " ")):
            fail(f"unexpected -9999 outside the documented methods sentences: "
                 f"...{window.strip()[:120]}...")
            break
    else:
        ok("-9999 appears only in the documented methods sentences")

    if "<<" in tex or "<%" in tex:
        fail("unrendered Jinja markers remain in paper.tex")
    else:
        ok("no unrendered Jinja markers")

    if re.search(r"\bn/a\b", tex):
        fail("'n/a' placeholder rendered into the paper")
    else:
        ok("no 'n/a' placeholders")

    # claims and jargon removed in the audit-r3 revision must not return
    retired = {
        "GMM": "the classifier is a constrained mixture fit, not a plain GMM",
        "no additional quenching": "quenching at fixed morphology is control-dependent",
        "unrelated to the star-formation": "missing sSFR is associated with crowding",
        "galaxy rank": "models use a BGG/satellite indicator",
        "causal": "no causal language",
        "mediation": "no mediation language",
        "mediator": "no mediation language",
        "estimand": "no statistical jargon",
        "positivity": "no statistical jargon",
        "upper bound": "the position-adjusted contrast is not a bound",
        "fewer AGN": "the high-[NII] class is not an AGN census",
    }
    body = tex.split("\\begin{document}", 1)[-1]
    for phrase, reason in retired.items():
        if phrase.lower() in body.lower():
            fail(f"retired phrase '{phrase}' in the paper ({reason})")
    ok("no retired phrases")


def check_sample_counts(tex: str) -> None:
    data = os.path.join(BASE, "data")
    cg_groups = pd.read_csv(os.path.join(data, "CG4_Groups.csv"))
    cg_gals = pd.read_csv(os.path.join(data, "CG4_Gals.csv"))
    nonsplit = cg_groups.loc[cg_groups["Class"] != "Split", "Group"]
    counts = {
        "CG4 groups (non-split)": int(nonsplit.nunique()),
        "CG4 galaxies (non-split)": int(cg_gals["Group"].isin(nonsplit).sum()),
    }
    for name, expected_groups in [("Control4B", None), ("Control4C", None),
                                  ("RG4", None)]:
        frame = pd.read_csv(os.path.join(data, f"{name}_Gals.csv"))
        frame = frame[frame["Group"] != 3688]
        counts[f"{name} groups"] = int(frame["Group"].nunique())
        counts[f"{name} galaxies"] = int(len(frame))

    build = json.load(open(BUILD))
    expectations = {
        "CG4_Groups_nonsplit_N": counts["CG4 groups (non-split)"],
        "CG4_Gals_nonsplit_N": counts["CG4 galaxies (non-split)"],
        "Control4B_Groups_N": counts["Control4B groups"],
        "Control4C_Groups_N": counts["Control4C groups"],
        "RG4_Groups_N": counts["RG4 groups"],
        "Control4B_Gals_N": counts["Control4B galaxies"],
        "Control4C_Gals_N": counts["Control4C galaxies"],
        "RG4_Gals_N": counts["RG4 galaxies"],
    }
    for key, expected in expectations.items():
        got = build.get(key)
        if got != expected:
            fail(f"{key}: JSON has {got}, data files give {expected}")
        else:
            ok(f"{key} = {expected} matches the data files")

    # the group counts must also appear verbatim in the rendered text
    for key in ["Control4B_Groups_N", "Control4C_Groups_N", "RG4_Groups_N"]:
        if str(expectations[key]) not in tex:
            fail(f"count {expectations[key]} ({key}) not found in paper.tex")
    ok("sample counts appear in the rendered text")


def check_headline_traceability(tex: str) -> None:
    results = json.load(open(RESULTS))
    es = results.get("extended_specialness", {})

    def expect_in_tex(value, digits, label):
        if value is None:
            return
        formatted = f"{value:.{digits}f}"
        if formatted not in tex:
            fail(f"{label} = {formatted} (results.json) not found in paper.tex")
        else:
            ok(f"{label} = {formatted} traces to results.json")

    contrasts = es.get("primary_contrasts", {}).get("contrasts", {})
    for control in ["Control4B", "Control4C", "RG4"]:
        model = contrasts.get(control, {}).get("elliptical_all", {})
        expect_in_tex(model.get("cg4_odds_ratio"), 2,
                      f"primary elliptical OR vs {control}")

    group_level = es.get("matched_controls", {}).get("group_level", {})
    expect_in_tex(group_level.get("delta_smooth_satellite_fraction"), 3,
                  "group-level matched smooth-satellite difference")

    sweep = es.get("morphology_threshold_sweep", {})
    if sweep.get("or_range"):
        expect_in_tex(sweep["or_range"][0], 2, "threshold-sweep OR minimum")

    host = es.get("host_controlled", {})
    model = host.get("models", {}).get("elliptical", {})
    expect_in_tex(model.get("cg_member_odds_ratio"), 2,
                  "within-host elliptical OR")

    for control in ["Control4B", "Control4C", "RG4"]:
        model = contrasts.get(control, {}).get("elliptical_satellites", {})
        expect_in_tex(model.get("cg4_odds_ratio"), 2, f"satellite elliptical OR vs {control}")
        radial = es.get("radial_position", {}).get("morphology", {}).get(control, {})
        expect_in_tex(radial.get("log_r", {}).get("fraction_control_standardised"), 2,
                      f"control E fraction at CG4 radii ({control})")
        quench = es.get("quenching_morphology", {}).get("models", {}).get(
            "satellites", {}).get(control, {}).get("quenched_given_class", {})
        expect_in_tex(quench.get("cg4_odds_ratio"), 2, f"quenched OR at fixed class vs {control}")
        halpha = es.get("halpha_emission", {}).get("models", {}).get(control, {})
        expect_in_tex(halpha.get("sf_emission", {}).get("mass_class", {}).get("cg4_odds_ratio"), 2,
                      f"SF-like strong H-alpha OR vs {control}")
        expect_in_tex(halpha.get("high_nii_emission", {}).get("mass_class", {}).get("cg4_odds_ratio"), 2,
                      f"high-[NII] strong H-alpha OR vs {control}")
    # the abstract must quote the same position-adjusted fractions as the body
    abstract = tex.split("\\abstract", 1)[-1].split("\\keywords", 1)[0]
    for control in ["Control4B", "Control4C", "RG4"]:
        value = es.get("radial_position", {}).get("morphology", {}).get(control, {}).get(
            "log_r", {}).get("fraction_control_standardised")
        if value is not None:
            if f"{value:.2f}" in abstract:
                ok(f"abstract fraction at CG4 radii ({control}) = {value:.2f}")
            else:
                fail(f"abstract does not quote the log R fraction {value:.2f} ({control})")
    crowd = es.get("crowding", {}).get("crowded_fraction", {}).get("CG4")
    if crowd is not None:
        expect_in_tex(100 * crowd, 0, "crowded CG4 percentage")


def main() -> int:
    if not os.path.exists(TEX):
        print("paper.tex not found; run the pipeline first")
        return 2
    tex = open(TEX).read()
    check_vocabulary(tex)
    check_sample_counts(tex)
    check_headline_traceability(tex)
    print()
    if FAILURES:
        print(f"CONSISTENCY GATE FAILED: {len(FAILURES)} problem(s)")
        return 1
    print("CONSISTENCY GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
