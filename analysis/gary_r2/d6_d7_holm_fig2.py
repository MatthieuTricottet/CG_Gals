"""D6 -- Holm bookkeeping across the three per-control tests; D7 -- Fig. 2 audit.

D6 stores Holm-adjusted p-values (values only) for (a) Table 3 group-level
permutation p-values, (b) galaxy-level adjusted elliptical odds-ratio
p-values, (c) galaxy-level adjusted quenched odds-ratio p-values.

D7 reports, for each Fig. 2 morphology bin and sample, N galaxies/groups,
median and mean p_el / p_cs, and the fraction classified E and S among
usable E/S classifications, flagging bins where the median and the
fraction/mean disagree qualitatively (i.e. the CG4-minus-control sign flips).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import CONTROLS, SAMPLES, load_results, load_sample, store_diagnostic, write_csv

from descriptive_trends import MORPHOLOGY_MASS_BINS  # noqa: E402
from extended_stats import holm_correction  # noqa: E402


def d6(results: dict) -> dict:
    es = results["extended_specialness"]
    mpc = es["matched_controls"]["group_level_per_control"]
    pcx = es["primary_contrasts"]["contrasts"]
    blocks = {
        "table3_group_permutation": {c: mpc[c]["p_permutation"] for c in CONTROLS},
        "galaxy_adjusted_elliptical_or": {c: pcx[c]["elliptical_all"]["cg4_p"] for c in CONTROLS},
        "galaxy_adjusted_quenched_or": {c: pcx[c]["quenched_all"]["cg4_p"] for c in CONTROLS},
    }
    out = {}
    rows = []
    for name, raw in blocks.items():
        adj = holm_correction([raw[c] for c in CONTROLS])
        out[name] = {"raw": raw, "holm": dict(zip(CONTROLS, adj)), "family_size": 3}
        for c, a in zip(CONTROLS, adj):
            rows.append(dict(family=name, control=c, p_raw=raw[c], p_holm=a))
        print(name, {c: (round(raw[c], 4), round(a, 4)) for c, a in zip(CONTROLS, adj)})
    write_csv(pd.DataFrame(rows), "d6_holm_per_control.csv")
    out["note"] = "Holm within each family of three per-control tests; values stored only, not adopted in the manuscript pending Gate A"
    return out


def d7(sample: dict) -> dict:
    rows = []
    bins = MORPHOLOGY_MASS_BINS
    for name in SAMPLES:
        g = sample[name + "_Gals"].copy()
        g["lgm"] = pd.to_numeric(g["lgm"], errors="coerce")
        g["p_E"] = pd.to_numeric(g["p_E"], errors="coerce")
        g["p_S"] = pd.to_numeric(g["p_S"], errors="coerce")
        g["mass_bin"] = pd.cut(g["lgm"], bins=bins, right=False, include_lowest=True)
        for interval, part in g.groupby("mass_bin", observed=False):
            votes = part.dropna(subset=["lgm", "p_E", "p_S"])
            usable = votes.loc[votes["morphology"].isin(["Elliptical", "Spiral"])]
            n_e = int((usable["morphology"] == "Elliptical").sum())
            n_s = int((usable["morphology"] == "Spiral").sum())
            rows.append(dict(
                sample=name, bin_left=float(interval.left), bin_right=float(interval.right),
                n_galaxies_votes=int(len(votes)), n_groups_votes=int(votes["Group"].nunique()),
                n_usable_ES=int(len(usable)), n_groups_usable=int(usable["Group"].nunique()),
                median_p_el=float(votes["p_E"].median()) if len(votes) else np.nan,
                median_p_cs=float(votes["p_S"].median()) if len(votes) else np.nan,
                mean_p_el=float(votes["p_E"].mean()) if len(votes) else np.nan,
                mean_p_cs=float(votes["p_S"].mean()) if len(votes) else np.nan,
                frac_E=n_e / len(usable) if len(usable) else np.nan,
                frac_S=n_s / len(usable) if len(usable) else np.nan,
                frac_uncertain_of_votes=float((votes["morphology"] == "Uncertain").mean()) if len(votes) else np.nan,
                displayed=bool(len(votes) >= 5 and votes["Group"].nunique() >= 3),
            ))
    tab = pd.DataFrame(rows)
    # qualitative disagreement: sign of (CG4 - control) under median vs fraction/mean
    flags = []
    cg = tab.loc[tab["sample"] == "CG4"].set_index("bin_left")
    for ctrl in CONTROLS:
        ct = tab.loc[tab["sample"] == ctrl].set_index("bin_left")
        for b in cg.index:
            if not (cg.loc[b, "displayed"] and ct.loc[b, "displayed"]):
                continue
            for stat_med, stat_alt in (("median_p_el", "frac_E"), ("median_p_el", "mean_p_el"),
                                       ("median_p_cs", "frac_S"), ("median_p_cs", "mean_p_cs")):
                d_med = cg.loc[b, stat_med] - ct.loc[b, stat_med]
                d_alt = cg.loc[b, stat_alt] - ct.loc[b, stat_alt]
                if np.sign(d_med) != np.sign(d_alt) and (abs(d_med) > 0.02 or abs(d_alt) > 0.02):
                    flags.append(dict(control=ctrl, bin_left=float(b), bin_right=float(cg.loc[b, "bin_right"]),
                                      median_stat=stat_med, alt_stat=stat_alt,
                                      delta_median=float(d_med), delta_alt=float(d_alt)))
    flags = pd.DataFrame(flags)
    write_csv(tab, "d7_fig2_morphology_bins_audit.csv")
    write_csv(flags, "d7_fig2_sign_disagreements.csv")
    pd.set_option("display.width", 250)
    print(tab[["sample", "bin_left", "bin_right", "n_galaxies_votes", "n_groups_votes", "n_usable_ES",
               "median_p_el", "mean_p_el", "frac_E", "median_p_cs", "mean_p_cs", "frac_S", "displayed"]].round(3).to_string())
    print("\nsign disagreements (CG4 - control):")
    print(flags.round(3).to_string() if len(flags) else "none")
    # medians equal to 0/1 pathologies: how often is the median vote pinned near 0 or 1?
    pinned = tab.loc[tab["displayed"] & ((tab["median_p_el"] <= 0.05) | (tab["median_p_el"] >= 0.95)
                                         | (tab["median_p_cs"] <= 0.05) | (tab["median_p_cs"] >= 0.95))]
    return {
        "bins": bins.tolist(),
        "table_file": "d7_fig2_morphology_bins_audit.csv",
        "rows": tab.to_dict(orient="records"),
        "sign_disagreements": flags.to_dict(orient="records"),
        "n_sign_disagreements": int(len(flags)),
        "n_displayed_bins_with_median_vote_near_0_or_1": int(len(pinned)),
        "note": ("Medians of bimodal vote fractions collapse toward 0/1 and hide the class mix; "
                 "fractions classified E/S among usable E/S rows and mean debiased votes track the "
                 "class mix directly. Flags list bins where the CG4-minus-control sign differs between "
                 "the median and the fraction/mean statistic (|Delta| > 0.02)."),
    }


def main() -> None:
    results = load_results()
    sample = load_sample()
    store_diagnostic("d6_holm_per_control", d6(results))
    store_diagnostic("d7_fig2_statistic_audit", d7(sample))


if __name__ == "__main__":
    main()
