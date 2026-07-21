"""T6 — why the Control4C satellite-mass offset arises (diagnostic).

Hypothesis (to test, not assert): within the shared Delta_m <= 3 window,
distance-ranked selection (Control4C) samples the eligible-satellite
luminosity function *democratically*, whereas brightness-ranked selection
(Control4B; RG4 is brightness-complete at N = 4; the HMCG selection is
luminosity-concordant) weights its bright end. That predicts satellites
that are slightly fainter / less massive in Control4C than in CG4/C4B/RG4
even after the T0 repair, and predicts that the much larger offset
measured on the defective (unrestricted) Control4C came from its
Delta_m > 3 interlopers.

Measurements
------------
1. Per-sample satellite Delta_m and log M* distributions (corrected
   sample, post-3688), plus CG4-minus-control differences of means.
2. The same for the *defective* (git HEAD, unrestricted) Control4C, and
   the decomposition: defective offset = interloper part + democratic
   part.
3. Democratic-sampling test: KS distance of the Control4C satellite
   Delta_m distribution against the full eligible-satellite Delta_m
   distribution of the same parent groups (prediction: close), and of
   Control4B against the same reference (prediction: strongly
   bright-shifted).

Outputs: referee/values/T6.json, referee/T6_satmass_table.csv.
"""

from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "referee"
QUANTS = [0.25, 0.5, 0.75]


def sat_frame(gals: pd.DataFrame) -> pd.DataFrame:
    sats = gals.loc[(gals["rank_M"] > 1) & (gals["Group"] != 3688)].copy()
    sats["dmag"] = sats["M_r"] - sats["M_BGG"]
    sats["lgm_num"] = pd.to_numeric(sats["lgm"], errors="coerce")
    return sats


def digest(sats: pd.DataFrame) -> dict:
    return {
        "n": int(len(sats)),
        "dmag_mean": float(sats["dmag"].mean()),
        "dmag_quantiles": {str(q): float(sats["dmag"].quantile(q))
                           for q in QUANTS},
        "lgm_mean": float(sats["lgm_num"].mean()),
        "lgm_quantiles": {str(q): float(sats["lgm_num"].quantile(q))
                          for q in QUANTS},
    }


def main() -> None:
    data = {name: pd.read_csv(ROOT / "data" / f"{name}_Gals.csv")
            for name in ["CG4", "Control4B", "Control4C", "RG4"]}
    pc = pd.read_csv(ROOT / "data" / "PC_Gals.csv")

    # analysis CG4 = the non-split 62-group sample used everywhere
    cg4_groups = pd.read_csv(ROOT / "data" / "CG4_Groups.csv")
    nonsplit = set(cg4_groups.loc[cg4_groups["Class"] != "Split", "Group"]) \
        if "Class" in cg4_groups else set(data["CG4"]["Group"])
    cg4 = data["CG4"][data["CG4"]["Group"].isin(nonsplit)]

    sats = {
        "CG4": sat_frame(cg4),
        "Control4B": sat_frame(data["Control4B"]),
        "Control4C": sat_frame(data["Control4C"]),
        "RG4": sat_frame(data["RG4"]),
    }

    # defective (unrestricted) Control4C from git HEAD for the decomposition
    head_c4c = pd.read_csv(StringIO(subprocess.run(
        ["git", "show", "HEAD:data/Control4C_Gals.csv"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout))
    sats["Control4C_defective"] = sat_frame(head_c4c)
    defective_interlopers = sats["Control4C_defective"].loc[
        sats["Control4C_defective"]["dmag"] > 3]

    values: dict = {"samples": {name: digest(frame)
                                for name, frame in sats.items()}}
    values["samples"]["Control4C_defective_interlopers_only"] = digest(
        defective_interlopers)

    cg4_lgm = values["samples"]["CG4"]["lgm_mean"]
    values["cg4_minus_control_mean_lgm"] = {
        name: cg4_lgm - values["samples"][name]["lgm_mean"]
        for name in ["Control4B", "Control4C", "RG4", "Control4C_defective"]
    }

    # democratic-sampling test against the eligible-satellite reference of
    # the same parent groups (Delta_m <= 3 satellites of C4C's hosts)
    hosts = set(data["Control4C"].loc[data["Control4C"]["Group"] != 3688,
                                      "Group"])
    parent = pc[pc["Group"].isin(hosts)].copy()
    parent["dmag"] = parent["M_r"] - parent["M_BGG"]
    reference = parent.loc[(parent["rank_M"] > 1) & (parent["dmag"] <= 3),
                           "dmag"].to_numpy()
    ks_c4c = stats.ks_2samp(sats["Control4C"]["dmag"].to_numpy(), reference)
    ks_c4b = stats.ks_2samp(sats["Control4B"]["dmag"].to_numpy(), reference)
    ks_cg4 = stats.ks_2samp(sats["CG4"]["dmag"].to_numpy(), reference)
    values["democratic_sampling_test"] = {
        "reference": "all Delta_m <= 3 satellites of the Control4C parent groups",
        "n_reference": int(len(reference)),
        "reference_dmag_mean": float(reference.mean()),
        "ks_Control4C_vs_reference": {"D": float(ks_c4c.statistic),
                                      "p": float(ks_c4c.pvalue)},
        "ks_Control4B_vs_reference": {"D": float(ks_c4b.statistic),
                                      "p": float(ks_c4b.pvalue)},
        "ks_CG4_vs_reference": {"D": float(ks_cg4.statistic),
                                "p": float(ks_cg4.pvalue)},
    }

    (OUT / "values").mkdir(exist_ok=True)
    with open(OUT / "values" / "T6.json", "w") as handle:
        json.dump(values, handle, indent=1, default=float)

    rows = [{"sample": name, **{k: v for k, v in d.items()
                                if not isinstance(v, dict)}}
            for name, d in values["samples"].items()]
    pd.DataFrame(rows).to_csv(OUT / "T6_satmass_table.csv", index=False)

    print(json.dumps(values["cg4_minus_control_mean_lgm"], indent=1))
    print(json.dumps(values["democratic_sampling_test"], indent=1))
    for name, d in values["samples"].items():
        print(f"{name:34s} n={d['n']:5d} dmag_mean={d['dmag_mean']:.3f} "
              f"lgm_mean={d['lgm_mean']:.3f}")


if __name__ == "__main__":
    main()
