"""T5 — symmetric extreme-assignment bounds for missing sSFR.

For each control comparison (all members and satellites only), recompute
the CG4-minus-control quenched-fraction difference under the four extreme
assignments in which ALL missing-sSFR galaxies on EACH side are set to
quenched (Q) or star-forming (SF). Denominators are consistent on both
sides: every galaxy of the sample, missing included, so CG4 is never
bounded against a classified-only control fraction. The report gives the
observed classified-only difference for reference, the per-side missing
fractions, the four corners, and the worst-case (sign-adverse) corner.

Outputs: referee/values/T5.json, referee/T5_bounds_table.csv.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "referee"
CONTROLS = ["Control4B", "Control4C", "RG4"]
CORNERS = {
    "both_quenched": ("Q", "Q"),
    "cg4_quenched_control_sf": ("Q", "SF"),
    "cg4_sf_control_quenched": ("SF", "Q"),
    "both_sf": ("SF", "SF"),
}


def counts(frame: pd.DataFrame) -> dict:
    status = frame["sSFR_status"].astype(str)
    return {
        "n_total": int(len(frame)),
        "n_quenched": int((status == "Quenched").sum()),
        "n_starforming": int((status == "Starforming").sum()),
        "n_missing": int((~status.isin(["Quenched", "Starforming"])).sum()),
    }


def corner_fraction(c: dict, assign: str) -> float:
    quenched = c["n_quenched"] + (c["n_missing"] if assign == "Q" else 0)
    return quenched / c["n_total"]


def main() -> None:
    with open(ROOT / "data" / "processed_sample.pkl", "rb") as handle:
        sample = pickle.load(handle)

    frames = {name: sample[f"{name}_Gals"] for name in ["CG4", *CONTROLS]}
    values = {"note": ("consistent denominators: all galaxies incl. missing "
                       "on both sides; corners assign every missing-sSFR "
                       "galaxy on a side to Q or SF"),
              "comparisons": {}}
    rows = []
    for scope in ["all", "satellites"]:
        scoped = {
            name: (frame if scope == "all"
                   else frame.loc[frame["rank_M"] > 1])
            for name, frame in frames.items()
        }
        c_cg4 = counts(scoped["CG4"])
        for control in CONTROLS:
            c_ctl = counts(scoped[control])
            observed = (
                c_cg4["n_quenched"]
                / (c_cg4["n_quenched"] + c_cg4["n_starforming"])
                - c_ctl["n_quenched"]
                / (c_ctl["n_quenched"] + c_ctl["n_starforming"])
            )
            entry = {
                "cg4_counts": c_cg4,
                "control_counts": c_ctl,
                "cg4_missing_pct": 100 * c_cg4["n_missing"] / c_cg4["n_total"],
                "control_missing_pct": 100 * c_ctl["n_missing"] / c_ctl["n_total"],
                "observed_classified_only_diff": observed,
                "corners": {},
            }
            for name, (a_cg4, a_ctl) in CORNERS.items():
                diff = (corner_fraction(c_cg4, a_cg4)
                        - corner_fraction(c_ctl, a_ctl))
                entry["corners"][name] = diff
            diffs = entry["corners"]
            entry["min_diff"] = min(diffs.values())
            entry["max_diff"] = max(diffs.values())
            entry["sign_preserved_at_worst_case"] = bool(
                entry["min_diff"] > 0 if observed > 0 else entry["max_diff"] < 0
            )
            values["comparisons"][f"{scope}:{control}"] = entry
            rows.append({
                "scope": scope, "control": control,
                "cg4_missing_pct": round(entry["cg4_missing_pct"], 1),
                "control_missing_pct": round(entry["control_missing_pct"], 1),
                "observed_diff": round(observed, 4),
                **{k: round(v, 4) for k, v in diffs.items()},
                "sign_preserved": entry["sign_preserved_at_worst_case"],
            })

    (OUT / "values").mkdir(exist_ok=True)
    with open(OUT / "values" / "T5.json", "w") as handle:
        json.dump(values, handle, indent=1, default=float)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "T5_bounds_table.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
