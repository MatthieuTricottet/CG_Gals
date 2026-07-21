"""T1 — adjusted per-control morphology families after the 55" exclusion.

Sect. 3.5's crowding test contrasts *raw* proportions, while the headline
per-control result (Sect. 3.2) is *adjusted*; the referee asks for the
robustness of the adjusted estimand on the adjusted estimand.

This script reruns the full Sect. 3.2 per-control adjusted binomial
families (all eight outcomes: elliptical/spiral/quenched x all-member,
satellite-only, plus starforming_satellites and elliptical_bgg) on the
sample that *excludes* every galaxy whose nearest projected same-group
neighbour lies within 55 arcsec (the existing Sect. 3.5 crowding flag),
separately vs Control4B, Control4C, and RG4. Covariates, clustering
(physical Lim group), and the Holm-within-contrast correction are
identical to the published families; the excluded-sample fits form a new
labelled sensitivity family and are never folded into the published one.

Outputs
-------
referee/values/T1.json        machine-readable values (consumed by the TeX)
referee/T1_crowding_table.csv side-by-side full vs excluded table
referee/T1_summary.md         <= 5-sentence referee-response summary
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extended_data import ensure_galaxy_frame  # noqa: E402
from morphology_robustness import (  # noqa: E402
    CROWDING_THRESHOLD_ARCSEC,
    _nearest_angular,
)
from primary_contrasts import run_primary_contrasts  # noqa: E402
from specialness_models import LABELS, MODEL_SPECS  # noqa: E402

OUT = ROOT / "referee"
FIELDS = ["cg4_odds_ratio", "cg4_ci95", "cg4_p", "cg4_p_adj", "n", "n_clusters"]


def main() -> None:
    with open(ROOT / "data" / "processed_sample.pkl", "rb") as handle:
        data = pickle.load(handle)
    frame = ensure_galaxy_frame(data)
    frame["nearest_angular_separation_arcsec"] = _nearest_angular(frame)
    close = (
        frame["nearest_angular_separation_arcsec"] < CROWDING_THRESHOLD_ARCSEC
    ).fillna(False)

    excluded_stats = (
        frame.assign(close=close)
        .groupby("sample", observed=True)["close"]
        .agg(n_close="sum", n_total="count")
        .assign(pct_close=lambda t: 100 * t["n_close"] / t["n_total"])
    )

    kept = frame.loc[~close].copy()
    sensitivity = run_primary_contrasts(data, frame=kept)

    with open(ROOT / "output" / "results.json") as handle:
        published = json.load(handle)["extended_specialness"]["primary_contrasts"]

    rows = []
    values = {
        "threshold_arcsec": CROWDING_THRESHOLD_ARCSEC,
        "excluded_by_sample": {
            sample: {
                "n_close": int(row["n_close"]),
                "n_total": int(row["n_total"]),
                "pct_close": float(row["pct_close"]),
            }
            for sample, row in excluded_stats.iterrows()
        },
        "family_note": (
            "sensitivity family: 55-arcsec exclusion; Holm within each "
            "contrast across the same eight outcomes as the published "
            "Sect. 3.2 families, which stay frozen"
        ),
        "contrasts": {},
    }
    for control in ["Control4B", "Control4C", "RG4"]:
        full_c = published["contrasts"][control]
        excl_c = sensitivity["contrasts"][control]
        values["contrasts"][control] = {}
        for model in MODEL_SPECS:
            full_m = {k: full_c.get(model, {}).get(k) for k in FIELDS}
            excl_m = {k: excl_c.get(model, {}).get(k) for k in FIELDS}
            values["contrasts"][control][model] = {
                "label": LABELS.get(model, model),
                "full": full_m,
                "excluded": excl_m,
                "full_status": full_c.get(model, {}).get("status"),
                "excluded_status": excl_c.get(model, {}).get("status"),
            }
            rows.append(
                {
                    "control": control,
                    "model": model,
                    **{f"full_{k}": full_m[k] for k in FIELDS},
                    **{f"excl_{k}": excl_m[k] for k in FIELDS},
                }
            )

    (OUT / "values").mkdir(exist_ok=True)
    with open(OUT / "values" / "T1.json", "w") as handle:
        json.dump(values, handle, indent=1)
    pd.DataFrame(rows).to_csv(OUT / "T1_crowding_table.csv", index=False)

    # console digest for the run log
    print("55\" exclusion by sample:")
    print(excluded_stats.round(1).to_string())
    for control in ["Control4B", "Control4C", "RG4"]:
        print(f"\n=== {control} (full -> excluded) ===")
        for model in MODEL_SPECS:
            entry = values["contrasts"][control][model]
            f, e = entry["full"], entry["excluded"]

            def fmt(m):
                if m["cg4_odds_ratio"] is None:
                    return "skipped"
                return (f"OR={m['cg4_odds_ratio']:.2f} "
                        f"[{m['cg4_ci95'][0]:.2f},{m['cg4_ci95'][1]:.2f}] "
                        f"p={m['cg4_p']:.4f} pH={m['cg4_p_adj']:.4f} "
                        f"n={m['n']}")

            print(f"{model:24s} {fmt(f)}  ->  {fmt(e)}")


if __name__ == "__main__":
    main()
