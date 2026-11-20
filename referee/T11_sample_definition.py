"""T11 — sample-definition sensitivities of the E-class contrasts.

Appendix H quotes two sensitivities of the per-control E-versus-S models:

* restoring the 16 Zheng--Shen split systems to the compact-group sample;
* retaining the anomalous Lim 3688 quartets in Control4B and Control4C.

The values were first computed outside the pipeline.  This script rebuilds
them from the inherited catalogues: the extra rows are taken from the raw
``data/*_Gals.csv`` / ``*_Groups.csv`` tables, receive their GZ1 votes from
the cached ``SDSS_withAGN`` table exactly as in ``main.load_data_build``,
and are appended to the processed sample.  The models are the published
``run_primary_contrasts`` families.  The original-label run must reproduce
the previously published odds ratios; the GZ1 DR7 recovery
(``src/gz1_recovery.py``) is then applied for the values quoted in the paper.

Output: referee/values/T11.json.
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

import config as co  # noqa: E402
import data_loader as dl  # noqa: E402
import morphologies as morph  # noqa: E402
from extended_data import ensure_galaxy_frame  # noqa: E402
from gz1_recovery import apply_to_sample  # noqa: E402
from host_controlled import _decision_boundary  # noqa: E402
from primary_contrasts import run_primary_contrasts  # noqa: E402

OUT = ROOT / "referee" / "values" / "T11.json"
MODELS = ["elliptical_all", "elliptical_satellites"]
CONTROLS = ["Control4B", "Control4C", "RG4"]
PUBLISHED_ORIGINAL = {
    "split_restored": {
        "Control4B": (1.97, 2.57),
        "Control4C": (1.49, 1.84),
        "RG4": (2.62, 2.96),
    },
    "lim3688_retained": {"Control4B": (2.18, 2.65), "Control4C": (1.68, 1.94)},
}


def _prepare(raw_gals: pd.DataFrame, sdss: pd.DataFrame) -> pd.DataFrame:
    """Process raw catalogue rows as the rebuild stage does for the paper."""

    gals = dl.sanitize_ssfr(raw_gals.copy())
    gals["objid"] = gals["objid"].astype("int64")
    votes = sdss[["objid", "p_E", "p_S"]].drop_duplicates("objid")
    gals = gals.merge(votes.assign(objid=votes["objid"].astype("int64")), on="objid", how="left")
    gals = morph.classify(gals)
    boundary = _decision_boundary()
    lgm = pd.to_numeric(gals["lgm"], errors="coerce")
    ssfr = pd.to_numeric(gals["sSFR"], errors="coerce")
    status = pd.Series(co.NosSFR_LABEL, index=gals.index, dtype=object)
    measured = lgm.notna() & ssfr.notna()
    if boundary is not None:
        below = ssfr[measured] < boundary(lgm[measured])
        status[measured] = np.where(below, co.sSFR_status[0], co.sSFR_status[1])
    gals["sSFR_status"] = status
    return gals


def _augmented(base: dict, raw: dict, scenario: str) -> dict:
    sample = {key: value.copy() for key, value in base.items()}
    if scenario == "split_restored":
        groups = raw["CG4_Groups"]
        extra_groups = groups[groups["Class"].eq("Split")]
        extra = raw["CG4_Gals"][raw["CG4_Gals"]["Group"].isin(extra_groups["Group"])]
        sample["CG4_Gals"] = pd.concat(
            [sample["CG4_Gals"], _prepare(extra, base["SDSS_withAGN"])],
            ignore_index=True, sort=False,
        )
        sample["CG4_Groups"] = pd.concat(
            [sample["CG4_Groups"], extra_groups], ignore_index=True, sort=False
        )
    elif scenario == "lim3688_retained":
        for name in ["Control4B", "Control4C"]:
            extra = raw[name + "_Gals"][raw[name + "_Gals"]["Group"].eq(3688)]
            extra_groups = raw[name + "_Groups"][raw[name + "_Groups"]["Group"].eq(3688)]
            sample[name + "_Gals"] = pd.concat(
                [sample[name + "_Gals"], _prepare(extra, base["SDSS_withAGN"])],
                ignore_index=True, sort=False,
            )
            sample[name + "_Groups"] = pd.concat(
                [sample[name + "_Groups"], extra_groups], ignore_index=True, sort=False
            )
    return dl.correct_group_distance_scales(sample)


def _summarise(result: dict, controls) -> dict:
    out = {}
    for control in controls:
        out[control] = {}
        for model in MODELS:
            fit = result["contrasts"][control].get(model, {})
            out[control][model] = {
                "status": fit.get("status"),
                "odds_ratio": fit.get("cg4_odds_ratio"),
                "ci95": fit.get("cg4_ci95"),
                "p": fit.get("cg4_p"),
                "n": fit.get("n"),
                "n_clusters": fit.get("n_clusters"),
            }
    return out


def _cached_indices(specobjids, cache_path=None):
    """Offline galSpecIndx: the extra rows have no cached indices (unused here)."""

    import spectral_indices

    cached = pd.read_csv(cache_path or spectral_indices.CACHE_PATH)
    cached["specobjid"] = cached["specobjid"].astype("int64")
    return cached


def main() -> None:
    import spectral_indices

    # Keep the run offline and leave data/galspecindx_dr12.csv untouched: the
    # E/S models do not use the spectral indices of the appended rows.
    spectral_indices.fetch_spectral_indices = _cached_indices
    with open(ROOT / "data" / "processed_sample.pkl", "rb") as handle:
        base = pickle.load(handle)
    raw = dl.load_previous_samples()
    values = {"note": __doc__.split("\n\n")[1].replace("\n", " "), "scenarios": {}}
    for scenario, published in PUBLISHED_ORIGINAL.items():
        controls = list(published)
        entry = {}
        for labels in ["original", "recovered"]:
            sample = _augmented(base, raw, scenario)
            if labels == "recovered":
                sample = apply_to_sample(sample)
            frame = ensure_galaxy_frame(sample)
            entry[labels] = _summarise(run_primary_contrasts(sample, frame=frame), controls)
            entry[labels]["n_cg4_galaxies"] = int(frame["is_CG4"].eq(1).sum())
        checks = {
            control: [
                round(entry["original"][control][model]["odds_ratio"], 2) == ref
                for model, ref in zip(MODELS, published[control])
            ]
            for control in controls
        }
        entry["reproduces_published_original"] = checks
        entry["published_original"] = {c: list(v) for c, v in published.items()}
        values["scenarios"][scenario] = entry
        print(scenario, json.dumps(checks))
        for control in controls:
            for model in MODELS:
                o = entry["original"][control][model]
                r = entry["recovered"][control][model]
                print(f"  {control:9s} {model:22s} original {o['odds_ratio']:.3f} "
                      f"recovered {r['odds_ratio']:.3f} {np.round(r['ci95'], 2)}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(values, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
