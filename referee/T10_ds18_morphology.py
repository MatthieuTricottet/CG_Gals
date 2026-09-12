"""Reproduce the DS18 morphology audit values used in the manuscript.

This is an isolated validation analysis, not a dependency of the main sample
construction.  It consumes the existing processed sample and the two caches
created and validated by ``notebooks/audit_gz1_vs_ds18.ipynb``; it never
downloads data and never modifies the source catalogues.

Run from the repository root with::

    python referee/T10_ds18_morphology.py

The sole output is ``referee/values/T10.json``, which the paper renderer loads
as part of its existing sensitivity-value context.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extended_data import ensure_galaxy_frame  # noqa: E402
from extended_stats import fit_logistic_model, safe_json  # noqa: E402
from specialness_models import _covariates  # noqa: E402


CACHE = Path(
    os.environ.get(
        "CG_GALS_DS18_CACHE", Path.home() / ".cache" / "cg_gals_audit"
    )
)
SAMPLE_PATH = REPO / "data" / "processed_sample.pkl"
BRIDGE_PATH = CACHE / "photoobjdr7_map.csv"
DS18_PATH = CACHE / "ds18" / "catalog.dat.gz"
OUTPUT_PATH = REPO / "referee" / "values" / "T10.json"

GALAXY_TABLES = ["CG4_Gals", "RG4_Gals", "Control4B_Gals", "Control4C_Gals"]
CONTROL_LABELS = ["Control4B", "Control4C", "RG4"]
SHORT_CLASS = {
    "Elliptical": "E",
    "Spiral": "S",
    "Uncertain": "U",
    "NoGZ": "X",
}
DS18_COLSPECS = [
    (0, 19),
    (20, 26),
    (27, 38),
    (39, 50),
    (51, 62),
    (63, 74),
    (75, 86),
    (87, 98),
    (99, 110),
    (111, 119),
    (120, 131),
]
DS18_COLUMNS = [
    "objID",
    "M15",
    "P_disk",
    "P_edge_on",
    "P_bar_GZ2",
    "P_bar_Nair10",
    "P_merg",
    "P_bulge",
    "P_cigar",
    "TType",
    "P_S0",
]


def _require_inputs() -> None:
    missing = [path for path in (SAMPLE_PATH, BRIDGE_PATH, DS18_PATH) if not path.exists()]
    if missing:
        joined = "\n  - ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing validated DS18-audit input(s):\n  - "
            + joined
            + "\nRun notebooks/audit_gz1_vs_ds18.ipynb to create the external cache."
        )


def _load_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    _require_inputs()
    with SAMPLE_PATH.open("rb") as stream:
        sample = pickle.load(stream)

    bridge = pd.read_csv(
        BRIDGE_PATH,
        dtype={"dr7objid": "int64", "dr8objid": "int64"},
    )
    if bridge["dr8objid"].duplicated().any() or bridge["dr7objid"].duplicated().any():
        raise ValueError("PhotoObjDR7 bridge must be one-to-one")

    with gzip.open(DS18_PATH, "rt") as stream:
        ds18 = pd.read_fwf(
            stream,
            colspecs=DS18_COLSPECS,
            names=DS18_COLUMNS,
            header=None,
            usecols=["objID", "TType", "P_S0"],
            dtype={"objID": str},
        )
    ds18["objID"] = ds18["objID"].str.strip().astype("int64")
    ds18["TType"] = pd.to_numeric(ds18["TType"], errors="coerce")
    ds18["P_S0"] = pd.to_numeric(ds18["P_S0"], errors="coerce")
    if ds18["objID"].duplicated().any():
        raise ValueError("DS18 objID values must be unique")
    return sample, bridge, ds18


def _unique_audit_frame(
    sample: dict, bridge: pd.DataFrame, ds18: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    columns = ["objid", "morphology"]
    for table in GALAXY_TABLES:
        part = sample[table][columns].copy()
        part["objid"] = part["objid"].astype("int64")
        part["sample"] = table.removesuffix("_Gals")
        rows.append(part)
    long = pd.concat(rows, ignore_index=True)

    first = long.drop_duplicates("objid").set_index("objid")
    membership = (
        long.pivot_table(
            index="objid", columns="sample", values="morphology", aggfunc="size"
        )
        .fillna(0)
        .gt(0)
    )
    membership.columns = [f"in_{column}" for column in membership]
    ours = first[["morphology"]].join(membership)
    ours["gz1_class"] = ours["morphology"].map(SHORT_CLASS)
    ours["is_CG"] = ours["in_CG4"]
    ours["is_control"] = ours[
        ["in_RG4", "in_Control4B", "in_Control4C"]
    ].any(axis=1) & ~ours["is_CG"]
    if (ours["is_CG"] & ours["is_control"]).any():
        raise ValueError("CG4 and deduplicated control populations must be disjoint")

    ds18_by_dr8 = bridge[["dr8objid", "dr7objid"]].merge(
        ds18,
        left_on="dr7objid",
        right_on="objID",
        how="inner",
        validate="1:1",
    )
    audited = (
        ours.reset_index()
        .merge(
            ds18_by_dr8[["dr8objid", "dr7objid", "TType", "P_S0"]],
            left_on="objid",
            right_on="dr8objid",
            how="inner",
            validate="1:1",
        )
        .set_index("objid")
    )
    audited["early"] = audited["TType"] <= 0
    return ours, audited


def _fraction(mask: pd.Series) -> float:
    return float(mask.mean())


def _composition_results(ours: pd.DataFrame, audited: pd.DataFrame) -> dict:
    cg = audited.loc[audited["is_CG"]]
    controls = audited.loc[audited["is_control"]]
    hi_s0 = audited.loc[audited["early"] & (audited["P_S0"] >= 0.8)]
    cg_hi = hi_s0.loc[hi_s0["is_CG"]]
    control_hi = hi_s0.loc[hi_s0["is_control"]]

    def matched_block(frame: pd.DataFrame, total: int) -> dict:
        return {
            "n_total": int(total),
            "n_matched": int(len(frame)),
            "fraction_matched": float(len(frame) / total),
        }

    def e_content(frame: pd.DataFrame) -> dict:
        selected = frame.loc[frame["gz1_class"] == "E"]
        return {
            "n": int(len(selected)),
            "fraction_early": _fraction(selected["early"]),
            "fraction_s0_like": _fraction(
                selected["early"] & (selected["P_S0"] > 0.5)
            ),
            "fraction_strong_s0_like": _fraction(
                selected["early"] & (selected["P_S0"] >= 0.8)
            ),
        }

    fisher_table = np.array(
        [
            [
                int((cg_hi["gz1_class"] == "E").sum()),
                int((cg_hi["gz1_class"] != "E").sum()),
            ],
            [
                int((control_hi["gz1_class"] == "E").sum()),
                int((control_hi["gz1_class"] != "E").sum()),
            ],
        ]
    )
    return {
        "unique_sample": int(len(ours)),
        "matched_unique": int(len(audited)),
        "matched_fraction": float(len(audited) / len(ours)),
        "matching_by_environment": {
            "CG4": matched_block(cg, int(ours["is_CG"].sum())),
            "deduplicated_controls": matched_block(
                controls, int(ours["is_control"].sum())
            ),
        },
        "matched_gz1_e": {
            "pooled": e_content(audited),
            "CG4": e_content(cg),
            "deduplicated_controls": e_content(controls),
        },
        "high_confidence_s0": {
            "definition": "TType <= 0 and P_S0 >= 0.8",
            "n": int(len(hi_s0)),
            "fraction_to_E": _fraction(hi_s0["gz1_class"] == "E"),
            "fraction_to_S": _fraction(hi_s0["gz1_class"] == "S"),
            "environment_mapping": {
                "CG4": {
                    "n": int(len(cg_hi)),
                    "fraction_to_E": _fraction(cg_hi["gz1_class"] == "E"),
                },
                "deduplicated_controls": {
                    "n": int(len(control_hi)),
                    "fraction_to_E": _fraction(
                        control_hi["gz1_class"] == "E"
                    ),
                },
                "fisher_E_vs_not_E_p": float(fisher_exact(fisher_table)[1]),
            },
        },
    }


def _support_diagnostics(
    work: pd.DataFrame, continuous: list[str]
) -> dict[str, dict[str, object]]:
    result = {}
    for column in continuous:
        cg = work.loc[work["is_CG4"] == 1, column]
        control = work.loc[work["is_CG4"] == 0, column]
        overlap_low = max(float(cg.min()), float(control.min()))
        overlap_high = min(float(cg.max()), float(control.max()))
        result[column] = {
            "CG4_range": [float(cg.min()), float(cg.max())],
            "control_range": [float(control.min()), float(control.max())],
            "overlap_range": [overlap_low, overlap_high],
            "CG4_fraction_within_control_range": float(
                cg.between(control.min(), control.max()).mean()
            ),
            "control_fraction_within_CG4_range": float(
                control.between(cg.min(), cg.max()).mean()
            ),
        }
    return result


def _independent_early_type_models(
    sample: dict, bridge: pd.DataFrame, ds18: pd.DataFrame
) -> dict:
    frame = ensure_galaxy_frame(sample)
    ds18_by_dr8 = bridge[["dr8objid", "dr7objid"]].merge(
        ds18[["objID", "TType"]],
        left_on="dr7objid",
        right_on="objID",
        how="inner",
        validate="1:1",
    )
    frame = frame.merge(
        ds18_by_dr8[["dr8objid", "TType"]],
        left_on="objid",
        right_on="dr8objid",
        how="left",
        validate="m:1",
    )
    frame["ds18_early"] = np.where(
        frame["TType"].notna(), (frame["TType"] <= 0).astype(float), np.nan
    )
    covariates, continuous = _covariates(frame)
    predictors = ["is_CG4", *covariates]
    results = {
        "outcome": "DS18 TType <= 0",
        "design": "separate CG4-versus-control contrasts",
        "covariates": covariates,
        "continuous_standardized": continuous,
        "cluster_unit": "physical_group",
        "contrasts": {},
    }
    for control in CONTROL_LABELS:
        panel = frame.loc[frame["sample"].isin(["CG4", control])].copy()
        required = ["ds18_early", *predictors]
        work = (
            panel[[*required, "physical_group"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=required)
        )
        model = fit_logistic_model(
            panel,
            "ds18_early",
            predictors,
            continuous=continuous,
        )
        no_sigma_covariates = [
            column for column in covariates if column != "velocity_dispersion"
        ]
        no_sigma_model = fit_logistic_model(
            panel,
            "ds18_early",
            ["is_CG4", *no_sigma_covariates],
            continuous=[
                column for column in continuous if column in no_sigma_covariates
            ],
        )
        results["contrasts"][control] = {
            "n": int(len(work)),
            "n_CG4": int((work["is_CG4"] == 1).sum()),
            "n_control": int((work["is_CG4"] == 0).sum()),
            "n_physical_groups": int(work["physical_group"].nunique()),
            "n_early_CG4": int(
                work.loc[work["is_CG4"] == 1, "ds18_early"].sum()
            ),
            "n_early_control": int(
                work.loc[work["is_CG4"] == 0, "ds18_early"].sum()
            ),
            "model": model,
            "without_sigma_v": no_sigma_model,
            "support": _support_diagnostics(work, continuous),
        }
    results["same_direction"] = all(
        block["model"].get("status") == "ok"
        and block["model"].get("cg4_odds_ratio", 0) > 1
        for block in results["contrasts"].values()
    )
    return results


def build_results() -> dict:
    sample, bridge, ds18 = _load_inputs()
    ours, audited = _unique_audit_frame(sample, bridge, ds18)
    return safe_json(
        {
            "status": "ok",
            "scope": "DR7-era DS18-matched subset; diagnostic validation only",
            "source_notebook": "notebooks/audit_gz1_vs_ds18.ipynb",
            "inputs": {
                "processed_sample": str(SAMPLE_PATH.relative_to(REPO)),
                "photoobjdr7_bridge": (
                    "${CG_GALS_DS18_CACHE:-~/.cache/cg_gals_audit}/"
                    "photoobjdr7_map.csv"
                ),
                "ds18_catalogue": (
                    "${CG_GALS_DS18_CACHE:-~/.cache/cg_gals_audit}/"
                    "ds18/catalog.dat.gz"
                ),
            },
            "composition": _composition_results(ours, audited),
            "early_type_robustness": _independent_early_type_models(
                sample, bridge, ds18
            ),
        }
    )


def main() -> None:
    results = build_results()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(OUTPUT_PATH)
    print(json.dumps(results, indent=2))
    print(f"\nWrote {OUTPUT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
