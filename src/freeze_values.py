"""Values cited from the referee freeze of 20 November 2026.

The compactness comparison, the spectral checks at comparable compactness
and the host-environment check are computed by
``results/referee_freeze_20261120/run.py`` and stored as CSV tables.  This
module reads those tables, without recomputing anything, and returns the
numbers that the manuscript cites.  The freeze directory is left untouched.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import config as co

FREEZE = Path(co.BASE_PATH) / "results" / "referee_freeze_20261120"
TABLES = FREEZE / "tables"
FIGURES = {
    "main": "fig_centerpiece_compactness.pdf",
    "raw": "fig_centerpiece_compactness_raw.pdf",
}
CENTRE_KPC = 175.0
LABELSET = "frozen"


def _estimate(row: pd.Series) -> dict:
    out = {
        key: float(row[key])
        for key in ("difference", "low", "high")
        if key in row and pd.notna(row[key])
    }
    for key in (
        "n", "quartets", "hosts", "cg_n", "control_n", "cg_quartets",
        "control_quartets", "cg_hosts", "control_hosts", "matched_pairs",
        "labelled_pairs", "components", "permutation_p",
    ):
        if key in row and pd.notna(row[key]):
            value = float(row[key])
            out[key] = int(value) if value.is_integer() and key != "permutation_p" else value
    return out


def _one(frame: pd.DataFrame, **conditions) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for key, value in conditions.items():
        if value is None:
            mask &= frame[key].isna()
        elif isinstance(value, float):
            mask &= np.isclose(frame[key].astype(float), value)
        else:
            mask &= frame[key].eq(value)
    selected = frame[mask]
    if len(selected) != 1:
        raise ValueError(f"expected one freeze row for {conditions}, found {len(selected)}")
    return selected.iloc[0]


def _compactness(primary: pd.DataFrame, geometry: pd.DataFrame) -> dict:
    frozen = primary[primary["labelset"].eq(LABELSET)]
    comp = frozen[frozen["analysis"].eq("compactness")]
    model = comp[comp["purity"].eq("all") & comp["row"].eq("contrast")]
    cg = geometry[geometry["sample"].eq("CG4")]
    control = geometry[geometry["sample"].eq("Control4C")]
    window = [
        float(control["mean_Rij"].quantile(0.05)),
        float(cg["mean_Rij"].quantile(0.95)),
    ]
    inside = geometry[geometry["window"]]
    out = {
        "centre_kpc": CENTRE_KPC,
        "window_kpc": window,
        "window_quartets": {
            "CG4": int(inside["sample"].eq("CG4").sum()),
            "Control4C": int(inside["sample"].eq("Control4C").sum()),
        },
        "window_hosts": {
            "CG4": int(inside.loc[inside["sample"].eq("CG4"), "physical_host_id"].nunique()),
            "Control4C": int(
                inside.loc[inside["sample"].eq("Control4C"), "physical_host_id"].nunique()
            ),
        },
        "median_mean_Rij_kpc": {
            "CG4": float(cg["mean_Rij"].median()),
            "Control4C": float(control["mean_Rij"].median()),
        },
        "full": _estimate(_one(model, design="full", radius=None)),
        "common_slope": _estimate(_one(model, design="common_slope", radius=CENTRE_KPC)),
        "separate_slopes": _estimate(_one(model, design="interaction", radius=CENTRE_KPC)),
        "matched": _estimate(_one(comp, design="compact_match", row="paired")),
        "matched_sigma": _estimate(_one(comp, design="compact_match_sigma", row="paired")),
    }
    purge = comp[comp["purity"].eq("HMCG_purged") & comp["row"].eq("contrast")]
    out["hmcg_purged"] = {
        "full": _estimate(_one(purge, design="full", radius=None)),
        "separate_slopes": _estimate(_one(purge, design="interaction", radius=CENTRE_KPC)),
    }
    return out


def _table5(primary: pd.DataFrame) -> dict:
    rows = primary[primary["analysis"].eq("Table 5") & primary["labelset"].eq(LABELSET)]
    return {row["control"]: _estimate(row) for _, row in rows.iterrows()}


def _galaxy_match(primary: pd.DataFrame) -> dict:
    rows = primary[primary["analysis"].eq("galaxy match") & primary["labelset"].eq(LABELSET)]
    out = {}
    for scope in ("all", "satellites"):
        part = rows[rows["scope"].eq(scope)]
        matched = _one(part, row="matched")
        decisive = _one(part, row="doubly_labelled")
        out[scope] = {
            "matched_pairs": int(matched["matched_pairs"]),
            "decisive_pairs": int(decisive["matched_pairs"]),
        }
    original = primary[
        primary["analysis"].eq("galaxy match")
        & primary["labelset"].eq("original")
        & primary["row"].eq("doubly_labelled")
    ]
    for _, row in original.iterrows():
        out[row["scope"]]["decisive_pairs_original"] = int(row["matched_pairs"])
    return out


def _spectral(table: pd.DataFrame) -> dict:
    frozen = table[
        table["labelset"].eq(LABELSET)
        & table["design"].eq("window")
        & table["row"].eq("contrast")
    ]

    def pick(scope, outcome, spec, exclusion, slope):
        return _estimate(
            _one(frozen, scope=scope, outcome=outcome, spec=spec, exclusion=exclusion, slope=slope)
        )

    mid = "10_to_10.5"
    failures = table[table["labelset"].eq(LABELSET) & table["row"].eq("failure")]
    variants = [
        ("historical", "strong_halpha", "baseline", "none"),
        ("exact", "strong_exact", "baseline", "none"),
        ("exact_es_common", "strong_exact", "morphology_common", "none"),
        ("exact_aperture", "strong_exact", "aperture", "none"),
        ("exact_aperture_kewley", "strong_exact", "aperture", "Kewley_AGN"),
        ("exact_aperture_kauffmann", "strong_exact", "aperture", "Kauffmann_AGN_composite"),
        ("quenched", "quenched", "baseline", "none"),
    ]
    supplement_rows = [
        {
            "key": key,
            "common": pick(mid, outcome, spec, exclusion, "common"),
            "separate": pick(mid, outcome, spec, exclusion, "interaction"),
        }
        for key, outcome, spec, exclusion in variants
    ]
    paired = table[
        table["labelset"].eq(LABELSET)
        & table["row"].eq("paired")
        & table["scope"].eq(mid)
        & table["spec"].eq("baseline")
        & table["exclusion"].eq("none")
    ]
    matched_rows = [
        {
            "key": outcome,
            "match": _estimate(_one(paired, outcome=outcome, design="compact_match")),
            "match_sigma": _estimate(_one(paired, outcome=outcome, design="compact_match_sigma")),
        }
        for outcome in ("strong_exact", "quenched")
    ]
    return {
        "supplement_rows": supplement_rows,
        "matched_rows": matched_rows,
        "all_satellites": {
            "exact_common": pick("satellites", "strong_exact", "baseline", "none", "common"),
            "quenched_common": pick("satellites", "quenched", "baseline", "none", "common"),
        },
        "intermediate_mass": {
            "exact_common": pick(mid, "strong_exact", "baseline", "none", "common"),
            "exact_separate": pick(mid, "strong_exact", "baseline", "none", "interaction"),
            "exact_common_sample": pick(mid, "strong_exact", "morphology_common", "none", "common"),
            "exact_aperture": pick(mid, "strong_exact", "aperture", "none", "common"),
            "exact_aperture_kewley": pick(mid, "strong_exact", "aperture", "Kewley_AGN", "common"),
            "exact_aperture_kauffmann": pick(
                mid, "strong_exact", "aperture", "Kauffmann_AGN_composite", "common"
            ),
            "quenched_common": pick(mid, "quenched", "baseline", "none", "common"),
            "quenched_separate": pick(mid, "quenched", "baseline", "none", "interaction"),
        },
        "n_singular_fits": int(len(failures)),
        "singular_designs": sorted(failures["design"].unique().tolist()),
    }


def _hosts(table: pd.DataFrame) -> dict:
    dist = table[table["row"].eq("distribution")]
    ranks = table[table["row"].eq("rank_sum")]
    out = {}
    for term in ("N_Lim", "host_mass"):
        out[term] = {
            sample: float(_one(dist, sample=sample, term=term)["median"])
            for sample in ("CG4", "retained", "excluded_cores")
        }
        out[term]["p_retained_vs_excluded"] = float(
            _one(ranks, sample="retained", term=term, control="excluded_cores")["p"]
        )
    return out


def _outer_bin(table: pd.DataFrame) -> dict:
    row = _one(table, sample="Control4C", labelset=LABELSET)
    return {
        "difference": float(row["difference"]),
        "low": float(row["low"]),
        "high": float(row["high"]),
        "n": int(row["upper_n"]),
        "hosts": int(row["upper_hosts"]),
    }


def load() -> dict:
    """Return the freeze values cited by the manuscript, or ``{}`` if absent."""

    if not TABLES.is_dir():
        return {}
    primary = pd.read_csv(TABLES / "02_frozen_primary_and_compactness.csv", low_memory=False)
    geometry = pd.read_csv(TABLES / "02_quartet_geometry.csv", low_memory=False)
    spectral = pd.read_csv(TABLES / "03_compactness_ha_quenching.csv", low_memory=False)
    hosts = pd.read_csv(TABLES / "04_host_environment_window_check.csv", low_memory=False)
    outer = pd.read_csv(TABLES / "06_outer_bin_contrasts.csv")
    return {
        "source": str(FREEZE.relative_to(Path(co.BASE_PATH))),
        "compactness": _compactness(primary, geometry),
        "table5": _table5(primary),
        "galaxy_match": _galaxy_match(primary),
        "spectral": _spectral(spectral),
        "hosts": _hosts(hosts),
        "outer_bin": _outer_bin(outer),
        "figures": dict(FIGURES),
    }


def copy_figures(destination: str) -> list[str]:
    """Copy the frozen compactness figures next to the generated paper."""

    copied = []
    for name in FIGURES.values():
        source = FREEZE / "figures" / name
        if source.exists():
            shutil.copy2(source, Path(destination) / name)
            copied.append(name)
    return copied
