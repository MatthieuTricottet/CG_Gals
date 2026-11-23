"""Read the numerical results and figure names cited by the manuscript.

The 20 November referee freeze supplies the full-sample results. Its
median-separation rerun supplies the comparisons that depend on compactness.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import config as co

FREEZE = Path(co.BASE_PATH) / "results" / "referee_freeze_20261120"
TABLES = FREEZE / "tables"
MEDIAN = FREEZE / "median_separation"
MEDIAN_TABLES = MEDIAN / "tables"
FIGURES = {
    "main": "fig_centerpiece_compactness.pdf",
    "raw": "fig_centerpiece_compactness_raw.pdf",
}
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
        "labelled_pairs", "components", "permutation_p", "ref_cg_n",
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


def _compactness(estimates: pd.DataFrame, geometry: pd.DataFrame, metadata: dict) -> dict:
    frozen = estimates[estimates["labelset"].eq(LABELSET)]
    comp = frozen[frozen["analysis"].eq("compactness")]
    model = comp[comp["purity"].eq("all") & comp["row"].eq("contrast")]
    cg = geometry[geometry["sample"].eq("CG4")]
    control = geometry[geometry["sample"].eq("Control4C")]
    centre = metadata["centre_kpc"]
    inside = geometry[geometry["window"]]
    out = {
        "centre_kpc": centre,
        "window_kpc": metadata["window_kpc"],
        "mass_ref": metadata["mass_ref"],
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
        "median_Rij_kpc": {
            "CG4": float(cg["med_Rij"].median()),
            "Control4C": float(control["med_Rij"].median()),
        },
        "full": _estimate(_one(model, design="full", radius=None)),
        "full_local": _estimate(_one(model, design="full", radius=centre)),
        "window_no_separation": _estimate(_one(model, design="window_no_separation", radius=centre)),
        "common_slope": _estimate(_one(model, design="common_slope", radius=centre)),
        "separate_slopes": _estimate(_one(model, design="interaction", radius=centre)),
        "matched": _estimate(_one(comp, design="compact_match", row="paired")),
        "matched_sigma": _estimate(_one(comp, design="compact_match_sigma", row="paired")),
    }
    purge = comp[comp["purity"].eq("HMCG_purged") & comp["row"].eq("contrast")]
    out["hmcg_purged"] = {
        "full": _estimate(_one(purge, design="full", radius=None)),
        "separate_slopes": _estimate(_one(purge, design="interaction", radius=centre)),
    }
    return out


def _primary_morphology(primary: pd.DataFrame) -> dict:
    """Read the frozen Table 4 intervals, without refitting their models."""
    rows = primary[
        primary["analysis"].eq("Table 4")
        & primary["labelset"].eq(LABELSET)
        & primary["row"].eq("coefficient")
        & primary["term"].eq("cg")
        & primary["sigma"].eq(True)
    ]
    return {
        control: {
            scope: {
                "status": "ok",
                "n": int(row["n"]),
                "n_clusters": int(row["hosts"]),
                "cg4_odds_ratio": float(row["OR"]),
                "cg4_ci95": [float(row["OR_low"]), float(row["OR_high"])],
                "cg4_p": float(row["p"]),
            }
            for scope in ("all", "satellites")
            for row in [_one(rows, control=control, scope=scope)]
        }
        for control in ("Control4B", "Control4C", "RG4")
    }


def _table5(primary: pd.DataFrame) -> dict:
    rows = primary[primary["analysis"].eq("Table 5") & primary["labelset"].eq(LABELSET)]
    comparisons = {row["control"]: _estimate(row) for _, row in rows.iterrows()}
    pooled_pairs = pd.read_csv(
        TABLES / "02_table5_pairs_pooled.csv", usecols=["control_quartet"]
    )
    control_samples = pooled_pairs["control_quartet"].str.split(":", n=1).str[0]
    comparisons["pooled"]["control_counts"] = control_samples.value_counts().to_dict()
    return comparisons


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
        "n_unestimable_fits": int(len(failures)),
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


def _inner_bins(bins: pd.DataFrame, upper: float = 2.0) -> dict:
    """Plotted Fig. 3a bins below log10(Rmed/kpc) = ``upper`` for CG4 and Control4C."""

    rows = bins[
        bins["labelset"].eq(LABELSET)
        & bins["sample"].isin(["CG4", "Control4C"])
        & bins["bin_right"].le(upper + 1e-9)
        & bins["hosts"].ge(3)
        & bins["raw"].notna()
    ]
    out = {}
    for _, row in rows.iterrows():
        out.setdefault(row["sample"], []).append({
            "bin_left": float(row["bin_left"]), "bin_right": float(row["bin_right"]),
            "n": int(row["n"]), "hosts": int(row["hosts"]),
            "raw": float(row["raw"]), "raw_low": float(row["raw_low"]), "raw_high": float(row["raw_high"]),
            "adjusted": float(row["adjusted"]), "adjusted_low": float(row["adjusted_low"]),
            "adjusted_high": float(row["adjusted_high"]),
        })
    return out


def _submission() -> dict:
    """Paired bootstraps and other checks written by the freeze submission module."""

    path = MEDIAN_TABLES / "07_submission_checks.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load() -> dict:
    """Return the freeze values cited by the manuscript, or ``{}`` if absent."""

    if not TABLES.is_dir():
        return {}
    primary = pd.read_csv(TABLES / "02_frozen_primary_and_compactness.csv", low_memory=False)
    estimates = pd.read_csv(MEDIAN_TABLES / "02_compactness_models.csv", low_memory=False)
    metadata = json.loads((MEDIAN / "provenance.json").read_text())
    geometry = pd.read_csv(MEDIAN_TABLES / "02_quartet_geometry.csv", low_memory=False)
    spectral = pd.read_csv(MEDIAN_TABLES / "03_compactness_ha_quenching.csv", low_memory=False)
    hosts = pd.read_csv(MEDIAN_TABLES / "04_host_environment_window_check.csv", low_memory=False)
    outer = pd.read_csv(MEDIAN_TABLES / "06_outer_bin_contrasts.csv")
    return {
        "source": str(FREEZE.relative_to(Path(co.BASE_PATH))),
        "compactness": _compactness(estimates, geometry, metadata),
        "primary_morphology": _primary_morphology(primary),
        "table5": _table5(primary),
        "galaxy_match": _galaxy_match(primary),
        "spectral": _spectral(spectral),
        "hosts": _hosts(hosts),
        "outer_bin": _outer_bin(outer),
        "submission": _submission(),
        "inner_bins": _inner_bins(pd.read_csv(MEDIAN_TABLES / "06_compactness_bins_audit.csv")),
        "figures": dict(FIGURES),
    }


def copy_figures(destination: str) -> list[str]:
    """Copy the frozen compactness figures next to the generated paper."""

    copied = []
    for name in FIGURES.values():
        source = MEDIAN / "figures" / name
        if source.exists():
            shutil.copy2(source, Path(destination) / name)
            copied.append(name)
    return copied
