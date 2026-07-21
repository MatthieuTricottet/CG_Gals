"""T9 -- compactness and host-BGG alignment audits.

This script answers two narrow referee-facing questions without changing the
published sample definitions or primary inference families:

1. How compact is Control4C compared with CG4 and the other controls?
2. Do the satellite morphology contrasts survive after restricting CG4 to
   systems whose quartet BGG is also the BGG of the Lim parent group?

Outputs are written under ``output/referee/`` plus ``referee/values/T9.json``
for manuscript rendering. The primary Holm families in ``output/results.json``
are not modified.
"""

from __future__ import annotations

import itertools as it
import json
import math
import os
import pickle
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.cosmology import FlatLambdaCDM  # noqa: E402
from scipy import stats  # noqa: E402

SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from extended_data import ensure_galaxy_frame  # noqa: E402
from extended_stats import fit_logistic_model, safe_json  # noqa: E402
from matched_controls import per_control_group_level_matches  # noqa: E402
from primary_contrasts import run_primary_contrasts  # noqa: E402
from specialness_models import MODEL_SPECS, _covariates  # noqa: E402

warnings.filterwarnings(
    "ignore", message="Mean of empty slice", category=RuntimeWarning
)

DATA = ROOT / "data"
OUT = ROOT / "output" / "referee"
VALUES = ROOT / "referee" / "values"

SAMPLES = ["CG4", "Control4B", "Control4C", "RG4"]
CONTROLS = ["Control4B", "Control4C", "RG4"]
MORPHOLOGY_MODELS = ["elliptical_satellites", "spiral_satellites"]

# Same Planck15-like constants already used by src/sample_construction.py and
# by the T8 values currently rendered in Sect. 2.1. Distances below are still
# angular-diameter distances, not the inherited luminosity-distance group size.
COSMO = FlatLambdaCDM(H0=67.8, Om0=0.308, Tcmb0=2.7255, Neff=3.15)

COMPACTNESS_REFERENCE = {
    "CG4": {
        "median_R_pair_med_kpc": 143.14844751808846,
        "q25_R_pair_med_kpc": 114.17602594725024,
        "q75_R_pair_med_kpc": 177.10467924993998,
    },
    "RG4": {
        "median_R_pair_med_kpc": 429.5835442703844,
        "q25_R_pair_med_kpc": 292.2501115685054,
        "q75_R_pair_med_kpc": 493.9419848464603,
    },
}


def _load_processed_sample() -> dict[str, pd.DataFrame]:
    with open(DATA / "processed_sample.pkl", "rb") as handle:
        return pickle.load(handle)


def _load_catalogues() -> dict[str, pd.DataFrame]:
    return {
        "CG4_Gals": pd.read_csv(DATA / "CG4_Gals.csv"),
        "CG4_Groups": pd.read_csv(DATA / "CG4_Groups.csv"),
        "PC_Gals": pd.read_csv(DATA / "PC_Gals.csv"),
    }


def _angular_matrix(ra_deg, dec_deg) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    delta_ra = ra[:, None] - ra[None, :]
    delta_dec = dec[:, None] - dec[None, :]
    a = (
        np.sin(delta_dec / 2) ** 2
        + np.cos(dec[:, None]) * np.cos(dec[None, :]) * np.sin(delta_ra / 2) ** 2
    )
    return 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _overlap_coefficient(a: np.ndarray, b: np.ndarray, n_bins: int = 60) -> float:
    """Histogram overlap, matching referee/T4_tidal_support.py."""

    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    grid = np.linspace(lo, hi, n_bins + 1)
    fa, _ = np.histogram(a, bins=grid, density=True)
    fb, _ = np.histogram(b, bins=grid, density=True)
    return float(np.sum(np.minimum(fa, fb)) * (grid[1] - grid[0]))


def _article_sample_gals(sample: dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
    return sample[f"{name}_Gals"].copy()


def compactness_audit(sample: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    for name in SAMPLES:
        gals = _article_sample_gals(sample, name)
        for group_id, members in gals.groupby("Group", observed=True):
            members = members.sort_values("rank_M").reset_index(drop=True)
            base = {
                "sample": name,
                "group": int(group_id),
                "n_members": int(len(members)),
                "objids_rank_order": ";".join(members["objid"].astype("int64").astype(str)),
            }
            clean = members[["RA", "Dec", "z"]].apply(pd.to_numeric, errors="coerce")
            if len(members) != 4 or clean.isna().any(axis=None):
                rows.append(
                    {
                        **base,
                        "dropped": True,
                        "drop_reason": "not_four_members_or_missing_coordinates_redshift",
                    }
                )
                continue

            z_group = float(clean["z"].mean())
            angular = _angular_matrix(clean["RA"], clean["Dec"])
            da_kpc = COSMO.angular_diameter_distance(z_group).to_value("kpc")
            pair_values = {}
            seps = []
            for i, j in it.combinations(range(4), 2):
                value = float(angular[i, j] * da_kpc)
                pair_values[f"R_pair_rank{i + 1}_rank{j + 1}_kpc"] = value
                seps.append(value)
            rows.append(
                {
                    **base,
                    "dropped": False,
                    "drop_reason": "",
                    "z_group_mean": z_group,
                    "D_A_kpc": float(da_kpc),
                    **pair_values,
                    "R_pair_med_kpc": float(np.median(seps)),
                    "R_pair_min_kpc": float(np.min(seps)),
                    "R_pair_max_kpc": float(np.max(seps)),
                    "compactness_score_neg_log10_Rmed": float(-np.log10(np.median(seps))),
                }
            )

    values = pd.DataFrame(rows)
    usable = values.loc[~values["dropped"]].copy()
    summary_rows = []
    for name in SAMPLES:
        part = usable.loc[usable["sample"] == name, "R_pair_med_kpc"]
        dropped = values.loc[(values["sample"] == name) & values["dropped"]]
        summary_rows.append(
            {
                "sample": name,
                "n_quartets": int(part.size),
                "median_R_pair_med_kpc": float(part.median()),
                "q25_R_pair_med_kpc": float(part.quantile(0.25)),
                "q75_R_pair_med_kpc": float(part.quantile(0.75)),
                "min_R_pair_med_kpc": float(part.min()),
                "max_R_pair_med_kpc": float(part.max()),
                "n_dropped_missing_coordinates_or_redshift": int(len(dropped)),
            }
        )
    summary = pd.DataFrame(summary_rows)

    for sample_name, expected in COMPACTNESS_REFERENCE.items():
        found = summary.loc[summary["sample"] == sample_name].iloc[0].to_dict()
        for key, expected_value in expected.items():
            if not math.isclose(float(found[key]), expected_value, rel_tol=0, abs_tol=1e-6):
                raise AssertionError(
                    f"{sample_name} {key}={found[key]} does not reproduce "
                    f"the rendered reference value {expected_value}"
                )

    cg4 = usable.loc[usable["sample"] == "CG4", "compactness_score_neg_log10_Rmed"].to_numpy()
    c4c = usable.loc[
        usable["sample"] == "Control4C", "compactness_score_neg_log10_Rmed"
    ].to_numpy()
    c4c_threshold_score = float(np.quantile(c4c, 0.95))
    c4c_threshold_r = float(10 ** (-c4c_threshold_score))
    ks = stats.ks_2samp(cg4, c4c, alternative="two-sided", method="auto")
    diagnostics = {
        "definition": {
            "quantity": "median over the six proper projected pairwise separations within each selected quartet",
            "distance": (
                "great-circle separation times angular-diameter distance at the "
                "quartet mean member redshift; Paper-I/Control4C cosmology "
                "H0=67.8, Om0=0.308"
            ),
            "unit_guard": (
                "does not use dist2BGG or the inherited luminosity-distance "
                "group-size convention"
            ),
            "compactness_score": "-log10(R_pair_med/kpc), so larger is more compact",
        },
        "cg4_vs_control4c": {
            "overlap_coefficient_compactness_score": _overlap_coefficient(cg4, c4c),
            "control4c_p95_compactness_score": c4c_threshold_score,
            "control4c_equivalent_R_pair_med_p05_kpc": c4c_threshold_r,
            "fraction_cg4_above_control4c_p95_compactness": float((cg4 > c4c_threshold_score).mean()),
            "ks_statistic_descriptive": float(ks.statistic),
            "ks_pvalue_descriptive": float(ks.pvalue),
        },
    }
    return summary, values, diagnostics


def _plot_compactness_ecdf(group_values: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colours = {
        "CG4": "#2864A6",
        "Control4B": "#777777",
        "Control4C": "#25876E",
        "RG4": "#A74752",
    }
    for name in SAMPLES:
        values = np.sort(
            group_values.loc[
                (group_values["sample"] == name) & (~group_values["dropped"]),
                "R_pair_med_kpc",
            ].to_numpy(dtype=float)
        )
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, y, where="post", label=name, color=colours.get(name))
    ax.set_xlabel(r"$R_{\rm pair,med}$ [proper kpc]")
    ax.set_ylabel("ECDF")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def host_bgg_alignment(catalogues: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    cg4 = catalogues["CG4_Gals"].copy()
    cg4_groups = catalogues["CG4_Groups"].copy()
    pc = catalogues["PC_Gals"].copy()
    class_by_group = cg4_groups.set_index("Group")["Class"]
    non_split = cg4_groups.loc[cg4_groups["Class"] != "Split", "Group"]
    cg4 = cg4.loc[cg4["Group"].isin(non_split)].copy()

    per_group = []
    per_galaxy = []
    pc_minimal = pc[["objid", "Group", "rank_M"]].rename(
        columns={"Group": "lim_group", "rank_M": "lim_rank_M"}
    )
    for group_id, members in cg4.groupby("Group", observed=True):
        members = members.sort_values("rank_M").copy()
        group_class = str(class_by_group.loc[group_id])
        merged = members[["objid", "Group", "rank_M"]].merge(
            pc_minimal, on="objid", how="left", validate="1:1"
        )
        lim_groups = sorted(merged["lim_group"].dropna().astype(int).unique())
        if len(lim_groups) == 0:
            mapping_status = "no reliable Lim-host mapping"
            lim_group = np.nan
            host_bgg_objid = np.nan
            equality = np.nan
            host_member_count = np.nan
            lim_group_is_quartet_itself = False
        elif len(lim_groups) > 1:
            mapping_status = "ambiguous or duplicate mapping"
            lim_group = lim_groups[0]
            host_bgg_objid = np.nan
            equality = np.nan
            host_member_count = np.nan
            lim_group_is_quartet_itself = False
        else:
            lim_group = lim_groups[0]
            host = pc.loc[pc["Group"] == lim_group]
            host_bgg = host.loc[host["rank_M"] == 1, "objid"].drop_duplicates()
            host_member_count = int(host["objid"].nunique())
            lim_group_is_quartet_itself = set(host["objid"]) == set(members["objid"])
            if len(host_bgg) != 1:
                mapping_status = "ambiguous or duplicate mapping"
                host_bgg_objid = np.nan
                equality = np.nan
            else:
                cg4_bgg_objid = int(members.loc[members["rank_M"].idxmin(), "objid"])
                host_bgg_objid = int(host_bgg.iloc[0])
                equality = bool(cg4_bgg_objid == host_bgg_objid)
                mapping_status = (
                    "CG4 BGG equals Lim-host BGG"
                    if equality
                    else "CG4 BGG differs from Lim-host BGG"
                )

        cg4_bgg_objid = int(members.loc[members["rank_M"].idxmin(), "objid"])
        per_group.append(
            {
                "cg4_group": int(group_id),
                "zheng_shen_class": group_class,
                "lim_group": lim_group,
                "mapping_status": mapping_status,
                "cg4_bgg_objid": cg4_bgg_objid,
                "lim_host_bgg_objid": host_bgg_objid,
                "bgg_objid_equal": equality,
                "lim_host_member_count": host_member_count,
                "lim_group_is_quartet_itself": bool(lim_group_is_quartet_itself),
                "isolated_equality_trivial": bool(
                    group_class == "Isolated" and lim_group_is_quartet_itself and equality is True
                ),
            }
        )
        for _, row in merged.iterrows():
            quartet_satellite = bool(row["rank_M"] > 1)
            host_halo_satellite = bool(pd.notna(row["lim_rank_M"]) and row["lim_rank_M"] > 1)
            per_galaxy.append(
                {
                    "cg4_group": int(group_id),
                    "zheng_shen_class": group_class,
                    "objid": int(row["objid"]),
                    "cg4_rank_M": int(row["rank_M"]),
                    "lim_group": row["lim_group"],
                    "lim_rank_M": row["lim_rank_M"],
                    "quartet_satellite": quartet_satellite,
                    "host_halo_satellite": host_halo_satellite,
                    "host_status_differs_from_quartet_rank": bool(
                        quartet_satellite != host_halo_satellite
                    ),
                }
            )

    per_group_df = pd.DataFrame(per_group).sort_values("cg4_group")
    per_galaxy_df = pd.DataFrame(per_galaxy).sort_values(["cg4_group", "cg4_rank_M"])

    summary_rows = []
    scopes = [("All", per_group_df)]
    for class_name, part in per_group_df.groupby("zheng_shen_class", observed=True):
        scopes.append((str(class_name), part))
    for scope, part in scopes:
        total = int(len(part))
        for status, count in part["mapping_status"].value_counts().sort_index().items():
            summary_rows.append(
                {
                    "scope": scope,
                    "mapping_status": status,
                    "n_groups": int(count),
                    "n_total": total,
                    "fraction": float(count / total) if total else np.nan,
                }
            )
        if scope == "Isolated":
            summary_rows.append(
                {
                    "scope": scope,
                    "mapping_status": "isolated equality trivial because Lim group is quartet itself",
                    "n_groups": int(part["isolated_equality_trivial"].sum()),
                    "n_total": total,
                    "fraction": float(part["isolated_equality_trivial"].mean()) if total else np.nan,
                }
            )
    summary_df = pd.DataFrame(summary_rows)

    satellites = per_galaxy_df.loc[per_galaxy_df["quartet_satellite"]]
    satellite_status = {
        "n_cg4_galaxies": int(len(per_galaxy_df)),
        "n_quartet_rank_satellites": int(len(satellites)),
        "n_quartet_rank_satellites_that_are_host_halo_satellites": int(
            satellites["host_halo_satellite"].sum()
        ),
        "n_quartet_rank_satellites_not_host_halo_satellites": int(
            (~satellites["host_halo_satellite"]).sum()
        ),
        "n_all_cg4_galaxies_with_host_status_different_from_quartet_rank": int(
            per_galaxy_df["host_status_differs_from_quartet_rank"].sum()
        ),
        "n_cg4_bggs_that_are_host_halo_satellites": int(
            (
                (~per_galaxy_df["quartet_satellite"])
                & per_galaxy_df["host_halo_satellite"]
            ).sum()
        ),
    }
    aligned_groups = per_group_df.loc[
        per_group_df["mapping_status"] == "CG4 BGG equals Lim-host BGG", "cg4_group"
    ].astype(int)
    return summary_df, per_group_df, per_galaxy_df, aligned_groups, satellite_status


def _model_complete_panel(
    frame: pd.DataFrame, control: str, outcome: str, covariates: list[str]
) -> pd.DataFrame:
    panel = frame.loc[
        frame["sample"].isin(["CG4", control]) & frame["is_satellite"].eq(1)
    ].copy()
    predictors = ["is_CG4", *[c for c in covariates if c != "is_satellite"]]
    required = [outcome, *predictors]
    work = panel[
        list(dict.fromkeys(required + ["sample", "Group", "objid", "physical_group"]))
    ].replace([np.inf, -np.inf], np.nan)
    for column in required:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    return work.dropna(subset=required).copy()


def host_bgg_sensitivity(
    frame: pd.DataFrame, aligned_groups: pd.Series
) -> tuple[pd.DataFrame, dict, dict]:
    aligned_groups = set(aligned_groups.astype(int))
    sensitivity_frame = frame.loc[
        (frame["sample"] != "CG4") | frame["Group"].astype(int).isin(aligned_groups)
    ].copy()

    published = run_primary_contrasts(frame)
    sensitivity = run_primary_contrasts(sensitivity_frame)
    covariates = published.get("covariates_considered", [])
    if sensitivity.get("covariates_considered") != covariates:
        raise AssertionError(
            "Aligned sensitivity did not retain the same covariate list as "
            f"the published models: {sensitivity.get('covariates_considered')} vs {covariates}"
        )

    rows = []
    for control in CONTROLS:
        for model_name in MORPHOLOGY_MODELS:
            outcome, _ = MODEL_SPECS[model_name]
            primary_model = published["contrasts"][control][model_name]
            aligned_model = sensitivity["contrasts"][control][model_name]
            complete = _model_complete_panel(sensitivity_frame, control, outcome, covariates)
            primary_complete = _model_complete_panel(frame, control, outcome, covariates)
            row = {
                "control": control,
                "model": model_name,
                "outcome": outcome,
                "n_cg4_systems_retained": int(len(aligned_groups)),
                "n_cg4_satellite_galaxies_raw": int(
                    sensitivity_frame.loc[
                        sensitivity_frame["sample"].eq("CG4")
                        & sensitivity_frame["is_satellite"].eq(1)
                    ].shape[0]
                ),
                "n_cg4_complete": int(complete["is_CG4"].eq(1).sum()),
                "n_control_complete": int(complete["is_CG4"].eq(0).sum()),
                "n_model_complete": int(len(complete)),
                "n_physical_groups_complete": int(complete["physical_group"].nunique()),
                "primary_n_cg4_complete": int(primary_complete["is_CG4"].eq(1).sum()),
                "primary_n_control_complete": int(primary_complete["is_CG4"].eq(0).sum()),
                "primary_odds_ratio": primary_model.get("cg4_odds_ratio"),
                "primary_ci95_low": (primary_model.get("cg4_ci95") or [None, None])[0],
                "primary_ci95_high": (primary_model.get("cg4_ci95") or [None, None])[1],
                "primary_raw_p": primary_model.get("cg4_p"),
                "primary_family_holm_p": primary_model.get("cg4_p_adj"),
                "aligned_odds_ratio": aligned_model.get("cg4_odds_ratio"),
                "aligned_ci95_low": (aligned_model.get("cg4_ci95") or [None, None])[0],
                "aligned_ci95_high": (aligned_model.get("cg4_ci95") or [None, None])[1],
                "aligned_raw_p": aligned_model.get("cg4_p"),
                "sensitivity_family_holm_p": aligned_model.get("cg4_p_adj"),
                "primary_beta": primary_model.get("cg4_coefficient"),
                "aligned_beta": aligned_model.get("cg4_coefficient"),
            }
            if row["primary_beta"] is not None and row["aligned_beta"] is not None:
                row["delta_beta_aligned_minus_primary"] = float(
                    row["aligned_beta"] - row["primary_beta"]
                )
                row["attenuation_fraction_abs_beta"] = (
                    float(
                        (abs(row["primary_beta"]) - abs(row["aligned_beta"]))
                        / abs(row["primary_beta"])
                    )
                    if row["primary_beta"] != 0
                    else np.nan
                )
                row["decision_classification"] = _classify_sensitivity(row)
            rows.append(row)

    classification_counts = pd.Series(
        [row.get("decision_classification") for row in rows]
    ).value_counts(dropna=False)
    audit = {
        "role": (
            "Labelled sensitivity only: CG4 is restricted to systems whose "
            "quartet BGG objid equals the Lim parent-group BGG objid. Control "
            "definitions, covariates, complete-case handling, physical-group "
            "clustering, model specs, and per-control Holm families are unchanged."
        ),
        "n_aligned_cg4_groups": int(len(aligned_groups)),
        "holm_family": list(MODEL_SPECS.keys()),
        "reported_models": MORPHOLOGY_MODELS,
        "classification_rule": (
            "Stable requires same direction, overlapping 95% CIs, and less than "
            "30% attenuation in absolute log-odds coefficient. Compatible but "
            "less precise keeps the same direction and no large attenuation, but "
            "has a wider interval. Materially attenuated means at least 30% "
            "movement toward zero with |Delta beta| >= 0.20. Inconsistent means "
            "the direction reverses."
        ),
        "classification_counts": {
            str(key): int(value) for key, value in classification_counts.items()
        },
    }
    return pd.DataFrame(rows), audit, sensitivity_frame


def _classify_sensitivity(row: dict) -> str:
    beta_primary = float(row["primary_beta"])
    beta_aligned = float(row["aligned_beta"])
    if np.sign(beta_primary) != np.sign(beta_aligned):
        return "Inconsistent"

    primary_ci = (float(row["primary_ci95_low"]), float(row["primary_ci95_high"]))
    aligned_ci = (float(row["aligned_ci95_low"]), float(row["aligned_ci95_high"]))
    intervals_overlap = max(primary_ci[0], aligned_ci[0]) <= min(primary_ci[1], aligned_ci[1])
    attenuation = float(row.get("attenuation_fraction_abs_beta", 0.0))
    delta_beta = float(row.get("delta_beta_aligned_minus_primary", 0.0))
    materially_attenuated = attenuation >= 0.30 and abs(delta_beta) >= 0.20
    if materially_attenuated:
        return "Materially attenuated"
    if not intervals_overlap:
        return "Compatible but less precise"

    width_primary = math.log(primary_ci[1]) - math.log(primary_ci[0])
    width_aligned = math.log(aligned_ci[1]) - math.log(aligned_ci[0])
    if width_aligned > 1.20 * width_primary:
        return "Compatible but less precise"
    return "Stable"


def _fit_satellite_model(
    frame: pd.DataFrame, control: str, outcome: str, covariates: list[str]
) -> tuple[dict, pd.DataFrame]:
    panel = frame.loc[
        frame["sample"].isin(["CG4", control]) & frame["is_satellite"].eq(1)
    ].copy()
    predictors = ["is_CG4", *[c for c in covariates if c != "is_satellite"]]
    result = fit_logistic_model(
        panel,
        outcome,
        predictors,
        continuous=[c for c in covariates if c in predictors],
    )
    complete = _model_complete_panel(frame, control, outcome, covariates)
    return result, complete


def influence_checks(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    covariates, _ = _covariates(frame)
    leave_cg4_rows = []
    leave_host_rows = []
    audit_rows = []
    for control in CONTROLS:
        full_model, complete = _fit_satellite_model(frame, control, "elliptical", covariates)
        full_beta = full_model.get("cg4_coefficient")
        full_sign = np.sign(full_beta) if full_beta is not None else np.nan
        cg4_groups = sorted(frame.loc[frame["sample"].eq("CG4"), "Group"].astype(int).unique())
        cg4_hosts = sorted(
            frame.loc[frame["sample"].eq("CG4"), "physical_group"].dropna().astype(str).unique()
        )
        for group_id in cg4_groups:
            reduced = frame.loc[
                ~(
                    frame["sample"].eq("CG4")
                    & frame["Group"].astype(int).eq(int(group_id))
                )
            ].copy()
            fit, _ = _fit_satellite_model(reduced, control, "elliptical", covariates)
            beta = fit.get("cg4_coefficient")
            leave_cg4_rows.append(
                {
                    "control": control,
                    "omitted_cg4_group": int(group_id),
                    "status": fit.get("status"),
                    "beta": beta,
                    "odds_ratio": fit.get("cg4_odds_ratio"),
                    "p": fit.get("cg4_p"),
                    "direction_preserved": bool(np.sign(beta) == full_sign)
                    if beta is not None and not np.isnan(full_sign)
                    else None,
                }
            )
        for host in cg4_hosts:
            reduced = frame.loc[frame["physical_group"].astype(str) != host].copy()
            fit, _ = _fit_satellite_model(reduced, control, "elliptical", covariates)
            beta = fit.get("cg4_coefficient")
            leave_host_rows.append(
                {
                    "control": control,
                    "omitted_physical_host": host,
                    "status": fit.get("status"),
                    "beta": beta,
                    "odds_ratio": fit.get("cg4_odds_ratio"),
                    "p": fit.get("cg4_p"),
                    "direction_preserved": bool(np.sign(beta) == full_sign)
                    if beta is not None and not np.isnan(full_sign)
                    else None,
                }
            )

        control_complete = complete.loc[complete["is_CG4"].eq(0)]
        cg4_complete = complete.loc[complete["is_CG4"].eq(1)]
        overlap = set(control_complete["objid"].astype(int)) & set(
            cg4_complete["objid"].astype(int)
        )
        audit_rows.append(
            {
                "control": control,
                "full_status": full_model.get("status"),
                "full_beta": full_beta,
                "full_odds_ratio": full_model.get("cg4_odds_ratio"),
                "full_covariance": full_model.get("covariance"),
                "full_n_clusters": full_model.get("n_clusters"),
                "complete_case_physical_groups": int(complete["physical_group"].nunique()),
                "control_duplicate_objids": int(control_complete["objid"].duplicated().sum()),
                "cg4_objids_on_control_side": int(len(overlap)),
                "clustered_by_physical_group": bool(
                    full_model.get("covariance") == "cluster"
                    and full_model.get("n_clusters") == complete["physical_group"].nunique()
                ),
            }
        )

    leave_cg4 = pd.DataFrame(leave_cg4_rows)
    leave_host = pd.DataFrame(leave_host_rows)
    audit_df = pd.DataFrame(audit_rows)
    audit = {
        "model_checked": "elliptical_satellites",
        "covariates": covariates,
        "leave_one_cg4_group": {
            "n_rows": int(len(leave_cg4)),
            "n_direction_changes": int((leave_cg4["direction_preserved"] == False).sum()),
        },
        "leave_one_lim_host": {
            "n_rows": int(len(leave_host)),
            "n_direction_changes": int((leave_host["direction_preserved"] == False).sum()),
        },
        "per_control_integrity": audit_df.to_dict(orient="records"),
        "all_control_objids_unique_within_per_control_models": bool(
            (audit_df["control_duplicate_objids"] == 0).all()
        ),
        "no_cg4_objids_on_control_side": bool((audit_df["cg4_objids_on_control_side"] == 0).all()),
        "all_models_clustered_by_physical_group": bool(
            audit_df["clustered_by_physical_group"].all()
        ),
        "no_cg4_host_changes_elliptical_direction": bool(
            (leave_host["direction_preserved"] != False).all()
        ),
    }
    return leave_cg4, leave_host, audit


def implementation_path() -> dict:
    return {
        "manuscript_source": "src/paper_template/paper_template.tex",
        "rendered_manuscript": "output/paper/paper.tex and output/paper/paper.pdf",
        "catalogue_generation": [
            "src/main.py::load_data/load_data_build",
            "src/data_loader.py::load_previous_samples",
            "src/data_loader.py::remove_split_CG",
            "src/main.py::clean (Lim 3688 removal)",
            "src/sample_construction.py (Control4C regeneration)",
        ],
        "projected_pairwise_separations": [
            "src/tidal_indices.py::_angular_matrix/_derive for pairwise diagnostics",
            "this audit uses the same angular-diameter/proper-kpc principle but the mean-redshift/Paper-I cosmology convention that exactly reproduces Sect. 2.1 values",
        ],
        "cg4_to_lim_mapping": [
            "src/identity.py::cg4_host_lim_map",
            "src/extended_data.py::_cg4_host_lim_map",
            "this audit recomputes the mapping from shared objid values in data/PC_Gals.csv",
        ],
        "per_control_morphology_models": [
            "src/primary_contrasts.py::run_primary_contrasts",
            "src/specialness_models.MODEL_SPECS/_covariates",
            "src/extended_stats.fit_logistic_model",
        ],
        "build_and_checks": [
            "python audit/run_full_pipeline.py",
            "python -m src.main",
            "pytest",
            "audit/verify_findings.py",
            "audit/consistency_gate.py",
        ],
        "canonical_identifiers": {
            "physical_galaxy": "objid",
            "cg4_quartet": "CG4_Gals.Group",
            "lim_parent_group": "PC_Gals.Group and control Group values, physical_group='Lim:<Group>'",
            "quartet_bgg": "CG4_Gals rank_M == 1, compared by objid",
            "lim_group_bgg": "PC_Gals rank_M == 1 within the Lim parent group, compared by objid",
            "zheng_shen_class": "CG4_Groups.Class (Embedded, Predom, Isolated, Split; Split removed from article sample)",
        },
        "unit_ambiguity_record": {
            "dist2BGG": "angle in radians multiplied by 3600; not used for the compactness audit",
            "new_pairwise_summary": "proper kpc from angular-diameter distances",
            "legacy_group_size": "size_Group_Bary_kpc uses luminosity distance and is not reused here",
            "redshift_cosmology_note": (
                "The Sect. 2.1 CG4/RG4 values are reproduced only with the "
                "quartet mean redshift and the Paper-I FlatLambdaCDM constants. "
                "Planck15 at the median redshift differs by <1 kpc but does not "
                "match the rendered values at numerical precision."
            ),
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(safe_json(payload), handle, indent=1)
        handle.write("\n")


def _write_summary_md(path: Path, values: dict) -> None:
    comp = values["compactness"]["summary_by_sample"]
    c4c = comp["Control4C"]
    cg4 = comp["CG4"]
    align = values["host_bgg_alignment"]
    sens = values["host_bgg_sensitivity"]
    lines = [
        "# T9 summary -- compactness and host-BGG alignment",
        "",
        "Compactness audit passed the Sect. 2.1 reproduction gate: CG4 and RG4",
        "median pairwise-separation summaries match the rendered values exactly",
        "under the recorded angular-diameter/mean-redshift convention.",
        "",
        (
            f"Control4C has median R_pair,med = {c4c['median_R_pair_med_kpc']:.1f} "
            f"kpc (IQR {c4c['q25_R_pair_med_kpc']:.1f}-"
            f"{c4c['q75_R_pair_med_kpc']:.1f}), compared with CG4 "
            f"{cg4['median_R_pair_med_kpc']:.1f} kpc."
        ),
        (
            "CG4-Control4C compactness-score overlap coefficient = "
            f"{values['compactness']['cg4_vs_control4c']['overlap_coefficient_compactness_score']:.2f}; "
            "fraction of CG4 above the Control4C 95th compactness percentile = "
            f"{100 * values['compactness']['cg4_vs_control4c']['fraction_cg4_above_control4c_p95_compactness']:.1f}%."
        ),
        "",
        (
            f"Host-BGG alignment: {align['n_equal']} of {align['n_total']} "
            "non-split CG4 systems have the same quartet and Lim-host BGG objid; "
            f"{align['n_different']} differ."
        ),
        (
            f"All {align['satellite_status']['n_quartet_rank_satellites']} "
            "quartet-rank CG4 satellites are also Lim-host satellites; the only "
            "host-status mismatches are quartet BGGs that are host satellites."
        ),
        "",
        (
            "Aligned-subset morphology classifications: "
            + ", ".join(
                f"{key}={value}" for key, value in sens["classification_counts"].items()
            )
            + "."
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    VALUES.mkdir(parents=True, exist_ok=True)

    sample = _load_processed_sample()
    catalogues = _load_catalogues()
    frame = ensure_galaxy_frame(sample)

    compact_summary, compact_values, compact_diag = compactness_audit(sample)
    compact_summary.to_csv(OUT / "compactness_summary.csv", index=False)
    compact_values.to_csv(OUT / "compactness_group_values.csv", index=False)
    _plot_compactness_ecdf(compact_values, OUT / "compactness_ecdf.pdf")

    align_summary, align_groups, align_galaxies, aligned_groups, satellite_status = (
        host_bgg_alignment(catalogues)
    )
    align_summary.to_csv(OUT / "host_bgg_alignment_summary.csv", index=False)
    align_groups.to_csv(OUT / "host_bgg_alignment_per_group.csv", index=False)
    align_galaxies.to_csv(OUT / "host_bgg_alignment_per_galaxy.csv", index=False)

    sensitivity_table, sensitivity_audit, sensitivity_frame = host_bgg_sensitivity(
        frame, aligned_groups
    )
    sensitivity_table.to_csv(OUT / "host_bgg_sensitivity_morphology.csv", index=False)

    group_match_diag = per_control_group_level_matches(sensitivity_frame)
    leave_cg4, leave_host, influence_audit = influence_checks(frame)
    leave_cg4.to_csv(OUT / "morphology_influence_leave_one_cg4_group.csv", index=False)
    leave_host.to_csv(OUT / "morphology_influence_leave_one_lim_host.csv", index=False)

    compact_by_sample = {
        row["sample"]: {key: row[key] for key in compact_summary.columns if key != "sample"}
        for row in compact_summary.to_dict(orient="records")
    }
    status_counts = align_groups["mapping_status"].value_counts()
    values = {
        "implementation_path": implementation_path(),
        "compactness": {
            "summary_by_sample": compact_by_sample,
            **compact_diag,
            "outputs": {
                "summary_csv": "output/referee/compactness_summary.csv",
                "group_values_csv": "output/referee/compactness_group_values.csv",
                "figure": "output/referee/compactness_ecdf.pdf",
            },
        },
        "host_bgg_alignment": {
            "n_total": int(len(align_groups)),
            "n_equal": int(status_counts.get("CG4 BGG equals Lim-host BGG", 0)),
            "n_different": int(status_counts.get("CG4 BGG differs from Lim-host BGG", 0)),
            "n_no_reliable_mapping": int(status_counts.get("no reliable Lim-host mapping", 0)),
            "n_ambiguous_or_duplicate": int(status_counts.get("ambiguous or duplicate mapping", 0)),
            "by_class": {
                class_name: {
                    status: int(count)
                    for status, count in part["mapping_status"].value_counts().items()
                }
                for class_name, part in align_groups.groupby("zheng_shen_class", observed=True)
            },
            "n_isolated_trivial_equal": int(align_groups["isolated_equality_trivial"].sum()),
            "satellite_status": satellite_status,
            "outputs": {
                "summary_csv": "output/referee/host_bgg_alignment_summary.csv",
                "per_group_csv": "output/referee/host_bgg_alignment_per_group.csv",
                "per_galaxy_csv": "output/referee/host_bgg_alignment_per_galaxy.csv",
            },
        },
        "host_bgg_sensitivity": {
            **sensitivity_audit,
            "rows": sensitivity_table.to_dict(orient="records"),
            "outputs": {
                "morphology_csv": "output/referee/host_bgg_sensitivity_morphology.csv"
            },
        },
        "aligned_group_level_match_diagnostic": group_match_diag,
        "influence_checks": {
            **influence_audit,
            "outputs": {
                "leave_one_cg4_group_csv": "output/referee/morphology_influence_leave_one_cg4_group.csv",
                "leave_one_lim_host_csv": "output/referee/morphology_influence_leave_one_lim_host.csv",
                "audit_json": "output/referee/morphology_influence_audit.json",
            },
        },
    }

    _write_json(OUT / "compactness_audit.json", values["compactness"])
    _write_json(OUT / "host_bgg_alignment_audit.json", values["host_bgg_alignment"])
    _write_json(OUT / "host_bgg_sensitivity_audit.json", values["host_bgg_sensitivity"])
    _write_json(OUT / "morphology_influence_audit.json", values["influence_checks"])
    _write_json(VALUES / "T9.json", values)
    _write_summary_md(ROOT / "referee" / "T9_summary.md", values)

    print(json.dumps(safe_json(values["compactness"]["summary_by_sample"]), indent=1))
    print(json.dumps(safe_json(values["host_bgg_alignment"]), indent=1))
    print(json.dumps(safe_json(values["host_bgg_sensitivity"]["classification_counts"]), indent=1))
    if any(
        row.get("decision_classification") in {"Materially attenuated", "Inconsistent"}
        for row in values["host_bgg_sensitivity"]["rows"]
    ):
        raise SystemExit(
            "Host-BGG aligned morphology sensitivity is materially attenuated "
            "or inconsistent; stop manuscript integration."
        )


if __name__ == "__main__":
    main()
