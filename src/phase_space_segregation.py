"""BGG-centric phase-space segregation tests for CG4 and RG4 satellites."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from extended_data import C_KMS, ensure_galaxy_frame, first_existing
    from extended_stats import (
        bootstrap_difference,
        fit_logistic_model,
        safe_float,
        safe_json,
        standardized_mean_difference,
    )
except ModuleNotFoundError:  # pragma: no cover
    from .extended_data import C_KMS, ensure_galaxy_frame, first_existing
    from .extended_stats import (
        bootstrap_difference,
        fit_logistic_model,
        safe_float,
        safe_json,
        standardized_mean_difference,
    )


MODULE_VERSION = "2026-06-13"
RNG_SEED = 20260613
SAMPLES = ["CG4", "RG4"]
DISTANCE_BIN_ORDER = ["inner", "middle", "outer"]
DISTANCE_BIN_LABELS = {
    "inner": "Nearest satellite",
    "middle": "Second satellite",
    "outer": "Outer satellites",
}
PHASE_BIN_ORDER = [
    "inner_low_velocity",
    "inner_high_velocity",
    "outer_low_velocity",
    "outer_high_velocity",
]
PHASE_BIN_LABELS = {
    "inner_low_velocity": "Inner, low |dv|",
    "inner_high_velocity": "Inner, high |dv|",
    "outer_low_velocity": "Outer, low |dv|",
    "outer_high_velocity": "Outer, high |dv|",
}
OUTCOMES = {
    "quenched": "quenched",
    "early_type": "early_type",
    "elliptical": "elliptical",
}


def _numeric(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _finite_median(values) -> float | None:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.median())


def _finite_mean(values) -> float | None:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _series_from_aliases(
    frame: pd.DataFrame, aliases: list[str], mapping: dict[str, str | None], key: str
) -> pd.Series:
    column = first_existing(frame, aliases)
    mapping[key] = column
    return _numeric(frame, column)


def _build_group_uid(frame: pd.DataFrame) -> pd.Series:
    if "group_uid" in frame:
        return frame["group_uid"].astype(str)
    group_column = first_existing(frame, ["Group", "Id_CG", "Id_LT", "group_id"])
    if group_column is None:
        return pd.Series(frame.index.astype(str), index=frame.index)
    sample = frame.get("sample", pd.Series("unknown", index=frame.index)).astype(str)
    return sample + ":" + frame[group_column].astype(str)


def _gapper_scale(values: Iterable[float]) -> float | None:
    clean = np.sort(pd.to_numeric(pd.Series(values), errors="coerce").dropna())
    n = len(clean)
    if n < 3:
        return None
    gaps = np.diff(clean)
    weights = np.arange(1, n) * np.arange(n - 1, 0, -1)
    sigma = math.sqrt(math.pi) * np.sum(weights * gaps) / (n * (n - 1))
    return float(sigma) if np.isfinite(sigma) and sigma > 0 else None


def compute_velocity_offsets(frame: pd.DataFrame) -> pd.DataFrame:
    """Add BGG-relative velocity offsets and normalized offsets to a frame."""

    work = frame.copy()
    if "group_uid" not in work:
        work["group_uid"] = _build_group_uid(work)
    if "rank" not in work:
        rank_column = first_existing(work, ["rank_M", "rank_M_CG", "rank_M_LT"])
        work["rank"] = _numeric(work, rank_column)
    if "z_numeric" not in work:
        z_column = first_existing(work, ["z", "zobs_CG", "z_gal"])
        work["z_numeric"] = _numeric(work, z_column)
    if "z_group_numeric" not in work:
        group_z_column = first_existing(
            work, ["z_group", "group_z_group", "Yang_z_CMB_group"]
        )
        work["z_group_numeric"] = _numeric(work, group_z_column)

    bgg_mask = work["rank"].eq(1)
    if "is_bgg" in work:
        bgg_mask = bgg_mask | pd.to_numeric(work["is_bgg"], errors="coerce").eq(1)
    if "dist2BGG_projected_kpc" in work:
        bgg_mask = bgg_mask | work["dist2BGG_projected_kpc"].eq(0)
    elif "dist2BGG_kpc" in work:
        bgg_mask = bgg_mask | work["dist2BGG_kpc"].eq(0)

    bgg_z = (
        work.loc[bgg_mask & work["z_numeric"].notna(), ["group_uid", "z_numeric"]]
        .drop_duplicates("group_uid")
        .set_index("group_uid")["z_numeric"]
    )
    work["z_bgg_numeric"] = work["group_uid"].map(bgg_z)
    missing_group_z = work["z_group_numeric"].isna()
    work.loc[missing_group_z, "z_group_numeric"] = work.groupby("group_uid")[
        "z_numeric"
    ].transform("median")[missing_group_z]
    missing_bgg_z = work["z_bgg_numeric"].isna()
    work.loc[missing_bgg_z, "z_bgg_numeric"] = work["z_group_numeric"][
        missing_bgg_z
    ]

    work["dv_to_bgg"] = (
        C_KMS
        * (work["z_numeric"] - work["z_bgg_numeric"])
        / (1 + work["z_group_numeric"])
    )
    work["abs_dv_to_bgg"] = work["dv_to_bgg"].abs()

    if "velocity_dispersion" not in work:
        sigma_column = first_existing(work, ["Vdisp", "sigma_v", "group_Vdisp"])
        work["velocity_dispersion"] = _numeric(work, sigma_column)
    sigma = pd.to_numeric(work["velocity_dispersion"], errors="coerce")
    sigma = sigma.where(sigma > 0)

    group_scale = {}
    for group_uid, group in work.groupby("group_uid"):
        velocities = (
            C_KMS
            * (group["z_numeric"] - group["z_numeric"].median())
            / (1 + group["z_numeric"].median())
        )
        group_scale[group_uid] = _gapper_scale(velocities)
    fallback_sigma = work["group_uid"].map(group_scale).astype(float)
    work["velocity_scale_source"] = np.where(sigma.notna(), "catalogue", "gapper")
    work["sigma_v_used"] = sigma.fillna(fallback_sigma).where(lambda value: value > 0)
    work["abs_dv_norm"] = work["abs_dv_to_bgg"] / work["sigma_v_used"]
    return work


def _prepare_frame(data) -> tuple[pd.DataFrame, dict[str, str | None], list[str]]:
    frame = ensure_galaxy_frame(data).replace([np.inf, -np.inf], np.nan).copy()
    mapping: dict[str, str | None] = {}
    warnings: list[str] = []
    if frame.empty:
        return frame, mapping, warnings

    if "sample" not in frame:
        if "is_CG4" in frame:
            frame["sample"] = np.where(frame["is_CG4"].eq(1), "CG4", "RG4")
        else:
            warnings.append("No sample label found; all rows were treated as RG4.")
            frame["sample"] = "RG4"
    if "is_CG4" not in frame:
        frame["is_CG4"] = frame["sample"].eq("CG4").astype(int)

    frame["group_uid"] = _build_group_uid(frame)
    frame["logMstar"] = (
        frame["logMstar"]
        if "logMstar" in frame
        else _series_from_aliases(
            frame, ["lgm_tot_p50", "lgm", "logMstar"], mapping, "stellar_mass"
        )
    )
    if "rank" not in frame:
        frame["rank"] = _series_from_aliases(
            frame, ["rank_M", "rank_M_CG", "rank_M_LT"], mapping, "mass_rank"
        )
    else:
        mapping["mass_rank"] = "rank"
        frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    if "is_satellite" not in frame:
        frame["is_satellite"] = np.where(frame["rank"].notna(), frame["rank"] > 1, np.nan)
    if "is_bgg" not in frame:
        frame["is_bgg"] = np.where(frame["rank"].notna(), frame["rank"].eq(1), np.nan)

    if "quenched" not in frame:
        status_column = first_existing(frame, ["sSFR_status", "SFRcategory"])
        mapping["sfr_class"] = status_column
        status = frame.get(status_column, pd.Series("", index=frame.index)).astype(str)
        status = status.str.lower()
        # only measured classes count; missing sSFR (NosSFR) stays NaN
        valid = status.isin(["quenched", "starforming", "q", "m", "g"])
        frame["quenched"] = np.where(
            valid,
            status.isin(["quenched", "q"]).astype(float),
            np.nan,
        )
    else:
        mapping["sfr_class"] = "quenched"
        frame["quenched"] = pd.to_numeric(frame["quenched"], errors="coerce")

    if "elliptical" not in frame:
        morphology_column = first_existing(frame, ["morphology", "morph_class"])
        mapping["morphology"] = morphology_column
        morphology = frame.get(morphology_column, pd.Series("", index=frame.index))
        morphology = morphology.astype(str).str.lower()
        frame["elliptical"] = np.where(
            morphology.isin(["elliptical", "spiral"]),
            morphology.eq("elliptical").astype(float),
            np.nan,
        )
    else:
        mapping["morphology"] = "elliptical"
        frame["elliptical"] = pd.to_numeric(frame["elliptical"], errors="coerce")
    frame["early_type"] = frame["elliptical"]
    warnings.append(
        "No S0/lenticular class is available; early_type is therefore elliptical-only."
    )

    if "dist2BGG_projected_kpc" not in frame:
        if "dist2BGG_kpc" in frame:
            frame["dist2BGG_projected_kpc"] = pd.to_numeric(
                frame["dist2BGG_kpc"], errors="coerce"
            )
            mapping["projected_distance"] = "dist2BGG_kpc"
        else:
            distance_column = first_existing(frame, ["dist2BGG", "R_BGG"])
            mapping["projected_distance"] = distance_column
            frame["dist2BGG_projected_kpc"] = _numeric(frame, distance_column)
            warnings.append(
                "Projected distance is not in kpc; using the available distance column as a ranking proxy."
            )
    else:
        mapping["projected_distance"] = "dist2BGG_projected_kpc"

    rank_distance_column = first_existing(frame, ["rank_dist", "distance_rank"])
    mapping["distance_rank"] = rank_distance_column
    frame["rank_dist_raw"] = _numeric(frame, rank_distance_column)

    if "z_numeric" not in frame:
        frame["z_numeric"] = _series_from_aliases(
            frame, ["z", "zobs_CG", "z_gal"], mapping, "galaxy_redshift"
        )
    else:
        mapping["galaxy_redshift"] = "z_numeric"
        frame["z_numeric"] = pd.to_numeric(frame["z_numeric"], errors="coerce")
    if "z_group_numeric" not in frame:
        frame["z_group_numeric"] = _series_from_aliases(
            frame,
            ["z_group", "group_z_group", "Yang_z_CMB_group"],
            mapping,
            "group_redshift",
        )
    else:
        mapping["group_redshift"] = "z_group_numeric"
        frame["z_group_numeric"] = pd.to_numeric(frame["z_group_numeric"], errors="coerce")

    sigma_column = first_existing(frame, ["velocity_dispersion", "Vdisp", "sigma_v"])
    mapping["velocity_dispersion"] = sigma_column
    if sigma_column and sigma_column != "velocity_dispersion":
        frame["velocity_dispersion"] = _numeric(frame, sigma_column)
    frame = compute_velocity_offsets(frame)
    return frame, mapping, warnings


def _assign_bins(satellites: pd.DataFrame) -> pd.DataFrame:
    work = satellites.copy()
    work["satellite_distance_rank"] = work.groupby("group_uid")[
        "dist2BGG_projected_kpc"
    ].rank(method="first")
    work["distance_bin"] = pd.Series(pd.NA, index=work.index, dtype=object)
    work.loc[work["satellite_distance_rank"].eq(1), "distance_bin"] = "inner"
    work.loc[work["satellite_distance_rank"].eq(2), "distance_bin"] = "middle"
    work.loc[work["satellite_distance_rank"].ge(3), "distance_bin"] = "outer"
    work["distance_binary"] = np.where(
        work["distance_bin"].eq("inner"), "inner", "outer"
    )
    velocity_complete = work["abs_dv_norm"].notna().mean()
    velocity_threshold = 1.0
    if velocity_complete < 0.5:
        velocity_threshold = _finite_median(work["abs_dv_norm"])
    if velocity_threshold is None:
        work["velocity_bin"] = np.nan
        work["phase_space_bin"] = np.nan
        return work
    work["velocity_bin"] = np.where(
        work["abs_dv_norm"] <= velocity_threshold,
        "low_velocity",
        "high_velocity",
    )
    work["phase_space_bin"] = work["distance_binary"] + "_" + work["velocity_bin"]
    work.attrs["velocity_threshold"] = float(velocity_threshold)
    return work


def prepare_phase_space_satellite_sample(data) -> pd.DataFrame:
    """Return the CG4/RG4 satellite sample with derived phase-space columns."""

    frame, _, _ = _prepare_frame(data)
    if frame.empty:
        return frame
    satellites = frame.loc[
        frame["sample"].isin(SAMPLES)
        & pd.to_numeric(frame["is_satellite"], errors="coerce").eq(1)
        & ~pd.to_numeric(frame["is_bgg"], errors="coerce").eq(1)
    ].copy()
    return _assign_bins(satellites)


def _cut_counts(frame: pd.DataFrame) -> dict[str, list[dict[str, int | str]]]:
    cuts = [
        ("input_cg4_rg4", lambda value: value["sample"].isin(SAMPLES)),
        (
            "satellites_excluding_bgg",
            lambda value: pd.to_numeric(value["is_satellite"], errors="coerce").eq(1)
            & ~pd.to_numeric(value["is_bgg"], errors="coerce").eq(1),
        ),
        ("valid_stellar_mass", lambda value: value["logMstar"].notna()),
        ("valid_sfr_class", lambda value: value["quenched"].notna()),
        (
            "valid_projected_distance",
            lambda value: value["dist2BGG_projected_kpc"].notna(),
        ),
        ("valid_redshift", lambda value: value["z_numeric"].notna()),
        ("valid_velocity_offset", lambda value: value["abs_dv_to_bgg"].notna()),
        ("valid_morphology", lambda value: value["early_type"].notna()),
    ]
    counts = {sample: [] for sample in SAMPLES}
    for sample in SAMPLES:
        work = frame.copy()
        for name, selector in cuts:
            work = work.loc[selector(work)]
            counts[sample].append(
                {"cut": name, "n": int(work.loc[work["sample"].eq(sample)].shape[0])}
            )
    return counts


def cluster_bootstrap_fraction(
    frame: pd.DataFrame,
    outcome: str,
    *,
    group_col: str = "group_uid",
    n_boot: int = 2000,
    seed: int = RNG_SEED,
) -> dict[str, object]:
    """Estimate a binary fraction with group-cluster bootstrap uncertainty."""

    required = [group_col, outcome]
    if any(column not in frame for column in required):
        return {
            "status": "skipped",
            "reason": "missing_required_columns",
            "fraction": None,
            "ci68": [None, None],
            "stderr": None,
            "n": 0,
            "n_groups": 0,
        }
    work = frame[required].replace([np.inf, -np.inf], np.nan).dropna()
    work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    work = work.loc[work[outcome].isin([0, 1])]
    if work.empty:
        return {
            "status": "skipped",
            "reason": "no_complete_cases",
            "fraction": None,
            "ci68": [None, None],
            "stderr": None,
            "n": 0,
            "n_groups": 0,
        }

    grouped = [group[outcome].to_numpy(dtype=float) for _, group in work.groupby(group_col)]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        draw = rng.integers(0, len(grouped), len(grouped))
        values = np.concatenate([grouped[position] for position in draw])
        boot[index] = values.mean()
    low, high = np.quantile(boot, [0.16, 0.84])
    return {
        "status": "ok",
        "fraction": float(work[outcome].mean()),
        "median_bootstrap": float(np.median(boot)),
        "ci68": [float(low), float(high)],
        "stderr": float(np.std(boot, ddof=1)),
        "n": int(len(work)),
        "n_groups": int(work[group_col].nunique()),
    }


def _cluster_bootstrap_difference(
    treated: pd.DataFrame,
    control: pd.DataFrame,
    outcome: str,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    tx = cluster_bootstrap_fraction(treated, outcome, n_boot=n_boot, seed=seed)
    cx = cluster_bootstrap_fraction(control, outcome, n_boot=n_boot, seed=seed + 1)
    if tx["fraction"] is None or cx["fraction"] is None:
        return {"delta": None, "ci68": [None, None], "stderr": None}

    def grouped_values(frame):
        work = frame[["group_uid", outcome]].dropna()
        return [
            group[outcome].to_numpy(dtype=float) for _, group in work.groupby("group_uid")
        ]

    tx_groups = grouped_values(treated)
    cx_groups = grouped_values(control)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        tx_draw = rng.integers(0, len(tx_groups), len(tx_groups))
        cx_draw = rng.integers(0, len(cx_groups), len(cx_groups))
        tx_values = np.concatenate([tx_groups[position] for position in tx_draw])
        cx_values = np.concatenate([cx_groups[position] for position in cx_draw])
        boot[index] = tx_values.mean() - cx_values.mean()
    low, high = np.quantile(boot, [0.16, 0.84])
    return {
        "delta": float(tx["fraction"] - cx["fraction"]),
        "ci68": [float(low), float(high)],
        "stderr": float(np.std(boot, ddof=1)),
    }


def _fraction_block(
    frame: pd.DataFrame,
    outcome: str,
    *,
    n_boot: int,
    min_total: int,
    min_per_sample: int,
    seed: int,
) -> dict[str, object]:
    cg = frame.loc[frame["sample"].eq("CG4")]
    rg = frame.loc[frame["sample"].eq("RG4")]
    result = {
        "CG4": cluster_bootstrap_fraction(cg, outcome, n_boot=n_boot, seed=seed),
        "RG4": cluster_bootstrap_fraction(rg, outcome, n_boot=n_boot, seed=seed + 1),
        "delta_CG4_minus_RG4": _cluster_bootstrap_difference(
            cg, rg, outcome, n_boot=n_boot, seed=seed + 2
        ),
    }
    result["low_N"] = bool(
        result["CG4"]["n"] + result["RG4"]["n"] < min_total
        or result["CG4"]["n"] < min_per_sample
        or result["RG4"]["n"] < min_per_sample
    )
    return result


def summarize_binned_fractions(
    satellites: pd.DataFrame,
    bin_col: str,
    bin_order: list[str],
    *,
    n_boot: int = 2000,
    min_total: int = 20,
    min_per_sample: int = 10,
) -> dict[str, dict[str, object]]:
    """Return clustered fractions for each outcome in each requested bin."""

    summaries: dict[str, dict[str, object]] = {}
    for bin_index, bin_name in enumerate(bin_order):
        part = satellites.loc[satellites[bin_col].eq(bin_name)]
        summaries[bin_name] = {
            outcome_name: _fraction_block(
                part.dropna(subset=[outcome_col]),
                outcome_col,
                n_boot=n_boot,
                min_total=min_total,
                min_per_sample=min_per_sample,
                seed=RNG_SEED + 17 * bin_index + 101 * outcome_index,
            )
            for outcome_index, (outcome_name, outcome_col) in enumerate(
                OUTCOMES.items()
            )
        }
    return summaries


def _global_statistics(satellites: pd.DataFrame, n_boot: int) -> dict[str, object]:
    result = {}
    for sample in SAMPLES:
        part = satellites.loc[satellites["sample"].eq(sample)]
        result[sample] = {
            "n_satellites": int(len(part)),
            "n_groups": int(part["group_uid"].nunique()),
            "median_logMstar": _finite_median(part["logMstar"]),
            "median_redshift": _finite_median(part["z_numeric"]),
            "quenched_fraction": cluster_bootstrap_fraction(
                part, "quenched", n_boot=n_boot
            ),
            "early_type_fraction": cluster_bootstrap_fraction(
                part, "early_type", n_boot=n_boot
            ),
            "elliptical_fraction": cluster_bootstrap_fraction(
                part, "elliptical", n_boot=n_boot
            ),
            "median_projected_distance_to_bgg_kpc": _finite_median(
                part["dist2BGG_projected_kpc"]
            ),
            "median_abs_dv_to_bgg_kms": _finite_median(part["abs_dv_to_bgg"]),
            "median_abs_dv_norm": _finite_median(part["abs_dv_norm"]),
        }
    return result


def _add_model_terms(satellites: pd.DataFrame, include_velocity: bool = False):
    work = satellites.copy()
    work["distance_bin"] = pd.Categorical(
        work["distance_bin"], DISTANCE_BIN_ORDER, ordered=True
    )
    dummies = pd.get_dummies(work["distance_bin"], prefix="distance", dtype=float)
    dummies = dummies.drop(columns=["distance_inner"], errors="ignore")
    for column in dummies:
        work[column] = dummies[column]
        work[f"is_CG4_x_{column}"] = work["is_CG4"].astype(float) * dummies[column]

    predictors = ["is_CG4", "logMstar", "z_numeric", *list(dummies.columns)]
    predictors += [f"is_CG4_x_{column}" for column in dummies.columns]
    continuous = ["logMstar", "z_numeric"]

    optional_controls = [
        "log_group_mass",
        "log_group_luminosity",
        "velocity_dispersion",
        "dominance",
    ]
    for column in optional_controls:
        if column in work:
            complete = pd.to_numeric(work[column], errors="coerce").notna().mean()
            if complete >= 0.6 and work[column].nunique(dropna=True) > 1:
                predictors.append(column)
                continuous.append(column)

    if include_velocity:
        work["is_CG4_x_abs_dv_norm"] = work["is_CG4"].astype(float) * work[
            "abs_dv_norm"
        ]
        predictors += ["abs_dv_norm", "is_CG4_x_abs_dv_norm"]
        continuous.append("abs_dv_norm")

    return work, predictors, continuous


def _fit_models(satellites: pd.DataFrame, velocity_available: bool) -> dict[str, object]:
    models = {}
    for outcome_name, outcome_col in [
        ("quenched", "quenched"),
        ("early_type", "early_type"),
    ]:
        work, predictors, continuous = _add_model_terms(satellites)
        models[f"{outcome_name}_distance"] = fit_logistic_model(
            work,
            outcome_col,
            predictors,
            continuous=continuous,
            cluster_col="group_uid",
            min_n=30,
            min_class=5,
        )
        if velocity_available:
            vwork, vpredictors, vcontinuous = _add_model_terms(
                satellites.dropna(subset=["abs_dv_norm"]), include_velocity=True
            )
            models[f"{outcome_name}_phase_space"] = fit_logistic_model(
                vwork,
                outcome_col,
                vpredictors,
                continuous=vcontinuous,
                cluster_col="group_uid",
                min_n=30,
                min_class=5,
            )
    models["quenched"] = models.get("quenched_phase_space") or models.get(
        "quenched_distance"
    )
    models["elliptical"] = models.get("early_type_phase_space") or models.get(
        "early_type_distance"
    )
    return models


def _match_pairs(
    frame: pd.DataFrame, mass_tolerance: float, z_tolerance: float
) -> list[dict[str, object]]:
    work = frame.dropna(subset=["logMstar", "z_numeric"]).copy()
    cg = work.loc[work["sample"].eq("CG4")]
    rg = work.loc[work["sample"].eq("RG4")]
    available = set(rg.index)
    pairs = []
    for cg_index, row in cg.sort_values(["logMstar", "z_numeric"]).iterrows():
        if not available:
            break
        candidates = rg.loc[list(available)]
        dm = (candidates["logMstar"] - row["logMstar"]).abs()
        dz = (candidates["z_numeric"] - row["z_numeric"]).abs()
        candidates = candidates.loc[(dm <= mass_tolerance) & (dz <= z_tolerance)]
        if candidates.empty:
            continue
        distance = np.sqrt(
            ((candidates["logMstar"] - row["logMstar"]) / mass_tolerance) ** 2
            + ((candidates["z_numeric"] - row["z_numeric"]) / z_tolerance) ** 2
        )
        control_index = distance.idxmin()
        available.remove(control_index)
        pairs.append(
            {
                "treated_index": cg_index,
                "control_index": control_index,
                "distance": float(distance.loc[control_index]),
            }
        )
    return pairs


def _balance(frame, treated, control) -> dict[str, object]:
    diagnostics = {}
    for column in ["logMstar", "z_numeric"]:
        diagnostics[column] = {
            "before": {
                "CG4_mean": _finite_mean(frame.loc[frame["sample"].eq("CG4"), column]),
                "RG4_mean": _finite_mean(frame.loc[frame["sample"].eq("RG4"), column]),
                "CG4_median": _finite_median(
                    frame.loc[frame["sample"].eq("CG4"), column]
                ),
                "RG4_median": _finite_median(
                    frame.loc[frame["sample"].eq("RG4"), column]
                ),
                "smd": standardized_mean_difference(
                    frame.loc[frame["sample"].eq("CG4"), column],
                    frame.loc[frame["sample"].eq("RG4"), column],
                ),
            },
            "after": {
                "CG4_mean": _finite_mean(treated[column]),
                "RG4_mean": _finite_mean(control[column]),
                "CG4_median": _finite_median(treated[column]),
                "RG4_median": _finite_median(control[column]),
                "smd": standardized_mean_difference(treated[column], control[column]),
            },
        }
    return diagnostics


def _matched_effect(treated, control, outcome, n_boot):
    effect = bootstrap_difference(
        treated[outcome], control[outcome], statistic=np.mean, paired=True, n_boot=n_boot
    )
    if effect["estimate"] is None:
        return {"status": "skipped", "reason": "no_complete_matched_pairs"}
    return {
        "status": "ok",
        "delta_CG4_minus_RG4": effect["estimate"],
        "ci95": effect["ci95"],
        "p": effect["p"],
        "n_pairs": effect["n"],
    }


def _matched_robustness(satellites: pd.DataFrame, n_boot: int) -> dict[str, object]:
    complete = satellites.dropna(subset=["logMstar", "z_numeric"]).copy()
    attempts = [(0.15, 0.005), (0.20, 0.0075), (0.25, 0.010)]
    pairs = []
    tolerances = attempts[0]
    for tolerances in attempts:
        pairs = _match_pairs(complete, *tolerances)
        if len(pairs) >= 10:
            break
    if len(pairs) < 10:
        return {
            "status": "skipped",
            "reason": "too_few_matches",
            "n_pairs": int(len(pairs)),
            "attempted_tolerances": [
                {"delta_logMstar": mass, "delta_z": redshift}
                for mass, redshift in attempts
            ],
        }

    treated = complete.loc[[pair["treated_index"] for pair in pairs]].reset_index(drop=True)
    control = complete.loc[[pair["control_index"] for pair in pairs]].reset_index(drop=True)
    effects = {
        "quenched": _matched_effect(treated, control, "quenched", n_boot),
        "early_type": _matched_effect(treated, control, "early_type", n_boot),
    }
    distance_bin_effects = {}
    for bin_name in ["inner", "outer"]:
        if bin_name == "outer":
            mask = treated["distance_bin"].isin(["middle", "outer"])
        else:
            mask = treated["distance_bin"].eq("inner")
        if int(mask.sum()) < 10:
            distance_bin_effects[bin_name] = {
                "status": "skipped",
                "reason": "too_few_pairs",
                "n_pairs": int(mask.sum()),
            }
            continue
        distance_bin_effects[bin_name] = {
            "quenched": _matched_effect(treated.loc[mask], control.loc[mask], "quenched", n_boot),
            "early_type": _matched_effect(
                treated.loc[mask], control.loc[mask], "early_type", n_boot
            ),
        }

    return {
        "status": "ok",
        "method": "greedy nearest-neighbour matching without replacement",
        "matching_variables": ["logMstar", "z_numeric"],
        "tolerances": {
            "delta_logMstar": tolerances[0],
            "delta_z": tolerances[1],
        },
        "preferred_tolerances": {"delta_logMstar": 0.15, "delta_z": 0.005},
        "n_cg4_matched": int(len(treated)),
        "n_rg4_matched": int(len(control)),
        "n_rg4_unique": int(control.index.nunique()),
        "median_match_distance": float(np.median([pair["distance"] for pair in pairs])),
        "balance": _balance(complete, treated, control),
        "effects": effects,
        "distance_bin_effects": distance_bin_effects,
    }


def _error(summary):
    if summary.get("fraction") is None:
        return (0.0, 0.0)
    low, high = summary.get("ci68", [None, None])
    if low is None or high is None:
        return (0.0, 0.0)
    value = summary["fraction"]
    return (max(0.0, value - low), max(0.0, high - value))


def _plot_distance_fraction(
    binned: dict[str, dict[str, object]], outcome: str, ylabel: str, path: str
) -> str | None:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = np.arange(len(DISTANCE_BIN_ORDER))
    colours = {"CG4": "#A74752", "RG4": "#2864A6"}
    offsets = {"CG4": -0.08, "RG4": 0.08}
    for sample in SAMPLES:
        y = []
        yerr_low = []
        yerr_high = []
        for bin_name in DISTANCE_BIN_ORDER:
            summary = binned[bin_name][outcome][sample]
            y.append(np.nan if summary["fraction"] is None else summary["fraction"])
            low, high = _error(summary)
            yerr_low.append(low)
            yerr_high.append(high)
        ax.errorbar(
            x + offsets[sample],
            y,
            yerr=[yerr_low, yerr_high],
            marker="o",
            linestyle="-",
            capsize=3,
            label=sample,
            color=colours[sample],
        )
    ax.set_xticks(x, [DISTANCE_BIN_LABELS[name] for name in DISTANCE_BIN_ORDER])
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Satellite projected-distance rank from BGG")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def _plot_phase_bins(
    binned: dict[str, dict[str, object]], outcome: str, ylabel: str, path: str
) -> str | None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(PHASE_BIN_ORDER))
    width = 0.34
    colours = {"CG4": "#A74752", "RG4": "#2864A6"}
    for offset, sample in [(-width / 2, "CG4"), (width / 2, "RG4")]:
        values = []
        lows = []
        highs = []
        for bin_name in PHASE_BIN_ORDER:
            summary = binned[bin_name][outcome][sample]
            values.append(0 if summary["fraction"] is None else summary["fraction"])
            low, high = _error(summary)
            lows.append(low)
            highs.append(high)
        ax.bar(
            x + offset,
            values,
            width,
            yerr=[lows, highs],
            capsize=3,
            label=sample,
            color=colours[sample],
            alpha=0.88,
        )
    ax.set_xticks(x, [PHASE_BIN_LABELS[name] for name in PHASE_BIN_ORDER], rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Projected phase-space bin")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def _plot_matching_balance(matched: dict[str, object], path: str) -> str | None:
    if matched.get("status") != "ok":
        return None
    balance = matched.get("balance", {})
    variables = list(balance)
    if not variables:
        return None
    before = [abs(balance[column]["before"]["smd"]) for column in variables]
    after = [abs(balance[column]["after"]["smd"]) for column in variables]
    y = np.arange(len(variables))
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    ax.scatter(before, y, label="Before", color="#A74752")
    ax.scatter(after, y, label="After", color="#25876E")
    ax.axvline(0.1, color="0.45", linestyle=":", linewidth=1)
    ax.set_yticks(y, [column.replace("_", " ") for column in variables])
    ax.set_xlabel("Absolute standardized mean difference")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def _text_summary(
    global_stats: dict[str, object],
    distance_bins: dict[str, dict[str, object]],
    models: dict[str, object],
    matched: dict[str, object],
    velocity_available: bool,
) -> dict[str, object]:
    inner_quenched = distance_bins["inner"]["quenched"]
    outer_quenched = distance_bins["outer"]["quenched"]
    inner_early = distance_bins["inner"]["early_type"]
    quenched_model = models.get("quenched_distance", {})
    early_model = models.get("early_type_distance", {})
    matched_quenched = matched.get("effects", {}).get("quenched", {})
    matched_early = matched.get("effects", {}).get("early_type", {})
    trend = "undetermined"
    inner_delta = inner_quenched["delta_CG4_minus_RG4"]["delta"]
    outer_delta = outer_quenched["delta_CG4_minus_RG4"]["delta"]
    if inner_delta is not None and outer_delta is not None:
        if abs(inner_delta - outer_delta) < 0.05:
            trend = "similar_at_inner_and_outer_ranks"
        elif abs(inner_delta) > abs(outer_delta):
            trend = "stronger_inner"
        else:
            trend = "stronger_outer"

    quenched_terms = quenched_model.get("terms", {})
    early_terms = early_model.get("terms", {})
    quenched_cg = quenched_terms.get("is_CG4", {})
    early_cg = early_terms.get("is_CG4", {})
    return {
        "cg_satellite_n": global_stats["CG4"]["n_satellites"],
        "rg_satellite_n": global_stats["RG4"]["n_satellites"],
        "cg_quenched_fraction_inner": inner_quenched["CG4"]["fraction"],
        "rg_quenched_fraction_inner": inner_quenched["RG4"]["fraction"],
        "delta_quenched_fraction_inner": inner_delta,
        "delta_quenched_fraction_inner_err": inner_quenched["delta_CG4_minus_RG4"][
            "stderr"
        ],
        "cg_quenched_fraction_outer": outer_quenched["CG4"]["fraction"],
        "rg_quenched_fraction_outer": outer_quenched["RG4"]["fraction"],
        "delta_quenched_fraction_outer": outer_delta,
        "cg_earlytype_fraction_inner": inner_early["CG4"]["fraction"],
        "rg_earlytype_fraction_inner": inner_early["RG4"]["fraction"],
        "delta_earlytype_fraction_inner": inner_early["delta_CG4_minus_RG4"]["delta"],
        "delta_earlytype_fraction_inner_err": inner_early["delta_CG4_minus_RG4"][
            "stderr"
        ],
        "quenched_model_sample_cg_or": quenched_cg.get("odds_ratio"),
        "quenched_model_sample_cg_ci_low": (
            quenched_cg.get("ci95", [None, None])[0] if quenched_cg else None
        ),
        "quenched_model_sample_cg_ci_high": (
            quenched_cg.get("ci95", [None, None])[1] if quenched_cg else None
        ),
        "quenched_model_sample_cg_p": quenched_cg.get("p"),
        "earlytype_model_sample_cg_or": early_cg.get("odds_ratio"),
        "earlytype_model_sample_cg_p": early_cg.get("p"),
        "matched_delta_quenched": matched_quenched.get("delta_CG4_minus_RG4"),
        "matched_delta_quenched_ci95": matched_quenched.get("ci95"),
        "matched_delta_earlytype": matched_early.get("delta_CG4_minus_RG4"),
        "matched_delta_earlytype_ci95": matched_early.get("ci95"),
        "quenched_radial_trend": trend,
        "velocity_analysis_available": bool(velocity_available),
    }


def _legacy_bin_results(phase_bins: dict[str, dict[str, object]]) -> dict[str, object]:
    legacy = {}
    for bin_name in PHASE_BIN_ORDER:
        quenched = phase_bins.get(bin_name, {}).get("quenched", {})
        legacy[bin_name] = {
            "cg4": {
                "n": quenched.get("CG4", {}).get("n", 0),
                "quenched_fraction": quenched.get("CG4", {}).get("fraction"),
            },
            "control": {
                "n": quenched.get("RG4", {}).get("n", 0),
                "quenched_fraction": quenched.get("RG4", {}).get("fraction"),
            },
            "low_N": quenched.get("low_N", True),
        }
    return legacy


def run_phase_space_segregation_analysis(
    data,
    output_dir: str | None = None,
    *,
    n_boot: int = 2000,
    min_satellites: int = 30,
    min_total_per_bin: int = 20,
    min_per_sample_per_bin: int = 10,
) -> dict[str, object]:
    """Test whether CG4-RG4 satellite differences depend on BGG-centric location."""

    frame, mapping, warnings = _prepare_frame(data)
    if frame.empty:
        return {"status": "skipped", "reason": "no_input_rows"}
    missing = [
        column
        for column in ["sample", "is_satellite", "logMstar", "quenched", "dist2BGG_projected_kpc"]
        if column not in frame
    ]
    if missing:
        return {
            "status": "skipped",
            "reason": "missing_required_columns",
            "missing_columns": missing,
        }

    satellites = prepare_phase_space_satellite_sample(frame)
    baseline = satellites.dropna(
        subset=["logMstar", "quenched", "dist2BGG_projected_kpc", "z_numeric"]
    ).copy()
    if len(baseline) < min_satellites:
        return {
            "status": "skipped",
            "reason": "too_few_satellites_after_baseline_cuts",
            "n": int(len(baseline)),
            "metadata": {"column_mapping": mapping, "warnings": warnings},
        }

    velocity_available = (
        baseline["abs_dv_norm"].notna().mean() >= 0.5
        and baseline.dropna(subset=["abs_dv_norm"]).shape[0] >= min_satellites
    )
    if not velocity_available:
        warnings.append(
            "Normalized velocity offsets are incomplete; projected phase-space bins are exploratory or unavailable."
        )

    global_stats = _global_statistics(baseline, n_boot)
    distance_bins = summarize_binned_fractions(
        baseline,
        "distance_bin",
        DISTANCE_BIN_ORDER,
        n_boot=n_boot,
        min_total=min_total_per_bin,
        min_per_sample=min_per_sample_per_bin,
    )
    phase_bins = (
        summarize_binned_fractions(
            baseline.dropna(subset=["phase_space_bin", "abs_dv_norm"]),
            "phase_space_bin",
            PHASE_BIN_ORDER,
            n_boot=n_boot,
            min_total=min_total_per_bin,
            min_per_sample=min_per_sample_per_bin,
        )
        if velocity_available
        else {}
    )
    models = _fit_models(baseline, velocity_available)
    matched = _matched_robustness(baseline, n_boot)

    figures = {}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        figures["quenched_fraction_by_distance"] = _plot_distance_fraction(
            distance_bins,
            "quenched",
            "Quenched satellite fraction",
            os.path.join(
                output_dir,
                "phase_space_satellite_quenched_fraction_by_distance.pdf",
            ),
        )
        figures["earlytype_fraction_by_distance"] = _plot_distance_fraction(
            distance_bins,
            "early_type",
            "Early-type satellite fraction",
            os.path.join(
                output_dir,
                "phase_space_satellite_earlytype_fraction_by_distance.pdf",
            ),
        )
        if velocity_available:
            figures["quenched_fraction_projected_phase_space"] = _plot_phase_bins(
                phase_bins,
                "quenched",
                "Quenched satellite fraction",
                os.path.join(
                    output_dir,
                    "phase_space_satellite_quenched_fraction_projected_phase_space.pdf",
                ),
            )
        figures["mass_redshift_balance"] = _plot_matching_balance(
            matched,
            os.path.join(output_dir, "phase_space_mass_redshift_balance.pdf"),
        )

    fixed_signal = False
    for bin_name in PHASE_BIN_ORDER:
        block = phase_bins.get(bin_name, {}).get("quenched", {})
        delta = block.get("delta_CG4_minus_RG4", {})
        stderr = delta.get("stderr")
        estimate = delta.get("delta")
        if safe_float(stderr) not in (None, 0) and safe_float(estimate) is not None:
            fixed_signal = fixed_signal or abs(estimate / stderr) > 2

    result = {
        "status": "ok",
        "metadata": {
            "module_version": MODULE_VERSION,
            "input_tables_used": ["CG4_Gals", "CG4_Groups", "RG4_Gals", "RG4_Groups"],
            "column_mapping": mapping,
            "filters_applied": _cut_counts(frame),
            "warnings": warnings,
            "minimum_bin_counts": {
                "total": min_total_per_bin,
                "per_sample": min_per_sample_per_bin,
            },
            "bootstrap": {
                "method": "group-cluster bootstrap",
                "n_boot": n_boot,
                "seed": RNG_SEED,
            },
        },
        "sample_counts": {
            sample: {
                "n_satellites": global_stats[sample]["n_satellites"],
                "n_groups": global_stats[sample]["n_groups"],
            }
            for sample in SAMPLES
        },
        "descriptive_global_statistics": global_stats,
        "distance_bin_fractions": distance_bins,
        "phase_space_bin_fractions": phase_bins,
        "regression_results": models,
        "matched_robustness": matched,
        "figure_paths": figures,
        "text_summary": _text_summary(
            global_stats, distance_bins, models, matched, velocity_available
        ),
        "coordinates_used": {
            "radius": mapping.get("projected_distance"),
            "distance_bin": "within-group satellite rank by projected distance to BGG",
            "velocity": "c * (z_gal - z_BGG) / (1 + z_group)",
            "velocity_scale": "catalogue Vdisp, with gapper fallback",
            "velocity_threshold": float(baseline.attrs.get("velocity_threshold", 1.0)),
        },
        "bin_thresholds": {"velocity_abs_dv_norm": 1.0},
        "velocity_analysis_available": bool(velocity_available),
        "bin_results": _legacy_bin_results(phase_bins),
        "logistic_models": {
            "quenched": models.get("quenched_phase_space", models.get("quenched_distance")),
            "elliptical": models.get(
                "early_type_phase_space", models.get("early_type_distance")
            ),
        },
        "fixed_phase_space_cg4_significant": bool(fixed_signal),
        "figure": figures.get("quenched_fraction_projected_phase_space"),
    }
    return safe_json(result)


def run_phase_space_analysis(data, output_dir: str | None = None):
    """Compatibility wrapper for older pipeline imports."""

    return run_phase_space_segregation_analysis(data, output_dir=output_dir)
