"""Galaxy-size analysis at fixed stellar mass for compact groups.

Primary estimand: the CG4 coefficient Delta (dex) in a cluster-robust linear
model of log10(size/kpc) at fixed stellar mass, redshift, rank, and available
group-scale covariates, pooled over the three regular-group controls.
Delta < 0 means CG4 galaxies are smaller at fixed mass.

Pre-registered Holm families:
  F1 (primary, Simard size): all galaxies, satellites only, BGGs only.
  F2 (secondary): Petrosian all, Petrosian satellites, concentration
     satellites.
  F3 (matched outcomes): Simard Dlog size, Petrosian Dlog size,
     Dconcentration.
Everything else is descriptive/exploratory (raw p, BH inside figures).
"""

from __future__ import annotations

import os

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

try:
    import config as co
    from extended_data import dedup_control_pool, ensure_galaxy_frame
    from extended_stats import (
        benjamini_hochberg,
        empirical_p_two_sided,
        fit_ols_with_optional_cluster_se,
        holm_correction,
        safe_float,
        safe_json,
        two_sample_summary,
    )
    from matched_controls import matched_pairs
    from morphology_robustness import CROWDING_THRESHOLD_ARCSEC, _nearest_angular
    from size_data import Z_MATCH_TOLERANCE, attach_size_columns
    from tidal_indices import _derive as _derive_tidal_indices
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from .extended_data import dedup_control_pool, ensure_galaxy_frame
    from .extended_stats import (
        benjamini_hochberg,
        empirical_p_two_sided,
        fit_ols_with_optional_cluster_se,
        holm_correction,
        safe_float,
        safe_json,
        two_sample_summary,
    )
    from .matched_controls import matched_pairs
    from .morphology_robustness import CROWDING_THRESHOLD_ARCSEC, _nearest_angular
    from .size_data import Z_MATCH_TOLERANCE, attach_size_columns
    from .tidal_indices import _derive as _derive_tidal_indices


SEED = 20260612
N_BOOT = 9999
REFERENCE_LOGMSTAR = 10.3
SAMPLES = ["CG4", "Control4B", "Control4C", "RG4"]
CONTROL_SAMPLES = ["Control4B", "Control4C", "RG4"]
PALETTE = {
    "CG4": "#2864A6",
    "Control4B": "#D17A22",
    "Control4C": "#25876E",
    "RG4": "#A74752",
}
PRIMARY_OUTCOME = "log_Rchl_r_kpc"
PETRO_OUTCOME = "log_petroR50_kpc"
CONCENTRATION_OUTCOME = "concentration_r90_r50"
# Group-scale covariates mirror specialness_models: kept only when at least
# 65 per cent complete on the analysis panel.
GROUP_COVARIATES = ["log_group_luminosity", "velocity_dispersion"]
COVARIATE_COMPLETENESS = 0.65


def _skipped(reason: str, **extra) -> dict[str, object]:
    return {"status": "skipped", "reason": reason, **extra}


def _standardize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    scale = float(values.std(ddof=0))
    if not np.isfinite(scale) or scale == 0:
        return values * 0.0
    return (values - float(values.mean())) / scale


def _term_summary(fitted, term: str) -> dict[str, object] | None:
    names = list(fitted.model.exog_names)
    if term not in names:
        return None
    index = names.index(term)
    ci = np.asarray(fitted.conf_int())
    coefficient = float(np.asarray(fitted.params)[index])
    return {
        "coefficient": coefficient,
        "standard_error": float(np.asarray(fitted.bse)[index]),
        "ci_low": float(ci[index, 0]),
        "ci_high": float(ci[index, 1]),
        "p": float(np.asarray(fitted.pvalues)[index]),
    }


def fit_size_model(
    frame: pd.DataFrame,
    outcome: str,
    *,
    use_mass: bool = True,
    use_luminosity: bool = False,
    include_satellite_flag: bool = True,
    include_group_covariates: bool = True,
    extra_covariates: tuple[str, ...] = (),
    interaction_with: str | None = None,
    log_outcome: bool = True,
    min_n: int = 30,
) -> dict[str, object]:
    """Fit the shared cluster-robust size regression and summarize is_CG4."""

    base_continuous = []
    if use_mass:
        base_continuous.append("logMstar")
    if use_luminosity:
        base_continuous.append("M_r")
    base_continuous.append("z_numeric")
    binary = ["is_satellite"] if include_satellite_flag else []
    group_covariates = list(GROUP_COVARIATES) if include_group_covariates else []

    cluster_col = "physical_group" if "physical_group" in frame.columns else "group_uid"
    required = [outcome, "is_CG4", "group_uid", cluster_col, *base_continuous, *binary]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return _skipped("missing_required_columns", missing_columns=missing)
    if interaction_with is not None and interaction_with not in frame.columns:
        return _skipped("missing_required_columns", missing_columns=[interaction_with])

    columns = list(
        dict.fromkeys(
            required
            + [c for c in group_covariates if c in frame.columns]
            + [c for c in extra_covariates if c in frame.columns]
            + ([interaction_with] if interaction_with else [])
        )
    )
    work = frame[columns].replace([np.inf, -np.inf], np.nan).copy()
    work = work.dropna(subset=[outcome, "is_CG4", cluster_col, *base_continuous, *binary])
    if interaction_with is not None:
        work = work.dropna(subset=[interaction_with])

    # Covariate-completeness handling mirrors specialness_models: sparse
    # group-scale covariates are dropped and the retained set is reported.
    kept_group = [
        column
        for column in group_covariates
        if column in work.columns
        and work[column].notna().mean() >= COVARIATE_COMPLETENESS
    ]
    kept_extra = [
        column
        for column in extra_covariates
        if column in work.columns
        and work[column].notna().mean() >= COVARIATE_COMPLETENESS
    ]
    work = work.dropna(subset=kept_group + kept_extra)
    if len(work) < min_n:
        return _skipped("too_few_complete_cases", n=int(len(work)))
    if work["is_CG4"].nunique() < 2:
        return _skipped("no_cg4_contrast", n=int(len(work)))

    terms = ["is_CG4"]
    covariates_used = []
    # The quadratic mass/luminosity term is the square of the standardized
    # linear term (then standardized again) to avoid the near-perfect
    # collinearity of raw logM and logM^2.
    for column, quadratic in [("logMstar", use_mass), ("M_r", use_luminosity)]:
        if column not in work.columns or column not in base_continuous:
            continue
        work[f"c_{column}"] = _standardize(work[column])
        terms.append(f"c_{column}")
        covariates_used.append(column)
        if quadratic:
            work[f"c_{column}_sq"] = _standardize(work[f"c_{column}"] ** 2)
            terms.append(f"c_{column}_sq")
            covariates_used.append(f"{column}^2")
    work["c_z"] = _standardize(work["z_numeric"])
    terms.append("c_z")
    covariates_used.append("z")
    for column in binary:
        if work[column].nunique(dropna=True) >= 2:
            terms.append(column)
            covariates_used.append(column)
    for column in kept_group + kept_extra:
        if work[column].nunique(dropna=True) < 2:
            continue
        work[f"c_{column}"] = _standardize(work[column])
        terms.append(f"c_{column}")
        covariates_used.append(column)
    if interaction_with is not None:
        terms.append(interaction_with)
        terms.append(f"is_CG4:{interaction_with}")
        covariates_used.append(interaction_with)

    formula = f"{outcome} ~ " + " + ".join(terms)
    # Cluster by physical Lim group so the same group under multiple control
    # labels counts as one cluster, consistent with the Sect. 5.2 convention.
    fitted = fit_ols_with_optional_cluster_se(formula, work, group_col=cluster_col)
    if fitted is None:
        return _skipped("model_fit_failed", n=int(len(work)))

    cg4 = _term_summary(fitted, "is_CG4")
    if cg4 is None:
        return _skipped("no_cg4_term", n=int(len(work)))
    result = {
        "status": "ok",
        "outcome": outcome,
        "formula": formula,
        "n": int(fitted.nobs),
        "cluster_unit": cluster_col,
        "n_groups": int(work[cluster_col].nunique()),
        "n_clusters": int(work[cluster_col].nunique()),
        "n_label_groups": int(work["group_uid"].nunique()),
        "n_cg4": int((work["is_CG4"] == 1).sum()),
        "covariance": fitted.cov_type,
        "covariates_used": covariates_used,
        "cg4_delta_dex" if log_outcome else "cg4_delta": cg4["coefficient"],
        "standard_error": cg4["standard_error"],
        "ci_low": cg4["ci_low"],
        "ci_high": cg4["ci_high"],
        "p": cg4["p"],
    }
    if log_outcome:
        result["pct_equivalent"] = float(100 * (10 ** cg4["coefficient"] - 1))
    if interaction_with is not None:
        interaction = _term_summary(fitted, f"is_CG4:{interaction_with}")
        result["interaction_term"] = interaction
    return result


def _holm_annotate(results: dict[str, dict], names: list[str]) -> None:
    """Attach Holm-corrected p-values across the given family in place."""

    ok = [n for n in names if results.get(n, {}).get("status") == "ok"]
    adjusted = holm_correction([results[n].get("p") for n in ok])
    for name, p_holm in zip(ok, adjusted):
        results[name]["p_holm"] = p_holm


# ---------------------------------------------------------------------------
# Block A: availability audit
# ---------------------------------------------------------------------------


def _availability_audit(frame: pd.DataFrame, output_dir: str | None) -> dict:
    audit = frame.attrs.get("size_attach_audit", {}).get("per_sample", {})
    if not audit:
        return _skipped("no_attach_audit")

    fractions = {}
    for sample_name, row in audit.items():
        n = max(int(row.get("n_rows", 0)), 1)
        fractions[sample_name] = {
            "simard_fraction": row.get("size_ok_simard", 0) / n,
            "petro_fraction": row.get("size_ok_petro", 0) / n,
        }

    tests = {}
    for measure in ["simard", "petro"]:
        cg = audit.get("CG4", {})
        cg_ok = int(cg.get(f"size_ok_{measure}", 0))
        cg_n = int(cg.get("n_rows", 0))
        for control in CONTROL_SAMPLES:
            row = audit.get(control, {})
            ok = int(row.get(f"size_ok_{measure}", 0))
            n = int(row.get("n_rows", 0))
            if cg_n == 0 or n == 0:
                continue
            table = [[cg_ok, cg_n - cg_ok], [ok, n - ok]]
            tests[f"{measure}_CG4_vs_{control}"] = {
                "cg4_fraction": cg_ok / cg_n,
                "control_fraction": ok / n,
                "fisher_p": float(stats.fisher_exact(table).pvalue),
            }

    available = frame["size_ok_simard"] == 1
    shifts = {}
    for label, column in [
        ("logMstar", "logMstar"),
        ("z", "z_numeric"),
        ("rank", "rank"),
        ("nearest_neighbour_arcsec", "nearest_angular_separation_arcsec"),
    ]:
        if column in frame.columns:
            shifts[label] = two_sample_summary(
                frame.loc[available, column], frame.loc[~available, column]
            )

    pooled_ok = sum(
        int(audit.get(c, {}).get("size_ok_simard", 0)) for c in CONTROL_SAMPLES
    )
    pooled_n = sum(int(audit.get(c, {}).get("n_rows", 0)) for c in CONTROL_SAMPLES)
    cg4_fraction = fractions.get("CG4", {}).get("simard_fraction", 0.0)
    pooled_fraction = pooled_ok / pooled_n if pooled_n else 0.0
    completeness_caveat = bool(abs(cg4_fraction - pooled_fraction) > 0.05)

    totals = {
        key: int(sum(int(row.get(key, 0)) for row in audit.values()))
        for key in [
            "n_rows",
            "petro_row_resolved",
            "dr7_bridge_resolved",
            "simard_matched",
            "z_mismatch",
            "shred_merge",
            "n_pegged",
            "simard_out_of_window",
            "petro_out_of_window",
            "size_ok_simard",
            "size_ok_petro",
        ]
    }
    result = {
        "status": "ok",
        "per_sample": audit,
        "totals": totals,
        "fractions": fractions,
        "cg4_simard_fraction": cg4_fraction,
        "pooled_control_simard_fraction": pooled_fraction,
        "two_proportion_tests": tests,
        "available_vs_unavailable_shifts": shifts,
        "completeness_caveat": completeness_caveat,
    }
    return result


# ---------------------------------------------------------------------------
# Block B: descriptive mass-size relations
# ---------------------------------------------------------------------------


def _mass_size(frame: pd.DataFrame, output_dir: str | None) -> dict:
    result = {"status": "ok", "reference_logMstar": REFERENCE_LOGMSTAR, "by_sample": {}}
    panels = {}
    for sample_name in SAMPLES:
        part = frame.loc[
            (frame["sample"] == sample_name)
            & frame[PRIMARY_OUTCOME].notna()
            & frame["logMstar"].notna()
        ]
        if len(part) < 10:
            result["by_sample"][sample_name] = _skipped(
                "too_few_galaxies", n=int(len(part))
            )
            continue
        coefficients = np.polyfit(part["logMstar"], part[PRIMARY_OUTCOME], 2)
        predicted_log = float(np.polyval(coefficients, REFERENCE_LOGMSTAR))
        result["by_sample"][sample_name] = {
            "status": "ok",
            "n": int(len(part)),
            "quadratic_coefficients": [float(c) for c in coefficients],
            "predicted_log_kpc_at_reference": predicted_log,
            "predicted_kpc_at_reference": float(10**predicted_log),
        }
        panels[sample_name] = (part, coefficients)
    if output_dir and panels:
        result["figure"] = _plot_mass_size(
            panels, os.path.join(output_dir, "size_mass_relation.pdf")
        )
    return result


def _plot_mass_size(panels: dict, path: str) -> str | None:
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.2), sharex=True, sharey=True)
    grid = np.linspace(9.0, 11.8, 100)
    for ax, sample_name in zip(axes.flat, SAMPLES):
        if sample_name not in panels:
            ax.set_visible(False)
            continue
        part, coefficients = panels[sample_name]
        ax.scatter(
            part["logMstar"],
            part[PRIMARY_OUTCOME],
            s=8,
            alpha=0.35,
            color=PALETTE[sample_name],
        )
        ax.plot(grid, np.polyval(coefficients, grid), color="0.2", linewidth=1.4)
        ax.set_title(f"{sample_name} (N={len(part)})")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\log(M_\star/M_\odot)$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10}(R_{\rm chl,r}/{\rm kpc})$")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


# ---------------------------------------------------------------------------
# Block C: primary adjusted models (family F1)
# ---------------------------------------------------------------------------


def _adjusted(frame: pd.DataFrame, outcome: str = PRIMARY_OUTCOME, **kwargs) -> dict:
    result = {"status": "ok", "family": "F1", "outcome": outcome}
    result["all"] = fit_size_model(frame, outcome, **kwargs)
    result["satellites"] = fit_size_model(
        frame.loc[frame["is_satellite"] == 1],
        outcome,
        include_satellite_flag=False,
        **kwargs,
    )
    result["bgg"] = fit_size_model(
        frame.loc[frame["is_bgg"] == 1],
        outcome,
        include_satellite_flag=False,
        **kwargs,
    )
    _holm_annotate(result, ["all", "satellites", "bgg"])
    if all(result[name].get("status") != "ok" for name in ["all", "satellites", "bgg"]):
        result["status"] = "skipped"
        result["reason"] = "no_fittable_variant"
    return result


# ---------------------------------------------------------------------------
# Block D: per-control fits (descriptive)
# ---------------------------------------------------------------------------


def _per_control(frame: pd.DataFrame, output_dir: str | None) -> dict:
    comparisons = {}
    comparisons["pooled"] = fit_size_model(
        frame, PRIMARY_OUTCOME, include_group_covariates=False
    )
    for control in CONTROL_SAMPLES:
        panel = frame.loc[frame["sample"].isin(["CG4", control])]
        comparisons[control] = fit_size_model(
            panel, PRIMARY_OUTCOME, include_group_covariates=False
        )
    ok = [k for k, v in comparisons.items() if v.get("status") == "ok"]
    adjusted = benjamini_hochberg([comparisons[k].get("p") for k in ok])
    for name, p_bh in zip(ok, adjusted):
        comparisons[name]["p_bh"] = p_bh
    result = {
        "status": "ok" if ok else "skipped",
        "note": "descriptive; mass+z adjusted; BH within this figure family",
        "comparisons": comparisons,
    }
    if output_dir and ok:
        result["figure"] = _plot_forest(
            comparisons, os.path.join(output_dir, "size_forest_per_control.pdf")
        )
    return result


def _plot_forest(comparisons: dict, path: str) -> str | None:
    labels = {
        "pooled": "Ordinary pooled",
        "Control4B": "Control4B",
        "Control4C": "Control4C",
        "RG4": "RG4",
    }
    rows = [
        (labels.get(k, k), v) for k, v in comparisons.items() if v.get("status") == "ok"
    ]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 0.6 * len(rows) + 1.6))
    y = np.arange(len(rows))
    values = np.array([row[1]["cg4_delta_dex"] for row in rows])
    lows = np.array([row[1]["ci_low"] for row in rows])
    highs = np.array([row[1]["ci_high"] for row in rows])
    ax.errorbar(
        values,
        y,
        xerr=[values - lows, highs - values],
        fmt="o",
        capsize=3,
        color="#2864A6",
    )
    ax.axvline(0, color="0.45", linestyle=":", linewidth=1)
    ax.set_yticks(y, [row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel(r"CG$_4$ size offset $\Delta$ (dex, 95% CI)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


# ---------------------------------------------------------------------------
# Block E: morphology strata (secondary)
# ---------------------------------------------------------------------------


def _morphology_strata(frame: pd.DataFrame) -> dict:
    satellites = frame.loc[frame["is_satellite"] == 1]
    result = {"status": "ok", "note": "secondary; Holm within this trio"}
    result["elliptical_satellites"] = fit_size_model(
        satellites.loc[satellites["elliptical"] == 1],
        PRIMARY_OUTCOME,
        include_satellite_flag=False,
    )
    result["spiral_satellites"] = fit_size_model(
        satellites.loc[satellites["spiral"] == 1],
        PRIMARY_OUTCOME,
        include_satellite_flag=False,
    )
    result["interaction"] = fit_size_model(
        satellites.loc[satellites["elliptical"].notna()],
        PRIMARY_OUTCOME,
        include_satellite_flag=False,
        interaction_with="elliptical",
    )
    family = ["elliptical_satellites", "spiral_satellites", "interaction"]
    # For the interaction model the family member is the interaction term.
    interaction = result["interaction"]
    if interaction.get("status") == "ok" and interaction.get("interaction_term"):
        interaction["p"] = interaction["interaction_term"]["p"]
    _holm_annotate(result, family)
    return result


# ---------------------------------------------------------------------------
# Block G: matched pairs (family F3)
# ---------------------------------------------------------------------------


def _paired_bootstrap(differences: np.ndarray, statistic, blocks=None) -> dict:
    rng = np.random.default_rng(SEED)
    n = differences.size
    boot = np.empty(N_BOOT)
    if blocks is not None:
        unique_blocks = pd.unique(np.asarray(blocks))
        members = {
            block: np.flatnonzero(np.asarray(blocks) == block)
            for block in unique_blocks
        }
        n_blocks = len(unique_blocks)
        for index in range(N_BOOT):
            drawn = rng.integers(0, n_blocks, n_blocks)
            rows = np.concatenate([members[unique_blocks[j]] for j in drawn])
            boot[index] = statistic(differences[rows])
    else:
        for index in range(N_BOOT):
            boot[index] = statistic(differences[rng.integers(0, n, n)])
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "estimate": float(statistic(differences)),
        "ci95": [float(low), float(high)],
        "p": empirical_p_two_sided(boot),
        "n_boot": int(N_BOOT),
        "p_floor": float(2 / (N_BOOT + 1)),
    }


def _matched(frame: pd.DataFrame) -> dict:
    pairs, _, caliper, prepared, variables = matched_pairs(frame)
    if not variables:
        return _skipped("no_matching_variables")
    pairs_repeat, _, _, _, _ = matched_pairs(frame)
    deterministic = pairs == pairs_repeat
    if len(pairs) < 10:
        return _skipped("too_few_matches", n_pairs=len(pairs))

    expected = frame.attrs.get("matched_controls_n_cg4_matched")
    pair_count_consistent = None
    if expected is not None:
        pair_count_consistent = bool(len(pairs) == int(expected))

    treated = prepared.loc[[pair["treated_index"] for pair in pairs]].reset_index(
        drop=True
    )
    control = prepared.loc[[pair["control_index"] for pair in pairs]].reset_index(
        drop=True
    )
    if "physical_group" in treated:
        treated_blocks = treated["physical_group"].astype(str).to_numpy()
    else:
        treated_blocks = treated["group_uid"].astype(str).to_numpy()

    outcomes = {
        "delta_log_Rchl_r": (PRIMARY_OUTCOME, True),
        "delta_log_petroR50": (PETRO_OUTCOME, True),
        "delta_concentration": (CONCENTRATION_OUTCOME, False),
    }
    effects = {}
    for name, (column, _is_log) in outcomes.items():
        if column not in treated or column not in control:
            effects[name] = _skipped("missing_outcome_column")
            continue
        tx = pd.to_numeric(treated[column], errors="coerce")
        cx = pd.to_numeric(control[column], errors="coerce")
        mask = tx.notna() & cx.notna()
        differences = (tx[mask] - cx[mask]).to_numpy(dtype=float)
        if differences.size < 10:
            effects[name] = _skipped(
                "too_few_pairs_with_sizes", n_pairs_with_sizes=int(differences.size)
            )
            continue
        pair_blocks = treated_blocks[mask.to_numpy()]
        mean_boot = _paired_bootstrap(differences, np.mean, blocks=pair_blocks)
        median_boot = _paired_bootstrap(differences, np.median, blocks=pair_blocks)
        effects[name] = {
            "status": "ok",
            "n_pairs": int(len(pairs)),
            "n_pairs_with_sizes": int(differences.size),
            "mean_delta": mean_boot["estimate"],
            "mean_ci95": mean_boot["ci95"],
            "median_delta": median_boot["estimate"],
            "median_ci95": median_boot["ci95"],
            "p": mean_boot["p"],
        }
    _holm_annotate(effects, list(outcomes))
    for name in outcomes:
        if effects[name].get("status") == "ok":
            effects[name]["p_adj"] = effects[name].get("p_holm")

    return {
        "status": "ok",
        "family": "F3",
        "note": (
            "Pairs recomputed with the paper's matching implementation and "
            "seed on the objid-deduplicated control pool; the published "
            "matched-control Holm family is untouched. Holm here spans the "
            "three size outcomes; p is the two-sided add-one empirical "
            "p-value of the group-blocked paired mean bootstrap."
        ),
        "matching_variables": variables,
        "propensity_caliper": safe_float(caliper),
        "deterministic": bool(deterministic),
        "n_pairs": int(len(pairs)),
        "expected_n_pairs_from_matched_controls": expected,
        "pair_count_consistent": pair_count_consistent,
        "effects": effects,
    }


# ---------------------------------------------------------------------------
# Block H: crowding robustness
# ---------------------------------------------------------------------------


def _crowding(frame: pd.DataFrame) -> dict:
    if "close_neighbour" not in frame.columns:
        return _skipped("missing_close_neighbour_flag")
    open_frame = frame.loc[frame["close_neighbour"] != 1]
    result = {
        "status": "ok",
        "threshold_arcsec": float(CROWDING_THRESHOLD_ARCSEC),
        "excluding_close": {
            "all": fit_size_model(open_frame, PRIMARY_OUTCOME),
            "satellites": fit_size_model(
                open_frame.loc[open_frame["is_satellite"] == 1],
                PRIMARY_OUTCOME,
                include_satellite_flag=False,
            ),
        },
        "with_close_flag": {
            "all": fit_size_model(
                frame, PRIMARY_OUTCOME, extra_covariates=("close_neighbour",)
            ),
            "satellites": fit_size_model(
                frame.loc[frame["is_satellite"] == 1],
                PRIMARY_OUTCOME,
                include_satellite_flag=False,
                extra_covariates=("close_neighbour",),
            ),
        },
    }
    return result


# ---------------------------------------------------------------------------
# Block I: Petrosian rerun and measure-delta diagnostic
# ---------------------------------------------------------------------------


def _petrosian(frame: pd.DataFrame) -> dict:
    result = {
        "status": "ok",
        "family": "F2",
        "note": "psfWidth_r enters as a standardized seeing covariate",
    }
    result["all"] = fit_size_model(
        frame, PETRO_OUTCOME, extra_covariates=("psfWidth_r",)
    )
    result["satellites"] = fit_size_model(
        frame.loc[frame["is_satellite"] == 1],
        PETRO_OUTCOME,
        include_satellite_flag=False,
        extra_covariates=("psfWidth_r",),
    )
    if all(result[name].get("status") != "ok" for name in ["all", "satellites"]):
        result["status"] = "skipped"
        result["reason"] = "no_fittable_variant"
    return result


def _measure_delta(frame: pd.DataFrame, output_dir: str | None) -> dict:
    valid = frame[PRIMARY_OUTCOME].notna() & frame[PETRO_OUTCOME].notna()
    work = frame.loc[valid].copy()
    if len(work) < 30:
        return _skipped("too_few_galaxies_with_both_sizes", n=int(len(work)))
    work["measure_delta"] = work[PRIMARY_OUTCOME] - work[PETRO_OUTCOME]

    spearman = None
    if "nearest_angular_separation_arcsec" in work.columns:
        panel = work[["measure_delta", "nearest_angular_separation_arcsec"]].dropna()
        if len(panel) >= 10:
            test = stats.spearmanr(
                panel["nearest_angular_separation_arcsec"], panel["measure_delta"]
            )
            spearman = {
                "n": int(len(panel)),
                "rho": float(test.statistic),
                "p": float(test.pvalue),
            }

    by_morphology = {}
    for label, mask in [
        ("elliptical", work.get("elliptical", pd.Series(dtype=float)) == 1),
        ("spiral", work.get("spiral", pd.Series(dtype=float)) == 1),
    ]:
        values = work.loc[mask, "measure_delta"]
        if len(values) >= 5:
            by_morphology[label] = {
                "n": int(len(values)),
                "mean": float(values.mean()),
            }
    by_sample = {
        str(name): {"n": int(len(part)), "mean": float(part["measure_delta"].mean())}
        for name, part in work.groupby("sample", observed=True)
    }
    result = {
        "status": "ok",
        "n": int(len(work)),
        "mean_delta_dex": float(work["measure_delta"].mean()),
        "spearman_vs_neighbour_separation": spearman,
        "mean_by_morphology": by_morphology,
        "mean_by_sample": by_sample,
    }
    if output_dir:
        result["figure"] = _plot_measure_delta(
            work, os.path.join(output_dir, "size_measure_delta.pdf")
        )
    return result


def _plot_measure_delta(work: pd.DataFrame, path: str) -> str | None:
    if "nearest_angular_separation_arcsec" not in work.columns:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for is_cg4, label, colour in [
        (0, "Controls", "#777777"),
        (1, "CG$_4$", "#2864A6"),
    ]:
        part = work.loc[work["is_CG4"] == is_cg4]
        ax.scatter(
            part["nearest_angular_separation_arcsec"],
            part["measure_delta"],
            s=9,
            alpha=0.4,
            color=colour,
            label=label,
        )
    ax.axhline(0, color="0.45", linestyle=":", linewidth=1)
    ax.axvline(CROWDING_THRESHOLD_ARCSEC, color="0.45", linestyle="--", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("Nearest projected neighbour separation (arcsec)")
    ax.set_ylabel(r"$\log R_{\rm chl,r} - \log R_{50,r}$ (dex)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


# ---------------------------------------------------------------------------
# Block J: tidal-index absorption
# ---------------------------------------------------------------------------


def _tidal(frame: pd.DataFrame) -> dict:
    required = ["RA", "Dec", "z_numeric", "logMstar", "group_uid"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return _skipped("missing_required_columns", missing_columns=missing)
    work = _derive_tidal_indices(frame)
    complete = work["log_tidal_index"].notna() & work[PRIMARY_OUTCOME].notna()
    panel = work.loc[complete]
    if len(panel) < 30:
        return _skipped("too_few_complete_cases", n=int(len(panel)))
    baseline = fit_size_model(panel, PRIMARY_OUTCOME)
    adjusted = fit_size_model(
        panel, PRIMARY_OUTCOME, extra_covariates=("log_tidal_index",)
    )
    attenuation = None
    if baseline.get("status") == "ok" and adjusted.get("status") == "ok":
        base = baseline.get("cg4_delta_dex")
        adj = adjusted.get("cg4_delta_dex")
        if base not in (None, 0):
            attenuation = float(1 - abs(adj) / abs(base))
    return {
        "status": "ok",
        "n_complete": int(len(panel)),
        "baseline": baseline,
        "with_tidal_index": adjusted,
        "attenuation_fraction": attenuation,
    }


# ---------------------------------------------------------------------------
# Block K: radial residual trend (exploratory)
# ---------------------------------------------------------------------------


def _radial(frame: pd.DataFrame, output_dir: str | None) -> dict:
    satellites = frame.loc[
        (frame["is_satellite"] == 1)
        & frame[PRIMARY_OUTCOME].notna()
        & frame["logMstar"].notna()
        & frame["z_numeric"].notna()
    ].copy()
    if "dist2BGG_kpc" not in satellites.columns:
        return _skipped("missing_dist2BGG_kpc")
    reference = satellites.loc[satellites["is_CG4"] == 0]
    if len(reference) < 30:
        return _skipped("too_few_reference_satellites", n=int(len(reference)))

    design = np.column_stack(
        [
            np.ones(len(reference)),
            reference["logMstar"],
            reference["logMstar"] ** 2,
            reference["z_numeric"],
        ]
    )
    coefficients, *_ = np.linalg.lstsq(
        design, reference[PRIMARY_OUTCOME].to_numpy(dtype=float), rcond=None
    )
    full_design = np.column_stack(
        [
            np.ones(len(satellites)),
            satellites["logMstar"],
            satellites["logMstar"] ** 2,
            satellites["z_numeric"],
        ]
    )
    satellites["size_residual"] = satellites[
        PRIMARY_OUTCOME
    ].to_numpy() - full_design.dot(coefficients)

    correlations = {}
    for label, part in [
        ("CG4", satellites.loc[satellites["is_CG4"] == 1]),
        ("pooled_controls", satellites.loc[satellites["is_CG4"] == 0]),
    ]:
        panel = part[["size_residual", "dist2BGG_kpc"]].dropna()
        if len(panel) < 10:
            correlations[label] = _skipped("too_few_satellites", n=int(len(panel)))
            continue
        test = stats.spearmanr(panel["dist2BGG_kpc"], panel["size_residual"])
        correlations[label] = {
            "status": "ok",
            "n": int(len(panel)),
            "spearman_rho": float(test.statistic),
            "spearman_p": float(test.pvalue),
        }
    result = {
        "status": "ok",
        "note": (
            "exploratory; residuals of log size versus the pooled-control "
            "satellite mass+z relation"
        ),
        "correlations": correlations,
    }
    if output_dir:
        result["figure"] = _plot_radial(
            satellites, os.path.join(output_dir, "size_radial.pdf")
        )
    return result


def _plot_radial(satellites: pd.DataFrame, path: str) -> str | None:
    panel = satellites[["size_residual", "dist2BGG_kpc", "is_CG4"]].dropna()
    if panel.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for is_cg4, label, colour, size in [
        (0, "Controls", "#777777", 8),
        (1, "CG$_4$", "#2864A6", 16),
    ]:
        part = panel.loc[panel["is_CG4"] == is_cg4]
        ax.scatter(
            part["dist2BGG_kpc"],
            part["size_residual"],
            s=size,
            alpha=0.4,
            color=colour,
            label=label,
        )
        if len(part) >= 20:
            bins = np.quantile(part["dist2BGG_kpc"], np.linspace(0, 1, 6))
            centers, medians = [], []
            for low, high in zip(bins[:-1], bins[1:]):
                mask = part["dist2BGG_kpc"].between(low, high)
                if mask.sum() >= 3:
                    centers.append(part.loc[mask, "dist2BGG_kpc"].median())
                    medians.append(part.loc[mask, "size_residual"].median())
            ax.plot(centers, medians, color=colour, linewidth=1.6)
    ax.axhline(0, color="0.45", linestyle=":", linewidth=1)
    ax.set_xlabel("Projected distance to the BGG (kpc)")
    ax.set_ylabel(r"$\Delta\log R_{\rm chl,r}$ at fixed mass and $z$ (dex)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


# ---------------------------------------------------------------------------
# Block L: concentration index
# ---------------------------------------------------------------------------


def _concentration(frame: pd.DataFrame) -> dict:
    result = {
        "status": "ok",
        "note": (
            "C=R90/R50 in linear units; the satellites test belongs to "
            "family F2, the all-galaxy fit is descriptive"
        ),
        "outcome_units": "concentration ratio (not dex)",
    }
    result["all"] = fit_size_model(
        frame,
        CONCENTRATION_OUTCOME,
        extra_covariates=("psfWidth_r",),
        log_outcome=False,
    )
    result["satellites"] = fit_size_model(
        frame.loc[frame["is_satellite"] == 1],
        CONCENTRATION_OUTCOME,
        include_satellite_flag=False,
        extra_covariates=("psfWidth_r",),
        log_outcome=False,
    )
    if all(result[name].get("status") != "ok" for name in ["all", "satellites"]):
        result["status"] = "skipped"
        result["reason"] = "no_fittable_variant"
    return result


def _apply_f2_holm(petrosian: dict, concentration: dict) -> list[str]:
    """Holm across the pre-registered F2 family, writing p_holm in place."""

    members = [
        ("petrosian_all", petrosian.get("all", {})),
        ("petrosian_satellites", petrosian.get("satellites", {})),
        ("concentration_satellites", concentration.get("satellites", {})),
    ]
    ok = [(name, entry) for name, entry in members if entry.get("status") == "ok"]
    adjusted = holm_correction([entry.get("p") for _, entry in ok])
    for (name, entry), p_holm in zip(ok, adjusted):
        entry["p_holm"] = p_holm
    return [name for name, _ in members]


# ---------------------------------------------------------------------------
# Block M: verdicts and the Re-n plane figure
# ---------------------------------------------------------------------------


def _same_sign(a, b) -> bool:
    return a is not None and b is not None and float(a) * float(b) > 0


def _verdicts(results: dict) -> dict:
    adjusted = results.get("adjusted", {})
    all_fit = adjusted.get("all", {})
    satellite_fit = adjusted.get("satellites", {})
    satellite_delta = satellite_fit.get("cg4_delta_dex")

    direction = None
    if satellite_delta is not None:
        direction = "smaller" if satellite_delta < 0 else "larger"

    matched_effect = (
        results.get("matched", {}).get("effects", {}).get("delta_log_Rchl_r", {})
    )
    survives_matching = bool(
        matched_effect.get("status") == "ok"
        and matched_effect.get("p_holm") is not None
        and matched_effect.get("p_holm") < co.P_LIMIT
        and _same_sign(matched_effect.get("mean_delta"), satellite_delta)
    )

    crowding = results.get("crowding", {})
    crowd_excl = crowding.get("excluding_close", {}).get("satellites", {})
    crowd_flag = crowding.get("with_close_flag", {}).get("satellites", {})
    survives_crowding = bool(
        crowd_excl.get("status") == "ok"
        and crowd_flag.get("status") == "ok"
        and crowd_excl.get("p") is not None
        and crowd_excl.get("p") < co.P_LIMIT
        and crowd_flag.get("p") is not None
        and crowd_flag.get("p") < co.P_LIMIT
        and _same_sign(crowd_excl.get("cg4_delta_dex"), satellite_delta)
        and _same_sign(crowd_flag.get("cg4_delta_dex"), satellite_delta)
    )

    petro_sat = results.get("petrosian", {}).get("satellites", {})
    petro_consistent = bool(
        petro_sat.get("status") == "ok"
        and petro_sat.get("p") is not None
        and petro_sat.get("p") < co.P_LIMIT
        and _same_sign(petro_sat.get("cg4_delta_dex"), satellite_delta)
    )

    tidal = results.get("tidal", {})
    tidal_adjusted = tidal.get("with_tidal_index", {})
    absorbed_by_tidal_index = bool(
        tidal.get("status") == "ok"
        and tidal.get("attenuation_fraction") is not None
        and tidal.get("attenuation_fraction") > 0.5
        and tidal_adjusted.get("p") is not None
        and tidal_adjusted.get("p") > co.P_LIMIT
    )

    return {
        "primary_all_significant": bool(
            all_fit.get("p_holm") is not None and all_fit.get("p_holm") < co.P_LIMIT
        ),
        "primary_satellites_significant": bool(
            satellite_fit.get("p_holm") is not None
            and satellite_fit.get("p_holm") < co.P_LIMIT
        ),
        "direction": direction,
        "survives_matching": survives_matching,
        "survives_crowding": survives_crowding,
        "petro_consistent": petro_consistent,
        "absorbed_by_tidal_index": absorbed_by_tidal_index,
        "completeness_caveat": bool(
            results.get("availability_audit", {}).get("completeness_caveat", False)
        ),
    }


def _plot_re_n_plane(frame: pd.DataFrame, path: str) -> str | None:
    panel = frame[[PRIMARY_OUTCOME, "simard_ng", "is_CG4"]].dropna()
    if len(panel) < 30:
        return None
    controls = panel.loc[panel["is_CG4"] == 0]
    cg4 = panel.loc[panel["is_CG4"] == 1]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    if len(controls):
        ax.hexbin(
            controls["simard_ng"],
            controls[PRIMARY_OUTCOME],
            gridsize=36,
            mincnt=1,
            bins="log",
            cmap="Greys",
            linewidths=0,
            alpha=0.62,
        )
    if len(cg4):
        ax.scatter(
            cg4["simard_ng"],
            cg4[PRIMARY_OUTCOME],
            s=28,
            alpha=0.9,
            facecolor="#D55E00",
            edgecolor="black",
            linewidth=0.35,
            label="CG$_4$ galaxies",
            zorder=4,
        )

    bins = np.array([0.5, 1.25, 2.0, 3.0, 4.5, 6.0, 8.0])

    def binned_median(part: pd.DataFrame, minimum: int):
        x_values = []
        y_values = []
        for left, right in zip(bins[:-1], bins[1:]):
            in_bin = part["simard_ng"].between(left, right, inclusive="left")
            values = part.loc[in_bin, PRIMARY_OUTCOME].dropna()
            if len(values) >= minimum:
                x_values.append((left + right) / 2)
                y_values.append(float(values.median()))
        return np.array(x_values), np.array(y_values)

    ctrl_x, ctrl_y = binned_median(controls, minimum=25)
    if len(ctrl_x):
        ax.plot(
            ctrl_x,
            ctrl_y,
            color="0.20",
            linewidth=1.9,
            marker="s",
            markersize=4.2,
            label="Control median",
            zorder=3,
        )
    cg_x, cg_y = binned_median(cg4, minimum=4)
    if len(cg_x):
        ax.plot(
            cg_x,
            cg_y,
            color="#0072B2",
            linewidth=2.1,
            marker="o",
            markersize=4.8,
            markeredgecolor="black",
            markeredgewidth=0.35,
            label="CG$_4$ median",
            zorder=5,
        )

    ax.set_xlabel("Sersic index $n_g$")
    ax.set_ylabel(r"$\log_{10}(R_{\rm chl,r}/{\rm kpc})$")
    ax.set_xlim(0.45, 8.05)
    y_min = float(panel[PRIMARY_OUTCOME].min())
    y_max = float(panel[PRIMARY_OUTCOME].max())
    padding = max(0.06, 0.04 * (y_max - y_min))
    ax.set_ylim(y_min - padding, y_max + padding)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="h",
            color="none",
            markerfacecolor="0.72",
            markeredgecolor="0.72",
            markersize=9,
            label="Pooled-control density",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#D55E00",
            markeredgecolor="black",
            markersize=6,
            label="CG$_4$ galaxies",
        ),
    ]
    if len(ctrl_x):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="0.20",
                marker="s",
                markersize=4,
                label="Control median",
            )
        )
    if len(cg_x):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#0072B2",
                marker="o",
                markersize=4,
                label="CG$_4$ median",
            )
        )
    ax.legend(handles=legend_handles, frameon=False, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_size_analysis(data, output_dir: str | None = None) -> dict[str, object]:
    """Run the galaxy-size analysis blocks and return one JSON-able dict."""

    frame = ensure_galaxy_frame(data)
    if frame.empty or "sample" not in frame:
        return _skipped("no_galaxy_samples")
    if PRIMARY_OUTCOME not in frame.columns:
        try:
            frame, _ = attach_size_columns(frame)
        except Exception as exc:
            return _skipped(
                "size_data_unavailable", error=f"{exc.__class__.__name__}: {exc}"
            )
    if PRIMARY_OUTCOME not in frame.columns:
        return _skipped("size_columns_missing_after_attach")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    frame = frame.copy()
    # Reuse the crowding-analysis nearest-neighbour geometry and threshold.
    if {"RA", "Dec", "group_uid"}.issubset(frame.columns):
        frame["nearest_angular_separation_arcsec"] = _nearest_angular(frame)
        frame["close_neighbour"] = (
            frame["nearest_angular_separation_arcsec"] < CROWDING_THRESHOLD_ARCSEC
        ).astype(float)

    # Deduplicate the control pool for all *pooled* adjusted models: one row
    # per physical galaxy, kept-label priority RG4 > Control4B > Control4C.
    # This mirrors the Sect. 5.2 convention (specialness_models.py) and ensures
    # cluster-robust SEs are by physical group rather than label-scoped group.
    # The full non-deduped frame is retained for per-control fits only.
    frame_dedup = dedup_control_pool(frame)

    results: dict[str, object] = {
        "status": "ok",
        "seed": SEED,
        "n_boot": N_BOOT,
        "primary_size_column": "Rchl_r (Simard et al. 2011 table3, pure Sersic)",
        "quality_cuts": {
            "size_min_kpc": float(co.SIZE_MIN_KPC),
            "size_max_kpc": float(co.SIZE_MAX_KPC),
            "ng_peg_low": float(co.NG_PEG_LOW),
            "ng_peg_high": float(co.NG_PEG_HIGH),
            "z_match_tolerance": float(Z_MATCH_TOLERANCE),
            "close_neighbour_arcsec": float(CROWDING_THRESHOLD_ARCSEC),
        },
        "holm_families": {
            "F1": ["adjusted.all", "adjusted.satellites", "adjusted.bgg"],
            "F2": [
                "petrosian.all",
                "petrosian.satellites",
                "concentration.satellites",
            ],
            "F3": [
                "matched.delta_log_Rchl_r",
                "matched.delta_log_petroR50",
                "matched.delta_concentration",
            ],
        },
    }

    blocks = [
        # availability_audit and mass_size use the full (non-deduped) frame
        # for descriptive counts and availability fractions that should reflect
        # all rows; the fitted models below use frame_dedup.
        ("availability_audit", lambda: _availability_audit(frame, output_dir)),
        ("mass_size", lambda: _mass_size(frame_dedup, output_dir)),
        # Primary and secondary adjusted models use the deduped frame:
        # one row per physical galaxy, clustered by physical_group.
        ("adjusted", lambda: _adjusted(frame_dedup)),
        # Per-control fits are label-scoped by design (one control type at a
        # time); they use the full frame so all galaxies within each label are
        # present and clustering is by physical_group within each label.
        ("per_control", lambda: _per_control(frame, output_dir)),
        ("morphology_strata", lambda: _morphology_strata(frame_dedup)),
        (
            "luminosity_version",
            lambda: {
                "status": "ok",
                "note": "descriptive analogue of the Coenda size-luminosity test",
                "all": fit_size_model(
                    frame_dedup, PRIMARY_OUTCOME, use_mass=False, use_luminosity=True
                ),
                "satellites": fit_size_model(
                    frame_dedup.loc[frame_dedup["is_satellite"] == 1],
                    PRIMARY_OUTCOME,
                    use_mass=False,
                    use_luminosity=True,
                    include_satellite_flag=False,
                ),
                "bgg": fit_size_model(
                    frame_dedup.loc[frame_dedup["is_bgg"] == 1],
                    PRIMARY_OUTCOME,
                    use_mass=False,
                    use_luminosity=True,
                    include_satellite_flag=False,
                ),
            },
        ),
        ("matched", lambda: _matched(frame_dedup)),
        ("crowding", lambda: _crowding(frame_dedup)),
        ("petrosian", lambda: _petrosian(frame_dedup)),
        ("measure_delta", lambda: _measure_delta(frame_dedup, output_dir)),
        ("tidal", lambda: _tidal(frame_dedup)),
        ("radial", lambda: _radial(frame_dedup, output_dir)),
        ("concentration", lambda: _concentration(frame_dedup)),
    ]
    for name, function in blocks:
        try:
            results[name] = function()
        except Exception as exc:  # One block must not sink the whole analysis.
            results[name] = _skipped(
                "block_exception", error=f"{exc.__class__.__name__}: {exc}"
            )

    results["holm_families"]["F2_members"] = _apply_f2_holm(
        results.get("petrosian", {}), results.get("concentration", {})
    )
    results["verdicts"] = _verdicts(results)
    if output_dir:
        try:
            results["re_n_plane_figure"] = _plot_re_n_plane(
                frame_dedup, os.path.join(output_dir, "size_re_n_plane.pdf")
            )
        except Exception:
            results["re_n_plane_figure"] = None
    return safe_json(results)
