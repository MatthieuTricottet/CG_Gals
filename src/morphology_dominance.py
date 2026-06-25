"""Morphology dependence on group class, magnitude gap and BGG dominance."""

from __future__ import annotations

import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
except ModuleNotFoundError:  # pragma: no cover
    sm = None
    PerfectSeparationWarning = Warning

try:
    import config as co
    from extended_data import ensure_galaxy_frame, first_existing
    from extended_stats import safe_float, safe_json
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from .extended_data import ensure_galaxy_frame, first_existing
    from .extended_stats import safe_float, safe_json


SAMPLES = ["CG4", "RG4"]
SUBSETS = {
    "all": "all galaxies",
    "bgg": "BGGs only",
    "satellites": "satellites only",
}
CLASS_ORDER = ["Split", "Isolated", "Predominant", "Embedded"]
CLASS_ALIASES = {
    "Predom": "Predominant",
    "predom": "Predominant",
    "Predominant": "Predominant",
    "Embedded": "Embedded",
    "Isolated": "Isolated",
    "Split": "Split",
}
PREDICTORS = {
    "Delta_m12": r"$\Delta m_{12}$",
    "f_L_BGG": r"$f_{L,\mathrm{BGG}}$",
}
CONTROL_COLUMNS = ["logMstar", "z_numeric"]
RNG_SEED = 20260625


def _numeric(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    column = first_existing(frame, names)
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_exp(value: float) -> float | None:
    with np.errstate(over="ignore", invalid="ignore"):
        return safe_float(np.exp(value))


def _significance_flag(p_value: float | None) -> str:
    p_value = safe_float(p_value)
    if p_value is None:
        return "inconclusive"
    if p_value < 0.05:
        return "significant"
    if p_value < 0.10:
        return "marginal"
    return "not_significant"


def _format_p_latex(p_value: float | None) -> str:
    p_value = safe_float(p_value)
    if p_value is None:
        return r"\(p\) unavailable"
    if p_value < 1e-6:
        return r"\(p < 10^{-6}\)"
    if p_value < 0.001:
        return rf"\(p = {p_value:.1e}\)"
    return rf"\(p = {p_value:.3f}\)"


def _format_or_latex(odds_ratio: float | None) -> str:
    odds_ratio = safe_float(odds_ratio)
    if odds_ratio is None:
        return r"OR unavailable"
    return rf"OR \(={odds_ratio:.2f}\)"


def _join_phrases(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _first_finite(frame: pd.DataFrame, names: list[str]) -> tuple[float, str | None]:
    for name in names:
        if name not in frame:
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        values = values[np.isfinite(values)]
        if not values.empty:
            return float(values.iloc[0]), name
    return np.nan, None


def _normalise_class(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    text = text.where(~text.str.lower().isin(["", "nan", "none", "<na>"]))
    return text.map(lambda value: CLASS_ALIASES.get(value, value) if pd.notna(value) else pd.NA)


def _morphology_labels(frame: pd.DataFrame) -> pd.Series:
    if "morphology" in frame:
        labels = frame["morphology"].astype("string").str.strip()
        labels = labels.where(labels.isin(["Elliptical", "Spiral"]))
        return labels

    labels = pd.Series(pd.NA, index=frame.index, dtype="string")
    elliptical = _numeric(frame, ["elliptical"])
    spiral = _numeric(frame, ["spiral"])
    labels.loc[elliptical.eq(1)] = "Elliptical"
    labels.loc[spiral.eq(1)] = "Spiral"
    return labels


def _rank_values(frame: pd.DataFrame) -> tuple[pd.Series, str | None]:
    column = first_existing(frame, ["rank", "rank_M", "rank_M_CG", "rank_M_LT"])
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float), None
    return pd.to_numeric(frame[column], errors="coerce"), column


def _class_values(frame: pd.DataFrame) -> tuple[pd.Series, str | None]:
    column = first_existing(
        frame,
        ["Class", "group_Class", "ZhengShenClass", "ZS_Class", "Zheng_Shen_class"],
    )
    if column is None:
        return pd.Series(pd.NA, index=frame.index, dtype="string"), None
    return _normalise_class(frame[column]), column


def _ensure_common_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "sample" not in out:
        out["sample"] = "unknown"
    if "group_uid" not in out:
        if {"sample", "Group"}.issubset(out.columns):
            out["group_uid"] = out["sample"].astype(str) + ":" + out["Group"].astype(str)
        elif "Group" in out:
            out["group_uid"] = out["Group"].astype(str)
        else:
            out["group_uid"] = out.index.astype(str)

    out["morphology_binary"] = _morphology_labels(out)
    is_elliptical = out["morphology_binary"].eq("Elliptical").fillna(False)
    is_spiral = out["morphology_binary"].eq("Spiral").fillna(False)
    out["elliptical_binary"] = np.where(
        is_elliptical,
        1.0,
        np.where(is_spiral, 0.0, np.nan),
    )

    rank, rank_source = _rank_values(out)
    out["rank_numeric"] = rank
    if "is_bgg" in out:
        out["is_bgg_numeric"] = pd.to_numeric(out["is_bgg"], errors="coerce")
    else:
        out["is_bgg_numeric"] = np.where(rank.eq(1), 1.0, np.nan)
    if "is_satellite" in out:
        out["is_satellite_numeric"] = pd.to_numeric(out["is_satellite"], errors="coerce")
    else:
        out["is_satellite_numeric"] = np.where(rank.notna(), rank.gt(1).astype(float), np.nan)

    class_label, class_source = _class_values(out)
    out["class_label"] = class_label
    out.attrs["rank_source"] = rank_source
    out.attrs["class_source"] = class_source
    return out


def _gap_from_group(group: pd.DataFrame) -> tuple[float, str]:
    magnitudes = _numeric(group, ["M_r", "M_r_CG", "M_r_LT", "absMag_r"])
    rank = pd.to_numeric(group.get("rank_numeric"), errors="coerce")

    ranked = pd.DataFrame({"rank": rank, "M_r": magnitudes}).dropna()
    if not ranked.empty:
        bgg = ranked.loc[ranked["rank"].eq(1), "M_r"]
        second = ranked.loc[ranked["rank"].eq(2), "M_r"]
        if not bgg.empty and not second.empty:
            return float(second.iloc[0] - bgg.iloc[0]), "rank_1_2_M_r"

    sorted_magnitudes = magnitudes.dropna().sort_values().to_numpy()
    if len(sorted_magnitudes) >= 2:
        return float(sorted_magnitudes[1] - sorted_magnitudes[0]), "sorted_M_r"
    return np.nan, "insufficient_valid_M_r"


def _bgg_fraction_from_group(group: pd.DataFrame) -> tuple[float, str]:
    lum_bgg, lum_bgg_source = _first_finite(
        group,
        ["Lum_BGG", "group_Lum_BGG", "Lum_BGG_CG", "Lum_BGG_LT"],
    )
    lum_group, lum_group_source = _first_finite(
        group,
        [
            "Lum_group",
            "group_Lum_group",
            "Lum_Group_CG",
            "Lum_Group_LT",
            "Lum_CG",
            "Lum_LT",
        ],
    )
    if np.isfinite(lum_bgg) and np.isfinite(lum_group) and lum_bgg > 0 and lum_group > 0:
        return lum_bgg / lum_group, f"{lum_bgg_source}/{lum_group_source}"

    frac, frac_source = _first_finite(
        group,
        ["FracLumBGG", "group_FracLumBGG", "f_L_BGG", "dominance"],
    )
    if np.isfinite(frac):
        return frac, frac_source or "fraction_column"

    luminosity = _numeric(group, ["Lum", "Lum_CG", "Lum_LT"])
    rank = pd.to_numeric(group.get("rank_numeric"), errors="coerce")
    ranked = pd.DataFrame({"rank": rank, "Lum": luminosity}).dropna()
    if not ranked.empty:
        bgg_lum = ranked.loc[ranked["rank"].eq(1), "Lum"]
        total_lum = luminosity.dropna().sum()
        if not bgg_lum.empty and total_lum > 0:
            return float(bgg_lum.iloc[0] / total_lum), "rank_1_Lum/sum_Lum"
    clean_luminosity = luminosity.dropna()
    if len(clean_luminosity) and clean_luminosity.sum() > 0:
        return float(clean_luminosity.max() / clean_luminosity.sum()), "max_Lum/sum_Lum"
    return np.nan, "missing_luminosity"


def compute_group_dominance_variables(
    data,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Attach group-level ``Delta_m12`` and ``f_L_BGG`` to a galaxy frame."""

    frame = _ensure_common_columns(ensure_galaxy_frame(data))
    if frame.empty:
        return frame, pd.DataFrame(), {"status": "skipped", "reason": "no_galaxies"}
    rank_source = frame.attrs.get("rank_source")
    class_source = frame.attrs.get("class_source")

    rows = []
    for group_uid, group in frame.groupby("group_uid", observed=True):
        first = group.iloc[0]
        gap, gap_source = _gap_from_group(group)
        frac, frac_source = _bgg_fraction_from_group(group)
        class_values = group["class_label"].dropna()
        rows.append(
            {
                "group_uid": group_uid,
                "sample": first.get("sample"),
                "Group": first.get("Group", group_uid),
                "Delta_m12": gap,
                "Delta_m12_source": gap_source,
                "f_L_BGG_raw": frac,
                "f_L_BGG_source": frac_source,
                "class_label": class_values.iloc[0] if not class_values.empty else pd.NA,
                "n_members": int(len(group)),
            }
        )

    groups = pd.DataFrame(rows)
    frac = pd.to_numeric(groups["f_L_BGG_raw"], errors="coerce")
    valid_frac = np.isfinite(frac) & (frac > 0) & (frac <= 1 + 1e-8)
    groups["f_L_BGG"] = frac.where(valid_frac)
    groups.loc[groups["f_L_BGG"] > 1, "f_L_BGG"] = 1.0

    frame = frame.drop(columns=["Delta_m12", "f_L_BGG"], errors="ignore").merge(
        groups[["group_uid", "Delta_m12", "f_L_BGG"]],
        on="group_uid",
        how="left",
        validate="m:1",
    )

    audit = {
        "status": "ok",
        "n_groups": int(len(groups)),
        "rank_source": rank_source,
        "class_source": class_source,
        "magnitude_gap_definition": "M_r(rank 2) - M_r(rank 1), with sorted M_r fallback",
        "bgg_fraction_definition": "Lum_BGG / Lum_group, with FracLumBGG and galaxy-Lum fallbacks",
        "gap_source_counts": groups["Delta_m12_source"].value_counts(dropna=False).to_dict(),
        "f_L_BGG_source_counts": groups["f_L_BGG_source"].value_counts(dropna=False).to_dict(),
        "n_groups_missing_Delta_m12": int(groups["Delta_m12"].isna().sum()),
        "n_groups_missing_or_invalid_f_L_BGG": int(groups["f_L_BGG"].isna().sum()),
        "n_groups_invalid_f_L_BGG": int((frac.notna() & ~valid_frac).sum()),
        "notes": [
            "The existing pipeline removes Zheng & Shen Split compact groups before analysis, so CG4 Split rows are expected to have zero counts.",
            "The current RG4 catalogue has no Zheng & Shen class labels; RG4 class-association tests are therefore reported as missing-class exclusions.",
        ],
    }
    return frame, groups, audit


def _subset(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "all":
        return frame.copy()
    if subset == "bgg":
        return frame.loc[pd.to_numeric(frame["is_bgg_numeric"], errors="coerce").eq(1)].copy()
    if subset == "satellites":
        return frame.loc[
            pd.to_numeric(frame["is_satellite_numeric"], errors="coerce").eq(1)
        ].copy()
    raise ValueError(f"Unknown subset: {subset}")


def _chi2_statistic(table: np.ndarray) -> float:
    return float(stats.chi2_contingency(table, correction=False).statistic)


def _permutation_chi2(table: np.ndarray, n_iter: int, seed: int) -> dict[str, object]:
    row_totals = table.sum(axis=1).astype(int)
    class_index = np.repeat(np.arange(table.shape[0]), row_totals)
    morph = []
    for e_count, sp_count in table.astype(int):
        morph.extend([1] * int(e_count))
        morph.extend([0] * int(sp_count))
    morph = np.asarray(morph)
    observed = _chi2_statistic(table)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_iter):
        shuffled = rng.permutation(morph)
        permuted = np.zeros_like(table, dtype=int)
        for row in range(table.shape[0]):
            values = shuffled[class_index == row]
            permuted[row, 0] = int(values.sum())
            permuted[row, 1] = int(len(values) - values.sum())
        if _chi2_statistic(permuted) >= observed - 1e-12:
            exceed += 1
    return {
        "test_used": "Monte Carlo permutation chi-square",
        "statistic": observed,
        "p_value": float((exceed + 1) / (n_iter + 1)),
        "n_permutations": int(n_iter),
    }


def test_morphology_class_association(
    table: pd.DataFrame | np.ndarray,
    *,
    n_iter: int = 5000,
    seed: int = RNG_SEED,
) -> dict[str, object]:
    """Test whether the E-vs-Sp mix depends on Zheng & Shen class."""

    values = np.asarray(table, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        return {"status": "skipped", "reason": "table_must_have_two_morphology_columns"}
    values = values[values.sum(axis=1) > 0]
    values = values[:, values.sum(axis=0) > 0]
    if values.shape[0] < 2 or values.shape[1] < 2:
        return {"status": "skipped", "reason": "insufficient_class_or_morphology_diversity"}

    try:
        chi2 = stats.chi2_contingency(values, correction=False)
    except ValueError as exc:
        return {"status": "skipped", "reason": "invalid_contingency_table", "error": str(exc)}

    expected = chi2.expected_freq
    if np.all(expected >= 5):
        return {
            "status": "ok",
            "test_used": "chi-square",
            "statistic": float(chi2.statistic),
            "p_value": float(chi2.pvalue),
            "degrees_of_freedom": int(chi2.dof),
            "min_expected": float(expected.min()),
        }
    if values.shape == (2, 2):
        fisher = stats.fisher_exact(values, alternative="two-sided")
        return {
            "status": "ok",
            "test_used": "Fisher exact",
            "statistic": float(fisher.statistic),
            "p_value": float(fisher.pvalue),
            "min_expected": float(expected.min()),
        }
    result = _permutation_chi2(values.astype(int), n_iter=n_iter, seed=seed)
    result.update({"status": "ok", "min_expected": float(expected.min())})
    return result


def summarize_morphology_by_class(
    frame: pd.DataFrame,
    sample_name: str,
    subset: str,
) -> dict[str, object]:
    """Return E-vs-Sp counts by Zheng & Shen class and the association test."""

    part = _subset(frame.loc[frame["sample"].eq(sample_name)], subset)
    valid_morph = part["elliptical_binary"].isin([0, 1])
    valid_class = part["class_label"].notna()
    used = part.loc[valid_morph & valid_class].copy()
    classes = CLASS_ORDER + sorted(
        set(used["class_label"].dropna().astype(str)) - set(CLASS_ORDER)
    )

    rows = []
    for class_name in classes:
        cls = used.loc[used["class_label"].eq(class_name)]
        n_e = int(cls["elliptical_binary"].eq(1).sum())
        n_sp = int(cls["elliptical_binary"].eq(0).sum())
        denom = n_e + n_sp
        rows.append(
            {
                "class": class_name,
                "N_E": n_e,
                "N_Sp": n_sp,
                "f_E": float(n_e / denom) if denom else None,
                "f_Sp": float(n_sp / denom) if denom else None,
            }
        )

    table = pd.DataFrame(rows)[["N_E", "N_Sp"]]
    test = test_morphology_class_association(table.to_numpy())
    return {
        "sample": sample_name,
        "subset": subset,
        "subset_label": SUBSETS[subset],
        "contingency_table": rows,
        "test": test,
        "interpretation_flag": _significance_flag(test.get("p_value")),
        "n_total": int(len(part)),
        "n_galaxies_used": int(len(used)),
        "n_groups_used": int(used["group_uid"].nunique()) if "group_uid" in used else None,
        "n_excluded_morphology_not_E_or_Sp": int((~valid_morph).sum()),
        "n_excluded_class_missing": int((valid_morph & ~valid_class).sum()),
    }


def _distribution_summary(work: pd.DataFrame, predictor: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for label, value in [("Elliptical", 1), ("Spiral", 0)]:
        vals = pd.to_numeric(
            work.loc[work["elliptical_binary"].eq(value), predictor], errors="coerce"
        ).dropna()
        out[label] = {
            "n": int(len(vals)),
            "median": safe_float(vals.median()),
            "mean": safe_float(vals.mean()),
            "q25": safe_float(vals.quantile(0.25)) if len(vals) else None,
            "q75": safe_float(vals.quantile(0.75)) if len(vals) else None,
        }
    e_vals = pd.to_numeric(
        work.loc[work["elliptical_binary"].eq(1), predictor], errors="coerce"
    ).dropna()
    sp_vals = pd.to_numeric(
        work.loc[work["elliptical_binary"].eq(0), predictor], errors="coerce"
    ).dropna()
    if len(e_vals) and len(sp_vals):
        out["mannwhitney_p"] = float(
            stats.mannwhitneyu(e_vals, sp_vals, alternative="two-sided").pvalue
        )
    else:
        out["mannwhitney_p"] = None
    return out


def _select_controls(work: pd.DataFrame, base_required: list[str]) -> list[str]:
    controls = []
    for column in CONTROL_COLUMNS:
        if column not in work:
            continue
        values = pd.to_numeric(work[column], errors="coerce")
        if values.notna().sum() < 10 or values.nunique(dropna=True) < 2:
            continue
        trial_required = base_required + controls + [column]
        complete = work[trial_required].replace([np.inf, -np.inf], np.nan).dropna()
        if len(complete) >= 10:
            controls.append(column)
    return controls


def _fit_logistic(
    frame: pd.DataFrame,
    predictor: str,
    *,
    subset: str,
    controls: list[str] | None = None,
    interaction: bool = False,
) -> dict[str, object]:
    if sm is None:
        return {"status": "skipped", "reason": "statsmodels_unavailable"}

    controls = controls or []
    predictor_term = predictor
    required = ["elliptical_binary", predictor_term, *controls]
    if interaction:
        required = ["elliptical_binary", "is_CG4_numeric", predictor_term, "interaction", *controls]
    columns = list(dict.fromkeys(required + (["group_uid"] if "group_uid" in frame else [])))
    missing = [column for column in required if column not in frame]
    if missing:
        return {"status": "skipped", "reason": "missing_required_columns", "missing": missing}

    work = frame[columns].replace([np.inf, -np.inf], np.nan).copy()
    for column in required:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    n_available_before_drop = int(len(work))
    work = work.dropna(subset=required)
    counts = work["elliptical_binary"].value_counts()
    if len(work) < 10:
        return {
            "status": "skipped",
            "reason": "too_few_complete_cases",
            "n_used": int(len(work)),
            "n_excluded": int(n_available_before_drop - len(work)),
            "controls_included": controls,
        }
    if len(counts) != 2 or int(counts.min()) < 3:
        return {
            "status": "skipped",
            "reason": "too_few_morphology_cases",
            "n_used": int(len(work)),
            "outcome_counts": {str(key): int(value) for key, value in counts.items()},
            "controls_included": controls,
        }

    xcols = [predictor_term]
    if interaction:
        xcols = ["is_CG4_numeric", predictor_term, "interaction"]
    xcols += [column for column in controls if work[column].nunique(dropna=True) >= 2]

    for column in controls:
        if column not in xcols:
            continue
        mean = float(work[column].mean())
        std = float(work[column].std(ddof=0))
        if math.isfinite(std) and std > 0:
            work[column] = (work[column] - mean) / std

    design = sm.add_constant(work[xcols].astype(float), has_constant="add")
    groups = work["group_uid"] if "group_uid" in work else None
    use_cluster = subset in {"all", "satellites"} and groups is not None and groups.nunique() >= 2
    covariance = "cluster" if use_cluster else "HC1"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("error", PerfectSeparationWarning)
            model = sm.GLM(
                work["elliptical_binary"].astype(float),
                design,
                family=sm.families.Binomial(),
            )
            if use_cluster:
                fit = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            else:
                fit = model.fit(cov_type="HC1")
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": "model_fit_failed",
            "error": str(exc),
            "n_used": int(len(work)),
            "n_excluded": int(n_available_before_drop - len(work)),
            "controls_included": controls,
        }

    if not np.isfinite(np.asarray(fit.params)).all() or np.max(np.abs(np.asarray(fit.params))) > 30:
        return {
            "status": "skipped",
            "reason": "perfect_or_quasi_separation",
            "n_used": int(fit.nobs),
            "controls_included": controls,
        }

    params = fit.params
    errors = fit.bse
    p_values = fit.pvalues
    ci = fit.conf_int()
    terms = {}
    for term in params.index:
        low, high = map(float, ci.loc[term])
        coefficient = float(params[term])
        terms[term] = {
            "coefficient": coefficient,
            "standard_error": float(errors[term]),
            "odds_ratio": _safe_exp(coefficient),
            "ci95": [_safe_exp(low), _safe_exp(high)],
            "p_value": float(p_values[term]),
        }

    target = "interaction" if interaction else predictor_term
    target_term = terms[target]
    formula_terms = xcols
    return {
        "status": "ok",
        "model_converged": bool(getattr(fit, "converged", True)),
        "formula": "elliptical_binary ~ " + " + ".join(formula_terms),
        "n_used": int(fit.nobs),
        "n_excluded": int(n_available_before_drop - fit.nobs),
        "n_groups": int(groups.nunique()) if groups is not None else None,
        "controls_included": [column for column in controls if column in xcols],
        "covariance": covariance,
        "clustered_errors": bool(use_cluster),
        "beta1": target_term["coefficient"],
        "standard_error": target_term["standard_error"],
        "odds_ratio": target_term["odds_ratio"],
        "ci95": target_term["ci95"],
        "p_value": target_term["p_value"],
        "interpretation_flag": _significance_flag(target_term["p_value"]),
        "terms": terms,
    }


def test_morphology_vs_continuous_dominance(
    frame: pd.DataFrame,
    sample_name: str,
    subset: str,
    predictor: str,
) -> dict[str, object]:
    """Fit morphology-vs-dominance logistic models for one sample/subset."""

    part = _subset(frame.loc[frame["sample"].eq(sample_name)], subset)
    valid_morph = part["elliptical_binary"].isin([0, 1])
    valid_predictor = np.isfinite(pd.to_numeric(part[predictor], errors="coerce"))
    work = part.loc[valid_morph & valid_predictor].copy()
    base_required = ["elliptical_binary", predictor]
    controls = _select_controls(work, base_required)
    return {
        "sample": sample_name,
        "subset": subset,
        "subset_label": SUBSETS[subset],
        "predictor": predictor,
        "predictor_label": PREDICTORS[predictor],
        "n_total": int(len(part)),
        "n_complete_for_descriptive": int(len(work)),
        "n_excluded_morphology_not_E_or_Sp": int((~valid_morph).sum()),
        "n_excluded_missing_or_invalid_predictor": int((valid_morph & ~valid_predictor).sum()),
        "descriptive_by_morphology": _distribution_summary(work, predictor),
        "models": {
            "unadjusted": _fit_logistic(work, predictor, subset=subset, controls=[]),
            "adjusted": _fit_logistic(work, predictor, subset=subset, controls=controls),
        },
    }


def _interaction_flag(model: dict[str, object]) -> str:
    if model.get("status") != "ok":
        return "inconclusive"
    p_value = model.get("p_value")
    coefficient = model.get("beta1")
    if p_value is None or coefficient is None:
        return "inconclusive"
    if p_value >= 0.05:
        return "no_detected_difference"
    return "stronger_in_CG4" if coefficient > 0 else "stronger_in_RG4"


def _continuous_interaction(
    frame: pd.DataFrame,
    subset: str,
    predictor: str,
) -> dict[str, object]:
    part = _subset(frame.loc[frame["sample"].isin(SAMPLES)], subset)
    part = part.copy()
    part["is_CG4_numeric"] = part["sample"].eq("CG4").astype(float)
    part["interaction"] = part["is_CG4_numeric"] * pd.to_numeric(
        part[predictor], errors="coerce"
    )
    valid = part["elliptical_binary"].isin([0, 1]) & np.isfinite(
        pd.to_numeric(part[predictor], errors="coerce")
    )
    work = part.loc[valid].copy()
    controls = _select_controls(work, ["elliptical_binary", "is_CG4_numeric", predictor, "interaction"])
    unadjusted = _fit_logistic(
        work,
        predictor,
        subset=subset,
        controls=[],
        interaction=True,
    )
    adjusted = _fit_logistic(
        work,
        predictor,
        subset=subset,
        controls=controls,
        interaction=True,
    )
    preferred = adjusted if adjusted.get("status") == "ok" else unadjusted
    return {
        "subset": subset,
        "subset_label": SUBSETS[subset],
        "predictor": predictor,
        "predictor_label": PREDICTORS[predictor],
        "n_total": int(len(part)),
        "n_complete_before_controls": int(len(work)),
        "models": {"unadjusted": unadjusted, "adjusted": adjusted},
        "interpretation_flag": _interaction_flag(preferred),
    }


def _preferred_model(result: dict[str, object]) -> dict[str, object]:
    models = result.get("models", {})
    adjusted = models.get("adjusted", {})
    if adjusted.get("status") == "ok":
        return adjusted
    return models.get("unadjusted", {})


def _class_interaction(frame: pd.DataFrame, subset: str) -> dict[str, object]:
    part = _subset(frame.loc[frame["sample"].isin(SAMPLES)], subset).copy()
    valid = part["elliptical_binary"].isin([0, 1]) & part["class_label"].notna()
    work = part.loc[valid].copy()
    class_by_sample = {
        sample: sorted(work.loc[work["sample"].eq(sample), "class_label"].dropna().unique())
        for sample in SAMPLES
    }
    if any(len(values) < 2 for values in class_by_sample.values()):
        return {
            "subset": subset,
            "subset_label": SUBSETS[subset],
            "status": "skipped",
            "reason": "class_labels_not_available_in_both_samples",
            "class_levels_by_sample": class_by_sample,
            "n_total": int(len(part)),
            "n_complete": int(len(work)),
            "interpretation_flag": "inconclusive",
        }
    return {
        "subset": subset,
        "subset_label": SUBSETS[subset],
        "status": "skipped",
        "reason": "class_interaction_not_implemented_for_current_catalogue",
        "class_levels_by_sample": class_by_sample,
        "n_total": int(len(part)),
        "n_complete": int(len(work)),
        "interpretation_flag": "inconclusive",
    }


def _flatten_class_results(results: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for sample_name, subset_results in results.items():
        for subset, result in subset_results.items():
            for row in result["contingency_table"]:
                rows.append(
                    {
                        "sample": sample_name,
                        "subset": subset,
                        "class": row["class"],
                        "N_E": row["N_E"],
                        "N_Sp": row["N_Sp"],
                        "f_E": row["f_E"],
                        "f_Sp": row["f_Sp"],
                        "n_used": result["n_galaxies_used"],
                        "test_used": result["test"].get("test_used"),
                        "p_value": result["test"].get("p_value"),
                    }
                )
    return rows


def _flatten_model_results(results: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for predictor, sample_results in results.items():
        for sample_name, subset_results in sample_results.items():
            for subset, result in subset_results.items():
                for model_name, model in result["models"].items():
                    rows.append(
                        {
                            "predictor": predictor,
                            "sample": sample_name,
                            "subset": subset,
                            "model": model_name,
                            "status": model.get("status"),
                            "beta1": model.get("beta1"),
                            "standard_error": model.get("standard_error"),
                            "odds_ratio": model.get("odds_ratio"),
                            "p_value": model.get("p_value"),
                            "n_used": model.get("n_used"),
                            "covariance": model.get("covariance"),
                            "controls": ",".join(model.get("controls_included", [])),
                            "reason": model.get("reason"),
                        }
                    )
    return rows


def _flatten_interactions(results: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for predictor, subset_results in results.get("continuous", {}).items():
        for subset, result in subset_results.items():
            for model_name, model in result["models"].items():
                rows.append(
                    {
                        "predictor": predictor,
                        "subset": subset,
                        "model": model_name,
                        "status": model.get("status"),
                        "interaction_beta": model.get("beta1"),
                        "interaction_p": model.get("p_value"),
                        "n_used": model.get("n_used"),
                        "interpretation_flag": result.get("interpretation_flag"),
                        "reason": model.get("reason"),
                    }
                )
    for subset, result in results.get("class", {}).items():
        rows.append(
            {
                "predictor": "class",
                "subset": subset,
                "model": "categorical_interaction",
                "status": result.get("status"),
                "interaction_beta": None,
                "interaction_p": None,
                "n_used": result.get("n_complete"),
                "interpretation_flag": result.get("interpretation_flag"),
                "reason": result.get("reason"),
            }
        )
    return rows


def _summary(
    class_results: dict[str, object],
    continuous_results: dict[str, object],
    interaction_results: dict[str, object],
) -> dict[str, object]:
    class_sig = []
    for sample_name, subset_results in class_results.items():
        for subset, result in subset_results.items():
            p_value = result["test"].get("p_value")
            if p_value is not None and p_value < 0.05:
                class_sig.append({"sample": sample_name, "subset": subset, "p_value": p_value})
    class_marginal = []
    for sample_name, subset_results in class_results.items():
        for subset, result in subset_results.items():
            p_value = result["test"].get("p_value")
            if p_value is not None and 0.05 <= p_value < 0.10:
                class_marginal.append(
                    {"sample": sample_name, "subset": subset, "p_value": p_value}
                )

    trend_sig = []
    trend_marginal = []
    for predictor, sample_results in continuous_results.items():
        for sample_name, subset_results in sample_results.items():
            for subset, result in subset_results.items():
                model = _preferred_model(result)
                p_value = model.get("p_value")
                if p_value is not None and p_value < 0.05:
                    trend_sig.append(
                        {
                            "predictor": predictor,
                            "sample": sample_name,
                            "subset": subset,
                            "p_value": p_value,
                            "odds_ratio": model.get("odds_ratio"),
                        }
                    )
                elif p_value is not None and p_value < 0.10:
                    trend_marginal.append(
                        {
                            "predictor": predictor,
                            "sample": sample_name,
                            "subset": subset,
                            "p_value": p_value,
                            "odds_ratio": model.get("odds_ratio"),
                        }
                    )

    interaction_flags = [
        {
            "predictor": predictor,
            "subset": subset,
            "interpretation_flag": result.get("interpretation_flag"),
            "p_value": (
                result.get("models", {}).get("adjusted", {}).get("p_value")
                if result.get("models", {}).get("adjusted", {}).get("status") == "ok"
                else result.get("models", {}).get("unadjusted", {}).get("p_value")
            ),
        }
        for predictor, subset_results in interaction_results.get("continuous", {}).items()
        for subset, result in subset_results.items()
    ]
    interaction_detected = [
        item
        for item in interaction_flags
        if item["interpretation_flag"] in {"stronger_in_CG4", "stronger_in_RG4"}
    ]
    any_interaction_tested = any(
        item["interpretation_flag"] != "inconclusive" for item in interaction_flags
    )
    tested_class_p = [
        result["test"].get("p_value")
        for subset_results in class_results.values()
        for result in subset_results.values()
        if result["test"].get("p_value") is not None
    ]
    tested_trend_p = {
        predictor: [
            _preferred_model(result).get("p_value")
            for subset_results in sample_results.values()
            for result in subset_results.values()
            if _preferred_model(result).get("p_value") is not None
        ]
        for predictor, sample_results in continuous_results.items()
    }
    return {
        "n_significant_class_tests": len(class_sig),
        "significant_class_tests": class_sig,
        "n_marginal_class_tests": len(class_marginal),
        "marginal_class_tests": class_marginal,
        "n_significant_continuous_trends": len(trend_sig),
        "significant_continuous_trends": trend_sig,
        "n_marginal_continuous_trends": len(trend_marginal),
        "marginal_continuous_trends": trend_marginal,
        "continuous_interaction_flags": interaction_flags,
        "n_detected_cg_rg_interactions": len(interaction_detected),
        "has_class_dependence": bool(class_sig) if tested_class_p else None,
        "has_gap_dependence": any(item["predictor"] == "Delta_m12" for item in trend_sig)
        if tested_trend_p.get("Delta_m12")
        else None,
        "has_lumfrac_dependence": any(item["predictor"] == "f_L_BGG" for item in trend_sig)
        if tested_trend_p.get("f_L_BGG")
        else None,
        "cg4_rg4_difference_detected": bool(interaction_detected)
        if any_interaction_tested
        else None,
    }


def _class_result_sentence(class_results: dict[str, object]) -> str:
    significant = []
    marginal = []
    cg_bits = []
    for subset, result in class_results.get("CG4", {}).items():
        p_value = result.get("test", {}).get("p_value")
        if p_value is None:
            continue
        label = SUBSETS.get(subset, subset)
        cg_bits.append(f"{label} ({_format_p_latex(p_value)})")
        flag = _significance_flag(p_value)
        if flag == "significant":
            significant.append(f"CG4 {label}")
        elif flag == "marginal":
            marginal.append(f"CG4 {label} ({_format_p_latex(p_value)})")

    rg_missing = sum(
        result.get("n_excluded_class_missing", 0)
        for result in class_results.get("RG4", {}).values()
    )
    rg_note = (
        " The RG4 class test is unavailable because the current RG4 catalogue has "
        "no Zheng--Shen class labels."
        if rg_missing
        else ""
    )
    if significant:
        return (
            r"The Zheng--Shen class test reaches \(p<0.05\) for "
            f"{_join_phrases(significant)}; the remaining class splits are treated as descriptive."
            + rg_note
        )
    if marginal:
        return (
            r"No Zheng--Shen class association reaches \(p<0.05\); the closest result is "
            f"{_join_phrases(marginal)}, so it is treated as marginal and descriptive."
            + rg_note
        )
    if cg_bits:
        return (
            "In CG4, the E-vs-Sp mix does not show a statistically significant dependence "
            f"on Zheng--Shen class across {_join_phrases(cg_bits)}."
            + rg_note
        )
    return "The Zheng--Shen class association is inconclusive because no class-labelled E/Sp sample is available."


def _continuous_result_sentence(
    continuous_results: dict[str, object],
    predictor: str,
) -> str:
    label = PREDICTORS[predictor]
    significant = []
    marginal = []
    for sample_name, subset_results in continuous_results.get(predictor, {}).items():
        for subset, result in subset_results.items():
            model = _preferred_model(result)
            p_value = model.get("p_value")
            if p_value is None:
                continue
            item = (
                f"{sample_name} {SUBSETS.get(subset, subset)} "
                f"({_format_or_latex(model.get('odds_ratio'))}, {_format_p_latex(p_value)})"
            )
            flag = _significance_flag(p_value)
            if flag == "significant":
                significant.append(item)
            elif flag == "marginal":
                marginal.append(item)

    if significant:
        sentence = (
            f"For {label}, significant morphology trends are found for "
            f"{_join_phrases(significant)}."
        )
    elif marginal:
        sentence = (
            rf"For {label}, no fitted trend reaches \(p<0.05\); the marginal result is "
            f"{_join_phrases(marginal)}."
        )
    else:
        sentence = (
            f"For {label}, no statistically significant morphology trend is found in "
            "CG4 or RG4 for the all-galaxy, BGG-only, or satellite-only fits."
        )
    if predictor == "f_L_BGG":
        sentence += (
            " The odds ratios are per unit luminosity fraction on a 0--1 scale, "
            "so broad-interval estimates should not be overread."
        )
    return sentence


def _interaction_result_sentence(interaction_results: dict[str, object]) -> str:
    detected = []
    inconclusive_class = []
    for predictor, subset_results in interaction_results.get("continuous", {}).items():
        for subset, result in subset_results.items():
            flag = result.get("interpretation_flag")
            if flag in {"stronger_in_CG4", "stronger_in_RG4"}:
                preferred = _preferred_model(result)
                direction = "CG4" if flag == "stronger_in_CG4" else "RG4"
                detected.append(
                    f"{PREDICTORS[predictor]} in {SUBSETS.get(subset, subset)} "
                    f"(stronger in {direction}, {_format_p_latex(preferred.get('p_value'))})"
                )
    for subset, result in interaction_results.get("class", {}).items():
        if result.get("interpretation_flag") == "inconclusive":
            inconclusive_class.append(SUBSETS.get(subset, subset))
    if detected:
        sentence = (
            "The CG4--RG4 interaction models detect a different morphology--dominance "
            f"relation for {_join_phrases(detected)}."
        )
    else:
        sentence = (
            r"The CG4--RG4 interaction models detect no significant difference in the "
            r"morphology--dominance relation for either \(\Delta m_{12}\) or "
            r"\(f_{L,\mathrm{BGG}}\), across all galaxies, BGGs only, and satellites only."
        )
    if inconclusive_class:
        sentence += (
            " The class interaction is inconclusive because RG4 class labels are unavailable."
        )
    return sentence


def _sentences(
    class_results: dict[str, object],
    continuous_results: dict[str, object],
    interaction_results: dict[str, object],
    summary: dict[str, object],
) -> dict[str, str]:
    class_sentence = _class_result_sentence(class_results)
    gap_sentence = _continuous_result_sentence(continuous_results, "Delta_m12")
    lumfrac_sentence = _continuous_result_sentence(continuous_results, "f_L_BGG")
    interaction_sentence = _interaction_result_sentence(interaction_results)
    if summary.get("n_significant_class_tests") or summary.get("n_significant_continuous_trends"):
        overall = (
            r"The morphology--dominance analysis finds at least one \(p<0.05\) relation, "
            "but the result should be interpreted by subset rather than as a global trend."
        )
    elif summary.get("n_marginal_class_tests") or summary.get("n_marginal_continuous_trends"):
        overall = (
            "The morphology--dominance analysis is dominated by null results; marginal "
            "associations are reported as descriptive rather than decisive."
        )
    else:
        overall = (
            "The morphology--dominance analysis finds no statistically significant dependence "
            "on group class, magnitude gap, or BGG luminosity fraction, and no detected CG4--RG4 "
            "difference in those relations."
        )
    return {
        "main_conclusion": overall,
        "overall_result_sentence": overall,
        "class_result_sentence": class_sentence,
        "gap_result_sentence": gap_sentence,
        "lumfrac_result_sentence": lumfrac_sentence,
        "interaction_result_sentence": interaction_sentence,
    }


def run_morphology_dominance_analysis(data, output_dir: str | None = None) -> dict[str, object]:
    """Run morphology-vs-class, magnitude-gap and luminosity-dominance analyses."""

    frame, groups, audit = compute_group_dominance_variables(data)
    if audit.get("status") == "skipped":
        return safe_json(audit)

    frame = frame.loc[frame["sample"].isin(SAMPLES)].copy()
    groups = groups.loc[groups["sample"].isin(SAMPLES)].copy()
    audit = dict(audit)
    audit["n_groups_all_catalogues"] = audit.get("n_groups")
    audit["n_groups"] = int(len(groups))
    audit["gap_source_counts"] = groups["Delta_m12_source"].value_counts(dropna=False).to_dict()
    audit["f_L_BGG_source_counts"] = groups["f_L_BGG_source"].value_counts(dropna=False).to_dict()
    audit["n_groups_missing_Delta_m12"] = int(groups["Delta_m12"].isna().sum())
    audit["n_groups_missing_or_invalid_f_L_BGG"] = int(groups["f_L_BGG"].isna().sum())
    frac = pd.to_numeric(groups["f_L_BGG_raw"], errors="coerce")
    audit["n_groups_invalid_f_L_BGG"] = int(
        (frac.notna() & ~(np.isfinite(frac) & (frac > 0) & (frac <= 1 + 1e-8))).sum()
    )

    class_results = {
        sample_name: {
            subset: summarize_morphology_by_class(frame, sample_name, subset)
            for subset in SUBSETS
        }
        for sample_name in SAMPLES
    }
    continuous_results = {
        predictor: {
            sample_name: {
                subset: test_morphology_vs_continuous_dominance(
                    frame, sample_name, subset, predictor
                )
                for subset in SUBSETS
            }
            for sample_name in SAMPLES
        }
        for predictor in PREDICTORS
    }
    interaction_results = {
        "continuous": {
            predictor: {
                subset: _continuous_interaction(frame, subset, predictor)
                for subset in SUBSETS
            }
            for predictor in PREDICTORS
        },
        "class": {subset: _class_interaction(frame, subset) for subset in SUBSETS},
    }
    summary = _summary(class_results, continuous_results, interaction_results)
    sentences = _sentences(class_results, continuous_results, interaction_results, summary)

    table_dir = output_dir or co.OUTPUT_PATH
    os.makedirs(table_dir, exist_ok=True)
    pd.DataFrame(_flatten_class_results(class_results)).to_csv(
        os.path.join(table_dir, "morphology_dominance_class_counts.csv"), index=False
    )
    pd.DataFrame(_flatten_model_results(continuous_results)).to_csv(
        os.path.join(table_dir, "morphology_dominance_continuous_models.csv"), index=False
    )
    pd.DataFrame(_flatten_interactions(interaction_results)).to_csv(
        os.path.join(table_dir, "morphology_dominance_interactions.csv"), index=False
    )
    groups.to_csv(os.path.join(table_dir, "morphology_dominance_group_variables.csv"), index=False)

    result = {
        "status": "ok",
        "samples": SAMPLES,
        "subsets": SUBSETS,
        "predictors": PREDICTORS,
        "dominance_audit": audit,
        "n_galaxies": int(len(frame)),
        "n_groups": int(len(groups)),
        "class_association": class_results,
        "continuous_association": continuous_results,
        "cg_rg_interactions": interaction_results,
        "summary": {**summary, **sentences},
        **sentences,
        "output_tables": {
            "class_counts": "morphology_dominance_class_counts.csv",
            "continuous_models": "morphology_dominance_continuous_models.csv",
            "interactions": "morphology_dominance_interactions.csv",
            "group_variables": "morphology_dominance_group_variables.csv",
        },
    }
    return safe_json(result)
