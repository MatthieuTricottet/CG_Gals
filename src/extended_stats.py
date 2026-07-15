"""Shared statistical helpers for the extended specialness analyses."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
except ModuleNotFoundError:  # pragma: no cover - documented fallback
    sm = None
    smf = None
    PerfectSeparationWarning = Warning


DEFAULT_SEED = 20260612


def safe_float(value):
    """Return a finite Python float, or None."""

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def safe_int(value):
    """Return a Python integer, or None."""

    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def safe_json(value):
    """Recursively convert NumPy/pandas values to strict JSON-compatible values."""

    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return [safe_json(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return safe_float(value)
    if value is pd.NA or value is None:
        return None
    return value


def holm_correction(p_values: Iterable[float | None]) -> list[float | None]:
    """Holm-adjust a family of p-values while preserving missing entries."""

    values = list(p_values)
    valid = [
        (index, float(value))
        for index, value in enumerate(values)
        if safe_float(value) is not None
    ]
    adjusted: list[float | None] = [None] * len(values)
    if not valid:
        return adjusted
    ordered = sorted(valid, key=lambda item: item[1])
    running = 0.0
    m = len(ordered)
    for rank, (index, p_value) in enumerate(ordered):
        running = max(running, (m - rank) * p_value)
        adjusted[index] = min(1.0, running)
    return adjusted


def benjamini_hochberg(p_values: Iterable[float | None]) -> list[float | None]:
    """Benjamini-Hochberg FDR-adjust a family of p-values, preserving gaps.

    Returns the step-up adjusted p-values (a.k.a. BH q-values) with the usual
    monotonicity enforced. Missing entries (``None``/non-finite) are passed
    through as ``None`` and excluded from the family size.
    """

    values = list(p_values)
    valid = [
        (index, float(value))
        for index, value in enumerate(values)
        if safe_float(value) is not None
    ]
    adjusted: list[float | None] = [None] * len(values)
    if not valid:
        return adjusted
    ordered = sorted(valid, key=lambda item: item[1])
    m = len(ordered)
    running = 1.0
    # Walk from the largest p-value down, keeping a running minimum of
    # m / rank * p so the adjusted sequence stays monotone non-decreasing.
    for rank in range(m, 0, -1):
        index, p_value = ordered[rank - 1]
        running = min(running, p_value * m / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def magnitude_gap(magnitudes) -> float:
    """Return the magnitude gap ``Delta m12 = M_r,2 - M_r,1`` (brightest first).

    Magnitudes are sorted ascending (brightest = most negative first) and the
    difference between the second-brightest and brightest is returned. ``nan``
    is returned when fewer than two finite magnitudes are available. This is the
    canonical definition shared by the fossilness and morphology-dominance
    analyses.
    """

    values = (
        pd.to_numeric(pd.Series(magnitudes), errors="coerce")
        .dropna()
        .sort_values()
        .to_numpy()
    )
    if values.size < 2:
        return float("nan")
    return float(values[1] - values[0])


def standardized_mean_difference(treated, control) -> float | None:
    """Return the pooled-standard-deviation mean difference."""

    treated = pd.to_numeric(pd.Series(treated), errors="coerce").dropna().to_numpy()
    control = pd.to_numeric(pd.Series(control), errors="coerce").dropna().to_numpy()
    if treated.size < 2 or control.size < 2:
        return None
    pooled = math.sqrt((np.var(treated, ddof=1) + np.var(control, ddof=1)) / 2)
    if not math.isfinite(pooled) or pooled == 0:
        return 0.0 if np.isclose(np.mean(treated), np.mean(control)) else None
    return float((np.mean(treated) - np.mean(control)) / pooled)


def cliffs_delta(x, y) -> float | None:
    """Return Cliff's delta, positive when x tends to exceed y."""

    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy()
    if x.size == 0 or y.size == 0:
        return None
    u_stat = stats.mannwhitneyu(
        x, y, alternative="two-sided", method="asymptotic"
    ).statistic
    return float(2 * u_stat / (x.size * y.size) - 1)


def empirical_p_two_sided(boot: np.ndarray) -> float:
    """Two-sided add-one empirical p-value for a bootstrap null-crossing.

    Uses p = 2 * min[(k_le + 1)/(B + 1), (k_ge + 1)/(B + 1)], capped at 1,
    where k_le/k_ge count draws at or across zero. The +1 terms enforce the
    Monte-Carlo floor: with B draws the smallest reportable value is
    2/(B + 1), never 0.
    """

    n_draws = boot.size
    k_le = int(np.sum(boot <= 0))
    k_ge = int(np.sum(boot >= 0))
    return float(
        min(1.0, 2 * min((k_le + 1) / (n_draws + 1), (k_ge + 1) / (n_draws + 1)))
    )


def bootstrap_difference(
    treated,
    control,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    paired: bool = False,
    n_boot: int = 9999,
    seed: int = DEFAULT_SEED,
    blocks=None,
) -> dict[str, object]:
    """Bootstrap a treated-minus-control effect with an add-one p-value.

    ``blocks`` (paired mode only) assigns each pair to a resampling block —
    e.g. the treated galaxy's physical group — and the bootstrap then
    resamples *blocks* rather than pairs, respecting the dependence of the
    four galaxies of one compact group. The reported p is the two-sided
    add-one empirical p-value (floor 2/(B+1)); the Monte-Carlo resolution is
    reported alongside so callers can never quote a value below it.
    """

    x = pd.to_numeric(pd.Series(treated), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(control), errors="coerce").to_numpy(dtype=float)
    block_ids = None
    if blocks is not None:
        block_ids = pd.Series(blocks).to_numpy()
        if block_ids.size != x.size:
            raise ValueError("blocks must align with the treated values")
    if paired:
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if block_ids is not None:
            block_ids = block_ids[mask]
    else:
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0 or (paired and x.size != y.size):
        return {"estimate": None, "ci95": [None, None], "p": None, "n": 0}

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    if paired and block_ids is not None:
        unique_blocks = pd.unique(block_ids)
        members = {
            block: np.flatnonzero(block_ids == block) for block in unique_blocks
        }
        n_blocks = len(unique_blocks)
        for index in range(n_boot):
            drawn = rng.integers(0, n_blocks, n_blocks)
            rows = np.concatenate([members[unique_blocks[j]] for j in drawn])
            boot[index] = statistic(x[rows]) - statistic(y[rows])
    elif paired:
        for index in range(n_boot):
            draw = rng.integers(0, x.size, x.size)
            boot[index] = statistic(x[draw]) - statistic(y[draw])
    else:
        for index in range(n_boot):
            xb = x[rng.integers(0, x.size, x.size)]
            yb = y[rng.integers(0, y.size, y.size)]
            boot[index] = statistic(xb) - statistic(yb)
    estimate = float(statistic(x) - statistic(y))
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci95": [float(low), float(high)],
        "p": empirical_p_two_sided(boot),
        "n": int(x.size if paired else min(x.size, y.size)),
        "n_boot": int(n_boot),
        "p_floor": float(2 / (n_boot + 1)),
        "resampling_unit": (
            "block" if (paired and block_ids is not None) else
            ("pair" if paired else "observation")
        ),
        "n_blocks": int(len(pd.unique(block_ids)))
        if (paired and block_ids is not None)
        else None,
    }


def two_sample_summary(
    x, y, *, n_boot: int = 2000, seed: int = DEFAULT_SEED
) -> dict[str, object]:
    """Summarize a continuous CG4-minus-control comparison."""

    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0 or y.size == 0:
        return {"status": "skipped", "reason": "no_complete_cases"}
    effect = bootstrap_difference(x, y, statistic=np.median, n_boot=n_boot, seed=seed)
    return {
        "status": "ok",
        "n_cg4": int(x.size),
        "n_control": int(y.size),
        "median_cg4": float(np.median(x)),
        "median_control": float(np.median(y)),
        "delta_median": effect["estimate"],
        "ci95": effect["ci95"],
        "mannwhitney_p": float(
            stats.mannwhitneyu(
                x, y, alternative="two-sided", method="asymptotic"
            ).pvalue
        ),
        "cliffs_delta": cliffs_delta(x, y),
    }


def _skipped(reason: str, **extra) -> dict[str, object]:
    return {"status": "skipped", "reason": reason, **extra}


def fit_ols_with_optional_cluster_se(
    formula: str,
    data: pd.DataFrame,
    group_col: str | None = "cluster_id",
    min_groups: int = 8,
    on_error: Callable[[str], None] | None = None,
):
    """Fit OLS and use cluster-robust errors when enough groups are present.

    Shared implementation used by the colour-robustness exploration and the
    galaxy-size analysis. Returns the fitted statsmodels result, or ``None``
    when the fit fails (the failure message goes to ``on_error`` if given).
    """

    if smf is None:  # pragma: no cover - documented fallback
        if on_error is not None:
            on_error(f"Skipping model '{formula}': statsmodels unavailable")
        return None
    try:
        model = smf.ols(formula, data=data, missing="drop")
        row_labels = model.data.row_labels
        if group_col and group_col in data.columns:
            groups = data.loc[row_labels, group_col]
            valid_groups = groups.dropna().nunique()
            if valid_groups >= min_groups and groups.notna().all():
                return model.fit(cov_type="cluster", cov_kwds={"groups": groups})
        return model.fit(cov_type="HC3")
    except Exception as error:
        if on_error is not None:
            on_error(f"Skipping model '{formula}': {error}")
        return None


def fit_logistic_model(
    frame: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    *,
    continuous: Iterable[str] = (),
    cluster_col: str | None = "physical_group",
    min_n: int = 30,
    min_class: int = 5,
) -> dict[str, object]:
    """Fit a guarded binomial GLM with cluster-robust standard errors.

    Clustering defaults to the *physical* group key so that the same Lim
    group appearing under several control labels forms a single cluster
    (never one pseudo-cluster per label).
    """

    if sm is None:
        return _skipped("statsmodels_unavailable")
    required = [outcome, *predictors]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return _skipped("missing_required_columns", missing_columns=missing)

    columns = list(
        dict.fromkeys(
            required + ([cluster_col] if cluster_col in frame.columns else [])
        )
    )
    work = frame[columns].replace([np.inf, -np.inf], np.nan).copy()
    for column in required:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=required)
    if len(work) < min_n:
        return _skipped("too_few_complete_cases", n=int(len(work)))
    counts = work[outcome].value_counts()
    if set(counts.index) - {0, 1}:
        return _skipped("outcome_not_binary")
    if len(counts) != 2 or int(counts.min()) < min_class:
        return _skipped(
            "too_few_outcome_cases",
            n=int(len(work)),
            outcome_counts={str(key): int(value) for key, value in counts.items()},
        )

    used_predictors = []
    standardized = []
    for predictor in predictors:
        if work[predictor].nunique(dropna=True) < 2:
            continue
        if predictor in set(continuous):
            std = float(work[predictor].std(ddof=0))
            if not math.isfinite(std) or std == 0:
                continue
            work[predictor] = (work[predictor] - float(work[predictor].mean())) / std
            standardized.append(predictor)
        used_predictors.append(predictor)
    if not used_predictors:
        return _skipped("no_variable_predictors", n=int(len(work)))

    design = sm.add_constant(work[used_predictors].astype(float), has_constant="add")
    groups = work[cluster_col] if cluster_col and cluster_col in work.columns else None
    covariance = "HC1"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            model = sm.GLM(
                work[outcome].astype(float), design, family=sm.families.Binomial()
            )
            if groups is not None and groups.nunique() >= 2:
                fitted = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
                covariance = "cluster"
            else:
                fitted = model.fit(cov_type="HC1")
    except Exception as exc:
        return _skipped("model_fit_failed", n=int(len(work)), error=str(exc))

    params = fitted.params
    errors = fitted.bse
    p_values = fitted.pvalues
    confidence = fitted.conf_int()
    if (
        not np.isfinite(np.asarray(params)).all()
        or np.max(np.abs(np.asarray(params))) > 20
    ):
        return _skipped("perfect_or_quasi_separation", n=int(len(work)))

    terms = {}
    for term in params.index:
        coefficient = float(params[term])
        low, high = map(float, confidence.loc[term])
        terms[term] = {
            "coefficient": coefficient,
            "standard_error": float(errors[term]),
            "odds_ratio": float(np.exp(coefficient)),
            "ci95": [float(np.exp(low)), float(np.exp(high))],
            "p": float(p_values[term]),
        }
    bic_value = getattr(fitted, "bic_llf", None)
    if bic_value is None:
        bic_value = getattr(fitted, "bic", None)
    result = {
        "status": "ok",
        "n": int(fitted.nobs),
        "formula": f"{outcome} ~ " + " + ".join(used_predictors),
        "predictors_used": used_predictors,
        "standardized_predictors": standardized,
        "covariance": covariance,
        "n_clusters": int(groups.nunique()) if groups is not None else None,
        "aic": safe_float(getattr(fitted, "aic", None)),
        "bic": safe_float(bic_value),
        "terms": terms,
    }
    if "is_CG4" in terms:
        cg = terms["is_CG4"]
        result.update(
            {
                "cg4_coefficient": cg["coefficient"],
                "cg4_standard_error": cg["standard_error"],
                "cg4_odds_ratio": cg["odds_ratio"],
                "cg4_ci95": cg["ci95"],
                "cg4_p": cg["p"],
            }
        )
    return result
