"""Rank-interaction inference for the Fig. 4 properties (bgg-split round).

Does the CG4 offset in the Fig. 4 properties differ between BGGs and
satellites?  Every choice below was fixed before any model was fitted
(results/diagnostics/bgg_split/PHASE0_AUDIT.md, decision D1):

* Support: log M* >= 10.5, where both ranks exist in every sample.
* Primary outcomes (one Holm family per model): strong H-alpha (logit),
  AGN-like BPT class (logit), D_n4000 (OLS), Simard log R_e (OLS).
  Sensitivity outcomes, outside the families: Petrosian log R50 and the
  Simard radius including Sersic-pegged fits.
* Mass adjustment: a natural cubic regression spline in log M*,
  ``cr(logMstar, df=3, constraints="center")``.  The unconstrained basis
  spans the intercept, so the centred one is used; it is built once per
  model family on the pooled four-sample frame and reused unchanged in
  every per-control fit and bootstrap draw.  Each basis column is scaled to
  unit pooled standard deviation: this reparametrisation leaves the fitted
  model and every CG term unchanged, and keeps the spline coefficients on
  the scale assumed by the |beta| > 20 separation guard of
  ``fit_logistic_model``.
* One fit per control, on CG4 plus that control's rows; the controls are
  never pooled.

Model I:  y ~ spline + CG + BGG + CG:BGG             (beta_CG:BGG)
Model R:  y ~ spline + BGG + CG_sat + CG_bgg         (same fit, rank-specific CG effects)
Model S:  y ~ spline + CG, satellites at all masses

Standard errors are cluster-robust on ``physical_group``.  For Model R a
2000-draw bootstrap resamples physical groups within each sample and gives
percentile 95% intervals for CG_sat, CG_bgg and their difference.  Holm is
applied across the 12 Model I interaction tests and, separately, across the
12 Model S tests; both families are secondary diagnostics.  Minimum
detectable effects use 2.8 standard errors (80% power, two-sided 0.05).
"""

from __future__ import annotations

import math
import os
import warnings
import zlib

import numpy as np
import pandas as pd

try:
    import config as co
    from descriptive_properties import _prepare as prepare_properties
    from descriptive_trends import MASS_BINS
    from extended_data import ensure_galaxy_frame
    from extended_stats import (
        DEFAULT_SEED,
        fit_logistic_model,
        fit_ols_with_optional_cluster_se,
        holm_correction,
        safe_json,
    )
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from .descriptive_properties import _prepare as prepare_properties
    from .descriptive_trends import MASS_BINS
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import (
        DEFAULT_SEED,
        fit_logistic_model,
        fit_ols_with_optional_cluster_se,
        holm_correction,
        safe_json,
    )

import patsy
import statsmodels.api as sm

CONTROLS = ("Control4B", "Control4C", "RG4")
SUPPORT_MIN_LOGM = 10.5
PRIMARY_OUTCOMES = (
    ("strong_halpha", "logit"),
    ("agn_like", "logit"),
    ("Dn4000", "ols"),
    ("log_Rchl_r_kpc", "ols"),
)
SENSITIVITY_OUTCOMES = (
    ("log_petroR50_kpc", "ols"),
    ("log_Rchl_r_kpc_incl_pegged", "ols"),
)
SPLINE_DF = 3
SPLINE_TERM = f"cr(x, df={SPLINE_DF}, constraints='center') - 1"
CLUSTER = "physical_group"
N_BOOT = 2000
MDE_FACTOR = 2.8  # z_0.975 + z_0.80
SEPARATION_LIMIT = 20.0  # same guard as extended_stats.fit_logistic_model
IDENTITY_TOLERANCE = 1e-6
CLASS_RULES = {
    "SAT": "CG_sat CI excludes 0, CG_bgg CI includes 0, interaction Holm p < 0.05",
    "SAT?": "CG_sat CI excludes 0, CG_bgg CI includes 0, interaction not significant",
    "HOST": "both exclude 0 with the same sign, interaction not significant",
    "OPP": "both exclude 0 with opposite signs",
    "BGG": "CG_bgg CI excludes 0, CG_sat CI includes 0",
    "NULL": "neither excludes 0",
}


# --------------------------------------------------------------------------
# data preparation
# --------------------------------------------------------------------------

def spline_basis(mass: pd.Series, prefix: str) -> pd.DataFrame:
    """Centred natural cubic regression spline basis evaluated on ``mass``."""

    design = patsy.dmatrix(
        SPLINE_TERM, {"x": mass.to_numpy(dtype=float)}, return_type="dataframe"
    )
    design.columns = [f"{prefix}{index}" for index in range(design.shape[1])]
    design.index = mass.index
    return (design - design.mean()) / design.std(ddof=0)


def prepare_frame(galaxies: pd.DataFrame) -> pd.DataFrame:
    """Outcome columns as in Fig. 4 plus the rank/sample indicators."""

    work = prepare_properties(galaxies).reset_index(drop=True)
    work["logMstar"] = pd.to_numeric(work["logMstar"], errors="coerce")
    work["is_CG4"] = (work["sample"] == "CG4").astype(float)
    work["is_bgg"] = np.where(work["rank"].notna(), work["rank"].eq(1).astype(float), np.nan)
    work["cg_x_bgg"] = work["is_CG4"] * work["is_bgg"]
    work["cg_sat"] = work["is_CG4"] * (1.0 - work["is_bgg"])
    work["cg_bgg"] = work["is_CG4"] * work["is_bgg"]
    for column, _kind in PRIMARY_OUTCOMES + SENSITIVITY_OUTCOMES:
        if column not in work:
            work[column] = np.nan
    return work


def attach_family_basis(work: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return the Model I/R frame and the Model S frame with their bases."""

    finite = np.isfinite(work["logMstar"])
    support = work.loc[finite & (work["logMstar"] >= SUPPORT_MIN_LOGM)].copy()
    support = support.join(spline_basis(support["logMstar"], "ms"))
    satellites = work.loc[finite & work["is_bgg"].eq(0)].copy()
    satellites = satellites.join(spline_basis(satellites["logMstar"], "ss"))
    info = {
        "spline": SPLINE_TERM.replace("x", "logMstar"),
        "basis_scaling": "each column standardised on the pooled rows (reparametrisation only)",
        "basis_built_on": {
            "models_I_R": f"pooled rows of the four samples with log M* >= {SUPPORT_MIN_LOGM}",
            "model_S": "pooled satellite rows of the four samples, all masses",
        },
        "n_basis_rows": {"models_I_R": int(len(support)), "model_S": int(len(satellites))},
    }
    return support, satellites, info


def _columns(prefix: str, frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]


# --------------------------------------------------------------------------
# analytic fits
# --------------------------------------------------------------------------

def _complete(frame: pd.DataFrame, outcome: str, predictors: list[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys([outcome, *predictors, CLUSTER, "sample", "is_bgg"]))
    data = frame[columns].replace([np.inf, -np.inf], np.nan)
    return data.dropna(subset=[outcome, *predictors, CLUSTER]).reset_index(drop=True)


def _counts(data: pd.DataFrame, outcome: str, kind: str) -> dict:
    cg = data["sample"] == "CG4"
    out = {
        "n": int(len(data)),
        "n_cg4_sat": int((cg & data["is_bgg"].eq(0)).sum()),
        "n_cg4_bgg": int((cg & data["is_bgg"].eq(1)).sum()),
        "n_control_sat": int((~cg & data["is_bgg"].eq(0)).sum()),
        "n_control_bgg": int((~cg & data["is_bgg"].eq(1)).sum()),
        "n_clusters": int(data[CLUSTER].nunique()),
        "n_cg4_clusters": int(data.loc[cg, CLUSTER].nunique()),
    }
    shared = set(data.loc[cg, CLUSTER]) & set(data.loc[~cg, CLUSTER])
    out["n_clusters_shared_cg4_control"] = len(shared)
    if kind == "logit":
        out["n_events_cg4_sat"] = int(data.loc[cg & data["is_bgg"].eq(0), outcome].sum())
        out["n_events_cg4_bgg"] = int(data.loc[cg & data["is_bgg"].eq(1), outcome].sum())
    return out


def _term(estimate, se, low, high, p) -> dict:
    return {
        "estimate": float(estimate),
        "se": float(se),
        "ci95": [float(low), float(high)],
        "p": float(p),
    }


def fit_terms(data: pd.DataFrame, outcome: str, kind: str, predictors: list[str],
              terms: list[str]) -> dict:
    """Fit one model on complete cases; return the requested terms."""

    if kind == "logit":
        fit = fit_logistic_model(data, outcome, predictors, continuous=(), cluster_col=CLUSTER)
        if fit.get("status") != "ok":
            return {"status": "skipped", "reason": fit.get("reason"), "detail": fit}
        if fit["predictors_used"] != predictors:
            return {"status": "skipped", "reason": "predictor_dropped",
                    "predictors_used": fit["predictors_used"]}
        out = {"status": "ok", "covariance": fit["covariance"], "n_clusters": fit["n_clusters"]}
        for term in terms:
            t = fit["terms"][term]
            low, high = (math.log(v) for v in t["ci95"])
            out[term] = _term(t["coefficient"], t["standard_error"], low, high, t["p"])
        return out

    formula = f"{outcome} ~ " + " + ".join(predictors)
    messages: list[str] = []
    result = fit_ols_with_optional_cluster_se(
        formula, data, group_col=CLUSTER, on_error=messages.append
    )
    if result is None:
        return {"status": "skipped", "reason": "model_fit_failed", "error": messages}
    out = {"status": "ok", "covariance": result.cov_type,
           "n_clusters": int(data[CLUSTER].nunique()), "use_t": bool(result.use_t)}
    confidence = result.conf_int()
    for term in terms:
        low, high = confidence.loc[term]
        out[term] = _term(result.params[term], result.bse[term], low, high, result.pvalues[term])
    return out


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------

def _fit_params(y: np.ndarray, X: np.ndarray, kind: str, start=None):
    if kind == "ols":
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta
    if y.min() == y.max():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = sm.GLM(y, X, family=sm.families.Binomial()).fit(start_params=start)
        except Exception:
            return None
    beta = np.asarray(result.params)
    if not np.isfinite(beta).all() or np.abs(beta).max() > SEPARATION_LIMIT:
        return None
    return beta


def bootstrap_model_r(data: pd.DataFrame, outcome: str, kind: str, predictors: list[str],
                      *, n_boot: int = N_BOOT, seed: int = DEFAULT_SEED) -> dict:
    """Physical-group bootstrap, stratified by sample, of the Model R terms."""

    X = np.column_stack([np.ones(len(data)), data[predictors].to_numpy(dtype=float)])
    y = data[outcome].to_numpy(dtype=float)
    i_sat = 1 + predictors.index("cg_sat")
    i_bgg = 1 + predictors.index("cg_bgg")
    start = _fit_params(y, X, kind)
    strata = []
    for _sample, part in data.groupby("sample", observed=True):
        members = [idx.to_numpy() for _key, idx in part.groupby(CLUSTER, observed=True).groups.items()]
        strata.append(members)
    rng = np.random.default_rng(seed)
    draws = np.full((n_boot, 3), np.nan)
    for b in range(n_boot):
        rows = np.concatenate([
            np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
            for members in strata
        ])
        beta = _fit_params(y[rows], X[rows], kind, start=start)
        if beta is None:
            continue
        draws[b] = (beta[i_sat], beta[i_bgg], beta[i_bgg] - beta[i_sat])
    ok = np.isfinite(draws).all(axis=1)
    out = {"n_boot": int(n_boot), "n_failed": int((~ok).sum()), "seed": int(seed),
           "resampling": "physical groups with replacement, within each sample"}
    for index, name in enumerate(("cg_sat", "cg_bgg", "interaction")):
        values = draws[ok, index]
        out[name] = {
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            if values.size else [None, None],
            "sd": float(values.std(ddof=1)) if values.size > 1 else None,
        }
    return out


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def _excludes_zero(term: dict) -> bool:
    low, high = term["ci95"]
    return low > 0 or high < 0


def classify(cg_sat: dict, cg_bgg: dict, interaction_holm_p: float | None) -> str:
    """Mechanical outcome code (see CLASS_RULES)."""

    sat, bgg = _excludes_zero(cg_sat), _excludes_zero(cg_bgg)
    significant = interaction_holm_p is not None and interaction_holm_p < 0.05
    if sat and bgg:
        if np.sign(cg_sat["estimate"]) != np.sign(cg_bgg["estimate"]):
            return "OPP"
        return "UNMATCHED" if significant else "HOST"
    if sat:
        return "SAT" if significant else "SAT?"
    if bgg:
        return "BGG"
    return "NULL"


def _mde(term: dict, kind: str) -> dict:
    mde = MDE_FACTOR * term["se"]
    if kind == "logit":
        return {"log_odds": mde, "odds_ratio_high": math.exp(mde), "odds_ratio_low": math.exp(-mde)}
    return {"delta": mde}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _seed(*parts) -> int:
    return (DEFAULT_SEED + zlib.crc32("|".join(map(str, parts)).encode())) % (2**32)


def fit_interaction_models(support: pd.DataFrame, outcome: str, kind: str, control: str,
                           *, n_boot: int = N_BOOT, extra: tuple[str, ...] = ()) -> dict:
    """Models I and R (and the Model R bootstrap) for one outcome and control."""

    spline = _columns("ms", support)
    part = support.loc[support["sample"].isin(["CG4", control])]
    predictors_i = [*spline, *extra, "is_CG4", "is_bgg", "cg_x_bgg"]
    predictors_r = [*spline, *extra, "is_bgg", "cg_sat", "cg_bgg"]
    data = _complete(part, outcome, list(dict.fromkeys(predictors_i + predictors_r)))
    out = {"outcome": outcome, "kind": kind, "control": control, "counts": _counts(data, outcome, kind)}
    model_i = fit_terms(data, outcome, kind, predictors_i, ["cg_x_bgg"])
    model_r = fit_terms(data, outcome, kind, predictors_r, ["cg_sat", "cg_bgg"])
    if model_i.get("status") != "ok" or model_r.get("status") != "ok":
        out.update({"status": "skipped", "model_I": model_i, "model_R": model_r})
        return out
    identity = (model_r["cg_bgg"]["estimate"] - model_r["cg_sat"]["estimate"]
                - model_i["cg_x_bgg"]["estimate"])
    if abs(identity) > IDENTITY_TOLERANCE:
        raise AssertionError(
            f"Model R does not reparametrise Model I for {outcome}/{control}: {identity:.3g}"
        )
    out.update({
        "status": "ok",
        "covariance": model_r["covariance"],
        "cg_sat": model_r["cg_sat"],
        "cg_bgg": model_r["cg_bgg"],
        "interaction": model_i["cg_x_bgg"],
        "identity_residual": float(identity),
        "mde": {"cg_sat": _mde(model_r["cg_sat"], kind), "cg_bgg": _mde(model_r["cg_bgg"], kind)},
    })
    if n_boot:
        out["bootstrap"] = bootstrap_model_r(
            data, outcome, kind, predictors_r, n_boot=n_boot,
            seed=_seed("rank_split", outcome, control, *extra),
        )
    return out


def fit_satellite_model(satellites: pd.DataFrame, outcome: str, kind: str, control: str) -> dict:
    spline = _columns("ss", satellites)
    part = satellites.loc[satellites["sample"].isin(["CG4", control])]
    predictors = [*spline, "is_CG4"]
    data = _complete(part, outcome, predictors)
    out = {"outcome": outcome, "kind": kind, "control": control,
           "counts": _counts(data, outcome, kind)}
    fit = fit_terms(data, outcome, kind, predictors, ["is_CG4"])
    if fit.get("status") != "ok":
        out.update({"status": "skipped", "fit": fit})
        return out
    out.update({"status": "ok", "covariance": fit["covariance"], "cg": fit["is_CG4"],
                "mde": _mde(fit["is_CG4"], kind)})
    return out


def _apply_holm(entries: list[dict], getter, key: str) -> None:
    p_values = [getter(entry) if entry.get("status") == "ok" else None for entry in entries]
    for entry, adjusted in zip(entries, holm_correction(p_values)):
        entry[key] = adjusted
    n_tests = sum(p is not None for p in p_values)
    for entry in entries:
        entry[f"{key}_family_size"] = n_tests


def _table(entries: list[dict], columns: list[str]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        row = {"outcome": entry["outcome"], "control": entry["control"], "status": entry["status"]}
        row.update({f"n_{k}": v for k, v in entry["counts"].items() if k.startswith("n_")})
        for column in columns:
            if column in entry and isinstance(entry[column], dict):
                row[f"{column}_estimate"] = entry[column].get("estimate")
                row[f"{column}_se"] = entry[column].get("se")
                row[f"{column}_ci_low"], row[f"{column}_ci_high"] = entry[column].get("ci95", [None, None])
                row[f"{column}_p"] = entry[column].get("p")
        for key in ("interaction_p_holm", "cg_p_holm", "class_code"):
            if key in entry:
                row[key] = entry[key]
        boot = entry.get("bootstrap", {})
        for name in ("cg_sat", "cg_bgg", "interaction"):
            if name in boot:
                row[f"boot_{name}_ci_low"], row[f"boot_{name}_ci_high"] = boot[name]["ci95"]
        rows.append(row)
    return pd.DataFrame(rows)


def run_rank_split_properties(galaxies, output_dir: str | None = None, *,
                              n_boot: int = N_BOOT) -> dict:
    frame = ensure_galaxy_frame(galaxies)
    if frame.empty:
        return {"status": "skipped", "reason": "no_galaxy_frame"}
    work = prepare_frame(frame)
    support, satellites, spline_info = attach_family_basis(work)

    cg_bgg_bins = (
        pd.cut(support.loc[(support["sample"] == "CG4") & support["is_bgg"].eq(1), "logMstar"],
               MASS_BINS[MASS_BINS >= SUPPORT_MIN_LOGM], right=False)
        .value_counts().sort_index()
    )
    if (cg_bgg_bins < 10).any():
        return {"status": "skipped", "reason": "fewer_than_10_cg4_bggs_in_an_upper_bin",
                "cg4_bgg_counts": {str(k): int(v) for k, v in cg_bgg_bins.items()}}

    primary = [
        fit_interaction_models(support, outcome, kind, control, n_boot=n_boot)
        for outcome, kind in PRIMARY_OUTCOMES for control in CONTROLS
    ]
    _apply_holm(primary, lambda e: e["interaction"]["p"], "interaction_p_holm")
    for entry in primary:
        if entry["status"] == "ok":
            entry["class_code"] = classify(entry["cg_sat"], entry["cg_bgg"], entry["interaction_p_holm"])

    satellite = [
        fit_satellite_model(satellites, outcome, kind, control)
        for outcome, kind in PRIMARY_OUTCOMES for control in CONTROLS
    ]
    _apply_holm(satellite, lambda e: e["cg"]["p"], "cg_p_holm")

    sensitivity = [
        fit_interaction_models(support, outcome, kind, control, n_boot=0)
        for outcome, kind in SENSITIVITY_OUTCOMES for control in CONTROLS
    ]
    sensitivity_satellite = [
        fit_satellite_model(satellites, outcome, kind, control)
        for outcome, kind in SENSITIVITY_OUTCOMES for control in CONTROLS
    ]
    for entry in sensitivity:
        if entry["status"] == "ok":
            entry["class_code_unadjusted"] = classify(entry["cg_sat"], entry["cg_bgg"], entry["interaction"]["p"])

    result = {
        "status": "ok",
        "support_min_logmstar": SUPPORT_MIN_LOGM,
        "cg4_bgg_counts_upper_bins": {str(k): int(v) for k, v in cg_bgg_bins.items()},
        "controls": list(CONTROLS),
        "primary_outcomes": [o for o, _ in PRIMARY_OUTCOMES],
        "sensitivity_outcomes": [o for o, _ in SENSITIVITY_OUTCOMES],
        "mass_adjustment": spline_info,
        "cluster": CLUSTER,
        "effect_scale": {"logit": "log odds ratio", "ols": "difference in outcome units"},
        "mde_factor_se": MDE_FACTOR,
        "holm_families": {
            "model_I_interaction": "12 tests (4 outcomes x 3 controls), secondary diagnostics",
            "model_S_cg": "12 tests (4 outcomes x 3 controls), secondary diagnostics",
        },
        "class_rules": CLASS_RULES,
        "class_rules_ci": "analytic cluster-robust 95% intervals of Model R",
        "interaction_models": primary,
        "satellite_models": satellite,
        "sensitivity_interaction_models": sensitivity,
        "sensitivity_satellite_models": sensitivity_satellite,
    }
    if output_dir:
        tables = pd.concat(
            [
                _table(primary, ["cg_sat", "cg_bgg", "interaction"]).assign(model="I_R"),
                _table(satellite, ["cg"]).assign(model="S"),
                _table(sensitivity, ["cg_sat", "cg_bgg", "interaction"]).assign(model="I_R_sensitivity"),
                _table(sensitivity_satellite, ["cg"]).assign(model="S_sensitivity"),
            ],
            ignore_index=True,
        )
        tables.to_csv(os.path.join(co.OUTPUT_PATH, "rank_split_properties.csv"), index=False)
        result["table_file"] = "rank_split_properties.csv"
    return safe_json(result)
