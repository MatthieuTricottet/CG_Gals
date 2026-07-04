"""Luminosity-function analysis for BGG and satellite galaxy samples.

Histograms in this module are visualization products only.  Schechter
parameters are estimated with an unbinned, truncated maximum likelihood so
the fitted values do not depend on histogram bin choices.

Truncation is handled PER OBJECT, following the sample construction:

* ``CG4`` / ``RG4`` -- the accordance criterion: every member lies within
  3 magnitudes of the brightest group galaxy, so satellites of group ``g``
  are truncated at ``M_BGG(g) + 3``.  Because group inclusion requires *all*
  satellites to pass this cut, the conditional likelihood factorises exactly
  into per-galaxy truncated densities (STY; Sandage, Tammann & Yahil 1979).
* ``Control4C`` -- companions are selected by projected distance regardless
  of luminosity, so the truncation is the spectroscopic flux limit at each
  galaxy's own distance, ``M_lim,i = M_r,i + (r_lim - r_i)``, taken from the
  parent-catalogue apparent magnitudes (``PC_Gals.csv``).
* Fallback -- when the required information is unavailable (unit tests,
  incomplete catalogues) the code falls back on a single common floor at the
  faintest observed galaxy of the sample and records this in the output.

``phi*`` cancels in the per-galaxy normalisation and is never estimated,
consistent with the science goal (``alpha`` and ``L*`` only).

BGGs are NOT fitted with a Schechter function.  A BGG magnitude is the
maximum of ~4 draws and the sample design truncates it at
``M_r,BGG <= -21.81``; the resulting distribution is peaked, and Schechter
fits rail against parameter bounds.  BGGs are summarised instead with a
Gaussian in ``log10 L`` truncated below at the design cut.

``Control4B`` is excluded: its faint cut-off is the magnitude of the parent
group's 5th-brightest member, an order statistic depending on the unobserved
richer membership, so its distribution is censored by richness rather than
truncated in luminosity.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence

if os.environ.get("MPLCONFIGDIR") is None:
    mpl_cache = os.path.join("/tmp", "cg_gals_matplotlib")
    os.makedirs(mpl_cache, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_cache
import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaincc, gammaln
from scipy.stats import norm

try:
    import config as co
    from extended_stats import safe_json
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from .extended_stats import safe_json


DEFAULT_SAMPLES = ("CG4", "RG4", "Control4C")
EXCLUDED_SAMPLES = {
    "Control4B": "excluded because selected as four brightest galaxies by construction"
}
REMOVED_COMPONENTS = {
    "all": "BGGs and satellites obey different selections; a joint Schechter fit "
    "mixes two differently truncated populations and is not reported"
}

# Solar absolute magnitude in SDSS r.  Kept equal to astro_utils.Mr_Sol so
# that L* values are directly comparable with the Lum/Lum_group catalogue
# columns (which use 4.68).
DEFAULT_M_SUN_R = 4.68
# Sample design cuts (Tricottet+25): members within 3 mag of the BGG, and
# M_r,BGG <= -21.81 for the volume/luminosity-complete selection.
ACCORDANCE_MAG = 3.0
BGG_DESIGN_MAG_CUT = -21.81
MIN_FIT_N = 8

ACCORDANCE_SAMPLES = ("CG4", "RG4")
FLUX_LIMIT_SAMPLES = ("Control4C",)
PARENT_PHOTOMETRY_FILE = "PC_Gals.csv"

COMPONENT_LABELS = {
    "bgg": "BGGs",
    "satellites": "Satellites",
}
FIGURE_NAMES = {
    "bgg": "fig_luminosity_function_bgg.pdf",
    "satellites": "fig_luminosity_function_satellites.pdf",
}
SAMPLE_COLOURS = {
    "CG4": "#2864A6",
    "RG4": "#D07C28",
    "Control4C": "#4D8C57",
}

LN10 = math.log(10.0)


def absolute_magnitude_to_log_luminosity(M_r, M_sun_r=DEFAULT_M_SUN_R):
    """Convert SDSS-r absolute magnitude to log10 solar luminosity.

    The default ``M_sun_r=4.68`` matches ``astro_utils.Mr_Sol`` (and hence the
    ``Lum`` columns of the catalogues); change both together if the paper
    adopts a different convention.
    """

    if isinstance(M_r, pd.Series):
        values = pd.to_numeric(M_r, errors="coerce")
        return -0.4 * (values - M_sun_r)
    if np.isscalar(M_r):
        value = pd.to_numeric(pd.Series([M_r]), errors="coerce").iloc[0]
        return float(-0.4 * (value - M_sun_r))
    values = pd.to_numeric(pd.Series(M_r), errors="coerce")
    return (-0.4 * (values - M_sun_r)).to_numpy(dtype=float)


def log_luminosity_to_absolute_magnitude(logL, M_sun_r=DEFAULT_M_SUN_R):
    """Inverse of :func:`absolute_magnitude_to_log_luminosity`."""

    return M_sun_r - 2.5 * np.asarray(logL, dtype=float)


# ---------------------------------------------------------------------------
# Generalised upper incomplete gamma, Gamma(a, x), any real a and x > 0
# ---------------------------------------------------------------------------
def upper_incomplete_gamma(a: float, x):
    """Return ``Gamma(a, x)`` for any real ``a`` (vectorised in ``x > 0``).

    scipy only covers ``a > 0``; for ``a <= 0`` (i.e. ``alpha <= -1``) the
    downward recurrence ``Gamma(a, x) = (Gamma(a+1, x) - x**a e**-x) / a``
    is applied from a positive starting order.
    """

    x = np.asarray(x, dtype=float)
    a = float(a)
    if a <= 0 and abs(a - round(a)) < 1e-9:  # poles at 0, -1, -2, ...
        a = a + 1e-9
    if a > 0:
        return np.exp(gammaln(a)) * gammaincc(a, x)
    k = int(np.ceil(-a)) + 1  # a + k lies in (1, 2]
    aa = a + k
    result = np.exp(gammaln(aa)) * gammaincc(aa, x)
    log_x = np.log(x)
    for _ in range(k):
        aa -= 1.0
        result = (result - np.exp(aa * log_x - x)) / aa
    return result


def _log_norm_per_object(alpha: float, logL_star: float, logL_lim: np.ndarray):
    """Return ``log Gamma(alpha+1, x_lim,i)`` per object, or ``None`` if invalid.

    The unique truncation limits are evaluated once and broadcast back, which
    keeps the bootstrap affordable for the larger samples.
    """

    unique_lims, inverse = np.unique(logL_lim, return_inverse=True)
    x_lim = 10.0 ** (unique_lims - logL_star)
    gamma_values = upper_incomplete_gamma(alpha + 1.0, x_lim)
    if (not np.all(np.isfinite(gamma_values))) or np.any(gamma_values <= 0.0):
        return None
    return np.log(gamma_values)[inverse]


def schechter_logpdf_logL(logL, alpha, logL_star, logL_lim):
    """Normalized truncated Schechter log-density in ``log10 L``.

    ``logL_lim`` may be a scalar (common floor) or an array of per-object
    lower truncation limits aligned with ``logL``.
    """

    scalar = np.isscalar(logL)
    values = _as_float_array([logL] if scalar else logL)
    lims = np.broadcast_to(
        _as_float_array(logL_lim), values.shape
    ).astype(float, copy=True)
    output = np.full(values.shape, -np.inf, dtype=float)
    if not all(
        math.isfinite(float(v)) for v in (alpha, logL_star)
    ) or not np.all(np.isfinite(lims)):
        return float(output[0]) if scalar else output

    log_norm = _log_norm_per_object(float(alpha), float(logL_star), lims)
    if log_norm is None:
        return float(output[0]) if scalar else output

    valid = np.isfinite(values) & (values >= lims)
    if np.any(valid):
        log_x = LN10 * (values[valid] - float(logL_star))
        with np.errstate(over="ignore", invalid="ignore"):
            x = np.exp(log_x)
        output[valid] = (
            math.log(LN10)
            + (float(alpha) + 1.0) * log_x
            - x
            - log_norm[valid]
        )
    return float(output[0]) if scalar else output


def negative_log_likelihood(params, logL, logL_lim):
    """Schechter negative log likelihood with per-object lower truncation."""

    try:
        alpha, logL_star = float(params[0]), float(params[1])
    except (TypeError, ValueError, IndexError):
        return float("inf")
    if not all(math.isfinite(value) for value in (alpha, logL_star)):
        return float("inf")

    values = _as_float_array(logL)
    lims = np.broadcast_to(_as_float_array(logL_lim), values.shape)
    keep = np.isfinite(values) & np.isfinite(lims)
    values, lims = values[keep], lims[keep]
    if values.size == 0 or np.any(values < lims):
        return float("inf")

    log_norm = _log_norm_per_object(alpha, logL_star, lims)
    if log_norm is None:
        return float("inf")
    log_x = LN10 * (values - logL_star)
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.exp(log_x)
    log_pdf = math.log(LN10) + (alpha + 1.0) * log_x - x - log_norm
    total = -np.sum(log_pdf)
    return float(total) if math.isfinite(float(total)) else float("inf")


def _as_float_array(values) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def _normalise_bounds(bounds, logL: np.ndarray):
    if bounds is None:
        low = float(np.min(logL)) - 0.5 if logL.size else 8.0
        high = float(np.max(logL)) + 1.5 if logL.size else 12.0
        return ((-3.0, 1.5), (low, high))
    if isinstance(bounds, Mapping):
        default_star = (
            float(np.min(logL)) - 0.5 if logL.size else 8.0,
            float(np.max(logL)) + 1.5 if logL.size else 12.0,
        )
        return (
            tuple(bounds.get("alpha", (-3.0, 1.5))),
            tuple(bounds.get("logL_star", default_star)),
        )
    return tuple(tuple(item) for item in bounds)


def _empty_fit(status: str, reason: str, **extra) -> dict[str, object]:
    return {
        "status": status,
        "reason": reason,
        "success": False,
        "alpha": None,
        "logL_star": None,
        "L_star": None,
        "n": int(extra.pop("n", 0)),
        "truncation": extra.pop("truncation", None),
        "negative_log_likelihood": None,
        "optimizer_message": extra.pop("optimizer_message", None),
        **extra,
    }


def fit_schechter_mle(
    logL, logL_lim=None, logL_min=None, bounds=None, starts=None
) -> dict[str, object]:
    """Fit a truncated Schechter function by unbinned maximum likelihood.

    ``logL_lim`` gives per-object lower truncation limits.  For backward
    compatibility a scalar ``logL_min`` (common floor) is also accepted; if
    neither is provided the observed minimum is used as a common floor.
    """

    values = _as_float_array(logL)
    finite = np.isfinite(values)

    if logL_lim is not None:
        lims = np.broadcast_to(_as_float_array(logL_lim), values.shape).astype(
            float, copy=True
        )
        truncation = "per_object"
    else:
        if logL_min is None:
            logL_min = float(np.min(values[finite])) if finite.any() else np.nan
        lims = np.full(values.shape, float(logL_min))
        truncation = "common_floor"

    keep = finite & np.isfinite(lims)
    values, lims = values[keep], lims[keep]
    if values.size == 0:
        return _empty_fit("skipped", "no_finite_luminosities", truncation=truncation)
    # A galaxy may sit marginally beyond its nominal limit (magnitude-system
    # mismatches); its own luminosity then bounds the effective limit.
    lims = np.minimum(lims, values - 1e-9)
    if values.size < MIN_FIT_N:
        return _empty_fit(
            "skipped",
            "too_few_galaxies",
            n=int(values.size),
            truncation=truncation,
            min_required=MIN_FIT_N,
        )

    fit_bounds = _normalise_bounds(bounds, values)
    if starts is None:
        starts = [
            (-1.0, np.percentile(values, 75)),
            (-0.5, np.percentile(values, 60)),
            (-1.5, np.percentile(values, 90)),
        ]
    starts = [
        (
            float(np.clip(alpha, fit_bounds[0][0] + 1e-4, fit_bounds[0][1] - 1e-4)),
            float(np.clip(star, fit_bounds[1][0] + 1e-4, fit_bounds[1][1] - 1e-4)),
        )
        for alpha, star in starts
    ]

    best = None
    message = None
    for start in starts:
        try:
            result = minimize(
                negative_log_likelihood,
                start,
                args=(values, lims),
                method="L-BFGS-B",
                bounds=fit_bounds,
                options={"maxiter": 300, "ftol": 1e-10},
            )
        except Exception as exc:  # pragma: no cover - defensive
            result = None
            message = f"{exc.__class__.__name__}: {exc}"
        else:
            message = str(result.message)
        if result is not None and math.isfinite(float(result.fun)):
            if best is None or float(result.fun) < float(best.fun):
                best = result

    if best is None or not math.isfinite(float(best.fun)):
        return _empty_fit(
            "failed",
            "optimizer_failed",
            n=int(values.size),
            truncation=truncation,
            optimizer_message=message or "no finite optimizer result",
        )

    alpha, logL_star = map(float, best.x)
    return {
        "status": "ok" if bool(best.success) else "failed",
        "reason": None if bool(best.success) else "optimizer_failed",
        "success": bool(best.success),
        "alpha": alpha if math.isfinite(alpha) else None,
        "logL_star": logL_star if math.isfinite(logL_star) else None,
        "L_star": float(10**logL_star) if math.isfinite(logL_star) else None,
        "M_star_r": float(log_luminosity_to_absolute_magnitude(logL_star))
        if math.isfinite(logL_star)
        else None,
        "n": int(values.size),
        "truncation": truncation,
        "logL_lim_min": float(np.min(lims)),
        "logL_lim_median": float(np.median(lims)),
        "logL_lim_max": float(np.max(lims)),
        "negative_log_likelihood": float(best.fun),
        "optimizer_message": str(best.message),
    }


# ---------------------------------------------------------------------------
# Truncated Gaussian for the BGGs
# ---------------------------------------------------------------------------
def fit_truncated_gaussian_mle(logL, logL_trunc) -> dict[str, object]:
    """Fit a Gaussian in ``log10 L`` truncated below at ``logL_trunc``.

    This is the model used for BGGs: the sample design keeps only groups with
    ``M_r,BGG <= -21.81``, i.e. ``logL >= logL_trunc``.
    """

    values = _as_float_array(logL)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return _empty_fit("skipped", "no_finite_luminosities", truncation="lower")
    trunc = float(min(float(logL_trunc), float(np.min(values)) - 1e-9))
    if values.size < MIN_FIT_N:
        return _empty_fit(
            "skipped",
            "too_few_galaxies",
            n=int(values.size),
            truncation="lower",
            min_required=MIN_FIT_N,
        )

    def nll(params):
        mu, sigma = float(params[0]), float(params[1])
        if not math.isfinite(mu) or not (0.01 < sigma < 3.0):
            return float("inf")
        z = (values - mu) / sigma
        log_z = norm.logcdf((mu - trunc) / sigma)  # P(X >= trunc)
        total = -np.sum(norm.logpdf(z) - math.log(sigma) - log_z)
        return float(total) if math.isfinite(float(total)) else float("inf")

    start = (float(np.mean(values) + 0.1), float(np.std(values) + 0.05))
    best = minimize(
        nll,
        start,
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 2000},
    )
    if not math.isfinite(float(best.fun)):
        return _empty_fit(
            "failed",
            "optimizer_failed",
            n=int(values.size),
            truncation="lower",
            optimizer_message=str(best.message),
        )
    mu, sigma = map(float, best.x)
    return {
        "status": "ok",
        "reason": None,
        "success": bool(best.success),
        "model": "truncated_gaussian_logL",
        "mu_logL": mu,
        "sigma_logL": sigma,
        "mu_M_r": float(log_luminosity_to_absolute_magnitude(mu)),
        "sigma_M_r": float(2.5 * sigma),
        "logL_trunc": trunc,
        "n": int(values.size),
        "negative_log_likelihood": float(best.fun),
        "optimizer_message": str(best.message),
    }


# ---------------------------------------------------------------------------
# Sample preparation: flags and per-object truncation limits
# ---------------------------------------------------------------------------
def _load_parent_photometry(path=None) -> pd.DataFrame | None:
    """Load parent-catalogue apparent magnitudes for the flux-limit scheme."""

    if path is None:
        path = os.path.join(co.DATA_PATH, PARENT_PHOTOMETRY_FILE)
    try:
        parent = pd.read_csv(path)
    except Exception:
        return None
    needed = {"objid", "rmag", "M_r"}
    if not needed.issubset(parent.columns):
        return None
    columns = ["objid", "rmag", "M_r"] + (["z"] if "z" in parent.columns else [])
    return parent[columns].dropna(subset=["rmag", "M_r"])


def _accordance_limits(work: pd.DataFrame, M_sun_r: float) -> tuple[pd.Series, dict]:
    """Per-object limits from the 3-mag accordance criterion.

    The brightest magnitude of each group is recomputed from the data (it
    coincides with the catalogue ``M_BGG`` for the real samples and keeps the
    function usable on synthetic inputs).
    """

    brightest = work.groupby("group_uid")["M_r"].transform("min")
    limit_mag = brightest + ACCORDANCE_MAG
    logL_lim = absolute_magnitude_to_log_luminosity(limit_mag, M_sun_r=M_sun_r)
    n_beyond = int((work["M_r"] > limit_mag + 1e-9).sum())
    meta = {
        "scheme": "accordance_3mag",
        "n_beyond_nominal_limit": n_beyond,
    }
    return logL_lim, meta


def _flux_limits(
    work: pd.DataFrame, parent: pd.DataFrame | None, M_sun_r: float
) -> tuple[pd.Series | None, dict]:
    """Per-object limits from the flux limit at each galaxy's distance."""

    if parent is None or "objid" not in work.columns:
        return None, {
            "scheme": "flux_limit",
            "status": "unavailable",
            "reason": "missing_parent_photometry_or_objid",
        }
    r_lim = float(getattr(co, "R_MAX", 17.77))
    matched = work[["objid"]].merge(
        parent[["objid", "rmag"]], on="objid", how="left"
    )
    matched.index = work.index
    limit_mag = work["M_r"] + (r_lim - matched["rmag"])

    missing = limit_mag.isna()
    n_fallback = int(missing.sum())
    if missing.any() and "z" in work.columns and "z" in parent.columns:
        parent_z = parent["z"].to_numpy(dtype=float)
        parent_dm = (parent["rmag"] - parent["M_r"]).to_numpy(dtype=float)
        for idx in work.index[missing]:
            z_value = work.at[idx, "z"]
            if not np.isfinite(z_value):
                continue
            window = np.abs(parent_z - float(z_value)) < 0.002
            if not window.any():
                order = np.argsort(np.abs(parent_z - float(z_value)))[:50]
                window = np.zeros(parent_z.size, dtype=bool)
                window[order] = True
            limit_mag.at[idx] = work.at[idx, "M_r"] + (
                r_lim - (work.at[idx, "M_r"] + float(np.median(parent_dm[window])))
            )
    still_missing = limit_mag.isna()
    logL_lim = absolute_magnitude_to_log_luminosity(limit_mag, M_sun_r=M_sun_r)
    meta = {
        "scheme": "flux_limit",
        "status": "ok",
        "r_limit": r_lim,
        "n_direct_match": int(len(work) - n_fallback),
        "n_redshift_fallback": int(n_fallback - still_missing.sum()),
        "n_unresolved": int(still_missing.sum()),
    }
    if still_missing.any():
        floor = np.nanmin(work["logL_r"].to_numpy(dtype=float))
        logL_lim = logL_lim.fillna(floor)
        meta["unresolved_filled_with_common_floor"] = float(floor)
    return logL_lim, meta


def _common_floor_limits(work: pd.DataFrame) -> tuple[pd.Series, dict]:
    floor = float(np.nanmin(work["logL_r"].to_numpy(dtype=float)))
    logL_lim = pd.Series(floor, index=work.index)
    meta = {"scheme": "common_floor", "logL_floor": floor}
    return logL_lim, meta


def prepare_lf_frame(
    sample: Mapping[str, object],
    samples: Sequence[str] = DEFAULT_SAMPLES,
    M_sun_r: float = DEFAULT_M_SUN_R,
    parent_photometry_path: str | None = None,
) -> pd.DataFrame:
    """Return a clean galaxy frame for luminosity-function work.

    The returned frame contains one globally unique ``group_uid`` per input
    group, recomputed BGG/satellite flags based on the brightest ``M_r`` in
    each group, and a per-object lower truncation limit ``logL_lim`` chosen
    according to the sample construction (see module docstring).  Rows with
    missing group identifiers or non-finite luminosities are dropped.  The
    input ``sample`` dictionary is not modified.
    """

    parent = None
    if any(name in FLUX_LIMIT_SAMPLES for name in samples):
        parent = _load_parent_photometry(parent_photometry_path)

    frames = []
    preparation = {}
    for sample_name in samples:
        key = f"{sample_name}_Gals"
        if key not in sample:
            preparation[sample_name] = {
                "status": "skipped",
                "reason": "missing_galaxy_frame",
            }
            continue

        data = sample[key]
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        missing = [column for column in ("Group", "M_r") if column not in data.columns]
        if missing:
            preparation[sample_name] = {
                "status": "skipped",
                "reason": "missing_required_columns",
                "missing_columns": missing,
            }
            continue

        work = data.copy()
        work["sample"] = sample_name
        work["original_row_order"] = np.arange(len(work), dtype=int)
        work["M_r"] = pd.to_numeric(work["M_r"], errors="coerce")
        work["logL_r"] = absolute_magnitude_to_log_luminosity(
            work["M_r"], M_sun_r=M_sun_r
        )

        valid_group = work["Group"].notna()
        valid_luminosity = np.isfinite(work["M_r"]) & np.isfinite(work["logL_r"])
        work = work.loc[valid_group & valid_luminosity].copy()
        if work.empty:
            preparation[sample_name] = {
                "status": "skipped",
                "reason": "no_finite_grouped_luminosities",
                "n_input": int(len(data)),
            }
            continue

        work["group_uid"] = sample_name + ":" + work["Group"].astype(str)

        if sample_name in ACCORDANCE_SAMPLES:
            logL_lim, truncation_meta = _accordance_limits(work, M_sun_r)
        elif sample_name in FLUX_LIMIT_SAMPLES:
            logL_lim, truncation_meta = _flux_limits(work, parent, M_sun_r)
            if logL_lim is None:
                logL_lim, floor_meta = _common_floor_limits(work)
                truncation_meta = {**truncation_meta, "fallback": floor_meta}
        else:
            logL_lim, truncation_meta = _common_floor_limits(work)
        # A galaxy may sit marginally beyond its nominal limit; its own
        # luminosity then bounds the effective limit (lower truncation).
        work["logL_lim"] = np.minimum(
            pd.to_numeric(logL_lim, errors="coerce"), work["logL_r"] - 1e-9
        )

        frames.append(work)
        preparation[sample_name] = {
            "status": "ok",
            "n_input": int(len(data)),
            "n_used": int(len(work)),
            "n_dropped": int(len(data) - len(work)),
            "n_groups": int(work["group_uid"].nunique()),
            "truncation": truncation_meta,
        }

    if frames:
        frame = pd.concat(frames, ignore_index=True, sort=False)
        frame = add_lf_bgg_flags(frame)
    else:
        frame = pd.DataFrame(
            columns=[
                "sample",
                "group_uid",
                "Group",
                "M_r",
                "logL_r",
                "logL_lim",
                "is_bgg_lf",
                "is_satellite_lf",
            ]
        )
    frame.attrs["sample_preparation"] = preparation
    return frame


def add_lf_bgg_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Add BGG/satellite flags using the brightest ``M_r`` per group.

    Ties are resolved deterministically by the original row order when that
    column is present, otherwise by the current row order.
    """

    required = {"group_uid", "M_r"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"LF BGG classification requires columns: {sorted(missing)}")

    result = frame.copy()
    if "original_row_order" in result:
        row_order = pd.to_numeric(result["original_row_order"], errors="coerce")
        row_order = row_order.fillna(pd.Series(np.arange(len(result)), index=result.index))
    else:
        row_order = pd.Series(np.arange(len(result)), index=result.index)
    result["_lf_row_order"] = row_order.to_numpy(dtype=float)
    result["M_r"] = pd.to_numeric(result["M_r"], errors="coerce")
    result["is_bgg_lf"] = False

    sorted_frame = result.sort_values(
        ["group_uid", "M_r", "_lf_row_order"], kind="mergesort"
    )
    bgg_indices = sorted_frame.groupby("group_uid", sort=False).head(1).index
    result.loc[bgg_indices, "is_bgg_lf"] = True
    result["is_satellite_lf"] = ~result["is_bgg_lf"]
    return result.drop(columns=["_lf_row_order"])


def _component_frame(frame: pd.DataFrame, component: str) -> pd.DataFrame:
    if component in {"bgg", "bggs", "BGG", "BGGs"}:
        return frame.loc[frame["is_bgg_lf"].astype(bool)].copy()
    if component in {"satellite", "satellites"}:
        return frame.loc[frame["is_satellite_lf"].astype(bool)].copy()
    raise ValueError(f"Unknown luminosity-function component: {component}")


def _bgg_truncation(values: np.ndarray, M_sun_r: float = DEFAULT_M_SUN_R) -> float:
    design = float(
        absolute_magnitude_to_log_luminosity(BGG_DESIGN_MAG_CUT, M_sun_r=M_sun_r)
    )
    if values.size:
        return min(design, float(np.min(values)) - 1e-9)
    return design


def _fit_component_sample(
    part: pd.DataFrame, component: str, sample_name: str
) -> dict[str, object]:
    rows = part.loc[part["sample"] == sample_name]
    values = rows["logL_r"].to_numpy(dtype=float)
    if component == "satellites":
        return fit_schechter_mle(values, logL_lim=rows["logL_lim"].to_numpy(dtype=float))
    return fit_truncated_gaussian_mle(values, _bgg_truncation(values))


# ---------------------------------------------------------------------------
# Group bootstrap
# ---------------------------------------------------------------------------
def bootstrap_schechter_by_group(
    frame: pd.DataFrame,
    component: str,
    sample_name: str,
    bootstrap_iterations: int = 500,
    random_state: int | np.random.Generator = 42,
) -> dict[str, object]:
    """Group-bootstrap fits for one sample/component.

    Whole groups are resampled with replacement (galaxies within a group are
    not independent).  Satellites are refitted with their per-object
    truncation limits; BGGs with the truncated Gaussian.
    """

    if bootstrap_iterations is None or int(bootstrap_iterations) <= 0:
        return {
            "status": "skipped",
            "reason": "no_bootstrap_requested",
            "n_bootstrap_success": 0,
        }

    part = _component_frame(frame, component)
    part = part.loc[part["sample"] == sample_name]
    if len(part) < MIN_FIT_N:
        return {
            "status": "skipped",
            "reason": "too_few_galaxies",
            "n": int(len(part)),
            "min_required": MIN_FIT_N,
            "n_bootstrap_success": 0,
        }

    grouped = {
        group_uid: (
            group["logL_r"].to_numpy(dtype=float),
            group["logL_lim"].to_numpy(dtype=float)
            if "logL_lim" in group
            else np.full(len(group), np.nan),
        )
        for group_uid, group in part.groupby("group_uid", sort=False)
    }
    group_ids = np.array(list(grouped), dtype=object)
    if group_ids.size == 0:
        return {"status": "skipped", "reason": "no_groups", "n_bootstrap_success": 0}

    trunc = _bgg_truncation(part["logL_r"].to_numpy(dtype=float))
    boot_starts = None
    if component == "satellites":
        full_fit = fit_schechter_mle(
            part["logL_r"].to_numpy(dtype=float),
            logL_lim=part["logL_lim"].to_numpy(dtype=float)
            if "logL_lim" in part
            else None,
        )
        if full_fit.get("status") == "ok":
            boot_starts = [(full_fit["alpha"], full_fit["logL_star"])]
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    fitted = []
    failures = 0
    for _ in range(int(bootstrap_iterations)):
        draw = rng.choice(group_ids, size=group_ids.size, replace=True)
        values = np.concatenate([grouped[uid][0] for uid in draw])
        if component == "satellites":
            lims = np.concatenate([grouped[uid][1] for uid in draw])
            fit = fit_schechter_mle(values, logL_lim=lims, starts=boot_starts)
            keys = ("alpha", "logL_star")
        else:
            fit = fit_truncated_gaussian_mle(values, trunc)
            keys = ("mu_logL", "sigma_logL")
        if fit.get("status") == "ok":
            fitted.append(tuple(fit[key] for key in keys))
        else:
            failures += 1

    if not fitted:
        return {
            "status": "failed",
            "reason": "no_successful_bootstrap_fits",
            "n_bootstrap_success": 0,
            "n_bootstrap_failed": int(failures),
        }

    estimates = np.asarray(fitted, dtype=float)
    first_p16, first_p50, first_p84 = np.percentile(estimates[:, 0], [16, 50, 84])
    second_p16, second_p50, second_p84 = np.percentile(estimates[:, 1], [16, 50, 84])
    if component == "satellites":
        labels = ("alpha", "logL_star")
    else:
        labels = ("mu_logL", "sigma_logL")
    summary: dict[str, object] = {
        "status": "ok",
        "n_bootstrap_success": int(len(fitted)),
        "n_bootstrap_failed": int(failures),
    }
    summary.update(
        {
            f"{labels[0]}_p16": float(first_p16),
            f"{labels[0]}_median": float(first_p50),
            f"{labels[0]}_p84": float(first_p84),
            f"{labels[1]}_p16": float(second_p16),
            f"{labels[1]}_median": float(second_p50),
            f"{labels[1]}_p84": float(second_p84),
        }
    )
    return summary


# ---------------------------------------------------------------------------
# Plots (visualization only; fits are unbinned)
# ---------------------------------------------------------------------------
def gehrels_intervals(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Small-number Poisson 1-sigma intervals (Gehrels 1986)."""

    counts = np.asarray(counts, dtype=float)
    upper = np.sqrt(counts + 0.75) + 1.0
    lower = np.where(counts > 0, np.sqrt(np.maximum(counts - 0.25, 0.0)), 0.0)
    return lower, upper


def _satellite_expected_curve(
    grid: np.ndarray, rows: pd.DataFrame, alpha: float, logL_star: float, width: float
) -> np.ndarray:
    """Expected counts per bin-width, summing per-object truncated densities."""

    lims = rows["logL_lim"].to_numpy(dtype=float)
    unique_lims, counts = np.unique(lims, return_counts=True)
    x_lim = 10.0 ** (unique_lims - logL_star)
    gamma_values = upper_incomplete_gamma(alpha + 1.0, x_lim)
    good = np.isfinite(gamma_values) & (gamma_values > 0)
    if not good.any():
        return np.full(grid.shape, np.nan)
    unique_lims, counts, gamma_values = (
        unique_lims[good],
        counts[good],
        gamma_values[good],
    )
    log_x = LN10 * (grid[:, None] - logL_star)
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.exp(log_x)
    pdf = LN10 * np.exp((alpha + 1.0) * log_x - x) / gamma_values[None, :]
    pdf[grid[:, None] < unique_lims[None, :]] = 0.0
    return pdf @ counts * width


def _fit_label(sample_name: str, values: np.ndarray, fit: Mapping[str, object]) -> str:
    if fit.get("status") != "ok":
        return f"{sample_name} (n={len(values)})"
    if fit.get("model") == "truncated_gaussian_logL":
        return (
            f"{sample_name} (n={len(values)}, mu={fit['mu_logL']:.2f}, "
            f"sigma={fit['sigma_logL']:.2f})"
        )
    return (
        f"{sample_name} (n={len(values)}, alpha={fit['alpha']:.2f}, "
        f"logL*={fit['logL_star']:.2f})"
    )


def plot_luminosity_histograms_with_fits(
    frame: pd.DataFrame,
    component: str,
    fits: Mapping[str, Mapping[str, object]],
    output_dir: str | os.PathLike | None = None,
    n_bins: int = 7,
    samples: Sequence[str] = DEFAULT_SAMPLES,
    filename: str | None = None,
) -> str | None:
    """Plot luminosity histograms with Poisson errors and fitted expectations."""

    output_dir = output_dir or co.FIGURES_PATH
    os.makedirs(output_dir, exist_ok=True)
    part = _component_frame(frame, component)
    if part.empty:
        return None

    all_values = pd.to_numeric(part["logL_r"], errors="coerce").dropna()
    if all_values.empty:
        return None
    lower, upper = float(all_values.min()), float(all_values.max())
    if upper <= lower:
        upper = lower + 0.2
    edges = np.linspace(lower, upper, int(n_bins) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(np.median(np.diff(edges)))

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    positive_values = []
    plotted = False
    for sample_name in samples:
        rows = part.loc[part["sample"] == sample_name]
        values = rows["logL_r"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        colour = SAMPLE_COLOURS.get(sample_name)
        counts, _ = np.histogram(values, bins=edges)
        visible_counts = np.where(counts > 0, counts, np.nan)
        ax.stairs(
            visible_counts,
            edges,
            linewidth=2.0,
            color=colour,
            label=_fit_label(sample_name, values, fits.get(sample_name, {})),
        )
        err_low, err_high = gehrels_intervals(counts)
        with_data = counts > 0
        ax.errorbar(
            centres[with_data],
            counts[with_data],
            yerr=[err_low[with_data], err_high[with_data]],
            fmt="none",
            ecolor=colour,
            elinewidth=1.0,
            capsize=2.0,
            alpha=0.85,
        )
        positive_values.extend(counts[counts > 0].tolist())
        plotted = True

        fit = fits.get(sample_name, {})
        if fit.get("status") == "ok":
            grid = np.linspace(edges[0], edges[-1], 300)
            if component == "satellites":
                expected = _satellite_expected_curve(
                    grid, rows, fit["alpha"], fit["logL_star"], bin_width
                )
            else:
                mu, sigma, trunc = (
                    fit["mu_logL"],
                    fit["sigma_logL"],
                    fit["logL_trunc"],
                )
                pdf = norm.pdf((grid - mu) / sigma) / sigma
                pdf /= norm.cdf((mu - trunc) / sigma)
                pdf[grid < trunc] = 0.0
                expected = float(fit.get("n", values.size)) * pdf * bin_width
            expected = np.where(expected > 0, expected, np.nan)
            ax.plot(grid, expected, color=colour, linewidth=1.5, alpha=0.8)
            positive_values.extend(expected[np.isfinite(expected)].tolist())

    if not plotted:
        plt.close(fig)
        return None

    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(L_r / L_\odot)$")
    ax.set_ylabel("Galaxies per bin")
    ax.set_title(COMPONENT_LABELS.get(component, component))
    ax.set_xlim(edges[0], edges[-1])
    positive_values = [value for value in positive_values if value > 0]
    if positive_values:
        ax.set_ylim(max(min(positive_values) / 1.5, 0.5), max(positive_values) * 2.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    filename = filename or FIGURE_NAMES.get(
        component, f"fig_luminosity_function_{component}.pdf"
    )
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


# ---------------------------------------------------------------------------
# Bookkeeping helpers
# ---------------------------------------------------------------------------
def _mass_availability(part: pd.DataFrame) -> dict[str, object]:
    availability = {}
    for column in ("lgm", "logMstar"):
        if column in part:
            values = pd.to_numeric(part[column], errors="coerce")
            availability[column] = {
                "n_available": int(np.isfinite(values).sum()),
                "fraction": float(np.isfinite(values).mean()) if len(values) else 0.0,
            }
    return availability


def _per_sample_counts(frame: pd.DataFrame) -> dict[str, object]:
    counts = {}
    for sample_name in DEFAULT_SAMPLES:
        part = frame.loc[frame["sample"] == sample_name]
        counts[sample_name] = {
            "n_galaxies": int(len(part)),
            "n_groups": int(part["group_uid"].nunique()) if "group_uid" in part else 0,
            "n_bgg": int(part["is_bgg_lf"].sum()) if "is_bgg_lf" in part else 0,
            "n_satellites": int(part["is_satellite_lf"].sum())
            if "is_satellite_lf" in part
            else 0,
            "stellar_mass_availability": _mass_availability(part),
        }
    return counts


def _component_counts(frame: pd.DataFrame, component: str) -> dict[str, int]:
    part = _component_frame(frame, component)
    return {
        sample_name: int((part["sample"] == sample_name).sum())
        for sample_name in DEFAULT_SAMPLES
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_luminosity_function_analysis(
    sample,
    output_dir=None,
    n_bins=7,
    bootstrap_iterations=500,
    random_state=42,
) -> dict[str, object]:
    """Run luminosity-function plots and unbinned truncated fits.

    Satellites: Schechter with per-object truncation limits.
    BGGs: Gaussian in ``log10 L`` truncated at the design cut.
    """

    output_dir = output_dir or co.FIGURES_PATH
    os.makedirs(output_dir, exist_ok=True)
    frame = prepare_lf_frame(sample, samples=DEFAULT_SAMPLES)
    result = {
        "status": "ok",
        "samples_used": list(DEFAULT_SAMPLES),
        "samples_in_frame": sorted(frame["sample"].unique().tolist())
        if "sample" in frame and not frame.empty
        else [],
        "samples_excluded": dict(EXCLUDED_SAMPLES),
        "components_removed": dict(REMOVED_COMPONENTS),
        "adopted_M_sun_r": DEFAULT_M_SUN_R,
        "bgg_design_mag_cut": BGG_DESIGN_MAG_CUT,
        "n_bins": int(n_bins),
        "sample_preparation": frame.attrs.get("sample_preparation", {}),
        "per_sample_counts": _per_sample_counts(frame),
        "components": {},
        "generated_figures": [],
    }
    if frame.empty:
        result["status"] = "skipped"
        result["reason"] = "no_usable_luminosity_data"
        return safe_json(result)

    rng = np.random.default_rng(random_state)
    for component in ("satellites", "bgg"):
        component_frame = _component_frame(frame, component)
        if component_frame.empty:
            result["components"][component] = {
                "status": "skipped",
                "reason": "no_luminosities",
                "fits": {},
                "n_by_sample": _component_counts(frame, component),
            }
            continue

        fits = {}
        for sample_name in DEFAULT_SAMPLES:
            fit = _fit_component_sample(component_frame, component, sample_name)
            seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            fit["bootstrap"] = bootstrap_schechter_by_group(
                frame,
                component,
                sample_name,
                bootstrap_iterations=bootstrap_iterations,
                random_state=seed,
            )
            fits[sample_name] = fit

        figure = plot_luminosity_histograms_with_fits(
            frame,
            component,
            fits,
            output_dir=output_dir,
            n_bins=n_bins,
            filename=FIGURE_NAMES[component],
        )
        if figure:
            result["generated_figures"].append(figure)

        result["components"][component] = {
            "status": "ok",
            "model": "schechter_per_object_truncation"
            if component == "satellites"
            else "truncated_gaussian_logL",
            "n_by_sample": _component_counts(frame, component),
            "fits": fits,
            "figure": figure,
        }

    if not result["generated_figures"]:
        result["status"] = "skipped"
        result["reason"] = "no_figures_generated"
    return safe_json(result)
