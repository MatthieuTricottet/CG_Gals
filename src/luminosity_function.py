"""Luminosity-function analysis for BGG and satellite galaxy samples.

Histograms in this module are visualization products only.  Schechter
parameters are estimated with an unbinned, lower-truncated maximum likelihood
so the fitted values do not depend on arbitrary histogram bin choices.
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
from scipy.integrate import quad
from scipy.optimize import minimize

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
DEFAULT_M_SUN_R = 4.65
MIN_FIT_N = 8

COMPONENT_LABELS = {
    "all": "All galaxies",
    "bgg": "BGGs",
    "satellites": "Satellites",
}
FIGURE_NAMES = {
    "all": "fig_luminosity_function_all.pdf",
    "bgg": "fig_luminosity_function_bgg.pdf",
    "satellites": "fig_luminosity_function_satellites.pdf",
}
SAMPLE_COLOURS = {
    "CG4": "#2864A6",
    "RG4": "#D07C28",
    "Control4C": "#4D8C57",
}


def absolute_magnitude_to_log_luminosity(M_r, M_sun_r=DEFAULT_M_SUN_R):
    """Convert SDSS-r absolute magnitude to log10 solar luminosity.

    The default ``M_sun_r=4.65`` is the adopted SDSS-r solar absolute
    magnitude; change it if the paper adopts a different convention.
    """

    if isinstance(M_r, pd.Series):
        values = pd.to_numeric(M_r, errors="coerce")
        return -0.4 * (values - M_sun_r)
    if np.isscalar(M_r):
        value = pd.to_numeric(pd.Series([M_r]), errors="coerce").iloc[0]
        return float(-0.4 * (value - M_sun_r))
    values = pd.to_numeric(pd.Series(M_r), errors="coerce")
    return (-0.4 * (values - M_sun_r)).to_numpy(dtype=float)


def prepare_lf_frame(
    sample: Mapping[str, object],
    samples: Sequence[str] = DEFAULT_SAMPLES,
    M_sun_r: float = DEFAULT_M_SUN_R,
) -> pd.DataFrame:
    """Return a clean galaxy frame for luminosity-function work.

    The returned frame contains one globally unique ``group_uid`` per input
    group and recomputed BGG/satellite flags based on the brightest ``M_r`` in
    each group.  Rows with missing group identifiers or non-finite luminosities
    are dropped.  The input ``sample`` dictionary is not modified.
    """

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
        frames.append(work)
        preparation[sample_name] = {
            "status": "ok",
            "n_input": int(len(data)),
            "n_used": int(len(work)),
            "n_dropped": int(len(data) - len(work)),
            "n_groups": int(work["group_uid"].nunique()),
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


def _as_float_array(values) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def _clean_log_luminosity(values, logL_min: float | None = None) -> np.ndarray:
    array = _as_float_array(values)
    array = array[np.isfinite(array)]
    if logL_min is not None and math.isfinite(float(logL_min)):
        array = array[array >= float(logL_min)]
    return array


def _schechter_log_normalization(
    alpha: float, logL_star: float, logL_min: float
) -> float | None:
    if not all(math.isfinite(float(value)) for value in (alpha, logL_star, logL_min)):
        return None
    t_min = math.log(10.0) * (float(logL_min) - float(logL_star))

    def integrand(t):
        if t > 50:
            return 0.0
        try:
            exp_t = math.exp(t)
        except OverflowError:
            return 0.0
        exponent = (float(alpha) + 1.0) * t - exp_t
        if exponent < -745:
            return 0.0
        if exponent > 709:
            return math.exp(709)
        return math.exp(exponent)

    try:
        value, _ = quad(
            integrand,
            t_min,
            np.inf,
            epsabs=1e-10,
            epsrel=1e-8,
            limit=200,
        )
    except Exception:
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return float(math.log(value))


def schechter_logpdf_logL(logL, alpha, logL_star, logL_min):
    """Return the normalized lower-truncated Schechter log-density in log10 L."""

    scalar = np.isscalar(logL)
    values = _as_float_array([logL] if scalar else logL)
    output = np.full(values.shape, -np.inf, dtype=float)
    log_norm = _schechter_log_normalization(alpha, logL_star, logL_min)
    if log_norm is None:
        return float(output[0]) if scalar else output

    valid = np.isfinite(values) & (values >= float(logL_min))
    if np.any(valid):
        log_y = math.log(10.0) * (values[valid] - float(logL_star))
        with np.errstate(over="ignore", invalid="ignore"):
            y = np.exp(log_y)
        log_unnormalized = (
            math.log(math.log(10.0)) + (float(alpha) + 1.0) * log_y - y
        )
        output[valid] = log_unnormalized - log_norm
    return float(output[0]) if scalar else output


def negative_log_likelihood(params, logL, logL_min):
    """Return the Schechter negative log likelihood for unbinned log-luminosities."""

    try:
        alpha, logL_star = float(params[0]), float(params[1])
    except (TypeError, ValueError, IndexError):
        return float("inf")
    if not all(math.isfinite(value) for value in (alpha, logL_star, float(logL_min))):
        return float("inf")
    values = _clean_log_luminosity(logL, logL_min=logL_min)
    if values.size == 0:
        return float("inf")
    logpdf = schechter_logpdf_logL(values, alpha, logL_star, logL_min)
    if not np.all(np.isfinite(logpdf)):
        return float("inf")
    return float(-np.sum(logpdf))


def _normalise_bounds(bounds, logL: np.ndarray, logL_min: float):
    if bounds is None:
        upper = float(np.max(logL)) + 1.5 if logL.size else float(logL_min) + 1.5
        return ((-3.0, 1.0), (float(logL_min) - 0.5, upper))
    if isinstance(bounds, Mapping):
        return (
            tuple(bounds.get("alpha", (-3.0, 1.0))),
            tuple(
                bounds.get(
                    "logL_star",
                    (
                        float(logL_min) - 0.5,
                        float(np.max(logL)) + 1.5 if logL.size else float(logL_min) + 1.5,
                    ),
                )
            ),
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
        "logL_min": extra.pop("logL_min", None),
        "negative_log_likelihood": None,
        "optimizer_message": extra.pop("optimizer_message", None),
        **extra,
    }


def fit_schechter_mle(logL, logL_min=None, bounds=None) -> dict[str, object]:
    """Fit a lower-truncated Schechter function by unbinned maximum likelihood."""

    values = _clean_log_luminosity(logL)
    if values.size == 0:
        return _empty_fit("skipped", "no_finite_luminosities")
    if logL_min is None:
        logL_min = float(np.min(values))
    else:
        logL_min = float(logL_min)
    values = values[values >= logL_min]
    if values.size < MIN_FIT_N:
        return _empty_fit(
            "skipped",
            "too_few_galaxies",
            n=int(values.size),
            logL_min=float(logL_min),
            min_required=MIN_FIT_N,
        )

    fit_bounds = _normalise_bounds(bounds, values, logL_min)
    logL_bounds = fit_bounds[1]
    starts = [
        (-1.0, np.percentile(values, 75)),
        (-0.5, np.percentile(values, 60)),
        (-1.5, np.percentile(values, 90)),
    ]
    starts = [
        (
            float(np.clip(alpha, fit_bounds[0][0] + 1e-4, fit_bounds[0][1] - 1e-4)),
            float(np.clip(star, logL_bounds[0] + 1e-4, logL_bounds[1] - 1e-4)),
        )
        for alpha, star in starts
    ]

    best = None
    for start in starts:
        try:
            result = minimize(
                negative_log_likelihood,
                start,
                args=(values, logL_min),
                method="L-BFGS-B",
                bounds=fit_bounds,
                options={"maxiter": 200, "ftol": 1e-8},
            )
        except Exception as exc:
            result = None
            message = f"{exc.__class__.__name__}: {exc}"
        else:
            message = str(result.message)
        if result is not None and math.isfinite(float(result.fun)):
            if best is None or float(result.fun) < float(best.fun):
                best = result
        elif best is None:
            best = type(
                "FailedResult",
                (),
                {
                    "success": False,
                    "message": message,
                    "fun": float("inf"),
                    "x": np.array([np.nan, np.nan]),
                },
            )()

    if best is None or not math.isfinite(float(best.fun)):
        return _empty_fit(
            "failed",
            "optimizer_failed",
            n=int(values.size),
            logL_min=float(logL_min),
            optimizer_message="no finite optimizer result",
        )

    alpha, logL_star = map(float, best.x)
    fit = {
        "status": "ok" if bool(best.success) else "failed",
        "reason": None if bool(best.success) else "optimizer_failed",
        "success": bool(best.success),
        "alpha": alpha if math.isfinite(alpha) else None,
        "logL_star": logL_star if math.isfinite(logL_star) else None,
        "L_star": float(10**logL_star) if math.isfinite(logL_star) else None,
        "n": int(values.size),
        "logL_min": float(logL_min),
        "negative_log_likelihood": float(best.fun),
        "optimizer_message": str(best.message),
    }
    return fit


def _component_frame(frame: pd.DataFrame, component: str) -> pd.DataFrame:
    if component == "all":
        return frame.copy()
    if component in {"bgg", "bggs", "BGG", "BGGs"}:
        return frame.loc[frame["is_bgg_lf"].astype(bool)].copy()
    if component in {"satellite", "satellites"}:
        return frame.loc[frame["is_satellite_lf"].astype(bool)].copy()
    raise ValueError(f"Unknown luminosity-function component: {component}")


def _common_logL_min(
    frame: pd.DataFrame, samples: Sequence[str] = DEFAULT_SAMPLES
) -> float | None:
    minima = []
    for sample_name in samples:
        values = _clean_log_luminosity(
            frame.loc[frame["sample"] == sample_name, "logL_r"]
        )
        if values.size:
            minima.append(float(np.min(values)))
    if not minima:
        return None
    return float(np.max(minima))


def bootstrap_schechter_by_group(
    frame: pd.DataFrame,
    component: str,
    sample_name: str,
    logL_min: float | None = None,
    bounds=None,
    bootstrap_iterations: int = 500,
    random_state: int | np.random.Generator = 42,
) -> dict[str, object]:
    """Group-bootstrap Schechter fits for one sample/component."""

    if bootstrap_iterations is None or int(bootstrap_iterations) <= 0:
        return {
            "status": "skipped",
            "reason": "no_bootstrap_requested",
            "n_bootstrap_success": 0,
        }

    part = _component_frame(frame, component)
    part = part.loc[part["sample"] == sample_name].copy()
    if logL_min is None:
        logL_min = _common_logL_min(part, samples=(sample_name,))
    if logL_min is None:
        return {
            "status": "skipped",
            "reason": "no_luminosities",
            "n_bootstrap_success": 0,
        }
    part = part.loc[pd.to_numeric(part["logL_r"], errors="coerce") >= float(logL_min)]
    if len(part) < MIN_FIT_N:
        return {
            "status": "skipped",
            "reason": "too_few_galaxies",
            "n": int(len(part)),
            "min_required": MIN_FIT_N,
            "n_bootstrap_success": 0,
        }

    grouped = {
        group_uid: group["logL_r"].to_numpy(dtype=float)
        for group_uid, group in part.groupby("group_uid", sort=False)
    }
    group_ids = np.array(list(grouped), dtype=object)
    if group_ids.size == 0:
        return {
            "status": "skipped",
            "reason": "no_groups",
            "n_bootstrap_success": 0,
        }

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    fitted = []
    failures = 0
    for _ in range(int(bootstrap_iterations)):
        draw = rng.choice(group_ids, size=group_ids.size, replace=True)
        values = np.concatenate([grouped[group_uid] for group_uid in draw])
        fit = fit_schechter_mle(values, logL_min=logL_min, bounds=bounds)
        if fit.get("status") == "ok":
            fitted.append((fit["alpha"], fit["logL_star"]))
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
    alpha_p16, alpha_p50, alpha_p84 = np.percentile(estimates[:, 0], [16, 50, 84])
    star_p16, star_p50, star_p84 = np.percentile(estimates[:, 1], [16, 50, 84])
    return {
        "status": "ok",
        "alpha_median": float(alpha_p50),
        "alpha_p16": float(alpha_p16),
        "alpha_p84": float(alpha_p84),
        "logL_star_median": float(star_p50),
        "logL_star_p16": float(star_p16),
        "logL_star_p84": float(star_p84),
        "n_bootstrap_success": int(len(fitted)),
        "n_bootstrap_failed": int(failures),
    }


def _fit_label(sample_name: str, values: np.ndarray, fit: Mapping[str, object]) -> str:
    if fit.get("status") == "ok":
        return (
            f"{sample_name} (n={len(values)}, alpha={fit['alpha']:.2f}, "
            f"logL*={fit['logL_star']:.2f})"
        )
    return f"{sample_name} (n={len(values)})"


def plot_luminosity_histograms_with_fits(
    frame: pd.DataFrame,
    component: str,
    fits: Mapping[str, Mapping[str, object]],
    output_dir: str | os.PathLike | None = None,
    n_bins: int = 7,
    logL_min: float | None = None,
    samples: Sequence[str] = DEFAULT_SAMPLES,
    filename: str | None = None,
) -> str | None:
    """Plot luminosity histograms and overlay fitted Schechter expectations."""

    output_dir = output_dir or co.FIGURES_PATH
    os.makedirs(output_dir, exist_ok=True)
    part = _component_frame(frame, component)
    if logL_min is None:
        logL_min = _common_logL_min(part, samples=samples)
    if logL_min is None:
        return None
    part = part.loc[pd.to_numeric(part["logL_r"], errors="coerce") >= float(logL_min)]
    if part.empty:
        return None

    upper = float(np.nanmax(part["logL_r"]))
    if not math.isfinite(upper) or upper <= float(logL_min):
        upper = float(logL_min) + 0.2
    edges = np.linspace(float(logL_min), upper, int(n_bins) + 1)
    bin_width = float(np.median(np.diff(edges)))

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    positive_values = []
    plotted = False
    for sample_name in samples:
        values = _clean_log_luminosity(
            part.loc[part["sample"] == sample_name, "logL_r"], logL_min=logL_min
        )
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
        positive_values.extend(counts[counts > 0].tolist())
        plotted = True

        fit = fits.get(sample_name, {})
        if fit.get("status") == "ok":
            grid = np.linspace(edges[0], edges[-1], 300)
            logpdf = schechter_logpdf_logL(
                grid, fit["alpha"], fit["logL_star"], logL_min
            )
            expected = float(fit.get("n", values.size)) * np.exp(logpdf) * bin_width
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
    filename = filename or FIGURE_NAMES.get(component, f"fig_luminosity_function_{component}.pdf")
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


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


def run_luminosity_function_analysis(
    sample,
    output_dir=None,
    n_bins=7,
    bootstrap_iterations=500,
    random_state=42,
) -> dict[str, object]:
    """Run luminosity-function plots and unbinned Schechter fits."""

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
        "adopted_M_sun_r": DEFAULT_M_SUN_R,
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
    for component in ("all", "bgg", "satellites"):
        component_frame = _component_frame(frame, component)
        logL_min = _common_logL_min(component_frame, samples=DEFAULT_SAMPLES)
        if logL_min is None:
            result["components"][component] = {
                "status": "skipped",
                "reason": "no_luminosities",
                "fits": {},
                "adopted_logL_min": None,
                "n_by_sample": _component_counts(frame, component),
            }
            continue

        fits = {}
        for sample_name in DEFAULT_SAMPLES:
            values = _clean_log_luminosity(
                component_frame.loc[
                    component_frame["sample"] == sample_name, "logL_r"
                ],
                logL_min=logL_min,
            )
            bounds = _normalise_bounds(None, values, logL_min) if values.size else None
            fit = fit_schechter_mle(values, logL_min=logL_min, bounds=bounds)
            seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            fit["bootstrap"] = bootstrap_schechter_by_group(
                frame,
                component,
                sample_name,
                logL_min=logL_min,
                bounds=bounds,
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
            logL_min=logL_min,
            filename=FIGURE_NAMES[component],
        )
        if figure:
            result["generated_figures"].append(figure)

        result["components"][component] = {
            "status": "ok",
            "adopted_logL_min": float(logL_min),
            "n_by_sample": _component_counts(frame, component),
            "fits": fits,
            "figure": figure,
        }

    if not result["generated_figures"]:
        result["status"] = "skipped"
        result["reason"] = "no_figures_generated"
    return safe_json(result)
