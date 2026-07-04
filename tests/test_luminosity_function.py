import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad

from src.luminosity_function import (
    absolute_magnitude_to_log_luminosity,
    add_lf_bgg_flags,
    fit_schechter_mle,
    fit_truncated_gaussian_mle,
    prepare_lf_frame,
    run_luminosity_function_analysis,
    upper_incomplete_gamma,
)


def test_absolute_magnitude_to_log_luminosity_orders_brightness():
    # M_sun_r follows astro_utils.Mr_Sol so L* matches the Lum columns.
    assert absolute_magnitude_to_log_luminosity(4.68) == 0.0
    assert absolute_magnitude_to_log_luminosity(-21.0) > absolute_magnitude_to_log_luminosity(
        -19.0
    )


def test_upper_incomplete_gamma_matches_quadrature():
    for a, x in [(-1.5, 0.3), (-0.2, 0.05), (0.5, 1.2), (1.0, 2.0)]:
        reference = quad(lambda t: t ** (a - 1.0) * math.exp(-t), x, np.inf, limit=200)[0]
        value = float(upper_incomplete_gamma(a, np.array([x]))[0])
        assert math.isclose(value, reference, rel_tol=1e-8)


def test_lf_bgg_flags_use_brightest_magnitude_and_stable_ties():
    frame = pd.DataFrame(
        {
            "sample": ["CG4"] * 6,
            "Group": [1, 1, 1, 2, 2, 2],
            "group_uid": ["CG4:1", "CG4:1", "CG4:1", "CG4:2", "CG4:2", "CG4:2"],
            "M_r": [-21.0, -21.0, -20.0, -19.0, -22.0, -21.5],
            "original_row_order": [0, 1, 2, 3, 4, 5],
        }
    )

    flagged = add_lf_bgg_flags(frame)

    assert flagged.groupby("group_uid")["is_bgg_lf"].sum().tolist() == [1, 1]
    assert bool(flagged.loc[0, "is_bgg_lf"]) is True
    assert bool(flagged.loc[1, "is_satellite_lf"]) is True
    assert bool(flagged.loc[4, "is_bgg_lf"]) is True


def _mock_sample(n_groups=5, members=4, include_control4b=False):
    sample = {}
    offsets = {"CG4": 0.0, "RG4": 0.15, "Control4C": -0.1, "Control4B": 0.3}
    for sample_name, offset in offsets.items():
        if sample_name == "Control4B" and not include_control4b:
            continue
        rows = []
        for group in range(n_groups):
            for member in range(members):
                rows.append(
                    {
                        "Group": 1000 + group,
                        "M_r": -22.2 + offset + 0.55 * member + 0.03 * group,
                        "lgm": 10.5 - 0.06 * member,
                    }
                )
        sample[f"{sample_name}_Gals"] = pd.DataFrame(rows)
        sample[f"{sample_name}_Groups"] = pd.DataFrame({"Group": np.arange(n_groups)})
    return sample


def test_prepare_lf_frame_excludes_control4b_by_default():
    sample = _mock_sample(include_control4b=True)

    frame = prepare_lf_frame(sample)

    assert set(frame["sample"]) == {"CG4", "RG4", "Control4C"}
    assert "Control4B" not in set(frame["sample"])


def test_prepare_lf_frame_assigns_truncation_schemes():
    sample = _mock_sample()

    frame = prepare_lf_frame(sample)
    preparation = frame.attrs["sample_preparation"]

    # CG4/RG4 follow the 3-mag accordance rule: limit = brightest member + 3.
    assert preparation["CG4"]["truncation"]["scheme"] == "accordance_3mag"
    cg4 = frame.loc[frame["sample"] == "CG4"]
    for _, group in cg4.groupby("group_uid"):
        expected = absolute_magnitude_to_log_luminosity(group["M_r"].min() + 3.0)
        assert np.allclose(group["logL_lim"], np.minimum(expected, group["logL_r"] - 1e-9))
    # Every galaxy respects its own limit (lower truncation in logL).
    assert (frame["logL_r"] >= frame["logL_lim"]).all()
    # Without parent photometry, Control4C falls back on a recorded common floor.
    c4c_truncation = preparation["Control4C"]["truncation"]
    assert c4c_truncation["scheme"] == "flux_limit"
    assert c4c_truncation["status"] == "unavailable"
    assert c4c_truncation["fallback"]["scheme"] == "common_floor"


def _draw_truncated_schechter(rng, alpha, logL_star, logL_lim, size):
    """Rejection-sample ``size`` values from a lower-truncated Schechter."""

    x_lim = 10.0 ** (np.asarray(logL_lim, dtype=float) - logL_star)
    out = np.empty(size)
    filled = 0
    while filled < size:
        draw = rng.gamma(shape=alpha + 1.0, scale=1.0, size=4 * (size - filled))
        draw = draw[draw >= np.min(x_lim)]
        take = min(draw.size, size - filled)
        out[filled : filled + take] = draw[:take]
        filled += take
    return logL_star + np.log10(out)


def test_schechter_recovery_with_per_object_truncation():
    """Simulation-based recovery test with the accordance-style selection.

    Satellites are drawn from a known Schechter above per-group limits (as in
    CG4/RG4, where the limit is M_BGG + 3).  The per-object estimator must
    recover the input parameters; the common-floor estimator, which ignores
    the per-group censoring, is biased shallow and serves as the contrast.
    """

    rng = np.random.default_rng(20260703)
    alpha_true, logL_star_true = -0.8, 10.4
    n_groups, sats_per_group = 500, 3

    group_lims = rng.normal(9.65, 0.20, size=n_groups)  # BGG logL - 1.2 dex
    logL, lims = [], []
    for lim in group_lims:
        accepted = []
        while len(accepted) < sats_per_group:
            draws = _draw_truncated_schechter(
                rng, alpha_true, logL_star_true, lim, 8
            )
            accepted.extend(draws[draws >= lim].tolist())
        logL.extend(accepted[:sats_per_group])
        lims.extend([lim] * sats_per_group)
    logL, lims = np.asarray(logL), np.asarray(lims)

    fit = fit_schechter_mle(logL, logL_lim=lims)
    assert fit["status"] == "ok"
    assert abs(fit["alpha"] - alpha_true) < 0.15
    assert abs(fit["logL_star"] - logL_star_true) < 0.12

    # The single-common-floor fit reads the per-group censoring as a real
    # faint-end decline: alpha comes out substantially too shallow.
    floor_fit = fit_schechter_mle(logL, logL_min=float(np.min(logL)))
    assert floor_fit["status"] == "ok"
    assert floor_fit["alpha"] - fit["alpha"] > 0.2


def test_schechter_fit_returns_finite_values_and_skips_small_samples():
    rng = np.random.default_rng(123)
    logL_star = 10.7
    logL_min = 9.6
    values = []
    while len(values) < 120:
        y = rng.gamma(shape=1.4, scale=1.0, size=200)
        draw = logL_star + np.log10(y[y > 0])
        values.extend(draw[draw >= logL_min].tolist())
    values = np.asarray(values[:120])

    fit = fit_schechter_mle(values, logL_min=logL_min)

    assert fit["status"] == "ok"
    assert np.isfinite(fit["alpha"])
    assert np.isfinite(fit["logL_star"])
    assert fit_schechter_mle(values[:7], logL_min=logL_min)["status"] == "skipped"


def test_truncated_gaussian_recovers_bgg_like_population():
    rng = np.random.default_rng(11)
    mu_true, sigma_true, trunc = 10.55, 0.20, 10.596  # design cut at M_r = -21.81
    draws = rng.normal(mu_true, sigma_true, size=6000)
    draws = draws[draws >= trunc][:800]

    fit = fit_truncated_gaussian_mle(draws, trunc)

    assert fit["status"] == "ok"
    assert abs(fit["mu_logL"] - mu_true) < 0.08
    assert abs(fit["sigma_logL"] - sigma_true) < 0.05
    assert fit["mu_M_r"] is not None


def test_luminosity_function_end_to_end_creates_figures(tmp_path):
    sample = _mock_sample(n_groups=5, members=4, include_control4b=True)

    result = run_luminosity_function_analysis(
        sample,
        output_dir=tmp_path,
        n_bins=6,
        bootstrap_iterations=5,
        random_state=7,
    )

    assert result["status"] == "ok"
    assert result["samples_excluded"]["Control4B"].startswith("excluded")
    assert "all" in result["components_removed"]
    # 5 mock BGGs per sample is below MIN_FIT_N, so the BGG fit is skipped.
    assert result["components"]["bgg"]["fits"]["CG4"]["status"] == "skipped"
    assert set(result["generated_figures"]) == {
        "fig_luminosity_function_bgg.pdf",
        "fig_luminosity_function_satellites.pdf",
    }
    for filename in result["generated_figures"]:
        assert (Path(tmp_path) / filename).is_file()
    json.dumps(result, allow_nan=False)
