"""Tests for the audit-r3 revision: positional, emission-line and covariate checks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.extended_stats import (
    fit_logistic_model,
    overlap_coefficient,
    restricted_cubic_spline,
    standardized_contrast,
)
from src.halpha_emission import classify_emission
from src.specialness_models import _covariates

ROOT = Path(__file__).resolve().parents[1]


def test_restricted_cubic_spline_is_linear_beyond_outer_knots():
    knots = (1.0, 2.0, 3.0)
    x = np.array([0.0, 0.5, 1.0])
    assert np.allclose(restricted_cubic_spline(x, knots), 0.0)
    far = np.array([10.0, 11.0, 12.0])
    values = restricted_cubic_spline(far, knots)
    # second differences vanish: the basis is linear above the last knot
    assert abs(values[2] - 2 * values[1] + values[0]) < 1e-9


def _two_sample_frame(seed=1, n_groups=120):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), 4)
    treated = (groups < 30).astype(int)
    mass = rng.normal(10.3, 0.4, groups.size)
    logit = -0.5 + 0.8 * treated + 1.2 * (mass - 10.3)
    outcome = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame(
        {
            "is_CG4": treated,
            "logMstar": mass,
            "elliptical": outcome.astype(float),
            "physical_group": [f"g{g}" for g in groups],
        }
    )


def test_standardized_contrast_reproduces_observed_treated_fraction():
    frame = _two_sample_frame()
    result = standardized_contrast(
        frame, "elliptical", ["is_CG4", "logMstar"], continuous=["logMstar"], n_boot=50
    )
    observed = frame.loc[frame["is_CG4"] == 1, "elliptical"].mean()
    assert result["status"] == "ok"
    assert result["fraction_cg4"] == pytest.approx(observed, abs=1e-6)
    assert result["difference"] > 0
    low, high = result["difference_ci95"]
    assert low <= result["difference"] <= high


def test_overlap_coefficient_limits():
    a = np.linspace(0, 1, 200)
    assert overlap_coefficient(a, a) == pytest.approx(1.0, abs=1e-6)
    assert overlap_coefficient(a, a + 10) == pytest.approx(0.0, abs=1e-6)


def test_covariate_set_fails_loudly_when_incomplete():
    frame = pd.DataFrame(
        {
            "logMstar": [10.0, 10.5, 11.0, 10.2],
            "z_numeric": [0.02, 0.03, 0.04, 0.03],
            "is_satellite": [0, 1, 1, 1],
            "log_group_luminosity": [np.nan, np.nan, np.nan, 11.0],
            "velocity_dispersion": [200.0, 250.0, 300.0, 220.0],
        }
    )
    with pytest.raises(ValueError, match="log_group_luminosity"):
        _covariates(frame)
    selected, _ = _covariates(frame, exclude=("log_group_luminosity",))
    assert "log_group_luminosity" not in selected


def test_fit_logistic_model_reports_dropped_predictors():
    frame = _two_sample_frame()
    frame["constant_column"] = 1.0
    result = fit_logistic_model(
        frame, "elliptical", ["is_CG4", "logMstar", "constant_column"],
        continuous=["logMstar"],
    )
    assert result["status"] == "ok"
    assert {"predictor": "constant_column", "reason": "constant_in_fitted_frame"} in result[
        "predictors_dropped"
    ]


def test_whan_classes_follow_width_ratio_and_signal_to_noise():
    frame = pd.DataFrame(
        {
            "specobjid": [1, 2, 3, 4],
            "z_numeric": [0.03] * 4,
            # SF-like, high-[NII], weak line, strong line with undetected [NII]
            "gsl_h_alpha_eqw": [-20.0, -8.0, -1.0, -10.0],
            "gsl_h_alpha_flux": [500.0, 100.0, 10.0, 300.0],
            "gsl_h_alpha_flux_err": [5.0, 5.0, 5.0, 5.0],
            "gsl_nii_6584_flux": [100.0, 80.0, 5.0, 1.0],
            "gsl_nii_6584_flux_err": [5.0, 5.0, 5.0, 5.0],
            "gsl_h_beta_flux": [150.0, 30.0, 3.0, 90.0],
            "gsl_h_beta_flux_err": [5.0, 5.0, 5.0, 5.0],
            "gsl_oiii_5007_flux": [100.0, 60.0, 3.0, 60.0],
            "gsl_oiii_5007_flux_err": [5.0, 5.0, 5.0, 5.0],
        }
    )
    work = classify_emission(frame)
    assert work["sf_emission"].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert work["high_nii_emission"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert work["retired"].tolist() == [0.0, 0.0, 1.0, 0.0]
    # strong = star-forming-like + high-[NII] (+ unclassifiable, none here)
    assert (work["strong_halpha"] == work["sf_emission"] + work["high_nii_emission"]).all()


def test_manuscript_template_has_no_dead_blocks_or_todo_markers():
    for name in ("paper_template.tex", "online_supplement_template.tex"):
        text = (ROOT / "src" / "paper_template" / name).read_text()
        assert "\\iffalse" not in text
        assert "TODO" not in text
        assert "/Users/" not in text


def test_spectral_index_cache_skips_ids_already_queried(tmp_path, monkeypatch):
    """Ids cached or listed as queried must not trigger an SDSS query."""

    import sys
    import types

    from src import spectral_indices

    cache = tmp_path / "galspecindx.csv"
    pd.DataFrame(
        {
            "specobjid": [11],
            "d4000_n": [1.5],
            "d4000_n_err": [0.01],
            "lick_hd_a": [1.0],
            "lick_hd_a_err": [0.2],
        }
    ).to_csv(cache, index=False)
    (tmp_path / "galspecindx_queried_ids.txt").write_text("22\n")

    calls = []
    fake = types.ModuleType("astroquery.sdss")
    fake.SDSS = types.SimpleNamespace(query_sql=lambda *a, **k: calls.append(a))
    monkeypatch.setitem(sys.modules, "astroquery.sdss", fake)
    rows = spectral_indices.fetch_spectral_indices([11, 22], cache_path=str(cache))
    assert rows["specobjid"].tolist() == [11]
    assert calls == []
