import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.agn_environment import run_agn_environment_analysis
from src.extended_stats import safe_json
from src.fossilness import run_fossilness_analysis
from src.matched_controls import run_matched_control_analysis
from src.phase_space import run_phase_space_analysis
from src.recent_quenching import run_recent_quenching_analysis
from src.selection_diagnostics import run_selection_diagnostics
from src.specialness_models import fit_logistic_specialness_models
from src.tidal_indices import run_tidal_indices_analysis


def synthetic_frame(seed=20260612):
    rng = np.random.default_rng(seed)
    n_groups = 40
    members = 4
    n = n_groups * members
    group_number = np.repeat(np.arange(n_groups), members)
    is_cg4_group = np.repeat(np.arange(n_groups) < 10, members)
    sample = np.where(is_cg4_group, "CG4", "RG4")
    rank = np.tile(np.arange(1, members + 1), n_groups)
    log_mass = rng.normal(10.5 + 0.1 * is_cg4_group, 0.45, n)
    z_group = np.repeat(rng.uniform(0.02, 0.045, n_groups), members)
    z = z_group + rng.normal(0, 0.00035, n)
    passive_probability = 1 / (
        1 + np.exp(-(-0.4 + 0.6 * is_cg4_group + 0.7 * (log_mass - 10.5)))
    )
    passive = rng.binomial(1, passive_probability)
    elliptical = rng.binomial(
        1, np.clip(0.25 + 0.2 * is_cg4_group + 0.2 * passive, 0.05, 0.9)
    )
    base_ra = np.repeat(rng.uniform(150, 151, n_groups), members)
    base_dec = np.repeat(rng.uniform(1, 2, n_groups), members)
    h_alpha = np.where(passive == 1, rng.normal(-1, 0.5, n), rng.normal(-15, 5, n))
    log_nii = rng.normal(-0.35 + 0.3 * passive, 0.25, n)
    log_oiii = rng.normal(-0.1 + 0.25 * passive, 0.25, n)
    frame = pd.DataFrame(
        {
            "sample": sample,
            "is_CG4": is_cg4_group.astype(int),
            "Group": group_number,
            "group_uid": [f"{s}:{g}" for s, g in zip(sample, group_number)],
            "rank": rank,
            "is_satellite": (rank > 1).astype(int),
            "is_bgg": (rank == 1).astype(int),
            "logMstar": log_mass,
            "z_numeric": z,
            "z_group_numeric": z_group,
            "passive": passive,
            "starforming": 1 - passive,
            "elliptical": elliptical,
            "spiral": 1 - elliptical,
            "R_norm": rng.uniform(0.05, 2.0, n),
            "V_norm": np.abs(rng.normal(0.8, 0.45, n)),
            "dist2BGG_kpc": rng.uniform(5, 350, n),
            "R_scale": rng.uniform(120, 300, n),
            "velocity_dispersion": rng.uniform(80, 350, n),
            "log_group_mass": np.repeat(rng.normal(12.7, 0.4, n_groups), members),
            "log_group_luminosity": np.repeat(rng.normal(11.0, 0.2, n_groups), members),
            "dominance": np.repeat(rng.uniform(0.35, 0.8, n_groups), members),
            "M_r": -19.5 - 0.8 * (4 - rank) + rng.normal(0, 0.15, n),
            "MS_res": rng.normal(0, 0.2, n),
            "RA": base_ra + rng.normal(0, 0.015, n),
            "Dec": base_dec + rng.normal(0, 0.015, n),
            "h_alpha_eqw": h_alpha,
            "h_beta_eqw": h_alpha / 3,
            "oiii_5007_eqw": h_alpha / 4,
            "nii_6584_eqw": h_alpha / 5,
            "log_NII_Ha": log_nii,
            "log_OIII_Hb": log_oiii,
            "is_AGN": (log_nii > -0.1),
            "u_minus_r": rng.normal(2.0 + 0.3 * passive, 0.2, n),
            "u_minus_g": rng.normal(1.2 + 0.2 * passive, 0.15, n),
            "g_minus_r": rng.normal(0.8 + 0.1 * passive, 0.1, n),
            "r_minus_i": rng.normal(0.4 + 0.05 * passive, 0.08, n),
        }
    )
    frame.loc[
        frame.index % 5 == 0, ["u_minus_r", "u_minus_g", "g_minus_r", "r_minus_i"]
    ] = np.nan
    return frame


def test_safe_json_is_strictly_serializable():
    payload = safe_json(
        {
            "finite": np.float64(1.2),
            "missing": np.nan,
            "array": np.array([1, 2]),
            "flag": np.bool_(True),
        }
    )
    json.dumps(payload, allow_nan=False)
    assert payload["missing"] is None


def test_all_modules_skip_gracefully_with_missing_columns():
    empty = pd.DataFrame()
    functions = [
        fit_logistic_specialness_models,
        run_matched_control_analysis,
        run_phase_space_analysis,
        run_fossilness_analysis,
        run_recent_quenching_analysis,
        run_agn_environment_analysis,
        run_tidal_indices_analysis,
        run_selection_diagnostics,
    ]
    for function in functions:
        assert function(empty)["status"] == "skipped"


def test_group_scale_availability_does_not_require_absent_mass():
    frame = synthetic_frame()
    frame["log_group_mass"] = np.nan

    result = run_selection_diagnostics(frame)

    assert result["availability_by_sample"]["CG4"]["group_scale_quantities"] == 1.0
    assert (
        "log_group_mass"
        in result["group_scale_column_audit"]["missing_or_sparse_columns"]
    )


def test_all_modules_execute_on_synthetic_data_and_create_figures(tmp_path):
    frame = synthetic_frame()
    results = {
        "specialness_models": fit_logistic_specialness_models(frame, tmp_path),
        "matched_controls": run_matched_control_analysis(frame, tmp_path, n_boot=100),
        "phase_space": run_phase_space_analysis(frame, tmp_path),
        "fossilness": run_fossilness_analysis(frame, tmp_path),
        "recent_quenching": run_recent_quenching_analysis(frame, tmp_path),
        "agn_environment": run_agn_environment_analysis(frame, tmp_path),
        "tidal_indices": run_tidal_indices_analysis(frame, tmp_path),
        "selection_diagnostics": run_selection_diagnostics(frame, tmp_path),
    }
    assert all(result["status"] in {"ok", "limited"} for result in results.values())
    json.dumps(safe_json(results), allow_nan=False)

    required_figures = [
        "fig_specialness_logistic_coefficients.pdf",
        "fig_matched_control_effects.pdf",
        "fig_matched_control_balance.pdf",
        "phase_space_satellite_passive_fraction_by_distance.pdf",
        "phase_space_satellite_earlytype_fraction_by_distance.pdf",
        "phase_space_satellite_passive_fraction_projected_phase_space.pdf",
        "phase_space_mass_redshift_balance.pdf",
        "fig_magnitude_gap_comparison.pdf",
        "fig_recent_quenching_diagnostics.pdf",
        "fig_agn_fraction_by_sample.pdf",
        "fig_tidal_index_outcomes.pdf",
        "fig_data_availability_by_sample.pdf",
        "fig_colour_matched_selection_bias.pdf",
    ]
    for filename in required_figures:
        assert (Path(tmp_path) / filename).is_file()
