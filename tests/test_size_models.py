import json

import numpy as np
import pandas as pd

from src.size_analysis import run_size_analysis

INJECTED_DELTA = -0.08
SCATTER_DEX = 0.18


def synthetic_size_frame(seed=20260612, n_cg4_groups=60, n_control_groups=150):
    """Four-member groups with a known CG4 size offset at fixed mass."""

    rng = np.random.default_rng(seed)
    members = 4
    n_groups = n_cg4_groups + n_control_groups
    n = n_groups * members
    group_number = np.repeat(np.arange(n_groups), members)
    is_cg4 = np.repeat(np.arange(n_groups) < n_cg4_groups, members)
    control_label = np.repeat(
        rng.choice(["Control4B", "Control4C", "RG4"], n_groups), members
    )
    sample = np.where(is_cg4, "CG4", control_label)
    rank = np.tile(np.arange(1, members + 1), n_groups)

    log_mass = rng.normal(10.4, 0.5, n)
    z_group = np.repeat(rng.uniform(0.01, 0.045, n_groups), members)
    z = z_group + rng.normal(0, 0.0003, n)
    log_group_luminosity = np.repeat(rng.normal(11.0, 0.3, n_groups), members)
    velocity_dispersion = np.repeat(rng.uniform(100, 500, n_groups), members)

    mass_term = 0.30 * (log_mass - 10.4) + 0.04 * (log_mass - 10.4) ** 2
    log_size = (
        0.45 + mass_term + INJECTED_DELTA * is_cg4 + rng.normal(0, SCATTER_DEX, n)
    )

    base_ra = np.repeat(rng.uniform(120, 240, n_groups), members)
    base_dec = np.repeat(rng.uniform(-5, 60, n_groups), members)
    ra = base_ra + rng.uniform(-0.03, 0.03, n)
    dec = base_dec + rng.uniform(-0.03, 0.03, n)

    elliptical = rng.binomial(1, 0.4, n).astype(float)
    uncertain = rng.random(n) < 0.15
    elliptical[uncertain] = np.nan
    spiral = np.where(np.isnan(elliptical), np.nan, 1 - elliptical)

    frame = pd.DataFrame(
        {
            "sample": sample,
            "is_CG4": is_cg4.astype(int),
            "group_uid": [f"{s}:{g}" for s, g in zip(sample, group_number)],
            "objid": np.arange(n) + 1237650000000000000,
            "rank": rank.astype(float),
            "is_satellite": (rank > 1).astype(float),
            "is_bgg": (rank == 1).astype(float),
            "logMstar": log_mass,
            "M_r": -20.5 - 2.0 * (log_mass - 10.4) + rng.normal(0, 0.2, n),
            "z_numeric": z,
            "z": z,
            "RA": ra,
            "Dec": dec,
            "dist2BGG_kpc": rng.uniform(10, 400, n),
            "log_group_luminosity": log_group_luminosity,
            "velocity_dispersion": velocity_dispersion,
            "elliptical": elliptical,
            "spiral": spiral,
            "log_Rchl_r_kpc": log_size,
            "log_petroR50_kpc": log_size - 0.05 + rng.normal(0, 0.05, n),
            "concentration_r90_r50": rng.normal(2.8, 0.35, n),
            "simard_ng": rng.uniform(0.8, 5.5, n),
            "psfWidth_r": rng.normal(1.2, 0.1, n),
            "size_ok_simard": 1.0,
            "size_ok_petro": 1.0,
        }
    )
    frame.attrs["size_attach_audit"] = {
        "per_sample": {
            name: {
                "n_rows": int((frame["sample"] == name).sum()),
                "petro_row_resolved": int((frame["sample"] == name).sum()),
                "dr7_bridge_resolved": int((frame["sample"] == name).sum()),
                "simard_matched": int((frame["sample"] == name).sum()),
                "z_mismatch": 0,
                "shred_merge": 0,
                "n_pegged": 0,
                "simard_out_of_window": 0,
                "petro_out_of_window": 0,
                "size_ok_simard": int((frame["sample"] == name).sum()),
                "size_ok_petro": int((frame["sample"] == name).sum()),
            }
            for name in ["CG4", "Control4B", "Control4C", "RG4"]
        }
    }
    return frame


def test_recovers_injected_offset_within_its_confidence_interval():
    frame = synthetic_size_frame()
    result = run_size_analysis(frame, output_dir=None)
    assert result["status"] == "ok"
    fit = result["adjusted"]["all"]
    assert fit["status"] == "ok"
    assert fit["ci_low"] <= INJECTED_DELTA <= fit["ci_high"]
    assert result["verdicts"]["direction"] == "smaller"
    assert result["verdicts"]["primary_all_significant"] is True


def test_holm_family_sizes_match_preregistration():
    frame = synthetic_size_frame()
    result = run_size_analysis(frame, output_dir=None)
    families = result["holm_families"]
    assert len(families["F1"]) == 3
    assert len(families["F2"]) == 3
    assert len(families["F3"]) == 3
    # F1: the three primary variants carry Holm-adjusted p-values.
    for variant in ["all", "satellites", "bgg"]:
        assert result["adjusted"][variant]["p_holm"] is not None
    # F2: Petrosian all/satellites plus concentration satellites.
    assert result["petrosian"]["all"]["p_holm"] is not None
    assert result["petrosian"]["satellites"]["p_holm"] is not None
    assert result["concentration"]["satellites"]["p_holm"] is not None
    assert "p_holm" not in result["concentration"]["all"]
    # F3: the three matched size outcomes.
    for outcome in [
        "delta_log_Rchl_r",
        "delta_log_petroR50",
        "delta_concentration",
    ]:
        effect = result["matched"]["effects"][outcome]
        assert effect["status"] == "ok"
        assert effect["p_holm"] is not None


def test_two_runs_produce_byte_identical_json():
    first = run_size_analysis(synthetic_size_frame(), output_dir=None)
    second = run_size_analysis(synthetic_size_frame(), output_dir=None)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_skips_cleanly_without_size_columns(monkeypatch):
    frame = synthetic_size_frame().drop(
        columns=[
            "log_Rchl_r_kpc",
            "log_petroR50_kpc",
            "concentration_r90_r50",
            "size_ok_simard",
            "size_ok_petro",
        ]
    )

    def failing_attach(_frame):
        raise RuntimeError("offline")

    from src import size_analysis as module

    monkeypatch.setattr(module, "attach_size_columns", failing_attach)
    result = run_size_analysis(frame, output_dir=None)
    assert result["status"] == "skipped"
    assert result["reason"] == "size_data_unavailable"
