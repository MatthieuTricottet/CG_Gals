import json

import numpy as np
import pandas as pd
import pytest

from src.phase_space_segregation import (
    C_KMS,
    cluster_bootstrap_fraction,
    compute_velocity_offsets,
    prepare_phase_space_satellite_sample,
    run_phase_space_segregation_analysis,
    summarize_binned_fractions,
)


def _small_phase_frame():
    rows = []
    for sample, is_cg4, group_offset in [("CG4", 1, 0), ("RG4", 0, 10)]:
        for group in range(4):
            z_bgg = 0.02 + 0.001 * group + 0.0001 * group_offset
            for rank in range(1, 5):
                rows.append(
                    {
                        "sample": sample,
                        "is_CG4": is_cg4,
                        "Group": group + group_offset,
                        "group_uid": f"{sample}:{group + group_offset}",
                        "rank": rank,
                        "is_satellite": int(rank > 1),
                        "is_bgg": int(rank == 1),
                        "logMstar": 10.0 + 0.12 * rank + 0.02 * is_cg4,
                        "z_numeric": z_bgg + 0.0001 * (rank - 1),
                        "z_group_numeric": z_bgg + 0.00015,
                        "quenched": int((rank + is_cg4) % 2 == 0),
                        "elliptical": int(rank <= 2 or is_cg4),
                        "early_type": int(rank <= 2 or is_cg4),
                        "dist2BGG_projected_kpc": 30.0 * (rank - 1),
                        "velocity_dispersion": 150.0,
                    }
                )
    return pd.DataFrame(rows)


def test_velocity_offset_formula_uses_bgg_redshift():
    frame = pd.DataFrame(
        [
            {
                "sample": "CG4",
                "group_uid": "CG4:1",
                "rank": 1,
                "z_numeric": 0.020,
                "z_group_numeric": 0.0205,
                "velocity_dispersion": 200.0,
            },
            {
                "sample": "CG4",
                "group_uid": "CG4:1",
                "rank": 2,
                "z_numeric": 0.021,
                "z_group_numeric": 0.0205,
                "velocity_dispersion": 200.0,
            },
        ]
    )

    result = compute_velocity_offsets(frame)

    expected = C_KMS * (0.021 - 0.020) / (1 + 0.0205)
    assert result.loc[1, "dv_to_bgg"] == pytest.approx(expected)
    assert result.loc[1, "abs_dv_norm"] == pytest.approx(abs(expected) / 200.0)


def test_prepare_phase_space_satellite_sample_excludes_bggs():
    satellites = prepare_phase_space_satellite_sample(_small_phase_frame())

    assert satellites["rank"].min() > 1
    assert satellites["is_bgg"].sum() == 0
    assert set(satellites["distance_bin"]) == {"inner", "middle", "outer"}


def test_binned_fraction_calculation_returns_clustered_summaries():
    satellites = prepare_phase_space_satellite_sample(_small_phase_frame())

    summary = summarize_binned_fractions(
        satellites,
        "distance_bin",
        ["inner"],
        n_boot=50,
        min_total=1,
        min_per_sample=1,
    )

    inner = summary["inner"]["quenched"]
    assert inner["CG4"]["status"] == "ok"
    assert inner["RG4"]["status"] == "ok"
    assert inner["delta_CG4_minus_RG4"]["delta"] is not None


def test_cluster_bootstrap_fraction_returns_finite_uncertainty():
    frame = pd.DataFrame(
        {
            "group_uid": np.repeat(["a", "b", "c", "d"], 3),
            "quenched": [1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
        }
    )

    result = cluster_bootstrap_fraction(frame, "quenched", n_boot=50)

    assert result["status"] == "ok"
    assert np.isfinite(result["stderr"])
    assert result["ci68"][0] <= result["fraction"] <= result["ci68"][1]


def test_phase_space_json_contains_expected_keys(tmp_path):
    result = run_phase_space_segregation_analysis(
        _small_phase_frame(),
        output_dir=tmp_path,
        n_boot=50,
        min_satellites=6,
        min_total_per_bin=1,
        min_per_sample_per_bin=1,
    )

    assert result["status"] == "ok"
    assert "phase_space_segregation" not in result
    assert "metadata" in result
    assert "text_summary" in result
    assert "distance_bin_fractions" in result
    assert "regression_results" in result
    assert result["text_summary"]["cg_satellite_n"] > 0
    assert json.dumps(result, allow_nan=False)
    assert (
        tmp_path / "phase_space_satellite_quenched_fraction_by_distance.pdf"
    ).is_file()
