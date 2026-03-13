import pandas as pd
import pytest

from src import exploration_ssfr as essfr


def _build_distance_inputs():
    gals = pd.DataFrame(
        [
            {"Group": 1, "z": 0.02, "dist2BGG": 0.0, "rank_M": 1, "sSFR": -11.0},
            {"Group": 1, "z": 0.02, "dist2BGG": 10.0, "rank_M": 2, "sSFR": -10.8},
            {"Group": 1, "z": 0.02, "dist2BGG": 20.0, "rank_M": 3, "sSFR": -10.6},
            {"Group": 1, "z": 0.02, "dist2BGG": 30.0, "rank_M": 4, "sSFR": -10.4},
            {"Group": 1, "z": 0.02, "dist2BGG": 40.0, "rank_M": 5, "sSFR": -10.2},
        ]
    )
    groups = pd.DataFrame([{"Group": 1, "size_Group_Bary_kpc": 100.0}])
    return gals, groups


def test_add_normalized_group_distances_adds_projected_and_scaled_columns():
    gals, groups = _build_distance_inputs()

    enriched = essfr.add_normalized_group_distances(gals, groups)

    assert "dist2BGG_kpc" in enriched.columns
    assert "norm_dist" in enriched.columns
    assert enriched.loc[1, "dist2BGG_kpc"] > 0
    assert enriched.loc[4, "norm_dist"] == pytest.approx(enriched.loc[4, "dist2BGG_kpc"] / 100.0)


def test_compute_distance_correlations_returns_both_distance_metrics(monkeypatch):
    monkeypatch.setattr(essfr.co, "SAMPLE", {"CG4": "cg"})

    gals, groups = _build_distance_inputs()
    sample = {"CG4_Gals": gals, "CG4_Groups": groups}

    correlations = essfr.compute_distance_correlations(sample)

    assert len(correlations) == 2
    assert {row["distance_key"] for row in correlations} == {"dist2BGG_kpc", "norm_dist"}
    for row in correlations:
        assert row["sample"] == "CG4"
        assert row["n"] == 4
        assert row["spearman_rho"] == pytest.approx(1.0)
