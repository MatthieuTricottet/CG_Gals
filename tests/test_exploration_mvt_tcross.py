import pandas as pd
import pytest

from src import exploration_mvt_tcross as tcross


def test_build_group_property_frame_adds_log_columns_and_classes():
    sample = {
        "CG4_Groups": pd.DataFrame(
            [
                {"t_cr": 0.5, "M_virial": 1.0e12, "Lum_group": 5.0e10, "M_virial_over_L": 20.0, "Class": "Compact"},
                {"t_cr": 1.0, "M_virial": 2.0e12, "Lum_group": 6.0e10, "M_virial_over_L": 30.0, "Class": "Loose"},
            ]
        ),
        "RG4_Groups": pd.DataFrame(
            [
                {"t_cr": 0.8, "M_virial": 1.5e12, "Lum_group": 7.0e10, "M_virial_over_L": 25.0},
            ]
        ),
    }

    frame = tcross.build_group_property_frame(sample)

    assert {"lg_t_cr", "lg_M_virial", "lg_Lum_group", "lg_M_virial_over_L"} <= set(frame.columns)
    assert "Compact" in frame["Class"].values
    assert r"RG$_4$" in frame["Class"].values


def test_add_group_ssfr_excess_summary_merges_group_medians():
    sample = {
        "CG4_Gals": pd.DataFrame(
            [
                {"Group": 1, "sSFR_excess": 0.2, "sSFR_status": "Starforming"},
                {"Group": 1, "sSFR_excess": float("nan"), "sSFR_status": "NosSFR"},
                {"Group": 2, "sSFR_excess": -0.2, "sSFR_status": "Quenched"},
                {"Group": 2, "sSFR_excess": 0.0, "sSFR_status": "Quenched"},
            ]
        ),
        "CG4_Groups": pd.DataFrame([{"Group": 1}, {"Group": 2}]),
    }

    enriched = tcross.add_group_ssfr_excess_summary(sample)
    groups = enriched["CG4_Groups"].set_index("Group")

    assert groups.loc[1, "sSFR_excess_median"] == pytest.approx(0.2)
    assert groups.loc[2, "sSFR_excess_median"] == pytest.approx(-0.1)
    assert bool(groups.loc[1, "has_nossfr"]) is True
    assert bool(groups.loc[2, "has_nossfr"]) is False


def test_compare_main_sequence_offset_by_sample_sets_direction_against_cg(monkeypatch):
    monkeypatch.setattr(tcross.co, "SAMPLE", {"CG4": "cg", "RG4": "rg"})

    sample = {
        "CG4_Gals": pd.DataFrame(
            {
                "sSFR_MS_offset": [0.1, 0.2, 0.3],
                "sSFR_status": ["Starforming", "Starforming", "Starforming"],
            }
        ),
        "RG4_Gals": pd.DataFrame(
            {
                "sSFR_MS_offset": [0.4, 0.5, 0.6],
                "sSFR_status": ["Starforming", "Starforming", "Starforming"],
            }
        ),
    }

    result = tcross.compare_main_sequence_offset_by_sample(sample).set_index("sample")

    assert pd.isna(result.loc["CG4", "wilcoxon_p_value_vs_CG4"])
    assert result.loc["RG4", "alternative"] == "greater"
    assert 0 <= result.loc["RG4", "wilcoxon_p_value_vs_CG4"] <= 1


def test_compute_global_tcross_correlations_returns_all_three_relations():
    frame = pd.DataFrame(
        {
            "lg_t_cr": [0.0, 0.5, 1.0, 1.5],
            "lg_M_virial_over_L": [1.0, 1.5, 2.0, 2.5],
            "lg_M_virial": [11.0, 11.5, 12.0, 12.5],
            "lg_Lum_group": [10.0, 10.5, 11.0, 11.5],
        }
    )

    rows = tcross.compute_global_tcross_correlations(frame)

    assert {row["y_key"] for row in rows} == {
        "lg_M_virial_over_L",
        "lg_M_virial",
        "lg_Lum_group",
    }
    assert all(row["rho"] == pytest.approx(1.0) for row in rows)
