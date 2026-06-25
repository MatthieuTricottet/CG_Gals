import math

import numpy as np
import pandas as pd
import pytest

from src.morphology_dominance import (
    compute_group_dominance_variables,
    run_morphology_dominance_analysis,
    summarize_morphology_by_class,
    test_morphology_vs_continuous_dominance as morphology_vs_continuous,
)


def _dominance_fixture(n_groups=16):
    rows = []
    class_cycle = ["Embedded", "Isolated", "Predom", "Embedded"]
    for group in range(n_groups):
        sample = "CG4" if group < n_groups // 2 else "RG4"
        class_name = class_cycle[group % len(class_cycle)] if sample == "CG4" else pd.NA
        frac = 0.42 + 0.02 * (group % 6)
        lum_group = 100.0 + group
        lum_bgg = frac * lum_group
        for rank in range(1, 5):
            morphology = "Elliptical" if (group + 2 * rank + (rank == 1)) % 3 == 0 else "Spiral"
            if group == 0 and rank == 4:
                morphology = "Uncertain"
            rows.append(
                {
                    "sample": sample,
                    "Group": group,
                    "group_uid": f"{sample}:{group}",
                    "rank": rank,
                    "is_bgg": float(rank == 1),
                    "is_satellite": float(rank > 1),
                    "M_r": -22.0 + 0.75 * (rank - 1) + 0.02 * group,
                    "Lum": lum_bgg if rank == 1 else (lum_group - lum_bgg) / 3.0,
                    "Lum_BGG": lum_bgg,
                    "Lum_group": lum_group,
                    "FracLumBGG": frac,
                    "Class": class_name,
                    "morphology": morphology,
                    "logMstar": 10.0 + 0.15 * rank + 0.01 * group,
                    "z_numeric": 0.02 + 0.0005 * group,
                    "elliptical": float(morphology == "Elliptical") if morphology != "Uncertain" else np.nan,
                    "spiral": float(morphology == "Spiral") if morphology != "Uncertain" else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_compute_group_dominance_uses_ranked_magnitudes_and_luminosities():
    frame = _dominance_fixture(4)

    galaxies, groups, audit = compute_group_dominance_variables(frame)

    first_group = groups.loc[groups["group_uid"].eq("CG4:0")].iloc[0]
    assert first_group["Delta_m12"] == 0.75
    assert first_group["Delta_m12_source"] == "rank_1_2_M_r"
    assert first_group["f_L_BGG"] == first_group["f_L_BGG_raw"]
    # f_L_BGG is now reused from the existing FracLumBGG column rather than
    # recomputed, and must match the Lum_BGG/Lum_group recomputation it replaces.
    assert first_group["f_L_BGG_source"] == "FracLumBGG"
    fixture_row = frame.loc[frame["group_uid"].eq("CG4:0")].iloc[0]
    recomputed_fraction = fixture_row["Lum_BGG"] / fixture_row["Lum_group"]
    assert first_group["f_L_BGG_raw"] == pytest.approx(recomputed_fraction, abs=1e-9)
    assert first_group["f_L_BGG_raw"] == pytest.approx(fixture_row["FracLumBGG"], abs=1e-9)
    assert audit["n_groups_invalid_f_L_BGG"] == 0
    # The gap reuse cross-checks the recomputed Delta_m12 against the catalogue.
    assert audit["n_groups_gap_crosschecked_against_DeltaR12"] >= 0
    assert {"Delta_m12", "f_L_BGG"}.issubset(galaxies.columns)


def test_class_summary_reports_missing_rg4_classes_and_excluded_morphology():
    frame, _, _ = compute_group_dominance_variables(_dominance_fixture(8))

    cg4 = summarize_morphology_by_class(frame, "CG4", "all")
    rg4 = summarize_morphology_by_class(frame, "RG4", "all")

    assert [row["class"] for row in cg4["contingency_table"]][:4] == [
        "Split",
        "Isolated",
        "Predominant",
        "Embedded",
    ]
    assert cg4["n_excluded_morphology_not_E_or_Sp"] == 1
    assert rg4["n_galaxies_used"] == 0
    assert rg4["n_excluded_class_missing"] > 0
    assert rg4["test"]["status"] == "skipped"


def test_continuous_models_return_guarded_logistic_results():
    frame, _, _ = compute_group_dominance_variables(_dominance_fixture(20))

    result = morphology_vs_continuous(frame, "CG4", "all", "Delta_m12")

    assert result["n_complete_for_descriptive"] > 0
    assert set(result["descriptive_by_morphology"]) >= {"Elliptical", "Spiral"}
    assert result["models"]["unadjusted"]["status"] in {"ok", "skipped"}
    if result["models"]["unadjusted"]["status"] == "ok":
        assert result["models"]["unadjusted"]["odds_ratio"] > 0
        assert 0 <= result["models"]["unadjusted"]["p_value"] <= 1


def test_run_morphology_dominance_analysis_is_json_safe(tmp_path, monkeypatch):
    monkeypatch.setattr("src.morphology_dominance.co.OUTPUT_PATH", str(tmp_path) + "/")

    result = run_morphology_dominance_analysis(_dominance_fixture(20), tmp_path)

    assert result["status"] == "ok"
    assert result["overall_result_sentence"]
    assert result["summary"]["main_conclusion"] == result["overall_result_sentence"]
    assert result["class_association"]["CG4"]["all"]["n_galaxies_used"] > 0
    assert result["class_association"]["CG4"]["all"]["interpretation_flag"] in {
        "significant",
        "marginal",
        "not_significant",
        "inconclusive",
    }
    assert result["class_association"]["RG4"]["all"]["n_excluded_class_missing"] > 0
    assert result["dominance_audit"]["n_groups_missing_Delta_m12"] == 0
    assert (tmp_path / "morphology_dominance_class_counts.csv").is_file()


def test_f_L_BGG_odds_ratio_reported_per_tenth():
    frame, _, _ = compute_group_dominance_variables(_dominance_fixture(20))

    result = morphology_vs_continuous(frame, "CG4", "all", "f_L_BGG")

    fitted = [
        model
        for model in result["models"].values()
        if model.get("status") == "ok"
    ]
    assert fitted, "expected at least one fitted f_L_BGG model"
    for model in fitted:
        odds_ratio = model["odds_ratio"]
        assert math.isfinite(odds_ratio)
        # The reported odds ratio is per 0.1 increase in the fraction, so it must
        # stay in a sane range rather than exploding on the native [0, 1] scale.
        assert 0.1 <= odds_ratio <= 10.0
        assert model["or_report_scale"] == pytest.approx(0.1)
        assert "0.1" in model["or_report_unit"]
        # exp(0.1 * beta) defines the reported OR; exp(beta) the per-unit raw OR.
        assert odds_ratio == pytest.approx(math.exp(0.1 * model["beta1"]), rel=1e-9)
        assert model["odds_ratio_raw"] == pytest.approx(
            math.exp(model["beta1"]), rel=1e-9
        )


def test_delta_m12_odds_ratio_reported_per_magnitude():
    frame, _, _ = compute_group_dominance_variables(_dominance_fixture(20))

    result = morphology_vs_continuous(frame, "CG4", "all", "Delta_m12")

    for model in result["models"].values():
        if model.get("status") != "ok":
            continue
        assert model["or_report_scale"] == pytest.approx(1.0)
        # Per-magnitude odds ratios are left untouched: OR == exp(beta).
        assert model["odds_ratio"] == pytest.approx(math.exp(model["beta1"]), rel=1e-9)
        assert model["odds_ratio"] == pytest.approx(model["odds_ratio_raw"], rel=1e-9)


def test_multiple_testing_block_has_bh_adjusted_p_values(tmp_path):
    result = run_morphology_dominance_analysis(_dominance_fixture(20), tmp_path)

    multiple_testing = result["multiple_testing"]
    assert multiple_testing["method"] == "Benjamini-Hochberg FDR"
    assert multiple_testing["n_tests"] > 0
    assert "n_tests_surviving_fdr" in multiple_testing
    assert multiple_testing["tests"], "expected the FDR family to be populated"
    for test in multiple_testing["tests"]:
        assert "p_value_bh" in test
        assert test["flag_bh"] in {
            "significant",
            "marginal",
            "not_significant",
            "inconclusive",
        }
    # The adjustment is written back onto the focal continuous models too.
    gap_models = result["continuous_association"]["Delta_m12"]["CG4"]["all"]["models"]
    preferred = (
        gap_models["adjusted"]
        if gap_models["adjusted"].get("status") == "ok"
        else gap_models["unadjusted"]
    )
    assert "p_value_bh" in preferred


def test_class_counts_csv_marks_rg4_as_unavailable(tmp_path):
    run_morphology_dominance_analysis(_dominance_fixture(20), tmp_path)

    counts = pd.read_csv(tmp_path / "morphology_dominance_class_counts.csv")
    rg4 = counts.loc[counts["sample"] == "RG4"]
    # RG4 has no class labels, so it must collapse to a single marker row rather
    # than one all-zero row per class and subset.
    assert len(rg4) == 1
    assert "unavailable" in str(rg4.iloc[0]["note"]).lower()

    cg4_split = counts.loc[(counts["sample"] == "CG4") & (counts["class"] == "Split")]
    assert not cg4_split.empty
    assert (cg4_split["note"].str.contains("construction")).all()
