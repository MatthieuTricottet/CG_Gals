import numpy as np
import pandas as pd

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
    assert first_group["f_L_BGG_source"] == "Lum_BGG/Lum_group"
    assert audit["n_groups_invalid_f_L_BGG"] == 0
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
