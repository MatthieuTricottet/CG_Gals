"""Regression checks for the submission-stage manuscript QC outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ["CG4", "Control4B", "Control4C", "RG4"]
CONTROLS = ["Control4B", "Control4C", "RG4"]


def _json(path: str) -> dict:
    with open(ROOT / path) as handle:
        return json.load(handle)


def _overlap_coefficient(a: np.ndarray, b: np.ndarray, n_bins: int = 60) -> float:
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    grid = np.linspace(lo, hi, n_bins + 1)
    fa, _ = np.histogram(a, bins=grid, density=True)
    fb, _ = np.histogram(b, bins=grid, density=True)
    return float(np.sum(np.minimum(fa, fb)) * (grid[1] - grid[0]))


def test_compactness_audit_reproduces_values_and_log10_score():
    t9 = _json("referee/values/T9.json")
    summary = t9["compactness"]["summary_by_sample"]
    expected = {
        "CG4": (143, 114, 177),
        "Control4B": (447, 323, 604),
        "Control4C": (293, 187, 436),
        "RG4": (430, 292, 494),
    }
    for sample, (median, q25, q75) in expected.items():
        row = summary[sample]
        assert round(row["median_R_pair_med_kpc"]) == median
        assert round(row["q25_R_pair_med_kpc"]) == q25
        assert round(row["q75_R_pair_med_kpc"]) == q75
        assert row["n_dropped_missing_coordinates_or_redshift"] == 0

    values = pd.read_csv(ROOT / "output/referee/compactness_group_values.csv")
    pair_cols = [c for c in values if c.startswith("R_pair_rank")]
    assert len(pair_cols) == 6
    assert (values["n_members"] == 4).all()
    assert values["objids_rank_order"].map(lambda s: len(set(str(s).split(";")))).eq(4).all()
    assert (values[pair_cols] > 0).all().all()
    assert not any("dist2BGG" in c for c in values.columns)
    assert np.allclose(
        values["compactness_score_neg_log10_Rmed"],
        -np.log10(values["R_pair_med_kpc"]),
    )

    cg4 = values.loc[values["sample"].eq("CG4"), "compactness_score_neg_log10_Rmed"].to_numpy()
    c4c = values.loc[
        values["sample"].eq("Control4C"), "compactness_score_neg_log10_Rmed"
    ].to_numpy()
    threshold = np.quantile(c4c, 0.95)
    frac = (cg4 > threshold).mean()
    assert threshold == pytest.approx(
        t9["compactness"]["cg4_vs_control4c"]["control4c_p95_compactness_score"]
    )
    assert (cg4 > threshold).sum() == 11
    assert frac == pytest.approx(11 / 62)
    assert frac == pytest.approx(
        t9["compactness"]["cg4_vs_control4c"][
            "fraction_cg4_above_control4c_p95_compactness"
        ]
    )
    assert _overlap_coefficient(cg4, c4c) == pytest.approx(0.3385949616849425)


def test_host_bgg_alignment_uses_objids_and_preserves_satellite_status():
    t9 = _json("referee/values/T9.json")
    alignment = t9["host_bgg_alignment"]
    assert alignment["n_total"] == 62
    assert alignment["n_equal"] == 56
    assert alignment["n_different"] == 6
    assert alignment["n_no_reliable_mapping"] == 0
    assert alignment["n_ambiguous_or_duplicate"] == 0
    assert alignment["by_class"]["Predom"]["CG4 BGG differs from Lim-host BGG"] == 6
    assert alignment["n_isolated_trivial_equal"] == 6

    per_group = pd.read_csv(ROOT / "output/referee/host_bgg_alignment_per_group.csv")
    assert per_group["cg4_group"].is_unique
    equal = per_group["mapping_status"].eq("CG4 BGG equals Lim-host BGG")
    different = per_group["mapping_status"].eq("CG4 BGG differs from Lim-host BGG")
    assert (per_group.loc[equal, "cg4_bgg_objid"].astype("int64")
            == per_group.loc[equal, "lim_host_bgg_objid"].astype("int64")).all()
    assert (per_group.loc[different, "cg4_bgg_objid"].astype("int64")
            != per_group.loc[different, "lim_host_bgg_objid"].astype("int64")).all()
    assert set(per_group.loc[different, "zheng_shen_class"]) == {"Predom"}
    assert per_group.loc[per_group["zheng_shen_class"].eq("Isolated"),
                         "isolated_equality_trivial"].all()

    per_gal = pd.read_csv(ROOT / "output/referee/host_bgg_alignment_per_galaxy.csv")
    assert per_gal["objid"].is_unique
    satellites = per_gal[per_gal["quartet_satellite"]]
    assert len(satellites) == 62 * 3 == 186
    assert satellites["host_halo_satellite"].all()
    assert alignment["satellite_status"][
        "n_quartet_rank_satellites_that_are_host_halo_satellites"
    ] == 186
    status_diff = per_gal[per_gal["host_status_differs_from_quartet_rank"]]
    assert len(status_diff) == 6
    assert (status_diff["cg4_rank_M"] == 1).all()


def test_aligned_morphology_sensitivities_are_labelled_and_separate():
    t9 = _json("referee/values/T9.json")
    rows = pd.DataFrame(t9["host_bgg_sensitivity"]["rows"])
    assert len(rows) == 6
    assert set(rows["control"]) == set(CONTROLS)
    assert set(rows["model"]) == {"elliptical_satellites", "spiral_satellites"}
    assert t9["host_bgg_sensitivity"]["n_aligned_cg4_groups"] == 56
    assert t9["host_bgg_sensitivity"]["classification_counts"] == {"Stable": 6}
    assert "Labelled sensitivity only" in t9["host_bgg_sensitivity"]["role"]

    for _, row in rows.iterrows():
        assert row["n_cg4_systems_retained"] == 56
        assert row["n_cg4_satellite_galaxies_raw"] == 56 * 3
        assert row["n_cg4_complete"] == 143
        assert row["sensitivity_family_holm_p"] >= row["aligned_raw_p"]
        assert row["delta_beta_aligned_minus_primary"] == pytest.approx(
            np.log(row["aligned_odds_ratio"]) - np.log(row["primary_odds_ratio"])
        )
        if row["model"] == "elliptical_satellites":
            assert row["aligned_odds_ratio"] > 1
        else:
            assert row["aligned_odds_ratio"] < 1

    c4c_e = rows.query("control == 'Control4C' and model == 'elliptical_satellites'").iloc[0]
    c4c_s = rows.query("control == 'Control4C' and model == 'spiral_satellites'").iloc[0]
    assert c4c_e["aligned_odds_ratio"] == pytest.approx(1.98, abs=0.005)
    assert c4c_s["aligned_odds_ratio"] == pytest.approx(0.51, abs=0.005)

    results = _json("output/results.json")["extended_specialness"]
    assert "host_bgg_sensitivity" not in results
    primary = results["primary_contrasts"]["contrasts"]
    for _, row in rows.iterrows():
        assert row["primary_odds_ratio"] == pytest.approx(
            primary[row["control"]][row["model"]]["cg4_odds_ratio"]
        )


def test_kitagawa_isolated_and_leave_one_outputs():
    additions = _json("output/paper_additions.json")
    quench = additions["quenched_by_morphology"]
    pqe = {sample: quench["per_sample"][sample]["Elliptical"]["p"] for sample in SAMPLES}
    assert pqe == pytest.approx(
        {"CG4": 0.9083333333, "Control4B": 0.9031078611,
         "Control4C": 0.8871841155, "RG4": 0.85}
    )
    assert quench["chi2_homogeneity_PQE"]["p"] == pytest.approx(0.3927829543)
    expected_cond = {
        "Control4B": (-0.0339856315, -0.0857162206, 0.0192449829),
        "Control4C": (-0.0160176243, -0.0683262411, 0.0385176513),
        "RG4": (0.0351211038, -0.0468715131, 0.1210957659),
    }
    for control, (cond, lo, hi) in expected_cond.items():
        item = quench["kitagawa"][control]
        assert item["conditional_term"] == pytest.approx(cond)
        assert item["conditional_ci95"] == pytest.approx([lo, hi])
        assert item["raw_delta_fQ"] == pytest.approx(
            item["mix_term"] + item["conditional_term"]
        )
        assert item["n_boot"] == 4000
        assert item["seed"] == 42
        assert item["blocked_by"].startswith("group within each sample")

    zheng = additions["zheng_shen"]
    assert zheng["per_class"]["Isolated"]["n_groups"] == 6
    assert zheng["per_class"]["Isolated"]["fE_all"]["p"] == pytest.approx(0.389, abs=0.001)
    assert zheng["per_class"]["Isolated"]["fE_sat"]["p"] == pytest.approx(0.286, abs=0.001)
    assert zheng["per_class"]["Embedded"]["fE_sat"]["p"] == pytest.approx(0.562, abs=0.001)
    assert zheng["per_class"]["Predominant"]["fE_sat"]["p"] == pytest.approx(0.638, abs=0.001)
    assert zheng["isolated_satellite_classification_support"][
        "n_isolated_groups_with_classified_satellites"
    ] == 5
    assert zheng["permutations"]["isolated_vs_rest_of_CG4"]["n_perm"] == 20000
    assert zheng["permutations"]["isolated_vs_rest_of_CG4"]["p"] == pytest.approx(0.0227988601)
    assert zheng["permutations"]["isolated_vs_RG4"]["p"] == 1.0
    assert zheng["permutations"]["isolated_vs_Control4C"]["p"] == pytest.approx(0.4007799610)
    loo = zheng["leave_one_isolated_group_out"]
    assert loo["isolated_vs_rest_of_CG4"]["n_omissions"] == 5
    assert loo["isolated_vs_RG4"]["n_omissions"] == 5
    assert len(loo["isolated_vs_rest_of_CG4"]["effect_diff_range"]) == 2
    assert len(loo["isolated_vs_RG4"]["p_range"]) == 2


def test_host_inclusive_tidal_and_integrity_checks():
    additions = _json("output/paper_additions.json")
    tidal = additions["tidal"]
    host = tidal["host_inclusive"]
    assert host["pooled"]["median_delta_dex"] == pytest.approx(0.0088547276)
    assert host["pooled"]["spearman_rho"] == pytest.approx(0.9111478886)
    assert max(v["median_delta_dex"] for v in host["per_sample"].values()) == pytest.approx(
        0.0232324781
    )
    assert host["per_sample"]["RG4"]["max_delta_dex"] == 0
    assert host["per_sample"]["RG4"]["median_extra_members"] == 0
    assert host["n_duplicate_lim_member_rows_dropped"] == 12
    assert tidal["published_inputs"]["residual_or"] == pytest.approx(1.3931206639)
    assert host["refit_elliptical_with_host_T"]["cg4_odds_ratio"] == pytest.approx(
        1.3287910685
    )
    assert "not used to support" in tidal["host_halo_tide_diagnostic"]["note"]

    lim = pd.read_csv(
        ROOT / "data/SDSS(L) galaxy.dat",
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[1, 2],
        names=["objid", "limgroup"],
    )
    dedup_lim = lim.drop_duplicates(["limgroup", "objid"])
    assert len(lim) - len(dedup_lim) == host["n_duplicate_lim_member_rows_dropped"]
    assert not dedup_lim.duplicated(["limgroup", "objid"]).any()

    cg4_ids = set(pd.read_csv(ROOT / "data/CG4_Gals.csv")["objid"].astype("int64"))
    for control in CONTROLS:
        ids = set(pd.read_csv(ROOT / f"data/{control}_Gals.csv")["objid"].astype("int64"))
        assert cg4_ids.isdisjoint(ids)


def test_influence_and_deduplication_policies():
    t9 = _json("referee/values/T9.json")
    influence = t9["influence_checks"]
    assert influence["leave_one_cg4_group"]["n_direction_changes"] == 0
    assert influence["leave_one_lim_host"]["n_direction_changes"] == 0
    assert influence["all_control_objids_unique_within_per_control_models"]
    assert influence["no_cg4_objids_on_control_side"]
    assert influence["all_models_clustered_by_physical_group"]

    cg4_loo = pd.read_csv(ROOT / "output/referee/morphology_influence_leave_one_cg4_group.csv")
    host_loo = pd.read_csv(ROOT / "output/referee/morphology_influence_leave_one_lim_host.csv")
    assert (cg4_loo["odds_ratio"] > 1).all()
    assert (host_loo["odds_ratio"] > 1).all()

    results = _json("output/results.json")["extended_specialness"]
    matched = results["matched_controls"]
    assert matched["control_pool_deduplicated_by_objid"]
    assert matched["n_control_rows_before_dedup"] > matched["n_control_galaxies_unique_pool"]
    template = (ROOT / "src/paper_template/paper_template.tex").read_text()
    assert "without} objid deduplication" in template
    assert "deduplicated by SDSS objid" in template
