"""GZ1 DR7 recovery and the 20 November 2026 referee freeze.

The pipeline applies ``src/gz1_recovery.py`` at load time.  These tests check
that it reproduces the frozen labels row by row, that the published Table 4
and Table 5 contrasts agree with the freeze CSVs, and that the values the
manuscript cites from the freeze are read without modification.
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FREEZE = ROOT / "results" / "referee_freeze_20261120" / "tables"
pytestmark = pytest.mark.skipif(
    not FREEZE.is_dir(), reason="referee freeze tables not available"
)
LABELS = {"Elliptical": "E", "Spiral": "S", "Uncertain": "U", "NoGZ": "missing"}


@pytest.fixture(scope="module")
def recovered():
    import gz1_recovery

    with open(ROOT / "data" / "processed_sample.pkl", "rb") as handle:
        sample = pickle.load(handle)
    return gz1_recovery.apply_to_sample(sample)


def test_recovery_reproduces_frozen_labels(recovered):
    frozen = pd.read_csv(FREEZE / "01_frozen_galaxy_labels.csv", dtype={"dr7objid": str})
    for name in ["CG4", "Control4B", "Control4C", "RG4"]:
        mine = recovered[name + "_Gals"][["objid", "morphology"]].copy()
        mine["label"] = mine["morphology"].map(LABELS)
        mine["objid"] = mine["objid"].astype("int64")
        ref = frozen.loc[frozen["sample"].eq(name), ["objid", "frozen_label"]].copy()
        ref["objid"] = ref["objid"].astype("int64")
        joined = mine.merge(ref, on="objid", how="outer", validate="1:1")
        assert joined["label"].notna().all() and joined["frozen_label"].notna().all()
        assert (joined["label"] == joined["frozen_label"]).all(), name


def test_recovery_counts(recovered):
    import gz1_recovery

    summary = gz1_recovery.summary(recovered)["samples"]
    assert summary["CG4"]["recovered"] == {"E": 6, "S": 10, "U": 1, "total": 17}
    assert summary["CG4"]["final"] == {"E": 130, "S": 96, "U": 8, "missing": 14}
    assert summary["CG4"]["n_dr16_without_verified_archive"] == 0
    assert [summary[s]["n_dr16_without_verified_archive"] for s in
            ["Control4B", "Control4C", "RG4"]] == [11, 9, 3]


def test_pipeline_contrasts_match_freeze():
    results = json.loads((ROOT / "output" / "results.json").read_text())
    es = results["extended_specialness"]
    primary = pd.read_csv(FREEZE / "02_frozen_primary_and_compactness.csv", low_memory=False)
    frozen = primary[primary["labelset"].eq("frozen")]
    table4 = frozen[
        frozen["analysis"].eq("Table 4") & frozen["row"].eq("coefficient")
        & frozen["term"].eq("cg") & frozen["sigma"].astype(str).eq("True")
    ]
    for _, row in table4.iterrows():
        fit = es["primary_contrasts"]["contrasts"][row["control"]][
            "elliptical_" + ("all" if row["scope"] == "all" else "satellites")
        ]
        assert fit["cg4_odds_ratio"] == pytest.approx(row["OR"], rel=1e-6)
        assert fit["n"] == row["n"]
    table5 = frozen[frozen["analysis"].eq("Table 5")]
    per_control = es["matched_controls"]["group_level_per_control"]
    for control in ["Control4B", "Control4C", "RG4"]:
        row = table5[table5["control"].eq(control)].iloc[0]
        fit = per_control[control]
        assert 100 * fit["delta_smooth_satellite_fraction"] == pytest.approx(row["difference"])
        assert 100 * fit["ci95"][0] == pytest.approx(row["low"])
        assert 100 * fit["ci95"][1] == pytest.approx(row["high"])
        assert fit["p_permutation"] == pytest.approx(row["permutation_p"])
        assert fit["n_matched_groups"] == row["matched_pairs"]


def test_freeze_values_cited_in_manuscript():
    import freeze_values

    values = freeze_values.load()
    compact = values["compactness"]
    assert compact["full"]["difference"] == pytest.approx(14.408, abs=1e-3)
    assert compact["separate_slopes"]["difference"] == pytest.approx(3.364, abs=1e-3)
    assert compact["matched"]["difference"] == pytest.approx(-0.417, abs=1e-3)
    assert compact["matched"]["labelled_pairs"] == 40
    assert compact["window_quartets"] == {"CG4": 48, "Control4C": 187}
    assert [round(x) for x in compact["window_kpc"]] == [95, 216]
    mid = values["spectral"]["intermediate_mass"]
    assert mid["exact_common"]["difference"] == pytest.approx(-16.007, abs=1e-3)
    assert mid["exact_aperture_kauffmann"]["difference"] == pytest.approx(-9.523, abs=1e-3)
    assert (mid["exact_aperture_kauffmann"]["cg_n"],
            mid["exact_aperture_kauffmann"]["control_n"]) == (29, 94)
    assert values["spectral"]["n_singular_fits"] == 16
    assert values["galaxy_match"]["all"] == {
        "matched_pairs": 234, "decisive_pairs": 207, "decisive_pairs_original": 180
    }
    assert values["outer_bin"]["n"] == 13 and values["outer_bin"]["hosts"] == 5
