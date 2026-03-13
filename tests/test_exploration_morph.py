import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from src import exploration_morph as emorph


def _build_morphology_fixture():
    rows = []
    specs = [
        (1, True, "Spiral", 10.1, ["Spiral", "Spiral", "Elliptical"]),
        (2, True, "Spiral", 10.3, ["Spiral", "Elliptical", "Spiral"]),
        (3, False, "Spiral", 10.5, ["Elliptical", "Spiral", "Elliptical"]),
        (4, False, "Spiral", 10.7, ["Spiral", "Spiral", "Elliptical"]),
        (5, True, "Elliptical", 10.2, ["Elliptical", "Elliptical", "Spiral"]),
        (6, True, "Elliptical", 10.4, ["Elliptical", "Spiral", "Elliptical"]),
        (7, False, "Elliptical", 10.6, ["Elliptical", "Elliptical", "Spiral"]),
        (8, False, "Elliptical", 10.8, ["Spiral", "Elliptical", "Elliptical"]),
    ]

    groups = []
    for group, is_dominated, bgg_morph, lgm_bgg, satellites in specs:
        rows.append({"Group": group, "rank_M": 1, "morphology": bgg_morph, "lgm": lgm_bgg})
        groups.append({"Group": group, "is_dominated": is_dominated})
        for rank, morph in enumerate(satellites, start=2):
            rows.append(
                {
                    "Group": group,
                    "rank_M": rank,
                    "morphology": morph,
                    "lgm": 9.5 + 0.1 * rank + 0.01 * group,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(groups)


def test_clean_morph_keeps_only_secure_labels():
    series = pd.Series(["Spiral", "Elliptical", "Uncertain", "Lenticular"])

    cleaned = emorph.clean_morph(series)

    assert cleaned.tolist()[:2] == ["Spiral", "Elliptical"]
    assert cleaned.isna().tolist()[2:] == [True, True]


def test_attach_dom_from_group_table_maps_group_flags():
    gals = pd.DataFrame({"Group": [1, 1, 2, 3], "rank_M": [1, 2, 1, 1]})
    groups = pd.DataFrame({"Group": [1, 2], "is_dominated": [True, False]})

    attached = emorph.attach_dom_from_group_table(gals, groups)

    assert attached["is_dominated"].dtype.name == "boolean"
    assert attached["is_dominated"].tolist() == [True, True, False, pd.NA]


def test_group_and_satellite_regressions_return_finite_effects():
    galaxy_df, _ = _build_morphology_fixture()

    group_result = emorph.group_level_binom(galaxy_df)
    satellite_result = emorph.satellite_level_cluster(galaxy_df)

    assert group_result["n_groups_used"] == 8
    assert group_result["OR"] > 0
    assert 0 <= group_result["p_glm"] <= 1
    assert group_result["error"] is None

    assert satellite_result["n_sat_used"] == 24
    assert satellite_result["n_groups_used"] == 8
    assert satellite_result["OR"] > 0
    assert 0 <= satellite_result["p_cluster"] <= 1
    assert satellite_result["error"] is None


@pytest.mark.filterwarnings("ignore:Perfect separation or prediction detected.*:statsmodels.tools.sm_exceptions.PerfectSeparationWarning")
def test_build_domination_results_returns_one_row_per_domination_class(monkeypatch):
    monkeypatch.setattr(emorph.co, "SAMPLE", {"CG4": "cg"})

    galaxy_df, group_df = _build_morphology_fixture()
    sample = {"CG4_Gals": galaxy_df, "CG4_Groups": group_df}

    results = emorph.build_domination_results(sample, min_groups=4)

    assert set(results["dom"]) == {"dominated", "non_dominated"}
    assert set(results["G_n_groups"]) == {4}
    assert results["error"].isna().all()
    assert np.isfinite(results["S_OR"]).all()
