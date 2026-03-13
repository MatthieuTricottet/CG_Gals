import numpy as np
import pandas as pd
import pytest

from src import exploration_dom as dom


def test_bh_fdr_matches_expected_adjusted_p_values():
    adjusted = dom._bh_fdr(np.array([0.04, 0.002, 0.03, 0.2]))

    assert adjusted == pytest.approx([0.0533333333, 0.008, 0.0533333333, 0.2])


def test_compute_pair_pvals_respects_minimum_pair_count():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5]})

    p_value, rho, n_points = dom._compute_pair_pvals(df, "x", "y", min_n_pair=3)
    assert n_points == 5
    assert rho == pytest.approx(1.0)
    assert p_value < 0.05

    p_value, rho, n_points = dom._compute_pair_pvals(df, "x", "y", min_n_pair=6)
    assert n_points == 5
    assert np.isnan(p_value)
    assert np.isnan(rho)


def test_cliffs_delta_tracks_effect_direction():
    assert dom.cliffs_delta(np.array([5, 6, 7]), np.array([1, 2, 3])) == pytest.approx(1.0)
    assert dom.cliffs_delta(np.array([1, 2, 3]), np.array([5, 6, 7])) == pytest.approx(-1.0)
    assert dom.cliffs_delta(np.array([1, 2, 3]), np.array([1, 2, 3])) == pytest.approx(0.0)


def test_compute_spiral_fraction_correlations_keeps_domination_specific_signals():
    sample = {
        "CG4_Groups": pd.DataFrame(
            [
                {"is_dominated": True, "S_frac": 0.2, "Offset_Bary": 1.0},
                {"is_dominated": True, "S_frac": 0.4, "Offset_Bary": 2.0},
                {"is_dominated": True, "S_frac": 0.6, "Offset_Bary": 3.0},
                {"is_dominated": False, "S_frac": 0.6, "Offset_Bary": 1.0},
                {"is_dominated": False, "S_frac": 0.4, "Offset_Bary": 2.0},
                {"is_dominated": False, "S_frac": 0.2, "Offset_Bary": 3.0},
            ]
        )
    }

    result = dom.compute_spiral_fraction_correlations(
        sample,
        main_quantity="S_frac",
        quantities=["Offset_Bary"],
    )

    dominated = result[result["Domination"] == "Dominated"].iloc[0]
    nondominated = result[result["Domination"] == "Non-dominated"].iloc[0]

    assert dominated["n_groups"] == 3
    assert dominated["Spear_corr"] == pytest.approx(1.0)
    assert nondominated["Spear_corr"] == pytest.approx(-1.0)


def test_summarize_distribution_tests_uses_lowest_adjusted_pvalue(monkeypatch):
    monkeypatch.setattr(dom.co, "SAMPLE", {"CG4": "cg", "RG4": "rg"})

    results = pd.DataFrame(
        [
            {
                "Sample": "CG4",
                "Quantity": "Offset_Bary",
                "n_dom": 8,
                "n_nondom": 8,
                "median_dom": 1.0,
                "median_nondom": 2.0,
                "BM_p": 0.02,
                "BM_p_adj": 0.01,
                "BM_signif_FDR": True,
                "Cliffs_delta": -0.6,
            },
            {
                "Sample": "CG4",
                "Quantity": "Vdisp",
                "n_dom": 8,
                "n_nondom": 8,
                "median_dom": 3.0,
                "median_nondom": 4.0,
                "BM_p": 0.03,
                "BM_p_adj": 0.04,
                "BM_signif_FDR": True,
                "Cliffs_delta": -0.3,
            },
            {
                "Sample": "RG4",
                "Quantity": "Vdisp",
                "n_dom": 8,
                "n_nondom": 8,
                "median_dom": 3.0,
                "median_nondom": 4.0,
                "BM_p": 0.3,
                "BM_p_adj": 0.3,
                "BM_signif_FDR": False,
                "Cliffs_delta": 0.1,
            },
        ]
    )

    summary = dom.summarize_distribution_tests(results)

    assert summary[0]["sample"] == "CG4"
    assert summary[0]["quantity"] == "Offset_Bary"
    assert summary[0]["has_significant_difference"] is True
    assert summary[1] == {
        "sample": "RG4",
        "sample_label": r"RG$_4$",
        "has_significant_difference": False,
    }
