import numpy as np
import pandas as pd
import pytest

from src import exploration_coulours as colours


def test_build_catalogue_colour_frame_matches_sdss_photometry():
    sample = {
        "SDSS": pd.DataFrame(
            {
                "objid": [1, 2, 3],
                "lgm": [9.0, 10.0, 11.0],
                "u_obs": [3.0, 4.0, 5.0],
                "g_obs": [2.0, 2.5, 3.0],
                "r_obs": [1.0, 1.2, 1.4],
                "i_obs": [0.8, 0.9, 1.0],
            }
        ),
        "CG4_Gals": pd.DataFrame(
            {
                "objid": [1, 3, 99],
                "lgm": [9.1, 11.1, 10.0],
            }
        ),
    }

    frame, matching = colours.build_catalogue_colour_frame(sample)
    cg4 = frame.loc[frame["catalogue"] == "CG4"].sort_values("_objid")

    assert matching.loc[0, "n_total"] == 3
    assert matching.loc[0, "n_matched"] == 2
    assert cg4["u_r"].tolist() == pytest.approx([2.0, 3.6])
    assert cg4["g_r"].tolist() == pytest.approx([1.0, 1.6])
    assert cg4["lgm"].tolist() == pytest.approx([9.1, 11.1])


def test_satellite_environment_tests_measure_offset_at_reference_mass():
    rows = []
    for environment, offset, slope_change in [
        ("Ordinary", 0.0, 0.0),
        ("Compact", -0.2, 0.08),
    ]:
        for group in range(16):
            for index, mass in enumerate([9.2, 10.0, 10.8]):
                noise = 0.015 * ((group + index) % 5 - 2)
                base = 1.1 + 0.25 * (mass - 10.0)
                value = base + offset + slope_change * (mass - 10.0) + noise
                rows.append(
                    {
                        "environment": environment,
                        "cluster_id": f"{environment}_{group}",
                        "lgm": mass,
                        "u_r": value,
                        "u_g": 0.8 * value,
                        "g_r": 0.2 * value,
                        "r_i": 0.1 * value,
                    }
                )

    result, reference_mass = colours.compute_satellite_environment_tests(pd.DataFrame(rows))
    by_colour = result.set_index("colour")

    assert reference_mass == pytest.approx(10.0)
    assert set(by_colour.index) == {"u-r", "u-g", "g-r", "r-i"}
    assert by_colour.loc["u-r", "compact_minus_ordinary"] == pytest.approx(-0.2, abs=0.01)
    assert by_colour.loc["u-r", "compact_slope"] > by_colour.loc["u-r", "ordinary_slope"]
    assert by_colour["difference_p_holm"].between(0, 1).all()


def test_bgg_domination_tests_report_raw_and_mass_adjusted_results():
    rows = []
    for dominated in [False, True]:
        for index in range(60):
            mass = 10.5 + 0.008 * index + (0.3 if dominated else 0.0)
            noise = 0.025 * ((index % 7) - 3)
            u_r = 1.5 + 0.4 * (mass - 10.5) + noise
            rows.append(
                {
                    "lgm": mass,
                    "is_dominated": dominated,
                    "domination": "Dominated" if dominated else "Non-dominated",
                    "u_r": u_r,
                    "u_g": 0.75 * u_r,
                    "g_r": 0.25 * u_r,
                    "r_i": 0.1 * u_r,
                }
            )

    mass_summary, mass_test, raw, adjusted, reference_mass = (
        colours.compute_bgg_domination_tests(pd.DataFrame(rows))
    )

    medians = mass_summary.set_index("domination")["median_log_mass"]
    assert medians["Dominated"] > medians["Non-dominated"]
    assert mass_test["significant"] is True
    assert reference_mass == pytest.approx(pd.DataFrame(rows)["lgm"].median())
    assert set(raw["colour"]) == {"u-r", "u-g", "g-r", "r-i"}
    assert set(adjusted["colour"]) == {"u-r", "u-g", "g-r", "r-i"}
    assert np.isfinite(adjusted["delta_at_reference_mass"]).all()
    assert adjusted["offset_p_holm"].between(0, 1).all()
