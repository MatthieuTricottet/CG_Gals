import pandas as pd
import pytest

from src.analysis import stats_comp_split


def _build_split():
    part_a_gals = pd.DataFrame(
        [
            {
                "sSFR": 1.0,
                "M_r": -20.0,
                "lgm": 10.5,
                "sSFR_status": "active",
                "morphology": "Spiral",
                "BGG_SFRcategory": "MS",
            },
            {
                "sSFR": 2.0,
                "M_r": -21.0,
                "lgm": 10.7,
                "sSFR_status": "quiescent",
                "morphology": "Elliptical",
                "BGG_SFRcategory": "Q",
            },
        ]
    )
    part_a_groups = pd.DataFrame(
        [
            {
                "Offset_Bary": 10.0,
                "Vdisp": 100.0,
                "Voffset": 200.0,
                "size_Group_Bary_kpc": 50.0,
                "M_group": 1.0e12,
                "M_virial": 5.0e12,
                "M_virial_over_L": 1.2,
                "t_cr": 0.5,
                "Prop_M_Sat": 0.1,
                "Prop_M_Tot": 0.2,
                "Prop_G_Sat": 0.1,
                "Prop_G_Tot": 0.3,
                "Prop_Q_Sat": 0.05,
                "Prop_Q_Tot": 0.4,
                "dom": 2.0,
                "Misfit_Bary": 0.3,
                "Vmisfit": 50.0,
                "lMass_200": 13.0,
                "r_200_kpc": 200.0,
            },
            {
                "Offset_Bary": 20.0,
                "Vdisp": 120.0,
                "Voffset": 220.0,
                "size_Group_Bary_kpc": 55.0,
                "M_group": 1.2e12,
                "M_virial": 5.2e12,
                "M_virial_over_L": 1.3,
                "t_cr": 0.6,
                "Prop_M_Sat": 0.2,
                "Prop_M_Tot": 0.3,
                "Prop_G_Sat": 0.15,
                "Prop_G_Tot": 0.35,
                "Prop_Q_Sat": 0.15,
                "Prop_Q_Tot": 0.45,
                "dom": 2.2,
                "Misfit_Bary": 0.35,
                "Vmisfit": 60.0,
                "lMass_200": 13.5,
                "r_200_kpc": 210.0,
            },
        ]
    )

    part_b_gals = pd.DataFrame(
        [
            {
                "sSFR": 0.5,
                "M_r": -19.5,
                "lgm": 9.5,
                "sSFR_status": "quiescent",
                "morphology": "Elliptical",
                "BGG_SFRcategory": "Q",
            },
            {
                "sSFR": 0.8,
                "M_r": -20.5,
                "lgm": 9.8,
                "sSFR_status": "quiescent",
                "morphology": "Spiral",
                "BGG_SFRcategory": "MS",
            },
            {
                "sSFR": 1.2,
                "M_r": -22.0,
                "lgm": 10.2,
                "sSFR_status": "active",
                "morphology": "Spiral",
                "BGG_SFRcategory": "MS",
            },
        ]
    )

    part_b_groups = pd.DataFrame(
        [
            {
                "Offset_Bary": 15.0,
                "Vdisp": 110.0,
                "Voffset": 210.0,
                "size_Group_Bary_kpc": 60.0,
                "M_group": 1.5e12,
                "M_virial": 6.0e12,
                "M_virial_over_L": 1.1,
                "t_cr": 0.55,
                "Prop_M_Sat": 0.12,
                "Prop_M_Tot": 0.22,
                "Prop_G_Sat": 0.11,
                "Prop_G_Tot": 0.31,
                "Prop_Q_Sat": 0.06,
                "Prop_Q_Tot": 0.41,
                "dom": 2.1,
                "Misfit_Bary": 0.33,
                "Vmisfit": 55.0,
                "lMass_200": 13.2,
                "r_200_kpc": 205.0,
            },
            {
                "Offset_Bary": 25.0,
                "Vdisp": 130.0,
                "Voffset": 230.0,
                "size_Group_Bary_kpc": 65.0,
                "M_group": 1.6e12,
                "M_virial": 6.2e12,
                "M_virial_over_L": 1.15,
                "t_cr": 0.65,
                "Prop_M_Sat": 0.18,
                "Prop_M_Tot": 0.28,
                "Prop_G_Sat": 0.12,
                "Prop_G_Tot": 0.32,
                "Prop_Q_Sat": 0.16,
                "Prop_Q_Tot": 0.46,
                "dom": 2.3,
                "Misfit_Bary": 0.37,
                "Vmisfit": 65.0,
                "lMass_200": 13.7,
                "r_200_kpc": 215.0,
            },
            {
                "Offset_Bary": 35.0,
                "Vdisp": 150.0,
                "Voffset": 250.0,
                "size_Group_Bary_kpc": 70.0,
                "M_group": 1.7e12,
                "M_virial": 6.5e12,
                "M_virial_over_L": 1.2,
                "t_cr": 0.7,
                "Prop_M_Sat": 0.2,
                "Prop_M_Tot": 0.35,
                "Prop_G_Sat": 0.2,
                "Prop_G_Tot": 0.37,
                "Prop_Q_Sat": 0.2,
                "Prop_Q_Tot": 0.5,
                "dom": 2.4,
                "Misfit_Bary": 0.4,
                "Vmisfit": 75.0,
                "lMass_200": 14.0,
                "r_200_kpc": 225.0,
            },
        ]
    )

    return {
        "part_a": {"Gals": part_a_gals, "Groups": part_a_groups},
        "part_b": {"Gals": part_b_gals, "Groups": part_b_groups},
    }


def test_stats_comp_split_computes_means_and_counts():
    split = _build_split()

    stats = stats_comp_split(split)

    assert set(stats.keys()) == set(split.keys())

    gal_mean_keys = ["sSFR", "M_r", "lgm"]
    gal_count_map = {
        "sSFR_status_counts": "sSFR_status",
        "morphology_counts": "morphology",
        "BGG_SFRcategory": "BGG_SFRcategory",
    }
    group_mean_keys = [
        "Offset_Bary",
        "Vdisp",
        "Voffset",
        "size_Group_Bary_kpc",
        "M_group",
        "M_virial",
        "M_virial_over_L",
        "t_cr",
        "Prop_M_Sat",
        "Prop_M_Tot",
        "Prop_G_Sat",
        "Prop_G_Tot",
        "Prop_Q_Sat",
        "Prop_Q_Tot",
        "dom",
        "Misfit_Bary",
        "Vmisfit",
        "lMass_200",
        "r_200_kpc",
    ]

    for part, frames in split.items():
        gals = frames["Gals"]
        groups = frames["Groups"]
        part_stats = stats[part]

        for key in gal_mean_keys:
            assert part_stats[key] == pytest.approx(gals[key].mean())

        for result_key, column in gal_count_map.items():
            assert part_stats[result_key] == gals[column].value_counts().to_dict()

        for key in group_mean_keys:
            assert part_stats[key] == pytest.approx(groups[key].mean())
