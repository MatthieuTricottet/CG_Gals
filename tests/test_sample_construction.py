"""Invariants and regression checks for the Control4C reconstruction."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sample_construction as sc  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="module")
def pc_gals():
    return pd.read_csv(os.path.join(DATA, "PC_Gals.csv"))


@pytest.fixture(scope="module")
def cg4_full():
    return pd.read_csv(os.path.join(DATA, "CG4_Gals.csv"))


@pytest.fixture(scope="module")
def c4c_gals():
    return pd.read_csv(os.path.join(DATA, "Control4C_Gals.csv"))


@pytest.fixture(scope="module")
def c4c_groups():
    return pd.read_csv(os.path.join(DATA, "Control4C_Groups.csv"))


def test_committed_control4c_matches_reconstruction(pc_gals, cg4_full, c4c_gals):
    rebuilt = sc.build_control4c_gals(pc_gals, cg4_full)
    committed = c4c_gals[rebuilt.columns]
    pd.testing.assert_frame_equal(
        committed.reset_index(drop=True), rebuilt.reset_index(drop=True),
        check_exact=False, rtol=1e-9,
    )


def test_control4c_counts(c4c_gals, c4c_groups):
    # 765 PC groups - 60 CG-contaminated = 705; Lim 3688 stays in the file
    # and is removed at load time (src/main.py::clean).
    assert c4c_gals["Group"].nunique() == 705
    assert len(c4c_gals) == 2820
    assert c4c_gals["objid"].is_unique
    assert len(c4c_groups) == 705
    assert (c4c_gals.groupby("Group").size() == 4).all()


def test_control4c_excludes_all_cg4_galaxies(c4c_gals, cg4_full):
    assert not set(c4c_gals["objid"]) & set(cg4_full["objid"])


def test_quartets_are_bgg_plus_three_closest(c4c_gals):
    assert set(c4c_gals["rank_dist"].unique()) == {1, 2, 3, 4}
    assert set(c4c_gals["rank_M"].unique()) == {1, 2, 3, 4}
    # BGG (rank_dist 1) is the quartet's most luminous member.
    bgg = c4c_gals[c4c_gals["rank_dist"] == 1]
    assert (bgg["rank_M"] == 1).all()
    assert (bgg["dist2BGG"] == 0).all()


def test_group_builder_reproduces_committed_rg4_groups(pc_gals):
    gals = pd.read_csv(os.path.join(DATA, "RG4_Gals.csv"))
    committed = pd.read_csv(os.path.join(DATA, "RG4_Groups.csv"))
    lm180 = pc_gals.drop_duplicates("Group").set_index("Group")["logM_180"]
    rebuilt = sc.build_group_table(gals, lm180_by_group=lm180)
    committed = committed.sort_values("Group").reset_index(drop=True)
    rebuilt = rebuilt.sort_values("Group").reset_index(drop=True)
    for column in sc.GROUP_COLUMNS:
        if column not in committed:
            continue
        a, b = committed[column], rebuilt[column]
        if a.dtype == object:
            assert (a.astype(str) == b.astype(str)).all(), column
        elif column in ("lMass_200", "r_200_kpc"):
            # legacy solver tolerance: <= 3e-3 dex / 0.3 per cent
            np.testing.assert_allclose(a, b, rtol=3e-3, err_msg=column)
        else:
            np.testing.assert_allclose(a, b, rtol=1e-6, err_msg=column)


def test_group_3688_velocity_dispersion_matches_manuscript(c4c_groups):
    # The manuscript justifies the Lim 3688 exclusion by its inflated
    # velocity dispersion (~1981 km/s in Control4C).
    vdisp = float(c4c_groups.loc[c4c_groups["Group"] == 3688, "Vdisp"].iloc[0])
    assert abs(vdisp - 1981) < 2
