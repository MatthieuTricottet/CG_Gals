"""Invariants for the canonical identity layer (audit Phase 1)."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import identity  # noqa: E402


@pytest.fixture(scope="module")
def frames():
    return identity.load_raw_samples()


@pytest.fixture(scope="module")
def catalog(frames):
    return identity.build_identity_catalog(frames)


def test_objids_unique_in_catalog(catalog):
    assert catalog["objid"].is_unique


def test_every_control_group_has_exactly_four_members(frames):
    for name in ["Control4B", "Control4C", "RG4"]:
        sizes = frames[name].groupby("Group")["objid"].nunique()
        assert (sizes == 4).all(), (
            f"{name}: groups without exactly 4 unique members: "
            f"{sizes[sizes != 4].to_dict()}"
        )


def test_membership_flags_match_raw_csvs(frames, catalog):
    for name in identity.SAMPLES:
        raw_ids = set(frames[name]["objid"])
        flagged = set(catalog.loc[catalog[f"in_{name}"], "objid"])
        assert flagged == raw_ids, f"membership flags for {name} disagree with CSV"


def test_ranks_match_raw_csvs(frames, catalog):
    for name in identity.SAMPLES:
        raw = frames[name].drop_duplicates().set_index("objid")["rank_M"]
        cat = catalog.set_index("objid").loc[raw.index, f"{name}_rank_M"]
        assert (cat.astype(float) == raw.astype(float)).all()


def test_nonsplit_cg4_groups_map_to_single_host_lim_group(frames):
    hosts = identity.cg4_host_lim_map(frames)
    classes = identity.cg4_class_map(frames)
    nonsplit = classes[classes != "Split"].index
    merged = frames["CG4"][["objid", "Group"]].merge(
        frames["PC"][["objid", "Group"]].rename(columns={"Group": "lim_group"}),
        on="objid",
        how="left",
    )
    per_group = merged[merged["Group"].isin(nonsplit)].groupby("Group")["lim_group"]
    assert (per_group.nunique() == 1).all()
    assert hosts.loc[nonsplit].notna().all()


def test_physical_group_key_unifies_control_labels(catalog):
    # A galaxy that appears in several control samples must carry a single
    # physical group key (the Lim group id), never a label-scoped one.
    multi = catalog[
        catalog[[f"in_{name}" for name in ["Control4B", "Control4C", "RG4"]]].sum(axis=1)
        > 1
    ]
    assert len(multi) > 0  # the overlap is real
    assert multi["physical_group"].str.startswith("Lim:").all()
    for name in ["Control4B", "Control4C", "RG4"]:
        rows = multi[multi[f"in_{name}"]]
        expected = "Lim:" + rows[f"{name}_group"].astype("Int64").astype(str)
        assert (rows["physical_group"] == expected).all()


def test_no_cg4_galaxy_in_any_control_sample(frames, catalog):
    # Paper I exclusion, restored by the Phase 2 regeneration: no control
    # sample may contain any galaxy of the full CG4 sample.
    assert identity.cg4_in_control4c_table(frames).empty
    for name in ["Control4B", "Control4C", "RG4"]:
        overlap = catalog["in_CG4"] & catalog[f"in_{name}"]
        assert not overlap.any(), f"CG4 galaxies found in {name}"


def test_cg4_in_pc_quartets_matches_paper_i_exclusion_counts(frames):
    # The Lim groups whose Control4C quartet would contain a CG4 galaxy are
    # exactly the groups excluded during the regeneration: 61 for the
    # committed PC_Gals.csv under the Delta_m <= 3-restricted quartet
    # selection, matching Paper I's published count (referee/T0; resolves
    # OPEN_QUESTIONS.md #1).
    table = identity.cg4_in_pc_quartets_table(frames)
    assert table["lim_group"].nunique() == 61
    c4c = frames["Control4C"]
    assert not set(table["lim_group"]) & set(c4c["Group"])
