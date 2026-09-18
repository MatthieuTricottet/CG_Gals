"""The Zheng--Shen class labels must follow the published definition.

Zheng & Shen (2021, Eq. 1): a non-isolated, non-split compact group is
*predominant* when it contributes at least half of its parent-group r-band
luminosity and *embedded* otherwise. The inherited Paper I export had the two
labels swapped (repaired 2026-09-17, see CHANGES.md); this test pins the
repaired convention to the Lim--Tempel membership in data/PC_Gals.csv.
"""

import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

import config as co  # noqa: E402


def _rederived():
    cg = pd.read_csv(co.DATA_PATH + "CG4_Gals.csv")
    pc = pd.read_csv(co.DATA_PATH + "PC_Gals.csv")
    groups = pd.read_csv(co.DATA_PATH + "CG4_Groups.csv").set_index("Group")
    merged = cg.merge(
        pc[["objid", "Group"]].rename(columns={"Group": "lim"}), on="objid", how="left"
    )
    per = merged.groupby("Group").agg(
        n_in_pc=("lim", lambda s: int(s.notna().sum())),
        n_hosts=("lim", "nunique"),
        host=("lim", lambda s: s.dropna().mode().iloc[0] if s.notna().any() else np.nan),
    )
    per["N_host"] = per["host"].map(pc.groupby("Group").size())
    per["f_L"] = groups["Lum_group"] / per["host"].map(pc.groupby("Group")["Lum"].sum())
    return groups.join(per)


def test_class_labels_follow_luminosity_fraction_rule():
    table = _rederived()
    nonsplit = table.loc[table["Class"] != "Split"]
    assert (nonsplit["n_in_pc"] == 4).all() and (nonsplit["n_hosts"] == 1).all()
    isolated = nonsplit.loc[nonsplit["Class"] == "Isolated"]
    assert (isolated["N_host"] == 4).all()
    predominant = nonsplit.loc[nonsplit["Class"] == "Predom"]
    embedded = nonsplit.loc[nonsplit["Class"] == "Embedded"]
    assert (predominant["f_L"] >= 0.5).all(), "Predominant CGs must dominate their host"
    assert (embedded["f_L"] < 0.5).all(), "Embedded CGs must not dominate their host"
    assert len(predominant) == 37 and len(embedded) == 19 and len(isolated) == 6


def test_processed_pickle_matches_csv_classes():
    import pickle

    with open(co.DATA_PATH + co.PROCESS_SAMPLES, "rb") as fh:
        sample = pickle.load(fh)
    csv = pd.read_csv(co.DATA_PATH + "CG4_Groups.csv").set_index("Group")["Class"]
    pk = sample["CG4_Groups"].set_index("Group")["Class"]
    assert (pk == csv.reindex(pk.index)).all()
