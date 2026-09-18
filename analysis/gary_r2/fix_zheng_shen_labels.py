"""One-off repair of the Zheng--Shen class labels (gary-r2, decision at Gate A).

Diagnostic D3 (results/diagnostics/gary_r2/SUMMARY.md) showed that the
inherited ``Class`` column of ``data/CG4_Groups.csv`` carries the labels
``Embedded`` and ``Predom`` swapped with respect to the definition of
Zheng & Shen (2021, ApJ 911, 105, Eq. 1) restated in Paper I (Sect. 3.3):
a compact group is *predominant* when it contributes at least half of its
parent-group r-band luminosity and *embedded* otherwise.

This script
  1. re-derives f_L = L_CG4 / L_host from the Lim--Tempel membership
     (data/PC_Gals.csv) and refuses to write unless every non-isolated,
     non-split group is unambiguously on the wrong side of 0.5;
  2. copies the original file to data/attic/CG4_Groups_paper1_export.csv
     (the attic is never read by code);
  3. rewrites the ``Class`` column of data/CG4_Groups.csv (schema unchanged);
  4. applies the same swap to ``CG4_Groups`` inside data/processed_sample.pkl,
     which the pipeline loads when REBUILD_SAMPLE is False.

Idempotent: a second run finds the labels already consistent and exits.
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "CG4_Groups.csv"
ATTIC = ROOT / "data" / "attic" / "CG4_Groups_paper1_export.csv"
PKL = ROOT / "data" / "processed_sample.pkl"
SWAP = {"Embedded": "Predom", "Predom": "Embedded"}


def rederived_classes() -> pd.Series:
    cg = pd.read_csv(ROOT / "data" / "CG4_Gals.csv")
    pc = pd.read_csv(ROOT / "data" / "PC_Gals.csv")
    groups = pd.read_csv(CSV).set_index("Group")
    m = cg.merge(pc[["objid", "Group"]].rename(columns={"Group": "lim"}), on="objid", how="left")
    per = m.groupby("Group").agg(n_in_pc=("lim", lambda s: int(s.notna().sum())),
                                 n_hosts=("lim", "nunique"),
                                 host=("lim", lambda s: s.dropna().mode().iloc[0] if s.notna().any() else np.nan))
    per["N_host"] = per["host"].map(pc.groupby("Group").size())
    per["L_host"] = per["host"].map(pc.groupby("Group")["Lum"].sum())
    per["f_L"] = groups["Lum_group"] / per["L_host"]

    def rule(row):
        if row["n_in_pc"] < 4 or row["n_hosts"] != 1:
            return "Split"
        if row["N_host"] == 4:
            return "Isolated"
        return "Predom" if row["f_L"] >= 0.5 else "Embedded"

    out = per.apply(rule, axis=1)
    out.name = "class_rederived"
    return out.to_frame().join(per[["f_L", "N_host"]])


def main() -> None:
    groups = pd.read_csv(CSV)
    ref = rederived_classes()
    check = groups.set_index("Group").join(ref)
    nonsplit = check.loc[check["Class"].ne("Split")]
    if (nonsplit["Class"] == nonsplit["class_rederived"]).all():
        print("labels already consistent with the Zheng & Shen definition; nothing to do")
        return
    swapped = nonsplit.loc[nonsplit["Class"].isin(SWAP.keys())]
    assert (swapped["Class"].map(SWAP) == swapped["class_rederived"]).all(), (
        "labels are not a clean swap; inspect before writing")
    assert (nonsplit.loc[nonsplit["Class"].eq("Isolated"), "class_rederived"] == "Isolated").all()
    assert (check.loc[check["Class"].eq("Split"), "class_rederived"] == "Split").all()
    margin = (swapped["f_L"] - 0.5).abs().min()
    print(f"{len(swapped)} groups to relabel; smallest distance from the 0.5 boundary: {margin:.3f}")

    ATTIC.parent.mkdir(exist_ok=True)
    if not ATTIC.exists():
        shutil.copy2(CSV, ATTIC)
    groups["Class"] = groups["Class"].replace(SWAP)
    groups.to_csv(CSV, index=False)
    print(f"rewrote {CSV.relative_to(ROOT)}; original kept at {ATTIC.relative_to(ROOT)}")
    print(groups["Class"].value_counts().to_dict())

    with open(PKL, "rb") as fh:
        sample = pickle.load(fh)
    pk = sample["CG4_Groups"]
    before = pk["Class"].value_counts().to_dict()
    # the pickle holds the non-split groups only; verify it matched the old file
    old = pd.read_csv(ATTIC).set_index("Group")["Class"]
    assert (pk.set_index("Group")["Class"] == old.reindex(pk["Group"]).to_numpy()).all()
    pk["Class"] = pk["Class"].replace(SWAP)
    sample["CG4_Groups"] = pk
    for key in sample:
        if key.endswith("_Gals") and "Class" in sample[key].columns:
            sample[key]["Class"] = sample[key]["Class"].replace(SWAP)
            print(f"also swapped Class in {key}")
    with open(PKL, "wb") as fh:
        pickle.dump(sample, fh)
    print(f"pickle CG4_Groups Class: {before} -> {pk['Class'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
