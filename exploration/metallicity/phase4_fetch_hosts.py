"""Phase 4 pre-step — fetch galSpecIndx/galSpecInfo/SpecObj for the non-CG members
of the Lim host groups of Embedded/Predominant CG4s (the within-host design of
src/host_controlled.py), reusing the Phase 1 query and cache conventions."""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
WORK = os.path.join(HERE, "work")
sys.path.insert(0, HERE)
from casutil import cas  # noqa: E402
import phase1_fetch as p1  # noqa: E402

CG_CLASSES = ["Embedded", "Predom"]


def host_members() -> pd.DataFrame:
    pc = pd.read_csv(os.path.join(REPO, "data", "PC_Gals.csv"))
    cg4_groups = pd.read_csv(os.path.join(REPO, "data", "CG4_Groups.csv"))
    with open(os.path.join(REPO, "data", "processed_sample.pkl"), "rb") as fh:
        pkl = pickle.load(fh)
    cg4 = pkl["CG4_Gals"]
    classes = cg4_groups.set_index("Group")["Class"]
    analysis_groups = [g for g in cg4["Group"].unique() if classes.get(g) in CG_CLASSES]
    cg_objids = set(cg4.loc[cg4["Group"].isin(analysis_groups), "objid"])
    hosts = cg4.loc[cg4["Group"].isin(analysis_groups), ["objid"]].merge(pc[["objid", "Group"]], on="objid")["Group"].unique()
    members = pc[pc["Group"].isin(hosts)].copy()
    members["is_CG_member"] = members["objid"].isin(cg_objids).astype(int)
    members["host_lim_group"] = members["Group"]
    members["cg4_group"] = members["objid"].map(cg4.set_index("objid")["Group"])
    # CG4 group -> class for the CG members
    members["cg4_class"] = members["cg4_group"].map(classes)
    members["objid"] = members["objid"].astype("int64")
    members["specobjid_csv"] = pd.to_numeric(members["specobjid"], errors="coerce").fillna(0).astype("uint64")
    # SDSS cache columns (GZ1 votes) as in host_controlled.build_host_frame
    sdss = pkl["SDSS_withAGN"][["objid", "p_E", "p_S"]].drop_duplicates("objid")
    members = members.merge(sdss, on="objid", how="left")
    members["morphology"] = np.where(members["p_E"] > 0.5, "Elliptical", np.where(members["p_S"] > 0.5, "Spiral",
                                     np.where(members["p_E"].notna(), "Uncertain", "NoGZ")))
    return members


def main() -> None:
    m = host_members()
    print(f"host members: {len(m)} rows in {m['host_lim_group'].nunique()} hosts; CG members={int(m['is_CG_member'].sum())}, non-CG={int((m['is_CG_member']==0).sum())}")
    raw_path = os.path.join(WORK, "cas_spectra_raw_hosts.csv")
    if not os.path.exists(raw_path):
        ids = sorted(set(int(o) for o in m["objid"]))
        pieces, log = [], []
        for start in range(0, len(ids), p1.CHUNK):
            chunk = ids[start:start + p1.CHUNK]
            t = time.time()
            df = cas(p1.build_query(chunk), dr=p1.DR)
            pieces.append(df)
            log.append({"start": start, "n_ids": len(chunk), "n_rows": int(len(df)), "sec": round(time.time() - t, 1)})
            print(f"  chunk {start//p1.CHUNK+1}/{(len(ids)-1)//p1.CHUNK+1}: {len(chunk)} objids -> {len(df)} spectra ({time.time()-t:.1f}s)", flush=True)
        raw = pd.concat(pieces, ignore_index=True).drop_duplicates("specObjID")
        raw["from_pass2"] = 0
        raw.to_csv(raw_path, index=False)
        with open(os.path.join(WORK, "cas_fetch_log_hosts.json"), "w") as fh:
            json.dump(log, fh, indent=2)
    raw = pd.read_csv(raw_path, dtype={"specObjID": str, "bestObjID": str})
    chosen = p1.choose_spectrum(raw, m)
    work = m.merge(chosen.drop(columns=["bestObjID_i", "pref"]), on="objid", how="left", validate="m:1")
    work = work.rename(columns={"lgm_tot_p50": "lgm"})
    work["is_bgg"] = (pd.to_numeric(work["rank_M"], errors="coerce") == 1).astype(int)
    work["is_sat"] = (pd.to_numeric(work["rank_M"], errors="coerce") > 1).astype(int)
    work = p1.attach_sizes(work)
    out = os.path.join(WORK, "host_members_worktable_raw.csv")
    work.to_csv(out, index=False)
    print(f"-> {out}: {work.shape}; spec_choice={chosen['spec_choice'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
