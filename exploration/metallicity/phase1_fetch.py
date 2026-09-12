"""Phase 1 — fetch MPA-JHU galSpecIndx / galSpecInfo + SpecObj columns for the
four samples and assemble the working table.

Read-only w.r.t. the repository: inputs come from data/processed_sample.pkl,
data/sdss_size_columns.csv and data/simard2011_subset.csv; every output goes
to exploration/metallicity/work/ or outputs/.

Acquisition path (approved at Gate A): SkyServer SQL on the MPA-JHU DR8
value-added tables.  The DR16 SkyServer host fails its TLS handshake, so we
query DR18; galSpec* are the unchanged MPA-JHU DR8 tables (documented as
covering run2d=26 spectra only) and SpecObjAll keys are stable for those.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import Planck15

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
WORK = os.path.join(HERE, "work")
OUT_JSON = os.path.join(HERE, "outputs", "metallicity_scoping.json")
sys.path.insert(0, HERE)
from casutil import cas  # noqa: E402

SAMPLES = ["CG4", "RG4", "Control4B", "Control4C"]
CHUNK = 200
DR = 18

# Indices requested (base names in galSpecIndx); we pull raw, _err, _sub, _sub_err, _model.
INDICES = ["lick_hd_a", "lick_hg_a", "lick_hb", "lick_mgb", "lick_mg2", "lick_fe5270", "lick_fe5335",
           "lick_fe5015", "lick_fe5406", "lick_hd_f", "lick_hg_f"]
D4000 = ["d4000_n", "d4000"]
SIMARD_SENTINEL = -99.99
Z_MATCH_TOLERANCE = 0.005  # src/size_data.py


def index_cols() -> list[str]:
    cols = []
    for b in INDICES:
        cols += [f"i.{b}", f"i.{b}_err", f"i.{b}_sub", f"i.{b}_sub_err", f"i.{b}_model"]
    for b in D4000:
        cols += [f"i.{b}", f"i.{b}_err", f"i.{b}_sub", f"i.{b}_sub_err", f"i.{b}_model"]
    return cols


SPEC_COLS = [
    "s.specObjID", "s.bestObjID", "s.plate", "s.mjd", "s.fiberID", "s.run2d", "s.sciencePrimary",
    "s.legacyPrimary", "s.z AS spec_z", "s.zErr AS spec_zErr", "s.zWarning", "s.velDisp", "s.velDispErr",
    "s.snMedian", "s.snMedian_r", "s.class AS spec_class", "s.instrument", "s.survey",
    "n.plateid AS mpa_plateid", "n.mjd AS mpa_mjd", "n.fiberid AS mpa_fiberid", "n.z AS mpa_z",
    "n.v_disp", "n.v_disp_err", "n.sn_median", "n.reliable", "n.spectrotype", "n.subclass AS mpa_subclass",
    "n.primtarget", "n.targettype",
]


def build_query(objids: list[int]) -> str:
    id_list = ",".join(str(int(o)) for o in objids)
    return (
        "SELECT " + ", ".join(SPEC_COLS + index_cols()) + " "
        "FROM SpecObjAll AS s "
        "LEFT JOIN galSpecInfo AS n ON n.specObjID = s.specObjID "
        "LEFT JOIN galSpecIndx AS i ON i.specObjID = s.specObjID "
        f"WHERE s.bestObjID IN ({id_list})"
    )


def build_query_by_specobjid(specobjids: list[int]) -> str:
    id_list = ",".join(str(int(o)) for o in specobjids)
    return (
        "SELECT " + ", ".join(SPEC_COLS + index_cols()) + " "
        "FROM SpecObjAll AS s "
        "LEFT JOIN galSpecInfo AS n ON n.specObjID = s.specObjID "
        "LEFT JOIN galSpecIndx AS i ON i.specObjID = s.specObjID "
        f"WHERE s.specObjID IN ({id_list})"
    )


def fetch_spectra(objids: np.ndarray, specobjids: np.ndarray) -> pd.DataFrame:
    """All SpecObjAll rows for the sample objids (by bestObjID), plus a by-specObjID
    pass for the CSV specobjids not returned by the first pass."""
    raw_path = os.path.join(WORK, "cas_spectra_raw.csv")
    log_path = os.path.join(WORK, "cas_fetch_log.json")
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path, dtype={"specObjID": str, "bestObjID": str})
    pieces, log = [], {"pass1_chunks": [], "pass2_chunks": [], "dr": DR, "chunk": CHUNK}
    ids = sorted(set(int(o) for o in objids))
    for start in range(0, len(ids), CHUNK):
        chunk = ids[start:start + CHUNK]
        t = time.time()
        df = cas(build_query(chunk), dr=DR)
        pieces.append(df)
        log["pass1_chunks"].append({"start": start, "n_ids": len(chunk), "n_rows": int(len(df)), "sec": round(time.time() - t, 1)})
        print(f"  pass1 chunk {start//CHUNK+1}/{(len(ids)-1)//CHUNK+1}: {len(chunk)} objids -> {len(df)} spectra ({time.time()-t:.1f}s)", flush=True)
    raw = pd.concat(pieces, ignore_index=True)
    # pass 2: CSV specobjids not seen in pass 1
    seen = set(raw["specObjID"].astype(str))
    missing = sorted(set(int(x) for x in specobjids if int(x) > 0 and str(int(x)) not in seen))
    log["pass2_n_missing_specobjid"] = len(missing)
    for start in range(0, len(missing), CHUNK):
        chunk = missing[start:start + CHUNK]
        t = time.time()
        df = cas(build_query_by_specobjid(chunk), dr=DR)
        df["from_pass2"] = 1
        pieces.append(df)
        log["pass2_chunks"].append({"start": start, "n_ids": len(chunk), "n_rows": int(len(df)), "sec": round(time.time() - t, 1)})
        print(f"  pass2 chunk {start//CHUNK+1}: {len(chunk)} specobjids -> {len(df)} spectra", flush=True)
    raw = pd.concat(pieces, ignore_index=True)
    if "from_pass2" not in raw:
        raw["from_pass2"] = 0
    raw["from_pass2"] = raw["from_pass2"].fillna(0).astype(int)
    raw = raw.drop_duplicates("specObjID").reset_index(drop=True)
    raw.to_csv(raw_path, index=False)
    with open(log_path, "w") as fh:
        json.dump(log, fh, indent=2)
    return raw


def load_samples() -> pd.DataFrame:
    with open(os.path.join(REPO, "data", "processed_sample.pkl"), "rb") as fh:
        pkl = pickle.load(fh)
    pieces = []
    for name in SAMPLES:
        g = pkl[f"{name}_Gals"].copy()
        G = pkl[f"{name}_Groups"][["Group", "z_group", "Vdisp", "NbGal"]].rename(
            columns={"z_group": "group_z", "Vdisp": "group_Vdisp"})
        g = g.merge(G, on="Group", how="left", validate="m:1")
        g["sample"] = name
        g["group_uid"] = ("HMCG:" if name == "CG4" else "Lim:") + g["Group"].astype(int).astype(str)
        pieces.append(g)
    frame = pd.concat(pieces, ignore_index=True, sort=False)
    frame["objid"] = frame["objid"].astype("int64")
    frame["specobjid_csv"] = pd.to_numeric(frame["specobjid"], errors="coerce").fillna(0).astype("uint64")
    frame["rank_M"] = pd.to_numeric(frame["rank_M"], errors="coerce")
    frame["is_bgg"] = (frame["rank_M"] == 1).astype(int)
    frame["is_sat"] = (frame["rank_M"] > 1).astype(int)
    keep = ["sample", "group_uid", "Group", "objid", "specobjid_csv", "rank_M", "is_bgg", "is_sat",
            "RA", "Dec", "z", "group_z", "group_Vdisp", "M_r", "lgm", "sSFR", "sSFR_status",
            "morphology", "p_E", "p_S", "is_dominated"]
    return frame[keep]


def choose_spectrum(raw: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """One spectrum per objid.  Preference order:
    1. specObjID == CSV specobjid and has MPA-JHU result
    2. any spectrum for this bestObjID with MPA-JHU result (legacy run2d=26), sciencePrimary first
    3. specObjID == CSV specobjid (no MPA result)
    4. any other spectrum (sciencePrimary first)
    """
    raw = raw.copy()
    raw["specObjID_u"] = raw["specObjID"].astype("uint64")
    raw["bestObjID_i"] = pd.to_numeric(raw["bestObjID"], errors="coerce").fillna(0).astype("int64")
    raw["has_mpa"] = (pd.to_numeric(raw["mpa_plateid"], errors="coerce").fillna(-1) > 0).astype(int)
    csv_map = frame.drop_duplicates("objid").set_index("objid")["specobjid_csv"]
    # map pass-2 rows (found by specObjID) back to their objid via the CSV key
    spec_to_obj = {int(v): int(k) for k, v in csv_map.items() if int(v) > 0}
    raw["objid_key"] = raw["bestObjID_i"]
    m = raw["objid_key"].eq(0) | ~raw["objid_key"].isin(csv_map.index)
    raw.loc[m, "objid_key"] = raw.loc[m, "specObjID_u"].map(lambda s: spec_to_obj.get(int(s), 0)).astype("int64")
    raw = raw[raw["objid_key"].isin(csv_map.index)].copy()
    raw["is_csv_spec"] = (raw["specObjID_u"] == raw["objid_key"].map(csv_map).astype("uint64")).astype(int)
    raw["pref"] = np.select(
        [raw["is_csv_spec"].eq(1) & raw["has_mpa"].eq(1), raw["has_mpa"].eq(1), raw["is_csv_spec"].eq(1)],
        [1, 2, 3], default=4)
    raw["sciencePrimary"] = pd.to_numeric(raw["sciencePrimary"], errors="coerce").fillna(0)
    raw = raw.sort_values(["objid_key", "pref", "sciencePrimary"], ascending=[True, True, False])
    n_spectra = raw.groupby("objid_key").size().rename("n_spectra_for_objid")
    chosen = raw.drop_duplicates("objid_key").set_index("objid_key")
    chosen = chosen.join(n_spectra)
    chosen["spec_choice"] = chosen["pref"].map({1: "csv_specobjid_with_mpa", 2: "other_legacy_with_mpa",
                                               3: "csv_specobjid_no_mpa", 4: "other_no_mpa"})
    return chosen.reset_index().rename(columns={"objid_key": "objid"})


def attach_sizes(frame: pd.DataFrame) -> pd.DataFrame:
    size = pd.read_csv(os.path.join(REPO, "data", "sdss_size_columns.csv"), dtype={"specObjID": str, "dr7objid": str})
    sim = pd.read_csv(os.path.join(REPO, "data", "simard2011_subset.csv"), dtype={"dr7objid": str})
    for c in ["Scale", "Rhlr", "Rchl_r"]:
        sim.loc[sim[c] == SIMARD_SENTINEL, c] = np.nan
    sim = sim.rename(columns={c: f"simard_{c}" for c in sim.columns if c != "dr7objid"})
    size = size.rename(columns={"specObjID": "size_specObjID"})
    size["objid"] = size["objid"].astype("int64")
    work = frame.merge(size, on="objid", how="left", validate="m:1")
    work = work.merge(sim, on="dr7objid", how="left", validate="m:1")
    z_mismatch = work["simard_Rchl_r"].notna() & ((work["simard_z"] - work["z"]).abs() > Z_MATCH_TOLERANCE)
    work["simard_z_mismatch"] = z_mismatch.astype(int)
    work.loc[z_mismatch, ["simard_Rchl_r", "simard_Rhlr", "simard_Scale"]] = np.nan
    # arcsec half-light radii (Simard: kpc / (kpc/arcsec)); Petrosian already arcsec
    work["Rchl_r_arcsec"] = work["simard_Rchl_r"] / work["simard_Scale"]
    z = pd.to_numeric(work["z"], errors="coerce").to_numpy(dtype=float)
    kpc_per_arcsec = np.full(z.shape, np.nan)
    ok = np.isfinite(z) & (z > 0)
    kpc_per_arcsec[ok] = Planck15.kpc_proper_per_arcmin(z[ok]).to(u.kpc / u.arcmin).value / 60.0
    work["kpc_per_arcsec_Planck15_DA"] = kpc_per_arcsec
    work["Rchl_r_kpc"] = work["Rchl_r_arcsec"] * kpc_per_arcsec
    work["petroR50_kpc"] = work["petroR50_r"] * kpc_per_arcsec
    return work


def main() -> None:
    t0 = time.time()
    frame = load_samples()
    objids = frame["objid"].unique()
    print(f"samples: {len(frame)} rows, {len(objids)} unique objids")
    raw = fetch_spectra(objids, frame["specobjid_csv"].unique())
    print(f"raw spectra fetched: {len(raw)} rows; columns={len(raw.columns)}")
    chosen = choose_spectrum(raw, frame)
    print(f"chosen spectra: {len(chosen)} objids; choice counts={chosen['spec_choice'].value_counts().to_dict()}")
    work = frame.merge(chosen.drop(columns=["bestObjID_i", "pref"]), on="objid", how="left", validate="m:1")
    work = attach_sizes(work)
    out = os.path.join(WORK, "metallicity_worktable_raw.csv")
    work.to_csv(out, index=False)
    print(f"-> {out}: {work.shape}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
