"""Phase 0 — read-only inventory of local data for the metallicity scoping study.

Writes exploration/metallicity/outputs/metallicity_scoping.json (section
"phase0") and renders exploration/metallicity/INVENTORY.md from it.  Every
pre-existing file is opened read-only; nothing outside exploration/metallicity/
is touched.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "outputs", "metallicity_scoping.json")
OUT_MD = os.path.join(HERE, "INVENTORY.md")
SAMPLES = ["CG4", "RG4", "Control4B", "Control4C"]

# Columns we will need later, grouped by what they answer.
NEEDED = {
    "spectral_indices": [
        "lick_hd_a", "lick_hg_a", "lick_hb", "lick_mg_b", "lick_fe5270",
        "lick_fe5335", "lick_mg2", "d4000_n", "Dn4000", "H_delta_A",
    ],
    "velocity_dispersion": ["velDisp", "velDispErr", "v_disp", "v_disp_err", "sigma_star"],
    "spectrum_sn": ["snMedian", "sn_median", "snMedian_r", "SN_MEDIAN"],
}


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()


def csv_summary(rel: str, sep: str = ",") -> dict:
    path = os.path.join(REPO, rel)
    with open(path, "r") as fh:
        header = fh.readline().rstrip("\n").split(sep)
        nrows = sum(1 for _ in fh)
    return {"path": rel, "format": "csv", "rows": nrows, "columns": header}


def dat_summary(rel: str) -> dict:
    path = os.path.join(REPO, rel)
    comments, nrows = [], 0
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("#"):
                comments.append(line.rstrip("\n"))
            else:
                nrows += 1
    return {"path": rel, "format": "ascii .dat (Lim+17 raw)", "rows": nrows,
            "columns": [c.lstrip("# ").strip() for c in comments]}


def decode_specobjid(sid: int) -> tuple[int, int, int, int]:
    sid = int(sid)
    return sid >> 50, (sid >> 38) & 0xFFF, ((sid >> 24) & 0x3FFF) + 50000, (sid >> 10) & 0x3FFF


def main() -> None:
    inv: dict = {}
    inv["meta"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_branch": sh(["git", "branch", "--show-current"]),
        "git_head": sh(["git", "rev-parse", "--short", "HEAD"]),
        "git_status_porcelain": sh(["git", "status", "--porcelain"]).splitlines(),
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
    }

    # ---- 1. tables present -------------------------------------------------
    tables = {}
    for rel in [
        "data/CG4_Gals.csv", "data/CG4_Groups.csv", "data/RG4_Gals.csv", "data/RG4_Groups.csv",
        "data/Control4B_Gals.csv", "data/Control4B_Groups.csv",
        "data/Control4C_Gals.csv", "data/Control4C_Groups.csv",
        "data/PC_Gals.csv", "data/PC_Groups.csv",
        "data/sdss_size_columns.csv", "data/simard2011_subset.csv",
        "audit/identity_catalog.csv",
    ]:
        tables[rel] = csv_summary(rel)
    tables["data/SDSS(L) galaxy.dat"] = dat_summary("data/SDSS(L) galaxy.dat")
    tables["data/SDSS(L) group.dat"] = dat_summary("data/SDSS(L) group.dat")

    with open(os.path.join(REPO, "data", "processed_sample.pkl"), "rb") as fh:
        pkl = pickle.load(fh)
    tables["data/processed_sample.pkl"] = {
        "path": "data/processed_sample.pkl", "format": "pickle dict of DataFrames",
        "keys": {k: {"rows": int(v.shape[0]), "columns": list(v.columns)} for k, v in pkl.items()},
    }
    inv["tables"] = tables

    # ---- 2. what is NOT here -------------------------------------------------
    all_cols = set()
    for t in tables.values():
        all_cols.update(t.get("columns", []))
        for sub in t.get("keys", {}).values():
            all_cols.update(sub["columns"])
    present = {grp: sorted(c for c in cols if c in all_cols) for grp, cols in NEEDED.items()}
    with open(os.path.join(REPO, "src", "data_loader.py")) as fh:
        loader = fh.read()
    with open(os.path.join(REPO, "output", "paper", "paper.tex")) as fh:
        paper = fh.read()
    inv["availability"] = {
        "galSpecIndx_downloaded": False,
        "galSpecIndx_evidence": {
            "columns_found_in_any_local_table": present["spectral_indices"],
            "data_loader_joins": [t for t in ["galSpecExtra", "galSpecLine", "galSpecIndx", "galSpecInfo", "zooSpec", "SpecObj", "PhotoObj"] if t in loader],
            "paper_states_not_retrieved": "galSpecIndx" in paper,
        },
        "velocity_dispersion_per_galaxy_local": bool(present["velocity_dispersion"]),
        "velocity_dispersion_columns_found": present["velocity_dispersion"],
        "note_Vdisp": "Every 'Vdisp' column in the *_Groups tables is the GROUP line-of-sight velocity dispersion (gapper), not a stellar sigma.",
        "spectrum_sn_per_galaxy_local": bool(present["spectrum_sn"]),
        "spectrum_sn_columns_found": present["spectrum_sn"],
        "sdss_query_in_pipeline": {
            "data_release": 16,
            "tables": ["SpecObj", "PhotoObj", "galSpecExtra", "galSpecLine", "zooSpec"],
            "join_type": "INNER (so galaxies without a zooSpec or galSpecExtra row are absent from the cache)",
            "columns_retrieved": ["specObjID", "z", "petroMag_{ugriz}-extinction", "objID",
                                  "sfr_tot_p50", "specsfr_tot_p50", "lgm_tot_p50",
                                  "h_alpha/h_beta/oiii_5007/nii_6584 eqw+flux", "p_el_debiased", "p_cs_debiased"],
        },
        "half_light_radius_used_by_pipeline": {
            "column": "Simard+11 Rchl_r (circular r-band half-light radius, pure-Sersic Table 3), via dr7objid bridge",
            "conversion": "Rchl_r / Scale -> arcsec -> Planck15 proper kpc via kpc_proper_per_arcmin (angular-diameter distance)",
            "source_file": "src/size_data.py::attach_size_columns",
            "fallback": "SDSS petroR50_r (arcsec) in data/sdss_size_columns.csv",
        },
    }

    # ---- 3. sample storage, flags, join keys -------------------------------
    sdss = pkl["SDSS_withAGN"]
    size = pd.read_csv(os.path.join(REPO, "data", "sdss_size_columns.csv"), dtype={"specObjID": str, "dr7objid": str})
    size["specObjID_u"] = size["specObjID"].astype("uint64")
    sim = pd.read_csv(os.path.join(REPO, "data", "simard2011_subset.csv"), dtype={"dr7objid": str})
    size_sim = size.merge(sim, on="dr7objid", how="left")

    samples = {}
    for name in SAMPLES:
        g = pkl[f"{name}_Gals"].copy()
        G = pkl[f"{name}_Groups"]
        raw = pd.read_csv(os.path.join(REPO, "data", f"{name}_Gals.csv"))
        rawG = pd.read_csv(os.path.join(REPO, "data", f"{name}_Groups.csv"))
        rank = pd.to_numeric(g["rank_M"], errors="coerce")
        so = pd.to_numeric(g["specobjid"], errors="coerce").fillna(0).astype("uint64")
        run2d = pd.Series([decode_specobjid(x)[3] if x > 0 else -1 for x in so])
        m_size = g[["objid"]].merge(size_sim, on="objid", how="left")
        so_size = m_size["specObjID_u"].fillna(0).astype("uint64").to_numpy()
        so = so.to_numpy()
        sat = rank > 1
        entry = {
            "raw_csv": {"gals_rows": int(len(raw)), "groups_rows": int(len(rawG))},
            "processed_pkl": {"gals_rows": int(len(g)), "groups_rows": int(len(G)),
                              "NbGal_values": {str(k): int(v) for k, v in G["NbGal"].value_counts().items()}},
            "N_groups": int(len(G)),
            "N_gals": int(len(g)),
            "N_unique_objid": int(g["objid"].nunique()),
            "N_BGG_rankM_eq_1": int((rank == 1).sum()),
            "N_sat_rankM_gt_1": int(sat.sum()),
            "N_rankM_missing": int(rank.isna().sum()),
            "N_in_SDSS_cache_by_objid": int(g["objid"].isin(sdss["objid"]).sum()),
            "N_in_SDSS_cache_by_specobjid": int(pd.Series(so).isin(sdss["specObjID"].astype("uint64")).sum()),
            "N_specobjid_zero": int((so == 0).sum()),
            "specobjid_run2d_distribution": {str(k): int(v) for k, v in run2d.value_counts().items()},
            "N_specobjid_ne_sizecache_specObjID": int(((so != so_size) & (so > 0)).sum()),
            "N_specobjid_ne_sizecache_same_plate_fiber_mjd": int(sum(
                (a > 0 and a != b and decode_specobjid(a)[:3] == decode_specobjid(b)[:3]) for a, b in zip(so, so_size))),
            "morphology_counts": {str(k): int(v) for k, v in g["morphology"].value_counts(dropna=False).items()},
            "sSFR_status_counts": {str(k): int(v) for k, v in g["sSFR_status"].value_counts(dropna=False).items()},
            "N_not_in_cache_but_lgm_present": int(g.loc[~g["objid"].isin(sdss["objid"]), "lgm"].notna().sum()),
            "size_cache": {
                "N_petroR50_valid": int((m_size["petroR50_r"] > 0).sum()),
                "N_dr7_bridged": int(m_size["dr7objid"].notna().sum()),
                "N_Rchl_r_valid": int(m_size["Rchl_r"].notna().sum()),
                "N_sat_Rchl_r_valid": int((sat.to_numpy() & m_size["Rchl_r"].notna().to_numpy()).sum()),
            },
        }
        samples[name] = entry
    inv["samples"] = samples

    all_ids = pd.concat([pkl[f"{k}_Gals"]["objid"] for k in SAMPLES]).unique()
    inv["join"] = {
        "canonical_galaxy_key": "objid (SDSS DR16 photometric objID; src/identity.py: one physical galaxy = one objid)",
        "canonical_group_key": "('HMCG', Group) for CG4, ('Lim', Group) for RG4/Control4B/Control4C (same Lim namespace)",
        "bgg_flag": "rank_M == 1 -> BGG; rank_M > 1 -> satellite (src/extended_data.py, 'is_bgg' / 'is_satellite')",
        "spectroscopic_key_in_sample_csvs": "specobjid (int64; DR16 SpecObj.specObjID encoding: plate<<50 | fiber<<38 | (mjd-50000)<<24 | run2d<<10)",
        "spectroscopic_key_in_SDSS_cache": "specObjID (uint64) — agrees with csv specobjid for every objid-matched row",
        "spectroscopic_key_in_size_cache": "specObjID from JOIN SpecObj ON bestObjID — differs from csv specobjid for the run2d=700 (BOSS v5_7_0) spectra, which the size query resolved to run2d=1300 (v5_13_0): same plate/fiber/mjd",
        "run2d_interpretation": {"26": "SDSS-I/II legacy spectro-1d rerun (the only spectra in MPA-JHU DR8 galSpec* tables)",
                                  "700": "BOSS v5_7_0 (SDSS-III) — not in MPA-JHU galSpec*",
                                  "1300": "BOSS v5_13_0 (DR16 re-reduction)"},
        "N_unique_objid_all_four_samples": int(len(all_ids)),
        "N_unique_objid_covered_by_size_cache": int(pd.Index(all_ids).isin(size["objid"]).sum()),
        "SDSS_cache": {"rows_withAGN": int(len(sdss)), "rows_nonAGN": int(len(pkl["SDSS"])),
                       "unique_specObjID": int(sdss["specObjID"].nunique()), "unique_objid": int(sdss["objid"].nunique()),
                       "z_min": float(sdss["z"].min()), "z_max": float(sdss["z"].max()),
                       "selection": "0.005<z<0.0452, r_petro-ext<=17.77, class=GALAXY, lgm_tot_p50>-1000"},
    }

    inv["cosmology"] = {
        "helper": "astropy.cosmology.Planck15; projected scales via Planck15.kpc_proper_per_arcmin (== angular-diameter-distance based)",
        "verified_in": ["src/size_data.py::_kpc_per_arcsec", "src/data_loader.py::correct_group_distance_scales", "src/extended_data.py"],
    }

    # ---- write JSON (phase0 section only; later phases append) --------------
    doc = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as fh:
            doc = json.load(fh)
    doc["phase0"] = inv
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(doc, fh, indent=2)

    # ---- render INVENTORY.md from JSON --------------------------------------
    L = []
    L.append("# Phase 0 — Inventory (auto-generated from outputs/metallicity_scoping.json)\n")
    m = inv["meta"]
    L.append(f"Generated {m['created_utc']} on branch `{m['git_branch']}` @ `{m['git_head']}`; "
             f"git status baseline: `{m['git_status_porcelain']}`\n")
    L.append("## 1. Local tables\n")
    L.append("| path | format | rows | n_cols | columns |")
    L.append("|---|---|---:|---:|---|")
    for rel, t in tables.items():
        if "keys" in t:
            for k, v in t["keys"].items():
                L.append(f"| `{rel}` [{k}] | {t['format']} | {v['rows']} | {len(v['columns'])} | `{', '.join(v['columns'])}` |")
        else:
            cols = t["columns"]
            shown = ", ".join(cols) if t["format"] == "csv" else "(see .dat header; " + str(len(cols)) + " commented column lines)"
            L.append(f"| `{rel}` | {t['format']} | {t['rows']} | {len(cols)} | `{shown}` |")
    L.append("")
    a = inv["availability"]
    L.append("## 2. What is NOT present locally\n")
    L.append(f"- **galSpecIndx downloaded: {a['galSpecIndx_downloaded']}**. Evidence: index columns found in any local table = "
             f"`{a['galSpecIndx_evidence']['columns_found_in_any_local_table']}`; `src/data_loader.py` joins "
             f"`{a['galSpecIndx_evidence']['data_loader_joins']}`; the manuscript explicitly states galSpecIndx was not retrieved: "
             f"{a['galSpecIndx_evidence']['paper_states_not_retrieved']}.")
    L.append(f"- **Per-galaxy stellar velocity dispersion local: {a['velocity_dispersion_per_galaxy_local']}** "
             f"(columns found: `{a['velocity_dispersion_columns_found']}`). {a['note_Vdisp']}")
    L.append(f"- **Per-spectrum median S/N local: {a['spectrum_sn_per_galaxy_local']}** (columns found: `{a['spectrum_sn_columns_found']}`).")
    q = a["sdss_query_in_pipeline"]
    L.append(f"- Pipeline SDSS query (DR{q['data_release']}): tables `{q['tables']}`, {q['join_type']}.")
    h = a["half_light_radius_used_by_pipeline"]
    L.append(f"- Half-light radius used by the pipeline: {h['column']}; {h['conversion']} (`{h['source_file']}`). Fallback: {h['fallback']}.")
    L.append("")
    L.append("## 3. Sample storage, flags and join keys\n")
    j = inv["join"]
    for k in ["canonical_galaxy_key", "canonical_group_key", "bgg_flag", "spectroscopic_key_in_sample_csvs",
              "spectroscopic_key_in_SDSS_cache", "spectroscopic_key_in_size_cache"]:
        L.append(f"- **{k}**: {j[k]}")
    L.append(f"- run2d field meaning: {j['run2d_interpretation']}")
    L.append(f"- Unique objids across the four samples: {j['N_unique_objid_all_four_samples']}; covered by `data/sdss_size_columns.csv`: {j['N_unique_objid_covered_by_size_cache']}")
    c = j["SDSS_cache"]
    L.append(f"- SDSS cache (`processed_sample.pkl['SDSS_withAGN']`): {c['rows_withAGN']} rows ({c['rows_nonAGN']} non-AGN), "
             f"{c['unique_specObjID']} unique specObjID, {c['unique_objid']} unique objid, z∈[{c['z_min']:.4f},{c['z_max']:.4f}]; selection: {c['selection']}")
    L.append(f"- Cosmology: {inv['cosmology']['helper']}")
    L.append("")
    L.append("## 4. Sample sizes currently in use (processed_sample.pkl)\n")
    L.append("| sample | N_groups | N_gals | N_BGG (rank_M=1) | N_sat (rank_M>1) | in SDSS cache (objid) | specobjid run2d=26 / 700 / 0 | not-in-cache but lgm present | Rchl_r valid (all / sat) | NoGZ | NosSFR |")
    L.append("|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|")
    for name, e in samples.items():
        r = e["specobjid_run2d_distribution"]
        L.append(f"| {name} | {e['N_groups']} | {e['N_gals']} | {e['N_BGG_rankM_eq_1']} | {e['N_sat_rankM_gt_1']} | "
                 f"{e['N_in_SDSS_cache_by_objid']} | {r.get('26',0)} / {r.get('700',0)} / {r.get('-1',0)} | {e['N_not_in_cache_but_lgm_present']} | "
                 f"{e['size_cache']['N_Rchl_r_valid']} / {e['size_cache']['N_sat_Rchl_r_valid']} | "
                 f"{e['morphology_counts'].get('NoGZ',0)} | {e['sSFR_status_counts'].get('NosSFR',0)} |")
    L.append("")
    L.append("Raw CSV vs processed rows: " + "; ".join(
        f"{n}: gals {e['raw_csv']['gals_rows']}→{e['processed_pkl']['gals_rows']}, groups {e['raw_csv']['groups_rows']}→{e['processed_pkl']['groups_rows']}"
        for n, e in samples.items()) + " (CG4 drops Split groups; Control4C drops Lim 3688).")
    L.append("")
    L.append("## 5. specobjid consistency\n")
    for name, e in samples.items():
        L.append(f"- {name}: {e['N_specobjid_ne_sizecache_specObjID']} rows where csv `specobjid` ≠ size-cache `specObjID`; "
                 f"of these {e['N_specobjid_ne_sizecache_same_plate_fiber_mjd']} share plate/fiber/MJD (run2d 700→1300 re-reduction), "
                 f"the remainder are objects with a different spectrum chosen by the bestObjID join; "
                 f"{e['N_specobjid_zero']} rows have specobjid=0.")
    L.append("")
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
