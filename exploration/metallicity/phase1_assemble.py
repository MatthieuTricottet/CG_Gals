"""Phase 1 (b) — sentinel handling, join accounting, working table, blinding.

Inputs : work/metallicity_worktable_raw.csv (from phase1_fetch.py)
Outputs: work/metallicity_worktable.csv        (true labels; NOT to be read in Phases 2-3)
         work/metallicity_worktable_blind.csv  (sample_blind / group_blind only)
         work/blind_key.json                    (permutation key)
         outputs/metallicity_scoping.json       (section "phase1")
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
OUT_JSON = os.path.join(HERE, "outputs", "metallicity_scoping.json")
SAMPLES = ["CG4", "RG4", "Control4B", "Control4C"]
BLIND_SEED = 20260912

# The seven indices of the brief (+ Mg2), MPA base names.
IDX = {"HdA": "lick_hd_a", "HgA": "lick_hg_a", "Hb": "lick_hb", "Mgb": "lick_mgb",
       "Fe5270": "lick_fe5270", "Fe5335": "lick_fe5335", "Mg2": "lick_mg2", "D4000n": "d4000_n"}
CORE = ["HdA", "HgA", "Hb", "Mgb", "Fe5270", "Fe5335", "D4000n"]  # Mg2 optional


def load_json() -> dict:
    with open(OUT_JSON) as fh:
        return json.load(fh)


def save_json(doc: dict) -> None:
    with open(OUT_JSON, "w") as fh:
        json.dump(doc, fh, indent=2)


def sentinel_census(w: pd.DataFrame) -> dict:
    """Empirical census of special values in the fetched columns (before cleaning)."""
    out = {}
    cols = [c for c in w.columns if c.startswith(("lick_", "d4000"))] + [
        "v_disp", "v_disp_err", "sn_median", "velDisp", "velDispErr", "snMedian", "reliable", "mpa_plateid"]
    for c in cols:
        v = pd.to_numeric(w[c], errors="coerce")
        out[c] = {
            "n_nan": int(v.isna().sum()),
            "n_exact_zero": int((v == 0).sum()),
            "n_le_-999": int((v <= -999).sum()),
            "n_negative": int((v < 0).sum()),
            "min": None if v.dropna().empty else float(v.min()),
            "max": None if v.dropna().empty else float(v.max()),
        }
    return out


def sentinel_notes(w: pd.DataFrame) -> dict:
    """Where do the err == -1 ('not measured') sentinels sit in redshift?"""
    out = {"err_minus1_convention": "index value 0.0 with <index>_err = -1.0 marks an index that was not measured (empirical; not in the datamodel text)"}
    z = pd.to_numeric(w["z"], errors="coerce")
    for base in ["lick_fe5335", "lick_mg2", "lick_hd_a", "lick_fe5270", "lick_mgb", "lick_hb", "lick_hg_a"]:
        m = pd.to_numeric(w[f"{base}_err"], errors="coerce") == -1
        out[base] = {"n_err_minus1": int(m.sum()),
                     "z_p10_p50_p90_of_flagged": None if m.sum() == 0 else [float(z[m].quantile(q)) for q in (.1, .5, .9)],
                     "obs_wavelength_of_red_continuum_at_median_flagged_z": None}
    red_cont = {"lick_fe5335": 5363.375, "lick_mg2": 5366.125}  # red pseudo-continuum upper edge (MPA bandpass table)
    for base, lam in red_cont.items():
        if out[base]["z_p10_p50_p90_of_flagged"]:
            out[base]["obs_wavelength_of_red_continuum_at_median_flagged_z"] = float(lam * (1 + out[base]["z_p10_p50_p90_of_flagged"][1]))
    out["interpretation"] = "flagged Fe5335/Mg2 cluster at z~0.040 where the red pseudo-continuum (5301-5366 A rest) reaches the 5577 A [OI] sky line -> redshift-dependent loss of Fe5335 and hence [MgFe]'"
    out["z_p10_p50_p90_all_rows"] = [float(z.quantile(q)) for q in (.1, .5, .9)]
    return out


def clean(w: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented missing-value conventions.

    * galSpecInfo/galSpecIndx rows are 'zeroed out' when the MPA pipeline was not
      run (datamodel), and ID values are -1 -> has_mpa = mpa_plateid > 0.
    * v_disp_err < 0 -> invalid fit (galSpecInfo datamodel).
    * index with err <= 0, or index == 0 and err == 0 -> missing.
    * D4000_n <= 0 -> missing.
    * sn_median <= 0 -> missing.
    """
    w = w.copy()
    num = lambda c: pd.to_numeric(w[c], errors="coerce")  # noqa: E731
    w["has_spectrum"] = w["specObjID"].notna().astype(int)
    w["has_mpa"] = (num("mpa_plateid").fillna(-1) > 0).astype(int)
    w["mpa_reliable"] = (num("reliable").fillna(0) == 1).astype(int)
    for short, base in IDX.items():
        for suffix, tag in [("", ""), ("_sub", "_sub")]:
            val = num(f"{base}{suffix}")
            err = num(f"{base}{suffix}_err")
            bad = (w["has_mpa"] == 0) | err.isna() | (err <= 0) | val.isna() | ((val == 0) & (err == 0))
            if short == "D4000n":
                bad |= val <= 0
            w[f"{short}{tag}"] = val.where(~bad)
            w[f"{short}{tag}_err"] = err.where(~bad)
        w[f"{short}_model"] = num(f"{base}_model").where(w["has_mpa"] == 1)
    # velocity dispersion: MPA copy of Schlegel's spZbest value, SpecObj copy as cross-check
    vd, vde = num("v_disp"), num("v_disp_err")
    ok = (w["has_mpa"] == 1) & (vde > 0) & (vd > 0)
    w["sigma_mpa"] = vd.where(ok)
    w["sigma_mpa_err"] = vde.where(ok)
    sd, sde = num("velDisp"), num("velDispErr")
    ok2 = (sd > 0) & (sde > 0)
    w["sigma_spec"] = sd.where(ok2)
    w["sigma_spec_err"] = sde.where(ok2)
    w["sigma"] = w["sigma_mpa"].fillna(w["sigma_spec"])
    w["sigma_err"] = w["sigma_mpa_err"].fillna(w["sigma_spec_err"])
    w["sigma_source"] = np.where(w["sigma_mpa"].notna(), "galSpecInfo.v_disp",
                                 np.where(w["sigma_spec"].notna(), "SpecObj.velDisp", "none"))
    sn = num("sn_median")
    w["SN"] = sn.where((w["has_mpa"] == 1) & (sn > 0))
    w["SN_spec_r"] = num("snMedian_r").where(num("snMedian_r") > 0)
    w["SN_spec_all"] = num("snMedian").where(num("snMedian") > 0)
    w["valid_core"] = w[[f"{k}" for k in CORE]].notna().all(axis=1).astype(int)
    w["valid_core_sub"] = w[[f"{k}_sub" for k in CORE]].notna().all(axis=1).astype(int)
    # combined indices (raw and emission-subtracted): [MgFe]' and Mgb/<Fe>
    for tag in ["", "_sub"]:
        mgb, f70, f35 = w[f"Mgb{tag}"], w[f"Fe5270{tag}"], w[f"Fe5335{tag}"]
        emgb, ef70, ef35 = w[f"Mgb{tag}_err"], w[f"Fe5270{tag}_err"], w[f"Fe5335{tag}_err"]
        fe_mix = 0.72 * f70 + 0.28 * f35
        efe_mix = np.sqrt((0.72 * ef70) ** 2 + (0.28 * ef35) ** 2)
        prod = mgb * fe_mix
        mgfe = np.sqrt(prod.where(prod > 0))
        # d[MgFe]' = 0.5/[MgFe]' * sqrt((fe_mix*eMgb)^2 + (mgb*efe_mix)^2)
        emgfe = 0.5 / mgfe * np.sqrt((fe_mix * emgb) ** 2 + (mgb * efe_mix) ** 2)
        w[f"MgFe{tag}"] = mgfe
        w[f"MgFe{tag}_err"] = emgfe
        fe_mean = 0.5 * (f70 + f35)
        efe_mean = 0.5 * np.sqrt(ef70 ** 2 + ef35 ** 2)
        ratio = (mgb / fe_mean).where(fe_mean > 0)
        w[f"MgbFe{tag}"] = ratio
        w[f"MgbFe{tag}_err"] = ratio.abs() * np.sqrt((emgb / mgb) ** 2 + (efe_mean / fe_mean) ** 2)
    return w


def join_accounting(w: pd.DataFrame) -> dict:
    out = {}
    for name in SAMPLES + ["ALL"]:
        part = w if name == "ALL" else w[w["sample"] == name]
        for tag, sub in [("all", part), ("satellites", part[part["is_sat"] == 1]), ("bgg", part[part["is_bgg"] == 1])]:
            out.setdefault(name, {})[tag] = {
                "N_input": int(len(sub)),
                "N_any_spectrum": int(sub["has_spectrum"].sum()),
                "N_csv_specobjid_found": int((sub["is_csv_spec"] == 1).sum()),
                "N_mpa_result": int(sub["has_mpa"].sum()),
                "N_mpa_reliable": int(((sub["has_mpa"] == 1) & (sub["mpa_reliable"] == 1)).sum()),
                "N_sigma_valid": int(sub["sigma"].notna().sum()),
                "N_SN_valid": int(sub["SN"].notna().sum()),
                "N_valid_core7": int(sub["valid_core"].sum()),
                "N_valid_core7_sub": int(sub["valid_core_sub"].sum()),
                "N_valid_core7_and_sigma_and_SN": int(((sub["valid_core"] == 1) & sub["sigma"].notna() & sub["SN"].notna()).sum()),
                "N_valid_MgFe": int(sub["MgFe"].notna().sum()),
                "N_Rchl_r_valid": int(sub["Rchl_r_arcsec"].notna().sum()),
                "N_petroR50_valid": int((sub["petroR50_r"] > 0).sum()),
                "spec_choice_counts": {str(k): int(v) for k, v in sub["spec_choice"].fillna("no_spectrum").value_counts().items()},
                "run2d_counts": {str(k): int(v) for k, v in pd.to_numeric(sub["run2d"], errors="coerce").fillna(-1).astype(int).value_counts().items()},
                "per_index_valid": {k: int(sub[k].notna().sum()) for k in list(IDX)},
                "per_index_valid_sub": {k: int(sub[f"{k}_sub"].notna().sum()) for k in list(IDX)},
            }
    # losses galaxy-by-galaxy: reason for the first failing stage
    reasons = np.select(
        [w["has_spectrum"] == 0, w["has_mpa"] == 0, w["valid_core"] == 0, w["sigma"].isna(), w["SN"].isna()],
        ["no_spectrum_in_SpecObjAll", "no_MPA_result(BOSS/not-run)", "core_index_missing_or_zeroed", "sigma_invalid", "SN_missing"],
        default="ok")
    w["loss_reason"] = reasons
    out["loss_reason_by_sample"] = {name: {str(k): int(v) for k, v in w.loc[w["sample"] == name, "loss_reason"].value_counts().items()} for name in SAMPLES}
    out["loss_reason_by_sample_satellites"] = {name: {str(k): int(v) for k, v in w.loc[(w["sample"] == name) & (w["is_sat"] == 1), "loss_reason"].value_counts().items()} for name in SAMPLES}
    return out


def sign_and_unit_checks(w: pd.DataFrame) -> dict:
    """Empirical checks of sign convention: for quenched (sSFR_status=='Quenched')
    galaxies with valid indices, Mgb should be positive (absorption), HdA typically
    negative (old populations), D4000n > 1.5."""
    q = w[(w["valid_core"] == 1) & (w["sSFR_status"] == "Quenched")]
    s = w[(w["valid_core"] == 1) & (w["sSFR_status"] == "Starforming")]
    med = lambda d, c: None if d[c].dropna().empty else float(d[c].median())  # noqa: E731
    return {
        "n_quenched_valid": int(len(q)), "n_starforming_valid": int(len(s)),
        "median_quenched": {k: med(q, k) for k in list(IDX)},
        "median_starforming": {k: med(s, k) for k in list(IDX)},
        "frac_Mgb_positive_quenched": None if q.empty else float((q["Mgb"] > 0).mean()),
        "frac_HdA_negative_quenched": None if q.empty else float((q["HdA"] < 0).mean()),
        "median_sigma_quenched": med(q, "sigma"), "median_sigma_starforming": med(s, "sigma"),
        "sigma_mpa_vs_spec": {
            "n_both": int((w["sigma_mpa"].notna() & w["sigma_spec"].notna()).sum()),
            "median_abs_diff_kms": None if (w["sigma_mpa"].notna() & w["sigma_spec"].notna()).sum() == 0 else float((w["sigma_mpa"] - w["sigma_spec"]).abs().median()),
            "frac_equal_within_1kms": None if (w["sigma_mpa"].notna() & w["sigma_spec"].notna()).sum() == 0 else float(((w["sigma_mpa"] - w["sigma_spec"]).abs() < 1).mean()),
        },
        "SN_mpa_vs_spec": {
            "n_both": int((w["SN"].notna() & w["SN_spec_all"].notna()).sum()),
            "median_ratio_mpa_over_spec": None if (w["SN"].notna() & w["SN_spec_all"].notna()).sum() == 0 else float((w["SN"] / w["SN_spec_all"]).median()),
        },
    }


def blind(w: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Group-level permutation of the sample label; opaque group ids."""
    rng = np.random.default_rng(BLIND_SEED)
    units = w[["sample", "group_uid"]].drop_duplicates().reset_index(drop=True)
    perm = rng.permutation(len(units))
    permuted = units["sample"].to_numpy()[perm]
    # neutral names for the blinded labels; the assignment itself is part of the key
    neutral = {name: f"S{k + 1}" for k, name in enumerate(rng.permutation(SAMPLES))}
    units["sample_blind"] = [neutral[x] for x in permuted]
    units["group_blind"] = ["G%05d" % k for k in rng.permutation(len(units))]
    key = {"seed": BLIND_SEED, "scheme": "permute 'sample' across (sample, group_uid) units, then rename the permuted label to a neutral S1..S4; opaque group ids",
           "neutral_name_of_permuted_label": neutral,
           "n_units": int(len(units)),
           "units": units.to_dict("records")}
    b = w.merge(units, on=["sample", "group_uid"], how="left", validate="m:1")
    b["gal_uid"] = ["g%05d" % k for k in range(len(b))]
    # opaque per-physical-galaxy key so pooled counts can be deduplicated
    b["obj_hash"] = [hashlib.sha1(f"{BLIND_SEED}:{o}".encode()).hexdigest()[:12] for o in b["objid"]]
    drop = ["sample", "group_uid", "Group", "RA", "Dec", "specobjid_csv", "objid", "specObjID", "bestObjID",
            "plate", "mjd", "fiberID", "mpa_plateid", "mpa_mjd", "mpa_fiberid", "size_specObjID", "dr7objid", "loss_reason"]
    b = b.drop(columns=[c for c in drop if c in b.columns])
    return b, key


def main() -> None:
    w = pd.read_csv(os.path.join(WORK, "metallicity_worktable_raw.csv"), dtype={"specObjID": str, "bestObjID": str, "dr7objid": str, "size_specObjID": str})
    census = sentinel_census(w)
    notes = sentinel_notes(w)
    w = clean(w)
    acct = join_accounting(w)
    checks = sign_and_unit_checks(w)
    w.to_csv(os.path.join(WORK, "metallicity_worktable.csv"), index=False)
    b, key = blind(w)
    b.to_csv(os.path.join(WORK, "metallicity_worktable_blind.csv"), index=False)
    with open(os.path.join(WORK, "blind_key.json"), "w") as fh:
        json.dump(key, fh, indent=2)
    with open(os.path.join(WORK, "cas_fetch_log.json")) as fh:
        fetch_log = json.load(fh)
    doc = load_json()
    doc["phase1"] = {
        "acquisition": {
            "path": "A: SkyServer SQL (SkyServerWS/SearchTools/SqlSearch), DR18 endpoint",
            "why_dr18": "skyserver.sdss.org/dr16 fails the TLS handshake (SSL EOF) from this host; galSpecIndx/galSpecInfo are the MPA-JHU DR8 value-added tables (unchanged since DR8, run2d=26 spectra only), so DR18 serves identical rows",
            "tables": ["SpecObjAll (s)", "galSpecInfo (n)", "galSpecIndx (i)"],
            "join": "LEFT JOIN on specObjID; selection s.bestObjID IN (sample objids) [pass 1], then s.specObjID IN (CSV specobjids not returned) [pass 2]",
            "n_unique_objid_requested": int(w["objid"].nunique()),
            "fetch_log": fetch_log,
            "chosen_spectrum_rule": "1) CSV specobjid with MPA result; 2) any other spectrum of the objid with MPA result (sciencePrimary first); 3) CSV specobjid without MPA; 4) other",
            "raw_cache": "work/cas_spectra_raw.csv",
        },
        "datamodel_verification": {
            "sources": {
                "mpa_dr7_gal_indx_doc": "work/mpa_dr7_SDSS_indx.html (https://wwwmpa.mpa-garching.mpg.de/SDSS/DR7/SDSS_indx.html)",
                "mpa_dr7_gal_info_doc": "work/mpa_dr7_SDSS_info.html",
                "sdss_datamodel_galSpecIndx": "work/sdss_datamodel_galSpecIndx.html (https://data.sdss.org/datamodel/files/SPECTRO_REDUX/galSpecIndx.html)",
                "sdss_datamodel_galSpecInfo": "work/sdss_datamodel_galSpecInfo.html",
                "sdss_dr17_mpajhu_page": "work/sdss_dr17_galaxy_mpajhu.html",
                "cas_DBColumns": ["work/cas_dr18_schema_galSpecIndx.csv", "work/cas_dr18_schema_galSpecInfo.csv"],
                "method_paper": "work/Kauffmann2003a_astroph0204055.pdf (+ .txt)",
            },
            "resolution_system": {
                "status": "UNRESOLVED (explicit statement absent); strong indirect evidence for native SDSS resolution",
                "evidence": [
                    "MPA gal_indx doc / SDSS datamodel: every column is 'Restframe index measurement' with Lick bandpass definitions (Worthey+94, Worthey&Ottaviani 97); no mention of a Lick/IDS resolution transformation or of Lick offsets",
                    "Kauffmann+03a Sect.2: 'The template spectra are convolved with a Gaussian to match the stellar velocity dispersion of each galaxy and rebinned to the SDSS dispersion' -- the MODELS are degraded to the data, the data are not transformed",
                ],
            },
            "velocity_dispersion_correction": {
                "status": "VERIFIED (indirect): no broadening correction is applied to the measured indices",
                "evidence": [
                    "Kauffmann+03a Sect.2 (quoted above): models broadened to each galaxy's sigma; the catalogue provides the sigma-matched model index as <index>_model",
                    "galSpecIndx datamodel: '<INDEX>_MODEL: Index of best fit model spectrum' -- no '_corr' or Lick-corrected column exists",
                ],
            },
            "units_and_sign": {
                "status": "VERIFIED from datamodel (units) + empirical check (sign)",
                "units": {"HdA": "A", "HgA": "A", "Hb": "A", "Mgb": "A", "Fe5270": "A", "Fe5335": "A", "Mg2": "mag", "D4000n": "dimensionless flux ratio (Balogh+99 windows 3850-3950 / 4000-4100)"},
                "sign": "Lick convention: equivalent width, positive = absorption; Balmer indices negative when emission-filled (empirical check in 'empirical_checks')",
                "D4000n_Fnu_or_Flambda": "UNRESOLVED from the catalogue documentation; Kauffmann+03a define D(4000) via 'the ratio of the average flux density F_nu' (Bruzual 83) and adopt the Balogh+99 narrow windows for Dn(4000)",
            },
            "emission_subtraction": {
                "status": "VERIFIED",
                "text": "'<INDEX>_SUB: Restframe index measurement on the data after subtracting all 3-sigma emission lines' (galSpecIndx datamodel). Raw '<INDEX>' is measured on the unmodified spectrum (MPA raw_data page: 'this file contains values measured off the unmodified spectrum')",
            },
            "missing_values": {
                "status": "VERIFIED",
                "text": "galSpecIndx/galSpecInfo datamodel: spectra without an MPA result 'have all values zeroed out'; ID values (plate/mjd/fiber) set to -1; RELIABLE=0 marks unreliable results. galSpecInfo: 'v_disp_err ... negative for invalid fit'. Coverage: 'run2d=26 (DR7 plates) ... not run on ... any BOSS spectra' (sdss dr17 MPA-JHU page)",
                "handling": "has_mpa = mpa_plateid>0; index missing if err<=0 or (val==0 & err==0); D4000n missing if <=0; sigma missing if v_disp_err<0 or v_disp<=0 (SpecObj.velDisp fallback); SN missing if sn_median<=0",
            },
            "velocity_dispersion_definition": "galSpecInfo.v_disp = 'Velocity dispersion from Schlegel' (spZbest, km/s); galSpecInfo.sn_median = 'Median S/N per pixel of the whole spectrum'",
        },
        "sentinel_census_raw": census,
        "sentinel_notes": notes,
        "join_accounting": acct,
        "empirical_checks": checks,
        "blinding": {"seed": BLIND_SEED, "scheme": key["scheme"], "n_units": key["n_units"],
                     "blind_table": "work/metallicity_worktable_blind.csv", "key": "work/blind_key.json"},
    }
    save_json(doc)
    # ---- console summary
    print("=== Phase 1 join accounting (N_input -> any spectrum -> MPA result -> reliable -> 7 core indices valid -> +sigma+SN) ===")
    for name in SAMPLES + ["ALL"]:
        for tag in ["all", "satellites"]:
            a = acct[name][tag]
            print(f"{name:10s} {tag:10s} {a['N_input']:5d} -> {a['N_any_spectrum']:5d} -> {a['N_mpa_result']:5d} -> {a['N_mpa_reliable']:5d} -> {a['N_valid_core7']:5d} -> {a['N_valid_core7_and_sigma_and_SN']:5d}   (sub: {a['N_valid_core7_sub']})")
    print("\nloss reasons (satellites):")
    for name in SAMPLES:
        print(f"  {name}: {acct['loss_reason_by_sample_satellites'][name]}")
    print("\nempirical sign/unit checks:", json.dumps(checks, indent=1)[:1500])
    print("\nblinded table:", b.shape, "columns:", list(b.columns)[:12], "...")


if __name__ == "__main__":
    main()
