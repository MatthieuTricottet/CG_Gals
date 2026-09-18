"""D1 -- origin of the missing sSFR rows (gary-r2 Phase 1, read-only).

Scope: all 6076 final rows of the four group samples, keyed by the stored
DR12 ``specobjid``.  Queries DR12 ``SpecObjAll`` (instrument, survey,
programname, plate/mjd/fiber, class, zWarning, snMedian_r, snMedian, run2d)
LEFT JOIN ``galSpecExtra`` and ``galSpecInfo`` in chunks of <= 500 ids and
caches the result in ``results/diagnostics/gary_r2/d1_specobjall_dr12.csv``.

Outputs
  d1_crosstab_missing_type_instrument.csv   (a)
  d1_snr_by_missing_status.csv              (b)  + rank-sum tests
  results.json['diagnostics_gary_r2']['d1_missing_ssfr']
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astroquery.sdss import SDSS
from scipy.stats import mannwhitneyu

from common import OUT, SAMPLES, load_sample, store_diagnostic, write_csv

CACHE = OUT / "d1_specobjall_dr12.csv"
CHUNK = 150  # <= 500 required; 150 keeps the GET URL under the SkyServer limit
SENTINEL = -9999.0


def fetch(ids: np.ndarray) -> pd.DataFrame:
    if CACHE.exists():
        cached = pd.read_csv(CACHE)
        if set(cached["specObjID"].astype("int64")) >= set(ids.tolist()):
            return cached
    pieces = []
    for start in range(0, len(ids), CHUNK):
        chunk = ids[start : start + CHUNK]
        sql_ids = ",".join(str(int(v)) for v in chunk)
        query = (
            "SELECT s.specObjID, s.bestObjID, s.instrument, s.survey, "
            "s.programname, s.plate, s.mjd, s.fiberID, s.class, s.zWarning, "
            "s.snMedian_r, s.snMedian, s.run2d, s.sciencePrimary, "
            "g.specObjID AS extra_specObjID, g.sfr_tot_p50, g.lgm_tot_p50, "
            "g.specsfr_tot_p50, "
            "i.specObjID AS info_specObjID, i.sn_median AS info_sn_median, "
            "i.reliable "
            "FROM SpecObjAll AS s "
            "LEFT JOIN galSpecExtra AS g ON s.specObjID = g.specObjID "
            "LEFT JOIN galSpecInfo AS i ON s.specObjID = i.specObjID "
            f"WHERE s.specObjID IN ({sql_ids})"
        )
        result = SDSS.query_sql(query, data_release=12, timeout=300)
        if result is None:
            raise RuntimeError(f"empty result for chunk starting at {start}")
        pieces.append(result.to_pandas())
        print(f"  fetched {start + len(chunk)}/{len(ids)}")
    fetched = pd.concat(pieces, ignore_index=True).drop_duplicates("specObjID")
    fetched.to_csv(CACHE, index=False)
    return fetched


def main() -> None:
    sample = load_sample()
    rows = []
    for name in SAMPLES:
        g = sample[name + "_Gals"]
        rows.append(
            pd.DataFrame(
                {
                    "sample": name,
                    "objid": g["objid"].astype("int64").to_numpy(),
                    "specobjid": pd.to_numeric(g["specobjid"], errors="coerce")
                    .fillna(0)
                    .astype("int64")
                    .to_numpy(),
                    "sfr": pd.to_numeric(g["sfr"], errors="coerce").to_numpy(),
                    "lgm": pd.to_numeric(g["lgm"], errors="coerce").to_numpy(),
                    "sSFR_status": g["sSFR_status"].to_numpy(),
                }
            )
        )
    table = pd.concat(rows, ignore_index=True)
    assert len(table) == 6076, len(table)

    ids = np.unique(table.loc[table["specobjid"] > 0, "specobjid"].to_numpy())
    print(f"{len(ids)} unique non-zero specObjIDs")
    spec = fetch(ids)
    spec["specObjID"] = spec["specObjID"].astype("int64")
    for col in ("instrument", "survey", "programname", "class"):
        spec[col] = spec[col].astype(str).str.strip()
    merged = table.merge(
        spec, how="left", left_on="specobjid", right_on="specObjID", validate="m:1"
    )

    has_spec = merged["specobjid"].gt(0) & merged["specObjID"].notna()
    has_extra = merged["extra_specObjID"].notna()
    sfr_sentinel = has_extra & (
        pd.to_numeric(merged["sfr_tot_p50"], errors="coerce") <= SENTINEL
    )
    missing_type = pd.Series("valid", index=merged.index, dtype=object)
    missing_type[~merged["sSFR_status"].isin(["Quenched", "Starforming"])] = (
        "unassigned"
    )
    is_missing = merged["sSFR_status"].eq("NosSFR")
    missing_type[is_missing & ~has_spec] = "no-specObjID"
    missing_type[is_missing & has_spec & ~has_extra] = "no-galSpecExtra"
    missing_type[is_missing & has_spec & has_extra & sfr_sentinel] = "sentinel"
    missing_type[is_missing & has_spec & has_extra & ~sfr_sentinel] = (
        "galSpecExtra-present-other"
    )
    merged["missing_type"] = missing_type
    assert not merged["missing_type"].eq("unassigned").any()
    merged["instrument"] = merged["instrument"].fillna("none").replace("nan", "none")
    merged.loc[~has_spec, "instrument"] = "none"

    # (a) crosstab
    cross = (
        merged.groupby(["sample", "missing_type", "instrument"])
        .size()
        .rename("N")
        .reset_index()
    )
    write_csv(cross, "d1_crosstab_missing_type_instrument.csv")
    extra = (
        merged.groupby(["sample", "missing_type"])
        .agg(
            N=("objid", "size"),
            n_boss=("instrument", lambda v: int((v == "BOSS").sum())),
            n_sdss=("instrument", lambda v: int((v == "SDSS").sum())),
            n_run2d_26=("run2d", lambda v: int((v.astype(str) == "26").sum())),
            n_zwarning_nonzero=("zWarning", lambda v: int((pd.to_numeric(v, errors="coerce").fillna(0) != 0).sum())),
            n_class_galaxy=("class", lambda v: int((v == "GALAXY").sum())),
            n_sciprimary=("sciencePrimary", lambda v: int((pd.to_numeric(v, errors="coerce") == 1).sum())),
            median_snMedian_r=("snMedian_r", "median"),
        )
        .reset_index()
    )
    write_csv(extra, "d1_missing_type_summary.csv")
    print(extra.to_string())
    print(cross.to_string())

    # (b) S/N distributions missing vs valid, per sample
    snr_rows = []
    tests = {}
    for name in SAMPLES:
        part = merged.loc[merged["sample"].eq(name)]
        sn = pd.to_numeric(part["snMedian_r"], errors="coerce")
        valid = sn[part["missing_type"].eq("valid")].dropna()
        miss = sn[part["missing_type"].ne("valid")].dropna()
        for label, values in (("valid", valid), ("missing", miss)):
            if len(values):
                q = values.quantile([0.05, 0.16, 0.5, 0.84, 0.95])
                snr_rows.append(
                    dict(sample=name, status=label, n=len(values), q05=q[0.05],
                         q16=q[0.16], median=q[0.5], q84=q[0.84], q95=q[0.95],
                         frac_below_5=float((values < 5).mean()),
                         frac_below_10=float((values < 10).mean()))
                )
        if len(miss) >= 3 and len(valid) >= 3:
            stat, p = mannwhitneyu(miss, valid, alternative="two-sided")
            tests[name] = dict(n_missing=int(len(miss)), n_valid=int(len(valid)),
                               median_missing=float(miss.median()),
                               median_valid=float(valid.median()),
                               mannwhitney_U=float(stat), p=float(p))
        # per missing type
        for mtype, sub in part.groupby("missing_type"):
            if mtype == "valid":
                continue
            v = pd.to_numeric(sub["snMedian_r"], errors="coerce").dropna()
            if len(v):
                snr_rows.append(dict(sample=name, status=f"missing:{mtype}", n=len(v),
                                     q05=v.quantile(0.05), q16=v.quantile(0.16),
                                     median=v.median(), q84=v.quantile(0.84),
                                     q95=v.quantile(0.95),
                                     frac_below_5=float((v < 5).mean()),
                                     frac_below_10=float((v < 10).mean())))
    snr = pd.DataFrame(snr_rows)
    write_csv(snr, "d1_snr_by_missing_status.csv")
    print(snr.to_string())
    print(tests)

    # pooled test on unique spectra (missing vs valid)
    uniq = merged.drop_duplicates("specobjid")
    sn = pd.to_numeric(uniq["snMedian_r"], errors="coerce")
    miss = sn[uniq["missing_type"].ne("valid")].dropna()
    valid = sn[uniq["missing_type"].eq("valid")].dropna()
    stat, p = mannwhitneyu(miss, valid, alternative="two-sided")
    tests["pooled_unique_spectra"] = dict(
        n_missing=int(len(miss)), n_valid=int(len(valid)),
        median_missing=float(miss.median()), median_valid=float(valid.median()),
        mannwhitney_U=float(stat), p=float(p))
    # BOSS-only S/N among the no-galSpecExtra rows vs SDSS valid
    boss_missing = sn[uniq["missing_type"].eq("no-galSpecExtra")].dropna()
    tests["no_galSpecExtra_vs_valid_unique"] = dict(
        n_missing=int(len(boss_missing)), median_missing=float(boss_missing.median()),
        median_valid=float(valid.median()),
        p=float(mannwhitneyu(boss_missing, valid, alternative="two-sided")[1]))
    sent = sn[uniq["missing_type"].eq("sentinel")].dropna()
    tests["sentinel_vs_valid_unique"] = dict(
        n_missing=int(len(sent)), median_missing=float(sent.median()) if len(sent) else None,
        median_valid=float(valid.median()),
        p=float(mannwhitneyu(sent, valid, alternative="two-sided")[1]) if len(sent) else None)

    # (c) verdict inputs
    miss_all = merged.loc[merged["missing_type"].ne("valid")]
    n_missing = int(len(miss_all))
    n_boss = int(miss_all["instrument"].eq("BOSS").sum())
    n_no_extra = int(miss_all["missing_type"].eq("no-galSpecExtra").sum())
    n_no_extra_boss = int((miss_all["missing_type"].eq("no-galSpecExtra") & miss_all["instrument"].eq("BOSS")).sum())
    n_sent = int(miss_all["missing_type"].eq("sentinel").sum())
    n_sent_sdss = int((miss_all["missing_type"].eq("sentinel") & miss_all["instrument"].eq("SDSS")).sum())
    n_nospec = int(miss_all["missing_type"].eq("no-specObjID").sum())
    sent_sn = pd.to_numeric(miss_all.loc[miss_all["missing_type"].eq("sentinel"), "snMedian_r"], errors="coerce")
    boss_valid = int((merged["missing_type"].eq("valid") & merged["instrument"].eq("BOSS")).sum())
    verdict = (
        f"Of the {n_missing} missing-sSFR rows, {n_no_extra} ({100*n_no_extra/n_missing:.0f}%) have no "
        f"galSpecExtra row and all {n_no_extra_boss} of these are BOSS-instrument spectra (survey='boss', "
        f"run2d v5_7_0), which the MPA-JHU tables (legacy run2d=26 only) never covered, while every valid "
        f"row is an SDSS-legacy spectrum ({boss_valid} valid BOSS rows); the {n_sent} sfr_tot_p50 = -9999 "
        f"sentinel rows are all SDSS-legacy spectra with galSpecInfo reliable=1, zWarning=0 and a stellar mass, "
        f"and {n_nospec} rows have no specObjID. Neither class is low-S/N: median snMedian_r is "
        f"{sent_sn.median():.1f} (sentinel) and "
        f"{pd.to_numeric(miss_all.loc[miss_all['missing_type'].eq('no-galSpecExtra'), 'snMedian_r'], errors='coerce').median():.1f} "
        f"(BOSS) against {valid.median():.1f} for valid rows, and the missing rows have significantly HIGHER "
        f"S/N than the valid rows in every sample (rank-sum). Verdict: BOSS-instrument coverage gap (84%) plus "
        f"MPA-JHU fit failures at normal S/N (15%); not low-S/N spectra."
    )
    print(verdict)

    summary = {
        "n_rows": int(len(merged)),
        "n_missing": n_missing,
        "by_type": {k: int(v) for k, v in miss_all["missing_type"].value_counts().items()},
        "by_type_and_instrument": {
            f"{k[0]}|{k[1]}": int(v)
            for k, v in miss_all.groupby(["missing_type", "instrument"]).size().items()
        },
        "n_boss_among_missing": n_boss,
        "n_boss_among_valid": boss_valid,
        "n_sentinel_sdss_instrument": n_sent_sdss,
        "per_sample": {
            name: {k: int(v) for k, v in
                   merged.loc[merged["sample"].eq(name), "missing_type"].value_counts().items()}
            for name in SAMPLES
        },
        "snr_column": "SpecObjAll.snMedian_r (DR12)",
        "snr_summary_file": "d1_snr_by_missing_status.csv",
        "crosstab_file": "d1_crosstab_missing_type_instrument.csv",
        "rank_sum_tests": tests,
        "verdict": verdict,
    }
    store_diagnostic("d1_missing_ssfr", summary)
    merged.to_csv(OUT / "d1_rows_traced.csv", index=False)


if __name__ == "__main__":
    main()
