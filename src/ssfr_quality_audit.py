"""Trace the exact spectra underlying the catalogue sSFR measurements.

The input catalogues store DR12 ``specObjID`` values.  The audit therefore
resolves those identifiers in DR12 ``SpecObjAll`` and left-joins the MPA--JHU
``galSpecExtra`` and ``galSpecInfo`` rows on that same identifier.  It never
substitutes a different spectrum sharing the photometric ``objID``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from astroquery.sdss import SDSS

try:
    import config as co
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co


SAMPLES = ("CG4", "Control4B", "Control4C", "RG4")
PROVENANCE_RELEASE = 12
CACHE_PATH = os.path.join(co.DATA_PATH, "sdss_spectral_provenance_dr12.csv")
PROVENANCE_CSV = os.path.join(co.OUTPUT_PATH, "ssfr_spectral_provenance.csv")
SUMMARY_CSV = os.path.join(co.OUTPUT_PATH, "ssfr_missing_quality_audit.csv")
SUMMARY_JSON = os.path.join(co.OUTPUT_PATH, "ssfr_missing_quality_audit.json")


def _analysis_specobjids(sample: dict[str, pd.DataFrame]) -> np.ndarray:
    """Return the unique, non-zero spectrum identifiers used by the samples."""

    identifiers = []
    for sample_name in SAMPLES:
        values = pd.to_numeric(
            sample[sample_name + co.GASUFF]["specobjid"], errors="coerce"
        ).dropna()
        identifiers.extend(values.loc[values.gt(0)].astype("int64").tolist())
    return np.unique(np.asarray(identifiers, dtype="int64"))


def _normalise_query_result(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    integer_columns = (
        "source_specobjid",
        "source_objid",
        "extra_specobjid",
        "info_specobjid",
        "plate",
        "mjd",
        "fiberid",
        "scienceprimary",
    )
    for column in integer_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(
                "Int64"
            )
    for column in (
        "source_sfr",
        "source_specsfr",
        "source_lgm",
        "sn_median",
        "reliable",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def fetch_spectral_provenance(
    sample: dict[str, pd.DataFrame],
    *,
    cache_path: str = CACHE_PATH,
    chunk_size: int = 160,
) -> pd.DataFrame:
    """Retrieve the exact DR12 spectrum and MPA--JHU rows for each stored ID."""

    identifiers = _analysis_specobjids(sample)
    pieces = []
    for start in range(0, len(identifiers), chunk_size):
        chunk = identifiers[start : start + chunk_size]
        sql_ids = ",".join(str(int(value)) for value in chunk)
        query = (
            "SELECT s.specObjID AS source_specobjid, "
            "s.bestObjID AS source_objid, s.plate, s.mjd, s.fiberID, "
            "s.run2d, s.sciencePrimary, "
            "g.specObjID AS extra_specobjid, "
            "g.sfr_tot_p50 AS source_sfr, "
            "g.specsfr_tot_p50 AS source_specsfr, "
            "g.lgm_tot_p50 AS source_lgm, "
            "i.specObjID AS info_specobjid, i.sn_median, i.reliable "
            "FROM SpecObjAll AS s "
            "LEFT JOIN galSpecExtra AS g ON s.specObjID = g.specObjID "
            "LEFT JOIN galSpecInfo AS i ON s.specObjID = i.specObjID "
            f"WHERE s.specObjID IN ({sql_ids})"
        )
        result = SDSS.query_sql(query, data_release=PROVENANCE_RELEASE)
        if result is not None:
            pieces.append(_normalise_query_result(result.to_pandas()))

    if not pieces:
        raise RuntimeError("SDSS returned no exact-spectrum provenance rows")
    fetched = pd.concat(pieces, ignore_index=True)
    fetched = fetched.drop_duplicates("source_specobjid").sort_values(
        "source_specobjid"
    )

    # Multiplicity is descriptive only. It establishes whether the same
    # photometric object has other spectra without ever using one of those
    # spectra as a replacement for the stored observation.
    objids = fetched["source_objid"].dropna().astype("int64").unique()
    multiplicity_pieces = []
    for start in range(0, len(objids), chunk_size):
        chunk = objids[start : start + chunk_size]
        sql_ids = ",".join(str(int(value)) for value in chunk)
        query = (
            "SELECT bestObjID AS source_objid, COUNT(*) AS n_spectra_for_objid "
            "FROM SpecObjAll "
            f"WHERE bestObjID IN ({sql_ids}) GROUP BY bestObjID"
        )
        result = SDSS.query_sql(query, data_release=PROVENANCE_RELEASE)
        if result is not None:
            current = result.to_pandas()
            current.columns = [str(column).lower() for column in current.columns]
            multiplicity_pieces.append(current)
    multiplicity = pd.concat(multiplicity_pieces, ignore_index=True)
    multiplicity["source_objid"] = pd.to_numeric(
        multiplicity["source_objid"], errors="raise"
    ).astype("Int64")
    multiplicity["n_spectra_for_objid"] = pd.to_numeric(
        multiplicity["n_spectra_for_objid"], errors="raise"
    ).astype("Int64")
    fetched = fetched.merge(
        multiplicity, how="left", on="source_objid", validate="m:1"
    )
    fetched.to_csv(cache_path, index=False)
    return fetched


def trace_catalogue_rows(
    sample: dict[str, pd.DataFrame], provenance: pd.DataFrame
) -> pd.DataFrame:
    """Attach exact-spectrum provenance to every final catalogue row."""

    source = provenance.copy()
    source.columns = [str(column).lower() for column in source.columns]
    source["source_specobjid"] = pd.to_numeric(
        source["source_specobjid"], errors="coerce"
    ).astype("Int64")
    for column in ("source_objid", "extra_specobjid", "info_specobjid"):
        source[column] = pd.to_numeric(source[column], errors="coerce").astype(
            "Int64"
        )
    for column in (
        "source_sfr",
        "source_specsfr",
        "source_lgm",
        "sn_median",
        "reliable",
    ):
        source[column] = pd.to_numeric(source[column], errors="coerce")

    pieces = []
    for sample_name in SAMPLES:
        frame = sample[sample_name + co.GASUFF].copy().reset_index(drop=False)
        frame = frame.rename(columns={frame.columns[0]: "catalogue_index"})
        frame["sample"] = sample_name
        frame["objid"] = pd.to_numeric(frame["objid"], errors="coerce").astype(
            "Int64"
        )
        frame["specobjid"] = pd.to_numeric(
            frame["specobjid"], errors="coerce"
        ).astype("Int64")
        keep = [
            "sample",
            "catalogue_index",
            "objid",
            "specobjid",
            "sfr",
            "lgm",
            "sSFR",
            "sSFR_status",
        ]
        joined = frame[keep].merge(
            source,
            how="left",
            left_on="specobjid",
            right_on="source_specobjid",
            validate="m:1",
        )
        pieces.append(joined)

    traced = pd.concat(pieces, ignore_index=True)
    traced["has_exact_spectrum"] = traced["source_specobjid"].notna()
    traced["exact_objid_match"] = (
        traced["has_exact_spectrum"]
        & traced["objid"].notna()
        & traced["source_objid"].notna()
        & traced["objid"].eq(traced["source_objid"])
    )
    traced["has_galspecextra"] = traced["extra_specobjid"].notna()
    traced["has_galspecinfo"] = traced["info_specobjid"].notna()

    source_sfr = pd.to_numeric(traced["source_sfr"], errors="coerce")
    source_lgm = pd.to_numeric(traced["source_lgm"], errors="coerce")
    catalogue_ssfr = pd.to_numeric(traced["sSFR"], errors="coerce")
    valid_components = source_sfr.gt(-9000) & source_lgm.gt(-9000)
    traced["source_derived_ssfr"] = (source_sfr - source_lgm).where(
        valid_components
    )
    traced["source_specsfr_is_sentinel"] = pd.to_numeric(
        traced["source_specsfr"], errors="coerce"
    ).le(-9000)
    traced["derived_ssfr_matches_catalogue"] = np.isclose(
        catalogue_ssfr,
        traced["source_derived_ssfr"],
        rtol=0,
        atol=1e-6,
        equal_nan=False,
    )
    conditions = (
        ~traced["has_exact_spectrum"],
        traced["has_exact_spectrum"] & ~traced["exact_objid_match"],
        ~traced["has_galspecextra"],
        traced["derived_ssfr_matches_catalogue"],
        traced["has_galspecextra"] & source_sfr.le(-9000),
    )
    choices = (
        "exact_spectrum_unresolved",
        "exact_spectrum_objid_mismatch",
        "no_galSpecExtra_sSFR",
        "valid_derived_sSFR",
        "galSpecExtra_sfr_-9999",
    )
    traced["provenance_category"] = np.select(
        conditions, choices, default="exact_source_value_mismatch"
    )
    return traced


def summarise(traced: pd.DataFrame) -> pd.DataFrame:
    """Summarise exact-source categories and available quality fields."""

    rows = []
    for sample_name in SAMPLES:
        sample_rows = traced.loc[traced["sample"] == sample_name]
        categories = (
            ("galSpecExtra_sfr_-9999", "galSpecExtra_sfr_-9999"),
            ("no_galSpecExtra_sSFR", "missing_no_galSpecExtra"),
            ("valid_derived_sSFR", "valid_derived_sSFR"),
            ("exact_spectrum_unresolved", "exact_spectrum_unresolved"),
            ("exact_spectrum_objid_mismatch", "exact_spectrum_objid_mismatch"),
            ("exact_source_value_mismatch", "exact_source_value_mismatch"),
        )
        for category, label in categories:
            current = sample_rows.loc[
                sample_rows["provenance_category"] == category
            ]
            sn = pd.to_numeric(
                current["sn_median"], errors="coerce"
            ).dropna().astype(float)
            reliable = pd.to_numeric(
                current["reliable"], errors="coerce"
            ).dropna().astype(float)
            rows.append(
                {
                    "sample": sample_name,
                    "sSFR_measurement": label,
                    "n_galaxies": int(len(current)),
                    "n_with_galSpecInfo": int(current["has_galspecinfo"].sum()),
                    "sn_median_q25": float(sn.quantile(0.25)),
                    "sn_median_median": float(sn.median()),
                    "sn_median_q75": float(sn.quantile(0.75)),
                    "sn_median_below_10_fraction": float((sn < 10).mean()),
                    "n_with_reliable": int(len(reliable)),
                    "reliable_0_count": int((reliable == 0).sum()),
                    "reliable_0_fraction": float((reliable == 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def run(
    sample: dict[str, pd.DataFrame],
    *,
    fetch: bool = False,
    cache_path: str = CACHE_PATH,
) -> dict:
    """Fetch if requested, then write row-level and summary audit outputs."""

    if fetch:
        provenance = fetch_spectral_provenance(sample, cache_path=cache_path)
    elif os.path.exists(cache_path):
        provenance = pd.read_csv(cache_path)
    else:
        return {
            "status": "skipped",
            "reason": "exact-spectrum provenance cache unavailable; run with fetch=True",
        }

    expected = _analysis_specobjids(sample)
    available = pd.to_numeric(
        provenance["source_specobjid"], errors="coerce"
    ).dropna()
    missing = np.setdiff1d(expected, available.astype("int64").to_numpy())
    traced = trace_catalogue_rows(sample, provenance)
    traced.to_csv(PROVENANCE_CSV, index=False)
    summary = summarise(traced)
    summary.to_csv(SUMMARY_CSV, index=False)
    serialisable_rows = (
        summary.astype(object).where(pd.notna(summary), None).to_dict(orient="records")
    )
    multiple = traced.loc[
        pd.to_numeric(traced["n_spectra_for_objid"], errors="coerce").gt(1)
    ]
    payload = {
        "status": "ok",
        "source": (
            "SDSS DR12 SpecObjAll with galSpecExtra and galSpecInfo left-joined "
            "on the exact stored specObjID"
        ),
        "provenance_rule": (
            "No alternative spectrum is substituted when a photometric objID "
            "has multiple spectra."
        ),
        "prior_failure_cause": (
            "The earlier audit queried DR16 with stored DR12 specObjIDs. Because "
            "specObjID encodes run2d, reprocessed BOSS observations can retain "
            "plate-MJD-fiber but have a different identifier in DR16."
        ),
        "catalogue_ssfr_source": "sfr_tot_p50 minus lgm_tot_p50",
        "fields": [
            "sfr_tot_p50",
            "specsfr_tot_p50",
            "lgm_tot_p50",
            "sn_median",
            "reliable",
        ],
        "n_final_catalogue_rows": int(len(traced)),
        "n_unique_specobjids": int(len(expected)),
        "n_specobjids_resolved_exactly": int(len(expected) - len(missing)),
        "n_specobjids_unresolved": int(len(missing)),
        "n_rows_with_objid_mismatch": int(
            (traced["has_exact_spectrum"] & ~traced["exact_objid_match"]).sum()
        ),
        "n_rows_with_source_value_mismatch": int(
            traced["provenance_category"].eq("exact_source_value_mismatch").sum()
        ),
        "n_rows_with_specsfr_sentinel_but_valid_derived_ssfr": int(
            (
                traced["source_specsfr_is_sentinel"]
                & traced["provenance_category"].eq("valid_derived_sSFR")
            ).sum()
        ),
        "n_unique_objids_with_multiple_spectra": int(
            multiple["objid"].nunique()
        ),
        "cache_file": os.path.basename(cache_path),
        "provenance_file": os.path.basename(PROVENANCE_CSV),
        "summary_file": os.path.basename(SUMMARY_CSV),
        "interpretation": (
            "The final catalogue sSFR is sfr_tot_p50 minus lgm_tot_p50, not "
            "specsfr_tot_p50. Its missing set contains both galSpecExtra "
            "sfr_tot_p50=-9999 rows and spectra with no galSpecExtra/galSpecInfo "
            "coverage; quality fields cannot establish the origin of the full "
            "missing set."
        ),
        "rows": serialisable_rows,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    return payload


def _load_processed_sample() -> dict[str, pd.DataFrame]:
    with open(os.path.join(co.DATA_PATH, co.PROCESS_SAMPLES), "rb") as stream:
        return pickle.load(stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="retrieve exact DR12 spectrum provenance before summarising",
    )
    arguments = parser.parse_args()
    result = run(_load_processed_sample(), fetch=arguments.fetch)
    print(json.dumps(result, indent=2))
