"""External size-catalogue retrieval and harmonization for the size analysis.

This module fetches the SDSS DR16 Petrosian/seeing columns and the
Simard et al. (2011, ApJS 196, 11) pure-Sersic structural catalogue for the
galaxies already present in the processed sample, joins them onto the
combined galaxy frame, and derives quality-masked physical sizes under the
project Planck15 convention. All network fetches are cached under
``co.DATA_PATH`` and are idempotent; the shared processed pickle is never
mutated.
"""

from __future__ import annotations

import gzip
import os
import urllib.request

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import Planck15

try:
    import config as co
    import generate_report as report
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from . import generate_report as report


# astroquery sends SkyServer SQL by GET; 150 eighteen-digit ids keep the URL
# safely below the server's ~8 kB limit (400-id chunks return HTTP 400).
SDSS_CHUNK_SIZE = 150
VIZIER_CHUNK_SIZE = 100
SIMARD_CATALOG = "J/ApJS/196/11"
SIMARD_SENTINEL = -99.99
# Simard tables are keyed on DR7 photometric identifiers; the DR16 objID must
# be bridged through a SkyServer cross-match table (never joined directly).
BRIDGE_CANDIDATES = ["SpecDR7", "PhotoObjDR7"]

# objid and dr7objid fit in int64; specObjID can exceed the signed 64-bit
# range, is never used for joins here, and is carried as a plain string.
SDSS_INT_ID_COLUMNS = ["objid", "dr7objid"]
SDSS_ID_COLUMNS = ["objid", "specObjID", "dr7objid"]
SDSS_VALUE_COLUMNS = [
    "petroR50_r",
    "petroR50Err_r",
    "petroR90_r",
    "petroR90Err_r",
    "petroRad_r",
    "psfWidth_r",
]
SIMARD_TABLE3_COLUMNS = [
    "objID",
    "z",
    "Sp",
    "Scale",
    "Rhlr",
    "Rchl_r",
    "e",
    "ng",
    "e_ng",
    "rg2d",
]
SIMARD_TABLE1_COLUMNS = ["objID", "PpS"]
SIMARD_VALUE_COLUMNS = [
    "z",
    "Sp",
    "Scale",
    "Rhlr",
    "Rchl_r",
    "e",
    "ng",
    "e_ng",
    "rg2d",
    "PpS",
]
Z_MATCH_TOLERANCE = 0.005


def _ids_to_int64(values) -> pd.Series:
    """Convert 18-digit identifier strings to nullable Int64 without float loss."""

    def convert(value):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return pd.NA
        text = str(value).strip()
        if text in ("", "nan", "None", "<NA>", "--"):
            return pd.NA
        return int(text)

    series = pd.Series(values)
    return pd.Series(
        pd.array([convert(value) for value in series], dtype="Int64"),
        index=series.index,
    )


def _ids_to_strings(series: pd.Series) -> pd.Series:
    """Render identifier values as plain digit strings (empty for missing)."""

    return pd.Series(
        ["" if pd.isna(value) else str(int(value)) for value in series],
        index=series.index,
    )


def _id_string(value) -> str:
    """Render one identifier as a digit string without a float round-trip."""

    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    text = str(value).strip()
    if text in ("", "nan", "None", "<NA>", "--"):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return text


def _column_to_list(table, name):
    """Extract an astropy column as a Python list, preserving masked entries."""

    column = table[name]
    mask = getattr(column, "mask", None)
    values = []
    for position, value in enumerate(column):
        if mask is not None and np.ndim(mask) > 0 and mask[position]:
            values.append(None)
        else:
            values.append(value)
    return values


def _record_build_provenance(key: str, value) -> None:
    """Safely add a provenance key to the persisted build-context JSON.

    ``report.append_json(..., build=True)`` assumes an open (unfinalized) JSON
    object and would corrupt the finalized build file that exists outside the
    sample-rebuild stage, so the file is rewritten atomically instead.
    """

    try:
        build_data = report._load_json(co.RESULTS_BUILD)
        if not build_data:
            return
        build_data[key] = value
        report._write_json(co.RESULTS_BUILD, build_data)
    except Exception:  # pragma: no cover - provenance must never stop a fetch
        pass


def _probe_dr7_bridge(query_sql) -> str | None:
    """Return the first usable DR7 cross-match table in the configured release."""

    probes = {
        "SpecDR7": "SELECT TOP 1 x.specObjID, x.dr7objid FROM SpecDR7 AS x",
        "PhotoObjDR7": "SELECT TOP 1 x.dr8objid, x.dr7objid FROM PhotoObjDR7 AS x",
    }
    for bridge in BRIDGE_CANDIDATES:
        try:
            result = query_sql(probes[bridge], data_release=co.DATA_RELEASE)
            if result is not None and len(result) > 0:
                return bridge
        except Exception:
            continue
    return None


def _sdss_size_query(chunk: list[int], bridge: str) -> str:
    """Build one chunked DR16 size query using the verified bridge table."""

    id_list = ", ".join(str(objid) for objid in chunk)
    if bridge == "SpecDR7":
        bridge_join = "LEFT JOIN SpecDR7 AS x ON x.specObjID = s.specObjID"
    else:
        bridge_join = "LEFT JOIN PhotoObjDR7 AS x ON x.dr8objid = p.objID"
    # psfWidth_r lives in the Field table in DR8+ schemas, not in PhotoObj.
    # dr7objid is coalesced to 0 so the CSV column stays integer-typed: a
    # single NULL would make astropy parse it as float64 and corrupt the
    # 18-digit identifiers.
    return (
        "SELECT p.objID, "
        "p.petroR50_r, p.petroR50Err_r, p.petroR90_r, p.petroR90Err_r, "
        "p.petroRad_r, f.psfWidth_r, s.specObjID, "
        "ISNULL(x.dr7objid, 0) AS dr7objid "
        "FROM PhotoObj AS p "
        "JOIN SpecObj AS s ON s.bestObjID = p.objID "
        "LEFT JOIN Field AS f ON p.fieldID = f.fieldID "
        f"{bridge_join} "
        f"WHERE p.objID IN ({id_list})"
    )


def _read_size_cache() -> pd.DataFrame | None:
    if not os.path.exists(co.SIZE_COLUMNS_FILE):
        return None
    cache = pd.read_csv(co.SIZE_COLUMNS_FILE, dtype={c: str for c in SDSS_ID_COLUMNS})
    for column in SDSS_INT_ID_COLUMNS:
        cache[column] = _ids_to_int64(cache.get(column, pd.Series(dtype=object)))
    cache["specObjID"] = (
        cache.get("specObjID", pd.Series(dtype=object)).fillna("").astype(str)
    )
    for column in SDSS_VALUE_COLUMNS:
        cache[column] = pd.to_numeric(cache.get(column), errors="coerce")
    return cache


def _write_size_cache(frame: pd.DataFrame) -> None:
    out = frame.copy()
    for column in SDSS_INT_ID_COLUMNS:
        out[column] = _ids_to_strings(out[column])
    out["specObjID"] = [_id_string(value) for value in out["specObjID"]]
    out.to_csv(co.SIZE_COLUMNS_FILE, index=False)


def fetch_sdss_size_columns(objids) -> pd.DataFrame:
    """Fetch (or load from cache) the DR16 Petrosian and seeing columns.

    Returns one row per requested ``objid``; galaxies that return no SkyServer
    row are kept as all-NaN rows so reruns stay offline once the cache covers
    the request.
    """

    requested = sorted({int(objid) for objid in pd.Series(objids).dropna()})
    cache = _read_size_cache()
    if cache is not None:
        covered = set(cache["objid"].dropna().astype("int64"))
        if set(requested).issubset(covered):
            subset = cache[cache["objid"].isin(requested)]
            return subset.drop_duplicates("objid").reset_index(drop=True)

    from astroquery.sdss import SDSS

    bridge = _probe_dr7_bridge(SDSS.query_sql)
    if bridge is None:
        raise RuntimeError("No DR7 cross-match table reachable in SkyServer")
    _record_build_provenance("size_dr7_bridge", bridge)

    pieces = []
    for start in range(0, len(requested), SDSS_CHUNK_SIZE):
        chunk = requested[start : start + SDSS_CHUNK_SIZE]
        result = SDSS.query_sql(
            _sdss_size_query(chunk, bridge), data_release=co.DATA_RELEASE
        )
        if result is None or len(result) == 0:
            continue
        piece = pd.DataFrame(
            {
                "objid": _ids_to_int64(_column_to_list(result, "objID")),
                "specObjID": [
                    _id_string(value) for value in _column_to_list(result, "specObjID")
                ],
                "dr7objid": _ids_to_int64(_column_to_list(result, "dr7objid")),
            }
        )
        piece.loc[piece["dr7objid"] == 0, "dr7objid"] = pd.NA
        for column in SDSS_VALUE_COLUMNS:
            piece[column] = pd.to_numeric(
                pd.Series(_column_to_list(result, column)), errors="coerce"
            )
        pieces.append(piece)

    fetched = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame(columns=SDSS_ID_COLUMNS + SDSS_VALUE_COLUMNS)
    )
    fetched = fetched.drop_duplicates("objid")
    missing = sorted(set(requested) - set(fetched["objid"].dropna().astype("int64")))
    if missing:
        placeholder = pd.DataFrame({"objid": _ids_to_int64(missing)})
        placeholder["specObjID"] = ""
        placeholder["dr7objid"] = pd.array([pd.NA] * len(missing), dtype="Int64")
        for column in SDSS_VALUE_COLUMNS:
            placeholder[column] = np.nan
        fetched = pd.concat([fetched, placeholder], ignore_index=True)

    if cache is not None:
        fetched = pd.concat(
            [cache[~cache["objid"].isin(fetched["objid"])], fetched],
            ignore_index=True,
        )
    fetched = fetched.drop_duplicates("objid").reset_index(drop=True)
    _write_size_cache(fetched)
    return fetched[fetched["objid"].isin(requested)].reset_index(drop=True)


def _read_simard_cache() -> pd.DataFrame | None:
    if not os.path.exists(co.SIMARD_SUBSET_FILE):
        return None
    cache = pd.read_csv(co.SIMARD_SUBSET_FILE, dtype={"dr7objid": str})
    cache["dr7objid"] = _ids_to_int64(cache["dr7objid"])
    for column in SIMARD_VALUE_COLUMNS:
        cache[column] = pd.to_numeric(cache.get(column), errors="coerce")
    return cache


def _write_simard_cache(frame: pd.DataFrame) -> None:
    out = frame.copy()
    out["dr7objid"] = _ids_to_strings(out["dr7objid"])
    out.to_csv(co.SIMARD_SUBSET_FILE, index=False)


def _clean_simard_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace CDS sentinels and unphysical sizes by NaN."""

    for column in SIMARD_VALUE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[np.isclose(frame[column], SIMARD_SENTINEL), column] = np.nan
    for column in ["Scale", "Rhlr", "Rchl_r"]:
        frame.loc[frame[column] <= 0, column] = np.nan
    return frame


def _fetch_simard_vizier(dr7objids: list[int]) -> pd.DataFrame:
    """Route A: chunked VizieR constraint queries on the Simard tables."""

    from astroquery.vizier import Vizier

    def query(table: str, columns: list[str]) -> pd.DataFrame:
        vizier = Vizier(columns=columns, row_limit=-1)
        pieces = []
        for start in range(0, len(dr7objids), VIZIER_CHUNK_SIZE):
            chunk = dr7objids[start : start + VIZIER_CHUNK_SIZE]
            # VizieR accepts comma-separated value lists for this column;
            # the pipe OR syntax silently matches nothing here.
            constraint = "=" + ",".join(str(objid) for objid in chunk)
            tables = vizier.query_constraints(
                catalog=f"{SIMARD_CATALOG}/{table}", objID=constraint
            )
            if not tables:
                continue
            result = tables[0]
            piece = pd.DataFrame(
                {"dr7objid": _ids_to_int64(_column_to_list(result, "objID"))}
            )
            for column in columns[1:]:
                piece[column] = pd.to_numeric(
                    pd.Series(_column_to_list(result, column)), errors="coerce"
                )
            pieces.append(piece)
        if not pieces:
            return pd.DataFrame(columns=["dr7objid"] + columns[1:])
        return pd.concat(pieces, ignore_index=True).drop_duplicates("dr7objid")

    table3 = query("table3", SIMARD_TABLE3_COLUMNS)
    table1 = query("table1", SIMARD_TABLE1_COLUMNS)
    return table3.merge(table1, on="dr7objid", how="left", validate="1:1")


def _fetch_simard_cds(dr7objids: list[int]) -> pd.DataFrame:
    """Route B: bulk CDS FTP download parsed with the CDS ReadMe format."""

    from astropy.io import ascii as astropy_ascii

    raw_dir = os.path.join(co.DATA_PATH, "simard2011_raw")
    os.makedirs(raw_dir, exist_ok=True)
    local = {}
    for name in ["ReadMe", "table3.dat.gz", "table1.dat.gz"]:
        path = os.path.join(raw_dir, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(co.SIMARD_FTP_URL + name, path)
        local[name] = path

    wanted = set(dr7objids)

    def read_table(name: str, columns: list[str]) -> pd.DataFrame:
        with gzip.open(local[f"{name}.dat.gz"], "rt") as handle:
            table = astropy_ascii.read(
                handle,
                format="cds",
                readme=local["ReadMe"],
                include_names=columns,
            )
        frame = table.to_pandas()
        frame["dr7objid"] = _ids_to_int64(frame.pop("objID"))
        return frame[frame["dr7objid"].isin(wanted)].drop_duplicates("dr7objid")

    table3 = read_table("table3", SIMARD_TABLE3_COLUMNS)
    table1 = read_table("table1", SIMARD_TABLE1_COLUMNS)
    return table3.merge(table1, on="dr7objid", how="left", validate="1:1")


def fetch_simard(dr7objids) -> pd.DataFrame:
    """Fetch (or load from cache) the Simard et al. (2011) subset.

    Sizes stay in the catalogue's own units here (kpc under the Simard
    cosmology, with ``Scale`` in kpc/arcsec); the mandatory re-conversion to
    Planck15 kpc happens in :func:`attach_size_columns`.
    """

    requested = sorted({int(objid) for objid in pd.Series(dr7objids).dropna()})
    cache = _read_simard_cache()
    if cache is not None:
        covered = set(cache["dr7objid"].dropna().astype("int64"))
        if set(requested).issubset(covered):
            subset = cache[cache["dr7objid"].isin(requested)]
            return subset.drop_duplicates("dr7objid").reset_index(drop=True)

    route = "vizier"
    try:
        fetched = _fetch_simard_vizier(requested)
    except Exception:
        route = "cds_ftp"
        fetched = _fetch_simard_cds(requested)
    _record_build_provenance("size_simard_route", route)

    fetched = _clean_simard_values(fetched.copy())
    missing = sorted(set(requested) - set(fetched["dr7objid"].dropna().astype("int64")))
    if missing:
        placeholder = pd.DataFrame({"dr7objid": _ids_to_int64(missing)})
        for column in SIMARD_VALUE_COLUMNS:
            placeholder[column] = np.nan
        fetched = pd.concat([fetched, placeholder], ignore_index=True)

    if cache is not None:
        fetched = pd.concat(
            [cache[~cache["dr7objid"].isin(fetched["dr7objid"])], fetched],
            ignore_index=True,
        )
    fetched = fetched.drop_duplicates("dr7objid").reset_index(drop=True)
    _write_simard_cache(fetched)
    return fetched[fetched["dr7objid"].isin(requested)].reset_index(drop=True)


def _kpc_per_arcsec(redshift: pd.Series) -> np.ndarray:
    """Project-standard Planck15 proper-kpc-per-arcsec conversion."""

    z = pd.to_numeric(redshift, errors="coerce").to_numpy(dtype=float)
    out = np.full(z.shape, np.nan)
    valid = np.isfinite(z) & (z > 0)
    if valid.any():
        out[valid] = (
            Planck15.kpc_proper_per_arcmin(z[valid]).to(u.kpc / u.arcmin).value / 60.0
        )
    return out


def _int_sum(series) -> int:
    return int(pd.Series(series).fillna(False).astype(bool).sum())


def attach_size_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join cached SDSS and Simard size measurements onto the galaxy frame.

    The merges run after the four samples are concatenated, so overlapping
    control galaxies each receive the same external values, and the external
    tables stay deduplicated on their own identifiers. Returns the enriched
    frame and the per-sample availability/rejection audit; the audit is also
    stored in ``frame.attrs['size_attach_audit']``.
    """

    if "objid" not in frame.columns:
        raise KeyError("galaxy frame lacks the 'objid' column")
    work = frame.copy()
    objids = _ids_to_int64(work["objid"])
    work["objid"] = objids.astype("int64")

    sdss = fetch_sdss_size_columns(objids.dropna().unique())
    sdss = sdss.rename(columns={"specObjID": "size_specObjID"})
    sdss["objid"] = sdss["objid"].astype("int64")
    work = work.merge(sdss, on="objid", how="left", validate="m:1")

    simard = fetch_simard(sdss["dr7objid"].dropna().unique())
    simard = simard.rename(
        columns={column: f"simard_{column}" for column in SIMARD_VALUE_COLUMNS}
    )
    work = work.merge(simard, on="dr7objid", how="left", validate="m:1")

    simard_value_columns = [f"simard_{column}" for column in SIMARD_VALUE_COLUMNS]
    has_simard = work["simard_Rchl_r"].notna()

    # Mismatch guard 1: the Simard redshift must agree with the catalogue one,
    # otherwise the DR7 bridge picked up a different object.
    z_catalogue = pd.to_numeric(work.get("z_numeric", work.get("z")), errors="coerce")
    z_mismatch = has_simard & (
        (work["simard_z"] - z_catalogue).abs() > Z_MATCH_TOLERANCE
    )

    # Mismatch guard 2: two distinct catalogue galaxies of one group resolving
    # to the same DR7 id indicate a blended (shredded/merged) DR7 detection.
    shred_merge = pd.Series(False, index=work.index)
    if "group_uid" in work.columns:
        keyed = work.loc[work["dr7objid"].notna()]
        # A collision needs at least two *different* objids behind one dr7objid.
        counts = keyed.groupby(["group_uid", "dr7objid"], observed=True)[
            "objid"
        ].transform("nunique")
        shred_merge.loc[keyed.index] = counts > 1

    rejected = z_mismatch | shred_merge
    work.loc[rejected, simard_value_columns] = np.nan

    # Units: catalogue sizes are kpc under the Simard cosmology. Route through
    # their own Scale column (kpc/arcsec) back to arcsec, then to kpc with the
    # project Planck15 convention. Petrosian radii are arcsec throughout.
    kpc_per_arcsec = _kpc_per_arcsec(z_catalogue)
    work["Rchl_r_kpc"] = (work["simard_Rchl_r"] / work["simard_Scale"]) * kpc_per_arcsec
    work["Rhlr_kpc"] = (work["simard_Rhlr"] / work["simard_Scale"]) * kpc_per_arcsec
    work["petroR50_kpc"] = work["petroR50_r"] * kpc_per_arcsec
    work["petroR90_kpc"] = work["petroR90_r"] * kpc_per_arcsec

    in_window_simard = work["Rchl_r_kpc"].between(co.SIZE_MIN_KPC, co.SIZE_MAX_KPC)
    in_window_petro = work["petroR50_kpc"].between(co.SIZE_MIN_KPC, co.SIZE_MAX_KPC)
    n_pegged = work["simard_ng"].notna() & (
        (work["simard_ng"] <= co.NG_PEG_LOW) | (work["simard_ng"] >= co.NG_PEG_HIGH)
    )
    petro_valid = (
        (work["petroR50_r"] > 0)
        & (work["petroR90_r"] > work["petroR50_r"])
        & (work["petroR50Err_r"] > 0)
    )

    work["n_pegged"] = n_pegged.astype(float)
    work["size_ok_simard"] = (
        work["Rchl_r_kpc"].notna() & in_window_simard & ~n_pegged
    ).astype(float)
    work["size_ok_simard_incl_pegged"] = (
        work["Rchl_r_kpc"].notna() & in_window_simard
    ).astype(float)
    work["size_ok_petro"] = (
        petro_valid & work["petroR50_kpc"].notna() & in_window_petro
    ).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        work["log_Rchl_r_kpc"] = np.where(
            work["size_ok_simard"] == 1, np.log10(work["Rchl_r_kpc"]), np.nan
        )
        work["log_Rchl_r_kpc_incl_pegged"] = np.where(
            work["size_ok_simard_incl_pegged"] == 1,
            np.log10(work["Rchl_r_kpc"]),
            np.nan,
        )
        work["log_Rhlr_kpc"] = np.where(
            work["Rhlr_kpc"] > 0, np.log10(work["Rhlr_kpc"]), np.nan
        )
        work["log_petroR50_kpc"] = np.where(
            work["size_ok_petro"] == 1, np.log10(work["petroR50_kpc"]), np.nan
        )
    work["concentration_r90_r50"] = np.where(
        petro_valid, work["petroR90_r"] / work["petroR50_r"], np.nan
    )

    audit = {"per_sample": {}}
    if "sample" in work.columns:
        for sample_name, part in work.groupby("sample", observed=True):
            index = part.index
            audit["per_sample"][str(sample_name)] = {
                "n_rows": int(len(part)),
                "petro_row_resolved": _int_sum(part["petroR50_r"].notna()),
                "dr7_bridge_resolved": _int_sum(part["dr7objid"].notna()),
                "simard_matched": _int_sum(
                    has_simard.loc[index] & ~rejected.loc[index]
                ),
                "z_mismatch": _int_sum(z_mismatch.loc[index]),
                "shred_merge": _int_sum(shred_merge.loc[index]),
                "n_pegged": _int_sum(n_pegged.loc[index]),
                "simard_out_of_window": _int_sum(
                    part["Rchl_r_kpc"].notna() & ~in_window_simard.loc[index]
                ),
                "petro_out_of_window": _int_sum(
                    petro_valid.loc[index]
                    & part["petroR50_kpc"].notna()
                    & ~in_window_petro.loc[index]
                ),
                "size_ok_simard": _int_sum(part["size_ok_simard"] == 1),
                "size_ok_petro": _int_sum(part["size_ok_petro"] == 1),
            }
    work.attrs = dict(frame.attrs)
    work.attrs["size_attach_audit"] = audit
    return work, audit
