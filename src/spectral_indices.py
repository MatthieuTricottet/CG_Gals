"""MPA-JHU ``galSpecIndx`` age-sensitive indices for the group samples.

``D_n4000`` (``d4000_n``) and ``H\\delta_A`` (``lick_hd_a``) with their errors
are retrieved from SDSS DR12 by the stored DR12 ``specobjid`` of every group
galaxy -- the same identifiers audited in Appendix A -- and cached in
``data/galspecindx_dr12.csv``.  Sentinel values (``-9999``, non-positive
errors) are treated as missing.  The columns are attached to the shared
galaxy frame by :func:`attach_spectral_indices`; they are used descriptively
only (Fig. F.2 availability row and the mass-binned property figure).  No
post-starburst classification is performed here.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    import config as co
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co

CACHE_PATH = co.DATA_PATH + "galspecindx_dr12.csv"
DATA_RELEASE = 12
CHUNK_SIZE = 150  # keeps the SkyServer GET URL under its length limit
INDEX_COLUMNS = ["d4000_n", "d4000_n_err", "lick_hd_a", "lick_hd_a_err"]
SENTINEL = -9999.0


def _clean_ids(values) -> np.ndarray:
    ids = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    ids = ids.loc[ids > 0].astype("int64")
    return np.unique(ids.to_numpy())


def fetch_spectral_indices(specobjids, cache_path: str = CACHE_PATH) -> pd.DataFrame:
    """Return the cached ``galSpecIndx`` rows, querying SDSS only for missing ids."""

    ids = _clean_ids(specobjids)
    cached = pd.DataFrame(columns=["specobjid", *INDEX_COLUMNS])
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        cached["specobjid"] = cached["specobjid"].astype("int64")
    missing = np.setdiff1d(ids, cached["specobjid"].to_numpy(dtype="int64"))
    if missing.size == 0:
        return cached
    try:
        from astroquery.sdss import SDSS
    except ImportError:  # pragma: no cover
        return cached
    pieces = []
    for start in range(0, missing.size, CHUNK_SIZE):
        chunk = missing[start : start + CHUNK_SIZE]
        sql_ids = ",".join(str(int(value)) for value in chunk)
        query = (
            "SELECT specObjID AS specobjid, d4000_n, d4000_n_err, lick_hd_a, "
            "lick_hd_a_err FROM galSpecIndx "
            f"WHERE specObjID IN ({sql_ids})"
        )
        try:
            result = SDSS.query_sql(query, data_release=DATA_RELEASE, timeout=300)
        except Exception as exc:  # network failure: keep what we have
            if co.VERBOSE:
                print(f"[spectral indices] query failed: {exc}")
            break
        if result is not None and len(result):
            pieces.append(result.to_pandas())
    if pieces:
        fetched = pd.concat(pieces, ignore_index=True)
        fetched.columns = [str(column).lower() for column in fetched.columns]
        fetched["specobjid"] = fetched["specobjid"].astype("int64")
        cached = (
            pd.concat([cached, fetched], ignore_index=True)
            .drop_duplicates("specobjid")
            .sort_values("specobjid")
        )
        cached.to_csv(cache_path, index=False)
    return cached


def attach_spectral_indices(frame: pd.DataFrame, cache_path: str = CACHE_PATH) -> pd.DataFrame:
    """Join ``Dn4000`` and ``HdeltaA`` (plus errors) onto ``frame`` by ``specobjid``.

    Rows without a stored spectrum identifier, without a ``galSpecIndx`` row
    (e.g. BOSS spectra outside the MPA-JHU coverage), or carrying sentinel
    values receive NaN.
    """

    if "specobjid" not in frame:
        return frame
    indices = fetch_spectral_indices(frame["specobjid"], cache_path=cache_path)
    work = indices.copy()
    for column in INDEX_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    for value_col, err_col in (("d4000_n", "d4000_n_err"), ("lick_hd_a", "lick_hd_a_err")):
        bad = (
            work[value_col].le(SENTINEL)
            | work[err_col].le(0)
            | ~np.isfinite(work[value_col])
            | ~np.isfinite(work[err_col])
        )
        work.loc[bad, [value_col, err_col]] = np.nan
    work = work.rename(
        columns={
            "d4000_n": "Dn4000",
            "d4000_n_err": "Dn4000_err",
            "lick_hd_a": "HdeltaA",
            "lick_hd_a_err": "HdeltaA_err",
        }
    )
    out = frame.copy()
    out["_specobjid_key"] = pd.to_numeric(out["specobjid"], errors="coerce")
    merged = out.merge(
        work.rename(columns={"specobjid": "_specobjid_key"}),
        on="_specobjid_key",
        how="left",
        validate="m:1",
    )
    return merged.drop(columns=["_specobjid_key"])
