"""MPA-JHU ``galSpecLine`` measurements with uncertainties for the group samples.

The H-alpha equivalent widths and line fluxes used in the paper come from the
DR16 reference query (``data_loader.load_SDSS``), joined to the group rows by
photometric ``objid``; for every group galaxy with line data this is the same
spectrum as the stored DR12 ``specobjid`` that provides its sSFR.  That query
carries no uncertainties, so the fluxes, equivalent widths and their errors of
the four diagnostic lines are retrieved here by ``specobjid`` and cached in
``data/galspecline_dr16.csv``.  Reruns are offline once the cache covers the
sample.  ``galSpecLine`` is the MPA-JHU (DR8 reduction) table, identical in
DR16 and DR18; the DR18 SkyServer endpoint is used because the DR16 host
refuses TLS connections from some clients.
"""

from __future__ import annotations

import io
import os
import subprocess
import time

import numpy as np
import pandas as pd

try:
    import config as co
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co

CACHE_PATH = os.path.join(co.DATA_PATH, "galspecline_dr16.csv")
SKYSERVER_URL = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
CHUNK_SIZE = 150  # keeps the SkyServer GET URL under its length limit
LINES = ["h_alpha", "nii_6584", "h_beta", "oiii_5007"]
LINE_COLUMNS = [
    f"{line}_{quantity}"
    for line in LINES
    for quantity in ("flux", "flux_err", "eqw", "eqw_err")
]


def _query(sql: str, attempts: int = 8, timeout: int = 120) -> pd.DataFrame:
    """Run one read-only SkyServer SQL query, retrying with a curl fallback."""

    last = None
    for attempt in range(attempts):
        try:
            import requests

            response = requests.get(
                SKYSERVER_URL, params={"cmd": sql, "format": "csv"}, timeout=timeout
            )
            text = response.text
            if response.status_code == 200 and not text.lstrip().startswith("<"):
                return _parse(text)
            last = RuntimeError(f"HTTP {response.status_code}")
        except Exception as exc:  # noqa: BLE001 - network errors of any kind
            last = exc
        try:
            out = subprocess.run(
                ["curl", "-s", "-m", str(timeout), "-G", SKYSERVER_URL,
                 "--data-urlencode", f"cmd={sql}", "--data-urlencode", "format=csv"],
                capture_output=True, text=True, check=False,
            )
            if out.returncode == 0 and out.stdout and not out.stdout.lstrip().startswith("<"):
                return _parse(out.stdout)
            last = RuntimeError(f"curl rc={out.returncode}")
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(min(1 + attempt, 8))
    raise RuntimeError(f"SkyServer query failed after {attempts} attempts: {last}")


def _parse(text: str) -> pd.DataFrame:
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    if not lines:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO("\n".join(lines)), dtype={"specObjID": str})


def fetch_line_measurements(specobjids, cache_path: str = CACHE_PATH) -> pd.DataFrame:
    """Return cached ``galSpecLine`` rows, querying SkyServer only for missing ids."""

    ids = pd.to_numeric(pd.Series(specobjids), errors="coerce").dropna()
    ids = np.unique(ids.loc[ids > 0].astype("int64").to_numpy())
    cached = pd.DataFrame(columns=["specobjid", *LINE_COLUMNS])
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        cached["specobjid"] = cached["specobjid"].astype("int64")
    # ids already queried but absent from galSpecLine are remembered so that
    # BOSS spectra (outside the MPA-JHU coverage) do not trigger new queries
    queried_path = cache_path.replace(".csv", "_queried_ids.txt")
    queried = set()
    if os.path.exists(queried_path):
        queried = {int(line) for line in open(queried_path) if line.strip()}
    missing = np.array(
        sorted(set(ids) - set(cached["specobjid"].tolist()) - queried), dtype="int64"
    )
    if missing.size == 0:
        return cached
    columns = ", ".join(LINE_COLUMNS)
    pieces = []
    for start in range(0, missing.size, CHUNK_SIZE):
        chunk = missing[start : start + CHUNK_SIZE]
        sql = (
            f"SELECT specObjID, {columns} FROM galSpecLine "
            f"WHERE specObjID IN ({','.join(str(int(v)) for v in chunk)})"
        )
        result = _query(sql)
        if len(result):
            result = result.rename(columns={"specObjID": "specobjid"})
            result["specobjid"] = result["specobjid"].astype("int64")
            pieces.append(result)
        queried.update(int(v) for v in chunk)
    if pieces:
        cached = (
            pd.concat([cached, *pieces], ignore_index=True)
            .drop_duplicates("specobjid")
            .sort_values("specobjid")
        )
        cached.to_csv(cache_path, index=False)
    with open(queried_path, "w") as handle:
        handle.write("\n".join(str(v) for v in sorted(queried)) + "\n")
    return cached
