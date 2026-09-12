"""Minimal, retrying SkyServer SQL client (read-only GET) for the scoping study.

The DR16 SkyServer host currently fails the TLS handshake outright; DR18 works
but its load balancer intermittently drops connections (SSL EOF).  We retry
with backoff and fall back to curl.  Results are cached as CSV under work/.
"""
from __future__ import annotations

import io
import os
import subprocess
import time

import pandas as pd
import requests

DEFAULT_DR = 18
URL = "https://skyserver.sdss.org/dr{dr}/SkyServerWS/SearchTools/SqlSearch"


def _parse(txt: str) -> pd.DataFrame:
    lines = [l for l in txt.splitlines() if not l.startswith("#")]
    if not lines:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO("\n".join(lines)), dtype={"specObjID": str, "bestObjID": str, "objID": str})


def cas(sql: str, dr: int = DEFAULT_DR, timeout: int = 300, attempts: int = 40) -> pd.DataFrame:
    """Run one SQL statement; return a DataFrame.  Raises after `attempts` failures."""
    last = None
    for k in range(attempts):
        try:
            with requests.Session() as s:
                r = s.get(URL.format(dr=dr), params={"cmd": sql, "format": "csv"}, timeout=timeout)
            if r.status_code == 200 and not r.text.lstrip().startswith("<"):
                return _parse(r.text)
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except Exception as exc:  # noqa: BLE001
            last = exc
        # curl fallback (different TLS stack)
        try:
            out = subprocess.run(
                ["curl", "-s", "-m", str(timeout), "-G", URL.format(dr=dr),
                 "--data-urlencode", f"cmd={sql}", "--data-urlencode", "format=csv"],
                capture_output=True, text=True, check=False)
            if out.returncode == 0 and out.stdout and not out.stdout.lstrip().startswith("<"):
                return _parse(out.stdout)
            last = RuntimeError(f"curl rc={out.returncode}: {out.stdout[:300]}")
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(min(1 + k, 8))
    raise RuntimeError(f"CAS query failed after {attempts} attempts: {last}")


def cas_cached(sql: str, cache_path: str, **kw) -> pd.DataFrame:
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, dtype={"specObjID": str, "bestObjID": str, "objID": str})
    df = cas(sql, **kw)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df
