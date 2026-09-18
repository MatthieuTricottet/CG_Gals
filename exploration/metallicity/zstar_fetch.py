"""Acquire and subset published direct stellar-metallicity catalogues.

All writes are confined to ``exploration/metallicity/work/catalogues``.
The parent CG_Gals sample is read from Claude's existing true-label worktable
without changing its rows or spectrum choice.

Products
--------
work/catalogues/gallazzi/
    Authoritative documentation, the DR4 metallicity/age catalogues, papers,
    and a small plate-MJD-fibre subset for the CG_Gals spectra.
work/catalogues/firefly/
    Authoritative DR16 documentation/paper and a targeted CAS extract from the
    DR16 SDSS/eBOSS FIREFLY VAC (queried through the working DR18 SkyServer
    endpoint, which republishes the DR16 VAC table).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
WORKTABLE = HERE / "work" / "metallicity_worktable.csv"
CAT = HERE / "work" / "catalogues"
GALLAZZI = CAT / "gallazzi"
FIREFLY = CAT / "firefly"
SDSS_AUX = CAT / "sdss_aux"
CAS_DR = 18  # DR16 endpoint fails TLS here; DR18 republishes sdssEbossFirefly.
CHUNK = 80

sys.path.insert(0, str(HERE))
from casutil import cas  # noqa: E402

DOWNLOADS = {
    GALLAZZI / "stellarmet.html": "https://wwwmpa.mpa-garching.mpg.de/SDSS/DR4/Data/stellarmet.html",
    GALLAZZI / "all_stat_z_log.dat.gz": "https://wwwmpa.mpa-garching.mpg.de/SDSS/DR4/Data/Gallazzi/all_stat_z_log.dat.gz",
    GALLAZZI / "all_stat_age.dat.gz": "https://wwwmpa.mpa-garching.mpg.de/SDSS/DR4/Data/Gallazzi/all_stat_age.dat.gz",
    GALLAZZI / "Gallazzi2005_arxiv.pdf": "https://arxiv.org/pdf/astro-ph/0506539",
    FIREFLY / "sdss_dr16_firefly.html": "https://www.sdss4.org/dr16/spectro/eboss-firefly-value-added-catalog/",
    FIREFLY / "sdss_eboss_firefly-DR16_datamodel.html": "https://data.sdss.org/datamodel/files/EBOSS_FIREFLY/FIREFLY_VER/sdss_eboss_firefly-DR16.html",
    FIREFLY / "Comparat2019_arxiv.pdf": "https://arxiv.org/pdf/1711.06575",
}

GALLAZZI_COLUMNS = [
    "plate", "mjd", "fiberid", "p2p5", "p16", "median", "p84",
    "p97p5", "mode", "dr4_index",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, path: Path, attempts: int = 8) -> None:
    """Download once, atomically, with retries."""
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    last: Exception | None = None
    for k in range(attempts):
        try:
            with requests.get(
                url,
                stream=True,
                timeout=180,
                headers={"User-Agent": "CG_Gals-Zstar-audit/1.0"},
            ) as response:
                response.raise_for_status()
                with part.open("wb") as fh:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            fh.write(block)
            os.replace(part, path)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            if part.exists():
                part.unlink()
            time.sleep(min(2 ** k, 30))
    raise RuntimeError(f"download failed after {attempts} attempts: {url}: {last}")


def extract_pdf_text(pdf: Path) -> None:
    txt = pdf.with_suffix(".txt")
    if txt.exists() and txt.stat().st_size > 0:
        return
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", str(pdf), str(txt)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        reader = PdfReader(pdf)
        txt.write_text("\n\f\n".join(page.extract_text() or "" for page in reader.pages))
        return
    if proc.returncode:
        raise RuntimeError(f"pdftotext failed for {pdf.name}: {proc.stderr}")


def target_pmf() -> pd.DataFrame:
    w = pd.read_csv(WORKTABLE, low_memory=False)
    cols = ["plate", "mjd", "fiberID"]
    target = w[cols].copy()
    for col in cols:
        target[col] = pd.to_numeric(target[col], errors="coerce")
    target = target.dropna().astype(int).drop_duplicates().sort_values(cols)
    target = target.rename(columns={"fiberID": "fiberid"}).reset_index(drop=True)
    return target


def target_objids() -> pd.Series:
    w = pd.read_csv(WORKTABLE, usecols=["objid"])
    return pd.to_numeric(w["objid"], errors="raise").astype("int64").drop_duplicates().sort_values().reset_index(drop=True)


def read_gallazzi(path: Path, prefix: str) -> pd.DataFrame:
    with gzip.open(path, "rt") as fh:
        frame = pd.read_csv(
            fh,
            sep=r"\s+",
            names=GALLAZZI_COLUMNS,
            dtype={"plate": "int32", "mjd": "int32", "fiberid": "int16", "dr4_index": "int32"},
        )
    return frame.rename(
        columns={c: f"{prefix}_{c}" for c in GALLAZZI_COLUMNS[3:-1]}
    )


def subset_gallazzi(target: pd.DataFrame) -> dict:
    z = read_gallazzi(GALLAZZI / "all_stat_z_log.dat.gz", "z_log_abs")
    age = read_gallazzi(GALLAZZI / "all_stat_age.dat.gz", "age_logyr_rband")
    keys = ["plate", "mjd", "fiberid", "dr4_index"]
    if len(z) != len(age):
        raise AssertionError("Gallazzi metallicity and age row counts differ")
    if not z[keys].equals(age[keys]):
        raise AssertionError("Gallazzi metallicity and age catalogue keys are not aligned")
    full = z.merge(age, on=keys, how="inner", validate="1:1")
    subset = target.merge(full, on=["plate", "mjd", "fiberid"], how="inner", validate="1:1")
    out = GALLAZZI / "gallazzi_sample_matches.csv"
    subset.to_csv(out, index=False)
    return {
        "catalogue_rows": int(len(full)),
        "unique_pmf": int(full[["plate", "mjd", "fiberid"]].drop_duplicates().shape[0]),
        "target_unique_pmf": int(len(target)),
        "matched_unique_pmf": int(len(subset)),
        "output": str(out.relative_to(HERE)),
        "metallicity_raw_quantiles": {
            str(q): float(full["z_log_abs_median"].quantile(q))
            for q in [0, 0.01, 0.5, 0.99, 1]
        },
    }


def firefly_query(rows: pd.DataFrame) -> str:
    terms = [
        f"(PLATE={int(r.plate)} AND MJD={int(r.mjd)} AND FIBERID={int(r.fiberid)})"
        for r in rows.itertuples(index=False)
    ]
    return "SELECT * FROM sdssEbossFirefly WHERE " + " OR ".join(terms)


def subset_firefly(target: pd.DataFrame) -> dict:
    query_dir = FIREFLY / "queries"
    query_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    query_log = []
    for start in range(0, len(target), CHUNK):
        rows = target.iloc[start : start + CHUNK]
        cache = query_dir / f"firefly_{start:05d}_{start + len(rows) - 1:05d}.csv"
        sql = firefly_query(rows)
        if cache.exists():
            got = pd.read_csv(cache, low_memory=False)
        else:
            t0 = time.time()
            got = cas(sql, dr=CAS_DR, timeout=180, attempts=12)
            got.to_csv(cache, index=False)
            print(f"FIREFLY {start:4d}/{len(target)}: {len(rows)} targets -> {len(got)} rows ({time.time()-t0:.1f}s)", flush=True)
        pieces.append(got)
        query_log.append({
            "start": start,
            "n_target": int(len(rows)),
            "n_returned": int(len(got)),
            "cache": str(cache.relative_to(HERE)),
            "query_sha256": hashlib.sha256(sql.encode()).hexdigest(),
        })
    raw = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    raw.columns = [str(c) for c in raw.columns]
    for col in ["PLATE", "MJD", "FIBERID"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["PLATE", "MJD", "FIBERID"])
    raw[["PLATE", "MJD", "FIBERID"]] = raw[["PLATE", "MJD", "FIBERID"]].astype(int)
    raw = raw.drop_duplicates(["PLATE", "MJD", "FIBERID"])
    matched = target.merge(
        raw,
        left_on=["plate", "mjd", "fiberid"],
        right_on=["PLATE", "MJD", "FIBERID"],
        how="inner",
        validate="1:1",
    )
    out = FIREFLY / "firefly_sample_matches.csv"
    matched.to_csv(out, index=False)
    with (FIREFLY / "query_log.json").open("w") as fh:
        json.dump(query_log, fh, indent=2)
    run2d_counts = {
        str(k): int(v)
        for k, v in matched["RUN2D"].astype(str).value_counts().items()
    }
    if sum(run2d_counts.values()) != len(matched):
        raise AssertionError("FIREFLY RUN2D counts do not sum to the matched spectrum count")
    return {
        "cas_data_release_endpoint": CAS_DR,
        "vac": "SDSS DR16 SDSS/eBOSS FIREFLY",
        "combined_dr16_vac_directory_version": "v1_1_1",
        "legacy_sdss_firefly_version": "v1_1_0",
        "eboss_firefly_version": "v1_1_1",
        "target_unique_pmf": int(len(target)),
        "matched_unique_pmf": int(len(matched)),
        "returned_run2d": run2d_counts,
        "columns": list(matched.columns),
        "output": str(out.relative_to(HERE)),
    }


def fetch_sdss_aux(objids: pd.Series) -> dict:
    """Fetch apparent/fibre magnitudes needed for selection/aperture audits."""
    query_dir = SDSS_AUX / "queries"
    query_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    chunk = 180
    for start in range(0, len(objids), chunk):
        ids = objids.iloc[start : start + chunk]
        cache = query_dir / f"photo_{start:05d}_{start + len(ids) - 1:05d}.csv"
        if cache.exists():
            got = pd.read_csv(cache, dtype={"objID": str})
        else:
            id_text = ",".join(str(int(value)) for value in ids)
            sql = (
                "SELECT objID, petroMag_r, fiberMag_r, modelMag_r, cModelMag_r, extinction_r "
                f"FROM PhotoObjAll WHERE objID IN ({id_text})"
            )
            got = cas(sql, dr=CAS_DR, timeout=180, attempts=12)
            got.to_csv(cache, index=False)
            print(f"photometry {start:4d}/{len(objids)}: {len(ids)} targets -> {len(got)} rows", flush=True)
        pieces.append(got)
    raw = pd.concat(pieces, ignore_index=True)
    raw["objID"] = pd.to_numeric(raw["objID"], errors="coerce")
    raw = raw.dropna(subset=["objID"])
    raw["objid"] = raw["objID"].astype("int64")
    raw = raw.drop(columns="objID").drop_duplicates("objid")
    out = SDSS_AUX / "sdss_photometry.csv"
    raw.to_csv(out, index=False)
    return {
        "target_unique_objid": int(len(objids)),
        "matched_unique_objid": int(len(raw)),
        "columns": list(raw.columns),
        "output": str(out.relative_to(HERE)),
    }


def main() -> None:
    GALLAZZI.mkdir(parents=True, exist_ok=True)
    FIREFLY.mkdir(parents=True, exist_ok=True)
    SDSS_AUX.mkdir(parents=True, exist_ok=True)
    for path, url in DOWNLOADS.items():
        print(f"download/check {path.name}", flush=True)
        download(url, path)
    for pdf in [GALLAZZI / "Gallazzi2005_arxiv.pdf", FIREFLY / "Comparat2019_arxiv.pdf"]:
        extract_pdf_text(pdf)
    target = target_pmf()
    gallazzi = subset_gallazzi(target)
    firefly = subset_firefly(target)
    sdss_aux = fetch_sdss_aux(target_objids())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_spectrum_definition": "Claude work/metallicity_worktable.csv chosen plate-MJD-fibre; parent rows unchanged",
        "matching_key": ["plate", "mjd", "fiberid"],
        "downloads": [
            {"path": str(path.relative_to(HERE)), "url": url, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path, url in DOWNLOADS.items()
        ],
        "gallazzi": gallazzi,
        "firefly": firefly,
        "sdss_aux": sdss_aux,
    }
    with (CAT / "catalogue_manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps({"gallazzi": gallazzi, "firefly": {k: v for k, v in firefly.items() if k != "columns"}, "sdss_aux": sdss_aux}, indent=2))


if __name__ == "__main__":
    main()
