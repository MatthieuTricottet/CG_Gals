"""Recover Galaxy Zoo 1 votes through the archived DR7 catalogue.

The group samples inherit their GZ1 debiased vote fractions (``p_E``,
``p_S``) from the DR16 ``zooSpec`` join on the photometric objid.  That join
returns no votes for part of the sample.  The original GZ1 catalogue is keyed
to DR7 objids, so those rows can be recovered through the SkyServer DR7
bridge cached in ``data/sdss_size_columns.csv``.  The rule frozen on
20 November 2026 (``results/referee_freeze_20261120``) is applied here:

* a DR7 counterpart is *verified* when its bridge identifier is not shared by
  several DR16 objects and the archived position lies within 1.5 arcsec of
  the catalogue position;
* a row without finite DR16 votes takes the verified archival votes, so it
  may become E, S or U;
* a finite-vote uncertain row takes the archival votes only when they are
  decisive (E or S);
* every other existing classification is kept, including previously
  classified rows without a verified archival counterpart.

``gz1_source`` records which catalogue supplied the votes.  The archive and
the existing DR16 classes agree for every verified row, so the second rule
changes no label in practice; it is kept to mirror the frozen rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.coordinates import Angle, angular_separation

import config as co

ARCHIVE_FILE = co.BASE_PATH + "results/referee_tests_20261117/inputs/gz1_paper1_crossmatch.csv"
BRIDGE_FILE = co.SIZE_COLUMNS_FILE
MAX_SEPARATION_ARCSEC = 1.5
SOURCE_DR16 = "DR16 zooSpec"
SOURCE_DR7 = "DR7 archive"
SOURCE_NONE = "none"
GROUP_TABLES = [name + co.GASUFF for name in co.SAMPLE]


def _vote_class(p_e: pd.Series, p_s: pd.Series) -> pd.Series:
    finite = p_e.notna() & p_s.notna()
    label = pd.Series("missing", index=p_e.index, dtype=object)
    label[finite] = "U"
    label[finite & (p_e > 0.5)] = "E"
    label[finite & ~(p_e > 0.5) & (p_s > 0.5)] = "S"
    return label


def _archive() -> pd.DataFrame:
    archive = pd.read_csv(ARCHIVE_FILE, dtype={"OBJID": "Int64"})
    if not archive["OBJID"].is_unique:
        raise ValueError("GZ1 archive cross-match must be unique in DR7 objid")
    archive["archive_ra"] = Angle(archive["RA"].tolist(), unit="hourangle").degree
    archive["archive_dec"] = Angle(archive["DEC"].tolist(), unit="deg").degree
    return archive.rename(
        columns={
            "OBJID": "dr7objid",
            "P_EL_DEBIASED": "archive_p_E",
            "P_CS_DEBIASED": "archive_p_S",
            "NVOTE": "archive_nvote",
        }
    )[["dr7objid", "archive_ra", "archive_dec", "archive_p_E", "archive_p_S", "archive_nvote"]]


def _bridge() -> pd.DataFrame:
    bridge = pd.read_csv(BRIDGE_FILE, dtype={"objid": str, "dr7objid": str})
    bridge = bridge.dropna(subset=["dr7objid"])
    bridge = bridge[bridge["dr7objid"] != "0"]
    shared = bridge.groupby("dr7objid")["objid"].nunique()
    bridge["bridge_collision"] = bridge["dr7objid"].map(shared).gt(1)
    bridge["objid"] = bridge["objid"].astype("int64")
    bridge["dr7objid"] = bridge["dr7objid"].astype("int64").astype("Int64")
    return bridge[["objid", "dr7objid", "bridge_collision"]].drop_duplicates("objid")


def archival_matches(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the verified archival GZ1 votes for the rows of ``frame``."""

    rows = frame[["objid", "RA", "Dec"]].copy()
    rows["objid"] = pd.to_numeric(rows["objid"]).astype("int64")
    rows = rows.merge(_bridge(), on="objid", how="left", validate="m:1")
    rows = rows.merge(_archive(), on="dr7objid", how="left", validate="m:1")
    separation = angular_separation(
        np.deg2rad(rows["RA"].to_numpy(float)),
        np.deg2rad(rows["Dec"].to_numpy(float)),
        np.deg2rad(rows["archive_ra"].to_numpy(float)),
        np.deg2rad(rows["archive_dec"].to_numpy(float)),
    )
    rows["archive_sep_arcsec"] = np.rad2deg(np.asarray(separation)) * 3600
    rows["archive_verified"] = (
        rows["archive_sep_arcsec"].le(MAX_SEPARATION_ARCSEC)
        & ~rows["bridge_collision"].eq(True)
        & rows[["archive_p_E", "archive_p_S"]].notna().all(axis=1)
    )
    rows.index = frame.index
    return rows


def recover_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill or resolve GZ1 votes of one galaxy table with verified DR7 votes."""

    out = frame.copy()
    matches = archival_matches(out)
    p_e = pd.to_numeric(out["p_E"], errors="coerce")
    p_s = pd.to_numeric(out["p_S"], errors="coerce")
    current = _vote_class(p_e, p_s)
    archive = _vote_class(matches["archive_p_E"], matches["archive_p_S"])
    verified = matches["archive_verified"]
    replace = verified & (
        current.eq("missing") | (current.eq("U") & archive.isin(["E", "S"]))
    )
    out["gz1_source"] = np.where(current.ne("missing"), SOURCE_DR16, SOURCE_NONE)
    out.loc[replace, "p_E"] = matches.loc[replace, "archive_p_E"].astype(float)
    out.loc[replace, "p_S"] = matches.loc[replace, "archive_p_S"].astype(float)
    out.loc[replace, "gz1_source"] = SOURCE_DR7
    out["gz1_archive_verified"] = verified.to_numpy()
    return out


def apply_to_sample(sample: dict, reclassify: bool = True) -> dict:
    """Apply the archival recovery to the four galaxy tables of ``sample``.

    The same galaxy can occur in several samples; its votes are recovered
    from the same archival row each time.  The SDSS reference tables are
    filled for those objids only, so a galaxy keeps one GZ1 class wherever
    it appears (the within-host analysis reads its votes from
    ``SDSS_withAGN``).  ``morphology`` is recomputed when ``reclassify``.
    """

    import morphologies

    recovered = []
    for name in GROUP_TABLES:
        if name not in sample:
            continue
        frame = recover_frame(sample[name])
        if reclassify:
            frame = morphologies.classify(frame)
        sample[name] = frame
        recovered.append(frame.loc[frame["gz1_source"].eq(SOURCE_DR7), ["objid", "p_E", "p_S"]])
    votes = pd.concat(recovered).drop_duplicates("objid") if recovered else None
    if votes is not None and len(votes):
        votes = votes.assign(objid=votes["objid"].astype("int64")).set_index("objid")
        for name in ["SDSS", "SDSS_withAGN"]:
            if name not in sample or "p_E" not in sample[name]:
                continue
            table = sample[name].copy()
            key = pd.to_numeric(table["objid"]).astype("int64")
            fill = key.isin(votes.index) & (
                table["p_E"].isna() | table["p_S"].isna()
            )
            table.loc[fill, "p_E"] = key[fill].map(votes["p_E"]).to_numpy()
            table.loc[fill, "p_S"] = key[fill].map(votes["p_S"]).to_numpy()
            if reclassify and "morphology" in table:
                table = morphologies.classify(table)
            sample[name] = table
    return sample


def recovered_votes(sample: dict) -> pd.DataFrame:
    """Votes recovered from the archive, one row per objid (after ``apply``)."""

    pieces = [
        sample[name].loc[sample[name]["gz1_source"].eq(SOURCE_DR7), ["objid", "p_E", "p_S"]]
        for name in GROUP_TABLES
        if name in sample and "gz1_source" in sample[name]
    ]
    if not pieces:
        return pd.DataFrame(columns=["objid", "p_E", "p_S"])
    votes = pd.concat(pieces).drop_duplicates("objid")
    return votes.assign(objid=votes["objid"].astype("int64"))


def summary(sample: dict) -> dict:
    """Counts used in the manuscript: recovered, retained and missing labels."""

    out = {
        "rule": (
            "DR7 bridge without shared identifiers, archived position within "
            f"{MAX_SEPARATION_ARCSEC} arcsec; missing votes take the archival votes, "
            "an uncertain class takes decisive archival votes, other classes are kept"
        ),
        "max_separation_arcsec": MAX_SEPARATION_ARCSEC,
        "samples": {},
    }
    for name in GROUP_TABLES:
        if name not in sample or "gz1_source" not in sample[name]:
            continue
        frame = sample[name]
        label = _vote_class(
            pd.to_numeric(frame["p_E"], errors="coerce"),
            pd.to_numeric(frame["p_S"], errors="coerce"),
        )
        dr7 = frame["gz1_source"].eq(SOURCE_DR7)
        dr16 = frame["gz1_source"].eq(SOURCE_DR16)
        satellite = pd.to_numeric(frame["rank_M"], errors="coerce").gt(1)
        block = {
            "n_galaxies": int(len(frame)),
            "n_dr16": int(dr16.sum()),
            "n_dr16_verified_archive": int((dr16 & frame["gz1_archive_verified"]).sum()),
            "n_dr16_without_verified_archive": int((dr16 & ~frame["gz1_archive_verified"]).sum()),
            "n_missing": int(label.eq("missing").sum()),
        }
        for key, mask in [("recovered", dr7), ("recovered_satellites", dr7 & satellite)]:
            block[key] = {cls: int((mask & label.eq(cls)).sum()) for cls in ["E", "S", "U"]}
            block[key]["total"] = int(mask.sum())
        block["final"] = {cls: int(label.eq(cls).sum()) for cls in ["E", "S", "U", "missing"]}
        out["samples"][name.removesuffix(co.GASUFF)] = block
    return out
