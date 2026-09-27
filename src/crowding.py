"""Close projected neighbours: the crowding flag and what it touches.

The nearest neighbour of every sample galaxy is searched in the full Lim et
al. (2017) SDSS DR7 galaxy catalogue (``data/SDSS(L) galaxy.dat``; all
redshifts, since fibre collisions and image blending depend on angular
separation only), not only among the other three quartet members.  A galaxy
is *crowded* when that neighbour lies within 55 arcsec, the SDSS fibre
collision scale.  The Lim catalogue is spectroscopic: companions that never
received a redshift are absent from it, so the flag is a lower limit on
crowding and cannot recover fibre-collision losses; it identifies the
galaxies whose own measurements are most exposed to blending and to the
collision pattern.

Three uses:

* the morphology contrast after excluding crowded galaxies, and with the flag
  as a covariate (E/(E+S) fractions use classified galaxies as denominator);
* the association of missing sSFR and missing Galaxy Zoo votes with
  crowding, luminosity and BGG status (the missing-data diagnostics);
* the flag shared by the size and selection diagnostics.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    import config as co
    from extended_data import dedup_control_pool, ensure_galaxy_frame
    from extended_stats import fit_logistic_model, safe_json
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from .extended_data import dedup_control_pool, ensure_galaxy_frame
    from .extended_stats import fit_logistic_model, safe_json

THRESHOLD_ARCSEC = 55.0
LIM_GALAXY_FILE = os.path.join(co.DATA_PATH, "SDSS(L) galaxy.dat")
LIM_COLUMNS = [
    "galaxy_id", "objid", "lim_group", "ra", "dec", "l", "b", "z_cmb",
    "z_edd", "z_comp", "z_src", "dist_nn", "log_L", "log_Ms", "colour",
]
SAMPLES = ["CG4", "Control4B", "Control4C", "RG4"]
CONTROLS = ["Control4B", "Control4C", "RG4"]
C_KMS = 299792.458


def load_lim_galaxies() -> pd.DataFrame:
    """The full Lim et al. (2017) SDSS galaxy catalogue (cached per process)."""

    cached = getattr(load_lim_galaxies, "_cache", None)
    if cached is not None:
        return cached
    table = pd.read_csv(
        LIM_GALAXY_FILE, sep=r"\s+", comment="#", header=None, names=LIM_COLUMNS
    )
    load_lim_galaxies._cache = table
    return table


def nearest_lim_neighbour(frame: pd.DataFrame) -> pd.DataFrame:
    """Angular separation (arcsec) and velocity offset of the nearest Lim galaxy.

    The galaxy itself is excluded by object identifier and by requiring a
    separation above 1 arcsec (duplicate photometric entries).
    """

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    lim = load_lim_galaxies()
    targets = pd.DataFrame(index=frame.index)
    objid = frame["objid"] if "objid" in frame else pd.Series(-1, index=frame.index)
    targets["objid"] = pd.to_numeric(objid, errors="coerce").fillna(-1).astype("int64")
    targets["RA"] = pd.to_numeric(frame["RA"], errors="coerce")
    targets["Dec"] = pd.to_numeric(frame["Dec"], errors="coerce")
    redshift = frame["z_numeric"] if "z_numeric" in frame else pd.Series(np.nan, index=frame.index)
    targets["z_numeric"] = pd.to_numeric(redshift, errors="coerce")
    coords = SkyCoord(targets["RA"].to_numpy() * u.deg, targets["Dec"].to_numpy() * u.deg)
    catalogue = SkyCoord(lim["ra"].to_numpy() * u.deg, lim["dec"].to_numpy() * u.deg)
    radius = 600.0 * u.arcsec  # wide enough for a true nearest-neighbour distance
    idx_target, idx_lim, sep, _ = catalogue.search_around_sky(coords, radius)
    pairs = pd.DataFrame(
        {"row": idx_target, "lim": idx_lim, "sep_arcsec": sep.to_value(u.arcsec)}
    )
    pairs = pairs.loc[
        (lim["objid"].to_numpy()[pairs["lim"]] != targets["objid"].to_numpy()[pairs["row"]])
        & (pairs["sep_arcsec"] > 1.0)
    ]
    nearest = pairs.sort_values("sep_arcsec").drop_duplicates("row")
    out = pd.DataFrame(index=frame.index)
    out["nearest_neighbour_arcsec"] = np.nan
    out["nearest_neighbour_dv_kms"] = np.nan
    rows = frame.index.to_numpy()[nearest["row"].to_numpy()]
    out.loc[rows, "nearest_neighbour_arcsec"] = nearest["sep_arcsec"].to_numpy()
    z_self = targets["z_numeric"].to_numpy()[nearest["row"].to_numpy()]
    z_nb = lim["z_cmb"].to_numpy()[nearest["lim"].to_numpy()]
    out.loc[rows, "nearest_neighbour_dv_kms"] = C_KMS * np.abs(z_nb - z_self) / (1 + z_self)
    return out


def attach_crowding(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the Lim nearest-neighbour separation and the 55-arcsec flag."""

    out = frame.copy()
    if "nearest_neighbour_arcsec" not in out:
        neighbours = nearest_lim_neighbour(out)
        out = out.join(neighbours)
    separation = out["nearest_neighbour_arcsec"]
    # NaN (no Lim galaxy within the search radius) compares False: not crowded
    out["crowded_55arcsec"] = (separation < THRESHOLD_ARCSEC).astype(float)
    out.attrs = dict(frame.attrs)  # keep upstream audits (e.g. size attachment)
    return out


def _quartet_nearest(frame: pd.DataFrame) -> pd.Series:
    """Nearest *quartet co-member* separation (the superseded definition)."""

    values = pd.Series(np.nan, index=frame.index)
    for _, group in frame.groupby("group_uid", observed=True):
        if len(group) < 2:
            continue
        ra = np.deg2rad(group["RA"].to_numpy(dtype=float))
        dec = np.deg2rad(group["Dec"].to_numpy(dtype=float))
        hav = (
            np.sin((dec[:, None] - dec[None, :]) / 2) ** 2
            + np.cos(dec[:, None]) * np.cos(dec[None, :])
            * np.sin((ra[:, None] - ra[None, :]) / 2) ** 2
        )
        angle = 2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
        np.fill_diagonal(angle, np.inf)
        values.loc[group.index] = np.degrees(angle.min(axis=1)) * 3600.0
    return values


def morphology_after_exclusion(frame: pd.DataFrame) -> dict:
    """E/(E+S) fractions and adjusted E-class odds ratios without crowded galaxies."""

    try:
        from primary_contrasts import run_primary_contrasts
    except ModuleNotFoundError:  # pragma: no cover
        from .primary_contrasts import run_primary_contrasts

    fractions = {}
    for sample_name in SAMPLES:
        part = frame.loc[frame["sample"] == sample_name]
        kept = part.loc[part["crowded_55arcsec"] != 1]
        fractions[sample_name] = {
            "n": int(len(part)),
            "n_crowded": int((part["crowded_55arcsec"] == 1).sum()),
            "crowded_fraction": float((part["crowded_55arcsec"] == 1).mean()),
            "E_fraction_classified_full": float(part["elliptical"].mean()),
            "E_fraction_classified_uncrowded": float(kept["elliptical"].mean()),
            "n_classified_uncrowded": int(kept["elliptical"].notna().sum()),
        }
    kept = frame.loc[frame["crowded_55arcsec"] != 1].copy()
    excluded = run_primary_contrasts(kept, frame=kept)
    with_flag = {}
    try:
        from specialness_models import _covariates
    except ModuleNotFoundError:  # pragma: no cover
        from .specialness_models import _covariates
    covariates, continuous = _covariates(frame)
    for control in CONTROLS:
        subset = frame.loc[frame["sample"].isin(["CG4", control])]
        with_flag[control] = fit_logistic_model(
            subset,
            "elliptical",
            ["is_CG4", "crowded_55arcsec", *covariates],
            continuous=continuous,
        )
    return {
        "fractions": fractions,
        "excluded_contrasts": excluded.get("contrasts", {}),
        "with_flag": with_flag,
    }


def missing_data_diagnostics(frame: pd.DataFrame) -> dict:
    """How missing sSFR and missing Galaxy Zoo votes relate to crowding.

    Rates are computed on the deduplicated pool (one row per physical
    galaxy).  The model uses the r-band absolute magnitude, which unlike the
    MPA-JHU stellar mass exists for every row, including the BOSS spectra
    that make up most of the missing class.
    """

    pool = dedup_control_pool(frame)
    pool = pool.assign(
        missing_ssfr=pool["sSFR_status"].astype(str).eq(co.NosSFR_LABEL).astype(float),
        missing_gz=pool["morphology"].astype(str).eq(co.NoMorphology_LABEL).astype(float),
        M_r_numeric=pd.to_numeric(pool["M_r"], errors="coerce"),
    )
    rates = {}
    for label, mask in (
        ("crowded", pool["crowded_55arcsec"] == 1),
        ("uncrowded", pool["crowded_55arcsec"] != 1),
        ("bgg", pool["is_bgg"] == 1),
        ("satellite", pool["is_satellite"] == 1),
    ):
        part = pool.loc[mask]
        rates[label] = {
            "n": int(len(part)),
            "missing_ssfr": float(part["missing_ssfr"].mean()),
            "missing_gz": float(part["missing_gz"].mean()),
        }
    missing = pool.loc[pool["missing_ssfr"] == 1]
    classified = pool.loc[pool["missing_ssfr"] == 0]
    by_sample = {}
    for sample_name in SAMPLES:
        part = frame.loc[frame["sample"] == sample_name]
        crowded = part["crowded_55arcsec"] == 1
        missing_mask = part["sSFR_status"].astype(str).eq(co.NosSFR_LABEL)
        by_sample[sample_name] = {
            "missing_ssfr_if_crowded": float(missing_mask[crowded].mean()),
            "missing_ssfr_if_uncrowded": float(missing_mask[~crowded].mean()),
        }
    model = fit_logistic_model(
        pool,
        "missing_ssfr",
        ["is_CG4", "crowded_55arcsec", "M_r_numeric", "z_numeric", "is_satellite"],
        continuous=["M_r_numeric", "z_numeric"],
    )
    return {
        "rates": rates,
        "crowded_fraction_among_missing_ssfr": float((missing["crowded_55arcsec"] == 1).mean()),
        "crowded_fraction_among_classified": float((classified["crowded_55arcsec"] == 1).mean()),
        "missing_gz_among_missing_ssfr": float(missing["missing_gz"].mean()),
        "logMstar_available_among_missing_ssfr": float(missing["logMstar"].notna().mean()),
        "by_sample": by_sample,
        "missingness_model": model,
    }


def run_crowding_analysis(data, output_dir: str | None = None) -> dict:
    """Crowding fractions, the morphology contrast without crowded galaxies,
    and the missing-data diagnostics."""

    del output_dir
    frame = attach_crowding(ensure_galaxy_frame(data))
    quartet = _quartet_nearest(frame) < THRESHOLD_ARCSEC
    return safe_json(
        {
            "status": "ok",
            "threshold_arcsec": THRESHOLD_ARCSEC,
            "neighbour_catalogue": "Lim et al. (2017) SDSS DR7 galaxies, all redshifts",
            "crowded_fraction": {
                s: float((frame.loc[frame["sample"] == s, "crowded_55arcsec"] == 1).mean())
                for s in SAMPLES
            },
            "crowded_fraction_quartet_only": {
                s: float(quartet[frame["sample"] == s].mean()) for s in SAMPLES
            },
            "crowded_fraction_pooled_controls": float(
                (dedup_control_pool(frame).query("is_CG4 == 0")["crowded_55arcsec"] == 1).mean()
            ),
            "median_neighbour_arcsec": {
                s: float(frame.loc[frame["sample"] == s, "nearest_neighbour_arcsec"].median())
                for s in SAMPLES
            },
            "morphology": morphology_after_exclusion(frame),
            "missing_data": missing_data_diagnostics(frame),
        }
    )
