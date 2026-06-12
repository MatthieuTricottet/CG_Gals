"""Catalogue harmonization for the extended specialness analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import Planck15


SAMPLE_KEYS = {
    "CG4": "CG4_Gals",
    "Control4B": "Control4B_Gals",
    "Control4C": "Control4C_Gals",
    "RG4": "RG4_Gals",
}
CONTROL_SAMPLES = ["Control4B", "Control4C", "RG4"]
C_KMS = 299792.458


def first_existing(frame: pd.DataFrame, names: list[str]) -> str | None:
    """Return the first available column name from an ordered alias list."""

    return next((name for name in names if name in frame.columns), None)


def _numeric(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    column = first_existing(frame, names)
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _merge_group_columns(galaxies: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    if "Group" not in galaxies or "Group" not in groups:
        return galaxies
    group_frame = groups.copy().drop_duplicates("Group")
    rename = {
        column: f"group_{column}"
        for column in group_frame.columns
        if column != "Group" and column in galaxies.columns
    }
    return galaxies.merge(
        group_frame.rename(columns=rename), on="Group", how="left", validate="m:1"
    )


def _merge_sdss_columns(
    frame: pd.DataFrame, samples: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    sdss = samples.get("SDSS_withAGN")
    if sdss is None or "objid" not in frame or "objid" not in sdss:
        return frame
    desired = [
        "objid",
        "h_alpha_eqw",
        "h_beta_eqw",
        "oiii_5007_eqw",
        "nii_6584_eqw",
        "h_alpha_flux",
        "h_beta_flux",
        "oiii_5007_flux",
        "nii_6584_flux",
        "log_NII_Ha",
        "log_OIII_Hb",
        "is_AGN",
        "u_obs",
        "g_obs",
        "r_obs",
        "i_obs",
        "z_obs",
    ]
    columns = [column for column in desired if column in sdss.columns]
    spectral = sdss[columns].drop_duplicates("objid").copy()
    rename = {
        column: f"sdss_{column}"
        for column in columns
        if column != "objid" and column in frame
    }
    return frame.merge(
        spectral.rename(columns=rename), on="objid", how="left", validate="m:1"
    )


def build_galaxy_frame(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine the four galaxy samples and derive common analysis columns."""

    pieces = []
    for label, galaxy_key in SAMPLE_KEYS.items():
        if galaxy_key not in samples:
            continue
        galaxies = samples[galaxy_key].copy()
        group_key = galaxy_key.replace("_Gals", "_Groups")
        if group_key in samples:
            galaxies = _merge_group_columns(galaxies, samples[group_key])
        galaxies["sample"] = label
        galaxies["is_CG4"] = int(label == "CG4")
        galaxies["group_uid"] = (
            label
            + ":"
            + galaxies.get(
                "Group", pd.Series(galaxies.index, index=galaxies.index)
            ).astype(str)
        )
        pieces.append(galaxies)
    if not pieces:
        return pd.DataFrame()

    frame = pd.concat(pieces, ignore_index=True, sort=False)
    frame = _merge_sdss_columns(frame, samples)

    frame["logMstar"] = _numeric(frame, ["lgm_tot_p50", "lgm", "logMstar"])
    frame["rank"] = _numeric(frame, ["rank_M_CG", "rank_M_LT", "rank_M"])
    frame["is_satellite"] = np.where(
        frame["rank"].notna(), (frame["rank"] > 1).astype(float), np.nan
    )
    frame["is_bgg"] = np.where(
        frame["rank"].notna(), (frame["rank"] == 1).astype(float), np.nan
    )

    status = (
        frame.get("sSFR_status", pd.Series("", index=frame.index))
        .astype(str)
        .str.lower()
    )
    frame["passive"] = np.where(
        status.ne("nan") & status.ne(""),
        status.isin(["passive", "quenched"]).astype(float),
        np.nan,
    )
    frame["starforming"] = np.where(
        status.ne("nan") & status.ne(""),
        status.eq("starforming").astype(float),
        np.nan,
    )
    morphology = (
        frame.get("morphology", pd.Series("", index=frame.index))
        .astype(str)
        .str.lower()
    )
    classified_morphology = morphology.isin(["elliptical", "spiral"])
    frame["elliptical"] = np.where(
        classified_morphology, morphology.eq("elliptical").astype(float), np.nan
    )
    frame["spiral"] = np.where(
        classified_morphology, morphology.eq("spiral").astype(float), np.nan
    )

    frame["z_numeric"] = _numeric(frame, ["z"])
    frame["z_group_numeric"] = _numeric(frame, ["z_group", "group_z_group"])
    missing_group_z = frame["z_group_numeric"].isna()
    frame.loc[missing_group_z, "z_group_numeric"] = frame.groupby("group_uid")[
        "z_numeric"
    ].transform("median")[missing_group_z]
    frame["log_group_mass"] = np.log10(
        _numeric(frame, ["M_group", "group_M_group"]).where(lambda values: values > 0)
    )
    frame["log_group_luminosity"] = np.log10(
        _numeric(frame, ["Lum_group", "group_Lum_group"]).where(
            lambda values: values > 0
        )
    )
    frame["velocity_dispersion"] = _numeric(frame, ["Vdisp", "sigma_v", "group_Vdisp"])
    frame["dominance"] = _numeric(
        frame, ["FracLumBGG", "group_FracLumBGG", "is_dominated", "group_is_dominated"]
    )

    angular_proxy = _numeric(frame, ["dist2BGG"])
    kpc_per_arcmin = (
        Planck15.kpc_proper_per_arcmin(
            frame["z_group_numeric"].fillna(frame["z_numeric"]).to_numpy()
        )
        .to(u.kpc / u.arcmin)
        .value
    )
    theta_arcmin = ((angular_proxy.to_numpy() / 3600.0) * u.rad).to(u.arcmin).value
    frame["dist2BGG_kpc"] = theta_arcmin * kpc_per_arcmin

    scale_column = first_existing(
        frame,
        [
            "R_180",
            "group_R_180",
            "mean_Rij",
            "group_mean_Rij",
            "<Rij>",
            "group_<Rij>",
            "size_Group_Bary_kpc",
            "group_size_Group_Bary_kpc",
        ],
    )
    if scale_column:
        frame["R_scale"] = pd.to_numeric(frame[scale_column], errors="coerce")
        frame["R_norm"] = frame["dist2BGG_kpc"] / frame["R_scale"]
        frame.attrs["radius_scale_column"] = scale_column
    else:
        frame["R_scale"] = np.nan
        frame["R_norm"] = frame["dist2BGG_kpc"]
        frame.attrs["radius_scale_column"] = None

    frame["delta_v"] = (
        C_KMS
        * (frame["z_numeric"] - frame["z_group_numeric"])
        / (1 + frame["z_group_numeric"])
    )
    frame["V_norm"] = frame["delta_v"].abs() / frame["velocity_dispersion"]

    for colour, left, right in [
        ("u_minus_r", "u_obs", "r_obs"),
        ("u_minus_g", "u_obs", "g_obs"),
        ("g_minus_r", "g_obs", "r_obs"),
        ("r_minus_i", "r_obs", "i_obs"),
    ]:
        if left in frame and right in frame:
            frame[colour] = pd.to_numeric(frame[left], errors="coerce") - pd.to_numeric(
                frame[right], errors="coerce"
            )
    return frame


def ensure_galaxy_frame(data) -> pd.DataFrame:
    """Accept either the pipeline sample dictionary or a prebuilt galaxy frame."""

    if isinstance(data, pd.DataFrame):
        return data.copy()
    return build_galaxy_frame(data)
