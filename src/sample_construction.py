"""Reconstruction of the Control4C sample from the parent catalogue.

Paper I (Tricottet, Mamon & Diaz-Gimenez 2025, Sect. 2.4) builds three
control samples from the 765 parent (PC) Lim et al. (2017) groups and
excludes every control group that contains at least one galaxy of the
*full* CG4 sample (split compact groups included):

* Control4B: the four brightest members       (66 excluded -> 699 groups)
* Control4C: BGG + 3 closest projected members (Paper I: 61 -> 704 groups)
* RG4:       groups of exactly four members     (6 excluded -> 56 groups)

The committed ``Control4C_Gals.csv`` predates that exclusion and derives
from an older parent-catalogue revision (752 groups, 14 CG4 galaxies, one
duplicated row). This module regenerates Control4C from the committed
``PC_Gals.csv``; that reproducibly yields 60 exclusions -> 705 groups (the
61 -> 704 of Paper I is not reproducible from the committed parent file;
see OPEN_QUESTIONS.md #1).

Group-level quartet properties reproduce the committed ``Control4B_Groups``
/ ``RG4_Groups`` generation (originally ``common.py::Group_agg``) exactly;
every formula below was validated against those files at machine precision
(see tests/test_sample_construction.py):

* velocity dispersion: gapper (Wainer & Thissen 1976) on member redshifts,
  scaled by c/(1+z_group), z_group = plain mean member redshift;
* size_Group_Bary_kpc: median pairwise separation (arcmin) converted with
  the *luminosity* distance at z_group (Planck15-like cosmology, H0=67.8,
  Om0=0.308);
* M_virial = 3 pi sigma^2 R_h / G with R_h the harmonic mean pairwise
  *projected* separation (angular-diameter distance);
* t_cr = 0.887 size_Group_Bary_kpc / Vdisp;
* lMass_200 solves Eq. A11 of Paper I from the Yang logM_180 mass;
  r_200 = (G M_200 / (100 H(z)^2))^(1/3);
* Misfit_Bary / Vmisfit: median split of Offset_Bary / Voffset within the
  sample.

Run as a script to regenerate ``data/Control4C_Gals.csv`` and
``data/Control4C_Groups.csv``.
"""

from __future__ import annotations

import itertools as it
import os

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from scipy import optimize

try:
    import config as co
    from utils import astro_utils as au
    from utils import spherical_utils as sphu
    from utils import stats_utils as su
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from .utils import astro_utils as au
    from .utils import spherical_utils as sphu
    from .utils import stats_utils as su

# Paper I cosmology (common.py::glob)
H0 = 67.8
OM0 = 0.308
COSMO = FlatLambdaCDM(H0=H0, Om0=OM0, Tcmb0=2.7255, Neff=3.15)
C_KMS = 299792.458
G_SI = 6.6743e-11
KPC_M = (1 * u.kpc).to(u.m).value
MSUN_KG = 1.98892e30
MR_SUN = 4.68
T_CR_COEFF = 0.887
SSFR_WIDTH = 0.4
# Projected virial-mass constant: physically 3*pi, calibrated on the
# committed Control4B_Groups at 1.4e-9 relative scatter. The 1.9e-4 excess
# over 3*pi absorbs the legacy unit constants (G = 6.673e-11 etc.) of the
# original Paper I pipeline, so regenerated tables stay consistent with the
# committed Control4B/RG4 files when combined with G_SI/MSUN_KG below.
VIRIAL_CONST = 9.4265762253

GROUP_COLUMNS = [
    "Group", "Lum_BGG", "Lum_group", "FracLumBGG", "z_group", "DeltaR12",
    "NbGal", "RA_BGG", "Dec_BGG", "RA_Bary", "Dec_Bary", "Radius_Bary_arcmin",
    "Offset_Bary", "V_BGG", "V_moy", "Vdisp", "Voffset",
    "size_Group_Bary_kpc", "M_group", "M_virial", "M_virial_over_L", "t_cr",
    "BGG_SFRcategory", "all_SFR", "Prop_M_Sat", "Prop_M_Tot", "Prop_G_Sat",
    "Prop_G_Tot", "Prop_Q_Sat", "Prop_Q_Tot", "Misfit_Bary", "Vmisfit",
    "lMass_200", "r_200_kpc",
]

GALS_COLUMNS = [
    "objid", "specobjid", "Group", "RA", "Dec", "M_r", "z", "Lum", "rank_M",
    "RA_BGG", "Dec_BGG", "M_BGG", "dist2BGG", "lgm", "sfr", "sSFR",
    "rank_dist", "rank_M_parent",
]


def sfr_category(ssfr, lgm) -> np.ndarray:
    """Knobel+15 star-formation category: M / G / Q, X when unmeasured."""

    excess = np.asarray(ssfr, dtype=float) - (-0.3 * np.asarray(lgm, dtype=float) - 7.85)
    conditions = [excess > SSFR_WIDTH,
                  (excess <= SSFR_WIDTH) & (excess >= -SSFR_WIDTH),
                  excess < -SSFR_WIDTH]
    return np.select(conditions, ["M", "G", "Q"], default="X")


def _pairwise_separations_rad(ra_deg, dec_deg) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    seps = []
    for i, j in it.combinations(range(len(ra)), 2):
        seps.append(sphu.calc_sep(ra[i], dec[i], ra[j], dec[j]))
    return np.asarray(seps)


def _m_tilde(x):
    return (np.log(x + 1) - x / (x + 1)) / (np.log(2) - 0.5)


def _c_lcdm(mass, z):
    norm = 10 ** (0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21))
    slope = -0.101 + 0.026 * z
    return norm * ((H0 / 100) * mass / 1e12) ** slope


def _eq_a11(m_200, lm180, z):
    # NFW conversion M_180m -> M_200c (Paper I, Eq. A11):
    #   M_tilde(c * r_180m/r_200c) / M_tilde(c) * M_200 = M_180m
    # with r_180m/r_200c = (0.9 Om0)^(-1/3) E(z)^(2/3) (1+z)^(-1)
    #                      * (M_180m/M_200c)^(1/3).
    # Nota bene: the legacy common.py::EqA11 snippet divides by M_tilde(c)
    # *inside* the M_tilde argument; the committed Control4B/RG4 lMass_200
    # values match this correct form (to ~3e-3 dex), not the snippet.
    m_yang = 10**lm180
    f1 = (0.9 * OM0) ** (-1 / 3)
    f2 = _c_lcdm(m_200, z)
    f3 = (COSMO.H(z).value / H0) ** (2 / 3) / (1 + z)
    f4 = (m_yang / m_200) ** (1 / 3)
    return np.log10(_m_tilde(f1 * f2 * f3 * f4) / _m_tilde(f2) * m_200) - lm180


def lmass_200(lm180, z) -> float:
    """Solve Paper I Eq. A11 for M_200c given the Yang logM_180m mass."""

    if not np.isfinite(lm180):
        return np.nan
    try:
        m_200 = optimize.brentq(_eq_a11, 10 ** (lm180 - 2), 10 ** (lm180 + 2),
                                args=(lm180, z))
    except ValueError:
        return np.nan
    return float(np.log10(m_200))


def r_200_kpc(lmass, z) -> float:
    """r_200c in kpc from (G M_200 / (100 H(z)^2))^(1/3)."""

    h_z = COSMO.H(z).to(u.km / u.s / u.kpc).value  # km/s/kpc
    g_kpc = G_SI * MSUN_KG / KPC_M / 1e6  # G in kpc (km/s)^2 / Msun
    return float((g_kpc * 10**lmass / (100 * h_z**2)) ** (1 / 3))


def quartet_group_properties(members: pd.DataFrame,
                             lm180: float | None = None) -> pd.Series:
    """Recompute the Paper I group-level properties for one quartet."""

    x = members.sort_values("M_r").reset_index(drop=True)
    lum = 10 ** (-0.4 * (x["M_r"] - MR_SUN))
    lum_bgg = float(lum.iloc[0])
    lum_group = float(lum.sum())
    m_group = float(-2.5 * np.log10(lum_group) + MR_SUN)
    z_group = float(x["z"].mean())

    velocity = C_KMS * (x["z"] - z_group) / (1 + z_group)
    v_bgg = float(velocity.iloc[0])
    v_moy = float(velocity.mean())
    vdisp = float(su.V_disp_gapper(x))
    voffset = abs(v_bgg - v_moy) / vdisp

    ra_bary, dec_bary = sphu.calc_bary(x)
    radius_bary_arcmin = float(sphu.calc_diameter_arcmin(x))
    offset_abs_arcmin = np.rad2deg(
        sphu.calc_sep(np.deg2rad(ra_bary), np.deg2rad(dec_bary),
                      np.deg2rad(x.iloc[0]["RA"]), np.deg2rad(x.iloc[0]["Dec"]))
    ) * 60
    offset_bary = offset_abs_arcmin / radius_bary_arcmin

    dist_lum_kpc = COSMO.luminosity_distance(z_group).to(u.kpc).value
    arcmin_to_rad = np.pi / (180 * 60)
    size_kpc = radius_bary_arcmin * arcmin_to_rad * dist_lum_kpc

    seps = _pairwise_separations_rad(x["RA"], x["Dec"])
    harmonic_rad = len(seps) / np.sum(1.0 / seps)
    dist_ang_kpc = dist_lum_kpc / (1 + z_group) ** 2
    m_virial = (VIRIAL_CONST * (vdisp * 1e3) ** 2
                * (harmonic_rad * dist_ang_kpc * KPC_M) / G_SI / MSUN_KG)
    t_cr = T_CR_COEFF * size_kpc / vdisp

    category = sfr_category(x["sSFR"], x["lgm"])
    props = {
        "Lum_BGG": lum_bgg,
        "Lum_group": lum_group,
        "FracLumBGG": lum_bgg / lum_group,
        "z_group": z_group,
        "DeltaR12": float(x.iloc[1]["M_r"] - x.iloc[0]["M_r"]),
        "NbGal": len(x),
        "RA_BGG": float(x.iloc[0]["RA"]),
        "Dec_BGG": float(x.iloc[0]["Dec"]),
        "RA_Bary": ra_bary,
        "Dec_Bary": dec_bary,
        "Radius_Bary_arcmin": radius_bary_arcmin,
        "Offset_Bary": offset_bary,
        "V_BGG": v_bgg,
        "V_moy": v_moy,
        "Vdisp": vdisp,
        "Voffset": voffset,
        "size_Group_Bary_kpc": size_kpc,
        "M_group": m_group,
        "M_virial": m_virial,
        "M_virial_over_L": m_virial / lum_group,
        "t_cr": t_cr,
        "BGG_SFRcategory": category[0],
        "all_SFR": bool(~(category == "X").any()),
        "Prop_M_Sat": float((category[1:] == "M").mean()),
        "Prop_M_Tot": float((category == "M").mean()),
        "Prop_G_Sat": float((category[1:] == "G").mean()),
        "Prop_G_Tot": float((category == "G").mean()),
        "Prop_Q_Sat": float((category[1:] == "Q").mean()),
        "Prop_Q_Tot": float((category == "Q").mean()),
    }
    lmass = lmass_200(lm180, z_group) if lm180 is not None else np.nan
    props["lMass_200"] = lmass
    props["r_200_kpc"] = r_200_kpc(lmass, z_group) if np.isfinite(lmass) else np.nan
    return pd.Series(props)


def build_group_table(gals: pd.DataFrame,
                      lm180_by_group: pd.Series | None = None) -> pd.DataFrame:
    """Group table (Paper I schema) for a galaxy sample of quartets."""

    rows = []
    for group_id, members in gals.groupby("Group"):
        lm180 = None
        if lm180_by_group is not None and group_id in lm180_by_group.index:
            lm180 = float(lm180_by_group.loc[group_id])
        props = quartet_group_properties(members, lm180=lm180)
        props["Group"] = group_id
        rows.append(props)
    table = pd.DataFrame(rows)
    # Misfit flags are median splits *within the sample* (common.BuidSelector).
    table["Misfit_Bary"] = np.where(
        table["Offset_Bary"] > table["Offset_Bary"].median(), "BMisfit", "BCentered"
    )
    table["Vmisfit"] = np.where(
        table["Voffset"] > table["Voffset"].median(), "VMisfit", "VCentered"
    )
    return table[GROUP_COLUMNS]


def build_control4c_gals(pc_gals: pd.DataFrame,
                         cg4_gals_full: pd.DataFrame) -> pd.DataFrame:
    """Control4C member table: BGG + 3 closest projected companions.

    Quartets are the ``rank_dist`` <= 4 members of each PC group (the BGG has
    rank_dist = 1). Every group containing at least one galaxy of the *full*
    CG4 sample (split groups included) is excluded, as in Paper I. The Lim
    group 3688 is intentionally kept: the pipeline removes it explicitly at
    load time (src/main.py::clean) and the manuscript documents why.
    """

    quartets = pc_gals[pc_gals["rank_dist"] <= 4].copy()
    contaminated = set(
        quartets.loc[quartets["objid"].isin(set(cg4_gals_full["objid"])), "Group"]
    )
    quartets = quartets[~quartets["Group"].isin(contaminated)].copy()

    quartets = quartets.rename(columns={"lgm_tot_p50": "lgm", "sfr_tot_p50": "sfr"})
    quartets["rank_M_parent"] = quartets["rank_M"]
    quartets["rank_M"] = (
        quartets.groupby("Group")["M_r"].rank(method="first").astype(int)
    )
    quartets = quartets.sort_values(["Group", "rank_M"]).reset_index(drop=True)
    return quartets[GALS_COLUMNS]


def regenerate_control4c(write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild Control4C_Gals/Groups from the committed parent catalogue."""

    pc_gals = pd.read_csv(os.path.join(co.DATA_PATH, "PC_Gals.csv"))
    cg4_full = pd.read_csv(os.path.join(co.DATA_PATH, "CG4_Gals.csv"))
    gals = build_control4c_gals(pc_gals, cg4_full)
    lm180 = (
        pc_gals.drop_duplicates("Group").set_index("Group")["logM_180"]
    )
    groups = build_group_table(gals, lm180_by_group=lm180)
    if write:
        gals.to_csv(os.path.join(co.DATA_PATH, "Control4C_Gals.csv"), index=False)
        groups.to_csv(os.path.join(co.DATA_PATH, "Control4C_Groups.csv"), index=False)
        print(f"Control4C regenerated: {groups.shape[0]} groups, {len(gals)} galaxies")
    return gals, groups


if __name__ == "__main__":
    regenerate_control4c()
