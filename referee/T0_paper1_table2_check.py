"""T0 addendum — which Control4C construction fed Paper I's published numbers?

Rebuilds both Control4C variants from the committed parent catalogue
(byte-identical to the public GitHub copy, MD5 ae4b89eb...):

  A) unrestricted: 3 nearest projected members regardless of magnitude
     (= the shipped/distributed Control4C_Gals.csv, 705 groups), and
  B) restricted:   Delta_m <= 3 filter first, then 3 nearest
     (704 groups after re-applying the CG4 exclusion),

then recomputes the published Paper I statistics with the paper's stated
methodology (Eq. 1 Rij = theta*D_A, Eq. 2 velocities, gapper sigma_v,
M_sun_r = 4.68):

  Table 2 medians (Control4C):   <Rij> 313, sigma_v 153, logL 11.01,
                                 DMr12 1.17, LBGG/L 0.61
  Table 3 (Control4C, N = 704):  T1 = 0.29, T2 = 0.65
  Sample counts:                 61 exclusions -> 704 groups

Result (2026-07-21 run): variant B reproduces every published value
(sigma_v 153.0, logL 11.011, DMr12 1.168, LBGG/L 0.612, T1 0.29, T2 0.65,
61 -> 704), with <Rij> = 293 kpc under D_A and 313 kpc under D_L --
i.e. the published sizes used the luminosity distance despite Eq. (1).
Variant A (the distributed CSV) matches none of them
(N 705, sigma_v 156.1, logL 10.967, DMr12 1.465, LBGG/L 0.694,
<Rij> 212 under D_A). Paper I's published analysis therefore used the
restricted construction; the distributed CSV implements the paper's
*literal prose* ("the three closest galaxies to the BGG (in projection)",
Sect. 2.4) which omits the Delta_m <= 3 restriction.
"""

from __future__ import annotations

import itertools as it
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COSMO = FlatLambdaCDM(H0=67.8, Om0=0.308, Tcmb0=2.7255, Neff=3.15)
C_KMS = 299792.458
MR_SUN = 4.68
DMAG_TOL = 1e-9


def sep_rad(ra0, dec0, ra, dec):
    c = (np.sin(np.deg2rad(dec0)) * np.sin(np.deg2rad(dec))
         + np.cos(np.deg2rad(dec0)) * np.cos(np.deg2rad(dec))
         * np.cos(np.deg2rad(ra - ra0)))
    return np.arccos(np.clip(c, -1.0, 1.0))


def gapper(v):
    v = np.sort(np.asarray(v, dtype=float))
    n = len(v)
    w = np.arange(1, n) * np.arange(n - 1, 0, -1)
    return np.sqrt(np.pi) / (n * (n - 1)) * np.sum(w * np.diff(v))


def quartet_props(members: pd.DataFrame) -> dict:
    x = members.sort_values("M_r")
    z_group = x["z"].mean()
    sig = gapper(C_KMS * (x["z"] - z_group) / (1 + z_group))
    lum = 10 ** (-0.4 * (x["M_r"] - MR_SUN))
    da_kpc = COSMO.angular_diameter_distance(z_group).to(u.kpc).value
    dl_kpc = COSMO.luminosity_distance(z_group).to(u.kpc).value
    seps = [sep_rad(x.iloc[i]["RA"], x.iloc[i]["Dec"],
                    x.iloc[j]["RA"], x.iloc[j]["Dec"])
            for i, j in it.combinations(range(len(x)), 2)]
    return {"Rij_DA": np.median(seps) * da_kpc,
            "Rij_DL": np.median(seps) * dl_kpc,
            "sigma_v": sig,
            "logL": np.log10(lum.sum()),
            "DMr12": x.iloc[1]["M_r"] - x.iloc[0]["M_r"],
            "LBGG_frac": lum.iloc[0] / lum.sum(),
            "M1": x.iloc[0]["M_r"],
            "span": x["M_r"].max() - x["M_r"].min()}


def report(name: str, quartets: list[pd.DataFrame]) -> None:
    t = pd.DataFrame([quartet_props(q) for q in quartets])
    med = t.median(numeric_only=True)
    t1 = t["M1"].std(ddof=1) / t["DMr12"].mean()
    t2 = t["DMr12"].std(ddof=1) / t["DMr12"].mean() / np.sqrt(0.677)
    print(f"{name}: N={len(t)}  <Rij>_DA={med.Rij_DA:.0f}  "
          f"<Rij>_DL={med.Rij_DL:.0f}  sigma_v={med.sigma_v:.1f}  "
          f"logL={med.logL:.3f}  DMr12={med.DMr12:.3f}  "
          f"LBGG/L={med.LBGG_frac:.3f}  T1={t1:.2f}  T2={t2:.2f}  "
          f"span>3: {(t['span'] > 3).sum()}")


def main() -> None:
    pc = pd.read_csv(DATA / "PC_Gals.csv")
    c4c = pd.read_csv(DATA / "Control4C_Gals.csv")
    cg4_ids = set(pd.read_csv(DATA / "CG4_Gals.csv")["objid"])

    shipped = [g for _, g in c4c.groupby("Group")]

    restricted, n_excluded = [], 0
    for _, parent in pc.groupby("Group"):
        bgg = parent[parent["rank_M"] == 1].iloc[0]
        m = parent[parent["objid"] != bgg["objid"]].copy()
        m["dmag"] = m["M_r"] - bgg["M_r"]
        el = m[m["dmag"] <= 3 + DMAG_TOL].copy()
        el["sep"] = sep_rad(bgg["RA"], bgg["Dec"], el["RA"], el["Dec"])
        quartet = pd.concat([parent[parent["objid"] == bgg["objid"]],
                             el.nsmallest(3, "sep")])
        if set(quartet["objid"]) & cg4_ids:
            n_excluded += 1
            continue
        restricted.append(quartet)

    print("Published (Paper I): N=704 (61 excl)  <Rij>=313  sigma_v=153  "
          "logL=11.010  DMr12=1.170  LBGG/L=0.610  T1=0.29  T2=0.65")
    report("A shipped/unrestricted CSV", shipped)
    report(f"B restricted ({n_excluded} excl)", restricted)


if __name__ == "__main__":
    main()
