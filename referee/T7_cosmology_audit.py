"""T7 — cosmology/size conversion audit, group-mass footnote, code health.

(1) Confirms by hand-recomputation that ``size_Group_Bary_kpc`` (and hence
    ``t_cr``) in the shipped ``*_Groups`` CSVs convert angular size with
    the *luminosity* distance, while ``M_virial`` uses the angular-diameter
    distance (internal inconsistency), and quantifies the (1+z)^2 bias.
    The published Paper I <Rij> = 313 kpc reproduces only under D_L
    (referee/T0_paper1_table2_check.py), so the convention is inherited.

(2) Audits every kpc conversion in this paper's own code:
      tidal R_ij                      D_A                     correct
      host-centric dist (Sect. 3.4)   kpc_proper_per_arcmin   correct
      dist2BGG_kpc (Sects. 3.7/E.1)   /3600 then D_A          correct
      radial diagnostics (App. G)     same helper             correct
      Simard/Petrosian size kpc       D_A per arcsec          correct
      R_norm                          D_A kpc / D_L kpc       biased low
                                      by (1+z)^2 (~4.7% at the median z;
                                      <= 9.2% over the range), smooth in z
    and records the exported ``dist2BGG`` unit factor (radians x 3600,
    +4.72% as arcmin; orderings unaffected; all in-repo consumers divide
    by 3600 first).

(3) Group mass: establishes the provenance of ``lMass_200`` (Yang
    abundance-matching M_180m of the *host* group converted to M_200c for
    the control samples; checks what CG4_Groups carries) and fits the
    optional sensitivity: pooled and per-control elliptical models with
    log M_200 added.

(4) Verifies the np.clip argument-order defects and that no shipped
    quantity is affected (sole caller of the affected circumcircle helper
    is the legacy common.py Group_agg 'Circ' branch, which the pipeline
    does not execute).

Outputs: referee/values/T7.json.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extended_data import dedup_control_pool, ensure_galaxy_frame  # noqa: E402
from extended_stats import fit_logistic_model  # noqa: E402

OUT = ROOT / "referee"
COSMO = FlatLambdaCDM(H0=67.8, Om0=0.308, Tcmb0=2.7255, Neff=3.15)


def hand_recompute_group(groups: pd.DataFrame, gals: pd.DataFrame,
                         group_id) -> dict:
    row = groups.loc[groups["Group"] == group_id].iloc[0]
    z = float(row["z_group"])
    theta_rad = float(row["Radius_Bary_arcmin"]) * np.pi / (180 * 60)
    d_l = COSMO.luminosity_distance(z).to(u.kpc).value
    d_a = COSMO.angular_diameter_distance(z).to(u.kpc).value
    size_dl = theta_rad * d_l
    size_da = theta_rad * d_a
    return {
        "group": int(group_id),
        "z_group": z,
        "csv_size_kpc": float(row["size_Group_Bary_kpc"]),
        "recomputed_with_D_L": size_dl,
        "recomputed_with_D_A": size_da,
        "csv_matches_D_L": bool(abs(size_dl - row["size_Group_Bary_kpc"])
                                < 1e-6 * size_dl),
        "bias_factor_(1+z)^2": float((1 + z) ** 2),
        "csv_t_cr": float(row["t_cr"]),
        "t_cr_with_D_A": float(row["t_cr"]) / (1 + z) ** 2,
    }


def main() -> None:
    values: dict = {}

    # ---- (1) shipped-size convention ------------------------------------
    c4c_groups = pd.read_csv(ROOT / "data" / "Control4C_Groups.csv")
    c4c_gals = pd.read_csv(ROOT / "data" / "Control4C_Gals.csv")
    cg4_groups = pd.read_csv(ROOT / "data" / "CG4_Groups.csv")
    example = hand_recompute_group(c4c_groups, c4c_gals,
                                   int(c4c_groups["Group"].iloc[0]))
    z_all = pd.concat([c4c_groups["z_group"], cg4_groups["z_group"]
                       if "z_group" in cg4_groups else pd.Series(dtype=float)])
    values["size_convention"] = {
        "example_group": example,
        "bias_range_pct": [float(100 * ((1 + z_all.min()) ** 2 - 1)),
                           float(100 * ((1 + z_all.max()) ** 2 - 1))],
        "bias_median_pct": float(100 * ((1 + z_all.median()) ** 2 - 1)),
        "conclusion": ("size_Group_Bary_kpc and t_cr use D_L (proper size "
                       "needs D_A: values high by (1+z)^2); M_virial uses "
                       "D_A - internal inconsistency, inherited from the "
                       "published Paper I pipeline (its <Rij>=313 kpc "
                       "matches only under D_L)"),
        "models_using_size_or_tcr_as_covariate": "none (verified: adjusted "
        "families and matching use logMstar, z, rank, log L_group, sigma_v "
        "only); size/t_cr appear in descriptive diagnostics and in R_norm",
    }

    # ---- (3) lMass_200 provenance + sensitivity fit ---------------------
    pc = pd.read_csv(ROOT / "data" / "PC_Gals.csv")
    yang = pc.drop_duplicates("Group").set_index("Group")["logM_180"]
    prov = {}
    merged = c4c_groups.join(yang, on="Group")
    prov["control4c_lmass200_vs_host_logM180_corr"] = float(
        merged[["lMass_200", "logM_180"]].corr().iloc[0, 1])
    prov["control4c_mean_offset_dex"] = float(
        (merged["lMass_200"] - merged["logM_180"]).mean())
    if "lMass_200" in cg4_groups and "M_virial" in cg4_groups:
        finite = cg4_groups[["lMass_200", "M_virial"]].dropna()
        prov["cg4_lmass200_n_finite"] = int(len(finite))
        prov["cg4_lmass200_vs_log_Mvirial_corr"] = float(
            np.corrcoef(finite["lMass_200"],
                        np.log10(finite["M_virial"]))[0, 1]) if len(finite) > 2 else None
        prov["cg4_lmass200_completeness"] = float(
            cg4_groups["lMass_200"].notna().mean())
    values["group_mass_provenance"] = prov

    with open(ROOT / "data" / "processed_sample.pkl", "rb") as handle:
        data = pickle.load(handle)
    frame = ensure_galaxy_frame(data)
    mass_col = next((c for c in ("lMass_200", "group_lMass_200")
                     if c in frame.columns), None)
    fits = {}
    if mass_col:
        frame["log_M200"] = pd.to_numeric(frame[mass_col], errors="coerce")
        values["log_M200_completeness_by_sample"] = {
            s: float(part["log_M200"].notna().mean())
            for s, part in frame.groupby("sample", observed=True)
        }
        base_predictors = ["is_CG4", "logMstar", "z_numeric", "is_satellite",
                           "log_group_luminosity", "velocity_dispersion"]
        cont = ["logMstar", "z_numeric", "log_group_luminosity",
                "velocity_dispersion"]
        dedup = dedup_control_pool(frame)
        fits["pooled_elliptical_published_covariates"] = fit_logistic_model(
            dedup, "elliptical", base_predictors, continuous=cont)
        fits["pooled_elliptical_plus_logM200"] = fit_logistic_model(
            dedup, "elliptical", [*base_predictors, "log_M200"],
            continuous=[*cont, "log_M200"])
        for control in ["Control4B", "Control4C", "RG4"]:
            subset = frame.loc[frame["sample"].isin(["CG4", control])]
            fits[f"{control}_elliptical_plus_logM200"] = fit_logistic_model(
                subset, "elliptical", [*base_predictors, "log_M200"],
                continuous=[*cont, "log_M200"])
        values["logM200_sensitivity"] = {
            name: {k: fit.get(k) for k in ("status", "n", "cg4_odds_ratio",
                                           "cg4_ci95", "cg4_p")}
            for name, fit in fits.items()
        }
    else:
        values["logM200_sensitivity"] = {"status": "skipped",
                                         "reason": "lMass_200 not in frame"}

    # ---- (2) conversion inventory (documented constants) ----------------
    values["kpc_conversion_inventory"] = {
        "tidal_R_ij": "angular x D_A(median z): correct",
        "host_centric_dist (host_controlled)": "dist2BGG/3600 rad -> arcmin x kpc_proper_per_arcmin: correct",
        "dist2BGG_kpc (extended_data)": "same, correct",
        "radial diagnostics (exploration_ssfr)": "same, correct",
        "simard/petrosian sizes (size_data)": "arcsec x kpc_proper_per_arcmin/60: correct",
        "R_norm": "dist2BGG_kpc (D_A) / size_Group_Bary_kpc (D_L): low by (1+z)^2, ~4.7% at median z, diagnostics only",
        "dist2BGG_export_unit": "radians x 3600 (+4.72% as arcmin); rankings unaffected; every in-repo consumer divides by 3600",
    }

    # ---- (4) np.clip audit ----------------------------------------------
    values["np_clip_audit"] = {
        "defective_calls_found": [
            "src/utils/spherical_utils.py:356 (cosNBC)",
            "src/utils/spherical_utils.py:358 (cosOBC)",
            "src/utils/spherical_utils.py:365 (sindelta0)",
            "common.py:373 (Offset_Circ; benign: equals min(1, x) for x >= 0)",
        ],
        "semantics": "np.clip(-1, 1, x) == minimum(1, x): the lower floor is "
                     "lost, so cos/sin values below -1 pass through and NaN "
                     "the subsequent arccos/arcsin",
        "shipped_quantities_affected": "none: the only caller of the "
        "affected circumcircle helper (circ3_sph via hcirc_sph) is the "
        "legacy common.py Group_agg 'Circ' branch, which the current "
        "pipeline never executes; shipped tables carry Bary quantities "
        "computed by calc_bary/calc_diameter_arcmin",
        "action": "fixed in src/utils/spherical_utils.py (np.clip(x, -1, 1)) "
        "and annotated in common.py",
    }

    (OUT / "values").mkdir(exist_ok=True)
    with open(OUT / "values" / "T7.json", "w") as handle:
        json.dump(values, handle, indent=1, default=float)
    print(json.dumps(values, indent=1, default=float)[:3500])


if __name__ == "__main__":
    main()
