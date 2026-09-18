"""Follow-up of the direct stellar-metallicity study (audit + physics).

Scope
-----
* Reads the historical products of ``zstar_analysis.py`` (never rewrites them)
  and the project sample tables (read-only) to attach group-level properties.
* Writes ONLY below ``outputs/zstar_followup``, ``figures/zstar_followup`` and
  ``ZSTAR_FOLLOWUP_REPORT.md`` (the report itself is written by hand from the
  saved tables; this script writes ``key_numbers.json`` for that purpose).

Sections (each writes one or more CSVs; the names are listed in ``main``):
  A  audit of the historical analysis (reproduction, N accounting, censoring,
     sample nesting, bootstrap-vs-CR1 widths)
  B  Gallazzi vs FIREFLY on exactly the same galaxies
  C  mass-interaction model  [Z/H] = f(M) + beta CG4 + gamma CG4 (M - M0)
  D  control-only MZR fit -> residuals of CG4 (bootstrap over control fit)
  E  CG4 vs RG4 (and vs Control4C) decomposition by SF state / morphology,
     joint models with interactions, Kitagawa/Oaxaca composition split
  F  population composition at fixed mass (covariate balance) and
     covariate-adjusted contrasts
  G  environmental sequence: RG4, Control4B, Control4C (+ subsets by core
     compactness and parent richness) and within-control environmental slopes
  H  mass-binned contrasts
  I  audit of the 11 star-forming CG4 BGGs
  J  exploratory light-/mass-weighted age contrasts
  L  robustness: median (quantile) regression, S/N-restricted RG4 contrast

All bootstrap intervals are cluster (group) bootstraps, resampling groups
within each sample, exactly as in the historical analysis; the seed differs so
that the follow-up is an independent realisation of the same procedure.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import zstar_analysis as za  # noqa: E402  (historical implementation, read-only use)

OUT = HERE / "outputs" / "zstar_followup"
FIG = HERE / "figures" / "zstar_followup"
HIST = HERE / "outputs" / "zstar"
SEED = 20260917
N_BOOT = 2000
PRIMARY_Y, PRIMARY_U = za.PRODUCTS[za.PRIMARY]
PALETTE = dict(za.PALETTE)
PALETTE.update({"C4C compact": "#1B5E4B", "C4C loose": "#6FC2A8", "C4B\\RG4": "#2864A6", "C4C\\RG4": "#25876E"})

# --------------------------------------------------------------------------
# data assembly
# --------------------------------------------------------------------------


def load() -> pd.DataFrame:
    """Historical worktable + the extra columns this follow-up needs."""
    w = za.prepare_worktable()
    m = pd.read_csv(
        HERE / "work" / "metallicity_worktable.csv",
        low_memory=False,
        usecols=["sample", "objid", "sSFR", "p_E", "p_S", "is_dominated", "Hb_sub", "mpa_subclass", "n_spectra_for_objid", "spec_choice"],
    )
    m = m[["sample", "objid"] + [c for c in m.columns if c not in w.columns]]
    w = w.merge(m, on=["sample", "objid"], how="left", validate="1:1")
    for col in ["sSFR", "p_E", "p_S", "is_dominated", "Hb_sub", "mpa_subclass", "n_spectra_for_objid", "spec_choice"]:
        assert col in w.columns, col

    # FIREFLY ages: the historical prepare_worktable() already carries every
    # FIREFLY column; ages are stored in years (grid 0.03-15 Gyr) -> log10(age/yr).
    for lib in ["MILES", "ELODIE"]:
        for wt, short in [("lightW", "lw"), ("massW", "mw")]:
            base = f"Chabrier_{lib}_age_{wt}"
            v, lo, hi = w[base], w[base + "_low_1sig"], w[base + "_up_1sig"]
            ok = (v > 0) & (lo > 0) & (hi > 0) & (lo <= v) & (v <= hi)
            with np.errstate(invalid="ignore"):
                w[f"ff_{lib.lower()}_{short}_logage"] = np.log10(v.where(ok))
                w[f"ff_{lib.lower()}_{short}_logage_unc"] = ((np.log10(hi) - np.log10(lo)) / 2).where(ok)
            w[f"ff_{lib.lower()}_{short}_age_at_grid_max"] = ok & (v >= 1.49e10)
            w[f"ff_{lib.lower()}_{short}_age_gt_universe"] = ok & (v > 1.38e10)
    # Gallazzi r-band light-weighted age (already merged): log10(age/yr).
    gok = w["gallazzi_valid"] & (w["age_logyr_rband_p16"] > 0) & (w["age_logyr_rband_median"] > 0) & (w["age_logyr_rband_p84"] > 0)
    w["gal_logage"] = w["age_logyr_rband_median"].where(gok)
    w["gal_logage_unc"] = ((w["age_logyr_rband_p84"] - w["age_logyr_rband_p16"]) / 2).where(gok)

    # Group-level properties (read-only from the project sample tables).
    pkl = pickle.load(open(REPO / "data" / "processed_sample.pkl", "rb"))
    pieces = []
    for sample, prefix in [("CG4", "HMCG:"), ("RG4", "Lim:"), ("Control4B", "Lim:"), ("Control4C", "Lim:")]:
        G = pkl[f"{sample}_Groups"][["Group", "size_Group_Bary_kpc", "Radius_Bary_arcmin", "Vdisp", "t_cr", "lMass_200", "z_group"]].copy()
        G["group_uid"] = prefix + G["Group"].astype(int).astype(str)
        G["sample"] = sample
        pieces.append(G.drop(columns="Group").rename(columns={"Vdisp": "group_Vdisp", "size_Group_Bary_kpc": "core_size_kpc", "z_group": "group_z"}))
    # the historical worktable already carries group_z/group_Vdisp (same source); replace them
    w = w.drop(columns=[c for c in ["group_z", "group_Vdisp"] if c in w.columns])
    w = w.merge(pd.concat(pieces, ignore_index=True), on=["sample", "group_uid"], how="left", validate="m:1")
    pc = pd.read_csv(REPO / "data" / "PC_Groups.csv", usecols=["Group", "NbGal"])
    pc["group_uid"] = "Lim:" + pc["Group"].astype(int).astype(str)
    w = w.merge(pc[["group_uid", "NbGal"]].rename(columns={"NbGal": "parent_NbGal"}), on="group_uid", how="left", validate="m:1")
    w["parent_NbGal"] = w["parent_NbGal"].where(w["sample"].ne("CG4"))  # CG4 parents are not Lim groups
    with np.errstate(divide="ignore", invalid="ignore"):
        w["log_core_size"] = np.log10(w["core_size_kpc"].where(w["core_size_kpc"] > 0))
        w["log_group_Vdisp"] = np.log10(w["group_Vdisp"].where(w["group_Vdisp"] > 0))
        w["log_parent_N"] = np.log10(w["parent_NbGal"])
        w["logSN"] = np.log10(w["SN"].where(w["SN"] > 0))
    rg4_groups = set(w.loc[w["sample"].eq("RG4"), "group_uid"])
    w["group_is_RG4"] = w["group_uid"].isin(rg4_groups)
    w["is_CG4"] = w["sample"].eq("CG4").astype(float)
    assert len(w) == 6076 and w["objid"].nunique() == 3857
    return w


# --------------------------------------------------------------------------
# generic mass-adjusted contrast with cluster bootstrap
# --------------------------------------------------------------------------


def _design(d: pd.DataFrame, m0: float, degree: int, extras: list[str], interaction: bool, centre: dict | None = None) -> tuple[np.ndarray, list[str]]:
    mass = d["lgm"].to_numpy(float) - m0
    cols = {"mass_c": mass}
    if degree == 2:
        cols["mass_c2"] = mass ** 2
    cols["is_CG4"] = d["is_CG4"].to_numpy(float)
    if interaction:
        cols["CG4_x_mass"] = cols["is_CG4"] * mass
    for e in extras:
        v = d[e].to_numpy(float)
        c = centre[e] if centre and e in centre else np.nanmedian(v)
        cols[e] = v - c
    names = ["const"] + list(cols)
    return np.column_stack([np.ones(len(d))] + list(cols.values())), names


def select(w: pd.DataFrame, role: str, control: str | list[str], ycol: str = PRIMARY_Y, ucol: str = PRIMARY_U, subset: pd.Series | None = None, extras: list[str] | None = None) -> pd.DataFrame:
    controls = [control] if isinstance(control, str) else list(control)
    keep = w[ucol] & w["role"].eq(role) & w["sample"].isin(["CG4"] + controls) & w["lgm"].notna() & w[ycol].notna()
    if subset is not None:
        keep &= subset.reindex(w.index).fillna(False).astype(bool)
    d = w.loc[keep].copy()
    if extras:
        d = d.dropna(subset=extras)
    return d


def contrast(d: pd.DataFrame, ycol: str, rng: np.random.Generator, m0: float, degree: int = 1, extras: list[str] | None = None, interaction: bool = False, n_boot: int = N_BOOT, coef: str = "is_CG4", min_n: int = 8) -> dict | None:
    """OLS with a common mass slope; CR1 SE; group bootstrap within sample."""
    extras = extras or []
    n_cg, n_ct = int(d["is_CG4"].sum()), int((1 - d["is_CG4"]).sum())
    if min(n_cg, n_ct) < min_n:
        return None
    centre = {e: float(np.nanmedian(d[e])) for e in extras}
    X, names = _design(d, m0, degree, extras, interaction, centre)
    y = d[ycol].to_numpy(float)
    fit = za.cr1_ols(y, X, names, d["group_uid"].to_numpy())
    out = {"n_CG4": n_cg, "n_control": n_ct, "groups_CG4": int(d.loc[d["is_CG4"].eq(1), "group_uid"].nunique()), "groups_control": int(d.loc[d["is_CG4"].eq(0), "group_uid"].nunique()), "degree": degree, "m0": m0}
    want = [coef] + (["CG4_x_mass"] if interaction else []) + extras
    for c in want:
        j = names.index(c)
        out[f"{c}_beta"] = float(fit["beta"][j])
        out[f"{c}_se"] = float(fit["se"][j])
    # cluster bootstrap within sample
    clusters: dict[tuple, list[int]] = {}
    for i, key in enumerate(zip(d["sample"], d["group_uid"])):
        clusters.setdefault(key, []).append(i)
    by_sample: dict[str, list] = {}
    for key in clusters:
        by_sample.setdefault(key[0], []).append(key)
    boots = {c: [] for c in want}
    for _ in range(n_boot):
        rows: list[int] = []
        for keys in by_sample.values():
            for pick in rng.integers(0, len(keys), len(keys)):
                rows.extend(clusters[keys[pick]])
        b = d.iloc[rows]
        Xb, _ = _design(b, m0, degree, extras, interaction, centre)
        try:
            coefb, *_ = np.linalg.lstsq(Xb, b[ycol].to_numpy(float), rcond=None)
        except np.linalg.LinAlgError:
            continue
        for c in want:
            boots[c].append(float(coefb[names.index(c)]))
    for c in want:
        arr = np.asarray(boots[c])
        out[f"{c}_ci68_lo"], out[f"{c}_ci68_hi"] = float(np.percentile(arr, 16)), float(np.percentile(arr, 84))
        out[f"{c}_ci95_lo"], out[f"{c}_ci95_hi"] = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    out["n_boot"] = len(boots[coef])
    # aliases for the main coefficient so that tables read like the historical ones
    for k in ["beta", "se", "ci68_lo", "ci68_hi", "ci95_lo", "ci95_hi"]:
        out[k] = out[f"{coef}_{k}"]
    return out


def common_mass_range(d: pd.DataFrame, q: float = 0.05) -> tuple[float, float]:
    cg, ct = d.loc[d["is_CG4"].eq(1), "lgm"], d.loc[d["is_CG4"].eq(0), "lgm"]
    return float(max(cg.quantile(q), ct.quantile(q))), float(min(cg.quantile(1 - q), ct.quantile(1 - q)))


def fmt(row: pd.Series | dict, key: str = "") -> str:
    p = key + "_" if key else ""
    return f"{row[p+'beta']:+.3f} [{row[p+'ci95_lo']:+.3f}, {row[p+'ci95_hi']:+.3f}]"


# --------------------------------------------------------------------------
# A. audit
# --------------------------------------------------------------------------


def section_audit(w: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    saved = pd.read_csv(HIST / "zstar_contrasts.csv")
    rows = []
    for product in [za.PRIMARY, "Gallazzi optical light-weighted"]:
        ycol, ucol = za.PRODUCTS[product]
        for role in ["Satellite", "BGG"]:
            for control in ["Control4C", "Control4B", "RG4"]:
                sv = saved[(saved["product"].eq(product)) & saved["role"].eq(role) & saved["control"].eq(control)].iloc[0]
                d = select(w, role, control, ycol, ucol)
                r = contrast(d, ycol, rng, m0=10.5, degree=int(sv["degree"]))
                usable = w[w[ucol] & w["role"].eq(role) & w["sample"].isin(["CG4", control])]
                rows.append({
                    "product": product, "role": role, "control": control,
                    "saved_beta": sv["beta"], "reproduced_beta": r["beta"], "abs_diff": abs(sv["beta"] - r["beta"]),
                    "saved_cluster_se": sv["cluster_se"], "reproduced_cluster_se": r["se"],
                    "saved_ci95_halfwidth": (sv["ci95_hi"] - sv["ci95_lo"]) / 2, "new_seed_ci95_halfwidth": (r["ci95_hi"] - r["ci95_lo"]) / 2,
                    "ci95_halfwidth_over_1.96se": ((sv["ci95_hi"] - sv["ci95_lo"]) / 2) / (1.96 * sv["cluster_se"]),
                    "usable_CG4": int(usable["sample"].eq("CG4").sum()), "usable_control": int(usable["sample"].ne("CG4").sum()),
                    "fit_CG4": r["n_CG4"], "fit_control": r["n_control"],
                    "dropped_no_lgm_CG4": int((usable["sample"].eq("CG4") & usable["lgm"].isna()).sum()),
                    "dropped_no_lgm_control": int((usable["sample"].ne("CG4") & usable["lgm"].isna()).sum()),
                })
    repro = pd.DataFrame(rows)

    # grid-ceiling / quantisation census for FIREFLY products
    cens = []
    for product, (ycol, ucol) in za.PRODUCTS.items():
        u = w[w[ucol]]
        cap = np.log10(1.99) if product.startswith("FIREFLY") else np.log10(0.05 / 0.02) - 1e-3
        for (role, sample), part in u.groupby(["role", "sample"]):
            cens.append({
                "product": product, "role": role, "sample": sample, "n": len(part),
                "frac_at_grid_ceiling": float((part[ycol] >= cap).mean()),
                "n_distinct_values": int(part[ycol].round(4).nunique()),
                "frac_zero_width_68": float((part[ycol.replace("_logz", "_logz_unc")] == 0).mean()) if ycol.replace("_logz", "_logz_unc") in part else np.nan,
                "median_logz": float(part[ycol].median()),
            })
    censoring = pd.DataFrame(cens)

    notes = {
        "RG4_objids_in_Control4C": int(w.loc[w["sample"].eq("RG4"), "objid"].isin(w.loc[w["sample"].eq("Control4C"), "objid"]).sum()),
        "RG4_objids_in_Control4B": int(w.loc[w["sample"].eq("RG4"), "objid"].isin(w.loc[w["sample"].eq("Control4B"), "objid"]).sum()),
        "RG4_n_objids": int(w["sample"].eq("RG4").sum()),
        "Control4B_Control4C_shared_objids": int(w.loc[w["sample"].eq("Control4B"), "objid"].isin(w.loc[w["sample"].eq("Control4C"), "objid"]).sum()),
        "duplicate_objid_within_sample": int(sum(w[w["sample"].eq(s)]["objid"].duplicated().sum() for s in za.SAMPLES)),
        "CG4_objids_in_any_control": int(w.loc[w["sample"].eq("CG4"), "objid"].isin(w.loc[w["sample"].ne("CG4"), "objid"]).sum()),
        "Control4C_groups_with_parent_N4": int(w[w["sample"].eq("Control4C")].drop_duplicates("group_uid")["parent_NbGal"].eq(4).sum()),
        "Control4C_groups_total": int(w[w["sample"].eq("Control4C")]["group_uid"].nunique()),
        "firefly_metallicity_is_linear_ZZsun": True,
        "firefly_metallicity_grid_max_linear": float(w["Chabrier_MILES_metallicity_lightW"].max()) if "Chabrier_MILES_metallicity_lightW" in w else np.nan,
        "gallazzi_logz_solar_max": float(w["gallazzi_logz_solar"].max()),
        "gallazzi_zsun": za.ZSUN_GALLAZZI,
    }
    return repro, censoring, notes


# --------------------------------------------------------------------------
# B. same objects
# --------------------------------------------------------------------------


def section_same_objects(w: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    both = w["gallazzi_usable"] & w["ff_miles_lw_usable"]
    for role in ["Satellite", "BGG"]:
        for control in ["Control4C", "RG4", "Control4B"]:
            d = select(w, role, control, "ff_miles_lw_logz", "ff_miles_lw_usable", subset=both)
            d["g_minus_f"] = d["gallazzi_logz_solar"] - d["ff_miles_lw_logz"]
            rf = contrast(d, "ff_miles_lw_logz", rng, m0=10.5)
            rg = contrast(d, "gallazzi_logz_solar", rng, m0=10.5)
            rd = contrast(d, "g_minus_f", rng, m0=10.5)
            full = contrast(select(w, role, control), "ff_miles_lw_logz", rng, m0=10.5)
            notg = contrast(select(w, role, control, subset=~w["gallazzi_usable"]), "ff_miles_lw_logz", rng, m0=10.5)
            rows.append({
                "role": role, "control": control, "n_CG4": rf["n_CG4"], "n_control": rf["n_control"],
                "firefly_same_objects": fmt(rf), "gallazzi_same_objects": fmt(rg), "gallazzi_minus_firefly_same_objects": fmt(rd),
                "firefly_full_sample": fmt(full) + f" (N={full['n_CG4']}+{full['n_control']})",
                "firefly_not_in_gallazzi": (fmt(notg) + f" (N={notg['n_CG4']}+{notg['n_control']})") if notg else "n/a",
                **{f"ff_{k}": rf[k] for k in ["beta", "se", "ci95_lo", "ci95_hi"]},
                **{f"gal_{k}": rg[k] for k in ["beta", "se", "ci95_lo", "ci95_hi"]},
                **{f"diff_{k}": rd[k] for k in ["beta", "se", "ci95_lo", "ci95_hi"]},
                "ff_full_beta": full["beta"], "ff_notgal_beta": notg["beta"] if notg else np.nan,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# C. mass interaction
# --------------------------------------------------------------------------


def section_interaction(w: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for product in [za.PRIMARY, "Gallazzi optical light-weighted"]:
        ycol, ucol = za.PRODUCTS[product]
        for role, control, m0 in [("Satellite", "Control4C", 10.2), ("BGG", "Control4C", 11.1), ("Satellite", "RG4", 10.2), ("Satellite", "Control4B", 10.2), ("BGG", "RG4", 11.1)]:
            d = select(w, role, control, ycol, ucol)
            base = contrast(d, ycol, rng, m0=m0)
            inter = contrast(d, ycol, rng, m0=m0, interaction=True)
            lo, hi = common_mass_range(d)
            rows.append({
                "product": product, "role": role, "control": control, "m0": m0, "n_CG4": base["n_CG4"], "n_control": base["n_control"],
                "common_mass_5_95_lo": lo, "common_mass_5_95_hi": hi,
                "beta_no_interaction": base["beta"], "beta_no_interaction_ci95_lo": base["ci95_lo"], "beta_no_interaction_ci95_hi": base["ci95_hi"],
                "beta_at_m0": inter["beta"], "beta_at_m0_ci95_lo": inter["ci95_lo"], "beta_at_m0_ci95_hi": inter["ci95_hi"],
                "gamma": inter["CG4_x_mass_beta"], "gamma_se": inter["CG4_x_mass_se"], "gamma_ci95_lo": inter["CG4_x_mass_ci95_lo"], "gamma_ci95_hi": inter["CG4_x_mass_ci95_hi"],
                "implied_effect_at_range_lo": inter["beta"] + inter["CG4_x_mass_beta"] * (lo - m0),
                "implied_effect_at_range_hi": inter["beta"] + inter["CG4_x_mass_beta"] * (hi - m0),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# D. control-only fit -> residuals
# --------------------------------------------------------------------------


def _mzr_fit(control: pd.DataFrame, ycol: str, degree: int) -> np.ndarray:
    m = control["lgm"].to_numpy(float) - 10.5
    X = np.column_stack([np.ones(len(control)), m] + ([m ** 2] if degree == 2 else []))
    b, *_ = np.linalg.lstsq(X, control[ycol].to_numpy(float), rcond=None)
    return b


def _predict(b: np.ndarray, lgm: np.ndarray) -> np.ndarray:
    m = lgm - 10.5
    X = np.column_stack([np.ones(len(m)), m] + ([m ** 2] if len(b) == 3 else []))
    return X @ b


def section_control_residuals(w: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, objects = [], []
    for product in [za.PRIMARY, "Gallazzi optical light-weighted"]:
        ycol, ucol = za.PRODUCTS[product]
        for role in ["Satellite", "BGG"]:
            for control in ["Control4C", "RG4"]:
                d = select(w, role, control, ycol, ucol)
                ctrl, cg = d[d["is_CG4"].eq(0)], d[d["is_CG4"].eq(1)]
                for degree in [1, 2]:
                    b = _mzr_fit(ctrl, ycol, degree)
                    d[f"resid_deg{degree}"] = d[ycol] - _predict(b, d["lgm"].to_numpy(float))
                    # bootstrap: resample control groups (refit) and CG4 groups (re-average)
                    cg_groups = cg.groupby("group_uid").indices
                    ct_groups = ctrl.groupby("group_uid").indices
                    cg_keys, ct_keys = list(cg_groups), list(ct_groups)
                    means, medians = [], []
                    for _ in range(N_BOOT):
                        ct_rows = np.concatenate([ct_groups[ct_keys[k]] for k in rng.integers(0, len(ct_keys), len(ct_keys))])
                        cg_rows = np.concatenate([cg_groups[cg_keys[k]] for k in rng.integers(0, len(cg_keys), len(cg_keys))])
                        bb = _mzr_fit(ctrl.iloc[ct_rows], ycol, degree)
                        r = cg.iloc[cg_rows][ycol].to_numpy(float) - _predict(bb, cg.iloc[cg_rows]["lgm"].to_numpy(float))
                        means.append(r.mean())
                        medians.append(np.median(r))
                    rc, rg = d.loc[d["is_CG4"].eq(0), f"resid_deg{degree}"], d.loc[d["is_CG4"].eq(1), f"resid_deg{degree}"]
                    rows.append({
                        "product": product, "role": role, "control": control, "degree": degree,
                        "n_CG4": len(cg), "n_control": len(ctrl), "groups_CG4": cg["group_uid"].nunique(), "groups_control": ctrl["group_uid"].nunique(),
                        "control_slope": b[1], "control_curvature": b[2] if degree == 2 else np.nan,
                        "mean_dZ_CG4": float(rg.mean()), "mean_dZ_CG4_ci95_lo": float(np.percentile(means, 2.5)), "mean_dZ_CG4_ci95_hi": float(np.percentile(means, 97.5)),
                        "median_dZ_CG4": float(rg.median()), "median_dZ_CG4_ci95_lo": float(np.percentile(medians, 2.5)), "median_dZ_CG4_ci95_hi": float(np.percentile(medians, 97.5)),
                        "median_dZ_control": float(rc.median()), "sd_dZ_control": float(rc.std(ddof=1)), "sd_dZ_CG4": float(rg.std(ddof=1)),
                        "iqr_dZ_control": float(rc.quantile(.75) - rc.quantile(.25)), "iqr_dZ_CG4": float(rg.quantile(.75) - rg.quantile(.25)),
                        "frac_CG4_below_control_median": float((rg < rc.median()).mean()),
                    })
                if product == za.PRIMARY:
                    objects.append(d[["objid", "sample", "group_uid", "role", "lgm", ycol, "resid_deg1", "resid_deg2", "sf_state", "morphology4"]].assign(control=control))
    return pd.DataFrame(rows), pd.concat(objects, ignore_index=True)


# --------------------------------------------------------------------------
# E. RG4 decomposition
# --------------------------------------------------------------------------

SUBSETS = {
    "all": lambda w: pd.Series(True, index=w.index),
    "SF=Quenched": lambda w: w["sf_state"].eq("Quenched"),
    "SF=Starforming": lambda w: w["sf_state"].eq("Starforming"),
    "morph=Elliptical": lambda w: w["morphology4"].eq("Elliptical"),
    "morph=Spiral": lambda w: w["morphology4"].eq("Spiral"),
    "morph=Uncertain": lambda w: w["morphology4"].eq("Uncertain"),
    "Quenched&Elliptical": lambda w: w["sf_state"].eq("Quenched") & w["morphology4"].eq("Elliptical"),
    "Quenched&Spiral": lambda w: w["sf_state"].eq("Quenched") & w["morphology4"].eq("Spiral"),
    "Starforming&Elliptical": lambda w: w["sf_state"].eq("Starforming") & w["morphology4"].eq("Elliptical"),
    "Starforming&Spiral": lambda w: w["sf_state"].eq("Starforming") & w["morphology4"].eq("Spiral"),
}


def section_rg4_decomposition(w: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub_rows, model_rows, oaxaca_rows = [], [], []
    for product, (ycol, ucol) in za.PRODUCTS.items():
        for role in ["Satellite", "BGG"]:
            for control in ["RG4", "Control4C"]:
                for label, fn in SUBSETS.items():
                    if role == "BGG" and "&" in label:
                        continue
                    d = select(w, role, control, ycol, ucol, subset=fn(w))
                    r = contrast(d, ycol, rng, m0=10.2 if role == "Satellite" else 11.1, min_n=6)
                    if r is None:
                        sub_rows.append({"product": product, "role": role, "control": control, "subset": label, "n_CG4": int(d["is_CG4"].sum()), "n_control": int((1 - d["is_CG4"]).sum()), "beta": np.nan})
                        continue
                    sub_rows.append({"product": product, "role": role, "control": control, "subset": label, **r})
                # joint models with a categorical and its interaction
                for cat, level, name in [("sf_state", "Starforming", "SF"), ("morphology4", "Spiral", "Spiral")]:
                    levels = ["Quenched", "Starforming"] if cat == "sf_state" else ["Elliptical", "Spiral"]
                    d = select(w, role, control, ycol, ucol, subset=w[cat].isin(levels))
                    d[f"x_{name}"] = d[cat].eq(level).astype(float)
                    d[f"CG4_x_{name}"] = d["is_CG4"] * d[f"x_{name}"]
                    m0 = 10.2 if role == "Satellite" else 11.1
                    r1 = contrast(d, ycol, rng, m0=m0, extras=[f"x_{name}"])
                    r2 = contrast(d, ycol, rng, m0=m0, extras=[f"x_{name}", f"CG4_x_{name}"])
                    if r1 is None or r2 is None:
                        continue
                    model_rows.append({
                        "product": product, "role": role, "control": control, "covariate": name, "n_CG4": r1["n_CG4"], "n_control": r1["n_control"],
                        "additive_CG4": fmt(r1), "additive_cov": fmt(r1, f"x_{name}"),
                        "interaction_CG4_at_reference": fmt(r2), "interaction_cov": fmt(r2, f"x_{name}"), "interaction_CG4_x_cov": fmt(r2, f"CG4_x_{name}"),
                        "add_CG4_beta": r1["beta"], "add_CG4_ci95_lo": r1["ci95_lo"], "add_CG4_ci95_hi": r1["ci95_hi"],
                        "int_CG4_ref_beta": r2["beta"], "int_CG4_ref_ci95_lo": r2["ci95_lo"], "int_CG4_ref_ci95_hi": r2["ci95_hi"],
                        "int_CG4xcov_beta": r2[f"CG4_x_{name}_beta"], "int_CG4xcov_se": r2[f"CG4_x_{name}_se"], "int_CG4xcov_ci95_lo": r2[f"CG4_x_{name}_ci95_lo"], "int_CG4xcov_ci95_hi": r2[f"CG4_x_{name}_ci95_hi"],
                    })
        # Kitagawa/Oaxaca split of the satellite total (primary product only)
        if product != za.PRIMARY:
            continue
        for control in ["RG4", "Control4C"]:
            for cat, levels in [("sf_state", ["Quenched", "Starforming"]), ("morphology4", ["Elliptical", "Spiral", "Uncertain", "NoGZ"])]:
                d = select(w, "Satellite", control, ycol, ucol)
                if cat == "sf_state":
                    d = d[d[cat].isin(levels)]
                ctrl = d[d["is_CG4"].eq(0)]
                b = _mzr_fit(ctrl, ycol, 1)
                d["r"] = d[ycol] - _predict(b, d["lgm"].to_numpy(float))

                def split(dd: pd.DataFrame) -> dict:
                    cgp, ctp = dd[dd["is_CG4"].eq(1)], dd[dd["is_CG4"].eq(0)]
                    total = cgp["r"].mean() - ctp["r"].mean()
                    within = composition = 0.0
                    for lev in levels:
                        wc, wt = (cgp[cat].eq(lev)).mean(), (ctp[cat].eq(lev)).mean()
                        rc = cgp.loc[cgp[cat].eq(lev), "r"].mean() if wc > 0 else 0.0
                        rt = ctp.loc[ctp[cat].eq(lev), "r"].mean() if wt > 0 else 0.0
                        within += wc * (rc - rt) if wc > 0 and wt > 0 else 0.0
                        composition += (wc - wt) * (rt - ctp["r"].mean()) if wt > 0 else 0.0
                    return {"total": total, "within": within, "composition": composition}

                point = split(d)
                boots = {k: [] for k in point}
                groups = d.groupby(["sample", "group_uid"]).indices
                keys_by_sample: dict[str, list] = {}
                for key in groups:
                    keys_by_sample.setdefault(key[0], []).append(key)
                for _ in range(N_BOOT):
                    rows_i: list[int] = []
                    for keys in keys_by_sample.values():
                        for k in rng.integers(0, len(keys), len(keys)):
                            rows_i.extend(groups[keys[k]])
                    bs = split(d.iloc[rows_i])
                    for k in point:
                        boots[k].append(bs[k])
                row = {"product": product, "role": "Satellite", "control": control, "stratifier": cat, "n_CG4": int(d["is_CG4"].sum()), "n_control": int((1 - d["is_CG4"]).sum())}
                for k in point:
                    row[k] = point[k]
                    row[f"{k}_ci95_lo"], row[f"{k}_ci95_hi"] = np.percentile(boots[k], 2.5), np.percentile(boots[k], 97.5)
                for lev in levels:
                    row[f"frac_{lev}_CG4"] = float(d.loc[d["is_CG4"].eq(1), cat].eq(lev).mean())
                    row[f"frac_{lev}_control"] = float(d.loc[d["is_CG4"].eq(0), cat].eq(lev).mean())
                oaxaca_rows.append(row)
    return pd.DataFrame(sub_rows), pd.DataFrame(model_rows), pd.DataFrame(oaxaca_rows)


# --------------------------------------------------------------------------
# F. composition / covariate balance at fixed mass
# --------------------------------------------------------------------------

COVARIATES = ["sSFR", "D4000n", "HdA_sub", "log_sigma", "ff_miles_lw_logage", "ff_miles_mw_logage", "fibre_light_fraction", "fibre_radius_over_R50", "R50_kpc", "z", "logSN", "ff_miles_lw_logz_unc", "p_E"]
FRACTIONS = {"frac_Quenched": ("sf_state", "Quenched"), "frac_Starforming": ("sf_state", "Starforming"), "frac_Elliptical": ("morphology4", "Elliptical"), "frac_Spiral": ("morphology4", "Spiral"), "frac_Uncertain": ("morphology4", "Uncertain"), "frac_NoGZ": ("morphology4", "NoGZ")}


def section_composition(w: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    balance, adjusted = [], []
    for role in ["Satellite", "BGG"]:
        m0 = 10.2 if role == "Satellite" else 11.1
        for control in ["RG4", "Control4C", "Control4B"]:
            d = select(w, role, control)
            for name, (col, level) in FRACTIONS.items():
                d[name] = d[col].eq(level).astype(float)
            for cov in COVARIATES + list(FRACTIONS):
                dd = d.dropna(subset=[cov])
                if dd[cov].std() == 0:
                    continue
                r = contrast(dd, cov, rng, m0=m0, n_boot=400)
                balance.append({
                    "role": role, "control": control, "covariate": cov, "n_CG4": r["n_CG4"], "n_control": r["n_control"],
                    "median_CG4": float(dd.loc[dd["is_CG4"].eq(1), cov].median()), "median_control": float(dd.loc[dd["is_CG4"].eq(0), cov].median()),
                    "mass_adjusted_diff": r["beta"], "diff_se": r["se"], "diff_ci95_lo": r["ci95_lo"], "diff_ci95_hi": r["ci95_hi"],
                    "smd": r["beta"] / float(dd[cov].std(ddof=1)),
                })
        # covariate-adjusted contrasts (total effect -> conditional effect)
        for control in ["RG4", "Control4C"]:
            for label, extras in [("none", []), ("z", ["z"]), ("logSN", ["logSN"]), ("z+logSN", ["z", "logSN"]), ("sSFR", ["sSFR"]), ("D4000n", ["D4000n"]), ("log_sigma", ["log_sigma"]), ("fibre_light_fraction", ["fibre_light_fraction"]), ("ff_miles_lw_logage", ["ff_miles_lw_logage"]), ("sSFR+D4000n+log_sigma", ["sSFR", "D4000n", "log_sigma"])]:
                d = select(w, role, control, extras=extras)
                r = contrast(d, PRIMARY_Y, rng, m0=m0, extras=extras)
                adjusted.append({"role": role, "control": control, "adjusted_for": label, "n_CG4": r["n_CG4"], "n_control": r["n_control"], "beta": r["beta"], "se": r["se"], "ci95_lo": r["ci95_lo"], "ci95_hi": r["ci95_hi"], "estimate": fmt(r)})
    return pd.DataFrame(balance), pd.DataFrame(adjusted)


# --------------------------------------------------------------------------
# G. environmental sequence
# --------------------------------------------------------------------------


def section_environment(w: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    cg_size90 = float(w[w["sample"].eq("CG4")].drop_duplicates("group_uid")["core_size_kpc"].quantile(.9))
    rg4_obj = set(w.loc[w["sample"].eq("RG4"), "objid"])
    definitions = [
        ("RG4", "RG4", None),
        ("Control4B", "Control4B", None),
        ("Control4B\\RG4", "Control4B", ~w["objid"].isin(rg4_obj)),
        ("Control4C", "Control4C", None),
        ("Control4C\\RG4", "Control4C", ~w["objid"].isin(rg4_obj)),
        ("C4C loose cores (size>CG4 p90)", "Control4C", w["core_size_kpc"].gt(cg_size90)),
        ("C4C compact cores (size<=CG4 p90)", "Control4C", w["core_size_kpc"].le(cg_size90)),
        ("C4C parent N=4 (=RG4)", "Control4C", w["parent_NbGal"].eq(4)),
        ("C4C parent N=5-7", "Control4C", w["parent_NbGal"].between(5, 7)),
        ("C4C parent N=8-12", "Control4C", w["parent_NbGal"].between(8, 12)),
        ("C4C parent N=13-24", "Control4C", w["parent_NbGal"].between(13, 24)),
        ("C4C parent N>=25", "Control4C", w["parent_NbGal"].ge(25)),
    ]
    rows = []
    for role in ["Satellite", "BGG"]:
        m0 = 10.2 if role == "Satellite" else 11.1
        for label, control, subset in definitions:
            sub = subset if subset is None else (subset | w["sample"].eq("CG4"))
            d = select(w, role, control, subset=sub)
            r = contrast(d, PRIMARY_Y, rng, m0=m0)
            dq = d[d["sf_state"].isin(["Quenched", "Starforming"])].copy()
            dq["x_SF"] = dq["sf_state"].eq("Starforming").astype(float)
            rq = contrast(dq, PRIMARY_Y, rng, m0=m0, extras=["x_SF"])
            ctrl = d[d["is_CG4"].eq(0)]
            rows.append({
                "role": role, "comparison": label, "n_CG4": r["n_CG4"], "n_control": r["n_control"], "groups_control": r["groups_control"],
                "beta": r["beta"], "se": r["se"], "ci68_lo": r["ci68_lo"], "ci68_hi": r["ci68_hi"], "ci95_lo": r["ci95_lo"], "ci95_hi": r["ci95_hi"],
                "beta_fixed_SF": rq["beta"], "beta_fixed_SF_ci95_lo": rq["ci95_lo"], "beta_fixed_SF_ci95_hi": rq["ci95_hi"],
                "control_median_core_size_kpc": float(ctrl["core_size_kpc"].median()), "control_median_group_Vdisp": float(ctrl["group_Vdisp"].median()),
                "control_median_parent_N": float(ctrl["parent_NbGal"].median()), "control_median_z": float(ctrl["z"].median()), "control_median_lgm": float(ctrl["lgm"].median()),
                "control_frac_quenched": float(ctrl["sf_state"].eq("Quenched").mean()), "control_frac_elliptical": float(ctrl["morphology4"].eq("Elliptical").mean()),
            })
    seq = pd.DataFrame(rows)
    cg = w[w["sample"].eq("CG4")].drop_duplicates("group_uid")
    seq.attrs["CG4_median_core_size_kpc"] = float(cg["core_size_kpc"].median())
    seq.attrs["CG4_median_group_Vdisp"] = float(cg["group_Vdisp"].median())
    seq.attrs["CG4_core_size_p90"] = cg_size90

    # within-control environmental slopes at fixed stellar mass
    slope_rows = []
    for control in ["Control4C", "Control4B"]:
        base = w[w[PRIMARY_U] & w["role"].eq("Satellite") & w["sample"].eq(control) & w["lgm"].notna()].copy()
        for label, fn in [("all", lambda x: pd.Series(True, index=x.index)), ("Quenched", lambda x: x["sf_state"].eq("Quenched")), ("Starforming", lambda x: x["sf_state"].eq("Starforming")), ("Elliptical", lambda x: x["morphology4"].eq("Elliptical")), ("Spiral", lambda x: x["morphology4"].eq("Spiral"))]:
            d = base[fn(base)]
            for env in ["log_parent_N", "log_core_size", "log_group_Vdisp"]:
                for extra_label, extras in [("", []), ("+z", ["z"]), ("+SF", ["x_SF"]), ("+z+SF", ["z", "x_SF"])]:
                    if "x_SF" in extras and label in ("Quenched", "Starforming"):
                        continue
                    dd = d.copy()
                    dd["x_SF"] = dd["sf_state"].eq("Starforming").astype(float)
                    if "x_SF" in extras:
                        dd = dd[dd["sf_state"].isin(["Quenched", "Starforming"])]
                    dd = dd.dropna(subset=[env] + extras)
                    dd["is_CG4"] = 0.0  # unused; contrast() needs the column
                    X = np.column_stack([np.ones(len(dd)), dd["lgm"] - 10.2, dd[env] - dd[env].median()] + [dd[e] - dd[e].median() for e in extras])
                    names = ["const", "mass", env] + extras
                    fit = za.cr1_ols(dd[PRIMARY_Y].to_numpy(float), X, names, dd["group_uid"].to_numpy())
                    j = names.index(env)
                    slope_rows.append({"control": control, "subset": label, "environment": env, "adjusted": extra_label or "mass only", "n": len(dd), "groups": dd["group_uid"].nunique(), "slope_per_dex": float(fit["beta"][j]), "se": float(fit["se"][j])})
    slopes = pd.DataFrame(slope_rows)

    # CG4 residual relative to a Control4C environmental model (mass + core size + group Vdisp)
    c = w[w[PRIMARY_U] & w["role"].eq("Satellite") & w["sample"].eq("Control4C") & w["lgm"].notna() & w["log_core_size"].notna() & w["log_group_Vdisp"].notna()].copy()
    Xc = np.column_stack([np.ones(len(c)), c["lgm"] - 10.2, c["log_core_size"] - 2.2, c["log_group_Vdisp"] - 2.2])
    bc, *_ = np.linalg.lstsq(Xc, c[PRIMARY_Y].to_numpy(float), rcond=None)
    env_rows = []
    for sample in ["CG4", "RG4", "Control4B", "Control4C"]:
        d = w[w[PRIMARY_U] & w["role"].eq("Satellite") & w["sample"].eq(sample) & w["lgm"].notna() & w["log_core_size"].notna() & w["log_group_Vdisp"].notna()]
        pred = bc[0] + bc[1] * (d["lgm"] - 10.2) + bc[2] * (d["log_core_size"] - 2.2) + bc[3] * (d["log_group_Vdisp"] - 2.2)
        r = (d[PRIMARY_Y] - pred).to_numpy(float)
        groups = d.groupby("group_uid").indices
        keys = list(groups)
        boots = []
        cgroups = c.groupby("group_uid").indices
        ckeys = list(cgroups)
        for _ in range(N_BOOT):
            crow = np.concatenate([cgroups[ckeys[k]] for k in rng.integers(0, len(ckeys), len(ckeys))])
            cb = c.iloc[crow]
            Xb = np.column_stack([np.ones(len(cb)), cb["lgm"] - 10.2, cb["log_core_size"] - 2.2, cb["log_group_Vdisp"] - 2.2])
            bb, *_ = np.linalg.lstsq(Xb, cb[PRIMARY_Y].to_numpy(float), rcond=None)
            drow = np.concatenate([groups[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
            db = d.iloc[drow]
            pb = bb[0] + bb[1] * (db["lgm"] - 10.2) + bb[2] * (db["log_core_size"] - 2.2) + bb[3] * (db["log_group_Vdisp"] - 2.2)
            boots.append(float((db[PRIMARY_Y] - pb).mean()))
        env_rows.append({"sample": sample, "n": len(d), "groups": len(keys), "mean_residual_vs_C4C_env_model": float(r.mean()), "ci95_lo": float(np.percentile(boots, 2.5)), "ci95_hi": float(np.percentile(boots, 97.5)), "median_core_size_kpc": float(d["core_size_kpc"].median()), "median_group_Vdisp": float(d["group_Vdisp"].median())})
    env = pd.DataFrame(env_rows)
    env.attrs["C4C_env_model_coefficients"] = {"const": bc[0], "lgm-10.2": bc[1], "log_core_size-2.2": bc[2], "log_group_Vdisp-2.2": bc[3]}
    seq.attrs["env_model"] = env.to_dict("records")
    seq.attrs["env_model_coefficients"] = env.attrs["C4C_env_model_coefficients"]
    return seq, slopes


# --------------------------------------------------------------------------
# H. mass bins
# --------------------------------------------------------------------------


def section_mass_bins(w: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    bins = {"Satellite": [(-np.inf, 10.0), (10.0, 10.5), (10.5, np.inf)], "BGG": [(-np.inf, 11.0), (11.0, 11.3), (11.3, np.inf)]}
    for role in ["Satellite", "BGG"]:
        for control in ["Control4C", "RG4"]:
            for lo, hi in bins[role]:
                d = select(w, role, control, subset=w["lgm"].gt(lo) & w["lgm"].le(hi))
                r = contrast(d, PRIMARY_Y, rng, m0=float(d["lgm"].median()))
                if r is None:
                    continue
                rows.append({"role": role, "control": control, "mass_bin": f"({lo:.1f}, {hi:.1f}]", "median_lgm": float(d["lgm"].median()), **r})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# I. star-forming BGG audit
# --------------------------------------------------------------------------


def section_bgg_sf(w: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = select(w, "BGG", "Control4C", subset=w["sf_state"].eq("Starforming"))
    ctrl = d[d["is_CG4"].eq(0)]
    b = _mzr_fit(ctrl, PRIMARY_Y, 1)
    d["resid_vs_C4C_SF_BGG"] = d[PRIMARY_Y] - _predict(b, d["lgm"].to_numpy(float))
    cols = ["objid", "Group", "plate", "mjd", "fiberID", "rank_M", "is_dominated", "lgm", "M_r", "z", "SN", "sigma", "sSFR", "morphology4", "p_E", "p_S", "D4000n", "HdA_sub", "mpa_subclass", "fibre_light_fraction", "fibre_radius_over_R50", "RUN2D", "spec_choice", "n_spectra_for_objid", "Chabrier_MILES_nComponentsSSP", "Chabrier_MILES_spm_EBV", "ff_miles_lw_logz", "ff_miles_lw_logz_unc", "ff_elodie_lw_logz", "ff_miles_mw_logz", "ff_elodie_mw_logz", "gallazzi_logz_solar", "gallazzi_usable", "ff_miles_lw_logage", "resid_vs_C4C_SF_BGG"]
    objects = d.loc[d["is_CG4"].eq(1), cols].sort_values("resid_vs_C4C_SF_BGG")
    objects["control_SF_BGG_resid_sd"] = float(d.loc[d["is_CG4"].eq(0), "resid_vs_C4C_SF_BGG"].std(ddof=1))

    full = contrast(d, PRIMARY_Y, rng, m0=11.1)
    loo = [{"dropped_objid": "none", "beta": full["beta"], "se": full["se"], "ci95_lo": full["ci95_lo"], "ci95_hi": full["ci95_hi"], "n_CG4": full["n_CG4"]}]
    for oid in objects["objid"]:
        r = contrast(d[d["objid"].ne(oid)], PRIMARY_Y, rng, m0=11.1, n_boot=500)
        loo.append({"dropped_objid": str(oid), "beta": r["beta"], "se": r["se"], "ci95_lo": r["ci95_lo"], "ci95_hi": r["ci95_hi"], "n_CG4": r["n_CG4"]})
    # also drop the two most extreme residuals together
    two = objects["objid"].iloc[:2].tolist()
    r = contrast(d[~d["objid"].isin(two)], PRIMARY_Y, rng, m0=11.1, n_boot=500)
    loo.append({"dropped_objid": "two lowest residuals", "beta": r["beta"], "se": r["se"], "ci95_lo": r["ci95_lo"], "ci95_hi": r["ci95_hi"], "n_CG4": r["n_CG4"]})

    prod = []
    for product, (ycol, ucol) in za.PRODUCTS.items():
        for label, subset in [("SF BGG", w["sf_state"].eq("Starforming")), ("SF BGG, S/N>20", w["sf_state"].eq("Starforming") & w["SN"].gt(20)), ("Spiral BGG", w["morphology4"].eq("Spiral")), ("SF BGG, Gallazzi∩FIREFLY", w["sf_state"].eq("Starforming") & w["gallazzi_usable"] & w["ff_miles_lw_usable"])]:
            dd = select(w, "BGG", "Control4C", ycol, ucol, subset=subset)
            r = contrast(dd, ycol, rng, m0=11.1, min_n=5)
            if r is None:
                prod.append({"product": product, "subset": label, "n_CG4": int(dd["is_CG4"].sum()), "n_control": int((1 - dd["is_CG4"]).sum()), "beta": np.nan, "estimate": "n/a (too few)"})
            else:
                prod.append({"product": product, "subset": label, **r, "estimate": fmt(r)})
    # age of the same subset
    for ycol, label in [("ff_miles_lw_logage", "MILES lw log age"), ("ff_miles_mw_logage", "MILES mw log age")]:
        dd = select(w, "BGG", "Control4C", ycol, PRIMARY_U, subset=w["sf_state"].eq("Starforming"))
        r = contrast(dd, ycol, rng, m0=11.1, min_n=5)
        prod.append({"product": label, "subset": "SF BGG", **r, "estimate": fmt(r)})
    return objects, pd.DataFrame(loo), pd.DataFrame(prod)


# --------------------------------------------------------------------------
# J. age
# --------------------------------------------------------------------------


def section_age(w: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    diag = []
    for ycol, ucol, name in [("ff_miles_lw_logage", "ff_miles_lw_usable", "FIREFLY MILES light-weighted age"), ("ff_miles_mw_logage", "ff_miles_mw_usable", "FIREFLY MILES mass-weighted age"), ("ff_elodie_lw_logage", "ff_elodie_lw_usable", "FIREFLY ELODIE light-weighted age"), ("gal_logage", "gallazzi_usable", "Gallazzi r-band light-weighted age")]:
        u = w[w[ucol] & w[ycol].notna()]
        for (role, sample), part in u.groupby(["role", "sample"]):
            diag.append({
                "age_product": name, "role": role, "sample": sample, "n": len(part), "unit": "log10(age/yr)",
                "median_log_age": float(part[ycol].median()), "median_age_Gyr": float(10 ** part[ycol].median() / 1e9),
                "median_unc_dex": float(part[ycol + "_unc"].median()),
                "frac_gt_13.8Gyr": float((10 ** part[ycol] > 1.38e10).mean()), "frac_at_grid_max": float(part[ycol.replace("_logage", "_age_at_grid_max")].mean()) if ycol.replace("_logage", "_age_at_grid_max") in part else np.nan,
                "frac_lt_1Gyr": float((10 ** part[ycol] < 1e9).mean()),
            })
    rows = []
    for ycol, ucol, name in [("ff_miles_lw_logage", "ff_miles_lw_usable", "FIREFLY MILES light-weighted age"), ("ff_miles_mw_logage", "ff_miles_mw_usable", "FIREFLY MILES mass-weighted age"), ("ff_elodie_lw_logage", "ff_elodie_lw_usable", "FIREFLY ELODIE light-weighted age"), ("gal_logage", "gallazzi_usable", "Gallazzi r-band light-weighted age")]:
        for role, control in [("Satellite", "Control4C"), ("Satellite", "RG4"), ("BGG", "Control4C"), ("BGG", "RG4")]:
            m0 = 10.2 if role == "Satellite" else 11.1
            for label, fn in SUBSETS.items():
                if "&" in label or label == "morph=Uncertain":
                    continue
                d = select(w, role, control, ycol, ucol, subset=fn(w))
                r = contrast(d, ycol, rng, m0=m0, min_n=6)
                if r is None:
                    rows.append({"age_product": name, "role": role, "control": control, "subset": label, "n_CG4": int(d["is_CG4"].sum()), "n_control": int((1 - d["is_CG4"]).sum()), "beta": np.nan, "estimate": "n/a"})
                    continue
                rows.append({"age_product": name, "role": role, "control": control, "subset": label, **r, "estimate": fmt(r)})
            # age adjusted for SF state and for D4000n (is the age offset a quenched-fraction effect?)
            for adj_label, extras, sub in [("+SF state", ["x_SF"], w["sf_state"].isin(["Quenched", "Starforming"])), ("+D4000n", ["D4000n"], None), ("+sSFR", ["sSFR"], None)]:
                d = select(w, role, control, ycol, ucol, subset=sub)
                d["x_SF"] = d["sf_state"].eq("Starforming").astype(float)
                d = d.dropna(subset=extras)
                r = contrast(d, ycol, rng, m0=m0, extras=extras)
                rows.append({"age_product": name, "role": role, "control": control, "subset": "all " + adj_label, **r, "estimate": fmt(r)})
    return pd.DataFrame(rows), pd.DataFrame(diag)


def section_age_same_objects(w: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """FIREFLY vs Gallazzi ages on exactly the same galaxies (+ the age/Z residual correlation)."""
    rows = []
    both = w["gallazzi_usable"] & w["ff_miles_lw_usable"] & w["gal_logage"].notna() & w["ff_miles_lw_logage"].notna()
    for role in ["Satellite", "BGG"]:
        m0 = 10.2 if role == "Satellite" else 11.1
        for control in ["Control4C", "RG4"]:
            for label, sub in [("all", pd.Series(True, index=w.index)), ("SF=Quenched", w["sf_state"].eq("Quenched"))]:
                d = select(w, role, control, "ff_miles_lw_logage", "ff_miles_lw_usable", subset=both & sub)
                d["g_minus_f"] = d["gal_logage"] - d["ff_miles_lw_logage"]
                rf = contrast(d, "ff_miles_lw_logage", rng, m0=m0, min_n=6)
                rg = contrast(d, "gal_logage", rng, m0=m0, min_n=6)
                rd = contrast(d, "g_minus_f", rng, m0=m0, min_n=6)
                if rf is None:
                    continue
                notg = contrast(select(w, role, control, "ff_miles_lw_logage", "ff_miles_lw_usable", subset=~w["gallazzi_usable"] & sub), "ff_miles_lw_logage", rng, m0=m0, min_n=6)
                rows.append({
                    "role": role, "control": control, "subset": label, "n_CG4": rf["n_CG4"], "n_control": rf["n_control"],
                    "firefly_age_same_objects": fmt(rf), "gallazzi_age_same_objects": fmt(rg), "gallazzi_minus_firefly": fmt(rd),
                    "firefly_age_not_in_gallazzi": (fmt(notg) + f" (N={notg['n_CG4']}+{notg['n_control']})") if notg else "n/a",
                    "median_offset_G_minus_F_dex": float(d["g_minus_f"].median()), "robust_scatter_G_minus_F_dex": float(1.4826 * np.median(np.abs(d["g_minus_f"] - d["g_minus_f"].median()))),
                })
    # age-metallicity residual correlation within Control4C satellites (degeneracy check)
    c = w[w[PRIMARY_U] & w["role"].eq("Satellite") & w["sample"].eq("Control4C") & w["lgm"].notna() & w["ff_miles_lw_logage"].notna()].copy()
    X = np.column_stack([np.ones(len(c)), c["lgm"] - 10.2])
    bz, *_ = np.linalg.lstsq(X, c[PRIMARY_Y].to_numpy(float), rcond=None)
    ba, *_ = np.linalg.lstsq(X, c["ff_miles_lw_logage"].to_numpy(float), rcond=None)
    rz, ra = c[PRIMARY_Y].to_numpy(float) - X @ bz, c["ff_miles_lw_logage"].to_numpy(float) - X @ ba
    slope = np.polyfit(ra, rz, 1)[0]
    rows.append({"role": "Satellite", "control": "Control4C", "subset": "within-control residual correlation", "n_control": len(c), "firefly_age_same_objects": f"corr(resid Z, resid age) = {np.corrcoef(rz, ra)[0, 1]:+.3f}; slope dZ/dlogage = {slope:+.3f} dex/dex", "gallazzi_age_same_objects": f"quenched only: corr = {np.corrcoef(rz[c['sf_state'].eq('Quenched').to_numpy()], ra[c['sf_state'].eq('Quenched').to_numpy()])[0, 1]:+.3f}"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# M. exploratory gas-phase metallicity (post-hoc addition)
# --------------------------------------------------------------------------


def section_gas_phase(w: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strong-line O/H (PP04 N2 and O3N2) for MPA-JHU BPT star-forming galaxies.

    Emission-line fluxes come from the project's cached MPA-JHU galSpecLine
    columns (data/processed_sample.pkl, read-only); the MPA `subclass`
    STARFORMING/STARBURST already requires S/N>3 in the four BPT lines and a
    position below the Kauffmann et al. (2003) line.  Only satellites have
    enough star-forming CG4 members.
    """
    pkl = pickle.load(open(REPO / "data" / "processed_sample.pkl", "rb"))
    S = pkl["SDSS_withAGN"][["objid", "h_alpha_flux", "nii_6584_flux", "oiii_5007_flux", "h_beta_flux", "specObjID"]].copy()
    S["objid"] = S["objid"].astype("int64")
    m = pd.read_csv(HERE / "work" / "metallicity_worktable.csv", low_memory=False, usecols=["sample", "objid", "specObjID"]).rename(columns={"specObjID": "chosen_specObjID"})
    d = w.merge(m, on=["sample", "objid"], how="left", validate="1:1").merge(S.rename(columns={"specObjID": "cache_specObjID"}), on="objid", how="left", validate="m:1")
    sf = d["mpa_subclass"].isin(["STARFORMING", "STARBURST"])
    pos = (d["h_alpha_flux"] > 0) & (d["nii_6584_flux"] > 0) & (d["oiii_5007_flux"] > 0) & (d["h_beta_flux"] > 0)
    same_spec = pd.to_numeric(d["cache_specObjID"], errors="coerce") == pd.to_numeric(d["chosen_specObjID"], errors="coerce")
    d["gas_usable"] = sf & pos & same_spec & d["lgm"].notna()
    with np.errstate(divide="ignore", invalid="ignore"):
        n2 = np.log10(d["nii_6584_flux"] / d["h_alpha_flux"])
        o3n2 = np.log10((d["oiii_5007_flux"] / d["h_beta_flux"]) / (d["nii_6584_flux"] / d["h_alpha_flux"]))
    d["oh_N2_PP04"] = (8.90 + 0.57 * n2).where(d["gas_usable"] & n2.between(-2.5, -0.3))
    d["oh_O3N2_PP04"] = (8.73 - 0.32 * o3n2).where(d["gas_usable"] & o3n2.between(-1, 1.9))
    d["log_sfr"] = d["sSFR"] + d["lgm"]
    census = d[d["gas_usable"]].groupby(["role", "sample"]).agg(n_gas_usable=("objid", "size"), n_N2=("oh_N2_PP04", "count"), n_O3N2=("oh_O3N2_PP04", "count"), median_oh_N2=("oh_N2_PP04", "median"), median_lgm=("lgm", "median")).reset_index()
    rows = []
    for ycol in ["oh_N2_PP04", "oh_O3N2_PP04"]:
        for control in ["Control4C", "RG4", "Control4B"]:
            for degree in [1, 2]:
                for adj_label, extras in [("mass only", []), ("mass + log SFR", ["log_sfr"]), ("mass + D4000n", ["D4000n"])]:
                    dd = select(d, "Satellite", control, ycol, "gas_usable", extras=extras)
                    r = contrast(dd, ycol, rng, m0=10.2, degree=degree, extras=extras)
                    if r is None:
                        continue
                    rows.append({"calibration": ycol, "control": control, "degree": degree, "adjusted": adj_label, **{k: r[k] for k in ["n_CG4", "n_control", "groups_CG4", "groups_control", "beta", "se", "ci68_lo", "ci68_hi", "ci95_lo", "ci95_hi"]}, "estimate": fmt(r)})
            # by morphology (spirals dominate the star-forming population)
            for label, sub in [("morph=Spiral", d["morphology4"].eq("Spiral")), ("morph=Elliptical", d["morphology4"].eq("Elliptical"))]:
                dd = select(d, "Satellite", control, ycol, "gas_usable", subset=sub)
                r = contrast(dd, ycol, rng, m0=10.2, degree=2, min_n=6)
                if r is None:
                    continue
                rows.append({"calibration": ycol, "control": control, "degree": 2, "adjusted": label, **{k: r[k] for k in ["n_CG4", "n_control", "groups_CG4", "groups_control", "beta", "se", "ci68_lo", "ci68_hi", "ci95_lo", "ci95_hi"]}, "estimate": fmt(r)})
    return pd.DataFrame(rows), census


# --------------------------------------------------------------------------
# L. robustness: quantile regression; S/N-restricted RG4 contrast
# --------------------------------------------------------------------------


def section_robustness(w: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for role, control in [("Satellite", "Control4C"), ("Satellite", "RG4"), ("BGG", "Control4C"), ("BGG", "RG4")]:
        m0 = 10.2 if role == "Satellite" else 11.1
        d = select(w, role, control)
        X = np.column_stack([np.ones(len(d)), d["lgm"] - m0, d["is_CG4"]])
        y = d[PRIMARY_Y].to_numpy(float)
        q = sm.QuantReg(y, X).fit(q=0.5, max_iter=5000)
        # cluster bootstrap for the median-regression CG4 coefficient
        groups = d.groupby(["sample", "group_uid"]).indices
        by_sample: dict[str, list] = {}
        for key in groups:
            by_sample.setdefault(key[0], []).append(key)
        boots = []
        for _ in range(400):
            rows_i: list[int] = []
            for keys in by_sample.values():
                for k in rng.integers(0, len(keys), len(keys)):
                    rows_i.extend(groups[keys[k]])
            b = d.iloc[rows_i]
            Xb = np.column_stack([np.ones(len(b)), b["lgm"] - m0, b["is_CG4"]])
            try:
                boots.append(float(sm.QuantReg(b[PRIMARY_Y].to_numpy(float), Xb).fit(q=0.5, max_iter=5000).params[2]))
            except Exception:  # noqa: BLE001
                continue
        ols = contrast(d, PRIMARY_Y, rng, m0=m0)
        rows.append({"role": role, "control": control, "variant": "median regression", "n_CG4": ols["n_CG4"], "n_control": ols["n_control"], "beta": float(q.params[2]), "ci95_lo": float(np.percentile(boots, 2.5)), "ci95_hi": float(np.percentile(boots, 97.5)), "ols_beta": ols["beta"], "ols_ci95_lo": ols["ci95_lo"], "ols_ci95_hi": ols["ci95_hi"]})
        # trimmed: drop the 2% lowest residuals overall (heavy low-Z tail)
        b_ctrl = _mzr_fit(d[d["is_CG4"].eq(0)], PRIMARY_Y, 1)
        resid = d[PRIMARY_Y] - _predict(b_ctrl, d["lgm"].to_numpy(float))
        dt = d[resid > resid.quantile(0.02)]
        r = contrast(dt, PRIMARY_Y, rng, m0=m0)
        rows.append({"role": role, "control": control, "variant": "OLS, lowest 2% residuals removed", "n_CG4": r["n_CG4"], "n_control": r["n_control"], "beta": r["beta"], "ci95_lo": r["ci95_lo"], "ci95_hi": r["ci95_hi"], "ols_beta": ols["beta"], "ols_ci95_lo": ols["ci95_lo"], "ols_ci95_hi": ols["ci95_hi"]})
        if control == "RG4":
            for label, sub in [("S/N>20", w["SN"].gt(20)), ("S/N>15", w["SN"].gt(15)), ("uncertainty<=0.1 dex", w["ff_miles_lw_logz_unc"].le(0.1)), ("z<=0.04", w["z"].le(0.04))]:
                dd = select(w, role, control, subset=sub)
                r = contrast(dd, PRIMARY_Y, rng, m0=m0)
                rows.append({"role": role, "control": control, "variant": label, "n_CG4": r["n_CG4"], "n_control": r["n_control"], "beta": r["beta"], "ci95_lo": r["ci95_lo"], "ci95_hi": r["ci95_hi"], "ols_beta": ols["beta"], "ols_ci95_lo": ols["ci95_lo"], "ols_ci95_hi": ols["ci95_hi"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def forest(ax, d: pd.DataFrame, label_col: str, title: str, xlabel: str = "CG4 − control at fixed mass (dex)") -> None:
    d = d.reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    ax.hlines(y, d["ci95_lo"], d["ci95_hi"], color="0.7", lw=2)
    if "ci68_lo" in d:
        ax.hlines(y, d["ci68_lo"], d["ci68_hi"], color="#2864A6", lw=5)
    ax.plot(d["beta"], y, "ko", ms=4)
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(y, d[label_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=.25)


def plot_sequence(seq: pd.DataFrame, slopes: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.2, 1]})
    d = seq[seq["role"].eq("Satellite")].copy()
    d["label"] = d.apply(lambda r: f"{r['comparison']}  (N={int(r['n_control'])}, size {r['control_median_core_size_kpc']:.0f} kpc, σ {r['control_median_group_Vdisp']:.0f})", axis=1)
    forest(axes[0], d, "label", "Satellites: CG4 minus each comparison set (FIREFLY MILES lw)")
    yy = np.arange(len(d))[::-1]
    axes[0].plot(d["beta_fixed_SF"], yy - 0.25, marker="s", ls="none", color="#A74752", ms=4, label="also at fixed SF state")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    s = slopes[slopes["control"].eq("Control4C") & slopes["adjusted"].eq("mass only")]
    envs = ["log_parent_N", "log_core_size", "log_group_Vdisp"]
    subsets = ["all", "Quenched", "Starforming", "Elliptical", "Spiral"]
    x = np.arange(len(subsets))
    for k, env in enumerate(envs):
        part = s[s["environment"].eq(env)].set_index("subset").loc[subsets]
        axes[1].errorbar(x + (k - 1) * 0.22, part["slope_per_dex"], yerr=1.96 * part["se"], fmt="o", capsize=3, label=env.replace("log_", "log "))
    axes[1].axhline(0, color="k", lw=.8)
    axes[1].set_xticks(x, subsets)
    axes[1].set_ylabel("d[Z/H] / d(environment) at fixed M* (dex per dex)")
    axes[1].set_title("Within Control4C satellites: slopes at fixed M* (95% CR1)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("Environmental sequence of satellite stellar metallicity")
    fig.tight_layout()
    fig.savefig(FIG / "environment_sequence.png", dpi=160)
    plt.close(fig)


def plot_decomposition(sub: pd.DataFrame, models: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True)
    for ax, control in zip(axes, ["RG4", "Control4C"]):
        d = sub[sub["product"].eq(za.PRIMARY) & sub["role"].eq("Satellite") & sub["control"].eq(control) & sub["beta"].notna()].copy()
        d["label"] = d.apply(lambda r: f"{r['subset']} (N={int(r['n_CG4'])}+{int(r['n_control'])})", axis=1)
        forest(ax, d, "label", f"Satellites: CG4 vs {control} by subset")
    fig.suptitle("Where does the CG4–RG4 satellite contrast live? (FIREFLY MILES lw; 68%/95% group bootstrap)")
    fig.tight_layout()
    fig.savefig(FIG / "rg4_decomposition.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cov in zip(axes, ["Spiral", "SF"]):
        d = models[models["role"].eq("Satellite") & models["covariate"].eq(cov)].copy()
        d["label"] = d["product"].str.replace("FIREFLY ", "").str.replace(" optical light-weighted", "") + " vs " + d["control"]
        d = d.rename(columns={"int_CG4xcov_beta": "beta", "int_CG4xcov_ci95_lo": "ci95_lo", "int_CG4xcov_ci95_hi": "ci95_hi"})
        forest(ax, d, "label", f"CG4 × {cov} interaction (satellites)", xlabel=f"extra CG4 offset for {cov} relative to reference class (dex)")
    fig.suptitle("Is the CG4 contrast different for spirals / star-forming galaxies? (all products)")
    fig.tight_layout()
    fig.savefig(FIG / "rg4_interactions_by_product.png", dpi=160)
    plt.close(fig)


def plot_residuals(objects: pd.DataFrame, summary: pd.DataFrame, w: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, control in enumerate(["Control4C", "RG4"]):
        d = objects[objects["control"].eq(control) & objects["role"].eq("Satellite")]
        ax = axes[0, col]
        bins = np.linspace(-0.6, 0.4, 41)
        for sample, color in [(control, PALETTE[control]), ("CG4", PALETTE["CG4"])]:
            part = d[d["sample"].eq(sample)]
            ax.hist(part["resid_deg1"], bins=bins, density=True, histtype="step", lw=2, color=color, label=f"{sample} (N={len(part)})")
            ax.axvline(part["resid_deg1"].median(), color=color, ls="--", lw=1)
        ax.set(xlabel=f"ΔZ = [Z/H] − Z_hat_{control}(M*)  (dex)", ylabel="density", title=f"Satellites: residuals from the {control}-only linear MZR")
        ax.legend(frameon=False)
        ax.grid(alpha=.25)
        ax = axes[1, col]
        cg = d[d["sample"].eq("CG4")]
        ct = d[d["sample"].eq(control)]
        edges = np.array([9.0, 9.6, 9.9, 10.2, 10.5, 10.8, 11.4])
        for part, color, label, off in [(ct, PALETTE[control], control, -0.02), (cg, PALETTE["CG4"], "CG4", 0.02)]:
            g = part.groupby(pd.cut(part["lgm"], edges), observed=True)["resid_deg1"]
            med = g.median()
            n = g.size()
            se = 1.253 * g.std() / np.sqrt(n.clip(lower=1))
            centres = np.array([iv.mid for iv in med.index]) + off
            ax.errorbar(centres, med, yerr=1.96 * se, fmt="o-", color=color, capsize=3, label=label)
            for xc, nn, mm in zip(centres, n, med):
                ax.annotate(str(nn), (xc, mm), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color=color)
        ax.axhline(0, color="k", lw=.8)
        ax.set(xlabel="log stellar mass", ylabel="median ΔZ (dex)", title=f"Binned medians (95% CI of the median); CG4 vs {control}")
        ax.legend(frameon=False)
        ax.grid(alpha=.25)
    fig.suptitle("Control-only MZR → residuals (FIREFLY MILES lw). Bins are wide on purpose.")
    fig.tight_layout()
    fig.savefig(FIG / "control_only_residuals.png", dpi=160)
    plt.close(fig)


def plot_mass_bins(bins: pd.DataFrame, inter: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, role in zip(axes, ["Satellite", "BGG"]):
        for control, color, off in [("Control4C", PALETTE["Control4C"], -0.02), ("RG4", PALETTE["RG4"], 0.02)]:
            d = bins[bins["role"].eq(role) & bins["control"].eq(control)]
            if d.empty:
                continue
            ax.errorbar(d["median_lgm"] + off, d["beta"], yerr=[d["beta"] - d["ci95_lo"], d["ci95_hi"] - d["beta"]], fmt="o", color=color, capsize=4, label=f"vs {control} (3 broad bins)")
            for _, r in d.iterrows():
                ax.annotate(f"{int(r['n_CG4'])}+{int(r['n_control'])}", (r["median_lgm"] + off, r["beta"]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7, color=color)
            ii = inter[inter["role"].eq(role) & inter["control"].eq(control) & inter["product"].eq(za.PRIMARY)].iloc[0]
            grid = np.linspace(ii["common_mass_5_95_lo"], ii["common_mass_5_95_hi"], 50)
            ax.plot(grid, ii["beta_at_m0"] + ii["gamma"] * (grid - ii["m0"]), color=color, lw=1.2, ls="--", label=f"linear interaction vs {control}: γ={ii['gamma']:+.3f}±{ii['gamma_se']:.3f}")
        ax.axhline(0, color="k", lw=.8)
        ax.set(xlabel="log stellar mass", ylabel="CG4 − control (dex)", title=role)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=.25)
    fig.suptitle("Mass dependence of the CG4 contrast (FIREFLY MILES lw; 95% group bootstrap)")
    fig.tight_layout()
    fig.savefig(FIG / "delta_z_vs_mass.png", dpi=160)
    plt.close(fig)


def plot_age(age: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, (role, control) in zip(axes, [("Satellite", "Control4C"), ("Satellite", "RG4"), ("BGG", "Control4C")]):
        d = age[age["role"].eq(role) & age["control"].eq(control) & age["beta"].notna() & age["subset"].isin(["all", "SF=Quenched", "SF=Starforming", "morph=Elliptical", "morph=Spiral", "all +SF state"])].copy()
        d["label"] = d["age_product"].str.replace("FIREFLY ", "").str.replace(" light-weighted age", " lw").str.replace(" mass-weighted age", " mw").str.replace("Gallazzi r-band lw", "Gallazzi") + ": " + d["subset"]
        forest(ax, d, "label", f"{role}: CG4 vs {control}", xlabel="Δ log age (dex) at fixed mass")
    fig.suptitle("Exploratory age contrasts (log10 age; FIREFLY and Gallazzi)")
    fig.tight_layout()
    fig.savefig(FIG / "age_effects.png", dpi=160)
    plt.close(fig)


def plot_gas(gas: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    for ax, cal in zip(axes, ["oh_N2_PP04", "oh_O3N2_PP04"]):
        d = gas[gas["calibration"].eq(cal) & gas["degree"].eq(2)].copy()
        d["label"] = "vs " + d["control"] + ": " + d["adjusted"] + " (N=" + d["n_CG4"].astype(str) + "+" + d["n_control"].astype(str) + ")"
        forest(ax, d, "label", cal.replace("oh_", "12+log(O/H) ").replace("_PP04", " (PP04)"), xlabel="CG4 − control at fixed mass (dex)")
    fig.suptitle("Exploratory gas-phase metallicity of BPT star-forming satellites (quadratic mass term; 68%/95% group bootstrap)")
    fig.tight_layout()
    fig.savefig(FIG / "gas_phase_effects.png", dpi=160)
    plt.close(fig)


def plot_same_objects(same: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    d = same.copy()
    d["label"] = d["role"] + " vs " + d["control"] + " (N=" + d["n_CG4"].astype(str) + "+" + d["n_control"].astype(str) + ")"
    y = np.arange(len(d))[::-1]
    for key, color, off, name in [("ff", "#2864A6", 0.18, "FIREFLY MILES lw"), ("gal", "#A74752", -0.18, "Gallazzi")]:
        ax.errorbar(d[f"{key}_beta"], y + off, xerr=[d[f"{key}_beta"] - d[f"{key}_ci95_lo"], d[f"{key}_ci95_hi"] - d[f"{key}_beta"]], fmt="o", color=color, capsize=3, label=name + " (same objects)")
    ax.plot(d["ff_full_beta"], y + 0.18, marker="x", ls="none", color="#2864A6", label="FIREFLY, full sample")
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(y, d["label"])
    ax.set_xlabel("CG4 − control at fixed mass (dex)")
    ax.set_title("Gallazzi vs FIREFLY on exactly the same galaxies (95% group bootstrap)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(FIG / "same_objects.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    w = load()

    repro, censoring, notes = section_audit(w, rng)
    same = section_same_objects(w, rng)
    inter = section_interaction(w, rng)
    resid_summary, resid_objects = section_control_residuals(w, rng)
    sub, models, oaxaca = section_rg4_decomposition(w, rng)
    balance, adjusted = section_composition(w, rng)
    seq, slopes = section_environment(w, rng)
    bins = section_mass_bins(w, rng)
    bgg_objects, bgg_loo, bgg_products = section_bgg_sf(w, rng)
    age, age_diag = section_age(w, rng)
    age_same = section_age_same_objects(w, rng)
    gas, gas_census = section_gas_phase(w, rng)
    robust = section_robustness(w, rng)

    env_model = pd.DataFrame(seq.attrs["env_model"])
    tables = {
        "audit_reproduction.csv": repro, "audit_censoring.csv": censoring,
        "same_objects.csv": same, "mass_interaction.csv": inter,
        "control_only_residuals.csv": resid_summary, "control_only_residuals_objects.csv": resid_objects,
        "rg4_subsets.csv": sub, "rg4_joint_models.csv": models, "rg4_composition_split.csv": oaxaca,
        "composition_balance.csv": balance, "adjusted_contrasts.csv": adjusted,
        "environment_sequence.csv": seq, "within_control_environment_slopes.csv": slopes, "environment_model_residuals.csv": env_model,
        "mass_bins.csv": bins,
        "bgg_sf_objects.csv": bgg_objects, "bgg_sf_leave_one_out.csv": bgg_loo, "bgg_sf_products.csv": bgg_products,
        "age_contrasts.csv": age, "age_grid_diagnostics.csv": age_diag, "age_same_objects.csv": age_same,
        "gas_phase_contrasts.csv": gas, "gas_phase_census.csv": gas_census,
        "robustness.csv": robust,
    }
    keep = ["sample", "group_uid", "objid", "role", "lgm", "z", "SN", "sf_state", "morphology4", "sSFR", "D4000n", "sigma", "p_E", "p_S", "core_size_kpc", "group_Vdisp", "parent_NbGal", "group_is_RG4", "ff_miles_lw_usable", "ff_miles_lw_logz", "ff_miles_lw_logz_unc", "ff_miles_mw_logz", "ff_elodie_lw_logz", "ff_elodie_mw_logz", "gallazzi_usable", "gallazzi_logz_solar", "ff_miles_lw_logage", "ff_miles_lw_logage_unc", "ff_miles_mw_logage", "ff_elodie_lw_logage", "gal_logage", "gal_logage_unc"]
    tables["followup_worktable.csv"] = w[keep]
    paths = []
    for name, frame in tables.items():
        p = OUT / name
        frame.to_csv(p, index=False)
        paths.append(p)

    plot_sequence(seq, slopes)
    plot_decomposition(sub, models)
    plot_residuals(resid_objects, resid_summary, w)
    plot_mass_bins(bins, inter)
    plot_age(age)
    plot_same_objects(same)
    plot_gas(gas)

    key = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": SEED, "n_boot": N_BOOT,
        "audit_notes": notes,
        "max_abs_beta_reproduction_diff": float(repro["abs_diff"].max()),
        "CG4_group_medians": {"core_size_kpc": seq.attrs["CG4_median_core_size_kpc"], "group_Vdisp": seq.attrs["CG4_median_group_Vdisp"], "core_size_p90": seq.attrs["CG4_core_size_p90"]},
        "env_model_coefficients": {k: float(v) for k, v in seq.attrs["env_model_coefficients"].items()},
    }
    (OUT / "key_numbers.json").write_text(json.dumps(key, indent=2, default=float))
    paths.append(OUT / "key_numbers.json")
    paths.extend(sorted(FIG.glob("*.png")))
    manifest = {
        "created_utc": key["created_utc"], "seed": SEED, "n_boot": N_BOOT,
        "scope": "writes only below exploration/metallicity/{outputs/zstar_followup,figures/zstar_followup}",
        "inputs": [str(p.relative_to(HERE)) for p in [HIST / "zstar_contrasts.csv", HERE / "work" / "metallicity_worktable.csv", HERE / "work" / "catalogues" / "firefly" / "firefly_sample_matches.csv", HERE / "work" / "catalogues" / "gallazzi" / "gallazzi_sample_matches.csv"]] + ["../../data/processed_sample.pkl (read-only)", "../../data/PC_Groups.csv (read-only)"],
        "files": [{"path": str(p.relative_to(HERE)), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
    }
    (OUT / "followup_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"tables": list(tables), "max_abs_beta_reproduction_diff": key["max_abs_beta_reproduction_diff"]}, indent=2))


if __name__ == "__main__":
    main()
