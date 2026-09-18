"""Direct stellar-metallicity extension of the historical index study.

This script is deliberately self-contained and writes only below this
directory.  It treats Control4C as the primary ordinary-group comparison,
keeps BGGs and satellites separate, reports effect sizes and intervals rather
than p-values, and never modifies the historical Claude products.
"""
from __future__ import annotations

import json
import math
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
CAT = WORK / "catalogues"
OUT = HERE / "outputs" / "zstar"
FIG = HERE / "figures" / "zstar"
SEED = 20260915
N_BOOT = 1000
ZSUN_GALLAZZI = 0.02
SAMPLES = ["CG4", "RG4", "Control4B", "Control4C"]
CONTROLS = ["Control4C", "Control4B", "RG4"]
ROLES = {"BGG": "is_bgg", "Satellite": "is_sat"}

PRODUCTS = {
    "FIREFLY MILES light-weighted": ("ff_miles_lw_logz", "ff_miles_lw_usable"),
    "FIREFLY ELODIE light-weighted": ("ff_elodie_lw_logz", "ff_elodie_lw_usable"),
    "FIREFLY MILES mass-weighted": ("ff_miles_mw_logz", "ff_miles_mw_usable"),
    "FIREFLY ELODIE mass-weighted": ("ff_elodie_mw_logz", "ff_elodie_mw_usable"),
    "Gallazzi optical light-weighted": ("gallazzi_logz_solar", "gallazzi_usable"),
}
PRIMARY = "FIREFLY MILES light-weighted"

PALETTE = {"CG4": "#A74752", "Control4C": "#25876E", "Control4B": "#2864A6", "RG4": "#7A5195"}


def finite(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """Row-wise finite mask that is false for missing columns."""
    cols = list(columns)
    if any(c not in frame for c in cols):
        return pd.Series(False, index=frame.index)
    return pd.Series(np.isfinite(frame[cols].to_numpy(dtype=float)).all(axis=1), index=frame.index)


def prepare_worktable() -> pd.DataFrame:
    """Attach catalogues, standardise units, and construct explicit flags."""
    w = pd.read_csv(WORK / "metallicity_worktable.csv", low_memory=False)
    f = pd.read_csv(CAT / "firefly" / "firefly_sample_matches.csv", low_memory=False)
    g = pd.read_csv(CAT / "gallazzi" / "gallazzi_sample_matches.csv", low_memory=False)
    p = pd.read_csv(CAT / "sdss_aux" / "sdss_photometry.csv", low_memory=False)

    f = f.rename(columns={"fiberid": "fiberID"})
    g = g.rename(columns={"fiberid": "fiberID"})
    keys = ["plate", "mjd", "fiberID"]
    w = w.merge(f, on=keys, how="left", validate="m:1", suffixes=("", "_ff"))
    w = w.merge(g, on=keys, how="left", validate="m:1")
    w = w.merge(p, on="objid", how="left", validate="m:1")

    w["role"] = np.where(w["is_bgg"].eq(1), "BGG", "Satellite")
    w["morphology4"] = w["morphology"].fillna("NoGZ").astype(str)
    w["sf_state"] = w["sSFR_status"].fillna("NosSFR").astype(str)
    w["r_petro_dered"] = w["petroMag_r"] - w["extinction_r"]
    w["fibre_light_fraction"] = 10 ** (-0.4 * (w["fiberMag_r"] - w["petroMag_r"]))
    w["R50_arcsec"] = w["Rchl_r_arcsec"].where(w["Rchl_r_arcsec"] > 0, w["petroR50_r"].where(w["petroR50_r"] > 0))
    w["R50_kpc"] = w["R50_arcsec"] * w["kpc_per_arcsec_Planck15_DA"]
    w["fibre_radius_kpc"] = 1.5 * w["kpc_per_arcsec_Planck15_DA"]
    w["fibre_radius_over_R50"] = 1.5 / w["R50_arcsec"]
    w["log_sigma"] = np.log10(w["sigma"].where(w["sigma"] > 0))

    # Gallazzi DR4 stores log10(Z), where Z is the absolute metal mass
    # fraction.  The catalogue page specifies Z_sun=0.02 for conversion.
    gz = ["z_log_abs_p16", "z_log_abs_median", "z_log_abs_p84"]
    w["gallazzi_match"] = w["z_log_abs_median"].notna()
    w["gallazzi_valid"] = finite(w, gz) & w[gz].gt(-10).all(axis=1)
    w["gallazzi_valid"] &= (w["z_log_abs_p16"] <= w["z_log_abs_median"]) & (w["z_log_abs_median"] <= w["z_log_abs_p84"])
    logzsun = math.log10(ZSUN_GALLAZZI)
    w["gallazzi_logz_solar"] = (w["z_log_abs_median"] - logzsun).where(w["gallazzi_valid"])
    w["gallazzi_logz_lo"] = (w["z_log_abs_p16"] - logzsun).where(w["gallazzi_valid"])
    w["gallazzi_logz_hi"] = (w["z_log_abs_p84"] - logzsun).where(w["gallazzi_valid"])
    w["gallazzi_logz_unc"] = ((w["gallazzi_logz_hi"] - w["gallazzi_logz_lo"]) / 2).where(w["gallazzi_valid"])
    # Gallazzi et al. demonstrate that S/N per pixel >=20 is required for
    # reliable metallicity constraints.  This is catalogue-specific.
    w["gallazzi_usable"] = w["gallazzi_valid"] & w["SN"].ge(20)

    w["firefly_match"] = w["PLATE"].notna()
    w["firefly_spectrum_quality"] = (
        w["CLASS_NOQSO"].eq("GALAXY")
        & w["Z_NOQSO"].gt(w["Z_ERR_NOQSO"])
        & w["Z_ERR_NOQSO"].gt(0)
    )
    ff_defs = {
        "ff_miles_lw": "Chabrier_MILES_metallicity_lightW",
        "ff_elodie_lw": "Chabrier_ELODIE_metallicity_lightW",
        "ff_miles_mw": "Chabrier_MILES_metallicity_massW",
        "ff_elodie_mw": "Chabrier_ELODIE_metallicity_massW",
    }
    for short, raw in ff_defs.items():
        lo, hi = raw + "_low_1sig", raw + "_up_1sig"
        valid = finite(w, [lo, raw, hi]) & w[[lo, raw, hi]].gt(0).all(axis=1)
        valid &= (w[lo] <= w[raw]) & (w[raw] <= w[hi])
        w[f"{short}_valid"] = valid
        # FIREFLY reports linear Z/Z_sun; compare in log10 solar units.
        w[f"{short}_logz"] = np.log10(w[raw].where(valid))
        w[f"{short}_logz_lo"] = np.log10(w[lo].where(valid))
        w[f"{short}_logz_hi"] = np.log10(w[hi].where(valid))
        w[f"{short}_logz_unc"] = (w[f"{short}_logz_hi"] - w[f"{short}_logz_lo"]) / 2
        w[f"{short}_usable"] = valid & w["firefly_spectrum_quality"]
        w[f"{short}_precision03"] = w[f"{short}_usable"] & w[f"{short}_logz_unc"].le(0.3)

    assert len(w) == 6076, "parent row count changed"
    assert w["objid"].nunique() == 3857, "parent physical-object identity changed"
    assert not (w.filter(regex="_logz$").eq(-9999).any().any()), "sentinel escaped unit conversion"
    return w


def pooled_smd(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    if min(len(a), len(b)) < 2:
        return np.nan
    den = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / den) if den > 0 else np.nan


def coverage_and_selection(w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    definitions = [
        ("Gallazzi DR4", "gallazzi_match", "gallazzi_valid", "gallazzi_usable", "gallazzi_logz_unc"),
        ("FIREFLY DR16 MILES light-weighted", "firefly_match", "ff_miles_lw_valid", "ff_miles_lw_usable", "ff_miles_lw_logz_unc"),
    ]
    for catalogue, match, valid, usable, unc in definitions:
        for sample in SAMPLES:
            for role, flag in ROLES.items():
                d = w[w["sample"].eq(sample) & w[flag].eq(1)]
                u = d[d[usable]]
                rows.append({
                    "catalogue": catalogue,
                    "sample": sample,
                    "role": role,
                    "parent_n": len(d),
                    "match_n": int(d[match].sum()),
                    "valid_n": int(d[valid].sum()),
                    "usable_n": int(d[usable].sum()),
                    "retained_fraction": float(d[usable].mean()),
                    "median_z": float(u["z"].median()) if len(u) else np.nan,
                    "median_logmstar": float(u["lgm"].median()) if len(u) else np.nan,
                    "median_SN": float(u["SN"].median()) if len(u) else np.nan,
                    "median_logz_unc_dex": float(u[unc].median()) if len(u) else np.nan,
                })
    coverage = pd.DataFrame(rows)

    # Use one row per physical object for the catalogue selection-function
    # audit; repeated control labels must not create pseudo-replication.
    d = w.assign(_priority=w["sample"].map({"CG4": 0, "RG4": 1, "Control4B": 2, "Control4C": 3})).sort_values("_priority").drop_duplicates("objid")
    numeric = ["lgm", "z", "r_petro_dered", "SN", "fibre_light_fraction", "R50_kpc"]
    smd_rows = []
    rate_rows = []
    for catalogue, _, _, usable, _ in definitions:
        for variable in numeric:
            smd_rows.append({
                "catalogue": catalogue,
                "variable": variable,
                "usable_minus_not_smd": pooled_smd(d.loc[d[usable], variable], d.loc[~d[usable], variable]),
                "usable_median": d.loc[d[usable], variable].median(),
                "not_usable_median": d.loc[~d[usable], variable].median(),
            })
        for variable in ["morphology4", "sf_state"]:
            for level, part in d.groupby(variable, dropna=False):
                rate_rows.append({
                    "catalogue": catalogue,
                    "variable": variable,
                    "level": str(level),
                    "n": len(part),
                    "usable_n": int(part[usable].sum()),
                    "retained_fraction": float(part[usable].mean()),
                })
        # Quintiles provide a transparent non-parametric dependence audit.
        for variable in numeric:
            ok = d[variable].notna()
            bins = pd.qcut(d.loc[ok, variable], 5, duplicates="drop")
            for interval, part in d.loc[ok].groupby(bins, observed=True):
                rate_rows.append({
                    "catalogue": catalogue,
                    "variable": variable + "_quintile",
                    "level": str(interval),
                    "n": len(part),
                    "usable_n": int(part[usable].sum()),
                    "retained_fraction": float(part[usable].mean()),
                })
    return coverage, pd.DataFrame(smd_rows), pd.DataFrame(rate_rows)


def cr1_ols(y: np.ndarray, X: np.ndarray, names: list[str], clusters: np.ndarray) -> dict:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    cid = pd.factorize(clusters)[0]
    G = int(cid.max() + 1)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for group in range(G):
        score = X[cid == group].T @ resid[cid == group]
        meat += np.outer(score, score)
    n, k = X.shape
    correction = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1
    cov = bread @ meat @ bread * correction
    return {"beta": beta, "se": np.sqrt(np.maximum(np.diag(cov), 0)), "names": names, "resid": resid, "n_clusters": G}


def design_matrix(d: pd.DataFrame, degree: int, extras: list[str] | None = None, morphology: bool = False) -> tuple[np.ndarray, list[str]]:
    cols = {"mass_c": (d["lgm"] - 10.5).to_numpy(float)}
    if degree == 2:
        cols["mass_c2"] = cols["mass_c"] ** 2
    cols["is_CG4"] = d["is_CG4"].to_numpy(float)
    for extra in extras or []:
        values = d[extra].to_numpy(float)
        cols[extra] = values - np.nanmedian(values)
    if morphology:
        for level in ["Spiral", "Uncertain", "NoGZ"]:
            cols["morph_" + level] = d["morphology4"].eq(level).to_numpy(float)
    names = ["const"] + list(cols)
    return np.column_stack([np.ones(len(d))] + list(cols.values())), names


def cv_degree(control: pd.DataFrame, ycol: str) -> tuple[int, float, float]:
    """Choose quadratic only for >=2% group-fold RMSE improvement."""
    d = control.dropna(subset=[ycol, "lgm", "group_uid"]).copy()
    group_codes = pd.factorize(d["group_uid"])[0]
    fold = group_codes % 5
    rmses = {}
    for degree in [1, 2]:
        errors = []
        for k in range(5):
            train, test = d[fold != k], d[fold == k]
            if min(len(train), len(test)) < 5:
                continue
            mt = train["lgm"].to_numpy() - 10.5
            mm = test["lgm"].to_numpy() - 10.5
            Xt = np.column_stack([np.ones(len(train)), mt] + ([mt**2] if degree == 2 else []))
            Xv = np.column_stack([np.ones(len(test)), mm] + ([mm**2] if degree == 2 else []))
            b, *_ = np.linalg.lstsq(Xt, train[ycol].to_numpy(), rcond=None)
            errors.extend((test[ycol].to_numpy() - Xv @ b).tolist())
        rmses[degree] = float(np.sqrt(np.mean(np.square(errors))))
    improve = (rmses[1] - rmses[2]) / rmses[1]
    return (2 if improve >= 0.02 else 1), rmses[1], rmses[2]


def cluster_bootstrap(d: pd.DataFrame, ycol: str, degree: int, extras: list[str], morphology: bool, rng: np.random.Generator) -> np.ndarray:
    clusters = {}
    for idx, (sample, group) in enumerate(zip(d["sample"], d["group_uid"])):
        clusters.setdefault((sample, group), []).append(idx)
    by_sample: dict[str, list[tuple[str, str]]] = {}
    for key in clusters:
        by_sample.setdefault(key[0], []).append(key)
    out = []
    for _ in range(N_BOOT):
        rows: list[int] = []
        for keys in by_sample.values():
            for pick in rng.integers(0, len(keys), len(keys)):
                rows.extend(clusters[keys[pick]])
        b = d.iloc[rows]
        X, names = design_matrix(b, degree, extras, morphology)
        try:
            coef, *_ = np.linalg.lstsq(X, b[ycol].to_numpy(float), rcond=None)
            out.append(float(coef[names.index("is_CG4")]))
        except np.linalg.LinAlgError:
            continue
    return np.asarray(out)


def fit_contrast(
    frame: pd.DataFrame,
    ycol: str,
    usable: pd.Series,
    role: str,
    control: str,
    rng: np.random.Generator,
    subset: pd.Series | None = None,
    extras: list[str] | None = None,
    morphology: bool = False,
    degree: int | None = None,
) -> dict | None:
    extras = extras or []
    flag = ROLES[role]
    keep = usable & frame[flag].eq(1) & frame["sample"].isin(["CG4", control])
    if subset is not None:
        keep &= subset
    required = [ycol, "lgm", "group_uid"] + extras
    d = frame.loc[keep].dropna(subset=required).copy()
    d["is_CG4"] = d["sample"].eq("CG4").astype(int)
    n_cg, n_ctrl = int(d["is_CG4"].sum()), int((1 - d["is_CG4"]).sum())
    if min(n_cg, n_ctrl) < 10:
        return None
    if degree is None:
        degree, rmse1, rmse2 = cv_degree(d[d["sample"].eq(control)], ycol)
    else:
        _, rmse1, rmse2 = cv_degree(d[d["sample"].eq(control)], ycol)
    X, names = design_matrix(d, degree, extras, morphology)
    fit = cr1_ols(d[ycol].to_numpy(float), X, names, d["group_uid"].to_numpy())
    j = names.index("is_CG4")
    beta, se = float(fit["beta"][j]), float(fit["se"][j])
    boots = cluster_bootstrap(d, ycol, degree, extras, morphology, rng)
    ctrl_sd = float(np.std(fit["resid"][d["sample"].eq(control).to_numpy()], ddof=1))
    return {
        "n": len(d), "n_CG4": n_cg, "n_control": n_ctrl,
        "groups_CG4": int(d.loc[d["is_CG4"].eq(1), "group_uid"].nunique()),
        "groups_control": int(d.loc[d["is_CG4"].eq(0), "group_uid"].nunique()),
        "degree": degree, "control_cv_rmse_linear": rmse1, "control_cv_rmse_quadratic": rmse2,
        "beta": beta, "cluster_se": se, "standardized_effect": beta / ctrl_sd if ctrl_sd else np.nan,
        "ci68_lo": float(np.percentile(boots, 16)), "ci68_hi": float(np.percentile(boots, 84)),
        "ci95_lo": float(np.percentile(boots, 2.5)), "ci95_hi": float(np.percentile(boots, 97.5)),
        "n_boot": len(boots), "extras": "+".join(extras) if extras else "none",
    }


def science_tables(w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    total_rows = []
    for product, (ycol, usable_col) in PRODUCTS.items():
        for role in ROLES:
            for control in CONTROLS:
                result = fit_contrast(w, ycol, w[usable_col], role, control, rng)
                if result:
                    total_rows.append({"product": product, "role": role, "control": control, "subset": "total", **result})
    total = pd.DataFrame(total_rows)

    decomposition_rows = []
    for product in [PRIMARY, "Gallazzi optical light-weighted"]:
        ycol, usable_col = PRODUCTS[product]
        for role in ROLES:
            tests = [("total", pd.Series(True, index=w.index), [], False)]
            for level in ["Elliptical", "Spiral"]:
                tests.append(("morphology=" + level, w["morphology4"].eq(level), [], False))
            for level in ["Quenched", "Starforming"]:
                tests.append(("SF=" + level, w["sf_state"].eq(level), [], False))
            tests.extend([
                ("Elliptical + sigma", w["morphology4"].eq("Elliptical") & w["log_sigma"].notna(), ["log_sigma"], False),
                ("Quenched + sigma", w["sf_state"].eq("Quenched") & w["log_sigma"].notna(), ["log_sigma"], False),
            ])
            for label, subset, extras, morph in tests:
                result = fit_contrast(w, ycol, w[usable_col], role, "Control4C", rng, subset=subset, extras=extras, morphology=morph)
                if result:
                    decomposition_rows.append({"product": product, "role": role, "control": "Control4C", "subset": label, **result})
    decomposition = pd.DataFrame(decomposition_rows)

    # Robustness variants keep FIREFLY MILES light-weighted fixed.  The base
    # FIREFLY selection has no imported Gallazzi S/N threshold; cuts below are
    # labelled sensitivities, not alternative primaries.
    ycol, usable_col = PRODUCTS[PRIMARY]
    usable = w[usable_col]
    variants: list[tuple[str, pd.Series, list[str]]] = [
        ("base usable", usable, []),
        ("metallicity uncertainty <=0.3 dex", w["ff_miles_lw_precision03"], []),
        ("S/N>5", usable & w["SN"].gt(5), []),
        ("S/N>20", usable & w["SN"].gt(20), []),
        ("legacy RUN2D=26 only", usable & w["RUN2D"].astype(str).eq("26"), []),
        ("adjust redshift", usable, ["z"]),
        ("adjust fibre light fraction", usable & w["fibre_light_fraction"].notna(), ["fibre_light_fraction"]),
        ("adjust physical fibre radius", usable & w["fibre_radius_kpc"].notna(), ["fibre_radius_kpc"]),
        ("adjust galaxy R50", usable & w["R50_kpc"].notna(), ["R50_kpc"]),
        ("adjust z + fibre fraction + R50", usable & w[["z", "fibre_light_fraction", "R50_kpc"]].notna().all(axis=1), ["z", "fibre_light_fraction", "R50_kpc"]),
    ]
    robustness_rows = []
    for role in ROLES:
        flag = ROLES[role]
        pair = w[usable & w[flag].eq(1) & w["sample"].isin(["CG4", "Control4C"]) & w["lgm"].notna()]
        lower = max(pair.loc[pair["sample"].eq("CG4"), "lgm"].min(), pair.loc[pair["sample"].eq("Control4C"), "lgm"].min())
        upper = min(pair.loc[pair["sample"].eq("CG4"), "lgm"].max(), pair.loc[pair["sample"].eq("Control4C"), "lgm"].max())
        q01, q99 = pair["lgm"].quantile([.01, .99])
        role_variants = variants + [
            ("common CG4-control mass support", usable & w["lgm"].between(lower, upper), []),
            ("central 98% mass range", usable & w["lgm"].between(q01, q99), []),
        ]
        for label, mask, extras in role_variants:
            result = fit_contrast(w, ycol, mask, role, "Control4C", rng, extras=extras)
            if result:
                robustness_rows.append({"product": PRIMARY, "role": role, "control": "Control4C", "variant": label, **result})
    robustness = pd.DataFrame(robustness_rows)
    return total, decomposition, robustness


def cross_calibration(w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    priority = {"CG4": 0, "RG4": 1, "Control4B": 2, "Control4C": 3}
    d = w.assign(_priority=w["sample"].map(priority)).sort_values("_priority").drop_duplicates("objid")
    overlap = d[d["gallazzi_usable"] & d["ff_miles_lw_usable"]].copy()
    overlap["g_minus_f"] = overlap["gallazzi_logz_solar"] - overlap["ff_miles_lw_logz"]
    centre = float(overlap["g_minus_f"].median())
    scatter = float(1.4826 * np.median(np.abs(overlap["g_minus_f"] - centre)))
    threshold = max(0.5, 3 * scatter)
    overlap["catastrophic_disagreement"] = (overlap["g_minus_f"] - centre).abs().gt(threshold)

    rows = [{
        "comparison": "Gallazzi minus FIREFLY MILES light-weighted",
        "stratum": "all overlap (Gallazzi S/N>=20)", "n": len(overlap),
        "median_offset_dex": centre, "robust_scatter_dex": scatter,
        "catastrophic_threshold_dex_from_median": threshold,
        "catastrophic_n": int(overlap["catastrophic_disagreement"].sum()),
    }]
    for variable in ["lgm", "z", "SN", "fibre_light_fraction"]:
        q = overlap[[variable, "g_minus_f"]].dropna()
        slope, intercept, lo, hi = stats.theilslopes(q["g_minus_f"], q[variable], 0.95)
        rows.append({
            "comparison": "Gallazzi minus FIREFLY MILES light-weighted",
            "stratum": "trend versus " + variable, "n": len(q),
            "theil_sen_slope": slope, "slope_ci95_lo": lo, "slope_ci95_hi": hi,
        })
    for variable in ["morphology4", "sf_state"]:
        for level, part in overlap.groupby(variable):
            rows.append({
                "comparison": "Gallazzi minus FIREFLY MILES light-weighted",
                "stratum": f"{variable}={level}", "n": len(part),
                "median_offset_dex": float(part["g_minus_f"].median()),
                "robust_scatter_dex": float(1.4826 * np.median(np.abs(part["g_minus_f"] - part["g_minus_f"].median()))),
            })

    # FIREFLY model-systematic checks use the same valid galaxies and retain
    # each model exactly once; no variant is chosen by its CG4 coefficient.
    comparisons = [
        ("MILES lw minus ELODIE lw", "ff_miles_lw_logz", "ff_elodie_lw_logz", "ff_miles_lw_usable", "ff_elodie_lw_usable"),
        ("MILES mw minus MILES lw", "ff_miles_mw_logz", "ff_miles_lw_logz", "ff_miles_mw_usable", "ff_miles_lw_usable"),
        ("ELODIE mw minus ELODIE lw", "ff_elodie_mw_logz", "ff_elodie_lw_logz", "ff_elodie_mw_usable", "ff_elodie_lw_usable"),
    ]
    for label, a, b, va, vb in comparisons:
        q = d[d[va] & d[vb]].copy()
        diff = q[a] - q[b]
        med = float(diff.median())
        rows.append({
            "comparison": label, "stratum": "all", "n": len(q),
            "median_offset_dex": med,
            "robust_scatter_dex": float(1.4826 * np.median(np.abs(diff - med))),
        })
        for run2d, part in q.groupby(q["RUN2D"].astype(str)):
            x = part[a] - part[b]
            med = float(x.median())
            rows.append({
                "comparison": label, "stratum": "RUN2D=" + run2d, "n": len(part),
                "median_offset_dex": med,
                "robust_scatter_dex": float(1.4826 * np.median(np.abs(x - med))),
            })
    outliers = overlap.loc[overlap["catastrophic_disagreement"], [
        "objid", "plate", "mjd", "fiberID", "sample", "role", "lgm", "z", "SN",
        "morphology4", "sf_state", "fibre_light_fraction", "gallazzi_logz_solar",
        "ff_miles_lw_logz", "g_minus_f",
    ]].sort_values("g_minus_f", key=lambda x: (x - centre).abs(), ascending=False)
    return pd.DataFrame(rows), outliers


def index_sensitivity(w: pd.DataFrame) -> pd.DataFrame:
    indices = {
        "Mgb": "Mgb_err", "Fe5270": "Fe5270_err", "Fe5335": "Fe5335_err",
        "[MgFe]'": "MgFe_err", "Mg2": "Mg2_err", "Dn4000": "D4000n_err",
        "Hdelta_A": "HdA_sub_err",
    }
    ymap = {"[MgFe]'": "MgFe", "Dn4000": "D4000n", "Hdelta_A": "HdA_sub"}
    rows = []
    for control in CONTROLS:
        base = w[w["is_sat"].eq(1) & w["sample"].isin(["CG4", control]) & w["valid_core"].eq(1) & w["sigma"].gt(0)].copy()
        base["is_CG4"] = base["sample"].eq("CG4").astype(int)
        for label, errcol in indices.items():
            ycol = ymap.get(label, label)
            d = base[base[errcol].gt(0)].dropna(subset=[ycol, "log_sigma", "group_uid"])
            for spec, mass_required in [("Claude-like", False), ("Claude-like, mass-complete", True), ("+ stellar mass", True)]:
                dd = d.dropna(subset=["lgm"]) if mass_required else d
                if min(dd["is_CG4"].sum(), (1 - dd["is_CG4"]).sum()) < 10:
                    continue
                m = dd["log_sigma"].to_numpy(float)
                cols = {
                    "is_CG4": dd["is_CG4"].to_numpy(float),
                    "log_sigma": m,
                    "log_sigma2": m**2,
                }
                for morph in ["Spiral", "Uncertain", "NoGZ"]:
                    cols["morph_" + morph] = dd["morphology4"].eq(morph).to_numpy(float)
                if spec == "+ stellar mass":
                    cols["lgm"] = dd["lgm"].to_numpy(float) - 10.5
                names = ["const"] + list(cols)
                X = np.column_stack([np.ones(len(dd))] + list(cols.values()))
                fit = cr1_ols(dd[ycol].to_numpy(float), X, names, dd["group_uid"].to_numpy())
                j = names.index("is_CG4")
                beta, se = float(fit["beta"][j]), float(fit["se"][j])
                rows.append({
                    "control": control, "index": label, "specification": spec,
                    "n": len(dd), "n_CG4": int(dd["is_CG4"].sum()), "n_clusters": fit["n_clusters"],
                    "beta_CG4": beta, "cluster_se": se,
                    "ci68_lo": beta - se, "ci68_hi": beta + se,
                    "ci95_lo": beta - 1.96 * se, "ci95_hi": beta + 1.96 * se,
                })
    return pd.DataFrame(rows)


def control_residuals(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for product in [PRIMARY, "Gallazzi optical light-weighted"]:
        ycol, usable_col = PRODUCTS[product]
        for role, flag in ROLES.items():
            d = w[w[usable_col] & w[flag].eq(1) & w["sample"].isin(["CG4", "Control4C"])].dropna(subset=[ycol, "lgm"]).copy()
            control = d[d["sample"].eq("Control4C")]
            degree, rmse1, rmse2 = cv_degree(control, ycol)
            m = control["lgm"].to_numpy() - 10.5
            X = np.column_stack([np.ones(len(control)), m] + ([m**2] if degree == 2 else []))
            beta, *_ = np.linalg.lstsq(X, control[ycol].to_numpy(), rcond=None)
            md = d["lgm"].to_numpy() - 10.5
            Xd = np.column_stack([np.ones(len(d)), md] + ([md**2] if degree == 2 else []))
            d["predicted_control_logz"] = Xd @ beta
            d["delta_logz"] = d[ycol] - d["predicted_control_logz"]
            for row in d[["objid", "sample", "group_uid", "role", "lgm", "z", ycol, "predicted_control_logz", "delta_logz"]].to_dict("records"):
                rows.append({"product": product, "control": "Control4C", "degree": degree, "control_cv_rmse_linear": rmse1, "control_cv_rmse_quadratic": rmse2, **row})
    return pd.DataFrame(rows)


def plot_coverage(coverage: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    order = [f"{s}\n{r}" for s in SAMPLES for r in ["BGG", "Satellite"]]
    for ax, (catalogue, d) in zip(axes, coverage.groupby("catalogue", sort=False)):
        d = d.assign(label=d["sample"] + "\n" + d["role"]).set_index("label").loc[order].reset_index()
        x = np.arange(len(d))
        ax.bar(x - 0.22, d["match_n"] / d["parent_n"], width=.22, label="matched", color="#9ecae1")
        ax.bar(x, d["valid_n"] / d["parent_n"], width=.22, label="valid Z*", color="#4292c6")
        ax.bar(x + 0.22, d["usable_n"] / d["parent_n"], width=.22, label="usable", color="#08519c")
        for i, row in d.iterrows():
            ax.text(i + .22, row["usable_n"] / row["parent_n"] + .025, str(int(row["usable_n"])), ha="center", fontsize=8)
        ax.set_ylim(0, 1.12); ax.set_ylabel("fraction of parent"); ax.set_title(catalogue); ax.grid(axis="y", alpha=.25)
    axes[0].legend(ncol=3, frameon=False)
    axes[-1].set_xticks(np.arange(len(order)), order)
    fig.suptitle("Direct stellar-metallicity catalogue coverage (numbers label usable N)")
    fig.tight_layout(); fig.savefig(FIG / "coverage.png", dpi=160); plt.close(fig)


def plot_selection(rates: pd.DataFrame) -> None:
    variables = ["lgm", "z", "r_petro_dered", "SN", "fibre_light_fraction", "R50_kpc"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, variable in zip(axes.ravel(), variables):
        d = rates[rates["variable"].eq(variable + "_quintile")]
        for catalogue, part in d.groupby("catalogue", sort=False):
            ax.plot(np.arange(1, len(part) + 1), part["retained_fraction"], marker="o", label=catalogue.split()[0])
        ax.set_title(variable); ax.set_xlabel("parent quintile (low to high)"); ax.set_ylabel("usable fraction"); ax.set_ylim(0, 1.02); ax.grid(alpha=.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Catalogue selection functions (one row per physical galaxy)")
    fig.tight_layout(); fig.savefig(FIG / "selection_functions.png", dpi=160); plt.close(fig)


def plot_crosscal(w: pd.DataFrame, summary: pd.DataFrame) -> None:
    d = w.assign(_p=w["sample"].map({"CG4": 0, "RG4": 1, "Control4B": 2, "Control4C": 3})).sort_values("_p").drop_duplicates("objid")
    d = d[d["gallazzi_usable"] & d["ff_miles_lw_usable"]].copy()
    d["diff"] = d["gallazzi_logz_solar"] - d["ff_miles_lw_logz"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].scatter(d["ff_miles_lw_logz"], d["gallazzi_logz_solar"], s=8, alpha=.35)
    lim = [-1.5, .6]; axes[0, 0].plot(lim, lim, color="k", lw=1); axes[0, 0].set(xlim=lim, ylim=lim, xlabel="FIREFLY MILES lightW [Z/H]", ylabel="Gallazzi [Z/H]")
    for ax, x, label in [(axes[0, 1], "lgm", "log stellar mass"), (axes[1, 0], "SN", "S/N"), (axes[1, 1], "fibre_light_fraction", "fibre/Petrosian flux")]:
        ax.scatter(d[x], d["diff"], s=8, alpha=.3); ax.axhline(d["diff"].median(), color="#A74752", lw=1.5); ax.axhline(0, color="k", lw=.8, ls="--")
        ax.set(xlabel=label, ylabel="Gallazzi − FIREFLY (dex)")
    for ax in axes.ravel(): ax.grid(alpha=.25)
    fig.suptitle("Gallazzi–FIREFLY cross-calibration, common high-S/N sample")
    fig.tight_layout(); fig.savefig(FIG / "cross_calibration.png", dpi=160); plt.close(fig)


def plot_mzr(w: pd.DataFrame, residuals: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex="col")
    for row, product in enumerate([PRIMARY, "Gallazzi optical light-weighted"]):
        ycol, usable = PRODUCTS[product]
        for col, (role, flag) in enumerate(ROLES.items()):
            ax = axes[row, col]
            d = w[w[usable] & w[flag].eq(1) & w["sample"].isin(["CG4", "Control4C"])].dropna(subset=[ycol, "lgm"])
            for sample in ["Control4C", "CG4"]:
                q = d[d["sample"].eq(sample)]
                ax.scatter(q["lgm"], q[ycol], s=10 if sample == "CG4" else 5, alpha=.55 if sample == "CG4" else .18, color=PALETTE[sample], label=sample)
            rr = residuals[(residuals["product"].eq(product)) & residuals["role"].eq(role)]
            degree = int(rr["degree"].iloc[0]); control = rr[rr["sample"].eq("Control4C")]
            grid = np.linspace(d["lgm"].quantile(.01), d["lgm"].quantile(.99), 100)
            # Recover the plotted control fit directly from saved predictions.
            m = control["lgm"].to_numpy() - 10.5
            X = np.column_stack([np.ones(len(control)), m] + ([m**2] if degree == 2 else []))
            b, *_ = np.linalg.lstsq(X, control["predicted_control_logz"].to_numpy(), rcond=None)
            mg = grid - 10.5; Xg = np.column_stack([np.ones(len(grid)), mg] + ([mg**2] if degree == 2 else []))
            ax.plot(grid, Xg @ b, color=PALETTE["Control4C"], lw=2, label=f"C4C degree {degree}")
            ax.set_xlim(d["lgm"].quantile(.005), d["lgm"].quantile(.995))
            ax.set_title(f"{product.replace('FIREFLY ', '')}: {role}"); ax.set_ylabel("stellar [Z/H] (dex)"); ax.grid(alpha=.25)
    axes[-1, 0].set_xlabel("log stellar mass"); axes[-1, 1].set_xlabel("log stellar mass"); axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Stellar mass–metallicity relations; Control4C defines the reference curve")
    fig.tight_layout(); fig.savefig(FIG / "mass_metallicity.png", dpi=160); plt.close(fig)


def forest(ax, d: pd.DataFrame, label_col: str, title: str) -> None:
    d = d.reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    ax.hlines(y, d["ci95_lo"], d["ci95_hi"], color="0.7", lw=2)
    ax.hlines(y, d["ci68_lo"], d["ci68_hi"], color="#2864A6", lw=5)
    ax.plot(d["beta"], y, "ko", ms=4); ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(y, d[label_col]); ax.set_title(title); ax.set_xlabel("CG4 − control at fixed mass (dex)"); ax.grid(axis="x", alpha=.25)


def plot_effects(total: pd.DataFrame, decomposition: pd.DataFrame, robustness: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, role in zip(axes, ROLES):
        d = total[(total["role"].eq(role)) & total["product"].isin([PRIMARY, "Gallazzi optical light-weighted"])] .copy()
        d["label"] = d["product"].str.replace("FIREFLY ", "", regex=False).str.replace(" optical light-weighted", "", regex=False) + " vs " + d["control"]
        forest(ax, d, "label", role)
    fig.suptitle("Mass-adjusted direct stellar-metallicity contrasts")
    fig.tight_layout(); fig.savefig(FIG / "direct_effects.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, role in zip(axes, ROLES):
        d = decomposition[(decomposition["product"].eq(PRIMARY)) & decomposition["role"].eq(role)].copy()
        forest(ax, d, "subset", role)
    fig.suptitle("FIREFLY MILES lightW decomposition; conditioned rows are sensitivities")
    fig.tight_layout(); fig.savefig(FIG / "decomposition.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, role in zip(axes, ROLES):
        d = robustness[robustness["role"].eq(role)].copy()
        forest(ax, d, "variant", role)
    fig.suptitle("Aperture, redshift, S/N, precision, and pipeline robustness")
    fig.tight_layout(); fig.savefig(FIG / "robustness.png", dpi=160); plt.close(fig)


def plot_index_sensitivity(index: pd.DataFrame) -> None:
    d = index[index["control"].eq("Control4C") & index["specification"].isin(["Claude-like, mass-complete", "+ stellar mass"])].copy()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7)); axes = axes.ravel()
    for ax, (name, part) in zip(axes, d.groupby("index", sort=False)):
        part = part.set_index("specification").loc[["Claude-like, mass-complete", "+ stellar mass"]].reset_index()
        y = [1, 0]
        ax.errorbar(part["beta_CG4"], y, xerr=1.96 * part["cluster_se"], fmt="o", capsize=3)
        ax.axvline(0, color="k", lw=.8); ax.set_yticks(y, ["Claude-like", "+ M*"]); ax.set_title(name); ax.grid(axis="x", alpha=.25)
    axes[-1].axis("off")
    fig.suptitle("Index-level CG4 coefficient before and after stellar-mass adjustment (satellites vs C4C; 95% CR1 intervals)")
    fig.tight_layout(); fig.savefig(FIG / "index_mass_sensitivity.png", dpi=160); plt.close(fig)


def plot_uncertainty_sn(w: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    configs = [
        ("FIREFLY MILES lightW", "ff_miles_lw_usable", "ff_miles_lw_logz_unc"),
        ("Gallazzi", "gallazzi_valid", "gallazzi_logz_unc"),
    ]
    for ax, (title, valid, unc) in zip(axes, configs):
        d = w[w[valid] & w["SN"].gt(0)].drop_duplicates("objid")
        ax.scatter(d["SN"], d[unc], s=7, alpha=.2)
        bins = pd.qcut(d["SN"], 8, duplicates="drop")
        med = d.groupby(bins, observed=True).agg(SN=("SN", "median"), unc=(unc, "median"))
        ax.plot(med["SN"], med["unc"], "o-", color="#A74752", lw=2)
        ax.axvline(20, color="k", ls="--", lw=.8)
        ax.set(xlabel="spectrum S/N", ylabel="half-width of 68% [Z/H] interval (dex)", title=title, xscale="log")
        ax.grid(alpha=.25)
    fig.suptitle("Reported metallicity precision versus spectrum quality")
    fig.tight_layout(); fig.savefig(FIG / "uncertainty_vs_sn.png", dpi=160); plt.close(fig)


def md_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    labels = columns
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join(["---"] * len(labels)) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                vals.append("–")
            elif col in formats:
                vals.append(format(value, formats[col]))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def effect_text(row: pd.Series) -> str:
    return f"{row.beta:+.3f} dex (95% group-bootstrap CI {row.ci95_lo:+.3f} to {row.ci95_hi:+.3f}; N={int(row.n_CG4)}+{int(row.n_control)})"


def build_report(
    w: pd.DataFrame,
    coverage: pd.DataFrame,
    smd: pd.DataFrame,
    rates: pd.DataFrame,
    crosscal: pd.DataFrame,
    outliers: pd.DataFrame,
    total: pd.DataFrame,
    decomposition: pd.DataFrame,
    robustness: pd.DataFrame,
    index: pd.DataFrame,
) -> None:
    primary = total[(total["product"].eq(PRIMARY)) & total["control"].eq("Control4C")].set_index("role")
    gallazzi = total[(total["product"].eq("Gallazzi optical light-weighted")) & total["control"].eq("Control4C")].set_index("role")
    cc = crosscal[(crosscal["comparison"].eq("Gallazzi minus FIREFLY MILES light-weighted")) & crosscal["stratum"].str.startswith("all overlap")].iloc[0]
    run2d = json.loads((CAT / "catalogue_manifest.json").read_text())["firefly"]["returned_run2d"]

    direct_table = total[total["product"].isin([PRIMARY, "Gallazzi optical light-weighted"])].copy()
    direct_table["estimate"] = direct_table.apply(effect_text, axis=1)
    dec = decomposition[decomposition["product"].eq(PRIMARY)].copy()
    dec["estimate"] = dec.apply(effect_text, axis=1)
    robust_ranges = robustness.groupby("role").agg(beta_min=("beta", "min"), beta_max=("beta", "max"), ci95_min=("ci95_lo", "min"), ci95_max=("ci95_hi", "max")).reset_index()

    idx = index[index["control"].eq("Control4C") & index["specification"].isin(["Claude-like, mass-complete", "+ stellar mass"])]
    piv = idx.pivot(index="index", columns="specification", values="beta_CG4").reset_index()
    piv["shift_after_mass"] = piv["+ stellar mass"] - piv["Claude-like, mass-complete"]

    strongest = smd.assign(abs_smd=smd["usable_minus_not_smd"].abs()).sort_values("abs_smd", ascending=False).groupby("catalogue", sort=False).head(3)
    ff_systematics = total[total["control"].eq("Control4C") & total["product"].str.startswith("FIREFLY")].copy()
    ff_systematics["estimate"] = ff_systematics.apply(effect_text, axis=1)
    categorical = rates[rates["variable"].isin(["morphology4", "sf_state"])]
    def retained(catalogue: str, variable: str, level: str) -> float:
        return float(categorical[(categorical["catalogue"].eq(catalogue)) & categorical["variable"].eq(variable) & categorical["level"].eq(level)]["retained_fraction"].iloc[0])
    text = rf"""# Direct stellar metallicity extension

## Bottom line

This is an exploratory extension of Claude's immutable [index-level feasibility study](REPORT.md), not a replacement for it. FIREFLY provides the decisive coverage advantage and is therefore the primary direct-$Z_\star$ catalogue. After stellar-mass adjustment against the project-defined Control4C analogue, the FIREFLY MILES light-weighted contrast is **{effect_text(primary.loc['Satellite'])} for satellites** and **{effect_text(primary.loc['BGG'])} for BGGs**. The independent Gallazzi estimates are **{effect_text(gallazzi.loc['Satellite'])}** and **{effect_text(gallazzi.loc['BGG'])}**, respectively. These are effect estimates with uncertainty, not detections selected by a significance threshold.

The object-level catalogues are not interchangeable: among {int(cc['n'])} common high-S/N objects, Gallazzi minus FIREFLY has median {cc['median_offset_dex']:+.3f} dex and robust scatter {cc['robust_scatter_dex']:.3f} dex. They are nevertheless useful as independent environmental checks because the comparison uses the same product and model within each catalogue.

## Acquisition and definitions

- Spectrum identity is plate–MJD–fibre throughout. FIREFLY `SPECOBJID` is deliberately ignored because the DR16 VAC documentation warns that it is corrupted.
- The combined SDSS/eBOSS DR16 VAC is the published product. Its legacy SDSS `RUN2D=26` spectra were fit with FIREFLY v1.1.0; the BOSS/eBOSS `RUN2D=v5_13_0` spectra used v1.1.1. The matched split is {run2d.get('26', 0)} + {run2d.get('v5_13_0', 0)} = {sum(run2d.values())}.
- Gallazzi DR4 stores $\log_{{10}} Z$ for absolute metal mass fraction $Z$. Following its catalogue page, this analysis subtracts $\log_{{10}}(0.02)$ to obtain $[Z/H]=\log_{{10}}(Z/Z_\odot)$. The model grid uses BC03/STELIB and interprets the single-$Z$ model as optical-light-weighted metallicity.
- FIREFLY quantities such as `Chabrier_MILES_metallicity_lightW` are linear solar metallicities ($Z/Z_\odot$), so they are transformed with $\log_{{10}}$. MILES light-weighted is primary: it matches Gallazzi's optical weighting most closely and MILES has broader wavelength coverage than ELODIE. ELODIE and both mass-weighted products are model-systematic checks fixed in advance.
- Gallazzi `-99` and FIREFLY `-9999` are invalid. Physical positivity, ordered 68% bounds, `CLASS_NOQSO=GALAXY`, and $Z>Z_{{err}}>0$ are enforced before use.
- Gallazzi's published S/N≥20 reliability result defines its usable sample. FIREFLY receives no automatic S/N=20 cut; valid galaxy/redshift/posterior rows are primary, with uncertainty≤0.3 dex and S/N cuts shown only as sensitivities.

Authoritative local snapshots: [Gallazzi catalogue documentation](work/catalogues/gallazzi/stellarmet.html), [Gallazzi et al. 2005](work/catalogues/gallazzi/Gallazzi2005_arxiv.pdf), [FIREFLY DR16 VAC](work/catalogues/firefly/sdss_dr16_firefly.html), [FIREFLY data model](work/catalogues/firefly/sdss_eboss_firefly-DR16_datamodel.html), and [Comparat et al. 2019](work/catalogues/firefly/Comparat2019_arxiv.pdf).

## Coverage audit

"""
    cov = coverage.copy()
    cov["retained_%"] = 100 * cov["retained_fraction"]
    text += md_table(cov, ["catalogue", "sample", "role", "parent_n", "match_n", "valid_n", "usable_n", "retained_%", "median_z", "median_logmstar", "median_SN", "median_logz_unc_dex"], {"retained_%": ".1f", "median_z": ".4f", "median_logmstar": ".2f", "median_SN": ".1f", "median_logz_unc_dex": ".3f"})
    text += "\n\n`match_n` is identity coverage; `valid_n` removes sentinels/non-physical posteriors; `usable_n` additionally applies the catalogue-specific rules above. Parent N is never silently redefined.\n\n"
    text += "The largest continuous usable-versus-not-usable imbalances (standardised mean difference; positive means larger among usable objects) are:\n\n"
    text += md_table(strongest, ["catalogue", "variable", "usable_minus_not_smd", "usable_median", "not_usable_median"], {"usable_minus_not_smd": "+.2f", "usable_median": ".3f", "not_usable_median": ".3f"})
    text += f"\n\nGallazzi retention is materially composition-dependent: {100*retained('Gallazzi DR4', 'morphology4', 'Elliptical'):.1f}% for ellipticals versus {100*retained('Gallazzi DR4', 'morphology4', 'Spiral'):.1f}% for spirals, and {100*retained('Gallazzi DR4', 'sf_state', 'Quenched'):.1f}% for quenched versus {100*retained('Gallazzi DR4', 'sf_state', 'Starforming'):.1f}% for star-forming galaxies. FIREFLY is {100*retained('FIREFLY DR16 MILES light-weighted', 'morphology4', 'Elliptical'):.1f}% complete for ellipticals and {100*retained('FIREFLY DR16 MILES light-weighted', 'morphology4', 'Spiral'):.1f}% for spirals, but falls to {100*retained('FIREFLY DR16 MILES light-weighted', 'morphology4', 'NoGZ'):.1f}% for objects without a GZ class and {100*retained('FIREFLY DR16 MILES light-weighted', 'sf_state', 'NosSFR'):.1f}% without a parent-catalogue SF state. Full rates are in `outputs/zstar/selection_rates.csv`; the non-parametric quintile curves in `figures/zstar/selection_functions.png` expose mass, redshift, magnitude, S/N, fibre-fraction, and size dependence without treating duplicated control labels as independent galaxies.\n\n"

    text += "## Cross-calibration\n\n"
    text += md_table(crosscal, ["comparison", "stratum", "n", "median_offset_dex", "robust_scatter_dex", "theil_sen_slope", "slope_ci95_lo", "slope_ci95_hi", "catastrophic_n"], {"median_offset_dex": "+.3f", "robust_scatter_dex": ".3f", "theil_sen_slope": "+.3f", "slope_ci95_lo": "+.3f", "slope_ci95_hi": "+.3f"})
    text += f"\n\nA catastrophic disagreement is defined before inspection as an absolute residual from the median larger than max(0.5 dex, three robust scatters); {len(outliers)} objects meet it and are listed in `crosscal_outliers.csv`. The MILES/ELODIE and light-/mass-weighted offsets show that stellar-library and weighting choice are material model systematics; they are not averaged away.\n\n"

    text += r"## Mass-adjusted environmental contrasts" + "\n\n" + r"The model is $[Z/H]=f(\log M_\star)+\beta_{CG}I(CG4)+\epsilon$, fit separately to satellites and BGGs. A linear Control4C relation is the default; a quadratic is used only when five-fold group CV reduces RMSE by at least 2%. Intervals come from 1,000 cluster-pairs bootstrap replicates, resampled within sample; the reported CR1 SE and standardised effect are retained in the CSV." + "\n\n"
    text += md_table(direct_table, ["product", "role", "control", "degree", "estimate"])
    text += "\n\nControl4C is primary because the repository defines it as the BGG plus the three nearest projected eligible companions—the closest ordinary-group analogue to a compact core. Control4B and RG4 answer secondary bright-member and equal-membership questions and are shown without pooling duplicated physical control groups.\n\n"
    text += "The pre-fixed FIREFLY model-systematic checks against Control4C are:\n\n"
    text += md_table(ff_systematics, ["product", "role", "degree", "estimate"])
    text += "\n\nTheir intervals overlap the primary result; the mass-weighted satellite point estimates are somewhat more positive but less precise. This is reported as model dependence, not a preferred effect.\n\n"

    text += "## Morphology and star-formation decomposition\n\nThe first row for each role is the total mass-adjusted contrast. The conditioned rows describe where the total signal sits; they are not automatically more causal, because morphology and SF state may be environmental outcomes. Sigma enters only alongside stellar mass in the early/passive sensitivity rows.\n\n"
    text += md_table(dec, ["role", "subset", "degree", "estimate"])
    text += "\n\nThe star-forming BGG row is the one conditioned estimate whose 95% interval excludes zero, but it contains only 11 CG4 BGGs, was inspected within a hierarchy of exploratory subsets, and is not supported by the total or quenched BGG contrasts. It is a follow-up lead, not standalone evidence.\n"

    text += "\n\n## Aperture, redshift, S/N, and model robustness\n\n"
    text += md_table(robust_ranges, ["role", "beta_min", "beta_max", "ci95_min", "ci95_max"], {"beta_min": "+.3f", "beta_max": "+.3f", "ci95_min": "+.3f", "ci95_max": "+.3f"})
    text += "\n\nThe full variant-by-variant table is `zstar_robustness.csv`, including common CG4–control mass support and central-98%-mass checks. Fibre flux fraction is computed exactly as $10^{-0.4(m_{fiber,r}-m_{Petro,r})}$; physical fibre radius is 1.5 arcsec times the existing Planck15 scale; galaxy size uses Simard circularised R50 with Petrosian R50 fallback. The base FIREFLY result is not inverse-variance weighted, preventing formally tiny catalogue errors or zero-width grid posteriors from dominating. The MZR figure shows the central 99% mass span for legibility; fits use the full stated samples.\n\n"

    text += "## Claude index sensitivity: adding stellar mass\n\nThe historical files are untouched. To isolate mass omission, the fair comparison below uses exactly the same mass-complete satellite rows under both specifications. The Claude-like model is index ~ CG4 + broad morphology + log sigma + (log sigma)^2; the second adds log stellar mass.\n\n"
    text += md_table(piv, ["index", "Claude-like, mass-complete", "+ stellar mass", "shift_after_mass"], {"Claude-like, mass-complete": "+.3f", "+ stellar mass": "+.3f", "shift_after_mass": "+.3f"})
    text += "\n\n" + r"These indices remain age/abundance-sensitive observables, not direct metallicity estimates. The table quantifies specification sensitivity and does not retroactively promote any index contrast to a $Z_\star$ measurement." + "\n\n"

    text += "## Figures and reproducibility\n\n- `coverage.png`: eight requested sample/role strata for both catalogues.\n- `selection_functions.png` and `uncertainty_vs_sn.png`: availability and quality diagnostics.\n- `cross_calibration.png`: object-level pipeline comparison.\n- `mass_metallicity.png`: Control4C reference MZR for satellites and BGGs.\n- `direct_effects.png`, `decomposition.png`, and `robustness.png`: 68%/95% bootstrap intervals.\n- `index_mass_sensitivity.png`: historical-index coefficient shift after adding mass.\n\nRun `zstar_fetch.py` (idempotent; existing downloads/query caches are reused), then `zstar_analysis.py`, then `build_zstar_notebook.py`. The notebook reads the generated products and can execute without network acquisition. All paths and outputs remain inside `exploration/metallicity/`.\n\n"
    text += "## Limits\n\nThis is a targeted feasibility extension, not a preregistered confirmatory test. Catalogue availability is selection-dependent, fibre spectra probe galaxy centres, Gallazzi and FIREFLY have different SSP machinery and weighting conventions, FIREFLY model grids can yield boundary/zero-width intervals, and repeated ordinary-group labels are never valid independent replicates. Reported comparisons are conditional on the parent catalogue, chosen spectra, available stellar masses, and the stated quality rules.\n"
    (HERE / "ZSTAR_REPORT.md").write_text(text)


def write_analysis_manifest(paths: list[Path]) -> None:
    records = []
    for path in paths:
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"path": str(path.relative_to(HERE)), "bytes": path.stat().st_size, "sha256": h})
    doc = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED, "cluster_bootstrap_replicates": N_BOOT,
        "scope": "All writes are below exploration/metallicity",
        "primary_product": PRIMARY, "primary_control": "Control4C",
        "files": records,
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(doc, indent=2))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    acquisition = json.loads((CAT / "catalogue_manifest.json").read_text())
    run2d = acquisition["firefly"]["returned_run2d"]
    assert sum(run2d.values()) == acquisition["firefly"]["matched_unique_pmf"] == 3835
    assert acquisition["sdss_aux"]["matched_unique_objid"] == acquisition["sdss_aux"]["target_unique_objid"] == 3857

    w = prepare_worktable()
    coverage, smd, rates = coverage_and_selection(w)
    crosscal, outliers = cross_calibration(w)
    total, decomposition, robustness = science_tables(w)
    index = index_sensitivity(w)
    residuals = control_residuals(w)

    keep = [
        "sample", "group_uid", "Group", "objid", "plate", "mjd", "fiberID", "rank_M", "role", "is_bgg", "is_sat",
        "z", "lgm", "M_r", "r_petro_dered", "SN", "morphology4", "sf_state", "sigma", "log_sigma",
        "fibre_light_fraction", "fibre_radius_kpc", "R50_arcsec", "R50_kpc", "fibre_radius_over_R50", "RUN2D",
        "gallazzi_match", "gallazzi_valid", "gallazzi_usable", "gallazzi_logz_solar", "gallazzi_logz_lo", "gallazzi_logz_hi", "gallazzi_logz_unc",
        "firefly_match", "firefly_spectrum_quality", "ff_miles_lw_valid", "ff_miles_lw_usable", "ff_miles_lw_precision03",
        "ff_miles_lw_logz", "ff_miles_lw_logz_lo", "ff_miles_lw_logz_hi", "ff_miles_lw_logz_unc",
        "ff_elodie_lw_logz", "ff_elodie_lw_usable", "ff_miles_mw_logz", "ff_miles_mw_usable", "ff_elodie_mw_logz", "ff_elodie_mw_usable",
        "valid_core", "Mgb", "Mgb_err", "Fe5270", "Fe5270_err", "Fe5335", "Fe5335_err", "MgFe", "MgFe_err", "Mg2", "Mg2_err", "D4000n", "D4000n_err", "HdA_sub", "HdA_sub_err",
    ]
    tables = {
        "zstar_worktable.csv": w[keep], "coverage.csv": coverage, "selection_smd.csv": smd,
        "selection_rates.csv": rates, "crosscal_summary.csv": crosscal, "crosscal_outliers.csv": outliers,
        "zstar_contrasts.csv": total, "zstar_decomposition.csv": decomposition,
        "zstar_robustness.csv": robustness, "index_mass_sensitivity.csv": index,
        "control_mzr_residuals.csv": residuals,
    }
    paths = []
    for name, frame in tables.items():
        path = OUT / name; frame.to_csv(path, index=False); paths.append(path)

    plot_coverage(coverage); plot_selection(rates); plot_uncertainty_sn(w)
    plot_crosscal(w, crosscal); plot_mzr(w, residuals)
    plot_effects(total, decomposition, robustness); plot_index_sensitivity(index)
    build_report(w, coverage, smd, rates, crosscal, outliers, total, decomposition, robustness, index)
    paths.extend(sorted(FIG.glob("*.png"))); paths.append(HERE / "ZSTAR_REPORT.md")
    write_analysis_manifest(paths)
    print(json.dumps({
        "rows": len(w), "coverage_rows": len(coverage), "crosscal_overlap": int(crosscal.iloc[0]["n"]),
        "primary_satellite": effect_text(total[(total["product"].eq(PRIMARY)) & total["role"].eq("Satellite") & total["control"].eq("Control4C")].iloc[0]),
        "primary_BGG": effect_text(total[(total["product"].eq(PRIMARY)) & total["role"].eq("BGG") & total["control"].eq("Control4C")].iloc[0]),
    }, indent=2))


if __name__ == "__main__":
    main()
