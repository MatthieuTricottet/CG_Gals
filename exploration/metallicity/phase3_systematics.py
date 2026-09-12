"""Phase 3 — systematics characterisation (BLINDED).  Run only after Gate B approval.

Reads ONLY work/metallicity_worktable_blind.csv.
Writes outputs/metallicity_scoping.json section "phase3" and figures/fig_*.png.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
FIG = os.path.join(HERE, "figures")
OUT_JSON = os.path.join(HERE, "outputs", "metallicity_scoping.json")
IDX = ["HdA", "HgA", "Hb", "Mgb", "Fe5270", "Fe5335", "Mg2", "D4000n", "MgFe", "MgbFe"]
FIBRE_RADIUS_ARCSEC = 1.5
SN_CUT = 20.0


def load() -> pd.DataFrame:
    b = pd.read_csv(os.path.join(WORK, "metallicity_worktable_blind.csv"))
    assert "sample" not in b.columns
    b["R50_arcsec"] = b["Rchl_r_arcsec"].where(b["Rchl_r_arcsec"] > 0, b["petroR50_r"].where(b["petroR50_r"] > 0))
    b["R50_source"] = np.where(b["Rchl_r_arcsec"] > 0, "simard_Rchl_r", np.where(b["petroR50_r"] > 0, "petroR50_r", "none"))
    b["R50_kpc"] = b["R50_arcsec"] * b["kpc_per_arcsec_Planck15_DA"]
    b["r_fib_kpc"] = FIBRE_RADIUS_ARCSEC * b["kpc_per_arcsec_Planck15_DA"]
    b["ap_frac"] = b["r_fib_kpc"] / b["R50_kpc"]
    b["log_ap_frac"] = np.log10(b["ap_frac"])
    b["log_sigma"] = np.log10(b["sigma"])
    b["cluster"] = b["sample_blind"] + ":" + b["group_blind"]
    b["morph"] = b["morphology"].fillna("NoGZ")
    return b


def ols_cluster(df: pd.DataFrame, formula: str) -> dict:
    df = df.dropna(subset=[c for c in df.columns if c in formula.replace("~", "+").replace("*", "+").replace(":", "+").replace("(", " ").replace(")", " ").replace("+", " ").split()]) if False else df
    m = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["cluster"])[0]})
    return {"n": int(m.nobs), "params": {k: float(v) for k, v in m.params.items()}, "bse": {k: float(v) for k, v in m.bse.items()},
            "r2": float(m.rsquared), "n_clusters": int(df["cluster"].nunique())}


def aperture(b: pd.DataFrame) -> dict:
    sat = b[(b["is_sat"] == 1) & (b["valid_core"] == 1)]
    out = {"fibre_radius_arcsec": FIBRE_RADIUS_ARCSEC, "distance": "Planck15 angular-diameter distance (kpc_proper_per_arcmin), same helper as src/size_data.py",
           "R50_source_counts": {str(k): int(v) for k, v in sat["R50_source"].value_counts().items()},
           "ap_frac_quantiles_sat": {k: float(sat["ap_frac"].quantile(p)) for k, p in [("p10", .1), ("p25", .25), ("p50", .5), ("p75", .75), ("p90", .9)]},
           "gradients": {}}
    for cut, tag in [(0, "SN_gt_0"), (SN_CUT, "SN_gt_20")]:
        s = sat[sat["SN"] > cut].dropna(subset=["log_ap_frac", "log_sigma", "z"])
        out["gradients"][tag] = {}
        for k in IDX:
            d = s.dropna(subset=[k])
            if len(d) < 30:
                continue
            m1 = ols_cluster(d, f"{k} ~ log_ap_frac")
            m2 = ols_cluster(d, f"{k} ~ log_ap_frac + log_sigma")
            m3 = ols_cluster(d, f"{k} ~ z")
            m4 = ols_cluster(d, f"{k} ~ z + log_sigma")
            out["gradients"][tag][k] = {
                "slope_per_dex_apfrac": m1["params"]["log_ap_frac"], "se": m1["bse"]["log_ap_frac"], "n": m1["n"],
                "slope_per_dex_apfrac_at_fixed_sigma": m2["params"]["log_ap_frac"], "se_fixed_sigma": m2["bse"]["log_ap_frac"],
                "slope_per_unit_z": m3["params"]["z"], "se_z": m3["bse"]["z"],
                "slope_per_unit_z_at_fixed_sigma": m4["params"]["z"], "se_z_fixed_sigma": m4["bse"]["z"],
                "implied_shift_over_IQR_apfrac": m2["params"]["log_ap_frac"] * float(np.log10(sat["ap_frac"].quantile(.75) / sat["ap_frac"].quantile(.25))),
                "implied_shift_over_IQR_z": m4["params"]["z"] * float(sat["z"].quantile(.75) - sat["z"].quantile(.25)),
            }
    # figure: indices vs log ap_frac and vs z, split by blinded label
    keys = ["HdA", "Hb", "Mgb", "MgFe", "D4000n", "MgbFe"]
    fig, axes = plt.subplots(len(keys), 2, figsize=(11, 2.6 * len(keys)))
    s = sat[sat["SN"] > SN_CUT]
    for i, k in enumerate(keys):
        for lab, part in s.groupby("sample_blind"):
            axes[i, 0].scatter(part["log_ap_frac"], part[k], s=6, alpha=0.5, label=lab)
            axes[i, 1].scatter(part["z"], part[k], s=6, alpha=0.5, label=lab)
        axes[i, 0].set_ylabel(k); axes[i, 0].set_xlabel("log10(r_fibre / R50)"); axes[i, 1].set_xlabel("z")
        for ax in axes[i]:
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, markerscale=2)
    fig.suptitle(f"Indices vs aperture fraction and redshift — satellites, S/N>{SN_CUT:.0f}, BLINDED labels")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(FIG, "fig_aperture_bias.png"); fig.savefig(p, dpi=130); plt.close(fig)
    out["figure"] = os.path.relpath(p, HERE)
    return out


def balance(b: pd.DataFrame) -> dict:
    covs = ["z", "R50_kpc", "ap_frac", "SN", "sigma", "lgm", "M_r"]
    out = {"note": "Labels are a group-level random permutation, so any imbalance here is noise by construction; the informative quantity is the selection function of the S/N cut (below), which determines whether the cut can break the TRUE matched balance.", "tests": {}, "selection_function": {}}
    sat = b[b["is_sat"] == 1]
    for tag, s in [("before_cut", sat[sat["valid_core"] == 1]), ("after_SN20", sat[(sat["valid_core"] == 1) & (sat["SN"] > SN_CUT)])]:
        out["tests"][tag] = {}
        labels = sorted(s["sample_blind"].unique())
        smallest = sorted(labels, key=lambda l: (s["sample_blind"] == l).sum())[0]
        for c in covs:
            groups = [s.loc[s["sample_blind"] == l, c].dropna() for l in labels]
            kw = stats.kruskal(*[g for g in groups if len(g) > 1])
            med = {l: float(g.median()) if len(g) else None for l, g in zip(labels, groups)}
            ks = {}
            for l, g in zip(labels, groups):
                if l != smallest and len(g) > 1:
                    r = stats.ks_2samp(s.loc[s["sample_blind"] == smallest, c].dropna(), g)
                    ks[l] = {"D": float(r.statistic), "p": float(r.pvalue)}
            out["tests"][tag][c] = {"median_by_label": med, "kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue), "ks_vs_smallest": ks, "n_by_label": {l: int(len(g)) for l, g in zip(labels, groups)}}
    # selection function: P(SN>20 | covariates) among satellites with valid indices
    s = sat[sat["valid_core"] == 1].dropna(subset=["z", "lgm", "M_r", "R50_kpc", "sigma"]).copy()
    s["surv"] = (s["SN"] > SN_CUT).astype(int)
    for c in ["z", "lgm", "M_r", "R50_kpc", "sigma"]:
        s[f"{c}_std"] = (s[c] - s[c].mean()) / s[c].std()
    X = sm.add_constant(s[[f"{c}_std" for c in ["z", "lgm", "M_r", "R50_kpc", "sigma"]]])
    try:
        m = sm.Logit(s["surv"], X).fit(disp=0)
        out["selection_function"] = {"model": "logit(P(SN>20)) ~ standardised z + lgm + M_r + R50_kpc + sigma (satellites, valid core indices)",
                                     "n": int(m.nobs), "coef_per_SD": {k: float(v) for k, v in m.params.items()}, "se": {k: float(v) for k, v in m.bse.items()},
                                     "surv_rate": float(s["surv"].mean())}
    except Exception as exc:  # noqa: BLE001
        out["selection_function"] = {"error": str(exc)}
    # univariate survival vs z and M_r deciles
    for c in ["z", "M_r", "lgm"]:
        dec = pd.qcut(s[c], 5, duplicates="drop")
        out["selection_function"][f"survival_by_{c}_quintile"] = [{"bin": str(i), "rate": float(v), "n": int(n)} for (i, v), n in zip(s.groupby(dec, observed=True)["surv"].mean().items(), s.groupby(dec, observed=True)["surv"].size())]
    return out


def degeneracy(b: pd.DataFrame) -> dict:
    s = b[(b["is_sat"] == 1) & (b["valid_core"] == 1) & (b["SN"] > SN_CUT)].dropna(subset=["MgFe"])
    out = {"n": int(len(s)), "SN_cut": SN_CUT}
    for y in ["HdA", "HdA_sub", "D4000n", "Hb_sub"]:
        d = s.dropna(subset=[y, "MgFe"])
        rho = stats.spearmanr(d["MgFe"], d[y])
        out[f"{y}_vs_MgFe"] = {"n": int(len(d)), "spearman": float(rho.statistic), "p": float(rho.pvalue),
                               "MgFe_p5_p95": [float(d["MgFe"].quantile(.05)), float(d["MgFe"].quantile(.95))],
                               f"{y}_p5_p95": [float(d[y].quantile(.05)), float(d[y].quantile(.95))],
                               "median_err_MgFe": float(d["MgFe_err"].median()), f"median_err_{y}": float(d[f"{y}_err"].median()),
                               "span_over_err_MgFe": float((d["MgFe"].quantile(.95) - d["MgFe"].quantile(.05)) / d["MgFe_err"].median()),
                               f"span_over_err_{y}": float((d[y].quantile(.95) - d[y].quantile(.05)) / d[f"{y}_err"].median())}
        # residual scatter of y at fixed MgFe (quadratic fit) vs its median error: is there structure beyond noise?
        if len(d) > 30:
            coef = np.polyfit(d["MgFe"], d[y], 2)
            resid = d[y] - np.polyval(coef, d["MgFe"])
            out[f"{y}_vs_MgFe"]["resid_sd_at_fixed_MgFe"] = float(resid.std())
            out[f"{y}_vs_MgFe"]["resid_sd_over_median_err"] = float(resid.std() / d[f"{y}_err"].median())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for j, y in enumerate(["HdA_sub", "D4000n"]):
        d = s.dropna(subset=[y, "MgFe", "sigma"])
        sc = axes[0, j].scatter(d["MgFe"], d[y], c=d["sigma"], s=8, cmap="viridis", vmin=50, vmax=300)
        axes[0, j].set_xlabel("[MgFe]' (A)"); axes[0, j].set_ylabel(y); axes[0, j].grid(alpha=0.3)
        plt.colorbar(sc, ax=axes[0, j], label="sigma (km/s)")
        for morph, col in [("Elliptical", "C3"), ("Spiral", "C0"), ("Uncertain", "C7"), ("NoGZ", "0.7")]:
            dd = d[d["morph"] == morph]
            axes[1, j].scatter(dd["MgFe"], dd[y], s=8, alpha=0.6, color=col, label=f"{morph} ({len(dd)})")
        axes[1, j].set_xlabel("[MgFe]' (A)"); axes[1, j].set_ylabel(y); axes[1, j].grid(alpha=0.3); axes[1, j].legend(fontsize=8)
        # typical error bar
        for ax in axes[:, j]:
            ax.errorbar([d["MgFe"].quantile(.05)], [d[y].quantile(.95)], xerr=[d["MgFe_err"].median()], yerr=[d[f"{y}_err"].median()], color="k", capsize=3)
    fig.suptitle(f"Age–metallicity plane occupancy — satellites, S/N>{SN_CUT:.0f} (black cross = median error)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(FIG, "fig_degeneracy_plane.png"); fig.savefig(p, dpi=130); plt.close(fig)
    out["figure"] = os.path.relpath(p, HERE)
    return out


def morphology_confound(b: pd.DataFrame) -> dict:
    out = {"model": "index ~ elliptical + log_sigma + log_sigma^2 (pooled over all blinded labels, satellites+BGGs with valid indices, cluster-robust SE by group)", "by_SN_cut": {}}
    d0 = b[(b["valid_core"] == 1) & b["morph"].isin(["Elliptical", "Spiral"])].dropna(subset=["log_sigma"]).copy()
    d0["elliptical"] = (d0["morph"] == "Elliptical").astype(int)
    d0["log_sigma2"] = d0["log_sigma"] ** 2
    for cut, tag in [(0, "SN_gt_0"), (SN_CUT, "SN_gt_20")]:
        d = d0[d0["SN"] > cut]
        out["by_SN_cut"][tag] = {"n": int(len(d)), "n_E": int(d["elliptical"].sum()), "n_S": int((1 - d["elliptical"]).sum()), "indices": {}}
        for k in IDX:
            dd = d.dropna(subset=[k])
            if len(dd) < 30:
                continue
            m = ols_cluster(dd, f"{k} ~ elliptical + log_sigma + log_sigma2")
            raw = float(dd.loc[dd["elliptical"] == 1, k].median() - dd.loc[dd["elliptical"] == 0, k].median())
            # within sigma bins
            bins = pd.cut(dd["sigma"], [0, 80, 120, 160, 220, 400])
            per_bin = {}
            for bn, part in dd.groupby(bins, observed=True):
                e, sp = part.loc[part["elliptical"] == 1, k], part.loc[part["elliptical"] == 0, k]
                if len(e) >= 5 and len(sp) >= 5:
                    per_bin[str(bn)] = {"E_minus_S_median": float(e.median() - sp.median()), "n_E": int(len(e)), "n_S": int(len(sp))}
            out["by_SN_cut"][tag]["indices"][k] = {"E_minus_S_at_fixed_sigma": m["params"]["elliptical"], "se": m["bse"]["elliptical"],
                                                   "E_minus_S_raw_median_diff": raw, "n": m["n"], "per_sigma_bin": per_bin}
    return out


def main() -> None:
    b = load()
    ap = aperture(b)
    bal = balance(b)
    deg = degeneracy(b)
    mor = morphology_confound(b)
    with open(OUT_JSON) as fh:
        doc = json.load(fh)
    # compare each systematic to Delta_min at SN>20 for the smallest blinded label
    pw = doc["phase2"]["power"]["by_threshold"][str(int(SN_CUT))]
    small = pw["two_smallest_labels"][0]
    comp = {}
    for k in IDX:
        dmin = pw["indices"].get(k, {}).get("per_blind_sample", {}).get(small, {}).get("delta_min")
        g = ap["gradients"]["SN_gt_20"].get(k, {})
        m = mor["by_SN_cut"]["SN_gt_20"]["indices"].get(k, {})
        comp[k] = {"delta_min_SN20_smallest": dmin,
                   "aperture_shift_over_IQR_at_fixed_sigma": g.get("implied_shift_over_IQR_apfrac"),
                   "z_shift_over_IQR_at_fixed_sigma": g.get("implied_shift_over_IQR_z"),
                   "morphology_E_minus_S_at_fixed_sigma": m.get("E_minus_S_at_fixed_sigma"),
                   "ratio_aperture_over_delta_min": None if not dmin or g.get("implied_shift_over_IQR_apfrac") is None else abs(g["implied_shift_over_IQR_apfrac"]) / dmin,
                   "ratio_morph_over_delta_min": None if not dmin or m.get("E_minus_S_at_fixed_sigma") is None else abs(m["E_minus_S_at_fixed_sigma"]) / dmin}
    doc["phase3"] = {"blinded": True, "aperture": ap, "balance": bal, "degeneracy": deg, "morphology_confound": mor, "systematics_vs_delta_min": comp}
    with open(OUT_JSON, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(json.dumps(comp, indent=1))
    print("aperture figure:", ap["figure"], "| degeneracy figure:", deg["figure"])
    print("selection function coef/SD:", bal["selection_function"].get("coef_per_SD"))


if __name__ == "__main__":
    main()
