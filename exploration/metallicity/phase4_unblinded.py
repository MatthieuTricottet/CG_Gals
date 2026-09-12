"""Phase 4 — UNBLINDED exploratory contrasts (no p-values, no multiplicity correction).

Reads work/metallicity_worktable.csv (true labels) and work/host_members_worktable_raw.csv.
Writes outputs/metallicity_scoping.json section "phase4" and figures/fig_phase4_contrasts.png.

Design (all satellites unless stated; galaxies need the 7 core indices + sigma):
  (a) galaxy-level adjusted contrast CG4 vs each control:
        y ~ is_CG4 + C(morph) + log sigma + (log sigma)^2, cluster-robust SE by physical group,
        group-bootstrap percentile CIs (groups resampled within each sample).
  (b) group-level mean contrast: residuals of y from the pooled (morph, sigma) trend, averaged
        per group; difference of CG4 and control group means; group bootstrap.
  (c) (a) restricted to GZ1 ellipticals only, and to spirals only (no morphology term).
  (d) within-host fixed-effects analogue of src/host_controlled.py:
        y ~ is_CG_member + C(morph) + log sigma + (log sigma)^2 + logMstar + rank_parent
            + dist_host_kpc + C(host_lim_group); cluster-robust SE by host; host bootstrap.
Two S/N variants: 'all_valid' (primary, per Phase 2-3) and 'SN_gt_20'.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy.cosmology import Planck15  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
FIG = os.path.join(HERE, "figures")
OUT_JSON = os.path.join(HERE, "outputs", "metallicity_scoping.json")
sys.path.insert(0, HERE)
import phase1_assemble as p1a  # noqa: E402

CONTROLS = ["Control4B", "Control4C", "RG4"]
SHORT = {"Control4B": "C4B", "Control4C": "C4C", "RG4": "RG4"}
IDX = ["Mgb", "Fe5270", "Fe5335", "MgFe", "MgbFe", "Mg2", "D4000n", "HdA_sub", "Hb_sub", "HdA"]
N_BOOT = 1000
SEED = 20260912
VARIANTS = {"all_valid": 0.0, "SN_gt_20": 20.0}


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["morph"] = df["morphology"].fillna("NoGZ").astype(str)
    df["log_sigma"] = np.log10(df["sigma"])
    df["log_sigma2"] = df["log_sigma"] ** 2
    return df


def load_main() -> pd.DataFrame:
    w = pd.read_csv(os.path.join(WORK, "metallicity_worktable.csv"), low_memory=False)
    w = w[(w["is_sat"] == 1) & (w["valid_core"] == 1) & w["sigma"].notna() & (w["sigma"] > 0)]
    w = prep(w)
    w["cluster"] = w["group_uid"]
    return w


def load_hosts() -> pd.DataFrame:
    h = pd.read_csv(os.path.join(WORK, "host_members_worktable_raw.csv"), low_memory=False)
    h = p1a.clean(h)
    h = h[(h["valid_core"] == 1) & h["sigma"].notna() & (h["sigma"] > 0)]
    h = prep(h)
    h["logMstar"] = pd.to_numeric(h["lgm"], errors="coerce")
    h["rank_parent"] = pd.to_numeric(h["rank_M"], errors="coerce")
    zg = pd.to_numeric(h["Yang_z_CMB_group"], errors="coerce").fillna(pd.to_numeric(h["z"], errors="coerce"))
    kpc_per_arcmin = Planck15.kpc_proper_per_arcmin(zg.to_numpy()).to(u.kpc / u.arcmin).value
    # src/host_controlled._dist_kpc convention (dist2BGG in arcsec -> arcmin)
    h["dist_host_kpc"] = pd.to_numeric(h["dist2BGG"], errors="coerce") / 60.0 * kpc_per_arcmin
    h["cluster"] = h["host_lim_group"].astype(str)
    return h


def design(df: pd.DataFrame, y: str, treat: str, morph_term: bool, extra: list[str] | None = None, fe: str | None = None):
    """Return (y, X, column names) as numpy arrays for fast bootstrap."""
    cols = {treat: df[treat].to_numpy(dtype=float), "log_sigma": df["log_sigma"].to_numpy(), "log_sigma2": df["log_sigma2"].to_numpy()}
    if morph_term:
        for m in ["Spiral", "Uncertain", "NoGZ"]:
            cols[f"morph_{m}"] = (df["morph"] == m).to_numpy(dtype=float)
    for e in extra or []:
        cols[e] = df[e].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df))] + list(cols.values()))
    names = ["const"] + list(cols.keys())
    if fe:
        # within transformation (demean by fixed-effect group) -- equivalent to C(fe) dummies
        g = pd.factorize(df[fe])[0]
        yv = df[y].to_numpy(dtype=float)
        def demean(a):
            a = np.asarray(a, dtype=float)
            if a.ndim == 1:
                return a - pd.Series(a).groupby(g).transform("mean").to_numpy()
            return np.column_stack([demean(a[:, j]) for j in range(a.shape[1])])
        return demean(yv), demean(X[:, 1:]), names[1:]
    return df[y].to_numpy(dtype=float), X, names


def ols_treat(yv, X, names, treat, clusters):
    """OLS coefficient on `treat` with cluster-robust (CR1) SE."""
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    cid = pd.factorize(clusters)[0]
    G = cid.max() + 1
    meat = np.zeros((X.shape[1], X.shape[1]))
    for gidx in range(G):
        m = cid == gidx
        s = X[m].T @ resid[m]
        meat += np.outer(s, s)
    n, k = X.shape
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    j = names.index(treat)
    return float(beta[j]), float(np.sqrt(V[j, j])), int(G)


def group_bootstrap(df, y, treat, morph_term, clusters_col, rng, extra=None, fe=None, strata_col=None):
    """Percentile CIs from resampling clusters (within strata if given)."""
    yv, X, names = design(df, y, treat, morph_term, extra, fe)
    j = names.index(treat)
    cl = df[clusters_col].to_numpy()
    strata = df[strata_col].to_numpy() if strata_col else np.zeros(len(df), dtype=int)
    idx_by_cluster = {}
    for i, (c, s) in enumerate(zip(cl, strata)):
        idx_by_cluster.setdefault((s, c), []).append(i)
    keys_by_stratum = {}
    for (s, c) in idx_by_cluster:
        keys_by_stratum.setdefault(s, []).append((s, c))
    boots = []
    for _ in range(N_BOOT):
        rows = []
        for s, keys in keys_by_stratum.items():
            pick = rng.integers(0, len(keys), size=len(keys))
            for p in pick:
                rows.extend(idx_by_cluster[keys[p]])
        rows = np.array(rows)
        if fe:
            # re-demean within the resampled fixed-effect groups (duplicated hosts stay distinct)
            sub = df.iloc[rows].copy()
            sub["_fe"] = sub[fe].astype(str) + "#" + pd.Series(rows).groupby(rows).cumcount().astype(str).to_numpy()
            yb, Xb, _ = design(sub, y, treat, morph_term, extra, fe="_fe")
        else:
            yb, Xb = yv[rows], X[rows]
        try:
            b, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
            boots.append(b[j])
        except np.linalg.LinAlgError:
            continue
    boots = np.array(boots)
    return {"ci68": [float(np.percentile(boots, 16)), float(np.percentile(boots, 84))],
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "n_boot": int(len(boots))}


def contrast_a(w: pd.DataFrame, rng, sn_cut: float) -> dict:
    out = {}
    for ctrl in CONTROLS:
        d = w[(w["sample"].isin(["CG4", ctrl])) & (w["SN"] > sn_cut)].copy()
        d["is_CG4"] = (d["sample"] == "CG4").astype(int)
        out[ctrl] = {"n_CG4": int(d["is_CG4"].sum()), "n_ctrl": int((1 - d["is_CG4"]).sum()),
                     "groups_CG4": int(d.loc[d["is_CG4"] == 1, "cluster"].nunique()), "groups_ctrl": int(d.loc[d["is_CG4"] == 0, "cluster"].nunique()),
                     "indices": {}}
        for k in IDX:
            dd = d.dropna(subset=[k])
            for subset, mt, sel in [("all_morph", True, dd), ("ellipticals", False, dd[dd["morph"] == "Elliptical"]), ("spirals", False, dd[dd["morph"] == "Spiral"])]:
                if sel["is_CG4"].sum() < 10 or (1 - sel["is_CG4"]).sum() < 10:
                    continue
                yv, X, names = design(sel, k, "is_CG4", mt)
                coef, se, G = ols_treat(yv, X, names, "is_CG4", sel["cluster"].to_numpy())
                bs = group_bootstrap(sel, k, "is_CG4", mt, "cluster", rng, strata_col="sample")
                out[ctrl]["indices"].setdefault(k, {})[subset] = {"coef": coef, "cluster_se": se, "n": int(len(sel)), "n_CG4": int(sel["is_CG4"].sum()), "n_clusters": G, **bs}
    return out


def contrast_b(w: pd.DataFrame, rng, sn_cut: float) -> dict:
    """Group-level means of residuals from the pooled morphology+sigma trend."""
    out = {}
    d = w[w["SN"] > sn_cut].copy()
    pooled = d.drop_duplicates("objid")
    for k in IDX:
        dd = pooled.dropna(subset=[k])
        m = smf.ols(f"{k} ~ C(morph) + log_sigma + log_sigma2", data=dd).fit()
        d[f"{k}_resid"] = d[k] - m.predict(d)
        gm = d.dropna(subset=[f"{k}_resid"]).groupby(["sample", "cluster"])[f"{k}_resid"].agg(["mean", "size"]).reset_index()
        cg = gm[gm["sample"] == "CG4"]["mean"].to_numpy()
        out[k] = {"CG4": {"N_g": int(len(cg)), "mean_of_group_means": float(cg.mean()), "se": float(cg.std(ddof=1) / np.sqrt(len(cg)))}}
        for ctrl in CONTROLS:
            cc = gm[gm["sample"] == ctrl]["mean"].to_numpy()
            diff = cg.mean() - cc.mean()
            se = np.sqrt(cg.var(ddof=1) / len(cg) + cc.var(ddof=1) / len(cc))
            boots = [rng.choice(cg, len(cg)).mean() - rng.choice(cc, len(cc)).mean() for _ in range(N_BOOT)]
            out[k][ctrl] = {"N_g": int(len(cc)), "diff_CG4_minus_ctrl": float(diff), "se": float(se),
                            "ci68": [float(np.percentile(boots, 16)), float(np.percentile(boots, 84))],
                            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    return out


def contrast_d(h: pd.DataFrame, rng, sn_cut: float) -> dict:
    out = {}
    extra = ["logMstar", "rank_parent", "dist_host_kpc"]
    d = h[h["SN"] > sn_cut].dropna(subset=extra + ["log_sigma"]).copy()
    for subset, sel in [("all_members", d), ("satellites_only", d[d["rank_parent"] > 1])]:
        out[subset] = {"n": int(len(sel)), "n_CG": int(sel["is_CG_member"].sum()), "n_hosts": int(sel["host_lim_group"].nunique()), "indices": {}}
        for k in IDX:
            dd = sel.dropna(subset=[k])
            # informative hosts only: both CG and non-CG members present
            inf = dd.groupby("host_lim_group")["is_CG_member"].nunique()
            dd = dd[dd["host_lim_group"].isin(inf[inf > 1].index)]
            if len(dd) < 30:
                continue
            yv, X, names = design(dd, k, "is_CG_member", True, extra, fe="host_lim_group")
            coef, se, G = ols_treat(yv, X, names, "is_CG_member", dd["host_lim_group"].to_numpy())
            bs = group_bootstrap(dd, k, "is_CG_member", True, "host_lim_group", rng, extra=extra, fe="host_lim_group")
            out[subset]["indices"][k] = {"coef": coef, "cluster_se": se, "n": int(len(dd)), "n_CG": int(dd["is_CG_member"].sum()), "n_informative_hosts": G, **bs}
    return out


def figure(res: dict, dmin: dict) -> str:
    keys = ["Mgb", "Fe5270", "Fe5335", "MgFe", "MgbFe", "Mg2", "D4000n", "HdA_sub", "Hb_sub"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, k in zip(axes.ravel(), keys):
        labels, vals, lo68, hi68, lo95, hi95 = [], [], [], [], [], []
        for ctrl in CONTROLS:
            for subset, tag in [("all_morph", "all"), ("ellipticals", "E"), ("spirals", "S")]:
                r = res["all_valid"]["a_galaxy_level"][ctrl]["indices"].get(k, {}).get(subset)
                if r:
                    labels.append(f"{SHORT[ctrl]} {tag}"); vals.append(r["coef"]); lo68.append(r["ci68"][0]); hi68.append(r["ci68"][1]); lo95.append(r["ci95"][0]); hi95.append(r["ci95"][1])
        r = res["all_valid"]["d_within_host"]["satellites_only"]["indices"].get(k)
        if r:
            labels.append("within-host"); vals.append(r["coef"]); lo68.append(r["ci68"][0]); hi68.append(r["ci68"][1]); lo95.append(r["ci95"][0]); hi95.append(r["ci95"][1])
        yy = np.arange(len(labels))
        ax.hlines(yy, lo95, hi95, color="0.7", lw=2)
        ax.hlines(yy, lo68, hi68, color="C0", lw=4)
        ax.plot(vals, yy, "ko", ms=4)
        ax.axvline(0, color="k", lw=0.8)
        dm = dmin.get(k)
        if dm:
            ax.axvspan(-dm, dm, color="C1", alpha=0.12, label="$\\pm\\Delta_{\\min}$ (Phase 2)")
        ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=8); ax.set_title(k); ax.grid(alpha=0.3, axis="x")
        ax.set_xlabel("CG4 − control (index units), fixed morphology & σ")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Phase 4 exploratory contrasts, satellites, no S/N cut — 68% (thick) / 95% (thin) group-bootstrap CIs; NO p-values", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(FIG, "fig_phase4_contrasts.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return os.path.relpath(p, HERE)


def main() -> None:
    rng = np.random.default_rng(SEED)
    w = load_main()
    h = load_hosts()
    with open(OUT_JSON) as fh:
        doc = json.load(fh)
    res = {}
    for name, cut in VARIANTS.items():
        print(f"variant {name} ...", flush=True)
        res[name] = {"sn_cut": cut, "a_galaxy_level": contrast_a(w, rng, cut), "b_group_level": contrast_b(w, rng, cut), "d_within_host": contrast_d(h, rng, cut)}
    # Delta_min reference (Phase 2, S/N>0 and >20, smallest blinded label) for the figure/report
    dmin = {}
    for k in IDX:
        e = doc["phase2"]["power"]["by_threshold"]["0"]
        small = e["two_smallest_labels"][0]
        v = e["indices"].get(k, {}).get("per_blind_sample", {}).get(small, {}).get("delta_min")
        if v:
            dmin[k] = v
    figpath = figure(res, dmin)
    # true-label usability census + explicit list of usable satellites (deliverable 'which ones')
    full = pd.read_csv(os.path.join(WORK, "metallicity_worktable.csv"), low_memory=False)
    sat = full[full["is_sat"] == 1].copy()
    sat["usable"] = ((sat["valid_core"] == 1) & sat["sigma"].notna() & (sat["sigma"] > 0)).astype(int)
    census = {}
    for name in ["CG4", "RG4", "Control4B", "Control4C"]:
        part = sat[sat["sample"] == name]
        census[name] = {"N_sat": int(len(part)), "N_usable": int(part["usable"].sum()),
                        "N_usable_SN_gt_20": int(((part["usable"] == 1) & (part["SN"] > 20)).sum()),
                        "N_usable_SN_gt_30": int(((part["usable"] == 1) & (part["SN"] > 30)).sum()),
                        "N_groups_with_usable_sat": int(part.loc[part["usable"] == 1, "group_uid"].nunique()),
                        "N_groups_with_usable_sat_SN_gt_20": int(part.loc[(part["usable"] == 1) & (part["SN"] > 20), "group_uid"].nunique()),
                        "median_M_r_usable": float(part.loc[part["usable"] == 1, "M_r"].median()),
                        "median_M_r_usable_SN_gt_20": float(part.loc[(part["usable"] == 1) & (part["SN"] > 20), "M_r"].median())}
    cols = ["sample", "group_uid", "objid", "specObjID", "rank_M", "z", "M_r", "lgm", "morphology", "sSFR_status", "sigma", "sigma_err", "SN",
            "usable", "valid_core", "valid_core_sub", "spec_choice", "loss_reason", "Mgb", "Mgb_err", "Fe5270", "Fe5335", "MgFe", "MgFe_err", "D4000n", "HdA_sub", "Hb_sub"]
    sat[cols].sort_values(["sample", "group_uid", "rank_M"]).to_csv(os.path.join(HERE, "outputs", "usable_satellites.csv"), index=False)
    doc["phase4"] = {"true_label_census_satellites": census, "usable_satellite_list": "outputs/usable_satellites.csv","unblinded": True, "statement": "EXPLORATORY SCOPING RUN, NOT AN INFERENCE: no p-values are reported and no multiple-comparison correction is applied, so that no prespecified Paper III analysis is contaminated. Point estimates with group-clustered SEs and group-bootstrap percentile CIs only.",
                     "preunblind_snapshot": "outputs/metallicity_scoping_preunblind_snapshot.json (sha256 in outputs/PREUNBLIND_SHA256.txt)",
                     "design": __doc__, "n_boot": N_BOOT, "seed": SEED, "delta_min_reference_SN0_smallest_label": dmin,
                     "host_frame": {"n_members_fetched": int(len(pd.read_csv(os.path.join(WORK, "host_members_worktable_raw.csv"), low_memory=False))), "n_usable": int(len(h)), "n_hosts_usable": int(h["host_lim_group"].nunique()), "n_CG_usable": int(h["is_CG_member"].sum())},
                     "results": res, "figure": figpath}
    with open(OUT_JSON, "w") as fh:
        json.dump(doc, fh, indent=2)
    # console
    for name in VARIANTS:
        a = res[name]["a_galaxy_level"]
        print(f"\n=== {name}: galaxy-level CG4 - control at fixed morphology & sigma (coef ± cluster SE [68% CI]) ===")
        for k in IDX:
            row = []
            for ctrl in CONTROLS:
                r = a[ctrl]["indices"].get(k, {}).get("all_morph")
                row.append(f"{SHORT[ctrl]}: {r['coef']:+.3f}±{r['cluster_se']:.3f} [{r['ci68'][0]:+.3f},{r['ci68'][1]:+.3f}]" if r else f"{ctrl}: -")
            print(f"  {k:8s} " + " | ".join(row) + f"   (Δmin≈{dmin.get(k, float('nan')):.3f})")
        print(f"  --- ellipticals only / spirals only (vs Control4C) ---")
        for k in IDX:
            e = a["Control4C"]["indices"].get(k, {}).get("ellipticals"); s = a["Control4C"]["indices"].get(k, {}).get("spirals")
            print(f"  {k:8s} E: {e['coef']:+.3f}±{e['cluster_se']:.3f} (n_CG4={e['n_CG4']}) | S: {s['coef']:+.3f}±{s['cluster_se']:.3f} (n_CG4={s['n_CG4']})" if e and s else f"  {k}: -")
        dwh = res[name]["d_within_host"]["satellites_only"]
        print(f"  --- within-host FE (satellites only; n={dwh['n']}, CG={dwh['n_CG']}, hosts={dwh['n_hosts']}) ---")
        for k in IDX:
            r = dwh["indices"].get(k)
            print(f"  {k:8s} {r['coef']:+.3f}±{r['cluster_se']:.3f} [68%: {r['ci68'][0]:+.3f},{r['ci68'][1]:+.3f}] informative hosts={r['n_informative_hosts']}" if r else f"  {k}: -")
    print("figure:", figpath)


if __name__ == "__main__":
    main()
