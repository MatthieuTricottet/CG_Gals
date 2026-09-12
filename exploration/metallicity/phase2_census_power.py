"""Phase 2 — usability census and power (BLINDED).

Reads ONLY work/metallicity_worktable_blind.csv (sample_blind / group_blind).
Writes outputs/metallicity_scoping.json section "phase2" and
figures/fig_power_vs_SN.png.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
FIG = os.path.join(HERE, "figures")
OUT_JSON = os.path.join(HERE, "outputs", "metallicity_scoping.json")

SN_THRESH = [0, 10, 15, 20, 25, 30]
IDX_RAW = ["HdA", "HgA", "Hb", "Mgb", "Fe5270", "Fe5335", "Mg2", "D4000n", "MgFe", "MgbFe"]
IDX_SUB = [f"{k}_sub" for k in IDX_RAW]
UNITS = {"HdA": "A", "HgA": "A", "Hb": "A", "Mgb": "A", "Fe5270": "A", "Fe5335": "A", "Mg2": "mag",
         "D4000n": "", "MgFe": "A", "MgbFe": ""}
N_BOOT = 2000
SEED = 7


def load_blind() -> pd.DataFrame:
    b = pd.read_csv(os.path.join(WORK, "metallicity_worktable_blind.csv"))
    assert "sample" not in b.columns and "group_uid" not in b.columns, "true labels leaked into blind table"
    return b


def q(x: pd.Series) -> dict:
    x = x.dropna()
    if x.empty:
        return {"n": 0}
    return {"n": int(len(x)), "p10": float(x.quantile(0.10)), "p25": float(x.quantile(0.25)), "p50": float(x.median()),
            "p75": float(x.quantile(0.75)), "p90": float(x.quantile(0.90)), "mean": float(x.mean())}


def census(b: pd.DataFrame) -> dict:
    out = {}
    pooled = b.drop_duplicates("obj_hash")
    for t in SN_THRESH:
        ok = (pooled["SN"] > t) & (pooled["valid_core"] == 1)
        entry = {"pooled_dedup": {
            "all": int(ok.sum()), "sat": int((ok & (pooled["is_sat"] == 1)).sum()), "bgg": int((ok & (pooled["is_bgg"] == 1)).sum()),
            "all_SN_only": int((pooled["SN"] > t).sum()), "sat_SN_only": int(((pooled["SN"] > t) & (pooled["is_sat"] == 1)).sum()),
        }, "by_blind_sample": {}}
        for lab, part in b.groupby("sample_blind"):
            okp = (part["SN"] > t) & (part["valid_core"] == 1)
            sat = okp & (part["is_sat"] == 1)
            gsz = part.loc[sat].groupby("group_blind").size()
            entry["by_blind_sample"][lab] = {
                "all": int(okp.sum()), "sat": int(sat.sum()), "bgg": int((okp & (part["is_bgg"] == 1)).sum()),
                "N_groups_total": int(part["group_blind"].nunique()),
                "N_groups_with_ge1_sat": int((gsz >= 1).sum()), "N_groups_with_ge2_sat": int((gsz >= 2).sum()),
                "N_groups_all3_sat": int((gsz == 3).sum()),
                "mean_sat_per_surviving_group": None if gsz.empty else float(gsz.mean()),
            }
        out[str(t)] = entry
    return out


def decisive(b: pd.DataFrame, t: float = 20.0) -> dict:
    pooled = b.drop_duplicates("obj_hash")
    sat = pooled[pooled["is_sat"] == 1]
    surv = sat[(sat["SN"] > t) & (sat["valid_core"] == 1)]
    out = {"threshold": t, "N_sat_full": int(len(sat)), "N_sat_surviving": int(len(surv)),
           "frac_sat_surviving": float(len(surv) / len(sat)) if len(sat) else None,
           "M_r_full": q(sat["M_r"]), "M_r_survivors": q(surv["M_r"]),
           "lgm_full": q(sat["lgm"]), "lgm_survivors": q(surv["lgm"]),
           "z_full": q(sat["z"]), "z_survivors": q(surv["z"])}
    # survival fraction by quartile of the FULL satellite distribution
    for col in ["M_r", "lgm"]:
        edges = sat[col].quantile([0, .25, .5, .75, 1]).to_numpy()
        labels = ["Q1", "Q2", "Q3", "Q4"]  # Q1 = lowest values (for M_r: brightest)
        cat = pd.cut(sat[col], bins=edges, labels=labels, include_lowest=True)
        surv_cat = pd.cut(surv[col], bins=edges, labels=labels, include_lowest=True)
        out[f"survival_by_{col}_quartile"] = {
            lab: {"n_full": int((cat == lab).sum()), "n_surv": int((surv_cat == lab).sum()),
                  "frac_surv": float((surv_cat == lab).sum() / max((cat == lab).sum(), 1)),
                  "edges": [float(edges[i]), float(edges[i + 1])]}
            for i, lab in enumerate(labels)}
        out[f"frac_of_survivors_in_{col}_Q1"] = float((surv_cat == "Q1").mean()) if len(surv) else None
        out[f"frac_of_survivors_in_{col}_Q4"] = float((surv_cat == "Q4").mean()) if len(surv) else None
    # per blinded sample
    out["by_blind_sample"] = {}
    for lab, part in b[b["is_sat"] == 1].groupby("sample_blind"):
        s = part[(part["SN"] > t) & (part["valid_core"] == 1)]
        out["by_blind_sample"][lab] = {"N_sat_full": int(len(part)), "N_sat_surviving": int(len(s)),
                                       "frac": float(len(s) / len(part)) if len(part) else None,
                                       "M_r_p50_full": float(part["M_r"].median()), "M_r_p50_surv": None if s.empty else float(s["M_r"].median()),
                                       "lgm_p50_full": float(part["lgm"].median()), "lgm_p50_surv": None if s.empty else float(s["lgm"].median())}
    return out


def error_budget(b: pd.DataFrame) -> dict:
    pooled = b.drop_duplicates("obj_hash")
    pooled = pooled[pooled["SN"].notna()]
    bins = [0, 5, 10, 15, 20, 25, 30, 40, 60, 1e9]
    pooled = pooled.assign(SN_bin=pd.cut(pooled["SN"], bins=bins))
    out = {"SN_bins": [[float(bins[i]), float(bins[i + 1])] for i in range(len(bins) - 1)], "median_err_by_bin": {}, "fit_err_eq_a_over_SN": {}}
    for k in IDX_RAW + IDX_SUB:
        e = pooled[f"{k}_err"]
        med = pooled.groupby("SN_bin", observed=False)[f"{k}_err"].median()
        n = pooled.groupby("SN_bin", observed=False)[f"{k}_err"].count()
        out["median_err_by_bin"][k] = [{"bin": str(i), "median_err": None if pd.isna(v) else float(v), "n": int(n[i])} for i, v in med.items()]
        m = e.notna() & (pooled["SN"] > 0)
        if m.sum() > 10:
            # err ~ a / SN  (least squares in log space)
            a = np.exp(np.median(np.log(e[m]) + np.log(pooled.loc[m, "SN"])))
            out["fit_err_eq_a_over_SN"][k] = {"a": float(a), "n": int(m.sum()),
                                              "err_at_SN20": float(a / 20), "err_at_SN10": float(a / 10)}
    # total scatter vs measurement error for satellites
    sat = pooled[pooled["is_sat"] == 1]
    out["satellite_scatter_vs_error"] = {}
    for k in IDX_RAW:
        v = sat[k].dropna()
        out["satellite_scatter_vs_error"][k] = {"n": int(len(v)), "sd_total": float(v.std()) if len(v) > 2 else None,
                                                "mad_sd_total": float(1.4826 * (v - v.median()).abs().median()) if len(v) > 2 else None,
                                                "median_err": float(sat[f"{k}_err"].median()) if len(v) > 2 else None}
    return out


def _icc_from_stats(n_i: np.ndarray, s_i: np.ndarray, q_i: np.ndarray) -> tuple[float, float, int]:
    """ICC (one-way ANOVA, Donner 1986 n0) from per-group counts, sums and sums of squares."""
    k = len(n_i)
    if k < 3:
        return np.nan, np.nan, int(k)
    N = n_i.sum()
    grand = s_i.sum() / N
    means = s_i / n_i
    ssb = float((n_i * (means - grand) ** 2).sum())
    ssw = float((q_i - s_i ** 2 / n_i).sum())
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    rho = (msb - msw) / (msb + (n0 - 1) * msw)
    return float(np.clip(rho, 0, 1)), float(n0), int(k)


def _group_stats(values: np.ndarray, groups: np.ndarray):
    df = pd.DataFrame({"y": values, "g": groups}).dropna()
    agg = df.groupby("g")["y"].agg(["size", "sum", lambda x: float((x ** 2).sum())])
    agg.columns = ["n", "s", "q"]
    agg = agg[agg["n"] >= 2]
    return agg["n"].to_numpy(dtype=float), agg["s"].to_numpy(dtype=float), agg["q"].to_numpy(dtype=float)


def icc_anova(values: np.ndarray, groups: np.ndarray) -> tuple[float, float, int]:
    n_i, s_i, q_i = _group_stats(values, groups)
    return _icc_from_stats(n_i, s_i, q_i)


def rho_with_ci(values: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> dict:
    n_i, s_i, q_i = _group_stats(values, groups)
    rho, n0, k = _icc_from_stats(n_i, s_i, q_i)
    if not np.isfinite(rho):
        return {"rho": None, "n0": None, "n_groups": k, "ci68": None, "ci95": None}
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, k, size=k)
        r, _, _ = _icc_from_stats(n_i[pick], s_i[pick], q_i[pick])
        boots.append(r)
    boots = np.array([x for x in boots if np.isfinite(x)])
    return {"rho": rho, "n0": n0, "n_groups": k,
            "ci68": [float(np.percentile(boots, 16)), float(np.percentile(boots, 84))] if len(boots) else None,
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if len(boots) else None}


def delta_min(sigma: float, rho: float, N_g: int, n_sat: float, k: float = 3.0) -> float:
    if N_g <= 0 or n_sat <= 0:
        return np.nan
    return k * sigma * np.sqrt((1 + rho * (n_sat - 1)) / (N_g * n_sat))


def power(b: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    out = {"formula": "Delta_min = 3 * sigma_idx * sqrt((1 + rho (n_sat - 1)) / (N_g n_sat)); sigma_idx = pooled within-blinded-sample SD of the index among surviving satellites (total scatter: intrinsic + measurement); rho = one-way-ANOVA ICC of surviving satellites within groups (pooled, groups with >=2 survivors); N_g = groups with >=1 surviving satellite; n_sat = mean survivors per such group",
           "two_sample": "Delta_min(A vs B) = 3 * sqrt(SE_A^2 + SE_B^2), SE = sigma_idx sqrt((1+rho(n-1))/(N_g n))",
           "by_threshold": {}}
    sat_all = b[b["is_sat"] == 1]
    labels = sorted(b["sample_blind"].unique())
    for t in SN_THRESH:
        sat = sat_all[(sat_all["SN"] > t) & (sat_all["valid_core"] == 1)]
        entry = {"indices": {}}
        # sample sizes per blinded label
        sizes = {}
        for lab in labels:
            part = sat[sat["sample_blind"] == lab]
            gsz = part.groupby("group_blind").size()
            sizes[lab] = {"N_g": int(len(gsz)), "n_sat": float(gsz.mean()) if len(gsz) else 0.0, "N_sat": int(len(part))}
        entry["sizes"] = sizes
        smallest2 = sorted(labels, key=lambda l: sizes[l]["N_sat"])[:2]
        entry["two_smallest_labels"] = smallest2
        for k in IDX_RAW + IDX_SUB:
            v = sat[[k, "sample_blind", "group_blind"]].dropna()
            if len(v) < 10:
                entry["indices"][k] = {"n": int(len(v))}
                continue
            # pooled within-sample SD
            resid = v[k] - v.groupby("sample_blind")[k].transform("mean")
            dof = len(v) - v["sample_blind"].nunique()
            sigma = float(np.sqrt((resid ** 2).sum() / dof))
            sigma_mad = float(1.4826 * (resid - resid.median()).abs().median())
            # ICC pooled over all blinded samples: cluster key must be unique across labels
            cl = (v["sample_blind"] + ":" + v["group_blind"]).to_numpy()
            rho = rho_with_ci(v[k].to_numpy(), cl, rng) if k in IDX_RAW else {"rho": icc_anova(v[k].to_numpy(), cl)[0]}
            r = rho["rho"] if rho["rho"] is not None and np.isfinite(rho["rho"]) else 0.0
            per = {}
            for lab in labels:
                s = sizes[lab]
                per[lab] = {"N_g": s["N_g"], "n_sat": s["n_sat"], "delta_min": float(delta_min(sigma, r, s["N_g"], s["n_sat"])),
                            "delta_min_mad": float(delta_min(sigma_mad, r, s["N_g"], s["n_sat"]))}
                per[lab]["delta_min_over_sigma"] = per[lab]["delta_min"] / sigma if sigma > 0 else None
            a, c = smallest2
            se = lambda lab, sg: sg * np.sqrt((1 + r * (sizes[lab]["n_sat"] - 1)) / max(sizes[lab]["N_g"] * sizes[lab]["n_sat"], 1e-9))  # noqa: E731
            two = float(3 * np.sqrt(se(a, sigma) ** 2 + se(c, sigma) ** 2))
            # smallest vs largest (CG-size vs big control)
            largest = sorted(labels, key=lambda l: -sizes[l]["N_sat"])[0]
            two_sl = float(3 * np.sqrt(se(smallest2[0], sigma) ** 2 + se(largest, sigma) ** 2))
            entry["indices"][k] = {"n": int(len(v)), "sigma_idx": sigma, "sigma_idx_mad": sigma_mad,
                                   "median_meas_err": float(sat[f"{k}_err"].median()),
                                   "rho": rho, "rho_used": r, "per_blind_sample": per,
                                   "two_sample_smallest_pair": {"labels": smallest2, "delta_min": two},
                                   "two_sample_smallest_vs_largest": {"labels": [smallest2[0], largest], "delta_min": two_sl},
                                   "unit": UNITS.get(k.replace("_sub", ""), "")}
        out["by_threshold"][str(t)] = entry
    return out


def yardsticks(b: pd.DataFrame, pw: dict) -> dict:
    """Data-internal scales for Delta_min: IQR of each index among surviving satellites and the
    empirical index-vs-log(sigma) slope (OLS, pooled over blinded labels).  No SSP models."""
    out = {"note": "Delta_min expressed relative to (a) the satellite IQR of the index and (b) the empirical slope d(index)/d(log10 sigma) among surviving satellites; both are internal to the data", "by_threshold": {}}
    sat_all = b[(b["is_sat"] == 1) & (b["valid_core"] == 1)]
    for t in [0, 20]:
        sat = sat_all[sat_all["SN"] > t].dropna(subset=["sigma"])
        sat = sat[sat["sigma"] > 0]
        e = pw["by_threshold"][str(t)]
        small = e["two_smallest_labels"][0]
        out["by_threshold"][str(t)] = {}
        for k in IDX_RAW:
            d = sat.dropna(subset=[k])
            if len(d) < 30:
                continue
            x = np.log10(d["sigma"].to_numpy()); y = d[k].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            iqr = float(d[k].quantile(.75) - d[k].quantile(.25))
            dmin = e["indices"].get(k, {}).get("per_blind_sample", {}).get(small, {}).get("delta_min")
            out["by_threshold"][str(t)][k] = {"n": int(len(d)), "IQR": iqr, "slope_per_dex_logsigma": float(slope),
                                              "delta_min_smallest": dmin,
                                              "delta_min_over_IQR": None if dmin is None else dmin / iqr,
                                              "delta_min_as_equiv_dex_logsigma": None if dmin is None or slope == 0 else abs(dmin / slope)}
    return out


def figure(pw: dict, cen: dict) -> str:
    keys = ["HdA", "HgA", "Hb", "Mgb", "Fe5270", "Fe5335", "D4000n", "MgFe", "MgbFe"]
    ts = [t for t in SN_THRESH]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5))
    for ax, k in zip(axes.ravel(), keys):
        # smallest blinded sample (fewest surviving satellites at t=0) and largest
        e0 = pw["by_threshold"]["0"]
        small = e0["two_smallest_labels"][0]
        large = sorted(e0["sizes"], key=lambda l: -e0["sizes"][l]["N_sat"])[0]
        d_small = [pw["by_threshold"][str(t)]["indices"].get(k, {}).get("per_blind_sample", {}).get(small, {}).get("delta_min", np.nan) for t in ts]
        d_two = [pw["by_threshold"][str(t)]["indices"].get(k, {}).get("two_sample_smallest_pair", {}).get("delta_min", np.nan) for t in ts]
        d_sl = [pw["by_threshold"][str(t)]["indices"].get(k, {}).get("two_sample_smallest_vs_largest", {}).get("delta_min", np.nan) for t in ts]
        d_sub = [pw["by_threshold"][str(t)]["indices"].get(f"{k}_sub", {}).get("per_blind_sample", {}).get(small, {}).get("delta_min", np.nan) for t in ts]
        ax.plot(ts, d_small, "o-", color="C0", label=f"1-sample, smallest blind label ({small})")
        ax.plot(ts, d_sl, "s--", color="C0", alpha=0.7, label="2-sample: smallest vs largest")
        ax.plot(ts, d_two, "^:", color="C1", label="2-sample: two smallest")
        ax.plot(ts, d_sub, "x-", color="C2", alpha=0.6, label="1-sample, emission-subtracted index")
        u = UNITS.get(k, "")
        ax.set_ylabel(f"$\\Delta_{{\\min}}$ ({k}) [{u}]" if u else f"$\\Delta_{{\\min}}$ ({k})")
        ax.set_xlabel("S/N threshold (galSpecInfo.sn_median)")
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        nsat = [pw["by_threshold"][str(t)]["sizes"][small]["N_sat"] for t in ts]
        ax2.bar(ts, nsat, width=2.5, color="0.8", alpha=0.5, zorder=0)
        ax2.set_ylabel(f"surviving satellites ({small})", color="0.4")
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        ax.set_title(k)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Minimum detectable group-level mean offset (3$\\sigma$) vs S/N cut — BLINDED labels", y=0.995)
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    path = os.path.join(FIG, "fig_power_vs_SN.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return os.path.relpath(path, HERE)


def main() -> None:
    b = load_blind()
    cen = census(b)
    dec = decisive(b, 20.0)
    eb = error_budget(b)
    pw = power(b)
    ys = yardsticks(b, pw)
    figpath = figure(pw, cen)
    with open(OUT_JSON) as fh:
        doc = json.load(fh)
    doc["phase2"] = {"blinded": True, "input": "work/metallicity_worktable_blind.csv", "sn_thresholds": SN_THRESH,
                     "census": cen, "decisive_SN20": dec, "error_budget": eb, "power": pw, "yardsticks": ys, "figure": figpath}
    with open(OUT_JSON, "w") as fh:
        json.dump(doc, fh, indent=2)
    # console summary
    print("=== S/N census (pooled, deduplicated; require 7 core indices valid) ===")
    print("thr   all    sat    bgg  | per blinded label: sat (N_g>=1sat)")
    for t in SN_THRESH:
        c = cen[str(t)]
        per = "  ".join(f"{lab}:{v['sat']}({v['N_groups_with_ge1_sat']})" for lab, v in c["by_blind_sample"].items())
        print(f"{t:>3}  {c['pooled_dedup']['all']:5d}  {c['pooled_dedup']['sat']:5d}  {c['pooled_dedup']['bgg']:5d}  | {per}")
    print("\n=== decisive (S/N>20, satellites, pooled dedup) ===")
    print(f"surviving {dec['N_sat_surviving']}/{dec['N_sat_full']} = {dec['frac_sat_surviving']:.3f}")
    print("M_r full p25/p50/p75:", [round(dec['M_r_full'][k], 2) for k in ['p25', 'p50', 'p75']], " survivors:", [round(dec['M_r_survivors'][k], 2) for k in ['p25', 'p50', 'p75']])
    print("lgm full p25/p50/p75:", [round(dec['lgm_full'][k], 2) for k in ['p25', 'p50', 'p75']], " survivors:", [round(dec['lgm_survivors'][k], 2) for k in ['p25', 'p50', 'p75']])
    print("survival by M_r quartile (Q1=brightest):", {k: round(v['frac_surv'], 3) for k, v in dec['survival_by_M_r_quartile'].items()})
    print("fraction of survivors in brightest M_r quartile:", round(dec['frac_of_survivors_in_M_r_Q1'], 3))
    print("\n=== power: Delta_min (index units), smallest blinded label, by threshold ===")
    for t in SN_THRESH:
        e = pw["by_threshold"][str(t)]
        small = e["two_smallest_labels"][0]
        row = []
        for k in IDX_RAW:
            d = e["indices"].get(k, {})
            row.append(f"{k}={d.get('per_blind_sample', {}).get(small, {}).get('delta_min', float('nan')):.3f}(rho={d.get('rho_used', float('nan')):.2f})")
        print(f"SN>{t:>2} [{small}: N_g={e['sizes'][small]['N_g']}, n={e['sizes'][small]['n_sat']:.2f}] " + " ".join(row))
    print("figure:", figpath)


if __name__ == "__main__":
    main()
