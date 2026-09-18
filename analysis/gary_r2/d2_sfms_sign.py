"""D2 -- the SFMS sign conflict (gary-r2 Phase 1, read-only).

Raw per-control median SFMS-residual offsets are positive (CG4 above the
main sequence) while the matched residual-sSFR difference is negative.
This script (a) verifies which sample the polynomial is fitted on, (b)
re-derives the matched star-forming pairs and their definition, (c)
recomputes raw medians/means on exactly the matched subsets and (d)
compares star-forming stellar-mass distributions.  No pipeline code is
modified; the matching is re-derived with ``matched_controls.matched_pairs``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from common import (CONTROLS, OUT, SAMPLES, load_results, load_results_build,
                    load_sample, store_diagnostic, write_csv)

import sSFR  # noqa: E402  (src on path via common)
from extended_data import build_galaxy_frame  # noqa: E402
from matched_controls import matched_pairs  # noqa: E402
from size_data import attach_size_columns  # noqa: E402


def main() -> None:
    sample = load_sample()
    results = load_results()
    build = load_results_build()
    out = {}

    # (a) fitting sample -----------------------------------------------------
    sdss = sample["SDSS"]
    sf_ref = sdss.loc[sdss["sSFR_status"].eq("Starforming"), ["lgm", "sSFR"]]
    model = sSFR.fit_ssfr_vs_lgm_poly(sf_ref, order=2)
    # verify the stored MS_res column reproduces this fit on the group samples
    max_dev = 0.0
    for name in SAMPLES:
        g = sample[name + "_Gals"]
        sf = g.loc[g["sSFR_status"].eq("Starforming")]
        pred = sf["sSFR"].to_numpy(float) - model.predict(sf["lgm"])
        max_dev = max(max_dev, float(np.nanmax(np.abs(pred - sf["MS_res"].to_numpy(float)))))
    out["fit"] = {
        "fitted_on": "SDSS non-AGN reference sample (sample['SDSS']), GMM class 'Starforming'",
        "n_fit": int(len(sf_ref)),
        "order": 2,
        "coeffs_highest_first": [float(c) for c in model.coeffs],
        "stored_MS_res_max_abs_deviation_from_refit": max_dev,
        "cv_rms_selected": build.get("Main_Sequence_polyfit_selected_cv_rms"),
        "residual_defined_for": "GMM star-forming galaxies only (NaN otherwise)",
    }
    print("fit:", out["fit"])

    # raw published offsets
    raw = {}
    for ctrl in CONTROLS:
        rec = build[f"pval_MSresiduals_{ctrl}_Gals"]
        raw[ctrl] = {k: rec[k] for k in ("Δmedian", "CI_16", "CI_84", "p_value",
                                          "n_CG4_galaxies", "n_control_galaxies")}
    out["raw_published"] = raw

    # (b) matched pairs ------------------------------------------------------
    frame = build_galaxy_frame(sample)
    try:
        frame, _ = attach_size_columns(frame)
    except Exception:
        pass
    pairs, work, caliper, prepared, variables = matched_pairs(frame)
    n_cg4_matched = results["extended_specialness"]["matched_controls"]["n_cg4_matched"]
    assert len(pairs) == n_cg4_matched, (len(pairs), n_cg4_matched)
    t_idx = [p["treated_index"] for p in pairs]
    c_idx = [p["control_index"] for p in pairs]
    treated = prepared.loc[t_idx].reset_index(drop=True)
    control = prepared.loc[c_idx].reset_index(drop=True)
    both_sf = treated["starforming"].eq(1) & control["starforming"].eq(1)
    t_sf_only = treated["starforming"].eq(1)
    c_sf_only = control["starforming"].eq(1)
    published = results["extended_specialness"]["matched_controls"]["effects"]["residual_sSFR_starforming"]
    tx = treated.loc[both_sf, "MS_res"].to_numpy(float)
    cx = control.loc[both_sf, "MS_res"].to_numpy(float)
    assert int(both_sf.sum()) == published["n_pairs"]
    assert abs(float(np.mean(tx - cx)) - published["delta_cg4_minus_control"]) < 1e-9
    out["matched_definition"] = {
        "pool": "objid-deduplicated pooled controls (RG4 > Control4B > Control4C priority), one global match, not per control",
        "matching_variables": variables,
        "propensity_caliper_logit": caliper,
        "n_pairs_total": len(pairs),
        "sf_restriction": "BOTH members of a pair must be GMM star-forming (starforming==1)",
        "n_pairs_both_sf": int(both_sf.sum()),
        "n_pairs_cg4_sf_only": int(t_sf_only.sum()),
        "n_pairs_control_sf_only": int(c_sf_only.sum()),
        "n_pairs_cg4_sf_control_not_sf": int((t_sf_only & ~c_sf_only).sum()),
        "statistic": "MEAN of paired differences of MS_res (np.mean), bootstrap blocked by treated CG group",
        "published_delta": published["delta_cg4_minus_control"],
        "published_ci95": published["ci95"],
        "published_p": published["p"],
        "matched_control_composition_both_sf": {
            k: int(v) for k, v in control.loc[both_sf, "sample"].value_counts().items()
        },
        "matched_control_composition_both_sf_physical_labels": {
            k: int(v) for k, v in control.loc[both_sf, "control_source_labels"].fillna("").value_counts().items()
        },
        "rank_composition_both_sf": {str(int(k)): int(v) for k, v in treated.loc[both_sf, "rank"].value_counts().sort_index().items()},
    }
    print("matched:", out["matched_definition"])

    # (c) raw statistics recomputed on the matched SF subsets ---------------
    def desc(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return dict(n=0)
        return dict(n=int(len(v)), median=float(np.median(v)), mean=float(np.mean(v)),
                    q16=float(np.quantile(v, 0.16)), q84=float(np.quantile(v, 0.84)),
                    sd=float(np.std(v, ddof=1)) if len(v) > 1 else None)

    sub = {
        "matched_both_sf": {
            "cg4": desc(tx), "control": desc(cx),
            "delta_median": float(np.median(tx) - np.median(cx)),
            "delta_mean": float(np.mean(tx) - np.mean(cx)),
            "paired_median_of_differences": float(np.median(tx - cx)),
            "paired_mean_of_differences": float(np.mean(tx - cx)),
            "wilcoxon_p": float(wilcoxon(tx - cx).pvalue),
            "mannwhitney_p": float(mannwhitneyu(tx, cx).pvalue),
            "n_pairs_with_cg4_above_control": int((tx > cx).sum()),
        },
        "matched_cg4_sf_all_treated": desc(treated.loc[t_sf_only, "MS_res"]),
        "matched_control_sf_all_controls": desc(control.loc[c_sf_only, "MS_res"]),
    }
    # all star-forming per sample (raw) for reference
    full_sf = {}
    for name in SAMPLES:
        g = sample[name + "_Gals"]
        full_sf[name] = desc(g.loc[g["sSFR_status"].eq("Starforming"), "MS_res"])
    sub["full_sample_sf"] = full_sf
    # CG4 SF galaxies: matched vs unmatched (are the matched SF CG4 galaxies special?)
    cg4 = prepared.loc[prepared["is_CG4"] == 1]
    matched_flags = cg4.index.isin(t_idx)
    cg4_sf = cg4.loc[cg4["starforming"].eq(1)]
    sub["cg4_sf_matched_vs_unmatched"] = {
        "matched": desc(cg4_sf.loc[cg4_sf.index.isin(t_idx), "MS_res"]),
        "unmatched": desc(cg4_sf.loc[~cg4_sf.index.isin(t_idx), "MS_res"]),
        "in_both_sf_pairs": desc(tx),
        "matched_but_control_not_sf": desc(
            treated.loc[t_sf_only & ~c_sf_only, "MS_res"]),
    }
    # per-control raw offsets in mean rather than median
    per_control_mean = {}
    for ctrl in CONTROLS:
        g = sample[ctrl + "_Gals"]
        cv = g.loc[g["sSFR_status"].eq("Starforming"), "MS_res"].to_numpy(float)
        tv = sample["CG4_Gals"].loc[sample["CG4_Gals"]["sSFR_status"].eq("Starforming"), "MS_res"].to_numpy(float)
        per_control_mean[ctrl] = {
            "delta_median": float(np.median(tv) - np.median(cv)),
            "delta_mean": float(np.mean(tv) - np.mean(cv)),
            "mannwhitney_p": float(mannwhitneyu(tv, cv).pvalue),
        }
    sub["per_control_full_sf_mean_vs_median"] = per_control_mean
    out["recomputed"] = sub
    print("recomputed:", sub)

    # (d) stellar-mass distributions of SF galaxies per sample --------------
    rows = []
    for name in SAMPLES:
        g = sample[name + "_Gals"]
        sf = g.loc[g["sSFR_status"].eq("Starforming")]
        m = pd.to_numeric(sf["lgm"], errors="coerce").dropna()
        rows.append(dict(sample=name, scope="all_SF", n=len(m), q16=m.quantile(.16),
                         median=m.median(), mean=m.mean(), q84=m.quantile(.84),
                         frac_sat=float((sf["rank_M"] > 1).mean())))
        sat = sf.loc[sf["rank_M"] > 1]
        m = pd.to_numeric(sat["lgm"], errors="coerce").dropna()
        rows.append(dict(sample=name, scope="SF_satellites", n=len(m), q16=m.quantile(.16),
                         median=m.median(), mean=m.mean(), q84=m.quantile(.84), frac_sat=1.0))
    m = treated.loc[both_sf, "logMstar"]
    rows.append(dict(sample="CG4", scope="matched_both_SF", n=len(m), q16=m.quantile(.16),
                     median=m.median(), mean=m.mean(), q84=m.quantile(.84),
                     frac_sat=float((treated.loc[both_sf, "rank"] > 1).mean())))
    m = control.loc[both_sf, "logMstar"]
    rows.append(dict(sample="matched controls", scope="matched_both_SF", n=len(m), q16=m.quantile(.16),
                     median=m.median(), mean=m.mean(), q84=m.quantile(.84),
                     frac_sat=float((control.loc[both_sf, "rank"] > 1).mean())))
    mass = pd.DataFrame(rows)
    write_csv(mass, "d2_sf_stellar_mass_distributions.csv")
    print(mass.to_string())
    out["sf_mass_distributions"] = mass.to_dict(orient="records")

    # residual vs mass: does MS_res depend on mass within the SF samples?
    dep = {}
    for name in SAMPLES:
        g = sample[name + "_Gals"]
        sf = g.loc[g["sSFR_status"].eq("Starforming")]
        lo = sf.loc[sf["lgm"] < 10.3, "MS_res"]
        hi = sf.loc[sf["lgm"] >= 10.3, "MS_res"]
        dep[name] = dict(n_lo=int(lo.notna().sum()), median_lo=float(lo.median()),
                         n_hi=int(hi.notna().sum()), median_hi=float(hi.median()))
    out["ms_res_by_mass_split_10p3"] = dep
    out["mass_split_logMstar"] = 10.3
    print(dep)

    # pair table for the record
    pair_tab = pd.DataFrame({
        "cg4_objid": treated.loc[both_sf, "objid"].to_numpy(),
        "cg4_group": treated.loc[both_sf, "group_uid"].to_numpy(),
        "cg4_rank": treated.loc[both_sf, "rank"].to_numpy(),
        "cg4_logMstar": treated.loc[both_sf, "logMstar"].to_numpy(),
        "cg4_MS_res": tx,
        "control_objid": control.loc[both_sf, "objid"].to_numpy(),
        "control_labels": control.loc[both_sf, "control_source_labels"].to_numpy(),
        "control_logMstar": control.loc[both_sf, "logMstar"].to_numpy(),
        "control_MS_res": cx,
    })
    write_csv(pair_tab, "d2_matched_sf_pairs.csv")

    both = sub["matched_both_sf"]
    explanation = (
        f"The two numbers are different estimands on different samples. (1) The raw offsets are MEDIAN "
        f"differences between all {full_sf['CG4']['n']} star-forming CG4 galaxies and all star-forming "
        f"galaxies of each control ({raw['Control4B']['n_control_galaxies']}, {raw['Control4C']['n_control_galaxies']}, "
        f"{raw['RG4']['n_control_galaxies']}), unmatched in mass/rank. (2) The matched value is the MEAN paired "
        f"difference over only {both['cg4']['n']} pairs in which BOTH members are star-forming, from the pooled "
        f"deduplicated galaxy-level match. On those same {both['cg4']['n']} pairs the median difference is "
        f"{both['delta_median']:+.3f} and the median of paired differences {both['paired_median_of_differences']:+.3f} "
        f"(mean {both['paired_mean_of_differences']:+.3f}; Wilcoxon p = {both['wilcoxon_p']:.2f}; "
        f"{both['n_pairs_with_cg4_above_control']}/{both['cg4']['n']} pairs have CG4 above its control). "
        f"The star-forming CG4 galaxies that enter both-SF pairs have a median residual of "
        f"{both['cg4']['median']:+.3f} against {full_sf['CG4']['median']:+.3f} for all star-forming CG4 galaxies, "
        f"and the matched star-forming controls sit at {both['control']['median']:+.3f} against "
        f"{full_sf['Control4B']['median']:+.3f}/{full_sf['Control4C']['median']:+.3f}/{full_sf['RG4']['median']:+.3f} "
        f"for the full control SF populations. Both signs are therefore correct for what they measure; the "
        f"conflict comes from (i) the small both-SF pair subset, (ii) mean versus median, and (iii) the matched "
        f"controls being a different (mass/rank-matched, pooled) population than the full control SF samples. "
        f"Only the raw Control4C offset reaches p < 0.05 (group-bootstrap p = {raw['Control4C']['p_value']:.3f}; "
        f"Control4B {raw['Control4B']['p_value']:.3f}, RG4 {raw['RG4']['p_value']:.2f}) and the matched contrast does not "
        f"(p = {published['p']:.3f}); this is not a bug. A fourth ingredient is the "
        f"mass dependence of the residuals about the FIELD-fitted main sequence: in every control the "
        f"star-forming galaxies with log M* >= 10.3 sit well below the field relation (median "
        f"{dep['Control4B']['median_hi']:+.2f}, {dep['Control4C']['median_hi']:+.2f}, {dep['RG4']['median_hi']:+.2f}) "
        f"while those below 10.3 sit slightly above it ({dep['Control4B']['median_lo']:+.2f}, "
        f"{dep['Control4C']['median_lo']:+.2f}, {dep['RG4']['median_lo']:+.2f}); the {dep['CG4']['n_hi']} massive "
        f"star-forming CG4 galaxies do not show this depression ({dep['CG4']['median_hi']:+.2f}), which drives the "
        f"positive raw medians, whereas the matched controls of the both-SF pairs are lower-mass than their CG4 "
        f"partners (16th percentiles {mass.loc[mass.scope.eq('matched_both_SF'),'q16'].iloc[1]:.2f} vs "
        f"{mass.loc[mass.scope.eq('matched_both_SF'),'q16'].iloc[0]:.2f}) and therefore inherit positive residuals."
    )
    out["explanation"] = explanation
    print(explanation)
    store_diagnostic("d2_sfms_sign", out)


if __name__ == "__main__":
    main()
