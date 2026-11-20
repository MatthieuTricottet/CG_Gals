"""Compactness-window star formation, host environment, and bin audits."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, ranksums
from statsmodels.stats.outliers_influence import variance_inflation_factor

from agn_environment import _classify
from compactness import COMPACT_BINS, COSMO, _angular_matrix, select_control4c_quartets
from freeze import (
    BASE,
    CENTRE,
    FROZEN,
    ORIGINAL,
    MASS_REF,
    arm_sizes,
    bootstrap,
    common_window,
    fit,
    model_rows,
    pair_outcome,
    save,
    sizes,
)
from host_controlled import _decision_boundary
from referee_common import predicted
from run import PRIOR, ROOT
from scipy.stats import t

BPT_CLASSES = ["starforming", "composite", "AGN_like", "unclassified"]
OUTCOMES = ("strong_halpha", "strong_exact", "quenched")


def spectral_frame(frame):
    lines = pd.read_csv(
        ROOT / "data/galspecline_dr16.csv", dtype={"specobjid": "Int64"}
    )
    assert lines.specobjid.is_unique
    lines = lines.rename(columns={"h_alpha_eqw": "exact_EW"})
    fields = [
        "specobjid",
        "exact_EW",
        "h_alpha_flux",
        "h_beta_flux",
        "nii_6584_flux",
        "oiii_5007_flux",
    ]
    historical = _classify(frame)
    f = historical.merge(
        lines[fields].add_prefix("cache_"),
        left_on="specobjid",
        right_on="cache_specobjid",
        how="left",
        validate="m:1",
    )
    f["strong_halpha"] = (
        f.h_alpha_eqw.le(-3).astype(float).where(f.h_alpha_eqw.gt(-9000))
    )
    f["strong_exact"] = (
        f.cache_exact_EW.le(-3).astype(float).where(f.cache_exact_EW.gt(-9000))
    )
    f["aperture"] = np.log10(1.5 / f.petroR50_r.where(f.petroR50_r.gt(0)))
    f["midmass"] = f.logMstar.ge(10) & f.logMstar.lt(10.5)
    f["common"] = (
        f[["quenched", "strong_halpha", "strong_exact", "aperture"]].notna().all(axis=1)
    )
    boundary = _decision_boundary()
    assert boundary is not None
    f["ssfr_gmm_offset"] = (f.sSFR - boundary(f.logMstar)).where(f.quenched.notna())
    # Match BPT line provenance to the EW join. Use the same two curves
    # and positive finite fluxes for the stored-spectrum diagnostic.
    raw = f[
        [
            "cache_" + x
            for x in ["h_alpha_flux", "h_beta_flux", "nii_6584_flux", "oiii_5007_flux"]
        ]
    ]
    positive = raw.where(raw.gt(0))
    exact_bpt = pd.DataFrame(
        {
            "log_NII_Ha": np.log10(
                positive.cache_nii_6584_flux / positive.cache_h_alpha_flux
            ),
            "log_OIII_Hb": np.log10(
                positive.cache_oiii_5007_flux / positive.cache_h_beta_flux
            ),
        }
    )
    f["exact_bpt_class"] = _classify(exact_bpt).bpt_class
    return f


def matched_frame(panel, pairs):
    links = pd.concat(
        [
            pairs[["pair", "component", "cg_quartet"]].rename(
                columns={"cg_quartet": "group_uid"}
            ),
            pairs[["pair", "component", "control_quartet"]].rename(
                columns={"control_quartet": "group_uid"}
            ),
        ]
    )
    return panel.merge(links, on="group_uid", how="inner", validate="m:1")


def complete_matched_pairs(frame, outcome, morph, aperture):
    cols = [
        outcome,
        "m",
        "z_numeric",
        "xc",
        "log_group_luminosity",
        "velocity_dispersion",
    ]
    if morph:
        cols.append("elliptical")
    if aperture:
        cols.append("aperture")
    used = frame.dropna(subset=cols)
    keeps = used.groupby("pair").cg.nunique()
    return used[used.pair.isin(keeps[keeps.eq(2)].index)]


def star_formation(panel, matches):
    rows = []
    for labelset in [FROZEN, ORIGINAL]:
        f = panel.assign(
            elliptical=panel.elliptical if labelset == FROZEN else panel.original_E
        )
        for scope, p in [("satellites", f), ("10_to_10.5", f[f.midmass])]:
            for outcome in OUTCOMES:
                bpt = "exact_bpt_class" if outcome == "strong_exact" else "bpt_class"
                specs = [
                    ("baseline", False, False, "none"),
                    ("morphology", True, False, "none"),
                    ("morphology_common", True, False, "none"),
                    ("aperture", True, True, "none"),
                ]
                specs += [
                    (base, True, ap, exclusion)
                    for base, ap in [
                        ("morphology", False),
                        ("morphology_common", False),
                        ("aperture", True),
                    ]
                    for exclusion in ["Kewley_AGN", "Kauffmann_AGN_composite"]
                ]
                for spec, morph, ap, exclusion in specs:
                    r = p if spec in ["baseline", "morphology"] else p[p.common]
                    if exclusion == "Kewley_AGN":
                        r = r[r[bpt].ne("AGN_like")]
                    if exclusion == "Kauffmann_AGN_composite":
                        r = r[~r[bpt].isin(["AGN_like", "composite"])]
                    rhs = (
                        BASE
                        + (" + elliptical" if morph else "")
                        + (" + aperture" if ap else "")
                    )
                    metadata = dict(
                        labelset=labelset,
                        scope=scope,
                        outcome=outcome,
                        spec=spec,
                        exclusion=exclusion,
                        bpt_source=bpt if exclusion != "none" else None,
                    )
                    for design, match in [("window", None), *matches.items()]:
                        if match is None:
                            used = r
                        else:
                            used = complete_matched_pairs(
                                matched_frame(r, match), outcome, morph, ap
                            )
                            rows.append(
                                pair_outcome(
                                    used,
                                    match,
                                    outcome,
                                    dict(**metadata, design=design, slope="unadjusted"),
                                )
                            )
                        for slope, terms in [
                            ("common", "cg + xc"),
                            ("interaction", "cg * xc"),
                        ]:
                            results, _, _ = model_rows(
                                used,
                                outcome,
                                terms + " + " + rhs,
                                dict(
                                    **metadata,
                                    design=design,
                                    slope=slope,
                                    eligible_pairs=(
                                        used.pair.nunique()
                                        if match is not None
                                        else None
                                    ),
                                ),
                                (CENTRE,),
                            )
                            rows.extend(results)
    save("03_compactness_ha_quenching", rows)


def discordance(frame):
    rows = []
    cohorts = []
    for labelset, binary in [(ORIGINAL, "original_E"), (FROZEN, "elliptical")]:
        for source, classifier, bpt in [
            ("historical", "strong_halpha", "bpt_class"),
            ("exact", "strong_exact", "exact_bpt_class"),
        ]:
            p = frame[frame.midmass].dropna(subset=[classifier, "quenched", "aperture"])
            for arm, g in p.groupby("sample"):
                for strong in (0, 1):
                    for quenched in (0, 1):
                        cell = g[g[classifier].eq(strong) & g.quenched.eq(quenched)]
                        base = dict(
                            labelset=labelset,
                            sample=arm,
                            ew_source=source,
                            strong=strong,
                            quenched=quenched,
                            cohort_n=len(g),
                            **sizes(cell),
                        )
                        for key in ("ssfr_gmm_offset", "Dn4000", "aperture"):
                            observed = cell.dropna(subset=[key])
                            base[f"median_{key}"] = (
                                observed[key].median() if len(observed) else np.nan
                            )
                            base.update(
                                {f"{key}_{k}": v for k, v in sizes(observed).items()}
                            )
                        base.update({b: cell[bpt].eq(b).sum() for b in BPT_CLASSES})
                        base.update(
                            E=cell[binary].eq(1).sum(),
                            S=cell[binary].eq(0).sum(),
                            GZ1_unresolved=cell[binary].isna().sum(),
                        )
                        rows.append(base)
                if labelset == FROZEN:
                    cohorts.append(
                        g[
                            [
                                "objid",
                                "specobjid",
                                "sample",
                                "group_uid",
                                "physical_host_id",
                                classifier,
                                "quenched",
                                "ssfr_gmm_offset",
                                "Dn4000",
                                "aperture",
                                "original_label",
                                "frozen_label",
                                bpt,
                            ]
                        ]
                        .rename(columns={classifier: "strong", bpt: "bpt"})
                        .assign(ew_source=source)
                    )
    save("05_ha_ssfr_discordance_audit", rows)
    save("05_ha_ssfr_discordance_rows", pd.concat(cohorts))


def describe_hosts(q, hosts, window):
    selection = pd.read_csv(PRIOR / "tables/hmcg_isolation_quartets.csv")
    selection["group_uid"] = np.where(
        selection["sample"].eq("CG4"), "CG4:", "Control4C:"
    ) + selection.group.astype(str)
    lookup = q.set_index("group_uid")
    selection["med_Rij"] = selection.group_uid.map(lookup.med_Rij)
    # The rejected candidates are absent from the final galaxy frame.
    # Reconstruct their four members using the original Control4C selection.
    excluded = selection.loc[selection.stage.eq("Excluded by CG4 overlap"), "group"]
    parent = pd.read_csv(ROOT / "data/PC_Gals.csv")
    cores = select_control4c_quartets(parent[parent.Group.isin(excluded)])
    excluded_separations = {}
    for gid, members in cores.groupby("Group"):
        angles = _angular_matrix(members.RA.to_numpy(), members.Dec.to_numpy())
        distance = COSMO.angular_diameter_distance(members.z.mean()).to_value("kpc")
        excluded_separations[gid] = np.median(angles[np.triu_indices(4, 1)] * distance)
    rejected = selection.stage.eq("Excluded by CG4 overlap")
    selection.loc[rejected, "med_Rij"] = selection.loc[rejected, "group"].map(
        excluded_separations
    )
    assert selection.loc[selection.stage.ne("Excluded Lim 3688"), "med_Rij"].notna().all()
    selection = selection.loc[selection.med_Rij.between(*window)].drop(
        columns=["in_overlap", "R_mean"]
    )
    is_cg = selection["sample"].eq("CG4")
    selection["physical_group"] = "Lim:" + selection.group.astype(str)
    selection.loc[is_cg, "physical_group"] = selection.loc[is_cg, "group_uid"].map(
        lookup.physical_group
    )
    selection = selection.merge(
        hosts[["physical_host_id", "host_mass", "N_Lim", "N_Lim_Tempel"]],
        left_on="physical_group",
        right_on="physical_host_id",
        validate="m:1",
    )
    is_cg = selection["sample"].eq("CG4")
    selection["class"] = selection.group_uid.map(lookup["class"]).where(is_cg)
    all_cg = selection[is_cg]
    overlap = selection[selection.stage.eq("Excluded by CG4 overlap")]
    retained = selection[selection.stage.eq("Retained Control4C")]
    passed = retained[retained.passes_isolation_proxy]
    failed = retained[~retained.passes_isolation_proxy]
    assert set(all_cg.group_uid) == set(
        q.loc[q.window & q["sample"].eq("CG4"), "group_uid"]
    )
    assert set(retained.group_uid) == set(
        q.loc[q.window & q["sample"].eq("Control4C"), "group_uid"]
    )
    save("04_host_environment_quartets", selection)
    descriptors = {
        "CG4": all_cg,
        "excluded_cores": overlap,
        "retained": retained,
        "retained_isolation_pass": passed,
        "retained_isolation_fail": failed,
        **{f"CG4_{kind}": g for kind, g in all_cg.groupby("class")},
    }
    rows = []
    comparison_names = [
        ("retained", "excluded_cores"),
        ("retained", "CG4"),
        ("excluded_cores", "CG4"),
        ("retained_isolation_pass", "retained_isolation_fail"),
    ]
    comparison_names += [
        ("retained", key) for key in descriptors if key.startswith("CG4_")
    ]
    for col in ("host_mass", "N_Lim", "N_Lim_Tempel"):
        for name, p in descriptors.items():
            rows.append(
                dict(
                    row="distribution",
                    sample=name,
                    term=col,
                    median=p[col].median(),
                    q25=p[col].quantile(0.25),
                    q75=p[col].quantile(0.75),
                    n=4 * len(p),
                    quartets=len(p),
                    hosts=p.physical_group.nunique(),
                )
            )
        for name, other in comparison_names:
            p, r = descriptors[name], descriptors[other]
            stacked = pd.concat([p.assign(cg=1.0), r.assign(cg=0.0)], ignore_index=True)
            stacked["ranks"] = rankdata(stacked[col])
            # Ordinary rank-sum is displayed as a descriptive reference.
            # Inference is on the midranks with shared-host clustering.
            mod, used, warning = fit(stacked, "ranks ~ cg", family="ols")
            ci = mod.conf_int().loc["cg"]
            rows.append(
                dict(
                    row="rank_sum",
                    sample=name,
                    control=other,
                    term=col,
                    rank_sum_p_uncorrected=ranksums(p[col], r[col]).pvalue,
                    p=mod.pvalues["cg"],
                    estimate=mod.params["cg"],
                    low=ci.iloc[0],
                    high=ci.iloc[1],
                    n=4 * len(used),
                    quartets=len(used),
                    hosts=used.physical_group.nunique(),
                    cg_n=4 * len(p),
                    cg_quartets=len(p),
                    cg_hosts=p.physical_group.nunique(),
                    control_n=4 * len(r),
                    control_quartets=len(r),
                    control_hosts=r.physical_group.nunique(),
                    shared_hosts=len(set(p.physical_group) & set(r.physical_group)),
                )
            )
    return rows


def environment_models(frame, q, window, hosts):
    rows = describe_hosts(q, hosts, window)
    cc = frame[frame["sample"].isin(["CG4", "Control4C"]) & frame.window]
    for labelset in [FROZEN, ORIGINAL]:
        p = cc.assign(elliptical=cc.elliptical if labelset == FROZEN else cc.original_E)
        for outcome in ("elliptical", *OUTCOMES):
            for spec, columns in [
                ("baseline", ()),
                ("mass", ("host_mass",)),
                ("richness", ("host_richness",)),
                ("both", ("host_mass", "host_richness")),
            ]:
                for slope, terms in [("common", "cg + xc"), ("interaction", "cg * xc")]:
                    results, model, obs = model_rows(
                        p,
                        outcome,
                        " + ".join([terms, BASE, *columns]),
                        dict(
                            analysis="CG4-Control4C",
                            labelset=labelset,
                            outcome=outcome,
                            spec=spec,
                            slope=slope,
                        ),
                        (CENTRE,),
                        columns,
                    )
                    rows.extend(results)
                    if spec == "both" and labelset == FROZEN and model is not None:
                        exog = model.model.exog
                        labels = list(model.model.exog_names)
                        for col in (*columns, "log_group_luminosity"):
                            rows.append(
                                dict(
                                    row="collinearity",
                                    analysis="CG4-Control4C",
                                    labelset=labelset,
                                    outcome=outcome,
                                    spec=spec,
                                    slope=slope,
                                    term=col,
                                    vif=variance_inflation_factor(
                                        exog, labels.index(col)
                                    ),
                                    corr_luminosity=obs[col].corr(
                                        obs.log_group_luminosity
                                    ),
                                    corr_mass=obs[col].corr(obs.host_mass),
                                    corr_richness=obs[col].corr(obs.host_richness),
                                    **arm_sizes(obs),
                                )
                            )
    overlap = common_window(q, ["RG4", "Control4C"])
    midpoint = np.sqrt(overlap[0] * overlap[1])
    rg = frame[
        frame["sample"].isin(["RG4", "Control4C"]) & frame.med_Rij.between(*overlap)
    ].copy()
    rg["cg"] = rg["sample"].eq("RG4").astype(float)
    rg_hosts = set(rg.loc[rg.cg.eq(1), "physical_group"])
    disjoint = rg[rg.cg.eq(1) | ~rg.physical_group.isin(rg_hosts)]
    for cohort, r in [("overlapping_catalogues", rg), ("disjoint_hosts", disjoint)]:
        for labelset, binary in [(FROZEN, "elliptical"), (ORIGINAL, "original_E")]:
            for outcome in ("elliptical", *OUTCOMES):
                for spec, cols in [
                    ("baseline", ()),
                    ("mass", ("host_mass",)),
                    ("richness", ("host_richness",)),
                    ("both", ("host_mass", "host_richness")),
                ]:
                    fit_input = r.assign(
                        elliptical=r[binary], xc=r.xc - np.log10(midpoint / CENTRE)
                    )
                    results, _, _ = model_rows(
                        fit_input,
                        outcome,
                        " + ".join(["cg + xc + m + z_numeric", *cols]),
                        dict(
                            analysis="RG4-Control4C",
                            labelset=labelset,
                            outcome=outcome,
                            spec=spec,
                            slope="common",
                            cohort=cohort,
                            window_low=overlap[0],
                            window_high=overlap[1],
                        ),
                    )
                    rows.extend(results)
    save("04_host_environment_window_check", rows)


def probability(result, used, target):
    prob, gradient = predicted(result, target)
    se = np.sqrt(max(0, gradient @ result.cov_params() @ gradient))
    half = t.ppf(0.975, used.physical_group.nunique() - 1) * se
    return prob, max(0, prob - half), min(1, prob + half)


def compactness_bins(sat, q):
    rows = []
    for arm, sample in sat.groupby("sample"):
        quartet = q[q["sample"].eq(arm)]
        for bin_index, (low, high) in enumerate(
            zip(COMPACT_BINS[:-1], COMPACT_BINS[1:])
        ):
            core = quartet[quartet.x.gt(low) & quartet.x.le(high)]
            p = sample[sample.group_uid.isin(core.group_uid)]
            for labelset, binary in [(FROZEN, "elliptical"), (ORIGINAL, "original_E")]:
                labelled = p.dropna(subset=[binary])
                est, lo, hi = bootstrap(labelled, binary)
                info = dict(
                    sample=arm,
                    labelset=labelset,
                    bin_left=round(low, 2),
                    bin_right=round(high, 2),
                    bin_index=bin_index,
                    sampled_x=p.x.median(),
                    stage_n=len(p),
                    stage_quartets=len(core),
                    stage_hosts=core.physical_group.nunique(),
                    raw=est,
                    raw_low=lo,
                    raw_high=hi,
                    **sizes(labelled),
                    median_mass=labelled.logMstar.median(),
                    median_z=labelled.z_numeric.median(),
                    median_N_Lim=core.N_Lim.median(),
                    median_host_mass=core.host_mass.median(),
                )
                rows.append(info)
    tbin = pd.DataFrame(rows)
    rises = []
    sat = sat.assign(bin=pd.cut(sat.x, COMPACT_BINS, labels=False))
    for (arm, labelset), dest in tbin.groupby(["sample", "labelset"]):
        binary = "elliptical" if labelset == FROZEN else "original_E"
        use = sat[sat["sample"].eq(arm)].dropna(subset=["bin", "m", binary]).copy()
        variation = use.groupby("bin")[binary].nunique()
        mixed = use[use.bin.isin(variation[variation.eq(2)].index)]
        mod, used, _ = fit(mixed, f"{binary} ~ C(bin) + m")
        for idx, entry in dest.iterrows():
            selected = use[use.bin.eq(entry.bin_index)]
            model_bin = used[used.bin.eq(entry.bin_index)]
            lo, hi = selected.logMstar.min(), selected.logMstar.max()
            tbin.loc[
                idx,
                [
                    "mass_complete_n",
                    "mass_complete_quartets",
                    "mass_complete_hosts",
                    "mass_low",
                    "mass_high",
                    "fit_n",
                    "fit_quartets",
                    "fit_hosts",
                ],
            ] = [
                len(selected),
                selected.group_uid.nunique(),
                selected.physical_group.nunique(),
                lo,
                hi,
                len(used),
                used.group_uid.nunique(),
                used.physical_group.nunique(),
            ]
            if model_bin.physical_group.nunique() < 3 or not lo <= MASS_REF <= hi:
                tbin.loc[idx, "standardization_status"] = (
                    "insufficient within-bin support"
                )
                continue
            pred = pd.DataFrame(
                {"bin": [float(entry.bin_index)], "m": [MASS_REF - 10.25]}
            )
            tbin.loc[idx, ["adjusted", "adjusted_low", "adjusted_high"]] = probability(
                mod, used, pred
            )
            tbin.loc[idx, "standardization_status"] = "ok"
        if arm in ("Control4B", "Control4C"):
            lower = dest[np.isclose(dest.bin_left, 2.8)].iloc[0]
            upper = dest[np.isclose(dest.bin_left, 3.0)].iloc[0]
            ref = pd.DataFrame(
                {
                    "bin": [float(lower.bin_index), float(upper.bin_index)],
                    "m": MASS_REF - 10.25,
                }
            )
            a, ga = predicted(mod, ref.iloc[:1])
            b, gb = predicted(mod, ref.iloc[1:])
            gradient = gb - ga
            se = np.sqrt(max(0, gradient @ mod.cov_params() @ gradient))
            half = t.ppf(0.975, used.physical_group.nunique() - 1) * se
            upper_n = sizes(used[used.bin.eq(upper.bin_index)])
            lower_n = sizes(used[used.bin.eq(lower.bin_index)])
            rises.append(
                dict(
                    sample=arm,
                    labelset=labelset,
                    difference=100 * (b - a),
                    low=100 * (b - a - half),
                    high=100 * (b - a + half),
                    **sizes(used),
                    **{f"upper_{k}": v for k, v in upper_n.items()},
                    **{f"lower_{k}": v for k, v in lower_n.items()},
                )
            )
    save("06_outer_bin_contrasts", rises)
    return save("06_compactness_bins_audit", tbin)


def analyse(frame, q, window, host_table, matches):
    sat = spectral_frame(frame[frame.is_satellite.eq(1)])
    panel = sat[sat["sample"].isin(["CG4", "Control4C"])]
    star_formation(panel[panel.window], matches)
    environment_models(sat, q, window, host_table)
    return compactness_bins(sat, q)
