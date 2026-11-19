"""Mass interactions, magnitude completeness sensitivity and SF diagnostics."""

from referee_common import *

from astropy.cosmology import Planck15
from statsmodels.stats.multitest import multipletests

CONTROLS = ["Control4C", "Control4B", "RG4"]
ENDPOINTS = {
    "elliptical": ("E-class fraction", "mean"),
    "quenched": ("Quenched fraction", "mean"),
    "strong_halpha": (r"Strong H$\alpha$ fraction", "mean"),
    "Dn4000": (r"Median $D_n4000$", "median"),
    "log_Rchl_r_kpc": (r"Median $\log_{10}(R_e/\mathrm{kpc})$", "median"),
    "total_ssfr": ("Median total log sSFR", "median"),
}


def join_tempel(frame):
    cache = pd.read_csv(HERE / "inputs/tempel_lowz.csv.gz").drop_duplicates("objid")
    return frame.merge(
        cache[["objid", "mag_r", "mag_g", "zobs"]],
        on="objid",
        how="left",
        validate="m:1",
    )


def completeness(sat):
    # An empirical high-M/L envelope from unique red Lim–Tempel parent
    # galaxies, including fainter members beyond the quartet eligibility cut.
    pc = pd.read_csv(DATA / "PC_Gals.csv")
    pc = join_tempel(pc)
    pc["mass_to_light_proxy"] = pc.lgm_tot_p50 + 0.4 * pc.M_r
    pc["colour"] = pc.mag_g - pc.mag_r
    valid = (
        pc.lgm_tot_p50.between(7, 13)
        & pc.mag_r.between(10, 17.77)
        & pc.colour.between(0.75, 1.3)
    )
    parent = pc[valid].copy()
    # m_r minus M_r minus distance modulus empirically reconciles the Tempel
    # photometry and Paper I's catalogue absolute-magnitude system.
    parent["kext_offset"] = parent.mag_r - parent.M_r - Planck15.distmod(parent.z).value
    mur = parent.mass_to_light_proxy.quantile(0.95)
    kext = parent.kext_offset.quantile(0.95)
    save(
        "completeness_calibration",
        [
            dict(
                red_g_minus_r_low=0.75,
                red_g_minus_r_high=1.3,
                n=len(parent),
                hosts=parent.Group.nunique(),
                ml_q90=parent.mass_to_light_proxy.quantile(0.90),
                ml_q95=mur,
                ml_q99=parent.mass_to_light_proxy.quantile(0.99),
                offset_q50=parent.kext_offset.median(),
                offset_q95=kext,
            )
        ],
    )
    s = sat.copy()
    s["z_selection"] = s.z_numeric
    s["M_faint_flux"] = 17.77 - Planck15.distmod(s.z_selection).value - kext
    s["M_faint_bgg"] = s.M_BGG + 3
    s["M_faint_lim"] = np.minimum(s.M_faint_flux, s.M_faint_bgg)
    s["mass_limit"] = mur - 0.4 * s.M_faint_lim
    s["mass_limit_flux"] = mur - 0.4 * s.M_faint_flux
    s["complete_region"] = s.logMstar.ge(s.mass_limit)
    s["complete_flux_only"] = s.logMstar.ge(s.mass_limit_flux)
    save(
        "completeness_per_satellite",
        s[
            [
                "sample",
                "objid",
                "group_uid",
                "physical_group",
                "z_selection",
                "logMstar",
                "M_faint_flux",
                "M_faint_bgg",
                "mass_limit",
                "mass_limit_flux",
                "complete_region",
                "complete_flux_only",
            ]
        ],
    )
    rows = []
    for (arm, massbin), p in s.groupby(
        ["sample", "mass_bin"], observed=True, dropna=False
    ):
        rows.append(
            dict(
                sample=arm,
                bin=massbin,
                all=len(p),
                retained=p.complete_region.sum(),
                flux_only=p.complete_flux_only.sum(),
                n_bgg_limit=p.M_faint_bgg.le(p.M_faint_flux).sum(),
                threshold_q10=p.mass_limit.quantile(0.1),
                threshold_median=p.mass_limit.median(),
                threshold_q90=p.mass_limit.quantile(0.9),
                **sizes(p[p.complete_region]),
                cg_valid_E_S=p.loc[p.complete_region, "elliptical"].count(),
                cg_valid_Q=p.loc[p.complete_region, "quenched"].count(),
            )
        )
    save("completeness_flow", rows)
    return s


def local_mass(s, mass):
    p = s[(s.logMstar - mass).abs() <= 0.25]
    cg = p[p.cg.eq(1)]
    cc = p[p.cg.eq(0)]
    from compactness import cdiv

    cols = ["z_numeric", "log_group_luminosity", "velocity_dispersion"]
    ok = cdiv(cg, cc, cols, np.array([0.006, 0.35, 120.0]))
    return cg.iloc[ok.any(1)], cc.iloc[ok.any(0)]


def mass_models(sat):
    terms, contrasts, flow, ceiling, tests, failures = [], [], [], [], [], []
    fig, axs = plt.subplots(
        2, 3, figsize=(11, 6.2), sharex=True, sharey=True, layout="constrained"
    )
    for j, control in enumerate(CONTROLS):
        for k, outcome in enumerate(["elliptical", "quenched"]):
            ax = axs[k, j]
            for region, p in [
                ("primary", sat),
                ("flux_only", sat[sat.complete_flux_only]),
                ("flux_and_BGG", sat[sat.complete_region]),
            ]:
                p = p[p["sample"].isin(["CG4", control])].dropna(
                    subset=[
                        outcome,
                        "m",
                        "z_numeric",
                        "log_group_luminosity",
                        "velocity_dispersion",
                    ]
                )
                limits = p.groupby("cg").logMstar.agg(["min", "max"])
                lo, hi = limits["min"].max(), limits["max"].min()
                supported = p[p.logMstar.between(lo, hi)]
                flow.append(
                    dict(
                        control=control,
                        outcome=outcome,
                        region=region,
                        mass_low=lo,
                        mass_high=hi,
                        **counts(supported),
                    )
                )
                for model, formula in [
                    (
                        "interaction",
                        f"{outcome} ~ cg * m + z_numeric + log_group_luminosity + velocity_dispersion",
                    ),
                    ("constant_OR", f"{outcome} ~ cg + {BASE}"),
                ]:
                    try:
                        result, used, w = fit(supported, formula)
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                        failures.append(
                            dict(
                                control=control,
                                outcome=outcome,
                                region=region,
                                model=model,
                                error=str(e),
                            )
                        )
                        continue
                    terms.extend(
                        dict(
                            outcome=outcome,
                            control=control,
                            region=region,
                            model=model,
                            **b,
                        )
                        for b in coefficients(result, used, w)
                    )
                    if model == "interaction":
                        tests.append(
                            dict(
                                outcome=outcome,
                                control=control,
                                region=region,
                                beta=result.params["cg:m"],
                                p=result.pvalues["cg:m"],
                            )
                        )
                    subset = []
                    for mass in np.unique(
                        np.r_[
                            np.linspace(lo, hi, 75), [9.75, 10, 10.25, 10.5, 10.75, 11]
                        ]
                    ):
                        if not lo <= mass <= hi:
                            continue
                        a, b = local_mass(used, mass)
                        if (
                            min(a.physical_group.nunique(), b.physical_group.nunique())
                            < 5
                        ):
                            continue
                        row = dict(
                            outcome=outcome,
                            control=control,
                            region=region,
                            model=model,
                            mass=mass,
                            **marginal(result, used, a.assign(m=mass - 10.25)),
                            **counts(pd.concat([a, b])),
                        )
                        subset.append(row)
                    contrasts.extend(subset)
                    if model == "interaction" and region == "primary" and subset:
                        z = pd.DataFrame(subset)
                        ax.plot(z.mass, z.difference, color=COLORS["CG4"])
                        ax.fill_between(
                            z.mass, z.low, z.high, color=COLORS["CG4"], alpha=0.18
                        )
                        for side, color, y in [
                            (1, COLORS["CG4"], 0.02),
                            (0, COLORS[control], 0.055),
                        ]:
                            ax.plot(
                                used.loc[used.cg.eq(side), "logMstar"],
                                np.full(used.cg.eq(side).sum(), y),
                                "|",
                                c=color,
                                alpha=0.22,
                                transform=ax.get_xaxis_transform(),
                                markersize=4,
                            )
                    if outcome == "quenched" and model == "constant_OR":
                        from scipy.special import logit

                        tgt = used[used.cg.eq(1)]
                        for bin_label, bp in [
                            ("pooled", tgt),
                            *list(tgt.groupby("mass_bin", observed=True)),
                        ]:
                            if bp.empty:
                                continue
                            cov = bp.assign(cg=0)
                            p0 = result.predict(cov).clip(1e-8, 1 - 1e-8)
                            p1 = expit(logit(p0) + np.log(1.5))
                            ceiling.append(
                                dict(
                                    control=control,
                                    region=region,
                                    bin=bin_label,
                                    observed_cg=bp.quenched.mean(),
                                    n=len(bp),
                                    counterfactual_control=p0.mean(),
                                    implied_delta_pp_for_OR_1_5=100 * (p1 - p0).mean(),
                                    **marginal(result, used, bp),
                                )
                            )
            ax.axhline(0, c=".5", lw=0.8)
            ax.axvspan(10, 10.5, color=".7", alpha=0.15)
            if k == 0:
                ax.set_title(f"vs {control}")
            if j == 0:
                ax.set_ylabel(
                    ("E-class" if k == 0 else "Quenched") + " difference (pp)"
                )
    fig.supxlabel(r"$\log_{10}(M_\star/M_\odot)$")
    figure(fig, "05_adjusted_mass_interactions")
    save("mass_model_coefficients", terms)
    save("mass_marginal_contrasts", contrasts)
    save("mass_interaction_flow", flow)
    save("quenching_ceiling", ceiling)
    save(
        "mass_model_failures",
        pd.DataFrame(
            failures, columns=["control", "outcome", "region", "model", "error"]
        ),
    )
    t = pd.DataFrame(tests)
    t["p_Holm"] = t.groupby("region").p.transform(
        lambda a: multipletests(a, method="holm")[1]
    )
    save("mass_interaction_tests", t)

    rows = []
    for control in CONTROLS:
        both = sat[sat["sample"].isin(["CG4", control])]
        for outcome in ["elliptical", "quenched"]:
            d = both.dropna(
                subset=[
                    outcome,
                    "m",
                    "z_numeric",
                    "log_group_luminosity",
                    "velocity_dispersion",
                ]
            )
            limits = d.groupby("cg").logMstar.agg(["min", "max"])
            d = d[d.logMstar.between(limits["min"].max(), limits["max"].min())]
            base = f"{outcome} ~ cg * m + z_numeric + log_group_luminosity"
            specifications = [("omit_velocity_dispersion", d, base)]
            if outcome == "quenched":
                g = d[d.elliptical.notna()]
                baseline = base + " + velocity_dispersion"
                specifications.extend(
                    [
                        ("ES_complete_baseline", g, baseline),
                        ("with_ES_covariate", g, baseline + " + elliptical"),
                    ]
                )
            for name, part, formula in specifications:
                result, used, w = fit(part, formula)
                rows.extend(
                    dict(control=control, outcome=outcome, spec=name, **r)
                    for r in coefficients(result, used, w)
                )
    save("mass_model_sensitivities", rows)


def sf_checks(sat):
    # Validate the sign against positive emission-line flux, and against the
    # exact DR16 spectrum when an exact ID survives the join.
    lines = pd.read_csv(DATA / "galspecline_dr16.csv", dtype={"specobjid": "Int64"})
    assert lines.specobjid.is_unique
    phot = load()[1]["SDSS_withAGN"][["objid", "specObjID"]].drop_duplicates("objid")
    phot = phot.astype({"specObjID": "Int64"})
    p = sat.astype({"specobjid": "Int64"}).merge(
        phot, on="objid", how="left", validate="m:1"
    )
    p["same_spectrum"] = (
        p.specObjID.astype("Int64").eq(p.specobjid.astype("Int64")).fillna(False)
    )
    p = p.merge(
        lines[["specobjid", "h_alpha_eqw", "h_alpha_flux"]].rename(
            columns={"h_alpha_eqw": "exact_EW", "h_alpha_flux": "exact_flux"}
        ),
        on="specobjid",
        how="left",
        validate="m:1",
    )
    p["strong_exact"] = np.where(
        p.exact_EW.gt(-9000), p.exact_EW.le(-3).astype(float), np.nan
    )
    p["aperture"] = np.log10(1.5 / p.petroR50_r.where(p.petroR50_r.gt(0)))
    p["recent_disagreement"] = np.where(
        p.quenched.notna() & p.strong_halpha.notna(),
        p.quenched.eq(p.strong_halpha).astype(float),
        np.nan,
    )
    p["weak_and_SF"] = np.where(
        p.recent_disagreement.notna(),
        (p.quenched.eq(0) & p.strong_halpha.eq(0)).astype(float),
        np.nan,
    )
    p["strong_and_Q"] = np.where(
        p.recent_disagreement.notna(),
        (p.quenched.eq(1) & p.strong_halpha.eq(1)).astype(float),
        np.nan,
    )
    cover = []
    for (arm, bin_label), d in p.groupby(
        ["sample", "mass_bin"], observed=True, dropna=False
    ):
        valid = d.exact_flux.gt(0) & d.exact_EW.gt(-9000)
        cover.append(
            dict(
                sample=arm,
                bin=bin_label,
                total=len(d),
                EW=d.strong_halpha.count(),
                EW_exact=d.strong_exact.count(),
                same_stored_ID_EW=(d.same_spectrum & d.strong_halpha.notna()).sum(),
                exact_EW_negative_positive_flux=d.loc[valid, "exact_EW"].lt(0).sum(),
                exact_positive_flux=valid.sum(),
                Dn4000=d.Dn4000.count(),
                sSFR=d.total_ssfr.count(),
                Q=d.quenched.count(),
                aperture=d.aperture.count(),
                disagreement=d.recent_disagreement.count(),
                weak_and_SF=d.weak_and_SF.sum(),
                strong_and_Q=d.strong_and_Q.sum(),
                n_fibre_ssfr=0,
            )
        )
    save("sf_coverage", cover)
    rows = []
    for control in CONTROLS:
        base = p[p["sample"].isin(["CG4", control]) & p.mass_bin.eq("10–10.5")]
        for outcome, family in [
            ("strong_halpha", "logit"),
            ("strong_exact", "logit"),
            ("quenched", "logit"),
            ("total_ssfr", "ols"),
            ("Dn4000", "ols"),
            ("recent_disagreement", "logit"),
        ]:
            for selection, label in [
                (base, "available"),
                (
                    base.dropna(subset=["aperture", outcome, "recent_disagreement"]),
                    "same_aperture_complete",
                ),
            ]:
                use_ap = "same_aperture_complete" == label
                for ap in [False, True] if use_ap else [False]:
                    formula = f"{outcome} ~ cg + m + z_numeric" + (
                        " + aperture" if ap else ""
                    )
                    try:
                        model, used, w = fit(selection, formula, family)
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                        rows.append(
                            dict(
                                control=control,
                                outcome=outcome,
                                selection=label,
                                aperture=ap,
                                warning=str(e),
                            )
                        )
                        continue
                    for row in coefficients(model, used, w):
                        if row["term"] != "cg":
                            continue
                        effect = marginal(
                            model,
                            used,
                            used[used.cg.eq(1)],
                            family == "logit",
                            100 if family == "logit" else 1,
                        )
                        rows.append(
                            dict(
                                control=control,
                                outcome=outcome,
                                selection=label,
                                aperture=ap,
                                **row,
                                **{f"effect_{k}": v for k, v in effect.items()},
                            )
                        )
    save("sf_mass_adjusted", rows)


def run(frame):
    sat = frame.loc[frame.is_satellite.eq(1)].copy()
    sat["strong_halpha"] = np.where(
        sat.h_alpha_eqw.gt(-9000), sat.h_alpha_eqw.le(-3).astype(float), np.nan
    )
    sat["total_ssfr"] = sat.sSFR.where(sat.quenched.notna())
    sat = completeness(sat)
    rows = []
    for region, work in [
        ("full", sat),
        ("flux_and_BGG", sat[sat.complete_region]),
        ("flux_only", sat[sat.complete_flux_only]),
    ]:
        for (arm, mbin), p in work.groupby(
            ["sample", "mass_bin"], observed=True, dropna=False
        ):
            for name, (_, stat) in ENDPOINTS.items():
                est, lo, hi = bootstrap(p, name, stat)
                d = p.dropna(subset=[name])
                rows.append(
                    dict(
                        region=region,
                        sample=arm,
                        mass_bin=mbin,
                        outcome=name,
                        n_all=len(p),
                        events=(
                            d[name].sum()
                            if name in ["elliptical", "quenched", "strong_halpha"]
                            else np.nan
                        ),
                        estimate=est,
                        low=lo,
                        high=hi,
                        **sizes(d),
                        median_mass=d.logMstar.median(),
                    )
                )
    bins = save("satellite_mass_bins", rows)
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6), layout="constrained")
    for ax, (name, (title, stat)) in zip(axs.flat, ENDPOINTS.items()):
        for j, arm in enumerate(SAMPLES):
            r = bins[
                (bins.region == "full")
                & (bins["sample"] == arm)
                & (bins.outcome == name)
                & bins.mass_bin.notna()
                & (bins.hosts >= 3)
            ].copy()
            xx = (
                r.mass_bin.map({k: i for i, k in enumerate(MASS_LABELS)}).astype(float)
                + (j - 1.5) * 0.10
            )
            errorbars(ax, xx, r, arm, label=arm)
        ax.set(title=title, xticks=np.arange(4), xticklabels=MASS_LABELS)
        if name in ["elliptical", "quenched", "strong_halpha"]:
            ax.set_ylim(0, 1.04)
    axs[0, 0].legend(ncols=2, fontsize=8)
    fig.supxlabel(r"$\log_{10}(M_\star/M_\odot)$")
    figure(fig, "04_satellite_mass_diagnostics")
    mass_models(sat)
    sf_checks(sat)
    return sat
