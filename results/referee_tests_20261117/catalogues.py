"""Auxiliary catalogue, missing-data and selection audits."""

from referee_common import *

from itertools import combinations

from astropy.cosmology import Planck15
from astropy.coordinates import Angle, angular_separation
from matched_controls import matched_pairs, matched_cluster_components
from compactness import _angular_matrix
from mass import CONTROLS, join_tempel

HMCG = HERE / "inputs"


def hmcg_tables():
    g = pd.read_csv(
        HMCG / "EugeTableD1.dat",
        sep=r"\s+",
        header=None,
        names=[
            "HMCG",
            "multiplicity",
            "ra",
            "dec",
            "z",
            "diam_arcmin",
            "mu",
            "sigma",
            "bright_r",
            "flag",
        ],
    )
    d = pd.read_csv(
        HMCG / "gal_in_SDSSCG18_ObjID.dat",
        sep=r"\s+",
        header=None,
        names=[
            "HMCG",
            "hm_rank",
            "ra_gal",
            "dec_gal",
            "z_gal",
            "r_hmcg",
            "g_hmcg",
            "objid",
        ],
        dtype={"objid": "int64"},
    )
    assert len(g) == 462 and len(d) == 2070 and g.flag.ne(0).sum() == 56
    assert d.groupby("HMCG").size().eq(g.set_index("HMCG").multiplicity).all()
    return g, d


def purity(f):
    g, d = hmcg_tables()
    inherited = pd.read_csv(DATA / "CG4_Gals.csv")[["objid", "Group"]]
    legacy = d.merge(inherited, on="objid", how="inner")
    assert legacy.Group.eq(legacy.HMCG).all()
    assert len(legacy) == 312
    hit = (
        f[f.cg.eq(0)][["sample", "objid", "Group", "group_uid", "physical_group"]]
        .merge(d[["objid", "HMCG"]], on="objid")
        .merge(g[["HMCG", "multiplicity", "flag"]], on="HMCG")
    )
    assert set(hit.objid).isdisjoint(inherited.objid)
    save("hmcg_control_members", hit)
    rows = []
    for arm in CONTROLS + ["all_controls"]:
        p = hit if arm == "all_controls" else hit[hit["sample"] == arm]
        for label, x in [
            ("any", p),
            ("N_ne_4", p[p.multiplicity.ne(4)]),
            ("flagged", p[p.flag.ne(0)]),
            ("N_eq_4", p[p.multiplicity.eq(4)]),
        ]:
            rows.append(
                dict(
                    sample=arm,
                    kind=label,
                    galaxies=x.objid.nunique(),
                    quartets=x.group_uid.nunique(),
                    hosts=x.physical_group.nunique(),
                    HMCGs=x.HMCG.nunique(),
                )
            )
    save("hmcg_control_counts", rows)
    results = []
    for arm in CONTROLS:
        p = f[f["sample"].isin(["CG4", arm]) & f.is_satellite.eq(1)]
        bad = hit.loc[hit["sample"] == arm, "group_uid"]
        for label, x in [
            ("baseline", p),
            ("exclude_affected_control_quartets", p[~p.group_uid.isin(bad)]),
        ]:
            result, used, w = fit(x, "elliptical ~ cg + " + BASE)
            row = next(r for r in coefficients(result, used, w) if r["term"] == "cg")
            results.append(
                dict(
                    control=arm,
                    restriction=label,
                    **row,
                    **{
                        f"effect_{k}": v
                        for k, v in marginal(result, used, used[used.cg.eq(1)]).items()
                    },
                )
            )
    save("hmcg_purity_morphology", results)
    q = pd.read_csv(TABLES / "quartets.csv")
    lo, hi = pd.read_csv(TABLES / "compact_support.csv").iloc[0][["from_kpc", "to_kpc"]]
    b = f[f["sample"].isin(["CG4", "Control4C"]) & f.is_satellite.eq(1)].merge(
        q[["group_uid", "R_mean", "x"]], on="group_uid", validate="m:1"
    )
    b = b[b.R_mean.between(lo, hi)].assign(xc=lambda x: x.x - np.log10(175))
    bad = hit.loc[hit["sample"] == "Control4C", "group_uid"]
    rows = []
    for lab, part in [("all", b), ("purged", b[~b.group_uid.isin(bad)])]:
        result, used, w = fit(part, "elliptical ~ cg * xc + " + BASE)
        rows.extend(dict(spec=lab, **r) for r in coefficients(result, used, w))
    save("hmcg_purity_compactness", rows)
    return g, d


def circle(xy):
    """Smallest Euclidean enclosing circle in a local sky tangent plane."""
    centers = [*xy, *[(a + b) / 2 for a, b in combinations(xy, 2)]]
    for a, b, c in combinations(xy, 3):
        mat = 2 * np.array([b - a, c - a])
        if abs(np.linalg.det(mat)) > 1e-12:
            centers.append(
                np.linalg.solve(mat, np.array([b @ b - a @ a, c @ c - a @ a]))
            )
    centers = np.array(centers)
    radii = np.linalg.norm(xy - centers[:, None], axis=2).max(1)
    j = radii.argmin()
    return centers[j], radii[j]


def isolation(frame, q, core, excluded, hmcg_g, hmcg_d):
    parent = pd.read_csv(HMCG / "tempel_lowz.csv.gz")
    parent = parent.set_index("objid", drop=False)
    pc = pd.read_csv(DATA / "PC_Gals.csv").set_index("objid")
    pc_xyz = pc[["RA", "Dec", "zobs", "rmag"]].rename(
        columns={"RA": "raj2000", "Dec": "dej2000", "rmag": "mag_r"}
    )
    # Add PC rows absent from Tempel, retaining their exact IDs. The archive
    # does not contain every added non-SDSS spectrum in the original HMCG search.
    phot = pd.concat([parent, pc_xyz[~pc_xyz.index.isin(parent.index)]])
    photo = phot[["mag_r", "zobs"]]
    sky_ra = parent.raj2000.to_numpy()
    sky_dec = parent.dej2000.to_numpy()
    sky_z = parent.zobs.to_numpy()
    sky_r = parent.mag_r.to_numpy()
    sky_ids = parent.objid.to_numpy()
    c4c = q[q["sample"].eq("Control4C")].set_index("Group")
    cg = q[q["sample"].eq("CG4")].set_index("Group")
    h1 = hmcg_g.set_index("HMCG")
    p_cg = frame[frame.cg.eq(1)][["Group", "objid", "RA", "Dec", "z"]]
    p_cg = p_cg.merge(
        hmcg_d[["objid", "HMCG", "z_gal", "r_hmcg"]],
        left_on=["objid", "Group"],
        right_on=["objid", "HMCG"],
        validate="1:1",
    )
    lo, hi = pd.read_csv(TABLES / "compact_support.csv").iloc[0][["from_kpc", "to_kpc"]]
    rows = []
    for arm, part in [("CG4", p_cg), ("Control4C_candidate", core)]:
        for gid, p in part.groupby("Group"):
            ra0, dec0 = p.RA.mean(), p.Dec.mean()
            fac = np.cos(np.deg2rad(dec0))
            xy = np.c_[(p.RA - ra0) * fac, p.Dec - dec0]
            centre, r = circle(xy)
            mag_z = photo.reindex(p.objid)
            mags = mag_z.mag_r.to_numpy()
            redshifts = mag_z.zobs.to_numpy()
            # Use the same Tempel photometry source on both arms. Retain a
            # documented HMCG fallback for sparse cross-match rows.
            if arm == "CG4":
                mags = np.where(np.isfinite(mags), mags, p.r_hmcg)
                redshifts = np.where(np.isfinite(redshifts), redshifts, p.z_gal)
            assert np.isfinite(mags).all() and np.isfinite(redshifts).all()
            z = np.median(redshifts)
            sky_dist_sq = ((sky_ra - ra0) * fac - centre[0]) ** 2 + (
                sky_dec - dec0 - centre[1]
            ) ** 2
            ok = (
                (sky_dist_sq <= (10 * r) ** 2)
                & (np.abs(sky_z - z) * 299792.458 / (1 + z) <= 1000)
                & (sky_r <= mags.min() + 3)
                & ~np.isin(sky_ids, p.objid)
            )
            ratio = np.sqrt(sky_dist_sq[ok].min()) / r if ok.any() else 10.0
            mu = -2.5 * np.log10(np.sum(10 ** (-0.4 * mags))) + 2.5 * np.log10(
                np.pi * (r * 3600) ** 2
            )
            table = cg if arm == "CG4" else c4c
            if gid in table.index:
                r_mean = table.loc[gid, "R_mean"]
            else:
                angles = _angular_matrix(p.RA.to_numpy(), p.Dec.to_numpy())
                r_mean = angles[
                    np.triu_indices(4, 1)
                ].mean() * Planck15.angular_diameter_distance(p.z.median()).to_value(
                    "kpc"
                )
            outcome = (
                "CG4"
                if arm == "CG4"
                else (
                    "Excluded by CG4 overlap"
                    if gid in excluded
                    else "Excluded Lim 3688" if gid == 3688 else "Retained Control4C"
                )
            )
            rows.append(
                dict(
                    sample=arm,
                    group=gid,
                    stage=outcome,
                    R_mean=r_mean,
                    in_overlap=lo <= r_mean <= hi,
                    mu_proxy=mu,
                    bright_r=mags.min(),
                    diam_arcmin=120 * r,
                    ratio_censored_at_10=ratio,
                    n_neighbours_within_3r=(sky_dist_sq[ok] <= (3 * r) ** 2).sum(),
                    passes_isolation_proxy=ratio > 3,
                    passes_surface_brightness_proxy=mu <= 26.33,
                    passes_apparent_concordance=(mags.max() - mags.min()) <= 3,
                    all_members_on_Tempel_index=p.objid.isin(parent.index).all(),
                    hmcg_catalogue_mu=h1.loc[gid, "mu"] if arm == "CG4" else np.nan,
                    hmcg_member_velocity_ok=(
                        np.abs(redshifts - z) * 299792.458 / (1 + z) <= 1000
                    ).all(),
                )
            )
    t = save("hmcg_isolation_quartets", rows)
    ss = []
    for (part, window), p in t.groupby(["stage", "in_overlap"]):
        ss.append(
            dict(
                stage=part,
                in_overlap=window,
                n=len(p),
                mu_median=p.mu_proxy.median(),
                ratio_median=p.ratio_censored_at_10.median(),
                n_pass_isolation=p.passes_isolation_proxy.sum(),
                n_pass_SB=p.passes_surface_brightness_proxy.sum(),
                n_pass_appearance=p.passes_apparent_concordance.sum(),
                n_pass_velocity=p.hmcg_member_velocity_ok.sum(),
                n_Tempel_all=p.all_members_on_Tempel_index.sum(),
            )
        )
    save("hmcg_isolation_summary", ss)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for name, color, mark in [
        ("Retained Control4C", COLORS["Control4C"], "o"),
        ("Excluded by CG4 overlap", "#61707b", "x"),
        ("CG4", COLORS["CG4"], "^"),
    ]:
        p = t[t.stage.eq(name) & t.in_overlap]
        ax.scatter(
            p.mu_proxy,
            p.ratio_censored_at_10,
            s=19,
            marker=mark,
            alpha=0.65,
            c=color,
            label=name,
        )
    ax.axhline(3, c=".5", lw=0.8)
    ax.axvline(26.33, c=".5", lw=0.8)
    ax.set(
        xlabel=r"Tempel-based surface brightness (mag arcsec$^{-2}$)",
        ylabel="Nearest outsider distance / group radius",
    )
    ax.legend()
    figure(fig, "07_hmcg_selection_in_overlap")


def archived_zoo(frame):
    gz = pd.read_csv(HMCG / "gz1_paper1_crossmatch.csv", dtype={"OBJID": "Int64"})
    assert gz.OBJID.is_unique
    gz["RA"] = Angle(gz.RA.to_list(), unit="hourangle").deg
    gz["DEC"] = Angle(gz.DEC.to_list(), unit="deg").deg
    gz = gz.rename(
        columns={"OBJID": "dr7objid", "RA": "GZ1_DR7_RA", "DEC": "GZ1_DR7_Dec"}
    )
    assert frame.dr7objid.dtype == "Int64"
    f = frame.merge(gz, on="dr7objid", how="left", validate="m:1")
    dr7_duplicates = f.groupby("dr7objid").objid.transform("nunique").gt(1)
    f["GZ1_DR7_sep_arcsec"] = (
        np.rad2deg(
            angular_separation(
                *np.deg2rad([f.RA, f.Dec]), *np.deg2rad([f.GZ1_DR7_RA, f.GZ1_DR7_Dec])
            )
        )
        * 3600
    )
    f["GZ1_DR7_verified"] = f.GZ1_DR7_sep_arcsec.le(1.5) & ~dr7_duplicates
    f["GZ1_DR7_label"] = np.where(
        f.P_EL_DEBIASED.gt(0.5), 1, np.where(f.P_CS_DEBIASED.gt(0.5), 0, np.nan)
    )
    f["paper1_recover_E_S"] = f.GZ1_DR7_label.where(f.GZ1_DR7_verified)
    f["expanded_E_S"] = f.elliptical.fillna(f.paper1_recover_E_S)
    counts = []
    for arm, p in f.groupby("sample"):
        missing = p[p.elliptical.isna()]
        valid = p[p.GZ1_DR7_verified & p.elliptical.notna() & p.GZ1_DR7_label.notna()]
        counts.append(
            dict(
                sample=arm,
                baseline_labels=p.elliptical.count(),
                verified_both_labels=len(valid),
                matching_labels=valid.elliptical.eq(valid.GZ1_DR7_label).sum(),
                verified_bridge_among_absent=missing.GZ1_DR7_verified.sum(),
                E_recovered=missing.paper1_recover_E_S.eq(1).sum(),
                S_recovered=missing.paper1_recover_E_S.eq(0).sum(),
                expanded_labels=p.expanded_E_S.count(),
            )
        )
    save("gz1_dr7_recovery_summary", counts)
    save(
        "gz1_dr7_recovered_rows",
        f.loc[
            f.elliptical.isna() & f.paper1_recover_E_S.notna(),
            [
                "sample",
                "objid",
                "dr7objid",
                "group_uid",
                "physical_group",
                "is_satellite",
                "logMstar",
                "NVOTE",
                "GZ1_DR7_sep_arcsec",
                "paper1_recover_E_S",
            ],
        ],
    )
    b = f[f.is_satellite.eq(1) & f["sample"].isin(["CG4", "Control4C"])].copy()
    bounds = pd.read_csv(TABLES / "compact_support.csv").iloc[0][["from_kpc", "to_kpc"]]
    b["xc"] = np.log10(b.R_mean / 175)
    rows = []
    for label, p, rhs in [
        ("full", b, "cg + " + BASE),
        ("shared_compactness", b[b.R_mean.between(*bounds)], "cg * xc + " + BASE),
    ]:
        for outcome in ["elliptical", "expanded_E_S"]:
            result, used, warning = fit(p, outcome + " ~ " + rhs)
            rows.extend(
                dict(sample_window=label, outcome=outcome, **row)
                for row in coefficients(result, used, warning)
            )
    save("gz1_dr7_recovery_models", rows)
    return f


def selection(frame, near, q):
    f = join_tempel(frame.merge(near, on=["group_uid", "objid"], validate="1:1"))
    f = f.merge(q[["group_uid", "R_mean"]], on="group_uid", validate="m:1")
    f["red_colour"] = f.mag_g - f.mag_r
    f["spectral_ID_available"] = f.specobjid.gt(0)
    f["sdss_reference_row"] = f.objid.isin(load()[1]["SDSS_withAGN"].objid)
    f["ES_usable"] = f.elliptical.notna()
    f["finite_gz_votes"] = f[["p_E", "p_S"]].notna().all(axis=1)
    f["close55"] = f.nearest_arcsec.lt(55)
    f["close62"] = f.nearest_arcsec.lt(62)
    f["R_dex"] = np.log10(f.R_mean)
    f["four_colours_finite"] = (
        f[["u_obs", "g_obs", "r_obs", "i_obs"]].notna().all(axis=1)
    )
    f = archived_zoo(f)
    cg = f[f.cg.eq(1)]
    bad = cg[~cg.ES_usable]
    assert len(cg) == 248 and len(bad) == 38
    save(
        "cg4_missing_GZ1",
        bad[
            [
                "objid",
                "Group",
                "specobjid",
                "physical_group",
                "is_satellite",
                "logMstar",
                "z_numeric",
                "nearest_kpc",
                "nearest_arcsec",
                "concentration_r90_r50",
                "red_colour",
                "u_minus_r",
                "spectral_ID_available",
                "sdss_reference_row",
                "close55",
                "close62",
                "p_E",
                "p_S",
                "finite_gz_votes",
                "dr7objid",
                "GZ1_DR7_sep_arcsec",
                "GZ1_DR7_verified",
                "NVOTE",
                "P_EL_DEBIASED",
                "P_CS_DEBIASED",
                "paper1_recover_E_S",
            ]
        ],
    )
    rows = []
    for arm, p in f.groupby("sample"):
        for selection in ["ES_usable", "colour_analysis_retained"]:
            for label, z in p.groupby(selection):
                for v in [
                    "logMstar",
                    "z_numeric",
                    "nearest_kpc",
                    "nearest_arcsec",
                    "concentration_r90_r50",
                    "red_colour",
                    "R_mean",
                    "elliptical",
                    "quenched",
                    "is_satellite",
                    "close55",
                    "close62",
                    "spectral_ID_available",
                    "sdss_reference_row",
                    "four_colours_finite",
                ]:
                    x = z[v].astype(float)
                    rows.append(
                        dict(
                            sample=arm,
                            selection=selection,
                            retained=label,
                            n=len(z),
                            variable=v,
                            n_available=x.count(),
                            mean=x.mean(),
                            median=x.median(),
                        )
                    )
    save("selection_summary", rows)
    cols = []
    for label, var, spec, p in [
        (
            "missing_GZ1",
            "ES_usable",
            "m + z_numeric + R_dex + close55 + is_satellite",
            cg,
        ),
        (
            "colour",
            "colour_analysis_retained",
            "m + z_numeric + elliptical + is_satellite + C(sample)",
            f,
        ),
        (
            "colour_CG4",
            "colour_analysis_retained",
            "m + z_numeric + elliptical + is_satellite",
            cg,
        ),
    ]:
        p = p.assign(selected=p[var].astype(int))
        result, used, w = fit(p, "selected ~ " + spec)
        cols.extend(dict(selection=label, **r) for r in coefficients(result, used, w))
    save("selection_coefficients", cols)
    bounds = []
    for population, part in [
        ("all", f),
        ("satellites", f[f.is_satellite.eq(1)]),
        (
            "satellites_compactness_window",
            f[
                f.is_satellite.eq(1)
                & f.R_mean.between(
                    *pd.read_csv(TABLES / "compact_support.csv").iloc[0][
                        ["from_kpc", "to_kpc"]
                    ]
                )
            ],
        ),
    ]:
        subs = {}
        for arm, p in part.groupby("sample"):
            n = len(p)
            e = p.elliptical.sum()
            missing = p.elliptical.isna().sum()
            subs[arm] = (p.elliptical.mean(), e / n, (e + missing) / n)
            bounds.append(
                dict(
                    population=population,
                    sample=arm,
                    valid=n - missing,
                    n=n,
                    events=e,
                    observed=subs[arm][0],
                    lower=subs[arm][1],
                    upper=subs[arm][2],
                )
            )
        for arm in CONTROLS:
            if arm in subs:
                a, b = subs["CG4"], subs[arm]
                bounds.append(
                    dict(
                        population=population,
                        sample="CG4 − " + arm,
                        observed=a[0] - b[0],
                        lower=a[1] - b[2],
                        upper=a[2] - b[1],
                    )
                )
    save("missing_morphology_bounds", bounds)
    # Common baseline available-case and colour-retained contrasts. This checks
    # sample restriction on the main morphology estimand, separately from colours.
    crows = []
    for arm in CONTROLS:
        p = f[f["sample"].isin(["CG4", arm]) & f.is_satellite.eq(1)]
        for label, x in [
            ("available", p),
            ("colour_retained", p[p.colour_analysis_retained]),
        ]:
            for outcome, kind in [
                ("elliptical", "logit"),
                ("u_minus_r", "ols"),
                ("g_minus_r", "ols"),
            ]:
                result, used, w = fit(x, f"{outcome} ~ cg + {BASE}", kind)
                row = next(
                    r for r in coefficients(result, used, w) if r["term"] == "cg"
                )
                crows.append(
                    dict(
                        control=arm,
                        restriction=label,
                        outcome=outcome,
                        **row,
                        **{
                            f"effect_{k}": v
                            for k, v in marginal(
                                result,
                                used,
                                used[used.cg.eq(1)],
                                kind == "logit",
                                100 if kind == "logit" else 1,
                            ).items()
                        },
                    )
                )
    save("colour_selection_sensitivity", crows)
    plotrows = []
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.9), layout="constrained")
    for j, name in enumerate(["ES_usable", "colour_analysis_retained"]):
        for i, arm in enumerate(SAMPLES):
            rows = []
            for label, p in f[f["sample"] == arm].groupby("mass_bin", observed=True):
                est, lo, hi = bootstrap(p, name)
                rows.append(
                    dict(
                        sample=arm,
                        selection=name,
                        mass_bin=label,
                        estimate=est,
                        low=lo,
                        high=hi,
                        **sizes(p),
                    )
                )
            plotrows.extend(rows)
            z = pd.DataFrame(rows)
            xx = (
                z.mass_bin.map({k: i for i, k in enumerate(MASS_LABELS)}).astype(float)
                + (i - 1.5) * 0.10
            )
            errorbars(axs[j], xx, z, arm, label=arm)
        axs[j].set(
            ylim=(0, 1.03),
            xticks=range(4),
            xticklabels=MASS_LABELS,
            title="Usable GZ1 E/S label" if j == 0 else "Matched photometry retained",
            ylabel="Fraction",
        )
    axs[1].legend(fontsize=8, ncols=2)
    fig.supxlabel(r"$\log_{10}(M_\star/M_\odot)$")
    figure(fig, "08_catalogue_selection")
    save("selection_by_mass", plotrows)


def legacy(frame):
    dtype = {
        x: "Int64" for x in ["source_specobjid", "source_objid", "extra_specobjid"]
    }
    old = pd.read_csv(DATA / "sdss_spectral_provenance_dr12.csv", dtype=dtype)
    total_valid = old.source_sfr.between(-15, 5) & old.source_lgm.between(7, 13)
    alternatives = old.loc[
        total_valid & old.source_specobjid.eq(old.extra_specobjid),
        [
            "source_objid",
            "source_specobjid",
            "source_sfr",
            "source_lgm",
            "run2d",
            "scienceprimary",
            "sn_median",
            "reliable",
        ],
    ].copy()
    alternatives = alternatives.rename(
        columns={"source_objid": "objid", "source_specobjid": "alternate_specobjid"}
    )
    alternatives["source"] = "DR12 spectral provenance cache"
    sdss = load()[1]["SDSS_withAGN"]
    ref = (
        sdss.loc[
            sdss.SFR.between(-15, 5) & sdss.lgm.between(7, 13),
            ["objid", "specObjID", "SFR", "lgm"],
        ]
        .rename(
            columns={
                "specObjID": "alternate_specobjid",
                "SFR": "source_sfr",
                "lgm": "source_lgm",
            }
        )
        .astype({"alternate_specobjid": "Int64"})
    )
    ref["source"] = "DR16 SDSS_withAGN with galSpecExtra join"
    aux = pd.read_csv(
        HMCG / "SDSS_to_Euge.csv", dtype={"specObjID": "Int64", "objid": "Int64"}
    ).rename(
        columns={
            "specObjID": "alternate_specobjid",
            "sfr_tot_p50": "source_sfr",
            "lgm_tot_p50": "source_lgm",
        }
    )
    aux = aux[aux.source_sfr.between(-15, 5) & aux.source_lgm.between(7, 13)][
        ref.columns.drop("source")
    ]
    aux["source"] = "Paper I SDSS_to_Euge"
    alternatives = pd.concat([alternatives, ref, aux], ignore_index=True)
    provenance = pd.read_csv(
        ROOT / "output/ssfr_spectral_provenance.csv",
        usecols=[
            "sample",
            "objid",
            "specobjid",
            "provenance_category",
            "n_spectra_for_objid",
            "run2d",
        ],
        dtype={"specobjid": "Int64"},
    )
    missing = (
        frame[frame.quenched.isna()][
            [
                "sample",
                "objid",
                "specobjid",
                "group_uid",
                "physical_group",
                "is_satellite",
            ]
        ]
        .astype({"specobjid": "Int64"})
        .merge(
            provenance, on=["sample", "objid", "specobjid"], how="left", validate="1:1"
        )
    )
    boss = missing[missing.provenance_category.eq("no_galSpecExtra_sSFR")].copy()
    assert len(boss) == 332
    found = boss.merge(alternatives, on="objid", how="inner")
    found = found[found.alternate_specobjid.ne(found.specobjid)].copy()
    found["candidate_ssfr"] = found.source_sfr - found.source_lgm
    # Only exact-spectrum covered MPA–JHU candidates are eligible. Flag
    # competing IDs and conflicting measurements rather than replacing rows.
    n = found.groupby(["sample", "objid"]).alternate_specobjid.nunique()
    boss = boss.join(n.rename("candidate_spectra"), on=["sample", "objid"])
    boss["candidate_spectra"] = boss.candidate_spectra.fillna(0).astype(int)
    save("boss_sfr_rows", boss)
    save("boss_alternate_candidates", found)
    save(
        "boss_sfr_summary",
        boss.groupby("sample")
        .agg(
            missing_BOSS=("objid", "size"),
            alternative_rows=("candidate_spectra", lambda a: a.gt(0).sum()),
            multiple_candidates=("candidate_spectra", lambda a: a.gt(1).sum()),
            known_repeats=("n_spectra_for_objid", lambda a: a.gt(1).sum()),
        )
        .reset_index(),
    )


def match_audit(frame):
    pairs, work, caliper, prepared, cols = matched_pairs(frame)
    a = prepared.loc[[p["treated_index"] for p in pairs]].reset_index(drop=True)
    b = prepared.loc[[p["control_index"] for p in pairs]].reset_index(drop=True)
    assert a.objid.is_unique and b.objid.is_unique and a["rank"].eq(b["rank"]).all()
    comp = matched_cluster_components(a.physical_group, b.physical_group)
    assert max(p["distance"] for p in pairs) <= caliper
    out = pd.DataFrame(
        {
            "cg_objid": a.objid,
            "control_objid": b.objid,
            "cg_quartet": a.group_uid,
            "control_quartet": b.group_uid,
            "cg_host": a.physical_group,
            "control_host": b.physical_group,
            "rank": a["rank"],
            "cg_outcome": a.elliptical,
            "control_outcome": b.elliptical,
            "component": comp,
            "distance": [p["distance"] for p in pairs],
        }
    )
    out["both_ES"] = out.cg_outcome.notna() & out.control_outcome.notna()
    save("paper_galaxy_pairs", out)
    rows = []
    for population in ["all", "satellites"]:
        ww = work if population == "all" else work[work["rank"].gt(1)]
        cand = 0
        eligible_cg = []
        eligible_controls = []
        for rank, p in ww.groupby("rank"):
            x = p[p.is_CG4.eq(1)]
            y = p[p.is_CG4.eq(0)]
            ok = (
                np.abs(
                    x.propensity_logit.to_numpy()[:, None]
                    - y.propensity_logit.to_numpy()[None, :]
                )
                <= caliper
            )
            cand += ok.sum()
            eligible_cg.extend(x.index[ok.any(1)])
            eligible_controls.extend(y.index[ok.any(0)])
        matching = out if population == "all" else out[out["rank"].gt(1)]
        for stage, r in [
            ("matched", matching),
            ("both_E_S", matching[matching.both_ES]),
        ]:
            row = dict(
                population=population,
                stage=stage,
                n_pairs=len(r),
                candidate_edges=cand,
                complete_cg=ww.is_CG4.eq(1).sum(),
                complete_controls=ww.is_CG4.eq(0).sum(),
                cg_with_candidate=len(eligible_cg),
                controls_with_candidate=len(eligible_controls),
                cg_quartets=r.cg_quartet.nunique(),
                control_quartets=r.control_quartet.nunique(),
                cg_hosts=r.cg_host.nunique(),
                control_hosts=r.control_host.nunique(),
                unique_hosts=len(set(r.cg_host) | set(r.control_host)),
                caliper=caliper,
            )
            if stage == "both_E_S":
                row.update(
                    paired_bootstrap(
                        (r.cg_outcome - r.control_outcome).to_numpy(),
                        r.component.to_numpy(),
                    )
                )
            rows.append(row)
    save("paper_match_denominators", rows)


def run(frame, q, nn, core, excluded):
    g, d = purity(frame)
    isolation(frame, q, core, excluded, g, d)
    selection(frame, nn, q)
    legacy(frame)
    match_audit(frame)
