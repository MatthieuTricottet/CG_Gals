"""Quartet geometry, common support, matching and compactness regression."""

from referee_common import *

from astropy.cosmology import Planck15
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tidal_indices import _angular_matrix
from sample_construction import select_control4c_quartets, COSMO
from matched_controls import matched_cluster_components
from second_order_battery import _satellite_frame
from extended_stats import fit_logistic_model
from T9_compactness_host_bgg import _overlap_coefficient

CALIPERS = {"x": 0.08, "quartet_z": 0.005, "sat_mass": 0.25, "bgg_abs": 0.55}
RCENTRE = 175.0
COMPACT_BINS = np.arange(1.4, 3.61, 0.2)


def geometry(frame):
    rows, neighbours = [], []
    for uid, p in frame.groupby("group_uid"):
        assert len(p) == 4 and set(p["rank"]) == {1, 2, 3, 4}
        p = p.sort_values("rank")
        angle = _angular_matrix(p.RA.to_numpy(), p.Dec.to_numpy())
        z = p.z.median()
        da = Planck15.angular_diameter_distance(z).to_value("kpc")
        dl = Planck15.luminosity_distance(z).to_value("kpc")
        np.testing.assert_allclose(dl / da, (1 + z) ** 2)
        ij = np.triu_indices(4, 1)
        r = angle[ij] * da
        r_table1 = angle[ij] * COSMO.angular_diameter_distance(p.z.mean()).to_value(
            "kpc"
        )
        for oid, near in zip(p.objid, np.where(np.eye(4), np.inf, angle).min(1)):
            neighbours.append(
                dict(
                    group_uid=uid,
                    objid=oid,
                    nearest_kpc=near * da,
                    nearest_arcsec=np.rad2deg(near) * 3600,
                )
            )
        rows.append(
            dict(
                group_uid=uid,
                sample=p["sample"].iloc[0],
                Group=p.Group.iloc[0],
                physical_group=p.physical_group.iloc[0],
                quartet_z=z,
                R_mean=r.mean(),
                R_median=np.median(r),
                table1_R_median=np.median(r_table1),
                DA_kpc=da,
                DL_kpc=dl,
                bgg_abs=p.M_r.iloc[0],
                sat_mass=p.loc[p.is_satellite.eq(1), "logMstar"].mean(),
                sat_mass_n=p.loc[p.is_satellite.eq(1), "logMstar"].count(),
                log_group_luminosity=p.log_group_luminosity.iloc[0],
                velocity_dispersion=p.velocity_dispersion.iloc[0],
                **{f"R{i+1}{j+1}_kpc": value for (i, j), value in zip(zip(*ij), r)},
            )
        )
    return pd.DataFrame(rows).assign(
        x=lambda p: np.log10(p.R_mean), xmed=lambda p: np.log10(p.R_median)
    ), pd.DataFrame(neighbours)


def candidates(q, compact=True, radius_caliper=0.08, include_sigma=False):
    tolerances = {**CALIPERS, "x": radius_caliper}
    if include_sigma:
        tolerances["velocity_dispersion"] = 120.0
    cols = [k for k in tolerances if compact or k != "x"]
    t1 = q[q["sample"].eq("CG4") & q.sat_mass_n.eq(3)].dropna(subset=cols)
    t0 = q[q["sample"].eq("Control4C") & q.sat_mass_n.eq(3)].dropna(subset=cols)
    tol = np.array([tolerances[k] for k in cols])
    distance = cdist(t1[cols] / tol, t0[cols] / tol)
    valid = cdist(t1[cols] / tol, t0[cols] / tol, metric="chebyshev") <= 1 + 1e-12
    valid &= (
        t1.physical_group.to_numpy()[:, None] != t0.physical_group.to_numpy()[None, :]
    )
    return t1, t0, valid, distance, cols


def matching(q, sat):
    means = sat.groupby("group_uid").elliptical.agg(["mean", "count"])
    status, balance, effects, pairs, pool = [], [], [], [], []
    for label, sub, compact, tolerance in [
        ("without_compactness_full", q, False, 0.08),
        ("without_compactness_same_pool", q[q.window], False, 0.08),
        ("compactness_matched", q[q.window], True, 0.08),
        ("tighter_radius", q[q.window], True, 0.05),
        ("wider_radius", q[q.window], True, 0.12),
        ("velocity_balanced", q[q.window], True, 0.08),
    ]:
        a, b, ok, distance, cols = candidates(
            sub, compact, tolerance, label == "velocity_balanced"
        )
        cost = np.where(ok, distance, 1e6)
        i, j = linear_sum_assignment(np.c_[cost, np.full((len(a), len(a)), 1000.0)])
        use = j < len(b)
        i, j = i[use], j[use]
        assert ok[i, j].all() and len(np.unique(j)) == len(j)
        left, right = a.iloc[i].reset_index(drop=True), b.iloc[j].reset_index(drop=True)
        lm, rm = [
            means.reindex(x.group_uid).reset_index(drop=True) for x in (left, right)
        ]
        observed = lm["mean"].notna() & rm["mean"].notna()
        components = matched_cluster_components(
            left.physical_group, right.physical_group
        )
        estimates = paired_bootstrap(
            (lm["mean"] - rm["mean"])[observed].to_numpy(), components[observed]
        )
        effects.append(
            dict(
                spec=label,
                pairs=len(i),
                pairs_both_outcomes=observed.sum(),
                cg_hosts=left.physical_group.nunique(),
                cc_hosts=right.physical_group.nunique(),
                cg_hosts_valid=left[observed].physical_group.nunique(),
                cc_hosts_valid=right[observed].physical_group.nunique(),
                cg_labels=lm.loc[observed, "count"].sum(),
                cc_labels=rm.loc[observed, "count"].sum(),
                cg_fraction=lm.loc[observed, "mean"].mean(),
                cc_fraction=rm.loc[observed, "mean"].mean(),
                **estimates,
            )
        )
        status.append(
            dict(
                spec=label,
                radius_caliper=tolerance if compact else np.nan,
                pool_cg=sub["sample"].eq("CG4").sum(),
                pool_cc=sub["sample"].eq("Control4C").sum(),
                complete_cg=len(a),
                complete_cc=len(b),
                admissible_edges=ok.sum(),
                admissible_cg=ok.any(1).sum(),
                admissible_cc=ok.any(0).sum(),
                matched_cg=len(i),
                matched_cc=len(j),
            )
        )
        for column in [*CALIPERS, "log_group_luminosity", "velocity_dispersion"]:
            scale = np.sqrt((a[column].var() + b[column].var()) / 2)
            for stage, x, y in [
                ("pool", a, b),
                ("matched", left, right),
                ("both_E_S", left[observed], right[observed]),
            ]:
                balance.append(
                    dict(
                        spec=label,
                        stage=stage,
                        covariate=column,
                        cg=x[column].mean(),
                        cc=y[column].mean(),
                        smd=(x[column].mean() - y[column].mean()) / scale,
                    )
                )
        for part, feasible, selected in [(a, ok.any(1), left), (b, ok.any(0), right)]:
            pool.append(
                part[["group_uid", "physical_group"]].assign(
                    spec=label,
                    has_candidate=feasible,
                    selected=part.group_uid.isin(selected.group_uid),
                )
            )
        for k in range(len(i)):
            pairs.append(
                dict(
                    spec=label,
                    cg_quartet=left.group_uid[k],
                    cc_quartet=right.group_uid[k],
                    cg_host=left.physical_group[k],
                    cc_host=right.physical_group[k],
                    cg_labels=lm["count"][k],
                    cc_labels=rm["count"][k],
                    component=components[k],
                    cg_fraction=lm["mean"][k],
                    cc_fraction=rm["mean"][k],
                    **{
                        f"diff_{c}": left[c][k] - right[c][k]
                        for c in [*CALIPERS, "velocity_dispersion"]
                    },
                )
            )
    save("compact_matching_flow", status)
    save("compact_matching_balance", balance)
    save("compact_matching_effects", effects)
    save("compact_matching_pairs", pairs)
    save("compact_matching_eligibility", pd.concat(pool))


def appendix_c(sample):
    pc = pd.read_csv(DATA / "PC_Gals.csv")
    full = pd.read_csv(DATA / "CG4_Gals.csv")
    core = select_control4c_quartets(pc)
    excluded = core.loc[core.objid.isin(full.objid), "Group"].unique()
    final = core[~core.Group.isin([*excluded, 3688])]
    assert len(excluded) == 61
    assert set(zip(final.Group, final.objid)) == set(
        zip(sample["Control4C_Gals"].Group, sample["Control4C_Gals"].objid)
    )
    overlap = core.loc[core.objid.isin(full.objid)].merge(
        full[["Group", "objid"]], on="objid", suffixes=("_Lim", "_CG4")
    )
    assert len(overlap) == 233
    save("appendix_c_exclusions", overlap)
    return core, excluded


def run(frame, sample):
    q, nn = geometry(frame)
    core, excluded = appendix_c(sample)
    save("quartets", q)
    b = q[q["sample"].isin(["CG4", "Control4C"])].copy()
    vals = (
        b.groupby("sample")
        .R_mean.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        .unstack()
    )
    lo, hi = vals[0.05].max(), vals[0.95].min()
    assert lo < RCENTRE < hi
    b["window"] = b.R_mean.between(lo, hi)
    save("compact_quantiles", vals.reset_index())
    summary = []
    for arm, p in b.groupby("sample"):
        other = vals.drop(index=arm).iloc[0]
        summary.append(
            dict(
                sample=arm,
                n=len(p),
                hosts=p.physical_group.nunique(),
                within_other_90=p.R_mean.between(other[0.05], other[0.95]).sum(),
                within_other_full=p.R_mean.between(other[0], other[1]).sum(),
                both_central_90=p.window.sum(),
                hosts_both_central_90=p[p.window].physical_group.nunique(),
                from_kpc=lo,
                to_kpc=hi,
                **{
                    f"n_{k}_{k+50}": p.R_mean.between(k, k + 50, inclusive="left").sum()
                    for k in [100, 150, 200]
                },
            )
        )
    save("compact_support", summary)
    save(
        "compact_overlap",
        [
            dict(
                metric=key,
                bins=n,
                overlap=_overlap_coefficient(
                    b.loc[b["sample"].eq("CG4"), key].to_numpy(),
                    b.loc[b["sample"].eq("Control4C"), key].to_numpy(),
                    n,
                ),
            )
            for key in ["x", "xmed"]
            for n in [20, 40, 60]
        ],
    )
    a, c, ok, _, _ = candidates(b[b.window])
    joint = set(a.group_uid[ok.any(1)]) | set(c.group_uid[ok.any(0)])
    b["joint_quartet"] = b.group_uid.isin(joint)
    sat = frame.loc[frame.is_satellite.eq(1)].merge(
        q[["group_uid", "x", "xmed", "R_mean", "R_median"]], on="group_uid"
    )
    sat["xc"] = sat.x - np.log10(RCENTRE)
    sat["xcm"] = sat.xmed - np.log10(RCENTRE)
    sat["window"] = sat.R_mean.between(lo, hi)
    sat["comp_bin"] = pd.cut(sat.x, COMPACT_BINS)
    save(
        "compact_support_covariates",
        b.groupby(["sample", "window", "joint_quartet"])[
            [
                "quartet_z",
                "bgg_abs",
                "sat_mass",
                "log_group_luminosity",
                "velocity_dispersion",
            ]
        ]
        .agg(["count", "min", "median", "max"])
        .stack(level=0, future_stack=True)
        .reset_index(),
    )
    bp = sat[sat["sample"].isin(["CG4", "Control4C"])]
    # Local counterfactual support in the model covariates, beyond quartet averages.
    joint_cols = ["xc", "z_numeric", "m", "log_group_luminosity", "velocity_dispersion"]
    joint_tol = np.array([0.10, 0.006, 0.25, 0.35, 120.0])
    work = bp[bp.window & bp.elliptical.notna()].dropna(subset=joint_cols)
    t1 = work[work.cg.eq(1)]
    t0 = work[work.cg.eq(0)]
    ok = cdiv(t1, t0, joint_cols, joint_tol)
    ids = set(t1.index[ok.any(1)]) | set(t0.index[ok.any(0)])
    stages = [
        ("full", bp),
        ("common_compactness", bp[bp.window]),
        ("quartet_joint_support", bp[bp.group_uid.isin(joint)]),
        ("satellite_joint_E_S_support", work.loc[sorted(ids)]),
    ]
    save(
        "compact_model_flow",
        [
            dict(
                stage=name,
                **sizes(p),
                **counts(p),
                **{f"es_{k}": v for k, v in counts(p[p.elliptical.notna()]).items()},
            )
            for name, p in stages
        ],
    )
    matching(b, sat)
    coefs, contrasts, binrows, appendix, influence = [], [], [], [], []
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    for name, p in b.groupby("sample"):
        ax.hist(
            p.x,
            bins=COMPACT_BINS,
            density=True,
            histtype="step",
            lw=2,
            color=COLORS[name],
            label=name,
        )
    ax.axvspan(
        np.log10(lo),
        np.log10(hi),
        color=".85",
        alpha=0.5,
        label="Shared central 90% range",
    )
    ax.set(
        xlabel=r"$\log_{10}(\overline{R}_{ij}/\mathrm{kpc})$",
        ylabel="Quartet density per dex",
    )
    ax.legend()
    figure(fig, "01_compactness_overlap")
    fig, axs = plt.subplots(
        2, 2, figsize=(8, 6), sharex=True, sharey=True, layout="constrained"
    )
    for name, ax in zip(SAMPLES, axs.flat):
        p = sat[sat["sample"].eq(name)]
        app = _satellite_frame(sample, name, 0.5)
        old = fit_logistic_model(
            app,
            "elliptical",
            ["log_R_pair_med", "logMstar"],
            continuous=["log_R_pair_med", "logMstar"],
        )
        term = old["terms"]["log_R_pair_med"]
        appendix.append(
            dict(
                sample=name,
                n=old["n"],
                quartet_clusters=old["n_clusters"],
                OR_per_sample_SD=term["odds_ratio"],
                low=term["ci95"][0],
                high=term["ci95"][1],
                p=term["p"],
            )
        )
        for key in ["xc", "xcm"]:
            result, used, w = fit(p, f"elliptical ~ {key} + m")
            coefs.extend(
                dict(spec=f"single_{key}", sample=name, **r)
                for r in coefficients(result, used, w)
            )
            if key == "xc":
                # Display the mass-standardized curve only where the sample is populated.
                xs = np.linspace(*used.x.quantile([0.05, 0.95]), 90)
                ys = [
                    predicted(result, used.assign(xc=x - np.log10(RCENTRE)))[0]
                    for x in xs
                ]
                ax.plot(xs, ys, color=COLORS[name], lw=1.5)
        for key, part in p.groupby("comp_bin", observed=True):
            est, l, h = bootstrap(part, "elliptical")
            binrows.append(
                dict(
                    sample=name,
                    bin_left=key.left,
                    bin_right=key.right,
                    x=part.x.median(),
                    estimate=est,
                    low=l,
                    high=h,
                    **sizes(part.dropna(subset=["elliptical"])),
                )
            )
        pp = pd.DataFrame(binrows).query("sample == @name and hosts >= 3")
        errorbars(ax, pp.x, pp, name)
        ax.set(title=name, ylim=(0, 1), xlim=(COMPACT_BINS[0], COMPACT_BINS[-1]))
    fig.supxlabel(r"$\log_{10}(\overline{R}_{ij}/\mathrm{kpc})$")
    fig.supylabel("Satellite E-class fraction")
    figure(fig, "02_morphology_compactness")
    save("appendix_f_reproduction", appendix)
    save("compact_morphology_bins", binrows)
    fig, ax = plt.subplots(figsize=(7, 4))
    specs = [
        ("full_no_compactness", bp, f"cg + {BASE}", "xc"),
        ("window_no_compactness", bp[bp.window], f"cg + {BASE}", "xc"),
        ("common_slope", bp[bp.window], f"cg + xc + {BASE}", "xc"),
        ("interaction", bp[bp.window], f"cg * xc + {BASE}", "xc"),
        ("minimal_interaction", bp[bp.window], "cg * xc + m + z_numeric", "xc"),
        (
            "exclude_sigma",
            bp[bp.window],
            "cg * xc + m + z_numeric + log_group_luminosity",
            "xc",
        ),
        ("quadratic_mass", bp[bp.window], f"cg * xc + {BASE} + I(m**2)", "xc"),
        ("median_interaction", bp[bp.window], f"cg * xcm + {BASE}", "xcm"),
        ("joint_satellite_support", work.loc[sorted(ids)], f"cg * xc + {BASE}", "xc"),
        (
            "without_shared_hosts",
            bp[
                bp.window
                & ~bp.physical_group.isin(
                    set(bp.loc[bp.cg.eq(1), "physical_group"])
                    & set(bp.loc[bp.cg.eq(0), "physical_group"])
                )
            ],
            f"cg * xc + {BASE}",
            "xc",
        ),
    ]
    for label, p, formula, key in specs:
        result, used, w = fit(p, "elliptical ~ " + formula)
        coefs.extend(
            dict(spec=label, sample="CG4–Control4C", **r)
            for r in coefficients(result, used, w)
        )
        if label.startswith("full_"):
            contrasts.append(
                dict(
                    spec=label,
                    kpc=np.nan,
                    **marginal(result, used, used[used.cg.eq(1)]),
                )
            )
            continue
        # For each coordinate use a locally supported outcome-complete CG target.
        # Both sides must have measurements within 0.10 dex and at least 3 hosts.
        xs = np.unique(np.r_[np.linspace(lo, hi, 46), [150, 175, 200, 225, 250]])
        rows = []
        for r in xs:
            x = np.log10(r / RCENTRE)
            if not lo <= r <= hi:
                continue
            local = used[(used[key] - x).abs() <= 0.10]
            lt = local[local.cg.eq(1)]
            lc = local[local.cg.eq(0)]
            ok = cdiv(lt, lc, joint_cols[1:], joint_tol[1:])
            lt, lc = lt.iloc[ok.any(1)], lc.iloc[ok.any(0)]
            if min(lt.physical_group.nunique(), lc.physical_group.nunique()) < 3:
                continue
            row = dict(
                spec=label,
                kpc=r,
                **counts(pd.concat([lt, lc])),
                **marginal(result, used, lt.assign(**{key: x})),
            )
            contrasts.append(row)
            rows.append(row)
        if label == "interaction":
            local = used[used[key].abs() <= 0.10]
            left, right = local[local.cg.eq(1)], local[local.cg.eq(0)]
            ok = cdiv(left, right, joint_cols[1:], joint_tol[1:])
            target = left.iloc[ok.any(1)].assign(**{key: 0.0})
            for host in used.loc[used.cg.eq(1), "physical_group"].unique():
                sensitivity, obs, warning = fit(
                    used[used.physical_group.ne(host)], "elliptical ~ " + formula
                )
                # Fix the covariate target so re-estimation drives the range.
                influence.append(
                    dict(
                        host_left_out=host,
                        beta=sensitivity.params["cg"],
                        slope_interaction=sensitivity.params["cg:xc"],
                        warning=warning,
                        **marginal(sensitivity, obs, target),
                    )
                )
        if label in ["common_slope", "interaction"]:
            rows = pd.DataFrame(rows)
            color = COLORS["CG4" if label == "interaction" else "Control4C"]
            ax.plot(
                rows.kpc,
                rows.difference,
                c=color,
                label=label.replace("_", " ").capitalize(),
            )
            ax.fill_between(rows.kpc, rows.low, rows.high, color=color, alpha=0.17)
    ax.axhline(0, color=".3", lw=0.8)
    ax.set(
        xlabel="Mean projected pair separation (kpc)",
        ylabel="CG4 − Control4C E fraction (pp)",
    )
    ax.legend()
    figure(fig, "03_adjusted_compactness_contrast")
    save("compact_treated_host_influence", influence)
    save("compact_model_coefficients", coefs)
    save("compact_marginal_contrasts", contrasts)
    return q, nn, core, excluded


def cdiv(left, right, columns, tolerance):
    return (
        cdist(left[columns] / tolerance, right[columns] / tolerance, metric="chebyshev")
        <= 1 + 1e-12
    )
