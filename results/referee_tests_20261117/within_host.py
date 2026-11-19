"""Reconstruct assignment and audit within-Lim-host identification."""

from referee_common import *

from astropy.cosmology import Planck15
from astropy.coordinates import angular_separation
from scipy.stats import norm
from statsmodels.discrete.conditional_models import ConditionalLogit
from host_controlled import build_host_frame
from sample_construction import DMAG_MAX


def separation(p, ra, dec):
    return angular_separation(
        *np.deg2rad([p.RA, p.Dec]), np.deg2rad(ra), np.deg2rad(dec)
    )


def overlap(p, radius):
    a = p.loc[p.is_CG_member.eq(1), radius].to_numpy()
    b = p.loc[p.is_CG_member.eq(0), radius].to_numpy()
    if not len(a) or not len(b):
        return dict(
            overlap=False,
            gap=np.nan,
            within25=0,
            within50=0,
            cg_in_range=0,
            other_in_range=0,
        )
    dist = np.abs(a[:, None] - b[None, :])
    lo, hi = max(a.min(), b.min()), min(a.max(), b.max())
    return dict(
        overlap=bool(lo <= hi),
        gap=dist.min(),
        within25=(dist <= 25).sum(),
        within50=(dist <= 50).sum(),
        cg_in_range=((a >= lo) & (a <= hi)).sum(),
        other_in_range=((b >= lo) & (b <= hi)).sum(),
    )


def fit_host(panel, label, radii, covars=("logMstar", "rank_parent")):
    variables = ["is_CG_member", *covars, *radii]
    c = panel.dropna(subset=["elliptical", *variables])
    n_unique = c.groupby("host_lim_group")[["elliptical", "is_CG_member"]].nunique()
    outcome_hosts = n_unique.index[n_unique.elliptical.eq(2)]
    both_hosts = n_unique.index[n_unique.eq(2).all(axis=1)]
    d = c[c.host_lim_group.isin(outcome_hosts)]
    notes = dict(
        spec=label,
        n_complete=len(c),
        hosts=c.host_lim_group.nunique(),
        n=len(d),
        outcome_informative=len(outcome_hosts),
        both_informative=len(both_hosts),
        n_with_both_variation=d.host_lim_group.isin(both_hosts).sum(),
        cg_complete=c.is_CG_member.sum(),
        noncg_complete=(1 - c.is_CG_member).sum(),
        cg=d.is_CG_member.sum(),
        noncg=(1 - d.is_CG_member).sum(),
        raw_E_cg=d.loc[d.is_CG_member.eq(1), "elliptical"].mean(),
        raw_E_other=d.loc[d.is_CG_member.eq(0), "elliptical"].mean(),
        eligible_range_overlap_hosts=sum(
            overlap(p[p.eligible], "bgg_kpc")["overlap"]
            for _, p in d.groupby("host_lim_group")
        ),
        formula=" + ".join(variables),
    )
    notes["raw_delta_pp"] = (notes["raw_E_cg"] - notes["raw_E_other"]) * 100
    if len(both_hosts) < 2 or len(d) < 20:
        return dict(**notes, status="insufficient support")
    X = d[variables].astype(float).copy()
    for col in variables[1:]:
        X[col] = (X[col] - X[col].mean()) / X[col].std(ddof=0)
    if not np.isfinite(X).all(axis=None):
        return dict(**notes, status="constant or nonfinite predictor")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            m = ConditionalLogit(
                d.elliptical.to_numpy(),
                X.to_numpy(),
                groups=d.host_lim_group.to_numpy(),
            )
            fit = m.fit(method="bfgs", disp=False, maxiter=300)
            score = float(np.abs(m.score(fit.params)).max() / len(d))
            beta, se = fit.params[0], fit.bse[0]
            smallest_eigenvalue = np.linalg.eigvalsh(-m.hessian(fit.params)).min()
        except (ValueError, np.linalg.LinAlgError, OverflowError) as err:
            return dict(**notes, status="estimation failed", warning=str(err))
    messages = "; ".join(str(w.message) for w in caught)
    return dict(
        **notes,
        status="ok" if not messages and score < 1e-4 and se < 5 else "unstable",
        warning=messages,
        score_per_galaxy=score,
        hessian_min_eigenvalue=smallest_eigenvalue,
        beta=beta,
        se=se,
        OR=np.exp(beta),
        low=np.exp(beta - 1.96 * se),
        high=np.exp(beta + 1.96 * se),
        p=2 * norm.sf(abs(beta / se)),
    )


def run(frame, sample, core):
    d = build_host_frame(sample)
    pgroup = pd.read_csv(DATA / "PC_Groups.csv").set_index("Group")
    status = sample["CG4_Groups"].set_index("Group").Class
    mapping = sample["CG4_Gals"][["objid", "Group"]].rename(
        columns={"Group": "CG4_group"}
    )
    d = d.merge(mapping, on="objid", how="left", validate="1:1")
    assert d.objid.is_unique
    assert d.is_CG_member.eq(d.CG4_group.notna()).all()
    d.CG4_group = d.CG4_group.astype("Int64")
    d["delta_m"] = d.M_r - d.M_BGG
    d["eligible"] = d.delta_m.le(DMAG_MAX)
    d["host_bgg"] = d.rank_parent.eq(1)
    d["is_core"] = d.objid.isin(core.objid)
    d["rad_selection_order"] = d.objid.map(core.set_index("objid").rank_dist)
    da = Planck15.angular_diameter_distance(d.Yang_z_CMB_group.fillna(d.z)).to_value(
        "kpc"
    )
    d["bgg_kpc"] = separation(d, d.RA_BGG, d.Dec_BGG) * da
    d["centroid_kpc"] = (
        separation(d, d.Group.map(pgroup.RA_Bary), d.Group.map(pgroup.Dec_Bary)) * da
    )
    np.testing.assert_allclose(d.bgg_kpc, d.dist_host_kpc, atol=0.008, rtol=1e-6)
    d["log_radius"] = np.log1p(d.bgg_kpc / 25)
    d["bgg_rank_eligible"] = (
        d.bgg_kpc.where(d.eligible).groupby(d.Group).rank(method="first")
    )
    d["population"] = np.select(
        [d.is_CG_member.eq(1), d.eligible],
        ["CG4 member", "Eligible nonmember"],
        default="Faint nonmember",
    )
    d["system_Lfrac"] = d.CG4_group.map(
        d[d.is_CG_member.eq(1)].groupby("CG4_group").Lum.sum()
    ) / d.Group.map(d.groupby("Group").Lum.sum())
    quartet_rows = []
    for gid, p in d[d.is_CG_member.eq(1)].groupby("CG4_group"):
        host = p.Group.iloc[0]
        h = d[d.Group.eq(host)].copy()
        h["is_CG_member"] = h.CG4_group.eq(gid).fillna(False).astype(int)
        c = h[h.is_core]
        n_overlap = len(set(p.objid) & set(c.objid))
        exact = n_overlap == 4
        assert p.Group.nunique() == 1
        assert (p.system_Lfrac.iloc[0] >= 0.5) == (status[gid] == "Predom")
        if exact:
            assert set(p.bgg_rank_eligible) == {1, 2, 3, 4}
        row = dict(
            CG4_group=gid,
            Lim_host=host,
            host_bgg_objid=h.loc[h.host_bgg, "objid"].item(),
            CG4_objids=";".join(p.objid.astype(str)),
            core_objids=";".join(
                c.sort_values("rad_selection_order").objid.astype(str)
            ),
            category=status[gid],
            L_fraction=p.system_Lfrac.iloc[0],
            overlap_n=n_overlap,
            exact=exact,
            host_members=len(h),
            eligible_nonmembers=(h.eligible & h.is_CG_member.eq(0)).sum(),
            fainter_nonmembers=(~h.eligible & h.is_CG_member.eq(0)).sum(),
            ineligible_cg=(~p.eligible).sum(),
        )
        for key, v in overlap(h[h.eligible & ~h.host_bgg], "bgg_kpc").items():
            row[f"eligible_satellite_{key}"] = v
        quartet_rows.append(row)
    quartets = save("within_host_quartets", quartet_rows)
    assert (
        len(quartets),
        quartets.exact.sum(),
        quartets.Lim_host.nunique(),
        len(d),
        d.is_CG_member.sum(),
    ) == (56, 46, 54, 958, 224)
    exact_hosts = set(quartets.loc[quartets.exact, "Lim_host"])
    nonexact_hosts = set(quartets.loc[~quartets.exact, "Lim_host"])
    d["host_type"] = np.select(
        [d.Group.isin(exact_hosts & nonexact_hosts), d.Group.isin(exact_hosts)],
        ["Mixed host", "Exact-core host"],
        default="Non-identical host",
    )
    hrows = []
    valid = d.dropna(subset=["elliptical", "logMstar", "rank_parent", "dist_host_kpc"])
    for hid, p in d.groupby("Group"):
        for radius in ["bgg_kpc", "centroid_kpc"]:
            for label, mask in [
                ("all_members", p.index == p.index),
                ("eligible_satellites", p.eligible & ~p.host_bgg),
                (
                    "eligible_satellites_complete_E_S",
                    p.eligible & ~p.host_bgg & p.index.isin(valid.index),
                ),
            ]:
                part = p.loc[mask]
                hrows.append(
                    dict(
                        Lim_host=hid,
                        host_type=p.host_type.iloc[0],
                        population=label,
                        radius=radius,
                        cg=part.is_CG_member.sum(),
                        other=(1 - part.is_CG_member).sum(),
                        E_classes=part.elliptical.nunique(),
                        **overlap(part, radius),
                    )
                )
    save("within_host_overlap", hrows)
    save(
        "within_host_members",
        d[
            [
                "objid",
                "Group",
                "CG4_group",
                "host_type",
                "population",
                "delta_m",
                "eligible",
                "host_bgg",
                "is_CG_member",
                "is_core",
                "rank_parent",
                "bgg_rank_eligible",
                "bgg_kpc",
                "centroid_kpc",
                "elliptical",
                "logMstar",
            ]
        ],
    )
    es = valid[valid.eligible & ~valid.host_bgg].copy()
    near_ids = {25: [], 50: []}
    for _, p in es.groupby("Group"):
        a = p[p.is_CG_member.eq(1)]
        b = p[p.is_CG_member.eq(0)]
        distances = np.abs(
            a.bgg_kpc.to_numpy()[:, None] - b.bgg_kpc.to_numpy()[None, :]
        )
        for k in near_ids:
            near_ids[k].extend(
                [*a.index[(distances <= k).any(1)], *b.index[(distances <= k).any(0)]]
            )
    specs = [
        ("original_no_radius", valid, []),
        ("original_radius", valid, ["dist_host_kpc"]),
        ("recomputed_BGG_radius", valid, ["bgg_kpc"]),
        ("centroid_sensitivity", valid, ["centroid_kpc"]),
        ("log_BGG_radius", valid, ["log_radius"]),
        (
            "eligible_nonmembers_only",
            valid[valid.eligible | valid.is_CG_member.eq(1)],
            ["bgg_kpc"],
        ),
        ("eligible_satellites_only", es, ["bgg_kpc"]),
        (
            "exact_hosts_all_members",
            valid[valid.host_type.eq("Exact-core host")],
            ["bgg_kpc"],
        ),
        (
            "exact_hosts_eligible_satellites",
            es[es.host_type.eq("Exact-core host")],
            ["bgg_kpc"],
        ),
        (
            "nonidentical_hosts_eligible_satellites",
            es[es.host_type.eq("Non-identical host")],
            ["bgg_kpc"],
        ),
        ("eligible_satellites_within_50_kpc", es.loc[near_ids[50]], ["bgg_kpc"]),
        ("eligible_satellites_within_25_kpc", es.loc[near_ids[25]], ["bgg_kpc"]),
    ]
    models = save(
        "within_host_models", [fit_host(p, name, cols) for name, p, cols in specs]
    )
    assert models.iloc[1].n == 791 and models.iloc[1].outcome_informative == 53
    np.testing.assert_allclose(models.OR.iloc[:2], [1.90, 1.47], atol=0.012)
    d["physical_group"] = d.Group.astype(str)
    order = ["Exact-core host", "Non-identical host", "Mixed host"]
    pop = ["CG4 member", "Eligible nonmember", "Faint nonmember"]
    colors = ["#92234e", "#b55a00", "#61707b"]
    rb = [0, 50, 100, 150, 250, 400, 650, 1000, np.inf]
    d["radius_bin"] = pd.cut(d.bgg_kpc, rb, right=False)
    fig, axs = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
    rows = []
    for j, kind in enumerate(order):
        for lab, colour in zip(pop, colors):
            p = d[(d.host_type == kind) & (d.population == lab)]
            if p.empty:
                continue
            x = np.sort(p.bgg_kpc)
            axs[0, j].step(
                x, np.arange(1, len(x) + 1) / len(x), where="post", label=lab, c=colour
            )
            for k, v in p.groupby("radius_bin", observed=True):
                mu, l, h = bootstrap(v, "elliptical")
                r = v.bgg_kpc.median()
                rows.append(
                    dict(
                        host_type=kind,
                        population=lab,
                        radial_bin=str(k),
                        r=r,
                        n=len(v),
                        n_E_S=v.elliptical.count(),
                        hosts=v[v.elliptical.notna()].Group.nunique(),
                        estimate=mu,
                        low=l,
                        high=h,
                    )
                )
                if v[v.elliptical.notna()].Group.nunique() >= 3:
                    axs[1, j].vlines(r, l, h, color=colour)
                    axs[1, j].scatter(r, mu, color=colour, s=13)
        axs[0, j].set_title(kind)
        for ax in axs[:, j]:
            ax.set(xlim=(0, 1000), ylim=(0, 1.03))
    axs[1, 2].text(
        0.5,
        0.5,
        "Fewer than three hosts per bin",
        ha="center",
        va="center",
        transform=axs[1, 2].transAxes,
        color=".35",
        fontsize=9,
    )
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].set_ylabel("Galaxy cumulative fraction")
    axs[1, 0].set_ylabel("E-class fraction")
    fig.supxlabel("Projected radius from Lim BGG (kpc)")
    figure(fig, "06_within_host_radial_support")
    save("within_host_radial_bins", rows)
