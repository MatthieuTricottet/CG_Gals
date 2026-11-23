"""Freeze GZ1 labels, construct exact host keys, and rerun primary morphology."""

import warnings

import numpy as np
import pandas as pd
from astropy.coordinates import Angle, angular_separation
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
from patsy import build_design_matrices
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from compactness import candidates, cdiv, geometry
from config import LGM_VALID_RANGE
from extended_stats import bootstrap_difference
from matched_controls import (
    GROUP_MATCHING_CANDIDATES,
    _group_table,
    _group_table_for_control,
    _permutation_p_component_sign_flip,
    _run_one_group_match,
    matched_cluster_components,
    matched_pairs,
)
from referee_common import (
    BASE,
    bootstrap,
    fit,
    marginal,
    paired_bootstrap,
    sizes,
)
from run import OUTPUT, PRIOR, ROOT

CONTROLS = ("Control4C", "Control4B", "RG4")
ORIGINAL = "original"
FROZEN = "frozen"
CENTRE = 150.0
# The fixed-mass curves and bins of the compactness figure are evaluated at
# the median stellar mass of the E/S-classified CG4 and Control4C satellites
# with valid masses, rounded to 0.01 dex. The 20 November mean-separation
# freeze used the hand-set value 10.35, kept here only to redraw that record.
MASS_REF_SAMPLES = ("CG4", "Control4C")
LEGACY_MASS_REF = 10.35


def reference_mass(frame):
    sat = frame[frame.is_satellite.eq(1) & frame["sample"].isin(MASS_REF_SAMPLES)]
    return round(float(sat.dropna(subset=["elliptical", "m"]).logMstar.median()), 2)


def save(name, rows):
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / f"{name}.csv", index=False, float_format="%.10g")
    return df


def arm_sizes(df):
    return {
        **sizes(df),
        **{
            f"{arm}_{key}": value
            for arm, part in (("cg", df[df.cg.eq(1)]), ("control", df[df.cg.eq(0)]))
            for key, value in sizes(part).items()
        },
    }


def restore_labels(frame, *, write_audit=True):
    archive = pd.read_csv(
        PRIOR / "inputs/gz1_paper1_crossmatch.csv", dtype={"OBJID": "Int64"}
    )
    assert archive.OBJID.is_unique
    archive["archive_ra"] = Angle(archive.RA.tolist(), unit="hourangle").degree
    archive["archive_dec"] = Angle(archive.DEC.tolist(), unit="deg").degree
    archive = archive.rename(columns={"OBJID": "dr7objid"})
    cols = [
        "dr7objid",
        "archive_ra",
        "archive_dec",
        "P_EL_DEBIASED",
        "P_CS_DEBIASED",
        "NVOTE",
    ]
    bridge = pd.read_csv(
        ROOT / "data/sdss_size_columns.csv", dtype={"objid": str, "dr7objid": str}
    )
    collisions = bridge.groupby("dr7objid").objid.nunique()
    bad = set(collisions[collisions.gt(1)].index.astype("int64"))
    f = frame.merge(archive[cols], on="dr7objid", how="left", validate="m:1")
    f["sep_arcsec"] = (
        np.rad2deg(
            angular_separation(
                *np.deg2rad([f.RA, f.Dec]), *np.deg2rad([f.archive_ra, f.archive_dec])
            )
        )
        * 3600
    )
    f["bridge_collision"] = f.dr7objid.isin(bad)
    verified = f.sep_arcsec.le(1.5) & ~f.bridge_collision
    valid_votes = f[["P_EL_DEBIASED", "P_CS_DEBIASED"]].notna().all(axis=1)
    f["original_label"] = f.morphology.map(
        {"Elliptical": "E", "Spiral": "S", "Uncertain": "U"}
    ).fillna("missing")
    dr7 = np.select(
        [f.P_EL_DEBIASED.gt(0.5), f.P_CS_DEBIASED.gt(0.5)], ["E", "S"], default="U"
    )
    f["archive_label"] = pd.Series(dr7, index=f.index).where(verified & valid_votes)
    f["original_E"] = f.elliptical
    # Preserve existing decisive classes. An archival decisive vote can
    # resolve a formerly uncertain row; an archival U can fill a missing row.
    replace = f.original_label.eq("missing") | (
        f.original_label.eq("U") & f.archive_label.isin(["E", "S"])
    )
    f["frozen_label"] = f.archive_label.where(replace).fillna(f.original_label)
    f["elliptical"] = f.frozen_label.map({"E": 1.0, "S": 0.0})
    f["physical_host_id"] = f.physical_group
    f["mass_valid"] = f.logMstar.between(*LGM_VALID_RANGE)
    f["ssfr_valid"] = f.quenched.notna()
    if not write_audit:
        return f
    rows = []
    for name, arm in f.groupby("sample", sort=False):
        for scope, p in [("all", arm), ("satellites", arm[arm.is_satellite.eq(1)])]:
            comparable = p[p.original_label.ne("missing") & p.archive_label.notna()]
            comp_es = comparable[comparable.original_label.isin(["E", "S"])]
            for label, subset in [("E/S/U", comparable), ("E/S", comp_es)]:
                agree = subset.original_label.eq(subset.archive_label).sum()
                rows.append(
                    dict(
                        sample=name,
                        scope=scope,
                        metric="agreement",
                        label=label,
                        successes=agree,
                        rate=agree / len(subset),
                        previously_labelled=p.original_label.isin(
                            list(label.replace("/", ""))
                        ).sum(),
                        **sizes(subset),
                    )
                )
            for label in ["E", "S", "U"]:
                g = p[p.frozen_label.eq(label) & p.original_label.ne(label)]
                for qualifier, sub in [
                    ("any", g),
                    ("mass", g[g.mass_valid]),
                    ("total_ssfr", g[g.ssfr_valid]),
                    ("mass_and_total_ssfr", g[g.mass_valid & g.ssfr_valid]),
                ]:
                    rows.append(
                        dict(
                            sample=name,
                            scope=scope,
                            metric="recovery",
                            label=label,
                            qualifier=qualifier,
                            **sizes(sub),
                        )
                    )
    save("01_gz1_recovery_audit", rows)
    changed = f.original_label.ne(f.frozen_label)
    discrepancy = (
        f.original_label.ne("missing")
        & f.archive_label.notna()
        & f.original_label.ne(f.archive_label)
    )
    audit_cols = [
        "sample",
        "Group",
        "physical_host_id",
        "objid",
        "specobjid",
        "dr7objid",
        "sep_arcsec",
        "NVOTE",
        "is_satellite",
        "original_label",
        "archive_label",
        "frozen_label",
        "mass_valid",
        "ssfr_valid",
        "logMstar",
        "quenched",
    ]
    save(
        "01_gz1_recovered_and_discrepant_rows", f.loc[changed | discrepancy, audit_cols]
    )
    return f


def build(frame):
    frame = restore_labels(frame, write_audit=False)
    frame["m"] = frame.logMstar - 10.25
    frame["cg"] = frame.is_CG4.astype(float)
    q, _ = geometry(frame)
    # The Table 1 calculation uses the Paper I Planck 2015 parameters
    # and the arithmetic mean of the member redshifts.
    q = q.rename(columns={"table1_R_median": "med_Rij"})
    q["x"] = np.log10(q.med_Rij)
    q = q.drop(columns=[
        "R_mean", "R_median", "xmed", "DA_kpc", "DL_kpc",
        "R12_kpc", "R13_kpc", "R14_kpc", "R23_kpc", "R24_kpc", "R34_kpc",
    ])
    q["physical_host_id"] = q.physical_group
    # Parent environment comes from the host table, including for off-centre CG4.
    pc = pd.read_csv(
        ROOT / "data/PC_Groups.csv", usecols=["Group", "lMass_200", "NbGal"]
    )
    lim = pd.read_csv(
        ROOT / "data/SDSS(L) group.dat",
        sep=r"\s+",
        skiprows=11,
        header=None,
        usecols=[0, 6],
        names=["Group", "N_Lim"],
    )
    assert lim.Group.is_unique and pc.Group.is_unique
    host = pc.merge(lim, on="Group", how="left", validate="1:1")
    host["physical_host_id"] = "Lim:" + host.Group.astype(str)
    host = host.rename(columns={"lMass_200": "host_mass", "NbGal": "N_Lim_Tempel"})
    assert host.N_Lim.notna().all() and host.N_Lim.ge(host.N_Lim_Tempel).all()
    host["host_richness"] = np.log10(host.N_Lim)
    cols = ["physical_host_id", "host_mass", "N_Lim", "N_Lim_Tempel", "host_richness"]
    q = q.merge(host[cols], on="physical_host_id", validate="m:1", how="left")
    assert q.host_mass.notna().all(), q[q.host_mass.isna()].group_uid.tolist()
    q["class"] = q.Group.map(
        pd.read_csv(ROOT / "data/CG4_Groups.csv").set_index("Group").Class
    ).where(q["sample"].eq("CG4"))
    window = common_window(q, ["CG4", "Control4C"])
    assert window[0] < CENTRE < window[1]
    q["window"] = q.med_Rij.between(*window)
    save("02_quartet_geometry", q)
    frame = frame.merge(
        q[["group_uid", "med_Rij", "x", "window", *cols[1:]]],
        on="group_uid",
        validate="m:1",
    )
    frame["xc"] = np.log10(frame.med_Rij / CENTRE)
    assert frame.groupby("sample").size().to_dict() == {
        "CG4": 248,
        "Control4B": 2792,
        "Control4C": 2812,
        "RG4": 224,
    }
    return frame, q, window, host


def common_window(quartets, samples):
    central = (
        quartets.loc[quartets["sample"].isin(samples)]
        .groupby("sample").med_Rij.quantile([0.05, 0.95]).unstack()
    )
    return float(central[0.05].max()), float(central[0.95].min())


def local_target(used, radius, environment=()):
    xc = np.log10(radius / CENTRE)
    local = used[(used.xc - xc).abs().le(0.10)]
    cg, control = local[local.cg.eq(1)], local[local.cg.eq(0)]
    cols = ["z_numeric", "m", "log_group_luminosity", "velocity_dispersion"]
    tol = [0.006, 0.25, 0.35, 120.0]
    if "host_mass" in environment:
        cols.append("host_mass")
        tol.append(0.4)
    if "host_richness" in environment:
        cols.append("host_richness")
        tol.append(0.3)
    ok = cdiv(cg, control, cols, np.array(tol))
    left, right = cg.iloc[ok.any(1)], control.iloc[ok.any(0)]
    return left.assign(xc=xc), right


def model_rows(
    frame, outcome, rhs, meta, radii=(), environment=(), min_local=3, *,
    local_within_window=False,
):
    """Keep fitting denominators separate from the target denominators."""
    formula = f"{outcome} ~ {rhs}"
    model_df = frame
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            result, used, warning = fit(model_df, formula)
        if (
            np.max(np.abs(result.predict() - 0.5)) == 0.5
            and np.abs(result.params["cg"]) > 15
        ):
            raise RuntimeError("unstable separation")
    except (
        ValueError,
        RuntimeError,
        np.linalg.LinAlgError,
        PerfectSeparationWarning,
    ) as exc:
        # Patsy complete cases are available before fitting and also for failed fits.
        from patsy import dmatrices

        y, _ = dmatrices(formula, frame, return_type="dataframe")
        return (
            [
                dict(
                    **meta,
                    row="failure",
                    status=str(exc),
                    formula=formula,
                    **arm_sizes(frame.loc[y.index]),
                )
            ],
            None,
            None,
        )
    n = arm_sizes(used)
    ci = result.conf_int()
    rows = [
        dict(
            **meta,
            row="coefficient",
            status="ok",
            formula=formula,
            warning=warning,
            term=term,
            estimate=beta,
            low=ci.loc[term, 0],
            high=ci.loc[term, 1],
            se=result.bse[term],
            p=result.pvalues[term],
            **(
                {
                    "OR": np.exp(beta),
                    "OR_low": np.exp(ci.loc[term, 0]),
                    "OR_high": np.exp(ci.loc[term, 1]),
                }
                if term == "cg"
                else {}
            ),
            **n,
        )
        for term, beta in result.params.items()
    ]
    local_used = used[used.window] if local_within_window else used
    for r in radii or (None,):
        if r is None:
            ref, reference_control = used[used.cg.eq(1)], used[used.cg.eq(0)]
        else:
            ref, reference_control = local_target(local_used, r, environment)
        rcounts = {
            f"ref_{k}": v
            for k, v in arm_sizes(pd.concat([ref, reference_control])).items()
        }
        base = dict(**meta, row="contrast", formula=formula, radius=r, **n, **rcounts)
        if (
            min(
                ref.physical_group.nunique(), reference_control.physical_group.nunique()
            )
            < min_local
        ):
            rows.append(dict(**base, status="insufficient local host support"))
            continue
        effect = marginal(result, used, ref)
        if "component" in used:
            # One-step paired bootstrap of the standardized effect. Resample
            # centered logistic scores by whole matching graph component.
            scores = pd.DataFrame(
                result.model.score_obs(result.params), index=used.index
            )
            sums = scores.groupby(used.component).sum().to_numpy()
            rng = np.random.default_rng(20261117)
            draw = rng.integers(len(sums), size=(4999, len(sums)))
            score = sums[draw].sum(1) - sums.sum(0)
            inverse = np.linalg.inv(-result.model.hessian(result.params))
            boot_params = result.params.to_numpy() + score @ inverse
            design = result.model.data.design_info
            treated, control = [
                np.asarray(build_design_matrices([design], ref.assign(cg=arm))[0])
                for arm in (1, 0)
            ]
            draws = 100 * (
                expit(boot_params @ treated.T) - expit(boot_params @ control.T)
            ).mean(1)
            effect["low"], effect["high"] = np.quantile(draws, [0.025, 0.975])
            effect["components"] = len(sums)
            effect["ci_method"] = "one_step_component_bootstrap"
        effect.setdefault("ci_method", "host_clustered_delta_method")
        rows.append(dict(**base, status="ok", **effect))
    return rows, result, used


def compact_pairs(q):
    result = {}
    out = []
    for name, sigma in [("compact_match", False), ("compact_match_sigma", True)]:
        left, right, ok, distance, _ = candidates(q[q.window], include_sigma=sigma)
        cost = np.where(ok, distance, 1e6)
        i, j = linear_sum_assignment(
            np.c_[cost, np.full((len(left), len(left)), 1000.0)]
        )
        chosen = j < len(right)
        i, j = i[chosen], j[chosen]
        assert ok[i, j].all() and len(set(j)) == len(j)
        left, right = left.iloc[i], right.iloc[j]
        pairs = pd.DataFrame(
            dict(
                pair=np.arange(len(i)),
                cg_quartet=left.group_uid.to_numpy(),
                control_quartet=right.group_uid.to_numpy(),
                cg_host=left.physical_group.to_numpy(),
                control_host=right.physical_group.to_numpy(),
                match=name,
            )
        )
        pairs["component"] = pair_components(pairs.cg_host, pairs.control_host)
        result[name] = pairs
        out.append(pairs)
    save("02_compact_matched_pair_ids", pd.concat(out))
    return result


def pair_components(left, right):
    components = matched_cluster_components(left, right)
    # The legacy bipartite helper treats the same host on opposite sides
    # as separate nodes. Unite those links when they exist.
    touched = {}
    for idx, host in enumerate(left):
        touched.setdefault(host, []).append(idx)
    for idx, host in enumerate(right):
        for prior in touched.get(host, ()):
            lo, hi = components[prior], components[idx]
            components[components == hi] = lo
    return components


def pair_outcome(f, pairs, outcome, meta):
    groups = f.groupby("group_uid")[outcome].agg(["mean", "count"])
    a = groups.reindex(pairs.cg_quartet).reset_index(drop=True)
    b = groups.reindex(pairs.control_quartet).reset_index(drop=True)
    valid = a["count"].fillna(0).gt(0) & b["count"].fillna(0).gt(0)
    pairs_kept = pairs.loc[valid]
    ids = set(pairs_kept.cg_quartet) | set(pairs_kept.control_quartet)
    used = f[f.group_uid.isin(ids) & f[outcome].notna()]
    left = a.loc[valid, "mean"].to_numpy()
    right = b.loc[valid, "mean"].to_numpy()
    components = pairs_kept.component.to_numpy()
    effect = paired_bootstrap(left - right, components)
    p_perm = np.nan
    if meta.get("analysis") == "Table 5":
        historical = bootstrap_difference(
            left,
            right,
            statistic=np.mean,
            paired=True,
            n_boot=9999,
            seed=20260612,
            blocks=components,
        )
        effect["low"], effect["high"] = historical["ci95"]
        p_perm = _permutation_p_component_sign_flip(
            left - right, components, seed=20260612, n_perm=9999
        )
    return dict(
        **{**meta, "outcome": outcome},
        row="paired",
        ci_method="paired_component_bootstrap",
        matched_pairs=len(pairs),
        labelled_pairs=valid.sum(),
        cg_est=a.loc[valid, "mean"].mean() * 100,
        control_est=b.loc[valid, "mean"].mean() * 100,
        **arm_sizes(used),
        difference=effect["estimate"] * 100,
        low=effect["low"] * 100,
        high=effect["high"] * 100,
        components=effect["blocks"],
        effective_components=effect["effective_blocks"],
        permutation_p=p_perm,
    )


def historical_quartet_match(frame, control):
    # The pinned function returns summaries without the matched row IDs.
    # Capture the propensity input and the pair index selections it uses.
    from unittest.mock import patch
    import matched_controls as historical

    table = (
        _group_table(frame)
        if control == "pooled"
        else _group_table_for_control(frame, control)
    )
    variables = [c for c in GROUP_MATCHING_CANDIDATES if table[c].notna().mean() >= 0.7]
    recorded = []

    def capture(left, right, *args, **kwargs):
        recorded.append((left, right))
        return bootstrap_difference(left, right, *args, **kwargs)

    # In the original function, t_frac/c_frac are plain ndarrays, so obtain
    # pair IDs through its first dispersion/distance sort using the same
    # deterministic propensity implementation and caliper.
    with patch.object(historical, "bootstrap_difference", side_effect=capture):
        m = _run_one_group_match(table, variables)
    w = table.dropna(subset=variables).query("n_sat_classified > 0").copy()
    design = (w[variables] - w[variables].mean()) / w[variables].std(ddof=0).replace(
        0, 1
    )
    model = historical.LogisticRegression(max_iter=2000, random_state=20260612)
    model.fit(design, w.is_CG4)
    p = np.clip(model.predict_proba(design)[:, 1], 1e-8, 1 - 1e-8)
    logit = np.log(p / (1 - p))
    treated = w[w.is_CG4.eq(1)]
    controls = w[w.is_CG4.eq(0)]
    d = np.abs(logit[w.is_CG4.eq(1)][:, None] - logit[w.is_CG4.eq(0)][None, :])
    available = set(range(len(controls)))
    i, j = [], []
    for k in np.argsort(d.min(axis=1)):
        if not available:
            break
        best = min(available, key=lambda n: d[k, n])
        if d[k, best] <= 0.2 * logit.std():
            i.append(k)
            j.append(best)
            available.remove(best)
    a, b = treated.iloc[i], controls.iloc[j]
    assert len(a) == m["n_matched_groups"]
    for p, obs in zip((a, b), recorded[0]):
        np.testing.assert_allclose(p.n_smooth_sat / p.n_sat_classified, obs)
    pairs = pd.DataFrame(
        dict(
            cg_quartet=a.group_uid.to_numpy(),
            control_quartet=b.group_uid.to_numpy(),
            cg_host=a.physical_group.to_numpy(),
            control_host=b.physical_group.to_numpy(),
        )
    )
    pairs["component"] = pair_components(pairs.cg_host, pairs.control_host)
    return pairs, m


def baseline(frame, q, window):
    rows = []
    matches = compact_pairs(q)
    # Keep the default ranks, covariate availability rule, and outcome definition.
    for labelset, label, binary in [
        (ORIGINAL, "original_label", "original_E"),
        (FROZEN, "frozen_label", "elliptical"),
    ]:
        f = frame.assign(elliptical=frame[binary])
        for arm, p in f.groupby("sample"):
            for scope, r in [
                ("all", p),
                ("satellites", p[p.is_satellite.eq(1)]),
                ("BGGs", p[p.is_satellite.eq(0)]),
            ]:
                for key in ["E", "S", "U", "missing"]:
                    rows.append(
                        dict(
                            analysis="Table 3",
                            row="raw_count",
                            labelset=labelset,
                            sample=arm,
                            scope=scope,
                            term=key,
                            estimate=r[label].eq(key).sum(),
                            **sizes(r),
                        )
                    )
        for control in CONTROLS:
            p = f[f["sample"].isin(["CG4", control])]
            for scope, r, term in [
                ("all", p, " + is_satellite"),
                ("satellites", p[p.is_satellite.eq(1)], ""),
            ]:
                for sigma in [True, False]:
                    rhs = BASE if sigma else BASE.removesuffix(" + velocity_dispersion")
                    results, _, _ = model_rows(
                        r,
                        "elliptical",
                        "cg + " + rhs + term,
                        dict(
                            analysis="Table 4",
                            labelset=labelset,
                            control=control,
                            scope=scope,
                            sigma=sigma,
                            outcome="elliptical",
                        ),
                    )
                    rows.extend(results)
        sat = f[f.is_satellite.eq(1)]
        if labelset == FROZEN:
            allpairs, work, caliper, prepared, cols = matched_pairs(f)
            ai = [x["treated_index"] for x in allpairs]
            bi = [x["control_index"] for x in allpairs]
            a, b = prepared.loc[ai].reset_index(drop=True), prepared.loc[
                bi
            ].reset_index(drop=True)
            components = pair_components(a.physical_group, b.physical_group)
            for scope, mask, p in [
                ("all", a.objid.notna(), work),
                ("satellites", a.is_satellite.eq(1), work[work["rank"].gt(1)]),
            ]:
                complete = prepared.loc[p.index]
                rows.append(
                    dict(
                        analysis="galaxy match",
                        row="complete_covariates",
                        labelset=labelset,
                        scope=scope,
                        match_covariates=" + ".join(cols),
                        **arm_sizes(complete),
                    )
                )
                for label_version, col in [
                    (ORIGINAL, "original_E"),
                    (FROZEN, "elliptical"),
                ]:
                    for stage, keep in [
                        ("matched", mask),
                        ("doubly_labelled", mask & a[col].notna() & b[col].notna()),
                    ]:
                        used = pd.concat([a[keep], b[keep]])
                        effect = (
                            paired_bootstrap(
                                (a[col] - b[col])[keep].to_numpy(), components[keep]
                            )
                            if stage == "doubly_labelled"
                            else {}
                        )
                        rows.append(
                            dict(
                                analysis="galaxy match",
                                row=stage,
                                labelset=label_version,
                                scope=scope,
                                matched_pairs=keep.sum(),
                                complete_cg=p.is_CG4.eq(1).sum(),
                                complete_control=p.is_CG4.eq(0).sum(),
                                match_covariates=" + ".join(cols),
                                **arm_sizes(used),
                                **{
                                    k: (
                                        v * 100
                                        if k in ("estimate", "low", "high")
                                        else v
                                    )
                                    for k, v in effect.items()
                                },
                            )
                        )
            pairs_out = (
                a[["objid", "group_uid", "physical_group", "original_E", "elliptical"]]
                .add_prefix("cg_")
                .join(
                    b[
                        [
                            "objid",
                            "group_uid",
                            "physical_group",
                            "original_E",
                            "elliptical",
                        ]
                    ].add_prefix("control_")
                )
            )
            save(
                "02_galaxy_matched_pair_ids",
                pairs_out.assign(
                    scope=np.where(a.is_satellite.eq(1), "satellites", "BGGs"),
                    component=components,
                ),
            )
        for control in (*CONTROLS, "pooled"):
            pairs, original_spec = historical_quartet_match(f, control)
            outcome = pair_outcome(
                sat,
                pairs,
                "elliptical",
                dict(analysis="Table 5", labelset=labelset, control=control),
            )
            outcome["original_spec_permutation_p"] = original_spec["p_permutation"]
            rows.append(outcome)
            if labelset == FROZEN:
                save(f"02_table5_pairs_{control}", pairs)
    _, compactness = compactness_models(frame, q, window, matches)
    table = save("02_frozen_primary_and_compactness", [*rows, *compactness])
    return matches, table


def compactness_models(frame, q, window, matches=None):
    """Refit just the morphology comparisons that feed the compactness figures."""
    rows = []
    if matches is None:
        matches = compact_pairs(q)
    purge = pd.read_csv(PRIOR / "tables/hmcg_control_members.csv")
    purged = set(purge.loc[purge.multiplicity.ne(4) | purge.flag.ne(0), "group_uid"])
    satellites = frame[frame.is_satellite.eq(1)]
    for labelset, binary in [(ORIGINAL, "original_E"), (FROZEN, "elliptical")]:
        sat = satellites.assign(elliptical=satellites[binary])
        for name, pairs in matches.items():
            rows.append(
                pair_outcome(
                    sat,
                    pairs,
                    "elliptical",
                    dict(
                        analysis="compactness",
                        labelset=labelset,
                        control="Control4C",
                        design=name,
                    ),
                )
            )
        p = sat[sat["sample"].isin(["CG4", "Control4C"])]
        for purity, panel in [
            ("all", p),
            ("HMCG_purged", p[~p.group_uid.isin(purged)]),
        ]:
            for design, r, radii, formula in [
                ("full", panel, (None, CENTRE), "cg + " + BASE),
                ("window_no_separation", panel[panel.window], (CENTRE,), "cg + " + BASE),
                (
                    "common_slope",
                    panel[panel.window],
                    (CENTRE, 175.0, 200.0),
                    "cg + xc + " + BASE,
                ),
                (
                    "interaction",
                    panel[panel.window],
                    (CENTRE, 175.0, 200.0),
                    "cg * xc + " + BASE,
                ),
            ]:
                meta = dict(
                    analysis="compactness",
                    labelset=labelset,
                    control="Control4C",
                    outcome="elliptical",
                    design=design,
                    purity=purity,
                )
                estimates, model, used = model_rows(
                    r, "elliptical", formula, meta, radii,
                    local_within_window=design == "full",
                )
                rows.extend(estimates)
                if labelset == FROZEN and purity == "all" and design == "interaction":
                    curve = []
                    for radius in np.linspace(*window, 100):
                        left, right = local_target(used, radius)
                        if (
                            min(
                                left.physical_group.nunique(),
                                right.physical_group.nunique(),
                            )
                            >= 3
                        ):
                            curve.append(
                                dict(
                                    radius=radius,
                                    **marginal(model, used, left),
                                    **{
                                        f"ref_{k}": v
                                        for k, v in arm_sizes(
                                            pd.concat([left, right])
                                        ).items()
                                    },
                                    **arm_sizes(used),
                                )
                            )
                    save("02_compactness_interaction_curve", curve)
    return matches, rows
