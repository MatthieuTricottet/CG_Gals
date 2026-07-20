"""Matched comparison of CG4 and ordinary-group galaxies.

Design (statistical audit, Phase 4):

* the control pool is deduplicated by objid *before* matching (one row per
  physical galaxy, kept-label priority RG4 > Control4B > Control4C, all
  source labels recorded), so the same galaxy can never serve as two
  different controls and a CG4 galaxy can never be matched to its own
  duplicate row — both failure modes of the pre-audit analysis;
* hard constraints are enforced and unit-tested: no control objid may
  coincide with a CG4 objid, and no control objid is used more than once;
* a provenance table records, for every matched control, its Lim group,
  the control definitions that contain it, and whether it is physically an
  RG4 galaxy;
* the group-level analysis (counts of elliptical-vote satellites per group,
  binomial fractions) is the cleaner primary: groups are the natural
  independent unit. The galaxy-level matched contrast is kept as a
  secondary check with group-blocked bootstrap inference (resampling
  treated CG groups, add-one empirical p-values with an explicit
  Monte-Carlo floor).
"""

from __future__ import annotations

import os

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    from extended_data import dedup_control_pool, ensure_galaxy_frame
    from extended_stats import (
        bootstrap_difference,
        empirical_p_two_sided,
        holm_correction,
        safe_json,
        standardized_mean_difference,
    )
except ModuleNotFoundError:  # pragma: no cover
    from .extended_data import dedup_control_pool, ensure_galaxy_frame
    from .extended_stats import (
        bootstrap_difference,
        empirical_p_two_sided,
        holm_correction,
        safe_json,
        standardized_mean_difference,
    )


MATCHING_CANDIDATES = [
    "logMstar",
    "z_numeric",
    "rank",
    "log_group_mass",
    "log_group_luminosity",
    "velocity_dispersion",
]
GROUP_MATCHING_CANDIDATES = [
    "z_group",
    "logMstar_bgg",
    "log_group_luminosity",
    "velocity_dispersion",
]
SPATIAL_DIAGNOSTICS = ["dist2BGG_kpc", "R_norm"]
OUTCOMES = {
    "quenched_fraction": ("quenched", np.mean),
    "starforming_fraction": ("starforming", np.mean),
    "elliptical_fraction": ("elliptical", np.mean),
    "spiral_fraction": ("spiral", np.mean),
    "residual_sSFR_starforming": ("MS_res", np.mean),
    "colour_residual_u_minus_r": ("delta_u_minus_r", np.mean),
}

EFFECT_LABELS = {
    "quenched_fraction": "quenched fraction",
    "starforming_fraction": "star-forming fraction",
    "elliptical_fraction": "elliptical fraction",
    "spiral_fraction": "spiral fraction",
    "residual_sSFR_starforming": "residual sSFR, star-forming",
    "colour_residual_u_minus_r": r"colour residual $u-r$",
}

N_BOOT_DEFAULT = 9999


def _physical_group(frame):
    if "physical_group" in frame:
        return frame["physical_group"].astype(str)
    return frame["group_uid"].astype(str)


def prepare_matched_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate the control pool and enforce the hard constraints."""

    prepared = dedup_control_pool(frame)
    if "objid" in prepared and "is_CG4" in prepared:
        controls = prepared.loc[prepared["is_CG4"] == 0, "objid"]
        assert controls.is_unique, "duplicate control objids after dedup"
        cg4 = set(prepared.loc[prepared["is_CG4"] == 1, "objid"])
        assert not (set(controls) & cg4), "control pool contains CG4 objids"
    return prepared


def _select_variables(frame):
    variables = []
    for column in MATCHING_CANDIDATES:
        if (
            column in frame
            and frame.loc[frame["is_CG4"] == 1, column].notna().mean() >= 0.7
        ):
            if frame.loc[frame["is_CG4"] == 0, column].notna().mean() >= 0.7:
                variables.append(column)
    return variables


def _greedy_match(frame, variables):
    required = ["is_CG4", *variables]
    work = frame[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    propensity_variables = list(variables)
    if not propensity_variables:
        return [], work, None
    means = work[propensity_variables].mean()
    scales = work[propensity_variables].std(ddof=0).replace(0, 1)
    design = (work[propensity_variables] - means) / scales
    propensity_model = LogisticRegression(max_iter=2000, random_state=20260612)
    propensity_model.fit(design, work["is_CG4"])
    propensity = np.clip(propensity_model.predict_proba(design)[:, 1], 1e-8, 1 - 1e-8)
    work["propensity_logit"] = np.log(propensity / (1 - propensity))
    caliper = 0.2 * float(work["propensity_logit"].std(ddof=0))

    treated = work.loc[work["is_CG4"] == 1]
    pairs = []
    for rank_value in sorted(treated["rank"].dropna().unique()):
        tx = work.loc[(work["is_CG4"] == 1) & (work["rank"] == rank_value)]
        cx = work.loc[(work["is_CG4"] == 0) & (work["rank"] == rank_value)]
        if tx.empty or cx.empty:
            continue
        distance = np.abs(
            tx["propensity_logit"].to_numpy()[:, None]
            - cx["propensity_logit"].to_numpy()[None, :]
        )
        order = np.argsort(distance.min(axis=1))
        available = set(range(len(cx)))
        for treated_position in order:
            candidates = sorted(
                available, key=lambda position: distance[treated_position, position]
            )
            if not candidates:
                break
            control_position = candidates[0]
            if distance[treated_position, control_position] > caliper:
                continue
            available.remove(control_position)
            pairs.append(
                {
                    "treated_index": tx.index[treated_position],
                    "control_index": cx.index[control_position],
                    "distance": float(distance[treated_position, control_position]),
                }
            )
    return pairs, work, caliper


def matched_pairs(frame: pd.DataFrame):
    """Shared entry point: dedup the control pool, then match.

    Returns ``(pairs, work, caliper, prepared_frame, variables)``. Used by
    both this module and the size analysis so the two always operate on the
    identical matched sample.
    """

    prepared = prepare_matched_frame(frame)
    variables = _select_variables(prepared)
    if not variables:
        return [], None, None, prepared, variables
    pairs, work, caliper = _greedy_match(prepared, variables)
    return pairs, work, caliper, prepared, variables


def _provenance_table(prepared, control_indices):
    controls = prepared.loc[control_indices]
    labels = controls.get(
        "control_source_labels", pd.Series("", index=controls.index)
    ).fillna(controls["sample"].astype(str))
    return pd.DataFrame(
        {
            "objid": controls.get("objid"),
            "lim_group": controls.get("Group"),
            "physical_group": _physical_group(controls),
            "kept_label": controls["sample"].astype(str),
            "control_definitions": labels,
            "physically_RG4": labels.str.contains("RG4"),
        }
    ).reset_index(drop=True)


def _plot_effects(effects, path):
    rows = [
        (key, value) for key, value in effects.items() if value.get("status") == "ok"
    ]
    if not rows:
        return None
    labels = [
        EFFECT_LABELS.get(
            key, key.replace("starforming", "star-forming").replace("_", " ")
        )
        for key, _ in rows
    ]
    values = np.array([value["delta_cg4_minus_control"] for _, value in rows])
    lows = np.array([value["ci95"][0] for _, value in rows])
    highs = np.array([value["ci95"][1] for _, value in rows])
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(rows) + 1.5))
    y = np.arange(len(rows))
    ax.errorbar(
        values,
        y,
        xerr=[values - lows, highs - values],
        fmt="o",
        capsize=3,
        color="#2864A6",
    )
    ax.axvline(0, color="0.45", linestyle=":", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("CG4 minus matched control")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


COVARIATE_LABELS = {
    "logMstar": r"$\log M_\star$",
    "z_numeric": r"$z$",
    "rank": "rank",
    "log_group_luminosity": r"$\log L_{\rm group}$",
    "velocity_dispersion": r"$\sigma_v$",
}


def _plot_balance(before, after, path):
    if not before:
        return None
    variables = list(before)
    y = np.arange(len(variables))
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(variables) + 1.5))
    ax.scatter([abs(before[v]) for v in variables], y, label="Before", color="#A74752")
    ax.scatter([abs(after[v]) for v in variables], y, label="After", color="#25876E")
    ax.axvline(0.1, color="0.45", linestyle=":", linewidth=1)
    ax.set_yticks(y, [COVARIATE_LABELS.get(v, v) for v in variables])
    ax.invert_yaxis()
    ax.set_xlabel("Absolute standardized mean difference")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def _complementarity_status(treated, control):
    """Summarize redundant binary outcomes in the matched complete-case subsets."""

    pairs = {
        "quenched_starforming": ("quenched", "starforming"),
        "elliptical_spiral": ("elliptical", "spiral"),
    }
    summary = {}
    for name, (left, right) in pairs.items():
        if not {left, right}.issubset(treated.columns) or not {
            left,
            right,
        }.issubset(control.columns):
            summary[name] = {"status": "skipped", "reason": "missing_columns"}
            continue
        mask = (
            treated[left].notna()
            & treated[right].notna()
            & control[left].notna()
            & control[right].notna()
        )
        if int(mask.sum()) == 0:
            summary[name] = {"status": "skipped", "reason": "no_complete_pairs"}
            continue
        treated_sum = treated.loc[mask, left] + treated.loc[mask, right]
        control_sum = control.loc[mask, left] + control.loc[mask, right]
        summary[name] = {
            "status": "ok",
            "n_pairs": int(mask.sum()),
            "exact_complements": bool(
                np.allclose(treated_sum, 1.0) and np.allclose(control_sum, 1.0)
            ),
        }
    return summary


def _group_table(frame, label_priority=("RG4", "Control4B", "Control4C")):
    """One row per group: elliptical-vote satellite counts and matching covariates.

    CG4 groups are label-scoped quartets; on the control side one quartet is
    kept per *physical* Lim group (priority RG4 > Control4B > Control4C)
    since different control definitions select different quartets of the
    same group.
    """

    rows = []
    seen_physical = set()
    order = {"CG4": -1, **{label: i for i, label in enumerate(label_priority)}}
    for (label, uid), group in sorted(
        frame.groupby(["sample", "group_uid"], observed=True),
        key=lambda item: order.get(item[0][0], 99),
    ):
        physical = _physical_group(group).iloc[0]
        is_cg4 = int(group["is_CG4"].iloc[0])
        if not is_cg4:
            if physical in seen_physical:
                continue
            seen_physical.add(physical)
        satellites = group.loc[group["rank"] > 1] if "rank" in group else group.iloc[1:]
        classified = satellites["elliptical"].notna()
        bgg = group.loc[group["rank"] == 1] if "rank" in group else group.iloc[:1]
        rows.append(
            {
                "group_uid": uid,
                "physical_group": physical,
                "sample": label,
                "is_CG4": is_cg4,
                "n_sat_classified": int(classified.sum()),
                "n_smooth_sat": int(satellites.loc[classified, "elliptical"].sum()),
                "z_group": float(group["z_group_numeric"].median())
                if "z_group_numeric" in group
                else np.nan,
                "logMstar_bgg": float(bgg["logMstar"].median()) if len(bgg) else np.nan,
                "log_group_luminosity": float(group["log_group_luminosity"].median())
                if "log_group_luminosity" in group
                else np.nan,
                "velocity_dispersion": float(group["velocity_dispersion"].median())
                if "velocity_dispersion" in group
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _permutation_p_sign_flip(differences, seed=20260612, n_perm=9999):
    """Paired sign-flip permutation p-value (add-one convention).

    Under the sharp null hypothesis of no effect, the sign of each pair
    difference is exchangeable. ``n_perm`` random sign assignments are drawn
    and the fraction at least as extreme as the observed mean difference is
    returned (two-sided, adding one to numerator and denominator).
    """
    rng = np.random.default_rng(seed)
    n = len(differences)
    obs = float(np.mean(differences))
    signs = 2 * rng.integers(0, 2, (n_perm, n)) - 1
    null_means = np.mean(signs * differences[np.newaxis, :], axis=1)
    n_extreme = int(np.sum(np.abs(null_means) >= abs(obs)))
    return float((n_extreme + 1) / (n_perm + 1))


def _group_table_for_control(prepared, control_label):
    """One row per group for CG4 vs a single control label (no cross-label dedup).

    Unlike _group_table, which keeps only one quartet per physical group across
    all control types, this function retains all CG4 groups and all groups of
    the specified control type, because each control type selects a different
    quartet of the same physical group.
    """

    rows = []
    for (label, uid), group in prepared.groupby(["sample", "group_uid"], observed=True):
        if label not in ("CG4", control_label):
            continue
        is_cg4 = int(group["is_CG4"].iloc[0])
        satellites = group.loc[group["rank"] > 1] if "rank" in group else group.iloc[1:]
        classified = satellites["elliptical"].notna()
        bgg = group.loc[group["rank"] == 1] if "rank" in group else group.iloc[:1]
        rows.append(
            {
                "group_uid": uid,
                "physical_group": _physical_group(group).iloc[0],
                "sample": label,
                "is_CG4": is_cg4,
                "n_sat_classified": int(classified.sum()),
                "n_smooth_sat": int(satellites.loc[classified, "elliptical"].sum()),
                "mean_sat_logMstar": float(satellites["logMstar"].mean())
                if "logMstar" in satellites
                else np.nan,
                "z_group": float(group["z_group_numeric"].median())
                if "z_group_numeric" in group
                else np.nan,
                "logMstar_bgg": float(bgg["logMstar"].median()) if len(bgg) else np.nan,
                "log_group_luminosity": float(group["log_group_luminosity"].median())
                if "log_group_luminosity" in group
                else np.nan,
                "velocity_dispersion": float(group["velocity_dispersion"].median())
                if "velocity_dispersion" in group
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _run_one_group_match(table, variables, seed=20260612, n_boot=N_BOOT_DEFAULT):
    """Fit 1:1 propensity group match and return full diagnostics.

    Returns a dict with n_matched, delta, CI, bootstrap p, sign-flip
    permutation p, SMD before/after for the four matching covariates, and the
    SMD of mean satellite log M* per group (diagnostic only).
    """

    if not variables:
        return {"status": "skipped", "reason": "no_matching_variables"}
    work = table.dropna(subset=variables).copy()
    work = work.loc[work["n_sat_classified"] > 0]
    if work["is_CG4"].sum() < 5 or (1 - work["is_CG4"]).sum() < 5:
        return {"status": "skipped", "reason": "too_few_groups"}

    # Propensity score model
    means = work[variables].mean()
    scales = work[variables].std(ddof=0).replace(0, 1)
    design = (work[variables] - means) / scales
    model = LogisticRegression(max_iter=2000, random_state=seed)
    model.fit(design, work["is_CG4"])
    propensity = np.clip(model.predict_proba(design)[:, 1], 1e-8, 1 - 1e-8)
    work["propensity_logit"] = np.log(propensity / (1 - propensity))
    caliper = 0.2 * float(work["propensity_logit"].std(ddof=0))

    # SMD before matching for each covariate
    treated_all = work.loc[work["is_CG4"] == 1]
    controls_all = work.loc[work["is_CG4"] == 0]
    smd_before = {
        col: standardized_mean_difference(treated_all[col], controls_all[col])
        for col in variables
    }
    if "mean_sat_logMstar" in work.columns:
        smd_before["mean_sat_logMstar"] = standardized_mean_difference(
            treated_all["mean_sat_logMstar"], controls_all["mean_sat_logMstar"]
        )

    # Greedy 1:1 nearest-neighbour matching without replacement
    treated = treated_all
    controls = controls_all
    distance = np.abs(
        treated["propensity_logit"].to_numpy()[:, None]
        - controls["propensity_logit"].to_numpy()[None, :]
    )
    order = np.argsort(distance.min(axis=1))
    available = set(range(len(controls)))
    pair_rows = []
    for position in order:
        candidates = sorted(available, key=lambda c: distance[position, c])
        if not candidates:
            break
        chosen = candidates[0]
        if distance[position, chosen] > caliper:
            continue
        available.remove(chosen)
        pair_rows.append((treated.index[position], controls.index[chosen]))

    n_pairs = len(pair_rows)
    if n_pairs < 5:
        return {
            "status": "skipped",
            "reason": "too_few_matched_groups",
            "n_matched_groups": n_pairs,
        }

    t_idx = [row[0] for row in pair_rows]
    c_idx = [row[1] for row in pair_rows]

    # SMD after matching
    smd_after = {
        col: standardized_mean_difference(work.loc[t_idx, col], work.loc[c_idx, col])
        for col in variables
    }
    if "mean_sat_logMstar" in work.columns:
        smd_after["mean_sat_logMstar"] = standardized_mean_difference(
            work.loc[t_idx, "mean_sat_logMstar"],
            work.loc[c_idx, "mean_sat_logMstar"],
        )

    t_frac = (
        work.loc[t_idx, "n_smooth_sat"] / work.loc[t_idx, "n_sat_classified"]
    ).to_numpy()
    c_frac = (
        work.loc[c_idx, "n_smooth_sat"] / work.loc[c_idx, "n_sat_classified"]
    ).to_numpy()
    differences = t_frac - c_frac
    obs_delta = float(np.mean(differences))

    # Bootstrap CI and p
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for index in range(n_boot):
        draw = rng.integers(0, n_pairs, n_pairs)
        boot[index] = float(np.mean(t_frac[draw]) - np.mean(c_frac[draw]))
    low, high = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))

    # Sign-flip permutation p
    p_perm = _permutation_p_sign_flip(differences, seed=seed, n_perm=n_boot)

    def _count_dist(subset):
        counts = work.loc[subset, "n_smooth_sat"].value_counts().sort_index()
        return {str(int(k)): int(v) for k, v in counts.items()}

    # Satellite-mass balance audit: the match conditions on group covariates
    # but never on satellite stellar mass, so quantify the per-pair gap in
    # mean satellite log M* (CG4 minus control) with a pair-resampling
    # bootstrap CI (each pair contains exactly one CG4 group, so pair
    # resampling is group-blocked). A fresh generator keeps the existing
    # bootstrap and permutation streams untouched.
    satellite_mass_balance = {"status": "skipped", "reason": "no_mean_sat_logMstar"}
    if "mean_sat_logMstar" in work.columns:
        t_mass = work.loc[t_idx, "mean_sat_logMstar"].to_numpy(dtype=float)
        c_mass = work.loc[c_idx, "mean_sat_logMstar"].to_numpy(dtype=float)
        mass_mask = np.isfinite(t_mass) & np.isfinite(c_mass)
        if int(mass_mask.sum()) < 5:
            satellite_mass_balance = {
                "status": "skipped",
                "reason": "too_few_complete_pairs",
            }
        else:
            mass_diff = t_mass[mass_mask] - c_mass[mass_mask]
            mass_rng = np.random.default_rng(seed + 1)
            n_mass = int(mass_diff.size)
            mass_boot = np.empty(n_boot)
            for index in range(n_boot):
                draw = mass_rng.integers(0, n_mass, n_mass)
                mass_boot[index] = float(np.mean(mass_diff[draw]))
            satellite_mass_balance = {
                "status": "ok",
                "n_pairs": n_mass,
                "mean_paired_diff_dex": float(np.mean(mass_diff)),
                "ci95": [
                    float(np.quantile(mass_boot, 0.025)),
                    float(np.quantile(mass_boot, 0.975)),
                ],
                "smd_before": smd_before.get("mean_sat_logMstar"),
                "smd_after": smd_after.get("mean_sat_logMstar"),
            }

    return {
        "status": "ok",
        "unit": "group",
        "matching_variables": variables,
        "n_cg4_groups_available": int(treated_all.shape[0]),
        "n_control_groups_available": int(controls_all.shape[0]),
        "n_matched_groups": n_pairs,
        "propensity_caliper_logit_sd": 0.2,
        "delta_smooth_satellite_fraction": obs_delta,
        "ci95": [float(low), float(high)],
        "p": empirical_p_two_sided(boot),
        "p_permutation": p_perm,
        "n_boot": int(n_boot),
        "p_floor": float(2 / (n_boot + 1)),
        "mean_fraction_cg4": float(np.mean(t_frac)),
        "mean_fraction_control": float(np.mean(c_frac)),
        "n_smooth_sat_distribution_cg4": _count_dist(t_idx),
        "n_smooth_sat_distribution_control": _count_dist(c_idx),
        "matched_control_composition": {
            key: int(value)
            for key, value in work.loc[c_idx, "sample"].value_counts().items()
        },
        "satellite_mass_balance": satellite_mass_balance,
        "balance": {
            "before": smd_before,
            "after": smd_after,
            "max_abs_smd_before": float(
                max((abs(v) for v in smd_before.values() if v is not None), default=0)
            ),
            "max_abs_smd_after": float(
                max((abs(v) for v in smd_after.values() if v is not None), default=0)
            ),
        },
    }


def group_level_matched_analysis(prepared, n_boot=N_BOOT_DEFAULT, seed=20260612):
    """Primary group-level contrast: elliptical-vote satellite counts per group.

    CG groups are matched 1:1 without replacement to control groups on
    redshift, BGG stellar mass, total quartet luminosity and velocity
    dispersion (propensity caliper 0.2 SD of the logit). The effect is the
    matched difference in per-group elliptical-vote satellite fraction, with a
    pair-resampling bootstrap (groups are the independent unit) and an
    add-one empirical p-value, plus a paired sign-flip permutation p
    (9999 permutations, add-one). SMD diagnostics are stored for the four
    matching covariates before and after matching, plus mean satellite
    log M* per group as an unrestricted balance diagnostic.
    The pooled match draws from all three control types combined;
    its 47/5/2 (Control4B/RG4/Control4C) composition is reported.
    """

    table = _group_table(prepared)
    # Augment with per-group mean satellite stellar mass (balance diagnostic).
    if "logMstar" in prepared.columns and "rank" in prepared.columns:
        sat_mass = (
            prepared.loc[prepared["rank"] > 1]
            .groupby("group_uid", observed=True)["logMstar"]
            .mean()
        )
        table["mean_sat_logMstar"] = table["group_uid"].map(sat_mass)
    if table.empty:
        return {"status": "skipped", "reason": "no_groups"}
    variables = [
        column
        for column in GROUP_MATCHING_CANDIDATES
        if column in table and table[column].notna().mean() >= 0.7
    ]
    if not variables:
        return {"status": "skipped", "reason": "no_matching_variables"}
    result = _run_one_group_match(table, variables, seed=seed, n_boot=n_boot)
    return result


def per_control_group_level_matches(prepared, n_boot=N_BOOT_DEFAULT, seed=20260612):
    """Per-control group-level matched contrasts.

    Runs 1:1 propensity group matching three times, each time restricting the
    control pool to a single control type (Control4B, Control4C, RG4). Within
    one control type, each physical Lim group contributes at most one quartet,
    so no cross-label deduplication is needed. Returns a dict keyed by control
    label, each with the same diagnostics as group_level_matched_analysis
    (n pairs, delta, CI, bootstrap p, sign-flip permutation p, SMD balance).
    """

    results = {}
    for control_label in ("Control4B", "Control4C", "RG4"):
        table = _group_table_for_control(prepared, control_label)
        if "logMstar" in prepared.columns and "rank" in prepared.columns:
            sat_mass = (
                prepared.loc[prepared["rank"] > 1]
                .groupby("group_uid", observed=True)["logMstar"]
                .mean()
            )
            table["mean_sat_logMstar"] = table["group_uid"].map(sat_mass)
        variables = [
            column
            for column in GROUP_MATCHING_CANDIDATES
            if column in table and table[column].notna().mean() >= 0.7
        ]
        results[control_label] = _run_one_group_match(
            table, variables, seed=seed, n_boot=n_boot
        )
        # Sensitivity variant (satellite-mass audit follow-up): repeat the
        # match with per-group mean satellite log M* added to the matching
        # covariates. The published primary match above is unchanged; this
        # variant asks whether the contrast survives when the audited
        # variable is balanced by design rather than modelled away.
        if (
            results[control_label].get("status") == "ok"
            and "mean_sat_logMstar" in table.columns
            and variables
        ):
            variant = _run_one_group_match(
                table,
                variables + ["mean_sat_logMstar"],
                seed=seed,
                n_boot=n_boot,
            )
            if isinstance(variant, dict):
                variant["role"] = (
                    "Sensitivity variant with mean satellite log M* added to "
                    "the matching covariates; unadjusted p, reported as a "
                    "diagnostic of the corresponding primary per-control match."
                )
            results[control_label]["satellite_mass_balanced_variant"] = variant
    return results


def _group_level_holm_sensitivity(per_control_group):
    """Holm sensitivity for the three per-control group-level p-value sets."""

    control_labels = ["Control4B", "Control4C", "RG4"]
    ok_rows = [
        (label, per_control_group.get(label, {}))
        for label in control_labels
        if per_control_group.get(label, {}).get("status") == "ok"
    ]
    boot_adjusted = holm_correction([row.get("p") for _, row in ok_rows])
    perm_adjusted = holm_correction([row.get("p_permutation") for _, row in ok_rows])
    return {
        "status": "ok" if ok_rows else "skipped",
        "family": control_labels,
        "note": (
            "Sensitivity only: the three control-specific group-level matches are "
            "kept as distinct estimands in the main analysis, while the pooled "
            "match is a secondary summary and is not part of this Holm family."
        ),
        "rows": [
            {
                "control": label,
                "p_bootstrap": row.get("p"),
                "p_bootstrap_holm": boot_adj,
                "p_permutation": row.get("p_permutation"),
                "p_permutation_holm": perm_adj,
            }
            for (label, row), boot_adj, perm_adj in zip(
                ok_rows, boot_adjusted, perm_adjusted
            )
        ],
    }


def run_matched_control_analysis(
    data, output_dir: str | None = None, n_boot: int = N_BOOT_DEFAULT
):
    """Run the deduplicated 1:1 matching and the group-level primary."""

    frame = ensure_galaxy_frame(data)
    if frame.empty or "is_CG4" not in frame:
        return {"status": "skipped", "reason": "no_galaxy_samples"}
    n_control_rows = int((frame["is_CG4"] == 0).sum())
    pairs, match_frame, caliper, prepared, variables = matched_pairs(frame)
    if not variables:
        return {"status": "skipped", "reason": "no_matching_variables"}
    if len(pairs) < 10:
        return {
            "status": "skipped",
            "reason": "too_few_matches",
            "matching_variables": variables,
            "n_cg4_matched": len(pairs),
        }

    treated_indices = [pair["treated_index"] for pair in pairs]
    control_indices = [pair["control_index"] for pair in pairs]
    treated = prepared.loc[treated_indices].reset_index(drop=True)
    control = prepared.loc[control_indices].reset_index(drop=True)

    # Hard constraints (also enforced upstream; asserted here so any
    # regression fails loudly rather than biasing the effect toward null).
    if "objid" in prepared:
        control_objids = prepared.loc[control_indices, "objid"]
        assert control_objids.is_unique, "a control galaxy was used twice"
        assert not (
            set(control_objids) & set(prepared.loc[prepared["is_CG4"] == 1, "objid"])
        ), "self-match: a CG4 galaxy entered the control side"

    before = {
        column: standardized_mean_difference(
            match_frame.loc[match_frame["is_CG4"] == 1, column],
            match_frame.loc[match_frame["is_CG4"] == 0, column],
        )
        for column in variables
    }
    after = {
        column: standardized_mean_difference(treated[column], control[column])
        for column in variables
    }

    treated_blocks = _physical_group(prepared.loc[treated_indices]).to_numpy()
    effects = {}
    for name, (column, statistic) in OUTCOMES.items():
        if column not in treated or column not in control:
            effects[name] = {"status": "skipped", "reason": "missing_outcome_column"}
            continue
        tx, cx = treated[column], control[column]
        blocks = treated_blocks
        if name == "residual_sSFR_starforming":
            mask = treated["starforming"].eq(1) & control["starforming"].eq(1)
            tx, cx = tx[mask], cx[mask]
            blocks = treated_blocks[mask.to_numpy()]
        effect = bootstrap_difference(
            tx, cx, statistic=statistic, paired=True, n_boot=n_boot, blocks=blocks
        )
        if effect["estimate"] is None:
            effects[name] = {"status": "skipped", "reason": "no_complete_matched_pairs"}
        else:
            effects[name] = {
                "status": "ok",
                "delta_cg4_minus_control": effect["estimate"],
                "ci95": effect["ci95"],
                "p": effect["p"],
                "p_floor": effect["p_floor"],
                "n_boot": effect["n_boot"],
                "resampling_unit": effect["resampling_unit"],
                "n_blocks": effect["n_blocks"],
                "n_pairs": effect["n"],
            }
    ok_names = [name for name, value in effects.items() if value.get("status") == "ok"]
    for name, adjusted in zip(
        ok_names, holm_correction([effects[name]["p"] for name in ok_names])
    ):
        effects[name]["p_adj"] = adjusted

    # Post-hoc decomposition of the all-pair matched elliptical-fraction
    # difference into its satellite component (audit question: is the
    # all-pair value diluted by BGG pairs, whose adjusted elliptical OR is
    # below 1?). Exact rank matching means both members of a pair share the
    # same rank, so restricting to treated satellites restricts both sides.
    # Reported with its own unadjusted p and kept OUT of the published
    # matched-outcome Holm family above, which is untouched.
    satellite_decomposition = {"status": "skipped", "reason": "missing_columns"}
    if "rank" in treated and "elliptical" in treated and "elliptical" in control:
        sat_mask = treated["rank"].gt(1) & control["rank"].gt(1)
        effect = bootstrap_difference(
            treated.loc[sat_mask, "elliptical"],
            control.loc[sat_mask, "elliptical"],
            statistic=np.mean,
            paired=True,
            n_boot=n_boot,
            blocks=treated_blocks[sat_mask.to_numpy()],
        )
        if effect["estimate"] is None:
            satellite_decomposition = {
                "status": "skipped",
                "reason": "no_complete_satellite_pairs",
            }
        else:
            satellite_decomposition = {
                "status": "ok",
                "outcome": "elliptical_fraction",
                "n_satellite_pairs_total": int(sat_mask.sum()),
                "n_pairs": effect["n"],
                "delta_cg4_minus_control": effect["estimate"],
                "ci95": effect["ci95"],
                "p": effect["p"],
                "p_floor": effect["p_floor"],
                "n_boot": effect["n_boot"],
                "resampling_unit": effect["resampling_unit"],
                "n_blocks": effect["n_blocks"],
                "multiplicity": (
                    "Unadjusted post-hoc decomposition diagnostic of the "
                    "published all-pair matched elliptical-fraction outcome; "
                    "not a member of the matched-outcome Holm family."
                ),
            }

    provenance = _provenance_table(prepared, control_indices)
    group_level = group_level_matched_analysis(frame, n_boot=n_boot)
    per_control_group = per_control_group_level_matches(frame, n_boot=n_boot)
    group_level_holm_sensitivity = _group_level_holm_sensitivity(per_control_group)

    result = {
        "status": "ok",
        "method": (
            "1:1 propensity-score nearest neighbour without replacement, exact "
            "rank strata, on the objid-deduplicated control pool"
        ),
        "replacement": False,
        "common_support": "enforced_by_caliper",
        "propensity_caliper_logit_sd": 0.2,
        "propensity_caliper": caliper,
        "matching_variables": variables,
        "propensity_variables": variables,
        "exact_matching_variables": ["rank"],
        "spatial_variables_not_matched": [
            column for column in SPATIAL_DIAGNOSTICS if column in prepared
        ],
        "spatial_exclusion_reason": (
            "Projected distance is a defining compactness/interaction variable and lacks "
            "adequate overlap; it is retained for phase-space and tidal diagnostics."
        ),
        "control_pool_deduplicated_by_objid": True,
        "cg4_objids_excluded_from_controls": True,
        "n_control_rows_before_dedup": n_control_rows,
        "n_control_galaxies_unique_pool": int((prepared["is_CG4"] == 0).sum()),
        "n_cg4_matched": int(len(treated)),
        "n_control_matched": int(len(control)),
        "n_control_unique": int(control["objid"].nunique())
        if "objid" in control
        else int(len(set(control_indices))),
        "matched_counts_by_sample": {
            key: int(value)
            for key, value in (
                prepared.loc[treated_indices + control_indices, "sample"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "matched_control_counts_by_sample": {
            key: int(value)
            for key, value in control["sample"].value_counts().sort_index().items()
        },
        "matched_control_counts_by_provenance": {
            key: int(value)
            for key, value in provenance["control_definitions"]
            .value_counts()
            .sort_index()
            .items()
        },
        "n_matched_controls_physically_RG4": int(provenance["physically_RG4"].sum()),
        "median_match_distance": float(np.median([pair["distance"] for pair in pairs])),
        "balance": {"before": before, "after": after},
        "max_abs_smd_before": max(
            abs(value) for value in before.values() if value is not None
        ),
        "max_abs_smd_after": max(
            abs(value) for value in after.values() if value is not None
        ),
        "effects": effects,
        "satellite_decomposition": satellite_decomposition,
        "group_level": group_level,
        "group_level_per_control": per_control_group,
        "group_level_multiplicity_policy": (
            "The three per-control group-level satellite-composition contrasts "
            "are treated as separate scientific estimands because Control4B, "
            "Control4C, and RG4 answer different control questions. P-values are "
            "therefore reported without adjustment across control definitions; "
            "the pooled group match is a secondary summary. The paired sign-flip "
            "permutation p-value is the primary paired test, while the bootstrap "
            "p-value and interval describe robustness and effect-size uncertainty."
        ),
        "group_level_holm_sensitivity": group_level_holm_sensitivity,
        "holm_correction_family": ok_names,
        "holm_correction_note": (
            "Quenched/star-forming and elliptical/spiral diagnostics are retained "
            "in the same matched-outcome Holm family. These paired binary outcomes "
            "are complementary on their complete-case subsets, so the correction is "
            "conservative rather than anti-conservative."
        ),
        "complementarity_audit": _complementarity_status(treated, control),
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result["effects_figure"] = _plot_effects(
            effects, os.path.join(output_dir, "fig_matched_control_effects.pdf")
        )
        result["balance_figure"] = _plot_balance(
            before, after, os.path.join(output_dir, "fig_matched_control_balance.pdf")
        )
        provenance_path = os.path.normpath(
            os.path.join(output_dir, os.pardir, "matched_control_provenance.csv")
        )
        provenance.to_csv(provenance_path, index=False)
        result["provenance_file"] = os.path.basename(provenance_path)
    return safe_json(result)
