"""Nearest-neighbour matched comparison of CG4 and ordinary-group galaxies."""

from __future__ import annotations

import os

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

try:
    from extended_data import ensure_galaxy_frame
    from extended_stats import (
        bootstrap_difference,
        holm_correction,
        safe_json,
        standardized_mean_difference,
    )
except ModuleNotFoundError:  # pragma: no cover
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import (
        bootstrap_difference,
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
SPATIAL_DIAGNOSTICS = ["dist2BGG_kpc", "R_norm"]
OUTCOMES = {
    "passive_fraction": ("passive", np.mean),
    "starforming_fraction": ("starforming", np.mean),
    "elliptical_fraction": ("elliptical", np.mean),
    "spiral_fraction": ("spiral", np.mean),
    "residual_sSFR_starforming": ("MS_res", np.mean),
    "colour_residual_u_minus_r": ("delta_u_minus_r", np.mean),
}

EFFECT_LABELS = {
    "passive_fraction": "passive fraction",
    "starforming_fraction": "star-forming fraction",
    "elliptical_fraction": "elliptical fraction",
    "spiral_fraction": "spiral fraction",
    "residual_sSFR_starforming": "residual sSFR, star-forming",
    "colour_residual_u_minus_r": r"colour residual $u-r$",
}


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
        "passive_starforming": ("passive", "starforming"),
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


def run_matched_control_analysis(
    data, output_dir: str | None = None, n_boot: int = 2000
):
    """Run deterministic 1:1 propensity-score matching without replacement."""

    frame = ensure_galaxy_frame(data)
    variables = _select_variables(frame)
    if not variables:
        return {"status": "skipped", "reason": "no_matching_variables"}
    pairs, match_frame, caliper = _greedy_match(frame, variables)
    if len(pairs) < 10:
        return {
            "status": "skipped",
            "reason": "too_few_matches",
            "matching_variables": variables,
            "n_cg4_matched": len(pairs),
        }

    treated_indices = [pair["treated_index"] for pair in pairs]
    control_indices = [pair["control_index"] for pair in pairs]
    treated = frame.loc[treated_indices].reset_index(drop=True)
    control = frame.loc[control_indices].reset_index(drop=True)
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

    effects = {}
    for name, (column, statistic) in OUTCOMES.items():
        if column not in treated or column not in control:
            effects[name] = {"status": "skipped", "reason": "missing_outcome_column"}
            continue
        tx, cx = treated[column], control[column]
        if name == "residual_sSFR_starforming":
            mask = treated["starforming"].eq(1) & control["starforming"].eq(1)
            tx, cx = tx[mask], cx[mask]
        effect = bootstrap_difference(
            tx, cx, statistic=statistic, paired=True, n_boot=n_boot
        )
        if effect["estimate"] is None:
            effects[name] = {"status": "skipped", "reason": "no_complete_matched_pairs"}
        else:
            effects[name] = {
                "status": "ok",
                "delta_cg4_minus_control": effect["estimate"],
                "ci95": effect["ci95"],
                "p": effect["p"],
                "n_pairs": effect["n"],
            }
    ok_names = [name for name, value in effects.items() if value.get("status") == "ok"]
    for name, adjusted in zip(
        ok_names, holm_correction([effects[name]["p"] for name in ok_names])
    ):
        effects[name]["p_adj"] = adjusted

    result = {
        "status": "ok",
        "method": "1:1 propensity-score nearest neighbour without replacement, exact rank strata",
        "replacement": False,
        "common_support": "enforced_by_caliper",
        "propensity_caliper_logit_sd": 0.2,
        "propensity_caliper": caliper,
        "matching_variables": variables,
        "propensity_variables": variables,
        "exact_matching_variables": ["rank"],
        "spatial_variables_not_matched": [
            column for column in SPATIAL_DIAGNOSTICS if column in frame
        ],
        "spatial_exclusion_reason": (
            "Projected distance is a defining compactness/interaction variable and lacks "
            "adequate overlap; it is retained for phase-space and tidal diagnostics."
        ),
        "n_cg4_matched": int(len(treated)),
        "n_control_matched": int(len(control)),
        "n_control_unique": int(len(set(control_indices))),
        "matched_counts_by_sample": {
            key: int(value)
            for key, value in (
                frame.loc[treated_indices + control_indices, "sample"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "matched_control_counts_by_sample": {
            key: int(value)
            for key, value in control["sample"].value_counts().sort_index().items()
        },
        "median_match_distance": float(np.median([pair["distance"] for pair in pairs])),
        "balance": {"before": before, "after": after},
        "max_abs_smd_before": max(
            abs(value) for value in before.values() if value is not None
        ),
        "max_abs_smd_after": max(
            abs(value) for value in after.values() if value is not None
        ),
        "effects": effects,
        "holm_correction_family": ok_names,
        "holm_correction_note": (
            "Passive/star-forming and elliptical/spiral diagnostics are retained "
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
    return safe_json(result)
