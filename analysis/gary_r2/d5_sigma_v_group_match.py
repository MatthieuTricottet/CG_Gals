"""D5 -- sigma_v sensitivity of the group-level match (gary-r2 Phase 1).

Re-derives the per-control group-level propensity matches with and without
``velocity_dispersion`` (identical code path to
``matched_controls._run_one_group_match``: standardised logistic propensity,
caliper 0.2 SD of the logit, greedy nearest neighbour without replacement)
and compares the two matched sets: shared control groups, SMD of every
covariate (including sigma_v and mean satellite log M*) and the direction of
the sigma_v imbalance in the no-sigma_v set.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import CONTROLS, ROOT, load_results, load_sample, store_diagnostic, write_csv

from extended_data import build_galaxy_frame  # noqa: E402
from extended_stats import standardized_mean_difference  # noqa: E402
from matched_controls import GROUP_MATCHING_CANDIDATES, _group_table_for_control  # noqa: E402

SEED = 20260612
COVARIATES = ["z_group", "logMstar_bgg", "log_group_luminosity", "velocity_dispersion",
              "mean_sat_logMstar"]


def match(table: pd.DataFrame, variables: list[str]):
    work = table.dropna(subset=variables).copy()
    work = work.loc[work["n_sat_classified"] > 0]
    means = work[variables].mean()
    scales = work[variables].std(ddof=0).replace(0, 1)
    design = (work[variables] - means) / scales
    model = LogisticRegression(max_iter=2000, random_state=SEED).fit(design, work["is_CG4"])
    p = np.clip(model.predict_proba(design)[:, 1], 1e-8, 1 - 1e-8)
    work["logit"] = np.log(p / (1 - p))
    caliper = 0.2 * float(work["logit"].std(ddof=0))
    treated = work.loc[work["is_CG4"] == 1]
    controls = work.loc[work["is_CG4"] == 0]
    dist = np.abs(treated["logit"].to_numpy()[:, None] - controls["logit"].to_numpy()[None, :])
    order = np.argsort(dist.min(axis=1))
    available = set(range(len(controls)))
    pairs = []
    for pos in order:
        cands = sorted(available, key=lambda c: dist[pos, c])
        if not cands:
            break
        c = cands[0]
        if dist[pos, c] > caliper:
            continue
        available.remove(c)
        pairs.append((treated.index[pos], controls.index[c]))
    t_idx = [a for a, _ in pairs]
    c_idx = [b for _, b in pairs]
    frac = lambda idx: (work.loc[idx, "n_smooth_sat"] / work.loc[idx, "n_sat_classified"]).to_numpy()
    delta = float(np.mean(frac(t_idx) - frac(c_idx)))
    return work, t_idx, c_idx, delta, caliper


def main() -> None:
    sample = load_sample()
    results = load_results()
    t3 = json.load(open(ROOT / "referee" / "values" / "T3.json"))
    pub = results["extended_specialness"]["matched_controls"]["group_level_per_control"]
    pub_nosig = t3["no_sigma_v"]["matched"]["per_control_group"]

    # the pipeline passes the NON-deduplicated frame to per_control_group_level_matches
    # (matched_controls.run_matched_control_analysis, line ~939)
    prepared = build_galaxy_frame(sample)
    sat_mass = prepared.loc[prepared["rank"] > 1].groupby("group_uid", observed=True)["logMstar"].mean()

    out = {}
    smd_rows = []
    for ctrl in CONTROLS:
        table = _group_table_for_control(prepared, ctrl)
        table["mean_sat_logMstar"] = table["group_uid"].map(sat_mass)
        with_sig = [c for c in GROUP_MATCHING_CANDIDATES if table[c].notna().mean() >= 0.7]
        no_sig = [c for c in with_sig if c != "velocity_dispersion"]
        sets = {}
        for label, variables in (("with_sigma_v", with_sig), ("without_sigma_v", no_sig)):
            work, t_idx, c_idx, delta, caliper = match(table, variables)
            ref = pub[ctrl]["delta_smooth_satellite_fraction"] if label == "with_sigma_v" else pub_nosig[ctrl]["delta"]
            assert abs(delta - ref) < 1e-9, (ctrl, label, delta, ref)
            smd = {}
            for cov in COVARIATES:
                smd[cov] = {
                    "before": standardized_mean_difference(
                        work.loc[work["is_CG4"] == 1, cov], work.loc[work["is_CG4"] == 0, cov]),
                    "after": standardized_mean_difference(work.loc[t_idx, cov], work.loc[c_idx, cov]),
                    "median_cg4": float(work.loc[t_idx, cov].median()),
                    "median_control": float(work.loc[c_idx, cov].median()),
                }
                smd_rows.append(dict(control=ctrl, match=label, covariate=cov, **smd[cov]))
            sets[label] = dict(
                variables=variables, n_pairs=len(t_idx), delta=delta, caliper_logit=caliper,
                control_groups=[str(g) for g in work.loc[c_idx, "group_uid"]],
                cg4_groups=[str(g) for g in work.loc[t_idx, "group_uid"]],
                mean_fraction_cg4=float(np.mean(work.loc[t_idx, "n_smooth_sat"] / work.loc[t_idx, "n_sat_classified"])),
                mean_fraction_control=float(np.mean(work.loc[c_idx, "n_smooth_sat"] / work.loc[c_idx, "n_sat_classified"])),
                smd=smd,
            )
        a, b = sets["with_sigma_v"], sets["without_sigma_v"]
        shared = set(a["control_groups"]) & set(b["control_groups"])
        shared_cg4 = set(a["cg4_groups"]) & set(b["cg4_groups"])
        # same-pair overlap
        pairs_a = set(zip(a["cg4_groups"], a["control_groups"]))
        pairs_b = set(zip(b["cg4_groups"], b["control_groups"]))
        sig_no = b["smd"]["velocity_dispersion"]
        out[ctrl] = {
            "with_sigma_v": {k: v for k, v in a.items() if k not in ("control_groups", "cg4_groups")},
            "without_sigma_v": {k: v for k, v in b.items() if k not in ("control_groups", "cg4_groups")},
            "n_shared_control_groups": len(shared),
            "fraction_shared_control_groups_of_with": len(shared) / a["n_pairs"],
            "fraction_shared_control_groups_of_without": len(shared) / b["n_pairs"],
            "n_shared_cg4_groups": len(shared_cg4),
            "n_identical_pairs": len(pairs_a & pairs_b),
            "sigma_v_imbalance_without": {
                "smd_after": sig_no["after"],
                "direction": ("controls have HIGHER sigma_v than CG4" if sig_no["after"] < 0
                              else "CG4 have HIGHER sigma_v than controls"),
                "median_cg4_kms": sig_no["median_cg4"], "median_control_kms": sig_no["median_control"],
            },
            "delta_change": b["delta"] - a["delta"],
            "mean_fraction_control_change": b["mean_fraction_control"] - a["mean_fraction_control"],
            "mean_fraction_cg4_change": b["mean_fraction_cg4"] - a["mean_fraction_cg4"],
        }
        print(ctrl, json.dumps({k: v for k, v in out[ctrl].items() if k not in ("with_sigma_v", "without_sigma_v")}, indent=1))
        for label in ("with_sigma_v", "without_sigma_v"):
            print("  ", label, sets[label]["n_pairs"], round(sets[label]["delta"], 3),
                  {c: round(sets[label]["smd"][c]["after"], 2) for c in COVARIATES})
    smd_tab = pd.DataFrame(smd_rows)
    write_csv(smd_tab, "d5_group_match_smd_with_vs_without_sigma_v.csv")

    c4b = out["Control4B"]
    interpretation = (
        f"For Control4B the two matched control sets share only {c4b['n_shared_control_groups']}/"
        f"{c4b['with_sigma_v']['n_pairs']} control groups ({100*c4b['fraction_shared_control_groups_of_with']:.0f}%) "
        f"and {c4b['n_identical_pairs']} identical pairs; dropping sigma_v changes the matched-control mean "
        f"elliptical-satellite fraction by {c4b['mean_fraction_control_change']:+.3f} and the CG4 side by "
        f"{c4b['mean_fraction_cg4_change']:+.3f}, so the shift of Delta from {c4b['with_sigma_v']['delta']:.3f} to "
        f"{c4b['without_sigma_v']['delta']:.3f} comes from which control quartets are selected, not from the CG4 side. "
        f"In the no-sigma_v set the residual sigma_v SMD is {c4b['sigma_v_imbalance_without']['smd_after']:+.2f} "
        f"({c4b['sigma_v_imbalance_without']['direction']}; medians "
        f"{c4b['sigma_v_imbalance_without']['median_cg4_kms']:.0f} vs {c4b['sigma_v_imbalance_without']['median_control_kms']:.0f} km/s). "
        f"Because Control4B quartets embedded in rich hosts carry high gapper sigma_v, matching ON sigma_v pulls in "
        f"controls from dynamically hotter (richer, more early-type-rich) hosts, which raises the control elliptical "
        f"fraction and lowers Delta; matching WITHOUT it selects controls from cooler hosts. With four tracers "
        f"sigma_v is a noisy covariate (Sect. 2.4), so neither Delta is 'the' answer: the pair of values brackets "
        f"the Control4B contrast, and the ordering Control4C < Control4B, RG4 holds in both. Note that the no-sigma_v "
        f"Control4B set is NOT better balanced overall: its redshift SMD grows to "
        f"{c4b['without_sigma_v']['smd']['z_group']['after']:+.2f} (from {c4b['with_sigma_v']['smd']['z_group']['after']:+.2f}), "
        f"so the published with-sigma_v match remains the primary one and the no-sigma_v value a labelled sensitivity."
    )
    print(interpretation)
    out["interpretation"] = interpretation
    out["smd_file"] = "d5_group_match_smd_with_vs_without_sigma_v.csv"
    store_diagnostic("d5_sigma_v_group_match", out)


if __name__ == "__main__":
    main()
