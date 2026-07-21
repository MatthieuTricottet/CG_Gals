"""T3 — velocity-dispersion documentation and no-sigma_v sensitivity.

Three parts.

(1) Monte Carlo noise of the sigma_v estimator actually used: the gapper
    (Wainer & Thissen 1976) on N = 4 line-of-sight velocities, as
    implemented for every sample's quartet (including Control4B/4C
    quartets embedded in richer hosts). We draw 1e5 Gaussian quartets and
    quote the median and 16-84% range of sigma_gapper/sigma_true.

(2) Enumeration of every fitted model family that includes sigma_v
    (``velocity_dispersion``), with the numeric completeness rule that
    governs its inclusion (adjusted models: covariate enters when
    non-missing for >= 65% of the model frame; matching: >= 70%), plus
    the realized completeness per family.

(3) Sensitivity reruns WITHOUT sigma_v (column dropped before the
    covariate/matching-variable auto-selection): the three per-control
    adjusted families, the pooled secondary models, the galaxy-level
    propensity match, and the group-level matched satellite-composition
    contrasts (pooled + per-control). Same seeds, B = 9999, cluster and
    Holm conventions; a labelled sensitivity family, never folded into
    the published ones.

Outputs: referee/values/T3.json, referee/T3_no_sigma_table.csv,
referee/T3_summary.md (hand-written afterwards).
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extended_data import ensure_galaxy_frame  # noqa: E402
from matched_controls import run_matched_control_analysis  # noqa: E402
from primary_contrasts import run_primary_contrasts  # noqa: E402
from specialness_models import MODEL_SPECS, fit_logistic_specialness_models  # noqa: E402

OUT = ROOT / "referee"
SEED = 20260721
N_MC = 100_000


def gapper_ratio_mc() -> dict:
    rng = np.random.default_rng(SEED)
    draws = np.sort(rng.standard_normal((N_MC, 4)), axis=1)
    gaps = np.diff(draws, axis=1)
    weights = np.array([3.0, 4.0, 3.0])  # i*(n-i) for n=4
    sigma = np.sqrt(np.pi) / 12.0 * gaps @ weights
    q = np.quantile(sigma, [0.16, 0.5, 0.84])
    return {
        "n_draws": N_MC,
        "seed": SEED,
        "median_ratio": float(q[1]),
        "p16_ratio": float(q[0]),
        "p84_ratio": float(q[2]),
        "mean_ratio": float(sigma.mean()),
        "prob_below_half": float((sigma < 0.5).mean()),
        "prob_above_1p5": float((sigma > 1.5).mean()),
    }


def sigma_uses_by_grep() -> list[dict]:
    uses = []
    for path in sorted((ROOT / "src").glob("*.py")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"velocity_dispersion", line) and "def " not in line:
                uses.append({"file": f"src/{path.name}", "line": number,
                             "code": line.strip()[:110]})
    return uses


def _model_digest(block: dict) -> dict:
    return {
        key: {
            "odds_ratio": block[key].get("cg4_odds_ratio"),
            "ci95": block[key].get("cg4_ci95"),
            "p": block[key].get("cg4_p"),
            "p_adj": block[key].get("cg4_p_adj"),
            "n": block[key].get("n"),
        }
        for key in MODEL_SPECS
        if isinstance(block.get(key), dict) and block[key].get("status") == "ok"
    }


def _matched_digest(mc: dict) -> dict:
    digest = {
        "matching_variables": mc.get("matching_variables"),
        "n_cg4_matched": mc.get("n_cg4_matched"),
        "effects": {},
        "per_control_group": {},
    }
    for name, effect in (mc.get("effects") or {}).items():
        if isinstance(effect, dict) and effect.get("status") == "ok":
            digest["effects"][name] = {
                "delta": effect.get("delta_cg4_minus_control"),
                "ci95": effect.get("ci95"),
                "p": effect.get("p"),
                "p_adj": effect.get("p_adj"),
                "n_pairs": effect.get("n_pairs"),
            }
    for label, row in (mc.get("group_level_per_control") or {}).items():
        if isinstance(row, dict) and row.get("status") == "ok":
            digest["per_control_group"][label] = {
                "variables": row.get("matching_variables"),
                "n_pairs": row.get("n_matched_groups"),
                "delta": row.get("delta_smooth_satellite_fraction"),
                "ci95": row.get("ci95"),
                "p_bootstrap": row.get("p"),
                "p_permutation": row.get("p_permutation"),
                "satellite_mass_balance": (row.get("satellite_mass_balance") or {}).get(
                    "mean_paired_diff_dex"),
            }
    return digest


def main() -> None:
    with open(ROOT / "data" / "processed_sample.pkl", "rb") as handle:
        data = pickle.load(handle)
    frame = ensure_galaxy_frame(data)
    frame_wo = frame.drop(columns=["velocity_dispersion"])

    with open(ROOT / "output" / "results.json") as handle:
        es = json.load(handle)["extended_specialness"]

    values: dict = {"mc_gapper": gapper_ratio_mc()}
    print("gapper MC:", values["mc_gapper"])

    # ---- (2) enumeration -------------------------------------------------
    completeness = {
        "rule_adjusted_models": "covariate used when notna fraction >= 0.65 of the model frame (specialness_models._covariates)",
        "rule_matching": "variable used when notna fraction >= 0.70 (both arms for the galaxy match; group tables >= 0.70)",
        "realized": {},
    }
    for control in ["Control4B", "Control4C", "RG4"]:
        subset = frame.loc[frame["sample"].isin(["CG4", control])]
        completeness["realized"][f"CG4+{control}"] = float(
            subset["velocity_dispersion"].notna().mean())
    completeness["realized"]["pooled_frame"] = float(
        frame["velocity_dispersion"].notna().mean())
    values["sigma_v_completeness"] = completeness
    values["sigma_v_model_families"] = {
        "primary_contrasts_covariates": es["primary_contrasts"].get(
            "covariates_considered"),
        "pooled_covariates": es["specialness_models"].get(
            "covariates_considered") or es["specialness_models"].get("covariates"),
        "galaxy_match_variables": es["matched_controls"].get("matching_variables"),
        "group_match_variables_by_control": {
            label: (row or {}).get("matching_variables")
            for label, row in (es["matched_controls"].get(
                "group_level_per_control") or {}).items()
            if isinstance(row, dict)
        },
        "other_uses_grep": sigma_uses_by_grep(),
    }

    # ---- (3) reruns without sigma_v -------------------------------------
    print("rerunning primary contrasts without sigma_v ...")
    primary_wo = run_primary_contrasts(None, frame=frame_wo)
    print("rerunning pooled models without sigma_v ...")
    pooled_wo = fit_logistic_specialness_models(frame_wo)
    print("rerunning matched analyses without sigma_v (9999 bootstraps) ...")
    matched_wo = run_matched_control_analysis(frame_wo)

    values["published"] = {
        "primary": {
            control: _model_digest(es["primary_contrasts"]["contrasts"][control])
            for control in ["Control4B", "Control4C", "RG4"]
        },
        "pooled": _model_digest(es["specialness_models"]),
        "matched": _matched_digest(es["matched_controls"]),
    }
    values["no_sigma_v"] = {
        "primary": {
            control: _model_digest(primary_wo["contrasts"][control])
            for control in ["Control4B", "Control4C", "RG4"]
        },
        "pooled": _model_digest(pooled_wo),
        "matched": _matched_digest(matched_wo),
    }

    rows = []
    for family in ["Control4B", "Control4C", "RG4"]:
        pub, alt = values["published"]["primary"][family], values["no_sigma_v"]["primary"][family]
        for model in sorted(set(pub) | set(alt)):
            rows.append({
                "family": f"primary:{family}", "model": model,
                "published_or": pub.get(model, {}).get("odds_ratio"),
                "published_p_adj": pub.get(model, {}).get("p_adj"),
                "no_sigma_or": alt.get(model, {}).get("odds_ratio"),
                "no_sigma_p_adj": alt.get(model, {}).get("p_adj"),
            })
    for model in sorted(set(values["published"]["pooled"]) | set(values["no_sigma_v"]["pooled"])):
        rows.append({
            "family": "pooled", "model": model,
            "published_or": values["published"]["pooled"].get(model, {}).get("odds_ratio"),
            "published_p_adj": values["published"]["pooled"].get(model, {}).get("p_adj"),
            "no_sigma_or": values["no_sigma_v"]["pooled"].get(model, {}).get("odds_ratio"),
            "no_sigma_p_adj": values["no_sigma_v"]["pooled"].get(model, {}).get("p_adj"),
        })
    table = pd.DataFrame(rows)
    flips = []
    for _, row in table.iterrows():
        a, b = row["published_p_adj"], row["no_sigma_p_adj"]
        if a is not None and b is not None and not (pd.isna(a) or pd.isna(b)):
            if (a < 0.05) != (b < 0.05):
                flips.append(f"{row['family']}/{row['model']}")
    values["holm_significance_flips"] = flips

    (OUT / "values").mkdir(exist_ok=True)
    with open(OUT / "values" / "T3.json", "w") as handle:
        json.dump(values, handle, indent=1, default=float)
    table.to_csv(OUT / "T3_no_sigma_table.csv", index=False)
    print(table.to_string(index=False))
    print("\nHolm significance flips (published vs no-sigma_v):", flips or "none")
    print("matched effects (no sigma):", json.dumps(values["no_sigma_v"]["matched"], indent=1)[:1200])


if __name__ == "__main__":
    main()
