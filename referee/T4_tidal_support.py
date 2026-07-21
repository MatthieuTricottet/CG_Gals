"""T4 — pairwise tidal index: exact definition, common support, refits.

(1) Extracts, from the code actually used (src/tidal_indices.py), the
    precise definition of T_i = sum_j M*_j / R_ij^3 and its complete-case
    rules, for Sect. 3.6:
      * co-members: the galaxy's own *selected quartet* (sample-scoped
        group), not the full Lim host; no velocity window;
      * R_ij: great-circle separation x D_A(median member z), proper kpc;
      * groups with any member missing RA/Dec/z are skipped whole;
        co-members with missing logM* contribute zero to T_i (nansum),
        so T_i is a lower bound in those groups; log T_i needs T_i > 0;
      * the models pool all control labels WITHOUT objid deduplication
        (unlike the audited pooled models), clustering by physical group.

(2) Common-support diagnostics for log T_i: per-sample quantiles, the
    overlap coefficient between CG4 and the controls, the CG4 fraction
    above the control 95th percentile, the point-biserial correlation of
    the CG4 indicator with log T_i, and the VIF of is_CG4 in the adjusted
    design.

(3) Sensitivity refits of the elliptical and quenched models with the
    tidal term: (a) trimmed to the common-support window (intersection of
    the CG4 and control [2.5, 97.5] percentile ranges), (b) natural cubic
    spline (df=4) in log T_i instead of the linear term, (c) control pool
    deduplicated by objid. CG4 ORs are reported side by side with the
    published linear/full-pool fits.

Outputs: referee/values/T4.json, referee/T4_support_table.csv,
referee/T4_logT_support.pdf, referee/T4_summary.md (hand-written).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extended_data import dedup_control_pool, ensure_galaxy_frame  # noqa: E402
from extended_stats import fit_logistic_model  # noqa: E402
from tidal_indices import _derive  # noqa: E402

OUT = ROOT / "referee"
QUANTILES = [0.05, 0.16, 0.25, 0.50, 0.75, 0.84, 0.95]


def overlap_coefficient(a: np.ndarray, b: np.ndarray, n_bins: int = 60) -> float:
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    grid = np.linspace(lo, hi, n_bins + 1)
    fa, _ = np.histogram(a, bins=grid, density=True)
    fb, _ = np.histogram(b, bins=grid, density=True)
    width = grid[1] - grid[0]
    return float(np.sum(np.minimum(fa, fb)) * width)


def natural_spline_basis(x: np.ndarray, df: int = 4) -> pd.DataFrame:
    """Natural cubic spline basis with df columns (quantile knots).

    Standard construction (Hastie et al., ESL eq. 5.4-5.5): K = df + 1
    knots at quantiles; basis {x, N_1..N_{K-2}} with
    d_j(x) = [(x-k_j)_+^3 - (x-k_K)_+^3]/(k_K - k_j), N_j = d_j - d_{K-1}.
    """

    knots = np.quantile(x, np.linspace(0.05, 0.95, df + 1))

    def d(index):
        return ((np.clip(x - knots[index], 0, None) ** 3
                 - np.clip(x - knots[-1], 0, None) ** 3)
                / (knots[-1] - knots[index]))

    columns = {"ns_1": x}
    d_last = d(len(knots) - 2)
    for j in range(len(knots) - 2):
        columns[f"ns_{j + 2}"] = d(j) - d_last
    return pd.DataFrame(columns, index=pd.RangeIndex(len(x)))


def fit_with_tidal(work: pd.DataFrame, outcome: str, tidal_cols: list[str],
                   continuous_extra: list[str]) -> dict:
    predictors = ["is_CG4", "logMstar", "is_satellite", *tidal_cols]
    return fit_logistic_model(
        work, outcome, predictors,
        continuous=["logMstar", *continuous_extra],
    )


def digest(fit: dict) -> dict:
    return {
        "status": fit.get("status"),
        "n": fit.get("n"),
        "cg4_odds_ratio": fit.get("cg4_odds_ratio"),
        "cg4_ci95": fit.get("cg4_ci95"),
        "cg4_p": fit.get("cg4_p"),
    }


def main() -> None:
    with open(ROOT / "data" / "processed_sample.pkl", "rb") as handle:
        data = pickle.load(handle)
    frame = ensure_galaxy_frame(data)
    work = _derive(frame)

    complete = work["nearest_projected_distance"].notna()
    values: dict = {
        "definition": {
            "co_members": "selected quartet only (sample-scoped group_uid), not the full Lim host",
            "velocity_window": "none: all quartet co-members enter regardless of velocity difference",
            "distance": "great-circle separation x angular-diameter distance at the median member redshift (proper kpc; astropy Planck15)",
            "missing_mass_rule": "co-members with missing logM* contribute zero to T_i (nansum): T_i is a lower bound there",
            "group_completeness_rule": "groups with any member missing RA/Dec/z are skipped whole",
            "pooling": "models pool all control labels without objid deduplication (rows duplicated across labels), clustered by physical group",
        },
        "n_rows_total": int(len(work)),
        "n_galaxies_with_pairs": int(complete.sum()),
    }

    # quantify the missing-mass lower-bound rule
    mass_missing = work["logMstar"].isna()
    groups_with_missing_mass = work.loc[mass_missing, "group_uid"].nunique()
    values["n_groups_with_missing_mass_member"] = int(groups_with_missing_mass)
    values["n_groups_total"] = int(work["group_uid"].nunique())

    modeled = work.dropna(subset=["log_tidal_index", "is_CG4"])
    cg4 = modeled.loc[modeled["is_CG4"] == 1, "log_tidal_index"].to_numpy()
    controls_all = modeled.loc[modeled["is_CG4"] == 0]
    ctrl = controls_all["log_tidal_index"].to_numpy()
    ctrl_unique = (controls_all.drop_duplicates("objid")["log_tidal_index"]
                   .to_numpy())

    values["support"] = {
        "quantiles_logT": {
            "CG4": {str(q): float(np.quantile(cg4, q)) for q in QUANTILES},
            "controls_pooled_rows": {str(q): float(np.quantile(ctrl, q))
                                     for q in QUANTILES},
            "controls_unique_objid": {str(q): float(np.quantile(ctrl_unique, q))
                                      for q in QUANTILES},
        },
        "overlap_coefficient_rows": overlap_coefficient(cg4, ctrl),
        "overlap_coefficient_unique": overlap_coefficient(cg4, ctrl_unique),
        "cg4_above_control_p95_rows": float(
            (cg4 > np.quantile(ctrl, 0.95)).mean()),
        "cg4_above_control_p95_unique": float(
            (cg4 > np.quantile(ctrl_unique, 0.95)).mean()),
        "point_biserial_r": float(np.corrcoef(
            modeled["is_CG4"].to_numpy(dtype=float),
            modeled["log_tidal_index"].to_numpy())[0, 1]),
    }
    design = modeled[["is_CG4", "logMstar", "is_satellite", "log_tidal_index"]].dropna()
    x = design[["logMstar", "is_satellite", "log_tidal_index"]].to_numpy(dtype=float)
    y = design["is_CG4"].to_numpy(dtype=float)
    x1 = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x1, y, rcond=None)
    r2 = 1 - np.sum((y - x1 @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
    values["support"]["vif_is_cg4"] = float(1.0 / (1.0 - r2))
    values["support"]["r2_is_cg4_on_covariates"] = float(r2)

    # ---- refits ---------------------------------------------------------
    models: dict = {}
    for outcome in ["elliptical", "quenched"]:
        entry = {}
        entry["published_linear_fullpool"] = digest(fit_with_tidal(
            work, outcome, ["log_tidal_index"], ["log_tidal_index"]))
        lo = max(np.quantile(cg4, 0.025), np.quantile(ctrl, 0.025))
        hi = min(np.quantile(cg4, 0.975), np.quantile(ctrl, 0.975))
        trimmed = work[(work["log_tidal_index"] >= lo)
                       & (work["log_tidal_index"] <= hi)]
        entry["trimmed_common_support"] = digest(fit_with_tidal(
            trimmed, outcome, ["log_tidal_index"], ["log_tidal_index"]))
        entry["trim_window_logT"] = [float(lo), float(hi)]

        splined = work.dropna(subset=["log_tidal_index"]).reset_index(drop=True)
        basis = natural_spline_basis(
            splined["log_tidal_index"].to_numpy(), df=4)
        splined = pd.concat([splined, basis], axis=1)
        spline_cols = list(basis.columns)
        entry["spline_logT"] = digest(fit_with_tidal(
            splined, outcome, spline_cols, spline_cols))

        dedup = dedup_control_pool(work)
        entry["dedup_control_pool"] = digest(fit_with_tidal(
            dedup, outcome, ["log_tidal_index"], ["log_tidal_index"]))
        models[outcome] = entry
    values["models"] = models

    # ---- figure ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    order = ["CG4", "Control4B", "Control4C", "RG4"]
    series = [modeled.loc[modeled["sample"] == s, "log_tidal_index"].dropna()
              for s in order]
    parts = ax.violinplot(series, showmedians=True, widths=0.8)
    for body in parts["bodies"]:
        body.set_alpha(0.6)
    ax.set_xticks(range(1, len(order) + 1), order)
    ax.set_ylabel(r"$\log_{10}\,T_i\ \, [\mathrm{M_\odot\,kpc^{-3}}]$")
    fig.tight_layout()
    fig.savefig(OUT / "T4_logT_support.pdf", bbox_inches="tight")
    plt.close(fig)

    (OUT / "values").mkdir(exist_ok=True)
    with open(OUT / "values" / "T4.json", "w") as handle:
        json.dump(values, handle, indent=1, default=float)

    rows = []
    for outcome, entry in models.items():
        for variant, fit in entry.items():
            if isinstance(fit, dict):
                rows.append({"outcome": outcome, "variant": variant, **fit})
    pd.DataFrame(rows).to_csv(OUT / "T4_support_table.csv", index=False)

    print(json.dumps(values["support"], indent=1))
    for outcome, entry in models.items():
        print(f"== {outcome}")
        for variant, fit in entry.items():
            if isinstance(fit, dict) and fit.get("status") == "ok":
                print(f"  {variant:28s} OR={fit['cg4_odds_ratio']:.2f} "
                      f"[{fit['cg4_ci95'][0]:.2f},{fit['cg4_ci95'][1]:.2f}] "
                      f"p={fit['cg4_p']:.4f} n={fit['n']}")
            elif isinstance(fit, dict):
                print(f"  {variant:28s} {fit.get('status')}")


if __name__ == "__main__":
    main()
