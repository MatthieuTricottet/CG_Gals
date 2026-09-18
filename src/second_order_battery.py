"""Exploratory second-order battery (gary-r2 A6; appendix material).

For the CG4 satellites, and for reference within each control sample, a
logistic model of elliptical-versus-spiral morphology (usable GZ1 E/S
classifications) adjusted for log M* and clustered by physical group is fitted
against each group-scale descriptor in turn:

  * log R_ij,med   -- median pairwise projected separation of the quartet (kpc)
  * sigma_v        -- quartet gapper velocity dispersion (km/s)
  * f_L,BGG        -- BGG luminosity fraction (continuous, per 0.1)
  * log t_cross    -- crossing time
  * f_L,BGG > med  -- BGG luminosity fraction above the CG4 median (binary)

Continuous descriptors are standardised inside ``fit_logistic_model``; the
reported odds ratios are therefore per standard deviation of the descriptor
in the fitted frame (the SD is stored).  Benjamini--Hochberg FDR is applied
across the whole battery (4 samples x 5 descriptors) and, separately, across
the five CG4 tests.

The appendix E.2 dominance distribution tests are also reproduced with a
median f_L,BGG split (in addition to the published 60% threshold) so that
the results can be compared with M. Bozzio's.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from astropy.cosmology import Planck15

try:
    import config as co
    from exploration_dom import DOMINATION_QUANTITIES, compute_distribution_tests
    from extended_stats import benjamini_hochberg, fit_logistic_model, safe_json
    from tidal_indices import _angular_matrix
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from .exploration_dom import DOMINATION_QUANTITIES, compute_distribution_tests
    from .extended_stats import benjamini_hochberg, fit_logistic_model, safe_json
    from .tidal_indices import _angular_matrix

SAMPLES = ["CG4", "Control4B", "Control4C", "RG4"]
DESCRIPTORS = {
    "log_R_pair_med": {"label": r"\log R_{ij,\rm med}", "continuous": True},
    "sigma_v": {"label": r"\sigma_v", "continuous": True},
    "f_L_BGG": {"label": r"f_{L,\rm BGG}", "continuous": True},
    "log_t_cross": {"label": r"\log t_{\rm cross}", "continuous": True},
    "f_L_BGG_above_cg4_median": {"label": r"f_{L,\rm BGG} > {\rm med}(\CG)", "continuous": False},
}


def group_median_separations(gals: pd.DataFrame) -> pd.Series:
    """Median of the pairwise projected separations (proper kpc) per quartet."""

    out = {}
    for gid, part in gals.groupby("Group"):
        ra = part["RA"].to_numpy(float)
        dec = part["Dec"].to_numpy(float)
        zs = pd.to_numeric(part["z"], errors="coerce").to_numpy(float)
        if len(part) < 2 or np.isnan(ra).any() or np.isnan(zs).any():
            continue
        ang = _angular_matrix(ra, dec)
        d_a = Planck15.angular_diameter_distance(float(np.median(zs))).to_value("kpc")
        iu = np.triu_indices(len(part), k=1)
        out[gid] = float(np.median(ang[iu] * d_a))
    return pd.Series(out, name="R_pair_med_kpc")


def _satellite_frame(sample: dict, name: str, fl_median_cg4: float) -> pd.DataFrame:
    gals = sample[name + co.GASUFF]
    groups = sample[name + co.GRSUFF].set_index("Group")
    sat = gals.loc[(gals["rank_M"] > 1) & gals["morphology"].isin(["Elliptical", "Spiral"])].copy()
    sat["elliptical"] = (sat["morphology"] == "Elliptical").astype(float)
    sat["logMstar"] = pd.to_numeric(sat["lgm"], errors="coerce")
    sat["physical_group"] = name + ":" + sat["Group"].astype(str)
    seps = group_median_separations(gals)
    sat["log_R_pair_med"] = np.log10(sat["Group"].map(seps))
    sat["sigma_v"] = pd.to_numeric(sat["Group"].map(groups["Vdisp"]), errors="coerce")
    sat["f_L_BGG"] = pd.to_numeric(sat["Group"].map(groups["FracLumBGG"]), errors="coerce")
    tcross = pd.to_numeric(sat["Group"].map(groups["t_cr"]), errors="coerce")
    sat["log_t_cross"] = np.log10(tcross.where(tcross > 0))
    sat["f_L_BGG_above_cg4_median"] = (sat["f_L_BGG"] > fl_median_cg4).astype(float)
    return sat


def run_second_order_battery(sample: dict, output_dir: str | None = None) -> dict:
    cg4_groups = sample["CG4" + co.GRSUFF]
    fl_median_cg4 = float(pd.to_numeric(cg4_groups["FracLumBGG"], errors="coerce").median())
    rows = []
    fits = {}
    for name in SAMPLES:
        sat = _satellite_frame(sample, name, fl_median_cg4)
        fits[name] = {}
        for key, spec in DESCRIPTORS.items():
            continuous = ["logMstar"] + ([key] if spec["continuous"] else [])
            fit = fit_logistic_model(sat, "elliptical", [key, "logMstar"], continuous=continuous, min_n=20)
            fits[name][key] = fit
            term = fit.get("terms", {}).get(key, {}) if fit.get("status") == "ok" else {}
            sd = float(sat[key].dropna().std(ddof=0)) if spec["continuous"] else None
            rows.append({
                "sample": name,
                "descriptor": key,
                "label": spec["label"],
                "n": fit.get("n"),
                "n_groups": fit.get("n_clusters"),
                "odds_ratio": term.get("odds_ratio"),
                "ci95_low": (term.get("ci95") or [None, None])[0],
                "ci95_high": (term.get("ci95") or [None, None])[1],
                "p": term.get("p"),
                "per_unit": (f"per {sd:.3g} SD of {key}" if sd is not None else "binary"),
                "descriptor_sd_in_frame": sd,
                "status": fit.get("status"),
                "reason": fit.get("reason"),
            })
    table = pd.DataFrame(rows)
    table["p_bh_battery"] = benjamini_hochberg(table["p"].tolist())
    cg4_mask = table["sample"].eq("CG4")
    within = pd.Series(np.nan, index=table.index, dtype=float)
    for name in SAMPLES:
        mask = table["sample"].eq(name)
        within.loc[mask] = benjamini_hochberg(table.loc[mask, "p"].tolist())
    table["p_bh_within_sample"] = within

    # E.2 dominance tables with a median f_L,BGG split (per sample median, as
    # the published split is applied per sample with a fixed 60% threshold)
    alt = {}
    for name in SAMPLES:
        groups = sample[name + co.GRSUFF].copy()
        groups["is_dominated"] = groups["FracLumBGG"] >= float(groups["FracLumBGG"].median())
        alt[name + co.GRSUFF] = groups
    median_split = compute_distribution_tests(alt, quantities=DOMINATION_QUANTITIES)
    published_split = compute_distribution_tests(sample, quantities=DOMINATION_QUANTITIES)
    thresholds = {
        name: {
            "median_f_L_BGG": float(sample[name + co.GRSUFF]["FracLumBGG"].median()),
            "n_dominated_median_split": int(alt[name + co.GRSUFF]["is_dominated"].sum()),
            "n_dominated_60pc_split": int(sample[name + co.GRSUFF]["is_dominated"].sum()),
            "n_groups": int(len(sample[name + co.GRSUFF])),
        }
        for name in SAMPLES
    }

    result = {
        "status": "ok",
        "model": "elliptical ~ descriptor + logMstar, satellites with usable E/S GZ1 class, "
                 "binomial GLM, cluster-robust SE by group; continuous descriptors standardised",
        "cg4_median_f_L_BGG": fl_median_cg4,
        "descriptors": DESCRIPTORS,
        "rows": table.to_dict(orient="records"),
        "n_tests_battery": int(table["p"].notna().sum()),
        "n_cg4_tests": int(cg4_mask.sum()),
        "min_p_bh_cg4": float(np.nanmin(table.loc[cg4_mask, "p_bh_battery"].astype(float))),
        "dominance_median_split": {
            "thresholds": thresholds,
            "distribution_tests": median_split.to_dict(orient="records"),
            "published_60pc_distribution_tests": published_split.to_dict(orient="records"),
        },
    }
    os.makedirs(co.OUTPUT_PATH, exist_ok=True)
    table.to_csv(os.path.join(co.OUTPUT_PATH, "second_order_battery.csv"), index=False)
    median_split.to_csv(os.path.join(co.OUTPUT_PATH, "domination_distribution_tests_median_split.csv"), index=False)
    return safe_json(result)
