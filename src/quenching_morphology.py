"""Quenching and morphology, each at fixed values of the other.

Within a GZ1 class the CG4 and control galaxies do not have the same
stellar-mass distribution (CG4 satellites are less massive than Control4B
satellites of the same class), and the quenched fraction rises steeply with
mass, so comparisons at fixed morphology must also hold stellar mass fixed.
Stellar mass enters continuously; the class-by-mass-tercile Kitagawa split
is kept only as a descriptive complement.

For each control and for satellites (primary) and all galaxies:

* quenched ~ CG4 + GZ1 class + covariates (quenching at fixed morphology),
  with a spline in stellar mass and a class x mass interaction as checks,
  and with the projected BGG distance added (satellites);
* E class ~ CG4 + sSFR class + covariates (morphology at fixed quenching);
* the missing-sSFR extreme assignments propagated to the adjusted models.

Covariates are those of Table 4 (log M*, redshift, BGG/satellite indicator,
quartet luminosity, sigma_v), standardised, with cluster-robust errors by
physical Lim group.  Standardised fraction differences use
``extended_stats.standardized_contrast``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from extended_data import ensure_galaxy_frame
    from extended_stats import (
        fit_logistic_model,
        restricted_cubic_spline,
        safe_json,
        standardized_contrast,
    )
except ModuleNotFoundError:  # pragma: no cover
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import (
        fit_logistic_model,
        restricted_cubic_spline,
        safe_json,
        standardized_contrast,
    )

CONTROLS = ["Control4B", "Control4C", "RG4"]
COVARIATES = ["logMstar", "z_numeric", "log_group_luminosity", "velocity_dispersion"]
MORPHOLOGIES = ["Elliptical", "Spiral", "Uncertain"]
N_BOOT = 2000


def _frame(frame: pd.DataFrame, control: str, population: str) -> pd.DataFrame:
    work = frame.loc[frame["sample"].isin(["CG4", control])].copy()
    if population == "satellites":
        work = work.loc[work["is_satellite"] == 1].copy()
    radius = pd.to_numeric(work["dist2BGG_kpc"], errors="coerce")
    work["log_R_bgg"] = np.log10(radius.where(radius > 0))
    knots = np.nanquantile(work["logMstar"], [0.1, 0.5, 0.9])
    work["logMstar_rcs"] = restricted_cubic_spline(work["logMstar"], knots)
    work["elliptical_x_logMstar"] = work["elliptical"] * (
        work["logMstar"] - work["logMstar"].mean()
    )
    return work


def _covariates(population: str) -> list[str]:
    return COVARIATES if population == "satellites" else [*COVARIATES, "is_satellite"]


def _continuous(predictors: list[str]) -> list[str]:
    return [
        p
        for p in predictors
        if p in (*COVARIATES, "log_R_bgg", "logMstar_rcs", "elliptical_x_logMstar")
    ]


def adjusted_models(frame: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """Mutually conditioned quenching and morphology models per control."""

    results = {}
    for population in ("satellites", "all"):
        results[population] = {}
        for control in CONTROLS:
            work = _frame(frame, control, population)
            # both classifications required, so the four models share a frame
            both = work.dropna(subset=["quenched", "elliptical"]).copy()
            covs = _covariates(population)
            specs = {
                "quenched_given_class": ("quenched", ["is_CG4", "elliptical", *covs]),
                "quenched_given_class_mass_spline": (
                    "quenched",
                    ["is_CG4", "elliptical", *covs, "logMstar_rcs"],
                ),
                "quenched_given_class_x_mass": (
                    "quenched",
                    ["is_CG4", "elliptical", *covs, "elliptical_x_logMstar"],
                ),
                "quenched_no_class": ("quenched", ["is_CG4", *covs]),
                "class_given_quenched": ("elliptical", ["is_CG4", "quenched", *covs]),
            }
            if population == "satellites":
                specs["quenched_given_class_radius"] = (
                    "quenched",
                    ["is_CG4", "elliptical", *covs, "log_R_bgg"],
                )
                specs["class_given_quenched_radius"] = (
                    "elliptical",
                    ["is_CG4", "quenched", *covs, "log_R_bgg"],
                )
            entry = {"n_frame": int(len(both))}
            for name, (outcome, predictors) in specs.items():
                entry[name] = fit_logistic_model(
                    both, outcome, predictors, continuous=_continuous(predictors)
                )
            if population == "satellites":
                predictors = ["is_CG4", "elliptical", *covs]
                entry["quenched_given_class_standardised"] = standardized_contrast(
                    both, "quenched", predictors,
                    continuous=_continuous(predictors), n_boot=n_boot,
                )
            results[population][control] = entry
    return results


def missing_ssfr_extremes(frame: pd.DataFrame) -> dict:
    """Satellite quenched models under the four extreme missing-sSFR assignments.

    Every CG4 satellite without an sSFR is set to quenched or to star-forming,
    and independently every control satellite without one; the model with
    and without the GZ1 class is refitted in each corner.
    """

    results = {}
    missing = frame["sSFR_status"].astype(str).eq("NosSFR")
    for control in CONTROLS:
        entry = {}
        for cg4_value in (0.0, 1.0):
            for control_value in (0.0, 1.0):
                work = frame.copy()
                fill = np.where(work["is_CG4"] == 1, cg4_value, control_value)
                work["quenched"] = np.where(missing, fill, work["quenched"])
                work = _frame(work, control, "satellites")
                key = (
                    f"cg4_{'Q' if cg4_value else 'SF'}__control_{'Q' if control_value else 'SF'}"
                )
                entry[key] = {
                    "quenched_no_class": fit_logistic_model(
                        work, "quenched", ["is_CG4", *COVARIATES], continuous=COVARIATES
                    ),
                    "quenched_given_class": fit_logistic_model(
                        work,
                        "quenched",
                        ["is_CG4", "elliptical", *COVARIATES],
                        continuous=COVARIATES,
                    ),
                }
        results[control] = entry
    return results


def class_mass_kitagawa(frame: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> dict:
    """Descriptive Kitagawa split of the satellite quenched-fraction difference.

    Cells are GZ1 class (E, S, uncertain) crossed with stellar-mass terciles
    of the pooled satellite frame; the conditional term compares quenched
    fractions within cells.  The tercile edges are arbitrary, so the result
    complements, and does not replace, the continuous-mass models.
    Intervals resample physical groups within each sample.
    """

    satellites = frame.loc[
        (frame["is_satellite"] == 1)
        & frame["quenched"].notna()
        & frame["morphology"].isin(MORPHOLOGIES)
    ].copy()
    edges = np.quantile(satellites["logMstar"], [0, 1 / 3, 2 / 3, 1])
    edges[0] -= 1.0
    edges[-1] += 1.0
    satellites["mass_tercile"] = pd.cut(satellites["logMstar"], edges, labels=False)

    def split(treated, control, cells):
        n1 = treated.groupby(cells).size()
        n0 = control.groupby(cells).size()
        q1 = treated.groupby(cells)["quenched"].mean()
        q0 = control.groupby(cells)["quenched"].mean()
        index = n1.index.union(n0.index)
        n1, n0 = n1.reindex(index, fill_value=0), n0.reindex(index, fill_value=0)
        q1, q0 = q1.reindex(index).fillna(0.0), q0.reindex(index).fillna(0.0)
        w1, w0 = n1 / n1.sum(), n0 / n0.sum()
        shared = (n1 > 0) & (n0 > 0)
        conditional = float((((w1 + w0) / 2) * (q1 - q0))[shared].sum())
        raw = float(treated["quenched"].mean() - control["quenched"].mean())
        return raw, conditional

    rng = np.random.default_rng(seed)
    results = {"tercile_edges": [float(e) for e in edges[1:-1]], "n_boot": n_boot, "seed": seed}
    cg4 = satellites.loc[satellites["sample"] == "CG4"]
    for control in CONTROLS:
        ctrl = satellites.loc[satellites["sample"] == control]
        entry = {}
        for label, cells in (("class", ["morphology"]), ("class_x_mass", ["morphology", "mass_tercile"])):
            raw, conditional = split(cg4, ctrl, cells)
            members_t = [g.index.to_numpy() for _, g in cg4.groupby("physical_group")]
            members_c = [g.index.to_numpy() for _, g in ctrl.groupby("physical_group")]
            draws = []
            for _ in range(n_boot):
                t = cg4.loc[np.concatenate([members_t[j] for j in rng.integers(0, len(members_t), len(members_t))])]
                c = ctrl.loc[np.concatenate([members_c[j] for j in rng.integers(0, len(members_c), len(members_c))])]
                draws.append(split(t, c, cells)[1])
            entry[label] = {
                "raw_difference": raw,
                "conditional_term": conditional,
                "conditional_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            }
        results[control] = entry
    return results


def run_quenching_morphology_analysis(data, output_dir: str | None = None, n_boot: int = N_BOOT):
    """Run the mutually conditioned quenching/morphology analysis."""

    del output_dir  # no figure
    frame = ensure_galaxy_frame(data)
    within_class_mass = {}
    for population, mask in (("satellites", frame["is_satellite"] == 1),):
        part = frame.loc[mask & frame["morphology"].isin(["Elliptical", "Spiral"])]
        within_class_mass[population] = {
            f"{sample}:{morph}": float(g["logMstar"].median())
            for (sample, morph), g in part.groupby(["sample", "morphology"])
        }
    return safe_json(
        {
            "status": "ok",
            "covariates": COVARIATES,
            "median_logMstar_by_sample_and_class": within_class_mass,
            "models": adjusted_models(frame, n_boot=n_boot),
            "missing_ssfr_extremes": missing_ssfr_extremes(frame),
            "kitagawa_class_mass": class_mass_kitagawa(frame),
        }
    )
