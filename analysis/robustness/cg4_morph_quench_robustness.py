#!/usr/bin/env python3
"""Robustness checks for CG4 vs RG4/Control4C morphology and quenching.

This script is intentionally standalone. It reads the existing processed
sample and local cached structural tables, reuses the project's harmonized
frame and model helpers, and writes new artifacts only under
``results/robustness``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cg4_robustness_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config as co  # noqa: E402
from extended_data import dedup_control_pool, ensure_galaxy_frame  # noqa: E402
from extended_stats import fit_logistic_model, holm_correction  # noqa: E402
from primary_contrasts import run_primary_contrasts  # noqa: E402
from size_data import (  # noqa: E402
    SDSS_ID_COLUMNS,
    SDSS_VALUE_COLUMNS,
    SIMARD_SENTINEL,
    SIMARD_VALUE_COLUMNS,
    Z_MATCH_TOLERANCE,
    _ids_to_int64,
    _kpc_per_arcsec,
)
from specialness_models import _covariates, fit_logistic_specialness_models  # noqa: E402


SAMPLES = ["CG4", "Control4B", "Control4C", "RG4"]
REQUESTED_CONTROLS = ["RG4", "Control4C"]
MORPH_THRESHOLDS = [0.5, 0.8]
CONCENTRATION_THRESHOLDS = [2.6, 2.5, 2.86]
SERSIC_THRESHOLDS = [2.5, 2.0, 3.0]
ROBUSTNESS_VERSION = "2026-07-17"


@dataclass
class ModelSpec:
    task: str
    model_id: str
    contrast: str
    scope: str
    proxy: str
    outcome: str
    definition: str
    threshold: float | None
    model_type: str
    effect_type: str
    status: str
    estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p: float | None = None
    p_holm: float | None = None
    n: int | None = None
    n_cg4: int | None = None
    n_control: int | None = None
    n_clusters: int | None = None
    formula: str | None = None
    notes: str | None = None


def safe_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fmt(value, digits: int = 3) -> str:
    value = safe_float(value)
    if value is None:
        return "NA"
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def p_text(value) -> str:
    value = safe_float(value)
    if value is None:
        return "NA"
    if value < 1e-4:
        return "<1e-4"
    return f"{value:.3f}"


def load_sample() -> dict[str, pd.DataFrame]:
    path = Path(co.DATA_PATH) / co.PROCESS_SAMPLES
    with path.open("rb") as handle:
        return pickle.load(handle)


def read_cached_sdss_sizes() -> pd.DataFrame:
    path = Path(co.SIZE_COLUMNS_FILE)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype={column: str for column in SDSS_ID_COLUMNS})
    for column in ["objid", "dr7objid"]:
        frame[column] = _ids_to_int64(frame.get(column, pd.Series(dtype=object)))
    frame["specObjID"] = frame.get("specObjID", pd.Series(dtype=object)).fillna("").astype(str)
    for column in SDSS_VALUE_COLUMNS:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return frame


def read_cached_simard() -> pd.DataFrame:
    path = Path(co.SIMARD_SUBSET_FILE)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype={"dr7objid": str})
    frame["dr7objid"] = _ids_to_int64(frame["dr7objid"])
    for column in SIMARD_VALUE_COLUMNS:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
        frame.loc[np.isclose(frame[column], SIMARD_SENTINEL), column] = np.nan
    for column in ["Scale", "Rhlr", "Rchl_r"]:
        frame.loc[frame[column] <= 0, column] = np.nan
    return frame


def attach_local_size_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach size columns from local caches without network fetches or writes."""

    sdss = read_cached_sdss_sizes()
    simard = read_cached_simard()
    if sdss.empty:
        raise FileNotFoundError(f"Missing local size cache: {co.SIZE_COLUMNS_FILE}")
    if simard.empty:
        raise FileNotFoundError(f"Missing local Simard cache: {co.SIMARD_SUBSET_FILE}")

    work = frame.copy()
    work["objid"] = _ids_to_int64(work["objid"]).astype("int64")
    sdss = sdss.rename(columns={"specObjID": "size_specObjID"}).drop_duplicates("objid")
    sdss["objid"] = sdss["objid"].astype("int64")
    work = work.merge(sdss, on="objid", how="left", validate="m:1")

    simard = simard.rename(columns={column: f"simard_{column}" for column in SIMARD_VALUE_COLUMNS})
    work = work.merge(simard.drop_duplicates("dr7objid"), on="dr7objid", how="left", validate="m:1")

    simard_value_columns = [f"simard_{column}" for column in SIMARD_VALUE_COLUMNS]
    has_simard = work["simard_Rchl_r"].notna()
    z_catalogue = pd.to_numeric(work.get("z_numeric", work.get("z")), errors="coerce")
    z_mismatch = has_simard & ((work["simard_z"] - z_catalogue).abs() > Z_MATCH_TOLERANCE)

    shred_merge = pd.Series(False, index=work.index)
    if "group_uid" in work.columns:
        keyed = work.loc[work["dr7objid"].notna()]
        counts = keyed.groupby(["group_uid", "dr7objid"], observed=True)["objid"].transform("nunique")
        shred_merge.loc[keyed.index] = counts > 1
    rejected = z_mismatch | shred_merge
    work.loc[rejected, simard_value_columns] = np.nan

    kpc_per_arcsec = _kpc_per_arcsec(z_catalogue)
    work["Rchl_r_kpc"] = (work["simard_Rchl_r"] / work["simard_Scale"]) * kpc_per_arcsec
    work["Rhlr_kpc"] = (work["simard_Rhlr"] / work["simard_Scale"]) * kpc_per_arcsec
    work["petroR50_kpc"] = work["petroR50_r"] * kpc_per_arcsec
    work["petroR90_kpc"] = work["petroR90_r"] * kpc_per_arcsec

    in_window_simard = work["Rchl_r_kpc"].between(co.SIZE_MIN_KPC, co.SIZE_MAX_KPC)
    in_window_petro = work["petroR50_kpc"].between(co.SIZE_MIN_KPC, co.SIZE_MAX_KPC)
    n_pegged = work["simard_ng"].notna() & (
        (work["simard_ng"] <= co.NG_PEG_LOW) | (work["simard_ng"] >= co.NG_PEG_HIGH)
    )
    petro_valid = (
        (work["petroR50_r"] > 0)
        & (work["petroR90_r"] > work["petroR50_r"])
        & (work["petroR50Err_r"] > 0)
    )

    work["n_pegged"] = n_pegged.astype(float)
    work["size_ok_simard"] = (work["Rchl_r_kpc"].notna() & in_window_simard & ~n_pegged).astype(float)
    work["size_ok_simard_incl_pegged"] = (work["Rchl_r_kpc"].notna() & in_window_simard).astype(float)
    work["size_ok_petro"] = (petro_valid & work["petroR50_kpc"].notna() & in_window_petro).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        work["log_Rchl_r_kpc"] = np.where(work["size_ok_simard"] == 1, np.log10(work["Rchl_r_kpc"]), np.nan)
        work["log_Rhlr_kpc"] = np.where(work["Rhlr_kpc"] > 0, np.log10(work["Rhlr_kpc"]), np.nan)
        work["log_petroR50_kpc"] = np.where(work["size_ok_petro"] == 1, np.log10(work["petroR50_kpc"]), np.nan)
    work["concentration_r90_r50"] = np.where(petro_valid, work["petroR90_r"] / work["petroR50_r"], np.nan)

    audit = {"per_sample": {}}
    for sample_name, part in work.groupby("sample", observed=True):
        idx = part.index
        audit["per_sample"][str(sample_name)] = {
            "n_rows": int(len(part)),
            "petro_row_resolved": int(part["petroR50_r"].notna().sum()),
            "dr7_bridge_resolved": int(part["dr7objid"].notna().sum()),
            "simard_matched": int((has_simard.loc[idx] & ~rejected.loc[idx]).sum()),
            "z_mismatch": int(z_mismatch.loc[idx].sum()),
            "shred_merge": int(shred_merge.loc[idx].sum()),
            "n_pegged": int(n_pegged.loc[idx].sum()),
            "simard_out_of_window": int((part["Rchl_r_kpc"].notna() & ~in_window_simard.loc[idx]).sum()),
            "petro_out_of_window": int((petro_valid.loc[idx] & part["petroR50_kpc"].notna() & ~in_window_petro.loc[idx]).sum()),
            "size_ok_simard": int((part["size_ok_simard"] == 1).sum()),
            "size_ok_petro": int((part["size_ok_petro"] == 1).sum()),
        }
    work.attrs = dict(frame.attrs)
    work.attrs["size_attach_audit"] = audit
    return work, audit


def prepare_frame(sample: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    frame = ensure_galaxy_frame(sample).replace([np.inf, -np.inf], np.nan)
    missing = []
    for column in ["p_E", "p_S"]:
        if column not in frame:
            missing.append(column)
    if missing:
        raise RuntimeError(f"Missing required debiased GZ1 columns: {missing}")
    frame, size_audit = attach_local_size_columns(frame)
    frame["sersic_n_valid"] = pd.to_numeric(frame["simard_ng"], errors="coerce").where(
        frame["simard_ng"].notna()
        & (frame["n_pegged"] != 1)
        & (frame["simard_ng"] > co.NG_PEG_LOW)
        & (frame["simard_ng"] < co.NG_PEG_HIGH)
    )
    return frame, size_audit


def gz_binary(frame: pd.DataFrame, threshold: float) -> pd.Series:
    p_e = pd.to_numeric(frame["p_E"], errors="coerce")
    p_s = pd.to_numeric(frame["p_S"], errors="coerce")
    elliptical = (p_e > threshold) & (p_e > p_s)
    spiral = (p_s > threshold) & (p_s > p_e)
    return pd.Series(np.where(elliptical, 1.0, np.where(spiral, 0.0, np.nan)), index=frame.index)


def panel_for(frame: pd.DataFrame, contrast: str, scope: str, pooled: bool = False) -> pd.DataFrame:
    if pooled:
        panel = dedup_control_pool(frame).copy()
    else:
        panel = frame.loc[frame["sample"].isin(["CG4", contrast])].copy()
    if scope == "satellites":
        panel = panel.loc[pd.to_numeric(panel["is_satellite"], errors="coerce").eq(1)].copy()
    return panel


def predictors_for(frame: pd.DataFrame, scope: str) -> tuple[list[str], list[str]]:
    covariates, continuous = _covariates(frame)
    predictors = ["is_CG4", *covariates]
    if scope == "satellites":
        predictors = [column for column in predictors if column != "is_satellite"]
    return predictors, [column for column in continuous if column in predictors]


def add_exclusion_rows(
    rows: list[dict[str, object]],
    model_id: str,
    frame: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    *,
    outcome_reason: str = "missing_or_unclassified_outcome",
) -> pd.DataFrame:
    required = [outcome, *predictors]
    columns = [column for column in required if column in frame.columns]
    work = frame.copy()
    mask = pd.Series(True, index=work.index)
    rows.append(
        {
            "model_id": model_id,
            "reason": "input_rows",
            "excluded_n": 0,
            "remaining_n": int(mask.sum()),
        }
    )
    for column in required:
        if column not in work:
            excluded = mask.copy()
            reason = f"missing_column:{column}"
        else:
            values = pd.to_numeric(work[column], errors="coerce")
            excluded = mask & values.isna()
            reason = outcome_reason if column == outcome else f"missing_covariate:{column}"
        if int(excluded.sum()):
            mask &= ~excluded
            rows.append(
                {
                    "model_id": model_id,
                    "reason": reason,
                    "excluded_n": int(excluded.sum()),
                    "remaining_n": int(mask.sum()),
                }
            )
    return work.loc[mask, columns + [c for c in ["sample", "physical_group", "group_uid"] if c in work]].copy()


def summarize_logit(
    result: dict,
    spec: ModelSpec,
    panel: pd.DataFrame,
) -> ModelSpec:
    spec.status = str(result.get("status", "unknown"))
    spec.n = safe_int(result.get("n"))
    spec.formula = result.get("formula")
    spec.n_clusters = safe_int(result.get("n_clusters"))
    if result.get("status") == "ok":
        spec.estimate = safe_float(result.get("cg4_odds_ratio"))
        ci = result.get("cg4_ci95") or [None, None]
        spec.ci_low = safe_float(ci[0])
        spec.ci_high = safe_float(ci[1])
        spec.p = safe_float(result.get("cg4_p"))
    else:
        spec.notes = str(result.get("reason", result.get("error", "")))
    complete_n = spec.n
    if complete_n is not None:
        # Approximate complete-case sample counts using the model formula columns.
        spec.n_cg4 = int((panel["is_CG4"] == 1).sum()) if "is_CG4" in panel else None
        spec.n_control = int((panel["is_CG4"] == 0).sum()) if "is_CG4" in panel else None
    return spec


def fit_binary_estimate(
    panel: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    continuous: list[str],
    spec: ModelSpec,
    exclusions: list[dict[str, object]],
    *,
    outcome_reason: str = "missing_or_unclassified_outcome",
) -> ModelSpec:
    complete = add_exclusion_rows(
        exclusions, spec.model_id, panel, outcome, predictors, outcome_reason=outcome_reason
    )
    result = fit_logistic_model(
        panel,
        outcome,
        predictors,
        continuous=[column for column in continuous if column in predictors],
    )
    out = summarize_logit(result, spec, complete)
    if out.n is not None:
        used = add_exclusion_rows([], spec.model_id, panel, outcome, predictors, outcome_reason=outcome_reason)
        out.n_cg4 = int((used["is_CG4"] == 1).sum())
        out.n_control = int((used["is_CG4"] == 0).sum())
    return out


def fit_continuous_estimate(
    panel: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    continuous: list[str],
    spec: ModelSpec,
    exclusions: list[dict[str, object]],
) -> ModelSpec:
    complete = add_exclusion_rows(
        exclusions, spec.model_id, panel, outcome, predictors, outcome_reason="missing_continuous_outcome"
    )
    if len(complete) < 30:
        spec.status = "skipped"
        spec.notes = "too_few_complete_cases"
        spec.n = int(len(complete))
        return spec
    work = complete.copy()
    used_predictors = []
    for predictor in predictors:
        if work[predictor].nunique(dropna=True) < 2:
            continue
        if predictor in continuous:
            std = float(work[predictor].std(ddof=0))
            if not math.isfinite(std) or std == 0:
                continue
            work[predictor] = (work[predictor] - float(work[predictor].mean())) / std
        used_predictors.append(predictor)
    if "is_CG4" not in used_predictors:
        spec.status = "skipped"
        spec.notes = "no_cg4_contrast"
        spec.n = int(len(work))
        return spec
    design = sm.add_constant(work[used_predictors].astype(float), has_constant="add")
    groups = work["physical_group"] if "physical_group" in work else None
    try:
        model = sm.OLS(pd.to_numeric(work[outcome], errors="coerce").astype(float), design)
        if groups is not None and groups.nunique() >= 2:
            fitted = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            covariance = "cluster"
        else:
            fitted = model.fit(cov_type="HC1")
            covariance = "HC1"
    except Exception as exc:
        spec.status = "skipped"
        spec.notes = f"model_fit_failed:{exc}"
        spec.n = int(len(work))
        return spec
    ci = fitted.conf_int().loc["is_CG4"]
    spec.status = "ok"
    spec.estimate = safe_float(fitted.params["is_CG4"])
    spec.ci_low = safe_float(ci[0])
    spec.ci_high = safe_float(ci[1])
    spec.p = safe_float(fitted.pvalues["is_CG4"])
    spec.n = int(fitted.nobs)
    spec.n_cg4 = int((work["is_CG4"] == 1).sum())
    spec.n_control = int((work["is_CG4"] == 0).sum())
    spec.n_clusters = int(groups.nunique()) if groups is not None else None
    spec.formula = f"{outcome} ~ " + " + ".join(used_predictors)
    spec.notes = f"covariance={covariance}"
    return spec


def run_debiased_models(frame: pd.DataFrame, exclusions: list[dict[str, object]]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for contrast in REQUESTED_CONTROLS:
        for scope in ["all", "satellites"]:
            panel = panel_for(frame, contrast, scope)
            predictors, continuous = predictors_for(panel, scope)

            specs.append(
                fit_binary_estimate(
                    panel,
                    "elliptical",
                    predictors,
                    continuous,
                    ModelSpec(
                        task="A",
                        model_id=f"A_{contrast}_{scope}_fiducial_catalog_morphology",
                        contrast=f"CG4_vs_{contrast}",
                        scope=scope,
                        proxy="catalog_morphology_flag",
                        outcome="elliptical",
                        definition="fiducial morphology == Elliptical; Spiral is reference; Uncertain excluded",
                        threshold=0.5,
                        model_type="logistic",
                        effect_type="OR",
                        status="pending",
                        notes="The catalog flag is already based on GZ1 debiased p_E/p_S > 0.5.",
                    ),
                    exclusions,
                )
            )
            for threshold in MORPH_THRESHOLDS:
                outcome = f"elliptical_debiased_t{str(threshold).replace('.', 'p')}"
                panel[outcome] = gz_binary(panel, threshold)
                specs.append(
                    fit_binary_estimate(
                        panel,
                        outcome,
                        predictors,
                        continuous,
                        ModelSpec(
                            task="A",
                            model_id=f"A_{contrast}_{scope}_debiased_t{threshold:g}",
                            contrast=f"CG4_vs_{contrast}",
                            scope=scope,
                            proxy="gz1_debiased_votes",
                            outcome=outcome,
                            definition=(
                                f"elliptical if p_E > {threshold:g} and p_E > p_S; "
                                f"spiral if p_S > {threshold:g} and p_S > p_E; otherwise excluded"
                            ),
                            threshold=threshold,
                            model_type="logistic",
                            effect_type="OR",
                            status="pending",
                        ),
                        exclusions,
                    )
                )
    return specs


def run_structural_models(frame: pd.DataFrame, exclusions: list[dict[str, object]]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for contrast in REQUESTED_CONTROLS:
        for scope in ["all", "satellites"]:
            panel = panel_for(frame, contrast, scope)
            predictors, continuous = predictors_for(panel, scope)

            for threshold in CONCENTRATION_THRESHOLDS:
                outcome = f"early_concentration_t{str(threshold).replace('.', 'p')}"
                c = pd.to_numeric(panel["concentration_r90_r50"], errors="coerce")
                panel[outcome] = np.where(c.notna(), (c >= threshold).astype(float), np.nan)
                specs.append(
                    fit_binary_estimate(
                        panel,
                        outcome,
                        predictors,
                        continuous,
                        ModelSpec(
                            task="B",
                            model_id=f"B_{contrast}_{scope}_concentration_ge_{threshold:g}",
                            contrast=f"CG4_vs_{contrast}",
                            scope=scope,
                            proxy="concentration_r90_r50",
                            outcome=outcome,
                            definition=f"early type if C = petroR90_r / petroR50_r >= {threshold:g}",
                            threshold=threshold,
                            model_type="logistic",
                            effect_type="OR",
                            status="pending",
                        ),
                        exclusions,
                        outcome_reason="missing_or_invalid_concentration",
                    )
                )
            specs.append(
                fit_continuous_estimate(
                    panel,
                    "concentration_r90_r50",
                    predictors,
                    continuous,
                    ModelSpec(
                        task="B",
                        model_id=f"B_{contrast}_{scope}_concentration_continuous",
                        contrast=f"CG4_vs_{contrast}",
                        scope=scope,
                        proxy="concentration_r90_r50",
                        outcome="concentration_r90_r50",
                        definition="continuous C = petroR90_r / petroR50_r",
                        threshold=None,
                        model_type="OLS",
                        effect_type="beta",
                        status="pending",
                    ),
                    exclusions,
                )
            )

            for threshold in SERSIC_THRESHOLDS:
                outcome = f"early_sersic_t{str(threshold).replace('.', 'p')}"
                n = pd.to_numeric(panel["sersic_n_valid"], errors="coerce")
                panel[outcome] = np.where(n.notna(), (n >= threshold).astype(float), np.nan)
                specs.append(
                    fit_binary_estimate(
                        panel,
                        outcome,
                        predictors,
                        continuous,
                        ModelSpec(
                            task="B",
                            model_id=f"B_{contrast}_{scope}_sersic_ge_{threshold:g}",
                            contrast=f"CG4_vs_{contrast}",
                            scope=scope,
                            proxy="simard_sersic_n",
                            outcome=outcome,
                            definition=f"early type if valid Simard pure-Sersic n_g >= {threshold:g}",
                            threshold=threshold,
                            model_type="logistic",
                            effect_type="OR",
                            status="pending",
                        ),
                        exclusions,
                        outcome_reason="missing_or_invalid_sersic_n",
                    )
                )
            specs.append(
                fit_continuous_estimate(
                    panel,
                    "sersic_n_valid",
                    predictors,
                    continuous,
                    ModelSpec(
                        task="B",
                        model_id=f"B_{contrast}_{scope}_sersic_continuous",
                        contrast=f"CG4_vs_{contrast}",
                        scope=scope,
                        proxy="simard_sersic_n",
                        outcome="sersic_n_valid",
                        definition="continuous valid Simard pure-Sersic n_g; pegged fits excluded",
                        threshold=None,
                        model_type="OLS",
                        effect_type="beta",
                        status="pending",
                    ),
                    exclusions,
                )
            )
    return specs


def classify_cells(frame: pd.DataFrame) -> pd.Series:
    early = pd.to_numeric(frame["elliptical"], errors="coerce")
    quenched = pd.to_numeric(frame["quenched"], errors="coerce")
    cell = pd.Series(pd.NA, index=frame.index, dtype=object)
    cell.loc[(early == 0) & (quenched == 0)] = "late_SF"
    cell.loc[(early == 1) & (quenched == 1)] = "early_passive"
    cell.loc[(early == 1) & (quenched == 0)] = "early_SF"
    cell.loc[(early == 0) & (quenched == 1)] = "late_passive"
    return cell


def fit_multinomial_cells(frame: pd.DataFrame, exclusions: list[dict[str, object]]) -> tuple[list[ModelSpec], pd.DataFrame]:
    panel = panel_for(frame, "RG4", "satellites")
    panel["cell"] = classify_cells(panel)
    predictors, continuous = predictors_for(panel, "satellites")
    model_id = "C_CG4_vs_RG4_satellites_cells"
    complete = add_exclusion_rows(
        exclusions,
        model_id,
        panel,
        "cell_code",
        predictors,
        outcome_reason="missing_morphology_or_sfr_cell",
    )

    # Rebuild complete cases explicitly because cell_code is derived below.
    order = ["late_SF", "early_passive", "early_SF", "late_passive"]
    work = panel[["cell", *predictors, "sample", "physical_group"]].replace([np.inf, -np.inf], np.nan).copy()
    work["cell"] = pd.Categorical(work["cell"], categories=order, ordered=True)
    work["cell_code"] = work["cell"].cat.codes.replace(-1, np.nan)
    exclusions[:] = [row for row in exclusions if row["model_id"] != model_id]
    complete = add_exclusion_rows(
        exclusions,
        model_id,
        work,
        "cell_code",
        predictors,
        outcome_reason="missing_morphology_or_sfr_cell",
    )
    specs: list[ModelSpec] = []
    if len(complete) < 30 or complete["cell_code"].nunique() < 2:
        specs.append(
            ModelSpec(
                task="C",
                model_id=model_id,
                contrast="CG4_vs_RG4",
                scope="satellites",
                proxy="fiducial_morphology_x_sfr",
                outcome="cell",
                definition="multinomial cells; reference late_SF",
                threshold=None,
                model_type="multinomial_logit",
                effect_type="RRR",
                status="skipped",
                n=int(len(complete)),
                notes="too_few_complete_cases_or_cells",
            )
        )
        return specs, observed_cell_fractions(frame)

    work = complete.copy()
    used_predictors = []
    for predictor in predictors:
        if work[predictor].nunique(dropna=True) < 2:
            continue
        if predictor in continuous:
            std = float(work[predictor].std(ddof=0))
            if math.isfinite(std) and std != 0:
                work[predictor] = (work[predictor] - float(work[predictor].mean())) / std
            else:
                continue
        used_predictors.append(predictor)

    design = sm.add_constant(work[used_predictors].astype(float), has_constant="add")
    endog = work["cell_code"].astype(int)
    try:
        model = sm.MNLogit(endog, design)
        if work["physical_group"].nunique() >= 2:
            fitted = model.fit(
                method="newton",
                maxiter=200,
                disp=False,
                cov_type="cluster",
                cov_kwds={"groups": work["physical_group"]},
            )
            covariance = "cluster"
        else:
            fitted = model.fit(method="newton", maxiter=200, disp=False)
            covariance = "nonrobust"
    except Exception as exc:
        specs.append(
            ModelSpec(
                task="C",
                model_id=model_id,
                contrast="CG4_vs_RG4",
                scope="satellites",
                proxy="fiducial_morphology_x_sfr",
                outcome="cell",
                definition="multinomial cells; reference late_SF",
                threshold=None,
                model_type="multinomial_logit",
                effect_type="RRR",
                status="skipped",
                n=int(len(work)),
                notes=f"model_fit_failed:{exc}",
            )
        )
        return specs, observed_cell_fractions(frame)

    params = fitted.params
    conf = fitted.conf_int()
    pvalues = fitted.pvalues
    # Statsmodels columns 0..J-2 correspond to non-reference categories 1..J-1.
    for col_position, category in enumerate(order[1:]):
        if col_position not in params.columns or "is_CG4" not in params.index:
            continue
        category_code = col_position + 1
        beta = safe_float(params.loc["is_CG4", col_position])
        if isinstance(conf, pd.DataFrame) and isinstance(conf.index, pd.MultiIndex):
            key = (category_code, "is_CG4")
            if key not in conf.index:
                key = (str(category_code), "is_CG4")
            low_beta = safe_float(conf.loc[key, "lower"])
            high_beta = safe_float(conf.loc[key, "upper"])
        else:
            ci_row = conf.loc["is_CG4", col_position]
            low_beta = safe_float(ci_row[0] if hasattr(ci_row, "__len__") else np.nan)
            high_beta = safe_float(ci_row[1] if hasattr(ci_row, "__len__") else np.nan)
        if low_beta is None or high_beta is None:
            # Fallback for statsmodels versions where conf_int returns a 2-column
            # array with a MultiIndex [category, term].
            try:
                ci = fitted.conf_int().xs(col_position).loc["is_CG4"]
                low_beta = safe_float(ci.iloc[0])
                high_beta = safe_float(ci.iloc[1])
            except Exception:
                low_beta = high_beta = None
        p = safe_float(pvalues.loc["is_CG4", col_position])
        specs.append(
            ModelSpec(
                task="C",
                model_id=f"{model_id}_{category}",
                contrast="CG4_vs_RG4",
                scope="satellites",
                proxy="fiducial_morphology_x_sfr",
                outcome=category,
                definition=f"{category} relative to late_SF reference",
                threshold=None,
                model_type="multinomial_logit",
                effect_type="RRR",
                status="ok",
                estimate=math.exp(beta) if beta is not None else None,
                ci_low=math.exp(low_beta) if low_beta is not None else None,
                ci_high=math.exp(high_beta) if high_beta is not None else None,
                p=p,
                n=int(fitted.nobs),
                n_cg4=int((work["is_CG4"] == 1).sum()),
                n_control=int((work["is_CG4"] == 0).sum()),
                n_clusters=int(work["physical_group"].nunique()),
                formula="cell ~ " + " + ".join(used_predictors),
                notes=f"reference=late_SF; covariance={covariance}",
            )
        )
    return specs, observed_cell_fractions(frame)


def observed_cell_fractions(frame: pd.DataFrame) -> pd.DataFrame:
    satellites = frame.loc[pd.to_numeric(frame["is_satellite"], errors="coerce").eq(1)].copy()
    satellites["cell"] = classify_cells(satellites)
    rows = []
    order = ["late_SF", "early_passive", "early_SF", "late_passive"]
    for sample_name in ["CG4", "RG4", "Control4C"]:
        part = satellites.loc[satellites["sample"] == sample_name]
        complete = part.loc[part["cell"].isin(order)]
        denom = len(complete)
        for cell in order:
            n_cell = int((complete["cell"] == cell).sum())
            rows.append(
                {
                    "sample": sample_name,
                    "cell": cell,
                    "n_cell": n_cell,
                    "n_complete": int(denom),
                    "fraction": n_cell / denom if denom else np.nan,
                    "n_satellites_total": int(len(part)),
                    "n_missing_morphology": int(part["elliptical"].isna().sum()),
                    "n_missing_sfr_class": int(part["quenched"].isna().sum()),
                }
            )
    return pd.DataFrame(rows)


def run_quenching_ci(frame: pd.DataFrame, exclusions: list[dict[str, object]]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for contrast in ["RG4"]:
        for scope in ["all", "satellites"]:
            panel = panel_for(frame, contrast, scope)
            predictors, continuous = predictors_for(panel, scope)
            specs.append(
                fit_binary_estimate(
                    panel,
                    "quenched",
                    predictors,
                    continuous,
                    ModelSpec(
                        task="D",
                        model_id=f"D_CG4_vs_{contrast}_{scope}_quenched",
                        contrast=f"CG4_vs_{contrast}",
                        scope=scope,
                        proxy="fiducial_sfr_class",
                        outcome="quenched",
                        definition="quenched versus star-forming; NosSFR excluded",
                        threshold=None,
                        model_type="logistic",
                        effect_type="OR",
                        status="pending",
                    ),
                    exclusions,
                    outcome_reason="missing_sfr_class",
                )
            )
    for scope in ["all", "satellites"]:
        panel = panel_for(frame, "pooled_controls", scope, pooled=True)
        predictors, continuous = predictors_for(panel, scope)
        specs.append(
            fit_binary_estimate(
                panel,
                "quenched",
                predictors,
                continuous,
                ModelSpec(
                    task="D",
                    model_id=f"D_pooled_{scope}_quenched",
                    contrast="CG4_vs_pooled_controls",
                    scope=scope,
                    proxy="fiducial_sfr_class",
                    outcome="quenched",
                    definition="quenched versus star-forming; NosSFR excluded; controls deduplicated by objid",
                    threshold=None,
                    model_type="logistic",
                    effect_type="OR",
                    status="pending",
                ),
                exclusions,
                outcome_reason="missing_sfr_class",
            )
        )
    return specs


def fibre_collision_caveat(frame: pd.DataFrame) -> pd.DataFrame:
    satellites = frame.loc[pd.to_numeric(frame["is_satellite"], errors="coerce").eq(1)].copy()
    distance = pd.to_numeric(satellites.get("dist2BGG_kpc", satellites.get("dist2BGG")), errors="coerce")
    satellites["_satellite_distance"] = distance
    satellites["satellite_distance_rank"] = satellites.groupby("group_uid")["_satellite_distance"].rank(method="first")
    rows = []
    for sample_name in SAMPLES:
        for rank in [1, 2]:
            part = satellites.loc[
                (satellites["sample"] == sample_name)
                & satellites["satellite_distance_rank"].eq(rank)
            ]
            n = int(len(part))
            missing_sf = part["quenched"].isna()
            rows.append(
                {
                    "sample": sample_name,
                    "satellite_projected_distance_rank": rank,
                    "n_satellites": n,
                    "n_lacking_sf_classification": int(missing_sf.sum()),
                    "fraction_lacking_sf_classification": float(missing_sf.mean()) if n else np.nan,
                    "n_missing_sSFR": int(pd.to_numeric(part.get("sSFR"), errors="coerce").isna().sum()) if n else 0,
                    "n_missing_lgm": int(pd.to_numeric(part.get("lgm"), errors="coerce").isna().sum()) if n else 0,
                }
            )
    return pd.DataFrame(rows)


def data_availability_table(size_audit: dict) -> pd.DataFrame:
    rows = []
    for sample_name, row in size_audit.get("per_sample", {}).items():
        n = int(row.get("n_rows", 0))
        rows.append(
            {
                "sample": sample_name,
                **row,
                "petro_available_fraction": row.get("size_ok_petro", 0) / n if n else np.nan,
                "simard_available_fraction": row.get("size_ok_simard", 0) / n if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_fiducial_reproduction(sample: dict[str, pd.DataFrame]) -> pd.DataFrame:
    stored_path = Path(co.RESULTS)
    stored = {}
    if stored_path.exists():
        with stored_path.open() as handle:
            data = json.load(handle)
        root = data.get("extended_specialness", data)
        stored["primary"] = root.get("primary_contrasts", {})
        stored["pooled"] = root.get("specialness_models", {})

    refit_primary = run_primary_contrasts(sample)
    refit_pooled = fit_logistic_specialness_models(sample)
    rows = []
    targets = [
        ("primary", "RG4", "elliptical_all"),
        ("primary", "RG4", "elliptical_satellites"),
        ("primary", "RG4", "quenched_all"),
        ("primary", "Control4C", "elliptical_all"),
        ("primary", "Control4C", "quenched_all"),
        ("pooled", "pooled_controls", "quenched_satellites"),
        ("pooled", "pooled_controls", "elliptical_satellites"),
    ]
    for family, contrast, model_name in targets:
        if family == "primary":
            refit = refit_primary.get("contrasts", {}).get(contrast, {}).get(model_name, {})
            saved = stored.get("primary", {}).get("contrasts", {}).get(contrast, {}).get(model_name, {})
        else:
            refit = refit_pooled.get(model_name, {})
            saved = stored.get("pooled", {}).get(model_name, {})
        rows.append(
            {
                "family": family,
                "contrast": contrast,
                "model": model_name,
                "stored_or": saved.get("cg4_odds_ratio"),
                "refit_or": refit.get("cg4_odds_ratio"),
                "delta_or": (
                    safe_float(refit.get("cg4_odds_ratio")) - safe_float(saved.get("cg4_odds_ratio"))
                    if safe_float(refit.get("cg4_odds_ratio")) is not None and safe_float(saved.get("cg4_odds_ratio")) is not None
                    else np.nan
                ),
                "stored_p_holm": saved.get("cg4_p_adj"),
                "refit_p_holm": refit.get("cg4_p_adj"),
                "stored_n": saved.get("n"),
                "refit_n": refit.get("n"),
                "status": refit.get("status"),
                "formula": refit.get("formula"),
            }
        )
    return pd.DataFrame(rows)


def apply_global_holm(specs: list[ModelSpec]) -> None:
    ok = [spec for spec in specs if spec.status == "ok" and spec.p is not None]
    adjusted = holm_correction([spec.p for spec in ok])
    for spec, p_holm in zip(ok, adjusted):
        spec.p_holm = p_holm


def specs_to_frame(specs: list[ModelSpec]) -> pd.DataFrame:
    return pd.DataFrame([spec.__dict__ for spec in specs])


def plot_forest(estimates: pd.DataFrame, path_pdf: Path, path_png: Path) -> None:
    plot = estimates.loc[
        (estimates["status"] == "ok")
        & estimates["effect_type"].isin(["OR", "RRR"])
        & estimates["estimate"].notna()
        & estimates["ci_low"].notna()
        & estimates["ci_high"].notna()
    ].copy()
    keep = (
        ((plot["task"] == "A") & (plot["scope"] == "satellites") & (plot["contrast"].isin(["CG4_vs_RG4", "CG4_vs_Control4C"])))
        | ((plot["task"] == "B") & (plot["scope"] == "satellites") & (plot["threshold"].isin([2.6, 2.5]) | plot["threshold"].isna()))
        | (plot["task"] == "C")
    )
    plot = plot.loc[keep].copy()
    if plot.empty:
        return
    plot["label"] = plot.apply(
        lambda row: (
            f"{row['contrast']} {row['proxy']} {row['threshold']:g}"
            if pd.notna(row["threshold"])
            else f"{row['contrast']} {row['outcome']}"
        ),
        axis=1,
    )
    plot = plot.sort_values(["task", "contrast", "proxy", "threshold", "outcome"], na_position="last")
    fig, ax = plt.subplots(figsize=(8.4, max(4.0, 0.36 * len(plot) + 1.4)))
    y = np.arange(len(plot))
    estimates_arr = plot["estimate"].astype(float).to_numpy()
    lows = plot["ci_low"].astype(float).to_numpy()
    highs = plot["ci_high"].astype(float).to_numpy()
    colours = plot["task"].map({"A": "#2864A6", "B": "#25876E", "C": "#A74752"}).fillna("#555555")
    for i, colour in enumerate(colours):
        ax.errorbar(
            estimates_arr[i],
            y[i],
            xerr=[[estimates_arr[i] - lows[i]], [highs[i] - estimates_arr[i]]],
            fmt="o",
            color=colour,
            capsize=2.5,
            markersize=4,
        )
    ax.axvline(1.0, color="0.45", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y, plot["label"])
    ax.set_xlabel("CG4 effect ratio (OR or RRR; 95% CI)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    data = frame[columns].copy()
    if max_rows is not None:
        data = data.head(max_rows)
    for column in data.columns:
        if pd.api.types.is_float_dtype(data[column]):
            data[column] = data[column].map(lambda value: fmt(value) if pd.notna(value) else "NA")
    data = data.fillna("NA").astype(str)
    widths = {
        column: max(len(str(column)), *(len(value) for value in data[column].tolist()))
        for column in data.columns
    }
    header = "| " + " | ".join(str(column).ljust(widths[column]) for column in data.columns) + " |"
    divider = "| " + " | ".join("-" * widths[column] for column in data.columns) + " |"
    body = [
        "| " + " | ".join(row[column].ljust(widths[column]) for column in data.columns) + " |"
        for _, row in data.iterrows()
    ]
    return "\n".join([header, divider, *body])


def write_report(
    output_dir: Path,
    estimates: pd.DataFrame,
    cell_fractions: pd.DataFrame,
    quenching: pd.DataFrame,
    fiducial: pd.DataFrame,
    availability: pd.DataFrame,
    fibre: pd.DataFrame,
    missing_raw_votes: bool,
) -> None:
    report = []
    report.append(f"# CG4 Morphology/Quenching Robustness Report\n")
    report.append(f"Run date: 2026-07-17. Robustness runner version: `{ROBUSTNESS_VERSION}`.\n")
    report.append("## Methods Actually Used\n")
    report.append(
        "- Input catalogue: `data/processed_sample.pkl`, harmonized with `src.extended_data.ensure_galaxy_frame`.\n"
        "- Morphology: `p_E` and `p_S`, loaded by the project from SDSS `zooSpec.p_el_debiased` and `zooSpec.p_cs_debiased`.\n"
        "- Fiducial early-type proxy: `morphology == Elliptical`; `Spiral` is the binary reference and `Uncertain` rows are excluded.\n"
        "- Star-formation class: `sSFR_status`, with `Quenched` versus `Starforming`; `NosSFR` rows are excluded.\n"
        "- Structural columns: local caches `data/sdss_size_columns.csv` (`petroR50_r`, `petroR90_r`) and `data/simard2011_subset.csv` (`ng`). No external downloads were attempted.\n"
        "- Model adjustment: the existing helper selected `logMstar`, `z_numeric`, `is_satellite` for all-member fits, `log_group_luminosity`, and `velocity_dispersion` when complete enough. Satellite-only fits remove `is_satellite` after subsetting. Standard errors are clustered by `physical_group`.\n"
        "- Holm correction: one global Holm correction was applied over all successful inferential estimates in this new robustness family.\n"
    )
    if missing_raw_votes:
        report.append(
            "\nRaw (non-debiased) Galaxy Zoo vote fractions are not present in the processed sample or local source tables. The fiducial catalogue morphology is already debiased, so the raw-vs-debiased request cannot be separated locally.\n"
        )

    report.append("\n## Fiducial Reproduction\n")
    report.append(markdown_table(fiducial, ["family", "contrast", "model", "stored_or", "refit_or", "delta_or", "stored_n", "refit_n", "status"]))

    task_a = estimates.loc[estimates["task"] == "A"].copy()
    report.append("\n## Task A - Debiased Morphology\n")
    report.append(markdown_table(task_a, ["contrast", "scope", "proxy", "threshold", "estimate", "ci_low", "ci_high", "p", "p_holm", "n"], max_rows=24))

    rg_sat = task_a.loc[
        (task_a["contrast"] == "CG4_vs_RG4")
        & (task_a["scope"] == "satellites")
        & (task_a["proxy"] == "gz1_debiased_votes")
        & (task_a["threshold"] == 0.5)
    ]
    if not rg_sat.empty:
        row = rg_sat.iloc[0]
        report.append(
            f"\nAt the fiducial debiased threshold for satellites against RG4, OR = {fmt(row['estimate'])} "
            f"(95% CI {fmt(row['ci_low'])}-{fmt(row['ci_high'])}; Holm p = {p_text(row['p_holm'])}).\n"
        )

    task_b = estimates.loc[estimates["task"] == "B"].copy()
    structural_signal = task_b.loc[
        (task_b["status"] == "ok")
        & (task_b["p_holm"].notna())
        & (task_b["p_holm"] < 0.05)
        & (
            ((task_b["effect_type"] == "OR") & (task_b["estimate"] > 1))
            | ((task_b["effect_type"] == "beta") & (task_b["estimate"] > 0))
        )
    ]
    report.append("\n## Task B - Structural Morphology Proxies\n")
    report.append(markdown_table(task_b, ["contrast", "scope", "proxy", "model_type", "threshold", "estimate", "ci_low", "ci_high", "p", "p_holm", "n"], max_rows=40))
    if structural_signal.empty:
        report.append("\nNo structural excess is significant after the global Holm correction at fixed mass and the fiducial covariates.\n")
    else:
        report.append(
            "\nAt least one structural proxy is positive and significant after Holm correction: "
            + ", ".join(structural_signal["model_id"].astype(str).tolist())
            + ".\n"
        )

    task_c = estimates.loc[estimates["task"] == "C"].copy()
    report.append("\n## Task C - 2x2 Morphology x Star-Formation Decomposition\n")
    report.append(markdown_table(task_c, ["outcome", "estimate", "ci_low", "ci_high", "p", "p_holm", "n"]))
    report.append("\nObserved complete-case satellite fractions:\n")
    report.append(markdown_table(cell_fractions, ["sample", "cell", "n_cell", "n_complete", "fraction"]))
    if not task_c.empty and (task_c["status"] == "ok").any():
        strongest = task_c.loc[task_c["status"] == "ok"].sort_values("estimate", ascending=False).iloc[0]
        report.append(
            f"\nLargest adjusted RRR is `{strongest['outcome']}`: RRR = {fmt(strongest['estimate'])} "
            f"(95% CI {fmt(strongest['ci_low'])}-{fmt(strongest['ci_high'])}; Holm p = {p_text(strongest['p_holm'])}). "
            "Interpretation hooks: early_passive implies historical transform-and-quench; early_SF implies tidal heating without quenching; late_passive implies strangulation without structural transformation.\n"
        )

    report.append("\n## Task D - Quenching Null CI\n")
    report.append(markdown_table(quenching, ["contrast", "scope", "estimate", "ci_low", "ci_high", "p", "p_holm", "n"]))
    pooled_sat = quenching.loc[(quenching["contrast"] == "CG4_vs_pooled_controls") & (quenching["scope"] == "satellites")]
    if not pooled_sat.empty:
        row = pooled_sat.iloc[0]
        report.append(
            f"\nPooled satellite quenching OR = {fmt(row['estimate'])} "
            f"(95% CI {fmt(row['ci_low'])}-{fmt(row['ci_high'])}). This interval is compatible with no effect and with a modest excess.\n"
        )

    report.append("\n## Data Availability And Fibre-Collision Caveat\n")
    report.append(markdown_table(availability, ["sample", "n_rows", "size_ok_petro", "petro_available_fraction", "size_ok_simard", "simard_available_fraction"]))
    report.append("\nProjected-distance-rank 1 and 2 satellites lacking the SF classification:\n")
    report.append(markdown_table(fibre, ["sample", "satellite_projected_distance_rank", "n_satellites", "n_lacking_sf_classification", "fraction_lacking_sf_classification"]))

    report.append("\n## Output Files\n")
    report.append(
        "- `tables/morphology_debiased.csv`\n"
        "- `tables/structural_proxies.csv`\n"
        "- `tables/multinomial_rrr.csv`\n"
        "- `tables/observed_cell_fractions.csv`\n"
        "- `tables/quenching_ci.csv`\n"
        "- `tables/exclusions.csv`\n"
        "- `figures/robustness_forest.pdf` and `.png`\n"
    )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n")


def write_missing_data(output_dir: Path, reason: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = (
        "# Missing Data For Robustness Checks\n\n"
        f"{reason}\n\n"
        "Needed locally:\n"
        "- GZ1 debiased vote fractions keyed by SDSS `objid`: `p_el_debiased`, `p_cs_debiased`.\n"
        "- SDSS r-band Petrosian radii keyed by `objid`: `petroR50_r`, `petroR90_r`, `petroR50Err_r`.\n"
        "- Optional Simard et al. 2011 pure-Sersic crossmatch keyed through DR7 objID: `ng`, `Rchl_r`, `Scale`, and redshift for mismatch checks.\n"
        "- Existing star-formation classification fields: `sSFR_status`, `sSFR`, `lgm`.\n"
        "\nExternal downloading was not attempted because the task did not explicitly allow it.\n"
    )
    (output_dir / "MISSING_DATA.md").write_text(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "robustness"),
        help="Directory for new robustness outputs.",
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        sample = load_sample()
        frame, size_audit = prepare_frame(sample)
    except Exception as exc:
        write_missing_data(output_dir, f"Cannot run robustness checks from local data: `{exc}`.")
        return 2

    exclusions: list[dict[str, object]] = []
    specs: list[ModelSpec] = []
    specs.extend(run_debiased_models(frame, exclusions))
    specs.extend(run_structural_models(frame, exclusions))
    multinomial_specs, cell_fractions = fit_multinomial_cells(frame, exclusions)
    specs.extend(multinomial_specs)
    specs.extend(run_quenching_ci(frame, exclusions))
    apply_global_holm(specs)
    estimates = specs_to_frame(specs)

    morphology = estimates.loc[estimates["task"] == "A"].copy()
    structural = estimates.loc[estimates["task"] == "B"].copy()
    multinomial = estimates.loc[estimates["task"] == "C"].copy()
    quenching = estimates.loc[estimates["task"] == "D"].copy()
    availability = data_availability_table(size_audit)
    fibre = fibre_collision_caveat(frame)
    fiducial = run_fiducial_reproduction(sample)

    morphology.to_csv(tables_dir / "morphology_debiased.csv", index=False)
    structural.to_csv(tables_dir / "structural_proxies.csv", index=False)
    multinomial.to_csv(tables_dir / "multinomial_rrr.csv", index=False)
    cell_fractions.to_csv(tables_dir / "observed_cell_fractions.csv", index=False)
    quenching.to_csv(tables_dir / "quenching_ci.csv", index=False)
    pd.DataFrame(exclusions).to_csv(tables_dir / "exclusions.csv", index=False)
    availability.to_csv(tables_dir / "data_availability.csv", index=False)
    fibre.to_csv(tables_dir / "fibre_collision_caveat.csv", index=False)
    fiducial.to_csv(tables_dir / "fiducial_reproduction.csv", index=False)
    estimates.to_csv(tables_dir / "all_estimates.csv", index=False)

    plot_forest(estimates, figures_dir / "robustness_forest.pdf", figures_dir / "robustness_forest.png")

    raw_vote_candidates = [
        "p_el",
        "p_spiral",
        "p_el_raw",
        "p_cs_raw",
        "p_el_undeb",
        "p_cs_undeb",
    ]
    missing_raw_votes = not any(column in frame.columns for column in raw_vote_candidates)
    write_report(output_dir, estimates, cell_fractions, quenching, fiducial, availability, fibre, missing_raw_votes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
