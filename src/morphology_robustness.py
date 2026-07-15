"""Targeted morphology robustness checks for crowding and CG subclasses."""

from __future__ import annotations

import os

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import config as co
    from extended_data import ensure_galaxy_frame
    from extended_stats import fit_logistic_model, safe_json
    from utils import labels_utils as lu
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import fit_logistic_model, safe_json
    from .utils import labels_utils as lu


SAMPLES = ["CG4", "Control4B", "Control4C", "RG4"]
CONTROL_SAMPLES = ["Control4B", "Control4C", "RG4"]
MORPH_OUTCOMES = ["elliptical", "spiral"]
CROWDING_THRESHOLD_ARCSEC = 55.0


def _nearest_angular(frame: pd.DataFrame) -> np.ndarray:
    """Nearest projected neighbour separation within each sample/group, in arcsec."""

    values = np.full(len(frame), np.nan)
    positions = {index: position for position, index in enumerate(frame.index)}
    for _, group in frame.groupby("group_uid", observed=True):
        if len(group) < 2 or not {"RA", "Dec"}.issubset(group.columns):
            continue
        coords = group[["RA", "Dec"]].apply(lambda column: np.asarray(column, dtype=float))
        if not np.isfinite(coords.to_numpy()).all():
            continue
        ra = np.deg2rad(coords["RA"].to_numpy())
        dec = np.deg2rad(coords["Dec"].to_numpy())
        delta_ra = ra[:, None] - ra[None, :]
        delta_dec = dec[:, None] - dec[None, :]
        hav = (
            np.sin(delta_dec / 2) ** 2
            + np.cos(dec[:, None]) * np.cos(dec[None, :]) * np.sin(delta_ra / 2) ** 2
        )
        angular = 2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
        np.fill_diagonal(angular, np.inf)
        nearest_arcsec = np.min(angular, axis=1) * 180.0 / np.pi * 3600.0
        for index, value in zip(group.index, nearest_arcsec):
            values[positions[index]] = value
    return values


def _fraction_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for sample_name in SAMPLES:
        part = frame.loc[frame["sample"] == sample_name]
        if part.empty:
            continue
        close = part["close_neighbour_55arcsec"].fillna(False).astype(bool)
        no_close = part.loc[~close]
        rows.append(
            {
                "sample": sample_name,
                "n_total": int(len(part)),
                "n_close_lt55": int(close.sum()),
                "close_fraction": float(close.mean()),
                "n_after_excluding_close": int(len(no_close)),
                "elliptical_n_after_excluding_close": int(no_close["elliptical"].eq(1).sum()),
                "spiral_n_after_excluding_close": int(no_close["spiral"].eq(1).sum()),
                "elliptical_fraction_after_excluding_close": (
                    float(no_close["elliptical"].eq(1).sum() / len(no_close))
                    if len(no_close)
                    else None
                ),
                "spiral_fraction_after_excluding_close": (
                    float(no_close["spiral"].eq(1).sum() / len(no_close))
                    if len(no_close)
                    else None
                ),
            }
        )
    return rows


def _exact_test_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    close = frame["close_neighbour_55arcsec"].fillna(False).astype(bool)
    no_close = frame.loc[~close]
    cg = no_close.loc[no_close["sample"] == "CG4"]
    for control in CONTROL_SAMPLES:
        ctrl = no_close.loc[no_close["sample"] == control]
        for outcome in MORPH_OUTCOMES:
            cg_success = int(cg[outcome].eq(1).sum())
            ctrl_success = int(ctrl[outcome].eq(1).sum())
            table = np.array(
                [
                    [cg_success, int(len(cg) - cg_success)],
                    [ctrl_success, int(len(ctrl) - ctrl_success)],
                ]
            )
            if table.min() < 0 or table.sum() == 0:
                p_value = None
                method = "skipped"
            else:
                try:
                    p_value = float(stats.barnard_exact(table, alternative="two-sided").pvalue)
                    method = "Barnard exact"
                except Exception:
                    p_value = float(stats.fisher_exact(table, alternative="two-sided").pvalue)
                    method = "Fisher exact"
            rows.append(
                {
                    "control": control,
                    "outcome": outcome,
                    "method": method,
                    "cg4_n": int(len(cg)),
                    "control_n": int(len(ctrl)),
                    "cg4_fraction": float(cg_success / len(cg)) if len(cg) else None,
                    "control_fraction": float(ctrl_success / len(ctrl)) if len(ctrl) else None,
                    "p_value": p_value,
                }
            )
    return rows


def _adjusted_models(frame: pd.DataFrame) -> dict[str, object]:
    covariates = []
    continuous = []
    for column, is_continuous in [
        ("logMstar", True),
        ("z_numeric", True),
        ("is_satellite", False),
        ("log_group_luminosity", True),
        ("velocity_dispersion", True),
    ]:
        if column in frame and frame[column].notna().mean() >= 0.65:
            covariates.append(column)
            if is_continuous:
                continuous.append(column)
    predictors = ["is_CG4", "close_neighbour_55arcsec", *covariates]
    models = {"covariates": covariates}
    for outcome in MORPH_OUTCOMES:
        models[outcome] = fit_logistic_model(
            frame,
            outcome,
            predictors,
            continuous=[column for column in continuous if column in predictors],
        )
    return models


def _plot_fraction_rows(rows: list[dict[str, object]], path: str) -> str | None:
    if not rows:
        return None
    labels = [row["sample"] for row in rows]
    elliptical = [row["elliptical_fraction_after_excluding_close"] or 0 for row in rows]
    spiral = [row["spiral_fraction_after_excluding_close"] or 0 for row in rows]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - width / 2, elliptical, width, label="Elliptical/smooth", color="#2864A6")
    ax.bar(x + width / 2, spiral, width, label="Spiral/features", color="#A74752")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Fraction after excluding <55 arcsec neighbours")
    ax.set_ylim(0, max(0.05, max(elliptical + spiral) * 1.25))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def _concentration_result(frame: pd.DataFrame) -> dict[str, object]:
    lower_columns = {column.lower(): column for column in frame.columns}
    r50 = lower_columns.get("petror50_r") or lower_columns.get("petror50r")
    r90 = lower_columns.get("petror90_r") or lower_columns.get("petror90r")
    if not r50 or not r90:
        return {
            "status": "skipped",
            "reason": "missing_petroR50_r_petroR90_r",
            "future_work": (
                "Extend the current matched data products to carry SDSS r-band "
                "Petrosian radii, then apply the corresponding quality cuts before "
                "using concentration as an independent proxy."
            ),
        }
    valid = frame[[r50, r90]].apply(pd.to_numeric, errors="coerce")
    good = (valid[r50] > 0) & (valid[r90] > 0)
    if int(good.sum()) < 30:
        return {"status": "skipped", "reason": "too_few_valid_petrosian_radii"}
    return {
        "status": "implemented_in_size_analysis",
        "see": "extended_specialness.size_analysis.concentration",
        "r50_column": r50,
        "r90_column": r90,
    }


def _cg_class_split(frame: pd.DataFrame) -> dict[str, object]:
    if "Class" not in frame.columns:
        return {"status": "skipped", "reason": "missing_CG_class_column"}
    cg = frame.loc[frame["sample"] == "CG4"].copy()
    if cg.empty:
        return {"status": "skipped", "reason": "no_CG4_rows"}
    cg["class_label"] = cg["Class"].map(lu.display_label).fillna(cg["Class"])
    rows = []
    for class_name, part in cg.groupby("class_label", observed=True):
        rows.append(
            {
                "class": str(class_name),
                "n_groups": int(part["group_uid"].nunique()),
                "n_galaxies": int(len(part)),
                "elliptical_fraction": float(part["elliptical"].eq(1).mean()),
                "spiral_fraction": float(part["spiral"].eq(1).mean()),
                "quenched_fraction": float(part["quenched"].eq(1).mean()) if "quenched" in part else None,
                "starforming_fraction": (
                    float(part["starforming"].eq(1).mean()) if "starforming" in part else None
                ),
            }
        )

    comparisons = []
    definitions = [
        ("Embedded+Predominant", ["Embedded", "Predominant"], "Isolated", ["Isolated"]),
        ("Embedded", ["Embedded"], "Predominant", ["Predominant"]),
    ]
    for label_a, classes_a, label_b, classes_b in definitions:
        a = cg.loc[cg["class_label"].isin(classes_a)]
        b = cg.loc[cg["class_label"].isin(classes_b)]
        if len(a) < 5 or len(b) < 5:
            comparisons.append(
                {
                    "comparison": f"{label_a} vs {label_b}",
                    "status": "skipped",
                    "reason": "too_few_galaxies",
                    "n_a": int(len(a)),
                    "n_b": int(len(b)),
                }
            )
            continue
        for outcome in ["elliptical", "spiral", "quenched", "starforming"]:
            if outcome not in cg.columns:
                continue
            a_success = int(a[outcome].eq(1).sum())
            b_success = int(b[outcome].eq(1).sum())
            table = [[a_success, int(len(a) - a_success)], [b_success, int(len(b) - b_success)]]
            comparisons.append(
                {
                    "comparison": f"{label_a} vs {label_b}",
                    "outcome": outcome,
                    "status": "ok",
                    "n_a": int(len(a)),
                    "n_b": int(len(b)),
                    "fraction_a": float(a_success / len(a)),
                    "fraction_b": float(b_success / len(b)),
                    "fisher_p": float(stats.fisher_exact(table, alternative="two-sided").pvalue),
                }
            )
    return {"status": "ok", "summary": rows, "comparisons": comparisons}


def run_morphology_robustness(data, output_dir: str | None = None) -> dict[str, object]:
    """Run the requested Galaxy Zoo crowding and lightweight structural checks."""

    frame = ensure_galaxy_frame(data)
    if frame.empty or "sample" not in frame:
        return {"status": "skipped", "reason": "no_galaxy_samples"}
    if not {"RA", "Dec", "group_uid"}.issubset(frame.columns):
        return {"status": "skipped", "reason": "missing_RA_Dec_or_group_uid"}
    frame = frame.copy()
    frame["nearest_angular_separation_arcsec"] = _nearest_angular(frame)
    frame["close_neighbour_55arcsec"] = (
        frame["nearest_angular_separation_arcsec"] < CROWDING_THRESHOLD_ARCSEC
    ).astype(float)

    fraction_rows = _fraction_rows(frame)
    exact_rows = _exact_test_rows(frame)
    adjusted = _adjusted_models(frame)

    os.makedirs(co.OUTPUT_PATH, exist_ok=True)
    pd.DataFrame(fraction_rows).to_csv(
        os.path.join(co.OUTPUT_PATH, "morphology_crowding_fractions.csv"), index=False
    )
    pd.DataFrame(exact_rows).to_csv(
        os.path.join(co.OUTPUT_PATH, "morphology_crowding_exact_tests.csv"), index=False
    )

    figure = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        figure = _plot_fraction_rows(
            fraction_rows,
            os.path.join(output_dir, "fig_morphology_crowding_robustness.pdf"),
        )

    class_split = _cg_class_split(frame)
    if class_split.get("status") == "ok":
        pd.DataFrame(class_split["summary"]).to_csv(
            os.path.join(co.OUTPUT_PATH, "cg_class_morphology_split.csv"), index=False
        )

    result = {
        "status": "ok",
        "threshold_arcsec": CROWDING_THRESHOLD_ARCSEC,
        "fractions_after_excluding_close": fraction_rows,
        "exact_tests_after_excluding_close": exact_rows,
        "adjusted_models_with_close_flag": adjusted,
        "figure": figure,
        "concentration_index": _concentration_result(frame),
        "cg_class_split": class_split,
    }
    return safe_json(result)
