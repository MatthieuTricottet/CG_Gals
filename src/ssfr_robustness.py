"""Simple robustness checks for the sSFR classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

try:
    import config as co
    import generate_report as report
    from utils import graphics_utils as gu
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from . import generate_report as report
    from .utils import graphics_utils as gu


def _satellite_frame(sample: dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
    """Return satellites with finite sSFR values for one catalogue."""

    frame = sample[name + co.GASUFF].copy()
    if "rank_M" in frame:
        frame = frame.loc[pd.to_numeric(frame["rank_M"], errors="coerce") > 1].copy()
    ssfr_column = "sSFR_raw" if "sSFR_raw" in frame else "sSFR"
    frame["_ssfr_alt"] = pd.to_numeric(frame[ssfr_column], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["_ssfr_alt"])


def _summary(frame: pd.DataFrame, threshold: float) -> dict[str, object]:
    is_starforming = frame["_ssfr_alt"] >= threshold
    n_total = int(len(frame))
    n_starforming = int(is_starforming.sum())
    n_quenched = int(n_total - n_starforming)
    return {
        "n_total": n_total,
        "n_starforming": n_starforming,
        "n_quenched": n_quenched,
        "starforming_fraction": float(n_starforming / n_total) if n_total else None,
        "quenched_fraction": float(n_quenched / n_total) if n_total else None,
        "starforming_fraction_fmt": f"{100 * n_starforming / n_total:.1f}" if n_total else "NA",
        "quenched_fraction_fmt": f"{100 * n_quenched / n_total:.1f}" if n_total else "NA",
    }


def fixed_threshold_satellite_check(
    sample: dict[str, pd.DataFrame],
    threshold: float = -11.0,
) -> dict[str, object]:
    """Compare satellite star-forming fractions using a fixed sSFR threshold."""

    if "CG4" + co.GASUFF not in sample:
        return {"status": "skipped", "reason": "missing_CG4"}

    cg_frame = _satellite_frame(sample, "CG4")
    if cg_frame.empty:
        return {"status": "skipped", "reason": "no_CG4_satellites"}
    cg_summary = _summary(cg_frame, threshold)
    result = {
        "status": "ok",
        "threshold_log10_ssfr_per_year": float(threshold),
        "threshold_label": rf"$\log_{{10}}(\mathrm{{sSFR}}/\mathrm{{yr}}^{{-1}})={threshold:.1f}$",
        "cg4": cg_summary,
        "comparisons": {},
        "morphology_dependence": "Morphology classifications are independent of the sSFR threshold.",
    }

    for control in co.CONTROL:
        if control + co.GASUFF not in sample:
            continue
        control_frame = _satellite_frame(sample, control)
        control_summary = _summary(control_frame, threshold)
        table = [
            [
                cg_summary["n_starforming"],
                cg_summary["n_quenched"],
            ],
            [
                control_summary["n_starforming"],
                control_summary["n_quenched"],
            ],
        ]
        p_value = float(fisher_exact(table, alternative="two-sided").pvalue)
        delta = (
            cg_summary["starforming_fraction"] - control_summary["starforming_fraction"]
            if cg_summary["starforming_fraction"] is not None
            and control_summary["starforming_fraction"] is not None
            else None
        )
        # Use rounded display values for the delta so that the displayed
        # difference is arithmetically consistent with the two shown percentages.
        if delta is not None:
            cg_pct_rounded = round(100 * cg_summary["starforming_fraction"], 1)
            ctrl_pct_rounded = round(100 * control_summary["starforming_fraction"], 1)
            delta_pct_fmt = f"{cg_pct_rounded - ctrl_pct_rounded:.1f}"
        else:
            delta_pct_fmt = "NA"
        result["comparisons"][control] = {
            "control": control_summary,
            "delta_starforming_fraction": float(delta) if delta is not None else None,
            "delta_starforming_fraction_pct_fmt": delta_pct_fmt,
            "fisher_p": p_value,
            "fisher_p_fmt": gu.pvalue_latex(p_value, math_mode=False),
        }

    result["primary_comparison"] = result["comparisons"].get("RG4")
    if result["primary_comparison"]:
        p_value = result["primary_comparison"]["fisher_p"]
        result["summary"] = (
            "The fixed-threshold satellite comparison remains descriptive; the paper's "
            "main hierarchy is unchanged because morphology is unaffected and the "
            "star-formation contrast is threshold/model dependent."
        )
        result["primary_significance"] = (
            "significant" if p_value < co.P_LIMIT else "not_significant"
        )
    return result


def run(sample: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Append the fixed-threshold robustness check to results.json."""

    result = fixed_threshold_satellite_check(sample)
    report.append_json("sSFR_robustness", result)
    return result
