"""Orchestration for the extended compact-group specialness analyses."""

from __future__ import annotations

import os
import traceback

try:
    import config as co
    import generate_report as report
    from agn_environment import run_agn_environment_analysis
    from extended_data import build_galaxy_frame
    from extended_stats import safe_json
    from fossilness import run_fossilness_analysis
    from matched_controls import run_matched_control_analysis
    from phase_space import run_phase_space_analysis
    from recent_quenching import run_recent_quenching_analysis
    from selection_diagnostics import run_selection_diagnostics
    from specialness_models import fit_logistic_specialness_models
    from tidal_indices import run_tidal_indices_analysis
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from . import generate_report as report
    from .agn_environment import run_agn_environment_analysis
    from .extended_data import build_galaxy_frame
    from .extended_stats import safe_json
    from .fossilness import run_fossilness_analysis
    from .matched_controls import run_matched_control_analysis
    from .phase_space import run_phase_space_analysis
    from .recent_quenching import run_recent_quenching_analysis
    from .selection_diagnostics import run_selection_diagnostics
    from .specialness_models import fit_logistic_specialness_models
    from .tidal_indices import run_tidal_indices_analysis


def _failed(exc):
    return {
        "status": "skipped",
        "reason": "analysis_exception",
        "error": f"{exc.__class__.__name__}: {exc}",
    }


def run_extended_specialness(sample, output_dir: str | None = None):
    """Run every extended analysis independently and append one JSON object."""

    output_dir = output_dir or co.FIGURES_PATH
    os.makedirs(output_dir, exist_ok=True)
    galaxies = build_galaxy_frame(sample)
    analyses = [
        ("specialness_models", fit_logistic_specialness_models),
        ("matched_controls", run_matched_control_analysis),
        ("phase_space", run_phase_space_analysis),
        ("fossilness", run_fossilness_analysis),
        ("recent_quenching", run_recent_quenching_analysis),
        ("agn_environment", run_agn_environment_analysis),
        ("tidal_indices", run_tidal_indices_analysis),
        ("selection_diagnostics", run_selection_diagnostics),
    ]
    results = {"status": "ok", "n_galaxies": int(len(galaxies))}
    for name, function in analyses:
        try:
            results[name] = function(galaxies, output_dir=output_dir)
        except Exception as exc:  # One exploratory analysis must not stop the paper.
            results[name] = _failed(exc)
            if co.VERBOSE:
                print(f"[extended specialness] {name} failed: {exc}")
                traceback.print_exc()

    adjusted = results["specialness_models"].get("passive_all", {})
    matched = results["matched_controls"].get("effects", {}).get("passive_fraction", {})
    results["interpretation"] = {
        "adjusted_passive_signal": adjusted.get("cg4_p_adj", 1) is not None
        and adjusted.get("cg4_p_adj", 1) < 0.05,
        "matched_passive_signal": matched.get("p_adj", 1) is not None
        and matched.get("p_adj", 1) < 0.05,
        "independent_compact_group_signal": (
            adjusted.get("cg4_p_adj", 1) is not None
            and adjusted.get("cg4_p_adj", 1) < 0.05
            and matched.get("p_adj", 1) is not None
            and matched.get("p_adj", 1) < 0.05
        ),
        "phase_space_signal": results["phase_space"].get(
            "fixed_phase_space_cg4_significant", False
        ),
        "magnitude_gap_signal": results["fossilness"].get(
            "magnitude_gap_significant", False
        ),
        "strong_selection_bias": results["selection_diagnostics"].get(
            "strong_selection_bias", False
        ),
    }
    results["skipped_analyses"] = [
        name for name, _ in analyses if results[name].get("status") == "skipped"
    ]
    results = safe_json(results)
    report.append_json("extended_specialness", results)
    return results
