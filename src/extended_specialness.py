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
    from morphology_dominance import run_morphology_dominance_analysis
    from morphology_robustness import run_morphology_robustness
    from morphology_threshold_sweep import run_morphology_threshold_sweep
    from phase_space_segregation import run_phase_space_segregation_analysis
    from recent_quenching import run_recent_quenching_analysis
    from selection_diagnostics import run_selection_diagnostics
    from size_analysis import run_size_analysis
    from size_data import attach_size_columns
    from primary_contrasts import run_primary_contrasts
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
    from .morphology_dominance import run_morphology_dominance_analysis
    from .morphology_robustness import run_morphology_robustness
    from .morphology_threshold_sweep import run_morphology_threshold_sweep
    from .phase_space_segregation import run_phase_space_segregation_analysis
    from .recent_quenching import run_recent_quenching_analysis
    from .selection_diagnostics import run_selection_diagnostics
    from .size_analysis import run_size_analysis
    from .size_data import attach_size_columns
    from .primary_contrasts import run_primary_contrasts
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
    try:
        # Enrich the shared frame with the cached size columns so both the
        # size analysis and the dormant concentration hook can see them; a
        # failed fetch must not stop the other analyses.
        galaxies, _ = attach_size_columns(galaxies)
    except Exception as exc:
        if co.VERBOSE:
            print(f"[extended specialness] size columns unavailable: {exc}")
    analyses = [
        ("primary_contrasts", run_primary_contrasts),
        ("specialness_models", fit_logistic_specialness_models),
        ("matched_controls", run_matched_control_analysis),
        ("morphology_robustness", run_morphology_robustness),
        ("morphology_threshold_sweep", run_morphology_threshold_sweep),
        ("morphology_dominance", run_morphology_dominance_analysis),
        ("phase_space_segregation", run_phase_space_segregation_analysis),
        ("fossilness", run_fossilness_analysis),
        ("recent_quenching", run_recent_quenching_analysis),
        ("agn_environment", run_agn_environment_analysis),
        ("tidal_indices", run_tidal_indices_analysis),
        ("selection_diagnostics", run_selection_diagnostics),
        ("size_analysis", run_size_analysis),
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
        if name == "matched_controls" and isinstance(results[name], dict):
            # The size analysis re-derives the same pairs and asserts count
            # consistency against this run's matched-control block.
            galaxies.attrs["matched_controls_n_cg4_matched"] = results[name].get(
                "n_cg4_matched"
            )

    adjusted = results["specialness_models"].get("quenched_all", {})
    matched = results["matched_controls"].get("effects", {}).get("quenched_fraction", {})
    results["interpretation"] = {
        "adjusted_quenched_signal": adjusted.get("cg4_p_adj", 1) is not None
        and adjusted.get("cg4_p_adj", 1) < 0.05,
        "matched_quenched_signal": matched.get("p_adj", 1) is not None
        and matched.get("p_adj", 1) < 0.05,
        "independent_compact_group_signal": (
            adjusted.get("cg4_p_adj", 1) is not None
            and adjusted.get("cg4_p_adj", 1) < 0.05
            and matched.get("p_adj", 1) is not None
            and matched.get("p_adj", 1) < 0.05
        ),
        "phase_space_signal": results["phase_space_segregation"].get(
            "fixed_phase_space_cg4_significant", False
        ),
        "magnitude_gap_signal": results["fossilness"].get(
            "magnitude_gap_significant", False
        ),
        "strong_selection_bias": results["selection_diagnostics"].get(
            "strong_selection_bias", False
        ),
    }
    results["phase_space"] = results.get("phase_space_segregation", {})
    selection = results.get("selection_diagnostics", {})
    matched = results.get("matched_controls", {})
    if (
        isinstance(selection, dict)
        and isinstance(matched, dict)
        and selection.get("status") == "ok"
        and matched.get("status") == "ok"
    ):
        audit = selection.get("sample_size_audit", {})
        for sample_name, count in matched.get("matched_counts_by_sample", {}).items():
            if sample_name in audit:
                audit[sample_name]["matched_N"] = count
        selection["sample_size_audit"] = audit
        results["sample_size_audit"] = audit
    try:
        try:
            from host_controlled import run_host_controlled_analysis
        except ModuleNotFoundError:  # pragma: no cover
            from .host_controlled import run_host_controlled_analysis
        results["host_controlled"] = run_host_controlled_analysis(
            sample if isinstance(sample, dict) else {}, output_dir=output_dir
        )
    except Exception as exc:
        results["host_controlled"] = _failed(exc)
        if co.VERBOSE:
            print(f"[extended specialness] host_controlled failed: {exc}")
            traceback.print_exc()
    results["skipped_analyses"] = [
        name for name, _ in analyses if results[name].get("status") == "skipped"
    ]
    results = safe_json(results)
    report.append_json("phase_space_segregation", results["phase_space_segregation"])
    report.append_json("morphology_dominance", results["morphology_dominance"])
    report.append_json("extended_specialness", results)
    return results
