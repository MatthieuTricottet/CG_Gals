import json

import numpy as np

from src.extended_stats import safe_json
from src.matched_controls import (
    _greedy_match,
    _select_variables,
    run_matched_control_analysis,
)
from src.size_analysis import _matched
from tests.test_size_models import synthetic_size_frame


def add_matched_outcome_columns(frame, seed=20260612):
    rng = np.random.default_rng(seed)
    frame = frame.copy()
    frame["quenched"] = rng.binomial(1, 0.5, len(frame)).astype(float)
    frame["starforming"] = 1 - frame["quenched"]
    return frame


def test_size_module_reproduces_matched_control_pairs():
    frame = add_matched_outcome_columns(synthetic_size_frame())
    variables = _select_variables(frame)
    direct_pairs, _, direct_caliper = _greedy_match(frame, variables)

    size_result = _matched(frame)
    assert size_result["status"] == "ok"
    assert size_result["matching_variables"] == variables
    assert size_result["n_pairs"] == len(direct_pairs)
    assert size_result["deterministic"] is True

    matched_controls_result = run_matched_control_analysis(frame)
    assert matched_controls_result["status"] == "ok"
    assert size_result["n_pairs"] == matched_controls_result["n_cg4_matched"]

    # The orchestrator hands the matched-control count through frame.attrs;
    # the size block must confirm consistency against it.
    frame.attrs["matched_controls_n_cg4_matched"] = matched_controls_result[
        "n_cg4_matched"
    ]
    size_result_checked = _matched(frame)
    assert size_result_checked["pair_count_consistent"] is True


def test_matched_controls_return_dict_unchanged_by_refactor():
    frame = add_matched_outcome_columns(synthetic_size_frame())
    first = safe_json(run_matched_control_analysis(frame))
    second = safe_json(run_matched_control_analysis(frame))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    expected_keys = {
        "status",
        "method",
        "replacement",
        "common_support",
        "propensity_caliper_logit_sd",
        "propensity_caliper",
        "matching_variables",
        "propensity_variables",
        "exact_matching_variables",
        "spatial_variables_not_matched",
        "spatial_exclusion_reason",
        "n_cg4_matched",
        "n_control_matched",
        "n_control_unique",
        "matched_counts_by_sample",
        "matched_control_counts_by_sample",
        "median_match_distance",
        "balance",
        "max_abs_smd_before",
        "max_abs_smd_after",
        "effects",
        "holm_correction_family",
        "holm_correction_note",
        "complementarity_audit",
    }
    assert set(first.keys()) == expected_keys
    assert set(first["effects"].keys()) == {
        "quenched_fraction",
        "starforming_fraction",
        "elliptical_fraction",
        "spiral_fraction",
        "residual_sSFR_starforming",
        "colour_residual_u_minus_r",
    }
