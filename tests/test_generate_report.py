import pytest

from src.generate_report import _build_render_context, deep_merge


def _minimal_results_context():
    return {
        "Colour_matching_summary": [],
        "phase_space_segregation": {"status": "ok"},
        "morphology_dominance": {"status": "ok"},
        "extended_specialness": {
            "specialness_models": {
                "elliptical_all": {"status": "ok"},
                "spiral_all": {"status": "ok"},
            },
            "matched_controls": {
                "effects": {
                    "elliptical_fraction": {"status": "ok"},
                    "spiral_fraction": {"status": "ok"},
                }
            },
            "morphology_robustness": {
                "status": "ok",
                "concentration_index": {"status": "skipped"},
                "cg_class_split": {"status": "ok"},
                "adjusted_models_with_close_flag": {},
                "exact_tests_after_excluding_close": [],
            },
            "morphology_dominance": {"status": "ok"},
            "sample_size_audit": {"CG4": {"total_galaxies": 248}},
        },
    }


def test_deep_merge_recurses_without_deleting_base_keys():
    base = {
        "legacy": 1,
        "nested": {"keep": "base", "replace": "old", "inner": {"a": 1}},
        "items": [1],
    }
    overlay = {
        "nested": {"replace": "new", "inner": {"b": 2}},
        "items": [2],
    }

    merged = deep_merge(base, overlay)

    assert merged == {
        "legacy": 1,
        "nested": {
            "keep": "base",
            "replace": "new",
            "inner": {"a": 1, "b": 2},
        },
        "items": [2],
    }
    assert base["nested"]["replace"] == "old"


def test_build_render_context_uses_single_merged_results_world():
    build_data = {
        "CG4_Groups_nonsplit_N": 62,
        "pval_MSresiduals_Control4B_Gals": {"p_value": 0.1},
        "CG4_Gals_N_Elliptical": 124,
        "nested": {"from_build": True},
    }
    results_data = _minimal_results_context() | {
        "nested": {"from_results": True},
    }

    ctx, render_data = _build_render_context(build_data, results_data)

    assert render_data["nested"] == {"from_build": True, "from_results": True}
    assert render_data["extended_specialness"]["morphology_robustness"]["status"] == "ok"
    assert ctx["r"] is render_data
    assert ctx["build"] is render_data
    assert ctx["extended_specialness"] is render_data["extended_specialness"]


def test_build_render_context_fails_loudly_for_missing_robustness_block():
    build_data = {
        "CG4_Groups_nonsplit_N": 62,
        "pval_MSresiduals_Control4B_Gals": {"p_value": 0.1},
        "CG4_Gals_N_Elliptical": 124,
    }
    results_data = _minimal_results_context()
    del results_data["extended_specialness"]["morphology_robustness"]

    with pytest.raises(KeyError, match="morphology_robustness"):
        _build_render_context(build_data, results_data)
