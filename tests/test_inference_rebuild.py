"""Invariants for the Phase 4 inference rebuild.

Covers: the Monte-Carlo p-value floor, block resampling, control-pool
deduplication, matching hard constraints, provenance reporting, physical
clustering and the per-control primary contrasts.
"""

import numpy as np
import pandas as pd
import pytest

from src.extended_data import dedup_control_pool
from src.extended_stats import (
    bootstrap_difference,
    empirical_p_two_sided,
    holm_correction,
)
from src.matched_controls import run_matched_control_analysis
from src.primary_contrasts import run_primary_contrasts
from tests.test_matched_pairs_stability import add_matched_outcome_columns
from tests.test_size_models import synthetic_size_frame


def test_empirical_p_never_returns_zero():
    boot = np.full(2000, 0.5)  # every draw on the same side of zero
    p = empirical_p_two_sided(boot)
    assert p == pytest.approx(2 / 2001)
    assert p > 0


def test_bootstrap_difference_reports_floor_and_never_underflows():
    rng = np.random.default_rng(0)
    treated = rng.normal(5.0, 0.1, 200)  # overwhelming difference
    control = rng.normal(0.0, 0.1, 200)
    result = bootstrap_difference(treated, control, paired=True, n_boot=999)
    assert result["p"] >= result["p_floor"] == pytest.approx(2 / 1000)
    assert result["n_boot"] == 999


def test_bootstrap_difference_block_mode():
    rng = np.random.default_rng(1)
    n = 120
    blocks = np.repeat(np.arange(30), 4)
    treated = rng.normal(0.4, 1.0, n)
    control = rng.normal(0.0, 1.0, n)
    result = bootstrap_difference(
        treated, control, paired=True, n_boot=499, blocks=blocks
    )
    assert result["resampling_unit"] == "block"
    assert result["n_blocks"] == 30


def test_holm_adjusted_never_below_raw():
    raw = [0.001, 0.02, 0.04, 0.2, 0.6]
    adjusted = holm_correction(raw)
    assert all(a >= r for a, r in zip(adjusted, raw))


def _frame_with_control_duplicates():
    frame = add_matched_outcome_columns(synthetic_size_frame())
    # duplicate a control galaxy under a second label, as the real control
    # samples do (all RG4 galaxies are also Control4B rows)
    controls = frame.loc[frame["is_CG4"] == 0]
    dup = controls.iloc[:40].copy()
    dup["sample"] = np.where(dup["sample"].eq("Control4B"), "Control4C", "Control4B")
    dup["group_uid"] = dup["sample"] + ":" + dup["group_uid"].str.split(":").str[1]
    out = pd.concat([frame, dup], ignore_index=True)
    labels = (
        out.loc[out["is_CG4"] == 0]
        .groupby("objid")["sample"]
        .agg(lambda v: "+".join(sorted(set(v))))
    )
    out["control_source_labels"] = out["objid"].map(labels)
    return out


def test_dedup_control_pool_collapses_duplicates_and_blocks_cg4():
    frame = _frame_with_control_duplicates()
    deduped = dedup_control_pool(frame)
    controls = deduped.loc[deduped["is_CG4"] == 0]
    assert controls["objid"].is_unique
    assert (deduped["is_CG4"] == 1).sum() == (frame["is_CG4"] == 1).sum()

    poisoned = frame.copy()
    cg4_objid = poisoned.loc[poisoned["is_CG4"] == 1, "objid"].iloc[0]
    poisoned.loc[poisoned.index[poisoned["is_CG4"] == 0][0], "objid"] = cg4_objid
    with pytest.raises(ValueError, match="Paper I exclusion"):
        dedup_control_pool(poisoned)


def test_matching_hard_constraints_and_provenance():
    frame = _frame_with_control_duplicates()
    result = run_matched_control_analysis(frame)
    assert result["status"] == "ok"
    assert result["control_pool_deduplicated_by_objid"] is True
    assert result["cg4_objids_excluded_from_controls"] is True
    # every matched control is a distinct physical galaxy
    assert result["n_control_unique"] == result["n_control_matched"]
    assert "matched_control_counts_by_provenance" in result
    assert result["n_matched_controls_physically_RG4"] >= 0
    for effect in result["effects"].values():
        if effect.get("status") == "ok":
            assert effect["p"] is None or effect["p"] >= effect["p_floor"]
            assert effect["resampling_unit"] == "block"


def test_group_level_analysis_reports_counts():
    frame = add_matched_outcome_columns(synthetic_size_frame())
    result = run_matched_control_analysis(frame)
    group_level = result["group_level"]
    assert group_level["status"] == "ok"
    assert group_level["unit"] == "group"
    assert group_level["n_matched_groups"] >= 10
    distribution = group_level["n_smooth_sat_distribution_cg4"]
    assert set(distribution) <= {"0", "1", "2", "3"}
    assert group_level["p"] >= group_level["p_floor"]
    assert set(result["group_level_per_control"]) == {"Control4B", "Control4C", "RG4"}
    ok_per_control = [
        item
        for item in result["group_level_per_control"].values()
        if item.get("status") == "ok"
    ]
    assert ok_per_control


def test_primary_contrasts_fit_three_families():
    frame = add_matched_outcome_columns(synthetic_size_frame())
    result = run_primary_contrasts(frame)
    assert result["status"] == "ok"
    assert set(result["contrasts"]) == {"Control4B", "Control4C", "RG4"}
    assert result["cluster_unit"] == "physical_group"
    for contrast in result["contrasts"].values():
        ok_models = [
            value
            for value in contrast.values()
            if isinstance(value, dict) and value.get("status") == "ok"
        ]
        assert ok_models, "every contrast should fit at least one model"
        for model in ok_models:
            if model.get("cg4_p_adj") is not None and model.get("cg4_p") is not None:
                assert model["cg4_p_adj"] >= model["cg4_p"]
