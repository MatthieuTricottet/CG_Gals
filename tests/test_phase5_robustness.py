"""Tests for the Phase 5 robustness additions."""

import numpy as np
import pandas as pd
import pytest

from src import config as co
from src.host_controlled import run_host_controlled_analysis
from src.morphology_threshold_sweep import (
    THRESHOLDS,
    _classify_at,
    run_morphology_threshold_sweep,
)
from tests.test_matched_pairs_stability import add_matched_outcome_columns
from tests.test_size_models import synthetic_size_frame


def test_classify_at_tie_rule_below_half():
    frame = pd.DataFrame({"p_E": [0.45, 0.41, 0.9, 0.2, np.nan],
                          "p_S": [0.41, 0.45, 0.05, 0.1, 0.5]})
    out = _classify_at(frame, 0.4)
    assert out.tolist()[:3] == [1.0, 0.0, 1.0]
    assert np.isnan(out.iloc[3])  # neither fraction passes the threshold
    assert np.isnan(out.iloc[4])  # missing vote fraction stays unclassified


def _frame_with_votes(seed=3):
    rng = np.random.default_rng(seed)
    frame = add_matched_outcome_columns(synthetic_size_frame())
    # vote fractions correlated with the injected elliptical indicator
    base = frame["elliptical"].fillna(0.5)
    frame["p_E"] = np.clip(rng.normal(0.2 + 0.6 * base, 0.15), 0, 1)
    frame["p_S"] = np.clip(1 - frame["p_E"] - rng.uniform(0, 0.1, len(frame)), 0, 1)
    return frame


def test_threshold_sweep_runs_all_thresholds_and_continuous_model():
    result = run_morphology_threshold_sweep(_frame_with_votes())
    assert result["status"] == "ok"
    assert set(result["thresholds"]) == {f"{t:.1f}" for t in THRESHOLDS}
    for entry in result["thresholds"].values():
        assert entry["n_classified"] > 0
    assert result["continuous"]["status"] == "ok"
    assert result["sersic_early_late"]["coverage"] > 0


def test_host_controlled_respects_config_toggle(monkeypatch):
    # host_controlled may bind either the flat `config` module (src on
    # sys.path) or `src.config`, depending on import order; patch both.
    import sys

    for name in ("config", "src.config"):
        if name in sys.modules:
            monkeypatch.setattr(
                sys.modules[name], "HOST_CONTROLLED_ANALYSIS", False, raising=False
            )
    monkeypatch.setattr(co, "HOST_CONTROLLED_ANALYSIS", False)
    result = run_host_controlled_analysis({"CG4_Gals": pd.DataFrame()})
    assert result == {"status": "skipped", "reason": "disabled_by_config"}


def test_host_controlled_runs_on_committed_data(monkeypatch):
    monkeypatch.setattr(co, "HOST_CONTROLLED_ANALYSIS", True)
    cg4 = pd.read_csv(co.DATA_PATH + "CG4_Gals.csv")
    groups = pd.read_csv(co.DATA_PATH + "CG4_Groups.csv")
    nonsplit = groups.loc[groups["Class"] != "Split", "Group"]
    sample = {"CG4_Gals": cg4[cg4["Group"].isin(nonsplit)]}
    result = run_host_controlled_analysis(sample)
    assert result["status"] == "ok"
    # 56 Embedded/Predominant CG4 groups produce 224 CG members; two hosts
    # each contain two CG4 groups, so there are fewer hosts than CG groups
    assert result["n_hosts"] > 40
    assert result["n_cg_members"] == 224
    assert result["n_hosts"] <= 56
    assert result["n_members"] > result["n_cg_members"]
    quenched = result["models"]["quenched"]
    if quenched.get("status") == "ok":
        assert quenched.get("cg_member_p_adj") is None or (
            quenched["cg_member_p_adj"] >= quenched["cg_member_p"]
        )
