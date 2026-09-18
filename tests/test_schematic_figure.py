"""Fig. 1 schematic: panel selection, exclusion status, and legibility guards.

The two hosts are chosen by documented rules (src/schematic_figure.py); the
tests pin those choices to the cached processed sample and check the
properties the caption and Appendix rely on.  They also guard against the
defect of the first version, whose in-panel inset hid three host members (one
of them a CG4 member that was outside the inset window as well).
"""

import os
import pickle
import sys

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

import config as co  # noqa: E402
import schematic_figure as sf  # noqa: E402

PKL = os.path.join(co.DATA_PATH, co.PROCESS_SAMPLES)
pytestmark = pytest.mark.skipif(not os.path.exists(PKL), reason="cached processed sample not available")


@pytest.fixture(scope="module")
def sample():
    with open(PKL, "rb") as fh:
        return pickle.load(fh)


@pytest.fixture(scope="module")
def result(sample, tmp_path_factory):
    out = tmp_path_factory.mktemp("schematic")
    res = sf.run_schematic_figure(sample, output_dir=str(out))
    assert os.path.exists(os.path.join(out, res["figure"]))
    return res


def test_panel_a_is_the_host_core_case(result):
    assert (result["cg4_group"], result["lim_host"]) == (204, 1117)
    assert result["n_cg4_in_would_be_control4c"] == 4
    assert result["cg4_bgg_is_host_bgg"]
    assert not result["control4b_retained"] and not result["control4c_retained"]


def test_panel_b_is_an_off_centre_embedded_group_with_retained_control4c(result):
    b = result["panel_b"]
    assert (b["cg4_group"], b["lim_host"]) == (330, 1289)
    assert b["n_cg4_in_would_be_control4c"] == 0
    assert b["control4c_retained"] and not b["control4b_retained"]
    assert b["n_cg4_in_would_be_control4b"] == 1
    assert not b["cg4_bgg_is_host_bgg"]
    assert b["cg4_centre_offset_kpc"] > 500
    assert sorted(b["candidate_cg4_groups"]) == [330, 337, 368]


def test_panel_b_control4c_quartet_is_the_final_sample_quartet(sample, result):
    b = result["panel_b"]
    final = sample["Control4C" + co.GASUFF]
    objids = set(final.loc[final["Group"] == b["lim_host"], "objid"].astype("int64"))
    assert objids == set(b["would_be_control4c_objids"])
    assert not objids & set(b["cg4_member_objids"])


def test_off_centre_summary_matches_appendix_claims(result):
    off = result["off_centre_embedded"]
    assert off["n"] == 5
    assert off["n_host_control4c_retained"] == 3
    assert off["n_host_control4c_with_other_cg4"] == 2
    assert off["n_host_control4c_retained"] + off["n_host_control4c_with_other_cg4"] == off["n"]
    assert off["n_cg4_bgg_in_would_be_control4b"] == off["n"]
    assert off["n_host_control4b_retained"] == 0


def test_status_annotations_hide_no_member_or_zoom_box(result):
    """The in-panel status text stays clear of every member and of the zoom box."""

    min_clear = 0.06 * 2 * result["half_width_kpc"]
    for panel in (result, result["panel_b"]):
        assert panel["status_clearance_kpc"] > min_clear


def test_zoom_window_contains_all_cg4_members(sample, result):
    pc = pd.read_csv(os.path.join(co.DATA_PATH, "PC_Gals.csv"))
    pc["objid"] = pc["objid"].astype("int64")
    for panel in (result, result["panel_b"]):
        data = sf._panel_data(sample, pc, panel["cg4_group"], panel["lim_host"])
        host = data["host"]
        cx, cy = panel["inset_centre_kpc"]
        half = result["inset_half_kpc"]
        members = host[host["objid"].isin(panel["cg4_member_objids"])]
        assert len(members) == 4
        assert ((members["dx"] - cx).abs() < half - sf.INSET_PAD_KPC / 2).all()
        assert ((members["dy"] - cy).abs() < half - sf.INSET_PAD_KPC / 2).all()
