import json
import os

import pytest
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src import config as co
from src import generate_report as gr

SIZE_STUB_OK = {
    "status": "ok",
    "quality_cuts": {
        "size_min_kpc": 0.1,
        "size_max_kpc": 50.0,
        "ng_peg_low": 0.55,
        "ng_peg_high": 7.9,
        "z_match_tolerance": 0.005,
        "close_neighbour_arcsec": 55.0,
    },
    "holm_families": {"F1": [], "F2": [], "F3": []},
    "availability_audit": {
        "status": "ok",
        "totals": {"z_mismatch": 0, "shred_merge": 0, "n_pegged": 0},
        "cg4_simard_fraction": 0.77,
        "pooled_control_simard_fraction": 0.72,
        "completeness_caveat": True,
    },
    "mass_size": {"status": "ok", "reference_logMstar": 10.3, "by_sample": {}},
    "adjusted": {"status": "ok", "all": {}, "satellites": {}, "bgg": {}},
    "per_control": {"status": "ok", "comparisons": {}},
    "morphology_strata": {"status": "ok"},
    "luminosity_version": {"status": "ok"},
    "matched": {"status": "skipped", "reason": "stub", "effects": {}},
    "crowding": {"status": "skipped", "reason": "stub"},
    "petrosian": {"status": "skipped", "reason": "stub"},
    "measure_delta": {"status": "skipped", "reason": "stub"},
    "tidal": {"status": "skipped", "reason": "stub"},
    "radial": {"status": "skipped", "reason": "stub"},
    "concentration": {"status": "skipped", "reason": "stub"},
    "verdicts": {
        "primary_all_significant": False,
        "primary_satellites_significant": False,
        "direction": None,
        "survives_matching": False,
        "survives_crowding": False,
        "petro_consistent": False,
        "absorbed_by_tidal_index": False,
        "completeness_caveat": True,
    },
}


def render_with_size_block(size_block):
    build_data = gr._load_json(co.RESULTS_BUILD)
    results_data = gr._load_json(co.RESULTS)
    if not build_data or not results_data:
        pytest.skip("committed results JSON files are unavailable")
    results_data.setdefault("extended_specialness", {})["size_analysis"] = size_block
    results_data.setdefault("size_dr7_bridge", "SpecDR7")

    ctx, _ = gr._build_render_context(build_data, results_data)
    env = Environment(
        loader=FileSystemLoader(co.TEMPLATE_PATH),
        undefined=StrictUndefined,
        block_start_string="<%",
        block_end_string="%>",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="<#",
        comment_end_string="#>",
    )
    env.filters["fmt"] = gr._format_number
    env.filters["pct"] = gr._format_percent
    template = env.get_template(co.TEMPLATE_FILE)
    return template.render(ctx)


def test_template_renders_with_ok_size_stub():
    rendered = render_with_size_block(json.loads(json.dumps(SIZE_STUB_OK)))
    assert r"\subsection{Galaxy sizes}" in rendered
    assert r"\subsection{Galaxy sizes at fixed stellar mass}" in rendered
    assert "<<" not in rendered


def test_template_renders_with_skipped_size_block():
    rendered = render_with_size_block(
        {"status": "skipped", "reason": "size_data_unavailable"}
    )
    assert r"\subsection{Galaxy sizes}" not in rendered
    assert "<<" not in rendered


def test_latex_build_is_out_of_scope_here():
    """The LaTeX binary step is exercised by the pipeline, not this test."""

    assert os.path.exists(os.path.join(co.TEMPLATE_PATH, co.TEMPLATE_FILE))
