"""Assertions for analysis inputs, derived tables and inferential contracts."""

import numpy as np
import pandas as pd
from pypdf import PdfReader

from referee_common import FIGURES, TABLES, paired_bootstrap
from compactness import CALIPERS


def read(name, **kwargs):
    return pd.read_csv(TABLES / f"{name}.csv", **kwargs)


def test_samples_and_shared_hosts():
    counts = read("sample_counts").set_index("sample")
    assert (
        counts.loc[
            "CG4", ["n", "quartets", "hosts", "E_S", "uncertain", "missing_votes"]
        ]
        == [248, 62, 60, 210, 7, 31]
    ).all()
    assert counts.loc["Control4C", "quartets"] == 703
    assert (counts.E_S + counts.uncertain + counts.missing_votes == counts.n).all()
    overlaps = read("sample_overlap")
    comparison = overlaps.query("sample=='CG4'")
    assert comparison.shared_objids.eq(0).all()
    assert comparison.query("other=='Control4C'").shared_hosts.item() == 3
    a = read("appendix_c_exclusions", dtype={"objid": "int64"})
    assert a.Group_Lim.nunique() == 61 and a.objid.is_unique and len(a) == 233


def test_distance_units():
    q = read("quartets")
    pairs = q[["R12_kpc", "R13_kpc", "R14_kpc", "R23_kpc", "R24_kpc", "R34_kpc"]]
    np.testing.assert_allclose(q.R_mean, pairs.mean(axis=1), rtol=1e-8)
    np.testing.assert_allclose(q.R_median, pairs.median(axis=1), rtol=1e-8)
    np.testing.assert_allclose(q.DL_kpc / q.DA_kpc, (1 + q.quartet_z) ** 2, rtol=1e-8)
    t1 = q.groupby("sample").table1_R_median.median().round().astype(int)
    assert t1.to_dict() == {"CG4": 143, "Control4B": 447, "Control4C": 293, "RG4": 430}


def test_reproducible_matches_and_bootstrap():
    p = read("compact_matching_pairs")
    for label, g in p.groupby("spec"):
        assert g.cg_quartet.is_unique and g.cc_quartet.is_unique
        assert g.cg_host.ne(g.cc_host).all()
        lims = {k: v for k, v in CALIPERS.items() if k != "x"}
        if not label.startswith("without"):
            lims["x"] = {
                "compactness_matched": 0.08,
                "tighter_radius": 0.05,
                "wider_radius": 0.12,
                "velocity_balanced": 0.08,
            }[label]
        if label == "velocity_balanced":
            lims["velocity_dispersion"] = 120.0
        for key, limit in lims.items():
            assert g[f"diff_{key}"].abs().le(limit + 1e-8).all()
        d = (g.cg_fraction - g.cc_fraction).dropna()
        x = paired_bootstrap(d.to_numpy(), g.loc[d.index, "component"].to_numpy())
        y = paired_bootstrap(d.to_numpy(), g.loc[d.index, "component"].to_numpy())
        assert x == y
        summary = read("compact_matching_effects").query("spec==@label").iloc[0]
        np.testing.assert_allclose(
            [x[k] for k in ("estimate", "low", "high")],
            summary[["estimate", "low", "high"]].to_numpy(dtype=float),
            atol=1e-8,
        )
    m = read("paper_match_denominators").set_index(["population", "stage"])
    assert m.n_pairs.to_dict() == {
        ("all", "matched"): 234,
        ("all", "both_E_S"): 180,
        ("satellites", "matched"): 177,
        ("satellites", "both_E_S"): 130,
    }


def test_host_geometry_and_fit_status():
    q = read("within_host_quartets")
    assert (len(q), q.Lim_host.nunique(), q.exact.sum()) == (56, 54, 46)
    assert not q.loc[q.exact, "eligible_satellite_overlap"].any()
    members = read("within_host_members", dtype={"objid": "int64"})
    assert members.objid.is_unique and len(members) == 958
    m = read("within_host_models").set_index("spec")
    assert m.loc["original_radius", "outcome_informative"] == 53
    assert m.loc["original_radius", "both_informative"] == 52
    np.testing.assert_allclose(
        m.loc[["original_no_radius", "original_radius"], "OR"], [1.90, 1.47], atol=0.012
    )
    assert (
        m.loc[m.index.str.endswith("_kpc"), "status"].eq("insufficient support").all()
    )
    assert (
        m.loc[m.status.eq("ok"), ["p", "OR", "score_per_galaxy"]].notna().all(axis=None)
    )
    assert read("mass_model_failures").empty


def test_missing_and_figures():
    m = read("cg4_missing_GZ1", dtype={"objid": "int64"})
    assert m.objid.is_unique and len(m) == 38 and m.is_satellite.sum() == 29
    assert m.finite_gz_votes.sum() == 7
    assert read("boss_sfr_summary").missing_BOSS.sum() == 332
    zoo = read("gz1_dr7_recovery_summary").set_index("sample").loc["CG4"]
    assert (
        zoo[
            [
                "baseline_labels",
                "matching_labels",
                "E_recovered",
                "S_recovered",
                "expanded_labels",
            ]
        ]
        == [210, 210, 6, 10, 226]
    ).all()
    files = list(FIGURES.glob("*.pdf"))
    assert len(files) == 8
    for f in files:
        assert (
            len(PdfReader(f).pages) == 1
            and f.with_suffix(".png").stat().st_size > 10000
        )
