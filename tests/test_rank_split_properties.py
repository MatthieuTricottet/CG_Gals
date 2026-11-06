import numpy as np
import pandas as pd
import pytest

from src import rank_split_properties as rs
from src.extended_stats import holm_correction


def _synthetic_frame(seed: int = 0, sat_effect: float = -2.5) -> pd.DataFrame:
    """Quartets with a CG4 effect on satellites only (strong H-alpha, Dn4000)."""

    rng = np.random.default_rng(seed)
    rows = []
    n_groups = {"CG4": 70, "Control4B": 150, "Control4C": 150, "RG4": 90}
    for sample, count in n_groups.items():
        for group in range(count):
            for rank in range(1, 5):
                bgg = rank == 1
                mass = rng.uniform(10.6, 11.6) if bgg else rng.uniform(9.6, 11.0)
                cg_sat = sample == "CG4" and not bgg
                logit = 1.5 - 2.0 * (mass - 10.5) + (sat_effect if cg_sat else 0.0)
                strong = rng.random() < 1 / (1 + np.exp(-logit))
                rows.append(
                    {
                        "sample": sample,
                        "Group": group,
                        "group_uid": f"{sample}:{group}",
                        "physical_group": f"{sample}:{group}",
                        "rank": rank,
                        "logMstar": mass,
                        "h_alpha_eqw": -8.0 if strong else -0.5,
                        "log_NII_Ha": rng.normal(-0.2, 0.3),
                        "log_OIII_Hb": rng.normal(0.0, 0.4),
                        "Dn4000": 1.4 + 0.3 * (mass - 10) + (0.15 if cg_sat else 0.0)
                        + rng.normal(0, 0.05),
                        "log_Rchl_r_kpc": rng.normal(0.5, 0.1),
                        "log_Rchl_r_kpc_incl_pegged": rng.normal(0.5, 0.1),
                        "log_petroR50_kpc": rng.normal(0.55, 0.1),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def result():
    # The BGG-level CI excludes 0 at about the nominal 5% rate by chance
    # (checked over 20 seeds, mean z of CG_bgg -0.3 +- 0.2), so the seed is
    # fixed to keep the classification test deterministic.
    return rs.run_rank_split_properties(_synthetic_frame(), None, n_boot=30)


def _entry(entries, outcome, control):
    return next(e for e in entries if e["outcome"] == outcome and e["control"] == control)


def test_satellite_only_effect_is_classified_sat(result):
    assert result["status"] == "ok"
    for control in rs.CONTROLS:
        strong = _entry(result["interaction_models"], "strong_halpha", control)
        assert strong["class_code"] in ("SAT", "SAT?")
        assert strong["cg_sat"]["estimate"] < 0
        dn = _entry(result["interaction_models"], "Dn4000", control)
        assert dn["class_code"] in ("SAT", "SAT?")
        assert dn["cg_sat"]["estimate"] > 0


def test_model_r_reparametrises_model_i(result):
    for entry in result["interaction_models"] + result["sensitivity_interaction_models"]:
        assert entry["status"] == "ok"
        difference = entry["cg_bgg"]["estimate"] - entry["cg_sat"]["estimate"]
        assert difference == pytest.approx(entry["interaction"]["estimate"], abs=1e-8)
        assert abs(entry["identity_residual"]) < rs.IDENTITY_TOLERANCE


def test_holm_families_have_twelve_tests(result):
    for entries, getter, key in (
        (result["interaction_models"], lambda e: e["interaction"]["p"], "interaction_p_holm"),
        (result["satellite_models"], lambda e: e["cg"]["p"], "cg_p_holm"),
    ):
        assert len(entries) == 12
        assert all(e[f"{key}_family_size"] == 12 for e in entries)
        expected = holm_correction([getter(e) for e in entries])
        assert [e[key] for e in entries] == pytest.approx(expected)
    # sensitivity outcomes stay outside the families
    assert all("interaction_p_holm" not in e for e in result["sensitivity_interaction_models"])


def test_spline_basis_is_centred_and_full_rank():
    mass = pd.Series(np.linspace(10.5, 11.8, 200))
    basis = rs.spline_basis(mass, "ms")
    assert basis.shape == (200, rs.SPLINE_DF)
    design = np.column_stack([np.ones(len(basis)), basis.to_numpy()])
    assert np.linalg.matrix_rank(design) == rs.SPLINE_DF + 1


@pytest.mark.parametrize(
    "sat, bgg, holm, code",
    [
        ((-1, 0.5), (-0.5, 0.5), 0.01, "SAT"),
        ((-1, 0.5), (-0.5, 0.5), 0.40, "SAT?"),
        ((-1, 0.5), (-1, 0.5), 0.40, "HOST"),
        ((-1, 0.5), (1, 0.5), 0.01, "OPP"),
        ((0.1, 0.5), (1, 0.5), 0.40, "BGG"),
        ((0.1, 0.5), (0.1, 0.5), 0.90, "NULL"),
        ((-1, 0.5), (-2, 0.5), 0.01, "UNMATCHED"),
    ],
)
def test_classification_rules(sat, bgg, holm, code):
    def term(estimate, se):
        return {"estimate": estimate, "ci95": [estimate - 1.96 * se, estimate + 1.96 * se]}

    assert rs.classify(term(*sat), term(*bgg), holm) == code


def test_rank_split_diagnostics_run_on_synthetic_data(tmp_path):
    from src import rank_split_diagnostics as rd

    frame = _synthetic_frame()
    rng = np.random.default_rng(5)
    frame["specobjid"] = np.arange(len(frame)) + 1000
    frame["z_numeric"] = rng.uniform(0.01, 0.04, len(frame))
    frame["petroR50_kpc"] = rng.uniform(1.0, 8.0, len(frame))
    cache = tmp_path / "galspecline.csv"
    lines = {"specobjid": frame["specobjid"]}
    for line in rd.BPT_LINES:
        lines[f"{line}_flux"] = rng.uniform(5, 50, len(frame))
        lines[f"{line}_flux_err"] = rng.uniform(0.5, 10, len(frame))
    pd.DataFrame(lines).to_csv(cache, index=False)
    work, info = rd.attach_line_sn(rs.prepare_frame(frame), cache_path=str(cache))
    assert info["status"] == "ok" and info["n_classified_with_errors"] == info["n_classified"]
    work = rd.add_diagnostic_outcomes(work)
    # WHAN AGN-like is a subset of AGN-like; retired rows never count
    assert (work["agn_like_whan"] <= work["agn_like"]).all()
    assert work.loc[work["retired_like"].eq(1), "agn_like_whan"].fillna(0).eq(0).all()
    # log_fib_frac = log10(1.5 arcsec in kpc / R50)
    from src.size_data import _kpc_per_arcsec
    expected = np.log10(1.5 * _kpc_per_arcsec(frame["z_numeric"]) / frame["petroR50_kpc"])
    assert np.allclose(work["log_fib_frac"], expected)
    shares = rd.retired_share(work)
    assert all(0 <= row["retired_share"] <= 1 for row in shares if row["retired_share"] is not None)
