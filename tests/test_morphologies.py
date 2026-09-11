import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import config as co  # noqa: E402
import morphologies as morph  # noqa: E402


def test_classify_keeps_missing_galaxy_zoo_separate_from_uncertain():
    frame = pd.DataFrame(
        {
            "p_E": [0.7, 0.2, 0.3, np.nan, 0.8],
            "p_S": [0.1, 0.8, 0.4, 0.6, np.nan],
        }
    )

    out = morph.classify(frame)

    assert out["morphology"].tolist() == [
        "Elliptical",
        "Spiral",
        "Uncertain",
        co.NoMorphology_LABEL,
        co.NoMorphology_LABEL,
    ]


def test_raw_morphology_stats_use_one_fisher_e_vs_sp_test(monkeypatch):
    captured = {}
    monkeypatch.setattr(morph.report, "append_json", lambda key, value, **_: captured.setdefault(key, value))
    monkeypatch.setattr(morph.gu, "pvalue_latex", lambda value: value)
    monkeypatch.setattr(co, "VERBOSE", False)

    def frame(n_elliptical, n_spiral, n_uncertain=0, n_missing=0):
        return pd.DataFrame(
            {
                "morphology": (
                    ["Elliptical"] * n_elliptical
                    + ["Spiral"] * n_spiral
                    + ["Uncertain"] * n_uncertain
                    + [co.NoMorphology_LABEL] * n_missing
                )
            }
        )

    samples = {
        "CG4" + co.GASUFF: frame(8, 2, n_uncertain=5, n_missing=3),
        "Control4B" + co.GASUFF: frame(2, 8, n_uncertain=40),
        "Control4C" + co.GASUFF: frame(4, 6, n_missing=20),
        "RG4" + co.GASUFF: frame(1, 9, n_uncertain=2, n_missing=1),
        "SDSS": frame(5, 5),
    }

    morph.stats(samples)

    expected = fisher_exact([[8, 2], [2, 8]], alternative="two-sided").pvalue
    assert captured["pval_Control4B_Elliptical_vs_CG_pc"] == expected
    assert "pval_Control4B_Spiral_vs_CG_pc" not in captured
    assert captured["CG4" + co.GASUFF + "_N_GZUsable"] == 10
    assert captured["CG4" + co.GASUFF + "_N_GZExcludedBinary"] == 8
