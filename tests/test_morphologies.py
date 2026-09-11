import os
import sys

import numpy as np
import pandas as pd

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
