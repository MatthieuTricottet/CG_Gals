import pandas as pd
import pytest

from src import sSFR


def _galaxies(statuses, morphologies=None):
    if morphologies is None:
        morphologies = ["Elliptical"] * len(statuses)
    return pd.DataFrame(
        {
            "rank_M": range(1, len(statuses) + 1),
            "sSFR_status": statuses,
            "morphology": morphologies,
        }
    )


def test_starforming_table_uses_non_starforming_failures():
    left = {"Quenched": 1, "Passive": 2, "Starforming": 3, "Total": 6}
    right = {"Quenched": 4, "Passive": 5, "Starforming": 6, "Total": 15}

    assert sSFR._starforming_vs_non_table(left, right) == [[3, 3], [6, 9]]


def test_validate_ssfr_table_counts_rejects_missing_morphology_class(monkeypatch):
    sample = {
        "CG4_Gals": _galaxies(["Quenched", "Passive", "Starforming"]),
        "Control4B_Gals": _galaxies(["Quenched", "Passive", "Starforming"]),
        "Control4C_Gals": _galaxies(
            ["Quenched", "Passive", "Starforming"], ["Elliptical", "bad", "Spiral"]
        ),
        "RG4_Gals": _galaxies(["Quenched", "Passive", "Starforming"]),
    }

    monkeypatch.setattr(
        sSFR.co,
        "SAMPLE",
        {"CG4": "CG4", "Control4B": "Control4B", "Control4C": "Control4C", "RG4": "RG4"},
    )

    with pytest.raises(AssertionError, match="morphology total"):
        sSFR.validate_ssfr_table_counts(sample)
