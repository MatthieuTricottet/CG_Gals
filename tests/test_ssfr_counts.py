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


def test_starforming_table_uses_quenched_as_failures():
    # Only classified galaxies enter the Fisher table: Total counts the
    # measured (Quenched + Starforming) rows, never the missing ones.
    left = {"Quenched": 3, "Starforming": 3, "Total": 6, "NosSFR": 2}
    right = {"Quenched": 9, "Starforming": 6, "Total": 15, "NosSFR": 1}

    assert sSFR._starforming_vs_non_table(left, right) == [[3, 3], [6, 9]]


def test_status_counts_reports_missing_separately():
    df = _galaxies(["Quenched", "Starforming", "NosSFR", "Starforming"])
    counts = sSFR._status_counts(df)
    assert counts == {
        "Quenched": 1,
        "Starforming": 2,
        "Total": 3,
        "NosSFR": 1,
    }


def test_validate_ssfr_table_counts_rejects_missing_morphology_class(monkeypatch):
    sample = {
        "CG4_Gals": _galaxies(["Quenched", "NosSFR", "Starforming"]),
        "Control4B_Gals": _galaxies(["Quenched", "NosSFR", "Starforming"]),
        "Control4C_Gals": _galaxies(
            ["Quenched", "NosSFR", "Starforming"], ["Elliptical", "bad", "Spiral"]
        ),
        "RG4_Gals": _galaxies(["Quenched", "NosSFR", "Starforming"]),
    }

    monkeypatch.setattr(
        sSFR.co,
        "SAMPLE",
        {"CG4": "CG4", "Control4B": "Control4B", "Control4C": "Control4C", "RG4": "RG4"},
    )

    with pytest.raises(AssertionError, match="morphology total"):
        sSFR.validate_ssfr_table_counts(sample)


def test_validate_ssfr_table_counts_accepts_consistent_sample(monkeypatch):
    sample = {
        name + "_Gals": _galaxies(["Quenched", "NosSFR", "Starforming"])
        for name in ["CG4", "Control4B", "Control4C", "RG4"]
    }
    monkeypatch.setattr(
        sSFR.co,
        "SAMPLE",
        {"CG4": "CG4", "Control4B": "Control4B", "Control4C": "Control4C", "RG4": "RG4"},
    )
    audit = sSFR.validate_ssfr_table_counts(sample)
    assert audit["CG4"]["all"]["Total"] == 2
    assert audit["CG4"]["all"]["NosSFR"] == 1
