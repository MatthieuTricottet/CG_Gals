import numpy as np
import pandas as pd

from src import descriptive_trends as dt
from src import ssfr_quality_audit as qa


def test_group_blocked_interval_reports_catalogue_units():
    frame = pd.DataFrame(
        {
            "Group": [1, 1, 2, 2, 3, 3],
            "value": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        }
    )
    estimate, low, high, n_galaxies, n_groups = dt._group_blocked_interval(
        frame,
        value_col="value",
        statistic="median",
        random_state=17,
        n_boot=100,
    )
    assert estimate == 0.5
    assert low <= estimate <= high
    assert n_galaxies == 6
    assert n_groups == 3


def test_mass_bin_uses_contributing_galaxies_median_mass():
    frame = pd.DataFrame(
        {
            "Group": [1, 1, 2, 2, 3, 3],
            "rank_M": [1, 2, 1, 2, 1, 2],
            "lgm": [7.1, 7.2, 7.3, 7.4, 7.5, 9.9],
            "value": [0.0, 0.2, 0.4, 0.6, 0.8, np.nan],
        }
    )
    rows = dt._mass_bin_rows(
        frame,
        sample_name="CG4",
        panel="test",
        value_col="value",
        statistic="median",
        bins=np.array([7.0, 10.0]),
        scope="all",
    )
    assert rows[0]["bin_centre"] == 8.5
    assert rows[0]["mass_location"] == 7.3
    assert rows[0]["n_galaxies"] == 5


def test_quality_summary_uses_exact_spectrum_provenance_categories():
    base = pd.DataFrame(
        {
            "objid": [100, 101, 102, 103],
            "specobjid": [10, 11, 12, 13],
            "sfr": [-9999.0, -0.9, 1.2, -9999.0],
            "lgm": [10.0, 10.1, 10.2, np.nan],
            "sSFR": [np.nan, -11.0, -9.0, np.nan],
            "sSFR_status": ["NosSFR", "Quenched", "Starforming", "NosSFR"],
        }
    )
    sample = {name + "_Gals": base.copy() for name in qa.SAMPLES}
    provenance = pd.DataFrame(
        {
            "source_specobjid": [10, 11, 12, 13],
            "source_objid": [100, 101, 102, 103],
            "extra_specobjid": [10, 11, 12, np.nan],
            "info_specobjid": [10, 11, 12, np.nan],
            "source_sfr": [-9999.0, -0.9, 1.2, np.nan],
            "source_specsfr": [-9999.0, -11.0, -9999.0, np.nan],
            "source_lgm": [10.0, 10.1, 10.2, np.nan],
            "sn_median": [5.0, 20.0, 30.0, np.nan],
            "reliable": [0, 1, 1, np.nan],
            "n_spectra_for_objid": [2, 1, 1, 1],
        }
    )
    traced = qa.trace_catalogue_rows(sample, provenance)
    summary = qa.summarise(traced)
    cg_sentinel = summary.loc[
        (summary["sample"] == "CG4")
        & (summary["sSFR_measurement"] == "galSpecExtra_sfr_-9999")
    ].iloc[0]
    cg_valid = summary.loc[
        (summary["sample"] == "CG4")
        & (summary["sSFR_measurement"] == "valid_derived_sSFR")
    ].iloc[0]
    cg_absent = summary.loc[
        (summary["sample"] == "CG4")
        & (summary["sSFR_measurement"] == "missing_no_galSpecExtra")
    ].iloc[0]
    assert cg_sentinel["n_galaxies"] == 1
    assert cg_sentinel["sn_median_median"] == 5.0
    assert cg_sentinel["reliable_0_fraction"] == 1.0
    assert cg_valid["n_galaxies"] == 2
    assert np.isclose(cg_valid["sn_median_median"], 25.0)
    assert cg_absent["n_galaxies"] == 1
    assert cg_absent["n_with_galSpecInfo"] == 0
