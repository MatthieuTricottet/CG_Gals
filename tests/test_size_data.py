import numpy as np
import pandas as pd
import pytest
from astropy.cosmology import Planck15

from src import size_data
from src.size_data import (
    _clean_simard_values,
    _ids_to_int64,
    _read_size_cache,
    _write_size_cache,
    attach_size_columns,
)


def synthetic_frame():
    """Six galaxies in two groups with 18-digit identifiers."""

    return pd.DataFrame(
        {
            "objid": [
                1237673808117498264,
                1237673808117498246,
                1237673808117498266,
                1237651249881939975,
                1237651249881940093,
                1237651250439979082,
            ],
            "sample": ["CG4"] * 3 + ["RG4"] * 3,
            "group_uid": ["CG4:7"] * 3 + ["RG4:12"] * 3,
            "z_numeric": [0.040, 0.041, 0.042, 0.030, 0.031, 0.032],
            "z": [0.040, 0.041, 0.042, 0.030, 0.031, 0.032],
        }
    )


def fake_sdss_table(frame):
    table = pd.DataFrame(
        {
            "objid": frame["objid"].astype("int64"),
            "specObjID": [str(10**19 + i) for i in range(len(frame))],
            "dr7objid": pd.array(
                [587739115771789687 + i for i in range(len(frame))], dtype="Int64"
            ),
            "petroR50_r": 3.0,
            "petroR50Err_r": 0.1,
            "petroR90_r": 9.0,
            "petroR90Err_r": 0.5,
            "petroRad_r": 8.0,
            "psfWidth_r": 1.1,
        }
    )
    return table


def fake_simard_table(sdss):
    return pd.DataFrame(
        {
            "dr7objid": sdss["dr7objid"],
            "z": [0.040, 0.041, 0.042, 0.030, 0.031, 0.032],
            "Sp": 2.0,
            "Scale": 0.8,
            "Rhlr": 4.0,
            "Rchl_r": 3.2,
            "e": 0.2,
            "ng": 2.5,
            "e_ng": 0.1,
            "rg2d": 15.0,
            "PpS": 0.3,
        }
    )


def patch_fetches(monkeypatch, sdss, simard):
    monkeypatch.setattr(
        size_data, "fetch_sdss_size_columns", lambda objids: sdss.copy()
    )
    monkeypatch.setattr(size_data, "fetch_simard", lambda dr7objids: simard.copy())


def test_sentinel_and_nonpositive_sizes_become_nan():
    frame = pd.DataFrame(
        {
            "z": [0.03, -99.99],
            "Sp": [2.0, 2.0],
            "Scale": [0.8, -1.0],
            "Rhlr": [-99.99, 4.0],
            "Rchl_r": [3.0, 0.0],
            "e": [0.2, 0.2],
            "ng": [2.0, 2.0],
            "e_ng": [0.1, 0.1],
            "rg2d": [15.0, 15.0],
            "PpS": [0.3, -99.99],
        }
    )
    cleaned = _clean_simard_values(frame.copy())
    assert np.isnan(cleaned.loc[1, "z"])
    assert np.isnan(cleaned.loc[0, "Rhlr"])
    assert np.isnan(cleaned.loc[1, "PpS"])
    assert np.isnan(cleaned.loc[1, "Scale"])
    assert np.isnan(cleaned.loc[1, "Rchl_r"])
    assert cleaned.loc[0, "Rchl_r"] == 3.0


def test_kpc_conversion_matches_direct_astropy(monkeypatch):
    frame = synthetic_frame()
    sdss = fake_sdss_table(frame)
    simard = fake_simard_table(sdss)
    patch_fetches(monkeypatch, sdss, simard)
    enriched, audit = attach_size_columns(frame)

    for _, row in enriched.iterrows():
        arcsec = 3.2 / 0.8
        expected = (
            arcsec * Planck15.kpc_proper_per_arcmin(row["z_numeric"]).value / 60.0
        )
        assert row["Rchl_r_kpc"] == pytest.approx(expected, rel=1e-12)
        expected_petro = (
            3.0 * Planck15.kpc_proper_per_arcmin(row["z_numeric"]).value / 60.0
        )
        assert row["petroR50_kpc"] == pytest.approx(expected_petro, rel=1e-12)
    assert audit["per_sample"]["CG4"]["simard_matched"] == 3
    assert (enriched["size_ok_simard"] == 1).all()
    assert (enriched["size_ok_petro"] == 1).all()


def test_z_mismatch_drops_simard_values(monkeypatch):
    frame = synthetic_frame()
    sdss = fake_sdss_table(frame)
    simard = fake_simard_table(sdss)
    simard.loc[0, "z"] = 0.052  # |dz| = 0.012 > 0.005 for the first CG4 galaxy
    patch_fetches(monkeypatch, sdss, simard)
    enriched, audit = attach_size_columns(frame)

    assert audit["per_sample"]["CG4"]["z_mismatch"] == 1
    assert np.isnan(enriched.loc[0, "Rchl_r_kpc"])
    assert enriched.loc[0, "size_ok_simard"] == 0
    # Petrosian values are untouched by the Simard guard.
    assert enriched.loc[0, "size_ok_petro"] == 1
    assert audit["per_sample"]["CG4"]["simard_matched"] == 2


def test_shred_merge_drops_all_involved_rows(monkeypatch):
    frame = synthetic_frame()
    sdss = fake_sdss_table(frame)
    # Two distinct CG4 galaxies of the same group share one DR7 detection.
    sdss.loc[1, "dr7objid"] = sdss.loc[0, "dr7objid"]
    simard = fake_simard_table(sdss).drop_duplicates("dr7objid")
    patch_fetches(monkeypatch, sdss, simard)
    enriched, audit = attach_size_columns(frame)

    assert audit["per_sample"]["CG4"]["shred_merge"] == 2
    assert np.isnan(enriched.loc[0, "Rchl_r_kpc"])
    assert np.isnan(enriched.loc[1, "Rchl_r_kpc"])
    assert enriched.loc[2, "Rchl_r_kpc"] > 0


def test_pegged_sersic_index_excluded_from_primary(monkeypatch):
    frame = synthetic_frame()
    sdss = fake_sdss_table(frame)
    simard = fake_simard_table(sdss)
    simard.loc[0, "ng"] = 7.95
    patch_fetches(monkeypatch, sdss, simard)
    enriched, audit = attach_size_columns(frame)

    assert audit["per_sample"]["CG4"]["n_pegged"] == 1
    assert enriched.loc[0, "size_ok_simard"] == 0
    assert enriched.loc[0, "size_ok_simard_incl_pegged"] == 1
    assert np.isnan(enriched.loc[0, "log_Rchl_r_kpc"])
    assert np.isfinite(enriched.loc[0, "log_Rchl_r_kpc_incl_pegged"])


def test_string_id_round_trip_through_csv(monkeypatch, tmp_path):
    cache_file = tmp_path / "sdss_size_columns.csv"
    monkeypatch.setattr(size_data.co, "SIZE_COLUMNS_FILE", str(cache_file))
    table = fake_sdss_table(synthetic_frame())
    table.loc[2, "dr7objid"] = pd.NA
    _write_size_cache(table)
    restored = _read_size_cache()

    assert restored["objid"].astype("int64").tolist() == table["objid"].tolist()
    assert restored.loc[0, "dr7objid"] == table.loc[0, "dr7objid"]
    assert pd.isna(restored.loc[2, "dr7objid"])
    # 18-digit values survive exactly (a float64 round-trip would corrupt).
    assert int(restored.loc[0, "objid"]) == 1237673808117498264
    assert restored.loc[0, "specObjID"] == str(10**19)


def test_ids_to_int64_rejects_float_text():
    with pytest.raises(ValueError):
        _ids_to_int64(["5.880070046973955e+17"])
    converted = _ids_to_int64(["587739115771789687", "", None, "nan"])
    assert converted[0] == 587739115771789687
    assert converted[1:].isna().all()
