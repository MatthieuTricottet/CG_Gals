import pickle

import pandas as pd
import pytest
from astropy import units as u
from astropy.cosmology import Planck15

from src import data_loader as dl


def test_load_sdss_falls_back_to_cached_processed_sample(tmp_path, monkeypatch):
    sdss_with_agn = pd.DataFrame({"objid": [1, 2], "p_E": [0.7, 0.2], "p_S": [0.2, 0.7]})
    sdss = pd.DataFrame({"objid": [1], "p_E": [0.7], "p_S": [0.2]})
    cache_path = tmp_path / "processed_sample.pkl"
    with cache_path.open("wb") as file:
        pickle.dump({"SDSS_withAGN": sdss_with_agn, "SDSS": sdss}, file)

    monkeypatch.setattr(dl.co, "DATA_PATH", str(tmp_path) + "/")
    monkeypatch.setattr(dl.co, "PROCESS_SAMPLES", "processed_sample.pkl")
    monkeypatch.setattr(dl.co, "VERBOSE", False)
    monkeypatch.setattr(dl.report, "append_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        dl.SDSS,
        "query_sql",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad SDSS CSV")),
    )

    loaded_with_agn, loaded_sdss = dl.load_SDSS()

    pd.testing.assert_frame_equal(loaded_with_agn, sdss_with_agn)
    pd.testing.assert_frame_equal(loaded_sdss, sdss)


def test_correct_group_distance_scales_uses_angular_diameter_distance():
    groups = pd.DataFrame(
        {
            "Radius_Bary_arcmin": [2.0],
            "z_group": [0.04],
            "Vdisp": [200.0],
            "size_Group_Bary_kpc": [999.0],
            "t_cr": [999.0],
        }
    )
    sample = {"CG4_Groups": groups}
    corrected = dl.correct_group_distance_scales(sample)["CG4_Groups"]
    expected_size = (
        2.0
        * 3.141592653589793
        / (180 * 60)
        * Planck15.angular_diameter_distance(0.04).to_value(u.kpc)
    )
    assert corrected.loc[0, "size_Group_Bary_kpc"] == pytest.approx(expected_size)
    assert corrected.loc[0, "t_cr"] == pytest.approx(0.887 * expected_size / 200.0)
