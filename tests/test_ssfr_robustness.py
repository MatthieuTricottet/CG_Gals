import pandas as pd

from src.ssfr_robustness import fixed_threshold_satellite_check


def test_fixed_threshold_satellite_check_reports_rg4_comparison():
    sample = {
        "CG4_Gals": pd.DataFrame(
            {
                "rank_M": [1, 2, 3, 4],
                "sSFR": [-12.0, -10.5, -11.5, -9.8],
            }
        ),
        "RG4_Gals": pd.DataFrame(
            {
                "rank_M": [1, 2, 3, 4],
                "sSFR": [-12.0, -10.4, -10.3, -10.2],
            }
        ),
        "Control4B_Gals": pd.DataFrame(
            {
                "rank_M": [1, 2, 3, 4],
                "sSFR": [-12.0, -10.2, -11.2, -10.1],
            }
        ),
        "Control4C_Gals": pd.DataFrame(
            {
                "rank_M": [1, 2, 3, 4],
                "sSFR": [-12.0, -10.2, -11.2, -10.1],
            }
        ),
    }

    result = fixed_threshold_satellite_check(sample, threshold=-11.0)

    assert result["status"] == "ok"
    assert result["cg4"]["n_total"] == 3
    assert result["cg4"]["n_starforming"] == 2
    assert result["primary_comparison"]["control"]["n_starforming"] == 3
    assert result["primary_comparison"]["fisher_p_fmt"] != "0.00"
