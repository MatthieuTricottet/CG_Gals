import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.luminosity_function import (
    absolute_magnitude_to_log_luminosity,
    add_lf_bgg_flags,
    fit_schechter_mle,
    prepare_lf_frame,
    run_luminosity_function_analysis,
)


def test_absolute_magnitude_to_log_luminosity_orders_brightness():
    assert absolute_magnitude_to_log_luminosity(4.65) == 0.0
    assert absolute_magnitude_to_log_luminosity(-21.0) > absolute_magnitude_to_log_luminosity(
        -19.0
    )


def test_lf_bgg_flags_use_brightest_magnitude_and_stable_ties():
    frame = pd.DataFrame(
        {
            "sample": ["CG4"] * 6,
            "Group": [1, 1, 1, 2, 2, 2],
            "group_uid": ["CG4:1", "CG4:1", "CG4:1", "CG4:2", "CG4:2", "CG4:2"],
            "M_r": [-21.0, -21.0, -20.0, -19.0, -22.0, -21.5],
            "original_row_order": [0, 1, 2, 3, 4, 5],
        }
    )

    flagged = add_lf_bgg_flags(frame)

    assert flagged.groupby("group_uid")["is_bgg_lf"].sum().tolist() == [1, 1]
    assert bool(flagged.loc[0, "is_bgg_lf"]) is True
    assert bool(flagged.loc[1, "is_satellite_lf"]) is True
    assert bool(flagged.loc[4, "is_bgg_lf"]) is True


def _mock_sample(n_groups=5, members=4, include_control4b=False):
    sample = {}
    offsets = {"CG4": 0.0, "RG4": 0.15, "Control4C": -0.1, "Control4B": 0.3}
    for sample_name, offset in offsets.items():
        if sample_name == "Control4B" and not include_control4b:
            continue
        rows = []
        for group in range(n_groups):
            for member in range(members):
                rows.append(
                    {
                        "Group": 1000 + group,
                        "M_r": -22.2 + offset + 0.55 * member + 0.03 * group,
                        "lgm": 10.5 - 0.06 * member,
                    }
                )
        sample[f"{sample_name}_Gals"] = pd.DataFrame(rows)
        sample[f"{sample_name}_Groups"] = pd.DataFrame({"Group": np.arange(n_groups)})
    return sample


def test_prepare_lf_frame_excludes_control4b_by_default():
    sample = _mock_sample(include_control4b=True)

    frame = prepare_lf_frame(sample)

    assert set(frame["sample"]) == {"CG4", "RG4", "Control4C"}
    assert "Control4B" not in set(frame["sample"])


def test_schechter_fit_returns_finite_values_and_skips_small_samples():
    rng = np.random.default_rng(123)
    logL_star = 10.7
    logL_min = 9.6
    values = []
    while len(values) < 120:
        y = rng.gamma(shape=1.4, scale=1.0, size=200)
        draw = logL_star + np.log10(y[y > 0])
        values.extend(draw[draw >= logL_min].tolist())
    values = np.asarray(values[:120])

    fit = fit_schechter_mle(values, logL_min=logL_min)

    assert fit["status"] == "ok"
    assert np.isfinite(fit["alpha"])
    assert np.isfinite(fit["logL_star"])
    assert fit_schechter_mle(values[:7], logL_min=logL_min)["status"] == "skipped"


def test_luminosity_function_end_to_end_creates_figures(tmp_path):
    sample = _mock_sample(n_groups=5, members=4, include_control4b=True)

    result = run_luminosity_function_analysis(
        sample,
        output_dir=tmp_path,
        n_bins=6,
        bootstrap_iterations=5,
        random_state=7,
    )

    assert result["status"] == "ok"
    assert result["samples_excluded"]["Control4B"].startswith("excluded")
    assert result["components"]["bgg"]["fits"]["CG4"]["status"] == "skipped"
    assert set(result["generated_figures"]) == {
        "fig_luminosity_function_all.pdf",
        "fig_luminosity_function_bgg.pdf",
        "fig_luminosity_function_satellites.pdf",
    }
    for filename in result["generated_figures"]:
        assert (Path(tmp_path) / filename).is_file()
    json.dumps(result, allow_nan=False)
