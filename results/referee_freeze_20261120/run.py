"""Recompute the Paper II compactness comparisons using median pair separations."""

import io
from importlib.metadata import version
import json
import os
import pickle
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT / "results/referee_tests_20261117"
MEDIAN = HERE / "median_separation"
OUTPUT = MEDIAN / "tables"
FIGURES = MEDIAN / "figures"
REVISION = "d768251"


def main():
    os.environ["CG_GALS_ROOT"] = str(ROOT)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.dont_write_bytecode = True
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paper2-src-") as work:
        archive = subprocess.check_output(
            ["git", "archive", REVISION, "src", "referee"], cwd=ROOT
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(work, filter="data")
        sys.path.insert(1, str(PRIOR))
        import referee_common  # establishes cache locations and script paths

        # The archived helper adds current checkout paths. Put the requested
        # source revision first before importing any Paper II source modules.
        sys.path[:0] = [str(Path(work) / folder) for folder in ("src", "referee")]
        import data_loader
        import morphologies
        import spectral_indices
        from extended_data import build_galaxy_frame
        from size_data import attach_size_columns
        from astroquery.sdss import SDSS
        from freeze import CENTRE, build, compactness_models, save
        from diagnostics import analyse
        from centerpiece import draw

        assert Path(data_loader.__file__).is_relative_to(work)
        with (ROOT / "data/processed_sample.pkl").open("rb") as handle:
            sample = pickle.load(handle)
        sample = data_loader.correct_group_distance_scales(sample)
        sample = morphologies.classify_all_samples(sample, record_build_counts=False)
        sample = morphologies.add_morphology_fractions_to_groups(sample)
        with patch(
            "socket.socket.connect", side_effect=RuntimeError("cache only")
        ), patch.object(
            SDSS, "query_sql", side_effect=RuntimeError("cache only")
        ), patch.object(
            spectral_indices, "attach_spectral_indices", side_effect=lambda frame: frame
        ):
            frame, _ = attach_size_columns(build_galaxy_frame(sample))
        frame, quartets, window, host_table = build(frame)
        matches, rows = compactness_models(frame, quartets, window)
        contrasts = save("02_compactness_models", rows)
        bins = analyse(frame, quartets, window, host_table, matches)
        draw(frame, quartets, window, bins, contrasts)

        provenance = {
            "git_source": subprocess.check_output(
                ["git", "rev-parse", REVISION], cwd=ROOT, text=True
            ).strip(),
            "software": {
                name: version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "statsmodels",
                    "astropy",
                    "matplotlib",
                    "scikit-learn",
                )
            },
            "compactness": "Median of six separations at mean member redshift (Paper I Planck 2015 parameters)",
            "window_kpc": list(window),
            "centre_kpc": CENTRE,
        }
        (MEDIAN / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
