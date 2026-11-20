"""Rebuild the Paper II freeze from versioned code and local catalogue caches."""

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
OUTPUT = HERE / "tables"
REVISION = "d768251"


def main():
    os.environ["CG_GALS_ROOT"] = str(ROOT)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.dont_write_bytecode = True
    OUTPUT.mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
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
        import pandas as pd
        import data_loader
        import morphologies
        import spectral_indices
        from extended_data import build_galaxy_frame
        from size_data import attach_size_columns
        from astroquery.sdss import SDSS
        from freeze import build, baseline
        from diagnostics import analyse
        from centerpiece import draw

        assert Path(data_loader.__file__).is_relative_to(work)
        with (ROOT / "data/processed_sample.pkl").open("rb") as handle:
            sample = pickle.load(handle)
        sample = data_loader.correct_group_distance_scales(sample)
        sample = morphologies.classify_all_samples(sample, record_build_counts=False)
        sample = morphologies.add_morphology_fractions_to_groups(sample)
        indices = pd.read_csv(ROOT / "data/galspecindx_dr12.csv")
        with patch(
            "socket.socket.connect", side_effect=RuntimeError("cache only")
        ), patch.object(
            SDSS, "query_sql", side_effect=RuntimeError("cache only")
        ), patch.object(
            spectral_indices, "fetch_spectral_indices", return_value=indices
        ):
            frame, _ = attach_size_columns(build_galaxy_frame(sample))
        frame, quartets, window, host_table = build(frame)
        matches, contrasts = baseline(frame, quartets, window)
        bins = analyse(frame, quartets, window, host_table, matches)
        draw(frame, quartets, window, bins, contrasts)
        import hashlib

        inputs = [
            "data/processed_sample.pkl",
            "data/sdss_size_columns.csv",
            "data/simard2011_subset.csv",
            "data/galspecindx_dr12.csv",
            "data/galspecline_dr16.csv",
            "data/PC_Groups.csv",
            "data/CG4_Groups.csv",
            "data/SDSS(L) group.dat",
            "output/results_build.json",
            "results/referee_tests_20261117/inputs/gz1_paper1_crossmatch.csv",
            "results/referee_tests_20261117/tables/hmcg_control_members.csv",
            "results/referee_tests_20261117/tables/hmcg_isolation_quartets.csv",
            "results/referee_tests_20261117/tables/compact_morphology_bins.csv",
        ]
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
            "inputs_sha256": {
                p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in inputs
            },
        }
        (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
