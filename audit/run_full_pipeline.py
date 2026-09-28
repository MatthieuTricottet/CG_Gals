"""Run the full analysis pipeline regardless of the RENDER_PAPER_ONLY flag.

Usage:
    python audit/run_full_pipeline.py [--rebuild]

``--rebuild`` also rebuilds the processed sample from the raw CSVs
(REBUILD_SAMPLE=True); otherwise the cached ``data/processed_sample.pkl``
is used. The committed ``src/config.py`` is not modified.
"""

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# This runner always targets the checkout that contains it, even if a shell
# inherited an override intended for another copy.
os.environ["CG_GALS_ROOT"] = BASE
sys.path.insert(0, os.path.join(BASE, "src"))

import config as co  # noqa: E402

co.RENDER_PAPER_ONLY = False
co.SHOW = False
if "--rebuild" in sys.argv:
    co.REBUILD_SAMPLE = True

import main  # noqa: E402

main.main()
