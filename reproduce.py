"""Reproduce every number, figure and the PDF of Paper II from the repository.

Usage (from the repository root, Python 3.13 environment of requirements.lock):

    python reproduce.py            # full run: analyses, value files, paper
    python reproduce.py --render   # re-render the paper from existing outputs

The steps, in dependency order, are the only provenance path of the
manuscript's numbers:

1. ``src/main.py`` (full pipeline): processed sample -> ``output/results.json``
   and every figure;
2. ``src/paper_additions.py``: ``output/paper_additions.json`` and the LaTeX
   macros ``output/paper/additions_macros.tex``;
3. the referee-round value files still cited by the manuscript
   (``referee/T3``, ``T4``, ``T5``, ``T7``, ``T9``, ``T10`` ->
   ``referee/values/*.json``);
4. the read-only diagnostics D1-D7 (``analysis/gary_r2`` ->
   ``results/diagnostics/gary_r2/diagnostics_gary_r2.json``);
5. rendering and compilation of ``output/paper/paper.pdf`` and the online
   supplement (pdflatex and bibtex required).

All inputs are tracked in ``data/`` except the SDSS reference query, which is
cached in ``data/processed_sample.pkl`` (see README).  No step needs the
network once the caches exist.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
ANALYSIS_STEPS = [
    ["audit/run_full_pipeline.py"],
    ["src/paper_additions.py"],
    ["referee/T3_sigma_v.py"],
    ["referee/T4_tidal_support.py"],
    ["referee/T5_missing_ssfr_bounds.py"],
    ["referee/T7_cosmology_audit.py"],
    ["referee/T9_compactness_host_bgg.py"],
    ["referee/T10_ds18_morphology.py"],
    ["analysis/gary_r2/d1_missing_ssfr.py"],
    ["analysis/gary_r2/d2_sfms_sign.py"],
    ["analysis/gary_r2/d3_zheng_shen_class.py"],
    ["analysis/gary_r2/d4_ds18_gz1_mapping.py"],
    ["analysis/gary_r2/d5_sigma_v_group_match.py"],
    ["analysis/gary_r2/d6_d7_holm_fig2.py"],
]
RENDER_STEP = ["src/main.py"]  # RENDER_PAPER_ONLY = True in src/config.py


def run(step: list[str]) -> None:
    start = time.time()
    env = dict(os.environ, MPLBACKEND="Agg")
    print(f"--> {' '.join(step)}", flush=True)
    result = subprocess.run([PYTHON, *step], cwd=ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(f"step failed ({result.returncode}): {' '.join(step)}")
    print(f"    done in {time.time() - start:.0f} s", flush=True)


def main() -> None:
    if sys.version_info[:2] != (3, 13):
        print(f"warning: tested with Python 3.13, running {sys.version.split()[0]}")
    steps = [] if "--render" in sys.argv else list(ANALYSIS_STEPS)
    for step in [*steps, RENDER_STEP]:
        run(step)


if __name__ == "__main__":
    main()
