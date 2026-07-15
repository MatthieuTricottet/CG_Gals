"""Run the manuscript consistency gate as part of the test suite.

Skipped when the paper has not been rendered (fresh clone without outputs);
after any full pipeline run the gate must pass.
"""

import os
import subprocess
import sys

import pytest

BASE = os.path.join(os.path.dirname(__file__), "..")
TEX = os.path.join(BASE, "output", "paper", "paper.tex")


@pytest.mark.skipif(not os.path.exists(TEX), reason="paper not rendered")
def test_consistency_gate_passes():
    proc = subprocess.run(
        [sys.executable, os.path.join(BASE, "audit", "consistency_gate.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
