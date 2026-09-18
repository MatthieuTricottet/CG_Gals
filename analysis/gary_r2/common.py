"""Shared helpers for the gary-r2 read-only diagnostics (Phase 1)."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT = ROOT / "results" / "diagnostics" / "gary_r2"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = ROOT / "output" / "results.json"
RESULTS_BUILD_JSON = ROOT / "output" / "results_build.json"
PKL = ROOT / "data" / "processed_sample.pkl"

SAMPLES = ["CG4", "Control4B", "Control4C", "RG4"]
CONTROLS = ["Control4B", "Control4C", "RG4"]


def load_sample() -> dict:
    with open(PKL, "rb") as fh:
        return pickle.load(fh)


def load_results() -> dict:
    with open(RESULTS_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def load_results_build() -> dict:
    with open(RESULTS_BUILD_JSON, encoding="utf-8") as fh:
        return json.load(fh)


DIAG_JSON = OUT / "diagnostics_gary_r2.json"


def _atomic_dump(path: Path, data, indent=None) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent)
        fh.write("\n")
    os.replace(tmp, path)


def store_diagnostic(key: str, value) -> None:
    """Persist ``value`` under key ``diagnostics_gary_r2.<key>``.

    The canonical copy lives in ``results/diagnostics/gary_r2/diagnostics_gary_r2.json``
    (results.json is rewritten by every pipeline run); the same block is also
    injected into ``output/results.json['diagnostics_gary_r2']`` so the template
    can consume it now.
    """

    from extended_stats import safe_json  # noqa: E402

    block = {}
    if DIAG_JSON.exists():
        with open(DIAG_JSON, encoding="utf-8") as fh:
            block = json.load(fh)
    block[key] = safe_json(value)
    _atomic_dump(DIAG_JSON, block, indent=1)

    with open(RESULTS_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    data["diagnostics_gary_r2"] = block
    _atomic_dump(RESULTS_JSON, data)


def write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = OUT / name
    frame.to_csv(path, index=False)
    return path
