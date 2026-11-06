"""Shared helpers for the bgg-split round (read-only frame reconstruction).

The galaxy frame is rebuilt exactly as ``extended_specialness`` sees it:
processed sample -> the non-rebuild ``main.load_data`` steps ->
``build_galaxy_frame`` -> ``attach_size_columns``.  Network access is blocked
so every external table comes from the committed caches and nothing under
``data/`` can be rewritten.
"""

from __future__ import annotations

import pickle
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT = ROOT / "results" / "diagnostics" / "bgg_split"
OUT.mkdir(parents=True, exist_ok=True)
PKL = ROOT / "data" / "processed_sample.pkl"
SAMPLES = ["CG4", "Control4B", "Control4C", "RG4"]
CONTROLS = ["Control4B", "Control4C", "RG4"]


class NetworkBlocked(RuntimeError):
    pass


def block_network() -> None:
    def _refuse(*_args, **_kwargs):
        raise NetworkBlocked("network access disabled in bgg_split diagnostics")

    socket.socket.connect = _refuse  # type: ignore[assignment]
    socket.create_connection = _refuse  # type: ignore[assignment]


def load_sample() -> dict:
    import data_loader as dl
    import morphologies as morph

    with open(PKL, "rb") as fh:
        sample = pickle.load(fh)
    sample = dl.correct_group_distance_scales(sample)
    sample = morph.classify_all_samples(sample, record_build_counts=False)
    sample = morph.add_morphology_fractions_to_groups(sample)
    return sample


def load_galaxy_frame():
    """Return the frame handed to ``run_descriptive_properties``."""

    block_network()
    from extended_data import build_galaxy_frame
    from size_data import attach_size_columns

    sample = load_sample()
    frame = build_galaxy_frame(sample)
    frame, _ = attach_size_columns(frame)
    return frame, sample
