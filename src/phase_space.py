"""Compatibility wrapper for the phase-space segregation analysis."""

from __future__ import annotations

try:
    from phase_space_segregation import (
        cluster_bootstrap_fraction,
        compute_velocity_offsets,
        prepare_phase_space_satellite_sample,
        run_phase_space_analysis,
        run_phase_space_segregation_analysis,
        summarize_binned_fractions,
    )
except ModuleNotFoundError:  # pragma: no cover
    from .phase_space_segregation import (
        cluster_bootstrap_fraction,
        compute_velocity_offsets,
        prepare_phase_space_satellite_sample,
        run_phase_space_analysis,
        run_phase_space_segregation_analysis,
        summarize_binned_fractions,
    )

__all__ = [
    "cluster_bootstrap_fraction",
    "compute_velocity_offsets",
    "prepare_phase_space_satellite_sample",
    "run_phase_space_analysis",
    "run_phase_space_segregation_analysis",
    "summarize_binned_fractions",
]
