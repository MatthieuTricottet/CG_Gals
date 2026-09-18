"""Create an exploratory Figure 2 variant with Gary Mamon's error-bar helper.

This script is deliberately separate from ``src/descriptive_trends.py`` and
does not write into ``output/paper``.  It imports ``binomialerrorplot`` and
``BinomialError`` directly from local snapshots of Gary Mamon's public
``python-codes`` repository, supplied on the command line.

The expanded layout retains the all-galaxy E/S morphology panels, adds
satellite-only and BGG-only E/S panels, and retains the satellite/BGG
quenched-fraction panels from the paper figure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cg-gals-mpl")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "output" / "exploratory" / "figure2_binomial"
DEFAULT_SAMPLE_PATH = ROOT / "data" / "processed_sample.pkl"

SAMPLES = ("CG4", "Control4B", "Control4C", "RG4")
MORPHOLOGY_MASS_BINS = np.array([7.0, 9.5, 10.0, 10.5, 11.0, 12.5])
QUENCHED_MASS_BINS = np.array([7.0, 10.0, 10.5, 11.0, 12.5])
MIN_GALAXIES_TO_PLOT = 5
MIN_GROUPS_TO_PLOT = 3

SAMPLE_STYLES = {
    "CG4": {"colour": "#000000", "marker": "o"},
    "Control4B": {"colour": "#0072B2", "marker": "s"},
    "Control4C": {"colour": "#D55E00", "marker": "^"},
    "RG4": {"colour": "#009E73", "marker": "D"},
}
SAMPLE_LABELS = {
    "CG4": r"CG$_4$",
    "Control4B": r"Control$_{4B}$",
    "Control4C": r"Control$_{4C}$",
    "RG4": r"RG$_4$",
}

UPSTREAM_REPOSITORY = "https://gitlab.com/gmamon/python-codes"
UPSTREAM_BRANCH = "gary"
UPSTREAM_COMMIT = "88fea77dbd6430a67c262ffa9ac5ab369ab351f3"
GRAPHUTILS_URL = (
    f"{UPSTREAM_REPOSITORY}/-/blob/{UPSTREAM_COMMIT}/graphutils.py"
)
MATHUTILS_URL = f"{UPSTREAM_REPOSITORY}/-/blob/{UPSTREAM_COMMIT}/mathutils.py"


@dataclass(frozen=True)
class Panel:
    key: str
    title: str
    population: str
    outcome: str
    success: str
    bins: np.ndarray


PANELS = (
    Panel("morph_e_all", "(a) GZ1 E class: all galaxies", "all", "morphology", "Elliptical", MORPHOLOGY_MASS_BINS),
    Panel("morph_s_all", "(b) GZ1 S class: all galaxies", "all", "morphology", "Spiral", MORPHOLOGY_MASS_BINS),
    Panel("morph_e_sat", "(c) GZ1 E class: satellites", "satellites", "morphology", "Elliptical", MORPHOLOGY_MASS_BINS),
    Panel("morph_s_sat", "(d) GZ1 S class: satellites", "satellites", "morphology", "Spiral", MORPHOLOGY_MASS_BINS),
    Panel("morph_e_bgg", "(e) GZ1 E class: BGGs", "bggs", "morphology", "Elliptical", MORPHOLOGY_MASS_BINS),
    Panel("morph_s_bgg", "(f) GZ1 S class: BGGs", "bggs", "morphology", "Spiral", MORPHOLOGY_MASS_BINS),
    Panel("quenched_sat", "(g) Quenched fraction: satellites", "satellites", "ssfr", "Quenched", QUENCHED_MASS_BINS),
    Panel("quenched_bgg", "(h) Quenched fraction: BGGs", "bggs", "ssfr", "Quenched", QUENCHED_MASS_BINS),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_upstream_plotting(graphutils_path: Path, mathutils_path: Path):
    """Load the exact upstream functions while stubbing unused dependencies."""

    for path in (graphutils_path, mathutils_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    # graphutils imports these optional modules at module load, but
    # binomialerrorplot does not use them.
    for name in ("mpltern", "langutils", "densutils"):
        sys.modules.setdefault(name, types.ModuleType(name))

    original_rc = mpl.rcParams.copy()
    mathutils = _module_from_path("mathutils", mathutils_path)
    graphutils = _module_from_path("graphutils", graphutils_path)
    # graphutils changes global fonts and enables TeX on import.  Restore the
    # project's plotting state; this does not alter binomialerrorplot itself.
    mpl.rcParams.update(original_rc)
    return graphutils.binomialerrorplot, mathutils.BinomialError


def _restrict_population(frame: pd.DataFrame, population: str) -> pd.DataFrame:
    rank = pd.to_numeric(frame["rank_M"], errors="coerce")
    if population == "satellites":
        return frame.loc[rank.gt(1)].copy()
    if population == "bggs":
        return frame.loc[rank.eq(1)].copy()
    if population == "all":
        return frame.copy()
    raise ValueError(f"Unknown population: {population}")


def compute_counts(sample: dict[str, pd.DataFrame], binomial_error) -> pd.DataFrame:
    """Compute n/N and the exact plotting coordinates used by the helper."""

    rows: list[dict[str, object]] = []
    for panel in PANELS:
        for sample_name in SAMPLES:
            frame = _restrict_population(sample[f"{sample_name}_Gals"], panel.population)
            frame["lgm"] = pd.to_numeric(frame["lgm"], errors="coerce")
            frame["mass_bin"] = pd.cut(
                frame["lgm"], bins=panel.bins, right=False, include_lowest=True
            )
            for interval in frame["mass_bin"].cat.categories:
                current = frame.loc[frame["mass_bin"] == interval].copy()
                if panel.outcome == "morphology":
                    contributors = current.loc[
                        current["morphology"].isin(("Elliptical", "Spiral"))
                    ]
                    success = contributors["morphology"].eq(panel.success)
                else:
                    contributors = current.loc[
                        current["sSFR_status"].isin(("Quenched", "Starforming"))
                    ]
                    success = contributors["sSFR_status"].eq(panel.success)

                total = int(len(contributors))
                successes = int(success.sum())
                n_groups = int(contributors["Group"].nunique())
                mass_location = (
                    float(contributors["lgm"].median()) if total else np.nan
                )
                if total:
                    plotted_p, plotted_error = binomial_error(
                        np.array([total], dtype=int),
                        np.array([successes], dtype=int),
                    )
                    plot_estimate = float(plotted_p[0])
                    plot_error = float(plotted_error[0])
                else:
                    plot_estimate = np.nan
                    plot_error = np.nan
                rows.append(
                    {
                        "panel": panel.key,
                        "title": panel.title,
                        "sample": sample_name,
                        "population": panel.population,
                        "outcome": panel.outcome,
                        "success_class": panel.success,
                        "bin_left": float(interval.left),
                        "bin_right": float(interval.right),
                        "mass_location": mass_location,
                        "n_success": successes,
                        "n_total": total,
                        "n_groups": n_groups,
                        "raw_fraction": successes / total if total else np.nan,
                        "plot_estimate": plot_estimate,
                        "plot_error": plot_error,
                        "limit_type": (
                            "upper" if total and successes == 0 else
                            "lower" if total and successes == total else
                            "two_sided" if total else "empty"
                        ),
                        "displayed": bool(
                            total >= MIN_GALAXIES_TO_PLOT
                            and n_groups >= MIN_GROUPS_TO_PLOT
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows)


def plot_variant(counts: pd.DataFrame, binomialerrorplot, output_base: Path) -> None:
    """Draw the expanded 4x2 figure with upstream binomialerrorplot calls."""

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(7.35, 9.35),
        sharex=True,
        sharey=True,
    )
    displayed = counts.loc[counts["displayed"]]
    for panel_index, (ax, panel) in enumerate(zip(axes.flat, PANELS)):
        subset = displayed.loc[displayed["panel"] == panel.key]
        for sample_name in SAMPLES:
            current = subset.loc[subset["sample"] == sample_name].sort_values(
                "mass_location"
            )
            if current.empty:
                continue
            style = SAMPLE_STYLES[sample_name]
            binomialerrorplot(
                current["mass_location"].to_numpy(dtype=float),
                current["n_total"].to_numpy(dtype=int),
                current["n_success"].to_numpy(dtype=int),
                color=style["colour"],
                marker=style["marker"],
                markersize=4.3,
                mec=style["colour"],
                mew=1.0,
                capsize=2.0,
                capthick=1.0,
                ecolor=style["colour"],
                label=SAMPLE_LABELS[sample_name] if panel_index == 0 else None,
                ax=ax,
                zorder=3,
            )
        ax.set_title(panel.title, fontsize=9)
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for row, label in enumerate(
        ("Class fraction", "Class fraction", "Class fraction", "Quenched fraction")
    ):
        axes[row, 0].set_ylabel(label, fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.text(
        0.5,
        0.006,
        (
            "Gary Mamon binomialerrorplot: interior points use 1σ binomial "
            "errors; arrows mark 95% Wilson edge limits."
        ),
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.972), h_pad=1.0, w_pad=0.9)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--graphutils", type=Path, required=True)
    parser.add_argument("--mathutils", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    binomialerrorplot, binomial_error = load_upstream_plotting(
        args.graphutils, args.mathutils
    )
    with args.sample.open("rb") as stream:
        sample = pickle.load(stream)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = compute_counts(sample, binomial_error)
    counts_path = args.output_dir / "fig2_binomial_satellite_bgg_counts.csv"
    counts.to_csv(counts_path, index=False)
    output_base = args.output_dir / "fig2_binomial_satellite_bgg"
    plot_variant(counts, binomialerrorplot, output_base)

    provenance = {
        "purpose": "Exploratory second version of Figure 2; not used by the paper build",
        "input_sample": str(args.sample.resolve()),
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_branch": UPSTREAM_BRANCH,
        "upstream_commit": UPSTREAM_COMMIT,
        "graphutils_url": GRAPHUTILS_URL,
        "mathutils_url": MATHUTILS_URL,
        "graphutils_sha256": _sha256(args.graphutils),
        "mathutils_sha256": _sha256(args.mathutils),
        "binomial_method": (
            "Unmodified upstream binomialerrorplot calling upstream BinomialError: "
            "Wald 1-sigma errors for 0<n<N; z=1.65 Wilson-centred one-sided "
            "limits for n=0 or n=N"
        ),
        "minimum_galaxies_displayed": MIN_GALAXIES_TO_PLOT,
        "minimum_groups_displayed": MIN_GROUPS_TO_PLOT,
        "outputs": {
            "pdf": str(output_base.with_suffix(".pdf").resolve()),
            "png": str(output_base.with_suffix(".png").resolve()),
            "counts": str(counts_path.resolve()),
        },
    }
    with (args.output_dir / "provenance.json").open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
