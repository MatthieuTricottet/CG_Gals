"""Render two-column A&A compactness figures."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.special import expit

from run import FIGURES, OUTPUT, ROOT

# When run.py uses archived analysis modules, the figure still uses the
# article's current plotting styles.
sys.path.insert(0, str(ROOT))
from src.utils.labels_utils import sample_tex_label
from src.utils.sample_styles import LINE_COLOURS, MARKER_EDGE, SAMPLE_STYLES

PLOT_SAMPLES = ("CG4", "Control4C", "RG4")


def draw(frame, q, window, bins, contrasts, mass_ref):
    from freeze import CENTRE, FROZEN, fit, save, sizes

    sat = frame[frame.is_satellite.eq(1)].dropna(subset=["elliptical", "m"])
    trends = {}
    coefficients = []
    for arm in PLOT_SAMPLES:
        model, used, _ = fit(sat[sat["sample"].eq(arm)], "elliptical ~ x + m")
        central = q.loc[q["sample"].eq(arm), "x"].quantile([0.05, 0.95]).to_numpy()
        xs = np.linspace(*central, 120)
        model_points = pd.DataFrame({"x": xs, "m": mass_ref - 10.25})
        trends[arm] = xs, model.predict(model_points)
        ci = model.conf_int()
        coefficients.extend(
            dict(
                sample=arm,
                term=term,
                beta=estimate,
                low=ci.loc[term, 0],
                high=ci.loc[term, 1],
                p=model.pvalues[term],
                central90_x_min=central[0],
                central90_x_max=central[1],
                **sizes(used),
            )
            for term, estimate in model.params.items()
        )
    save("06_figure_trend_coefficients", coefficients)
    curve = pd.read_csv(OUTPUT / "02_compactness_interaction_curve.csv")
    _plot(q, window, bins, contrasts, curve, trends, FIGURES, mass_ref,
          centre_kpc=CENTRE, labelset=FROZEN)


def redraw_saved(freeze_dir: Path, figure_dir: Path, *, legacy: bool = False) -> None:
    """Draw from the frozen estimates and coefficients without refitting."""

    from freeze import LEGACY_MASS_REF

    tables = freeze_dir / "tables"
    q = pd.read_csv(tables / "02_quartet_geometry.csv")
    centre_kpc = None
    if legacy:
        central = q.groupby("sample").mean_Rij.quantile([0.05, 0.95]).unstack()
        window = central.loc["Control4C", 0.05], central.loc["CG4", 0.95]
        contrast_table = "02_frozen_primary_and_compactness.csv"
        mass_ref = LEGACY_MASS_REF
    else:
        provenance = json.loads((freeze_dir / "provenance.json").read_text())
        window = provenance["window_kpc"]
        mass_ref = provenance["mass_ref"]
        centre_kpc = provenance["centre_kpc"]
        contrast_table = "02_compactness_models.csv"
    coefficients = pd.read_csv(tables / "06_figure_trend_coefficients.csv")
    trends = {}
    for arm in PLOT_SAMPLES:
        terms = coefficients.loc[coefficients["sample"].eq(arm)].set_index("term")
        anchor = terms.loc["Intercept"]
        xs = np.linspace(anchor.central90_x_min, anchor.central90_x_max, 120)
        trends[arm] = xs, expit(anchor.beta + terms.loc["x", "beta"] * xs
                                + terms.loc["m", "beta"] * (mass_ref - 10.25))
    bins = pd.read_csv(tables / "06_compactness_bins_audit.csv")
    contrasts = pd.read_csv(tables / contrast_table, low_memory=False)
    curve = pd.read_csv(tables / "02_compactness_interaction_curve.csv")
    _plot(q, window, bins, contrasts, curve, trends, figure_dir, mass_ref,
          centre_kpc=centre_kpc, legacy=legacy)


def _plot(q, window, bins, contrasts, curve, trends, figure_dir, mass_ref, *,
          centre_kpc, labelset="frozen", legacy=False):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    effects = contrasts[
        (contrasts.labelset == labelset) & contrasts.analysis.eq("compactness")
    ]
    all_contrasts = effects[effects.row.eq("contrast") & effects.purity.eq("all")]
    centre = None if legacy else all_contrasts.loc[
        all_contrasts.design.eq("interaction") & all_contrasts.radius.eq(centre_kpc)
    ].squeeze(axis=0)
    marks = [
        effects[
            (effects.design == "full")
            & effects.row.eq("contrast")
            & effects.purity.eq("all")
            & effects.radius.isna()
        ].iloc[0],
        effects[effects.design.eq("compact_match") & effects.row.eq("paired")].iloc[0],
        effects[
            effects.design.eq("compact_match_sigma") & effects.row.eq("paired")
        ].iloc[0],
    ]
    lo, hi = np.log10(window)
    separation = r"\overline{R}_{ij}" if legacy else r"\langle R_{ij}\rangle"
    separation_label = "Mean" if legacy else "Median"
    crimson = SAMPLE_STYLES["CG4"]["colour"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    for raw in (False, True):
        fig = plt.figure(figsize=(7.08, 3.15))
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[1.0, 4.5],
            # Panel (a) keeps its width; panel (b) takes the spare gutter so
            # that the three two-line category labels do not overlap.
            width_ratios=[1, 1.11],
            left=0.098,
            right=0.988,
            bottom=0.185,
            top=0.93,
            hspace=0.04,
            wspace=0.20,
        )
        dist = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[1, 0], sharex=dist)
        # Panel (b): the model curve in kpc on the left, and the three summary
        # contrasts, which have no separation coordinate, on a categorical
        # axis to the right sharing the same vertical scale.
        panel_b = gs[1, 1].subgridspec(1, 2, width_ratios=[0.55, 0.45], wspace=0.06)
        delta = fig.add_subplot(panel_b[0, 0])
        summary = fig.add_subplot(panel_b[0, 1], sharey=delta)
        for area in [dist, ax]:
            area.axvspan(lo, hi, color="#eaecef", zorder=0)
        edges = np.arange(1.4, 3.2, 0.10)
        for arm in ("CG4", "Control4C"):
            quartets = q.loc[q["sample"].eq(arm)]
            colour = LINE_COLOURS[arm]
            # Each sample's histogram is normalised to its own quartet count
            # so that the 62 CG4 quartets remain visible next to the ~700
            # Control4C quartets.
            dist.hist(
                quartets.x,
                bins=edges,
                weights=np.full(len(quartets), 1.0 / len(quartets)),
                histtype="step",
                color=colour,
                linestyle=SAMPLE_STYLES[arm]["linestyle"],
                linewidth=1.0,
            )
            if not legacy:
                median_kpc = quartets.med_Rij.median()
                x_median = np.log10(median_kpc)
                dist.axvline(x_median, color=colour, linestyle=":", linewidth=1.3, zorder=5)
                dist.text(
                    x_median, 1.04, f"{median_kpc:.0f} kpc",
                    transform=dist.get_xaxis_transform(),
                    color="#8A5F00" if arm == "Control4C" else colour,
                    fontsize=7.5, ha="center", va="bottom", clip_on=False,
                )
        dist.set_ylabel("Quartet\nfraction")
        dist.tick_params(axis="x", bottom=False, labelbottom=False)
        dist.spines["bottom"].set_visible(False)
        dist.set_yticks([0.1, 0.2])
        dist.text(0.02, 0.93, "(a)", va="top", transform=dist.transAxes, fontweight="bold")
        handles = []
        for arm in PLOT_SAMPLES:
            style = SAMPLE_STYLES[arm]
            xs, y = trends[arm]
            ax.plot(xs, y, color=LINE_COLOURS[arm], ls=style["linestyle"], lw=1.2)
            b = bins[bins["sample"].eq(arm) & bins.labelset.eq(labelset)]
            key = "raw" if raw else "adjusted"
            b = b[b[key].notna() & b.hosts.ge(3)]
            ax.vlines(
                b.sampled_x,
                b[key + "_low"],
                b[key + "_high"],
                colors=LINE_COLOURS[arm],
                lw=0.8,
            )
            ax.scatter(
                b.sampled_x, b[key], c=style["colour"], marker=style["marker"], s=15,
                edgecolors=MARKER_EDGE, linewidths=0.45, zorder=5 if arm == "CG4" else 4,
            )
            handles.append(Line2D(
                [], [], color=LINE_COLOURS[arm], ls=style["linestyle"], lw=1.2,
                marker=style["marker"], markersize=3.8, markerfacecolor=style["colour"],
                markeredgecolor=MARKER_EDGE, markeredgewidth=0.45, label=sample_tex_label(arm),
            ))
        ax.set(
            xlabel=(rf"$\log_{{10}}({separation}/{{\rm kpc}})$" if legacy
                    else rf"${separation}$ (kpc)"),
            ylabel=(
                "Raw satellite E fraction"
                if raw
                else "Satellite E fraction\n" + rf"at $\log M_\star={mass_ref:.2f}$"
            ),
            xlim=(1.45, 3.16),
            ylim=(0, 1.0),
        )
        if not legacy:
            separation_ticks = [50, window[0], centre_kpc, window[1], 400, 800]
            ax.set_xticks(np.log10(separation_ticks), [f"{kpc:.0f}" for kpc in separation_ticks])
        ax.legend(handles=handles, loc="upper right", frameon=False)
        title_style = dict(fontsize=7, color="#4A4F55", pad=4)
        for area in (delta, summary):
            area.axhline(0, color="#646971", lw=0.7, ls="--")
            area.spines["top"].set_visible(False)
        delta.plot(curve.radius, curve.difference, lw=1.3, color=crimson)
        delta.fill_between(
            curve.radius, curve.low, curve.high, color=crimson, alpha=0.15, lw=0
        )
        if centre is not None:
            delta.errorbar(
                centre.radius, centre.difference,
                yerr=[[centre.difference - centre.low], [centre.high - centre.difference]],
                fmt="o", ms=4.6, mfc="white", mec=crimson, mew=1.1,
                ecolor="#252b31", elinewidth=1.2, capsize=2.5, zorder=5,
            )
        margin = 0.025 * (window[1] - window[0])
        delta.set(
            xlabel=f"{separation_label} projected pair\nseparation ${separation}$ (kpc)",
            ylabel=f"{sample_tex_label('CG4')} − {sample_tex_label('Control4C')} E fraction (pp)",
            xlim=(window[0] - margin, window[1] + margin),
            xticks=[100, 125, 150, 175, 200],
        )
        delta.spines["right"].set_visible(False)
        delta.set_title(f"Model ({window[0]:.0f}–{window[1]:.0f} kpc)", **title_style)
        delta.text(0.04, 0.95, "(b)", va="top", transform=delta.transAxes,
                   fontweight="bold")
        for x, row, symbol in zip(range(3), marks, ("o", "s", "D")):
            summary.vlines(x, row.low, row.high, colors="#252b31", lw=1)
            summary.plot(x, row.difference, marker=symbol, c="#252b31", ms=4.2)
        summary.set(xlim=(-0.55, 2.55), xticks=[0, 1, 2])
        summary.set_xticklabels(
            ["Full\nsample", "Compact.\nmatch", "Match\n" + r"+ $\sigma_v$"],
            fontsize=6.5,
            linespacing=1.05,
        )
        summary.tick_params(axis="x", length=2)
        summary.tick_params(axis="y", left=True, right=True, labelleft=False)
        summary.spines["right"].set_visible(True)
        summary.set_title("Summary", **title_style)
        # Leave headroom above the band for the panel label; both sub-axes
        # share these limits.
        low = min(curve.low.min(), min(row.low for row in marks))
        high = max(curve.high.max(), max(row.high for row in marks))
        span = high - low
        delta.set_ylim(low - 0.05 * span, high + 0.16 * span)
        fname = "fig_centerpiece_compactness" + ("_raw" if raw else "")
        for suffix in ("pdf", "png"):
            fig.savefig(figure_dir / f"{fname}.{suffix}", dpi=300)
        plt.close(fig)
