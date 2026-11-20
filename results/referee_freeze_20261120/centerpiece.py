"""Render two-column A&A compactness figures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freeze import FROZEN, MASS_REF, fit, save, sizes
from run import HERE, OUTPUT

COLOURS = {"CG4": "#9d3500", "Control4C": "#006a9f", "RG4": "#00734f"}
MARKERS = {"CG4": "o", "Control4C": "s", "RG4": "^"}


def draw(frame, q, window, bins, contrasts):
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
    curve = pd.read_csv(OUTPUT / "02_compactness_interaction_curve.csv")
    effects = contrasts[
        (contrasts.labelset == FROZEN) & contrasts.analysis.eq("compactness")
    ]
    marks = [
        effects[
            (effects.design == "full")
            & effects.row.eq("contrast")
            & effects.purity.eq("all")
        ].iloc[0],
        effects[effects.design.eq("compact_match") & effects.row.eq("paired")].iloc[0],
        effects[
            effects.design.eq("compact_match_sigma") & effects.row.eq("paired")
        ].iloc[0],
    ]
    sat = frame[frame.is_satellite.eq(1)].dropna(subset=["elliptical", "m"])
    lo, hi = np.log10(window)
    trends = {}
    coefficients = []
    for arm in COLOURS:
        model, used, _ = fit(sat[sat["sample"].eq(arm)], "elliptical ~ x + m")
        central = q.loc[q["sample"].eq(arm), "x"].quantile([0.05, 0.95]).to_numpy()
        xs = np.linspace(*central, 120)
        model_points = pd.DataFrame({"x": xs, "m": MASS_REF - 10.25})
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
    for raw in (False, True):
        fig = plt.figure(figsize=(7.08, 3.15))
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[1.0, 4.5],
            width_ratios=[1, 1],
            left=0.098,
            right=0.988,
            bottom=0.16,
            top=0.93,
            hspace=0.04,
            wspace=0.32,
        )
        dist = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[1, 0], sharex=dist)
        delta = fig.add_subplot(gs[1, 1])
        legend_area = fig.add_subplot(gs[0, 1])
        legend_area.axis("off")
        for area in [dist, ax]:
            area.axvspan(lo, hi, color="#eaecef", zorder=0)
        edges = np.arange(1.4, 3.2, 0.10)
        for arm in ("CG4", "Control4C"):
            dist.hist(
                q[q["sample"].eq(arm)].x,
                bins=edges,
                histtype="step",
                color=COLOURS[arm],
                linewidth=1.0,
            )
        dist.set_ylabel("Quartets")
        dist.tick_params(axis="x", bottom=False, labelbottom=False)
        dist.spines["bottom"].set_visible(False)
        dist.set_yticks([50, 100])
        dist.text(0.02, 0.88, "(a)", transform=dist.transAxes, fontweight="bold")
        for arm in COLOURS:
            xs, y = trends[arm]
            ax.plot(xs, y, color=COLOURS[arm], lw=1.2, label=arm)
            b = bins[bins["sample"].eq(arm) & bins.labelset.eq(FROZEN)]
            key = "raw" if raw else "adjusted"
            b = b[b[key].notna() & b.hosts.ge(3)]
            ax.vlines(
                b.sampled_x,
                b[key + "_low"],
                b[key + "_high"],
                colors=COLOURS[arm],
                lw=0.8,
            )
            ax.scatter(
                b.sampled_x, b[key], c=COLOURS[arm], marker=MARKERS[arm], s=13, zorder=4
            )
        ax.set(
            xlabel=r"$\log_{10}(\overline{R}_{ij}/{\rm kpc})$",
            ylabel=(
                "Raw satellite E fraction"
                if raw
                else "Satellite E fraction\n" + r"at $\log M_\star=10.35$"
            ),
            xlim=(1.45, 3.16),
            ylim=(0, 1.0),
        )
        ax.legend(loc="upper right", frameon=False)
        delta.axhline(0, color="#646971", lw=0.7, ls="--")
        delta.plot(curve.radius, curve.difference, lw=1.3, color=COLOURS["CG4"])
        delta.fill_between(
            curve.radius, curve.low, curve.high, color=COLOURS["CG4"], alpha=0.15, lw=0
        )
        delta.set(
            xlabel="Mean projected pair separation (kpc)",
            ylabel="CG4 − Control4C E fraction (pp)",
            xlim=(92, 276),
            xticks=[100, 125, 150, 175, 200],
        )
        delta.text(0.025, 0.92, "(b)", transform=delta.transAxes, fontweight="bold")
        delta.axvline(221, color="#c9ced2", lw=0.6)
        for i, (x, row, symbol, label) in enumerate(
            zip(
                (229, 247, 265),
                marks,
                ("o", "s", "D"),
                ("Full sample", "Compactness match", r"Match + $\sigma_v$ caliper"),
            )
        ):
            delta.vlines(x, row.low, row.high, colors="#252b31", lw=1)
            delta.plot(x, row.difference, marker=symbol, c="#252b31", ms=4.2)
            column, line = (i % 2) * 0.5, i // 2
            legend_area.plot(
                [column + 0.015],
                [0.72 - 0.34 * line],
                marker=symbol,
                color="#252b31",
                ms=4,
                transform=legend_area.transAxes,
            )
            legend_area.text(
                column + 0.05,
                0.72 - 0.34 * line,
                label,
                transform=legend_area.transAxes,
                va="center",
                fontsize=7,
            )
        fname = "fig_centerpiece_compactness" + ("_raw" if raw else "")
        for suffix in ("pdf", "png"):
            fig.savefig(HERE / "figures" / f"{fname}.{suffix}", dpi=300)
        plt.close(fig)
