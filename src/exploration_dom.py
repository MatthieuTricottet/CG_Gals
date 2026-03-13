from __future__ import annotations

import os
from itertools import combinations

import matplotlib
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
from scipy.stats import brunnermunzel, ks_2samp, mannwhitneyu, ranksums, spearmanr
from statsmodels.stats.multitest import multipletests

try:
    import config as co
    import generate_report as report
    from utils import graphics_utils as gu
    from utils import labels_utils as lu
    from utils import pandas_utils as pu
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from . import generate_report as report
    from .utils import graphics_utils as gu
    from .utils import labels_utils as lu
    from .utils import pandas_utils as pu


DOMINATION_QUANTITIES = [
    "size_Group_Bary_kpc",
    "Offset_Bary",
    "Vdisp",
    "Voffset",
    "Lum_group",
]


def compute_significant_quantities(
    sample: dict[str, pd.DataFrame],
    quantities: list[str] | None = None,
    p_limit: float | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Identify which group quantities show a domination split in at least one sample."""

    quantities = quantities or DOMINATION_QUANTITIES
    p_limit = co.P_LIMIT if p_limit is None else p_limit

    rows: list[dict[str, object]] = []
    significant: list[str] = []

    for quantity in quantities:
        is_significant = False
        for cat in co.SAMPLE.keys():
            grp_key = cat + co.GRSUFF
            if grp_key not in sample or quantity not in sample[grp_key].columns:
                continue

            df = sample[grp_key]
            dom = pd.to_numeric(df.loc[df["is_dominated"], quantity], errors="coerce").dropna()
            nondom = pd.to_numeric(df.loc[~df["is_dominated"], quantity], errors="coerce").dropna()

            statistic = np.nan
            p_value = np.nan
            alternative = np.nan
            if len(dom) and len(nondom):
                alternative = "less" if dom.median() < nondom.median() else "greater"
                statistic, p_value = ranksums(dom, nondom, alternative=alternative)
                is_significant = is_significant or (p_value < p_limit)

            rows.append(
                {
                    "Sample": cat,
                    "Quantity": quantity,
                    "n_dom": int(len(dom)),
                    "n_nondom": int(len(nondom)),
                    "median_dom": float(dom.median()) if len(dom) else np.nan,
                    "median_nondom": float(nondom.median()) if len(nondom) else np.nan,
                    "alternative": alternative,
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )

        if is_significant:
            significant.append(quantity)

    return significant, pd.DataFrame(rows)


def smart_log_formatter(x: float, _pos: int) -> str:
    """Format values that are plotted in log10 space back into readable linear labels."""

    val = 10 ** x
    if 1e-2 <= val <= 1e3:
        if val >= 100:
            return f"{val:.0f}"
        if val >= 10:
            return f"{val:.1f}".rstrip("0").rstrip(".")
        return f"{val:.2f}".rstrip("0").rstrip(".")

    exp = int(np.floor(np.log10(val)))
    mant = val / 10 ** exp
    if np.isclose(mant, 1.0):
        return rf"$10^{{{exp}}}$"
    if np.isclose(mant, np.round(mant)):
        return rf"${int(np.round(mant))}\times10^{{{exp}}}$"
    return rf"${mant:.1f}\times10^{{{exp}}}$"


def log_value_formatter(x: float, _pos: int) -> str:
    """Show the raw log10 values when the last column is explicitly log-labelled."""

    if np.isfinite(x):
        return f"{x:.2f}".rstrip("0").rstrip(".")
    return ""


def _ensure_axes_2d(axes, nrows: int, ncols: int) -> np.ndarray:
    if nrows == 1 and ncols == 1:
        return np.array([[axes]])
    if nrows == 1:
        return axes[np.newaxis, :]
    if ncols == 1:
        return axes[:, np.newaxis]
    return axes


def plot_domination_distributions(
    sample: dict[str, pd.DataFrame],
    quantities: list[str],
    output_path: str,
) -> str | None:
    """Reproduce the notebook's dominated vs non-dominated histogram grid."""

    if not quantities:
        return None

    sns.set_theme(style="ticks", context="paper")

    dom_color = "#4C72B0"
    nondom_color = "#DD8452"
    nb_bin = {"CG4": 10, "Control4B": 25, "Control4C": 25, "RG4": 10}
    custom_major_ticks = {"Lum_group": [3e10, 1e11, 3e11]}

    nrows = len(co.SAMPLE)
    ncols = len(quantities)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 4 * nrows),
        gridspec_kw={"hspace": 0, "wspace": 0},
    )
    axes = _ensure_axes_2d(axes, nrows, ncols)

    col_xmin = np.full(ncols, np.inf)
    col_xmax = np.full(ncols, -np.inf)
    row_ymax = np.zeros(nrows)

    for col, quantity in enumerate(quantities):
        for row, cat in enumerate(co.SAMPLE.keys()):
            ax = axes[row, col]
            grp_key = cat + co.GRSUFF
            if grp_key not in sample or quantity not in sample[grp_key].columns:
                ax.axis("off")
                continue

            df = sample[grp_key]
            dom_vals = pd.to_numeric(df.loc[df["is_dominated"], quantity], errors="coerce").to_numpy()
            nondom_vals = pd.to_numeric(df.loc[~df["is_dominated"], quantity], errors="coerce").to_numpy()
            dom_vals = dom_vals[np.isfinite(dom_vals) & (dom_vals > 0)]
            nondom_vals = nondom_vals[np.isfinite(nondom_vals) & (nondom_vals > 0)]

            p_value = np.nan
            if dom_vals.size and nondom_vals.size:
                alternative = "less" if np.median(dom_vals) < np.median(nondom_vals) else "greater"
                _, p_value = ranksums(dom_vals, nondom_vals, alternative=alternative)

            vals = pd.to_numeric(df[quantity], errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size == 0:
                ax.axis("off")
                continue

            bins = np.linspace(np.log10(vals.min()), np.log10(vals.max()), nb_bin[cat])
            col_xmin[col] = min(col_xmin[col], bins.min())
            col_xmax[col] = max(col_xmax[col], bins.max())

            if dom_vals.size:
                ax.hist(
                    np.log10(dom_vals),
                    bins=bins,
                    density=True,
                    color=dom_color,
                    alpha=0.5,
                    label="Dominated",
                )
                ax.axvline(np.log10(np.median(dom_vals)), color=dom_color, linestyle="--", linewidth=2.0)

            if nondom_vals.size:
                ax.hist(
                    np.log10(nondom_vals),
                    bins=bins,
                    density=True,
                    color=nondom_color,
                    alpha=0.5,
                    label="Non-Dominated",
                )
                ax.axvline(
                    np.log10(np.median(nondom_vals)),
                    color=nondom_color,
                    linestyle="--",
                    linewidth=2.0,
                )

            row_ymax[row] = max(row_ymax[row], ax.get_ylim()[1])

            if col == 0:
                ax.set_ylabel(f"{cat} Groups", fontsize=18)

            if row == nrows - 1:
                label_key = "logLum_group" if quantity == "Lum_group" and col == ncols - 1 else quantity
                ax.set_xlabel(lu.formatted_label(label_key), fontsize=16)

            if np.isfinite(p_value):
                ptxt = f"p={gu.tex_form(p_value)}"
                pcol = "red" if p_value <= co.P_LIMIT else "black"
            else:
                ptxt = "p=NA"
                pcol = "black"

            ax.text(
                0.05,
                0.97,
                ptxt,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=14,
                color=pcol,
            )

            if col == ncols - 1 and row == 0:
                ax.legend(fontsize=13)

    for col in range(ncols):
        if not np.isfinite(col_xmin[col]) or not np.isfinite(col_xmax[col]):
            continue
        for row in range(nrows):
            axes[row, col].set_xlim(col_xmin[col], col_xmax[col])

    for row in range(nrows):
        ymax = row_ymax[row] if row_ymax[row] > 0 else 1.0
        for col in range(ncols):
            axes[row, col].set_ylim(0, ymax)

    for col, quantity in enumerate(quantities):
        xmin = col_xmin[col]
        xmax = col_xmax[col]
        if not np.isfinite(xmin) or not np.isfinite(xmax):
            continue

        if quantity in custom_major_ticks:
            ticks = np.log10(np.array(custom_major_ticks[quantity], dtype=float))
        else:
            span = xmax - xmin
            if ncols >= 5:
                ntarget = 4
            elif ncols == 4:
                ntarget = 5
            else:
                ntarget = 6

            if span <= 1.5:
                ticks = np.linspace(xmin, xmax, ntarget)
            else:
                dmin = int(np.floor(xmin))
                dmax = int(np.ceil(xmax))
                decades = np.arange(dmin, dmax + 1, 1.0)
                if len(decades) <= ntarget:
                    mids = decades[:-1] + np.log10(3)
                    ticks = np.sort(np.concatenate([decades, mids]))
                else:
                    ticks = decades
                if len(ticks) > ntarget:
                    idx = np.linspace(0, len(ticks) - 1, ntarget).round().astype(int)
                    ticks = ticks[idx]
                ticks = ticks[(ticks >= xmin) & (ticks <= xmax)]

        for row in range(nrows):
            ax = axes[row, col]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            formatter = log_value_formatter if col == ncols - 1 else smart_log_formatter
            ax.xaxis.set_major_formatter(FuncFormatter(formatter))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.tick_params(
                which="both",
                top=True,
                bottom=True,
                left=True,
                right=True,
                direction="in",
                labelsize=12,
            )
            ax.tick_params(axis="x", which="minor", length=3)
            ax.tick_params(axis="x", which="major", length=7)
            if row != nrows - 1:
                ax.tick_params(axis="x", labelbottom=False)
            if col != 0:
                ax.tick_params(axis="y", labelleft=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if co.SHOW:
        plt.show()
    plt.close(fig)
    return output_path


def compute_pairwise_spearman_correlations(
    sample: dict[str, pd.DataFrame],
    quantities: list[str] | None = None,
) -> pd.DataFrame:
    """Compute the pairwise group-property correlations split by domination."""

    quantities = quantities or DOMINATION_QUANTITIES
    corr = pd.DataFrame(
        columns=["Sample", "Domination", "Quantity1", "Quantity2", "Spear_corr", "Spear_pval"]
    )

    for cat in co.SAMPLE.keys():
        grp_key = cat + co.GRSUFF
        if grp_key not in sample:
            continue
        grp = sample[grp_key]

        for dom in [True, False]:
            df = grp[grp["is_dominated"] == dom]
            for q0, q1 in combinations(quantities, 2):
                if q0 not in df.columns or q1 not in df.columns:
                    continue
                x = pd.to_numeric(df[q0], errors="coerce").to_numpy()
                y = pd.to_numeric(df[q1], errors="coerce").to_numpy()
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 3:
                    spearcorr, spearpval = np.nan, np.nan
                else:
                    spearcorr, spearpval = spearmanr(x[mask], y[mask])

                corr = pu.append_df(
                    corr,
                    {
                        "Sample": cat,
                        "Domination": dom,
                        "Quantity1": q0,
                        "Quantity2": q1,
                        "Spear_corr": spearcorr,
                        "Spear_pval": spearpval,
                    },
                )

    return corr.sort_values(by="Spear_pval", ascending=True, na_position="last").reset_index(drop=True)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size used in the domination notebook."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum((ai > b).sum() for ai in a)
    lt = sum((ai < b).sum() for ai in a)
    return (gt - lt) / (len(a) * len(b))


def compute_distribution_tests(
    sample: dict[str, pd.DataFrame],
    quantities: list[str] | None = None,
) -> pd.DataFrame:
    """Run the non-parametric domination tests from the notebook."""

    quantities = quantities or DOMINATION_QUANTITIES
    rows: list[dict[str, object]] = []

    for cat in co.SAMPLE.keys():
        grp_key = cat + co.GRSUFF
        if grp_key not in sample:
            continue
        grp = sample[grp_key]
        dom = grp[grp["is_dominated"]]
        nod = grp[~grp["is_dominated"]]

        for quantity in quantities:
            if quantity not in grp.columns:
                continue
            a = pd.to_numeric(dom[quantity], errors="coerce").to_numpy()
            b = pd.to_numeric(nod[quantity], errors="coerce").to_numpy()
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]

            if len(a) < 5 or len(b) < 5:
                continue

            _, p_mwu = mannwhitneyu(a, b, alternative="two-sided")
            _, p_bm = brunnermunzel(a, b, alternative="two-sided")
            _, p_ks = ks_2samp(a, b, alternative="two-sided")

            rows.append(
                {
                    "Sample": cat,
                    "Quantity": quantity,
                    "n_dom": len(a),
                    "n_nondom": len(b),
                    "median_dom": np.median(a),
                    "median_nondom": np.median(b),
                    "MWU_p": p_mwu,
                    "BM_p": p_bm,
                    "KS_p": p_ks,
                    "Cliffs_delta": cliffs_delta(a, b),
                }
            )

    results = pd.DataFrame(rows)
    if not results.empty:
        reject, p_adj, _, _ = multipletests(results["BM_p"].to_numpy(), method="fdr_bh")
        results["BM_p_adj"] = p_adj
        results["BM_signif_FDR"] = reject
        results = results.sort_values(["BM_p_adj", "BM_p"], ascending=True).reset_index(drop=True)
    return results


def summarize_distribution_tests(results: pd.DataFrame) -> list[dict[str, object]]:
    """Keep the strongest FDR-significant domination split for each sample."""

    summary: list[dict[str, object]] = []

    for sample_name in co.SAMPLE.keys():
        sub = results[(results["Sample"] == sample_name) & (results["BM_signif_FDR"] == True)].copy()
        if sub.empty:
            summary.append(
                {
                    "sample": sample_name,
                    "sample_label": lu.formatted_sample_name(sample_name),
                    "has_significant_difference": False,
                }
            )
            continue

        sub = sub.sort_values(["BM_p_adj", "BM_p"], ascending=True)
        row = sub.iloc[0]
        summary.append(
            {
                "sample": sample_name,
                "sample_label": lu.formatted_sample_name(sample_name),
                "has_significant_difference": True,
                "quantity": row["Quantity"],
                "quantity_label": lu.formatted_label(row["Quantity"]),
                "median_dom": float(row["median_dom"]),
                "median_dom_fmt": f"{row['median_dom']:.3g}",
                "median_nondom": float(row["median_nondom"]),
                "median_nondom_fmt": f"{row['median_nondom']:.3g}",
                "n_dom": int(row["n_dom"]),
                "n_nondom": int(row["n_nondom"]),
                "p_adj": float(row["BM_p_adj"]),
                "p_adj_fmt": gu.tex_form(row["BM_p_adj"]),
                "cliffs_delta": float(row["Cliffs_delta"]),
                "cliffs_delta_fmt": f"{row['Cliffs_delta']:.2f}",
            }
        )

    return summary


def summarize_spiral_fraction_correlations(results: pd.DataFrame) -> list[dict[str, object]]:
    """Collect the significant spiral-fraction correlations used in the domination discussion."""

    summary: list[dict[str, object]] = []

    if results.empty:
        return summary

    significant = results[results["Spear_pval"] < co.P_LIMIT].sort_values("Spear_pval", ascending=True)
    for _, row in significant.iterrows():
        summary.append(
            {
                "sample": row["Sample"],
                "sample_label": lu.formatted_sample_name(row["Sample"]),
                "domination": row["Domination"],
                "quantity": row["Quantity"],
                "quantity_label": lu.formatted_label(row["Quantity"]),
                "n_groups": int(row["n_groups"]),
                "rho": float(row["Spear_corr"]),
                "rho_fmt": f"{row['Spear_corr']:.2f}",
                "p_value": float(row["Spear_pval"]),
                "p_value_fmt": gu.tex_form(row["Spear_pval"]),
            }
        )

    return summary


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    nvals = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    qvals = ranked * nvals / (np.arange(1, nvals + 1))
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    qvals = np.clip(qvals, 0.0, 1.0)
    out = np.empty_like(qvals)
    out[order] = qvals
    return out


def _bootstrap_spearman_rho(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n_points = x.size
    if n_points < 6:
        return np.array([], dtype=float)

    idx = rng.integers(0, n_points, size=(n_boot, n_points))
    rhos = np.empty(n_boot, dtype=float)
    for idx_row, boot_idx in enumerate(idx):
        rhos[idx_row] = spearmanr(x[boot_idx], y[boot_idx], nan_policy="omit").statistic
    return rhos[np.isfinite(rhos)]


def _compute_pair_pvals(df: pd.DataFrame, q1: str, q2: str, min_n_pair: int = 10):
    x = pd.to_numeric(df[q1], errors="coerce").to_numpy()
    y = pd.to_numeric(df[q2], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    n_points = int(mask.sum())
    if n_points < min_n_pair:
        return np.nan, np.nan, n_points
    stats = spearmanr(x[mask], y[mask], nan_policy="omit")
    return float(stats.pvalue), float(stats.statistic), n_points


def _format_p(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "-"
    if p_value < 1e-3:
        return f"{p_value:.1e}"
    return f"{p_value:.3f}"


def plot_xor_corrpairs_halfviolins_onefig(
    sample: dict[str, pd.DataFrame],
    quantities: list[str] | None = None,
    alpha_select: float = 0.10,
    alpha_report: float = 0.05,
    do_block_fdr: bool = True,
    min_n_pair: int = 10,
    n_boot: int = 2000,
    random_state: int = 0,
    output_path: str | None = None,
    max_pairs: int | None = None,
) -> str | None:
    """Single-figure half-violin plot for XOR domination correlations."""

    quantities = quantities or DOMINATION_QUANTITIES
    rng = np.random.default_rng(random_state)
    cats = list(co.SAMPLE.keys())
    pairs = list(combinations(quantities, 2))

    p_raw: dict[tuple[str, bool, tuple[str, str]], float] = {}
    p_used: dict[tuple[str, bool, tuple[str, str]], float] = {}
    n_pair: dict[tuple[str, bool, tuple[str, str]], int] = {}

    for cat in cats:
        grp_key = cat + co.GRSUFF
        if grp_key not in sample:
            continue
        grp = sample[grp_key]
        for dom in [True, False]:
            df = grp[grp["is_dominated"] == dom]
            for q1, q2 in pairs:
                if q1 not in df.columns or q2 not in df.columns:
                    p_raw[(cat, dom, (q1, q2))] = np.nan
                    n_pair[(cat, dom, (q1, q2))] = 0
                    continue
                p_value, _, n_points = _compute_pair_pvals(df, q1, q2, min_n_pair=min_n_pair)
                p_raw[(cat, dom, (q1, q2))] = p_value
                n_pair[(cat, dom, (q1, q2))] = n_points

    p_used.update(p_raw)
    if do_block_fdr:
        for cat in cats:
            for dom in [True, False]:
                block = np.array([p_raw.get((cat, dom, pair), np.nan) for pair in pairs], dtype=float)
                mask = np.isfinite(block)
                if mask.sum() == 0:
                    continue
                adjusted = np.full_like(block, np.nan)
                adjusted[mask] = _bh_fdr(block[mask])
                for idx, pair in enumerate(pairs):
                    p_used[(cat, dom, pair)] = adjusted[idx]

    xor_pairs: list[tuple[str, str]] = []
    for pair in pairs:
        any_xor = False
        for cat in cats:
            p_dom = p_used.get((cat, True, pair), np.nan)
            p_non = p_used.get((cat, False, pair), np.nan)
            sig_dom = np.isfinite(p_dom) and (p_dom < alpha_select)
            sig_non = np.isfinite(p_non) and (p_non < alpha_select)
            if (sig_dom != sig_non) and (sig_dom or sig_non):
                any_xor = True
                break
        if any_xor:
            xor_pairs.append(pair)

    if not xor_pairs:
        return None

    def best_xor_p(pair: tuple[str, str]) -> float:
        best = np.inf
        for cat in cats:
            p_dom = p_used.get((cat, True, pair), np.nan)
            p_non = p_used.get((cat, False, pair), np.nan)
            sig_dom = np.isfinite(p_dom) and (p_dom < alpha_select)
            sig_non = np.isfinite(p_non) and (p_non < alpha_select)
            if sig_dom != sig_non:
                candidate = p_dom if sig_dom else p_non
                if np.isfinite(candidate):
                    best = min(best, candidate)
        return best

    xor_pairs = sorted(xor_pairs, key=best_xor_p)
    if max_pairs is not None:
        xor_pairs = xor_pairs[:max_pairs]

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    nrows = len(xor_pairs)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(10.8, max(3.0, 1.9 * nrows)),
        sharex=True,
        sharey=True,
    )
    if nrows == 1:
        axes = np.array([axes])

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.99, hspace=0.55)
    sample_names = [lu.formatted_sample_name(cat) for cat in cats]
    x_positions = np.arange(len(cats))
    legend_done = False

    for idx, (q1, q2) in enumerate(xor_pairs):
        ax = axes[idx]
        rows: list[dict[str, object]] = []
        sig_rows: list[dict[str, object]] = []

        for cat in cats:
            grp_key = cat + co.GRSUFF
            if grp_key not in sample:
                continue
            grp = sample[grp_key]
            sample_name = lu.formatted_sample_name(cat)

            for dom in [True, False]:
                df = grp[grp["is_dominated"] == dom]
                if q1 not in df.columns or q2 not in df.columns:
                    continue
                rhos = _bootstrap_spearman_rho(
                    pd.to_numeric(df[q1], errors="coerce").to_numpy(),
                    pd.to_numeric(df[q2], errors="coerce").to_numpy(),
                    n_boot=n_boot,
                    rng=rng,
                )
                dom_label = "Dominated" if dom else "Non-dominated"
                for rho in rhos:
                    rows.append({"Sample": sample_name, "Domination": dom_label, "rho": rho})
                if rhos.size:
                    mu = float(np.mean(rhos))
                    sigma = float(np.std(rhos, ddof=1)) if rhos.size > 1 else 0.0
                    sig_rows.append(
                        {
                            "Sample": sample_name,
                            "Domination": dom_label,
                            "lo3": mu - 3 * sigma,
                            "hi3": mu + 3 * sigma,
                        }
                    )

        plot_df = pd.DataFrame(rows)
        sig_df = pd.DataFrame(sig_rows)

        if plot_df.empty:
            ax.text(0.5, 0.5, "Not enough data", transform=ax.transAxes, ha="center", va="center")
            ax.axis("off")
            continue

        sns.violinplot(
            data=plot_df,
            x="Sample",
            y="rho",
            hue="Domination",
            split=True,
            inner="quartile",
            cut=0,
            linewidth=1.0,
            ax=ax,
        )

        legend = ax.get_legend()
        if not legend_done and legend is not None:
            legend.set_title("")
            legend.set_frame_on(False)
            legend_done = True
        elif legend is not None:
            legend.remove()

        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticklabels(["-1.0", "-0.5", "0.0", "0.5", "1.0"])
        ax.set_ylabel(r"$\rho_\mathrm{S}$" if idx == nrows // 2 else "")
        ax.set_title(f"{lu.formatted_label(q1)}  vs  {lu.formatted_label(q2)}", pad=8)
        ax.axhline(0, lw=0.9, alpha=0.5)

        offset = {"Dominated": -0.12, "Non-dominated": 0.12}
        for _, row in sig_df.iterrows():
            xpos = sample_names.index(row["Sample"]) + offset[row["Domination"]]
            ax.vlines(xpos, row["lo3"], row["hi3"], linewidth=1.1, alpha=0.9)
            ax.hlines([row["lo3"], row["hi3"]], xpos - 0.03, xpos + 0.03, linewidth=1.1, alpha=0.9)

        ax.set_xticks(x_positions)
        if idx == nrows - 1:
            labels = []
            for cat, sample_name in zip(cats, sample_names):
                pair = (q1, q2)
                p_dom = p_used.get((cat, True, pair), np.nan)
                p_non = p_used.get((cat, False, pair), np.nan)
                n_dom = n_pair.get((cat, True, pair), 0)
                n_non = n_pair.get((cat, False, pair), 0)
                sig_dom = np.isfinite(p_dom) and (p_dom < alpha_report)
                sig_non = np.isfinite(p_non) and (p_non < alpha_report)
                p_dom_txt = _format_p(p_dom) + ("*" if (sig_dom and not sig_non) else "")
                p_non_txt = _format_p(p_non) + ("*" if (sig_non and not sig_dom) else "")
                labels.append(f"{sample_name}\nN={n_dom}/{n_non}\npD={p_dom_txt}  pN={p_non_txt}")
            ax.set_xticklabels(labels, rotation=0)
            ax.tick_params(axis="x", pad=10)
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

        sns.despine(ax=ax)

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if co.SHOW:
        plt.show()
    plt.close(fig)
    return output_path


def compute_spiral_fraction_correlations(
    sample: dict[str, pd.DataFrame],
    main_quantity: str = "S_frac",
    quantities: list[str] | None = None,
) -> pd.DataFrame:
    """Correlation table between group spiral fraction and the domination variables."""

    quantities = quantities or DOMINATION_QUANTITIES
    rows: list[dict[str, object]] = []

    for cat in co.SAMPLE.keys():
        grp_key = cat + co.GRSUFF
        if grp_key not in sample or main_quantity not in sample[grp_key].columns:
            continue
        df = sample[grp_key]
        for dom in [True, False]:
            subset = df[df["is_dominated"] == dom]
            for quantity in quantities:
                if quantity not in subset.columns:
                    continue
                clean = subset[[quantity, main_quantity]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(clean) < 3:
                    rho, p_value = np.nan, np.nan
                else:
                    rho, p_value = spearmanr(clean[quantity], clean[main_quantity])
                rows.append(
                    {
                        "Sample": cat,
                        "Domination": "Dominated" if dom else "Non-dominated",
                        "Quantity": quantity,
                        "main_quantity": main_quantity,
                        "n_groups": int(len(clean)),
                        "Spear_corr": rho,
                        "Spear_pval": p_value,
                    }
                )

    return pd.DataFrame(rows)


def run(
    sample: dict[str, pd.DataFrame],
    quantities: list[str] | None = None,
) -> dict[str, object]:
    """Entry point used by the main pipeline."""

    quantities = quantities or DOMINATION_QUANTITIES
    os.makedirs(co.FIGURES_PATH, exist_ok=True)
    os.makedirs(co.OUTPUT_PATH, exist_ok=True)
    os.makedirs(co.DATA_PATH, exist_ok=True)

    significant_quantities, significance_table = compute_significant_quantities(sample, quantities=quantities)
    significance_table.to_csv(os.path.join(co.OUTPUT_PATH, "domination_significance_summary.csv"), index=False)

    figures = {
        "domination_histograms": plot_domination_distributions(
            sample,
            significant_quantities,
            os.path.join(co.FIGURES_PATH, "Dom_vs_NonDom.pdf"),
        ),
        "xor_halfviolins": plot_xor_corrpairs_halfviolins_onefig(
            sample,
            quantities=quantities,
            output_path=os.path.join(co.FIGURES_PATH, "Dom_vs_NonDom_XOR_halfviolin_ONEFIG.pdf"),
        ),
    }

    pairwise_corr = compute_pairwise_spearman_correlations(sample, quantities=quantities)
    pairwise_corr.to_csv(os.path.join(co.OUTPUT_PATH, "domination_spearman_pairs.csv"), index=False)
    pairwise_corr[pairwise_corr["Spear_pval"] < co.P_LIMIT].sort_values(
        by=["Quantity1", "Quantity2"],
        ascending=True,
    ).to_csv(os.path.join(co.DATA_PATH, "Spearman_Dom_vs_NonDom.csv"), index=False)

    distribution_tests = compute_distribution_tests(sample, quantities=quantities)
    distribution_tests.to_csv(os.path.join(co.OUTPUT_PATH, "domination_distribution_tests.csv"), index=False)

    spiral_fraction_corr = compute_spiral_fraction_correlations(sample, quantities=quantities)
    spiral_fraction_corr.to_csv(
        os.path.join(co.OUTPUT_PATH, "domination_spiral_fraction_correlations.csv"),
        index=False,
    )

    distribution_summary = summarize_distribution_tests(distribution_tests)
    spiral_summary = summarize_spiral_fraction_correlations(spiral_fraction_corr)

    report.append_json("Dom_significant_quantities", significant_quantities)
    report.append_json(
        "Dom_significant_quantities_labels",
        [lu.formatted_label(quantity) for quantity in significant_quantities],
    )
    report.append_json("Dom_distribution_summary_by_sample", distribution_summary)
    report.append_json("Dom_spiral_fraction_significant_correlations", spiral_summary)
    report.append_json("Dom_spiral_fraction_significant_count", len(spiral_summary))

    return {
        "significant_quantities": significant_quantities,
        "significance_table": significance_table,
        "pairwise_correlations": pairwise_corr,
        "distribution_tests": distribution_tests,
        "distribution_summary": distribution_summary,
        "spiral_fraction_correlations": spiral_fraction_corr,
        "spiral_fraction_summary": spiral_summary,
        "figures": figures,
    }
