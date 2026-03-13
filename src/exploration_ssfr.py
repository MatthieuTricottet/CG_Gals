from __future__ import annotations

import math
import os

import astropy.units as u
import matplotlib
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.cosmology import Planck15
from scipy.stats import pearsonr, spearmanr

try:
    import config as co
    import generate_report as report
    from utils import graphics_utils as gu
    from utils import labels_utils as lu
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from . import generate_report as report
    from .utils import graphics_utils as gu
    from .utils import labels_utils as lu


def add_dist2bgg_kpc_legacy(
    gdf: pd.DataFrame,
    group_col: str = "Group",
    z_col: str = "z",
    dist_ang_col: str = "dist2BGG",
    out_kpc_col: str = "dist2BGG_kpc",
    z_agg: str = "median",
) -> pd.DataFrame:
    """Convert the legacy angular separation proxy into projected proper distance."""

    if z_agg not in {"median", "mean"}:
        raise ValueError("z_agg must be 'median' or 'mean'.")

    df = gdf.copy()
    df[dist_ang_col] = pd.to_numeric(df[dist_ang_col], errors="coerce")
    df[z_col] = pd.to_numeric(df[z_col], errors="coerce")

    if z_agg == "median":
        z_group = df.groupby(group_col)[z_col].transform("median")
    else:
        z_group = df.groupby(group_col)[z_col].transform("mean")

    kpc_per_arcmin = Planck15.kpc_proper_per_arcmin(z_group.to_numpy())
    theta_rad = df[dist_ang_col].to_numpy() / 3600.0
    theta_arcmin = (theta_rad * u.rad).to(u.arcmin).value
    df[out_kpc_col] = theta_arcmin * kpc_per_arcmin.to(u.kpc / u.arcmin).value
    return df


def _mini_joint(
    fig: plt.Figure,
    outer_spec,
    df: pd.DataFrame,
    x: str,
    y: str = "sSFR",
    title: str = "",
    x_label: str = "",
    y_label: str = "sSFR",
    add_kde: bool = True,
    add_reg: bool = True,
    ci: int = 95,
    bins: int = 25,
    show_ylabel: bool = True,
    inner_hspace: float = 0.02,
    inner_wspace: float = 0.02,
):
    """Create the subplot-friendly joint distribution panel used in the notebook."""

    gs = outer_spec.subgridspec(
        2,
        2,
        height_ratios=(1, 4),
        width_ratios=(4, 1),
        hspace=inner_hspace,
        wspace=inner_wspace,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_joint = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    d = df[[x, y]].copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    n_points = len(d)

    sns.scatterplot(data=d, x=x, y=y, ax=ax_joint, s=18, alpha=0.45, linewidth=0)

    if add_reg and n_points >= 3:
        sns.regplot(
            data=d,
            x=x,
            y=y,
            ax=ax_joint,
            scatter=False,
            ci=None,
            truncate=False,
            line_kws={"linewidth": 2},
        )

    if add_kde and n_points >= 20:
        kde_df = d if n_points <= 1500 else d.sample(1500, random_state=0)
        sns.kdeplot(data=kde_df, x=x, y=y, ax=ax_joint, levels=5, fill=False, linewidths=1)

    sns.histplot(data=d, x=x, ax=ax_top, bins=bins, kde=False)
    sns.histplot(data=d, y=y, ax=ax_right, bins=bins, kde=False)

    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_top.set_xlabel("")
    ax_top.set_ylabel("")
    ax_right.set_xlabel("")
    ax_right.set_ylabel("")

    ax_top.set_xlim(ax_joint.get_xlim())
    ax_right.set_ylim(ax_joint.get_ylim())
    ax_joint.set_xlabel(x_label)
    ax_joint.set_ylabel(y_label if show_ylabel else "")

    if n_points >= 3:
        rho_s, p_s = spearmanr(d[x], d[y])
        r_p, p_p = pearsonr(d[x], d[y])
        txt = (
            f"n={n_points}\n"
            f"Spearman rho={rho_s:.2f}, p={gu.tex_form(p_s)}\n"
            f"Pearson r={r_p:.2f}, p={gu.tex_form(p_p)}"
        )
    else:
        txt = f"n={n_points}\n(not enough data)"

    ax_joint.text(
        0.02,
        0.98,
        txt,
        transform=ax_joint.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.8", "alpha": 0.9},
    )

    if title:
        ax_top.set_title(title, fontsize=11)

    return ax_joint


def _make_grid_figure(
    panels: list[tuple[str, pd.DataFrame, str, str]],
    y: str = "sSFR",
    figsize_per_panel: float = 4.0,
    outer_wspace: float = 0.20,
    outer_hspace: float = 0.25,
    inner_hspace: float = 0.02,
    inner_wspace: float = 0.02,
) -> plt.Figure:
    """Lay out all sample panels on a roughly square grid."""

    if not panels:
        raise ValueError("No panels to plot.")

    sns.set_theme(style="whitegrid", context="notebook")

    n_panels = len(panels)
    ncols = math.ceil(math.sqrt(n_panels))
    nrows = math.ceil(n_panels / ncols)

    fig = plt.figure(figsize=(figsize_per_panel * ncols, figsize_per_panel * nrows))
    outer = fig.add_gridspec(nrows, ncols, wspace=outer_wspace, hspace=outer_hspace)

    for idx, (cat, df, xcol, xlabel) in enumerate(panels):
        row = idx // ncols
        col = idx % ncols
        _mini_joint(
            fig,
            outer[row, col],
            df,
            x=xcol,
            y=y,
            title=lu.formatted_sample_name(cat),
            x_label=xlabel,
            y_label=y,
            show_ylabel=(col == 0),
            inner_hspace=inner_hspace,
            inner_wspace=inner_wspace,
        )

    for idx in range(n_panels, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(outer[row, col])
        ax.axis("off")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95)
    return fig


def make_two_figures(
    sample: dict[str, pd.DataFrame],
    group_col: str = "Group",
    z_col: str = "z",
    dist_ang_col: str = "dist2BGG",
    y_col: str = "sSFR",
    rank_cut: int = 1,
    group_scale_col: str = "size_Group_Bary_kpc",
    figsize_per_panel: float = 4.0,
):
    """Build the two distance-versus-sSFR figures from the notebook."""

    panels_kpc: list[tuple[str, pd.DataFrame, str, str]] = []
    panels_norm: list[tuple[str, pd.DataFrame, str, str]] = []

    for cat in co.SAMPLE.keys():
        gal_key = cat + co.GASUFF
        grp_key = cat + co.GRSUFF

        if gal_key not in sample or grp_key not in sample:
            continue

        gdf = sample[gal_key].copy()
        rdf = sample[grp_key].copy()

        required_gal_cols = [group_col, z_col, dist_ang_col, "rank_M", y_col]
        if any(col not in gdf.columns for col in required_gal_cols):
            continue
        if group_col not in rdf.columns or group_scale_col not in rdf.columns:
            continue

        gdf = add_dist2bgg_kpc_legacy(
            gdf,
            group_col=group_col,
            z_col=z_col,
            dist_ang_col=dist_ang_col,
            out_kpc_col="dist2BGG_kpc",
            z_agg="median",
        )

        gdf_sat = gdf.loc[gdf["rank_M"] > rank_cut].copy()
        panels_kpc.append((cat, gdf_sat, "dist2BGG_kpc", "Distance to BGG (kpc)"))

        rdf_min = rdf[[group_col, group_scale_col]].copy()
        rdf_min[group_scale_col] = pd.to_numeric(rdf_min[group_scale_col], errors="coerce")
        rdf_min = rdf_min.groupby(group_col, as_index=False)[group_scale_col].median()

        if group_scale_col in gdf.columns:
            gdf = gdf.drop(columns=[group_scale_col])
        gdf = gdf.merge(rdf_min, on=group_col, how="left", validate="m:1")
        gdf["norm_dist"] = pd.to_numeric(gdf["dist2BGG_kpc"], errors="coerce") / pd.to_numeric(
            gdf[group_scale_col], errors="coerce"
        )
        gdf_sat_norm = gdf.loc[gdf["rank_M"] > rank_cut].copy()
        panels_norm.append(
            (
                cat,
                gdf_sat_norm,
                "norm_dist",
                r"$\Delta_{\mathrm{BGG}}/\langle R_{ij}\rangle$",
            )
        )

        sample[gal_key] = gdf

    fig_kpc = _make_grid_figure(panels_kpc, y=y_col, figsize_per_panel=figsize_per_panel) if panels_kpc else None
    fig_norm = _make_grid_figure(panels_norm, y=y_col, figsize_per_panel=figsize_per_panel) if panels_norm else None
    return fig_kpc, fig_norm


def compute_distance_correlations(sample: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    """Summarize the sSFR-distance correlations shown in the exploration figures."""

    rows: list[dict[str, object]] = []

    for cat in co.SAMPLE.keys():
        gal_key = cat + co.GASUFF
        grp_key = cat + co.GRSUFF
        if gal_key not in sample or grp_key not in sample:
            continue

        gdf = sample[gal_key].copy()
        rdf = sample[grp_key].copy()
        required = {"Group", "z", "dist2BGG", "rank_M", "sSFR"}
        if not required.issubset(gdf.columns):
            continue
        if not {"Group", "size_Group_Bary_kpc"}.issubset(rdf.columns):
            continue

        gdf = add_dist2bgg_kpc_legacy(gdf)
        if "size_Group_Bary_kpc" in gdf.columns:
            gdf = gdf.drop(columns=["size_Group_Bary_kpc"])
        rdf_min = rdf[["Group", "size_Group_Bary_kpc"]].copy()
        rdf_min["size_Group_Bary_kpc"] = pd.to_numeric(rdf_min["size_Group_Bary_kpc"], errors="coerce")
        rdf_min = rdf_min.groupby("Group", as_index=False)["size_Group_Bary_kpc"].median()
        gdf = gdf.merge(rdf_min, on="Group", how="left", validate="m:1")
        gdf["norm_dist"] = pd.to_numeric(gdf["dist2BGG_kpc"], errors="coerce") / pd.to_numeric(
            gdf["size_Group_Bary_kpc"], errors="coerce"
        )
        gdf = gdf[gdf["rank_M"] > 1]

        for distance_key, distance_label in [
            ("dist2BGG_kpc", "projected distance to the BGG"),
            ("norm_dist", r"distance to the BGG normalized by $\langle R_{ij}\rangle$"),
        ]:
            data = (
                gdf[[distance_key, "sSFR"]]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(data) < 3:
                continue

            rho_s, p_s = spearmanr(data[distance_key], data["sSFR"])
            r_p, p_p = pearsonr(data[distance_key], data["sSFR"])
            rows.append(
                {
                    "sample": cat,
                    "sample_label": lu.formatted_sample_name(cat),
                    "distance_key": distance_key,
                    "distance_label": distance_label,
                    "n": int(len(data)),
                    "spearman_rho": float(rho_s),
                    "spearman_rho_fmt": f"{rho_s:.2f}",
                    "spearman_p": float(p_s),
                    "spearman_p_fmt": gu.tex_form(p_s),
                    "pearson_r": float(r_p),
                    "pearson_r_fmt": f"{r_p:.2f}",
                    "pearson_p": float(p_p),
                    "pearson_p_fmt": gu.tex_form(p_p),
                }
            )

    return rows


def _save_or_close(fig: plt.Figure | None, path: str | None) -> str | None:
    if fig is None or path is None:
        return None
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if co.SHOW:
        plt.show()
    plt.close(fig)
    return path


def run(sample: dict[str, pd.DataFrame], output_dir: str | None = None) -> dict[str, str | None]:
    """Entry point used by the main pipeline."""

    if output_dir is None:
        output_dir = co.FIGURES_PATH

    os.makedirs(output_dir, exist_ok=True)
    fig_kpc, fig_norm = make_two_figures(sample, rank_cut=1, figsize_per_panel=4.0)
    correlations = compute_distance_correlations(sample)
    significant = [row for row in correlations if row["spearman_p"] < co.P_LIMIT]

    report.append_json("Dist2BGG_correlations", correlations)
    report.append_json("Dist2BGG_significant_correlations", significant)
    report.append_json("Dist2BGG_significant_count", len(significant))

    return {
        "dist2BGG_kpc": _save_or_close(fig_kpc, os.path.join(output_dir, "dist2BGG_kpc_vs_sSFR.pdf")),
        "dist2BGG_norm": _save_or_close(fig_norm, os.path.join(output_dir, "dist2BGG_norm_vs_sSFR.pdf")),
        "correlations": correlations,
        "significant_correlations": significant,
    }
