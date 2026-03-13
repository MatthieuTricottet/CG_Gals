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
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import ranksums
from scipy.stats import spearmanr

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


GROUP_QUANTITIES = ["t_cr", "M_virial", "Lum_group", "M_virial_over_L"]
STARFORMING = "Starforming"


def log10_formatter(x: float, _pos: int, n_min: int = -2, n_max: int = 2) -> str:
    """Format log10 axis ticks back into linear values."""

    if x < n_min or x > n_max:
        return rf"$10^{{{int(x)}}}$"
    return f"{10 ** int(x):.0f}"


def build_group_property_frame(sample: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Collect the group properties used in the movement/crossing-time notebook."""

    frames: list[pd.DataFrame] = []

    for cat in ["CG4", "Control4B", "RG4"]:
        key = cat + co.GRSUFF
        if key not in sample:
            continue

        df = sample[key].copy()
        required = set(GROUP_QUANTITIES)
        if not required.issubset(df.columns):
            continue

        if cat == "CG4" and "Class" in df.columns:
            frame = df[GROUP_QUANTITIES + ["Class"]].copy()
        else:
            frame = df[GROUP_QUANTITIES].copy()
            frame["Class"] = lu.formatted_sample_name(cat)

        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    for quantity in GROUP_QUANTITIES:
        data[f"lg_{quantity}"] = np.log10(pd.to_numeric(data[quantity], errors="coerce"))
    return data.replace([np.inf, -np.inf], np.nan).dropna()


def plot_group_relation(
    data: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    xticks: np.ndarray,
    yticks: np.ndarray,
) -> str | None:
    """Scatter+KDE plot used for the t_cross relations."""

    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=data, x=x, y=y, hue="Class", s=20, ax=ax)
    sns.kdeplot(data=data, x=x, y=y, hue="Class", levels=5, linewidths=1.5, ax=ax, legend=False)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(FuncFormatter(log10_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(log10_formatter))
    ax.legend(fontsize=12)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if co.SHOW:
        plt.show()
    plt.close(fig)
    return output_path


def plot_group_relation_3d(data: pd.DataFrame, output_path: str) -> str | None:
    """3D view of the t_cross, virial mass and group luminosity space."""

    if data.empty:
        return None

    x = data["lg_t_cr"].to_numpy()
    y = data["lg_M_virial"].to_numpy()
    z = data["lg_Lum_group"].to_numpy()
    classes = data["Class"].astype(str).to_numpy()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for label in np.unique(classes):
        mask = classes == label
        ax.scatter(x[mask], y[mask], z[mask], s=10, alpha=0.8, label=label)

    ax.set_xlabel(r"$t_\mathrm{cross}$ (Gyr)", fontsize=14)
    ax.set_ylabel(r"$M_\mathrm{VT}$ (solar)", fontsize=14)
    ax.set_zlabel(r"$L_\mathrm{group}$ (solar)", fontsize=14, rotation=270, labelpad=-12)
    ax.set_xticks(np.arange(-1, 1))
    ax.set_yticks(np.arange(11, 13))
    ax.set_zticks(np.arange(10, 11))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: rf"$10^{{{int(val)}}}$"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: rf"$10^{{{int(val)}}}$"))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda val, pos: rf"$10^{{{int(val)}}}$"))
    ax.view_init(elev=22, azim=35)
    ax.legend(title="Class", fontsize=12)
    plt.subplots_adjust(left=0.18, right=0.98, bottom=0.08, top=0.98)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if co.SHOW:
        plt.show()
    plt.close(fig)
    return output_path


def add_group_ssfr_excess_summary(sample: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add median sSFR excess and quenched flags to each group catalogue."""

    def loc_agg(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "sSFR_excess_median": float(np.nanmedian(group["sSFR_excess"])),
                "has_quenched": bool((group["sSFR_status"] == "Quenched").sum() > 0),
            }
        )

    for cat in co.SAMPLE.keys():
        gals_key = cat + co.GASUFF
        grp_key = cat + co.GRSUFF
        if gals_key not in sample or grp_key not in sample:
            continue
        if "sSFR_excess" not in sample[gals_key].columns or "sSFR_status" not in sample[gals_key].columns:
            continue

        group_stats = sample[gals_key].groupby("Group").apply(loc_agg).reset_index()
        groups = sample[grp_key].drop(columns=["sSFR_excess_median", "has_quenched"], errors="ignore")
        sample[grp_key] = groups.merge(group_stats, on="Group", how="left")

    return sample


def compare_main_sequence_offset_by_sample(sample: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pairwise rank-sum comparison against the CG sample."""

    cg = sample["CG4" + co.GASUFF]
    cg = cg[cg["sSFR_status"] == STARFORMING]
    rows: list[dict[str, object]] = []

    for cat in co.SAMPLE.keys():
        key = cat + co.GASUFF
        if key not in sample:
            continue
        df = sample[key]
        if "sSFR_MS_offset" not in df.columns or "sSFR_status" not in df.columns:
            continue
        df = df[df["sSFR_status"] == STARFORMING]

        row: dict[str, object] = {
            "sample": cat,
            "n_starforming": int(len(df)),
            "median_sSFR_MS_offset": float(df["sSFR_MS_offset"].median()) if len(df) else np.nan,
            "wilcoxon_statistic_vs_CG4": np.nan,
            "wilcoxon_p_value_vs_CG4": np.nan,
            "alternative": np.nan,
        }

        if cat != "CG4" and len(df) and len(cg):
            alternative = "less" if df["sSFR_MS_offset"].median() < cg["sSFR_MS_offset"].median() else "greater"
            statistic, p_value = ranksums(
                df["sSFR_MS_offset"],
                cg["sSFR_MS_offset"],
                alternative=alternative,
            )
            row.update(
                {
                    "wilcoxon_statistic_vs_CG4": float(statistic),
                    "wilcoxon_p_value_vs_CG4": float(p_value),
                    "alternative": alternative,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def compare_main_sequence_offset_by_cg_class(sample: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pairwise class-by-class rank-sum comparison inside the CG sample."""

    if "CG4_Groups" not in sample or "CG4_Gals" not in sample:
        return pd.DataFrame()
    if "Class" not in sample["CG4_Groups"].columns:
        return pd.DataFrame()

    df = sample["CG4_Gals"]
    if "sSFR_MS_offset" not in df.columns or "sSFR_status" not in df.columns:
        return pd.DataFrame()

    df = df[df["sSFR_status"] == STARFORMING]
    groups = sample["CG4_Groups"][["Group", "Class"]].dropna().drop_duplicates()
    df = df.merge(groups, on="Group", how="inner")

    rows: list[dict[str, object]] = []
    for class_a, class_b in combinations(sorted(df["Class"].unique()), 2):
        part_a = df[df["Class"] == class_a]["sSFR_MS_offset"]
        part_b = df[df["Class"] == class_b]["sSFR_MS_offset"]
        alternative = "less" if part_a.median() < part_b.median() else "greater"
        statistic, p_value = ranksums(part_a, part_b, alternative=alternative)
        rows.append(
            {
                "class_a": class_a,
                "class_b": class_b,
                "n_class_a": int(len(part_a)),
                "n_class_b": int(len(part_b)),
                "median_class_a": float(part_a.median()),
                "median_class_b": float(part_b.median()),
                "alternative": alternative,
                "wilcoxon_statistic": float(statistic),
                "wilcoxon_p_value": float(p_value),
            }
        )

    return pd.DataFrame(rows)


def compute_global_tcross_correlations(group_frame: pd.DataFrame) -> list[dict[str, object]]:
    """Quantify the trends shown in the crossing-time figures."""

    rows: list[dict[str, object]] = []

    if group_frame.empty:
        return rows

    for y_key, y_label in [
        ("lg_M_virial_over_L", r"$\log(\mathcal{M}_\mathrm{VT}/L_r)$"),
        ("lg_M_virial", r"$\log(\mathcal{M}_\mathrm{VT})$"),
        ("lg_Lum_group", r"$\log(L_\mathrm{group})$"),
    ]:
        clean = group_frame[["lg_t_cr", y_key]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 3:
            continue
        rho, p_value = spearmanr(clean["lg_t_cr"], clean[y_key])
        rows.append(
            {
                "x_key": "lg_t_cr",
                "y_key": y_key,
                "y_label": y_label,
                "n_groups": int(len(clean)),
                "rho": float(rho),
                "rho_fmt": f"{rho:.2f}",
                "p_value": float(p_value),
                "p_value_fmt": gu.tex_form(p_value),
            }
        )

    return rows


def run(sample: dict[str, pd.DataFrame], output_dir: str | None = None) -> dict[str, object]:
    """Entry point used by the main pipeline."""

    if output_dir is None:
        output_dir = co.FIGURES_PATH

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(co.OUTPUT_PATH, exist_ok=True)

    add_group_ssfr_excess_summary(sample)
    group_frame = build_group_property_frame(sample)
    tcross_correlations = compute_global_tcross_correlations(group_frame)

    figures = {
        "tcr_vs_mvirial_over_l": plot_group_relation(
            group_frame,
            x="lg_t_cr",
            y="lg_M_virial_over_L",
            xlabel=r"$t_\mathrm{cross}$ (Gyr)",
            ylabel=r"$M_\mathrm{VT}/L_r$ (solar)",
            output_path=os.path.join(output_dir, "tcr_vs_mvirial_over_l.pdf"),
            xticks=np.arange(-1, 2),
            yticks=np.arange(-1, 4),
        ),
        "tcr_vs_mvirial": plot_group_relation(
            group_frame,
            x="lg_t_cr",
            y="lg_M_virial",
            xlabel=r"$t_\mathrm{cross}$ (Gyr)",
            ylabel=r"$M_\mathrm{VT}$ (solar)",
            output_path=os.path.join(output_dir, "tcr_vs_mvirial.pdf"),
            xticks=np.arange(-1, 2),
            yticks=np.arange(11, 16),
        ),
        "tcr_mvirial_lgroup_3d": plot_group_relation_3d(
            group_frame,
            os.path.join(output_dir, "tcr_mvirial_lgroup_3d.pdf"),
        ),
    }

    by_sample = compare_main_sequence_offset_by_sample(sample)
    by_sample.to_csv(os.path.join(co.OUTPUT_PATH, "main_sequence_offset_by_sample.csv"), index=False)

    by_class = compare_main_sequence_offset_by_cg_class(sample)
    by_class.to_csv(os.path.join(co.OUTPUT_PATH, "main_sequence_offset_by_cg_class.csv"), index=False)

    sample_summary = []
    for _, row in by_sample.iterrows():
        sample_summary.append(
            {
                "sample": row["sample"],
                "sample_label": lu.formatted_sample_name(row["sample"]),
                "n_starforming": int(row["n_starforming"]),
                "median_ms_offset": float(row["median_sSFR_MS_offset"]),
                "median_ms_offset_fmt": f"{row['median_sSFR_MS_offset']:.3f}",
                "alternative": row["alternative"] if pd.notna(row["alternative"]) else None,
                "p_value": float(row["wilcoxon_p_value_vs_CG4"]) if pd.notna(row["wilcoxon_p_value_vs_CG4"]) else np.nan,
                "p_value_fmt": gu.tex_form(row["wilcoxon_p_value_vs_CG4"]) if pd.notna(row["wilcoxon_p_value_vs_CG4"]) else "NA",
            }
        )

    class_summary = []
    for _, row in by_class.iterrows():
        class_summary.append(
            {
                "class_a": row["class_a"],
                "class_b": row["class_b"],
                "n_class_a": int(row["n_class_a"]),
                "n_class_b": int(row["n_class_b"]),
                "median_class_a": float(row["median_class_a"]),
                "median_class_a_fmt": f"{row['median_class_a']:.3f}",
                "median_class_b": float(row["median_class_b"]),
                "median_class_b_fmt": f"{row['median_class_b']:.3f}",
                "alternative": row["alternative"],
                "p_value": float(row["wilcoxon_p_value"]),
                "p_value_fmt": gu.tex_form(row["wilcoxon_p_value"]),
            }
        )

    significant_sample_summary = [
        row for row in sample_summary
        if row["sample"] != "CG4" and np.isfinite(row["p_value"]) and row["p_value"] < co.P_LIMIT
    ]
    significant_class_summary = [
        row for row in class_summary
        if np.isfinite(row["p_value"]) and row["p_value"] < co.P_LIMIT
    ]

    report.append_json("Tcross_global_correlations", tcross_correlations)
    report.append_json("Main_sequence_offset_by_sample", sample_summary)
    report.append_json("Main_sequence_offset_by_sample_significant", significant_sample_summary)
    report.append_json("Main_sequence_offset_by_cg_class", class_summary)
    report.append_json("Main_sequence_offset_by_cg_class_significant", significant_class_summary)

    return {
        "figures": figures,
        "tcross_correlations": tcross_correlations,
        "main_sequence_offset_by_sample": by_sample,
        "main_sequence_offset_by_sample_summary": sample_summary,
        "main_sequence_offset_by_cg_class": by_class,
        "main_sequence_offset_by_cg_class_summary": class_summary,
    }
