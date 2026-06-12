from __future__ import annotations

import os

import matplotlib
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

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


COLOUR_SPECS = [
    ("u_r", "u-r", r"$(u-r)$"),
    ("u_g", "u-g", r"$(u-g)$"),
    ("g_r", "g-r", r"$(g-r)$"),
    ("r_i", "r-i", r"$(r-i)$"),
]
CATALOGUE_LABELS = {
    "CG4_Gals": "CG4",
    "Control4B_Gals": "Control4B",
    "Control4C_Gals": "Control4C",
    "RG4_Gals": "RG4",
}
CATALOGUE_ORDER = ["CG4", "Control4B", "Control4C", "RG4", "SDSS"]
CONTROL_LABELS = ["Control4B", "Control4C", "RG4"]
PALETTE = {
    "SDSS": "0.65",
    "CG4": "#2864A6",
    "Control4B": "#D17A22",
    "Control4C": "#25876E",
    "RG4": "#A74752",
    "Compact": "#2864A6",
    "Ordinary": "#555555",
    "Dominated": "#A74752",
    "Non-dominated": "#555555",
}


def _as_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _colour_label(colour: str) -> str:
    return rf"$({colour})$"


def _finite_float(value) -> float | None:
    value = float(np.asarray(value).item())
    return value if np.isfinite(value) else None


def _format_p(value: float | None) -> str:
    return (
        gu.latex_number(value, precision=3, math_mode=False)
        if value is not None and np.isfinite(value)
        else "NA"
    )


def _save_figure(fig: plt.Figure, output_path: str) -> str:
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if co.SHOW:
        plt.show()
    plt.close(fig)
    return output_path


def build_sdss_colour_lookup(sample: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a unique SDSS photometry lookup with the four colour indices."""

    required = {"objid", "lgm", "u_obs", "g_obs", "r_obs", "i_obs"}
    if "SDSS" not in sample or not required.issubset(sample["SDSS"].columns):
        return pd.DataFrame(columns=["_objid", "lgm", *[spec[0] for spec in COLOUR_SPECS]])

    sdss = sample["SDSS"].copy()
    sdss["_objid"] = _as_int(sdss["objid"])
    for column in ["lgm", "u_obs", "g_obs", "r_obs", "i_obs"]:
        sdss[column] = pd.to_numeric(sdss[column], errors="coerce")

    sdss["u_r"] = sdss["u_obs"] - sdss["r_obs"]
    sdss["u_g"] = sdss["u_obs"] - sdss["g_obs"]
    sdss["g_r"] = sdss["g_obs"] - sdss["r_obs"]
    sdss["r_i"] = sdss["r_obs"] - sdss["i_obs"]
    columns = ["_objid", "lgm", *[spec[0] for spec in COLOUR_SPECS]]
    return sdss[columns].dropna(subset=["_objid"]).drop_duplicates("_objid")


def build_catalogue_colour_frame(
    sample: dict[str, pd.DataFrame],
    lookup: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match each catalogue to SDSS photometry and return match statistics."""

    lookup = build_sdss_colour_lookup(sample) if lookup is None else lookup
    frames = [lookup.copy().assign(catalogue="SDSS")]
    match_rows = []

    for key, label in CATALOGUE_LABELS.items():
        if key not in sample or not {"objid", "lgm"}.issubset(sample[key].columns):
            continue

        sub = sample[key].copy()
        sub["_objid"] = _as_int(sub["objid"])
        sub["lgm"] = pd.to_numeric(sub["lgm"], errors="coerce")
        merged = (
            sub[["_objid", "lgm"]]
            .merge(lookup.drop(columns="lgm"), on="_objid", how="inner")
            .drop_duplicates("_objid")
        )
        merged["catalogue"] = label
        frames.append(merged)
        match_rows.append(
            {
                "sample": label,
                "sample_label": lu.formatted_sample_name(label),
                "n_total": int(len(sub)),
                "n_matched": int(len(merged)),
                "matched_fraction": float(len(merged) / len(sub)) if len(sub) else np.nan,
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    return combined, pd.DataFrame(match_rows)


def compute_catalogue_colour_mass_tests(
    colour_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare catalogue colour-mass slopes with HC3-robust ANCOVA."""

    slope_rows = []
    global_rows = []
    pairwise_rows = []

    for colour_key, colour_label, _ in COLOUR_SPECS:
        panel = colour_frame[["catalogue", "lgm", colour_key]].dropna().copy()
        panel["catalogue"] = pd.Categorical(
            panel["catalogue"],
            categories=CATALOGUE_ORDER,
            ordered=True,
        )
        if len(panel) < 10:
            continue

        model = smf.ols(
            f'{colour_key} ~ lgm * C(catalogue, Treatment(reference="CG4"))',
            data=panel,
        ).fit(cov_type="HC3")

        for label, group in panel.groupby("catalogue", observed=False):
            if len(group) < 2:
                continue
            slope, intercept = np.polyfit(group["lgm"], group[colour_key], deg=1)
            slope_rows.append(
                {
                    "colour": colour_label,
                    "colour_label": _colour_label(colour_label),
                    "catalogue": str(label),
                    "n": int(len(group)),
                    "slope": float(slope),
                    "intercept": float(intercept),
                }
            )

        terms = [
            f'lgm:C(catalogue, Treatment(reference="CG4"))[T.{label}]'
            for label in ["Control4B", "Control4C", "RG4", "SDSS"]
            if f'lgm:C(catalogue, Treatment(reference="CG4"))[T.{label}]' in model.params.index
        ]
        if terms:
            global_test = model.wald_test([f"{term} = 0" for term in terms], scalar=True)
            global_rows.append(
                {
                    "colour": colour_label,
                    "colour_label": _colour_label(colour_label),
                    "n_total": int(len(panel)),
                    "statistic": _finite_float(global_test.statistic),
                    "p_value": _finite_float(global_test.pvalue),
                }
            )

        current_pairwise = []
        for label in ["Control4B", "Control4C", "RG4", "SDSS"]:
            term = f'lgm:C(catalogue, Treatment(reference="CG4"))[T.{label}]'
            if term not in model.params.index:
                continue

            delta_slope = -float(model.params[term])
            p_two_sided = float(model.pvalues[term])
            p_one_sided = p_two_sided / 2 if delta_slope > 0 else 1 - p_two_sided / 2
            current_pairwise.append(
                {
                    "colour": colour_label,
                    "colour_label": _colour_label(colour_label),
                    "comparison": label,
                    "comparison_label": (
                        "SDSS" if label == "SDSS" else lu.formatted_sample_name(label)
                    ),
                    "delta_slope": delta_slope,
                    "test_statistic": float(delta_slope / model.bse[term]),
                    "p_one_sided": p_one_sided,
                }
            )

        if current_pairwise:
            adjusted = multipletests(
                [row["p_one_sided"] for row in current_pairwise],
                method="holm",
            )[1]
            for row, p_holm in zip(current_pairwise, adjusted):
                row["p_holm"] = float(p_holm)
                row["significant"] = bool(p_holm < co.P_LIMIT)
                pairwise_rows.append(row)

    return (
        pd.DataFrame(slope_rows),
        pd.DataFrame(global_rows),
        pd.DataFrame(pairwise_rows),
    )


def build_satellite_colour_frame(
    sample: dict[str, pd.DataFrame],
    lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Collect SDSS-matched satellites with catalogue and group identifiers."""

    lookup = build_sdss_colour_lookup(sample) if lookup is None else lookup
    photometry = lookup.drop(columns="lgm")
    frames = []

    for key, label in CATALOGUE_LABELS.items():
        required = {"objid", "Group", "rank_M", "lgm"}
        if key not in sample or not required.issubset(sample[key].columns):
            continue

        sub = sample[key].copy()
        sub["_objid"] = _as_int(sub["objid"])
        sub["lgm"] = pd.to_numeric(sub["lgm"], errors="coerce")
        merged = sub[["Group", "rank_M", "lgm", "_objid"]].merge(
            photometry,
            on="_objid",
            how="inner",
        )
        merged = merged.loc[(merged["rank_M"] > 1) & np.isfinite(merged["lgm"])].copy()
        merged["catalogue"] = label
        merged["environment"] = "Compact" if label == "CG4" else "Ordinary"
        merged["cluster_id"] = label + "_" + merged["Group"].astype(str)
        frames.append(merged)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def compute_satellite_environment_tests(
    satellite_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Compare compact and ordinary satellites at fixed stellar mass."""

    reference_mass = float(satellite_frame["lgm"].median())
    rows = []

    for colour_key, colour_label, _ in COLOUR_SPECS:
        panel = satellite_frame[["environment", "cluster_id", "lgm", colour_key]].dropna().copy()
        panel["environment"] = pd.Categorical(
            panel["environment"],
            categories=["Ordinary", "Compact"],
            ordered=True,
        )
        model = smf.ols(
            f'{colour_key} ~ lgm * C(environment, Treatment(reference="Ordinary"))',
            data=panel,
        ).fit(cov_type="cluster", cov_kwds={"groups": panel["cluster_id"]})

        offset_term = 'C(environment, Treatment(reference="Ordinary"))[T.Compact]'
        interaction_term = 'lgm:C(environment, Treatment(reference="Ordinary"))[T.Compact]'
        ordinary_slope = float(model.params["lgm"])
        compact_slope = ordinary_slope + float(model.params[interaction_term])
        offset = float(model.params[offset_term] + reference_mass * model.params[interaction_term])
        offset_test = model.t_test(
            f"{offset_term} + {reference_mass:.12f} * {interaction_term} = 0"
        )

        rows.append(
            {
                "colour": colour_label,
                "colour_label": _colour_label(colour_label),
                "ordinary_slope": ordinary_slope,
                "compact_slope": compact_slope,
                "compact_minus_ordinary": offset,
                "interaction_p_raw": float(model.pvalues[interaction_term]),
                "difference_p_raw": float(np.asarray(offset_test.pvalue).item()),
                "n_compact": int((panel["environment"] == "Compact").sum()),
                "n_ordinary": int((panel["environment"] == "Ordinary").sum()),
            }
        )

    result = pd.DataFrame(rows)
    result["interaction_p_holm"] = multipletests(result["interaction_p_raw"], method="holm")[1]
    result["difference_p_holm"] = multipletests(result["difference_p_raw"], method="holm")[1]
    result["interaction_significant"] = result["interaction_p_holm"] < co.P_LIMIT
    result["difference_significant"] = result["difference_p_holm"] < co.P_LIMIT
    return result, reference_mass


def build_group_colour_pairs(
    sample: dict[str, pd.DataFrame],
    lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pair each BGG colour with the mean colour of its matched satellites."""

    lookup = build_sdss_colour_lookup(sample) if lookup is None else lookup
    photometry = lookup.drop(columns="lgm")
    rows = []

    for key, label in CATALOGUE_LABELS.items():
        required = {"objid", "Group", "rank_M"}
        if key not in sample or not required.issubset(sample[key].columns):
            continue

        sub = sample[key].copy()
        sub["_objid"] = _as_int(sub["objid"])
        merged = sub[["Group", "rank_M", "_objid"]].merge(photometry, on="_objid", how="inner")

        for group_id, group in merged.groupby("Group"):
            bgg = group.loc[group["rank_M"] == 1]
            satellites = group.loc[group["rank_M"] > 1]
            if len(bgg) != 1 or satellites.empty:
                continue

            row = {
                "catalogue": label,
                "Group": group_id,
                "matched_satellites": int(len(satellites)),
            }
            for colour_key, _, _ in COLOUR_SPECS:
                bgg_value = pd.to_numeric(bgg[colour_key], errors="coerce").iloc[0]
                sat_values = pd.to_numeric(satellites[colour_key], errors="coerce")
                sat_values = sat_values[np.isfinite(sat_values)]
                if not np.isfinite(bgg_value) or sat_values.empty:
                    break
                row[f"bgg_{colour_key}"] = float(bgg_value)
                row[f"sat_mean_{colour_key}"] = float(sat_values.mean())
            else:
                rows.append(row)

    return pd.DataFrame(rows)


def compute_bgg_satellite_correlations(group_pairs: pd.DataFrame) -> pd.DataFrame:
    """Measure BGG-versus-mean-satellite colour correlations."""

    rows = []
    sample_frames = [("All", group_pairs)] + [
        (label, group_pairs.loc[group_pairs["catalogue"] == label])
        for label in CATALOGUE_ORDER[:-1]
    ]

    for colour_key, colour_label, _ in COLOUR_SPECS:
        x_key = f"bgg_{colour_key}"
        y_key = f"sat_mean_{colour_key}"
        for label, frame in sample_frames:
            panel = frame[[x_key, y_key, "matched_satellites"]].dropna()
            if len(panel) < 3:
                continue

            pearson = stats.pearsonr(panel[x_key], panel[y_key])
            spearman = stats.spearmanr(panel[x_key], panel[y_key])
            slope, intercept = np.polyfit(panel[x_key], panel[y_key], deg=1)
            rows.append(
                {
                    "colour": colour_label,
                    "colour_label": _colour_label(colour_label),
                    "sample": label,
                    "sample_label": lu.formatted_sample_name(label) if label != "All" else "all samples",
                    "n_groups": int(len(panel)),
                    "mean_matched_satellites": float(panel["matched_satellites"].mean()),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "slope": float(slope),
                    "intercept": float(intercept),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["pearson_p_holm"] = np.nan
    result["spearman_p_holm"] = np.nan
    for _, indices in result.groupby("sample").groups.items():
        idx = list(indices)
        result.loc[idx, "pearson_p_holm"] = multipletests(
            result.loc[idx, "pearson_p"],
            method="holm",
        )[1]
        result.loc[idx, "spearman_p_holm"] = multipletests(
            result.loc[idx, "spearman_p"],
            method="holm",
        )[1]
    return result


def compute_bgg_satellite_slope_tests(group_pairs: pd.DataFrame) -> pd.DataFrame:
    """Test whether the BGG-satellite colour slope differs from CG4."""

    rows = []
    for colour_key, colour_label, _ in COLOUR_SPECS:
        x_key = f"bgg_{colour_key}"
        y_key = f"sat_mean_{colour_key}"
        panel = group_pairs[["catalogue", x_key, y_key]].dropna().copy()
        panel["catalogue"] = pd.Categorical(
            panel["catalogue"],
            categories=["CG4", *CONTROL_LABELS],
            ordered=True,
        )
        model = smf.ols(
            f'{y_key} ~ {x_key} * C(catalogue, Treatment(reference="CG4"))',
            data=panel,
        ).fit(cov_type="HC3")
        cg4_slope = float(model.params[x_key])
        current = []

        for label in CONTROL_LABELS:
            term = f'{x_key}:C(catalogue, Treatment(reference="CG4"))[T.{label}]'
            if term not in model.params.index:
                continue
            delta = float(model.params[term])
            current.append(
                {
                    "colour": colour_label,
                    "colour_label": _colour_label(colour_label),
                    "comparison": label,
                    "comparison_label": lu.formatted_sample_name(label),
                    "cg4_n": int((panel["catalogue"] == "CG4").sum()),
                    "comparison_n": int((panel["catalogue"] == label).sum()),
                    "cg4_slope": cg4_slope,
                    "comparison_slope": cg4_slope + delta,
                    "delta_slope": delta,
                    "p_raw": float(model.pvalues[term]),
                }
            )

        if current:
            adjusted = multipletests([row["p_raw"] for row in current], method="holm")[1]
            for row, p_holm in zip(current, adjusted):
                row["p_holm"] = float(p_holm)
                row["significant"] = bool(p_holm < co.P_LIMIT)
                rows.append(row)

    return pd.DataFrame(rows)


def build_bgg_colour_frame(
    sample: dict[str, pd.DataFrame],
    lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Collect matched BGG colours and domination flags."""

    lookup = build_sdss_colour_lookup(sample) if lookup is None else lookup
    photometry = lookup.drop(columns="lgm")
    frames = []

    for key, label in CATALOGUE_LABELS.items():
        required = {"objid", "Group", "rank_M", "lgm", "is_dominated"}
        if key not in sample or not required.issubset(sample[key].columns):
            continue

        sub = sample[key].copy()
        sub["_objid"] = _as_int(sub["objid"])
        sub["lgm"] = pd.to_numeric(sub["lgm"], errors="coerce")
        merged = sub[["Group", "rank_M", "lgm", "is_dominated", "_objid"]].merge(
            photometry,
            on="_objid",
            how="inner",
        )
        merged = merged.loc[(merged["rank_M"] == 1) & np.isfinite(merged["lgm"])].copy()
        merged["is_dominated"] = (
            pd.to_numeric(merged["is_dominated"], errors="coerce").fillna(0).astype(bool)
        )
        merged["domination"] = np.where(
            merged["is_dominated"],
            "Dominated",
            "Non-dominated",
        )
        merged["catalogue"] = label
        frames.append(merged)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def compute_bgg_domination_tests(
    bgg_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame, float]:
    """Compare dominated and non-dominated BGG colours before and after mass adjustment."""

    mass_rows = []
    for label in ["Dominated", "Non-dominated"]:
        masses = bgg_frame.loc[bgg_frame["domination"] == label, "lgm"].dropna()
        mass_rows.append(
            {
                "domination": label,
                "n_bgg": int(len(masses)),
                "median_log_mass": float(masses.median()),
                "mean_log_mass": float(masses.mean()),
            }
        )
    mass_summary = pd.DataFrame(mass_rows)
    dominated_mass = bgg_frame.loc[bgg_frame["domination"] == "Dominated", "lgm"].dropna()
    non_dominated_mass = bgg_frame.loc[
        bgg_frame["domination"] == "Non-dominated",
        "lgm",
    ].dropna()
    mass_test = stats.mannwhitneyu(
        dominated_mass,
        non_dominated_mass,
        alternative="two-sided",
        method="asymptotic",
    )
    mass_result = {
        "p_value": float(mass_test.pvalue),
        "significant": bool(mass_test.pvalue < co.P_LIMIT),
    }

    raw_rows = []
    for colour_key, colour_label, _ in COLOUR_SPECS:
        panel = bgg_frame[[colour_key, "domination"]].dropna()
        dominated = panel.loc[panel["domination"] == "Dominated", colour_key]
        non_dominated = panel.loc[panel["domination"] == "Non-dominated", colour_key]
        test = stats.mannwhitneyu(
            dominated,
            non_dominated,
            alternative="two-sided",
            method="asymptotic",
        )
        raw_rows.append(
            {
                "colour": colour_label,
                "colour_label": _colour_label(colour_label),
                "median_dominated": float(dominated.median()),
                "median_non_dominated": float(non_dominated.median()),
                "delta_median": float(dominated.median() - non_dominated.median()),
                "p_raw": float(test.pvalue),
            }
        )
    raw = pd.DataFrame(raw_rows)
    raw["p_holm"] = multipletests(raw["p_raw"], method="holm")[1]
    raw["significant"] = raw["p_holm"] < co.P_LIMIT

    reference_mass = float(bgg_frame["lgm"].median())
    adjusted_rows = []
    for colour_key, colour_label, _ in COLOUR_SPECS:
        panel = bgg_frame[["lgm", colour_key, "is_dominated"]].dropna().copy()
        panel["dominated"] = panel["is_dominated"].astype(int)
        model = smf.ols(f"{colour_key} ~ lgm * dominated", data=panel).fit(cov_type="HC3")
        params = model.params
        covariance = model.cov_params()
        names = list(params.index)
        weights = np.zeros(len(names))
        for index, name in enumerate(names):
            if name == "dominated":
                weights[index] = 1.0
            elif name == "lgm:dominated":
                weights[index] = reference_mass

        offset = float(weights @ params.to_numpy())
        offset_se = float(np.sqrt(weights @ covariance.to_numpy() @ weights))
        offset_z = offset / offset_se if offset_se > 0 else np.nan
        offset_p = float(2 * stats.norm.sf(abs(offset_z))) if np.isfinite(offset_z) else np.nan
        adjusted_rows.append(
            {
                "colour": colour_label,
                "colour_label": _colour_label(colour_label),
                "slope_non_dominated": float(params.get("lgm", 0.0)),
                "slope_dominated": float(params.get("lgm", 0.0) + params.get("lgm:dominated", 0.0)),
                "slope_difference_p": float(model.pvalues.get("lgm:dominated", np.nan)),
                "delta_at_reference_mass": offset,
                "delta_ci_low": offset - 1.96 * offset_se,
                "delta_ci_high": offset + 1.96 * offset_se,
                "offset_p_raw": offset_p,
            }
        )
    adjusted = pd.DataFrame(adjusted_rows)
    adjusted["offset_p_holm"] = multipletests(adjusted["offset_p_raw"], method="holm")[1]
    adjusted["significant"] = adjusted["offset_p_holm"] < co.P_LIMIT
    return mass_summary, mass_result, raw, adjusted, reference_mass


def _style_axes(ax: plt.Axes) -> None:
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(alpha=0.18)


def plot_colour_mass_relations(colour_frame: pd.DataFrame, output_path: str) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.6), sharex=True)

    for ax, (colour_key, colour_label, axis_label) in zip(axes.flat, COLOUR_SPECS):
        panel = colour_frame[["catalogue", "lgm", colour_key]].dropna()
        for label in CATALOGUE_ORDER:
            group = panel.loc[panel["catalogue"] == label].sort_values("lgm")
            if group.empty:
                continue
            ax.scatter(
                group["lgm"],
                group[colour_key],
                s=1 if label == "SDSS" else 8,
                color=PALETTE[label],
                alpha=0.18 if label == "SDSS" else 0.65,
                linewidths=0,
                rasterized=(label == "SDSS"),
            )
            if len(group) >= 2:
                slope, intercept = np.polyfit(group["lgm"], group[colour_key], deg=1)
                x_fit = np.linspace(group["lgm"].quantile(0.01), group["lgm"].quantile(0.99), 100)
                ax.plot(
                    x_fit,
                    slope * x_fit + intercept,
                    linestyle="--",
                    linewidth=1.0 if label == "SDSS" else 1.4,
                    color=PALETTE[label],
                )

        ax.set_title(colour_label)
        ax.set_ylabel(axis_label)
        y_low, y_high = panel[colour_key].quantile([0.01, 0.99])
        ax.set_ylim(y_low - 0.15, y_high + 0.15)
        _style_axes(ax)

    for ax in axes[-1]:
        ax.set_xlabel(r"$\log(M_\star/M_\odot)$")

    handles = [
        Line2D([0], [0], color=PALETTE[label], linestyle="--", linewidth=1.4, label=label)
        for label in CATALOGUE_ORDER
    ]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_satellite_environment_relations(
    satellite_frame: pd.DataFrame,
    output_path: str,
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.6), sharex=True)

    for ax, (colour_key, colour_label, axis_label) in zip(axes.flat, COLOUR_SPECS):
        panel = satellite_frame[["environment", "lgm", colour_key]].dropna()
        for label in ["Ordinary", "Compact"]:
            group = panel.loc[panel["environment"] == label].sort_values("lgm")
            ax.scatter(
                group["lgm"],
                group[colour_key],
                s=5 if label == "Ordinary" else 10,
                color=PALETTE[label],
                alpha=0.22 if label == "Ordinary" else 0.7,
                linewidths=0,
                rasterized=True,
            )
            if len(group) >= 2:
                slope, intercept = np.polyfit(group["lgm"], group[colour_key], deg=1)
                x_fit = np.linspace(group["lgm"].quantile(0.01), group["lgm"].quantile(0.99), 100)
                ax.plot(
                    x_fit,
                    slope * x_fit + intercept,
                    linestyle="--",
                    linewidth=1.5,
                    color=PALETTE[label],
                )

        ax.set_title(colour_label)
        ax.set_ylabel(axis_label)
        y_low, y_high = panel[colour_key].quantile([0.01, 0.99])
        ax.set_ylim(y_low - 0.15, y_high + 0.15)
        _style_axes(ax)

    for ax in axes[-1]:
        ax.set_xlabel(r"$\log(M_\star/M_\odot)$")
    axes[0, 0].legend(
        handles=[
            Line2D([0], [0], color=PALETTE[label], linestyle="--", label=label)
            for label in ["Compact", "Ordinary"]
        ],
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_bgg_satellite_colours(group_pairs: pd.DataFrame, output_path: str) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.6))

    for ax, (colour_key, colour_label, axis_label) in zip(axes.flat, COLOUR_SPECS):
        x_key = f"bgg_{colour_key}"
        y_key = f"sat_mean_{colour_key}"
        panel = group_pairs[["catalogue", x_key, y_key]].dropna()
        values = pd.concat([panel[x_key], panel[y_key]], ignore_index=True)
        low, high = values.quantile([0.01, 0.99])
        ax.plot([low, high], [low, high], color="0.65", linestyle=":", linewidth=1)

        for label in CATALOGUE_ORDER[:-1]:
            group = panel.loc[panel["catalogue"] == label].sort_values(x_key)
            if group.empty:
                continue
            ax.scatter(
                group[x_key],
                group[y_key],
                s=14,
                color=PALETTE[label],
                alpha=0.72,
                linewidths=0,
            )
            if len(group) >= 3:
                slope, intercept = np.polyfit(group[x_key], group[y_key], deg=1)
                x_fit = np.linspace(group[x_key].min(), group[x_key].max(), 100)
                ax.plot(x_fit, slope * x_fit + intercept, "--", color=PALETTE[label], linewidth=1.2)

        ax.set_title(colour_label)
        ax.set_xlabel(f"BGG {axis_label}")
        ax.set_ylabel(f"Mean satellite {axis_label}")
        ax.set_xlim(low - 0.1, high + 0.1)
        ax.set_ylim(low - 0.1, high + 0.1)
        _style_axes(ax)

    axes[0, 0].legend(
        handles=[
            Line2D([0], [0], color=PALETTE[label], marker="o", linestyle="--", label=label)
            for label in CATALOGUE_ORDER[:-1]
        ],
        frameon=False,
        fontsize=7,
    )
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_bgg_domination_relations(bgg_frame: pd.DataFrame, output_path: str) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.6), sharex=True)

    for ax, (colour_key, colour_label, axis_label) in zip(axes.flat, COLOUR_SPECS):
        panel = bgg_frame[["lgm", colour_key, "domination"]].dropna()
        for label in ["Non-dominated", "Dominated"]:
            group = panel.loc[panel["domination"] == label].sort_values("lgm")
            ax.scatter(
                group["lgm"],
                group[colour_key],
                s=13,
                color=PALETTE[label],
                alpha=0.7,
                linewidths=0,
            )
            if len(group) >= 3:
                slope, intercept = np.polyfit(group["lgm"], group[colour_key], deg=1)
                x_fit = np.linspace(group["lgm"].min(), group["lgm"].max(), 100)
                ax.plot(x_fit, slope * x_fit + intercept, "--", color=PALETTE[label], linewidth=1.4)

        ax.set_title(colour_label)
        ax.set_ylabel(axis_label)
        y_low, y_high = panel[colour_key].quantile([0.01, 0.99])
        ax.set_ylim(y_low - 0.12, y_high + 0.12)
        _style_axes(ax)

    for ax in axes[-1]:
        ax.set_xlabel(r"$\log(M_\star/M_\odot)$")
    axes[0, 0].legend(
        handles=[
            Line2D([0], [0], color=PALETTE[label], marker="o", linestyle="--", label=label)
            for label in ["Dominated", "Non-dominated"]
        ],
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    return _save_figure(fig, output_path)


def _format_records(frame: pd.DataFrame, p_columns: list[str] | None = None) -> list[dict[str, object]]:
    p_columns = p_columns or []
    records = []
    for row in frame.to_dict(orient="records"):
        record = {}
        for key, value in row.items():
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                record[key] = None
            elif isinstance(value, (np.integer, np.floating, np.bool_)):
                record[key] = value.item()
            else:
                record[key] = value
        for key in p_columns:
            record[f"{key}_fmt"] = _format_p(record.get(key))
        for key in [
            "slope",
            "intercept",
            "delta_slope",
            "ordinary_slope",
            "compact_slope",
            "compact_minus_ordinary",
            "pearson_r",
            "spearman_rho",
            "cg4_slope",
            "comparison_slope",
            "median_log_mass",
            "mean_log_mass",
            "median_dominated",
            "median_non_dominated",
            "delta_median",
            "slope_non_dominated",
            "slope_dominated",
            "delta_at_reference_mass",
            "delta_ci_low",
            "delta_ci_high",
        ]:
            if key in record and record[key] is not None:
                record[f"{key}_fmt"] = f"{record[key]:.3f}"
        if "compact_minus_ordinary" in record and record["compact_minus_ordinary"] is not None:
            offset = record["compact_minus_ordinary"]
            record["offset_abs_fmt"] = f"{abs(offset):.3f}"
            record["offset_direction"] = "redder" if offset > 0 else "bluer" if offset < 0 else "unchanged"
        records.append(record)
    return records


def run(sample: dict[str, pd.DataFrame], output_dir: str | None = None) -> dict[str, object]:
    """Run the colour exploration, save its products, and populate results.json."""

    output_dir = co.FIGURES_PATH if output_dir is None else output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(co.OUTPUT_PATH, exist_ok=True)

    lookup = build_sdss_colour_lookup(sample)
    catalogue_frame, matching = build_catalogue_colour_frame(sample, lookup)
    slopes, global_tests, pairwise_tests = compute_catalogue_colour_mass_tests(catalogue_frame)

    satellite_frame = build_satellite_colour_frame(sample, lookup)
    satellite_tests, satellite_reference_mass = compute_satellite_environment_tests(satellite_frame)

    group_pairs = build_group_colour_pairs(sample, lookup)
    correlations = compute_bgg_satellite_correlations(group_pairs)
    slope_tests = compute_bgg_satellite_slope_tests(group_pairs)

    bgg_frame = build_bgg_colour_frame(sample, lookup)
    mass_summary, mass_test, raw_domination, adjusted_domination, bgg_reference_mass = (
        compute_bgg_domination_tests(bgg_frame)
    )

    tables = {
        "colour_matching_summary.csv": matching,
        "colour_mass_slopes.csv": slopes,
        "colour_mass_global_tests.csv": global_tests,
        "colour_mass_pairwise_tests.csv": pairwise_tests,
        "colour_satellite_environment_tests.csv": satellite_tests,
        "colour_bgg_satellite_correlations.csv": correlations,
        "colour_bgg_satellite_slope_tests.csv": slope_tests,
        "colour_bgg_domination_mass_summary.csv": mass_summary,
        "colour_bgg_domination_raw_tests.csv": raw_domination,
        "colour_bgg_domination_adjusted_tests.csv": adjusted_domination,
    }
    for filename, frame in tables.items():
        frame.to_csv(os.path.join(co.OUTPUT_PATH, filename), index=False)

    figure_names = {
        "colour_mass": "colour_mass_relations.pdf",
        "satellite_environment": "satellite_colour_mass_environment.pdf",
        "bgg_satellite": "bgg_satellite_colours.pdf",
        "bgg_domination": "bgg_colour_domination.pdf",
    }
    figures = {
        "colour_mass": plot_colour_mass_relations(
            catalogue_frame,
            os.path.join(output_dir, figure_names["colour_mass"]),
        ),
        "satellite_environment": plot_satellite_environment_relations(
            satellite_frame,
            os.path.join(output_dir, figure_names["satellite_environment"]),
        ),
        "bgg_satellite": plot_bgg_satellite_colours(
            group_pairs,
            os.path.join(output_dir, figure_names["bgg_satellite"]),
        ),
        "bgg_domination": plot_bgg_domination_relations(
            bgg_frame,
            os.path.join(output_dir, figure_names["bgg_domination"]),
        ),
    }

    matching_records = _format_records(matching)
    for record in matching_records:
        record["matched_fraction_fmt"] = f"{100 * record['matched_fraction']:.1f}"

    global_records = _format_records(global_tests, ["p_value"])
    for record in global_records:
        record["significant"] = bool(record["p_value"] < co.P_LIMIT)

    pairwise_records = _format_records(pairwise_tests, ["p_one_sided", "p_holm"])
    satellite_records = _format_records(
        satellite_tests,
        ["interaction_p_raw", "difference_p_raw", "interaction_p_holm", "difference_p_holm"],
    )
    correlation_records = _format_records(
        correlations,
        ["pearson_p", "spearman_p", "pearson_p_holm", "spearman_p_holm"],
    )
    slope_test_records = _format_records(slope_tests, ["p_raw", "p_holm"])
    mass_records = _format_records(mass_summary)
    mass_test_record = {
        **mass_test,
        "p_value_fmt": _format_p(mass_test["p_value"]),
    }
    raw_records = _format_records(raw_domination, ["p_raw", "p_holm"])
    adjusted_records = _format_records(
        adjusted_domination,
        ["slope_difference_p", "offset_p_raw", "offset_p_holm"],
    )

    report.append_json("Colour_matching_summary", matching_records)
    report.append_json("Colour_mass_slopes", _format_records(slopes))
    report.append_json("Colour_mass_global_tests", global_records)
    report.append_json(
        "Colour_mass_significant_global_tests",
        [row for row in global_records if row["significant"]],
    )
    report.append_json("Colour_mass_pairwise_tests", pairwise_records)
    report.append_json(
        "Colour_mass_significant_pairwise_tests",
        [row for row in pairwise_records if row["significant"]],
    )
    report.append_json("Colour_satellite_reference_log_mass", satellite_reference_mass)
    report.append_json("Colour_satellite_reference_log_mass_fmt", f"{satellite_reference_mass:.3f}")
    report.append_json("Colour_satellite_environment_tests", satellite_records)
    report.append_json(
        "Colour_satellite_significant_offsets",
        [row for row in satellite_records if row["difference_significant"]],
    )
    report.append_json(
        "Colour_satellite_significant_slope_differences",
        [row for row in satellite_records if row["interaction_significant"]],
    )
    report.append_json("Colour_bgg_satellite_correlations", correlation_records)
    report.append_json(
        "Colour_bgg_satellite_overall_correlations",
        [row for row in correlation_records if row["sample"] == "All"],
    )
    report.append_json("Colour_bgg_satellite_slope_tests", slope_test_records)
    report.append_json(
        "Colour_bgg_satellite_significant_slope_tests",
        [row for row in slope_test_records if row["significant"]],
    )
    report.append_json(
        "Colour_bgg_satellite_significant_slope_count",
        sum(bool(row["significant"]) for row in slope_test_records),
    )
    report.append_json("Colour_bgg_domination_mass_summary", mass_records)
    report.append_json("Colour_bgg_domination_mass_test", mass_test_record)
    report.append_json("Colour_bgg_domination_raw_tests", raw_records)
    report.append_json(
        "Colour_bgg_domination_significant_raw",
        [row for row in raw_records if row["significant"]],
    )
    report.append_json("Colour_bgg_reference_log_mass", bgg_reference_mass)
    report.append_json("Colour_bgg_reference_log_mass_fmt", f"{bgg_reference_mass:.3f}")
    report.append_json("Colour_bgg_domination_adjusted_tests", adjusted_records)
    report.append_json(
        "Colour_bgg_domination_significant_adjusted",
        [row for row in adjusted_records if row["significant"]],
    )
    report.append_json("Colour_figures", figure_names)

    return {
        "matching": matching,
        "catalogue_slopes": slopes,
        "catalogue_global_tests": global_tests,
        "catalogue_pairwise_tests": pairwise_tests,
        "satellite_tests": satellite_tests,
        "bgg_satellite_correlations": correlations,
        "bgg_satellite_slope_tests": slope_tests,
        "bgg_domination_raw": raw_domination,
        "bgg_domination_adjusted": adjusted_domination,
        "figures": figures,
    }
