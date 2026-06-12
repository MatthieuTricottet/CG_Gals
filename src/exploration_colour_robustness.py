"""Exploratory robustness tests for optical colours in compact groups.

This module is intentionally notebook-oriented: functions return tidy tables,
models, and matplotlib figures, while missing inputs produce explicit messages
instead of hard failures.  It does not write report JSON or overwrite the
production colour analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


SAMPLE_KEYS = {
    "CG4": "CG4_Gals",
    "Control4B": "Control4B_Gals",
    "Control4C": "Control4C_Gals",
    "RG4": "RG4_Gals",
}
GROUP_KEYS = {label: key.replace("_Gals", "_Groups") for label, key in SAMPLE_KEYS.items()}
SAMPLE_ORDER = list(SAMPLE_KEYS)
CONTROL_SAMPLES = ["Control4B", "Control4C", "RG4"]
COLOUR_COLUMNS = ["u_minus_r", "u_minus_g", "g_minus_r", "r_minus_i"]
COLOUR_LABELS = {
    "u_minus_r": r"$(u-r)$",
    "u_minus_g": r"$(u-g)$",
    "g_minus_r": r"$(g-r)$",
    "r_minus_i": r"$(r-i)$",
}
PALETTE = {
    "CG4": "#2864A6",
    "Control4B": "#D17A22",
    "Control4C": "#25876E",
    "RG4": "#A74752",
    "Ordinary": "#555555",
}
GROUP_QUANTITIES = [
    "t_cr",
    "Vdisp",
    "Lum_group",
    "M_virial",
    "M_virial_over_L",
    "is_dominated",
    "Offset_Bary",
    "Voffset",
]


def _message(text: str) -> None:
    print(f"[colour exploration] {text}")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _as_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _coef_value(result, term: str, attribute: str = "params") -> float:
    values = getattr(result, attribute)
    if hasattr(values, "index") and term in values.index:
        return float(values.loc[term])
    names = list(getattr(result.model, "exog_names", []))
    if term not in names:
        return np.nan
    return float(np.asarray(values)[names.index(term)])


def _coef_ci(result, term: str, alpha: float = 0.05) -> tuple[float, float]:
    names = list(getattr(result.model, "exog_names", []))
    if term not in names:
        return np.nan, np.nan
    ci = np.asarray(result.conf_int(alpha=alpha))
    index = names.index(term)
    return float(ci[index, 0]), float(ci[index, 1])


def _bootstrap_median_ci(
    values,
    confidence: float = 0.95,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    values = _finite(values)
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(20260612) if rng is None else rng
    medians = np.empty(n_boot)
    for index in range(n_boot):
        medians[index] = np.median(rng.choice(values, size=values.size, replace=True))
    alpha = 1 - confidence
    low, high = np.quantile(medians, [alpha / 2, 1 - alpha / 2])
    return float(np.median(values)), float(low), float(high)


def bootstrap_median_diff(
    x,
    y,
    n_boot: int = 2000,
    random_state: int = 20260612,
) -> dict[str, float]:
    """Bootstrap median(x)-median(y), with 68% and 95% intervals."""

    x = _finite(x)
    y = _finite(y)
    if x.size == 0 or y.size == 0:
        return {
            "median_diff": np.nan,
            "ci68_low": np.nan,
            "ci68_high": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
    rng = np.random.default_rng(random_state)
    diffs = np.empty(n_boot)
    for index in range(n_boot):
        diffs[index] = np.median(rng.choice(x, size=x.size, replace=True)) - np.median(
            rng.choice(y, size=y.size, replace=True)
        )
    q = np.quantile(diffs, [0.025, 0.16, 0.84, 0.975])
    return {
        "median_diff": float(np.median(x) - np.median(y)),
        "ci68_low": float(q[1]),
        "ci68_high": float(q[2]),
        "ci95_low": float(q[0]),
        "ci95_high": float(q[3]),
    }


def safe_mannwhitney(x, y, alternative: str = "two-sided") -> float:
    """Return a Mann-Whitney p-value, or NaN for an unusable comparison."""

    x = _finite(x)
    y = _finite(y)
    if x.size == 0 or y.size == 0:
        return np.nan
    try:
        return float(
            stats.mannwhitneyu(
                x,
                y,
                alternative=alternative,
                method="asymptotic",
            ).pvalue
        )
    except ValueError:
        return np.nan


def cliffs_delta(x, y) -> float:
    """Cliff's delta; negative values mean x tends to be smaller than y."""

    x = _finite(x)
    y = _finite(y)
    if x.size == 0 or y.size == 0:
        return np.nan
    u_statistic = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic").statistic
    return float(2 * u_statistic / (x.size * y.size) - 1)


def permutation_median_pvalue(
    x,
    y,
    n_perm: int = 2000,
    random_state: int = 20260612,
) -> float:
    x = _finite(x)
    y = _finite(y)
    if x.size == 0 or y.size == 0:
        return np.nan
    observed = abs(np.median(x) - np.median(y))
    joined = np.concatenate([x, y])
    rng = np.random.default_rng(random_state)
    exceedances = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(joined)
        difference = abs(np.median(shuffled[: x.size]) - np.median(shuffled[x.size :]))
        exceedances += difference >= observed
    return float((exceedances + 1) / (n_perm + 1))


def fit_ols_with_optional_cluster_se(
    formula: str,
    data: pd.DataFrame,
    group_col: str | None = "cluster_id",
    min_groups: int = 8,
):
    """Fit OLS and use cluster-robust errors when enough groups are present."""

    try:
        model = smf.ols(formula, data=data, missing="drop")
        row_labels = model.data.row_labels
        if group_col and group_col in data.columns:
            groups = data.loc[row_labels, group_col]
            valid_groups = groups.dropna().nunique()
            if valid_groups >= min_groups and groups.notna().all():
                return model.fit(cov_type="cluster", cov_kwds={"groups": groups})
        return model.fit(cov_type="HC3")
    except Exception as error:
        _message(f"Skipping model '{formula}': {error}")
        return None


def _sdss_photometry_lookup(sample: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "SDSS" not in sample:
        raise KeyError("sample['SDSS'] is required to match optical photometry.")
    required = {"objid", "u_obs", "g_obs", "r_obs", "i_obs"}
    missing = sorted(required - set(sample["SDSS"].columns))
    if missing:
        raise KeyError(f"SDSS photometry is missing columns: {missing}")

    lookup = sample["SDSS"].copy()
    lookup["_objid"] = _as_int(lookup["objid"])
    for column in ["u_obs", "g_obs", "r_obs", "i_obs"]:
        lookup[column] = _numeric(lookup[column])
    lookup["u_minus_r"] = lookup["u_obs"] - lookup["r_obs"]
    lookup["u_minus_g"] = lookup["u_obs"] - lookup["g_obs"]
    lookup["g_minus_r"] = lookup["g_obs"] - lookup["r_obs"]
    lookup["r_minus_i"] = lookup["r_obs"] - lookup["i_obs"]
    columns = ["_objid", "u_obs", "g_obs", "r_obs", "i_obs", *COLOUR_COLUMNS]
    return lookup[columns].dropna(subset=["_objid"]).drop_duplicates("_objid")


def _attach_distances(galaxies: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    required_gal = {"Group", "z", "dist2BGG"}
    required_group = {"Group", "size_Group_Bary_kpc"}
    if not required_gal.issubset(galaxies.columns):
        _message("Projected/normalized BGG distances skipped: galaxy columns are incomplete.")
        return galaxies
    if not required_group.issubset(groups.columns):
        _message("Normalized BGG distance skipped: group scale is unavailable.")
        return galaxies
    try:
        from exploration_ssfr import add_normalized_group_distances

        return add_normalized_group_distances(galaxies, groups)
    except Exception as error:
        _message(f"Projected/normalized BGG distances skipped: {error}")
        return galaxies


def build_harmonized_colour_frame(
    sample: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build one left-matched galaxy frame retaining colour non-matches."""

    lookup = _sdss_photometry_lookup(sample)
    frames = []

    for label, galaxy_key in SAMPLE_KEYS.items():
        if galaxy_key not in sample:
            _message(f"{label} skipped: sample['{galaxy_key}'] is missing.")
            continue
        galaxies = sample[galaxy_key].copy()
        if "objid" not in galaxies:
            _message(f"{label} skipped: objid is missing.")
            continue

        group_key = GROUP_KEYS[label]
        groups = sample.get(group_key, pd.DataFrame())
        galaxies = _attach_distances(galaxies, groups) if not groups.empty else galaxies
        galaxies["_objid"] = _as_int(galaxies["objid"])
        overlap = [column for column in lookup.columns if column != "_objid" and column in galaxies]
        if overlap:
            galaxies = galaxies.drop(columns=overlap)
        merged = galaxies.merge(lookup, on="_objid", how="left", validate="m:1")

        if not groups.empty and "Group" in merged and "Group" in groups:
            group_columns = [
                column
                for column in groups.columns
                if column == "Group" or column not in merged.columns
            ]
            group_frame = groups[group_columns].drop_duplicates("Group")
            merged = merged.merge(group_frame, on="Group", how="left", validate="m:1")

        rank_column = "rank_M" if "rank_M" in merged else "rank_M_CG"
        merged["sample"] = label
        merged["is_CG4"] = label == "CG4"
        merged["is_control"] = label in CONTROL_SAMPLES[:2]
        merged["is_RG4"] = label == "RG4"
        merged["is_satellite"] = (
            _numeric(merged[rank_column]).gt(1) if rank_column in merged else pd.NA
        )
        merged["is_BGG"] = (
            _numeric(merged[rank_column]).eq(1) if rank_column in merged else pd.NA
        )
        merged["logM"] = _numeric(merged["lgm"]) if "lgm" in merged else np.nan
        merged["z_harmonized"] = _numeric(merged["z"]) if "z" in merged else np.nan
        merged["sSFR_harmonized"] = _numeric(merged["sSFR"]) if "sSFR" in merged else np.nan
        merged["sSFR_class"] = (
            merged["sSFR_status"].astype("string") if "sSFR_status" in merged else pd.NA
        )
        merged["morphology_harmonized"] = (
            merged["morphology"].astype("string") if "morphology" in merged else pd.NA
        )
        merged["group_id"] = merged["Group"] if "Group" in merged else pd.NA
        merged["cluster_id"] = label + "_" + merged["group_id"].astype("string")
        merged["has_colour"] = merged[COLOUR_COLUMNS].notna().all(axis=1)
        frames.append(merged)

    if not frames:
        raise ValueError("No galaxy catalogues could be harmonized.")
    result = pd.concat(frames, ignore_index=True, sort=False)
    result["ordinary_sample"] = np.where(result["is_CG4"], "CG4", "Ordinary")
    return result


def colour_missingness_table(frame: pd.DataFrame, small_n: int = 20) -> pd.DataFrame:
    rows = []
    for label in SAMPLE_ORDER:
        part = frame.loc[frame["sample"] == label]
        satellites = part.loc[part["is_satellite"].fillna(False)]
        row = {
            "sample": label,
            "n_galaxies": int(len(part)),
            "n_all_four_colours": int(part["has_colour"].sum()),
            "n_logM": int(part["logM"].notna().sum()),
            "n_z": int(part["z_harmonized"].notna().sum()),
            "n_sSFR_class": int(part["sSFR_class"].notna().sum()),
            "n_satellites_with_colours": int(satellites["has_colour"].sum()),
        }
        rows.append(row)
        if label == "CG4" and row["n_satellites_with_colours"] < small_n:
            warnings.warn(
                "CG4 satellite colour statistics use fewer than "
                f"{small_n} galaxies (N={row['n_satellites_with_colours']}).",
                RuntimeWarning,
            )
    return pd.DataFrame(rows)


def matching_bias_tests(
    frame: pd.DataFrame,
    n_boot: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare colour-matched and unmatched galaxies within each sample."""

    continuous_rows = []
    categorical_rows = []
    rng = np.random.default_rng(20260612)
    continuous = {
        "logM": "logM",
        "z": "z_harmonized",
        "sSFR": "sSFR_harmonized",
        "M_r": "M_r",
    }
    categorical = {
        "morphology": "morphology_harmonized",
        "sSFR_class": "sSFR_class",
        "galaxy_role": "galaxy_role",
    }
    work = frame.copy()
    work["galaxy_role"] = np.select(
        [work["is_BGG"].fillna(False), work["is_satellite"].fillna(False)],
        ["BGG", "Satellite"],
        default="Unknown",
    )

    for label in SAMPLE_ORDER:
        part = work.loc[work["sample"] == label]
        for display_name, column in continuous.items():
            if column not in part:
                _message(f"Matching-bias test skips {display_name}: column unavailable.")
                continue
            matched = _finite(part.loc[part["has_colour"], column])
            unmatched = _finite(part.loc[~part["has_colour"], column])
            if matched.size == 0 or unmatched.size == 0:
                continue
            med_m, lo_m, hi_m = _bootstrap_median_ci(matched, n_boot=n_boot, rng=rng)
            med_u, lo_u, hi_u = _bootstrap_median_ci(unmatched, n_boot=n_boot, rng=rng)
            pooled_iqr = stats.iqr(np.concatenate([matched, unmatched]), nan_policy="omit")
            continuous_rows.append(
                {
                    "sample": label,
                    "variable": display_name,
                    "n_matched": int(matched.size),
                    "n_unmatched": int(unmatched.size),
                    "median_matched": med_m,
                    "median_matched_ci_low": lo_m,
                    "median_matched_ci_high": hi_m,
                    "median_unmatched": med_u,
                    "median_unmatched_ci_low": lo_u,
                    "median_unmatched_ci_high": hi_u,
                    "median_diff_over_iqr": (
                        float((med_m - med_u) / pooled_iqr) if pooled_iqr > 0 else np.nan
                    ),
                    "mannwhitney_p": safe_mannwhitney(matched, unmatched),
                    "cliffs_delta": cliffs_delta(matched, unmatched),
                }
            )

        for display_name, column in categorical.items():
            panel = part[["has_colour", column]].dropna()
            table = pd.crosstab(panel["has_colour"], panel[column])
            if table.shape[0] != 2 or table.shape[1] < 2:
                continue
            if table.shape == (2, 2):
                _, p_value = stats.fisher_exact(table.to_numpy())
                test_name = "Fisher exact"
            else:
                _, p_value, _, _ = stats.chi2_contingency(table.to_numpy())
                test_name = "chi-square"
            chi2 = stats.chi2_contingency(table.to_numpy(), correction=False)[0]
            denominator = table.to_numpy().sum() * min(table.shape[0] - 1, table.shape[1] - 1)
            categorical_rows.append(
                {
                    "sample": label,
                    "variable": display_name,
                    "n": int(table.to_numpy().sum()),
                    "test": test_name,
                    "p_value": float(p_value),
                    "cramers_v": float(np.sqrt(chi2 / denominator)) if denominator else np.nan,
                    "matched_fractions": (
                        panel.loc[panel["has_colour"], column].value_counts(normalize=True).to_dict()
                    ),
                    "unmatched_fractions": (
                        panel.loc[~panel["has_colour"], column].value_counts(normalize=True).to_dict()
                    ),
                }
            )
    return pd.DataFrame(continuous_rows), pd.DataFrame(categorical_rows)


def fit_colour_availability_model(frame: pd.DataFrame):
    required = [
        "has_colour",
        "sample",
        "logM",
        "z_harmonized",
        "sSFR_class",
        "morphology_harmonized",
        "is_satellite",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        _message(f"Colour-availability logistic model skipped; missing {missing}.")
        return None, pd.DataFrame()
    panel = frame[required].dropna().copy()
    panel["has_colour"] = panel["has_colour"].astype(int)
    panel["is_satellite"] = panel["is_satellite"].astype(int)
    if len(panel) < 50 or panel["has_colour"].nunique() < 2:
        _message("Colour-availability logistic model skipped: insufficient outcome variation.")
        return None, pd.DataFrame()
    formula = (
        "has_colour ~ C(sample) + logM + z_harmonized + "
        "C(sSFR_class) + C(morphology_harmonized) + is_satellite"
    )
    try:
        result = smf.glm(
            formula,
            data=panel,
            family=sm.families.Binomial(),
        ).fit(cov_type="HC3")
    except Exception as error:
        _message(f"Colour-availability logistic model skipped: {error}")
        return None, pd.DataFrame()
    ci = result.conf_int()
    table = pd.DataFrame(
        {
            "term": result.params.index,
            "coefficient": result.params.values,
            "odds_ratio": np.exp(result.params.values),
            "ci_low": np.exp(ci.iloc[:, 0].values),
            "ci_high": np.exp(ci.iloc[:, 1].values),
            "p_value": result.pvalues.values,
        }
    )
    return result, table


def plot_matching_bias(frame: pd.DataFrame):
    work = frame.copy()
    work["colour_availability"] = np.where(work["has_colour"], "Matched", "Unmatched")
    fig_cont, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, label in zip(axes.flat, SAMPLE_ORDER):
        part = work.loc[work["sample"] == label]
        sns.histplot(
            data=part,
            x="logM",
            hue="colour_availability",
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            ax=ax,
        )
        ax.set_title(f"{label}: stellar mass")
        ax.set_xlabel(r"$\log(M_\star/M_\odot)$")
    fig_cont.tight_layout()

    fig_z, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, label in zip(axes.flat, SAMPLE_ORDER):
        part = work.loc[work["sample"] == label]
        sns.histplot(
            data=part,
            x="z_harmonized",
            hue="colour_availability",
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            ax=ax,
        )
        ax.set_title(f"{label}: redshift")
        ax.set_xlabel("Redshift")
    fig_z.tight_layout()

    categorical = []
    for variable, column in [
        ("Morphology", "morphology_harmonized"),
        ("sSFR class", "sSFR_class"),
    ]:
        counts = (
            work.dropna(subset=[column])
            .groupby(["sample", "colour_availability", column], observed=True)
            .size()
            .rename("n")
            .reset_index()
        )
        counts["fraction"] = counts["n"] / counts.groupby(
            ["sample", "colour_availability"], observed=True
        )["n"].transform("sum")
        counts["variable"] = variable
        counts["category"] = counts[column].astype(str)
        counts["sample_subset"] = (
            counts["sample"].astype(str) + "\n" + counts["colour_availability"].astype(str)
        )
        categorical.append(counts)
    bar_data = pd.concat(categorical, ignore_index=True)
    fig_bar, axes = plt.subplots(2, 1, figsize=(12, 8))
    for ax, variable in zip(axes, ["Morphology", "sSFR class"]):
        sns.barplot(
            data=bar_data.loc[bar_data["variable"] == variable],
            x="sample_subset",
            y="fraction",
            hue="category",
            errorbar=None,
            ax=ax,
        )
        ax.set_title(f"{variable} fractions in matched and unmatched subsets")
        ax.set_ylabel("Fraction")
        ax.set_xlabel("")
        ax.legend(title=variable, frameon=False, ncol=3)
    fig_bar.tight_layout()
    return fig_cont, fig_z, fig_bar


@dataclass
class ResidualModels:
    baseline: dict[str, object]
    full: dict[str, object]


def make_colour_residuals(
    frame: pd.DataFrame,
    reference_samples: list[str] | None = None,
    satellites_only: bool = True,
) -> tuple[pd.DataFrame, ResidualModels]:
    """Fit ordinary-satellite colour relations and add baseline/full residuals."""

    reference_samples = CONTROL_SAMPLES if reference_samples is None else reference_samples
    result = frame.copy()
    reference = result["sample"].isin(reference_samples)
    if satellites_only:
        reference &= result["is_satellite"].fillna(False)
    baseline_models: dict[str, object] = {}
    full_models: dict[str, object] = {}

    for colour in COLOUR_COLUMNS:
        base_columns = [colour, "logM", "z_harmonized"]
        training = result.loc[reference, base_columns].dropna()
        if len(training) < 30:
            _message(f"{colour} residuals skipped: fewer than 30 reference galaxies.")
            result[f"delta_{colour}"] = np.nan
            result[f"delta_{colour}_full"] = np.nan
            continue
        base_model = smf.ols(
            f"{colour} ~ logM + z_harmonized",
            data=training,
        ).fit(cov_type="HC3")
        baseline_models[colour] = base_model
        valid = result[base_columns].notna().all(axis=1)
        result[f"delta_{colour}"] = np.nan
        result.loc[valid, f"delta_{colour}"] = (
            result.loc[valid, colour] - base_model.predict(result.loc[valid])
        )

        full_columns = base_columns + ["morphology_harmonized", "sSFR_class"]
        full_training = result.loc[reference, full_columns].dropna()
        result[f"delta_{colour}_full"] = np.nan
        if (
            len(full_training) < 50
            or full_training["morphology_harmonized"].nunique() < 2
            or full_training["sSFR_class"].nunique() < 2
        ):
            _message(f"Full {colour} residual model skipped: insufficient category coverage.")
            continue
        full_model = smf.ols(
            f"{colour} ~ logM + z_harmonized + "
            "C(morphology_harmonized) + C(sSFR_class)",
            data=full_training,
        ).fit(cov_type="HC3")
        full_models[colour] = full_model
        valid_full = result[full_columns].notna().all(axis=1)
        try:
            result.loc[valid_full, f"delta_{colour}_full"] = (
                result.loc[valid_full, colour] - full_model.predict(result.loc[valid_full])
            )
        except Exception as error:
            _message(f"Full residual prediction skipped for {colour}: {error}")

    return result, ResidualModels(baseline=baseline_models, full=full_models)


def compare_satellite_residuals(
    frame: pd.DataFrame,
    split_column: str | None = None,
    n_boot: int = 1000,
    n_perm: int = 1000,
) -> pd.DataFrame:
    """Compare CG4 satellite residuals with pooled and individual controls."""

    satellites = frame.loc[frame["is_satellite"].fillna(False)].copy()
    split_values = ["All"] if split_column is None else list(
        satellites[split_column].dropna().astype(str).unique()
    )
    rows = []
    comparisons = [("Ordinary pooled", CONTROL_SAMPLES)] + [
        (label, [label]) for label in CONTROL_SAMPLES
    ]
    for split_value in split_values:
        current = satellites
        if split_column is not None:
            current = current.loc[current[split_column].astype(str) == split_value]
        for colour in COLOUR_COLUMNS:
            residual = f"delta_{colour}"
            if residual not in current:
                continue
            compact = current.loc[current["sample"] == "CG4", residual].dropna()
            for comparison_label, comparison_samples in comparisons:
                ordinary = current.loc[current["sample"].isin(comparison_samples), residual].dropna()
                if len(compact) < 3 or len(ordinary) < 3:
                    continue
                boot = bootstrap_median_diff(compact, ordinary, n_boot=n_boot)
                rows.append(
                    {
                        "split_variable": split_column or "none",
                        "split_value": split_value,
                        "colour": colour,
                        "comparison": comparison_label,
                        "n_CG4": int(len(compact)),
                        "n_ordinary": int(len(ordinary)),
                        "median_CG4": float(compact.median()),
                        "median_ordinary": float(ordinary.median()),
                        **boot,
                        "mannwhitney_p": safe_mannwhitney(compact, ordinary),
                        "permutation_p": permutation_median_pvalue(
                            compact,
                            ordinary,
                            n_perm=n_perm,
                        ),
                        "cliffs_delta": cliffs_delta(compact, ordinary),
                    }
                )
    return pd.DataFrame(rows)


def plot_satellite_residuals(frame: pd.DataFrame):
    satellites = frame.loc[frame["is_satellite"].fillna(False)].copy()
    plot_frames = []
    for label, samples in [("CG4", ["CG4"]), ("Ordinary", CONTROL_SAMPLES)]:
        part = satellites.loc[satellites["sample"].isin(samples)].copy()
        part["environment"] = label
        plot_frames.append(part)
    plot_data = pd.concat(plot_frames, ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, colour in zip(axes.flat, COLOUR_COLUMNS):
        residual = f"delta_{colour}"
        sns.histplot(
            data=plot_data,
            x=residual,
            hue="environment",
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            palette={"CG4": PALETTE["CG4"], "Ordinary": PALETTE["Ordinary"]},
            ax=ax,
        )
        ax.axvline(0, color="0.4", linestyle=":", linewidth=1)
        ax.set_title(f"Satellite residual {COLOUR_LABELS[colour]}")
        ax.set_xlabel(r"$\Delta$ colour (mag)")
    fig.tight_layout()
    return fig


def satellite_colour_regressions(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit base and sSFR/morphology-adjusted satellite colour models."""

    satellites = frame.loc[frame["is_satellite"].fillna(False)].copy()
    rows = []
    comparisons = [("Ordinary pooled", CONTROL_SAMPLES)] + [
        (label, [label]) for label in CONTROL_SAMPLES
    ]
    for comparison_label, comparison_samples in comparisons:
        panel = satellites.loc[
            satellites["sample"].isin(["CG4", *comparison_samples])
        ].copy()
        panel["is_CG4_numeric"] = (panel["sample"] == "CG4").astype(int)
        for colour in COLOUR_COLUMNS:
            formulas = {
                "mass+redshift": f"{colour} ~ is_CG4_numeric + logM + z_harmonized",
                "full": (
                    f"{colour} ~ is_CG4_numeric + logM + z_harmonized + "
                    "C(morphology_harmonized) + C(sSFR_class)"
                ),
            }
            for model_name, formula in formulas.items():
                result = fit_ols_with_optional_cluster_se(formula, panel)
                if result is None or "is_CG4_numeric" not in result.model.exog_names:
                    continue
                low, high = _coef_ci(result, "is_CG4_numeric")
                rows.append(
                    {
                        "comparison": comparison_label,
                        "colour": colour,
                        "model": model_name,
                        "n": int(result.nobs),
                        "n_groups": int(panel.loc[result.model.data.row_labels, "cluster_id"].nunique()),
                        "is_CG4_coefficient": _coef_value(result, "is_CG4_numeric"),
                        "ci_low": low,
                        "ci_high": high,
                        "p_value": _coef_value(result, "is_CG4_numeric", "pvalues"),
                    }
                )
    return pd.DataFrame(rows)


def morphology_colour_regressions(frame: pd.DataFrame) -> pd.DataFrame:
    satellites = frame.loc[frame["is_satellite"].fillna(False)].copy()
    satellites["is_CG4_numeric"] = (satellites["sample"] == "CG4").astype(int)
    rows = []
    for colour in COLOUR_COLUMNS:
        formulas = {
            "morphology_adjusted": (
                f"{colour} ~ is_CG4_numeric + logM + z_harmonized + "
                "C(morphology_harmonized)"
            ),
            "morphology_interaction": (
                f"{colour} ~ is_CG4_numeric * C(morphology_harmonized) + "
                "logM + z_harmonized"
            ),
        }
        for model_name, formula in formulas.items():
            result = fit_ols_with_optional_cluster_se(formula, satellites)
            if result is None:
                continue
            for term in result.model.exog_names:
                if "is_CG4_numeric" not in term:
                    continue
                low, high = _coef_ci(result, term)
                rows.append(
                    {
                        "colour": colour,
                        "model": model_name,
                        "term": term,
                        "coefficient": _coef_value(result, term),
                        "ci_low": low,
                        "ci_high": high,
                        "p_value": _coef_value(result, term, "pvalues"),
                        "n": int(result.nobs),
                    }
                )
    return pd.DataFrame(rows)


def plot_residuals_by_category(
    frame: pd.DataFrame,
    category: str,
    title: str,
    colour: str = "u_minus_r",
):
    satellites = frame.loc[frame["is_satellite"].fillna(False)].copy()
    satellites["environment"] = np.where(satellites["is_CG4"], "CG4", "Ordinary")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    sns.boxplot(
        data=satellites,
        x=category,
        y=f"delta_{colour}",
        hue="environment",
        showfliers=False,
        palette={"CG4": PALETTE["CG4"], "Ordinary": PALETTE["Ordinary"]},
        ax=ax,
    )
    ax.axhline(0, color="0.4", linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(rf"$\Delta${COLOUR_LABELS[colour]} (mag)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def distance_colour_analysis(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    satellites = frame.loc[frame["is_satellite"].fillna(False)].copy()
    distance_columns = [
        column for column in ["dist2BGG_kpc", "norm_dist", "dist2BGG"] if column in satellites
    ]
    if not distance_columns:
        _message("Distance analysis skipped: no BGG-distance column is available.")
        return pd.DataFrame(), pd.DataFrame()
    correlation_rows = []
    model_rows = []
    for distance in distance_columns:
        for colour in COLOUR_COLUMNS[:3]:
            residual = f"delta_{colour}"
            for label in SAMPLE_ORDER:
                panel = satellites.loc[
                    satellites["sample"] == label,
                    [distance, residual],
                ].dropna()
                if len(panel) < 5 or panel[distance].nunique() < 3:
                    continue
                pearson = stats.pearsonr(panel[distance], panel[residual])
                spearman = stats.spearmanr(panel[distance], panel[residual])
                correlation_rows.append(
                    {
                        "sample": label,
                        "distance": distance,
                        "colour": colour,
                        "n": int(len(panel)),
                        "pearson_r": float(pearson.statistic),
                        "pearson_p": float(pearson.pvalue),
                        "spearman_rho": float(spearman.statistic),
                        "spearman_p": float(spearman.pvalue),
                    }
                )
                sample_model = fit_ols_with_optional_cluster_se(
                    f"{residual} ~ {distance} + logM + z_harmonized",
                    satellites.loc[satellites["sample"] == label],
                )
                if sample_model is not None and distance in sample_model.model.exog_names:
                    model_rows.append(
                        {
                            "sample": label,
                            "distance": distance,
                            "colour": colour,
                            "model": "within_sample",
                            "term": distance,
                            "coefficient": _coef_value(sample_model, distance),
                            "p_value": _coef_value(sample_model, distance, "pvalues"),
                        }
                    )

            pooled = satellites.copy()
            pooled["is_CG4_numeric"] = (pooled["sample"] == "CG4").astype(int)
            interaction = fit_ols_with_optional_cluster_se(
                f"{residual} ~ is_CG4_numeric * {distance} + logM + z_harmonized",
                pooled,
            )
            term = f"is_CG4_numeric:{distance}"
            if interaction is not None and term in interaction.model.exog_names:
                model_rows.append(
                    {
                        "sample": "Pooled interaction",
                        "distance": distance,
                        "colour": colour,
                        "model": "CG4_distance_interaction",
                        "term": term,
                        "coefficient": _coef_value(interaction, term),
                        "p_value": _coef_value(interaction, term, "pvalues"),
                    }
                )
    return pd.DataFrame(correlation_rows), pd.DataFrame(model_rows)


def plot_distance_residuals(frame: pd.DataFrame, distance: str = "dist2BGG_kpc"):
    if distance not in frame:
        _message(f"Distance plot skipped: {distance} is unavailable.")
        return None
    satellites = frame.loc[frame["is_satellite"].fillna(False)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, colour in zip(axes, COLOUR_COLUMNS[:3]):
        residual = f"delta_{colour}"
        for label in SAMPLE_ORDER:
            part = satellites.loc[satellites["sample"] == label, [distance, residual]].dropna()
            ax.scatter(
                part[distance],
                part[residual],
                s=12,
                alpha=0.45,
                color=PALETTE[label],
                label=f"{label} (N={len(part)})",
            )
        ax.axhline(0, color="0.4", linestyle=":", linewidth=1)
        ax.set_title(COLOUR_LABELS[colour])
        ax.set_xlabel(distance)
        ax.set_ylabel(r"$\Delta$ colour (mag)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Satellite colour residuals versus distance to the BGG")
    fig.tight_layout()
    return fig


def build_group_colour_summary(
    frame: pd.DataFrame,
    blue_threshold: float = -0.1,
) -> pd.DataFrame:
    residual = "delta_u_minus_r"
    required = {"sample", "group_id", "is_satellite", residual}
    if not required.issubset(frame.columns):
        _message("Group colour summary skipped: group ID or residual colour is unavailable.")
        return pd.DataFrame()
    satellites = frame.loc[frame["is_satellite"].fillna(False)].dropna(
        subset=["group_id", residual]
    )
    if satellites.empty:
        return pd.DataFrame()
    summary = (
        satellites.groupby(["sample", "group_id"], observed=True)[residual]
        .agg(
            n_satellites="size",
            mean_delta_u_minus_r="mean",
            median_delta_u_minus_r="median",
            colour_scatter="std",
        )
        .reset_index()
    )
    blue = (
        satellites.assign(is_blue=satellites[residual] < blue_threshold)
        .groupby(["sample", "group_id"], observed=True)["is_blue"]
        .mean()
        .rename("satellite_blue_fraction")
        .reset_index()
    )
    summary = summary.merge(blue, on=["sample", "group_id"], how="left")
    group_columns = [
        column for column in [*GROUP_QUANTITIES, "Class"] if column in satellites.columns
    ]
    if group_columns:
        metadata = (
            satellites.groupby(["sample", "group_id"], observed=True)[group_columns]
            .first()
            .reset_index()
        )
        summary = summary.merge(metadata, on=["sample", "group_id"], how="left")
    return summary


def _bootstrap_spearman_ci(
    x,
    y,
    n_boot: int = 1000,
    random_state: int = 20260612,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    values = []
    for _ in range(n_boot):
        indices = rng.integers(0, x.size, x.size)
        if np.unique(x[indices]).size < 2 or np.unique(y[indices]).size < 2:
            continue
        values.append(stats.spearmanr(x[indices], y[indices]).statistic)
    if not values:
        return np.nan, np.nan
    return tuple(map(float, np.quantile(values, [0.025, 0.975])))


def group_colour_correlations(
    group_summary: pd.DataFrame,
    n_boot: int = 1000,
) -> pd.DataFrame:
    if group_summary.empty:
        return pd.DataFrame()
    outcomes = [
        "mean_delta_u_minus_r",
        "median_delta_u_minus_r",
        "satellite_blue_fraction",
        "colour_scatter",
    ]
    quantities = [column for column in GROUP_QUANTITIES if column in group_summary]
    rows = []
    sample_sets = [("All", group_summary)] + [
        (label, group_summary.loc[group_summary["sample"] == label]) for label in SAMPLE_ORDER
    ]
    for sample_label, part in sample_sets:
        for outcome in outcomes:
            for quantity in quantities:
                panel = part[[outcome, quantity]].copy()
                panel[quantity] = _numeric(panel[quantity])
                panel = panel.dropna()
                if len(panel) < 5 or panel[quantity].nunique() < 3:
                    continue
                test = stats.spearmanr(panel[quantity], panel[outcome])
                low, high = _bootstrap_spearman_ci(
                    panel[quantity],
                    panel[outcome],
                    n_boot=n_boot,
                )
                rows.append(
                    {
                        "sample": sample_label,
                        "outcome": outcome,
                        "group_quantity": quantity,
                        "n_groups": int(len(panel)),
                        "spearman_rho": float(test.statistic),
                        "ci95_low": low,
                        "ci95_high": high,
                        "p_value": float(test.pvalue),
                    }
                )
    return pd.DataFrame(rows)


def group_class_summary(group_summary: pd.DataFrame) -> pd.DataFrame:
    if "Class" not in group_summary:
        _message("CG class summary skipped: Class is unavailable.")
        return pd.DataFrame()
    cg = group_summary.loc[group_summary["sample"] == "CG4"].dropna(subset=["Class"])
    if cg.empty:
        return pd.DataFrame()
    return (
        cg.groupby("Class", observed=True)
        .agg(
            n_groups=("group_id", "nunique"),
            mean_delta_u_minus_r=("mean_delta_u_minus_r", "median"),
            satellite_blue_fraction=("satellite_blue_fraction", "median"),
            colour_scatter=("colour_scatter", "median"),
        )
        .reset_index()
    )


def plot_group_colour_dynamics(group_summary: pd.DataFrame):
    if "t_cr" not in group_summary:
        _message("Group-dynamics plot skipped: crossing time is unavailable.")
        return None
    outcomes = [
        ("mean_delta_u_minus_r", r"Mean satellite $\Delta(u-r)$"),
        ("satellite_blue_fraction", "Satellite blue fraction"),
        ("colour_scatter", r"Satellite $\Delta(u-r)$ scatter"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (outcome, label_y) in zip(axes, outcomes):
        for label in SAMPLE_ORDER:
            part = group_summary.loc[group_summary["sample"] == label, ["t_cr", outcome]].dropna()
            ax.scatter(
                part["t_cr"],
                part[outcome],
                s=24,
                alpha=0.65,
                color=PALETTE[label],
                label=label,
            )
        ax.set_xlabel("Crossing time")
        ax.set_ylabel(label_y)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Group-level satellite colour diagnostics")
    fig.tight_layout()
    return fig


def colour_ssfr_outliers(
    frame: pd.DataFrame,
    blue_threshold: float = -0.1,
    red_threshold: float = 0.1,
) -> pd.DataFrame:
    satellites = frame.loc[frame["is_satellite"].fillna(False)].copy()
    residual = satellites["delta_u_minus_r"]
    conditions = {
        "blue + low sSFR": (residual < blue_threshold)
        & satellites["sSFR_class"].isin(["Quenched", "Passive"]),
        "red + high sSFR": (residual > red_threshold)
        & satellites["sSFR_class"].eq("Starforming"),
        "blue elliptical": (residual < blue_threshold)
        & satellites["morphology_harmonized"].eq("Elliptical"),
        "red spiral": (residual > red_threshold)
        & satellites["morphology_harmonized"].eq("Spiral"),
    }
    rows = []
    for label in SAMPLE_ORDER:
        sample_mask = satellites["sample"].eq(label)
        denominator = int(sample_mask.sum())
        for outlier_class, condition in conditions.items():
            count = int((sample_mask & condition.fillna(False)).sum())
            rows.append(
                {
                    "sample": label,
                    "outlier_class": outlier_class,
                    "count": count,
                    "fraction_of_satellites": count / denominator if denominator else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_colour_ssfr_plane(frame: pd.DataFrame):
    satellites = frame.loc[frame["is_satellite"].fillna(False)].copy()
    satellites["sSFR_plot"] = _numeric(satellites["sSFR_harmonized"]).clip(lower=-14.5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, colour in zip(axes, ["u_minus_r", "u_minus_g"]):
        residual = f"delta_{colour}"
        sns.scatterplot(
            data=satellites,
            x="sSFR_plot",
            y=residual,
            hue="sample",
            style="morphology_harmonized",
            hue_order=SAMPLE_ORDER,
            palette=PALETTE,
            s=28,
            alpha=0.65,
            ax=ax,
        )
        ax.axhline(0, color="0.4", linestyle=":", linewidth=1)
        ax.set_title(rf"$\Delta${COLOUR_LABELS[colour]} versus sSFR")
        ax.set_xlabel(r"$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$")
        ax.set_ylabel(r"$\Delta$ colour (mag)")
        if ax is axes[1] and ax.legend_ is not None:
            ax.legend_.remove()
    axes[0].legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def _regression_row(
    panel: pd.DataFrame,
    colour: str,
    check: str,
    comparison: str,
) -> dict[str, object] | None:
    panel = panel.copy()
    panel["is_CG4_numeric"] = (panel["sample"] == "CG4").astype(int)
    result = fit_ols_with_optional_cluster_se(
        f"{colour} ~ is_CG4_numeric + logM + z_harmonized",
        panel,
    )
    if result is None or "is_CG4_numeric" not in result.model.exog_names:
        return None
    low, high = _coef_ci(result, "is_CG4_numeric")
    return {
        "check": check,
        "comparison": comparison,
        "colour": colour,
        "n": int(result.nobs),
        "coefficient": _coef_value(result, "is_CG4_numeric"),
        "ci_low": low,
        "ci_high": high,
        "p_value": _coef_value(result, "is_CG4_numeric", "pvalues"),
    }


def _cluster_bootstrap_coefficients(
    panel: pd.DataFrame,
    colour: str,
    n_boot: int = 500,
    random_state: int = 20260612,
) -> np.ndarray:
    required = [colour, "sample", "cluster_id", "logM", "z_harmonized"]
    work = panel[required].dropna().copy()
    if work.empty:
        return np.array([])
    work["environment"] = np.where(work["sample"] == "CG4", "CG4", "Ordinary")
    group_frames = {
        environment: {
            group: values for group, values in part.groupby("cluster_id", observed=True)
        }
        for environment, part in work.groupby("environment", observed=True)
    }
    if set(group_frames) != {"CG4", "Ordinary"}:
        return np.array([])
    rng = np.random.default_rng(random_state)
    coefficients = []
    for iteration in range(n_boot):
        pieces = []
        for environment, groups in group_frames.items():
            ids = np.array(list(groups), dtype=object)
            sampled_ids = rng.choice(ids, size=len(ids), replace=True)
            for draw, group_id in enumerate(sampled_ids):
                piece = groups[group_id].copy()
                piece["bootstrap_cluster"] = f"{environment}_{iteration}_{draw}"
                pieces.append(piece)
        boot = pd.concat(pieces, ignore_index=True)
        boot["is_CG4_numeric"] = (boot["environment"] == "CG4").astype(int)
        try:
            fitted = smf.ols(
                f"{colour} ~ is_CG4_numeric + logM + z_harmonized",
                data=boot,
            ).fit()
            coefficients.append(float(fitted.params["is_CG4_numeric"]))
        except Exception:
            continue
    return np.asarray(coefficients)


def robustness_checks(
    frame: pd.DataFrame,
    n_group_boot: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    satellites = frame.loc[frame["is_satellite"].fillna(False)].copy()
    pooled = satellites.loc[satellites["sample"].isin(["CG4", *CONTROL_SAMPLES])].copy()
    rows = []
    leave_one_rows = []
    for colour in COLOUR_COLUMNS:
        valid = pooled.dropna(subset=[colour, "logM", "z_harmonized"]).copy()
        compact = valid.loc[valid["sample"] == "CG4"]
        ordinary = valid.loc[valid["sample"].isin(CONTROL_SAMPLES)]
        if compact.empty or ordinary.empty:
            continue
        mass_low = max(compact["logM"].min(), ordinary["logM"].min())
        mass_high = min(compact["logM"].max(), ordinary["logM"].max())
        z_low = max(compact["z_harmonized"].min(), ordinary["z_harmonized"].min())
        z_high = min(compact["z_harmonized"].max(), ordinary["z_harmonized"].max())
        residual = f"delta_{colour}"
        trim_low, trim_high = valid[residual].quantile([0.005, 0.995])
        checks = {
            "baseline": valid,
            "common stellar-mass range": valid.loc[valid["logM"].between(mass_low, mass_high)],
            "common redshift range": valid.loc[valid["z_harmonized"].between(z_low, z_high)],
            "exclude uncertain morphology": valid.loc[
                valid["morphology_harmonized"].ne("Uncertain")
            ],
            "trim residual 0.5%-99.5%": valid.loc[valid[residual].between(trim_low, trim_high)],
        }
        for check, panel in checks.items():
            row = _regression_row(panel, colour, check, "Ordinary pooled")
            if row:
                rows.append(row)
        for control in CONTROL_SAMPLES:
            panel = valid.loc[valid["sample"].isin(["CG4", control])]
            row = _regression_row(panel, colour, "separate control", control)
            if row:
                rows.append(row)

        boot = _cluster_bootstrap_coefficients(valid, colour, n_boot=n_group_boot)
        if boot.size:
            rows.append(
                {
                    "check": "bootstrap by group",
                    "comparison": "Ordinary pooled",
                    "colour": colour,
                    "n": int(len(valid)),
                    "coefficient": float(np.median(boot)),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                    "p_value": float(
                        2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
                    ),
                }
            )

        cg_groups = compact["cluster_id"].dropna().unique()
        for group_id in cg_groups:
            reduced = valid.loc[valid["cluster_id"] != group_id]
            row = _regression_row(
                reduced,
                colour,
                "leave one CG4 group out",
                str(group_id),
            )
            if row:
                leave_one_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(leave_one_rows)


def plot_leave_one_group_out(leave_one: pd.DataFrame):
    if leave_one.empty:
        _message("Leave-one-group-out plot skipped: no estimates are available.")
        return None
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, colour in zip(axes.flat, COLOUR_COLUMNS):
        values = leave_one.loc[leave_one["colour"] == colour, "coefficient"].dropna()
        sns.histplot(values, bins=min(15, max(5, len(values) // 4)), ax=ax)
        ax.axvline(0, color="0.3", linestyle=":", linewidth=1)
        ax.set_title(f"{COLOUR_LABELS[colour]} (N={len(values)} omissions)")
        ax.set_xlabel("CG4 coefficient after omitting one CG4 group")
    fig.suptitle("Leave-one-CG4-group-out robustness")
    fig.tight_layout()
    return fig


def make_machine_summary(
    matching_continuous: pd.DataFrame,
    matching_categorical: pd.DataFrame,
    regressions: pd.DataFrame,
    ssfr_split: pd.DataFrame,
    morphology_split: pd.DataFrame,
    distance_correlations: pd.DataFrame,
    robustness: pd.DataFrame,
) -> str:
    """Create a concise, deliberately conservative textual interpretation."""

    bias_flags = []
    if not matching_continuous.empty:
        bias_flags.append((matching_continuous["cliffs_delta"].abs() >= 0.147).any())
    if not matching_categorical.empty:
        bias_flags.append((matching_categorical["cramers_v"] >= 0.1).any())
    biased = any(bias_flags)

    main = regressions.loc[
        (regressions["comparison"] == "Ordinary pooled")
        & (regressions["model"] == "mass+redshift")
    ]
    blue_colours = main.loc[
        (main["is_CG4_coefficient"] < 0) & (main["p_value"] < 0.05),
        "colour",
    ].tolist()
    full = regressions.loc[
        (regressions["comparison"] == "Ordinary pooled") & (regressions["model"] == "full")
    ]
    full_blue = full.loc[
        (full["is_CG4_coefficient"] < 0) & (full["p_value"] < 0.05),
        "colour",
    ].tolist()

    sf_pooled = ssfr_split.loc[ssfr_split["comparison"] == "Ordinary pooled"]
    sf_classes = sf_pooled.loc[
        (sf_pooled["median_diff"] < 0) & (sf_pooled["permutation_p"] < 0.05),
        "split_value",
    ].unique().tolist()
    morph_pooled = morphology_split.loc[morphology_split["comparison"] == "Ordinary pooled"]
    morph_classes = morph_pooled.loc[
        (morph_pooled["median_diff"] < 0) & (morph_pooled["permutation_p"] < 0.05),
        "split_value",
    ].unique().tolist()

    cg_distance = distance_correlations.loc[
        (distance_correlations["sample"] == "CG4")
        & (distance_correlations["spearman_p"] < 0.05)
    ]
    robust_blue = robustness.loc[
        (robustness["comparison"] == "Ordinary pooled")
        & (robustness["coefficient"] < 0)
        & (robustness["ci_high"] < 0)
    ]
    separate_controls = regressions.loc[
        (regressions["comparison"].isin(CONTROL_SAMPLES))
        & (regressions["model"] == "full")
    ]
    significant_control_blue = separate_controls.loc[
        (separate_controls["is_CG4_coefficient"] < 0)
        & (separate_controls["p_value"] < 0.05),
        "comparison",
    ].unique().tolist()
    starforming_or_spiral_signal = bool(
        {"Starforming", "Spiral"}.intersection(set(sf_classes + morph_classes))
    )

    if blue_colours and not robust_blue.empty and starforming_or_spiral_signal:
        interpretation = (
            "The offset is stable and concentrated in star-forming or spiral satellites, "
            "which is consistent with residual or interaction-triggered recent star formation."
        )
    elif full_blue and not blue_colours:
        interpretation = (
            "A negative coefficient appears only after population controls and is not stable "
            "in the baseline pooled robustness checks. Because the split signal is not "
            "concentrated in star-forming spirals, interaction-triggered star formation is "
            "not the preferred explanation; selection, control-sample heterogeneity, "
            "rejuvenation, dust/metallicity, or classification effects require investigation."
        )
    else:
        interpretation = (
            "The expanded tests do not support a robust blue compact-group satellite offset."
        )

    lines = [
        "Colour exploration summary:",
        "",
        "1. The colour-matched subset appears "
        + ("potentially biased" if biased else "broadly representative")
        + " with respect to the audited galaxy properties.",
        "2. At fixed mass and redshift, CG4 satellites are "
        + (
            "significantly bluer in " + ", ".join(blue_colours)
            if blue_colours
            else "not significantly bluer in the pooled comparison"
        )
        + ".",
        "3. After controlling for sSFR class and morphology, a negative offset "
        + (
            "emerges in " + ", ".join(full_blue)
            if full_blue
            else "is not significant"
        )
        + ".",
        "4. Significant negative residual differences by sSFR class occur in "
        + (", ".join(map(str, sf_classes)) if sf_classes else "no class")
        + "; by morphology they occur in "
        + (", ".join(map(str, morph_classes)) if morph_classes else "no class")
        + ".",
        "5. CG4 colour residuals "
        + ("do" if not cg_distance.empty else "do not")
        + " show a nominal Spearman correlation with BGG distance.",
        "6. Preferred interpretation: " + interpretation,
        "7. Main caveats: SDSS colour matching, small CG4 subsamples after splitting, "
        "multiple exploratory tests, and possible residual dust/metallicity confounding. "
        f"{len(robust_blue)} pooled robustness estimates have a wholly negative 95% interval; "
        "significant negative full-model comparisons occur against "
        + (
            ", ".join(significant_control_blue)
            if significant_control_blue
            else "no individual control catalogue"
        )
        + ".",
    ]
    return "\n".join(lines)
