from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu

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


GROUP_COL = "Group"
RANK_COL = "rank_M"
MASS_COL = "lgm"
DOM_COL = "is_dominated"


def clean_morph(series: pd.Series) -> pd.Series:
    """Keep only the two secure morphology labels used in the notebook analysis."""

    return series.where(series.isin(["Spiral", "Elliptical"]))


def attach_dom_from_group_table(
    df_gal: pd.DataFrame,
    df_grp: pd.DataFrame,
    group_col: str = GROUP_COL,
    dom_col: str = DOM_COL,
) -> pd.DataFrame:
    """Attach domination labels from the group table to the galaxy table."""

    out = df_gal.copy()
    dom_map = (
        df_grp[[group_col, dom_col]]
        .drop_duplicates(subset=[group_col])
        .set_index(group_col)[dom_col]
    )
    out[dom_col] = out[group_col].map(dom_map).astype("boolean")
    return out


def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the galaxy table for the morphology-vs-domination regressions."""

    out = df.copy()
    out["morph_clean"] = clean_morph(out["morphology"])
    out["_is_bgg"] = out[RANK_COL].eq(1)
    return out


def _empty_group_level_result(error: str | None = None) -> dict[str, object]:
    """Return the placeholder payload used when the group-level fit cannot run."""

    return {
        "n_groups_used": 0,
        "median_fSp_sat_BGGSp": np.nan,
        "median_fSp_sat_BGGEll": np.nan,
        "OR": np.nan,
        "CI95_low": np.nan,
        "CI95_high": np.nan,
        "p_glm": np.nan,
        "p_mwu": np.nan,
        "error": error,
    }


def _empty_satellite_level_result(error: str | None = None) -> dict[str, object]:
    """Return the placeholder payload used when the satellite-level fit cannot run."""

    return {
        "n_sat_used": 0,
        "n_groups_used": 0,
        "OR": np.nan,
        "CI95_low": np.nan,
        "CI95_high": np.nan,
        "p_cluster": np.nan,
        "error": error,
    }


def group_level_binom(df: pd.DataFrame, control_lgm_bgg: bool = True) -> dict[str, object]:
    """Replicate the notebook's group-level binomial GLM."""

    df = prep_df(df).dropna(subset=[GROUP_COL, "morph_clean"])

    bgg = df[df["_is_bgg"]][[GROUP_COL, "morph_clean", MASS_COL]].dropna(
        subset=[GROUP_COL, "morph_clean"]
    )
    bgg = bgg.rename(columns={"morph_clean": "bgg_morph", MASS_COL: "lgm_bgg"})

    sat = df[~df["_is_bgg"]].copy()
    grp = (
        sat.groupby(GROUP_COL)
        .agg(
            n_sat=("morph_clean", "size"),
            k_sp_sat=("morph_clean", lambda x: (x == "Spiral").sum()),
        )
        .reset_index()
    )
    grp = grp.merge(bgg, on=GROUP_COL, how="inner")
    grp = grp[grp["n_sat"] > 0].copy()

    if grp.empty:
        return _empty_group_level_result("no_satellite_groups")

    grp["f_sp_sat"] = grp["k_sp_sat"] / grp["n_sat"]
    grp["bgg_is_sp"] = (grp["bgg_morph"] == "Spiral").astype(int)

    xcols = ["bgg_is_sp"]
    if control_lgm_bgg and "lgm_bgg" in grp.columns:
        xcols.append("lgm_bgg")

    a = grp.loc[grp["bgg_is_sp"] == 1, "f_sp_sat"]
    b = grp.loc[grp["bgg_is_sp"] == 0, "f_sp_sat"]

    result = _empty_group_level_result()
    result.update(
        {
            "n_groups_used": int(len(grp)),
            "median_fSp_sat_BGGSp": float(a.median()) if len(a) else np.nan,
            "median_fSp_sat_BGGEll": float(b.median()) if len(b) else np.nan,
        }
    )

    if len(a) and len(b):
        result["p_mwu"] = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)

    try:
        x = sm.add_constant(grp[xcols])
        y = grp["f_sp_sat"]
        w = grp["n_sat"]
        fit = sm.GLM(y, x, family=sm.families.Binomial(), var_weights=w).fit()
    except Exception as exc:  # pragma: no cover - numerical edge cases
        result["error"] = str(exc)
        return result

    beta = fit.params["bgg_is_sp"]
    se = fit.bse["bgg_is_sp"]
    result.update(
        {
            "OR": float(np.exp(beta)),
            "CI95_low": float(np.exp(beta - 1.96 * se)),
            "CI95_high": float(np.exp(beta + 1.96 * se)),
            "p_glm": float(fit.pvalues["bgg_is_sp"]),
        }
    )
    return result


def satellite_level_cluster(
    df: pd.DataFrame,
    control_lgm_bgg: bool = True,
    control_lgm_sat: bool = False,
) -> dict[str, object]:
    """Replicate the notebook's satellite-level clustered logistic regression."""

    df = prep_df(df).dropna(subset=[GROUP_COL, "morph_clean"])

    bgg = df[df["_is_bgg"]][[GROUP_COL, "morph_clean", MASS_COL]].dropna(
        subset=[GROUP_COL, "morph_clean"]
    )
    bgg = bgg.rename(columns={"morph_clean": "bgg_morph", MASS_COL: "lgm_bgg"})

    sat = df[~df["_is_bgg"]].merge(bgg, on=GROUP_COL, how="inner")
    sat = sat.dropna(subset=["bgg_morph", "morph_clean"])

    if sat.empty:
        return _empty_satellite_level_result("no_satellites")

    sat["y_sp"] = (sat["morph_clean"] == "Spiral").astype(int)
    sat["bgg_is_sp"] = (sat["bgg_morph"] == "Spiral").astype(int)

    xcols = ["bgg_is_sp"]
    if control_lgm_bgg and "lgm_bgg" in sat.columns:
        xcols.append("lgm_bgg")
    if control_lgm_sat and MASS_COL in sat.columns:
        xcols.append(MASS_COL)

    try:
        x = sm.add_constant(sat[xcols])
        y = sat["y_sp"]
        fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(
            cov_type="cluster",
            cov_kwds={"groups": sat[GROUP_COL]},
        )
    except Exception as exc:  # pragma: no cover - numerical edge cases
        result = _empty_satellite_level_result(str(exc))
        result.update(
            {
                "n_sat_used": int(len(sat)),
                "n_groups_used": int(sat[GROUP_COL].nunique()),
            }
        )
        return result

    beta = fit.params["bgg_is_sp"]
    se = fit.bse["bgg_is_sp"]
    return {
        "n_sat_used": int(len(sat)),
        "n_groups_used": int(sat[GROUP_COL].nunique()),
        "OR": float(np.exp(beta)),
        "CI95_low": float(np.exp(beta - 1.96 * se)),
        "CI95_high": float(np.exp(beta + 1.96 * se)),
        "p_cluster": float(fit.pvalues["bgg_is_sp"]),
        "error": None,
    }


def build_domination_results(sample: dict[str, pd.DataFrame], min_groups: int = 10) -> pd.DataFrame:
    """Build the table saved by the morphology exploration notebook."""

    rows: list[dict[str, object]] = []

    for cat in co.SAMPLE.keys():
        gal_key = cat + co.GASUFF
        grp_key = cat + co.GRSUFF

        if gal_key not in sample or grp_key not in sample:
            continue

        df_gal = sample[gal_key].copy()
        df_grp = sample[grp_key].copy()

        needed_gal = {GROUP_COL, RANK_COL, "morphology", MASS_COL}
        needed_grp = {GROUP_COL, DOM_COL}
        if not needed_gal.issubset(df_gal.columns):
            rows.append(
                {
                    "cat": cat,
                    "dom": "ALL",
                    "error": f"Missing in GAL: {sorted(needed_gal - set(df_gal.columns))}",
                }
            )
            continue
        if not needed_grp.issubset(df_grp.columns):
            rows.append(
                {
                    "cat": cat,
                    "dom": "ALL",
                    "error": f"Missing in GRP: {sorted(needed_grp - set(df_grp.columns))}",
                }
            )
            continue

        df = attach_dom_from_group_table(df_gal, df_grp)

        for dom_value, dom_label in [(True, "dominated"), (False, "non_dominated")]:
            dfi = df[df[DOM_COL] == dom_value].copy()
            n_groups = int(dfi[GROUP_COL].nunique())
            if n_groups < min_groups:
                rows.append(
                    {
                        "cat": cat,
                        "dom": dom_label,
                        "warning": f"too_few_groups ({n_groups})",
                    }
                )
                continue

            out_g = group_level_binom(dfi, control_lgm_bgg=True)
            out_s = satellite_level_cluster(
                dfi,
                control_lgm_bgg=True,
                control_lgm_sat=False,
            )

            rows.append(
                {
                    "cat": cat,
                    "dom": dom_label,
                    "G_n_groups": out_g["n_groups_used"],
                    "G_med_fSp_sat_BGGSp": out_g["median_fSp_sat_BGGSp"],
                    "G_med_fSp_sat_BGGEll": out_g["median_fSp_sat_BGGEll"],
                    "G_OR": out_g["OR"],
                    "G_CI95": (
                        f"[{out_g['CI95_low']:.2f}, {out_g['CI95_high']:.2f}]"
                        if np.isfinite(out_g["CI95_low"]) and np.isfinite(out_g["CI95_high"])
                        else np.nan
                    ),
                    "G_p": out_g["p_glm"],
                    "G_p_MWU": out_g["p_mwu"],
                    "S_n_sat": out_s["n_sat_used"],
                    "S_n_groups": out_s["n_groups_used"],
                    "S_OR": out_s["OR"],
                    "S_CI95": (
                        f"[{out_s['CI95_low']:.2f}, {out_s['CI95_high']:.2f}]"
                        if np.isfinite(out_s["CI95_low"]) and np.isfinite(out_s["CI95_high"])
                        else np.nan
                    ),
                    "S_p": out_s["p_cluster"],
                    "error": out_g.get("error") or out_s.get("error"),
                }
            )

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(["cat", "dom"], na_position="last").reset_index(drop=True)
    return results


def run(sample: dict[str, pd.DataFrame], output_path: str | None = None) -> pd.DataFrame:
    """Entry point used by the main pipeline."""

    if output_path is None:
        output_path = os.path.join(co.OUTPUT_PATH, "morphology_by_domination.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = build_domination_results(sample)
    results.to_csv(output_path, index=False)

    summary_rows: list[dict[str, object]] = []
    if not results.empty:
        for _, row in results.iterrows():
            if not (
                (pd.notna(row.get("G_p")) and row["G_p"] < co.P_LIMIT)
                or (pd.notna(row.get("S_p")) and row["S_p"] < co.P_LIMIT)
            ):
                continue

            summary_rows.append(
                {
                    "sample": row["cat"],
                    "sample_label": lu.formatted_sample_name(row["cat"]),
                    "domination": row["dom"],
                    "domination_label": row["dom"].replace("_", " "),
                    "group_n": int(row["G_n_groups"]),
                    "group_or": float(row["G_OR"]) if pd.notna(row["G_OR"]) else np.nan,
                    "group_or_fmt": f"{row['G_OR']:.2f}" if pd.notna(row["G_OR"]) else "NA",
                    "group_p": float(row["G_p"]) if pd.notna(row["G_p"]) else np.nan,
                    "sat_n_groups": int(row["S_n_groups"]),
                    "sat_n": int(row["S_n_sat"]),
                    "sat_or": float(row["S_OR"]) if pd.notna(row["S_OR"]) else np.nan,
                    "sat_or_fmt": f"{row['S_OR']:.2f}" if pd.notna(row["S_OR"]) else "NA",
                    "sat_p": float(row["S_p"]) if pd.notna(row["S_p"]) else np.nan,
                    "group_median_spiral_bgg": float(row["G_med_fSp_sat_BGGSp"]) if pd.notna(row["G_med_fSp_sat_BGGSp"]) else np.nan,
                    "group_median_elliptical_bgg": float(row["G_med_fSp_sat_BGGEll"]) if pd.notna(row["G_med_fSp_sat_BGGEll"]) else np.nan,
                    "group_median_spiral_bgg_fmt": f"{row['G_med_fSp_sat_BGGSp']:.2f}" if pd.notna(row["G_med_fSp_sat_BGGSp"]) else "NA",
                    "group_median_elliptical_bgg_fmt": f"{row['G_med_fSp_sat_BGGEll']:.2f}" if pd.notna(row["G_med_fSp_sat_BGGEll"]) else "NA",
                }
            )

    for row in summary_rows:
        if pd.notna(row["group_p"]):
            row["group_p_fmt"] = gu.tex_form(row["group_p"])
        else:
            row["group_p_fmt"] = "NA"
        if pd.notna(row["sat_p"]):
            row["sat_p_fmt"] = gu.tex_form(row["sat_p"])
        else:
            row["sat_p_fmt"] = "NA"

    report.append_json("Morphology_domination_effects", summary_rows)
    report.append_json("Morphology_domination_effects_count", len(summary_rows))
    return results
