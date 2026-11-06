"""bgg-split Phase 0: read-only audit (items 0.2-0.5, 0.8 data checks).

Writes ``results/diagnostics/bgg_split/phase0_audit.json`` and
``phase0_tables.md``; touches nothing else.  Run from the repo root with
``./.venv_clean/bin/python analysis/bgg_split/phase0_audit.py``.
"""

from __future__ import annotations

import json
import pickle
import subprocess

import numpy as np
import pandas as pd

from common import OUT, PKL, ROOT, SAMPLES, load_galaxy_frame

from descriptive_properties import compute_property_trends  # noqa: E402
from descriptive_trends import MASS_BINS  # noqa: E402
from extended_stats import safe_json  # noqa: E402

TOL = 1e-12
R_SIMARD_BRIGHT = 14.0
PCTS = (5, 16, 50, 84, 95)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def reproduce_fig4(frame: pd.DataFrame) -> dict:
    new = compute_property_trends(frame)
    ref = pd.read_csv(ROOT / "output" / "descriptive_property_trends.csv")
    key = ["panel", "sample", "bin_left"]
    merged = ref.merge(new, on=key, suffixes=("_ref", "_new"), validate="1:1")
    out = {
        "n_rows_ref": int(len(ref)),
        "n_rows_new": int(len(new)),
        "n_rows_matched": int(len(merged)),
        "columns_identical": list(ref.columns) == list(new.columns),
    }
    for column in ("n_galaxies", "n_groups"):
        out[f"{column}_mismatches"] = int(
            (merged[f"{column}_ref"] != merged[f"{column}_new"]).sum()
        )
    for column in ("estimate", "ci16", "ci84", "mass_location"):
        a, b = merged[f"{column}_ref"], merged[f"{column}_new"]
        out[f"{column}_nan_mismatches"] = int((a.isna() != b.isna()).sum())
        out[f"{column}_max_abs_diff"] = float((a - b).abs().max())
    out["displayed_mismatches"] = int(
        (merged["displayed_ref"] != merged["displayed_new"]).sum()
    )
    out["pass"] = bool(
        out["n_rows_matched"] == 64
        and out["n_galaxies_mismatches"] == 0
        and out["n_groups_mismatches"] == 0
        and all(
            out[f"{c}_max_abs_diff"] < TOL and out[f"{c}_nan_mismatches"] == 0
            for c in ("estimate", "ci16", "ci84")
        )
    )
    return out


def composition(frame: pd.DataFrame) -> dict:
    work = frame.loc[np.isfinite(frame["logMstar"])].copy()
    work["mass_bin"] = pd.cut(work["logMstar"], MASS_BINS, right=False)
    work["rank_class"] = np.where(work["rank"].eq(1), "bgg", "sat")
    out = {}
    for sample_name in SAMPLES:
        part = work.loc[work["sample"] == sample_name]
        whole = frame.loc[frame["sample"] == sample_name]
        table = pd.crosstab(part["mass_bin"], part["rank_class"]).reindex(
            columns=["bgg", "sat"], fill_value=0
        )
        out[sample_name] = {
            "n_rows": int(len(whole)),
            "n_finite_mass": int(len(part)),
            "n_group_selections": int(whole["group_uid"].nunique()),
            "n_physical_groups": int(whole["physical_group"].nunique()),
            "n_bgg_rows": int(whole["rank"].eq(1).sum()),
            "rank_missing": int(whole["rank"].isna().sum()),
            "bins": {
                f"{iv.left:g}-{iv.right:g}": {
                    "bgg": int(table.loc[iv, "bgg"]),
                    "sat": int(table.loc[iv, "sat"]),
                }
                for iv in table.index
            },
        }
    return out


def _ovl(a: np.ndarray, b: np.ndarray, width: float = 0.1) -> float:
    lo = np.floor(min(a.min(), b.min()) / width) * width
    hi = np.ceil(max(a.max(), b.max()) / width) * width + width
    edges = np.arange(lo, hi + 1e-9, width)
    ha, _ = np.histogram(a, edges)
    hb, _ = np.histogram(b, edges)
    return float(np.minimum(ha / ha.sum(), hb / hb.sum()).sum())


def bgg_mass(frame: pd.DataFrame) -> dict:
    from scipy import stats

    bggs = frame.loc[frame["rank"].eq(1) & np.isfinite(frame["logMstar"])]
    cg = bggs.loc[bggs["sample"] == "CG4", "logMstar"].to_numpy()
    cg_lo, cg_hi = np.percentile(cg, [5, 95])
    out = {}
    for sample_name in SAMPLES:
        values = bggs.loc[bggs["sample"] == sample_name, "logMstar"].to_numpy()
        row = {
            "n": int(values.size),
            "percentiles": {str(p): float(np.percentile(values, p)) for p in PCTS},
            "n_below_10p5": int((values < 10.5).sum()),
        }
        if sample_name != "CG4":
            lo, hi = np.percentile(values, [5, 95])
            row.update(
                {
                    "median_minus_cg4": float(np.median(values) - np.median(cg)),
                    "ovl_hist_0p1dex": _ovl(cg, values),
                    "frac_control_in_cg4_p5_p95": float(
                        ((values >= cg_lo) & (values <= cg_hi)).mean()
                    ),
                    "frac_cg4_in_control_p5_p95": float(
                        ((cg >= lo) & (cg <= hi)).mean()
                    ),
                    "ks_p": float(stats.ks_2samp(cg, values).pvalue),
                }
            )
        out[sample_name] = row
    return out


def size_coverage(frame: pd.DataFrame) -> dict:
    simard_cache = pd.read_csv(ROOT / "data" / "simard2011_subset.csv", dtype={"dr7objid": str})
    # the fetcher caches unmatched ids as all-NaN rows: only rows with a
    # radius are genuine Simard entries
    matched = simard_cache.loc[simard_cache["Rchl_r"].notna(), "dr7objid"]
    in_simard = set(pd.to_numeric(matched, errors="coerce").dropna().astype("int64"))
    work = frame.copy()
    dr7 = pd.to_numeric(work["dr7objid"], errors="coerce")
    work["has_dr7"] = dr7.notna()
    work["in_simard_catalogue"] = dr7.isin(in_simard)
    work["simard_value"] = work["simard_Rchl_r"].notna()
    work["rank_class"] = np.where(work["rank"].eq(1), "bgg", "sat")
    r = pd.to_numeric(work["r_obs"], errors="coerce")
    work["r_class"] = np.select(
        [r.isna(), r < R_SIMARD_BRIGHT], ["r_missing", "r_lt_14"], default="r_ge_14"
    )
    out = {}
    for (sample_name, rank_class, r_class), part in work.groupby(
        ["sample", "rank_class", "r_class"]
    ):
        n = int(len(part))
        cov = lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())  # noqa: E731
        out.setdefault(sample_name, {}).setdefault(rank_class, {})[r_class] = {
            "n": n,
            "log_Rchl_r_kpc": cov(np.isfinite(part["log_Rchl_r_kpc"])),
            "log_Rchl_r_kpc_incl_pegged": cov(
                np.isfinite(part["log_Rchl_r_kpc_incl_pegged"])
            ),
            "petroR50_kpc": cov(np.isfinite(part["petroR50_kpc"])),
            "log_petroR50_kpc": cov(np.isfinite(part["log_petroR50_kpc"])),
            # attribution of the Simard losses (mutually exclusive, in order)
            "loss_no_dr7_bridge": cov(~part["has_dr7"]),
            "loss_not_in_simard": cov(part["has_dr7"] & ~part["in_simard_catalogue"]),
            "loss_rejected_z_or_shred": cov(
                part["in_simard_catalogue"] & ~part["simard_value"]
            ),
            "loss_out_of_window": cov(
                part["simard_value"]
                & ~part["Rchl_r_kpc"].between(0.1, 50.0)
            ),
            "loss_pegged_only": cov(
                np.isfinite(part["log_Rchl_r_kpc_incl_pegged"])
                & ~np.isfinite(part["log_Rchl_r_kpc"])
            ),
        }
    return out


def line_error_columns(frame: pd.DataFrame) -> dict:
    with open(PKL, "rb") as fh:
        sample = pickle.load(fh)
    lines = ("h_alpha", "h_beta", "oiii_5007", "nii_6584")
    withagn = list(sample["SDSS_withAGN"].columns)
    err_cols = [c for c in withagn if "err" in c.lower()]
    data_files = sorted(p.name for p in (ROOT / "data").glob("*.csv"))
    audit_branch_files = git("ls-tree", "-r", "--name-only", "audit-r3", "data").split()
    return {
        "sdss_withagn_line_columns": [c for c in withagn if c.startswith(lines)],
        "sdss_withagn_error_columns": err_cols,
        "frame_error_columns": [c for c in frame.columns if c.endswith("_err") and c.startswith(lines)],
        "data_csv_files_main": data_files,
        "audit_r3_branch_line_caches": [
            f for f in audit_branch_files if "galspecline" in f
        ],
    }


def main() -> None:
    frame, _ = load_galaxy_frame()
    result = {
        "head": git("rev-parse", "HEAD"),
        "status_porcelain": git("status", "--porcelain"),
        "fig4_reproduction": reproduce_fig4(frame),
        "composition": composition(frame),
        "bgg_mass": bgg_mass(frame),
        "size_coverage": size_coverage(frame),
        "line_errors": line_error_columns(frame),
    }
    with open(OUT / "phase0_audit.json", "w", encoding="utf-8") as fh:
        json.dump(safe_json(result), fh, indent=1)
        fh.write("\n")
    print(json.dumps(safe_json(result["fig4_reproduction"]), indent=1))


if __name__ == "__main__":
    main()
