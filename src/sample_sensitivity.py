"""Sample-definition sensitivities of the adjusted morphology contrast.

Two choices of the sample construction are undone in turn and the
per-control E-class models of Table 4 refitted, with the same covariates
and clustering:

* the 16 Zheng & Shen (2021) *split* compact groups, removed from CG4, are
  restored from ``data/CG4_Gals.csv`` / ``CG4_Groups.csv``;
* the Control4B and Control4C quartets of Lim group 3688, removed because an
  outlying-redshift member inflates their velocity dispersion, are restored
  from the raw control catalogues.

Restored rows receive their Galaxy Zoo classes and sSFR classes exactly as
the published rows: debiased GZ1 votes from the SDSS reference table by
``objid`` and the adopted equal-posterior sSFR boundary.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    import config as co
    import generate_report as report
    from extended_data import build_galaxy_frame
    from extended_stats import safe_json
    from primary_contrasts import run_primary_contrasts
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from . import generate_report as report
    from .extended_data import build_galaxy_frame
    from .extended_stats import safe_json
    from .primary_contrasts import run_primary_contrasts

LIM_3688 = 3688
REPORTED_MODELS = ["elliptical_all", "elliptical_satellites"]


def _classify_rows(rows: pd.DataFrame, sample: dict) -> pd.DataFrame:
    """GZ1 morphology and sSFR status for restored catalogue rows."""

    try:
        from morphologies import classify
    except ModuleNotFoundError:  # pragma: no cover
        from .morphologies import classify

    votes = sample["SDSS_withAGN"][["objid", "p_E", "p_S"]].drop_duplicates("objid")
    out = rows.drop(columns=["p_E", "p_S"], errors="ignore").copy()
    out["objid"] = out["objid"].astype("int64")
    votes = votes.assign(objid=votes["objid"].astype("int64"))
    out = classify(out.merge(votes, on="objid", how="left"))
    boundary = report.decode_interp1d(report._load_json(co.RESULTS_BUILD)["sSFR_interp"])
    ssfr = pd.to_numeric(out["sSFR"], errors="coerce")
    mass = pd.to_numeric(out["lgm"], errors="coerce")
    valid = ssfr.between(*co.sSFR_VALID_RANGE) & mass.between(*co.LGM_VALID_RANGE)
    limit = boundary(mass.to_numpy(dtype=float))
    out["sSFR_status"] = np.where(
        valid, np.where(ssfr.to_numpy() <= limit, "Quenched", "Starforming"), co.NosSFR_LABEL
    )
    out["lgm"] = mass.where(mass.between(*co.LGM_VALID_RANGE))
    return out


def with_split_groups(sample: dict) -> dict:
    """The processed sample with the split compact groups added back."""

    groups = pd.read_csv(os.path.join(co.DATA_PATH, "CG4_Groups.csv"))
    galaxies = pd.read_csv(os.path.join(co.DATA_PATH, "CG4_Gals.csv"))
    split_ids = groups.loc[groups["Class"] == "Split", "Group"]
    extra = _classify_rows(galaxies.loc[galaxies["Group"].isin(split_ids)], sample)
    out = dict(sample)
    out["CG4_Gals"] = pd.concat([sample["CG4_Gals"], extra], ignore_index=True, sort=False)
    out["CG4_Groups"] = pd.concat(
        [sample["CG4_Groups"], groups.loc[groups["Group"].isin(split_ids)]],
        ignore_index=True,
        sort=False,
    )
    return out


def with_lim_3688(sample: dict) -> dict:
    """The processed sample with the Lim 3688 control quartets restored."""

    out = dict(sample)
    for name in ("Control4B", "Control4C"):
        galaxies = pd.read_csv(os.path.join(co.DATA_PATH, f"{name}_Gals.csv"))
        groups = pd.read_csv(os.path.join(co.DATA_PATH, f"{name}_Groups.csv"))
        extra = _classify_rows(galaxies.loc[galaxies["Group"] == LIM_3688], sample)
        out[f"{name}_Gals"] = pd.concat(
            [sample[f"{name}_Gals"], extra], ignore_index=True, sort=False
        )
        out[f"{name}_Groups"] = pd.concat(
            [sample[f"{name}_Groups"], groups.loc[groups["Group"] == LIM_3688]],
            ignore_index=True,
            sort=False,
        )
    return out


def _digest(contrasts: dict) -> dict:
    return {
        control: {
            model: {
                key: contrasts[control][model].get(key)
                for key in ("cg4_odds_ratio", "cg4_ci95", "cg4_p", "n", "n_clusters")
            }
            for model in REPORTED_MODELS
        }
        for control in contrasts
    }


def run_sample_sensitivity(sample: dict) -> dict:
    """Refit the Table 4 morphology models on the two altered samples."""

    split_sample = with_split_groups(sample)
    lim_sample = with_lim_3688(sample)
    n_split_groups = int(split_sample["CG4_Groups"]["Group"].nunique())
    split = run_primary_contrasts(None, frame=build_galaxy_frame(split_sample))
    lim = run_primary_contrasts(None, frame=build_galaxy_frame(lim_sample))
    sigma = {
        name: float(
            lim_sample[f"{name}_Groups"].loc[
                lim_sample[f"{name}_Groups"]["Group"] == LIM_3688, "Vdisp"
            ].iloc[0]
        )
        for name in ("Control4B", "Control4C")
    }
    members = pd.read_csv(os.path.join(co.DATA_PATH, "PC_Gals.csv"), usecols=["Group", "z"])
    z_members = members.loc[members["Group"] == LIM_3688, "z"].sort_values().to_numpy()
    offsets = np.abs(z_members - np.median(z_members))
    outlier = int(np.argmax(offsets))
    return safe_json(
        {
            "status": "ok",
            "split_groups_restored": {
                "n_cg4_groups": n_split_groups,
                "n_cg4_galaxies": int(len(split_sample["CG4_Gals"])),
                "contrasts": _digest(split["contrasts"]),
            },
            "lim_3688_retained": {
                "sigma_v_kms": sigma,
                "n_members": int(len(z_members)),
                "z_outlier": float(z_members[outlier]),
                "z_others_range": [
                    float(np.delete(z_members, outlier).min()),
                    float(np.delete(z_members, outlier).max()),
                ],
                "contrasts": _digest(lim["contrasts"]),
            },
        }
    )
