"""Pipeline-side checks cited in the A&A submission of Paper II.

Run from the repository root after the main pipeline::

    python src/submission_checks.py

It reads the processed sample (with the DR7 GZ1 recovery applied at load
time, as in ``src/main.py``), never queries a remote service, and writes
``output/submission_checks.json``, which the renderer exposes as ``subm``.

Blocks
------
* ``halpha_global``: strong-H-alpha odds ratios of the full-sample models of
  Sect. 3.5 with the emission-line join by photometric objid (as published)
  and by the stored spectrum identifier, adding redshift, the fibre-to-galaxy
  aperture ratio and group covariates;
* ``distance_scale``: the decrease of group radii and crossing times when the
  legacy luminosity-distance conversion is replaced by the angular-diameter
  distance;
* ``pegged_sersic``: the Simard fits excluded at the GIM2D index bounds;
* ``tidal_baseline``: the deduplicated E-class model without the tidal index
  on the sample of the published model with the index;
* ``battery_per_dex``: per-dex odds ratios of the median pair separation;
* ``robustness_satellites``: satellite-only versions of the robustness rows;
* ``agn``: AGN-like odds ratios against each control and on the
  deduplicated pool.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import pickle
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.environ.setdefault("MPLBACKEND", "Agg")

import config as co  # noqa: E402
import data_loader  # noqa: E402
import morphologies  # noqa: E402
import morphology_robustness as mr  # noqa: E402
import morphology_threshold_sweep as mts  # noqa: E402
import second_order_battery as sob  # noqa: E402
from agn_environment import _classify  # noqa: E402
from extended_data import build_galaxy_frame, dedup_control_pool  # noqa: E402
from extended_stats import fit_logistic_model, fit_ols_with_optional_cluster_se  # noqa: E402
from gz1_recovery import apply_to_sample  # noqa: E402
from size_data import attach_size_columns  # noqa: E402
from specialness_models import _covariates  # noqa: E402
from tidal_indices import _derive  # noqa: E402

OUTPUT = Path(co.OUTPUT_PATH) / "submission_checks.json"
CONTROLS = ("Control4B", "Control4C", "RG4")
SAMPLES = ("CG4", *CONTROLS)
DS18_CACHE = Path(os.environ.get("CG_GALS_DS18_CACHE", Path.home() / ".cache" / "cg_gals_audit"))


def _digest(model: dict, term: str | None = None) -> dict:
    if model.get("status") != "ok":
        return {"status": model.get("status"), "reason": model.get("reason")}
    if term is None:
        return {
            "status": "ok", "n": model["n"], "n_clusters": model.get("n_clusters"),
            "odds_ratio": model["cg4_odds_ratio"], "ci95": model["cg4_ci95"],
            "p": model["cg4_p"],
        }
    entry = model["terms"][term]
    return {
        "status": "ok", "n": model["n"], "n_clusters": model.get("n_clusters"),
        "odds_ratio": entry["odds_ratio"], "ci95": entry["ci95"], "p": entry["p"],
        "coefficient": entry["coefficient"], "standard_error": entry["standard_error"],
    }


def load_sample() -> dict:
    with open(Path(co.DATA_PATH) / co.PROCESS_SAMPLES, "rb") as handle:
        sample = pickle.load(handle)
    # Same loading sequence as src/main.py with REBUILD_SAMPLE = False.
    sample = data_loader.correct_group_distance_scales(sample)
    sample = morphologies.classify_all_samples(sample, record_build_counts=False)
    sample = morphologies.add_morphology_fractions_to_groups(sample)
    sample = apply_to_sample(sample)
    return morphologies.add_morphology_fractions_to_groups(sample)


def galaxy_frame(sample: dict) -> pd.DataFrame:
    with patch("socket.socket.connect", side_effect=RuntimeError("offline")):
        frame, _ = attach_size_columns(build_galaxy_frame(sample))
    return frame


# ---------------------------------------------------------------------------
def halpha_global(frame: pd.DataFrame) -> dict:
    work = frame.copy()
    objid_ew = pd.to_numeric(work["h_alpha_eqw"], errors="coerce")
    work["strong_objid"] = np.where(np.isfinite(objid_ew), (objid_ew <= -3).astype(float), np.nan)
    lines = pd.read_csv(Path(co.DATA_PATH) / "galspecline_dr16.csv", dtype={"specobjid": "Int64"})
    lines = lines[["specobjid", "h_alpha_eqw"]].rename(columns={"h_alpha_eqw": "spec_ew"})
    work["specobjid"] = pd.to_numeric(work["specobjid"], errors="coerce").astype("Int64")
    work = work.merge(lines, on="specobjid", how="left", validate="m:1")
    work["strong_specobjid"] = work.spec_ew.le(-3).astype(float).where(work.spec_ew.gt(-9000))
    radius = pd.to_numeric(work["petroR50_r"], errors="coerce")
    work["aperture"] = np.log10(1.5 / radius.where(radius > 0))

    both = work[["strong_objid", "strong_specobjid"]].dropna()
    only_spec = work.strong_objid.isna() & work.strong_specobjid.notna()
    out = {
        "agreement": {"n_both": int(len(both)),
                      "n_equal": int((both.strong_objid == both.strong_specobjid).sum())},
        "specobjid_only_rows": {s: int(only_spec[work["sample"].eq(s)].sum()) for s in SAMPLES},
        "objid_missing_rows": int(work.strong_objid.isna().sum()),
        "specobjid_only_total": int(only_spec.sum()),
        "fractions": {
            join: {
                s: {
                    "n": int(work.loc[work["sample"].eq(s), f"strong_{join}"].notna().sum()),
                    "fraction": float(work.loc[work["sample"].eq(s), f"strong_{join}"].mean()),
                }
                for s in SAMPLES
            }
            for join in ("objid", "specobjid")
        },
        "models": {},
    }
    specs = {
        "mass_class": ["is_CG4", "logMstar", "elliptical"],
        "plus_z_aperture": ["is_CG4", "logMstar", "elliptical", "z_numeric", "aperture"],
        "plus_z_aperture_group": [
            "is_CG4", "logMstar", "elliptical", "z_numeric", "aperture", "is_satellite",
            "log_group_luminosity", "velocity_dispersion",
        ],
    }
    continuous = ["logMstar", "z_numeric", "aperture", "log_group_luminosity", "velocity_dispersion"]
    for join in ("objid", "specobjid"):
        for control in CONTROLS:
            panel = work[work["sample"].isin(["CG4", control])]
            for name, predictors in specs.items():
                model = fit_logistic_model(
                    panel, f"strong_{join}", predictors,
                    continuous=[c for c in continuous if c in predictors],
                )
                out["models"][f"{join}|{control}|{name}"] = _digest(model)
    return out


def distance_scale(sample: dict) -> dict:
    redshifts = pd.concat(
        [pd.to_numeric(sample[f"{s}{co.GRSUFF}"]["z_group"], errors="coerce") for s in SAMPLES]
    ).dropna()
    decrease = 100 * (1 - (1 + redshifts) ** -2)
    overestimate = 100 * ((1 + redshifts) ** 2 - 1)
    return {
        "n_groups": int(len(redshifts)),
        "decrease_percent": {"min": float(decrease.min()), "median": float(decrease.median()),
                             "max": float(decrease.max())},
        "legacy_overestimate_percent": {"min": float(overestimate.min()),
                                        "median": float(overestimate.median()),
                                        "max": float(overestimate.max())},
        "z_group_max": float(redshifts.max()),
        "n_z_group_above_limit": int((redshifts > co.Z_MAX).sum()),
    }


def pegged_sersic(frame: pd.DataFrame) -> dict:
    ng = pd.to_numeric(frame["simard_ng"], errors="coerce")
    low, high = ng.le(co.NG_PEG_LOW), ng.ge(co.NG_PEG_HIGH)
    out = {"per_sample": {}}
    for s in SAMPLES:
        part = frame["sample"].eq(s)
        out["per_sample"][s] = {
            "simard_matched": int(ng[part].notna().sum()),
            "low": int((low & part).sum()), "high": int((high & part).sum()),
        }
        labels = frame.loc[part & (low | high), "morphology"].value_counts().to_dict()
        out["per_sample"][s]["morphology"] = {str(k): int(v) for k, v in labels.items()}
    pegged = frame[low | high]
    out["rows"] = int(len(pegged))
    out["unique_galaxies"] = int(pegged["objid"].nunique())
    unique = frame.drop_duplicates("objid")
    ung = pd.to_numeric(unique["simard_ng"], errors="coerce")
    out["unique_low"] = int(ung.le(co.NG_PEG_LOW).sum())
    out["unique_high"] = int(ung.ge(co.NG_PEG_HIGH).sum())
    controls = frame[~frame["sample"].eq("CG4")].drop_duplicates("objid")
    cng = pd.to_numeric(controls["simard_ng"], errors="coerce")
    out["controls_unique"] = {
        "simard_matched": int(cng.notna().sum()),
        "low": int(cng.le(co.NG_PEG_LOW).sum()), "high": int(cng.ge(co.NG_PEG_HIGH).sum()),
    }
    return out


def tidal_baseline(sample: dict) -> dict:
    with patch("socket.socket.connect", side_effect=RuntimeError("offline")):
        work = _derive(build_galaxy_frame(sample))
    dedup = dedup_control_pool(work)
    same = dedup.dropna(subset=["elliptical", "is_CG4", "logMstar", "is_satellite", "log_tidal_index"])
    with_index = fit_logistic_model(
        dedup, "elliptical", ["is_CG4", "logMstar", "is_satellite", "log_tidal_index"],
        continuous=["logMstar", "log_tidal_index"],
    )
    without = fit_logistic_model(
        same, "elliptical", ["is_CG4", "logMstar", "is_satellite"], continuous=["logMstar"]
    )
    return {"with_index": _digest(with_index), "without_index_same_sample": _digest(without),
            "n_cg4": int(same.is_CG4.sum())}


def battery_per_dex(sample: dict) -> dict:
    fl = float(pd.to_numeric(sample["CG4" + co.GRSUFF]["FracLumBGG"], errors="coerce").median())
    out = {}
    for s in SAMPLES:
        sat = sob._satellite_frame(sample, s, fl)
        model = fit_logistic_model(sat, "elliptical", ["log_R_pair_med", "logMstar"],
                                   continuous=["logMstar"], min_n=20)
        out[s] = _digest(model, "log_R_pair_med")
    a, b = out["CG4"], out["Control4C"]
    z = (a["coefficient"] - b["coefficient"]) / math.hypot(a["standard_error"], b["standard_error"])
    out["cg4_vs_control4c_slope"] = {"z": float(z), "p": float(2 * norm.sf(abs(z))),
                                     "assumption": "independent samples (approximate)"}
    return out


def robustness_satellites(frame: pd.DataFrame, sample: dict) -> dict:
    out = {}
    work = frame.copy()
    work["nearest_angular_separation_arcsec"] = mr._nearest_angular(work)
    work["close_neighbour_55arcsec"] = (
        work["nearest_angular_separation_arcsec"] < mr.CROWDING_THRESHOLD_ARCSEC
    ).astype(float)
    out["close_flag"] = _digest(mr._adjusted_models(work[work.is_satellite.eq(1)])["elliptical"])
    pool = dedup_control_pool(frame)
    covariates, continuous = _covariates(pool)
    predictors = ["is_CG4", *covariates]
    sat = pool[pool.is_satellite.eq(1)]
    sweep = {}
    for threshold in mts.THRESHOLDS:
        w = sat.copy()
        w["elliptical_t"] = mts._classify_at(w, threshold)
        sweep[f"{threshold:.1f}"] = _digest(fit_logistic_model(
            w, "elliptical_t", predictors, continuous=[c for c in continuous if c in predictors]))
    ors = [v["odds_ratio"] for v in sweep.values() if v.get("status") == "ok"]
    out["sweep"] = {"thresholds": sweep, "or_range": [min(ors), max(ors)],
                    "p_max": max(v["p"] for v in sweep.values() if v.get("status") == "ok")}
    w = sat.copy()
    w["p_E_numeric"] = pd.to_numeric(w["p_E"], errors="coerce")
    usable = [p for p in predictors if w[p].nunique() > 1]
    fitted = fit_ols_with_optional_cluster_se(
        "p_E_numeric ~ " + " + ".join(usable), w, group_col="physical_group", min_groups=8)
    ci = fitted.conf_int().loc["is_CG4"]
    out["continuous"] = {"coefficient": float(fitted.params["is_CG4"]), "ci95": [float(ci[0]), float(ci[1])],
                         "p": float(fitted.pvalues["is_CG4"]), "n": int(fitted.nobs)}
    w = sat.copy()
    ng = pd.to_numeric(w["simard_ng"], errors="coerce")
    w["early_sersic"] = np.where(ng.notna(), (ng > 2.5).astype(float), np.nan)
    out["sersic"] = _digest(fit_logistic_model(
        w, "early_sersic", predictors, continuous=[c for c in continuous if c in predictors]))
    out["sersic"]["coverage"] = float(ng.notna().mean())
    out["ds18"] = _ds18_satellites(sample)
    return out


def _ds18_satellites(sample: dict) -> dict:
    bridge_path = DS18_CACHE / "photoobjdr7_map.csv"
    ds18_path = DS18_CACHE / "ds18" / "catalog.dat.gz"
    if not (bridge_path.exists() and ds18_path.exists()):
        return {"status": "skipped", "reason": "DS18 cache unavailable"}
    sys.path.insert(0, str(Path(co.BASE_PATH) / "referee"))
    import T10_ds18_morphology as t10

    bridge = pd.read_csv(bridge_path, dtype={"dr7objid": "int64", "dr8objid": "int64"})
    with gzip.open(ds18_path, "rt") as stream:
        ds18 = pd.read_fwf(stream, colspecs=t10.DS18_COLSPECS, names=t10.DS18_COLUMNS,
                           header=None, usecols=["objID", "TType"], dtype={"objID": str})
    ds18["objID"] = ds18["objID"].str.strip().astype("int64")
    ds18["TType"] = pd.to_numeric(ds18["TType"], errors="coerce")
    with patch("socket.socket.connect", side_effect=RuntimeError("offline")):
        frame = build_galaxy_frame(sample)
    mapping = bridge[["dr8objid", "dr7objid"]].merge(
        ds18, left_on="dr7objid", right_on="objID", how="inner", validate="1:1")
    frame = frame.merge(mapping[["dr8objid", "TType"]], left_on="objid", right_on="dr8objid",
                        how="left", validate="m:1")
    frame["ds18_early"] = np.where(frame.TType.notna(), (frame.TType <= 0).astype(float), np.nan)
    covariates, continuous = _covariates(frame)
    out = {"status": "ok"}
    for control in CONTROLS:
        panel = frame[frame["sample"].isin(["CG4", control]) & frame.is_satellite.eq(1)]
        out[control] = _digest(fit_logistic_model(panel, "ds18_early", ["is_CG4", *covariates],
                                                  continuous=continuous))
    return out


def agn(frame: pd.DataFrame) -> dict:
    work = _classify(frame)
    predictors = ["is_CG4", "logMstar"] + [
        c for c in ("is_satellite", "dist2BGG_kpc", "dominance")
        if c in work and work[c].notna().mean() >= 0.6
    ]
    continuous = [c for c in predictors if c not in {"is_CG4", "is_satellite"}]
    valid = np.isfinite(work["log_NII_Ha"]) & np.isfinite(work["log_OIII_Hb"])
    out = {
        "predictors": predictors,
        "flag_only_agn": int((work["bpt_class"].eq("AGN_like") & ~valid).sum()),
        "pooled_rows": _digest(fit_logistic_model(work, "agn_like", predictors, continuous=continuous)),
        "pooled_dedup": _digest(fit_logistic_model(dedup_control_pool(work), "agn_like", predictors,
                                                   continuous=continuous)),
    }
    for control in CONTROLS:
        out[control] = _digest(fit_logistic_model(
            work[work["sample"].isin(["CG4", control])], "agn_like", predictors, continuous=continuous))
    return out


def control_overlap(frame: pd.DataFrame) -> dict:
    ids = {s: set(frame.loc[frame["sample"].eq(s), "objid"]) for s in SAMPLES}
    controls = [frame["sample"].eq(s).sum() for s in CONTROLS]
    return {
        "rg4_in_control4b": bool(ids["RG4"] <= ids["Control4B"]),
        "rg4_in_control4c": bool(ids["RG4"] <= ids["Control4C"]),
        "control4b_and_control4c": int(len(ids["Control4B"] & ids["Control4C"])),
        "unique_controls": int(len(ids["Control4B"] | ids["Control4C"] | ids["RG4"])),
        "control_rows": int(sum(controls)),
    }


def satellite_fisher(frame: pd.DataFrame) -> dict:
    """Two-sided Fisher tests of the satellite E/(E+S) fractions (Table 3 counts)."""
    from scipy.stats import fisher_exact

    sat = frame[frame.is_satellite.eq(1) & frame.morphology.isin(["Elliptical", "Spiral"])]
    counts = {
        s: [int(sat.loc[sat["sample"].eq(s), "morphology"].eq(m).sum()) for m in ("Elliptical", "Spiral")]
        for s in SAMPLES
    }
    out = {"counts": counts}
    for control in CONTROLS:
        out[control] = float(fisher_exact([counts["CG4"], counts[control]]).pvalue)
    return out


def main() -> dict:
    warnings.filterwarnings("ignore")
    sample = load_sample()
    frame = galaxy_frame(sample)
    result = {
        "control_overlap": control_overlap(frame),
        "satellite_fisher": satellite_fisher(frame),
        "halpha_global": halpha_global(frame),
        "distance_scale": distance_scale(sample),
        "pegged_sersic": pegged_sersic(frame),
        "tidal_baseline": tidal_baseline(sample),
        "battery_per_dex": battery_per_dex(sample),
        "robustness_satellites": robustness_satellites(frame, sample),
        "agn": agn(frame),
    }
    OUTPUT.write_text(json.dumps(result, indent=1, sort_keys=True, default=float) + "\n")
    print(f"Wrote {OUTPUT}")
    return result


if __name__ == "__main__":
    main()
