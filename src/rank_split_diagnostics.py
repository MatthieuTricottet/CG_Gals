"""Diagnostics exposed by the rank split of Fig. 4 (bgg-split round, Phase 3).

1. Retired galaxies in the BPT "AGN-like" class.  The WHAN scheme of Cid
   Fernandes et al. (2011) calls a galaxy retired when |EW(H-alpha)| < 3 A;
   such galaxies can reach AGN-like line ratios without an AGN.  We report
   their share of the AGN-like class and refit Model R of
   ``rank_split_properties`` with ``agn_like_whan`` (AGN-like and
   EW(H-alpha) <= -3 A, emission negative).  An S/N >= 3 cut in all four BPT
   lines, from the cached MPA-JHU galSpecLine errors, is a sensitivity run.
2. Fibre aperture.  The 3-arcsec SDSS fibre samples a smaller fraction of a
   larger galaxy; ``log_fib_frac = log10(R_fib / R50)`` with R_fib = 1.5
   arcsec and the Petrosian R50 (available for every BGG) is added to Model R
   for the three spectroscopic outcomes.

These are secondary diagnostics: no Holm family, analytic cluster-robust
intervals only.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    import config as co
    from descriptive_trends import MASS_BINS
    from extended_data import ensure_galaxy_frame
    from extended_stats import safe_json
    from rank_split_properties import (
        CONTROLS,
        SUPPORT_MIN_LOGM,
        attach_family_basis,
        fit_interaction_models,
        prepare_frame,
    )
    from size_data import _kpc_per_arcsec
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from .descriptive_trends import MASS_BINS
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import safe_json
    from .rank_split_properties import (
        CONTROLS,
        SUPPORT_MIN_LOGM,
        attach_family_basis,
        fit_interaction_models,
        prepare_frame,
    )
    from .size_data import _kpc_per_arcsec

RETIRED_EW = 3.0
MIN_SN = 3.0
FIBRE_RADIUS_ARCSEC = 1.5
BPT_LINES = ("h_alpha", "h_beta", "oiii_5007", "nii_6584")
LINE_CACHE = os.path.join(co.DATA_PATH, "galspecline_dr16.csv")
FIBRE_OUTCOMES = (("strong_halpha", "logit"), ("agn_like", "logit"), ("Dn4000", "ols"))


def attach_line_sn(work: pd.DataFrame, cache_path: str = LINE_CACHE) -> tuple[pd.DataFrame, dict]:
    """Minimum S/N over the four BPT lines, from the cached galSpecLine table.

    The cache is read as is (never fetched here); rows without a cached
    measurement get NaN.
    """

    if not os.path.exists(cache_path):
        work = work.copy()
        work["bpt_min_sn"] = np.nan
        return work, {"status": "unavailable", "path": os.path.basename(cache_path)}
    lines = pd.read_csv(cache_path)
    sn = []
    for line in BPT_LINES:
        flux = pd.to_numeric(lines[f"{line}_flux"], errors="coerce")
        error = pd.to_numeric(lines[f"{line}_flux_err"], errors="coerce")
        sn.append(np.where(error > 0, flux / error, np.nan))
    table = pd.DataFrame(
        {"_key": pd.to_numeric(lines["specobjid"], errors="coerce").astype(float),
         # a line without a usable error fails the cut (NaN propagates)
         "bpt_min_sn": np.min(np.column_stack(sn), axis=1) if len(lines) else []}
    ).drop_duplicates("_key")
    out = work.copy()
    out["_key"] = pd.to_numeric(out["specobjid"], errors="coerce").astype(float)
    out = out.merge(table, on="_key", how="left", validate="m:1").drop(columns="_key")
    classified = out["agn_like"].notna()
    return out, {
        "status": "ok",
        "path": os.path.basename(cache_path),
        "n_classified": int(classified.sum()),
        "n_classified_with_errors": int((classified & out["bpt_min_sn"].notna()).sum()),
        "n_classified_sn_ge_3": int((classified & (out["bpt_min_sn"] >= MIN_SN)).sum()),
    }


def add_diagnostic_outcomes(work: pd.DataFrame) -> pd.DataFrame:
    out = work.copy()
    ew = pd.to_numeric(out["h_alpha_eqw"], errors="coerce")
    agn = out["agn_like"]
    out["retired_like"] = np.where(np.isfinite(ew), (ew.abs() < RETIRED_EW).astype(float), np.nan)
    out["agn_like_whan"] = np.where(
        agn.isna(), np.nan,
        np.where(agn.eq(1), np.where(np.isfinite(ew), (ew <= -RETIRED_EW).astype(float), np.nan), 0.0),
    )
    passes = out["bpt_min_sn"] >= MIN_SN
    out["agn_like_sn3"] = np.where(passes, out["agn_like"], np.nan)
    out["agn_like_whan_sn3"] = np.where(passes, out["agn_like_whan"], np.nan)
    petro = pd.to_numeric(out["petroR50_kpc"], errors="coerce") if "petroR50_kpc" in out else np.nan
    r_fib = FIBRE_RADIUS_ARCSEC * _kpc_per_arcsec(out["z_numeric"])
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_fib_frac"] = np.where(petro > 0, np.log10(r_fib / petro), np.nan)
    return out


def retired_share(work: pd.DataFrame) -> list[dict]:
    """Share of AGN-like galaxies with |EW(H-alpha)| < 3 A by sample, rank, mass bin."""

    agn = work.loc[work["agn_like"].eq(1)].copy()
    agn["rank_class"] = np.where(agn["is_bgg"].eq(1), "bgg", "satellite")
    agn["mass_bin"] = pd.cut(agn["logMstar"], MASS_BINS, right=False)
    rows = []
    for (sample, rank_class), part in agn.groupby(["sample", "rank_class"], observed=True):
        groups = [("all", part)] + [
            (f"{iv.left:g}-{iv.right:g}", part.loc[part["mass_bin"] == iv])
            for iv in part["mass_bin"].cat.categories
        ]
        for label, cell in groups:
            measured = cell["retired_like"].notna()
            rows.append({
                "sample": sample, "rank": rank_class, "mass_bin": label,
                "n_agn_like": int(len(cell)),
                "n_with_ew": int(measured.sum()),
                "n_retired_like": int(cell.loc[measured, "retired_like"].sum()),
                "retired_share": float(cell.loc[measured, "retired_like"].mean())
                if measured.any() else None,
            })
    return rows


def _event_counts(support: pd.DataFrame, outcome: str, control: str) -> dict:
    part = support.loc[support["sample"].isin(["CG4", control]) & support[outcome].notna()]
    table = part.groupby(["sample", "is_bgg"])[outcome].agg(["size", "sum"])
    return {f"{s}|{'bgg' if b == 1 else 'sat'}": {"n": int(r["size"]), "events": int(r["sum"])}
            for (s, b), r in table.iterrows()}


def _refit(support: pd.DataFrame, outcome: str, kind: str, extra=()) -> list[dict]:
    fits = []
    for control in CONTROLS:
        entry = fit_interaction_models(support, outcome, kind, control, n_boot=0, extra=extra)
        if kind == "logit":
            entry["event_counts"] = _event_counts(support, outcome, control)
        fits.append(entry)
    return fits


def fibre_aperture(support: pd.DataFrame) -> dict:
    usable = support.loc[np.isfinite(support["log_fib_frac"])]
    summary = (
        usable.assign(rank_class=np.where(usable["is_bgg"].eq(1), "bgg", "satellite"))
        .groupby(["sample", "rank_class"])["log_fib_frac"]
        .agg(["size", "median"])
    )
    out = {
        "fibre_radius_arcsec": FIBRE_RADIUS_ARCSEC,
        "n_support_rows": int(len(support)),
        "n_rows_with_log_fib_frac": int(len(usable)),
        "log_fib_frac_median": {
            f"{s}|{r}": {"n": int(v["size"]), "median": float(v["median"])}
            for (s, r), v in summary.iterrows()
        },
        "models": [],
    }
    for outcome, kind in FIBRE_OUTCOMES:
        base = _refit(usable, outcome, kind)
        adjusted = _refit(usable, outcome, kind, extra=("log_fib_frac",))
        for without, with_fib in zip(base, adjusted):
            row = {"outcome": outcome, "control": without["control"],
                   "status": "ok" if without["status"] == with_fib["status"] == "ok" else "skipped"}
            if row["status"] == "ok":
                for term in ("cg_sat", "cg_bgg"):
                    row[term] = {
                        "without": without[term],
                        "with_log_fib_frac": with_fib[term],
                        "shift": with_fib[term]["estimate"] - without[term]["estimate"],
                        "shift_in_se": (with_fib[term]["estimate"] - without[term]["estimate"])
                        / without[term]["se"],
                    }
                row["n"] = with_fib["counts"]["n"]
            out["models"].append(row)
    return out


def run_rank_split_diagnostics(galaxies, output_dir: str | None = None) -> dict:
    frame = ensure_galaxy_frame(galaxies)
    if frame.empty:
        return {"status": "skipped", "reason": "no_galaxy_frame"}
    work, sn_info = attach_line_sn(prepare_frame(frame))
    work = add_diagnostic_outcomes(work)
    support, _satellites, spline_info = attach_family_basis(work)
    result = {
        "status": "ok",
        "support_min_logmstar": SUPPORT_MIN_LOGM,
        "mass_adjustment": spline_info,
        "retired_criterion": f"|EW(Halpha)| < {RETIRED_EW:g} A (WHAN retired; emission negative)",
        "retired_share": retired_share(work),
        "agn_like_whan_definition": "agn_like == 1 and EW(Halpha) <= -3 A; 0 for other BPT-classified rows",
        "agn_like_whan_models": _refit(support, "agn_like_whan", "logit"),
        "line_sn": sn_info,
    }
    if sn_info.get("status") == "ok":
        result["sn_cut"] = f"S/N >= {MIN_SN:g} in H-alpha, H-beta, [O III] 5007 and [N II] 6584"
        result["agn_like_sn3_models"] = _refit(support, "agn_like_sn3", "logit")
        result["agn_like_whan_sn3_models"] = _refit(support, "agn_like_whan_sn3", "logit")
    else:
        result["sn_cut"] = "none available: BPT classes carry no S/N cut"
    result["fibre_aperture"] = fibre_aperture(support)
    return safe_json(result)
