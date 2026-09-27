"""H-alpha emission of the group galaxies: strength, and what the strong emitters are.

Line measurements are the MPA-JHU ``galSpecLine`` fluxes, equivalent widths
and errors of the stored spectrum of each galaxy (``emission_lines``).
Following the WHAN scheme of Cid Fernandes et al. (2011), a galaxy is a
strong H-alpha emitter when W(Halpha) >= 3 A (EW <= -3 A in the SDSS sign
convention) with S/N >= 3 in the H-alpha flux, and strong emitters are split
at log([N II]/Halpha) = -0.4 into star-forming-like and high-[N II]/Halpha
("AGN-like" in WHAN) emission; the ratio requires S/N >= 3 in [N II] or a
3-sigma upper limit below -0.4.  Galaxies below the strong threshold are the
weak-line ("retired") class.  The high-[N II]/Halpha class mixes LINER- and
Seyfert-like, composite and metal-rich star-forming spectra, so it is not an
AGN census; the BPT classes of the strong emitters (S/N >= 3 in all four
lines; Kauffmann et al. 2003 and Kewley et al. 2001 lines) are recorded to
show that mix.

The split of the strong-Halpha deficit into these classes was examined after
the deficit itself had been found, so it is reported as exploratory.
Models are binomial GLMs (OLS for line strengths) with cluster-robust errors
by physical Lim group, CG4 against each control separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.cosmology import Planck15

try:
    from emission_lines import fetch_line_measurements
    from extended_data import ensure_galaxy_frame
    from extended_stats import fit_logistic_model, fit_ols_with_optional_cluster_se, safe_json
    from scipy.stats import fisher_exact
except ModuleNotFoundError:  # pragma: no cover
    from .emission_lines import fetch_line_measurements
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import fit_logistic_model, fit_ols_with_optional_cluster_se, safe_json
    from scipy.stats import fisher_exact

CONTROLS = ["Control4B", "Control4C", "RG4"]
SAMPLES = ["CG4", *CONTROLS]
STRONG_W_ANGSTROM = 3.0
STRONG_AGN_W_ANGSTROM = 6.0
NII_HALPHA_SPLIT = -0.4
MIN_SN = 3.0
TABLE4_COVARIATES = ["logMstar", "z_numeric", "log_group_luminosity", "velocity_dispersion"]
OUTCOMES = ["strong_halpha", "sf_emission", "high_nii_emission", "retired"]


def attach_line_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    """Join the cached galSpecLine columns (prefix ``gsl_``) by ``specobjid``."""

    lines = fetch_line_measurements(frame["specobjid"])
    lines = lines.rename(
        columns={c: f"gsl_{c}" for c in lines.columns if c != "specobjid"}
    )
    out = frame.copy()
    out["_specobjid_key"] = pd.to_numeric(out["specobjid"], errors="coerce")
    lines = lines.rename(columns={"specobjid": "_specobjid_key"})
    lines["_specobjid_key"] = lines["_specobjid_key"].astype(float)
    merged = out.merge(lines, on="_specobjid_key", how="left", validate="m:1")
    merged = merged.drop(columns=["_specobjid_key"])
    merged.index = frame.index
    merged.attrs = dict(frame.attrs)  # keep upstream audits (e.g. size attachment)
    return merged


def classify_emission(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the strong / star-forming-like / high-[N II] / weak-line classes."""

    work = frame.copy()
    if "gsl_h_alpha_eqw" not in work:
        work = attach_line_measurements(work)
    flux = pd.to_numeric(work["gsl_h_alpha_flux"], errors="coerce")
    flux_err = pd.to_numeric(work["gsl_h_alpha_flux_err"], errors="coerce")
    nii = pd.to_numeric(work["gsl_nii_6584_flux"], errors="coerce")
    nii_err = pd.to_numeric(work["gsl_nii_6584_flux_err"], errors="coerce")
    width = -pd.to_numeric(work["gsl_h_alpha_eqw"], errors="coerce")
    measured = width.notna() & flux_err.gt(0)
    sn_halpha = flux / flux_err.where(flux_err > 0)
    sn_nii = nii / nii_err.where(nii_err > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log10(nii.where(nii > 0) / flux.where(flux > 0))
        ratio_limit = np.log10((3.0 * nii_err) / flux.where(flux > 0))
    strong = measured & width.ge(STRONG_W_ANGSTROM) & sn_halpha.ge(MIN_SN)
    ratio_ok = strong & sn_nii.ge(MIN_SN)
    star_forming = (ratio_ok & ratio.lt(NII_HALPHA_SPLIT)) | (
        strong & ~sn_nii.ge(MIN_SN) & ratio_limit.lt(NII_HALPHA_SPLIT)
    )
    high_nii = ratio_ok & ratio.ge(NII_HALPHA_SPLIT)
    work["W_halpha"] = width.where(measured)
    work["log_nii_halpha_sn"] = ratio.where(ratio_ok)
    work["strong_halpha"] = np.where(measured, strong.astype(float), np.nan)
    work["sf_emission"] = np.where(measured, star_forming.astype(float), np.nan)
    work["high_nii_emission"] = np.where(measured, high_nii.astype(float), np.nan)
    work["high_nii_strong_w6"] = np.where(
        measured, (high_nii & width.ge(STRONG_AGN_W_ANGSTROM)).astype(float), np.nan
    )
    work["high_nii_weak_w3_6"] = np.where(
        measured, (high_nii & width.lt(STRONG_AGN_W_ANGSTROM)).astype(float), np.nan
    )
    work["strong_unclassified"] = np.where(
        measured, (strong & ~star_forming & ~high_nii).astype(float), np.nan
    )
    work["retired"] = np.where(measured, (~strong).astype(float), np.nan)

    # BPT class of spectra with all four lines at S/N >= 3
    def sn(line):
        value = pd.to_numeric(work[f"gsl_{line}_flux"], errors="coerce")
        error = pd.to_numeric(work[f"gsl_{line}_flux_err"], errors="coerce")
        return value / error.where(error > 0)

    four = (
        sn("h_alpha").ge(MIN_SN) & sn("nii_6584").ge(MIN_SN)
        & sn("h_beta").ge(MIN_SN) & sn("oiii_5007").ge(MIN_SN)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(
            pd.to_numeric(work["gsl_nii_6584_flux"], errors="coerce")
            / pd.to_numeric(work["gsl_h_alpha_flux"], errors="coerce")
        )
        y = np.log10(
            pd.to_numeric(work["gsl_oiii_5007_flux"], errors="coerce")
            / pd.to_numeric(work["gsl_h_beta_flux"], errors="coerce")
        )
        above_kauffmann = (y > 0.61 / (x - 0.05) + 1.3) | (x >= 0.05)
        above_kewley = (y > 0.61 / (x - 0.47) + 1.19) | (x >= 0.47)
    work["bpt_class_sn3"] = np.where(
        four,
        np.where(above_kewley, "AGN", np.where(above_kauffmann, "composite", "starforming")),
        "unclassified",
    )
    distance_cm = Planck15.luminosity_distance(
        pd.to_numeric(work["z_numeric"], errors="coerce").to_numpy()
    ).to("cm").value
    with np.errstate(divide="ignore", invalid="ignore"):
        work["log_L_halpha_fibre"] = np.log10(
            4.0 * np.pi * distance_cm**2 * flux.where(flux > 0) * 1e-17
        )
        work["log_W_halpha"] = np.log10(width.where(width > 0))
    return work


def _fraction_table(work: pd.DataFrame) -> dict:
    table = {}
    for scope, mask in (
        ("all", pd.Series(True, index=work.index)),
        ("bgg", work["is_bgg"] == 1),
        ("satellites", work["is_satellite"] == 1),
    ):
        table[scope] = {}
        for sample_name in SAMPLES:
            part = work.loc[mask & (work["sample"] == sample_name) & work["strong_halpha"].notna()]
            table[scope][sample_name] = {
                "n": int(len(part)),
                **{
                    outcome: float(part[outcome].mean()) if len(part) else None
                    for outcome in [*OUTCOMES, "high_nii_strong_w6", "high_nii_weak_w3_6", "strong_unclassified"]
                },
            }
    return table


def _bpt_of_strong(work: pd.DataFrame) -> dict:
    out = {}
    strong = work.loc[work["strong_halpha"] == 1]
    for sample_name in SAMPLES:
        part = strong.loc[strong["sample"] == sample_name, "bpt_class_sn3"]
        counts = part.value_counts()
        out[sample_name] = {
            "n_strong": int(len(part)),
            **{k: int(counts.get(k, 0)) for k in ("starforming", "composite", "AGN", "unclassified")},
        }
    return out


def _fisher(work: pd.DataFrame, outcome: str) -> dict:
    out = {}
    cg4 = work.loc[(work["sample"] == "CG4"), outcome].dropna()
    for control in CONTROLS:
        other = work.loc[(work["sample"] == control), outcome].dropna()
        table = [[int(cg4.sum()), int((1 - cg4).sum())], [int(other.sum()), int((1 - other).sum())]]
        out[control] = {
            "fraction_cg4": float(cg4.mean()),
            "fraction_control": float(other.mean()),
            "fisher_p": float(fisher_exact(table).pvalue),
        }
    return out


def adjusted_models(work: pd.DataFrame) -> dict:
    """CG4 odds ratios for each emission class, per control."""

    specifications = {
        # primary: stellar mass and GZ1 class, as in the published strong-Halpha fit
        "mass_class": (None, ["is_CG4", "logMstar", "elliptical"], ["logMstar"]),
        "mass_class_satellite": (
            None, ["is_CG4", "logMstar", "elliptical", "is_satellite"], ["logMstar"]
        ),
        "table4_class": (
            None,
            ["is_CG4", "elliptical", "is_satellite", *TABLE4_COVARIATES],
            TABLE4_COVARIATES,
        ),
        "satellites_table4_class": (
            "satellites", ["is_CG4", "elliptical", *TABLE4_COVARIATES], TABLE4_COVARIATES
        ),
        "bgg_mass_class": ("bgg", ["is_CG4", "logMstar", "elliptical"], ["logMstar"]),
    }
    results = {}
    for control in CONTROLS:
        subset = work.loc[work["sample"].isin(["CG4", control])]
        results[control] = {}
        for outcome in [*OUTCOMES, "high_nii_strong_w6", "high_nii_weak_w3_6"]:
            results[control][outcome] = {}
            for name, (scope, predictors, continuous) in specifications.items():
                panel = subset
                if scope == "satellites":
                    panel = subset.loc[subset["is_satellite"] == 1]
                elif scope == "bgg":
                    panel = subset.loc[subset["is_bgg"] == 1]
                results[control][outcome][name] = fit_logistic_model(
                    panel, outcome, predictors, continuous=continuous
                )
    return results


def emitter_strength(work: pd.DataFrame) -> dict:
    """Line strength of star-forming-like emitters at fixed mass (OLS)."""

    results = {}
    emitters = work.loc[work["sf_emission"] == 1].copy()
    for control in CONTROLS:
        subset = emitters.loc[emitters["sample"].isin(["CG4", control])].dropna(
            subset=["logMstar", "elliptical", "z_numeric"]
        )
        results[control] = {}
        for column in ("log_W_halpha", "log_L_halpha_fibre"):
            fitted = fit_ols_with_optional_cluster_se(
                f"{column} ~ is_CG4 + logMstar + z_numeric + elliptical + is_satellite",
                subset,
                group_col="physical_group",
            )
            if fitted is None:
                results[control][column] = {"status": "failed"}
                continue
            results[control][column] = {
                "status": "ok",
                "cg4_offset_dex": float(fitted.params["is_CG4"]),
                "cg4_se": float(fitted.bse["is_CG4"]),
                "cg4_p": float(fitted.pvalues["is_CG4"]),
                "n": int(fitted.nobs),
            }
    return results


def run_halpha_emission_analysis(data, output_dir: str | None = None) -> dict:
    """Strong-Halpha fractions, their WHAN/BPT decomposition and adjusted models."""

    del output_dir
    frame = ensure_galaxy_frame(data)
    work = classify_emission(frame)
    measured = work["strong_halpha"].notna()
    strong = work["strong_halpha"] == 1
    return safe_json(
        {
            "status": "ok",
            "line_source": "MPA-JHU galSpecLine of the stored spectrum (data/galspecline_dr16.csv)",
            "thresholds": {
                "strong_W_halpha_angstrom": STRONG_W_ANGSTROM,
                "min_sn": MIN_SN,
                "log_nii_halpha_split": NII_HALPHA_SPLIT,
                "strong_agn_like_W_angstrom": STRONG_AGN_W_ANGSTROM,
            },
            "exploratory": True,
            "n_measured": {s: int((measured & (work["sample"] == s)).sum()) for s in SAMPLES},
            "n_strong_nii_below_sn3": int((strong & work["log_nii_halpha_sn"].isna()).sum()),
            "n_strong_unclassified": int(work["strong_unclassified"].eq(1).sum()),
            "fractions": _fraction_table(work),
            "fisher_strong_halpha": _fisher(work, "strong_halpha"),
            "bpt_of_strong_emitters": _bpt_of_strong(work),
            "models": adjusted_models(work),
            "sf_emitter_strength": emitter_strength(work),
        }
    )
