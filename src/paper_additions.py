"""Submission additions: every new quantity added to the manuscript at submission.

This is the single script behind the referee-anticipating additions of the
``submission-additions`` branch. It recomputes, from the committed processed
sample and committed catalogues only, the quantities inserted into the paper
(separation gradation, conditional quenched fractions and their Kitagawa
decomposition, Zheng--Shen class diagnostics, tidal-index gaps and the
host-inclusive tidal-index robustness) and writes

* ``output/paper_additions.json``  -- full-precision values + provenance,
* ``output/paper/additions_macros.tex`` -- one ``\\newcommand`` per macro key,
  mechanically converted from the JSON ``macros`` block; the manuscript
  references the new numbers only through these macros.

No published number, seed, or sample definition is modified. Published
values quoted for internal-consistency displays are read from
``output/results.json`` and re-emitted unchanged. Every stochastic step uses
``numpy.random.default_rng(42)`` (a fresh generator per block, so blocks are
order-independent); bootstraps and permutations are vectorised with
per-group count arrays. Runtime well under 5 minutes.

Run with either
    python src/paper_additions.py
    (cd src && python paper_additions.py)
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import config as co  # noqa: E402
from astropy.cosmology import Planck15  # noqa: E402
from scipy.stats import chi2_contingency, spearmanr  # noqa: E402

from extended_data import build_galaxy_frame  # noqa: E402
from extended_stats import fit_logistic_model  # noqa: E402
from tidal_indices import _angular_matrix, _derive  # noqa: E402

SEED = 42
SAMPLES = ["CG4", "Control4B", "Control4C", "RG4"]
MORPHS = ["Elliptical", "Spiral", "Uncertain"]
LGM_TIDAL_WINDOW = (6.0, 13.0)  # paper convention: outside -> zero mass
JSON_PATH = os.path.join(co.OUTPUT_PATH, "paper_additions.json")
MACROS_PATH = os.path.join(co.REPORT_PATH, "additions_macros.tex")

PUBLISHED_COUNTS = {
    "CG4": dict(groups=62, gals=248, E=124, Sp=86, U=38, Q=146, SF=84, N=18),
    "Control4B": dict(groups=698, gals=2792, E=1106, Sp=1266, U=420,
                      Q=1593, SF=1019, N=180),
    "Control4C": dict(groups=703, gals=2812, E=1118, Sp=1229, U=465,
                      Q=1560, SF=1061, N=191),
    "RG4": dict(groups=56, gals=224, E=60, Sp=129, U=35, Q=101, SF=118, N=5),
}

REGRESSION_CHECKS: list[dict] = []


def check(name, computed, reference, tol, note=""):
    """Record a recomputed value against its independent regression reference."""

    status = "PASS" if abs(computed - reference) <= tol else "FLAG"
    REGRESSION_CHECKS.append(
        dict(name=name, computed=float(computed), reference=float(reference),
             tol=float(tol), status=status, note=note)
    )
    print(f"  [{status}] {name}: {computed:.4g} (ref {reference:.4g} ± {tol:g})"
          f"{'  -- ' + note if note else ''}")
    return status


# --------------------------------------------------------------------------
# gates and shared helpers
# --------------------------------------------------------------------------

def load_samples() -> dict:
    with open(co.DATA_PATH + co.PROCESS_SAMPLES, "rb") as fh:
        return pickle.load(fh)


def canonical_gate(sample: dict) -> dict:
    """Verify EXACT agreement with the published sample counts before anything."""

    report = {}
    for name, ref in PUBLISHED_COUNTS.items():
        g = sample[name + "_Gals"]
        got = dict(
            groups=int(g["Group"].nunique()), gals=int(len(g)),
            E=int((g["morphology"] == "Elliptical").sum()),
            Sp=int((g["morphology"] == "Spiral").sum()),
            U=int((g["morphology"] == "Uncertain").sum()),
            Q=int((g["sSFR_status"] == "Quenched").sum()),
            SF=int((g["sSFR_status"] == "Starforming").sum()),
            N=int((g["sSFR_status"] == "NosSFR").sum()),
        )
        if got != ref:
            raise SystemExit(
                f"CANONICAL-SAMPLE GATE FAILED for {name}: got {got}, "
                f"published {ref}. Aborting; nothing was written."
            )
        report[name] = got
    bgg = sample["CG4_Gals"].query("rank_M == 1")
    bgg_split = tuple(int((bgg["sSFR_status"] == s).sum())
                      for s in ("Quenched", "Starforming", "NosSFR"))
    if bgg_split != (45, 11, 6):
        raise SystemExit(f"CANONICAL gate: CG4 BGG Q/SF/N {bgg_split} != (45, 11, 6)")
    report["CG4_BGG_Q_SF_N"] = list(bgg_split)
    print("Canonical-sample gate: PASS (all four samples exact, incl. CG4 BGGs)")
    return report


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(centre - half), float(centre + half))


# --------------------------------------------------------------------------
# Task 1 -- per-group median pairwise projected separations
# --------------------------------------------------------------------------

def group_median_separations(gals: pd.DataFrame) -> pd.Series:
    """Median of the 6 pairwise projected separations (proper kpc) per quartet.

    Convention as in the paper: haversine angle x angular-diameter distance
    at the quartet's median redshift (Planck 2015).
    """

    out = {}
    for gid, part in gals.groupby("Group"):
        ra = part["RA"].to_numpy(float)
        dec = part["Dec"].to_numpy(float)
        zs = part["z"].to_numpy(float)
        if len(part) < 2 or np.isnan(ra).any() or np.isnan(zs).any():
            continue
        ang = _angular_matrix(ra, dec)
        d_a = Planck15.angular_diameter_distance(float(np.median(zs))).to_value("kpc")
        iu = np.triu_indices(len(part), k=1)
        out[gid] = float(np.median(ang[iu] * d_a))
    return pd.Series(out, name="R_pair_med_kpc")


def separations_block(sample: dict) -> dict:
    print("\n== Task 1: median pairwise separations ==")
    block = {}
    for name in SAMPLES:
        med = group_median_separations(sample[name + "_Gals"])
        block[name] = dict(
            n_groups=int(len(med)),
            median_kpc=float(med.median()),
            q25_kpc=float(med.quantile(0.25)),
            q75_kpc=float(med.quantile(0.75)),
        )
    check("sep_median_CG4_kpc", block["CG4"]["median_kpc"], 143, 2)
    check("sep_q25_CG4_kpc", block["CG4"]["q25_kpc"], 114, 2)
    check("sep_q75_CG4_kpc", block["CG4"]["q75_kpc"], 177, 2)
    check("sep_median_RG4_kpc", block["RG4"]["median_kpc"], 430, 2)
    check("sep_q25_RG4_kpc", block["RG4"]["q25_kpc"], 292, 2)
    check("sep_q75_RG4_kpc", block["RG4"]["q75_kpc"], 494, 2)
    check("sep_median_Control4B_kpc", block["Control4B"]["median_kpc"], 448, 10)
    check("sep_median_Control4C_kpc", block["Control4C"]["median_kpc"], 212, 40,
          note="reference is the pre-repair pickle variant; the canonical "
               "Delta_m<=3 Control4C is expected less compact (see report)")
    block["span_factor"] = float(
        max(v["median_kpc"] for k, v in block.items() if k in SAMPLES)
        / min(v["median_kpc"] for k, v in block.items() if k in SAMPLES)
    )
    block["loose_pair_mean_kpc"] = float(
        (block["Control4B"]["median_kpc"] + block["RG4"]["median_kpc"]) / 2
    )
    return block


# --------------------------------------------------------------------------
# Task 2 -- conditional quenched fractions and Kitagawa decomposition
# --------------------------------------------------------------------------

def group_count_arrays(gals: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-group (classified, quenched) counts by morphology, for blocked draws."""

    groups = np.sort(gals["Group"].unique())
    cl = gals[gals["sSFR_status"].isin(["Quenched", "Starforming"])]
    n = (cl.pivot_table(index="Group", columns="morphology", aggfunc="size",
                        fill_value=0)
         .reindex(index=groups, columns=MORPHS, fill_value=0))
    q = (cl[cl["sSFR_status"] == "Quenched"]
         .pivot_table(index="Group", columns="morphology", aggfunc="size",
                      fill_value=0)
         .reindex(index=groups, columns=MORPHS, fill_value=0))
    return n.to_numpy(float), q.to_numpy(float)


def _mix_cond(n_cg, q_cg, n_ct, q_ct):
    """Kitagawa split of Delta f_Q into mix and conditional terms (exact)."""

    f_cg = n_cg / n_cg.sum(axis=-1, keepdims=True)
    f_ct = n_ct / n_ct.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        qm_cg = np.where(n_cg > 0, q_cg / np.where(n_cg > 0, n_cg, 1), 0.0)
        qm_ct = np.where(n_ct > 0, q_ct / np.where(n_ct > 0, n_ct, 1), 0.0)
    mix = ((f_cg - f_ct) * (qm_cg + qm_ct) / 2).sum(axis=-1)
    cond = ((f_cg + f_ct) / 2 * (qm_cg - qm_ct)).sum(axis=-1)
    return mix, cond


def quenched_block(sample: dict, n_boot: int = 4000) -> dict:
    print("\n== Task 2: P(Q|morphology), homogeneity, Kitagawa ==")
    block = {"per_sample": {}, "chi2_homogeneity_PQE": {}, "kitagawa": {}}
    counts = {}
    for name in SAMPLES:
        g = sample[name + "_Gals"]
        cl = g[g["sSFR_status"].isin(["Quenched", "Starforming"])]
        per = {}
        for m in MORPHS:
            part = cl[cl["morphology"] == m]
            k, n = int((part["sSFR_status"] == "Quenched").sum()), int(len(part))
            lo, hi = wilson(k, n)
            per[m] = dict(quenched=k, classified=n, p=k / n if n else np.nan,
                          wilson_lo=lo, wilson_hi=hi)
        block["per_sample"][name] = per
        counts[name] = per

    refs_e = {"CG4": (0.908, 0.02), "Control4B": (0.903, 0.02),
              "Control4C": (0.874, 0.03), "RG4": (0.850, 0.02)}
    refs_s = {"CG4": (0.326, 0.02), "Control4B": (0.357, 0.02),
              "Control4C": (0.315, 0.03), "RG4": (0.295, 0.02)}
    for name in SAMPLES:
        note = "reference from the pre-repair pickle variant" \
            if name == "Control4C" else ""
        check(f"PQE_{name}", counts[name]["Elliptical"]["p"], *refs_e[name], note)
        check(f"PQSp_{name}", counts[name]["Spiral"]["p"], *refs_s[name], note)

    table = np.array([[counts[s]["Elliptical"]["quenched"],
                       counts[s]["Elliptical"]["classified"]
                       - counts[s]["Elliptical"]["quenched"]] for s in SAMPLES])
    chi2, p_hom, dof, _ = chi2_contingency(table)
    block["chi2_homogeneity_PQE"] = dict(chi2=float(chi2), p=float(p_hom),
                                         dof=int(dof))
    if not (0.03 <= p_hom <= 0.3):
        check("chi2_PQE_homogeneity_p", p_hom, 0.10, 0.0,
              note="outside the stated 0.03-0.3 plausibility window")
    else:
        check("chi2_PQE_homogeneity_p", p_hom, p_hom, 1.0,
              note="within the stated 0.03-0.3 window")

    kit_refs = {"Control4B": (-0.034, 0.02, ""),
                "Control4C": (0.001, 0.03,
                              "reference from the pre-repair pickle variant"),
                "RG4": (0.035, 0.02, "")}
    n_cg, q_cg = group_count_arrays(sample["CG4_Gals"])
    for ctrl in ["Control4B", "Control4C", "RG4"]:
        n_ct, q_ct = group_count_arrays(sample[ctrl + "_Gals"])
        mix, cond = _mix_cond(n_cg.sum(0), q_cg.sum(0), n_ct.sum(0), q_ct.sum(0))
        raw = (q_cg.sum() / n_cg.sum()) - (q_ct.sum() / n_ct.sum())
        assert abs(raw - (mix + cond)) < 1e-12, "Kitagawa identity violated"

        rng = np.random.default_rng(SEED)
        g_cg, g_ct = len(n_cg), len(n_ct)
        w_cg = rng.multinomial(g_cg, np.full(g_cg, 1 / g_cg), size=n_boot)
        w_ct = rng.multinomial(g_ct, np.full(g_ct, 1 / g_ct), size=n_boot)
        mix_b, cond_b = _mix_cond(w_cg @ n_cg, w_cg @ q_cg,
                                  w_ct @ n_ct, w_ct @ q_ct)
        block["kitagawa"][ctrl] = dict(
            raw_delta_fQ=float(raw), mix_term=float(mix),
            conditional_term=float(cond),
            conditional_ci95=[float(np.percentile(cond_b, 2.5)),
                              float(np.percentile(cond_b, 97.5))],
            mix_ci95=[float(np.percentile(mix_b, 2.5)),
                      float(np.percentile(mix_b, 97.5))],
            n_boot=n_boot, seed=SEED,
            blocked_by="group within each sample (multinomial group weights)",
        )
        ref, tol, note = kit_refs[ctrl]
        check(f"kitagawa_conditional_{ctrl}", cond, ref, tol, note)
    return block


# --------------------------------------------------------------------------
# Task 3 -- Zheng--Shen classes
# --------------------------------------------------------------------------

def _perm_test(a: np.ndarray, b: np.ndarray, n_perm: int = 20000,
               seed: int = SEED) -> dict:
    """Two-sided group-level permutation test on the difference of means."""

    obs = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    perm = rng.permuted(np.broadcast_to(pooled, (n_perm, len(pooled))).copy(),
                        axis=1)
    stat = perm[:, :len(a)].mean(axis=1) - perm[:, len(a):].mean(axis=1)
    p = float((np.sum(np.abs(stat) >= abs(obs) - 1e-12) + 1) / (n_perm + 1))
    return dict(observed_diff_of_group_means=obs, p=p, n_perm=n_perm, seed=seed,
                n_a=int(len(a)), n_b=int(len(b)))


def _group_sat_fe(gals: pd.DataFrame) -> pd.Series:
    """Per-group satellite elliptical fraction f_E = E/(E+Sp), classified only."""

    sat = gals[gals["rank_M"] > 1]
    cl = sat[sat["morphology"].isin(["Elliptical", "Spiral"])]
    return cl.groupby("Group")["morphology"].agg(lambda v: (v == "Elliptical").mean())


def zheng_shen_block(sample: dict) -> dict:
    print("\n== Task 3: Zheng--Shen classes ==")
    gals, groups = sample["CG4_Gals"], sample["CG4_Groups"]
    cls = groups.set_index("Group")["Class"]
    lm200 = groups.set_index("Group")["lMass_200"]
    block = {"per_class": {}, "permutations": {}}
    display = {"Isolated": "Isolated", "Embedded": "Embedded",
               "Predom": "Predominant"}
    for cname in ["Isolated", "Embedded", "Predom"]:
        gset = cls.index[cls == cname]
        part = gals[gals["Group"].isin(gset)]
        sat = part[part["rank_M"] > 1]
        entry = {"n_groups": int(len(gset)),
                 "median_host_lM200": float(lm200.loc[gset].median())}
        for scope, d in (("all", part), ("sat", sat)):
            cl = d[d["morphology"].isin(["Elliptical", "Spiral"])]
            k, n = int((cl["morphology"] == "Elliptical").sum()), int(len(cl))
            lo, hi = wilson(k, n)
            entry[f"fE_{scope}"] = dict(n_E=k, n_classified=n,
                                        p=k / n if n else np.nan,
                                        wilson_lo=lo, wilson_hi=hi)
        block["per_class"][display[cname]] = entry

    iso = block["per_class"]["Isolated"]
    check("fE_isolated_all", iso["fE_all"]["p"], 7 / 18, 1e-9)
    check("fE_isolated_sat", iso["fE_sat"]["p"], 4 / 14, 1e-9)
    check("lM200_isolated", iso["median_host_lM200"], 12.86, 0.02)
    check("lM200_embedded", block["per_class"]["Embedded"]["median_host_lM200"],
          13.13, 0.02)
    check("lM200_predominant",
          block["per_class"]["Predom" if "Predom" in block["per_class"]
                             else "Predominant"]["median_host_lM200"],
          13.83, 0.02)

    fe_cg = _group_sat_fe(gals)
    iso_groups = set(cls.index[cls == "Isolated"])
    a = fe_cg[fe_cg.index.isin(iso_groups)].to_numpy()
    b = fe_cg[~fe_cg.index.isin(iso_groups)].to_numpy()
    block["permutations"]["isolated_vs_rest_of_CG4"] = _perm_test(a, b)
    fe_rg = _group_sat_fe(sample["RG4_Gals"]).to_numpy()
    block["permutations"]["isolated_vs_RG4"] = _perm_test(a, fe_rg)
    fe_cc = _group_sat_fe(sample["Control4C_Gals"]).to_numpy()
    block["permutations"]["isolated_vs_Control4C"] = _perm_test(a, fe_cc)
    block["statistic"] = ("difference of the means over groups of the per-group "
                          "satellite elliptical fraction E/(E+Sp); two-sided, "
                          "add-one convention p=(k+1)/(B+1)")

    check("perm_iso_vs_rest_p",
          block["permutations"]["isolated_vs_rest_of_CG4"]["p"], 0.022, 0.01)
    check("perm_iso_vs_RG4_p", block["permutations"]["isolated_vs_RG4"]["p"],
          1.0, 0.1)
    check("perm_iso_vs_C4C_p",
          block["permutations"]["isolated_vs_Control4C"]["p"], 0.3, 0.15,
          note="reference approximate; Control4C is the repaired sample")
    return block


# --------------------------------------------------------------------------
# Task 4 -- tidal-index gaps, standardisation, host-inclusive robustness
# --------------------------------------------------------------------------

def load_lim_members() -> pd.DataFrame:
    dat = pd.read_csv(co.DATA_PATH + "SDSS(L) galaxy.dat", sep=r"\s+",
                      comment="#", header=None, usecols=[1, 2, 3, 4],
                      names=["objid", "limgroup", "RA", "Dec"])
    dat["objid"] = dat["objid"].astype("int64")
    return dat


def load_sdss_masses() -> pd.Series:
    sdss = pd.read_csv(co.OUTPUT_PATH + "SDSS_processed.csv",
                       usecols=["objid", "lgm"])
    sdss["objid"] = sdss["objid"].astype("int64")
    lgm = sdss.set_index("objid")["lgm"]
    return lgm[(lgm > LGM_TIDAL_WINDOW[0]) & (lgm < LGM_TIDAL_WINDOW[1])]


def host_inclusive_block(sample: dict, work: pd.DataFrame) -> dict:
    """T_i over quartet co-members plus all spectroscopic Lim host members.

    Membership and positions come from the Lim catalogue (``SDSS(L)
    galaxy.dat``); stellar masses of the additional members come from the
    SDSS selection (``output/SDSS_processed.csv``); members absent from that
    selection contribute zero mass, so the host-inclusive T_i is a lower
    bound. Separations use the same convention as the quartet T_i (haversine
    x angular-diameter distance at the *quartet's* median redshift).
    """

    print("\n== Task 4d: host-inclusive tidal index ==")
    dat = load_lim_members()
    lgm = load_sdss_masses()
    members_by_group = {gid: part for gid, part in dat.groupby("limgroup")}

    cg_modal = (sample["CG4_Gals"][["objid", "Group"]]
                .merge(dat[["objid", "limgroup"]], on="objid", how="left")
                .groupby("Group")["limgroup"]
                .agg(lambda v: v.dropna().mode().iloc[0] if v.notna().any()
                     else np.nan))

    extra_sum = pd.Series(0.0, index=work.index)
    n_extra_members = pd.Series(0, index=work.index)
    for uid, part in work.groupby("group_uid", observed=True):
        label, gid_str = uid.split(":", 1)
        gid = float(gid_str)
        host = cg_modal.get(gid, np.nan) if label == "CG4" else gid
        if pd.isna(host) or host not in members_by_group:
            continue
        mem = members_by_group[host]
        extra = mem[~mem["objid"].isin(set(part["objid"].astype("int64")))]
        extra = extra[extra["objid"].isin(lgm.index)]
        if extra.empty:
            continue
        ra = np.concatenate([part["RA"].to_numpy(float),
                             extra["RA"].to_numpy(float)])
        dec = np.concatenate([part["Dec"].to_numpy(float),
                              extra["Dec"].to_numpy(float)])
        if np.isnan(part[["RA", "Dec", "z_numeric"]].to_numpy(float)).any():
            continue
        ang = _angular_matrix(ra, dec)[: len(part), len(part):]
        z_med = float(np.median(part["z_numeric"].to_numpy(float)))
        d_kpc = ang * Planck15.angular_diameter_distance(z_med).to_value("kpc")
        masses = np.power(10.0, lgm.loc[extra["objid"]].to_numpy(float))
        with np.errstate(divide="ignore"):
            contrib = (masses[None, :] / d_kpc ** 3).sum(axis=1)
        extra_sum.loc[part.index] = contrib
        n_extra_members.loc[part.index] = len(extra)

    t_host = work["tidal_index_sum"] + extra_sum
    # keep the estimation sample identical to the published model: rows whose
    # quartet-only T is undefined stay undefined under the host-inclusive T
    log_t_host = np.log10(t_host.where(t_host > 0)).where(
        work["log_tidal_index"].notna())
    delta = log_t_host - work["log_tidal_index"]

    block = {"per_sample": {}, "n_lim_members_catalogue": int(len(dat))}
    ok = delta.notna()
    for name in SAMPLES:
        m = (work["sample"] == name) & ok
        d = delta[m]
        rho = spearmanr(work.loc[m, "log_tidal_index"], log_t_host[m]).statistic
        block["per_sample"][name] = dict(
            n=int(m.sum()), median_delta_dex=float(d.median()),
            q90_delta_dex=float(d.quantile(0.90)),
            frac_delta_above_03=float((d > 0.3).mean()),
            max_delta_dex=float(d.max()), spearman_rho=float(rho),
            median_extra_members=float(n_extra_members[m].median()),
        )
    pooled_rho = spearmanr(work.loc[ok, "log_tidal_index"],
                           log_t_host[ok]).statistic
    block["pooled"] = dict(n=int(ok.sum()),
                           median_delta_dex=float(delta[ok].median()),
                           spearman_rho=float(pooled_rho))

    for name in SAMPLES:
        check(f"hostT_median_delta_{name}",
              block["per_sample"][name]["median_delta_dex"], 0.0, 0.03,
              note="criterion: median shift <= 0.03 dex")
        check(f"hostT_spearman_{name}",
              block["per_sample"][name]["spearman_rho"], 1.0, 0.05,
              note="criterion: rho >= 0.95")
    if block["per_sample"]["RG4"]["max_delta_dex"] != 0.0:
        raise SystemExit("GATE: RG4 host-inclusive T must equal quartet T "
                         "exactly (groups of exactly four members)")
    print("  RG4 internal check: host-inclusive T identical to quartet T (exact)")

    refit_frame = work.copy()
    refit_frame["log_tidal_index"] = log_t_host
    refit = fit_logistic_model(
        refit_frame, "elliptical", ["is_CG4", "logMstar", "is_satellite",
                                    "log_tidal_index"],
        continuous=["logMstar", "log_tidal_index"],
    )
    block["refit_elliptical_with_host_T"] = dict(
        cg4_odds_ratio=refit.get("cg4_odds_ratio"),
        cg4_ci95=refit.get("cg4_ci95"), cg4_p=refit.get("cg4_p"),
        n=refit.get("n"),
        note="identical specification to the published with-tidal-index model "
             "(fit_logistic_model, standardised continuous terms, cluster-"
             "robust by physical_group), with log T_i replaced by the "
             "host-inclusive log T_i",
    )
    return block


def tidal_block(sample: dict, results: dict) -> tuple[dict, pd.DataFrame]:
    print("\n== Task 4: tidal-index gaps and standardisation ==")
    frame = build_galaxy_frame(sample)
    work = _derive(frame)

    pub = results["extended_specialness"]["tidal_indices"]
    pub_e = pub["models"]["elliptical"]
    block = {"published_inputs": dict(
        baseline_or=pub_e["baseline"]["cg4_odds_ratio"],
        residual_or=pub_e["with_tidal_index"]["cg4_odds_ratio"],
        tidal_term_or=pub_e["with_tidal_index"]["terms"]["log_tidal_index"]
        ["odds_ratio"],
        pooled_median_gap_dex=pub["summary_by_sample"]["log_tidal_index"]
        ["delta_median"],
    )}

    med = work.groupby("sample")["log_tidal_index"].median()
    check("tidal_median_logT_CG4_recomputed", med["CG4"],
          pub["summary_by_sample"]["log_tidal_index"]["median_cg4"], 1e-6,
          note="recomputation must reproduce the published pipeline value")
    gaps = {c: float(med["CG4"] - med[c]) for c in
            ["Control4B", "Control4C", "RG4"]}
    block["median_logT_by_sample"] = {k: float(v) for k, v in med.items()}
    block["gap_vs_control_dex"] = gaps
    check("tidal_gap_Control4B", gaps["Control4B"], 1.34, 0.1)
    check("tidal_gap_Control4C", gaps["Control4C"], 0.53, 0.2,
          note="reference from the pre-repair pickle variant")
    check("tidal_gap_RG4", gaps["RG4"], 1.22, 0.1)

    model_cols = ["elliptical", "is_CG4", "logMstar", "is_satellite",
                  "log_tidal_index"]
    cc = work[model_cols].replace([np.inf, -np.inf], np.nan).dropna()
    sd = float(cc["log_tidal_index"].std(ddof=0))
    block["logT_sd_in_elliptical_model_frame"] = dict(
        sd_dex=sd, n=int(len(cc)),
        note="population SD (ddof=0) of log10 T_i over the with-tidal-index "
             "elliptical model complete cases; the model standardises the "
             "regressor, so the published OR 1.42 is per this SD, not per dex")
    if len(cc) != pub_e["with_tidal_index"]["n"]:
        check("tidal_model_complete_cases", len(cc),
              pub_e["with_tidal_index"]["n"], 0,
              note="complete-case reconstruction mismatch")
    else:
        print(f"  model complete-case reconstruction: n={len(cc)} matches "
              f"published n={pub_e['with_tidal_index']['n']}")

    gap = block["published_inputs"]["pooled_median_gap_dex"]
    consistency = (block["published_inputs"]["residual_or"]
                   * block["published_inputs"]["tidal_term_or"]
                   ** (gap / sd))
    block["internal_consistency"] = dict(
        product_or=float(consistency),
        exponent_gap_over_sd=float(gap / sd),
        formula="residual_OR * tidal_OR^(pooled median gap / SD of log T)",
        compare_to_baseline_or=block["published_inputs"]["baseline_or"],
    )
    print(f"  consistency: {block['published_inputs']['residual_or']:.2f} x "
          f"{block['published_inputs']['tidal_term_or']:.2f}^"
          f"({gap:.2f}/{sd:.2f}) = {consistency:.2f} "
          f"(baseline {block['published_inputs']['baseline_or']:.2f})")

    # group-level lMass_200 / r_200_kpc / Class are already merged into the
    # harmonised frame for the CG4 rows by build_galaxy_frame
    cgj = work[(work["sample"] == "CG4") & (work["dist2BGG_kpc"] > 0)]
    emb = cgj[cgj["Class"].isin(["Embedded", "Predom"])].dropna(
        subset=["tidal_index_sum", "lMass_200"])
    block["host_halo_tide_diagnostic"] = dict(
        n_embedded_predominant_satellites=int(len(emb)),
        median_log10_ratio_pointmass_at_bgg_distance=float(np.median(
            np.log10((10 ** emb["lMass_200"] / emb["dist2BGG_kpc"] ** 3)
                     / emb["tidal_index_sum"]))),
        median_log10_ratio_mean_within_r200=float(np.median(
            np.log10((10 ** emb["lMass_200"] / emb["r_200_kpc"] ** 3)
                     / emb["tidal_index_sum"]))),
        note="supports the qualitative 'order of magnitude' host-tide clause; "
             "convention-dependent (point-mass at the galaxy's BGG-centric "
             "distance versus mean density within r200), diagnostic only",
    )

    block["host_inclusive"] = host_inclusive_block(sample, work)
    refit_or = block["host_inclusive"]["refit_elliptical_with_host_T"][
        "cg4_odds_ratio"]
    check("hostT_refit_or_shift",
          refit_or - block["published_inputs"]["residual_or"], 0.0, 0.05,
          note="criterion: refitted CG4 OR shift <= 0.05")
    return block, work


# --------------------------------------------------------------------------
# macro emission
# --------------------------------------------------------------------------

def _fmt(x: float, nd: int, sign: bool = False) -> str:
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s


def _cell(entry: dict, nd: int = 3) -> str:
    return (f"{entry['p']:.{nd}f}\\,[{entry['wilson_lo']:.{nd}f}, "
            f"{entry['wilson_hi']:.{nd}f}]")


def build_macros(sep, quench, zheng, tidal) -> dict:
    m = {}
    short = {"CG4": "CG", "Control4B": "CB", "Control4C": "CC", "RG4": "RG"}
    for name, sh in short.items():
        m[f"Sep{sh}"] = _fmt(sep[name]["median_kpc"], 0)
        m[f"Sep{sh}Iqr"] = (f"{sep[name]['q25_kpc']:.0f}--"
                            f"{sep[name]['q75_kpc']:.0f}")
    m["SepSpan"] = _fmt(sep["span_factor"], 1)
    m["SepLoose"] = f"{round(sep['loose_pair_mean_kpc'] / 10) * 10:d}"

    for name, sh in short.items():
        per = quench["per_sample"][name]
        m[f"qE{sh.lower()}"] = _fmt(per["Elliptical"]["p"], 3)
        m[f"qSp{sh.lower()}"] = _fmt(per["Spiral"]["p"], 3)
        m[f"qU{sh.lower()}"] = _fmt(per["Uncertain"]["p"], 3)
        m[f"qE{sh.lower()}Cell"] = _cell(per["Elliptical"])
        m[f"qSp{sh.lower()}Cell"] = _cell(per["Spiral"])
        m[f"qU{sh.lower()}Cell"] = _cell(per["Uncertain"])
        m[f"nU{sh.lower()}"] = str(per["Uncertain"]["classified"])
    m["pQEhom"] = _fmt(quench["chi2_homogeneity_PQE"]["p"], 2)
    for ctrl, sh in (("Control4B", "CB"), ("Control4C", "CC"), ("RG4", "RG")):
        kit = quench["kitagawa"][ctrl]
        m[f"cond{sh}"] = _fmt(kit["conditional_term"], 3, sign=True)
        m[f"cond{sh}lo"] = _fmt(kit["conditional_ci95"][0], 3, sign=True)
        m[f"cond{sh}hi"] = _fmt(kit["conditional_ci95"][1], 3, sign=True)
        m[f"condBound{sh}"] = _fmt(max(abs(kit["conditional_ci95"][0]),
                                       abs(kit["conditional_ci95"][1])), 2)

    zs_short = {"Isolated": "Iso", "Embedded": "Emb", "Predominant": "Pre"}
    for cname, sh in zs_short.items():
        entry = zheng["per_class"][cname]
        m[f"feAll{sh}"] = _cell(entry["fE_all"])
        m[f"feSat{sh}"] = _cell(entry["fE_sat"])
        m[f"lM{sh}"] = _fmt(entry["median_host_lM200"], 2)
        m[f"nGr{sh}"] = str(entry["n_groups"])
    m["pIsoRest"] = _fmt(zheng["permutations"]["isolated_vs_rest_of_CG4"]["p"], 3)
    m["pIsoRG"] = _fmt(zheng["permutations"]["isolated_vs_RG4"]["p"], 2)
    m["pIsoCC"] = _fmt(zheng["permutations"]["isolated_vs_Control4C"]["p"], 2)

    for ctrl, sh in (("Control4B", "CB"), ("Control4C", "CC"), ("RG4", "RG")):
        m[f"gap{sh}"] = _fmt(tidal["gap_vs_control_dex"][ctrl], 2, sign=True)
    m["sdlogT"] = _fmt(tidal["logT_sd_in_elliptical_model_frame"]["sd_dex"], 2)
    m["ORbase"] = _fmt(tidal["published_inputs"]["baseline_or"], 2)
    m["ORresid"] = _fmt(tidal["published_inputs"]["residual_or"], 2)
    m["ORtidal"] = _fmt(tidal["published_inputs"]["tidal_term_or"], 2)
    m["gapPooled"] = _fmt(tidal["published_inputs"]["pooled_median_gap_dex"], 2)
    m["ORconsistency"] = _fmt(tidal["internal_consistency"]["product_or"], 2)
    m["expTid"] = _fmt(tidal["internal_consistency"]["exponent_gap_over_sd"], 2)

    hi = tidal["host_inclusive"]
    m["dTall"] = _fmt(hi["pooled"]["median_delta_dex"], 3)
    m["dTmaxSamp"] = _fmt(max(v["median_delta_dex"]
                              for v in hi["per_sample"].values()), 2)
    m["rhoTT"] = _fmt(hi["pooled"]["spearman_rho"], 2)
    m["ORresidAll"] = _fmt(hi["refit_elliptical_with_host_T"]["cg4_odds_ratio"], 2)
    return m


def write_macros(macros: dict) -> None:
    lines = [
        "% additions_macros.tex -- generated by src/paper_additions.py.",
        "% One \\newcommand per key of the 'macros' block of",
        "% output/paper_additions.json (mechanical conversion). Do not edit;",
        "% rerun the script instead. Provenance: CHANGES_SUBMISSION.md.",
    ]
    for name in sorted(macros):
        lines.append(f"\\newcommand{{\\{name}}}{{{macros[name]}}}")
    with open(MACROS_PATH, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote {len(macros)} macros to {MACROS_PATH}")


def main() -> None:
    t0 = time.time()
    sample = load_samples()
    gate = canonical_gate(sample)
    with open(co.RESULTS) as fh:
        results = json.load(fh)

    sep = separations_block(sample)
    quench = quenched_block(sample)
    zheng = zheng_shen_block(sample)
    tidal, _ = tidal_block(sample, results)

    macros = build_macros(sep, quench, zheng, tidal)
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=co.BASE_PATH).stdout.strip()
    except Exception:
        head = "unknown"
    payload = dict(
        meta=dict(script="src/paper_additions.py", seed=SEED,
                  generated_utc=pd.Timestamp.utcnow().isoformat(),
                  git_head=head, runtime_s=round(time.time() - t0, 1)),
        canonical_gate=gate,
        separations=sep,
        quenched_by_morphology=quench,
        zheng_shen=zheng,
        tidal=tidal,
        regression_checks=REGRESSION_CHECKS,
        macros=macros,
    )
    with open(JSON_PATH, "w") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")
    print(f"Wrote {JSON_PATH}")
    write_macros(macros)

    flags = [c for c in REGRESSION_CHECKS if c["status"] == "FLAG"]
    print(f"\nRegression checks: {len(REGRESSION_CHECKS) - len(flags)} PASS, "
          f"{len(flags)} FLAG; runtime {time.time() - t0:.1f} s")
    for c in flags:
        print(f"  FLAG {c['name']}: {c['computed']:.4g} vs {c['reference']:.4g} "
              f"± {c['tol']:g}  ({c['note']})")


if __name__ == "__main__":
    main()
