"""bgg-split Phase 2/3: tables for PHASE2_3_REPORT.md from a results.json.

Usage: python analysis/bgg_split/phase2_3_tables.py <results.json>

Writes ``phase2_3_tables.md`` and ``preliminary_reconciliation.json`` next to
the report.  The reconciliation refits the preliminary specifications
(B-spline ``bs(logMstar, df=3)`` for Models I/R; a linear mass term on the
10.0-10.5 satellite bin for the Model S proxy) on the rebuilt frame, so every
difference larger than one standard error can be attributed.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import OUT, load_galaxy_frame

from extended_stats import safe_json  # noqa: E402
from rank_split_properties import SUPPORT_MIN_LOGM, prepare_frame  # noqa: E402

LABEL = {"Control4B": "C4B", "Control4C": "C4C", "RG4": "RG4"}
NAME = {
    "strong_halpha": "strong Hα", "agn_like": "AGN-like", "Dn4000": "D_n4000",
    "log_Rchl_r_kpc": "log R_e (Simard)", "log_petroR50_kpc": "log R50 (Petrosian)",
    "log_Rchl_r_kpc_incl_pegged": "log R_e (Simard, incl. pegged)",
}
# (outcome, control): (CG_sat, se, CG_bgg, se, interaction, se, raw p)
PRELIM_IR = {
    ("strong_halpha", "Control4B"): (-1.03, .51, -.44, .39, .59, .66, .37),
    ("strong_halpha", "Control4C"): (-.89, .51, -.44, .39, .45, .67, .50),
    ("strong_halpha", "RG4"): (-1.33, .70, -.35, .54, .98, .95, .30),
    ("agn_like", "Control4B"): (.12, .32, -.23, .31, -.35, .45, .44),
    ("agn_like", "Control4C"): (.17, .33, -.23, .31, -.40, .45, .38),
    ("agn_like", "RG4"): (.35, .57, .06, .48, -.29, .72, .69),
    ("Dn4000", "Control4B"): (.074, .022, -.009, .027, -.083, .034, .014),
    ("Dn4000", "Control4C"): (.054, .023, -.010, .027, -.064, .035, .064),
    ("Dn4000", "RG4"): (.079, .037, -.002, .039, -.081, .055, .14),
    ("log_Rchl_r_kpc", "Control4B"): (-.014, .028, .006, .027, .019, .039, .62),
    ("log_Rchl_r_kpc", "Control4C"): (-.001, .029, .004, .027, .005, .039, .89),
    ("log_Rchl_r_kpc", "RG4"): (-.035, .039, .016, .035, .051, .053, .35),
    ("log_petroR50_kpc", "Control4B"): (-.018, .028, -.007, .019, .011, .035, .75),
}
# satellites 10.0 <= log M < 10.5, linear mass term: (estimate, se)
PRELIM_S = {
    ("strong_halpha", "Control4B"): (-1.53, .28), ("strong_halpha", "Control4C"): (-.98, .28),
    ("strong_halpha", "RG4"): (-1.67, .42),
    ("agn_like", "Control4B"): (1.12, .28), ("agn_like", "Control4C"): (.84, .27),
    ("agn_like", "RG4"): (1.00, .46),
    ("Dn4000", "Control4B"): (.097, .031), ("Dn4000", "Control4C"): (.043, .030),
    ("Dn4000", "RG4"): (.090, .048),
    ("log_Rchl_r_kpc", "Control4B"): (-.070, .022), ("log_Rchl_r_kpc", "Control4C"): (-.013, .021),
    ("log_Rchl_r_kpc", "RG4"): (-.035, .035),
}
LOGIT = {"strong_halpha", "agn_like"}


def f(x, digits=3, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def p_text(p):
    if p is None:
        return "—"
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def ci(term, digits=3):
    low, high = term["ci95"]
    return f"[{low:+.{digits}f}, {high:+.{digits}f}]"


def mde_text(mde, logit):
    return f"OR {mde['odds_ratio_low']:.2f}/{mde['odds_ratio_high']:.2f}" if logit else f"±{mde['delta']:.3f}"


def model_ir_table(entries, holm=True) -> list[str]:
    head = ("| outcome | ctrl | n CG4 sat/BGG | CG_sat (95% CI) | boot CI | CG_bgg (95% CI) | boot CI "
            "| interaction (95% CI) | boot CI | raw p | Holm p | MDE sat | MDE bgg | code |")
    lines = [head, "|" + "---|" * 14]
    for e in entries:
        if e["status"] != "ok":
            lines.append(f"| {NAME[e['outcome']]} | {LABEL[e['control']]} | skipped: "
                         f"{e.get('model_R', {}).get('reason')} |" + " |" * 11)
            continue
        c, lg = e["counts"], e["outcome"] in LOGIT
        d = 2 if lg else 3
        boot = e.get("bootstrap", {})
        bt = lambda name: (f"[{boot[name]['ci95'][0]:+.{d}f}, {boot[name]['ci95'][1]:+.{d}f}]"  # noqa: E731
                           if name in boot else "—")
        lines.append(
            f"| {NAME[e['outcome']]} | {LABEL[e['control']]} | {c['n_cg4_sat']}/{c['n_cg4_bgg']} "
            f"| {f(e['cg_sat']['estimate'], d)} ± {f(e['cg_sat']['se'], d, False)} {ci(e['cg_sat'], d)} | {bt('cg_sat')} "
            f"| {f(e['cg_bgg']['estimate'], d)} ± {f(e['cg_bgg']['se'], d, False)} {ci(e['cg_bgg'], d)} | {bt('cg_bgg')} "
            f"| {f(e['interaction']['estimate'], d)} ± {f(e['interaction']['se'], d, False)} {ci(e['interaction'], d)} | {bt('interaction')} "
            f"| {p_text(e['interaction']['p'])} | {p_text(e.get('interaction_p_holm')) if holm else 'n/a'} "
            f"| {mde_text(e['mde']['cg_sat'], lg)} | {mde_text(e['mde']['cg_bgg'], lg)} "
            f"| {e.get('class_code', e.get('class_code_unadjusted', '—') + (' (raw p)' if not holm else ''))} |"
        )
    return lines


def model_s_table(entries, holm=True) -> list[str]:
    lines = ["| outcome | ctrl | n CG4 / control sat | CG (95% CI) | raw p | Holm p | MDE |",
             "|---|---|---|---|---|---|---|"]
    for e in entries:
        if e["status"] != "ok":
            lines.append(f"| {NAME[e['outcome']]} | {LABEL[e['control']]} | skipped | | | | |")
            continue
        c, lg = e["counts"], e["outcome"] in LOGIT
        d = 2 if lg else 3
        lines.append(
            f"| {NAME[e['outcome']]} | {LABEL[e['control']]} | {c['n_cg4_sat']}/{c['n_control_sat']} "
            f"| {f(e['cg']['estimate'], d)} ± {f(e['cg']['se'], d, False)} {ci(e['cg'], d)} "
            f"| {p_text(e['cg']['p'])} | {p_text(e.get('cg_p_holm')) if holm else 'n/a'} | {mde_text(e['mde'], lg)} |"
        )
    return lines


def reconcile(block: dict) -> dict:
    """Refit the preliminary specifications on the rebuilt frame."""

    frame, _ = load_galaxy_frame()
    work = prepare_frame(frame)
    work = work.loc[np.isfinite(work["logMstar"])]
    out = {"model_I_R": [], "model_S_proxy": []}
    entries = {(e["outcome"], e["control"]): e
               for e in block["interaction_models"] + block["sensitivity_interaction_models"]}
    support = work.loc[work["logMstar"] >= SUPPORT_MIN_LOGM]
    for (outcome, control), prelim in PRELIM_IR.items():
        e = entries[(outcome, control)]
        part = support.loc[support["sample"].isin(["CG4", control])].dropna(
            subset=[outcome, "physical_group"])
        formula = f"{outcome} ~ bs(logMstar, df=3) + is_bgg + cg_sat + cg_bgg"
        if outcome in LOGIT:
            import statsmodels.api as sm
            model = smf.glm(formula, part, family=sm.families.Binomial())
        else:
            model = smf.ols(formula, part)
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(part["physical_group"])[0]})
        refit = {"cg_sat": float(fit.params["cg_sat"]), "cg_bgg": float(fit.params["cg_bgg"]),
              "interaction": float(fit.params["cg_bgg"] - fit.params["cg_sat"])}
        row = {"outcome": outcome, "control": control}
        for index, term in enumerate(("cg_sat", "cg_bgg", "interaction")):
            mine = e[term]["estimate"]
            se = e[term]["se"]
            row[term] = {
                "prelim": prelim[2 * index], "prelim_se": prelim[2 * index + 1],
                "cr_centred": mine, "se": se, "bs_df3_refit": refit[term],
                "delta_in_se": (mine - prelim[2 * index]) / se,
                "bs_refit_minus_prelim": refit[term] - prelim[2 * index],
            }
        out["model_I_R"].append(row)
    sats = work.loc[work["is_bgg"].eq(0)]
    proxy_bin = sats.loc[(sats["logMstar"] >= 10.0) & (sats["logMstar"] < 10.5)]
    model_s = {(e["outcome"], e["control"]): e for e in block["satellite_models"]}
    for (outcome, control), (estimate, se) in PRELIM_S.items():
        part = proxy_bin.loc[proxy_bin["sample"].isin(["CG4", control])].dropna(
            subset=[outcome, "physical_group"])
        formula = f"{outcome} ~ logMstar + is_CG4"
        if outcome in LOGIT:
            import statsmodels.api as sm
            model = smf.glm(formula, part, family=sm.families.Binomial())
        else:
            model = smf.ols(formula, part)
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(part["physical_group"])[0]})
        s = model_s[(outcome, control)]
        out["model_S_proxy"].append({
            "outcome": outcome, "control": control,
            "prelim": estimate, "prelim_se": se,
            "proxy_refit": float(fit.params["is_CG4"]), "proxy_refit_se": float(fit.bse["is_CG4"]),
            "model_S": s["cg"]["estimate"], "model_S_se": s["cg"]["se"],
            "model_S_minus_prelim_in_se": (s["cg"]["estimate"] - estimate) / s["cg"]["se"],
        })
    return out


def main(path: str) -> None:
    with open(path, encoding="utf-8") as fh:
        es = json.load(fh)["extended_specialness"]
    block, diag = es["rank_split_properties"], es["rank_split_diagnostics"]
    md = ["## Model I/R (log M* ≥ 10.5), primary outcomes", ""]
    md += model_ir_table(block["interaction_models"])
    md += ["", "## Model I/R, sensitivity outcomes (outside the Holm family; code uses raw p)", ""]
    md += model_ir_table(block["sensitivity_interaction_models"], holm=False)
    md += ["", "## Model S (satellites, all masses)", ""]
    md += model_s_table(block["satellite_models"])
    md += ["", "## Model S, sensitivity outcomes", ""]
    md += model_s_table(block["sensitivity_satellite_models"], holm=False)

    md += ["", "## 3.1 Retired share of AGN-like galaxies (|EW(Hα)| < 3 Å)", "",
           "| sample | rank | all | <10.0 | 10.0–10.5 | 10.5–11.0 | ≥11.0 |", "|---|---|---|---|---|---|---|"]
    cells = {(r["sample"], r["rank"], r["mass_bin"]): r for r in diag["retired_share"]}
    for sample in ("CG4", "Control4B", "Control4C", "RG4"):
        for rank in ("bgg", "satellite"):
            row = [sample, rank]
            for label in ("all", "7-10", "10-10.5", "10.5-11", "11-12.5"):
                r = cells.get((sample, rank, label))
                row.append(f"{r['n_retired_like']}/{r['n_with_ew']}" if r and r["n_with_ew"] else "—")
            md.append("| " + " | ".join(row) + " |")
    for key, title in (("agn_like_whan_models", "agn_like_whan (AGN-like and EW ≤ −3 Å)"),
                       ("agn_like_sn3_models", "agn_like, S/N ≥ 3 in all four lines"),
                       ("agn_like_whan_sn3_models", "agn_like_whan, S/N ≥ 3")):
        md += ["", f"### Model R: {title}", "",
               "| ctrl | n | CG4 sat events/n | CG4 BGG events/n | CG_sat | CG_bgg | interaction (raw p) |",
               "|---|---|---|---|---|---|---|"]
        for e in diag.get(key, []):
            ev = e.get("event_counts", {})
            s, b = ev.get("CG4|sat", {}), ev.get("CG4|bgg", {})
            if e["status"] != "ok":
                md.append(f"| {LABEL[e['control']]} | skipped ({e.get('model_R', {}).get('reason')}) "
                          f"| {s.get('events')}/{s.get('n')} | {b.get('events')}/{b.get('n')} | | | |")
                continue
            md.append(
                f"| {LABEL[e['control']]} | {e['counts']['n']} | {s.get('events')}/{s.get('n')} | {b.get('events')}/{b.get('n')} "
                f"| {f(e['cg_sat']['estimate'], 2)} ± {f(e['cg_sat']['se'], 2, False)} "
                f"| {f(e['cg_bgg']['estimate'], 2)} ± {f(e['cg_bgg']['se'], 2, False)} "
                f"| {f(e['interaction']['estimate'], 2)} ({p_text(e['interaction']['p'])}) |")
    fa = diag["fibre_aperture"]
    md += ["", "## 3.2 Fibre aperture", "",
           f"Rows with log_fib_frac: {fa['n_rows_with_log_fib_frac']} of {fa['n_support_rows']} support rows.", "",
           "| sample | BGG median log(R_fib/R50) | satellite median |", "|---|---|---|"]
    med = fa["log_fib_frac_median"]
    for sample in ("CG4", "Control4B", "Control4C", "RG4"):
        md.append(f"| {sample} | {med[sample + '|bgg']['median']:+.2f} (n={med[sample + '|bgg']['n']}) "
                  f"| {med[sample + '|satellite']['median']:+.2f} (n={med[sample + '|satellite']['n']}) |")
    md += ["", "| outcome | ctrl | CG_sat without → with | shift (SE) | CG_bgg without → with | shift (SE) |",
           "|---|---|---|---|---|---|"]
    for m in fa["models"]:
        if m["status"] != "ok":
            md.append(f"| {NAME[m['outcome']]} | {LABEL[m['control']]} | skipped | | | |")
            continue
        d = 2 if m["outcome"] in LOGIT else 3
        md.append(
            f"| {NAME[m['outcome']]} | {LABEL[m['control']]} "
            f"| {f(m['cg_sat']['without']['estimate'], d)} → {f(m['cg_sat']['with_log_fib_frac']['estimate'], d)} "
            f"| {m['cg_sat']['shift_in_se']:+.2f} "
            f"| {f(m['cg_bgg']['without']['estimate'], d)} → {f(m['cg_bgg']['with_log_fib_frac']['estimate'], d)} "
            f"| {m['cg_bgg']['shift_in_se']:+.2f} |")

    rec = reconcile(block)
    with open(OUT / "preliminary_reconciliation.json", "w", encoding="utf-8") as fh:
        json.dump(safe_json(rec), fh, indent=1)
    md += ["", "## Reconciliation with the preliminary values", "",
           "| outcome | ctrl | term | prelim | this run (cr) | Δ/SE | bs(df=3) refit | bs − prelim |",
           "|---|---|---|---|---|---|---|---|"]
    for row in rec["model_I_R"]:
        for term in ("cg_sat", "cg_bgg", "interaction"):
            r = row[term]
            flag = " **>1 SE**" if abs(r["delta_in_se"]) > 1 else ""
            md.append(f"| {NAME[row['outcome']]} | {LABEL[row['control']]} | {term} | {r['prelim']:+.3f} "
                      f"| {r['cr_centred']:+.3f} | {r['delta_in_se']:+.2f}{flag} | {r['bs_df3_refit']:+.3f} "
                      f"| {r['bs_refit_minus_prelim']:+.3f} |")
    md += ["", "| outcome | ctrl | prelim (10.0–10.5, linear) | proxy refit | Model S (spline, all masses) | (S − prelim)/SE |",
           "|---|---|---|---|---|---|"]
    for r in rec["model_S_proxy"]:
        flag = " **>1 SE**" if abs(r["model_S_minus_prelim_in_se"]) > 1 else ""
        md.append(f"| {NAME[r['outcome']]} | {LABEL[r['control']]} | {r['prelim']:+.3f} ± {r['prelim_se']:.3f} "
                  f"| {r['proxy_refit']:+.3f} ± {r['proxy_refit_se']:.3f} | {r['model_S']:+.3f} ± {r['model_S_se']:.3f} "
                  f"| {r['model_S_minus_prelim_in_se']:+.2f}{flag} |")
    (OUT / "phase2_3_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1])
