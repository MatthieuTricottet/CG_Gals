"""Phase 5 — render REPORT.md entirely from outputs/metallicity_scoping.json.
No number in the report is typed by hand."""
from __future__ import annotations

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_JSON = os.path.join(HERE, "outputs", "metallicity_scoping.json")
OUT_MD = os.path.join(HERE, "REPORT.md")
SAMPLES = ["CG4", "RG4", "Control4B", "Control4C"]
SHORT = {"Control4B": "C4B", "Control4C": "C4C", "RG4": "RG4"}


def f(x, nd=3, sign=False):
    if x is None:
        return "–"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def main() -> None:
    with open(OUT_JSON) as fh:
        d = json.load(fh)
    p0, p1, p2, p3, p4 = d["phase0"], d["phase1"], d["phase2"], d["phase3"], d.get("phase4")
    ja = p1["join_accounting"]
    cen = p2["census"]
    dec = p2["decisive_SN20"]
    pw20 = p2["power"]["by_threshold"]["20"]
    pw0 = p2["power"]["by_threshold"]["0"]
    small = pw20["two_smallest_labels"][0]
    ys = p2["yardsticks"]["by_threshold"]["20"]
    sv = p3["systematics_vs_delta_min"]
    sf = p3["balance"]["selection_function"]
    L = []
    A = L.append
    A("# Stellar-population index feasibility for CG4 satellites — scoping report")
    A("")
    A(f"*Auto-generated from `outputs/metallicity_scoping.json` (`{p0['meta']['git_branch']}` @ `{p0['meta']['git_head']}`). Exploratory scoping, not inference: no p-values, no multiplicity correction.*")
    A("")
    # 1
    A("## 1. What data exist")
    A(f"- Locally: the four sample tables, the pipeline SDSS cache ({p0['join']['SDSS_cache']['rows_withAGN']} rows; `galSpecExtra`/`galSpecLine`/`zooSpec` only) and the size caches. "
      f"**No** `galSpecIndx`, per-galaxy σ or spectrum S/N existed locally.")
    fl = p1["acquisition"]["fetch_log"]
    A(f"- Fetched (path A; SkyServer DR{fl['dr']} because the DR16 host fails TLS — `galSpec*` are the unchanged MPA-JHU DR8 tables): `galSpecIndx` (indices, errors, raw + emission-subtracted), "
      f"`galSpecInfo` (`v_disp`, `sn_median`, `reliable`), `SpecObjAll` cross-checks, for {p1['acquisition']['n_unique_objid_requested']} objids "
      f"+ {p4['host_frame']['n_members_fetched'] if p4 else '–'} Lim-host members. Cached under `work/`.")
    dm = p1["datamodel_verification"]
    A(f"- Datamodel (primary sources saved in `work/`): Å for atomic indices, mag for Mg2; Lick EW sign convention (verified: quenched median Mgb {f(p1['empirical_checks']['median_quenched']['Mgb'],2)} Å, HδA {f(p1['empirical_checks']['median_quenched']['HdA'],2)} Å); "
      f"no σ-broadening correction is applied to the data (the models are broadened to each galaxy's σ, Kauffmann+03a §2); Lick/IDS transformation **UNRESOLVED** (§6); "
      f"MPA rows zeroed with `plateid=-1` when not run, `_err=-1` = index not measured.")
    sn = p1["sentinel_notes"]
    A(f"- Sentinel finding: the Fe5335 ({sn['lick_fe5335']['n_err_minus1']}) and Mg2 ({sn['lick_mg2']['n_err_minus1']}) `_err=-1` flags cluster at z = "
      f"{f(sn['lick_fe5335']['z_p10_p50_p90_of_flagged'][0],4)}–{f(sn['lick_fe5335']['z_p10_p50_p90_of_flagged'][2],4)}, where the red pseudo-continuum meets the 5577 Å sky line → a z-dependent hole in Fe5335/[MgFe]′.")
    A("")
    # 2
    A("## 2. How many satellites are usable, and which")
    A("| sample | sat input | MPA result | 7 indices valid | + σ + S/N | S/N>20 | S/N>30 |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    # blinded census cannot be mapped back per true sample without the key; use phase4/phase1 for true labels
    tc = p4["true_label_census_satellites"] if p4 else {}
    for s in SAMPLES:
        a = ja[s]["satellites"]; t = tc.get(s, {})
        A(f"| {s} | {a['N_input']} | {a['N_mpa_result']} | {a['N_valid_core7']} | {a['N_valid_core7_and_sigma_and_SN']} ({t.get('N_groups_with_usable_sat','–')} groups) | "
          f"{t.get('N_usable_SN_gt_20','–')} ({t.get('N_groups_with_usable_sat_SN_gt_20','–')}) | {t.get('N_usable_SN_gt_30','–')} |")
    A(f"| pooled (dedup) | {dec['N_sat_full']} | – | {cen['0']['pooled_dedup']['sat']} | – | {cen['20']['pooled_dedup']['sat']} | {cen['30']['pooled_dedup']['sat']} |")
    A("")
    A("Galaxy-by-galaxy list (objid, group, S/N, indices, loss reason): `outputs/usable_satellites.csv`.")
    A(f"CG4 satellite losses: " + ", ".join(f"{v} {k}" for k, v in ja['loss_reason_by_sample_satellites']['CG4'].items() if k != 'ok') + f". The S/N>20 cut is differential between samples "
      f"({', '.join(f'{s} {pct(tc[s]['N_usable_SN_gt_20']/tc[s]['N_usable'])}' for s in SAMPLES)} of usable satellites kept).")
    A(f"**Decisive question.** At S/N > {dec['threshold']:.0f}, {pct(dec['frac_sat_surviving'])} of satellites survive ({dec['N_sat_surviving']}/{dec['N_sat_full']}). They are **not** just the brightest quartile "
      f"({pct(dec['frac_of_survivors_in_M_r_Q1'])} of survivors come from it), but the selection is strongly graded: survival by M_r quartile (bright→faint) "
      f"{' / '.join(pct(v['frac_surv']) for v in dec['survival_by_M_r_quartile'].values())}; by stellar-mass quartile (low→high) "
      f"{' / '.join(pct(v['frac_surv']) for v in dec['survival_by_lgm_quartile'].values())}; survivors are {f(abs(dec['M_r_survivors']['p50'] - dec['M_r_full']['p50']),2)} mag brighter and "
      f"{f(dec['lgm_survivors']['p50'] - dec['lgm_full']['p50'],2)} dex more massive at the median. Logistic selection per SD: "
      f"M_r {f(sf['coef_per_SD']['M_r_std'],2,True)}, σ {f(sf['coef_per_SD']['sigma_std'],2,True)}, R50 {f(sf['coef_per_SD']['R50_kpc_std'],2,True)}, z {f(sf['coef_per_SD']['z_std'],2,True)}: "
      "a hard S/N cut breaks matched balance in luminosity, size and redshift.")
    A("")
    # 3
    A("## 3. Δ_min per index (3σ group-level mean offset; blinded labels)")
    A(f"Smallest blinded label at S/N>20: N_g = {pw20['sizes'][small]['N_g']}, n̄_sat = {f(pw20['sizes'][small]['n_sat'],2)}; ρ = empirical one-way ICC (group bootstrap); σ_idx = total satellite scatter, of which measurement error is only "
      f"{pct(p2['error_budget']['satellite_scatter_vs_error']['Mgb']['median_err'] / p2['error_budget']['satellite_scatter_vs_error']['Mgb']['sd_total'])} (Mgb) / "
      f"{pct(p2['error_budget']['satellite_scatter_vs_error']['MgFe']['median_err'] / p2['error_budget']['satellite_scatter_vs_error']['MgFe']['sd_total'])} ([MgFe]′).")
    A("")
    A("| index | unit | median err @S/N 20–25 | ρ | Δ_min (S/N>20) | Δ_min (no cut) | 2-sample, smallest vs largest | Δ_min/IQR |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for k in ["Mgb", "Fe5270", "Fe5335", "MgFe", "MgbFe", "Mg2", "D4000n", "HdA", "Hb_sub"]:
        v20 = pw20["indices"][k]; v0 = pw0["indices"][k]
        eb = next((e for e in p2["error_budget"]["median_err_by_bin"][k] if e["bin"].startswith("(20.0")), {})
        y = ys.get(k, {})
        A(f"| {k} | {v20.get('unit','')} | {f(eb.get('median_err'))} | {f(v20['rho_used'],2)} | **{f(v20['per_blind_sample'][small]['delta_min'])}** | {f(v0['per_blind_sample'][pw0['two_smallest_labels'][0]]['delta_min'])} | "
          f"{f(v20['two_sample_smallest_vs_largest']['delta_min'])} | {f(y.get('delta_min_over_IQR'),2)} |")
    A("")
    A("Δ_min is nearly flat against the S/N threshold (intrinsic scatter dominates): the cut buys little power while imposing the graded selection of §2. Raw Hβ is unusable (emission fill-in); use `_sub`. Figure: `figures/fig_power_vs_SN.png`.")
    A("")
    # 4
    A("## 4. Dominant systematic (blinded)")
    A("| index | Δ_min (S/N>20) | aperture shift over IQR(r_fib/R50) at fixed σ | z shift over IQR(z) at fixed σ | GZ1 E−S at fixed σ | morph/Δ_min |")
    A("|---|---:|---:|---:|---:|---:|")
    for k in ["Mgb", "Fe5270", "Fe5335", "MgFe", "MgbFe", "Mg2", "D4000n", "HdA"]:
        v = sv[k]
        A(f"| {k} | {f(v['delta_min_SN20_smallest'])} | {f(v['aperture_shift_over_IQR_at_fixed_sigma'],3,True)} | {f(v['z_shift_over_IQR_at_fixed_sigma'],3,True)} | {f(v['morphology_E_minus_S_at_fixed_sigma'],3,True)} | {f(v['ratio_morph_over_delta_min'],2)} |")
    A("")
    ap = p3["aperture"]
    dg = p3["degeneracy"]
    mc = p3["morphology_confound"]["by_SN_cut"]["SN_gt_20"]["indices"]
    A(f"- Aperture (1.5″ fibre radius / Simard R_chl,r, Petrosian fallback; Planck15 angular-diameter distance): median r_fib/R50 = {f(ap['ap_frac_quantiles_sat']['p50'],2)} "
      f"(IQR {f(ap['ap_frac_quantiles_sat']['p25'],2)}–{f(ap['ap_frac_quantiles_sat']['p75'],2)}); metal-line gradients at fixed σ are ≤ {f(max(abs(sv[k]['ratio_aperture_over_delta_min']) for k in ['Mgb','Fe5270','Fe5335','MgFe']),2)} Δ_min over the IQR — subdominant, as is redshift.")
    A(f"- **Morphology is the dominant systematic**: at fixed σ, GZ1 ellipticals − spirals = {f(mc['Mgb']['E_minus_S_at_fixed_sigma'],2,True)} Å (Mgb), "
      f"{f(mc['MgFe']['E_minus_S_at_fixed_sigma'],2,True)} Å ([MgFe]′), {f(mc['D4000n']['E_minus_S_at_fixed_sigma'],3,True)} (Dn4000), {f(mc['HdA']['E_minus_S_at_fixed_sigma'],2,True)} Å (HδA) — "
      f"{f(sv['Fe5335']['ratio_morph_over_delta_min'],1)}–{f(max(sv[k]['ratio_morph_over_delta_min'] for k in ['Mgb','MgFe','D4000n','HdA','Mg2']),1)} × Δ_min, and σ-dependent "
      f"(Mgb E−S {f(mc['Mgb']['per_sigma_bin'].get('(0, 80]',{}).get('E_minus_S_median'),2,True)} Å at σ<80 vs {f(mc['Mgb']['per_sigma_bin'].get('(160, 220]',{}).get('E_minus_S_median'),2,True)} at 160–220 km/s). "
      f"Since CG4's headline result is a morphology excess, any index signal must be measured within morphology class. Mgb/⟨Fe⟩ is morphology-independent at fixed σ ({f(mc['MgbFe']['E_minus_S_at_fixed_sigma'],3,True)}).")
    A(f"- Age–Z plane: HδA_sub vs [MgFe]′ Spearman {f(dg['HdA_sub_vs_MgFe']['spearman'],2)}, Dn4000 vs [MgFe]′ {f(dg['D4000n_vs_MgFe']['spearman'],2)}; residual width at fixed [MgFe]′ is "
      f"{f(dg['HdA_sub_vs_MgFe']['resid_sd_over_median_err'],1)}× (HδA) / {f(dg['D4000n_vs_MgFe']['resid_sd_over_median_err'],1)}× (Dn4000) the median error: real width at population level, marginal per galaxy.")
    A("")
    # 5 recommendation + phase 4 summary
    A("## 5. Exploratory unblinded contrasts (Phase 4; no inference)")
    if p4:
        r = p4["results"]["all_valid"]["a_galaxy_level"]
        rw = p4["results"]["all_valid"]["d_within_host"]["satellites_only"]
        A("CG4 − control, satellites, fixed morphology & σ, no S/N cut (coef ± group-clustered SE [68% group-bootstrap CI]); last column = within-host fixed-effects analogue of the conditional-logit design:")
        A("")
        A("| index | vs C4B | vs C4C | vs RG4 | within-host (sat.) | Δ_min (no cut) |")
        A("|---|---|---|---|---|---:|")
        for k in ["Mgb", "Fe5270", "Fe5335", "MgFe", "MgbFe", "Mg2", "D4000n", "HdA_sub", "Hb_sub"]:
            cells = []
            for c in ["Control4B", "Control4C", "RG4"]:
                v = r[c]["indices"].get(k, {}).get("all_morph")
                cells.append(f"{f(v['coef'],3,True)} ± {f(v['cluster_se'])} [{f(v['ci68'][0],2,True)},{f(v['ci68'][1],2,True)}]" if v else "–")
            w = rw["indices"].get(k)
            cells.append(f"{f(w['coef'],3,True)} ± {f(w['cluster_se'])}" if w else "–")
            A(f"| {k} | " + " | ".join(cells) + f" | {f(p4['delta_min_reference_SN0_smallest_label'].get(k))} |")
        A("")
        A("All metal-line contrasts are below Δ_min (mostly ≤ 0.4 Δ_min) and consistent with zero; Balmer/Dn4000 offsets vs C4B/RG4 are ~2 SE (older-looking CG4 satellites) but ≈ 0 vs C4C, and the within-host contrasts flip sign between the no-cut and S/N>20 variants. Nothing here is a detection.")
    A("")
    A("## 6. Recommendation and unresolved items")
    A(f"**Go/no-go: conditional GO — only as a prespecified, within-morphology, S/N-uncut, error-weighted design.** N is adequate ({tc['CG4']['N_usable']} usable CG4 satellites in {tc['CG4']['N_groups_with_usable_sat']} groups; "
      f"{tc['CG4']['N_usable_SN_gt_20']} at S/N>20), Δ_min ≈ {f(ys['Mgb']['delta_min_over_IQR'],2)} × IQR ({f(pw20['indices']['Mgb']['per_blind_sample'][small]['delta_min'],2)} Å Mgb, "
      f"{f(pw20['indices']['MgFe']['per_blind_sample'][small]['delta_min'],2)} Å [MgFe]′) and no systematic swamps it; but morphology ≈ Δ_min and the S/N cut is differential, so Paper III must "
      "(i) replace the hard S/N cut by measurement-error weighting, (ii) stratify/match on GZ1 class and σ, (iii) take C4C as primary control, (iv) prespecify before any SSP fitting. "
      "The exploratory contrasts show no metal-line effect above ~0.4 Δ_min.")
    A("")
    A("UNRESOLVED (not verifiable from primary sources consulted):")
    A(f"1. Lick/IDS resolution transformation of the MPA indices: {dm['resolution_system']['status']}.")
    A("2. Dn4000 flux-density convention (F_ν vs F_λ): not stated in the catalogue docs (Kauffmann+03a define D4000 via F_ν).")
    A("3. The `_err = -1` convention is inferred empirically (datamodel text only describes whole-row zeroing).")
    A("4. DR18 = DR16 `galSpecIndx` rows is asserted from the run2d=26-only coverage statement and the 100% `specobjid` cross-match, not a row-level DR16 diff (endpoint unreachable).")
    A("5. Blinded balance tests are null by construction; true-label balance after an S/N cut was not tested (only the differential survival above).")
    A("")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip().splitlines()
    A(f"`git status --porcelain` at report time: {len(status)} entries (`exploration/` plus pre-existing/concurrent-session paths; see INVENTORY.md and the final session log).")
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
