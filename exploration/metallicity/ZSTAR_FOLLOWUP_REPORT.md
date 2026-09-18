# Direct stellar metallicity — follow-up audit and physical interpretation

*Continuation of [ZSTAR_REPORT.md](ZSTAR_REPORT.md) (Codex, 2026-09-15) and [REPORT.md](REPORT.md) (Claude, 2026-09-12). Neither historical product is modified. Everything new lives in `outputs/zstar_followup/`, `figures/zstar_followup/`, `zstar_followup.py` and this file. Seed 20260917, 2 000 group-bootstrap replicates (groups resampled within sample, as before). Reproduce with `python zstar_followup.py` (~7 min, no network). All numbers below are FIREFLY MILES light-weighted [Z/H] at fixed stellar mass unless stated otherwise; brackets are 95 % group-bootstrap intervals; N = N(CG4)+N(control).*

Labelling convention used throughout: **[planned]** = requested in the brief; **[robustness]** = alternative specification of a planned test; **[post-hoc]** = exploration suggested by the data during this session.

---

## 0. Ten answers in one page

| # | Question | Answer |
|---|---|---|
| 1 | Does the CG4–Control4C null survive? | **Yes, as a net offset.** Satellites +0.003 [−0.019, +0.023]; control-only-fit residuals +0.003 [−0.018, +0.023]; median regression −0.003 [−0.020, +0.027]; every covariate adjustment −0.008…+0.009; mass bins −0.004…+0.016; four other products +0.003…+0.023. BGGs −0.015 [−0.044, +0.014], equally stable. **But the null is a cancellation**: CG4 ellipticals −0.021 [−0.051, +0.008] vs CG4 spirals +0.018 [−0.007, +0.044]; interaction CG4×Spiral = +0.040 [+0.003, +0.077]. |
| 2 | Mass interaction? | **None measurable.** Satellites vs C4C γ = +0.016 ± 0.028 dex/dex; vs RG4 γ = −0.036 ± 0.036; BGGs γ = +0.11 ± 0.08 over a 0.6-dex range (implied effect −0.05…+0.02). Three broad mass bins are flat. Gallazzi alone gives γ = +0.21 ± 0.10 for satellites (N=63) — not seen by FIREFLY on the same objects. |
| 3 | Why do Gallazzi and FIREFLY BGGs differ (−0.067 vs −0.015)? | **Half subsample, half pipeline, neither significant.** On the same 27+279 BGGs: FIREFLY −0.035 [−0.084, +0.013], Gallazzi −0.067 [−0.154, +0.002], difference −0.032 [−0.089, +0.019]. FIREFLY on the 30 CG4 BGGs Gallazzi does not have: +0.000 [−0.032, +0.030]. The Gallazzi subsample simply contains the more negative half. |
| 4 | Is the CG4–RG4 satellite contrast (+0.033) robust? | **Statistically yes, physically it is not "CG4 vs ordinary groups".** +0.033 [+0.004, +0.065]; median regression +0.028 [+0.007, +0.062]; S/N>15 +0.030; z ≤ 0.04 +0.023; uncertainty ≤ 0.1 dex +0.032; adjusting z or S/N changes nothing. Seen in 4/5 products (ELODIE lw +0.017 [−0.007, +0.040]). **RG4 ⊂ Control4C** (same 56 groups, 224 galaxies): it is the poorest, loosest, lowest-σ end of the control population, not an independent "ordinary group" sample. |
| 5 | Who carries it? | **Star-forming/spiral satellites; quenched/elliptical ones do not.** vs RG4: quenched −0.000 [−0.041, +0.035], star-forming +0.042 [−0.005, +0.086]; ellipticals −0.035 [−0.078, +0.007], spirals +0.053 [+0.017, +0.091]; CG4×Spiral +0.088 [+0.029, +0.146]. Kitagawa split: composition (CG4 has 2× the elliptical fraction of RG4) +0.024 [+0.009, +0.042], within-class +0.007 [−0.026, +0.036] (morphology strata) — or +0.014/+0.015 with SF strata. |
| 6 | Is Control4C closer to CG4 than RG4 is? | **Yes, and the whole control population forms one sequence.** CG4 minus: RG4 +0.033; C4C loose cores +0.011; C4C −0.001…+0.003; C4C compact cores (≤218 kpc) −0.016 [−0.043, +0.010]; C4C cores of N ≥ 25 parents −0.027 [−0.056, +0.000]. Within Control4C, satellite [Z/H] at fixed M★ rises with parent richness (+0.052 ± 0.010 dex/dex) and falls with core size (−0.054 ± 0.015 dex/dex). CG4 (median core 153 kpc, σ 160 km s⁻¹) sits at −0.013 [−0.038, +0.011] from the Control4C environmental model — i.e. exactly where an ordinary compact core of that size would sit. |
| 7 | Star-forming CG4 BGGs (−0.082, N=11)? | **Not an artefact, still not a result.** All 11 are rank-1 by mass; 9 spirals, 2 unclassified; no object dominates (LOO −0.065…−0.097; dropping the two lowest −0.049 [−0.103, +0.008]); same sign in all 5 products (ELODIE lw −0.047, MILES mw −0.122, ELODIE mw −0.067, Gallazzi −0.265 on 6 objects); survives D4000n/sSFR adjustment (−0.067 ± 0.028). It remains a post-hoc, N = 11, ≈2.5σ lead. Their ages are not younger (+0.03 ± 0.05). |
| 8 | Is age more informative than Z★? | **Slightly, and more coherent.** CG4 satellites are older at fixed mass by +0.035 [−0.000, +0.069] dex (MILES lw), +0.036 [+0.007, +0.064] (MILES mw), +0.032 [−0.003, +0.064] (ELODIE lw) vs Control4C — *inside the quenched class*: +0.036, +0.028, +0.040 [+0.016, +0.066]. vs RG4 quenched: +0.085 [+0.023, +0.147]. Gallazzi ages (S/N ≥ 20 subsample, N=63) show nothing (−0.001 ± 0.031), FIREFLY on the same 63 objects +0.028 ± 0.029 — not discriminating. A ~8–20 % older light-weighted age of quenched CG4 satellites is the most physically interesting number produced so far, but it is one pipeline and ≈2σ. |
| 9 | Realistic path to [α/Fe]? | **One catalogue, on request; otherwise Paper III work.** Gallazzi et al. (2021, MNRAS 502, 4457): SDSS DR7 MGS, S/N ≥ 20, 113 307 galaxies, semi-empirical [α/Fe] from Δ(Mgb/⟨Fe⟩) calibrated on TMK04; "available upon reasonable request to the corresponding author". Expected coverage here: ≈53 CG4 BGGs, ≈116 CG4 satellites (85 quenched, 75 E), 636+1262 Control4C, 52+75 RG4. All other public [α/Fe] catalogues (MOSES/Thomas+2010, Johansson+2012, SPIDER/La Barbera+2014) are at z ≥ 0.05, above our z ≤ 0.045 sample. FIREFLY has no [α/Fe]. |
| 10 | My reading | Satellite Z★ at fixed mass in the CG_Gals volume is a **smooth function of local group density** (richness, core compactness), mostly through the quenched/spiral mix and partly within spirals. CG4 satellites follow that relation; they are ordinary compact-core satellites in Z★, older by a few hundred Myr–1 Gyr if FIREFLY ages are right, and possibly slightly gas-metal-rich when star-forming (post-hoc O3N2 hint, +0.02–0.04 dex). The morphology excess of Paper II is therefore **not** accompanied by a chemical anomaly beyond what compactness already predicts. |

---

## 1. Audit of the historical analysis [planned]

`audit_reproduction.csv`, `audit_censoring.csv`, `key_numbers.json`.

**Transformations.** Gallazzi DR4 stores log₁₀ Z (absolute); the catalogue page instructs to subtract log₁₀(0.02); the maximum after conversion is +0.394, i.e. the BC03 grid edge Z = 0.05. Correct. FIREFLY `*_metallicity_*` are linear Z/Z☉: the values populate [0.008, 1.9967] with a hard ceiling at 1.9967 ≈ 2 (the M11-MILES Z = 0.04 template) and 245 distinct values in 3 728 spectra — they are light-weighted means of linear Z over ≤ 9 SSP components drawn from {0.005, 0.05, 0.5, 1, 2} Z☉. log₁₀ of that is what the previous analysis used; it is a legitimate transformation but the quantity is log(⟨Z⟩_L), not ⟨log Z⟩_L (Jensen: biased high relative to Gallazzi's median log Z; irrelevant for within-catalogue contrasts, relevant for the −0.037 dex Gallazzi−FIREFLY offset). FIREFLY ages are in **years** (grid 0.03–15 Gyr).

**A ceiling the previous report under-stated.** 13.9 % of Control4C BGGs and 10.0 % of CG4 BGGs sit exactly at the FIREFLY grid ceiling [Z/H] = +0.30 (183 of 1 467 BGG rows at 0.300, 141 at 0.211, 107 at 0.243: the BGG distribution is quantised). Any BGG contrast is a comparison of censored, discretised values; a −0.015 dex mean offset is compressed by the ceiling. Satellites are unaffected (0–1.2 % at the ceiling). Gallazzi has no ceiling pile-up.

**Counts.** Usable → fitted: CG4 satellites 180 → 176 (4 without MPA stellar mass), CG4 BGGs 60 → 57 (3), Control4C satellites 2042 → 1978 (64), BGGs 678 → 648 (30). Correct and now tabulated.

**Repeated objects.** Within each sample no objid is duplicated; CG4 shares no object with any control. Between samples, however, **RG4 is a strict subset of both Control4B and Control4C** (all 224 galaxies, same 56 groups, same BGG/satellite roles) and Control4B∩Control4C = 1 995 objects. The previous report treated "vs RG4" and "vs Control4C" as independent checks; they are not (RG4 is 8 % of Control4C satellites). No pseudo-replication occurs inside any single fit, because each comparison uses one control at a time.

**Bootstrap.** Cluster-pairs bootstrap within sample, degree fixed from the full-data CV, percentile intervals. With a new seed the 95 % half-widths change by < 5 %; the ratio half-width / (1.96 × CR1 SE) is 0.92–1.04 for all 12 primary contrasts, so the cluster-robust SE and the bootstrap agree. BGG "clusters" are single galaxies (one BGG per group), so the BGG bootstrap is an ordinary stratified bootstrap.

**Linear vs quadratic.** The CV rule (quadratic only if ≥ 2 % RMSE gain) picked degree 1 everywhere for the primary product. Control-only fits with degree 2 change no satellite result by more than 0.003 dex. The one place the rule mattered — Gallazzi satellites vs RG4, degree 2 on 43 control galaxies — should be read with the linear value (+0.031 [−0.057, +0.120] instead of +0.025).

**Mass control.** The model has a common M★ slope fitted jointly; the control-only version (§4) gives identical answers, so the contrast is a mass-controlled one and does not depend on which sample sets the slope.

**Reproduction.** All 12 historical point estimates reproduce to 10⁻¹⁶; N's match.

**Two things I would have done differently.** (i) The report's "model-systematic" ELODIE/mass-weighted rows are quoted as reassuring; for the RG4 contrast they range from +0.017 to +0.076, i.e. the *size* of the effect is model-dependent by a factor 4 even if the sign is not. (ii) The morphology/SF decomposition was framed as "where the signal sits"; it should have been read as a sign-changing interaction (see §5), which the "total ≈ 0" statement hides.

## 2. Gallazzi vs FIREFLY on exactly the same galaxies [planned]

`same_objects.csv`, `figures/zstar_followup/same_objects.png`.

| role | control | N (same objects) | FIREFLY, same objects | Gallazzi, same objects | Gallazzi − FIREFLY | FIREFLY, full | FIREFLY, objects *not* in Gallazzi |
|---|---|---|---|---|---|---|---|
| BGG | Control4C | 27+279 | −0.035 [−0.084, +0.013] | −0.067 [−0.154, +0.002] | −0.032 [−0.089, +0.019] | −0.015 (57+648) | +0.000 [−0.032, +0.030] (30+369) |
| BGG | RG4 | 27+30 | −0.002 [−0.066, +0.069] | −0.059 [−0.153, +0.028] | −0.057 [−0.130, +0.009] | +0.016 (57+53) | +0.034 (30+23) |
| Satellite | Control4C | 63+545 | −0.025 [−0.063, +0.010] | +0.014 [−0.051, +0.072] | +0.039 [−0.023, +0.097] | +0.003 (176+1978) | +0.016 [−0.011, +0.043] (113+1433) |
| Satellite | RG4 | 63+43 | −0.015 [−0.056, +0.029] | +0.032 [−0.050, +0.123] | +0.048 [−0.042, +0.139] | +0.033 (176+166) | **+0.054 [+0.015, +0.094]** (113+123) |

Reading: for BGGs, moving from the full FIREFLY sample to the Gallazzi subsample shifts FIREFLY from −0.015 to −0.035 (selection: the 30 CG4 BGGs absent from DR4 are the ones with no offset); the remaining −0.032 is pipeline, with an interval that comfortably includes zero. **Neither half is significant; the −0.067 is not a discrepancy to be explained.** For satellites the more interesting fact is the last column: the CG4–RG4 contrast is carried by the galaxies that fail Gallazzi's S/N ≥ 20 cut — fainter, more star-forming — which is why Gallazzi cannot see it (§5).

## 3. Mass interaction [planned]

`mass_interaction.csv`, `mass_bins.csv`, `figures/zstar_followup/delta_z_vs_mass.png`. Model [Z/H] = f(M★) + β I_CG4 + γ I_CG4 (log M★ − M₀), M₀ = 10.2 (satellites; CG4 median) or 11.1 (BGGs).

| product | role | control | common 5–95 % mass range | β at M₀ | γ (dex/dex) | implied effect at range ends |
|---|---|---|---|---|---|---|
| FIREFLY | Sat | C4C | 9.49–10.83 | +0.003 [−0.018, +0.023] | +0.016 ± 0.028 [−0.036, +0.073] | −0.009 → +0.013 |
| FIREFLY | Sat | RG4 | 9.49–10.83 | +0.031 [+0.001, +0.057] | −0.036 ± 0.036 [−0.106, +0.037] | +0.057 → +0.008 |
| FIREFLY | Sat | C4B | 9.53–10.83 | +0.011 [−0.012, +0.032] | +0.013 ± 0.028 | +0.003 → +0.019 |
| FIREFLY | BGG | C4C | 10.82–11.40 | −0.017 [−0.048, +0.013] | +0.110 ± 0.080 [−0.072, +0.255] | −0.048 → +0.016 |
| FIREFLY | BGG | RG4 | 10.82–11.21 | +0.024 [−0.033, +0.065] | +0.162 ± 0.091 | −0.022 → +0.041 |
| Gallazzi | Sat | C4C | 9.80–10.84 | −0.018 [−0.099, +0.058] | +0.210 ± 0.098 [+0.028, +0.422] | −0.102 → +0.116 |
| Gallazzi | BGG | C4C | 10.85–11.37 | −0.064 [−0.152, +0.005] | +0.214 ± 0.103 | −0.118 → −0.006 |

Broad bins (satellites vs C4C): < 10.0: −0.001 [−0.045, +0.040] (52+679); 10.0–10.5: −0.004 [−0.037, +0.029] (78+760); > 10.5: +0.016 [−0.015, +0.045] (46+539). vs RG4: +0.043, +0.020, +0.034. BGGs: −0.012, −0.019, −0.013. **No structure with mass in FIREFLY.** The positive Gallazzi interaction (2σ, N=63) is the kind of coefficient a 27-object BGG sample and a S/N-selected satellite sample can produce; FIREFLY on the same objects does not show it. A mass-dependent effect hidden by a near-zero mean is not supported.

## 4. Control-only MZR → CG4 residuals [planned]

`control_only_residuals.csv`, `control_only_residuals_objects.csv`, `figures/zstar_followup/control_only_residuals.png`. Bootstrap resamples control groups (refitting the MZR each time) and CG4 groups.

| role | control | slope of control MZR | mean ΔZ★(CG4) | median ΔZ★(CG4) − median ΔZ★(control) | IQR CG4 / control | fraction of CG4 below control median |
|---|---|---|---|---|---|---|
| Sat | C4C | 0.227 | +0.003 [−0.018, +0.023] | +0.016 − 0.019 = −0.003 | 0.159 / 0.164 | 0.51 |
| Sat | RG4 | 0.279 | +0.031 [+0.001, +0.060] | +0.047 − 0.017 = +0.030 | 0.152 / 0.185 | 0.41 |
| BGG | C4C | 0.118 | −0.015 [−0.044, +0.013] | +0.009 − 0.014 = −0.005 | 0.127 / 0.127 | 0.54 |
| BGG | RG4 | 0.066 | +0.027 [−0.035, +0.070] | +0.053 − 0.003 | 0.129 / 0.134 | 0.28 |
| Sat, Gallazzi | C4C | 0.371 | +0.013 [−0.053, +0.076] | +0.054 − 0.052 | 0.253 / 0.199 | 0.49 |
| BGG, Gallazzi | C4C | 0.098 | −0.067 [−0.146, +0.002] | −0.037 − 0.009 = −0.046 | 0.136 / 0.122 | 0.59 |

The joint-model and control-only estimates agree to ≤ 0.004 dex; medians tell the same story. The residual histograms show one qualitative difference: Control4C satellites have a low-Z★ tail (ΔZ < −0.3) that CG4 lacks (SD 0.148 vs 0.126 dex, IQR nearly equal); median regression and 2 %-trimming (§11) confirm this tail does not drive anything.

## 5. Anatomy of the CG4–RG4 satellite contrast [planned + post-hoc]

`rg4_subsets.csv`, `rg4_joint_models.csv`, `rg4_composition_split.csv`, `figures/zstar_followup/rg4_decomposition.png`, `rg4_interactions_by_product.png`.

### 5.1 Subsets (FIREFLY MILES lw, satellites)

| subset | N CG4 / RG4 / C4C | CG4 − RG4 | CG4 − C4C |
|---|---|---|---|
| all | 176 / 166 / 1978 | +0.033 [+0.004, +0.064] | +0.003 [−0.018, +0.024] |
| quenched | 101 / 63 / 1013 | −0.000 [−0.041, +0.035] | −0.014 [−0.042, +0.011] |
| star-forming | 72 / 103 / 948 | +0.042 [−0.005, +0.086] | +0.007 [−0.026, +0.042] |
| elliptical | 88 / 39 / 704 | −0.035 [−0.078, +0.007] | −0.021 [−0.051, +0.008] |
| spiral | 68 / 103 / 1028 | **+0.053 [+0.017, +0.091]** | +0.018 [−0.007, +0.044] |
| uncertain | 6 / 13 / 125 | −0.044 [−0.311, +0.187] | −0.053 [−0.247, +0.131] |
| quenched & E | 74 / 31 / 589 | −0.028 [−0.078, +0.022] | −0.031 [−0.062, −0.000] |
| quenched & S | 20 / 24 / 287 | +0.037 [−0.026, +0.097] | +0.015 [−0.033, +0.059] |
| star-forming & E | 11 / 8 / 109 | −0.104 [−0.191, −0.023] | −0.014 [−0.086, +0.066] |
| star-forming & S | 48 / 79 / 739 | **+0.056 [+0.009, +0.103]** | +0.017 [−0.014, +0.049] |

The earlier un-bootstrapped hints are confirmed: the contrast is zero for quenched galaxies and lives in star-forming spirals; and — the part the earlier check did not say — **it changes sign between ellipticals and spirals**, against both controls.

### 5.2 Joint models (satellites)

| model | vs RG4 | vs C4C |
|---|---|---|
| Z ~ f(M) + CG4 + SF | CG4 +0.022 [−0.008, +0.052]; SF −0.076 | CG4 −0.005 [−0.028, +0.015]; SF −0.093 |
| … + CG4×SF | CG4(quenched) −0.000 [−0.040, +0.036]; CG4×SF +0.042 [−0.019, +0.104] | CG4(quenched) −0.014 [−0.043, +0.012]; CG4×SF +0.021 [−0.023, +0.069] |
| Z ~ f(M) + CG4 + Spiral | CG4 +0.018 [−0.011, +0.048]; Spiral −0.057 | CG4 −0.003 [−0.026, +0.016]; Spiral −0.060 |
| … + CG4×Spiral | CG4(E) −0.035 [−0.077, +0.008]; **CG4×Spiral +0.088 [+0.029, +0.146]** | CG4(E) −0.021 [−0.051, +0.006]; **CG4×Spiral +0.040 [+0.003, +0.077]** |

Across products the CG4×Spiral interaction is +0.088 (MILES lw), +0.112 [+0.011, +0.223] (MILES mw), +0.068 [−0.013, +0.143] (ELODIE mw), +0.082 [−0.068, +0.260] (Gallazzi), but −0.012 [−0.068, +0.042] (ELODIE lw) vs RG4; vs C4C +0.040, +0.053, +0.085 [+0.034, +0.134], +0.039, +0.019. Four of five estimators agree on the sign; ELODIE light-weighted does not see it. This is the robustness status of the whole spiral result: real in most fits, not universal, and never larger than ≈0.1 dex.

### 5.3 Composition vs within-class (Kitagawa/Oaxaca split of the mass-adjusted residual difference)

| control | strata | total | within-class (CG4 weights) | composition |
|---|---|---|---|---|
| RG4 | SF state | +0.029 [−0.002, +0.059] | +0.015 [−0.016, +0.045] | +0.014 [+0.003, +0.029] |
| RG4 | morphology | +0.031 [−0.001, +0.062] | +0.007 [−0.026, +0.036] | **+0.024 [+0.009, +0.042]** |
| C4C | SF state | +0.001 | −0.004 | +0.005 |
| C4C | morphology | +0.003 | −0.006 | +0.009 [+0.004, +0.014] |

CG4 satellites are 58 % quenched / 50 % elliptical; RG4 38 % / 24 %. The only component whose interval excludes zero is the composition term. **Total effect: +0.03 dex; conditional effect at fixed class: ≈ +0.01 and compatible with zero; but with opposite signs for E and S.** Morphology and SF state are treated here as descriptors, not confounders — they may well be the environmental outcome (§9).

### 5.4 Within-class balance (are CG4 spirals different spirals?) [post-hoc]

At fixed mass, CG4 spirals vs Control4C spirals: p_E +0.03 ± 0.02, log σ −0.01 ± 0.02, D4000n +0.02 ± 0.02, sSFR −0.09 ± 0.08, R50 +0.09 ± 0.24 kpc, fibre fraction +0.01 ± 0.01, S/N −0.9 ± 1.2, z +0.002 ± 0.001. All |SMD| < 0.25. Within control spirals, Z★ rises with p_E (+0.175 ± 0.029 per unit): the CG4 p_E shift buys +0.005 dex of the +0.018. So the spiral offset is not a bulge-fraction, aperture, S/N or redshift artefact of the classification.

## 6. Population composition at fixed mass [planned]

`composition_balance.csv`, `adjusted_contrasts.csv`. Satellites, mass-adjusted CG4 − control (SMD in parentheses):

| covariate | CG4 − RG4 | CG4 − C4C |
|---|---|---|
| sSFR (dex) | −0.21 ± 0.08 (−0.22) | −0.11 ± 0.06 (−0.12) |
| D4000n | +0.056 ± 0.024 (+0.20) | +0.025 ± 0.017 (+0.09) |
| HδA (Å) | −0.82 ± 0.26 (−0.28) | −0.36 ± 0.18 (−0.13) |
| log σ★ | +0.013 ± 0.019 (+0.05) | −0.003 ± 0.013 (−0.01) |
| log age (MILES lw) | +0.045 ± 0.026 (+0.20) | +0.035 ± 0.018 (+0.14) |
| fibre light fraction | +0.013 ± 0.010 (+0.13) | +0.013 ± 0.007 (+0.13) |
| z | −0.003 ± 0.001 (−0.48) | +0.001 ± 0.001 (+0.19) |
| log S/N | +0.067 ± 0.020 (+0.30) | −0.001 ± 0.014 (0.00) |
| [Z/H] uncertainty (dex) | −0.019 ± 0.006 (−0.27) | −0.002 ± 0.004 (−0.03) |
| p_E | +0.154 ± 0.031 (+0.47) | +0.112 ± 0.024 (+0.33) |
| quenched fraction | +0.11 ± 0.05 | +0.06 ± 0.03 |
| elliptical fraction | +0.22 ± 0.05 | +0.14 ± 0.04 |

Against Control4C, CG4 satellites differ only in morphology/SF mix (the Paper II result) and, mildly, in FIREFLY age; every nuisance variable (z, S/N, σ★, aperture, size) is balanced. Against RG4 they additionally differ in redshift (RG4 median 0.040 vs 0.037), S/N (18.9 vs 25.8) and hence [Z/H] precision — but adjusting for z and/or log S/N leaves the RG4 contrast at +0.029…+0.034, and within Control4C the S/N dependence of Z★ at fixed mass (+0.13 ± 0.03 dex per dex of S/N) is a physical surface-brightness correlation, not a bias that could produce the RG4 offset.

Adjusted contrasts (satellites): vs RG4 +0.033 → z +0.029, sSFR +0.016 [−0.013, +0.044], D4000n +0.021, σ★ +0.026, age +0.043 [+0.015, +0.074], sSFR+D4000n+σ★ +0.011 [−0.017, +0.039]. vs C4C: +0.003 → −0.008…+0.009 for every adjustment. Note the age adjustment *increases* the RG4 contrast: the older CG4 populations would, through the FIREFLY age–metallicity degeneracy (residual correlation r = −0.28, slope −0.17 dex/dex within Control4C), *lower* their light-weighted Z★ by ≈0.01 dex.

## 7. Does compact-core selection explain the RG4 difference? [planned]

`environment_sequence.csv`, `within_control_environment_slopes.csv`, `environment_model_residuals.csv`, `figures/zstar_followup/environment_sequence.png`. Group properties are the quartet's own (barycentric size, gapper σ of the four members — identical definitions for CG4 and controls); "parent N" is the Lim et al. parent richness, available for control groups only.

**Satellites, CG4 minus comparison set** (medians of the control set: core size, σ_quartet, parent N, quenched fraction):

| comparison set | N | size / σ / N / f_Q | at fixed M★ | at fixed M★ + SF state |
|---|---|---|---|---|
| RG4 (= C4C parents with N = 4) | 166 (56 gr) | 459 kpc / 98 / 4 / 0.38 | **+0.033 [+0.004, +0.065]** | +0.022 [−0.008, +0.051] |
| C4C parent N = 5–7 | 652 (229) | 361 / 124 / 6 / 0.41 | +0.014 [−0.010, +0.039] | +0.000 |
| Control4B | 1976 (695) | 479 / 139 / 9 / 0.53 | +0.011 [−0.011, +0.032] | −0.005 |
| Control4B \ RG4 | 1810 (639) | 482 / 144 / 9 / 0.55 | +0.007 | −0.010 |
| C4C loose cores (size > 218 kpc) | 1405 (491) | 401 / 136 / 8 / 0.46 | +0.011 [−0.011, +0.032] | −0.000 |
| Control4C | 1978 (699) | 315 / 150 / 9 / 0.51 | +0.003 [−0.019, +0.023] | −0.005 |
| Control4C \ RG4 | 1812 (643) | 304 / 155 / 9 / 0.52 | −0.001 [−0.023, +0.020] | −0.008 |
| C4C parent N = 8–12 | 566 (199) | 296 / 156 / 10 / 0.50 | −0.005 [−0.028, +0.016] | −0.013 |
| C4C parent N = 13–24 | 402 (144) | 273 / 191 / 17 / 0.62 | −0.004 [−0.027, +0.022] | −0.004 |
| C4C compact cores (size ≤ 218 kpc = CG4 p90) | 573 (208) | 153 / 195 / 13 / 0.64 | **−0.016 [−0.043, +0.010]** | −0.015 |
| C4C parent N ≥ 25 | 192 (71) | 173 / 339 / 40 / 0.80 | **−0.027 [−0.056, +0.000]** | −0.023 |

CG4 groups: median core 153 kpc, σ_quartet 160 km s⁻¹, 58 % quenched satellites. The sequence is monotonic in local density and CG4 sits in the middle of it — level with cores of N ≈ 8–24 parents and slightly *below* ordinary cores of their own size.

**Within Control4C satellites, at fixed M★** (dex per dex; CR1 SE):

| environment | all | quenched | star-forming | elliptical | spiral |
|---|---|---|---|---|---|
| log parent N | +0.052 ± 0.010 | +0.013 ± 0.010 | +0.061 ± 0.022 | +0.019 ± 0.015 | +0.046 ± 0.016 |
| log core size | −0.054 ± 0.015 | −0.038 ± 0.013 | −0.012 ± 0.031 | −0.010 ± 0.018 | −0.060 ± 0.018 |
| log σ_quartet | +0.028 ± 0.013 | +0.002 ± 0.013 | +0.018 ± 0.022 | +0.001 ± 0.016 | +0.007 ± 0.017 |

Adding SF state halves the richness slope (+0.028 ± 0.010) and the size slope (−0.028 ± 0.015); adding z does not remove them. The same trends hold in Control4B (+0.043 ± 0.013, −0.023 ± 0.015). **The environmental dependence of Z★ acts mostly on spirals** (−0.060 ± 0.018 per dex of core size; ellipticals −0.010 ± 0.018) — which is exactly the pattern of §5: CG4 spirals, living in cores 0.31 dex smaller than the Control4C median, are predicted +0.019 dex more metal-rich by the control trend, and are observed at +0.018.

A Control4C environmental model (M★, log core size, log σ_quartet; coefficients +0.224, −0.050, +0.013) leaves mean residuals: Control4C 0.000 [−0.010, +0.011]; Control4B +0.004; **CG4 −0.013 [−0.038, +0.011]**; RG4 −0.027 [−0.054, −0.001] (the linear model under-predicts how metal-poor the N = 4 groups are; richness is not in the model because CG4 has no Lim parent).

**Answer to the question posed:** Control4C already looks like CG4 because it selects compact cores, and compact cores of ordinary groups host satellites whose Z★ at fixed mass follows a density sequence that CG4 obeys. The +0.033 vs RG4 is the difference between the dense and the sparse end of that sequence, not a compact-group property. This is the Pasquali et al. (2010) / Peng et al. (2015) satellite trend (higher Z★ at fixed M★ in denser environments, driven by quenched fraction and by star-forming galaxies being more enriched where gas supply is cut) reproduced inside our own control sample.

**BGGs** show the same sequence in weaker form: CG4 minus RG4 +0.016, minus C4C parents N = 5–7 −0.004, N = 8–12 −0.031 [−0.062, −0.000], compact cores −0.024, N ≥ 25 −0.064 [−0.096, −0.014] — but the BGG ceiling (§1) makes the rich-parent BGG values unreliable (98 % quenched, many at [Z/H] = +0.30).

## 8. Mass dependence without over-parametrisation [planned]

§3 and the binned medians in `control_only_residuals.png` (six 0.3-dex bins, 95 % CI of the median): CG4 satellites track the Control4C medians within ±0.03 dex in every bin; against RG4 the CG4 medians are above in five of six bins by 0.02–0.07 dex with no trend. **No structure with mass appears; I say so plainly.**

## 9. Star-forming CG4 BGGs [planned]

`bgg_sf_objects.csv`, `bgg_sf_leave_one_out.csv`, `bgg_sf_products.csv`.

The 11 objects: HMCG groups 61, 368, 25, 158, 175, 33, 265, 405, 173, 128, 330; all rank_M = 1 (BGG by stellar mass); 5 flagged `is_dominated`; 9 GZ1 spirals, 2 unclassified; log M★ 10.55–11.21; S/N 19–48 (one below 20); σ★ 111–187 km s⁻¹; sSFR −11.3…−9.8; MPA subclass: 4 STARFORMING, 1 STARBURST, 2 AGN, 1 BROADLINE, 3 unclassified; fibre fraction 0.03–0.22 (normal); FIREFLY E(B−V) 0.03–0.65; 3–7 SSP components. Residuals vs the Control4C star-forming-BGG MZR: −0.26, −0.21, −0.16, −0.15, −0.12, −0.08, −0.08, +0.01, +0.03, +0.05, +0.06 (control scatter 0.11).

| test | result |
|---|---|
| full | −0.083 [−0.149, −0.018] (11+106) |
| leave-one-out | −0.065 … −0.097; dropping the two lowest residuals −0.049 [−0.103, +0.008] |
| S/N > 20 | −0.100 [−0.160, −0.037] (10+105) |
| ELODIE lw / MILES mw / ELODIE mw | −0.047 [−0.099, +0.011] / −0.122 [−0.246, −0.004] / −0.067 [−0.146, +0.001] |
| Gallazzi (6 objects) / FIREFLY on the same 6 | −0.265 [−0.514, −0.052] / −0.141 [−0.215, −0.062] |
| all spiral BGGs (17+194) | −0.043 [−0.106, +0.014] |
| adjusted for D4000n / sSFR | −0.067 ± 0.028 / −0.073 ± 0.031 |
| light-weighted age of the same 11 | +0.030 [−0.061, +0.124] (not younger) |
| covariate balance vs control SF BGGs | D4000n −0.07 ± 0.07, sSFR +0.10 ± 0.15, S/N −1 ± 2, σ★ −11 ± 8 |

Verdict: **not fragile in the sense of being one object or one pipeline, but fragile in the sense that matters** — N = 11, found inside a hierarchy of subsets, and ≈2.5σ. The 6 objects common to Gallazzi are the low-Z half, so the Gallazzi number is selection, not confirmation. It is not explained by younger light. If anything follows, it is a gas-phase check (only 4 of the 11 have MPA star-forming BPT classes — too few) or an independent sample. Classified as a lead; no further budget recommended.

## 10. Age: exploratory [planned]

`age_contrasts.csv`, `age_grid_diagnostics.csv`, `age_same_objects.csv`, `figures/zstar_followup/age_effects.png`.

**Definitions verified.** FIREFLY `age_lightW`/`age_massW` in years, M11 grid 0.03–15 Gyr (above the age of the Universe; 0–1.8 % of BGGs and 0–2.2 % of satellites have light-weighted ages > 13.8 Gyr, ≈0.2 % at the 15-Gyr edge; mass-weighted ≈3–5 %). Analysed as log₁₀(age/yr); median 68 % half-width 0.07 dex (BGGs) – 0.10–0.13 dex (satellites; RG4 largest). Gallazzi `all_stat_age`: r-band light-weighted log(age/yr) from the same PDF machinery as Z, median uncertainty 0.09–0.12 dex, no grid pile-up. Object-to-object Gallazzi − FIREFLY(MILES lw): median −0.08 dex, robust scatter 0.18 dex — the two age scales are not interchangeable.

| age product | sat vs C4C, all | sat vs C4C, quenched | sat vs RG4, all | sat vs RG4, quenched | BGG vs C4C |
|---|---|---|---|---|---|
| MILES lw | +0.035 [−0.000, +0.069] | +0.036 [+0.000, +0.070] | +0.045 [−0.006, +0.095] | +0.085 [+0.023, +0.147] | +0.009 [−0.029, +0.045] |
| MILES mw | +0.036 [+0.007, +0.064] | +0.028 [−0.003, +0.057] | +0.038 [−0.004, +0.077] | +0.059 [+0.007, +0.110] | +0.011 |
| ELODIE lw | +0.032 [−0.003, +0.064] | +0.040 [+0.016, +0.066] | +0.017 [−0.031, +0.057] | +0.075 [+0.034, +0.118] | −0.013 |
| Gallazzi (N=63/46) | −0.001 [−0.062, +0.059] | +0.002 [−0.039, +0.039] | +0.053 [−0.025, +0.132] | +0.033 [−0.024, +0.090] | −0.039 [−0.105, +0.019] |

Adjusting for SF state (MILES lw, vs C4C): +0.027 [−0.008, +0.060]; for D4000n +0.027; for sSFR +0.026. Ellipticals: +0.033 (C4C), +0.099 [+0.024, +0.169] (RG4). Star-forming: +0.018, −0.016. On the 63+545 objects in common, FIREFLY gives +0.028 ± 0.029 and Gallazzi −0.001 ± 0.031 (difference −0.029 ± 0.033): the catalogues neither confirm nor contradict each other at that precision.

**Is age more interesting than Z★?** Modestly: the age signal has the same nominal significance as the RG4 metallicity contrast but (i) it is present against the primary control, (ii) it is present *within* the quenched class rather than being a mix effect, (iii) it is consistent across the three FIREFLY products, and (iv) it points the same way as the index-level hints of REPORT.md (HδA lower, Dn4000 higher in CG4). +0.035 dex ≈ 8 % ≈ 0.6 Gyr at 7 Gyr; +0.085 dex vs RG4 ≈ 1.3 Gyr. It is what "quenched earlier" would look like; it is also what a slightly different dust/α-abundance treatment could produce. One pipeline, ≈2σ, exploratory.

## 11. [α/Fe] feasibility [planned]

| resource | reference | data / release | coverage & selection | identifiers | expected CG_Gals N | definition | quality |
|---|---|---|---|---|---|---|---|
| Gallazzi, Pasquali, Zibetti, La Barbera 2021 | MNRAS 502, 4457 (arXiv:2010.04733) | SDSS DR7 MGS, legacy spectra | 113 307 galaxies with the five G05 indices measured and S/N ≥ 20; star-forming *and* quiescent | DR7 plate–MJD–fibre (matched to MPA-JHU) | ≈53 CG4 BGG + 116 CG4 sat (85 Q, 75 E); ≈636 + 1262 C4C; 52 + 75 RG4 (estimated from our S/N ≥ 20 legacy spectra with indices) | Δ(Mgb/⟨Fe⟩) between the observed spectrum and the BC03 (scaled-solar) PDF-weighted models, converted to [α/Fe] with TMK04 per age and Z; zero-point relative to the MW pattern, biased low at [Z/H] ≲ −0.4 | index-ratio error ≈0.14, PDF half-range ≈0.025 → σ([α/Fe]) ≈ 0.1 dex per galaxy; **available upon reasonable request to the corresponding author** |
| MOSES: Thomas et al. 2010; Johansson, Thomas & Maraston 2012 | MNRAS 404, 1775; 421, 1908 | SDSS DR4/DR6 early types | 0.05 ≤ z ≤ 0.1 (ETGs) | — | **0** (our sample is z ≤ 0.045) | Lick indices + TMB03/TMJ11 | — |
| SPIDER: La Barbera et al. 2014 | MNRAS 445, 1977 | SDSS DR7 ETGs | 0.05 < z < 0.095 | — | **0** | — | — |
| Scholz-Díaz, Martín-Navarro & Falcón-Barroso 2022/2023 | MNRAS 511, 4900; 518, 6325 | SDSS DR7 + Yang groups | *central* galaxies only | — | BGGs at best; not public | full-spectrum fitting, [Mg/Fe] | — |
| FIREFLY DR16 (used here) | Comparat et al. 2019 | — | — | — | — | **no [α/Fe]** (M11-MILES, scaled-solar) | — |
| MaNGA VACs (Pipe3D, FIREFLY-MaStar) | DR17 | IFU | ~10⁴ galaxies | — | a handful of CG4 at most | — | — |

Home-made routes (not attempted, per the brief): (a) Lick-index inversion of our own MPA `Mgb`, `Fe5270`, `Fe5335`, `Hβ` with TMJ11 α-variable models — feasible, but the unresolved Lick/IDS transformation and σ-broadening treatment of the MPA indices (REPORT.md §6) must be fixed first; (b) pPXF with sMILES (Knowles et al. 2021/2023) or α-MILES (Vazdekis et al. 2015) on the ≈250 CG4 + ≈700 compact-core control spectra, [α/Fe] as a free parameter, using the same fibre spectra. Either is Paper III work. **Realistic path: write to A. Gallazzi.**

## 12. Being a scientist about it [post-hoc unless stated]

### 12.1 What histories fit "different morphology, same Z★"?
The Paper II result is a morphology (and quenched-fraction) excess in CG4. Z★ at fixed mass is, to ±0.02 dex, what ordinary compact cores show. Three readings are compatible:

1. **Morphological transformation without chemical consequence.** Whatever turns CG4 spirals into ellipticals (tidal harassment, minor mergers, disc fading) acts on stellar systems whose chemical enrichment was already set; light-weighted Z★ changes by the ≈0.03 dex it changes anywhere a spiral quenches. The E/S sign flip of §5 is a natural corollary: "new" ellipticals carry spiral-like Z★ (CG4 ellipticals slightly metal-poor, −0.02 to −0.03), while the surviving spirals are the more evolved, gas-poor ones (CG4 spirals slightly metal-rich, +0.02 to +0.05; gas-phase O3N2 hint +0.02–0.04). This is the reading I favour, and it makes a prediction: CG4 ellipticals should be *younger* than control ellipticals at fixed mass. FIREFLY says the opposite or nothing (+0.033 ± 0.022 vs C4C) — a genuine tension worth a dedicated test, unless the morphology transformation happened > 5 Gyr ago, in which case light-weighted ages cannot see it.
2. **Same population, different sampling.** CG4 are the compact cores of ordinary groups caught at pericentre (the Paper I "embedded/predominant" picture). Then all stellar-population properties should match Control4C at the CG4 compactness, which is what §7 finds (residual −0.013 ± 0.012). The age excess (+0.035) is the only thing this reading does not obviously produce.
3. **Early formation in dense environments** (Gallazzi et al. 2021's "ancient infallers"): older and slightly α-enhanced satellites at fixed Z★. Predicts the age excess *and* an [α/Fe] excess among quenched CG4 satellites, with Z★ ≈ 0 — consistent with everything here; the [α/Fe] request (§11) is the discriminating measurement.

### 12.2 What would discriminate
- **[α/Fe] of quenched CG4 satellites vs compact-core controls** (reading 3 predicts +, readings 1–2 predict 0). Gallazzi et al. 2021 catalogue; ≈85 CG4 quenched satellites.
- **Ages of CG4 ellipticals vs control ellipticals at fixed M★ and σ★** (reading 1 predicts younger). Needs a second age estimator; the pPXF/sMILES run of §11 would give ages, Z★ and [α/Fe] in one go.
- **Gas-phase metallicity of star-forming satellites at fixed M★ and SFR** (reading 1 predicts +, strangulation). Post-hoc first look below.
- **Position in the age–Z★ plane relative to the compact-core sequence** rather than a Z★-only contrast: the density sequence of §7 defines the expectation; a CG4 deviation would have to be measured against it, not against RG4.

### 12.3 A cheap test we had not run: gas-phase metallicity [post-hoc]
`gas_phase_contrasts.csv`, `gas_phase_census.csv`, `figures/zstar_followup/gas_phase_effects.png`. MPA-JHU fluxes already cached by the project (same fibre spectra; MPA BPT class STARFORMING/STARBURST, which enforces S/N > 3 in the four lines), Pettini & Pagel (2004) N2 and O3N2 calibrations, quadratic mass term, group bootstrap. Satellites only (4 CG4 star-forming BGGs).

| calibration | vs C4C (55+747) | vs RG4 (55+80) | vs C4B (55+693) | + log SFR (vs C4C / RG4) |
|---|---|---|---|---|
| N2 | +0.001 [−0.010, +0.011] (52+651) | +0.006 [−0.008, +0.020] | +0.003 | +0.002 / +0.008 |
| O3N2 | +0.019 [−0.002, +0.041] | **+0.044 [+0.018, +0.071]** | +0.023 [+0.001, +0.047] | +0.019 / +0.044 |

O3N2 gives a +0.02–0.04 dex excess for CG4 star-forming satellites at fixed mass and SFR — the size and sign of the Pasquali et al. (2012) satellite excess and of the stellar spiral offset of §5 (+0.017 vs C4C, +0.056 vs RG4); N2 gives nothing, but N2 saturates at the near-solar metallicities of these galaxies (median 12+log O/H ≈ 8.63) while O3N2 also responds to ionisation parameter. **Suggestive, not established**; the proper version uses the MPA `oh_p50` (Tremonti et al. 2004) or Curti et al. (2020) calibrations and a fibre-covering-fraction control. It is the single cheapest next measurement.

### 12.4 Anomalies and corrections to the previous work
- RG4 ⊂ Control4C/Control4B was not stated; "three controls" are two nested populations. Interpretation of the RG4 contrast changes accordingly (§7).
- FIREFLY BGG metallicities are censored at the grid ceiling (10–14 % of BGGs at +0.30) and strongly quantised; BGG contrasts are compressed. Not a bug, but the report's BGG numbers deserve that caveat.
- The morphology/SF "decomposition" hides a sign-changing interaction; the "total ≈ 0 vs Control4C" statement is a cancellation, not an absence of structure.
- No numerical error found: every historical number reproduces; the bootstrap, clustering, and mass control are sound.

## 13. Robustness table (satellites, FIREFLY MILES lw) [robustness]

`robustness.csv`.

| variant | vs C4C | vs RG4 |
|---|---|---|
| OLS (reference) | +0.003 [−0.019, +0.024] | +0.033 [+0.003, +0.064] |
| median regression | −0.003 [−0.020, +0.027] | +0.028 [+0.007, +0.062] |
| lowest 2 % residuals removed | −0.006 [−0.027, +0.013] | +0.021 [−0.007, +0.048] |
| S/N > 20 / > 15 | −0.004 (historical) | +0.018 [−0.016, +0.054] (120+79) / +0.030 [−0.002, +0.062] |
| [Z/H] uncertainty ≤ 0.1 dex | −0.000 (historical, ≤ 0.3) | +0.032 [−0.001, +0.064] |
| z ≤ 0.04 | — | +0.023 [−0.016, +0.062] (125+81) |

BGGs: median regression −0.007 [−0.035, +0.027] vs C4C; +0.012 vs RG4.

## 14. Files

`outputs/zstar_followup/`: `audit_reproduction.csv`, `audit_censoring.csv`, `same_objects.csv`, `mass_interaction.csv`, `mass_bins.csv`, `control_only_residuals.csv` (+`_objects.csv`), `rg4_subsets.csv`, `rg4_joint_models.csv`, `rg4_composition_split.csv`, `composition_balance.csv`, `adjusted_contrasts.csv`, `environment_sequence.csv`, `within_control_environment_slopes.csv`, `environment_model_residuals.csv`, `bgg_sf_objects.csv`, `bgg_sf_leave_one_out.csv`, `bgg_sf_products.csv`, `age_contrasts.csv`, `age_grid_diagnostics.csv`, `age_same_objects.csv`, `gas_phase_contrasts.csv`, `gas_phase_census.csv`, `robustness.csv`, `followup_worktable.csv` (one row per sample×galaxy with the group-level covariates used here), `key_numbers.json`, `followup_manifest.json` (sha256 of every product).
`figures/zstar_followup/`: `environment_sequence.png`, `rg4_decomposition.png`, `rg4_interactions_by_product.png`, `control_only_residuals.png`, `delta_z_vs_mass.png`, `same_objects.png`, `age_effects.png`, `gas_phase_effects.png`.
`work/catalogues/alpha_fe/`: Gallazzi et al. 2021 PDF and extracted text (the only download of this session).

Inputs are read-only: the historical `outputs/zstar/*`, `work/metallicity_worktable.csv`, the cached FIREFLY/Gallazzi matches, and the project's `data/processed_sample.pkl` and `data/PC_Groups.csv` (group sizes, quartet σ, parent richness, MPA line fluxes).

## 15. Limits

Same as before, plus: the environmental sequence of §7 is derived inside a control sample whose cores were *selected* (three nearest companions within 3 mag), so "core size" partly encodes richness and the selection; the O3N2 result is post-hoc and calibration-dependent; FIREFLY ages carry an unknown α-abundance/dust systematic; every subset in §5 and §9 was looked at because the data suggested it, and the intervals are not corrected for that.
