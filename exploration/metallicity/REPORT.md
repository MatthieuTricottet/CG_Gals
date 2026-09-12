# Stellar-population index feasibility for CG4 satellites — scoping report

*Auto-generated from `outputs/metallicity_scoping.json` (`submission-qc-float-fix` @ `7367e51`). Exploratory scoping, not inference: no p-values, no multiplicity correction.*

## 1. What data exist
- Locally: the four sample tables, the pipeline SDSS cache (69995 rows; `galSpecExtra`/`galSpecLine`/`zooSpec` only) and the size caches. **No** `galSpecIndx`, per-galaxy σ or spectrum S/N existed locally.
- Fetched (path A; SkyServer DR18 because the DR16 host fails TLS — `galSpec*` are the unchanged MPA-JHU DR8 tables): `galSpecIndx` (indices, errors, raw + emission-subtracted), `galSpecInfo` (`v_disp`, `sn_median`, `reliable`), `SpecObjAll` cross-checks, for 3857 objids + 958 Lim-host members. Cached under `work/`.
- Datamodel (primary sources saved in `work/`): Å for atomic indices, mag for Mg2; Lick EW sign convention (verified: quenched median Mgb 4.19 Å, HδA -1.95 Å); no σ-broadening correction is applied to the data (the models are broadened to each galaxy's σ, Kauffmann+03a §2); Lick/IDS transformation **UNRESOLVED** (§6); MPA rows zeroed with `plateid=-1` when not run, `_err=-1` = index not measured.
- Sentinel finding: the Fe5335 (104) and Mg2 (115) `_err=-1` flags cluster at z = 0.0398–0.0408, where the red pseudo-continuum meets the 5577 Å sky line → a z-dependent hole in Fe5335/[MgFe]′.

## 2. How many satellites are usable, and which
| sample | sat input | MPA result | 7 indices valid | + σ + S/N | S/N>20 | S/N>30 |
|---|---:|---:|---:|---:|---:|---:|
| CG4 | 186 | 177 | 172 | 168 (62 groups) | 120 (60) | 61 |
| RG4 | 168 | 166 | 163 | 159 (56 groups) | 78 (48) | 27 |
| Control4B | 2094 | 1995 | 1943 | 1917 (696 groups) | 1510 (666) | 993 |
| Control4C | 2109 | 1996 | 1945 | 1899 (700 groups) | 1284 (651) | 697 |
| pooled (dedup) | 3092 | – | 2864 | – | 2055 | 1241 |

Galaxy-by-galaxy list (objid, group, S/N, indices, loss reason): `outputs/usable_satellites.csv`.
CG4 satellite losses: 9 no_MPA_result(BOSS/not-run), 5 core_index_missing_or_zeroed, 4 sigma_invalid. The S/N>20 cut is differential between samples (CG4 71%, RG4 49%, Control4B 79%, Control4C 68% of usable satellites kept).
**Decisive question.** At S/N > 20, 66% of satellites survive (2055/3092). They are **not** just the brightest quartile (34% of survivors come from it), but the selection is strongly graded: survival by M_r quartile (bright→faint) 91% / 80% / 60% / 35%; by stellar-mass quartile (low→high) 27% / 65% / 90% / 97%; survivors are 0.32 mag brighter and 0.19 dex more massive at the median. Logistic selection per SD: M_r -2.10, σ +2.48, R50 -1.43, z -1.04: a hard S/N cut breaks matched balance in luminosity, size and redshift.

## 3. Δ_min per index (3σ group-level mean offset; blinded labels)
Smallest blinded label at S/N>20: N_g = 52, n̄_sat = 2.17; ρ = empirical one-way ICC (group bootstrap); σ_idx = total satellite scatter, of which measurement error is only 34% (Mgb) / 30% ([MgFe]′).

| index | unit | median err @S/N 20–25 | ρ | Δ_min (S/N>20) | Δ_min (no cut) | 2-sample, smallest vs largest | Δ_min/IQR |
|---|---|---:|---:|---:|---:|---:|---:|
| Mgb | A | 0.429 | 0.17 | **0.265** | 0.290 | 0.275 | 0.22 |
| Fe5270 | A | 0.446 | 0.16 | **0.161** | 0.191 | 0.168 | 0.29 |
| Fe5335 | A | 0.453 | 0.16 | **0.202** | 0.302 | 0.209 | 0.36 |
| MgFe | A | 0.276 | 0.18 | **0.188** | 0.205 | 0.196 | 0.27 |
| MgbFe |  | 0.241 | 0.10 | **0.127** | 0.602 | 0.132 | 0.35 |
| Mg2 | mag | 0.014 | 0.19 | **0.019** | 0.019 | 0.020 | 0.22 |
| D4000n |  | 0.026 | 0.18 | **0.077** | 0.077 | 0.080 | 0.22 |
| HdA | A | 0.799 | 0.17 | **0.612** | 0.646 | 0.636 | 0.22 |
| Hb_sub | A | 0.420 | 0.15 | **0.217** | 0.237 | 0.226 | – |

Δ_min is nearly flat against the S/N threshold (intrinsic scatter dominates): the cut buys little power while imposing the graded selection of §2. Raw Hβ is unusable (emission fill-in); use `_sub`. Figure: `figures/fig_power_vs_SN.png`.

## 4. Dominant systematic (blinded)
| index | Δ_min (S/N>20) | aperture shift over IQR(r_fib/R50) at fixed σ | z shift over IQR(z) at fixed σ | GZ1 E−S at fixed σ | morph/Δ_min |
|---|---:|---:|---:|---:|---:|
| Mgb | 0.265 | +0.032 | -0.056 | +0.206 | 0.78 |
| Fe5270 | 0.161 | +0.008 | -0.038 | +0.116 | 0.72 |
| Fe5335 | 0.202 | +0.015 | -0.046 | +0.113 | 0.56 |
| MgFe | 0.188 | +0.021 | -0.046 | +0.158 | 0.84 |
| MgbFe | 0.127 | +0.015 | +0.034 | +0.016 | 0.13 |
| Mg2 | 0.019 | +0.004 | -0.002 | +0.020 | 1.08 |
| D4000n | 0.077 | +0.027 | -0.023 | +0.105 | 1.37 |
| HdA | 0.612 | -0.240 | +0.054 | -0.827 | 1.35 |

- Aperture (1.5″ fibre radius / Simard R_chl,r, Petrosian fallback; Planck15 angular-diameter distance): median r_fib/R50 = 0.33 (IQR 0.24–0.47); metal-line gradients at fixed σ are ≤ 0.12 Δ_min over the IQR — subdominant, as is redshift.
- **Morphology is the dominant systematic**: at fixed σ, GZ1 ellipticals − spirals = +0.21 Å (Mgb), +0.16 Å ([MgFe]′), +0.105 (Dn4000), -0.83 Å (HδA) — 0.6–1.4 × Δ_min, and σ-dependent (Mgb E−S +0.57 Å at σ<80 vs +0.11 at 160–220 km/s). Since CG4's headline result is a morphology excess, any index signal must be measured within morphology class. Mgb/⟨Fe⟩ is morphology-independent at fixed σ (+0.016).
- Age–Z plane: HδA_sub vs [MgFe]′ Spearman -0.82, Dn4000 vs [MgFe]′ 0.84; residual width at fixed [MgFe]′ is 2.0× (HδA) / 5.9× (Dn4000) the median error: real width at population level, marginal per galaxy.

## 5. Exploratory unblinded contrasts (Phase 4; no inference)
CG4 − control, satellites, fixed morphology & σ, no S/N cut (coef ± group-clustered SE [68% group-bootstrap CI]); last column = within-host fixed-effects analogue of the conditional-logit design:

| index | vs C4B | vs C4C | vs RG4 | within-host (sat.) | Δ_min (no cut) |
|---|---|---|---|---|---:|
| Mgb | +0.105 ± 0.058 [+0.05,+0.16] | +0.044 ± 0.058 [-0.02,+0.10] | +0.128 ± 0.092 [+0.04,+0.22] | -0.008 ± 0.077 | 0.290 |
| Fe5270 | +0.009 ± 0.049 [-0.04,+0.06] | -0.003 ± 0.049 [-0.05,+0.04] | -0.065 ± 0.095 [-0.16,+0.02] | -0.069 ± 0.089 | 0.191 |
| Fe5335 | +0.001 ± 0.060 [-0.06,+0.06] | -0.008 ± 0.062 [-0.07,+0.05] | +0.121 ± 0.144 [-0.02,+0.25] | -0.095 ± 0.087 | 0.302 |
| MgFe | +0.040 ± 0.045 [-0.00,+0.09] | +0.002 ± 0.045 [-0.04,+0.05] | +0.040 ± 0.075 [-0.04,+0.12] | -0.045 ± 0.069 | 0.205 |
| MgbFe | -0.064 ± 0.080 [-0.14,+0.02] | -0.050 ± 0.059 [-0.10,+0.01] | -0.105 ± 0.155 [-0.27,+0.05] | -0.052 ± 0.079 | 0.602 |
| Mg2 | +0.007 ± 0.003 [+0.00,+0.01] | +0.004 ± 0.003 [+0.00,+0.01] | +0.005 ± 0.005 [-0.00,+0.01] | -0.002 ± 0.004 | 0.019 |
| D4000n | +0.029 ± 0.015 [+0.02,+0.04] | +0.007 ± 0.015 [-0.01,+0.02] | +0.024 ± 0.024 [+0.00,+0.05] | -0.037 ± 0.021 | 0.077 |
| HdA_sub | -0.466 ± 0.175 [-0.63,-0.28] | -0.203 ± 0.174 [-0.38,-0.03] | -0.554 ± 0.285 [-0.86,-0.28] | +0.446 ± 0.217 | 0.780 |
| Hb_sub | -0.146 ± 0.053 [-0.20,-0.09] | -0.068 ± 0.052 [-0.12,-0.02] | -0.224 ± 0.096 [-0.32,-0.14] | +0.134 ± 0.088 | 0.237 |

All metal-line contrasts are below Δ_min (mostly ≤ 0.4 Δ_min) and consistent with zero; Balmer/Dn4000 offsets vs C4B/RG4 are ~2 SE (older-looking CG4 satellites) but ≈ 0 vs C4C, and the within-host contrasts flip sign between the no-cut and S/N>20 variants. Nothing here is a detection.

## 6. Recommendation and unresolved items
**Go/no-go: conditional GO — only as a prespecified, within-morphology, S/N-uncut, error-weighted design.** N is adequate (168 usable CG4 satellites in 62 groups; 120 at S/N>20), Δ_min ≈ 0.22 × IQR (0.26 Å Mgb, 0.19 Å [MgFe]′) and no systematic swamps it; but morphology ≈ Δ_min and the S/N cut is differential, so Paper III must (i) replace the hard S/N cut by measurement-error weighting, (ii) stratify/match on GZ1 class and σ, (iii) take C4C as primary control, (iv) prespecify before any SSP fitting. The exploratory contrasts show no metal-line effect above ~0.4 Δ_min.

UNRESOLVED (not verifiable from primary sources consulted):
1. Lick/IDS resolution transformation of the MPA indices: UNRESOLVED (explicit statement absent); strong indirect evidence for native SDSS resolution.
2. Dn4000 flux-density convention (F_ν vs F_λ): not stated in the catalogue docs (Kauffmann+03a define D4000 via F_ν).
3. The `_err = -1` convention is inferred empirically (datamodel text only describes whole-row zeroing).
4. DR18 = DR16 `galSpecIndx` rows is asserted from the run2d=26-only coverage statement and the 100% `specobjid` cross-match, not a row-level DR16 diff (endpoint unreachable).
5. Blinded balance tests are null by construction; true-label balance after an S/N cut was not tested (only the differential survival above).

`git status --porcelain` at report time: 9 entries (`exploration/` plus pre-existing/concurrent-session paths; see INVENTORY.md and the final session log).
