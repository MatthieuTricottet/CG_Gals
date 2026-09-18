# gary-r2 Phase 1 — read-only diagnostics (2026-09-17)

Scripts: `analysis/gary_r2/d*.py` (run from the repo root with `.venv_clean`).
Scalars: `diagnostics_gary_r2.json` here (canonical) and mirrored into
`output/results.json['diagnostics_gary_r2']` (results.json is rewritten by every
pipeline run; Phase 2 should make `generate_report` load the canonical file).
No pipeline code, data file, or manuscript text was modified.

## D1 — Missing-sSFR origin (394 / 6076 rows)

| type | N | instrument | run2d | median snMedian_r |
|---|---|---|---|---|
| no-galSpecExtra | 332 (84 %) | **BOSS, all 332** (survey `boss`) | v5_7_0 | 45.9 |
| sentinel `sfr_tot_p50 = −9999` | 60 (15 %) | SDSS legacy | 26 | 40.1 |
| no-specObjID | 2 (Control4B) | — | — | — |
| valid (5682) | | SDSS legacy, **0 BOSS** | 26 | 39.0 |

Per sample (no-galSpecExtra / sentinel / no-id): CG4 14/4/0, Control4B 149/29/2,
Control4C 164/27/0, RG4 5/0/0. Every missing row is assigned a category.

S/N: the missing rows have **higher**, not lower, `snMedian_r` than the valid rows
in every sample (rank-sum p = 7×10⁻⁴, 1×10⁻⁵, 4×10⁻¹⁰, 2×10⁻³ for CG4/C4B/C4C/RG4;
pooled unique spectra p = 2×10⁻¹²). The 60 sentinel rows are legacy spectra with
`galSpecInfo.reliable = 1`, `zWarning = 0` and a valid `lgm_tot_p50`.

**Verdict:** BOSS-instrument spectra that the MPA-JHU tables (run2d = 26 only) never
covered (84 %), plus MPA-JHU SFR fit failures at normal S/N (15 %). Not low-S/N spectra.
→ The Sect. 2.2 / App. A sentence "the available cross-match is therefore insufficient
to establish the origin" can be replaced by this statement.

## D2 — SFMS sign conflict

(a) The order-2 polynomial is fitted on the **SDSS non-AGN reference** star-forming
galaxies (N = 44 192; `main.sSFR_properties`); stored `MS_res` reproduces the refit to
1e-16. (b) "Matched star-forming galaxies" = pairs of the pooled deduplicated
galaxy-level match in which **both** members are GMM star-forming: **39 pairs** (of 234;
84 CG4 members are SF, 45 of them are paired with a non-SF control); the statistic is the
**mean** paired difference (published −0.147, p = 0.064). (c) On the same 39 pairs the
median difference is −0.045, the median of paired differences −0.078 (Wilcoxon p = 0.26;
15/39 pairs have CG4 above its control). The raw published offsets are medians over all
SF galaxies (84 vs 1019 / 1061 / 118): +0.056 (p = 0.064), +0.058 (p = 0.030), +0.046
(p = 0.59); as means they are +0.040, +0.046, +0.001. (d) SF stellar masses: CG4 median
9.99 vs C4B 10.27, C4C 10.05, RG4 9.94; the matched SF controls (9.85, 16th pct 9.29)
are less massive than their CG4 partners (9.95, 9.64).

**Explanation:** not a bug — different estimands (median over all SF galaxies vs mean over
39 both-SF pairs), different populations (full control SF samples vs mass/rank-matched
pooled controls), and a strong mass dependence of the residuals about the *field*
main sequence: control SF galaxies with log M* ≥ 10.3 lie 0.19–0.21 dex **below** the
field relation while those below 10.3 lie slightly above; the 23 massive SF CG4 galaxies
show no such depression (+0.00), which drives the positive raw medians. Only the raw
C4C offset reaches p < 0.05. Recommendation: report both with their definitions, drop
"the largest", and fit the SFMS panel of Fig. D.1 on the reference sample as now.

## D3 — Zheng–Shen class vs host mass  ⚠ changes a claim

(a) Definition (Zheng & Shen 2021, Sect. 2.3, Eq. 1; Paper I Sect. 3.3): *embedded* =
CG contributes **less than half** of the parent-group luminosity; *predominant* = half or
more; *isolated* = same membership as the parent; *split* = several parents.
Implementation: the inherited `Class` column of `data/CG4_Groups.csv`, used verbatim
everywhere. (b/c) Re-deriving f_L = L_CG4 / L_host from the Lim–Tempel membership:

| stored label | N | median host log M200c | host richness | f_L range |
|---|---|---|---|---|
| Isolated | 6 | 12.86 | 4 | 1 |
| "Embedded" | 37 | 13.13 | 5–24 | **0.51–0.98** |
| "Predom" | 19 | 13.83 | 13–93 | **0.06–0.48** |

The stored "Predominant" hosts are indeed the more massive ones, **but this contradicts
the definitions: all 56 non-isolated labels are swapped** (0.5 boundary reproduced
exactly, no ambiguous group). Under the correct labels there are 37 Predominant and 19
Embedded CG4s, and Embedded hosts are the massive ones, as the definitions imply.
Paper I's Sect. 3.3 counts (19 Predominant / 37 Embedded) and Sect. 4.3 prose are also
affected (author-level decision). Affected here: Table E.2 (row labels), Table E.1,
App. C Table C.1 rows, Sect. 3.4 text ("Embedded and Predominant" jointly — unaffected),
class permutation tests (Isolated vs rest — unaffected).
(d) CG4-satellite E-vs-S logistic (157 satellites, 59 groups, cluster-robust): vs Isolated,
OR ≈ 4.4 (Predominant-correct) and 4.7 (Embedded-correct), both p < 0.001; adding
log M200c: 5.0 / 6.5, log M200c OR 0.84 (p = 0.40). Embedded vs Predominant only: OR 0.97
(p = 0.94), with log M200c 0.74 (p = 0.44). Host mass adds nothing; only the 6-group
Isolated class differs (14 classified satellites — exploratory).

## D4 — DS18 → GZ1 mapping (pooled, environment-agnostic)

2436 / 3857 unique galaxies matched to DS18. TType ≤ 0 and P_S0 > 0.5 (N = 730):
E 70.0 %, S 18.8 %, U 6.2 %, NoGZ 5.1 %. P_S0 ≥ 0.8 (N = 429): E 62.9 %, S 24.5 %,
U 7.7 %, NoGZ 4.9 %. Inclination (DS18 P_edge_on): f_E = 0.77 (0.75) for P_edge_on < 0.1
vs 0.47 (0.38) for ≥ 0.5 at the two thresholds; in the 0.7–1 bin E and S are equal
(0.41/0.41 and 0.33/0.48); rank-sum P_edge_on E vs S: p = 1×10⁻⁵ / 4×10⁻⁷.
Our GZ1 E class (942 matched): 92 % TType ≤ 0, 54 % early with P_S0 > 0.5, 29 % ≥ 0.8,
17 % early with P_S0 < 0.2. No CG4/control split computed.

## D5 — σ_v sensitivity of the group-level match

Control4B: the with- and without-σ_v matched sets share only **7/54 control groups**
(0 identical pairs); the CG4 side is identical (54 groups), the matched-control mean
elliptical-satellite fraction drops by 0.123 → Δ 0.160 → 0.284. Without σ_v the residual
σ_v SMD is +0.27 (CG4 hotter: medians 160 vs 129 km s⁻¹) and the redshift SMD worsens
(−0.08 → +0.25); with σ_v all |SMD| ≤ 0.17. Control4C: 10/54 shared, Δ 0.114 → 0.102,
σ_v SMD −0.11. RG4: 21 shared, Δ 0.196 → 0.253, σ_v SMD +0.52 (28 → 31 pairs).
Interpretation: matching on σ_v selects Control4B quartets from dynamically hotter
(richer, more early-type-rich) hosts, raising the control E fraction; without it the
controls are cooler and less well balanced in z. The two values bracket the C4B
contrast; the ordering C4C < C4B, RG4 holds either way. Keep the with-σ_v match primary.

## D6 — Holm across the three per-control tests (stored only)

| family | C4B raw → Holm | C4C | RG4 |
|---|---|---|---|
| Table 3 permutation p | 0.016 → 0.047 | 0.080 → 0.080 | 0.026 → 0.053 |
| adjusted elliptical OR p | 1e-6 → 3e-6 | 6e-4 → 6e-4 | 6e-6 → 1e-5 |
| adjusted quenched OR p | 0.0013 → 0.0038 | 0.17 → 0.35 | 0.24 → 0.35 |

RG4's Table 3 permutation p crosses 0.05 after Holm (0.053).

## D7 — Fig. 2 statistic audit

All 20 morphology bins are displayed (≥ 5 galaxies, ≥ 3 groups). Medians of the
bimodal vote fractions collapse toward 0/1 and exaggerate the CG4–control gap in the
intermediate bins (e.g. 10.0–10.5: median p_el 0.65 vs 0.26 for C4B, whereas f_E is
0.62 vs 0.36 and mean p_el 0.56 vs 0.36). Qualitative sign disagreements (CG4 − control)
between median and fraction/mean occur in 11 (control, bin) combinations, all in the
lowest bin (7.0–9.5: CG4 has 10 galaxies, 0 classified E; the median says CG4 is *more*
elliptical, the fraction says less) and the highest bin vs C4C (11.0–12.5, |Δ| ≈ 0.02).
Fractions and means agree everywhere. → Recommend fraction classified E/S (A1), with
mean debiased votes as open markers.
