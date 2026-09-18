# gary-r2 — section-by-section change log (Gate C, 2026-09-17)

Compiled PDF: `output/paper/paper.pdf` (21 pages: main text + references 11,
appendices 10). Main-text words 7277 → 4706 (texcount, −35 %). All numbers
render from JSON/macros; `validate_template.py`, `audit/consistency_gate.py`,
`analysis/gary_r2/style_checks.py`, `analysis/gary_r2/phase4_checks.py` and
the 127-test pytest suite pass.

## Front matter
* Abstract rewritten: 277 words, no citations; the matched-quartet sentence now
  reads "significant against Control4B and RG4, same sign but not significant
  against Control4C (Δ = 0.11, p = 0.08)" from `results.json`.
* Title unchanged.

## 1 Introduction
* Last paragraph shortened; section roadmap merged into it.

## 2 Data and classifications
* 2.1 Samples: Zheng–Shen classes defined in the text (isolated / predominant ≥ ½
  host luminosity / embedded) with the corrected counts (6/37/19); new Fig. 1
  (sample schematic: embedded CG4 204 in Lim host 1117 with the would-be
  Control4B/4C quartets; RG4 10914); new Table 1 (groups, galaxies, median
  R_pair per sample); exclusion statement now "51 of the 56 non-isolated"
  (rendered); Lim 3688 sentence kept; file path removed; separation numbers
  moved to Table 1.
* 2.2 Classifications (ex "Star-formation and morphology classes"): missing-sSFR
  origin from D1 (BOSS coverage gap + sentinels, higher S/N), citation for the
  −9999 convention now Brinchmann+04 / Kauffmann+03b; what the E class contains
  (DS18 cross-match: 92 % TType ≤ 0, 54 % / 29 % S0-like; pooled DS18 S0 → E/S
  fractions and the inclination dependence from D4); Dn4000/HδA retrieval
  mentioned. No CG4-vs-control S0 contrast anywhere.
* 2.3 Galaxy sizes: condensed (same content).
* 2.4 Statistical approach: two paragraphs; p-value conventions, completeness
  rule, gapper Monte Carlo, overlap coefficient moved to new Appendix D.

## 3 Results
* 3.1 Morphology (ex 3.1 raw, 3.2 adjusted, 3.3 matched, 3.4 within-host):
  raw fractions (59 % vs 47/48/32 %) with Fisher tests; Fig. 2 with the
  classified-fraction statistic and mean-vote open markers (caption updated);
  per-control adjusted odds ratios with CIs (Fig. 3); pooled satellite/BGG
  sentence; group-level match (Table 4, now with a Holm column; Control4C stated
  as same sign, not significant); no-σ_v explanation from D5 (7/54 shared
  controls); individual match; within-host paragraph with the 56 → 54 host
  explanation (two Lim groups host two CG4s each) and the radius-free contrast.
* 3.2 Star formation at fixed morphology (ex 3.2 quenched ORs, 4.3 Kitagawa,
  3.7 SFMS): raw and adjusted quenched contrasts; P(Q|E) homogeneity and
  Kitagawa terms moved here; matched quenched difference; SFMS paragraph with
  the D2 explanation ("the largest" removed) and the new quenched-sequence
  panel (Fig. 4, ex D.1, caption without the normalisation sentence, filled
  median symbols); fixed-threshold sentence merged.
* 3.3 Compact-group classes (new): Fig. 5 (satellite f_E by class with control
  bands and host log M200c); isolated permutation tests; predominant-vs-embedded
  OR 0.97 and the log M200c non-effect from D3; note on the Paper I label swap.
* 3.4 Local projected density (ex 3.6 tidal index): compressed to six sentences
  keeping "entangled with selection, attenuation ≠ mediation" and "residual
  term not robust on common support"; the Sect. 4.1 repetition removed.
* 3.5 Other galaxy properties (ex 3.7/3.8): Fig. 6 (strong-Hα fraction, AGN-like
  fraction, median Dn4000, median log R_chl,r vs M*); Hα, AGN, Dn4000 (rendered
  bin values), sizes condensed; magnitude gaps, phase space, colours and
  group-scale diagnostics moved to Appendix F.
* 3.6 Robustness (ex 3.5): one paragraph + Table 5 (now with a "without σ_v"
  row); details in Appendix I.

## 4 Interpretation and comparison with previous work (retitled)
* 4.1: tidal-index numbers no longer repeated in full; S-PLUS citations added;
  isolated-group and M200c statements; two-sentence metallicity deferral
  (Peng+15; single-fibre aperture).
* 4.2: S0-degeneracy discussion shortened; the CG4-vs-control S0 share removed
  (reserved for the blind Paper III analysis).
* 4.3: "Control4C is higher still" replaced by "highest close-pair fraction of
  the controls"; aperture-correction citation Brinchmann+04 + Salim+07;
  Kitagawa numbers now only in 3.2.

## 5 Conclusion
* Condensed; Dn4000 added to the list of non-drivers; "elliptical/spiral"
  wording throughout.

## Data availability
* File paths (data_loader, ssfr_quality_audit, spectral_indices, diagnostics)
  collected here.

## Appendices
* A SDSS data query: galSpecIndx join added; D1 result replaces "insufficient
  to establish the origin" (BOSS/sentinel/no-id counts per sample, S/N test).
* B sSFR classification details: unchanged content; Fig. B.1 panels now
  Elliptical/Spiral/Uncertain; Table B.1 bounds; Table B.2 P(Q|morphology).
* C Projected cores: statement now "37/37 predominant and 14/19 embedded"
  (rendered); table rows follow the corrected labels.
* D Statistical conventions (new): independence unit, p-value conventions,
  covariates and gapper Monte Carlo, overlap coefficient.
* E Main-sequence and quenched-sequence details: order selection for both
  sequences; class-split sentence.
* F Secondary and environmental diagnostics: magnitude gaps (screen sample
  stated: all 1519 quartets), projected phase space, distance to BGG,
  dominance (60 % split + new median-f_L,BGG split), morphology and
  domination, Zheng–Shen tables (corrected labels, table foot explains that
  embedded CGs live in the richer hosts), crossing time (colour sentence
  removed), optical colours (colour-audit sentence added, p = 0.050).
* G Second-order battery (new): Table G.1 (20 fits, BH across the battery).
* H Pooled models, matching details, selection diagnostics: pooled forest
  (Fig. H.1) moved here from the main text; matching details; Fig. H.3
  availability with the Dn4000/HδA column; T_i host-inclusive check.
* I Robustness details (new): crowding, no-σ_v reruns with the D5
  interpretation, within-host estimators and host-BGG alignment (6 misaligned
  systems are embedded), DS18 classification.
* J Extended size diagnostics: unchanged content.

## Figures (all regenerated)
Fig. 1 schematic (new) · Fig. 2 mass trends (new statistic) · Fig. 3 per-control
forest (labels, colours, no minor ticks) · Fig. 4 residual ECDFs (two panels) ·
Fig. 5 CG4 classes (new) · Fig. 6 property trends (new) · B.1 (E/S labels) ·
F.1 colour coefficients (legend labels) · H.1 pooled forest (ticks) · H.2 balance ·
H.3 availability (row + labels) · J.1 size forest (labels).
