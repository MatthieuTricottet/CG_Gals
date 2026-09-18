# Revision round gary-r2 (2026-09-17) — readability review by G. Mamon

Starting point: tag `pre-gary-r2` (= 71011cd). Nothing was tuned to reproduce
a previous number; every manuscript value is rendered from `output/results.json`,
`output/paper/additions_macros.tex`, `referee/values/*.json`, or the new
read-only diagnostics file `results/diagnostics/gary_r2/diagnostics_gary_r2.json`
(exposed to the template as `diag`, mirrored into `results.json` under
`diagnostics_gary_r2` on every pipeline run).

## Phase 1 — read-only diagnostics (`analysis/gary_r2/d*.py`, `results/diagnostics/gary_r2/`)

| Item | Finding |
|---|---|
| D1 missing sSFR | 332/394 missing rows are BOSS-instrument spectra outside the MPA-JHU coverage, 60 are legacy `sfr_tot_p50 = -9999` sentinels, 2 have no spectrum id; the missing rows have *higher* S/N than the classified ones. |
| D2 SFMS sign | Raw median offsets (+0.05 dex, all SF galaxies) and the matched mean over 39 both-SF pairs (−0.15 dex) are different statistics of different subsets, plus a strong mass dependence of the residuals about the field-fitted relation. Not a bug. |
| D3 Zheng–Shen classes | **The inherited `Class` labels had Embedded/Predom swapped** relative to Zheng & Shen (2021, Eq. 1) and Paper I Sect. 3.3 (all 56 non-isolated groups). Repaired (below). |
| D4 DS18 → GZ1 | TType ≤ 0 & P_S0 > 0.5: 70 % E / 19 % S / 6 % U / 5 % NoGZ (63/25/8/5 % for P_S0 ≥ 0.8); f_E drops from 0.77 (face-on) to 0.47 (inclined). Pooled only. |
| D5 σ_v match | With/without-σ_v Control4B group matches share 7/54 control groups; without σ_v the controls are dynamically cooler and less balanced in z. With-σ_v stays primary. |
| D6 Holm | Table 4 permutation p: 0.047 / 0.080 / 0.053 (C4B / C4C / RG4). |
| D7 Fig. 2 statistic | Medians of the bimodal vote fractions exaggerated the gap; fractions and means agree. |

## Data repair

* `data/CG4_Groups.csv` and `data/processed_sample.pkl`: `Class` labels
  `Embedded` ↔ `Predom` swapped back to the published definition
  (`analysis/gary_r2/fix_zheng_shen_labels.py`; original kept in
  `data/attic/CG4_Groups_paper1_export.csv`; regression test
  `tests/test_zheng_shen_classes.py`). Correct counts: 6 Isolated, 19 Embedded
  (median host log M200c 13.83), 37 Predominant (13.13). Paper I's Sect. 3.3
  counts and Table 5 class rows are affected (author decision); the isolated-group
  results are not.

## Analyses and figures

* Fig. 2 top row: fraction classified E / S among usable classifications
  (group-blocked 16–84 % intervals) with mean debiased votes as open markers
  (`src/descriptive_trends.py`).
* Fig. 4 (ex-D.1): two panels; new quenched-sequence fit (order selected by
  5-fold CV RMS on the reference quenched galaxies) with per-control median
  offsets (`descriptive_mass_trends.quenched_sequence`).
* `src/spectral_indices.py`: MPA-JHU `galSpecIndx` D_n4000 / HδA (+ errors) by
  stored DR12 specObjID, cached in `data/galspecindx_dr12.csv`, sentinels → NaN;
  availability row in Fig. F.2. Post-starburst classification stays off
  (`config.POST_STARBURST_CLASSIFICATION = False`).
* New figure `fig_property_trends.pdf` (`src/descriptive_properties.py`):
  strong-Hα fraction, AGN-like fraction, median D_n4000, median log R_chl,r vs M*.
* New figure `fig_cg4_classes.pdf` (`src/paper_additions.py`): satellite f_E by
  class with Wilson intervals and control bands.
* New appendix battery (`src/second_order_battery.py`): E-vs-S satellite
  logistic vs log R_ij,med, σ_v, f_L,BGG, log t_cross, f_L,BGG > median; BH-FDR
  across 20 tests; dominance tables also with a median f_L,BGG split
  (`output/second_order_battery.csv`, `domination_distribution_tests_median_split.csv`).
* New Fig. 1 schematic (`src/schematic_figure.py`): embedded CG4 204 in Lim host
  1117 with the would-be Control4B/4C quartets, and RG4 group 10914
  (`results.json['schematic_figure']`). Rebuilt 2026-09-18 (RG4 panel dropped
  earlier): the in-panel inset hid three host members, one of them a CG4 member
  that was also outside the inset window, so only three of the four CG4 members
  were visible. Now two hosts on a common scale plus a zoom column — (a) Lim
  1117 / CG4 204, the host-core case (both quartets excluded); (b) Lim 1289 /
  CG4 330, an off-centre embedded group whose host Control4C quartet contains
  no CG4 galaxy and is *retained* while its Control4B quartet (which holds the
  CG4's brightest member) is excluded — chosen among the 3 such hosts as the one
  closest in richness to (a). All members drawn as luminosity-scaled discs;
  in-panel exclusion status; caption and Appendix C numbers rendered from
  `schematic_figure.off_centre_embedded` (5 off-centre embedded systems: 3
  hosts keep Control4C, 2 also contain a core-forming CG4, all 5 would-be
  Control4B quartets contain the CG4 BGG). `python src/schematic_figure.py`
  regenerates the figure and the JSON entry; `tests/test_schematic_figure.py`.
* Holm-adjusted values stored next to the per-control permutation and adjusted
  odds-ratio p-values (`p_permutation_holm_across_controls`,
  `cg4_p_holm_across_controls`); Table 4 shows the Holm column.
* Forest plots: no minor log-tick labels; all figure legends use the manuscript
  labels (`labels_utils.sample_tex_label`) and elliptical/spiral; extended
  figures drawn under matplotlib's default style.
* `host_controlled`: bookkeeping of hosts containing two CG4s (56 systems → 54 hosts).

## Manuscript

Restructured: Sect. 2 (Samples with Fig. 1 and Table 1 / Classifications incl.
what the E class contains / Sizes / Statistical approach in two paragraphs),
Sect. 3 (Morphology / Star formation at fixed morphology incl. Kitagawa and
Fig. 4 / Compact-group classes / Local projected density / Other properties /
Robustness), Sect. 4 retitled "Interpretation and comparison with previous work".
New appendices: D (statistical conventions), G (second-order battery), I
(robustness details). Specific fixes: abstract ≤ 300 words without citations and
with the Control4C matched value; "Control4C is higher still" corrected; SFMS
"largest offset" removed and the sign difference explained; gap-correlation
screen states its sample; colour-audit sentence moved to the colour appendix
(p to 3 decimals); Fig. 4 caption without the normalisation sentence;
Zeraatgari citation replaced by Brinchmann+04 / Kauffmann+03b; aperture
corrections cite Brinchmann+04 and Salim+07; S-PLUS citations added;
56 → 54 hosts explained; Dn4000 sentence updated; metallicity deferral
justified; file paths moved to Data availability.
Style rules checked by `analysis/gary_r2/style_checks.py`; count bookkeeping and
references by `analysis/gary_r2/phase4_checks.py`. Main-text words (texcount "words in text",
\maketitle to Data availability): 7277 → 4706 (−35 %; cap +5 %); captions 423 → 689;
compiled PDF 21 pages (main text + references 11, appendices 10 — the appendix
block grew by two pages because the secondary diagnostics moved there).

## Proposed commits (not made autonomously)

1. `gary-r2: read-only diagnostics D1–D7` — `analysis/gary_r2/{common,d1…d7,style_checks,phase4_checks}.py`, `results/diagnostics/gary_r2/`.
2. `gary-r2: Zheng–Shen label repair, new analyses and figures` — data repair (+attic, test), `src/{descriptive_trends,descriptive_properties,schematic_figure,second_order_battery,spectral_indices,extended_data,extended_specialness,selection_diagnostics,recent_quenching,config,main,matched_controls,primary_contrasts,host_controlled,sSFR,paper_additions,generate_report}.py`, `data/galspecindx_dr12.csv`, regenerated `output/` JSON/CSV/figures.
3. `gary-r2: manuscript restructure` — `src/paper_template/paper_template.tex`, rendered `output/paper/`, `audit/consistency_gate.py`, `tests/test_size_render_smoke.py`, `tests/test_submission_qc.py`.
4. `gary-r2: figure notation fixes and docs` — `src/utils/labels_utils.py`, `src/{specialness_models,size_analysis,exploration_coulours,phase_space_segregation}.py`, `README.md`, `CHANGES.md`.

# Statistical refactor — branch `refactor/statistical-audit`

This branch repairs the sample-construction and inference defects identified
by the external statistical audit, implements the authors' new sSFR-handling
decisions, and regenerates the entire paper. The audit trail lives in
`audit/FINDINGS.md` (curated verification record), `audit/verify_findings.py`
(re-runnable defect checks), `audit/consistency_gate.py` (manuscript gate)
and `OPEN_QUESTIONS.md`. Nothing was tuned to reproduce a previous number.
(The notes of the earlier presentation-only `presubmission-polish` pass are
in the git history of this file.)

## What was fixed

1. **Control4C regenerated** from the committed parent catalogue
   (`src/sample_construction.py`): the committed file was an older lineage
   containing 14 CG4 galaxies, a duplicated row, and 75 groups absent from
   the parent sample. New: 705 groups / 2,820 galaxies, zero CG4
   contamination (704 / 2,816 after the documented Lim-3688 removal).
   Group-level quartet properties reproduce the committed Control4B/RG4
   builders to <1e-6 (lMass_200/r_200 to <3e-3 dex). Paper I's "61 excluded"
   is not reproducible from the committed parent file (we get 60); see
   `OPEN_QUESTIONS.md` #1.
2. **Missing sSFR is missing data.** The −9999 sentinel / NaN never forms a
   class: unmeasured galaxies are excluded from the GMM, from every sSFR
   figure and from all fractions (denominators are classified galaxies) and
   are reported as counts. The measured low-sSFR class is renamed
   **Quenched** (two-class scheme); every `Passive` key, label and figure was
   renamed in lockstep.
3. **Physical-group inference.** The same Lim group appearing under several
   control labels is one cluster, not three; all cluster-robust SEs and all
   resampling now use the physical group key (CG4 systems cluster with their
   host Lim group). Pooled analyses deduplicate the overlapping controls to
   one row per physical galaxy and are demoted to secondary summaries.
4. **Matching rebuilt.** Control pool deduplicated by objid before matching;
   hard constraints enforced and unit-tested (no CG4 objid among controls —
   this kills the 5 self-pairs of the old run — and no control reused —
   kills the 5 duplicates); per-galaxy provenance table released
   (`output/matched_control_provenance.csv`). A group-level matched contrast
   (smooth-satellite counts per group) is the new primary matched estimand.
5. **Valid p-values.** All bootstrap p-values use the add-one rule
   `p=(k+1)/(B+1)` with B reported (B = 9,999), resampling blocked by group;
   Holm can only increase p; displayed p-values are floored at 1e-4. Every
   `p<10^{-6}` claim is gone.
6. New robustness: Galaxy Zoo threshold sweep + continuous vote-fraction
   model + Sérsic cross-check; within-host CG-member experiment (config
   toggle `HOST_CONTROLLED_ANALYSIS`); tidal-index section rewritten with an
   explicit estimand paragraph (conditional attenuation, not mediation).

## Headline numbers, before → after

"Before" = committed state at the start of the audit (`main` /
`presubmission-polish`, snapshotted in `baseline/`); "after" = this branch's
full rebuild. Baseline pooled models pseudo-replicated overlapping controls
and clustered on label-scoped ids; baseline matched effects contained
self-pairs and duplicated controls and used the invalid sign-crossing p.

| Quantity | Before (invalid where noted) | After |
|---|---|---|
| Control4C sample | 752 groups / 3,008 rows, 14 CG4 galaxies, 1 duplicated row | 705 / 2,820, clean (704 / 2,816 after 3688) |
| Raw smooth/elliptical fraction | CG4 50.0 %, C4B 39.6 %, C4C 40.4 %, RG4 26.8 % | CG4 50.0 %, C4B 39.6 %, C4C 41.1 %, RG4 26.8 % |
| Pooled adjusted elliptical OR (all) | 1.62 [1.21, 2.16], Holm p = 0.007 (pseudo-replicated) | 1.50 [1.13, 1.99], Holm p = 0.032 (dedup, secondary) |
| Per-control elliptical OR (all) | — (not fitted) | vs C4B 2.14 [1.58, 2.90], Holm p = 5.7e-6; vs C4C 1.30 [0.98, 1.73], n.s.; vs RG4 2.81 [1.79, 4.38], Holm p = 4.8e-5 |
| Per-control quenched OR (all) | — (not fitted) | vs C4B 1.80 [1.26, 2.58], Holm p = 0.0025; vs C4C 0.91, n.s.; vs RG4 1.40, n.s. |
| Matched elliptical-fraction difference | +0.197 [0.090, 0.303], "Holm p < 1e-6" (literal p = 0; self-pairs; duplicated controls) | +0.087 [−0.005, 0.179], Holm p = 0.36 (group-blocked, B = 9,999) — **does not survive** |
| Matched quenched-fraction difference | +0.098 [0.017, 0.175], Holm p = 0.054 ("passive") | −0.022 [−0.100, 0.055], Holm p = 1 — **gone** |
| Group-level matched smooth-satellite fraction | — (not fitted) | Δ = +0.160 [0.019, 0.299], p = 0.027 (54 group pairs) |
| Matched-control composition | {C4B 164, C4C 70, RG4 0} with 5 self-pairs, 5 duplicates, 17 hidden RG4 | {C4B 141, C4C 77, RG4 16}, all unique physical galaxies, 16 physically RG4 (declared) |
| Quenched / star-forming fractions | three-class with sentinel "quenched": CG4 7.3 % / 58.9 % / 33.9 % of all | two-class among classified: CG4 63.5 % / 36.5 % (18 unclassified); RG4 46.1 % / 53.9 % |
| Tidal-index attenuation (elliptical OR) | 1.65 → 1.15 ("reframes the signal") | 1.64 → 1.13, described as conditional attenuation with explicit estimands |
| GZ threshold sweep | — | OR 1.46–1.98 over thresholds 0.4–0.8, stable; continuous p_E model p = 0.0014; Sérsic OR 1.42 |
| Within-host CG-member test | — | elliptical OR 1.36 [0.92, 2.00], Holm p = 0.25; quenched OR 0.75, n.s. |

## Which conclusions survive, weaken, or disappear

**Survives (and sharpens).** The compact-group morphology excess survives
the corrected analysis where it is genuinely testable: strongly against the
luminous cores of richer groups (Control4B) and against true four-member
groups (RG4), at the group level in the matched comparison, across Galaxy
Zoo thresholds, in a threshold-free vote-fraction model, and in a
Sérsic-index cross-check.

**Weakens.** The galaxy-level matched morphology contrast — previously the
headline "p < 1e-6" result — is directionally consistent but no longer
significant (Holm p = 0.36) once self-pairs and duplicated controls are
removed and the four galaxies of a group are resampled together. The pooled
adjusted ORs shrink (1.62 → 1.50) after deduplication and physical
clustering. Crucially, the excess is absent against the projected-core
control (Control4C) and absent within shared host groups: the signal looks
like "dense projected cores of groups", not "compact-group membership".

**Disappears.** The matched quenched/star-forming-fraction difference
(previously borderline, Holm p = 0.054) is gone (−0.02, Holm p = 1). The
adjusted star-formation contrast survives only against Control4B. The old
sentinel-based "quenched" class, which silently converted missing
measurements into a physical population, ceases to exist.

## Candidate titles (for the authors to decide — not changed unilaterally)

1. *Are galaxies in compact groups special? Morphology in dense group cores
   rather than compact-group membership*
2. *Galaxies in compact groups of four: a morphology excess concentrated in
   the densest projected configurations*
3. *Are galaxies in compact groups special? A control-matched re-analysis of
   morphology and star formation*

## Verification record

- `audit/verify_findings.py`: every audited defect fails to reproduce
  (remaining "defects" in `audit/FINDINGS_raw.md`: none; F8/README fixed in
  Phase 5, D6-tex purged in Phase 4).
- `audit/consistency_gate.py`: passes (no stale vocabulary; counts match the
  data files; headline numbers trace to `results.json`).
- `pytest`: 106 tests pass, including identity invariants, sample
  post-conditions, matching hard constraints, p-value floors, and render
  smoke tests.
- Determinism: all stochastic steps seeded (documented in code); the paper
  builds with 0 undefined references/citations.
