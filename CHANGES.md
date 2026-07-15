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
