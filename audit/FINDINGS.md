# Audit findings — independent verification record (Phase 0)

Verified on branch `refactor/statistical-audit` with `audit/verify_findings.py`
(run it with `--write-md` to regenerate the raw table; this file is the curated
record with reconciliation notes). Status is relative to the *defect*: a
"present" defect must fail to reproduce after the refactor.

| Finding | Status at Phase 0 | Detail |
|---|---|---|
| A1 (C4B lineage) | verified OK | Control4B_Gals: 699 groups / 2,796 rows / 0 CG4 objids — matches Paper I (66 excluded from 765). |
| A1 (RG4 lineage) | verified OK | RG4_Gals: 56 groups / 224 rows / 0 CG4 objids — matches Paper I (6 excluded from 62 four-member PC groups). |
| A1 (C4C lineage) | DEFECT PRESENT | Control4C_Gals: 752 groups / 3,008 rows, 14 CG4 galaxies in Lim groups {1685, 1741, 3859, 5240} = CG4 groups {206, 309, 315, 323}, all class Embedded. |
| A1 (C4C rebuild) | reconciled | Rebuilding C4C from PC_Gals (rank_dist ≤ 4) and excluding groups containing any **full-sample** CG4 galaxy (312 objids, split CGs included — this convention exactly reproduces Paper I's 66→699 for C4B and 6→56 for RG4) yields 60 contaminated → **705** clean groups vs Paper I's 61 → 704. See OPEN_QUESTIONS.md #1. |
| A1 (composition drift) | DEFECT PRESENT | 75 committed-C4C groups absent from committed PC; 88 PC groups absent from committed C4C; only 200/224 RG4 galaxies in committed C4C. Committed C4C derives from an older parent-catalogue revision. |
| A1+ (duplicate row, found in Phase 1) | DEFECT PRESENT | Committed Control4C contains an exact duplicate row (objid 1237657591393157322, Lim group 3103, rank_dist 3 twice), so that quartet has only 3 unique galaxies. Not in the external audit; fixed by the Phase 2 regeneration. |
| A2 (old CSV malformed) | DEFECT PRESENT | Control4C_Gals_old.csv data columns shifted left by one: 100 % of its `objid` values are `specobjid` values. |
| A3 (control overlap) | verified (info) | After the Lim-3688 removal: C4B∩C4C = 1,536 objids; RG4⊂C4B = 224/224; RG4∩C4C = 200; pooled control table 6,020 rows / 4,259 unique galaxies (audit numbers reproduce exactly). The overlap is astrophysical; the defect is pooling duplicate rows as independent (B4). |
| B4 (pseudoreplication) | DEFECT PRESENT | `extended_data.py` builds `group_uid = label + ":" + Group`; `extended_stats.fit_logistic_model` clusters on it by default. Same physical Lim group counted as up to three "independent" clusters. |
| C5 (matching) | DEFECT PRESENT | Saved matched analysis (234 pairs, {C4B: 164, C4C: 70, RG4: 0}) reproduced exactly by reimplementation. Diagnostics: 17 matched controls are physically RG4 galaxies (entering via C4B duplicate rows); 5 duplicate control objids; **5 self-pairs** (CG4 treated galaxy matched to its own duplicate row in C4C). The non-split and Lim-3688 filters live in `src/main.py` (`load_data_build` → `dl.remove_split_CG`; `clean()`), i.e. upstream of the pickled sample — not in `extended_data.py`/`matched_controls.py`. |
| D6 (bootstrap p-values) | DEFECT PRESENT | `bootstrap_difference` resamples pairs with B = 2,000 and `p = 2·min(frac≤0, frac≥0)` → literal p = 0 for elliptical/spiral matched fractions; Holm-adjusted values equal to 0 as well; 15 occurrences of `p<10^{-6}` in the rendered paper. |
| E7 (sSFR sentinel) | DEFECT PRESENT | Sample CSVs contain NaN (plus a handful of ≤ −9000 sentinels in raw files); `data_loader.sSFR_floor` fabricates status "Quenched" from the sentinel; `sSFR.sSFR_status` defaults NaN-sSFR rows to "Quenched"; `flattens_quenched` maps −9999 → −15 for display; `common.good_sfr` contains a no-op `np.where` reassignment loop. Missingness (BGG; satellites): CG4 5/62; 9/186 · C4B 48/698; 103/2094 · C4C 55/751; 110/2253 · RG4 3/56; 2/168 — matches the audit exactly. Template states the −9999→quenched rule in 2 places. |
| F8 (README) | DEFECT PRESENT | README.md describes "TRICOTTET-GAM-CG2" with a wrong structure. |
| F9 (tidal wording) | partially present | OR attenuation confirmed: elliptical 1.649 → 1.150 with the projected tidal-index proxy. The rendered text does not use "explains/accounts for" near "tidal" (the polish branch already softened it), but the estimand framing (total association vs association at fixed projected local density) is still missing — Phase 5 rewrites it. |

## Baseline reproducibility

- Committed `output/results.json` reproduced by a full pipeline run
  (`audit/run_full_pipeline.py`, cached processed sample) except **19 keys**,
  all `*_median_err` / `*_median_ci_*` values from an **unseeded**
  `stats_utils.bootstrap_median_error` — a determinism defect fixed in
  Phase 4.
- Snapshots: `baseline/output_committed/` (pre-run), `baseline/output_rerun/`
  (fresh run), `baseline/processed_sample_committed.pkl` (all git-ignored).
- Lim group 3688: the exclusion **is** documented in the manuscript methods
  (spurious group; one z = 0.048 outlier among z ≈ 0.032–0.034 members
  inflates σ_v to ≈ 1,980 km/s). No open question.

## Phase 2 notes

- `src/sample_construction.py` regenerates Control4C (705 groups / 2,820
  galaxies, zero CG4 contamination, Lim 3688 kept in-file and removed at
  load). Group-level properties reproduce the committed Control4B/RG4
  builders at < 1e-6 relative precision for every column except
  `lMass_200`/`r_200_kpc` (< 3e-3 dex / 0.3 %, legacy solver tolerance).
- The regenerated Control4C quartet for Lim 3688 has Vdisp = 1980.6 km/s,
  matching the manuscript's quoted 1981 km/s.
- `common.py::EqA11` as committed is *not* what produced the committed
  `lMass_200` columns: the committed values match the correct NFW
  M_180m -> M_200c conversion (`M_tilde(f1 f2 f3 f4)/M_tilde(f2) * M_200`),
  while the snippet divides by `M_tilde(c)` inside the argument. The new
  implementation uses the correct form.
- Control4C `rank_M` is now the **within-quartet** luminosity rank (1-4,
  same convention as CG4/Control4B/RG4); the parent-group rank is kept as
  `rank_M_parent`. The committed file carried parent ranks (up to ~30),
  which broke exact rank matching in the propensity analysis.
- `Control4C_Gals_old.csv` retired to `data/attic/` with a README.

## Phase 3 notes

- Missing sSFR is now missing data end-to-end: `sanitize_ssfr` converts
  sentinels/unphysical values to NaN at load; the GMM is fitted on measured
  SDSS galaxies only; unmeasured galaxies carry the internal `NosSFR` token,
  are excluded from every sSFR figure and fraction, and are reported as
  counts (results key family `*_NNosSFR*`, `sSFR_missingness`).
- Classification requires both a valid sSFR and a valid stellar mass, so the
  unclassified counts slightly exceed the raw sSFR-NaN counts (CG4: 18 vs
  14; the extra 4 have sSFR outside [-25, -5] and no valid mass).
- The measured low-sSFR class is named **Quenched** (config
  `sSFR_status = ['Quenched', 'Starforming']`); every `*Passive*` results
  key, figure label, column and template placeholder was renamed in
  lockstep (`quenched` in the extended pipeline).
- `flattens_quenched`, `sSFR_floor` and the -15 display floor are gone; the
  sentinel-based "quenched" class ceases to exist.

## Conventions established

- "CG4 galaxy" for control-exclusion purposes = any galaxy of the **full**
  CG4 sample (312 objids, split groups included), matching Paper I's counts.
- The Control4C rebuild selects `rank_dist ≤ 4` per PC group (BGG has
  rank_dist = 1).
