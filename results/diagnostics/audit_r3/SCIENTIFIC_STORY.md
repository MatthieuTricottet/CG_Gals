# Revision round audit-r3 — internal note on the scientific story (Phase 2)

Internal working note, not part of the paper. Baseline: tag `pre-audit-r3`
(= a3d08b2). Every number below was recomputed from the repository data with
the project's own galaxy frame (`src/extended_data.build_galaxy_frame`), not
copied from `audit_claude_referee/`. Satellite models use the Table 4
covariates (log M*, z, log L_quartet, sigma_v; standardised), cluster-robust
by physical Lim group. Scratch scripts: session scratchpad `p1/`; the
accepted analyses are re-implemented in `src/` (Phase 3).

## Robust findings (keep, lead with them)

* **Satellite E-class excess.** Table 4 reproduces exactly (satellite ORs
  2.63 / 1.93 / 3.03 vs CB / CC / RG). Survives: full-Lim-catalogue crowding
  exclusion (2.14→2.14/1.67→1.71/2.81→2.83 all galaxies), sigma_v SIMEX
  (unchanged), cluster bootstrap, split groups restored, Lim 3688 retained,
  DS18, Sersic, threshold sweep. All-quartet group-level binomial regression
  (cluster-robust): OR 2.14 [1.54, 2.96], 1.80 [1.30, 2.48], 2.72 [1.56, 4.75].
* **The excess is a satellite phenomenon**: BGG ORs 0.78 [0.40, 1.54],
  0.87 [0.44, 1.75], 1.87 [0.63, 5.52].
* **Morphology excess at fixed sSFR class** (satellites, E | Q + covariates):
  1.94 [1.30, 2.90], 1.68 [1.14, 2.47], 2.41 [1.26, 4.63].

## Findings weakened by adjustment

* **Projected BGG-centric radius.** CG4 satellites lie at 4–319 kpc (median
  123) vs medians 381 / 226 / 338 kpc for CB / CC / RG satellites (only 42 /
  65 / 46 % of control satellites inside the CG4 maximum). E-class fraction
  falls with radius in every sample. Satellite ORs at fixed radius: log R
  1.80 / 1.39 / 1.95; restricted cubic spline 1.80 / 1.40 / 1.93;
  log(R/r200) 1.70 / 1.34 / 1.87; common support (R ≤ 319 kpc) + spline
  1.82 / 1.42 / 1.90. Standardised to the CG4 satellites' masses, redshifts,
  quartet luminosities, sigma_v and radii, control models predict
  f_E = 0.43 / 0.48 / 0.42 vs 0.56 observed: residual differences
  +0.13 [0.05, 0.21], +0.08 [0.01, 0.15], +0.14 [0.00, 0.26]
  (group bootstrap), compared with +0.22 / +0.15 / +0.24 without radius.
  Report as two quantities (total contrast; contrast at common projected
  position). NOT "x% explained", NOT an "upper bound" (radius is projected,
  noisy, and partly defined by the compact-group selection).
* **Quenching.** At fixed GZ1 class and continuous stellar mass (satellites):
  OR 1.85 [1.13, 3.02] vs CB, 1.29 [0.83, 2.01] vs CC, 1.39 [0.65, 2.97] vs RG;
  stable to a mass spline and a class×mass interaction. Adding log R:
  1.49 [0.88, 2.53], 1.07, 1.11. All galaxies: 1.56 (p = 0.05), 1.20, 0.98.
  The published morphology-only Kitagawa (conditional ≈ 0) is confounded by
  mass within class (CG4 E satellites median log M* 10.34 vs 10.62 in CB).
  Class×mass Kitagawa (tercile edges arbitrary): CB +0.067 [−0.005, 0.128]
  (audit edges: +0.074 [0.004, 0.140]) — descriptive only.

## Findings that disappear / must be deleted

* "At fixed morphology the quenched fractions do not differ" / "no
  additional quenching at fixed morphology" / "the sSFR-defined quenching
  result is the morphology mix seen through a different lens": delete. True
  only against CC and RG and only without mass; against CB there is an excess
  at fixed class and mass that weakens at fixed radius.
* "Matched quartets ... not significant against CC (Δ = 0.11, p = 0.08)" as a
  headline: the matched estimator uses 54 of ~630 control quartets, its CB
  value moves 0.16→0.10 when the control pool is deduplicated, 0.16→0.28
  without sigma_v, and under re-drawn sigma_v noise the CC p < 0.05 in ~60 %
  of realisations. All-quartet regression is clear against all three. Move
  matching to a documented robustness appendix (penalised sklearn propensity
  C = 1, greedy nearest-first, caliper 0.2 SD logit).
* Within-host "no longer statistically clear once central location is held
  fixed" based on the linear-radius OR 1.47: specification-dependent. Log /
  spline radius give 0.85–1.08; satellites-only 0.93–1.01. Radius alone
  predicts CG membership (AUC 0.87); only 17 % of non-member satellites lie
  inside their host's largest CG-member radius. At fixed radius members and
  co-members have similar E fractions, but the leverage is limited — make
  the poor overlap the interpretation.
* "missing class ... unrelated to the star-formation properties": delete.
  29 % of galaxies with a Lim neighbour < 55" lack sSFR vs 3 % otherwise
  (missing-row model with M_r, not MPA mass which is absent for 85 % of the
  missing rows: crowding OR ≈ 12; BGG excess vanishes at fixed luminosity).
* Crowding test description ("nearest catalogued neighbour"): the flag uses
  quartet co-members only. Rebuild on the full Lim catalogue (22 / 12 / 14 /
  7 % crowded). "CG4 all-row E fraction rises after exclusion" is a
  denominator artefact (NoGZ rows are crowded); E/(E+S) 0.590 → 0.583.

## Genuinely new (exploratory, post hoc)

* **H-alpha.** The strong-Hα deficit at fixed mass and GZ1 class is carried
  entirely by strong emitters with log [N II]/Hα ≥ −0.4 (WHAN "AGN-like"):
  ORs 0.39 / 0.41 / 0.37; SF-like strong emission is not deficient
  (1.25 / 1.57 / 1.64, intervals include 1); weak-line (retired) galaxies are
  correspondingly more common. All W ≥ 3 Å emitters have S/N ≥ 3 in Hα and
  [N II] (galSpecLine errors fetched), so S/N does not drive it. Among strong
  emitters CG4 are 75 % BPT-SF vs 56–61 %; the deficient class is a mix of
  composites, LINER/Seyfert-like and metal-rich SF spectra → do NOT write
  "fewer AGN". SF-like emitters have the same W(Hα) and fibre L(Hα) at fixed
  mass. BGGs show no difference. Label exploratory.

## Exploratory / demote

* Isolated CG4s (6 groups, 14 classified satellites, 4 E): appendix only.
* sSFR classifier: KL-minimised constrained two-component mixture on a
  50×50 binned (log M*, log sSFR) reference; unconstrained EM on the non-AGN
  reference puts both components on the SF sequence. Stored group sSFR
  (sfr_p50 − lgm_p50) exceeds specsfr_p50 by ~0.05 dex. Variants change
  classes for ~2–4 % of galaxies; results stable → one compact sentence.

## Claims to keep but re-word

* "rank" → BGG/satellite indicator (that is what the models use).
* log_group_mass: `M_group` is an absolute magnitude; the covariate was
  silently all-NaN. Remove; make covariate drops loud.
* No temporal ordering between morphological change and quenching.

## Addendum — final pipeline values (supersede the scratch values above)

The numbers above come from the Phase-1 scratch scripts. The paper quotes the
pipeline values (`python reproduce.py`), which differ where the final
implementation differs:

* Crowding flag on the full Lim catalogue at any redshift: 27 / 14 / 16 / 10 %
  crowded (CG4 / CB / CC / RG; quartet co-members only: 19 / 6 / 10 / 5 %).
  After exclusion E/(E+S) 0.590 → 0.577 (CG4), 0.476 → 0.456 (CC).
* Missing sSFR, model with M_r: crowding OR 14 [10, 20]; satellite 1.37,
  CG4 0.72; MPA mass absent for 84 % of the missing rows.
* Fractions at the CG4 satellite covariates incl. log R_BGG: 0.43 / 0.49 /
  0.42 vs 0.56 observed; differences 0.22 / 0.15 / 0.24 → 0.13 / 0.07 / 0.14.
* H-alpha from galSpecLine (W ≥ 3 Å with S/N ≥ 3, galaxies with line data):
  strong emitters 34 % vs 41 / 43 / 56 %; adjusted ORs 0.55 / 0.65 / 0.54;
  high-[N II] 0.38 / 0.40 / 0.37; SF-like 1.25 / 1.57 / 1.59; BPT-SF among
  strong emitters 79 % vs 60 / 67 / 69 %.
* Within hosts: satellites with log R 0.93 [0.55, 1.57], spline 0.85, common
  support 0.74, all members with linear R 1.47; AUC of radius for membership
  0.87; 17 % of co-member satellites inside the members' largest radius.
