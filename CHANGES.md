# Pre-submission polish — branch `presubmission-polish`

All edits were made to the authoring sources — the Jinja2 template
`src/paper_template/paper_template.tex` (delimiters `<% %>` / `<< >>`), the
hand-maintained bibliography `output/paper/paper.bib`, and the plotting code
under `src/` — and the paper was regenerated with the repository's own
pipeline (`python -m src.main` with `RENDER_PAPER_ONLY = True`). No analysis
was re-run and no results JSON was modified: `output/results.json` and
`output/results_build.json` are byte-identical to their state on `main`.

Baseline build (before any edit): 30 pages, 0 BibTeX warnings, 0 undefined
references/citations. Final build: 34 pages (the growth comes from the
acknowledgements block and from rendering the two appendix domination figures
at full text width for legibility), 0 BibTeX warnings, 0 undefined
references/citations, abstract exactly 300 words with no `\cite`.

## Commits

| Commit | Scope |
|---|---|
| `Polish A: …` | Task group A (scientific presentation) |
| `Polish B: …` | Task group B (references and citations) |
| `Polish C: …` | Task group C (A&A compliance) |
| `Polish D: …` | This file + final verification build |
| `appendix-trim: …` | Task C7, isolated for co-author review (revert wholesale with `git revert`) |

## Figure map (paper figure → producing code)

Main-text figures:

| Fig. | File | Producer |
|---|---|---|
| 1 | `sSFR_classification.pdf` | `src/sSFR.py::plot_classification` (via `src/main.py`) |
| 2 | `galaxies_sfr.pdf` | `src/sSFR.py::plot_galaxies` (via `src/main.py`) |
| 3 | `main_sequence_residuals.pdf` | `src/sSFR.py::plot_main_sequence_residuals` |
| 4 | `colour_robustness_coefficients.pdf` | `src/exploration_coulours.py` |
| 5 | `fig_specialness_logistic_coefficients.pdf` | `src/specialness_models.py` |
| 6 | `fig_morphology_crowding_robustness.pdf` | `src/morphology_robustness.py` |
| 7 | `fig_matched_control_effects.pdf` | `src/matched_controls.py` |
| 8 | `fig_matched_control_balance.pdf` | `src/matched_controls.py` |
| 9, 10 | `phase_space_satellite_{passive,earlytype}_fraction_by_distance.pdf` | `src/phase_space_segregation.py` |
| 11 | `fig_tidal_index_outcomes.pdf` | `src/tidal_indices.py` |
| B.1 | `BPT_diagram.pdf` | `src/data_loader.py::plot_bpt` |

Appendix C figures:

| Figs. | Files | Producer |
|---|---|---|
| C.1 | `residual_sSFR_histogram.pdf` | `src/sSFR.py::plot_residual_distribution` |
| C.2–C.4 | `ssfr_class.pdf`, `BGG_ssfr_class.pdf`, `Satellites_by_BGG_ssfr_class.pdf` | `src/sSFR.py::split_by_*fertility` |
| C.5 | `Main_Sequence_polyfits.pdf` | `src/sSFR.py::plot_main_sequence_models` |
| C.6, C.7 | `dist2BGG_{kpc,norm}_vs_sSFR.pdf` | `src/exploration_ssfr.py` |
| C.8 | `Dom_vs_NonDom.pdf` | `src/exploration_dom.py::plot_domination_distributions` |
| C.9 | `Dom_vs_NonDom_XOR_halfviolin_ONEFIG.pdf` | `src/exploration_dom.py::plot_xor_corrpairs_halfviolins_onefig` |
| C.10–C.12 | `tcr_vs_mvirial_over_l.pdf`, `tcr_vs_mvirial.pdf`, `tcr_mvirial_lgroup_3d.pdf` | `src/exploration_mvt_tcross.py` |
| C.13–C.17 | `colour_mass_relations.pdf`, `satellite_colour_mass_environment.pdf`, `colour_residual_distance.pdf`, `bgg_satellite_colours.pdf`, `bgg_colour_domination.pdf` | `src/exploration_coulours.py` |
| C.18–C.23 | `size_availability.pdf`, `size_mass_relation.pdf`, `size_forest_per_control.pdf`, `size_measure_delta.pdf`, `size_re_n_plane.pdf`, `size_radial.pdf` | `src/size_analysis.py` |
| C.24 | `phase_space_satellite_passive_fraction_projected_phase_space.pdf` | `src/phase_space_segregation.py` |
| C.25, C.26 | `fig_magnitude_gap_comparison.pdf`, `fig_magnitude_gap_vs_passive_fraction.pdf` | `src/fossilness.py` |
| C.27 | `fig_recent_quenching_diagnostics.pdf` | `src/recent_quenching.py` |
| C.28 | `fig_agn_fraction_by_sample.pdf` | `src/agn_environment.py` |
| C.29, C.30 | `fig_data_availability_by_sample.pdf`, `fig_colour_matched_selection_bias.pdf` | `src/selection_diagnostics.py` |

## Task group A — scientific presentation

- **A1 (`Quenched` → `No sSFR`)** — Table 1 header hard-coded to `No sSFR`
  in the template; Sect. 2.2 now defines the class with the crowding
  explanation (the 7 %, 10 %, 1 % fractions are rendered from the results
  JSON, not hard-coded) and cross-references Sect. 6.4 via
  `\ref{sec:decoupling}`; Fig. 1 and Fig. 2 captions reworded; Appendix B
  paragraph renamed; the Fig. 2 legend renaming is done upstream in
  `src/utils/labels_utils.py::display_label` (`"Quenched" → "No sSFR"`).
  Figs. C.2–C.4 verified to plot only Passive/Star-forming (indices 1, 2 of
  `co.sSFR_status`) — no change needed. The unused `\Quenched` macro was
  removed. Remaining "quenched" occurrences are the allowed Sect. 5.6
  contexts only (`grep -n uenched output/paper/paper.tex`).
- **A2 (move Sect. 2.3 → 3.1)** — "Raw sSFR fractions" and Table 1 moved to
  the head of Sect. 3; Fig. 1 stays with the sSFR-classification subsection.
  Cross-references are `\label`/`\ref` based; the one hard-coded subsection
  reference found ("Sect.~5.2" in the discussion) now uses
  `\ref{sec:morph_crowding}`.
- **A3 (main-sequence residual statistics)** — the test behind the quoted
  p-values is `stats_utils.bootstrap_median_difference`: a **two-sided
  bootstrap sign-crossing probability for the median difference** (10 000
  resamples), and the quoted interval is the **central 68 % (16th–84th
  percentile) bootstrap interval**. The paragraph before the bullets now
  states the Δ sign convention (CG4 − control), the interval level, and the
  test name; the apparent Control4B inconsistency ([0.027, 0.098] excluding
  zero with p = 0.056) is resolved by making the 68 % level explicit — a
  68 % interval excluding zero is compatible with a two-sided p slightly
  above 0.05. All numbers unchanged.
- **A4 (AGN wording)** — `src/agn_environment.py` uses *both* demarcations:
  composite = between Kauffmann et al. (2003) and Kewley et al. (2001),
  AGN-like = above the Kewley line (a pre-existing AGN flag fills
  otherwise-unclassified spectra). Sect. 5.7 now states both demarcations
  with citations (`Kauffmann+03a`, `Kewley+01`); "broad AGN-like" is gone
  everywhere; Appendix B keeps its own (deliberately more inclusive,
  Kauffmann-based) calibration flag and now says so explicitly, pointing to
  Sect. 5.7 (`\ref{sec:agn_env}`).
- **A5 (Sect. 4.3)** — the two control cases (Control4B and Control4C
  dominated groups) are now a proper `itemize` list, followed by the
  identical-odds-ratio-by-construction clause.
- **A6 (SDSS selection)** — definition added at the end of Sect. 2.1 with
  the redshift/magnitude limits rendered from the JSON (`Z_MIN`, `Z_MAX`,
  `R_MAX`), matching the Appendix A query (which also requires
  `lgm_tot_p50 > -1000` and the `zooSpec` join); Table 1 row label and the
  Fig. 1 caption fixed.
- **A7 (XOR criterion)** — Fig. C.9 caption now defines it per the
  implemented logic in `plot_xor_corrpairs_halfviolins_onefig`: pairs whose
  Spearman correlation, after per-subsample Benjamini–Hochberg adjustment
  across pairs, is significant (p_adj < 0.10) in exactly one of the
  dominated/non-dominated subsamples of at least one sample.
- **A8 (cosmology)** — Planck 2015 (Planck Collaboration XIII 2016; existing
  key `Planck15`) stated once at the end of Sect. 2.1; the sizes subsection
  points back with `\ref{sec:samples}`.
- **A9 (caveats)** — halo-mass caveat added in Sect. 6.5 (pointing to
  `\ref{sec:selection_diagnostics}`); the matched-balance covariate is
  unambiguous from `results_build.json` (`balance.after`:
  log_group_luminosity = −0.144, all others ≤ 0.03), so the Sect. 5.3
  sentence names \(\log L_\mathrm{group}\).
- **A10** — Fig. C.29 caption now uses `\ref{sec:optical_colours}` (was
  "Sect. 4.5"); tidal-index units (M⊙ kpc⁻³) added in Sect. 5.8 and on the
  Fig. 11 x-axis; "SDSS DR 16" → "SDSS DR16" (template spacing around
  `<<DATA_RELEASE>>`).

## Task group B — references and citations

- Already present under different keys (cited, not duplicated):
  `McConnachie+09` (MNRAS 395, 255), `Planck15` (A&A 594, A13), `Kewley+01`
  (ApJ 556, 121), `DiazGimenez+20` (MNRAS 492, 2588), `Taverna+23`
  (MNRAS 520, 6367).
- Added (verified against publisher DOI metadata via doi.org content
  negotiation): `Ahumada+20` (ApJS 249, 3), `Lintott+08` (MNRAS 389, 1179),
  `MendesdeOliveira&Hickson94` (ApJ 427, 684).
- Montaguth et al. 2026a = `Montaguth2026MassSize` (ApJ 999, 160, DOI
  10.3847/1538-4357/ae42ca) and 2026b = `Montaguth2026NonIsolated`
  (ApJ 998, 91, DOI 10.3847/1538-4357/ae2bdd) — both verified against DOI
  metadata. `McLachlanPeel2000` gained `address = {New York}`.
- Text: `\citep` → `\citet` for the "cores of richer groups in Tricottet et
  al. (2025)" sentence (no other `in (\citep…)` slips found); DR16, MPA-JHU
  (Brinchmann+04; Kauffmann+03b — the stellar-masses paper, distinct from
  the BPT paper Kauffmann+03a), and Galaxy Zoo (Lintott+08 + the 2011 data
  release, existing key `GalaxyZoo1`) citations added at first mention;
  Sect. 6.2 McConnachie fixed; Introduction gained the Mendes de Oliveira &
  Hickson (1994) sentence and DiazGimenez+20/Taverna+23 in the
  transient-alignment citation list.

## Task group C — A&A compliance

- **C1** — abstract rewritten per the approved draft: exactly 300 words
  (script-stripped count), no citations, acronyms defined (CG4, RG4,
  Control4B, Control4C, BGG, Sloan Digital Sky Survey, AGN). Trims were
  taken from the Aims observable list and small connectives, as suggested.
  Note: the previous abstract's size sentence was data-conditional Jinja;
  it is now static text asserting the current verdicts (no pooled adjusted
  offset; matched-pair preference for smaller sizes is control-dependent).
  A template comment marks this for revisiting if the size analysis changes.
- **C2** — `acknowledgements` environment inserted before the references
  with `TODO(MT)` placeholders (personal thanks, funding, verbatim SDSS-IV
  text). All eight acknowledged packages verified as imported by the
  analysis code; seven new bib entries added (Astropy2022, Ginsburg2019,
  Harris2020, Virtanen2020, Hunter2007, McKinney2010, Seabold2010 — note
  the pre-existing `Hunter14` is an unrelated JCAP paper), Pedregosa2011
  already existed.
- **C3** — "Code availability" merged into the single unnumbered "Data
  availability" section (CDS at publication, GitHub URL, Zenodo DOI TODO).
- **C4** — `aa.cls` in the repo has no ORCID command, so the three author
  ORCIDs are hooked with `\thanks{ORCID: \url{...}}` + `TODO(MT)`
  placeholders.
- **C5** — "per cent" → `\%` (3 spots); "artifact" → "artefact" (3 spots);
  every rendered value that can be negative and sat in text mode is now
  wrapped in `\(...\)` (size-section coefficients and CIs, matched size
  effects, magnitude-gap medians, percentage-point differences, −9999 in
  Appendix B) so minus signs are typeset as math minus. No sentence-start
  "Sect."/"Fig." violations and no "Tab." abbreviations were found. The
  reported Table 8 "checkMorphology" run-in is a PDF text-extraction
  artefact: the source cells are separated by `&` and the current PDF
  renders/wraps them correctly — no change needed.
- **C6** — every appendix C subsection now opens with a prose sentence
  naming its figures (`\ref`-based, robust to renumbering);
  `\usepackage{placeins}` + `\FloatBarrier` at appendix subsection
  boundaries (compatible with aa.cls; build clean). Figure fixes:
  - Fig. C.27: `\AA` literal → `$\mathrm{\AA}$` (renders Å correctly).
  - Fig. 8: y-ticks now `$\log M_\star$, $z$, rank, $\log L_{\rm group}$,
    $\sigma_v$` (mapping in `matched_controls.py`).
  - Fig. 2: legend "Compact Groups galaxies" → "CG$_4$ galaxies" (folded
    into the A1 re-render of the same legend).
  - Fig. C.15: legend moved above the panels (was overlapping points at
    lower left).
  - Figs. C.8/C.9 promoted to full text width (they are 19.9 in and 10.4 in
    wide; at column width their fonts shrank to 2.5–4.4 pt); C.8 fonts
    additionally raised (labels 22 pt, ticks 18 pt, p-annotations 20 pt,
    legend repositioned); C.18 canvas shrunk to 5.2×3.6 in with 10 pt
    annotations; C.29 annotations 7.5 → 9 pt. All re-rendered figures were
    visually inspected at publication scale.
  - Sect. 5.3 propensity-covariate list now uses math symbols.
- **C7** — see the separate `appendix-trim` commit (below).

## Figures re-rendered vs. patched-only

Re-rendered (label/legend/font changes only; all analyses behind them are
deterministic with fixed seeds, and key recomputed statistics were checked
against the published JSON — e.g. tidal n_pairs = 6268, matched
max-SMD-after = 0.1437): `galaxies_sfr.pdf`, `fig_tidal_index_outcomes.pdf`,
`fig_matched_control_balance.pdf` (+ sibling `fig_matched_control_effects.pdf`),
`fig_recent_quenching_diagnostics.pdf`, `fig_data_availability_by_sample.pdf`
(+ sibling `fig_colour_matched_selection_bias.pdf`), `size_availability.pdf`
(+ sibling size figures), `Dom_vs_NonDom.pdf`,
`Dom_vs_NonDom_XOR_halfviolin_ONEFIG.pdf`, `colour_residual_distance.pdf`.
The re-rendering used a plotting-only driver against the frozen
`data/processed_sample.pkl`; `exploration_dom.run` / `exploration_coulours.run`
were deliberately *not* called because they append to the results JSONs.
Sibling figures re-rendered by shared entry points are content-identical.

Patched-only: none — every figure whose code was touched was re-rendered.

## Digit-touching diff review (ground rule 1)

`git diff main...HEAD -- src/paper_template/paper_template.tex` was reviewed
hunk by hunk for digits. Every digit-touching change is one of:
1. minus-sign/math-mode typography around **unchanged** Jinja placeholders;
2. new static explanatory constants that are not results (68 % interval
   level, 10 000 bootstrap resamples, the conventional 0.1 SMD threshold,
   the p_adj < 0.10 XOR selection level — all read from the code);
3. pure relabeling ("No sSFR", "SDSS selection", "CG$_4$ galaxies");
4. previously hard-coded manuscript numbers replaced by template variables
   (the abstract's group count now renders from `CG4_Groups_nonsplit_N`;
   the Sect. 2.2 no-sSFR fractions render from the JSON counts).
No table value, p-value, CI, count, or coefficient changed.
`output/results.json` and `output/results_build.json` are byte-identical to
`main`.

## Verification (task group D)

- Clean rebuild via `python -m src.main` (render-only): pdflatex ×3 + bibtex
  exit 0; `paper.log` has **zero** undefined references, undefined
  citations, or multiply-defined labels (only two pre-existing font-shape
  warnings from the `ae` package); `paper.blg` has zero warnings.
- Abstract: 300 words (LaTeX-stripped count), contains no `\cite`.
- Grep gates — all return no hits in the template and the rendered
  manuscript: `per cent`, `artifact`, `DR 16`, `select'n`, bare
  `Zoo morphologies`, `broad AGN`, `logMstar` in prose,
  `(Tricottet et al. 2025), show`, `Code availability`.
- `pytest`: **77 passed** (14 warnings, all pre-existing deprecations).

## Open TODOs for the authors

1. **SDSS acknowledgement** — paste the official SDSS-IV/DR16 funding text
   verbatim from https://www.sdss4.org/collaboration/citing-sdss/ into the
   acknowledgements (placeholder comment in the template).
2. **Funding grant numbers** — MT, GAM, and EDG (CONICET / SECyT–UNC grant
   numbers) in the acknowledgements.
3. **ORCID iDs** — replace the three `https://orcid.org/TODO-*` placeholders
   in the author block; the corresponding author's iD is required at
   submission.
4. **Zenodo DOI** — mint the archival DOI for the code release and replace
   the TODO in the Data availability section.
5. **ADS check** — Montaguth et al. (2026b) and the two new co-author-lineage
   entries (`DiazGimenez+20`, `Taverna+23`) were verified against publisher
   DOI metadata only; please confirm the exact ADS author lists before
   submission (no ADS API token was available).
6. **Appendix trim (C7)** — the separate `appendix-trim` commit drops
   C.3, C.11, C.12, C.16, and C.17 and needs explicit co-author approval;
   revert it wholesale if declined.
7. **A&A page-charge policy** — verify the current pricing (main-text /
   appendix page thresholds) on aanda.org; the 12+8 figure quoted during
   this polish was not independently verified.
8. **Abstract size sentence** — now static; if the size analysis or its
   verdicts change, the Results sentence on galaxy sizes must be updated by
   hand (guard comment in the template).
