# Conservative manuscript revision — 26 October 2026

Starting HEAD: `36e04bd`. Changes were made in the authoritative Jinja2 source, `src/paper_template/paper_template.tex`, and rendered to `output/paper/paper.tex`. The original template reproduced the existing rendered source byte for byte before editing. Unrelated working-tree changes were preserved; no commit was made.

## A. Implemented scientific changes

- **Section 4.1:** Connected the 46 exact projected-core identities among 56 non-isolated systems to the separate, stellar-mass-adjusted satellite compactness associations in Table F.1. The interpretation is continuity with dense group cores, allowing host properties, selection, and different evolutionary pathways. It explicitly avoids assuming common slopes or intercepts or inferring causation. The separation statistic is the **median of the six projected pair separations**, as already defined in Table 1, not their arithmetic mean.
- **Section 4.1:** Added one sentence relating the bright-BGG luminosity selection to high stellar masses and already large quenched fractions, as a possible limit on observable environmental contrast. The selection is not a stellar-mass threshold. Replaced the categorical statement that halo mass “adds nothing” with the observed stability under adjustment and its limited implication.
- **Sections 3.3 and 4.1:** Retained the six-system isolated result, replacing “indistinguishable” for the two non-isolated classes with “do not differ detectably”. No evolutionary age or equivalence is inferred from the isolated comparison or its p = 1.00.
- **Section 4.3:** Distinguished line-to-continuum EW in the 3-arcsec legacy SDSS fibre from estimated total sSFR, including photometric aperture corrections and weak-line methods. Non-star-forming excitation is acknowledged. These are possible measurement contributions to different contrasts, not evidence for central quenching. Section 4.1's fixed physical fibre-size description was replaced with redshift-dependent coverage.
- **Sections 3.5 and 4.3:** Made the unequal available-pair samples explicit: 166 Simard pairs versus 234 Petrosian pairs. Compared the size results with Coenda et al.'s concentration-selected morphological subsamples, including their C > 3 check without a clear environmental trend. Estimator/profile dependence is a possible methodological contribution; modest intrinsic offsets remain allowed. Appendix I's null neighbour-separation correlation no longer excludes a blending systematic.

## B. Technical corrections

- **Section 3.2:** Corrected the Kitagawa reference to Appendix B (`app:ssfr_method`), Table B.3 (`tab:quenching_decomposition`). Referenced Appendix E (`app:baseline_details`) at the actual sequence-fitting discussion, so it remains correctly cited.
- **Section 2.1 footnote:** Restored a qualified historical clarification. Paper I Sect. 3.3 reports 19 predominant and 37 embedded systems, whereas its Table 5 lists 19 embedded and 37 predominant systems. The footnote therefore says “in parts” of Paper I; it does not claim every published table used the opposite labels. The already repaired catalogue labels and all class counts were preserved.
- **Section 3.4:** Foregrounded the existing deduplicated pooled fit: OR 1.27, 95% CI [0.94, 1.71], p = 0.12, N = 3260. Specified the established retained-quartet priority RG4 > Control4B > Control4C, stellar mass and satellite-status covariates, and physical-group clustering. Kept the catalogue-row fit as an explicit secondary check, with its weighting and support diagnostics labelled accordingly. Appendix G.3's host-inclusive check is explicitly identified with that catalogue-row sensitivity.
- **Appendix D:** Corrected the overlap-coefficient method description from kernel densities to normalised histograms on common bins, matching both T4 and T9 code.
- **Preamble/graphics:** Removed duplicate `graphicx` loading and five redundant figure-directory prefixes, using the existing `graphicspath`. Class options, fonts, margins, spacing, tables, macros, and bibliography configuration were preserved.
- **Existing policy test:** Updated only the outdated tidal-pooling assertion in `tests/test_submission_qc.py` to require primary deduplication and a labelled catalogue-row alternative.

Table B.1's CG/control missingness ordering was already correct in both the source and the pre-existing PDF. Its values were checked against the sample; no numeric change was needed. The acknowledgements already name Astropy only once. No unjustified table-spacing adjustment was made.

## C. Validation and suggestions not adopted

### Numerical checks

Direct cached-data refits reproduced:

| Diagnostic | Reproduced estimate |
|---|---|
| Deduplicated tidal fit, without / with index | OR 1.8584 / 1.2672; with-index CI [0.9365, 1.7148], p = 0.124863; N = 3260 in both |
| Catalogue-row tidal fit, with index | OR 1.3931, CI [1.0304, 1.8836], p = 0.031210; N = 5117 |
| Compactness, CG4 / Control4B / Control4C | OR 0.6526 / 0.8001 / 0.7023; published BH-adjusted p = 0.004757 / 0.000122 / 3.09e-10 |
| Strong H-alpha, same control order | OR 0.5486 / 0.6544 / 0.5627; CI [0.3730, 0.8067] / [0.4490, 0.9538] / [0.3010, 1.0521]; p = 0.00228 / 0.02739 / 0.07170 |
| Adjusted satellite Simard size | −0.009964 dex, CI [−0.039426, 0.019497], p = 0.507393; N = 2575 |
| Matched Simard mean | −0.019008 dex, CI [−0.063897, 0.026478], p = 0.400; 166 pairs |
| Matched Petrosian mean | −0.033385 dex, CI [−0.064548, −0.003198], p = 0.0298; 234 pairs |
| Simard-minus-Petrosian, E / S | 0.074418 / 0.041610 dex; N = 1091 / 1453 |

The 5828 control catalogue rows represent 3609 unique galaxies: 1771 occur twice, 224 three times, and 1614 once. Repeated rows agree in mass, morphology, rank, and physical host, but 1093 objects have different quartet tidal indices. Deduplication therefore also fixes which quartet defines the index; host clustering alone cannot correct weighting or assignment. The original no-index catalogue-row fit has 5118 complete cases, one more than its with-index fit. The deduplicated comparison has the same 3260 complete cases with and without the index.

The overlap bookkeeping is 46 exact identities, five partial overlaps, and five zero overlaps. The independently verified mass-adjusted compactness fits use usable E/S satellites and within-sample standardisation; their ORs cannot be compared as unstandardised slopes.

BGG median log stellar masses are 11.119, 11.100, 11.100, and 10.976 for CG4, Control4B, Control4C, and RG4. Respectively 88.7%, 89.3%, 89.2%, and 87.5% exceed 10.75, rather than all doing so. Quenched fractions among classified BGGs are 80.4%, 83.3%, 83.4%, and 71.7%. This supports limited dynamic range in quenching, without asserting morphological saturation.

### Rejected or narrowed interpretations

- No universal compactness law, equal slopes/intercepts, causal compactness effect, universal surface-brightness failure, or halo-mass irrelevance was asserted.
- No hard mass cut, universal BGG saturation, or environmental equivalence was inferred.
- No fixed physical fibre aperture, universal sSFR timescale, gas-exhaustion mechanism, or accelerated central quenching was introduced.
- Binary composition mixing gives only (0.074 − 0.042) × 0.16 ≈ 0.005 dex. It cannot explain the entire nominal 0.014-dex difference between the published matched means; those means also use different available-pair subsets. This arithmetic was not presented as a fitted decomposition. No historical size “puzzle” was declared solved.
- No new E/S0 or post-starburst classification was added. Existing deferrals were retained.
- `ae`, `aecompl`, and their associated declarations were retained: `ae.sty` overrides the text-family defaults after `txfonts`. Removing it would change the established typography and metrics, contrary to the explicit constraint. The final and baseline PDFs have identical embedded font-family sets.

Primary literature consulted: [Coenda et al. 2012, especially Sects. 3 and 4.2](https://arxiv.org/pdf/1205.3428), [MPA-JHU DR7 methodology](https://wwwmpa.mpa-garching.mpg.de/SDSS/DR7/) and [SFR notes](https://wwwmpa.mpa-garching.mpg.de/SDSS/DR7/sfrs.html), [legacy SDSS fibre specifications](https://www.sdss2.org/dr7/products/spectra/index.php), [Paper I, Sects. 3.3 and 4.3/Table 5](https://arxiv.org/pdf/2407.20357), and [Zheng & Shen 2021, Eq. 1](https://arxiv.org/pdf/2102.12804). The indexed [official A&A guide](https://www.aanda.org/doc_journal/instructions/aadoc.pdf), including its class/font, graphics, and table instructions, was consulted; direct full-page retrieval from A&A returned HTTP 403. No bibliography entries were added or altered.

## D. Regression verification

| Metric | Before | After |
|---|---:|---:|
| Main-paper words | 4621 | 4540 |
| Total body-text words | 9549 | 9472 |
| Main-paper typeset pages, including references | 11 | 11 |
| Total PDF pages | 18 | 18 |
| Unresolved references/citations | 0 | 0 |
| Compilation/package/bibliography warnings | 0 | 0 |
| Overfull boxes | 0 | 0 |
| A&A configuration | Existing official `aa` class | Preserved |

Counts use `texcount -merge`, reporting **Words in text**. Main text runs from `maketitle` through Data availability, before acknowledgements, following the project's convention. Total counts use the full source, including abstract, acknowledgements, and appendices; headers, captions/table material, mathematics, and bibliography are excluded by this statistic. The main article and references occupy pages 1–11; appendices start on page 12. The 12-page target is respected.

Both baseline and final builds used this established sequence, in the directory containing the source, class, macros, bibliography, and figures:

```sh
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -synctex=0 paper.tex
bibtex paper
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -synctex=0 paper.tex
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -synctex=0 paper.tex
```

The template was rendered with the project's unchanged `_build_render_context`, filters, and StrictUndefined Jinja configuration without writing the merged context back to `results_build.json`. The normal documented render-only entry point remains `.venv_clean/bin/python -m src.main`.

Completed checks:

- All four final build commands succeeded; final LaTeX and BibTeX logs have no warnings.
- Targeted existing suite: **14 passed** (`test_size_render_smoke.py`, `test_zheng_shen_classes.py`, `test_submission_qc.py`, `test_consistency_gate.py`).
- `audit/consistency_gate.py` passed. Active source has no dangling references or uncited figures; all nine appendices are cited.
- Baseline/final pages 2 and 7–10 were visually compared; Table B.1 and the revised final appendix were inspected at readable resolution. All 18 final pages were screened for downstream layout problems. No new visual defects were found.
- `git diff --check` passed. Hash checks preserved all existing data, stored result JSON, figures, macros, bibliography, journal class/style, supplement, and unrelated modified/untracked files.
- The saved baseline source, template, PDF, initial diff/status, file hashes, refit outputs, and build logs are under `/private/tmp/cg_gals_conservative_revision/`.

## E. Outstanding issues

No scientific analysis or compilation problem blocks these changes. The retained legacy font configuration is the sole deliberately unimplemented production suggestion because changing it would conflict with the typography constraint.

Two older auxiliary checkers retain **identical pre-existing flags**, rather than new manuscript failures: `phase4_checks.py` parses Table 2 using the obsolete column order and sees a figure inside an inactive `iffalse` block; `style_checks.py` flags two vocabulary counts and one existing single-sentence paragraph. The actual table headers/counts, active cross-references, compiler diagnostics, and current targeted tests were checked independently. These older checkers were not rewritten in this surgical revision.
