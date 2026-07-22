# Submission additions — number provenance

One line per macro inserted at submission: LaTeX macro → key path in
`output/paper_additions.json` → producing function in
`src/paper_additions.py`. All macros are generated mechanically by
`build_macros()`/`write_macros()` into `output/paper/additions_macros.tex`
(loaded from the template preamble); regenerate with
`python src/paper_additions.py` (seed 42, ~12 s). "Used in" gives the
manuscript location; macros marked *defined only* are emitted for
completeness (e.g. the space-conditional consistency identity of Task 4b,
dropped because the main body sits at the 12-page cap) and can be cited in
the referee response without touching the pipeline.

Cell variants (`...Cell`) render `p^{+hi-p}_{-(p-lo)}` with Wilson 95%
offsets; scalar variants render the fraction alone. Values re-emitted from
`output/results.json` (published, unchanged) are marked †.

## Separations (function `separations_block`; JSON `separations.*`)

| Macro | JSON key | Used in |
|---|---|---|
| `\SepCC` | `separations.Control4C.median_kpc` | Sect. 3.2 gradation clause |
| `\SepLoose` | `separations.loose_pair_mean_kpc` (rounded to 10) | Sect. 3.2 gradation clause |
| `\SepCG`, `\SepCB`, `\SepRG` (+ `\Sep*Iqr`) | `separations.<sample>.{median,q25,q75}_kpc` | defined only (Sect. 2.1 already quotes these via referee `T9.compactness`) |
| `\SepSpan` | `separations.span_factor` | defined only (cut in the page-budget compression) |

## Star formation at fixed morphology (function `quenched_block`; JSON `quenched_by_morphology.*`)

| Macro | JSON key | Used in |
|---|---|---|
| `\qEcg`, `\qEcb`, `\qEcc`, `\qErg` | `per_sample.<sample>.Elliptical.p` | Sect. 4.3 replacement ¶ |
| `\qE*Cell`, `\qSp*Cell` | `per_sample.<sample>.{Elliptical,Spiral}.{p,wilson_lo,wilson_hi}` | Table B.2 (`tab:quenched_by_morphology`) |
| `\qUcg`, `\qUcb`, `\qUcc`, `\qUrg`; `\nU*` | `per_sample.<sample>.Uncertain.{p,classified}` | Table B.2 footnote |
| `\pQEhom` | `chi2_homogeneity_PQE.p` | Sect. 4.3 ¶ and App. B paragraph |
| `\condCB/CC/RG` (+`lo`,`hi`) | `kitagawa.<control>.{conditional_term,conditional_ci95}` | Sect. 4.3 replacement ¶ |
| `\qSpcg…`, `\qU*Cell`, `\condBound*` | same blocks | defined only |

## Zheng–Shen classes (function `zheng_shen_block`; JSON `zheng_shen.*`)

| Macro | JSON key | Used in |
|---|---|---|
| `\feAllIso/Emb/Pre`, `\feSatIso/Emb/Pre` | `per_class.<class>.fE_{all,sat}` | Table E.3 (`tab:zheng_shen_fe`) |
| `\lMIso/Emb/Pre` | `per_class.<class>.median_host_lM200` | Table E.3 |
| `\nGrIso/Emb/Pre` | `per_class.<class>.n_groups` | Table E.3; Sect. 4.1 sentences (`\nGrIso`) |
| `\pIsoRest`, `\pIsoRG`, `\pIsoCC` | `permutations.<test>.p` | App. E.3 paragraph; Sect. 4.1 sentences (`\pIsoRest`, `\pIsoRG`) |

## Tidal index (functions `tidal_block`, `host_inclusive_block`; JSON `tidal.*`)

| Macro | JSON key | Used in |
|---|---|---|
| `\sdlogT` | `tidal.logT_sd_in_elliptical_model_frame.sd_dex` | Sect. 3.6 unit clarification ("per \sdlogT-dex SD") |
| `\gapCB`, `\gapCC`, `\gapRG` | `tidal.gap_vs_control_dex.<control>` | Sect. 3.6 common-support sentence |
| `\dTall` | `tidal.host_inclusive.pooled.median_delta_dex` | App. F host-inclusive passage |
| `\dTmaxSamp` | max over `tidal.host_inclusive.per_sample.*.median_delta_dex` | App. F host-inclusive passage |
| `\rhoTT` | `tidal.host_inclusive.pooled.spearman_rho` | App. F host-inclusive passage |
| `\ORresidAll` | `tidal.host_inclusive.refit_elliptical_with_host_T.cg4_odds_ratio` | App. F host-inclusive passage |
| `\ORresid` † | `tidal.published_inputs.residual_or` | App. F host-inclusive passage |
| `\ORbase` †, `\ORtidal` †, `\gapPooled` †, `\expTid`, `\ORconsistency` | `tidal.{published_inputs,internal_consistency}.*` | defined only (Task 4b identity, dropped for space: `\ORresid × \ORtidal^{\expTid} = \ORconsistency ≈ \ORbase`) |
