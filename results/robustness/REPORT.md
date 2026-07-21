# CG4 Morphology/Quenching Robustness Report

Run date: 2026-07-17. Robustness runner version: `2026-07-17`.

## Methods Actually Used

- Input catalogue: `data/processed_sample.pkl`, harmonized with `src.extended_data.ensure_galaxy_frame`.
- Morphology: `p_E` and `p_S`, loaded by the project from SDSS `zooSpec.p_el_debiased` and `zooSpec.p_cs_debiased`.
- Fiducial early-type proxy: `morphology == Elliptical`; `Spiral` is the binary reference and `Uncertain` rows are excluded.
- Star-formation class: `sSFR_status`, with `Quenched` versus `Starforming`; `NosSFR` rows are excluded.
- Structural columns: local caches `data/sdss_size_columns.csv` (`petroR50_r`, `petroR90_r`) and `data/simard2011_subset.csv` (`ng`). No external downloads were attempted.
- Model adjustment: the existing helper selected `logMstar`, `z_numeric`, `is_satellite` for all-member fits, `log_group_luminosity`, and `velocity_dispersion` when complete enough. Satellite-only fits remove `is_satellite` after subsetting. Standard errors are clustered by `physical_group`.
- Holm correction: one global Holm correction was applied over all successful inferential estimates in this new robustness family.


Raw (non-debiased) Galaxy Zoo vote fractions are not present in the processed sample or local source tables. The fiducial catalogue morphology is already debiased, so the raw-vs-debiased request cannot be separated locally.


## Fiducial Reproduction

| family  | contrast        | model                 | stored_or | refit_or | delta_or | stored_n | refit_n | status |
| ------- | --------------- | --------------------- | --------- | -------- | -------- | -------- | ------- | ------ |
| primary | RG4             | elliptical_all        | 2.805     | 2.805    | 0.000    | 399      | 399     | ok     |
| primary | RG4             | elliptical_satellites | 3.031     | 3.031    | 0.000    | 299      | 299     | ok     |
| primary | RG4             | quenched_all          | 1.404     | 1.404    | 0.000    | 449      | 449     | ok     |
| primary | Control4C       | elliptical_all        | 1.298     | 1.298    | 0.000    | 2493     | 2493    | ok     |
| primary | Control4C       | quenched_all          | 0.908     | 0.908    | 0.000    | 2867     | 2867    | ok     |
| pooled  | pooled_controls | quenched_satellites   | 1.166     | 1.166    | 0.000    | 3268     | 3268    | ok     |
| pooled  | pooled_controls | elliptical_satellites | 1.707     | 1.707    | 0.000    | 2848     | 2848    | ok     |

## Task A - Debiased Morphology

| contrast         | scope      | proxy                   | threshold | estimate | ci_low | ci_high | p        | p_holm   | n    |
| ---------------- | ---------- | ----------------------- | --------- | -------- | ------ | ------- | -------- | -------- | ---- |
| CG4_vs_RG4       | all        | catalog_morphology_flag | 0.500     | 2.805    | 1.795  | 4.385   | 5.99e-06 | 3.05e-04 | 399  |
| CG4_vs_RG4       | all        | gz1_debiased_votes      | 0.500     | 2.805    | 1.795  | 4.385   | 5.99e-06 | 3.05e-04 | 399  |
| CG4_vs_RG4       | all        | gz1_debiased_votes      | 0.800     | 4.761    | 2.206  | 10.272  | 6.99e-05 | 0.003    | 228  |
| CG4_vs_RG4       | satellites | catalog_morphology_flag | 0.500     | 3.031    | 1.719  | 5.344   | 1.27e-04 | 0.006    | 299  |
| CG4_vs_RG4       | satellites | gz1_debiased_votes      | 0.500     | 3.031    | 1.719  | 5.344   | 1.27e-04 | 0.006    | 299  |
| CG4_vs_RG4       | satellites | gz1_debiased_votes      | 0.800     | 10.831   | 3.069  | 38.224  | 2.13e-04 | 0.010    | 150  |
| CG4_vs_Control4C | all        | catalog_morphology_flag | 0.500     | 1.298    | 0.975  | 1.729   | 0.074    | 1.000    | 2493 |
| CG4_vs_Control4C | all        | gz1_debiased_votes      | 0.500     | 1.298    | 0.975  | 1.729   | 0.074    | 1.000    | 2493 |
| CG4_vs_Control4C | all        | gz1_debiased_votes      | 0.800     | 1.866    | 1.231  | 2.829   | 0.003    | 0.135    | 1409 |
| CG4_vs_Control4C | satellites | catalog_morphology_flag | 0.500     | 1.395    | 1.010  | 1.927   | 0.043    | 1.000    | 1836 |
| CG4_vs_Control4C | satellites | gz1_debiased_votes      | 0.500     | 1.395    | 1.010  | 1.927   | 0.043    | 1.000    | 1836 |
| CG4_vs_Control4C | satellites | gz1_debiased_votes      | 0.800     | 2.262    | 1.415  | 3.616   | 6.50e-04 | 0.029    | 862  |

At the fiducial debiased threshold for satellites against RG4, OR = 3.031 (95% CI 1.719-5.344; Holm p = 0.006).


## Task B - Structural Morphology Proxies

| contrast         | scope      | proxy                 | model_type | threshold | estimate | ci_low   | ci_high | p        | p_holm | n    |
| ---------------- | ---------- | --------------------- | ---------- | --------- | -------- | -------- | ------- | -------- | ------ | ---- |
| CG4_vs_RG4       | all        | concentration_r90_r50 | logistic   | 2.600     | 1.142    | 0.762    | 1.712   | 0.519    | 1.000  | 453  |
| CG4_vs_RG4       | all        | concentration_r90_r50 | logistic   | 2.500     | 1.055    | 0.696    | 1.599   | 0.802    | 1.000  | 453  |
| CG4_vs_RG4       | all        | concentration_r90_r50 | logistic   | 2.860     | 1.858    | 1.066    | 3.238   | 0.029    | 1.000  | 453  |
| CG4_vs_RG4       | all        | concentration_r90_r50 | OLS        | NA        | 0.105    | 7.20e-04 | 0.209   | 0.048    | 1.000  | 453  |
| CG4_vs_RG4       | all        | simard_sersic_n       | logistic   | 2.500     | 1.932    | 1.099    | 3.396   | 0.022    | 0.866  | 393  |
| CG4_vs_RG4       | all        | simard_sersic_n       | logistic   | 2.000     | 1.581    | 0.912    | 2.738   | 0.102    | 1.000  | 393  |
| CG4_vs_RG4       | all        | simard_sersic_n       | logistic   | 3.000     | 1.908    | 1.072    | 3.397   | 0.028    | 1.000  | 393  |
| CG4_vs_RG4       | all        | simard_sersic_n       | OLS        | NA        | 0.489    | 0.182    | 0.796   | 0.002    | 0.075  | 393  |
| CG4_vs_RG4       | satellites | concentration_r90_r50 | logistic   | 2.600     | 1.133    | 0.644    | 1.994   | 0.665    | 1.000  | 343  |
| CG4_vs_RG4       | satellites | concentration_r90_r50 | logistic   | 2.500     | 1.037    | 0.618    | 1.740   | 0.891    | 1.000  | 343  |
| CG4_vs_RG4       | satellites | concentration_r90_r50 | logistic   | 2.860     | 1.882    | 0.943    | 3.756   | 0.073    | 1.000  | 343  |
| CG4_vs_RG4       | satellites | concentration_r90_r50 | OLS        | NA        | 0.123    | -0.003   | 0.250   | 0.056    | 1.000  | 343  |
| CG4_vs_RG4       | satellites | simard_sersic_n       | logistic   | 2.500     | 2.005    | 1.039    | 3.866   | 0.038    | 1.000  | 332  |
| CG4_vs_RG4       | satellites | simard_sersic_n       | logistic   | 2.000     | 1.892    | 1.052    | 3.405   | 0.033    | 1.000  | 332  |
| CG4_vs_RG4       | satellites | simard_sersic_n       | logistic   | 3.000     | 2.287    | 1.129    | 4.630   | 0.022    | 0.864  | 332  |
| CG4_vs_RG4       | satellites | simard_sersic_n       | OLS        | NA        | 0.573    | 0.236    | 0.910   | 8.69e-04 | 0.038  | 332  |
| CG4_vs_Control4C | all        | concentration_r90_r50 | logistic   | 2.600     | 0.868    | 0.663    | 1.137   | 0.305    | 1.000  | 2887 |
| CG4_vs_Control4C | all        | concentration_r90_r50 | logistic   | 2.500     | 0.900    | 0.681    | 1.190   | 0.461    | 1.000  | 2887 |
| CG4_vs_Control4C | all        | concentration_r90_r50 | logistic   | 2.860     | 1.188    | 0.860    | 1.642   | 0.296    | 1.000  | 2887 |
| CG4_vs_Control4C | all        | concentration_r90_r50 | OLS        | NA        | 0.014    | -0.058   | 0.086   | 0.704    | 1.000  | 2887 |
| CG4_vs_Control4C | all        | simard_sersic_n       | logistic   | 2.500     | 1.348    | 1.001    | 1.815   | 0.049    | 1.000  | 2291 |
| CG4_vs_Control4C | all        | simard_sersic_n       | logistic   | 2.000     | 1.177    | 0.852    | 1.627   | 0.323    | 1.000  | 2291 |
| CG4_vs_Control4C | all        | simard_sersic_n       | logistic   | 3.000     | 1.208    | 0.933    | 1.563   | 0.152    | 1.000  | 2291 |
| CG4_vs_Control4C | all        | simard_sersic_n       | OLS        | NA        | 0.181    | -0.031   | 0.393   | 0.095    | 1.000  | 2291 |
| CG4_vs_Control4C | satellites | concentration_r90_r50 | logistic   | 2.600     | 0.880    | 0.629    | 1.231   | 0.456    | 1.000  | 2175 |
| CG4_vs_Control4C | satellites | concentration_r90_r50 | logistic   | 2.500     | 0.873    | 0.635    | 1.199   | 0.401    | 1.000  | 2175 |
| CG4_vs_Control4C | satellites | concentration_r90_r50 | logistic   | 2.860     | 1.208    | 0.862    | 1.693   | 0.273    | 1.000  | 2175 |
| CG4_vs_Control4C | satellites | concentration_r90_r50 | OLS        | NA        | 0.014    | -0.074   | 0.103   | 0.752    | 1.000  | 2175 |
| CG4_vs_Control4C | satellites | simard_sersic_n       | logistic   | 2.500     | 1.359    | 1.001    | 1.844   | 0.049    | 1.000  | 2043 |
| CG4_vs_Control4C | satellites | simard_sersic_n       | logistic   | 2.000     | 1.259    | 0.907    | 1.747   | 0.169    | 1.000  | 2043 |
| CG4_vs_Control4C | satellites | simard_sersic_n       | logistic   | 3.000     | 1.246    | 0.950    | 1.635   | 0.112    | 1.000  | 2043 |
| CG4_vs_Control4C | satellites | simard_sersic_n       | OLS        | NA        | 0.199    | -0.030   | 0.428   | 0.089    | 1.000  | 2043 |

At least one structural proxy is positive and significant after Holm correction: B_RG4_satellites_sersic_continuous.


## Task C - 2x2 Morphology x Star-Formation Decomposition

| outcome       | estimate | ci_low | ci_high | p     | p_holm | n   |
| ------------- | -------- | ------ | ------- | ----- | ------ | --- |
| early_passive | 3.426    | 1.624  | 7.226   | 0.001 | 0.052  | 296 |
| early_SF      | 2.199    | 0.774  | 6.245   | 0.139 | 1.000  | 296 |
| late_passive  | 1.349    | 0.541  | 3.366   | 0.521 | 1.000  | 296 |

Observed complete-case satellite fractions:

| sample    | cell          | n_cell | n_complete | fraction |
| --------- | ------------- | ------ | ---------- | -------- |
| CG4       | late_SF       | 49     | 154        | 0.318    |
| CG4       | early_passive | 74     | 154        | 0.481    |
| CG4       | early_SF      | 11     | 154        | 0.071    |
| CG4       | late_passive  | 20     | 154        | 0.130    |
| RG4       | late_SF       | 79     | 142        | 0.556    |
| RG4       | early_passive | 31     | 142        | 0.218    |
| RG4       | early_SF      | 8      | 142        | 0.056    |
| RG4       | late_passive  | 24     | 142        | 0.169    |
| Control4C | late_SF       | 692    | 1673       | 0.414    |
| Control4C | early_passive | 608    | 1673       | 0.363    |
| Control4C | early_SF      | 130    | 1673       | 0.078    |
| Control4C | late_passive  | 243    | 1673       | 0.145    |

Largest adjusted RRR is `early_passive`: RRR = 3.426 (95% CI 1.624-7.226; Holm p = 0.052). Interpretation hooks: early_passive implies historical transform-and-quench; early_SF implies tidal heating without quenching; late_passive implies strangulation without structural transformation.


## Task D - Quenching Null CI

| contrast               | scope      | estimate | ci_low | ci_high | p     | p_holm | n    |
| ---------------------- | ---------- | -------- | ------ | ------- | ----- | ------ | ---- |
| CG4_vs_RG4             | all        | 1.404    | 0.799  | 2.467   | 0.238 | 1.000  | 449  |
| CG4_vs_RG4             | satellites | 1.854    | 0.994  | 3.456   | 0.052 | 1.000  | 340  |
| CG4_vs_pooled_controls | all        | 1.075    | 0.800  | 1.444   | 0.631 | 1.000  | 3969 |
| CG4_vs_pooled_controls | satellites | 1.166    | 0.859  | 1.584   | 0.325 | 1.000  | 3268 |

Pooled satellite quenching OR = 1.166 (95% CI 0.859-1.584). This interval is compatible with no effect and with a modest excess.


## Data Availability And Fibre-Collision Caveat

| sample    | n_rows | size_ok_petro | petro_available_fraction | size_ok_simard | simard_available_fraction |
| --------- | ------ | ------------- | ------------------------ | -------------- | ------------------------- |
| CG4       | 248    | 248           | 1.000                    | 192            | 0.774                     |
| Control4B | 2792   | 2792          | 1.000                    | 1916           | 0.686                     |
| Control4C | 2816   | 2807          | 0.997                    | 2100           | 0.746                     |
| RG4       | 224    | 224           | 1.000                    | 197            | 0.879                     |

Projected-distance-rank 1 and 2 satellites lacking the SF classification:

| sample    | satellite_projected_distance_rank | n_satellites | n_lacking_sf_classification | fraction_lacking_sf_classification |
| --------- | --------------------------------- | ------------ | --------------------------- | ---------------------------------- |
| CG4       | 1                                 | 62           | 2                           | 0.032                              |
| CG4       | 2                                 | 62           | 6                           | 0.097                              |
| Control4B | 1                                 | 698          | 50                          | 0.072                              |
| Control4B | 2                                 | 698          | 36                          | 0.052                              |
| Control4C | 1                                 | 704          | 46                          | 0.065                              |
| Control4C | 2                                 | 704          | 33                          | 0.047                              |
| RG4       | 1                                 | 56           | 0                           | 0.000                              |
| RG4       | 2                                 | 56           | 1                           | 0.018                              |

## Output Files

- `tables/morphology_debiased.csv`
- `tables/structural_proxies.csv`
- `tables/multinomial_rrr.csv`
- `tables/observed_cell_fractions.csv`
- `tables/quenching_ci.csv`
- `tables/exclusions.csv`
- `figures/robustness_forest.pdf` and `.png`

