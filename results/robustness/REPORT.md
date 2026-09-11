# CG4 Morphology/Quenching Robustness Report

Run date: 2026-09-11. Robustness runner version: `2026-09-11`.

## Methods Actually Used

- Input catalogue: `data/processed_sample.pkl`, harmonized with `src.extended_data.ensure_galaxy_frame`.
- Morphology: `p_E` and `p_S`, loaded by the project from SDSS `zooSpec.p_el_debiased` and `zooSpec.p_cs_debiased`.
- Fiducial early-type proxy: `morphology == Elliptical`; `Spiral` is the binary reference and `Uncertain` rows are excluded.
- Star-formation class: `sSFR_status`, with `Quenched` versus `Starforming`; `NosSFR` rows are excluded.
- Structural columns: local caches `data/sdss_size_columns.csv` (`petroR50_r`, `petroR90_r`) and `data/simard2011_subset.csv` (`ng`). No external downloads were attempted.
- Model adjustment: the existing helper selected `logMstar`, `z_numeric`, `is_satellite` for all-member fits, `log_group_luminosity`, and `velocity_dispersion` when complete enough. Satellite-only fits remove `is_satellite` after subsetting. Standard errors are clustered by `physical_group`.
- These are labelled robustness checks of the fitted direction, magnitude, and interval; they are not pooled into an additional multiple-testing family.


Raw (non-debiased) Galaxy Zoo vote fractions are not present in the processed sample or local source tables. The fiducial catalogue morphology is already debiased, so the raw-vs-debiased request cannot be separated locally.


## Fiducial Reproduction

| family  | contrast        | model                 | stored_or | refit_or | delta_or | stored_n | refit_n | status |
| ------- | --------------- | --------------------- | --------- | -------- | -------- | -------- | ------- | ------ |
| primary | RG4             | elliptical_all        | 2.805     | 2.805    | 0.000    | 399      | 399     | ok     |
| primary | RG4             | elliptical_satellites | 3.031     | 3.031    | 0.000    | 299      | 299     | ok     |
| primary | RG4             | quenched_all          | 1.404     | 1.404    | 0.000    | 449      | 449     | ok     |
| primary | Control4C       | elliptical_all        | 1.671     | 1.671    | 0.000    | 2557     | 2557    | ok     |
| primary | Control4C       | quenched_all          | 1.245     | 1.245    | 0.000    | 2851     | 2851    | ok     |
| pooled  | pooled_controls | quenched_satellites   | 1.526     | 1.526    | 0.000    | 2908     | 2908    | ok     |
| pooled  | pooled_controls | elliptical_satellites | 2.082     | 2.082    | 0.000    | 2604     | 2604    | ok     |

## Task A - Debiased Morphology

| contrast         | scope      | proxy                   | threshold | estimate | ci_low | ci_high | p        | n    |
| ---------------- | ---------- | ----------------------- | --------- | -------- | ------ | ------- | -------- | ---- |
| CG4_vs_RG4       | all        | catalog_morphology_flag | 0.500     | 2.805    | 1.795  | 4.385   | 5.99e-06 | 399  |
| CG4_vs_RG4       | all        | gz1_debiased_votes      | 0.500     | 2.805    | 1.795  | 4.385   | 5.99e-06 | 399  |
| CG4_vs_RG4       | all        | gz1_debiased_votes      | 0.800     | 4.761    | 2.206  | 10.272  | 6.99e-05 | 228  |
| CG4_vs_RG4       | satellites | catalog_morphology_flag | 0.500     | 3.031    | 1.719  | 5.344   | 1.27e-04 | 299  |
| CG4_vs_RG4       | satellites | gz1_debiased_votes      | 0.500     | 3.031    | 1.719  | 5.344   | 1.27e-04 | 299  |
| CG4_vs_RG4       | satellites | gz1_debiased_votes      | 0.800     | 10.831   | 3.069  | 38.224  | 2.13e-04 | 150  |
| CG4_vs_Control4C | all        | catalog_morphology_flag | 0.500     | 1.671    | 1.247  | 2.238   | 5.86e-04 | 2557 |
| CG4_vs_Control4C | all        | gz1_debiased_votes      | 0.500     | 1.671    | 1.247  | 2.238   | 5.86e-04 | 2557 |
| CG4_vs_Control4C | all        | gz1_debiased_votes      | 0.800     | 2.158    | 1.427  | 3.264   | 2.68e-04 | 1588 |
| CG4_vs_Control4C | satellites | catalog_morphology_flag | 0.500     | 1.932    | 1.403  | 2.660   | 5.42e-05 | 1901 |
| CG4_vs_Control4C | satellites | gz1_debiased_votes      | 0.500     | 1.932    | 1.403  | 2.660   | 5.42e-05 | 1901 |
| CG4_vs_Control4C | satellites | gz1_debiased_votes      | 0.800     | 2.909    | 1.883  | 4.495   | 1.49e-06 | 1042 |

At the fiducial debiased threshold for satellites against RG4, OR = 3.031 (95% CI 1.719-5.344; p = 0.000).


## Task B - Structural Morphology Proxies

| contrast         | scope      | proxy                 | model_type | threshold | estimate | ci_low   | ci_high | p        | n    |
| ---------------- | ---------- | --------------------- | ---------- | --------- | -------- | -------- | ------- | -------- | ---- |
| CG4_vs_RG4       | all        | concentration_r90_r50 | logistic   | 2.600     | 1.142    | 0.762    | 1.712   | 0.519    | 453  |
| CG4_vs_RG4       | all        | concentration_r90_r50 | logistic   | 2.500     | 1.055    | 0.696    | 1.599   | 0.802    | 453  |
| CG4_vs_RG4       | all        | concentration_r90_r50 | logistic   | 2.860     | 1.858    | 1.066    | 3.238   | 0.029    | 453  |
| CG4_vs_RG4       | all        | concentration_r90_r50 | OLS        | NA        | 0.105    | 7.20e-04 | 0.209   | 0.048    | 453  |
| CG4_vs_RG4       | all        | simard_sersic_n       | logistic   | 2.500     | 1.932    | 1.099    | 3.396   | 0.022    | 393  |
| CG4_vs_RG4       | all        | simard_sersic_n       | logistic   | 2.000     | 1.581    | 0.912    | 2.738   | 0.102    | 393  |
| CG4_vs_RG4       | all        | simard_sersic_n       | logistic   | 3.000     | 1.908    | 1.072    | 3.397   | 0.028    | 393  |
| CG4_vs_RG4       | all        | simard_sersic_n       | OLS        | NA        | 0.489    | 0.182    | 0.796   | 0.002    | 393  |
| CG4_vs_RG4       | satellites | concentration_r90_r50 | logistic   | 2.600     | 1.133    | 0.644    | 1.994   | 0.665    | 343  |
| CG4_vs_RG4       | satellites | concentration_r90_r50 | logistic   | 2.500     | 1.037    | 0.618    | 1.740   | 0.891    | 343  |
| CG4_vs_RG4       | satellites | concentration_r90_r50 | logistic   | 2.860     | 1.882    | 0.943    | 3.756   | 0.073    | 343  |
| CG4_vs_RG4       | satellites | concentration_r90_r50 | OLS        | NA        | 0.123    | -0.003   | 0.250   | 0.056    | 343  |
| CG4_vs_RG4       | satellites | simard_sersic_n       | logistic   | 2.500     | 2.005    | 1.039    | 3.866   | 0.038    | 332  |
| CG4_vs_RG4       | satellites | simard_sersic_n       | logistic   | 2.000     | 1.892    | 1.052    | 3.405   | 0.033    | 332  |
| CG4_vs_RG4       | satellites | simard_sersic_n       | logistic   | 3.000     | 2.287    | 1.129    | 4.630   | 0.022    | 332  |
| CG4_vs_RG4       | satellites | simard_sersic_n       | OLS        | NA        | 0.573    | 0.236    | 0.910   | 8.69e-04 | 332  |
| CG4_vs_Control4C | all        | concentration_r90_r50 | logistic   | 2.600     | 0.934    | 0.705    | 1.237   | 0.636    | 2879 |
| CG4_vs_Control4C | all        | concentration_r90_r50 | logistic   | 2.500     | 1.010    | 0.747    | 1.365   | 0.949    | 2879 |
| CG4_vs_Control4C | all        | concentration_r90_r50 | logistic   | 2.860     | 1.188    | 0.858    | 1.646   | 0.300    | 2879 |
| CG4_vs_Control4C | all        | concentration_r90_r50 | OLS        | NA        | 0.026    | -0.045   | 0.097   | 0.467    | 2879 |
| CG4_vs_Control4C | all        | simard_sersic_n       | logistic   | 2.500     | 1.549    | 1.144    | 2.097   | 0.005    | 2241 |
| CG4_vs_Control4C | all        | simard_sersic_n       | logistic   | 2.000     | 1.509    | 1.079    | 2.112   | 0.016    | 2241 |
| CG4_vs_Control4C | all        | simard_sersic_n       | logistic   | 3.000     | 1.300    | 1.005    | 1.681   | 0.046    | 2241 |
| CG4_vs_Control4C | all        | simard_sersic_n       | OLS        | NA        | 0.283    | 0.073    | 0.492   | 0.008    | 2241 |
| CG4_vs_Control4C | satellites | concentration_r90_r50 | logistic   | 2.600     | 0.986    | 0.698    | 1.392   | 0.936    | 2168 |
| CG4_vs_Control4C | satellites | concentration_r90_r50 | logistic   | 2.500     | 1.009    | 0.720    | 1.413   | 0.959    | 2168 |
| CG4_vs_Control4C | satellites | concentration_r90_r50 | logistic   | 2.860     | 1.269    | 0.908    | 1.773   | 0.163    | 2168 |
| CG4_vs_Control4C | satellites | concentration_r90_r50 | OLS        | NA        | 0.040    | -0.047   | 0.126   | 0.369    | 2168 |
| CG4_vs_Control4C | satellites | simard_sersic_n       | logistic   | 2.500     | 1.587    | 1.163    | 2.166   | 0.004    | 1993 |
| CG4_vs_Control4C | satellites | simard_sersic_n       | logistic   | 2.000     | 1.640    | 1.173    | 2.292   | 0.004    | 1993 |
| CG4_vs_Control4C | satellites | simard_sersic_n       | logistic   | 3.000     | 1.361    | 1.036    | 1.788   | 0.027    | 1993 |
| CG4_vs_Control4C | satellites | simard_sersic_n       | OLS        | NA        | 0.317    | 0.093    | 0.540   | 0.005    | 1993 |

Positive structural-proxy fits whose 95% confidence intervals exclude the null are: B_RG4_all_concentration_ge_2.86, B_RG4_all_concentration_continuous, B_RG4_all_sersic_ge_2.5, B_RG4_all_sersic_ge_3, B_RG4_all_sersic_continuous, B_RG4_satellites_sersic_ge_2.5, B_RG4_satellites_sersic_ge_2, B_RG4_satellites_sersic_ge_3, B_RG4_satellites_sersic_continuous, B_Control4C_all_sersic_ge_2.5, B_Control4C_all_sersic_ge_2, B_Control4C_all_sersic_ge_3, B_Control4C_all_sersic_continuous, B_Control4C_satellites_sersic_ge_2.5, B_Control4C_satellites_sersic_ge_2, B_Control4C_satellites_sersic_ge_3, B_Control4C_satellites_sersic_continuous. These deliberately overlapping proxy definitions are robustness diagnostics, not separate discoveries.


## Task C - 2x2 Morphology x Star-Formation Decomposition

| outcome       | estimate | ci_low | ci_high | p     | n   |
| ------------- | -------- | ------ | ------- | ----- | --- |
| early_passive | 3.426    | 1.624  | 7.226   | 0.001 | 296 |
| early_SF      | 2.199    | 0.774  | 6.245   | 0.139 | 296 |
| late_passive  | 1.349    | 0.541  | 3.366   | 0.521 | 296 |

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
| Control4C | late_SF       | 742    | 1736       | 0.427    |
| Control4C | early_passive | 593    | 1736       | 0.342    |
| Control4C | early_SF      | 111    | 1736       | 0.064    |
| Control4C | late_passive  | 290    | 1736       | 0.167    |

Largest adjusted RRR is `early_passive`: RRR = 3.426 (95% CI 1.624-7.226; p = 0.001). Interpretation hooks: early_passive implies historical transform-and-quench; early_SF implies tidal heating without quenching; late_passive implies strangulation without structural transformation.


## Task D - Quenching Null CI

| contrast               | scope      | estimate | ci_low | ci_high | p     | n    |
| ---------------------- | ---------- | -------- | ------ | ------- | ----- | ---- |
| CG4_vs_RG4             | all        | 1.404    | 0.799  | 2.467   | 0.238 | 449  |
| CG4_vs_RG4             | satellites | 1.854    | 0.994  | 3.456   | 0.052 | 340  |
| CG4_vs_pooled_controls | all        | 1.348    | 0.978  | 1.859   | 0.068 | 3608 |
| CG4_vs_pooled_controls | satellites | 1.526    | 1.103  | 2.113   | 0.011 | 2908 |

Pooled satellite quenching OR = 1.526 (95% CI 1.103-2.113). This pooled estimate points to a modest excess, but the control-specific fits show that it is not stable across comparison samples.


## Data Availability And Fibre-Collision Caveat

| sample    | n_rows | size_ok_petro | petro_available_fraction | size_ok_simard | simard_available_fraction |
| --------- | ------ | ------------- | ------------------------ | -------------- | ------------------------- |
| CG4       | 248    | 248           | 1.000                    | 192            | 0.774                     |
| Control4B | 2792   | 2792          | 1.000                    | 1917           | 0.687                     |
| Control4C | 2812   | 2809          | 0.999                    | 2030           | 0.722                     |
| RG4       | 224    | 224           | 1.000                    | 197            | 0.879                     |

Projected-distance-rank 1 and 2 satellites lacking the SF classification:

| sample    | satellite_projected_distance_rank | n_satellites | n_lacking_sf_classification | fraction_lacking_sf_classification |
| --------- | --------------------------------- | ------------ | --------------------------- | ---------------------------------- |
| CG4       | 1                                 | 62           | 2                           | 0.032                              |
| CG4       | 2                                 | 62           | 6                           | 0.097                              |
| Control4B | 1                                 | 698          | 50                          | 0.072                              |
| Control4B | 2                                 | 698          | 36                          | 0.052                              |
| Control4C | 1                                 | 703          | 52                          | 0.074                              |
| Control4C | 2                                 | 703          | 41                          | 0.058                              |
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

