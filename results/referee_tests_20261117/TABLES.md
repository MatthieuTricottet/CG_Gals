# Referee reanalysis tables

Intervals are 95% unless marked otherwise. Estimates in the matching and raw fraction tables are on a 0–1 scale. Marginal differences for binary outcomes are in percentage points. The CSV files retain model formulas, warnings, exact denominators and additional sensitivity results.

## Sample construction and compactness support

### Sample counts

[Full CSV](tables/sample_counts.csv)

| sample | n | quartets | hosts | E_S | uncertain | missing_votes | Q_SF | colour_retained | derived_colours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CG4 | 248 | 62 | 60 | 210 | 7 | 31 | 230 | 110 | 217 |
| Control4C | 2812 | 703 | 703 | 2347 | 144 | 321 | 2621 | 1153 | 2491 |
| Control4B | 2792 | 698 | 698 | 2372 | 116 | 304 | 2612 | 931 | 2488 |
| RG4 | 224 | 56 | 56 | 189 | 15 | 20 | 219 | 102 | 204 |

### Compact support

[Full CSV](tables/compact_support.csv)

| sample | n | hosts | within_other_90 | within_other_full | both_central_90 | hosts_both_central_90 | from_kpc | to_kpc | n_100_150 | n_150_200 | n_200_250 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CG4 | 62 | 60 | 52 | 62 | 48 | 46 | 95.25 | 215.9 | 22 | 24 | 5 |
| Control4C | 703 | 703 | 217 | 249 | 187 | 187 | 95.25 | 215.9 | 73 | 79 | 91 |

### Compact overlap

[Full CSV](tables/compact_overlap.csv)

| metric | bins | overlap |
| --- | --- | --- |
| x | 20 | 0.3661 |
| x | 40 | 0.3575 |
| x | 60 | 0.3467 |
| xmed | 20 | 0.3955 |
| xmed | 40 | 0.3684 |
| xmed | 60 | 0.3353 |

### Compact model flow

[Full CSV](tables/compact_model_flow.csv)

| stage | n | quartets | hosts | cg_n | cg_quartets | cg_hosts | control_n | control_quartets | control_hosts | es_cg_n | es_cg_quartets | es_cg_hosts | es_control_n | es_control_quartets | es_control_hosts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 2295 | 765 | 760 | 186 | 62 | 60 | 2109 | 703 | 703 | 157 | 59 | 58 | 1744 | 679 | 679 |
| common_compactness | 705 | 235 | 233 | 144 | 48 | 46 | 561 | 187 | 187 | 122 | 45 | 44 | 453 | 181 | 181 |
| quartet_joint_support | 453 | 151 | 150 | 123 | 41 | 40 | 330 | 110 | 110 | 110 | 39 | 39 | 281 | 107 | 107 |
| satellite_joint_E_S_support | 494 | 200 | 199 | 122 | 45 | 44 | 372 | 155 | 155 | 122 | 45 | 44 | 372 | 155 | 155 |

### Appendix f reproduction

[Full CSV](tables/appendix_f_reproduction.csv)

| sample | n | quartet_clusters | OR_per_sample_SD | low | high | p |
| --- | --- | --- | --- | --- | --- | --- |
| CG4 | 157 | 59 | 0.6526 | 0.5042 | 0.8448 | 0.001189 |
| RG4 | 142 | 55 | 0.9227 | 0.6369 | 1.337 | 0.6705 |
| Control4B | 1772 | 677 | 0.8001 | 0.7225 | 0.886 | 1.831e-05 |
| Control4C | 1744 | 679 | 0.7023 | 0.6337 | 0.7782 | 1.546e-11 |

## Compactness model and matching

### Compact model coefficients

[Full CSV](tables/compact_model_coefficients.csv)

| spec | term | beta | low | high | p | cg_n | control_n | hosts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| common_slope | cg | 0.1954 | -0.2062 | 0.5969 | 0.3387 | 122 | 453 | 225 |
| common_slope | xc | -2.182 | -4.062 | -0.3027 | 0.02307 | 122 | 453 | 225 |
| interaction | cg | 0.1004 | -0.3697 | 0.5706 | 0.6742 | 122 | 453 | 225 |
| interaction | xc | -1.965 | -4.078 | 0.1487 | 0.06828 | 122 | 453 | 225 |
| interaction | cg:xc | -1.41 | -5.35 | 2.529 | 0.4812 | 122 | 453 | 225 |

### Compact marginal contrasts

[Full CSV](tables/compact_marginal_contrasts.csv)

| spec | kpc | cg_est | control_est | difference | low | high | cg_hosts | control_hosts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| common_slope | 150 | 52.36 | 47.68 | 4.678 | -4.91 | 14.27 | 33 | 95 |
| common_slope | 175 | 48.68 | 44.04 | 4.64 | -4.904 | 14.18 | 29 | 107 |
| common_slope | 200 | 46.37 | 41.8 | 4.569 | -4.852 | 13.99 | 19 | 73 |
| interaction | 150 | 52.31 | 47.64 | 4.666 | -4.99 | 14.32 | 33 | 95 |
| interaction | 175 | 46.76 | 44.38 | 2.381 | -8.771 | 13.53 | 29 | 107 |
| interaction | 200 | 42.86 | 42.43 | 0.433 | -13.95 | 14.82 | 19 | 73 |
| joint_satellite_support | 150 | 51.91 | 48.58 | 3.333 | -6.436 | 13.1 | 33 | 93 |
| joint_satellite_support | 175 | 46.69 | 45.9 | 0.796 | -10.39 | 11.98 | 29 | 104 |
| joint_satellite_support | 200 | 43.5 | 44.84 | -1.342 | -15.9 | 13.21 | 19 | 72 |

### Compact matching flow

[Full CSV](tables/compact_matching_flow.csv)

| spec | radius_caliper | pool_cg | pool_cc | complete_cg | complete_cc | admissible_edges | admissible_cg | admissible_cc | matched_cg | matched_cc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| without_compactness_full | — | 62 | 703 | 53 | 594 | 4871 | 53 | 557 | 53 | 53 |
| without_compactness_same_pool | — | 48 | 187 | 42 | 152 | 1025 | 42 | 132 | 42 | 42 |
| compactness_matched | 0.08 | 48 | 187 | 42 | 152 | 442 | 41 | 110 | 41 | 41 |
| tighter_radius | 0.05 | 48 | 187 | 42 | 152 | 275 | 41 | 100 | 41 | 41 |
| wider_radius | 0.12 | 48 | 187 | 42 | 152 | 629 | 41 | 122 | 41 | 41 |
| velocity_balanced | 0.08 | 48 | 187 | 42 | 152 | 272 | 41 | 95 | 41 | 41 |

### Compact matching effects

[Full CSV](tables/compact_matching_effects.csv)

| spec | pairs | pairs_both_outcomes | cg_labels | cc_labels | cg_fraction | cc_fraction | estimate | low | high | effective_blocks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| without_compactness_full | 53 | 50 | 140 | 131 | 0.5367 | 0.3567 | 0.18 | 0.07667 | 0.2767 | 50 |
| without_compactness_same_pool | 42 | 40 | 113 | 110 | 0.5083 | 0.4292 | 0.07917 | -0.07083 | 0.2208 | 40 |
| compactness_matched | 41 | 37 | 105 | 96 | 0.5135 | 0.5135 | -6.001e-18 | -0.1261 | 0.1264 | 37 |
| tighter_radius | 41 | 37 | 104 | 93 | 0.5045 | 0.4775 | 0.02703 | -0.1126 | 0.1712 | 37 |
| wider_radius | 41 | 38 | 107 | 100 | 0.5088 | 0.5132 | -0.004386 | -0.1186 | 0.1053 | 38 |
| velocity_balanced | 41 | 39 | 110 | 99 | 0.5043 | 0.4701 | 0.03419 | -0.09829 | 0.1752 | 39 |

### Compact matching balance

[Full CSV](tables/compact_matching_balance.csv)

| stage | covariate | cg | cc | smd |
| --- | --- | --- | --- | --- |
| pool | x | 2.175 | 2.193 | -0.1989 |
| matched | x | 2.172 | 2.178 | -0.06711 |
| both_E_S | x | 2.171 | 2.177 | -0.06522 |
| pool | quartet_z | 0.03691 | 0.03467 | 0.3643 |
| matched | quartet_z | 0.037 | 0.03708 | -0.01262 |
| both_E_S | quartet_z | 0.03657 | 0.03659 | -0.002476 |
| pool | sat_mass | 10.17 | 10.22 | -0.195 |
| matched | sat_mass | 10.16 | 10.17 | -0.02815 |
| both_E_S | sat_mass | 10.16 | 10.17 | -0.01645 |
| pool | bgg_abs | -22.27 | -22.27 | -0.006498 |
| matched | bgg_abs | -22.25 | -22.26 | 0.03417 |
| both_E_S | bgg_abs | -22.27 | -22.25 | -0.03642 |
| pool | log_group_luminosity | 11 | 11.01 | -0.06176 |
| matched | log_group_luminosity | 10.99 | 10.98 | 0.03063 |
| both_E_S | log_group_luminosity | 10.99 | 10.99 | 0.03778 |
| pool | velocity_dispersion | 177.4 | 208.2 | -0.268 |
| matched | velocity_dispersion | 178.4 | 232.6 | -0.4711 |
| both_E_S | velocity_dispersion | 173.6 | 224.5 | -0.4435 |

## Mass, completeness and star formation

### Mass interaction tests

[Full CSV](tables/mass_interaction_tests.csv)

| outcome | control | region | beta | p | p_Holm |
| --- | --- | --- | --- | --- | --- |
| elliptical | Control4C | primary | 0.8502 | 0.07202 | 0.3561 |
| elliptical | Control4C | flux_only | 0.7724 | 0.1585 | 0.7413 |
| elliptical | Control4C | flux_and_BGG | 1.005 | 0.2152 | 1 |
| quenched | Control4C | primary | 1.571 | 0.005451 | 0.03271 |
| quenched | Control4C | flux_only | 1.518 | 0.02398 | 0.1439 |
| quenched | Control4C | flux_and_BGG | 0.9934 | 0.3039 | 1 |
| elliptical | Control4B | primary | 0.6699 | 0.1561 | 0.3561 |
| elliptical | Control4B | flux_only | 0.3508 | 0.5221 | 1 |
| elliptical | Control4B | flux_and_BGG | 0.4169 | 0.6112 | 1 |
| quenched | Control4B | primary | 1.027 | 0.07121 | 0.3561 |
| quenched | Control4B | flux_only | 0.6938 | 0.3023 | 0.9068 |
| quenched | Control4B | flux_and_BGG | 0.1717 | 0.8559 | 1 |
| elliptical | RG4 | primary | 0.3843 | 0.5435 | 0.5435 |
| elliptical | RG4 | flux_only | -0.0774 | 0.9289 | 1 |
| elliptical | RG4 | flux_and_BGG | -1.116 | 0.4212 | 1 |
| quenched | RG4 | primary | 1.277 | 0.08058 | 0.3561 |
| quenched | RG4 | flux_only | 1.396 | 0.1483 | 0.7413 |
| quenched | RG4 | flux_and_BGG | 0.1216 | 0.93 | 1 |

### Satellite mass bins

[Full CSV](tables/satellite_mass_bins.csv)

| sample | mass_bin | outcome | n | events | estimate | low | high | hosts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CG4 | <10 | elliptical | 45 | 14 | 0.3111 | 0.1778 | 0.4545 | 34 |
| CG4 | <10 | quenched | 53 | 9 | 0.1698 | 0.07692 | 0.2857 | 37 |
| CG4 | <10 | strong_halpha | 48 | 38 | 0.7917 | 0.6667 | 0.9 | 35 |
| CG4 | <10 | Dn4000 | 53 | — | 1.309 | 1.266 | 1.372 | 37 |
| CG4 | <10 | log_Rchl_r_kpc | 50 | — | 0.3979 | 0.3515 | 0.4635 | 36 |
| CG4 | <10 | total_ssfr | 53 | — | -10.02 | -10.2 | -9.905 | 37 |
| CG4 | 10–10.5 | elliptical | 71 | 44 | 0.6197 | 0.5147 | 0.7246 | 48 |
| CG4 | 10–10.5 | quenched | 75 | 52 | 0.6933 | 0.5853 | 0.7975 | 51 |
| CG4 | 10–10.5 | strong_halpha | 72 | 18 | 0.25 | 0.1562 | 0.3507 | 49 |
| CG4 | 10–10.5 | Dn4000 | 78 | — | 1.766 | 1.68 | 1.836 | 52 |
| CG4 | 10–10.5 | log_Rchl_r_kpc | 73 | — | 0.3802 | 0.3365 | 0.4088 | 49 |
| CG4 | 10–10.5 | total_ssfr | 75 | — | -11.64 | -11.92 | -11.3 | 51 |
| CG4 | 10.5–11 | elliptical | 39 | 28 | 0.7179 | 0.575 | 0.8462 | 31 |
| CG4 | 10.5–11 | quenched | 44 | 38 | 0.8636 | 0.7619 | 0.9545 | 34 |
| CG4 | 10.5–11 | strong_halpha | 41 | 6 | 0.1463 | 0.04105 | 0.2791 | 33 |
| CG4 | 10.5–11 | Dn4000 | 44 | — | 1.892 | 1.861 | 1.914 | 34 |
| CG4 | 10.5–11 | log_Rchl_r_kpc | 43 | — | 0.5634 | 0.4786 | 0.6347 | 34 |
| CG4 | 10.5–11 | total_ssfr | 44 | — | -12.16 | -12.27 | -12.06 | 34 |
| CG4 | ≥11 | elliptical | 2 | 2 | 1 | 1 | 1 | 2 |
| CG4 | ≥11 | quenched | 2 | 2 | 1 | 1 | 1 | 2 |
| CG4 | ≥11 | strong_halpha | 2 | 0 | 0 | 0 | 0 | 2 |
| CG4 | ≥11 | Dn4000 | 2 | — | 1.958 | 1.936 | 1.979 | 2 |
| CG4 | ≥11 | log_Rchl_r_kpc | 2 | — | 0.9402 | 0.928 | 0.9523 | 2 |
| CG4 | ≥11 | total_ssfr | 2 | — | -12.49 | -12.72 | -12.27 | 2 |
| Control4B | <10 | elliptical | 337 | 64 | 0.1899 | 0.1494 | 0.2305 | 247 |
| Control4B | <10 | quenched | 394 | 70 | 0.1777 | 0.1369 | 0.2188 | 279 |
| Control4B | <10 | strong_halpha | 371 | 302 | 0.814 | 0.7705 | 0.8579 | 263 |
| Control4B | <10 | Dn4000 | 399 | — | 1.289 | 1.274 | 1.321 | 283 |
| Control4B | <10 | log_Rchl_r_kpc | 372 | — | 0.4602 | 0.4322 | 0.4834 | 269 |
| Control4B | <10 | total_ssfr | 394 | — | -9.988 | -10.03 | -9.943 | 279 |
| Control4B | 10–10.5 | elliptical | 580 | 209 | 0.3603 | 0.3201 | 0.4 | 400 |
| Control4B | 10–10.5 | quenched | 629 | 271 | 0.4308 | 0.3906 | 0.4703 | 429 |
| Control4B | 10–10.5 | strong_halpha | 604 | 350 | 0.5795 | 0.5403 | 0.6189 | 411 |
| Control4B | 10–10.5 | Dn4000 | 639 | — | 1.589 | 1.554 | 1.629 | 434 |
| Control4B | 10–10.5 | log_Rchl_r_kpc | 606 | — | 0.4828 | 0.4642 | 0.5123 | 423 |
| Control4B | 10–10.5 | total_ssfr | 629 | — | -10.79 | -10.94 | -10.68 | 429 |
| Control4B | 10.5–11 | elliptical | 696 | 334 | 0.4799 | 0.4415 | 0.5199 | 432 |
| Control4B | 10.5–11 | quenched | 780 | 571 | 0.7321 | 0.6986 | 0.7645 | 469 |
| Control4B | 10.5–11 | strong_halpha | 730 | 210 | 0.2877 | 0.2531 | 0.3242 | 444 |
| Control4B | 10.5–11 | Dn4000 | 783 | — | 1.851 | 1.835 | 1.866 | 470 |
| Control4B | 10.5–11 | log_Rchl_r_kpc | 634 | — | 0.586 | 0.5728 | 0.6029 | 408 |
| Control4B | 10.5–11 | total_ssfr | 780 | — | -11.99 | -12.04 | -11.93 | 469 |
| Control4B | ≥11 | elliptical | 159 | 94 | 0.5912 | 0.5125 | 0.6667 | 121 |
| Control4B | ≥11 | quenched | 169 | 148 | 0.8757 | 0.8235 | 0.9222 | 125 |
| Control4B | ≥11 | strong_halpha | 164 | 27 | 0.1646 | 0.1118 | 0.2222 | 123 |
| Control4B | ≥11 | Dn4000 | 169 | — | 1.926 | 1.919 | 1.945 | 125 |
| Control4B | ≥11 | log_Rchl_r_kpc | 82 | — | 0.7355 | 0.6958 | 0.7658 | 72 |
| Control4B | ≥11 | total_ssfr | 169 | — | -12.31 | -12.37 | -12.22 | 125 |
| Control4C | <10 | elliptical | 580 | 146 | 0.2517 | 0.2163 | 0.2877 | 414 |
| Control4C | <10 | quenched | 676 | 178 | 0.2633 | 0.2283 | 0.2999 | 458 |
| Control4C | <10 | strong_halpha | 641 | 478 | 0.7457 | 0.708 | 0.7819 | 438 |
| Control4C | <10 | Dn4000 | 684 | — | 1.349 | 1.321 | 1.362 | 463 |
| Control4C | <10 | log_Rchl_r_kpc | 640 | — | 0.424 | 0.4051 | 0.4472 | 446 |
| Control4C | <10 | total_ssfr | 676 | — | -10.11 | -10.2 | -10.05 | 458 |
| Control4C | 10–10.5 | elliptical | 689 | 315 | 0.4572 | 0.4209 | 0.4964 | 481 |
| Control4C | 10–10.5 | quenched | 759 | 432 | 0.5692 | 0.5326 | 0.6047 | 518 |
| Control4C | 10–10.5 | strong_halpha | 722 | 335 | 0.464 | 0.4265 | 0.5014 | 497 |
| Control4C | 10–10.5 | Dn4000 | 766 | — | 1.684 | 1.655 | 1.716 | 522 |
| Control4C | 10–10.5 | log_Rchl_r_kpc | 725 | — | 0.4195 | 0.4028 | 0.4453 | 503 |
| Control4C | 10–10.5 | total_ssfr | 759 | — | -11.33 | -11.45 | -11.18 | 518 |
| Control4C | 10.5–11 | elliptical | 416 | 213 | 0.512 | 0.4647 | 0.5598 | 326 |
| Control4C | 10.5–11 | quenched | 480 | 359 | 0.7479 | 0.7074 | 0.7876 | 371 |
| Control4C | 10.5–11 | strong_halpha | 445 | 122 | 0.2742 | 0.2325 | 0.3183 | 344 |
| Control4C | 10.5–11 | Dn4000 | 482 | — | 1.859 | 1.841 | 1.875 | 372 |
| Control4C | 10.5–11 | log_Rchl_r_kpc | 409 | — | 0.5643 | 0.5488 | 0.5843 | 317 |
| Control4C | 10.5–11 | total_ssfr | 480 | — | -12.01 | -12.06 | -11.96 | 371 |
| Control4C | ≥11 | elliptical | 59 | 36 | 0.6102 | 0.4912 | 0.7241 | 56 |
| Control4C | ≥11 | quenched | 62 | 54 | 0.871 | 0.7832 | 0.9508 | 58 |
| Control4C | ≥11 | strong_halpha | 61 | 11 | 0.1803 | 0.09524 | 0.2759 | 57 |
| Control4C | ≥11 | Dn4000 | 62 | — | 1.912 | 1.888 | 1.941 | 58 |
| Control4C | ≥11 | log_Rchl_r_kpc | 33 | — | 0.7141 | 0.6826 | 0.7623 | 31 |
| Control4C | ≥11 | total_ssfr | 62 | — | -12.27 | -12.37 | -12.08 | 58 |
| RG4 | <10 | elliptical | 64 | 10 | 0.1562 | 0.07576 | 0.2424 | 43 |
| RG4 | <10 | quenched | 77 | 13 | 0.1688 | 0.08747 | 0.2598 | 48 |
| RG4 | <10 | strong_halpha | 73 | 63 | 0.863 | 0.7763 | 0.9412 | 46 |
| RG4 | <10 | Dn4000 | 77 | — | 1.292 | 1.261 | 1.34 | 48 |
| RG4 | <10 | log_Rchl_r_kpc | 73 | — | 0.4205 | 0.3553 | 0.5062 | 47 |
| RG4 | <10 | total_ssfr | 77 | — | -9.936 | -10.1 | -9.8 | 48 |
| RG4 | 10–10.5 | elliptical | 51 | 15 | 0.2941 | 0.1627 | 0.4211 | 36 |
| RG4 | 10–10.5 | quenched | 57 | 26 | 0.4561 | 0.3158 | 0.6 | 40 |
| RG4 | 10–10.5 | strong_halpha | 54 | 32 | 0.5926 | 0.4717 | 0.7255 | 37 |
| RG4 | 10–10.5 | Dn4000 | 57 | — | 1.561 | 1.501 | 1.691 | 40 |
| RG4 | 10–10.5 | log_Rchl_r_kpc | 57 | — | 0.4422 | 0.3558 | 0.5089 | 40 |
| RG4 | 10–10.5 | total_ssfr | 57 | — | -10.9 | -11.43 | -10.54 | 40 |
| RG4 | 10.5–11 | elliptical | 26 | 14 | 0.5385 | 0.32 | 0.75 | 23 |
| RG4 | 10.5–11 | quenched | 31 | 23 | 0.7419 | 0.5625 | 0.9 | 25 |
| RG4 | 10.5–11 | strong_halpha | 27 | 8 | 0.2963 | 0.1429 | 0.4483 | 23 |
| RG4 | 10.5–11 | Dn4000 | 31 | — | 1.832 | 1.75 | 1.869 | 25 |
| RG4 | 10.5–11 | log_Rchl_r_kpc | 30 | — | 0.5942 | 0.5395 | 0.6861 | 24 |
| RG4 | 10.5–11 | total_ssfr | 31 | — | -11.82 | -12.18 | -11.6 | 25 |
| RG4 | ≥11 | elliptical | 1 | 0 | 0 | 0 | 0 | 1 |
| RG4 | ≥11 | quenched | 1 | 1 | 1 | 1 | 1 | 1 |
| RG4 | ≥11 | strong_halpha | 1 | 0 | 0 | 0 | 0 | 1 |
| RG4 | ≥11 | Dn4000 | 1 | — | 1.984 | 1.984 | 1.984 | 1 |
| RG4 | ≥11 | log_Rchl_r_kpc | 0 | — | — | — | — | 0 |
| RG4 | ≥11 | total_ssfr | 1 | — | -12.56 | -12.56 | -12.56 | 1 |

### Completeness flow

[Full CSV](tables/completeness_flow.csv)

| sample | bin | all | retained | flux_only | threshold_median | quartets | hosts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CG4 | <10 | 53 | 1 | 28 | 9.994 | 1 | 1 |
| CG4 | 10–10.5 | 78 | 55 | 78 | 10.09 | 41 | 41 |
| CG4 | 10.5–11 | 44 | 44 | 44 | 10.1 | 35 | 34 |
| CG4 | ≥11 | 2 | 2 | 2 | 10.36 | 2 | 2 |
| CG4 | — | 9 | 0 | 0 | 10.24 | 0 | 0 |
| Control4B | <10 | 399 | 15 | 212 | 10 | 15 | 15 |
| Control4B | 10–10.5 | 640 | 556 | 640 | 10.05 | 403 | 403 |
| Control4B | 10.5–11 | 783 | 783 | 783 | 10.1 | 470 | 470 |
| Control4B | ≥11 | 169 | 169 | 169 | 10.2 | 125 | 125 |
| Control4B | — | 103 | 0 | 0 | 10.09 | 0 | 0 |
| Control4C | <10 | 684 | 23 | 364 | 10.02 | 22 | 22 |
| Control4C | 10–10.5 | 766 | 567 | 766 | 10.1 | 430 | 430 |
| Control4C | 10.5–11 | 482 | 480 | 482 | 10.1 | 372 | 372 |
| Control4C | ≥11 | 62 | 62 | 62 | 10.16 | 58 | 58 |
| Control4C | — | 115 | 0 | 0 | 10.12 | 0 | 0 |
| RG4 | <10 | 77 | 4 | 23 | 9.969 | 4 | 4 |
| RG4 | 10–10.5 | 57 | 52 | 57 | 9.982 | 37 | 37 |
| RG4 | 10.5–11 | 31 | 31 | 31 | 9.967 | 25 | 25 |
| RG4 | ≥11 | 1 | 1 | 1 | 10.01 | 1 | 1 |
| RG4 | — | 2 | 0 | 0 | 9.976 | 0 | 0 |

### Mass marginal contrasts

[Full CSV](tables/mass_marginal_contrasts.csv)

| outcome | control | region | mass | cg_hosts | control_hosts | difference | low | high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| elliptical | Control4C | primary | 9.75 | 29 | 345 | 6.14 | -4.083 | 16.36 |
| elliptical | Control4C | primary | 10.25 | 48 | 474 | 17.48 | 8.607 | 26.35 |
| elliptical | Control4C | primary | 10.75 | 31 | 309 | 23.83 | 12.12 | 35.54 |
| elliptical | Control4C | primary | 11 | 13 | 133 | 23.79 | 12.42 | 35.17 |
| elliptical | Control4C | flux_and_BGG | 10.25 | 36 | 390 | 16.06 | 2.743 | 29.38 |
| elliptical | Control4C | flux_and_BGG | 10.75 | 31 | 309 | 25.81 | 12.78 | 38.85 |
| elliptical | Control4C | flux_and_BGG | 11 | 13 | 133 | 29.19 | 13.04 | 45.33 |
| quenched | Control4C | primary | 9.75 | 32 | 383 | -6.03 | -15.15 | 3.088 |
| quenched | Control4C | primary | 10.25 | 51 | 508 | 10.82 | 1.494 | 20.14 |
| quenched | Control4C | primary | 10.75 | 34 | 352 | 13.99 | 7.705 | 20.28 |
| quenched | Control4C | primary | 11 | 14 | 150 | 10.49 | 6.271 | 14.7 |
| quenched | Control4C | flux_and_BGG | 10.25 | 40 | 418 | 11.82 | -1.681 | 25.32 |
| quenched | Control4C | flux_and_BGG | 10.75 | 34 | 352 | 13.65 | 5.204 | 22.1 |
| quenched | Control4C | flux_and_BGG | 11 | 14 | 150 | 11.23 | 3.655 | 18.81 |
| elliptical | Control4B | primary | 9.75 | 29 | 215 | 13.52 | 3.248 | 23.79 |
| elliptical | Control4B | primary | 10.25 | 48 | 396 | 24.67 | 15.79 | 33.56 |
| elliptical | Control4B | primary | 10.75 | 31 | 417 | 29.19 | 17.73 | 40.65 |
| elliptical | Control4B | primary | 11 | 13 | 225 | 28.29 | 16.91 | 39.68 |
| elliptical | Control4B | flux_and_BGG | 10.25 | 36 | 365 | 24.87 | 11.62 | 38.12 |
| elliptical | Control4B | flux_and_BGG | 10.75 | 31 | 417 | 27.54 | 14.56 | 40.53 |
| elliptical | Control4B | flux_and_BGG | 11 | 13 | 225 | 27.14 | 11.22 | 43.06 |
| quenched | Control4B | primary | 9.75 | 32 | 239 | 5.701 | -3.3 | 14.7 |
| quenched | Control4B | primary | 10.25 | 51 | 424 | 20.68 | 11.13 | 30.22 |
| quenched | Control4B | primary | 10.75 | 34 | 452 | 16.21 | 10.07 | 22.34 |
| quenched | Control4B | primary | 11 | 14 | 245 | 11 | 6.846 | 15.15 |
| quenched | Control4B | flux_and_BGG | 10.25 | 40 | 391 | 20.95 | 7.442 | 34.46 |
| quenched | Control4B | flux_and_BGG | 10.75 | 34 | 452 | 12.82 | 4.876 | 20.76 |
| quenched | Control4B | flux_and_BGG | 11 | 14 | 245 | 8.175 | 1.402 | 14.95 |
| elliptical | RG4 | primary | 9.75 | 27 | 36 | 16.71 | 4.092 | 29.33 |
| elliptical | RG4 | primary | 10.25 | 45 | 36 | 26.03 | 12.25 | 39.81 |
| elliptical | RG4 | primary | 10.75 | 29 | 23 | 27.43 | 6.499 | 48.36 |
| elliptical | RG4 | primary | 11 | 9 | 13 | 23.46 | 0.7808 | 46.13 |
| elliptical | RG4 | flux_and_BGG | 10.25 | 34 | 33 | 35.49 | 15.7 | 55.28 |
| elliptical | RG4 | flux_and_BGG | 10.75 | 29 | 23 | 21.25 | -4.327 | 46.83 |
| elliptical | RG4 | flux_and_BGG | 11 | 9 | 13 | 12.15 | -19.34 | 43.65 |
| quenched | RG4 | primary | 9.75 | 30 | 40 | 1.986 | -10.56 | 14.53 |
| quenched | RG4 | primary | 10.25 | 48 | 39 | 18.08 | 2.293 | 33.86 |
| quenched | RG4 | primary | 10.75 | 32 | 25 | 16.73 | 1.906 | 31.54 |
| quenched | RG4 | primary | 11 | 11 | 15 | 12.16 | -0.9579 | 25.27 |
| quenched | RG4 | flux_and_BGG | 10.25 | 38 | 36 | 24.09 | 3.924 | 44.26 |
| quenched | RG4 | flux_and_BGG | 10.75 | 32 | 25 | 15.53 | -4.098 | 35.16 |
| quenched | RG4 | flux_and_BGG | 11 | 11 | 15 | 10.71 | -9.643 | 31.05 |

### Quenching ceiling

[Full CSV](tables/quenching_ceiling.csv)

| control | bin | n | counterfactual_control | implied_delta_pp_for_OR_1_5 | difference | low | high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Control4C | pooled | 174 | 0.5176 | 8.047 | 6.29 | -0.1028 | 12.68 |
| Control4C | <10 | 53 | 0.2628 | 7.957 | 6.099 | -0.4395 | 12.64 |
| Control4C | 10–10.5 | 75 | 0.5365 | 9.336 | 7.319 | -0.05988 | 14.7 |
| Control4C | 10.5–11 | 44 | 0.7761 | 6.149 | 4.911 | 0.1924 | 9.63 |
| Control4C | ≥11 | 2 | 0.8719 | 3.838 | 3.086 | 0.09048 | 6.082 |
| Control4B | pooled | 174 | 0.4328 | 7.498 | 14.76 | 8.229 | 21.3 |
| Control4B | <10 | 53 | 0.1577 | 5.719 | 12.5 | 6.107 | 18.89 |
| Control4B | 10–10.5 | 75 | 0.4361 | 9.238 | 18.11 | 10.1 | 26.13 |
| Control4B | 10.5–11 | 44 | 0.741 | 6.787 | 12.05 | 7.513 | 16.59 |
| Control4B | ≥11 | 2 | 0.8228 | 5.023 | 8.731 | 5.309 | 12.15 |
| RG4 | pooled | 174 | 0.4687 | 7.354 | 11.18 | -0.2146 | 22.57 |
| RG4 | <10 | 53 | 0.1678 | 5.969 | 9.616 | 0.1991 | 19.03 |
| RG4 | 10–10.5 | 75 | 0.4791 | 9.402 | 14.21 | -0.231 | 28.65 |
| RG4 | 10.5–11 | 44 | 0.7936 | 5.735 | 8.204 | -0.9983 | 17.41 |
| RG4 | ≥11 | 2 | 0.9055 | 2.914 | 4.095 | -1.749 | 9.939 |

### Sf mass adjusted

[Full CSV](tables/sf_mass_adjusted.csv)

| outcome | selection | aperture | n | cg_n | control_n | effect_difference | effect_low | effect_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strong_halpha | available | False | 794 | 72 | 722 | -22.35 | -32.76 | -11.94 |
| strong_halpha | same_aperture_complete | False | 787 | 69 | 718 | -21.47 | -32.03 | -10.91 |
| strong_halpha | same_aperture_complete | True | 787 | 69 | 718 | -17.77 | -27.14 | -8.393 |
| strong_exact | available | False | 844 | 78 | 766 | -19.96 | -30.49 | -9.427 |
| strong_exact | same_aperture_complete | False | 787 | 69 | 718 | -21.47 | -32.03 | -10.91 |
| strong_exact | same_aperture_complete | True | 787 | 69 | 718 | -17.77 | -27.14 | -8.393 |
| quenched | available | False | 834 | 75 | 759 | 12.93 | 1.554 | 24.3 |
| quenched | same_aperture_complete | False | 787 | 69 | 718 | 15.73 | 4.246 | 27.21 |
| quenched | same_aperture_complete | True | 787 | 69 | 718 | 11.8 | 1.807 | 21.8 |
| total_ssfr | available | False | 834 | 75 | 759 | -0.2117 | -0.4162 | -0.00721 |
| total_ssfr | same_aperture_complete | False | 787 | 69 | 718 | -0.2759 | -0.4723 | -0.07956 |
| total_ssfr | same_aperture_complete | True | 787 | 69 | 718 | -0.2092 | -0.3768 | -0.04171 |
| Dn4000 | available | False | 844 | 78 | 766 | 0.04604 | -0.01454 | 0.1066 |
| Dn4000 | same_aperture_complete | False | 787 | 69 | 718 | 0.05887 | 0.0002302 | 0.1175 |
| Dn4000 | same_aperture_complete | True | 787 | 69 | 718 | 0.03988 | -0.01089 | 0.09065 |
| recent_disagreement | available | False | 787 | 69 | 718 | -6.805 | -11.4 | -2.21 |
| recent_disagreement | same_aperture_complete | False | 787 | 69 | 718 | -6.805 | -11.4 | -2.21 |
| recent_disagreement | same_aperture_complete | True | 787 | 69 | 718 | -6.013 | -10.52 | -1.508 |

## Within-host geometry and fits

### Within host models

[Full CSV](tables/within_host_models.csv)

| spec | n_complete | hosts | n | outcome_informative | both_informative | cg | noncg | eligible_range_overlap_hosts | OR | low | high | p | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_no_radius | 791 | 53 | 791 | 53 | 52 | 192 | 599 | 6 | 1.902 | 1.227 | 2.946 | 0.004016 | ok |
| original_radius | 791 | 53 | 791 | 53 | 52 | 192 | 599 | 6 | 1.468 | 0.9189 | 2.346 | 0.1082 | ok |
| recomputed_BGG_radius | 791 | 53 | 791 | 53 | 52 | 192 | 599 | 6 | 1.468 | 0.9189 | 2.346 | 0.1082 | ok |
| centroid_sensitivity | 791 | 53 | 791 | 53 | 52 | 192 | 599 | 6 | 1.629 | 1.031 | 2.573 | 0.0364 | ok |
| log_BGG_radius | 791 | 53 | 791 | 53 | 52 | 192 | 599 | 6 | 1.069 | 0.6458 | 1.77 | 0.7946 | ok |
| eligible_nonmembers_only | 470 | 53 | 453 | 49 | 42 | 178 | 275 | 6 | 1.822 | 1.022 | 3.249 | 0.04191 | ok |
| eligible_satellites_only | 418 | 53 | 397 | 46 | 42 | 126 | 271 | 6 | 1.965 | 1.062 | 3.637 | 0.03143 | ok |
| exact_hosts_all_members | 552 | 43 | 552 | 43 | 42 | 154 | 398 | 0 | 1.365 | 0.7742 | 2.408 | 0.2819 | ok |
| exact_hosts_eligible_satellites | 291 | 43 | 273 | 37 | 33 | 99 | 174 | 0 | 1.951 | 0.8791 | 4.331 | 0.1003 | ok |
| nonidentical_hosts_eligible_satellites | 99 | 8 | 96 | 7 | 7 | 19 | 77 | 5 | 2.502 | 0.7076 | 8.849 | 0.1546 | ok |
| eligible_satellites_within_50_kpc | 32 | 9 | 17 | 4 | 4 | 8 | 9 | 1 | — | — | — | — | insufficient support |
| eligible_satellites_within_25_kpc | 21 | 7 | 14 | 4 | 4 | 7 | 7 | 1 | — | — | — | — | insufficient support |

## Catalogue checks

### Hmcg control counts

[Full CSV](tables/hmcg_control_counts.csv)

| sample | kind | galaxies | quartets | hosts | HMCGs |
| --- | --- | --- | --- | --- | --- |
| Control4C | any | 185 | 58 | 58 | 56 |
| Control4C | N_ne_4 | 147 | 43 | 43 | 41 |
| Control4C | flagged | 14 | 5 | 5 | 5 |
| Control4C | N_eq_4 | 38 | 15 | 15 | 15 |
| Control4B | any | 166 | 67 | 67 | 65 |
| Control4B | N_ne_4 | 129 | 46 | 46 | 44 |
| Control4B | flagged | 11 | 5 | 5 | 5 |
| Control4B | N_eq_4 | 37 | 21 | 21 | 21 |
| RG4 | any | 4 | 2 | 2 | 2 |
| RG4 | N_ne_4 | 3 | 1 | 1 | 1 |
| RG4 | flagged | 0 | 0 | 0 | 0 |
| RG4 | N_eq_4 | 1 | 1 | 1 | 1 |
| all_controls | any | 221 | 127 | 67 | 65 |
| all_controls | N_ne_4 | 173 | 90 | 46 | 44 |
| all_controls | flagged | 17 | 10 | 5 | 5 |
| all_controls | N_eq_4 | 48 | 37 | 21 | 21 |

### Hmcg isolation summary

[Full CSV](tables/hmcg_isolation_summary.csv)

| stage | in_overlap | n | mu_median | ratio_median | n_pass_isolation | n_pass_SB | n_pass_appearance | n_pass_velocity | n_Tempel_all |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CG4 | False | 14 | 24.16 | 4.727 | 13 | 14 | 14 | 14 | 14 |
| CG4 | True | 48 | 25.58 | 3.592 | 42 | 47 | 45 | 48 | 48 |
| Excluded Lim 3688 | False | 1 | 27.17 | 2.65 | 0 | 0 | 1 | 0 | 1 |
| Excluded by CG4 overlap | False | 21 | 24.71 | 3.449 | 12 | 15 | 21 | 21 | 21 |
| Excluded by CG4 overlap | True | 40 | 25.63 | 3.79 | 35 | 39 | 40 | 40 | 40 |
| Retained Control4C | False | 516 | 27.53 | 1.517 | 83 | 52 | 501 | 512 | 516 |
| Retained Control4C | True | 187 | 25.62 | 1.731 | 21 | 152 | 181 | 184 | 187 |

### Boss sfr summary

[Full CSV](tables/boss_sfr_summary.csv)

| sample | missing_BOSS | alternative_rows | multiple_candidates | known_repeats |
| --- | --- | --- | --- | --- |
| CG4 | 14 | 0 | 0 | 1 |
| Control4B | 149 | 0 | 0 | 9 |
| Control4C | 164 | 0 | 0 | 8 |
| RG4 | 5 | 0 | 0 | 0 |

### Paper match denominators

[Full CSV](tables/paper_match_denominators.csv)

| population | stage | n_pairs | candidate_edges | complete_cg | cg_with_candidate | cg_hosts | control_hosts | blocks | effective_blocks | estimate | low | high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | matched | 234 | 24092 | 234 | 234 | 60 | 204 | — | — | — | — | — |
| all | both_E_S | 180 | 24092 | 234 | 234 | 58 | 161 | 33 | 14.29 | 0.1389 | 0.04969 | 0.2394 |
| satellites | matched | 177 | 19467 | 177 | 177 | 60 | 158 | — | — | — | — | — |
| satellites | both_E_S | 130 | 19467 | 177 | 177 | 57 | 118 | 33 | 15.17 | 0.1923 | 0.07874 | 0.3305 |

### Gz1 dr7 recovery summary

[Full CSV](tables/gz1_dr7_recovery_summary.csv)

| sample | baseline_labels | verified_both_labels | matching_labels | verified_bridge_among_absent | E_recovered | S_recovered | expanded_labels |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CG4 | 210 | 210 | 210 | 24 | 6 | 10 | 226 |
| Control4B | 2372 | 2362 | 2362 | 233 | 55 | 59 | 2486 |
| Control4C | 2347 | 2339 | 2339 | 259 | 50 | 60 | 2457 |
| RG4 | 189 | 186 | 186 | 28 | 6 | 6 | 201 |

### Missing morphology bounds

[Full CSV](tables/missing_morphology_bounds.csv)

| population | sample | valid | n | observed | lower | upper |
| --- | --- | --- | --- | --- | --- | --- |
| all | CG4 | 210 | 248 | 0.5905 | 0.5 | 0.6532 |
| all | Control4B | 2372 | 2792 | 0.4663 | 0.3961 | 0.5466 |
| all | Control4C | 2347 | 2812 | 0.4764 | 0.3976 | 0.5629 |
| all | RG4 | 189 | 224 | 0.3175 | 0.2679 | 0.4241 |
| all | CG4 − Control4C | — | — | 0.1141 | -0.06294 | 0.2556 |
| all | CG4 − Control4B | — | — | 0.1242 | -0.04656 | 0.2571 |
| all | CG4 − RG4 | — | — | 0.273 | 0.07589 | 0.3854 |
| satellites | CG4 | 157 | 186 | 0.5605 | 0.4731 | 0.629 |
| satellites | Control4B | 1772 | 2094 | 0.3956 | 0.3348 | 0.4885 |
| satellites | Control4C | 1744 | 2109 | 0.4071 | 0.3367 | 0.5097 |
| satellites | RG4 | 142 | 168 | 0.2746 | 0.2321 | 0.3869 |
| satellites | CG4 − Control4C | — | — | 0.1534 | -0.0366 | 0.2924 |
| satellites | CG4 − Control4B | — | — | 0.1649 | -0.01542 | 0.2943 |
| satellites | CG4 − RG4 | — | — | 0.2859 | 0.08621 | 0.3969 |
| satellites_compactness_window | CG4 | 122 | 144 | 0.5328 | 0.4514 | 0.6042 |
| satellites_compactness_window | Control4B | 103 | 135 | 0.4175 | 0.3185 | 0.5556 |
| satellites_compactness_window | Control4C | 453 | 561 | 0.4901 | 0.3957 | 0.5882 |
| satellites_compactness_window | RG4 | 2 | 3 | 0 | 0 | 0.3333 |
| satellites_compactness_window | CG4 − Control4C | — | — | 0.04272 | -0.1368 | 0.2084 |
| satellites_compactness_window | CG4 − Control4B | — | — | 0.1153 | -0.1042 | 0.2856 |
| satellites_compactness_window | CG4 − RG4 | — | — | 0.5328 | 0.1181 | 0.6042 |

### Colour selection sensitivity

[Full CSV](tables/colour_selection_sensitivity.csv)

| control | restriction | outcome | cg_n | control_n | effect_difference | effect_low | effect_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Control4C | available | elliptical | 157 | 1744 | 15.37 | 7.969 | 22.77 |
| Control4C | available | u_minus_r | 163 | 1869 | -0.08627 | -0.1701 | -0.002436 |
| Control4C | available | g_minus_r | 163 | 1869 | -0.001597 | -0.0224 | 0.01921 |
| Control4C | colour_retained | elliptical | 97 | 975 | 12.87 | 3.645 | 22.1 |
| Control4C | colour_retained | u_minus_r | 102 | 1052 | -0.07926 | -0.1948 | 0.03629 |
| Control4C | colour_retained | g_minus_r | 102 | 1052 | -0.008121 | -0.03887 | 0.02263 |
| Control4B | available | elliptical | 157 | 1772 | 22.02 | 14.51 | 29.54 |
| Control4B | available | u_minus_r | 163 | 1869 | 0.03623 | -0.04786 | 0.1203 |
| Control4B | available | g_minus_r | 163 | 1869 | 0.02924 | 0.005485 | 0.053 |
| Control4B | colour_retained | elliptical | 97 | 775 | 20.6 | 10.86 | 30.33 |
| Control4B | colour_retained | u_minus_r | 102 | 831 | 0.05206 | -0.09481 | 0.1989 |
| Control4B | colour_retained | g_minus_r | 102 | 831 | 0.02709 | -0.01695 | 0.07113 |
| RG4 | available | elliptical | 157 | 142 | 23.69 | 12.12 | 35.26 |
| RG4 | available | u_minus_r | 163 | 155 | 0.09716 | 0.01189 | 0.1824 |
| RG4 | available | g_minus_r | 163 | 155 | 0.03482 | 0.00641 | 0.06322 |
| RG4 | colour_retained | elliptical | 97 | 82 | 21.96 | 5.527 | 38.4 |
| RG4 | colour_retained | u_minus_r | 102 | 92 | 0.08518 | -0.03467 | 0.205 |
| RG4 | colour_retained | g_minus_r | 102 | 92 | 0.03238 | -0.00607 | 0.07082 |

### Paper1 distance audit

[Full CSV](tables/paper1_distance_audit.csv)

| sample | n | raw_distance_residual_DA | raw_distance_residual_DL | true_percent_decrease_median | rho_tc_Mvir_raw | rho_tc_Mvir_DA |
| --- | --- | --- | --- | --- | --- | --- |
| CG4 | 62 | 10.59 | 0.1321 | 6.945 | -0.6826 | -0.6842 |
| Control4C | 703 | 19.94 | 0.2691 | 6.716 | -0.4589 | -0.4578 |
| Control4B | 698 | 30.66 | 0.4129 | 6.689 | -0.3982 | -0.3972 |
| RG4 | 56 | 29.54 | 0.3977 | 7.543 | -0.5598 | -0.5576 |
