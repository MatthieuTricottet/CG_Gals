# Phase 0 — Inventory (auto-generated from outputs/metallicity_scoping.json)

Generated 2026-09-12T07:25:04+00:00 on branch `submission-qc-float-fix` @ `7367e51`; git status baseline: `['M output/paper/paper.aux', ' M output/paper/paper.pdf', ' M output/paper/paper.tex', ' M src/paper_template/paper_template.tex', ' M tests/test_size_render_smoke.py', '?? exploration/', '?? referee/T10_ds18_morphology.py', '?? referee/values/T10.json', '?? tmp/']`

## 1. Local tables

| path | format | rows | n_cols | columns |
|---|---|---:|---:|---|
| `data/CG4_Gals.csv` | csv | 312 | 17 | `objid, specobjid, Group, RA, Dec, M_r, Lum, z, dist2BGG, lgm, sfr, sSFR, rank_dist, rank_M, RA_BGG, Dec_BGG, M_BGG` |
| `data/CG4_Groups.csv` | csv | 78 | 36 | `Group, Lum_BGG, Lum_group, FracLumBGG, z_group, DeltaR12, NbGal, RA_BGG, Dec_BGG, RA_Bary, Dec_Bary, Radius_Bary_arcmin, Offset_Bary, V_BGG, V_moy, Vdisp, Voffset, size_Group_Bary_kpc, M_group, M_virial, M_virial_over_L, t_cr, BGG_SFRcategory, all_SFR, Prop_M_Sat, Prop_M_Tot, Prop_G_Sat, Prop_G_Tot, Prop_Q_Sat, Prop_Q_Tot, dom, Misfit_Bary, Vmisfit, lMass_200, r_200_kpc, Class` |
| `data/RG4_Gals.csv` | csv | 224 | 18 | `objid, specobjid, Group, RA, Dec, M_r, Lum, z, dist2BGG, lgm, sfr, sSFR, rank_dist, BGG_ID, rank_M, RA_BGG, Dec_BGG, M_BGG` |
| `data/RG4_Groups.csv` | csv | 56 | 34 | `Group, Lum_BGG, Lum_group, FracLumBGG, z_group, DeltaR12, NbGal, RA_BGG, Dec_BGG, RA_Bary, Dec_Bary, Radius_Bary_arcmin, Offset_Bary, V_BGG, V_moy, Vdisp, Voffset, size_Group_Bary_kpc, M_group, M_virial, M_virial_over_L, t_cr, BGG_SFRcategory, all_SFR, Prop_M_Sat, Prop_M_Tot, Prop_G_Sat, Prop_G_Tot, Prop_Q_Sat, Prop_Q_Tot, Misfit_Bary, Vmisfit, lMass_200, r_200_kpc` |
| `data/Control4B_Gals.csv` | csv | 2796 | 18 | `objid, specobjid, Group, RA, Dec, M_r, Lum, z, dist2BGG, lgm, sfr, sSFR, rank_dist, BGG_ID, rank_M, RA_BGG, Dec_BGG, M_BGG` |
| `data/Control4B_Groups.csv` | csv | 699 | 33 | `Group, Lum_BGG, Lum_group, FracLumBGG, z_group, DeltaR12, NbGal, RA_BGG, Dec_BGG, RA_Bary, Dec_Bary, Radius_Bary_arcmin, Offset_Bary, V_BGG, V_moy, Vdisp, Voffset, size_Group_Bary_kpc, M_group, M_virial, M_virial_over_L, t_cr, all_SFR, Prop_M_Sat, Prop_M_Tot, Prop_G_Sat, Prop_G_Tot, Prop_Q_Sat, Prop_Q_Tot, Misfit_Bary, Vmisfit, lMass_200, r_200_kpc` |
| `data/Control4C_Gals.csv` | csv | 2816 | 18 | `objid, specobjid, Group, RA, Dec, M_r, z, Lum, rank_M, RA_BGG, Dec_BGG, M_BGG, dist2BGG, lgm, sfr, sSFR, rank_dist, rank_M_parent` |
| `data/Control4C_Groups.csv` | csv | 704 | 34 | `Group, Lum_BGG, Lum_group, FracLumBGG, z_group, DeltaR12, NbGal, RA_BGG, Dec_BGG, RA_Bary, Dec_Bary, Radius_Bary_arcmin, Offset_Bary, V_BGG, V_moy, Vdisp, Voffset, size_Group_Bary_kpc, M_group, M_virial, M_virial_over_L, t_cr, BGG_SFRcategory, all_SFR, Prop_M_Sat, Prop_M_Tot, Prop_G_Sat, Prop_G_Tot, Prop_Q_Sat, Prop_Q_Tot, Misfit_Bary, Vmisfit, lMass_200, r_200_kpc` |
| `data/PC_Gals.csv` | csv | 11087 | 26 | `Id, objid, specobjid, Group, RA, Dec, M_r, z, zobs, Yang_z_CMB_group, BGG_ID, Yang_logM, logM_180, rmag, Lum, rank_M, RA_BGG, Dec_BGG, M_BGG, dist2BGG, lgm_tot_p50, sfr_tot_p50, sSFR, SFRexcess, SFRcategory, rank_dist` |
| `data/PC_Groups.csv` | csv | 765 | 32 | `Group, Lum_BGG, Lum_group, FracLumBGG, z_group, DeltaR12, NbGal, RA_BGG, Dec_BGG, RA_Bary, Dec_Bary, Radius_Bary_arcmin, Offset_Bary, V_BGG, V_moy, Vdisp, Voffset, size_Group_Bary_kpc, M_group, M_virial, M_virial_over_L, t_cr, BGG_SFRcategory, all_SFR, Prop_M_Sat, Prop_M_Tot, Prop_G_Sat, Prop_G_Tot, Prop_Q_Sat, Prop_Q_Tot, lMass_200, r_200_kpc` |
| `data/sdss_size_columns.csv` | csv | 4830 | 9 | `objid, specObjID, dr7objid, petroR50_r, petroR50Err_r, petroR90_r, petroR90Err_r, petroRad_r, psfWidth_r` |
| `data/simard2011_subset.csv` | csv | 4573 | 11 | `dr7objid, z, Sp, Scale, Rhlr, Rchl_r, e, ng, e_ng, rg2d, PpS` |
| `audit/identity_catalog.csv` | csv | 11123 | 25 | `objid, in_CG4, CG4_group, CG4_rank_M, CG4_rank_dist, in_Control4B, Control4B_group, Control4B_rank_M, Control4B_rank_dist, in_Control4C, Control4C_group, Control4C_rank_M, Control4C_rank_dist, in_RG4, RG4_group, RG4_rank_M, RG4_rank_dist, in_PC, PC_group, PC_rank_M, PC_rank_dist, CG4_class, CG4_host_lim_group, lim_group, physical_group` |
| `data/SDSS(L) galaxy.dat` | ascii .dat (Lim+17 raw) | 586025 | 22 | `(see .dat header; 22 commented column lines)` |
| `data/SDSS(L) group.dat` | ascii .dat (Lim+17 raw) | 446496 | 10 | `(see .dat header; 10 commented column lines)` |
| `data/processed_sample.pkl` [CG4_Gals] | pickle dict of DataFrames | 248 | 25 | `objid, specobjid, Group, RA, Dec, M_r, Lum, z, dist2BGG, lgm, sfr, sSFR, rank_dist, rank_M, RA_BGG, Dec_BGG, M_BGG, p_E, p_S, is_dominated, morphology, sSFR_status, sSFR_MS_offset, MS_res, sSFR_excess` |
| `data/processed_sample.pkl` [Control4B_Gals] | pickle dict of DataFrames | 2792 | 26 | `objid, specobjid, Group, RA, Dec, M_r, Lum, z, dist2BGG, lgm, sfr, sSFR, rank_dist, BGG_ID, rank_M, RA_BGG, Dec_BGG, M_BGG, p_E, p_S, is_dominated, morphology, sSFR_status, sSFR_MS_offset, MS_res, sSFR_excess` |
| `data/processed_sample.pkl` [Control4C_Gals] | pickle dict of DataFrames | 2812 | 26 | `objid, specobjid, Group, RA, Dec, M_r, z, Lum, rank_M, RA_BGG, Dec_BGG, M_BGG, dist2BGG, lgm, sfr, sSFR, rank_dist, rank_M_parent, p_E, p_S, is_dominated, morphology, sSFR_status, sSFR_MS_offset, MS_res, sSFR_excess` |
| `data/processed_sample.pkl` [RG4_Gals] | pickle dict of DataFrames | 224 | 26 | `objid, specobjid, Group, RA, Dec, M_r, Lum, z, dist2BGG, lgm, sfr, sSFR, rank_dist, BGG_ID, rank_M, RA_BGG, Dec_BGG, M_BGG, p_E, p_S, is_dominated, morphology, sSFR_status, sSFR_MS_offset, MS_res, sSFR_excess` |
| `data/processed_sample.pkl` [CG4_Groups] | pickle dict of DataFrames | 62 | 45 | `Group, Lum_BGG, Lum_group, FracLumBGG, z_group, DeltaR12, NbGal, RA_BGG, Dec_BGG, RA_Bary, Dec_Bary, Radius_Bary_arcmin, Offset_Bary, V_BGG, V_moy, Vdisp, Voffset, size_Group_Bary_kpc, M_group, M_virial, M_virial_over_L, t_cr, BGG_SFRcategory, all_SFR, Prop_M_Sat, Prop_M_Tot, Prop_G_Sat, Prop_G_Tot, Prop_Q_Sat, Prop_Q_Tot, dom, Misfit_Bary, Vmisfit, lMass_200, r_200_kpc, Class, is_dominated, E_frac_NoU_NoBGG, S_frac_NoU_NoBGG, E_frac_NoU, S_frac_NoU, E_frac_NoBGG, S_frac_NoBGG, E_frac, S_frac` |
| `data/processed_sample.pkl` [Control4B_Groups] | pickle dict of DataFrames | 698 | 42 | `Group, Lum_BGG, Lum_group, FracLumBGG, z_group, DeltaR12, NbGal, RA_BGG, Dec_BGG, RA_Bary, Dec_Bary, Radius_Bary_arcmin, Offset_Bary, V_BGG, V_moy, Vdisp, Voffset, size_Group_Bary_kpc, M_group, M_virial, M_virial_over_L, t_cr, all_SFR, Prop_M_Sat, Prop_M_Tot, Prop_G_Sat, Prop_G_Tot, Prop_Q_Sat, Prop_Q_Tot, Misfit_Bary, Vmisfit, lMass_200, r_200_kpc, is_dominated, E_frac_NoU_NoBGG, S_frac_NoU_NoBGG, E_frac_NoU, S_frac_NoU, E_frac_NoBGG, S_frac_NoBGG, E_frac, S_frac` |
| `data/processed_sample.pkl` [Control4C_Groups] | pickle dict of DataFrames | 703 | 43 | `Group, Lum_BGG, Lum_group, FracLumBGG, z_group, DeltaR12, NbGal, RA_BGG, Dec_BGG, RA_Bary, Dec_Bary, Radius_Bary_arcmin, Offset_Bary, V_BGG, V_moy, Vdisp, Voffset, size_Group_Bary_kpc, M_group, M_virial, M_virial_over_L, t_cr, BGG_SFRcategory, all_SFR, Prop_M_Sat, Prop_M_Tot, Prop_G_Sat, Prop_G_Tot, Prop_Q_Sat, Prop_Q_Tot, Misfit_Bary, Vmisfit, lMass_200, r_200_kpc, is_dominated, E_frac_NoU_NoBGG, S_frac_NoU_NoBGG, E_frac_NoU, S_frac_NoU, E_frac_NoBGG, S_frac_NoBGG, E_frac, S_frac` |
| `data/processed_sample.pkl` [RG4_Groups] | pickle dict of DataFrames | 56 | 43 | `Group, Lum_BGG, Lum_group, FracLumBGG, z_group, DeltaR12, NbGal, RA_BGG, Dec_BGG, RA_Bary, Dec_Bary, Radius_Bary_arcmin, Offset_Bary, V_BGG, V_moy, Vdisp, Voffset, size_Group_Bary_kpc, M_group, M_virial, M_virial_over_L, t_cr, BGG_SFRcategory, all_SFR, Prop_M_Sat, Prop_M_Tot, Prop_G_Sat, Prop_G_Tot, Prop_Q_Sat, Prop_Q_Tot, Misfit_Bary, Vmisfit, lMass_200, r_200_kpc, is_dominated, E_frac_NoU_NoBGG, S_frac_NoU_NoBGG, E_frac_NoU, S_frac_NoU, E_frac_NoBGG, S_frac_NoBGG, E_frac, S_frac` |
| `data/processed_sample.pkl` [SDSS] | pickle dict of DataFrames | 52531 | 26 | `specObjID, z, r_obs, u_obs, g_obs, i_obs, z_obs, objid, SFR, sSFR, lgm, h_alpha_eqw, h_beta_eqw, oiii_5007_eqw, nii_6584_eqw, h_alpha_flux, h_beta_flux, oiii_5007_flux, nii_6584_flux, p_E, p_S, log_NII_Ha, log_OIII_Hb, is_AGN, morphology, sSFR_status` |
| `data/processed_sample.pkl` [SDSS_withAGN] | pickle dict of DataFrames | 69995 | 24 | `specObjID, z, r_obs, u_obs, g_obs, i_obs, z_obs, objid, SFR, sSFR, lgm, h_alpha_eqw, h_beta_eqw, oiii_5007_eqw, nii_6584_eqw, h_alpha_flux, h_beta_flux, oiii_5007_flux, nii_6584_flux, p_E, p_S, log_NII_Ha, log_OIII_Hb, is_AGN` |

## 2. What is NOT present locally

- **galSpecIndx downloaded: False**. Evidence: index columns found in any local table = `[]`; `src/data_loader.py` joins `['galSpecExtra', 'galSpecLine', 'zooSpec', 'SpecObj', 'PhotoObj']`; the manuscript explicitly states galSpecIndx was not retrieved: True.
- **Per-galaxy stellar velocity dispersion local: False** (columns found: `[]`). Every 'Vdisp' column in the *_Groups tables is the GROUP line-of-sight velocity dispersion (gapper), not a stellar sigma.
- **Per-spectrum median S/N local: False** (columns found: `[]`).
- Pipeline SDSS query (DR16): tables `['SpecObj', 'PhotoObj', 'galSpecExtra', 'galSpecLine', 'zooSpec']`, INNER (so galaxies without a zooSpec or galSpecExtra row are absent from the cache).
- Half-light radius used by the pipeline: Simard+11 Rchl_r (circular r-band half-light radius, pure-Sersic Table 3), via dr7objid bridge; Rchl_r / Scale -> arcsec -> Planck15 proper kpc via kpc_proper_per_arcmin (angular-diameter distance) (`src/size_data.py::attach_size_columns`). Fallback: SDSS petroR50_r (arcsec) in data/sdss_size_columns.csv.

## 3. Sample storage, flags and join keys

- **canonical_galaxy_key**: objid (SDSS DR16 photometric objID; src/identity.py: one physical galaxy = one objid)
- **canonical_group_key**: ('HMCG', Group) for CG4, ('Lim', Group) for RG4/Control4B/Control4C (same Lim namespace)
- **bgg_flag**: rank_M == 1 -> BGG; rank_M > 1 -> satellite (src/extended_data.py, 'is_bgg' / 'is_satellite')
- **spectroscopic_key_in_sample_csvs**: specobjid (int64; DR16 SpecObj.specObjID encoding: plate<<50 | fiber<<38 | (mjd-50000)<<24 | run2d<<10)
- **spectroscopic_key_in_SDSS_cache**: specObjID (uint64) — agrees with csv specobjid for every objid-matched row
- **spectroscopic_key_in_size_cache**: specObjID from JOIN SpecObj ON bestObjID — differs from csv specobjid for the run2d=700 (BOSS v5_7_0) spectra, which the size query resolved to run2d=1300 (v5_13_0): same plate/fiber/mjd
- run2d field meaning: {'26': 'SDSS-I/II legacy spectro-1d rerun (the only spectra in MPA-JHU DR8 galSpec* tables)', '700': 'BOSS v5_7_0 (SDSS-III) — not in MPA-JHU galSpec*', '1300': 'BOSS v5_13_0 (DR16 re-reduction)'}
- Unique objids across the four samples: 3857; covered by `data/sdss_size_columns.csv`: 3857
- SDSS cache (`processed_sample.pkl['SDSS_withAGN']`): 69995 rows (52531 non-AGN), 69995 unique specObjID, 69995 unique objid, z∈[0.0054,0.0452]; selection: 0.005<z<0.0452, r_petro-ext<=17.77, class=GALAXY, lgm_tot_p50>-1000
- Cosmology: astropy.cosmology.Planck15; projected scales via Planck15.kpc_proper_per_arcmin (== angular-diameter-distance based)

## 4. Sample sizes currently in use (processed_sample.pkl)

| sample | N_groups | N_gals | N_BGG (rank_M=1) | N_sat (rank_M>1) | in SDSS cache (objid) | specobjid run2d=26 / 700 / 0 | not-in-cache but lgm present | Rchl_r valid (all / sat) | NoGZ | NosSFR |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| CG4 | 62 | 248 | 62 | 186 | 217 | 234 / 14 / 0 | 17 | 197 / 173 | 31 | 18 |
| RG4 | 56 | 224 | 56 | 168 | 204 | 219 / 5 / 0 | 15 | 199 / 162 | 20 | 5 |
| Control4B | 698 | 2792 | 698 | 2094 | 2488 | 2641 / 149 / 2 | 153 | 1944 / 1721 | 304 | 180 |
| Control4C | 703 | 2812 | 703 | 2109 | 2491 | 2648 / 164 / 0 | 157 | 2065 / 1842 | 321 | 191 |

Raw CSV vs processed rows: CG4: gals 312→248, groups 78→62; RG4: gals 224→224, groups 56→56; Control4B: gals 2796→2792, groups 699→698; Control4C: gals 2816→2812, groups 704→703 (CG4 drops Split groups; Control4C drops Lim 3688).

## 5. specobjid consistency

- CG4: 14 rows where csv `specobjid` ≠ size-cache `specObjID`; of these 14 share plate/fiber/MJD (run2d 700→1300 re-reduction), the remainder are objects with a different spectrum chosen by the bestObjID join; 0 rows have specobjid=0.
- RG4: 5 rows where csv `specobjid` ≠ size-cache `specObjID`; of these 5 share plate/fiber/MJD (run2d 700→1300 re-reduction), the remainder are objects with a different spectrum chosen by the bestObjID join; 0 rows have specobjid=0.
- Control4B: 152 rows where csv `specobjid` ≠ size-cache `specObjID`; of these 149 share plate/fiber/MJD (run2d 700→1300 re-reduction), the remainder are objects with a different spectrum chosen by the bestObjID join; 2 rows have specobjid=0.
- Control4C: 170 rows where csv `specobjid` ≠ size-cache `specObjID`; of these 164 share plate/fiber/MJD (run2d 700→1300 re-reduction), the remainder are objects with a different spectrum chosen by the bestObjID join; 0 rows have specobjid=0.
