# bgg-split — Phase 0 read-only audit (GATE A)

Date 2026-09-29. Script: `analysis/bgg_split/phase0_audit.py`, which uses
`analysis/bgg_split/common.py`. Machine-readable output:
`phase0_audit.json` in this directory. The script rebuilds the galaxy frame
with network access blocked. It changes no existing file. `git status` after
the run shows only the two new untracked directories, `analysis/bgg_split/`
and `results/diagnostics/bgg_split/`.

## 0.1 Repo state

* `git rev-parse HEAD`: `36e04bddd7adb91256df02df186ba4ac1abae70a` (main, up to date with origin/main).
* `git status` before this round: clean. No tag has been created yet; `pre-rank-split` will be tagged before the first edit.
* **Caption, as it stands** (`src/paper_template/paper_template.tex:468`):
  > Other galaxy properties versus stellar mass: (a) strong-Hα fraction; (b) AGN-like fraction among BPT-classified galaxies; (c) median D_n4000; (d) median seeing-corrected effective radius log10(R_e,r/kpc). Points use the mass bins of Fig. 2; bars here are 16th–84th percentile intervals from a bootstrap blocked by physical group. These comparisons are descriptive.
* The tree is clean, so the local caption edit you mentioned is the version committed in 36e04bd. Compared with a3d08b2, "AGN fraction" became "AGN-like fraction" and the interval sentence was rewritten. No sentence was deleted. **If you have caption edits that are not committed (e.g. in another copy), please apply them before Phase 4.**
* **`sec:other_properties`** (template lines 456–462, one paragraph). As rendered in `output/paper/paper.tex:582`, it contains:
  * strong-Hα raw fractions: 70/217 = 32 % in CG4, against 41/43/57 %, with Fisher p = 0.012, 0.002, <1e-4. These are injected from `recent_quenching`.
  * **hard-coded** morphology-adjusted ORs of 0.55, 0.65 and 0.56 (see 0.6).
  * AGN-like fraction 38 %, with adjusted OR 1.17 (p = 0.364). These are injected from `agn_environment`.
  * the D_n4000 sentence: "CG4 above the controls only in the 10.0–10.5 bin, 1.77 against 1.59, 1.68, 1.56". This is injected from `props.rows` (see 0.7).

## 0.2 Reproduce Fig. 4 — PASS

I rebuilt the frame from the pickle as follows:
1. Apply the non-rebuild `main.load_data` steps (`correct_group_distance_scales`, `classify_all_samples(record_build_counts=False)`, `add_morphology_fractions_to_groups`).
2. Run `build_galaxy_frame`, then `attach_size_columns` (6076 rows).
3. Run `descriptive_properties.compute_property_trends`.

Compared with the committed `output/descriptive_property_trends.csv`:

| check | result |
|---|---|
| rows matched (panel × sample × bin) | 64 / 64 |
| `n_galaxies` / `n_groups` mismatches | 0 / 0 |
| max \|Δ\| on `estimate`, `ci16`, `ci84` | 2.2e-16, 2.2e-16, 6.9e-17 (all < 1e-12) |
| NaN pattern and `displayed` mismatches | 0 |

## 0.3 Composition — matches the reference exactly

Counts are BGG / satellite galaxies with finite `logMstar`, per `MASS_BINS` bin.

| Sample | < 10.0 | 10.0–10.5 | 10.5–11.0 | ≥ 11.0 | group selections | physical groups |
|---|---|---|---|---|---|---|
| CG4 | 0 / 53 | 0 / 78 | 13 / 44 | 44 / 2 | 62 | 60 |
| Control4B | 2 / 399 | 3 / 640 | 176 / 783 | 469 / 169 | 698 | 698 |
| Control4C | 2 / 684 | 3 / 766 | 176 / 482 | 473 / 62 | 703 | 703 |
| RG4 | 1 / 77 | 0 / 57 | 30 / 31 | 22 / 1 | 56 | 56 |

* HEAD is the reference commit and every count agrees, so no stop is needed.
* Every group has exactly one `rank == 1` row, and `rank` is never missing. 5 of the 62 CG4 BGGs have no `logMstar`: 57 have finite mass, out of 234 finite-mass CG4 rows in 248.
* The two pairs of CG4 selections that share a physical host are Lim:36 (CG4 165 and 167) and Lim:757 (CG4 27 and 28). They are what makes 62 selections into 60 physical groups.
* Every BGG satisfies the M_r ≤ −21.81 cut, but not every BGG has log M⋆ ≥ 10.5. There are 11 control BGGs below 10.5 (2+3, 2+3, 1). No CG4 BGG is below 10.5. In the BGG column of Fig. 4 these bins stay undisplayed (≤ 3 galaxies).
* **Phase 2 support check:** there are 13 and 44 CG4 BGGs in the two upper bins, so both clear the ≥ 10 requirement and Phase 2 is feasible as prespecified.

## 0.4 BGG stellar-mass distributions

| Sample | n | p5 | p16 | p50 | p84 | p95 | Δmedian vs CG4 | OVL (0.1-dex hist) | ctrl in CG4 p5–p95 | CG4 in ctrl p5–p95 | KS p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CG4 | 57 | 10.82 | 10.94 | 11.12 | 11.30 | 11.46 | — | — | — | — | — |
| Control4B | 650 | 10.79 | 10.93 | 11.10 | 11.28 | 11.40 | −0.019 | 0.85 | 0.90 | 0.88 | 0.72 |
| Control4C | 654 | 10.79 | 10.94 | 11.10 | 11.29 | 11.40 | −0.019 | 0.85 | 0.90 | 0.88 | 0.73 |
| RG4 | 53 | 10.66 | 10.82 | 10.98 | 11.12 | 11.21 | **−0.143** | 0.63 | 0.87 | 0.70 | 2.5e-5 |

The BGG masses of Control4B and Control4C are essentially identical to those of CG4. The RG4 BGGs are 0.14 dex lighter, which confirms the reference value. Only 70 % of CG4 BGGs fall inside the RG4 5–95 % range. Any CG_bgg contrast against RG4 therefore leans on the spline extrapolating into the top ~0.25 dex.

## 0.5 Size coverage (sample × rank, split at r_obs = 14)

`r_obs` is the DR16 Petrosian r minus extinction, taken from `SDSS_withAGN`. It is missing for 9–13 % of rows, namely those absent from that table. Each cell gives the counts with a Simard `log_Rchl_r_kpc` / Simard including pegged / Petrosian `petroR50_kpc`.

| Sample | rank | all | r < 14 | r ≥ 14 | r missing |
|---|---|---|---|---|---|
| CG4 | BGG | **24**/24/62 of 62 | 3/3/34 of 34 | 20/20/20 of 20 | 1/1/8 of 8 |
| CG4 | sat | 168/173/186 of 186 | 2/2/4 of 4 | 152/157/159 of 159 | 14/14/23 of 23 |
| Control4B | BGG | 223/223/698 of 698 | 27/27/427 of 427 | 186/186/192 of 192 | 10/10/79 of 79 |
| Control4B | sat | 1694/1716/2094 of 2094 | 30/30/235 of 235 | 1575/1597/1634 of 1634 | 89/89/225 of 225 |
| Control4C | BGG | 223/223/703 of 703 | 27/27/430 of 430 | 186/186/192 of 192 | 10/10/81 of 81 |
| Control4C | sat | 1807/1838/2109 of 2109 | 12/13/96 of 96 | 1704/1733/1773 of 1773 | 91/92/240 of 240 |
| RG4 | BGG | 37/37/56 of 56 | 1/1/13 of 13 | 34/34/36 of 36 | 2/2/7 of 7 |
| RG4 | sat | 160/162/168 of 168 | 0/0/1 of 1 | 150/152/154 of 154 | 10/10/13 of 13 |

The next table attributes each galaxy that lacks a primary Simard size. The categories are mutually exclusive.

| Sample | rank | no Simard size | not in Simard, r < 14 | not in Simard, other r | no DR7 bridge | z-mismatch / shred | Sérsic-pegged |
|---|---|---|---|---|---|---|---|
| CG4 | BGG | 38 | **31** | 2 (r missing) | 5 | 0 | **0** |
| CG4 | sat | 18 | 2 | 2 | 9 | 0 | 5 |
| Control4B | BGG | 475 | 400 | 23 | 52 | 0 | 0 |
| Control4B | sat | 400 | 205 | 55 | 113 | 5 | 22 |
| Control4C | BGG | 480 | 403 | 24 | 53 | 0 | 0 |
| Control4C | sat | 302 | 83 | 54 | 130 | 4 | 31 |
| RG4 | BGG | 19 | 12 | 4 | 3 | 0 | 0 |
| RG4 | sat | 8 | 1 | 3 | 2 | 0 | 2 |

* **Confirmed: the r = 14 bright limit of Simard et al. (2011) causes the BGG losses. The Sérsic-pegging cut does not.** It removes no BGG in any sample and only 5/22/31/2 satellites.
  * CG4: 24 of 62 BGGs have a Simard size. 31 of the 38 losses are r < 14 galaxies absent from the catalogue. Every CG4 BGG at r ≥ 14 has a size (20/20).
  * The 5 with no DR7 bridge have no `r_obs` either. They are consistent with the non-Legacy (BOSS) spectra identified in gary-r2 D1.
* **Petrosian `petroR50_kpc` covers 100 % of every sample × rank.** `log_petroR50_kpc` passes its validity window everywhere except 3 Control4C satellites.
* Consequence for the size panel: Simard sizes exist for only 32 % of the Control4B/4C BGGs, 42 % of CG4 BGGs and 66 % of RG4 BGGs. They are selected towards the fainter BGGs, which also means less massive ones. The Petrosian rows planned in 1.4 are the right check.

## 0.6 Hard-coded strong-Hα odds ratios (0.55, 0.65, 0.56)

* **No generating code exists on `main`.** The sentence arrived in a3d08b2 ("gary-r2: sharpen manuscript claims"), which touched only `paper_template.tex`, `output/paper/paper.tex` and `paper.pdf`. Nothing in `src/`, `analysis/`, `audit/` or `output/results*.json` fits a morphology-adjusted strong-Hα model.
* **A reproduction exists on the unmerged `audit-r3` branch:** `audit_claude_referee/a4c_halpha.py` with `a4c_output.txt`, committed in 0e39e72. It labels the values "paper (hard-coded)", and its row "logM + E (E/S only, clustered by physical group)" reproduces them to two decimals:

  | control | OR | 95 % CI | p | n |
  |---|---|---|---|---|
  | Control4B | 0.55 | [0.37, 0.81] | 0.0023 | 2582 |
  | Control4C | 0.65 | [0.45, 0.95] | 0.027 | 2557 |
  | RG4 | 0.56 | [0.30, 1.05] | 0.072 | 399 |

  * **Predictors:**
    * `is_CG4`;
    * standardised `logM`, from the catalogue `lgm` rather than `lgm_tot_p50`;
    * a binary GZ1 `E`: E = 1 and S = 0, with Uncertain and NoGZ rows dropped.
    * **Satellite status is not included.** With `sat` added, the same script gives 0.51, 0.64 and 0.59.
  * **Clustering:** cluster-robust binomial GLM on the physical group, i.e. the host Lim group for CG4 (HMCG otherwise), which is the same definition as `physical_group`.
  * **Sample:** CG4 ∪ one control. All ranks are included. Rows need a finite EW(Hα), `logM` and an E/S class. The outcome is EW(Hα) ≤ −3 Å with no S/N cut.
  * "First two statistically clear" rests on raw p-values, with no multiplicity correction.
* That script is an independent referee loader: it does not import `src/`. It is a reproduction, not the original generator. On `audit-r3` itself, the sentence was later replaced by values injected from a new `src/halpha_emission.py`, which uses a different definition (W ≥ 3 Å with S/N ≥ 3 from galSpecLine).
* **Verdict for 4.5:** the source was not found on `main`. As specified, the default is to insert `% TODO(Matt): untraced ORs` (see decision D3).

## 0.7 Template selectors that read `props.rows`

`props` is set once, at `paper_template.tex:146` (`es.get('descriptive_properties', {})`). Neither `online_supplement_template.tex`, `generate_report.py`, `audit/` nor `validate_template.py` reads it.

| line | expression | reads rows? | ambiguity once rows carry three scopes |
|---|---|---|---|
| 462 | `props.get('rows', [])\|selectattr('panel','equalto','dn4000')\|selectattr('bin_left','equalto',10.0)\|list` → `dn_rows` | yes | the result contains three rows per sample (all / bggs / satellites) |
| 462 | `dn_rows\|selectattr('sample','equalto', S)\|list\|first` for S in CG4, Control4B, Control4C, RG4 → `dn_cg`, `dn_cb`, `dn_cc`, `dn_rg` | yes | `first` silently takes whichever scope is emitted first. It prints `.bin_left`, `.bin_right` and `.estimate` (currently 1.77 / 1.59 / 1.68 / 1.56) |
| 464 | `props.get('figure')` (guard) | no | none |
| 467 | `props.get('figure')` (includegraphics) | no | none |

That is the only row selector. In the 10.0–10.5 bin, CG4 has 0 BGGs, and the controls have 3, 3 and 0. The satellite-scope values will therefore be close to the current ones but not identical. Per 4.2, the selector will read `scope == 'satellites'`.

## 0.8 Line-flux errors

* **On `main` (working tree and pickle), there are no line-flux error columns.**
  * `SDSS_withAGN` carries only `*_flux`, `*_eqw`, `log_NII_Ha`, `log_OIII_Hb` and `is_AGN`. The `load_SDSS` query does not select `*_err`.
  * No galSpecLine cache exists in `data/`.
* The BPT classes (`agn_environment._classify`) require only finite log ratios built from positive fluxes. There is **no S/N cut**. `is_AGN` rows without line ratios are also counted as AGN-like.
* **However**, the `audit-r3` branch holds `data/galspecline_dr16.csv` together with `…_queried_ids.txt`. It contains galSpecLine fluxes, EWs **and errors** for Hα, Hβ, [O III] 5007 and [N II] 6584, keyed by stored `specobjid`. It was fetched with your approval during audit-r3.
  * I read it offline with `git show`, without any network access.
  * It covers **all 4949 BPT-ratio rows** of the main frame.
  * Hα fluxes agree with `SDSS_withAGN` for all 5400 rows the two tables share: 0 differ by more than 1 %.
  * An S/N ≥ 3 sensitivity in 3.1 is therefore possible with no fetch, if the file is brought onto `main` (decision D4).

## Tests and templates at GATE A

`pytest -q`: 137 passed. `validate_template.py` passes on both templates: 42 refs for the paper, 3 for the supplement. The tree is unchanged by either.

---

## Decisions needed at GATE A

**D1 — The prespecified spline is infeasible as written.**
* patsy `cr(logMstar, df=3)` without constraints returns 3 columns that sum to 1 row by row. Together with the model intercept, the design is rank-deficient: rank 3 for 4 columns, checked with patsy 1.0.2. Both `fit_logistic_model` and `smf.ols` always add a constant.
* Following the rules, I stopped rather than choosing a fix. Options:
  * **(a) (recommended)** Use `cr(logMstar, df=3, constraints="center")`. This gives 3 non-constant columns (linear plus 2 non-linear), which matches R's `ns(x, df=3)` and the df of the preliminary `bs(…, df=3)`.
  * (b) Use the unconstrained `cr(df=3)` columns and drop the intercept. This gives only 2 non-constant df.
* For the spec's "identical in every fit": I propose to build the basis **once per model family** on the pooled four-sample frame and reuse the columns unchanged in every per-control fit and every bootstrap draw.
  * For Models I/R, the frame is all rows with log M⋆ ≥ 10.5.
  * For Model S, it is all satellites.
  * Knots stay fixed at the pooled quantiles.

**D2 — 1.1 blocking key.**
* Blocking on `physical_group` affects CG4 only in content: 9 CG4 rows lose one block (Lim:36, Lim:757). The estimates and `displayed` do not change.
* Because the bootstrap draws index the sorted group keys, switching from integer `Group` to string `"Lim:<id>"` also reorders the controls' blocks. All 64 intervals then move by Monte-Carlo noise. In a dry run the largest shift was 0.021, on RG4 AGN-like at ≥ 11.0, which is a pure reordering effect. The largest CG4 shift was 0.016.
* This means Phase 1's "identical `n_groups`" will fail by construction in those 9 CG4 rows.
* Proposal: block on `physical_group` directly, and report the CG4 shifts separately from the control-only reordering noise.
* The alternative is an order-preserving integer key, which keeps the control rows bit-identical but is less transparent.

**D3 — Hard-coded ORs (4.5).**
* Default per spec: add the `% TODO(Matt): untraced ORs` marker and leave the numbers alone.
* Alternative, outside the current plan: port the a4c model into the pipeline and inject the values. Note that it omits satellite status.

**D4 — galSpecLine errors.**
* Bring `data/galspecline_dr16.csv` and its sidecar onto `main` with `git checkout audit-r3 -- data/galspecline_dr16*.csv/txt`. This is local and needs no network. It would enable the S/N ≥ 3 run in 3.1.
* Otherwise, 3.1 records that "BPT classes carry no S/N cut".

**Notes (no decision needed unless you object):**
* `_mass_bin_rows` seeds the bootstrap with (sample, panel, bin) and not with scope, so the BGG and satellite rows reuse the same RNG streams as "all". This is harmless and keeps the "all" rows unchanged, so I will leave it.
* Fig. 2 (`descriptive_trends.compute_mass_trends`) also blocks on CG4 `Group`, even though its `interval_method` string says "physical-group-blocked". Fig. 2 plots Mamon binomial errors, so the figure is unaffected; only the CSV intervals are. This is out of scope and is flagged only.
