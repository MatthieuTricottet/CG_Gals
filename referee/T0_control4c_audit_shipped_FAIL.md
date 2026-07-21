# T0 — Control4C construction audit (BLOCKING)

**Verdict: FAIL.** The shipped `data/Control4C_Gals.csv` (the file the pipeline reads, `src/data_loader.py:89`) does **not** implement the Paper I cell-59 construction. Companions are the 3 nearest projected members **regardless of magnitude**, not the 3 nearest among Δm ≤ 3 members.

## 1. Root cause (code level)

`src/sample_construction.py::build_control4c_gals` (line 272) selects `PC_Gals.rank_dist <= 4` with no Δm filter. `PC_Gals.rank_dist` is demonstrably the **unrestricted** distance rank:

- members with Δm > 3 and rank_dist ≤ 4: **718** (a restricted rank would give 0);
- rank_dist is dense 1..N over *all* members per group: True; dense over the Δm ≤ 3 subset: False.

Paper I cell 59 filtered to Δm ≤ 3 **first** and ranked distance within that subset (`rank_distMag`); the deprecated unrestricted variant (cell 57, `Control_4C`) survives only as commented-out code. The shipped selection matches the deprecated variant (see Sect. 4 for where it entered).

## 2. Shipped-data assertions (per group, recomputed from PC_Gals)

- groups audited: **705** (2820 galaxies)
- **(a) Δm ≤ 3 violated in 457 groups** (64.8%), 670 companion galaxies with Δm > 3 (31.7% of companions);
- **(b) 'not the 3 nearest among Δm ≤ 3 members' in 457 groups** (tie tolerance 1e-06 arcmin);
- groups with any violation: **457**; groups whose corrected quartet differs (as a set) from the shipped one: **457** — the (a), (b), and membership-change group sets coincide exactly: every group free of Δm > 3 companions ships precisely the 3 nearest eligible members;
- per-group max Δm of shipped companions: median 3.30, 90% 4.32, 95% 4.59, max 5.76; groups with max Δm > 3: 457, > 4: 139, > 5: 10;
- ineligible (Δm > 3) members sitting closer to the BGG than the 3rd nearest eligible member: 1225 across 457 groups;
- BGG identity matches PC rank_M = 1 in all groups: True.

Per-group detail: `referee/T0_control4c_per_group.csv`.

### Side-finding: `dist2BGG` unit factor

`PC_Gals.dist2BGG` (inherited by every `*_Gals` export) equals the great-circle separation in radians × 3600 (measured ratio 3599.9998–3600.0000), i.e. it converts radians to 'arcmin' with 1 rad = 3600′ instead of 3437.75′: values are a uniform **+4.72 % too large** as arcmin. The factor is global, so all distance *rankings* (incl. rank_dist) are unaffected; any use of dist2BGG as a numeric angle or its conversion to kpc inherits the +4.72 % (flagged for T7.2; this repo's own `r2arcmin` utility is correct).

## 3. Control4B and RG4 (pass)

- Control4B: 699 groups; not the 4 brightest of the parent: 0; Δm > 3: 0 (max Δm = 2.998);
- RG4: 56 groups; parent multiplicity ≠ 4: 0; membership mismatch: 0; Δm > 3: 0 (max Δm = 2.989);
- parent eligibility (cell 52, 4th-brightest Δm ≤ 3) violations among 765 PC groups: 0.

## 4. Where the defect entered

The Paper I era export (git `b0d5791`, 752 groups) **already contains 690 companions with Δm > 3** (max Δm = 5.758). The shipped Control4C has therefore implemented the *unrestricted* nearest-3 selection since Paper I: the defect predates this repository. The 2026 regeneration (commit `c5d80a3`) faithfully reproduced the shipped construction via the unrestricted `rank_dist` column — but that construction matches the deprecated cell-57 variant, not the cell-59 code (Δm ≤ 3 filter first, then `rank_distMag`) that the Paper I notebook documents as the final one. In other words, **the exported Control4C CSV disagrees with the notebook's documented construction**, and the referee's question uncovers a real defect inherited from the Paper I data export, not a regeneration artefact of this paper.

**The corrected construction reproduces Paper I's published counts exactly, resolving OPEN_QUESTIONS.md #1.** Re-running the CG4-contamination exclusion on corrected (Δm ≤ 3-filtered) quartets excludes 61 groups → 704 (Paper I published: 61 → 704; the unrestricted shipped construction gives 60 → 705). The corrected exclusion set is a strict superset of the current one (the same 60 groups + Lim 442, whose corrected quartet picks up a CG4 galaxy). Paper I's *published* sample was therefore built with the restricted construction; the CSV handed down to this paper was out of sync with the published analysis.

## 5. Consequence and required decision

Every Control4C-dependent result in the current build is computed on the unrestricted sample: raw tables, per-control adjusted models, both matchings, the crowding test, pooled models, and the tidal comparison. Per the task instructions, **all downstream referee tasks are halted**. Two repair routes exist and the choice is an author-level decision because it touches the Paper I data legacy:

1. **Correct to the documented construction** (Δm ≤ 3 filter before distance ranking) — *recommended by the evidence above*: changes the quartet membership of 457/705 shipped groups (670 companion swaps). Every parent group retains ≥ 3 eligible companions (min = 3), so no group drops out for lack of members; the corrected sample has 61 exclusions → **704 groups**, matching Paper I's published 704. All Control4C numbers in the paper change, but the sample *converges to* the one Paper I actually published.
2. **Keep the shipped CSV's sample** and re-document it honestly: Control4C = BGG + 3 nearest projected members *regardless of relative magnitude*, answering the referee that the Δm ≤ 3 range is respected by construction in Control4B/RG4 but not in Control4C, and quantifying the faint tail (31.7 % of companions, median per-group max Δm 3.30). This avoids recomputation but perpetuates a sample that matches neither the notebook's construction nor Paper I's published counts.

## 6. Addendum (same day): verification against the published Paper I

Checked with `referee/T0_paper1_table2_check.py` against the *published*
A&A 699, A329 numbers, on data byte-identical to the public GitHub copies
(MD5-verified). The Δm ≤ 3-restricted construction reproduces **every**
published Control4C statistic; the distributed CSV reproduces **none**:

| statistic | published | shipped CSV (unrestricted) | restricted rebuild |
|---|---|---|---|
| groups (exclusions) | 704 (61) | 705 (60) | **704 (61)** |
| Table 2 σ_v (km/s) | 153 | 156.1 | **153.0** |
| Table 2 log L | 11.01 | 10.967 | **11.011** |
| Table 2 ΔMr12 | 1.17 | 1.465 | **1.168** |
| Table 2 L_BGG/L | 0.61 | 0.694 | **0.612** |
| Table 2 ⟨Rij⟩ (kpc) | 313 | 212 (D_A) / 226 (D_L) | 293 (D_A) / **313 (D_L)** |
| Table 3 T1, T2 | 0.29, 0.65 | 0.23, 0.68 | **0.29, 0.65** |

Two consequences. (i) Paper I's published analysis used the restricted
sample; its prose (Sect. 2.4, Table 1, Conclusions: "the BGG and the three
closest galaxies to the BGG (in projection)") omits the Δm ≤ 3
restriction, and the distributed CSV implements that literal prose rather
than the analysed sample. (ii) The published ⟨Rij⟩ matches only when
angular separations are converted with the *luminosity* distance —
Paper I's Eq. (1) states D_A but the published sizes used D_L (+6.8% at
median z): the T7.1 size-conversion issue is present in the published
Paper I values themselves, not only in this repository's Groups files.
