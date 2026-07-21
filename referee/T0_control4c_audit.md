# T0 — Control4C construction audit (BLOCKING)

**Verdict: PASS.** Every companion in `data/Control4C_Gals.csv` satisfies Δm ≤ 3, and every quartet consists of the BGG plus exactly the 3 nearest recomputed projected separations among Δm ≤ 3 members (Paper I cell-59 construction). The historical audit of the defective shipped sample is preserved in `referee/T0_control4c_audit_shipped_FAIL.md`.

## 1. `PC_Gals.rank_dist` semantics (context)

`PC_Gals.rank_dist` is the **unrestricted** distance rank, so any regeneration must recompute separations within the Δm ≤ 3 subset rather than reuse that column:

- members with Δm > 3 and rank_dist ≤ 4: **718** (a restricted rank would give 0);
- rank_dist is dense 1..N over *all* members per group: True; dense over the Δm ≤ 3 subset: False.

Paper I cell 59 filtered to Δm ≤ 3 **first** and ranked distance within that subset (`rank_distMag`); the deprecated unrestricted variant (cell 57, `Control_4C`) survives only as commented-out code.

## 2. Shipped-data assertions (per group, recomputed from PC_Gals)

- groups audited: **704** (2816 galaxies)
- **(a) Δm ≤ 3 violated in 0 groups** (0.0%), 0 companion galaxies with Δm > 3 (0.0% of companions);
- **(b) 'not the 3 nearest among Δm ≤ 3 members' in 0 groups** (tie tolerance 1e-06 arcmin);
- groups with any violation: **0**; groups whose corrected quartet differs (as a set) from the shipped one: **0** — the (a), (b), and membership-change group sets coincide exactly: every group free of Δm > 3 companions ships precisely the 3 nearest eligible members;
- per-group max Δm of shipped companions: median 2.64, 90% 2.95, 95% 2.98, max 3.00; groups with max Δm > 3: 0, > 4: 0, > 5: 0;
- ineligible (Δm > 3) members sitting closer to the BGG than the 3rd nearest eligible member: 1222 across 456 groups;
- BGG identity matches PC rank_M = 1 in all groups: True.

Per-group detail: `referee/T0_control4c_per_group.csv`.

### Side-finding: `dist2BGG` unit factor

`PC_Gals.dist2BGG` (inherited by every `*_Gals` export) equals the great-circle separation in radians × 3600 (measured ratio 3599.9998–3600.0000), i.e. it converts radians to 'arcmin' with 1 rad = 3600′ instead of 3437.75′: values are a uniform **+4.72 % too large** as arcmin. The factor is global, so all distance *rankings* (incl. rank_dist) are unaffected; any use of dist2BGG as a numeric angle or its conversion to kpc inherits the +4.72 % (flagged for T7.2; this repo's own `r2arcmin` utility is correct).

## 3. Control4B and RG4 (pass)

- Control4B: 699 groups; not the 4 brightest of the parent: 0; Δm > 3: 0 (max Δm = 2.998);
- RG4: 56 groups; parent multiplicity ≠ 4: 0; membership mismatch: 0; Δm > 3: 0 (max Δm = 2.989);
- parent eligibility (cell 52, 4th-brightest Δm ≤ 3) violations among 765 PC groups: 0.

## 4. History

The Paper I era export (git `b0d5791`, 752 groups) contains 690 companions with Δm > 3 (max Δm = 5.758): the *distributed* Control4C implemented the unrestricted nearest-3 selection since Paper I, and this repository's first regeneration (commit `c5d80a3`) faithfully reproduced it via the unrestricted `rank_dist` column. The restricted construction reproduces Paper I's *published* counts exactly (61 exclusions → 704 groups; Paper I: 61 → 704) and every published Control4C statistic (Table 2 medians, Table 3 T1/T2 — see `referee/T0_paper1_table2_check.py`), resolving OPEN_QUESTIONS.md #1: Paper I's published analysis used the restricted sample, while its distributed CSV implemented the deprecated variant. Full defect record: `referee/T0_control4c_audit_shipped_FAIL.md`.

## 5. Status

The committed sample implements the corrected construction (704 groups, matching Paper I's published 704); this audit is the acceptance gate for the regeneration and passes with zero violations.
