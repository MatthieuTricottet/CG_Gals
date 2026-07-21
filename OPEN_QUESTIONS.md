# Open questions for the authors

## 1. Control4C off-by-one vs Paper I (705 vs 704 groups) — RESOLVED 2026-07-21

**Resolution (referee task T0):** the off-by-one was caused by the quartet
selection, not by a parent-catalogue revision. `PC_Gals.rank_dist` is the
*unrestricted* projected-distance rank; the construction behind Paper I's
published statistics filters members to Δm ≤ 3 **before** ranking
(notebook cell 59). Rebuilding Control4C with the restricted construction
from the committed `PC_Gals.csv` flags **61** contaminated groups → **704**
clean groups, exactly Paper I's published counts (the exclusion set is a
strict superset of the unrestricted one: the same 60 groups + Lim 442),
and reproduces every published Control4C statistic (Table 2 medians,
Table 3 T1/T2 — `referee/T0_paper1_table2_check.py`). The CSV distributed
with Paper I (and this repo's Phase 2 regeneration, which trusted
`rank_dist ≤ 4`) implemented the deprecated unrestricted variant: 457 of
its 705 groups contained companions with Δm > 3 (up to 5.76 mag;
`referee/T0_control4c_audit_shipped_FAIL.md`).

**Decision taken (2026-07-21, author-approved):** `sample_construction.py`
now implements the restricted construction (61 excluded → 704 groups; 703
after the Lim-3688 removal), the manuscript documents the Δm ≤ 3
eligibility explicitly, and `referee/T0_control4c_audit.py` is the
acceptance gate (exit 0 on the committed data).

The earlier investigation notes are kept below for the record: the
*pre-Phase-2 committed* `Control4C_Gals.csv` additionally derived from an
earlier parent-catalogue revision (75 groups absent from the committed
`PC_Gals.csv`, 200/224 RG4 galaxies), which compounded the discrepancy but
was not the cause of the 60-vs-61 gap.
