# data/attic — retired input files (do not use)

## Control4C_Gals_old.csv

Retired in the 2026 statistical audit (branch `refactor/statistical-audit`).
The file is **malformed**: its data columns are shifted left by one relative
to the header (the `objid` column actually holds `specobjid` values, and so
on). It also predates the Paper I CG-exclusion step. It is kept only as a
provenance record; no code may read it.

The current `data/Control4C_Gals.csv` / `data/Control4C_Groups.csv` are
regenerated from `data/PC_Gals.csv` by `src/sample_construction.py`
(BGG + 3 closest projected companions, Paper I CG exclusion applied).
