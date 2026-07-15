# Open questions for the authors

## 1. Control4C off-by-one vs Paper I (705 vs 704 groups)

Rebuilding Control4C from the committed `data/PC_Gals.csv` (BGG + 3 closest
projected companions, i.e. `rank_dist` ≤ 4) and excluding every group that
contains at least one galaxy of the full CG4 sample (split groups included —
the convention that exactly reproduces Paper I's Control4B 66 → 699 and
RG4 6 → 56) flags **60** contaminated groups → **705** clean groups.
Paper I (Sect. 2.4) reports **61** → **704**.

Ruled out by direct checks on the committed files:

- no projected-distance tie at any quartet boundary that involves a CG4
  galaxy (nearest miss: Lim group 51, CG4 member at rank_dist 8);
- no CG4 galaxy present in an unflagged quartet under a different objid
  (positional cross-match at 2 arcsec) or via a shared specobjid;
- no PC group hosting members of two distinct CG4 groups (60 group-pairs =
  60 groups, so pair-counting cannot give 61).

What we did find: the *committed* `Control4C_Gals.csv` contains 75 groups
that do not exist in the committed `PC_Gals.csv` at all, and only 200/224
RG4 galaxies, so it derives from an **earlier revision of the parent
catalogue**. Paper I's "61" was presumably counted on that earlier revision
and is not reproducible from the data in this repository.

**Decision taken (flagged for author sign-off):** the pipeline now uses the
reproducible construction from the committed `PC_Gals.csv` (60 excluded →
705 groups; 704 after the Lim-3688 removal), and the manuscript reports
those counts. If the authors can recover the Paper I parent catalogue, the
discrepant group can be identified by diffing the two PC revisions.
(Coincidentally 705 − 1 = 704 after removing Lim 3688, but that is *not*
the same accounting as Paper I's pre-removal 704.)
