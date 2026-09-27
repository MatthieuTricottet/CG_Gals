# Superseded referee-round artefacts

Kept for the record of the earlier response draft (`../RESPONSE_DRAFT.md`);
not read by the paper renderer and not run by `reproduce.py`.

* `T1_*` — crowding refits with a 55-arcsec flag computed from the *other
  quartet members only*.  Superseded (2026-09-27, audit-r3) by
  `src/crowding.py`, which searches the full Lim et al. (2017) catalogue;
  the manuscript numbers come from `results.json['extended_specialness']['crowding']`.
* `T2_*` — within-host models with and without the linear host-centric
  radius.  Superseded by `host_controlled.radial_specifications`
  (`results.json['extended_specialness']['host_controlled']`), which adds the
  logarithmic, spline and common-support specifications.
