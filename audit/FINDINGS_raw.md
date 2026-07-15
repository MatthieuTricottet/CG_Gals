# Audit findings verification (raw, machine-generated)

| Finding | Status | Detail |
|---|---|---|
| A1-B | info | Control4B: 699 groups, 2796 rows, 0 CG4 objids -> matches Paper I |
| A1-R | info | RG4: 56 groups, 224 rows, 0 CG4 objids -> matches Paper I |
| A1-C | defect absent | Control4C: 705 groups, 2820 rows, 0 CG4 galaxies in groups [] (CG4 groups [], classes []) |
| A1-rebuild | info | Rebuilt C4C from PC: 765 groups (sizes 4-4), 56 CG-contaminated -> 709 clean (Paper I says 61 -> 704) |
| A1-drift | defect absent | 0 committed-C4C groups not in PC; 60 PC groups absent from committed C4C (56 of those CG-contaminated) |
| A1-rg4c4c | info | 224/224 RG4 galaxies appear in committed C4C (should be 224 for a faithful BGG+3-closest quartet of 4-member groups) |
| A2 | defect absent | Control4C_Gals_old.csv retired (absent from data/) |
| A3 | info | Overlaps by objid: C4B&C4C=1628, RG4&C4B=224/224, RG4&C4C=224; pooled rows=5840, unique=3988 (overlap itself is a result; the defect is pooling duplicates as independent - see B4) |
| B4 | defect absent | physical_group defined in extended_data.py: True; fit_logistic_model clusters by physical_group by default: True (label-scoped default remaining: False) |
| C5-json | info | results.json matched_controls: n=234, composition={'Control4B': 141, 'Control4C': 77, 'RG4': 16} -> differs from audited composition |
| C5-fix | defect absent | matching hard constraints recorded in results.json: dedup_by_objid=True, cg4_objids_excluded=True, provenance=yes |
| C5-impl | defect absent | reimplemented matching: 234 pairs, composition={'Control4B': 141, 'Control4C': 77, 'RG4': 16}; 16 controls physically RG4, 0 duplicate control objids, 0 self-pairs (treated objid == control objid), 0 controls that are CG4 objids |
| D6-code | defect absent | bootstrap_difference uses sign-crossing p that can be exactly 0: False; add-one (k+1)/(B+1) rule present: True |
| D6-tex | defect absent | 'p<10^{-6}' occurrences in output/paper/paper.tex: 0 |
| D6-json | defect absent | matched effects with literal p==0 in results.json: none |
| E7-data | info | missing-sSFR (BGG miss/N | sat miss/N): CG4 5/62 | 9/186; Control4B 48/698 | 103/2094; Control4C 49/704 | 105/2112; RG4 3/56 | 2/168 -> matches audit; raw files contain -9999 sentinels: True |
| E7-code | defect absent | flattens_quenched present: False; sentinel-based 3-class config ['Quenched','Passive','Starforming']: False; sSFR_floor fabricates 'Quenched' from sentinel: False; good_sfr no-op -9999 loop in common.py: False |
| E7-tex | defect absent | template still states the sentinel->class/display-floor rule in 0 place(s) |
| F8 | DEFECT PRESENT | README describes 'TRICOTTET-GAM-CG2': True |
| F9-json | info | tidal attenuation of elliptical OR: baseline=1.636, with_tidal_index=1.128 |
| F9-tex | defect absent | 'explains/accounts for' within 400 chars of 'tidal' in paper.tex: 0 |
