# Paper II referee reanalysis

The analysis uses the 6 November 2026 Paper II revision in the CG_Gals checkout and writes only here. See [the scientific report](REPORT.md), [readable tables](TABLES.md), [CSV tables](tables/), [figures](figures/) and [input checksums](provenance.json).

## Reproduce

From the CG_Gals repository root, using Python 3.13 and the repository scientific environment:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv_clean/bin/python -u results/referee_tests_20261117/run.py
PYTHONDONTWRITEBYTECODE=1 .venv_clean/bin/python -m pytest -q -p no:cacheprovider results/referee_tests_20261117/test_referee.py
```

On another machine, use an environment installed from the repository's requirements with NumPy, pandas, SciPy, statsmodels, Patsy, Matplotlib, Astropy and the repository's transitive imports. The tests also use pytest and pypdf. The exact versions and checksums from the run are recorded in `provenance.json`. No network access is needed. The original SDSS and Simard sizes are loaded using the November rank-split pipeline. Spectral indices use a cache-only read through the same column validation and join as the paper. The project caches must be present at the paths listed in the manifest. Only the Paper I ancillary inputs used here have been copied or subset in this directory.

## Inputs and provenance

- Paper II `output/paper/paper.tex`, `paper.pdf`, `online_supplement.tex` and `online_supplement.pdf` at commit `d768251`. The September `output/results*.json` supply older frozen results and the sSFR boundary. The newer rank-split source is present in the November revision. Code and provenance distinguish these layers.
- `data/processed_sample.pkl` supplies the final quartets and SDSS measurements. The CSVs supply native members, the full pre-split CG4 list, Lim parent geometry and sizes, and the input to Control4C reconstruction. `data/galspecindx_dr12.csv`, `galspecline_dr16.csv`, `sdss_spectral_provenance_dr12.csv`, `sdss_size_columns.csv` and `simard2011_subset.csv` supply the cached external measurements. `output/ssfr_spectral_provenance.csv` identifies unavailable legacy products by released spectrum ID.
- Paper I material came from the user's local Google Drive archive under `_Astrophysics/Tricottet-Gam-CG-1/Data/`. The package includes copies of `HMCG/EugeTableD1.dat`, `HMCG/gal_in_SDSSCG18_ObjID.dat` and `HMCG/SDSS_to_Euge.csv`. These contain published catalogue identifiers and previously queried public SDSS values. [HMCG catalogue and publication](https://vizier.cds.unistra.fr/cgi-bin/VizieR-3?-source=J%2FA%2BA%2F618%2FA157)
- `inputs/tempel_lowz.csv.gz` contains the columns `objid,specobjid,raj2000,dej2000,mag_g,mag_r,zobs,zcmb,absmag_r` and rows with `0 ≤ zcmb ≤ .065` from the archive's `Tempel/Tempel-HMCG.csv`. This range covers the velocity search around all tested quartets. `inputs/gz1_paper1_crossmatch.csv` contains `OBJID,RA,DEC,NVOTE,P_EL_DEBIASED,P_CS_DEBIASED` from the archive's `GalaxyZoo1_DR_table2.csv`, subset to the DR16 and DR7 identifier columns of `data/sdss_size_columns.csv`. The recovery sensitivity requires an unambiguous DR7 identifier and sky positions within 1.5 arcsec. [Galaxy Zoo 1 catalogue](https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/MNRAS/410/166)
- The Paper I definitions, Table 2 and contradictory class counts in Sect. 3.3 and Table 5 were checked against arXiv v5, in press at *A&A* 699, A329. The repository README's Paper I DOI ends in `202451727`. The bibliographic entry and published record use `202451713`. [Paper I, arXiv v5](https://arxiv.org/html/2407.20357v5)

## Analysis and outputs

- [compactness.py](compactness.py) reconstructs actual mean and median proper separations, reconstructs Appendix C, reproduces the Appendix F compactness rows, and executes overlap regressions, leave-one-CG-host-out checks and matching.
- [mass.py](mass.py) computes the mass-bin diagnostics, continuous tests, red-galaxy completeness envelope and fixed-mass spectroscopy and aperture diagnostics.
- [within_host.py](within_host.py) reconstructs eligible companions and host radius, audits support and fits conditional logistic models.
- [catalogues.py](catalogues.py) does purity, isolation, GZ1 missingness, alternate spectrum, galaxy matching and colour-selection checks.
- [referee_common.py](referee_common.py) holds modelling and interval helpers. Between-sample regression uses physical-host cluster-robust covariance and cluster-based t intervals, and marginal contrasts use the delta method. Descriptive fractions and medians resample whole hosts. Matched effects resample host components from both sides. All bootstrap tables use 4,999 draws, seed 20261117 and 2.5/97.5 percentiles. Matched effects condition on the selected pairs. Conditional-logit Wald intervals use the stratified likelihood and independent Lim hosts. Figure 02 shows pointwise binned intervals and the mass-standardized response, Figure 03 and 05 show pointwise adjusted intervals. Bootstrap limits can coincide at an observed fraction of zero or one.
- [run.py](run.py) checks sample sizes, identity and cosmology, runs the modules, and writes [provenance.json](provenance.json) and [TABLES.md](TABLES.md). [test_referee.py](test_referee.py) checks persisted counts, clustering identities, distances, matching calipers, reproducible bootstrap estimates, structural support and regression status.

CSV files retaining galaxy identifiers should be imported with an exact integer or string type. Spreadsheet defaults and binary floating-point imports can round SDSS objIDs. Galaxy rows across control definitions intentionally repeat. Within each definition the galaxy identifier is unique.
