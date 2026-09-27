# CG_Gals — Are galaxies in compact groups special? (Paper II)

Analysis pipeline and manuscript source for *"Are galaxies in compact groups
special?"* (Tricottet, Mamon & Díaz-Giménez, in prep.), the companion paper to
[Tricottet, Mamon & Díaz-Giménez 2025, A&A 699, A329](https://doi.org/10.1051/0004-6361/202451727)
(Paper I). It compares the member galaxies of 62 non-split compact groups of
four galaxies (CG4, 248 galaxies, from the HMCG catalogue of
Díaz-Giménez et al. 2018) against three control samples drawn from the
Lim et al. (2017) group catalogue:

| Sample | Definition | Groups (after Lim 3688 removal) |
|---|---|---|
| Control4B | four brightest members of each parent group | 698 |
| Control4C | BGG + three closest projected companions among members within 3 mag of the BGG | 703 |
| RG4 | regular groups of exactly four members | 56 |

Control groups containing any CG4 galaxy are excluded, as in Paper I.
`Control4C` is regenerated from the parent catalogue by
`src/sample_construction.py`; the audit record for the 2026 statistical
refactor lives in `audit/` and `CHANGES.md`.

## Repository layout

```
CG_Gals/
├── data/                 # input catalogues (CG4, controls, parent PC, caches)
│   └── attic/            # retired files — never read by code
├── src/                  # analysis pipeline
│   ├── main.py           # entry point: analyses + paper rendering
│   ├── config.py         # paths, flags (REBUILD_SAMPLE, RENDER_PAPER_ONLY, ...)
│   ├── identity.py       # canonical objid/group identity layer
│   ├── sample_construction.py  # Control4C regeneration from PC_Gals
│   ├── primary_contrasts.py    # CG4 vs each control (primary inference)
│   ├── matched_controls.py     # deduplicated matching + group-level primary
│   ├── host_controlled.py      # within-host CG-member experiment
│   ├── paper_template/   # Jinja2 LaTeX template (A&A)
│   └── utils/            # shared helpers
├── reproduce.py          # single entry point: every number, figure and the PDF
├── audit/                # 2026 statistical-audit records and verification
├── analysis/gary_r2/     # 2026-09 revision-round diagnostics and checks
├── results/diagnostics/  # read-only diagnostic tables + diagnostics_gary_r2.json
├── tests/                # pytest suite (identity, samples, inference, render)
├── output/               # generated results.json, figures, paper/
└── notebooks/            # exploratory notebooks (not part of the pipeline)
```

## Reproduction

Python 3.13 (the pinned `scipy==1.15.1` has no wheels for 3.14) with the
exact versions of `requirements.lock`; pdflatex and bibtex for the paper.

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock
python reproduce.py            # every analysis, value file, figure, and the PDFs
python reproduce.py --render   # re-render the paper from existing outputs only
pytest                         # invariants, statistics helpers, render checks
```

`reproduce.py` is the single provenance path of the manuscript: it runs the
full pipeline (`src/main.py` via `audit/run_full_pipeline.py`, writing
`output/results.json` and the figures), `src/paper_additions.py` (macros),
the referee value scripts still cited by the paper (`referee/T3`, `T4`, `T5`,
`T7`, `T9`, `T10` -> `referee/values/`), the read-only diagnostics
(`analysis/gary_r2/d1`-`d7`), and then renders
`output/paper/paper.pdf` and `output/paper/online_supplement.pdf` from
`src/paper_template/`. Every stochastic step uses a fixed, documented seed;
never edit `output/paper/paper.tex` by hand. `python -m src.main` alone
honours the flags in `src/config.py` (`RENDER_PAPER_ONLY = True` renders
without re-running analyses).

## Data

All inputs are tracked in `data/`:

- the compact-group and control catalogues (`CG4_*`, `Control4B_*`,
  `Control4C_*`, `RG4_*`), the parent Lim groups (`PC_*`), and the full
  Lim et al. (2017) SDSS galaxy and group catalogues (`SDSS(L) *.dat`, used
  for the host groups and the 55-arcsec crowding flag);
- `processed_sample.pkl`: the processed samples plus the cached SDSS DR16
  reference query (MPA-JHU masses and SFRs, Galaxy Zoo 1, `galSpecLine`);
- cached public-catalogue retrievals, refreshed only for missing identifiers:
  `sdss_size_columns.csv` (DR16 Petrosian and seeing columns),
  `simard2011_subset.csv` (Simard et al. 2011, VizieR `J/ApJS/196/11`),
  `galspecindx_dr12.csv` (MPA-JHU `d4000_n`, `lick_hd_a`),
  `galspecline_dr16.csv` (MPA-JHU line fluxes, equivalent widths and errors
  of the stored spectra, `src/emission_lines.py`),
  `sdss_spectral_provenance_dr12.csv` (spectrum provenance and
  `specsfr_tot_p50`), `ds18_subset.csv` and `photoobjdr7_map.csv`
  (Dominguez Sanchez et al. 2018, VizieR `J/MNRAS/476/3661`, and the DR8->DR7
  identifier bridge);
- `galspecindx_dr12_queried_ids.txt` and `galspecline_dr16_queried_ids.txt`:
  identifiers already queried, including the BOSS spectra that have no
  MPA-JHU row, so that they are not queried again.

Once these caches exist the whole reproduction runs offline.

## Verification

`audit/verify_findings.py` re-checks every defect identified by the 2026
statistical audit against the current data, code and outputs
(`--write-md` refreshes `audit/FINDINGS_raw.md`; the curated record is
`audit/FINDINGS.md`). Open questions for the authors are tracked in
`OPEN_QUESTIONS.md`.
