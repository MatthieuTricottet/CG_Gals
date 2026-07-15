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
| Control4C | BGG + three closest projected companions | 704 |
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
├── audit/                # 2026 statistical-audit records and verification
├── tests/                # pytest suite (identity, samples, inference, render)
├── output/               # generated results.json, figures, paper/
└── notebooks/            # exploratory notebooks (not part of the pipeline)
```

## Reproduction

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or requirements.lock for exact pins
python audit/run_full_pipeline.py      # analyses + paper from cached sample
python audit/run_full_pipeline.py --rebuild   # additionally rebuild the
                                              # processed sample (SDSS query,
                                              # falls back to the cache)
pytest                                 # invariants and render smoke tests
```

`python -m src.main` honours the flags in `src/config.py`
(`RENDER_PAPER_ONLY = True` renders the paper from the existing JSON without
re-running analyses). The paper is compiled to `output/paper/paper.pdf`
(pdflatex + bibtex required). Every stochastic step uses a fixed, documented
seed; `output/results.json` and `output/results_build.json` are regenerated
by the pipeline and feed the Jinja2 template — never edit
`output/paper/paper.tex` by hand.

## External data caches

The galaxy-size analysis (`src/size_data.py`, `src/size_analysis.py`) uses two
external catalogues fetched on first run and cached under `data/`:

- **SDSS DR16 Petrosian and seeing columns** (`data/sdss_size_columns.csv`):
  `petroR50_r`, `petroR90_r`, their uncertainties, `petroRad_r`, the field
  seeing `psfWidth_r`, and the DR7 cross-match identifier `dr7objid`.
- **Simard et al. (2011, ApJS 196, 11) structural subset**
  (`data/simard2011_subset.csv`): pure-Sérsic half-light radii and indices
  from VizieR `J/ApJS/196/11`, with a CDS FTP fallback into
  `data/simard2011_raw/` (gitignored).

Both fetches are idempotent: once the caches cover the sample's object IDs,
reruns are fully offline. The SDSS spectroscopic sample itself is cached in
`data/processed_sample.pkl`.

## Verification

`audit/verify_findings.py` re-checks every defect identified by the 2026
statistical audit against the current data, code and outputs
(`--write-md` refreshes `audit/FINDINGS_raw.md`; the curated record is
`audit/FINDINGS.md`). Open questions for the authors are tracked in
`OPEN_QUESTIONS.md`.
