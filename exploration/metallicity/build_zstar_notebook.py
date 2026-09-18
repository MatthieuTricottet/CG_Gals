"""Build the executable direct-Zstar continuation notebook."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
NB = nbf.v4.new_notebook()
NB["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}


def md(text: str) -> None:
    NB.cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    NB.cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# Direct stellar metallicity: Gallazzi DR4 and FIREFLY DR16

This notebook continues—not replaces—the historical [index-level feasibility notebook](metallicity_scoping.ipynb) and its [immutable report](REPORT.md). It carries the same chosen spectra and parent rows into direct stellar-metallicity catalogues, analyses satellites and BGGs separately, and takes Control4C as the primary compact-core analogue.

The calculations live in `zstar_analysis.py` so that the notebook and command-line run share one implementation. Set `REFRESH=True` below to regenerate all small analysis products from the cached catalogue matches. Catalogue acquisition itself is isolated in the idempotent `zstar_fetch.py`.
""")

code("""
from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, Markdown, display
import zstar_analysis as za

HERE = Path.cwd()
if not (HERE / "zstar_analysis.py").exists():
    raise RuntimeError("Run this notebook from exploration/metallicity")
REFRESH = False
required = HERE / "outputs" / "zstar" / "zstar_contrasts.csv"
if REFRESH or not required.exists():
    za.main()
print("Analysis products:", required.parent)
""")

md("""
## 1. Acquisition integrity and conventions

Plate–MJD–fibre is the authoritative spectrum key. The FIREFLY `SPECOBJID` is not used. Gallazzi's absolute `log10(Z)` values are converted using `Z_sun=0.02`; FIREFLY's linear `Z/Z_sun` values are converted with `log10`. Sentinel values are removed before either transformation.
""")

code("""
manifest = json.loads((HERE / "work/catalogues/catalogue_manifest.json").read_text())
run2d = manifest["firefly"]["returned_run2d"]
assert sum(run2d.values()) == manifest["firefly"]["matched_unique_pmf"]
assert manifest["sdss_aux"]["matched_unique_objid"] == 3857
display(pd.DataFrame({
    "check": ["Gallazzi matched PMF", "FIREFLY matched PMF", "FIREFLY RUN2D count sum", "SDSS photometry objid"],
    "N": [manifest["gallazzi"]["matched_unique_pmf"], manifest["firefly"]["matched_unique_pmf"], sum(run2d.values()), manifest["sdss_aux"]["matched_unique_objid"]],
}))
print("RUN2D:", run2d)
""")

md("""
## 2. Coverage and selection

`usable` means Gallazzi valid plus S/N≥20, reflecting that catalogue's published reliability study. FIREFLY uses valid ordered posteriors plus a galaxy classification and valid no-QSO redshift; S/N cuts are sensitivities, not the primary selection.
""")

code("""
coverage = pd.read_csv(HERE / "outputs/zstar/coverage.csv")
display(coverage.style.format({"retained_fraction": "{:.1%}", "median_z": "{:.4f}", "median_logmstar": "{:.2f}", "median_SN": "{:.1f}", "median_logz_unc_dex": "{:.3f}"}))
display(Image(filename=str(HERE / "figures/zstar/coverage.png")))
""")

code("""
smd = pd.read_csv(HERE / "outputs/zstar/selection_smd.csv")
display(smd.assign(abs_smd=smd.usable_minus_not_smd.abs()).sort_values("abs_smd", ascending=False).head(12))
display(Image(filename=str(HERE / "figures/zstar/selection_functions.png")))
display(Image(filename=str(HERE / "figures/zstar/uncertainty_vs_sn.png")))
""")

md("""
## 3. Gallazzi–FIREFLY cross-calibration

The common scale is `[Z/H]`. The primary object-level comparison uses Gallazzi's S/N-qualified sample and FIREFLY MILES light-weighted metallicity. ELODIE and mass-weighted combinations are fixed model-systematic checks rather than candidates selected by environmental effect size.
""")

code("""
cross = pd.read_csv(HERE / "outputs/zstar/crosscal_summary.csv")
display(cross)
display(Image(filename=str(HERE / "figures/zstar/cross_calibration.png")))
""")

md("""
## 4. Mass–metallicity relation and environmental contrasts

The total contrast is `[Z/H] ~ f(log M*) + CG4`. A quadratic is admitted only if five-fold group CV improves Control4C RMSE by at least 2%. Groups are resampled within sample for 1,000 bootstrap replicates; no p-values or model shopping are used.
""")

code("""
contrasts = pd.read_csv(HERE / "outputs/zstar/zstar_contrasts.csv")
primary = contrasts[(contrasts["product"] == za.PRIMARY) & (contrasts["control"] == "Control4C")]
display(primary[["role", "n_CG4", "n_control", "degree", "beta", "cluster_se", "standardized_effect", "ci68_lo", "ci68_hi", "ci95_lo", "ci95_hi"]])
display(Image(filename=str(HERE / "figures/zstar/mass_metallicity.png")))
display(Image(filename=str(HERE / "figures/zstar/direct_effects.png")))
""")

md("""
## 5. Morphology, SF state, and velocity dispersion

The unconditioned mass-adjusted contrast remains primary. Subsets show how the total result decomposes. Sigma is added only alongside mass in early/passive sensitivities and is not treated as a replacement for mass.
""")

code("""
decomp = pd.read_csv(HERE / "outputs/zstar/zstar_decomposition.csv")
display(decomp[decomp["product"] == za.PRIMARY][["role", "subset", "n_CG4", "n_control", "beta", "ci95_lo", "ci95_hi"]])
display(Image(filename=str(HERE / "figures/zstar/decomposition.png")))
""")

md("""
## 6. Aperture, redshift, S/N, and model robustness
""")

code("""
robust = pd.read_csv(HERE / "outputs/zstar/zstar_robustness.csv")
display(robust[["role", "variant", "n_CG4", "n_control", "beta", "ci95_lo", "ci95_hi"]])
display(Image(filename=str(HERE / "figures/zstar/robustness.png")))
""")

md("""
## 7. Historical index sensitivity to stellar mass

This is a new sensitivity table; it does not overwrite Claude's historical output. The paired comparison uses identical mass-complete satellite rows under the Claude-like morphology-plus-sigma model and after adding stellar mass.
""")

code("""
indices = pd.read_csv(HERE / "outputs/zstar/index_mass_sensitivity.csv")
c4c = indices[(indices["control"] == "Control4C") & indices["specification"].isin(["Claude-like, mass-complete", "+ stellar mass"])]
display(c4c.pivot(index="index", columns="specification", values=["beta_CG4", "cluster_se"]))
display(Image(filename=str(HERE / "figures/zstar/index_mass_sensitivity.png")))
""")

md("""
## 8. Full interpretation

The generated report contains the complete numerical audit, catalogue documentation, caveats, and file map.
""")

code("""
display(Markdown((HERE / "ZSTAR_REPORT.md").read_text()))
""")

(HERE / "metallicity_zstar.ipynb").write_text(nbf.writes(NB))
print(HERE / "metallicity_zstar.ipynb")
