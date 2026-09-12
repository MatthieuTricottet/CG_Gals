"""Build metallicity_scoping.ipynb (nbformat). Every phase cell runs the phase script
(idempotent: CAS results are cached under work/) and every displayed number is read
from outputs/metallicity_scoping.json."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: C.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# Stellar-metallicity feasibility scoping — CG4 satellites vs controls

**Isolated, read-only exploration.** Everything this notebook writes lives under `exploration/metallicity/`;
every pre-existing repository file is opened read-only. No number below is typed by hand: each cell runs the
corresponding phase script and then renders tables from `outputs/metallicity_scoping.json`.

| phase | script | writes |
|---|---|---|
| 0 inventory | `phase0_inventory.py` | `INVENTORY.md`, JSON `phase0` |
| 1 extraction | `phase1_fetch.py`, `phase1_assemble.py` | `work/*`, blind table + key, JSON `phase1` |
| 2 census & power (blinded) | `phase2_census_power.py` | `figures/fig_power_vs_SN.png`, JSON `phase2` |
| 3 systematics (blinded) | `phase3_systematics.py` | `figures/fig_aperture_bias.png`, `figures/fig_degeneracy_plane.png`, JSON `phase3` |
| 4 unblinded contrasts | `phase4_fetch_hosts.py`, `phase4_unblinded.py` | `figures/fig_phase4_contrasts.png`, `outputs/usable_satellites.csv`, JSON `phase4` |
| 5 report | `phase5_report.py` | `REPORT.md` |

Gates A/B/C are recorded in the session log; the JSON frozen before unblinding is `outputs/metallicity_scoping_preunblind_snapshot.json`
(sha256 in `outputs/PREUNBLIND_SHA256.txt`).""")

code("""import json, os, subprocess, sys
from IPython.display import Image, Markdown, display
import pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 3)

HERE = os.path.abspath("")                       # exploration/metallicity
assert HERE.endswith(os.path.join("exploration", "metallicity")), HERE
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PY = sys.executable
JSON_PATH = os.path.join(HERE, "outputs", "metallicity_scoping.json")

def run(script, tail=12):
    r = subprocess.run([PY, os.path.join(HERE, script)], cwd=REPO, capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if "Warning" not in l and "warnings.warn" not in l]
    print("\\n".join(lines[-tail:]))
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed (exit {r.returncode})")

def J():
    with open(JSON_PATH) as fh:
        return json.load(fh)

print(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout)""")

md("## Phase 0 — Inventory")
code("""run("phase0_inventory.py", tail=3)
p0 = J()["phase0"]
rows = []
for s, e in p0["samples"].items():
    r = e["specobjid_run2d_distribution"]
    rows.append({"sample": s, "N_groups": e["N_groups"], "N_gals": e["N_gals"], "N_BGG": e["N_BGG_rankM_eq_1"], "N_sat": e["N_sat_rankM_gt_1"],
                 "in_SDSS_cache": e["N_in_SDSS_cache_by_objid"], "run2d=26": r.get("26", 0), "run2d=700 (BOSS)": r.get("700", 0),
                 "Rchl_r_valid_sat": e["size_cache"]["N_sat_Rchl_r_valid"], "NoGZ": e["morphology_counts"].get("NoGZ", 0)})
display(pd.DataFrame(rows))
a = p0["availability"]
display(Markdown(f"**galSpecIndx downloaded:** {a['galSpecIndx_downloaded']} — **per-galaxy σ local:** {a['velocity_dispersion_per_galaxy_local']} — "
                 f"**spectrum S/N local:** {a['spectrum_sn_per_galaxy_local']}  \\nJoin key: {p0['join']['canonical_galaxy_key']}  \\nBGG flag: {p0['join']['bgg_flag']}"))""")

md("## Phase 1 — Index extraction (MPA-JHU `galSpecIndx` + `galSpecInfo` + `SpecObjAll`)\nThe fetch is cached in `work/cas_spectra_raw.csv`; rerunning is offline.")
code("""run("phase1_fetch.py", tail=4)
run("phase1_assemble.py", tail=0)
p1 = J()["phase1"]
ja = p1["join_accounting"]
rows = []
for s in ["CG4", "RG4", "Control4B", "Control4C", "ALL"]:
    for tag in ["all", "satellites"]:
        a = ja[s][tag]
        rows.append({"sample": s, "subset": tag, "N_input": a["N_input"], "any_spectrum": a["N_any_spectrum"], "MPA_result": a["N_mpa_result"],
                     "reliable": a["N_mpa_reliable"], "core7_valid": a["N_valid_core7"], "core7+sigma+SN": a["N_valid_core7_and_sigma_and_SN"], "MgFe_valid": a["N_valid_MgFe"]})
display(pd.DataFrame(rows))
display(pd.DataFrame(ja["loss_reason_by_sample_satellites"]).fillna(0).astype(int).T)
dm = p1["datamodel_verification"]
display(Markdown("\\n".join(f"- **{k}**: {v['status'] if isinstance(v, dict) and 'status' in v else v}" for k, v in dm.items() if k != "sources")))
sn = {k: v for k, v in p1["sentinel_notes"].items() if isinstance(v, dict)}
display(pd.DataFrame(sn).T[["n_err_minus1", "z_p10_p50_p90_of_flagged"]])
display(Markdown(p1["sentinel_notes"]["interpretation"]))
display(Markdown(f"Blinding: {p1['blinding']}"))""")

md("## Phase 2 — Usability census and power (BLINDED labels S1–S4)")
code("""run("phase2_census_power.py", tail=0)
p2 = J()["phase2"]
cen = p2["census"]
rows = []
for t in p2["sn_thresholds"]:
    c = cen[str(t)]
    row = {"S/N >": t, "pooled all": c["pooled_dedup"]["all"], "pooled sat": c["pooled_dedup"]["sat"], "pooled BGG": c["pooled_dedup"]["bgg"]}
    for lab, v in c["by_blind_sample"].items():
        row[f"{lab} sat (groups)"] = f"{v['sat']} ({v['N_groups_with_ge1_sat']})"
    rows.append(row)
display(pd.DataFrame(rows))
dec = p2["decisive_SN20"]
display(Markdown(f"**S/N>{dec['threshold']:.0f}:** {dec['N_sat_surviving']}/{dec['N_sat_full']} = {100*dec['frac_sat_surviving']:.1f}% of satellites survive; "
                 f"survival by M_r quartile (bright→faint): {[round(v['frac_surv'],2) for v in dec['survival_by_M_r_quartile'].values()]}; "
                 f"by log M* quartile (low→high): {[round(v['frac_surv'],2) for v in dec['survival_by_lgm_quartile'].values()]}; "
                 f"fraction of survivors in the brightest quartile: {dec['frac_of_survivors_in_M_r_Q1']:.2f}"))
eb = p2["error_budget"]
display(pd.DataFrame({k: {e["bin"]: e["median_err"] for e in v} for k, v in eb["median_err_by_bin"].items() if not k.endswith("_sub")}).T)
display(pd.DataFrame(eb["satellite_scatter_vs_error"]).T)
pw = p2["power"]["by_threshold"]
rows = []
for t, e in pw.items():
    small = e["two_smallest_labels"][0]
    row = {"S/N >": int(t), "smallest label": small, "N_g": e["sizes"][small]["N_g"], "n_sat": round(e["sizes"][small]["n_sat"], 2)}
    for k in ["Mgb", "Fe5270", "Fe5335", "MgFe", "MgbFe", "Mg2", "D4000n", "HdA", "Hb_sub"]:
        v = e["indices"].get(k, {})
        row[f"Δmin {k}"] = v.get("per_blind_sample", {}).get(small, {}).get("delta_min")
        row[f"ρ {k}"] = v.get("rho_used")
    rows.append(row)
display(pd.DataFrame(rows).set_index("S/N >"))
display(pd.DataFrame(p2["yardsticks"]["by_threshold"]["20"]).T)
display(Image(os.path.join(HERE, p2["figure"]), width=1000))""")

md("> **GATE B** — reported in the session log; approval delegated by the author (\"Do what you decide is the most relevant\"). Decision: proceed.")
md("## Phase 3 — Systematics (BLINDED)")
code("""run("phase3_systematics.py", tail=0)
p3 = J()["phase3"]
display(pd.DataFrame(p3["systematics_vs_delta_min"]).T)
g = p3["aperture"]["gradients"]["SN_gt_20"]
display(pd.DataFrame({k: {"slope/dex apfrac": v["slope_per_dex_apfrac"], "±": v["se"], "slope/dex apfrac @σ": v["slope_per_dex_apfrac_at_fixed_sigma"], "± ": v["se_fixed_sigma"],
                          "slope/z @σ": v["slope_per_unit_z_at_fixed_sigma"], "±  ": v["se_z_fixed_sigma"]} for k, v in g.items()}).T)
sf = p3["balance"]["selection_function"]
display(Markdown(f"Selection function P(S/N>20) logit coefficients per SD: {dict((k, round(v, 2)) for k, v in sf['coef_per_SD'].items())}"))
m = p3["morphology_confound"]["by_SN_cut"]["SN_gt_20"]["indices"]
display(pd.DataFrame({k: {"E−S @σ": v["E_minus_S_at_fixed_sigma"], "±": v["se"], "raw median diff": v["E_minus_S_raw_median_diff"]} for k, v in m.items()}).T)
dg = p3["degeneracy"]
display(pd.DataFrame({k: v for k, v in dg.items() if isinstance(v, dict)}).T[["n", "spearman", "span_over_err_MgFe", "resid_sd_over_median_err"]])
display(Image(os.path.join(HERE, p3["aperture"]["figure"]), width=900))
display(Image(os.path.join(HERE, dg["figure"]), width=900))""")

md("""> **GATE C** — unblinding. The blinded JSON was frozen first (`outputs/PREUNBLIND_SHA256.txt`). Approval delegated by the author; Phase 4 proceeds only because no
> Phase 3 systematic exceeds Δ_min by more than the morphology term, which the Phase 4 design controls by construction.
>
> **Phase 4 is an exploratory scoping run, not an inference: no p-values, no multiple-comparison correction.** Set `UNBLIND = False` to skip it.""")
code("""UNBLIND = True
if UNBLIND:
    run("phase4_fetch_hosts.py", tail=2)
    run("phase4_unblinded.py", tail=0)
    p4 = J()["phase4"]
    display(Markdown(f"**{p4['statement']}**"))
    display(pd.DataFrame(p4["true_label_census_satellites"]).T)
    for variant in ["all_valid", "SN_gt_20"]:
        a = p4["results"][variant]["a_galaxy_level"]; w = p4["results"][variant]["d_within_host"]["satellites_only"]
        rows = []
        for k in ["Mgb", "Fe5270", "Fe5335", "MgFe", "MgbFe", "Mg2", "D4000n", "HdA_sub", "Hb_sub"]:
            row = {"index": k}
            for c in ["Control4B", "Control4C", "RG4"]:
                for sub in ["all_morph", "ellipticals", "spirals"]:
                    v = a[c]["indices"].get(k, {}).get(sub)
                    row[f"{c[:3]}{c[-2:]} {sub[:3]}"] = None if v is None else f"{v['coef']:+.3f}±{v['cluster_se']:.3f}"
            v = w["indices"].get(k)
            row["within-host"] = None if v is None else f"{v['coef']:+.3f}±{v['cluster_se']:.3f}"
            row["Δmin (no cut)"] = p4["delta_min_reference_SN0_smallest_label"].get(k)
            rows.append(row)
        display(Markdown(f"**Variant `{variant}`** (coef ± group-clustered SE; CG4 − control at fixed morphology & σ)"))
        display(pd.DataFrame(rows).set_index("index"))
    display(Image(os.path.join(HERE, p4["figure"]), width=1000))""")

md("## Phase 5 — Report and repository check")
code("""run("phase5_report.py", tail=0)
display(Markdown(open(os.path.join(HERE, "REPORT.md")).read()))
status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout
print(status)
mine = [l for l in status.splitlines() if l.strip().endswith("exploration/")]
others = [l for l in status.splitlines() if not l.strip().endswith("exploration/")]
print("untracked path written by this study:", mine)
print("other entries (pre-existing tmp/ or concurrent-session edits; not written by this study):", others)""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "exploration/metallicity/metallicity_scoping.ipynb")
print("notebook written")
