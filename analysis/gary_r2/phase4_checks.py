"""gary-r2 Phase 4 consistency checks on the rendered manuscript.

Usage: python analysis/gary_r2/phase4_checks.py

1. Every figure, table, and appendix label in output/paper/paper.tex is
   referenced at least once, and every \\ref resolves (no dangling refs).
2. Numeric literals in the template that were not present at tag
   pre-gary-r2 are listed for review (they must be method constants, not
   results).
3. Count bookkeeping recomputed from the committed data files and compared
   with the rendered tables: sample sizes (Table 1), sSFR classes (Table 2),
   morphology classes (Table 3), Zheng--Shen class counts (App. tables),
   overlap table (App. C), and the availability totals of Fig. F.2.
Exit status 1 on any failure.
"""

from __future__ import annotations

import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "output" / "paper" / "paper.tex"
TEMPLATE = ROOT / "src" / "paper_template" / "paper_template.tex"
FAILED = False


def report(ok: bool, message: str) -> None:
    global FAILED
    print(f"  {'PASS' if ok else 'FAIL'}: {message}")
    FAILED |= not ok


def check_references(tex: str) -> None:
    print("references")
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = set(re.findall(r"\\(?:page)?ref\{([^}]+)\}", tex))
    for prefix in ("fig:", "tab:"):
        unreferenced = sorted(l for l in labels if l.startswith(prefix) and l not in refs)
        report(not unreferenced, f"all {prefix} labels referenced" + (f" (missing: {unreferenced})" if unreferenced else ""))
    dangling = sorted(r for r in refs if r not in set(labels))
    report(not dangling, "no dangling references" + (f" ({dangling})" if dangling else ""))
    # appendices: every \section inside the appendix environment carries an app: label that is referenced
    appendix = tex[tex.index("\\begin{appendix}"):]
    sections = re.findall(r"\\section\{[^}]*\}\\label\{([^}]+)\}", appendix)
    missing = [s for s in sections if s not in refs]
    report(not missing, f"all {len(sections)} appendices referenced" + (f" (missing: {missing})" if missing else ""))


def numeric_literals(template: str) -> set[str]:
    text = re.sub(r"<<.*?>>", " ", template, flags=re.S)
    text = re.sub(r"<%.*?%>", " ", text, flags=re.S)
    text = re.sub(r"<#.*?#>", " ", text, flags=re.S)
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\(cite[pt]?|citealt|label|ref|url|href|includegraphics|input|bibliography)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    return set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", text))


def check_literals() -> None:
    print("template numeric literals")
    old = subprocess.run(["git", "show", "pre-gary-r2:src/paper_template/paper_template.tex"],
                         capture_output=True, text=True, cwd=ROOT).stdout
    new = TEMPLATE.read_text()
    added = sorted(numeric_literals(new) - numeric_literals(old), key=lambda v: float(v))
    print(f"  INFO: literals present now but not at pre-gary-r2: {added}")
    removed = sorted(numeric_literals(old) - numeric_literals(new), key=lambda v: float(v))
    print(f"  INFO: literals removed since pre-gary-r2: {removed}")


def check_counts(tex: str) -> None:
    print("count bookkeeping")
    with open(ROOT / "data" / "processed_sample.pkl", "rb") as fh:
        sample = pickle.load(fh)
    results = json.load(open(ROOT / "output" / "results.json"))
    groups = pd.read_csv(ROOT / "data" / "CG4_Groups.csv")
    samples = ["CG4", "Control4B", "Control4C", "RG4"]

    # Table 1: groups / galaxies
    tab1 = tex[tex.index("\\label{tab:samples}"):]
    tab1 = tab1[: tab1.index("\\end{tabular}")]
    for name, tex_name in zip(samples, ["\\CG", "\\CB", "\\CC", "\\RG"]):
        gals = sample[name + "_Gals"]
        n_g, n_gal = int(gals["Group"].nunique()), int(len(gals))
        row = re.search(re.escape(tex_name) + r"\{\}\s*&\s*(\d+)\s*&\s*(\d+)", tab1)
        report(row is not None and (int(row.group(1)), int(row.group(2))) == (n_g, n_gal),
               f"Table 1 {name}: {n_g} groups / {n_gal} galaxies")

    # Table 2 (sSFR classes) and Table 3 (morphology classes)
    for name in samples:
        gals = sample[name + "_Gals"]
        n_miss = int((gals["sSFR_status"] == "NosSFR").sum())
        n_q = int((gals["sSFR_status"] == "Quenched").sum())
        n_sf = int((gals["sSFR_status"] == "Starforming").sum())
        key = {"CG4": "\\CG", "Control4B": "\\CB", "Control4C": "\\CC", "RG4": "\\RG"}[name]
        tab2 = tex[tex.index("\\label{tab:sSFR_status}"):]
        tab2 = tab2[: tab2.index("\\end{tabular}")]
        row = re.search(re.escape(key) + r"\s*&\s*(\d+)\s*\(\d+\\%\)\s*&\s*(\d+)\s*\(\d+\\%\)\s*&\s*(\d+)", tab2)
        report(row is not None and tuple(map(int, row.groups())) == (n_miss, n_q, n_sf),
               f"Table 2 {name}: {n_miss}/{n_q}/{n_sf} (no sSFR/quenched/star-forming)")
        counts = gals["morphology"].value_counts()
        tab3 = tex[tex.index("\\label{tab:morphologies}"):]
        tab3 = tab3[: tab3.index("\\end{tabular}")]
        row = re.search(re.escape(key) + r"\s*&\s*(\d+)\s*\(\d+\\%\)\s*&\s*(\d+)\s*\(\d+\\%\)\s*&\s*(\d+)\s*\(\d+\\%\)\s*&\s*(\d+)", tab3)
        expected = tuple(int(counts.get(k, 0)) for k in ["Elliptical", "Spiral", "Uncertain", "NoGZ"])
        report(row is not None and tuple(map(int, row.groups())) == expected,
               f"Table 3 {name}: E/S/U/NoGZ = {expected}")

    # Zheng--Shen class counts (tab:zheng_shen_fe) and class-count table
    n_by_class = groups.loc[groups["Class"] != "Split", "Class"].value_counts().to_dict()
    macros_tex = (ROOT / "output" / "paper" / "additions_macros.tex").read_text()
    for cname, sh in [("Isolated", "Iso"), ("Embedded", "Emb"), ("Predom", "Pre")]:
        m = re.search(r"\\newcommand\{\\nGr" + sh + r"\}\{(\d+)\}", macros_tex)
        report(m is not None and int(m.group(1)) == n_by_class.get(cname, 0),
               f"Table zheng_shen_fe {cname}: {n_by_class.get(cname, 0)} groups (macro nGr{sh})")
    tabc = tex[tex.index("\\label{tab:morphology_dominance_class_counts}"):]
    tabc = tabc[: tabc.index("\\end{tabular}")]
    cg = sample["CG4_Gals"].merge(groups[["Group", "Class"]], on="Group", how="left")
    for cname in ["Isolated", "Embedded", "Predominant"]:
        stored = "Predom" if cname == "Predominant" else cname
        part = cg.loc[cg["Class"] == stored]
        n_e = int((part["morphology"] == "Elliptical").sum())
        n_s = int((part["morphology"] == "Spiral").sum())
        row = re.search(cname + r"\s*&\s*(\d+)/(\d+)", tabc)
        report(row is not None and (int(row.group(1)), int(row.group(2))) == (n_e, n_s),
               f"class-count table {cname}: E/S = {n_e}/{n_s}")

    # Overlap table (App. C)
    overlap = results.get("cg4_pc_quartet_overlap", {})
    if overlap:
        tabo = tex[tex.index("\\label{tab:cg_core_overlap}"):]
        tabo = tabo[: tabo.index("\\end{tabular}")]
        totals = groups["Class"].value_counts().to_dict()
        for cname, row in overlap.get("by_class", {}).items():
            label = "Predominant" if cname == "Predom" else cname
            m = re.search(label + r"\s*&\s*(\d+)\s*/\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)", tabo)
            expected = (row["n_cg4_groups"], totals.get(cname, 0), row["n_lim_groups"], row["n_galaxies"])
            report(m is not None and tuple(map(int, m.groups())) == expected,
                   f"overlap table {label}: {expected}")

    # Fig. F.2 availability totals (per-sample n_total equals the sample size)
    avail = results["extended_specialness"]["selection_diagnostics"]["availability_counts_by_sample"]
    for name in samples:
        totals = {q: v["n_total"] for q, v in avail[name].items()}
        report(all(t == len(sample[name + "_Gals"]) for t in totals.values()),
               f"availability denominators {name} = {len(sample[name + '_Gals'])}")
    # P(Q|E) table B.2 cell counts from macros vs data
    macros = (ROOT / "output" / "paper" / "additions_macros.tex").read_text()
    for name, sh in [("CG4", "cg"), ("Control4B", "cb"), ("Control4C", "cc"), ("RG4", "rg")]:
        gals = sample[name + "_Gals"]
        part = gals.loc[gals["morphology"].eq("Elliptical") & gals["sSFR_status"].isin(["Quenched", "Starforming"])]
        expected = (part["sSFR_status"] == "Quenched").mean()
        m = re.search(r"\\newcommand\{\\qE" + sh + r"\}\{([0-9.]+)\}", macros)
        report(m is not None and abs(float(m.group(1)) - expected) < 5e-4,
               f"Table B.2 P(Q|E) {name}: {expected:.3f}")


def main() -> int:
    tex = TEX.read_text()
    check_references(tex)
    check_literals()
    check_counts(tex)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
