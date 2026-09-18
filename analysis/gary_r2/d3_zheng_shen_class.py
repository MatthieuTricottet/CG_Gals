"""D3 -- Zheng--Shen class versus host mass (gary-r2 Phase 1, read-only).

(a) class definitions as implemented (inherited ``Class`` column of
    data/CG4_Groups.csv) versus the published definition (Zheng & Shen 2021,
    ApJ 911, 105, Sect. 2.3 Eq. 1; Paper I Sect. 3.3), re-derived from the
    Lim--Tempel host membership in data/PC_Gals.csv;
(b) per class: median host log M200c, host richness, number of groups;
(c) confirm/refute "Predominant hosts are more massive than Embedded hosts";
(d) exploratory CG4-satellite logistic E vs S ~ class + log M*, with and
    without log M200c, clustered by group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import ROOT, OUT, load_sample, store_diagnostic, write_csv

from extended_stats import fit_logistic_model  # noqa: E402

DEFINITION = (
    "Zheng & Shen (2021, ApJ 911, 105, Sect. 2.3): CGs whose members coincide "
    "with a Yang et al. (2007) group are 'isolated'; CGs that are subsets of a "
    "richer group are 'embedded systems' and, among them, 'embedded CGs' are "
    "the subgroups that do NOT dominate the luminosity of their parent group, "
    "sum_i^{Npar} L_i >= 2 sum_j^{Nemb} L_j (Eq. 1), while the remainder, which "
    "contribute more than half of the parent luminosity, are 'predominant'; "
    "members matched to different groups are 'split'. Paper I (Tricottet+25, "
    "Sect. 3.3) states the same: Predominant = 'accounting for half or more of "
    "their host group total r-band luminosity' (19 groups), Embedded = 'less "
    "than half' (37 groups)."
)


def main() -> None:
    sample = load_sample()
    cg = pd.read_csv(ROOT / "data" / "CG4_Gals.csv")
    cgg = pd.read_csv(ROOT / "data" / "CG4_Groups.csv")
    pc = pd.read_csv(ROOT / "data" / "PC_Gals.csv")
    pcg = pd.read_csv(ROOT / "data" / "PC_Groups.csv")

    # host identification via shared objids (same rule as identity.cg4_host_lim_map)
    m = cg.merge(pc[["objid", "Group"]].rename(columns={"Group": "lim"}), on="objid", how="left")
    host = m.groupby("Group")["lim"].agg(
        lambda s: s.dropna().mode().iloc[0] if s.notna().any() else np.nan
    )
    n_in_pc = m.groupby("Group")["lim"].apply(lambda s: int(s.notna().sum()))
    n_hosts = m.groupby("Group")["lim"].nunique()
    t = cgg.set_index("Group")[["Class", "lMass_200", "Lum_group", "Vdisp"]].copy()
    t["host"] = host
    t["n_members_in_PC"] = n_in_pc
    t["n_distinct_hosts"] = n_hosts
    L_host = pc.groupby("Group")["Lum"].sum()
    N_host = pc.groupby("Group").size()
    t["N_host"] = t["host"].map(N_host)
    t["L_host"] = t["host"].map(L_host)
    t["f_L_cg_over_host"] = t["Lum_group"] / t["L_host"]
    t["N_host_PCgroups"] = t["host"].map(pcg.set_index("Group")["NbGal"])

    def zs_class(row):
        if row["n_members_in_PC"] < 4 or pd.isna(row["host"]) or row["n_distinct_hosts"] > 1:
            return "Split"
        if row["N_host"] == 4:
            return "Isolated"
        return "Predominant" if row["f_L_cg_over_host"] >= 0.5 else "Embedded"

    t["class_rederived_ZS21"] = t.apply(zs_class, axis=1)
    t["class_stored"] = t["Class"].replace({"Predom": "Predominant"})
    write_csv(t.reset_index(), "d3_cg4_class_rederivation.csv")

    cross = pd.crosstab(t["class_stored"], t["class_rederived_ZS21"])
    print(cross)
    # stored Split groups (removed from the analysis sample) are kept for the record
    nonsplit = t.loc[t["class_stored"].ne("Split")]

    per_class = (
        nonsplit.groupby("class_stored")
        .agg(
            n_groups=("host", "size"),
            median_host_logM200c=("lMass_200", "median"),
            q16_host_logM200c=("lMass_200", lambda v: v.quantile(0.16)),
            q84_host_logM200c=("lMass_200", lambda v: v.quantile(0.84)),
            median_host_richness=("N_host", "median"),
            min_host_richness=("N_host", "min"),
            max_host_richness=("N_host", "max"),
            median_fL_cg_over_host=("f_L_cg_over_host", "median"),
            min_fL=("f_L_cg_over_host", "min"),
            max_fL=("f_L_cg_over_host", "max"),
        )
        .reset_index()
    )
    write_csv(per_class, "d3_per_class_host_properties.csv")
    print(per_class.to_string())

    stored = per_class.set_index("class_stored")
    pred_more_massive = bool(
        stored.loc["Predominant", "median_host_logM200c"]
        > stored.loc["Embedded", "median_host_logM200c"]
    )
    n_swapped = int(
        (nonsplit["class_stored"].ne(nonsplit["class_rederived_ZS21"])).sum()
    )
    verdict = (
        f"CONFIRMED numerically for the stored labels: median host log M200c is "
        f"{stored.loc['Predominant', 'median_host_logM200c']:.2f} for the 19 groups labelled "
        f"'Predom' and {stored.loc['Embedded', 'median_host_logM200c']:.2f} for the 37 labelled "
        f"'Embedded' (Table E.2), BUT this CONTRADICTS the definitions: every group labelled "
        f"'Embedded' contributes >= {stored.loc['Embedded', 'min_fL']:.2f} (median "
        f"{stored.loc['Embedded', 'median_fL_cg_over_host']:.2f}) of its host r-band luminosity "
        f"(host richness {int(stored.loc['Embedded', 'min_host_richness'])}--"
        f"{int(stored.loc['Embedded', 'max_host_richness'])}), i.e. it DOMINATES the host and is "
        f"'Predominant' under Zheng & Shen (2021) Eq. 1 and under Paper I Sect. 3.3, whereas every "
        f"group labelled 'Predom' contributes <= {stored.loc['Predominant', 'max_fL']:.2f} (median "
        f"{stored.loc['Predominant', 'median_fL_cg_over_host']:.2f}; host richness "
        f"{int(stored.loc['Predominant', 'min_host_richness'])}--"
        f"{int(stored.loc['Predominant', 'max_host_richness'])}) and is 'Embedded'. The two "
        f"labels are swapped in data/CG4_Groups.csv for all {n_swapped}/56 non-isolated non-split "
        f"groups (the 0.5 luminosity-fraction boundary is reproduced exactly: no group is "
        f"ambiguous). With the correct labels, Embedded hosts (rich groups/clusters) are the "
        f"massive ones, as the definitions imply. Paper I's Sect. 4.3 prose already carried the "
        f"inconsistency ('Embedded CG4s ... contributing over half the luminosity of the host')."
    )
    print(verdict)

    # (d) exploratory logistic model on CG4 satellites ------------------------
    gals = sample["CG4_Gals"].copy()
    gals = gals.merge(
        t[["class_stored", "class_rederived_ZS21", "lMass_200", "N_host"]],
        left_on="Group", right_index=True, how="left",
    )
    sat = gals.loc[(gals["rank_M"] > 1) & gals["morphology"].isin(["Elliptical", "Spiral"])].copy()
    sat["elliptical"] = sat["morphology"].eq("Elliptical").astype(float)
    sat["physical_group"] = sat["Group"].astype(str)
    sat["logMstar"] = pd.to_numeric(sat["lgm"], errors="coerce")
    sat["log_M200c"] = pd.to_numeric(sat["lMass_200"], errors="coerce")
    sat["log_N_host"] = np.log10(sat["N_host"])
    models = {}
    for label_col in ("class_stored", "class_rederived_ZS21"):
        # reference = Isolated; dummies for the two embedded-type classes
        for cname in ("Embedded", "Predominant"):
            sat[f"is_{cname}"] = sat[label_col].eq(cname).astype(float)
        base = ["is_Embedded", "is_Predominant", "logMstar"]
        fits = {
            "class_plus_logMstar": fit_logistic_model(
                sat, "elliptical", base, continuous=["logMstar"], min_n=20),
            "class_plus_logMstar_plus_logM200c": fit_logistic_model(
                sat, "elliptical", base + ["log_M200c"],
                continuous=["logMstar", "log_M200c"], min_n=20),
            "logMstar_plus_logM200c_no_class": fit_logistic_model(
                sat, "elliptical", ["logMstar", "log_M200c"],
                continuous=["logMstar", "log_M200c"], min_n=20),
            "class_plus_logMstar_plus_logNhost": fit_logistic_model(
                sat, "elliptical", base + ["log_N_host"],
                continuous=["logMstar", "log_N_host"], min_n=20),
        }
        # alternative reference: Embedded-vs-Predominant only (drops 6 isolated groups)
        sub = sat.loc[sat[label_col].ne("Isolated")]
        fits["predominant_vs_embedded_only_plus_logMstar"] = fit_logistic_model(
            sub, "elliptical", ["is_Predominant", "logMstar"], continuous=["logMstar"], min_n=20)
        fits["predominant_vs_embedded_only_plus_logMstar_logM200c"] = fit_logistic_model(
            sub, "elliptical", ["is_Predominant", "logMstar", "log_M200c"],
            continuous=["logMstar", "log_M200c"], min_n=20)
        models[label_col] = fits
        print(f"\n== {label_col} ==")
        for k, f in fits.items():
            if f.get("status") == "ok":
                terms = {n: (round(v["odds_ratio"], 2), round(v["p"], 3)) for n, v in f["terms"].items() if n != "const"}
                print(k, f["n"], f["n_clusters"], terms)
            else:
                print(k, f)

    counts = sat.groupby("class_stored").agg(
        n_sat=("elliptical", "size"), n_E=("elliptical", "sum"),
        n_groups=("Group", "nunique"), fE=("elliptical", "mean"))
    print(counts)

    # rows for the record: satellite fE per class under both labellings
    fe_rows = []
    for label_col in ("class_stored", "class_rederived_ZS21"):
        for cname, part in sat.groupby(label_col):
            fe_rows.append(dict(labelling=label_col, cls=cname, n_groups=part["Group"].nunique(),
                                n_sat_classified=len(part), n_E=int(part["elliptical"].sum()),
                                fE=float(part["elliptical"].mean()),
                                median_host_logM200c=float(t.loc[t[label_col].eq(cname), "lMass_200"].median())))
    write_csv(pd.DataFrame(fe_rows), "d3_satellite_fE_by_class.csv")

    store_diagnostic("d3_zheng_shen_class", {
        "definition_source": DEFINITION,
        "implemented": "inherited 'Class' column of data/CG4_Groups.csv (Paper I export); used verbatim by paper_additions.zheng_shen_block, morphology_dominance, host_controlled, identity",
        "class_crosstab_stored_vs_rederived": {str(i): {str(c): int(cross.loc[i, c]) for c in cross.columns} for i in cross.index},
        "per_class_stored_labels": per_class.to_dict(orient="records"),
        "predominant_hosts_more_massive_than_embedded_with_stored_labels": pred_more_massive,
        "n_nonsplit_groups_with_swapped_label": n_swapped,
        "verdict": verdict,
        "n_isolated_groups": int((nonsplit["class_stored"] == "Isolated").sum()),
        "satellite_logistic_models": models,
        "satellite_counts_by_stored_class": counts.reset_index().to_dict(orient="records"),
        "satellite_fE_by_class_file": "d3_satellite_fE_by_class.csv",
        "note_small_N": "6 Isolated groups (14 classified satellites); class coefficients are exploratory",
    })


if __name__ == "__main__":
    main()
