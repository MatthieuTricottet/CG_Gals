"""Build readable tables from the machine-readable outputs."""

from referee_common import HERE, TABLES, pd

SECTIONS = {
    "Sample construction and compactness support": {
        "sample_counts": None,
        "compact_support": None,
        "compact_overlap": None,
        "compact_model_flow": None,
        "appendix_f_reproduction": None,
    },
    "Compactness model and matching": {
        "compact_model_coefficients": (
            "spec in ['common_slope','interaction'] and term in ['cg','xc','cg:xc']",
            "spec term beta low high p cg_n control_n hosts",
        ),
        "compact_marginal_contrasts": (
            "kpc in [150,175,200] and spec in ['common_slope','interaction','joint_satellite_support']",
            "spec kpc cg_est control_est difference low high cg_hosts control_hosts",
        ),
        "compact_matching_flow": None,
        "compact_matching_effects": (
            None,
            "spec pairs pairs_both_outcomes cg_labels cc_labels cg_fraction cc_fraction estimate low high effective_blocks",
        ),
        "compact_matching_balance": (
            "spec == 'compactness_matched'",
            "stage covariate cg cc smd",
        ),
    },
    "Mass, completeness and star formation": {
        "mass_interaction_tests": None,
        "satellite_mass_bins": (
            "region == 'full' and mass_bin.notna()",
            "sample mass_bin outcome n events estimate low high hosts",
        ),
        "completeness_flow": (
            None,
            "sample bin all retained flux_only threshold_median quartets hosts",
        ),
        "mass_marginal_contrasts": (
            "mass in [9.75,10.25,10.75,11] and model == 'interaction' and region in ['primary','flux_and_BGG']",
            "outcome control region mass cg_hosts control_hosts difference low high",
        ),
        "quenching_ceiling": (
            "region == 'primary'",
            "control bin n counterfactual_control implied_delta_pp_for_OR_1_5 difference low high",
        ),
        "sf_mass_adjusted": (
            "control == 'Control4C'",
            "outcome selection aperture n cg_n control_n effect_difference effect_low effect_high",
        ),
    },
    "Within-host geometry and fits": {
        "within_host_models": (
            None,
            "spec n_complete hosts n outcome_informative both_informative cg noncg eligible_range_overlap_hosts OR low high p status",
        ),
    },
    "Catalogue checks": {
        "hmcg_control_counts": None,
        "hmcg_isolation_summary": None,
        "boss_sfr_summary": None,
        "paper_match_denominators": (
            None,
            "population stage n_pairs candidate_edges complete_cg cg_with_candidate cg_hosts control_hosts blocks effective_blocks estimate low high",
        ),
        "gz1_dr7_recovery_summary": None,
        "missing_morphology_bounds": (
            None,
            "population sample valid n observed lower upper",
        ),
        "colour_selection_sensitivity": (
            None,
            "control restriction outcome cg_n control_n effect_difference effect_low effect_high",
        ),
        "paper1_distance_audit": (
            None,
            "sample n raw_distance_residual_DA raw_distance_residual_DL true_percent_decrease_median rho_tc_Mvir_raw rho_tc_Mvir_DA",
        ),
    },
}


def markdown_table(frame):
    def fmt(x):
        if pd.isna(x):
            return "—"
        if isinstance(x, float):
            return f"{x:.4g}"
        return str(x).replace("|", r"\|")

    return "\n".join(
        [
            "| " + " | ".join(frame.columns) + " |",
            "| " + " | ".join("---" for _ in frame.columns) + " |",
            *(
                "| " + " | ".join(fmt(x) for x in row) + " |"
                for row in frame.itertuples(index=False, name=None)
            ),
        ]
    )


def write_summary():
    lines = [
        "# Referee reanalysis tables",
        "",
        "Intervals are 95% unless marked otherwise. Estimates in the matching and raw fraction tables are on a 0–1 scale. Marginal differences for binary outcomes are in percentage points. The CSV files retain model formulas, warnings, exact denominators and additional sensitivity results.",
        "",
    ]
    for section, tables in SECTIONS.items():
        lines += ["## " + section, ""]
        for name, settings in tables.items():
            d = pd.read_csv(TABLES / f"{name}.csv")
            if settings:
                query, columns = settings
                if query:
                    d = d.query(query, engine="python")
                d = d[columns.split()]
            lines += [
                f"### {name.replace('_',' ').capitalize()}",
                "",
                f"[Full CSV](tables/{name}.csv)",
                "",
                markdown_table(d),
                "",
            ]
    (HERE / "TABLES.md").write_text("\n".join(lines), encoding="utf-8")
