"""Canonical galaxy/group identity layer.

One physical galaxy = one SDSS ``objid``. One physical group = one key
``("HMCG", group)`` for compact groups or ``("Lim", group)`` for the parent
catalogue and every control sample (the three control samples are all
selections of Lim et al. 2017 groups, so their ``Group`` ids share the Lim
namespace).

The identity catalogue has one row per unique objid appearing in any of
CG4 / Control4B / Control4C / RG4 / PC, with membership flags, per-sample
ranks and group ids, the CG4 class, and the CG4 -> host Lim group
cross-reference. It is the single source of truth used by the inference
layer (cluster-robust SEs, group-level resampling, control deduplication).

Run as a script to (re)write the audit products:

    python -m identity            # from src/, or
    python src/identity.py

writes ``audit/identity_catalog.csv``, ``audit/overlap_matrix.csv`` and
``audit/cg4_in_control4c_diagnostic.csv``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    import config as co
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co

SAMPLES = ["CG4", "Control4B", "Control4C", "RG4", "PC"]
AUDIT_DIR = os.path.join(co.BASE_PATH, "audit")


def load_raw_samples() -> dict[str, pd.DataFrame]:
    """Load the raw galaxy CSVs (no filtering) keyed by sample name."""

    frames = {}
    for name in SAMPLES:
        frames[name] = pd.read_csv(os.path.join(co.DATA_PATH, f"{name}_Gals.csv"))
    frames["CG4_Groups"] = pd.read_csv(os.path.join(co.DATA_PATH, "CG4_Groups.csv"))
    return frames


def cg4_class_map(frames: dict[str, pd.DataFrame]) -> pd.Series:
    """Map CG4 group id -> class (Embedded / Predom / Isolated / Split)."""

    groups = frames["CG4_Groups"]
    return groups.set_index("Group")["Class"]


def cg4_host_lim_map(frames: dict[str, pd.DataFrame]) -> pd.Series:
    """Map CG4 group id -> host Lim group id via shared objids with PC.

    Every non-split CG4 group has all four members inside exactly one Lim
    parent group. Split groups may span several or none; the modal Lim group
    is returned, or NaN when no member is in PC.
    """

    merged = frames["CG4"][["objid", "Group"]].merge(
        frames["PC"][["objid", "Group"]].rename(columns={"Group": "lim_group"}),
        on="objid",
        how="left",
    )

    def modal(series: pd.Series):
        series = series.dropna()
        return series.mode().iloc[0] if len(series) else np.nan

    return merged.groupby("Group")["lim_group"].agg(modal)


def build_identity_catalog(frames: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Return one row per unique objid with membership flags and ranks."""

    if frames is None:
        frames = load_raw_samples()

    all_ids = pd.Index(
        pd.concat([frames[name]["objid"] for name in SAMPLES]).unique(), name="objid"
    )
    catalog = pd.DataFrame(index=all_ids)

    for name in SAMPLES:
        # The committed Control4C contains one exact duplicate row (objid
        # 1237657591393157322 in Lim group 3103, part of the corrupted-lineage
        # finding A1); exact duplicates collapse to one physical galaxy, but
        # an objid recurring with *conflicting* data is a hard error.
        sample = frames[name].drop_duplicates()
        dup = sample.loc[sample["objid"].duplicated(), "objid"]
        if len(dup):
            raise ValueError(
                f"{name}_Gals.csv has objids with conflicting rows: {dup.tolist()}"
            )
        indexed = sample.set_index("objid")
        catalog[f"in_{name}"] = catalog.index.isin(indexed.index)
        catalog[f"{name}_group"] = indexed["Group"].reindex(catalog.index)
        if "rank_M" in indexed:
            catalog[f"{name}_rank_M"] = indexed["rank_M"].reindex(catalog.index)
        if "rank_dist" in indexed:
            catalog[f"{name}_rank_dist"] = indexed["rank_dist"].reindex(catalog.index)

    classes = cg4_class_map(frames)
    hosts = cg4_host_lim_map(frames)
    catalog["CG4_class"] = catalog["CG4_group"].map(classes)
    catalog["CG4_host_lim_group"] = catalog["CG4_group"].map(hosts)

    # Canonical physical-group key. Control and PC group ids share the Lim
    # namespace; a CG4 galaxy that is also a Lim-catalogue member unifies
    # with its host Lim group so overlapping rows cluster together.
    lim_group = catalog["PC_group"]
    for name in ["Control4B", "Control4C", "RG4"]:
        lim_group = lim_group.fillna(catalog[f"{name}_group"])
    lim_group = lim_group.fillna(catalog["CG4_host_lim_group"])
    catalog["lim_group"] = lim_group
    catalog["physical_group"] = np.where(
        lim_group.notna(),
        "Lim:" + lim_group.astype("Int64").astype(str),
        "HMCG:" + catalog["CG4_group"].astype("Int64").astype(str),
    )
    return catalog.reset_index()


def overlap_matrix(frames: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Pairwise overlap between samples, in galaxies and in groups.

    Group overlap counts, for each pair (A, B), the physical Lim groups that
    contribute galaxies to both samples (for CG4 the host Lim group is used).
    """

    if frames is None:
        frames = load_raw_samples()
    catalog = build_identity_catalog(frames)

    rows = []
    for a in SAMPLES:
        for b in SAMPLES:
            in_a = catalog[f"in_{a}"]
            in_b = catalog[f"in_{b}"]
            galaxies = int((in_a & in_b).sum())
            lim_a = set(catalog.loc[in_a, "lim_group"].dropna())
            lim_b = set(catalog.loc[in_b, "lim_group"].dropna())
            rows.append(
                {
                    "sample_a": a,
                    "sample_b": b,
                    "galaxies_shared": galaxies,
                    "lim_groups_shared": len(lim_a & lim_b),
                }
            )
    return pd.DataFrame(rows)


def cg4_in_control4c_table(frames: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Diagnostic table of CG4 galaxies present in the committed Control4C.

    These are the contaminated quartets that Paper I's construction excludes;
    they are scientifically interesting because they show embedded CGs are
    the projected cores of regular groups.
    """

    if frames is None:
        frames = load_raw_samples()
    catalog = build_identity_catalog(frames)
    rows = catalog[catalog["in_CG4"] & catalog["in_Control4C"]]
    table = rows[
        [
            "CG4_group",
            "CG4_host_lim_group",
            "Control4C_group",
            "objid",
            "CG4_rank_M",
            "Control4C_rank_M",
            "Control4C_rank_dist",
            "CG4_class",
        ]
    ].sort_values(["CG4_group", "CG4_rank_M"])
    return table.rename(
        columns={
            "CG4_group": "cg4_group",
            "CG4_host_lim_group": "host_lim_group",
            "Control4C_group": "control4c_group",
            "CG4_rank_M": "cg4_rank_M",
            "Control4C_rank_M": "control4c_rank_M",
            "Control4C_rank_dist": "control4c_rank_dist",
            "CG4_class": "cg4_class",
        }
    )


def cg4_in_pc_quartets_table(frames: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """CG4 galaxies that fall inside a would-be Control4C quartet of PC.

    This is the lineage-independent version of the contamination diagnostic:
    the groups listed here are exactly the ones the Paper I construction
    excludes from Control4C (60 groups for the committed PC_Gals.csv). The
    embedded/predominant classes dominating this table show that compact
    groups are frequently the projected cores of ordinary groups.
    """

    if frames is None:
        frames = load_raw_samples()
    pc = frames["PC"]
    quartets = pc[pc["rank_dist"] <= 4]
    cg4 = frames["CG4"][["objid", "Group", "rank_M"]].rename(
        columns={"Group": "cg4_group", "rank_M": "cg4_rank_M"}
    )
    merged = quartets.merge(cg4, on="objid")
    classes = cg4_class_map(frames)
    merged["cg4_class"] = merged["cg4_group"].map(classes)
    table = merged[
        ["cg4_group", "Group", "objid", "cg4_rank_M", "rank_M", "rank_dist", "cg4_class"]
    ].rename(columns={"Group": "lim_group", "rank_M": "pc_rank_M", "rank_dist": "pc_rank_dist"})
    return table.sort_values(["cg4_group", "cg4_rank_M"]).reset_index(drop=True)


def write_audit_products() -> None:
    """Write the identity catalogue and derived audit tables to audit/."""

    os.makedirs(AUDIT_DIR, exist_ok=True)
    frames = load_raw_samples()
    catalog = build_identity_catalog(frames)
    catalog.to_csv(os.path.join(AUDIT_DIR, "identity_catalog.csv"), index=False)
    overlap_matrix(frames).to_csv(
        os.path.join(AUDIT_DIR, "overlap_matrix.csv"), index=False
    )
    diagnostic = cg4_in_control4c_table(frames)
    diagnostic.to_csv(
        os.path.join(AUDIT_DIR, "cg4_in_control4c_diagnostic.csv"), index=False
    )
    quartet_table = cg4_in_pc_quartets_table(frames)
    quartet_table.to_csv(
        os.path.join(AUDIT_DIR, "cg4_in_pc_quartets.csv"), index=False
    )
    print(
        f"identity catalog: {len(catalog)} unique objids; "
        f"CG4 galaxies in current Control4C: {len(diagnostic)}; "
        f"CG4 galaxies in PC quartets: {len(quartet_table)} "
        f"({quartet_table['lim_group'].nunique()} Lim groups)"
    )


if __name__ == "__main__":
    write_audit_products()
