"""D4 -- DS18 -> GZ1 mapping, environment-agnostic (gary-r2 Phase 1).

Pooled over the deduplicated sample (3857 unique galaxies, 2436 matched to
DS18).  For DS18 galaxies with TType <= 0 and P_S0 > 0.5, and separately
P_S0 >= 0.8, report the fractions in GZ1 E / S / Uncertain / NoGZ with
Wilson intervals, and the inclination (DS18 P_edge_on) dependence of the
GZ1 class at both thresholds.  NO CG4-vs-control split (reserved for the
blind Paper III analysis).
"""

from __future__ import annotations

import gzip
import sys

import numpy as np
import pandas as pd

from common import ROOT, load_sample, store_diagnostic, write_csv

sys.path.insert(0, str(ROOT / "referee"))
import T10_ds18_morphology as t10  # noqa: E402

from paper_additions import wilson  # noqa: E402

CLASSES = ["E", "S", "U", "X"]
EDGE_BINS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0001]
EDGE_LABELS = ["0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-1"]


def main() -> None:
    sample, bridge, ds18_min = t10._load_inputs()
    with gzip.open(t10.DS18_PATH, "rt") as stream:
        ds18 = pd.read_fwf(stream, colspecs=t10.DS18_COLSPECS, names=t10.DS18_COLUMNS,
                           header=None, usecols=["objID", "TType", "P_S0", "P_edge_on"],
                           dtype={"objID": str})
    ds18["objID"] = ds18["objID"].str.strip().astype("int64")
    for c in ("TType", "P_S0", "P_edge_on"):
        ds18[c] = pd.to_numeric(ds18[c], errors="coerce")
    ours, audited = t10._unique_audit_frame(sample, bridge, ds18_min)
    audited = audited.join(
        bridge.merge(ds18[["objID", "P_edge_on"]], left_on="dr7objid", right_on="objID")
        .set_index("dr8objid")["P_edge_on"], how="left")
    assert audited["P_edge_on"].notna().all()

    out = {"n_unique_galaxies": int(len(ours)), "n_matched_ds18": int(len(audited)),
           "thresholds": {}}
    rows, inc_rows = [], []
    for name, thr, op in (("P_S0>0.5", 0.5, "gt"), ("P_S0>=0.8", 0.8, "ge")):
        mask = audited["early"] & (audited["P_S0"] > thr if op == "gt" else audited["P_S0"] >= thr)
        sel = audited.loc[mask]
        n = int(len(sel))
        block = {"n": n, "fractions": {}}
        for c in CLASSES:
            k = int((sel["gz1_class"] == c).sum())
            lo, hi = wilson(k, n)
            block["fractions"][c] = {"n": k, "fraction": k / n, "wilson_lo": lo, "wilson_hi": hi}
            rows.append(dict(threshold=name, gz1_class=c, n=k, n_total=n, fraction=k / n,
                             wilson_lo=lo, wilson_hi=hi))
        # inclination dependence: GZ1 class fractions per P_edge_on bin
        bins = pd.cut(sel["P_edge_on"], EDGE_BINS, labels=EDGE_LABELS, right=False)
        inc = {}
        for b, part in sel.groupby(bins, observed=False):
            nb = int(len(part))
            entry = {"n": nb}
            for c in CLASSES:
                k = int((part["gz1_class"] == c).sum())
                lo, hi = wilson(k, nb) if nb else (np.nan, np.nan)
                entry[c] = {"n": k, "fraction": k / nb if nb else None, "wilson_lo": lo, "wilson_hi": hi}
                inc_rows.append(dict(threshold=name, P_edge_on_bin=str(b), n_bin=nb, gz1_class=c,
                                     n=k, fraction=k / nb if nb else np.nan, wilson_lo=lo, wilson_hi=hi))
            inc[str(b)] = entry
        block["inclination"] = inc
        # medians of P_edge_on per GZ1 class + rank-sum E vs S
        from scipy.stats import mannwhitneyu
        e = sel.loc[sel["gz1_class"] == "E", "P_edge_on"]
        s = sel.loc[sel["gz1_class"] == "S", "P_edge_on"]
        block["P_edge_on_median_by_class"] = {
            c: float(sel.loc[sel["gz1_class"] == c, "P_edge_on"].median()) for c in CLASSES
            if (sel["gz1_class"] == c).any()}
        block["P_edge_on_E_vs_S_mannwhitney_p"] = float(mannwhitneyu(e, s).pvalue) if len(e) and len(s) else None
        # fraction E among face-on (P_edge_on<0.1) vs inclined (>=0.5)
        face = sel.loc[sel["P_edge_on"] < 0.1]
        incl = sel.loc[sel["P_edge_on"] >= 0.5]
        block["fE_face_on_lt0p1"] = {"n": int(len(face)), "fE": float((face["gz1_class"] == "E").mean())}
        block["fE_inclined_ge0p5"] = {"n": int(len(incl)), "fE": float((incl["gz1_class"] == "E").mean())}
        out["thresholds"][name] = block
        print(name, n, {c: round(block["fractions"][c]["fraction"], 3) for c in CLASSES},
              "face-on fE", block["fE_face_on_lt0p1"], "inclined fE", block["fE_inclined_ge0p5"],
              "p(E vs S edge-on)", block["P_edge_on_E_vs_S_mannwhitney_p"])
    # reference bands for orientation: pure E (P_S0<0.2) and late types
    pure = audited.loc[audited["early"] & (audited["P_S0"] < 0.2)]
    late = audited.loc[~audited["early"]]
    out["reference"] = {
        "early_P_S0<0.2": {"n": int(len(pure)), **{c: float((pure["gz1_class"] == c).mean()) for c in CLASSES}},
        "late_TType>0": {"n": int(len(late)), **{c: float((late["gz1_class"] == c).mean()) for c in CLASSES}},
    }
    # what our GZ1 E class contains (P(DS18 | E)), pooled
    e_all = audited.loc[audited["gz1_class"] == "E"]
    out["gz1_E_content_pooled"] = {
        "n_E_matched": int(len(e_all)),
        "fraction_TType_le0": float(e_all["early"].mean()),
        "fraction_TType_gt0": float((~e_all["early"]).mean()),
        "fraction_early_and_P_S0_gt0.5": float((e_all["early"] & (e_all["P_S0"] > 0.5)).mean()),
        "fraction_early_and_P_S0_ge0.8": float((e_all["early"] & (e_all["P_S0"] >= 0.8)).mean()),
        "fraction_early_and_P_S0_lt0.2": float((e_all["early"] & (e_all["P_S0"] < 0.2)).mean()),
        "median_TType": float(e_all["TType"].median()),
    }
    print(out["gz1_E_content_pooled"])
    write_csv(pd.DataFrame(rows), "d4_ds18_s0_to_gz1_fractions.csv")
    write_csv(pd.DataFrame(inc_rows), "d4_ds18_s0_to_gz1_inclination.csv")
    out["environment_split"] = "NOT computed (reserved for blind Paper III analysis)"
    store_diagnostic("d4_ds18_gz1_mapping", out)


if __name__ == "__main__":
    main()
