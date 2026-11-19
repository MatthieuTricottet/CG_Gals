"""Regenerate the referee package without mutating catalogue or paper files."""

from referee_common import DATA, FIGURES, HERE, ROOT, TABLES, load, np, pd, save

import hashlib
import json
import platform
from importlib.metadata import version

import catalogues
import compactness
import mass
import within_host
from astropy.cosmology import Planck15
from scipy.stats import spearmanr


def provenance(frame, sample):
    expected = {
        "CG4": (248, 62, 60),
        "Control4C": (2812, 703, 703),
        "Control4B": (2792, 698, 698),
        "RG4": (224, 56, 56),
    }
    rows, overlaps, distances = [], [], []
    for name, (n, q, h) in expected.items():
        p = frame[frame["sample"].eq(name)]
        assert (len(p), p.group_uid.nunique(), p.physical_group.nunique()) == (n, q, h)
        assert p.objid.is_unique
        assert p.groupby("group_uid").physical_group.nunique().eq(1).all()
        rows.append(
            dict(
                sample=name,
                n=n,
                quartets=q,
                hosts=h,
                E_S=p.elliptical.count(),
                uncertain=p.morphology.eq("Uncertain").sum(),
                missing_votes=(~p[["p_E", "p_S"]].notna().all(axis=1)).sum(),
                Q_SF=p.quenched.count(),
                colour_retained=p.colour_analysis_retained.sum(),
                derived_colours=p[["u_obs", "g_obs", "i_obs", "r_obs"]]
                .notna()
                .all(axis=1)
                .sum(),
            )
        )
        for other in expected:
            if other == name:
                continue
            r = frame[frame["sample"].eq(other)]
            galaxy_overlap = set(r.objid) & set(p.objid)
            if name == "CG4":
                assert not galaxy_overlap
            q2 = {frozenset(g.objid) for _, g in r.groupby("group_uid")}
            q1 = {frozenset(g.objid) for _, g in p.groupby("group_uid")}
            host_overlap = set(r.physical_group) & set(p.physical_group)
            overlaps.append(
                dict(
                    sample=name,
                    other=other,
                    shared_objids=len(galaxy_overlap),
                    identical_quartets=len(q1 & q2),
                    shared_hosts=len(host_overlap),
                    shared_host_ids=";".join(sorted(host_overlap)),
                )
            )
        raw = pd.read_csv(DATA / f"{name}_Groups.csv")
        raw = raw[raw.Group.isin(p.Group)].sort_values("Group")
        current = sample[f"{name}_Groups"].sort_values("Group")
        assert raw.Group.tolist() == current.Group.tolist()
        theta = np.deg2rad(raw.Radius_Bary_arcmin.to_numpy() / 60.0)
        da = theta * Planck15.angular_diameter_distance(
            raw.z_group.to_numpy()
        ).to_value("kpc")
        dl = theta * Planck15.luminosity_distance(raw.z_group.to_numpy()).to_value(
            "kpc"
        )
        np.testing.assert_allclose(da, current.size_Group_Bary_kpc, rtol=1e-10)
        decrease = (1 - da / dl) * 100
        delta_raw = 100 * (1 - da / raw.size_Group_Bary_kpc)
        distances.append(
            dict(
                sample=name,
                n=len(raw),
                raw_distance_residual_DA=np.median(abs(raw.size_Group_Bary_kpc - da)),
                raw_distance_residual_DL=np.median(abs(raw.size_Group_Bary_kpc - dl)),
                true_percent_decrease_median=np.median(decrease),
                true_percent_decrease_min=decrease.min(),
                true_percent_decrease_max=decrease.max(),
                actually_changed_in_raw_median=np.median(delta_raw),
                rho_tc_Mvir_raw=spearmanr(raw.t_cr, raw.M_virial).statistic,
                rho_tc_Mvir_DA=spearmanr(current.t_cr, current.M_virial).statistic,
            )
        )
    save("sample_counts", rows)
    save("sample_overlap", overlaps)
    save("paper1_distance_audit", distances)
    sources = sorted(HERE.joinpath("inputs").glob("*")) + [
        DATA / name
        for name in (
            "processed_sample.pkl",
            "PC_Gals.csv",
            "PC_Groups.csv",
            "CG4_Gals.csv",
            "CG4_Groups.csv",
            "Control4C_Gals.csv",
            "Control4C_Groups.csv",
            "Control4B_Groups.csv",
            "RG4_Groups.csv",
            "galspecindx_dr12.csv",
            "galspecline_dr16.csv",
            "sdss_spectral_provenance_dr12.csv",
            "sdss_size_columns.csv",
            "simard2011_subset.csv",
        )
    ]
    sources.extend(
        ROOT / "output" / name
        for name in (
            "paper/paper.tex",
            "paper/paper.pdf",
            "paper/online_supplement.tex",
            "paper/online_supplement.pdf",
            "ssfr_spectral_provenance.csv",
            "results.json",
            "results_build.json",
        )
    )
    manifest = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }
    (HERE / "provenance.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "versions": {
                    p: version(p)
                    for p in [
                        "numpy",
                        "pandas",
                        "scipy",
                        "statsmodels",
                        "astropy",
                        "matplotlib",
                        "patsy",
                    ]
                },
                "sha256": manifest,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    for d in [TABLES, FIGURES]:
        d.mkdir(exist_ok=True)
    frame, sample = load()
    provenance(frame, sample)
    print("Compactness", flush=True)
    q, nn, core, excluded = compactness.run(frame, sample)
    print("Within-host geometry", flush=True)
    within_host.run(frame, sample, core)
    print("Mass interactions", flush=True)
    mass.run(frame)
    print("Catalogue audits", flush=True)
    catalogues.run(frame, q, nn, core, excluded)
    from summaries import write_summary

    write_summary()
    print(f"Completed outputs in {HERE}", flush=True)


if __name__ == "__main__":
    main()
