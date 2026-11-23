"""Submission checks on the frozen compactness comparison (Paper II, A&A pass).

Called from ``run.py`` after the freeze tables are written. Writes
``median_separation/tables/07_submission_checks.json`` with

* ``paired``: host-bootstrap contrasts that refit every model on the same
  resampled physical Lim hosts and evaluate it on one fixed reference
  population of CG4 satellites (the satellites locally comparable at the
  evaluation separation), with paired differences between designs;
* ``below_window``: CG4 and Control4C support below the common range and a
  direct contrast there;
* ``host_environment``: common-range contrasts adding Lim host richness or
  host luminosity;
* ``halpha``: full-sample versus common-range stored-spectrum H-alpha
  contrasts for satellites;
* ``magnitude_span``: primary models without the three compact groups whose
  members span slightly more than 3 mag in catalogue absolute r;
* ``compact_match``: calipers and support of the compactness matches.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrices
from scipy.special import expit

from compactness import CALIPERS, candidates
from freeze import BASE, CENTRE, local_target, model_rows
from run import OUTPUT, ROOT

SEED = 20261117
DRAWS = 4999
MAGNITUDE_SPAN_GROUPS = (173, 229, 354)


def _p_cross(draws: np.ndarray) -> float:
    k_le, k_ge = (draws <= 0).sum(), (draws >= 0).sum()
    n = len(draws)
    return float(min(1.0, 2 * min((k_le + 1) / (n + 1), (k_ge + 1) / (n + 1))))


def _summary(point: float, draws: np.ndarray) -> dict:
    draws = draws[np.isfinite(draws)]
    low, high = np.quantile(draws, [0.025, 0.975])
    return dict(
        estimate=float(point), low=float(low), high=float(high),
        p_cross=_p_cross(draws), draws=int(len(draws)),
    )


class PairedBootstrap:
    """Refit logistic models on shared host resamples and evaluate them on fixed references."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)
        hosts = self.frame["physical_group"].to_numpy()
        self.unique_hosts, inverse = np.unique(hosts, return_inverse=True)
        self.members = [np.flatnonzero(inverse == k) for k in range(len(self.unique_hosts))]
        self.specs = {}

    def add(self, name, mask, outcome, formula, references):
        sub = self.frame[mask]
        y, X = dmatrices(f"{outcome} ~ {formula}", sub, return_type="dataframe")
        fit = sm.Logit(y.iloc[:, 0], X).fit(disp=0, method="newton", maxiter=100)
        position = np.full(len(self.frame), -1)
        position[y.index.to_numpy()] = np.arange(len(y))
        design = X.design_info
        refs = {}
        for ref_name, ref in references.items():
            target = ref.assign(xc=0.0)
            refs[ref_name] = tuple(
                np.asarray(build_design_matrices([design], target.assign(cg=arm))[0])
                for arm in (1.0, 0.0)
            )
        self.specs[name] = dict(
            y=y.iloc[:, 0].to_numpy(), X=X.to_numpy(), position=position,
            start=fit.params.to_numpy(), refs=refs,
            # The redshift coefficient is large in natural units; exclude it
            # from the separation guard.
            guard=np.array([c != "z_numeric" for c in design.column_names]),
            n=len(y), hosts=int(sub.loc[y.index, "physical_group"].nunique()),
        )

    @staticmethod
    def _contrast(beta, refs):
        treated, control = refs
        return 100 * float(np.mean(expit(treated @ beta) - expit(control @ beta)))

    def run(self, draws=DRAWS, seed=SEED):
        self.points = {
            f"{name}|{ref}": self._contrast(spec["start"], refs)
            for name, spec in self.specs.items() for ref, refs in spec["refs"].items()
        }
        self.draws = {key: np.full(draws, np.nan) for key in self.points}
        self.failures = {name: 0 for name in self.specs}
        rng = np.random.default_rng(seed)
        for b in range(draws):
            pick = rng.integers(len(self.unique_hosts), size=len(self.unique_hosts))
            index = np.concatenate([self.members[k] for k in pick])
            for name, spec in self.specs.items():
                rows = spec["position"][index]
                rows = rows[rows >= 0]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = sm.Logit(spec["y"][rows], spec["X"][rows]).fit(
                            disp=0, method="newton", maxiter=60, start_params=spec["start"]
                        )
                    beta = result.params
                    if (
                        not result.mle_retvals["converged"]
                        or not np.all(np.isfinite(beta))
                        or np.max(np.abs(beta[spec["guard"]])) > 50
                    ):
                        raise RuntimeError("unstable refit")
                except Exception:
                    self.failures[name] += 1
                    continue
                for ref, refs in spec["refs"].items():
                    self.draws[f"{name}|{ref}"][b] = self._contrast(beta, refs)
        return self

    def estimate(self, key):
        return _summary(self.points[key], self.draws[key])

    def difference(self, a, b):
        both = np.isfinite(self.draws[a]) & np.isfinite(self.draws[b])
        return _summary(self.points[a] - self.points[b], (self.draws[a] - self.draws[b])[both])

    def sizes(self):
        return {name: dict(n=s["n"], hosts=s["hosts"], failures=self.failures[name])
                for name, s in self.specs.items()}


def _host_luminosity(frame):
    pc = pd.read_csv(ROOT / "data/PC_Groups.csv", usecols=["Group", "Lum_group"])
    pc["physical_host_id"] = "Lim:" + pc.Group.astype(str)
    pc = pc.rename(columns={"Lum_group": "host_luminosity"})
    out = frame.merge(pc[["physical_host_id", "host_luminosity"]], on="physical_host_id",
                      how="left", validate="m:1")
    out["host_log_luminosity"] = np.log10(out.host_luminosity)
    return out


def _reference(frame, formula, outcome="elliptical"):
    """The CG4 satellites locally comparable at the centre in the common range."""
    _, _, used = model_rows(frame[frame.window], outcome, formula, {}, (CENTRE,))
    reference, _ = local_target(used, CENTRE)
    return used, reference


def morphology_paired(frame):
    sat = _host_luminosity(frame)
    sat = sat[sat.is_satellite.eq(1) & sat["sample"].isin(["CG4", "Control4C"])].reset_index(drop=True)
    _, _, full_used = model_rows(sat, "elliptical", "cg + " + BASE, {}, (None,))
    _, local = _reference(sat, "cg * xc + " + BASE)
    full_reference = full_used[full_used.cg.eq(1)]
    boot = PairedBootstrap(sat)
    window, cg = sat.window, sat.cg.eq(1)
    everyone = np.ones(len(sat), bool)
    ref = {"local": local}
    boot.add("full", everyone, "elliptical", "cg + " + BASE, {"local": local, "all": full_reference})
    for name, mask, formula in [
        ("window_no_separation", window, "cg + " + BASE),
        ("common_slope", window, "cg + xc + " + BASE),
        ("separate_slopes", window, "cg * xc + " + BASE),
        ("control_restricted", cg | window, "cg + " + BASE),
        ("cg4_restricted", ~cg | window, "cg + " + BASE),
        ("window_no_separation_richness", window, "cg + " + BASE + " + host_richness"),
        ("separate_slopes_richness", window, "cg * xc + " + BASE + " + host_richness"),
        ("window_no_separation_host_luminosity", window, "cg + " + BASE + " + host_log_luminosity"),
        ("separate_slopes_host_luminosity", window, "cg * xc + " + BASE + " + host_log_luminosity"),
    ]:
        boot.add(name, mask, "elliptical", formula, ref)
    boot.run()
    estimates = {key: boot.estimate(key) for key in boot.points}
    differences = {
        f"{a}-{b}": boot.difference(f"{a}|local", f"{b}|local")
        for a, b in [
            ("full", "separate_slopes"), ("full", "common_slope"),
            ("full", "window_no_separation"), ("window_no_separation", "separate_slopes"),
            ("full", "control_restricted"), ("full", "cg4_restricted"),
            ("separate_slopes_richness", "separate_slopes"),
            ("window_no_separation_richness", "window_no_separation"),
            ("separate_slopes_host_luminosity", "separate_slopes"),
            ("window_no_separation_host_luminosity", "window_no_separation"),
        ]
    }
    return dict(
        reference=dict(
            local_cg_satellites=int(len(local)), local_quartets=int(local.group_uid.nunique()),
            local_hosts=int(local.physical_group.nunique()),
            all_cg_satellites=int(len(full_reference)),
        ),
        hosts_resampled=int(len(boot.unique_hosts)), draws=DRAWS, seed=SEED,
        sizes=boot.sizes(), estimates=estimates, differences=differences,
        host_terms=_host_terms(sat),
    )


def _host_terms(sat):
    """Delta-method contrasts and per-dex host-term odds ratios in the common range."""
    out = {}
    window = sat[sat.window]
    for label, term in [("richness", "host_richness"), ("host_luminosity", "host_log_luminosity")]:
        for design, formula in [("no_separation", "cg + " + BASE), ("separate_slopes", "cg * xc + " + BASE)]:
            rows, result, _ = model_rows(window, "elliptical", f"{formula} + {term}", {}, (CENTRE,))
            contrast = next(r for r in rows if r["row"] == "contrast")
            ci = result.conf_int().loc[term]
            out[f"{label}|{design}"] = dict(
                difference=float(contrast["difference"]), low=float(contrast["low"]),
                high=float(contrast["high"]),
                term_or_per_dex=float(np.exp(result.params[term])),
                term_or_low=float(np.exp(ci[0])), term_or_high=float(np.exp(ci[1])),
                term_p=float(result.pvalues[term]),
            )
    q = window.drop_duplicates("group_uid")
    out["median_N_Lim"] = {
        "CG4": float(q.loc[q.cg.eq(1), "N_Lim"].median()),
        "Control4C": float(q.loc[q.cg.eq(0), "N_Lim"].median()),
    }
    return out


def below_window(frame, quartets, window):
    low = window[0]
    pct = (
        quartets[quartets["sample"].isin(["CG4", "Control4C"])]
        .groupby("sample").med_Rij.quantile([0.05, 0.95]).unstack()
    )
    out = {
        "percentiles_kpc": {s: {"p05": float(r[0.05]), "p95": float(r[0.95])} for s, r in pct.iterrows()},
        "samples": {},
    }
    for sample in ("CG4", "Control4C", "Control4B", "RG4"):
        q = quartets[quartets["sample"].eq(sample) & quartets.med_Rij.lt(low)]
        sat = frame[frame["sample"].eq(sample) & frame.is_satellite.eq(1) & frame.group_uid.isin(q.group_uid)]
        es = sat[sat.elliptical.notna()]
        entry = dict(
            quartets=int(len(q)), hosts=int(q.physical_group.nunique()),
            satellites=int(len(sat)), es_satellites=int(len(es)),
            E=int(es.elliptical.sum()), S=int((es.elliptical == 0).sum()),
            fE=float(es.elliptical.mean()) if len(es) else None,
        )
        if sample == "CG4":
            entry["classes"] = q["class"].value_counts().to_dict()
        out["samples"][sample] = entry
    sat = frame[
        frame.is_satellite.eq(1) & frame["sample"].isin(["CG4", "Control4C"]) & frame.med_Rij.lt(low)
    ].dropna(subset=["elliptical"])
    observed = sat.loc[sat.cg.eq(1), "elliptical"].mean() - sat.loc[sat.cg.eq(0), "elliptical"].mean()
    groups = {h: g for h, g in sat.groupby("physical_group")}
    hosts = np.array(list(groups))
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(DRAWS):
        b = pd.concat([groups[h] for h in rng.choice(hosts, len(hosts))])
        a, c = b[b.cg.eq(1)], b[b.cg.eq(0)]
        if len(a) and len(c):
            draws.append(a.elliptical.mean() - c.elliptical.mean())
    out["raw_difference"] = _summary(100 * observed, 100 * np.array(draws))
    d = sat.dropna(subset=["m", "z_numeric", "log_group_luminosity", "velocity_dispersion"])
    import statsmodels.formula.api as smf

    fit = smf.logit("elliptical ~ cg + " + BASE, d).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d.physical_group}, use_t=True
    )
    ci = fit.conf_int().loc["cg"]
    out["adjusted"] = dict(
        odds_ratio=float(np.exp(fit.params.cg)), low=float(np.exp(ci[0])),
        high=float(np.exp(ci[1])), p=float(fit.pvalues.cg), n=int(len(d)),
        hosts=int(d.physical_group.nunique()),
    )
    return out


def halpha_paired(frame):
    from diagnostics import spectral_frame

    spec = spectral_frame(frame)
    sat = spec[spec.is_satellite.eq(1) & spec["sample"].isin(["CG4", "Control4C"])].reset_index(drop=True)
    out = {}
    for label, extra, need in [
        ("baseline", "", []),
        ("morphology", " + elliptical", ["elliptical"]),
        ("morphology_aperture", " + elliptical + aperture", ["elliptical", "aperture"]),
    ]:
        mask = sat[["strong_exact", *need]].notna().all(axis=1).to_numpy()
        subset = sat[mask]
        _, local = _reference(subset, "cg + xc + " + BASE + extra, outcome="strong_exact")
        boot = PairedBootstrap(sat)
        for name, design_mask, formula in [
            ("full", mask, "cg + " + BASE + extra),
            ("common_slope", mask & sat.window.to_numpy(), "cg + xc + " + BASE + extra),
            ("separate_slopes", mask & sat.window.to_numpy(), "cg * xc + " + BASE + extra),
        ]:
            boot.add(name, design_mask, "strong_exact", formula, {"local": local})
        boot.run()
        out[label] = dict(
            reference_cg_satellites=int(len(local)),
            estimates={k.split("|")[0]: boot.estimate(k) for k in boot.points},
            differences={
                "full-common_slope": boot.difference("full|local", "common_slope|local"),
                "full-separate_slopes": boot.difference("full|local", "separate_slopes|local"),
            },
            sizes=boot.sizes(),
        )
    return out


def magnitude_span(frame):
    reduced = frame[~(frame["sample"].eq("CG4") & frame.Group.isin(MAGNITUDE_SPAN_GROUPS))]
    out = {"groups": list(MAGNITUDE_SPAN_GROUPS), "table4": {}, "compactness": {}}
    for control in ("Control4B", "Control4C", "RG4"):
        panel = reduced[reduced["sample"].isin(["CG4", control])]
        for scope, data, term in [("all", panel, " + is_satellite"),
                                  ("satellites", panel[panel.is_satellite.eq(1)], "")]:
            rows, _, _ = model_rows(data, "elliptical", "cg + " + BASE + term, {})
            row = next(r for r in rows if r["row"] == "coefficient" and r["term"] == "cg")
            out["table4"][f"{control}|{scope}"] = dict(
                odds_ratio=float(row["OR"]), low=float(row["OR_low"]),
                high=float(row["OR_high"]), p=float(row["p"]), n=int(row["n"]),
            )
    sat = reduced[reduced.is_satellite.eq(1) & reduced["sample"].isin(["CG4", "Control4C"])]
    for design, data, radii, formula in [
        ("full", sat, (None,), "cg + " + BASE),
        ("window_no_separation", sat[sat.window], (CENTRE,), "cg + " + BASE),
        ("separate_slopes", sat[sat.window], (CENTRE,), "cg * xc + " + BASE),
    ]:
        rows, _, _ = model_rows(data, "elliptical", formula, {}, radii)
        row = next(r for r in rows if r["row"] == "contrast")
        out["compactness"][design] = {k: float(row[k]) for k in ("difference", "low", "high")}
    return out


def compact_match(quartets, matches):
    inside = quartets[quartets.window]
    out = {"calipers": dict(CALIPERS, velocity_dispersion=120.0), "designs": {}}
    for name, sigma in [("compact_match", False), ("compact_match_sigma", True)]:
        cg, control, valid, _, _ = candidates(inside, include_sigma=sigma)
        counts = valid.sum(axis=1)
        out["designs"][name] = dict(
            cg4_in_window=int(inside["sample"].eq("CG4").sum()),
            cg4_with_three_satellite_masses=int(len(cg)),
            control4c_eligible=int(len(control)),
            cg4_with_candidate=int((counts > 0).sum()),
            median_candidates=float(np.median(counts[counts > 0])),
            pairs=int(len(matches[name])),
        )
    a, b = matches["compact_match"], matches["compact_match_sigma"]
    out["same_cg4_quartets"] = bool(set(a.cg_quartet) == set(b.cg_quartet))
    out["identical_pairs"] = int(len(set(zip(a.cg_quartet, a.control_quartet))
                                     & set(zip(b.cg_quartet, b.control_quartet))))
    out["shared_control_quartets"] = int(len(set(a.control_quartet) & set(b.control_quartet)))
    return out


def run_checks(frame, quartets, window, matches):
    result = {
        "paired": morphology_paired(frame),
        "below_window": below_window(frame, quartets, window),
        "halpha": halpha_paired(frame),
        "magnitude_span": magnitude_span(frame),
        "compact_match": compact_match(quartets, matches),
    }
    path = OUTPUT / "07_submission_checks.json"
    path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result
