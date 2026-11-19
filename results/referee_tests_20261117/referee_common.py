"""Read-only inputs and shared host-aware inference."""

import os
import sys
import warnings
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cg-referee-mpl")
)
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data"
TABLES = HERE / "tables"
FIGURES = HERE / "figures"
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "referee"),
]

import importlib.util
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy.special import expit
from scipy.stats import t

SAMPLES = ["CG4", "RG4", "Control4B", "Control4C"]
COLORS = dict(zip(SAMPLES, ["#92234e", "#00796b", "#4868a2", "#b55a00"]))
SEED = 20261117
DRAWS = 4999
BASE = "m + z_numeric + log_group_luminosity + velocity_dispersion"
MASS_BINS = [-np.inf, 10, 10.5, 11, np.inf]
MASS_LABELS = ["<10", "10–10.5", "10.5–11", "≥11"]
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 280,
    }
)


@lru_cache
def load():
    # Reuse the November revision's cache-only reconstruction. Scope the socket
    # guard so local bootstrap and plotting code need no global patch.
    spec = importlib.util.spec_from_file_location(
        "revision_inputs", ROOT / "analysis/bgg_split/common.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import spectral_indices

    indices = pd.read_csv(DATA / "galspecindx_dr12.csv")
    with patch("socket.socket.connect"), patch(
        "socket.create_connection"
    ), patch.object(spectral_indices, "fetch_spectral_indices", return_value=indices):
        module.block_network()
        frame, sample = module.load_galaxy_frame()
    frame["m"] = frame.logMstar - 10.25
    frame["mass_bin"] = pd.cut(
        frame.logMstar, MASS_BINS, labels=MASS_LABELS, right=False
    )
    frame["cg"] = frame.is_CG4.astype(float)
    return frame, sample


def save(name, rows):
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / f"{name}.csv", index=False, float_format="%.10g")
    return table


def figure(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def sizes(frame):
    return dict(
        n=len(frame),
        quartets=frame.group_uid.nunique(),
        hosts=frame.physical_group.nunique(),
    )


def counts(frame):
    return {
        f"{arm}_{key}": value
        for arm, part in [
            ("cg", frame[frame.cg.eq(1)]),
            ("control", frame[frame.cg.eq(0)]),
        ]
        for key, value in sizes(part).items()
    }


def bootstrap(frame, value, statistic="mean"):
    v = frame.dropna(subset=[value])
    if v.empty:
        return np.nan, np.nan, np.nan
    blocks = [p[value].to_numpy() for _, p in v.groupby("physical_group")]
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(blocks), size=(DRAWS, len(blocks)))
    if statistic == "mean":
        numerator = np.array([b.sum() for b in blocks])
        denominator = np.array([len(b) for b in blocks])
        estimates = numerator[draws].sum(1) / denominator[draws].sum(1)
        point = v[value].mean()
    else:
        estimates = [
            np.median(np.concatenate([blocks[j] for j in row])) for row in draws
        ]
        point = v[value].median()
    return point, *np.quantile(estimates, [0.025, 0.975])


def paired_bootstrap(delta, components):
    table = (
        pd.DataFrame({"delta": delta, "component": components})
        .groupby("component")
        .delta.agg(["sum", "size"])
    )
    if table.empty:
        return dict(
            estimate=np.nan, low=np.nan, high=np.nan, blocks=0, effective_blocks=0
        )
    rng = np.random.default_rng(SEED)
    idx = rng.integers(len(table), size=(DRAWS, len(table)))
    bs = table["sum"].to_numpy()[idx].sum(1) / table["size"].to_numpy()[idx].sum(1)
    return dict(
        estimate=np.mean(delta),
        low=np.quantile(bs, 0.025),
        high=np.quantile(bs, 0.975),
        blocks=len(table),
        effective_blocks=table["size"].sum() ** 2 / (table["size"] ** 2).sum(),
    )


def fit(frame, formula, family="logit"):
    model = (smf.logit if family == "logit" else smf.ols)(
        formula, frame, missing="drop"
    )
    used = frame.loc[model.data.row_labels]
    if not used.index.is_unique or used.physical_group.nunique() <= 2:
        raise ValueError(f"Invalid estimation frame for {formula}")
    if np.linalg.matrix_rank(model.exog) < model.exog.shape[1]:
        raise ValueError(f"Rank-deficient model {formula}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        options = dict(
            cov_type="cluster", cov_kwds={"groups": used.physical_group}, use_t=True
        )
        result = (
            model.fit(disp=False, maxiter=200, **options)
            if family == "logit"
            else model.fit(**options)
        )
    if family == "logit" and not result.mle_retvals["converged"]:
        raise RuntimeError(f"Failed to converge for {formula}")
    if not np.isfinite(result.bse).all():
        raise RuntimeError(f"Non-finite uncertainty for {formula}")
    return result, used, "; ".join(str(w.message) for w in caught)


def coefficients(result, used, warning):
    ci = result.conf_int()
    return [
        dict(
            term=term,
            beta=beta,
            se=result.bse[term],
            low=ci.loc[term, 0],
            high=ci.loc[term, 1],
            p=result.pvalues[term],
            formula=result.model.formula,
            warning=warning,
            **sizes(used),
            **counts(used),
        )
        for term, beta in result.params.items()
    ]


def predicted(result, reference, probability=True):
    matrix = np.asarray(
        build_design_matrices([result.model.data.design_info], reference)[0]
    )
    eta = matrix @ result.params
    values = expit(eta) if probability else eta
    gradient = matrix * (values * (1 - values))[:, None] if probability else matrix
    return values.mean(), gradient.mean(axis=0)


def marginal(result, used, reference, probability=True, scale=100):
    p1, g1 = predicted(result, reference.assign(cg=1), probability)
    p0, g0 = predicted(result, reference.assign(cg=0), probability)
    delta = p1 - p0
    g = g1 - g0
    se = np.sqrt(max(0, g @ result.cov_params() @ g))
    half = t.ppf(0.975, used.physical_group.nunique() - 1) * se
    return dict(
        cg_est=p1 * scale,
        control_est=p0 * scale,
        difference=delta * scale,
        low=(delta - half) * scale,
        high=(delta + half) * scale,
        reference_n=len(reference),
    )


def errorbars(ax, x, table, sample, **kwargs):
    ax.vlines(x, table.low, table.high, color=COLORS[sample], linewidth=1.1)
    ax.scatter(x, table.estimate, color=COLORS[sample], s=17, zorder=4, **kwargs)
