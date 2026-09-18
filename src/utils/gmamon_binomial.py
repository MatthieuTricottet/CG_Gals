"""Binomial plotting helper by Gary A. Mamon.

This is the small, self-contained subset of ``graphutils.py`` and
``mathutils.py`` used by Fig. 2.  It is reproduced from commit
88fea77dbd6430a67c262ffa9ac5ab369ab351f3 of
https://gitlab.com/gmamon/python-codes so the published figure is generated
by the requested routine without depending on a user-specific checkout.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def BinomialError(N, n, Wilson_nsigma=1.65, Wilson_center_to_zero=False):
    """Return Gary Mamon's binomial point estimates and uncertainties."""

    if not isinstance(N, np.ndarray) or not isinstance(n, np.ndarray):
        raise ValueError("N and n must be numpy arrays")
    if len(n) != len(N):
        raise ValueError("N and n must have same length")
    N_tmp = np.where(N == 0, 1, N)
    p_orig = np.where(N == 0, -1, n / N_tmp)
    error_p_orig = np.where(N == 0, -1, np.sqrt(p_orig * (1 - p_orig) / N_tmp))
    z2 = Wilson_nsigma * Wilson_nsigma
    if np.max(Wilson_nsigma) > 0:
        if Wilson_center_to_zero:
            p = np.select([n == 0, n == N], [0, 1], p_orig)
        else:
            p = np.where((n == 0) | (n == N), (n + 0.5 * z2) / (N + z2), p_orig)
        error_p = np.where(
            (n == 0) | (n == N),
            Wilson_nsigma * np.sqrt(n * (1 - n / N_tmp) + 0.25 * z2) / (N + z2),
            error_p_orig,
        )
    else:
        p, error_p = p_orig, error_p_orig
    return p, error_p


def binomialerrorplot(
    x,
    N,
    n,
    color="k",
    marker="o",
    markersize=30,
    mec="k",
    mew=1,
    capsize=0,
    capthick=1,
    ecolor="k",
    label=None,
    scale="linear",
    eyFactor=None,
    line_apex=None,
    line_color="darkorange",
    ax=None,
    zorder=None,
    yoffset=0,
    verbose=0,
):
    """Gary Mamon's binomial-fraction plotting routine (upstream API)."""

    del line_apex, line_color
    condGood = N > 0
    x, N, n = x[condGood], N[condGood], n[condGood]
    p, ep = BinomialError(N, n)
    condUpper = n == 0
    condLower = n == N
    condPoints = np.logical_not(np.logical_or(condUpper, condLower))
    if ax is None:
        ax = plt.gca()
    eb1 = ax.errorbar(
        x[condPoints], p[condPoints] + yoffset, ep[condPoints], marker=marker,
        mec=mec, mfc=color, ecolor=ecolor, ls="none", ms=markersize, mew=mew,
        capsize=capsize, capthick=capthick, label=label, zorder=zorder,
    )
    if scale == "log":
        eyFactor = 0.5 if eyFactor is None else eyFactor
        ep = p * (1 - 10 ** (-eyFactor))
    else:
        eyFactor = 0.075 if eyFactor is None else eyFactor
        ep = np.where(p - eyFactor < 0, p / 2, eyFactor)
    ax.errorbar(
        x[condUpper], p[condUpper] + yoffset, ep[condUpper], marker=",",
        mec=mec, mfc=color, ecolor=ecolor, ls="none", capsize=capsize,
        capthick=capthick, uplims=np.full(len(x[condUpper]), True), zorder=zorder,
    )
    if scale == "log":
        ep = p * 10 ** eyFactor
    else:
        ep = np.where(p + eyFactor > 1, (1 - p) / 2, eyFactor)
    ax.errorbar(
        x[condLower], p[condLower] + yoffset, ep[condLower], marker=",",
        mec=mec, mfc=color, ecolor=ecolor, ls="none", capsize=capsize,
        capthick=capthick, lolims=np.full(len(x[condLower]), True), zorder=zorder,
    )
    if scale != "linear":
        ax.set_yscale(scale)
    if verbose:
        print(np.transpose([x, p, ep, condPoints]))
    return eb1
