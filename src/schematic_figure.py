"""Schematic of the sample definitions (new Fig. 1, gary-r2 A7).

The figure shows one Embedded CG4 (in the corrected Zheng--Shen sense: the compact
group contributes less than half of its host luminosity) whose Lim--Tempel
host has at least eight members and whose redshift is the closest to the
CG4 sample median.  All host members are plotted in projected physical kpc
relative to the host BGG; the CG4 members, the would-be Control4B quartet
(BGG plus the three brightest members within 3 mag) and the would-be
Control4C quartet (BGG plus the three nearest projected members within
3 mag) are marked.  Such hosts are excluded from the actual control samples
because they contain CG4 galaxies.  Positions only; no image cutouts.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15
from matplotlib.path import Path

try:
    import config as co
    from extended_stats import safe_json
    from utils import labels_utils as lu
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from .extended_stats import safe_json
    from .utils import labels_utils as lu

MIN_HOST_MEMBERS = 8
MAG_WINDOW = 3.0
CENTRED_TRIANGLE = Path(
    np.array([[0.0, 2.0 / 3.0], [-1.0 / np.sqrt(3), -1.0 / 3.0],
              [1.0 / np.sqrt(3), -1.0 / 3.0], [0.0, 2.0 / 3.0]])
)


def _projected_kpc(ra, dec, ra0, dec0, z):
    """Small-angle projected physical offsets (kpc) relative to (ra0, dec0)."""

    kpc_per_arcmin = Planck15.kpc_proper_per_arcmin(float(z)).value
    dx = (np.asarray(ra, float) - ra0) * np.cos(np.deg2rad(dec0)) * 60.0 * kpc_per_arcmin
    dy = (np.asarray(dec, float) - dec0) * 60.0 * kpc_per_arcmin
    return dx, dy


def _would_be_quartets(host: pd.DataFrame):
    """Would-be Control4B and Control4C quartets of a Lim host (objid lists)."""

    host = host.sort_values("M_r").reset_index(drop=True)
    bgg = host.iloc[0]
    eligible = host.loc[host["M_r"] <= bgg["M_r"] + MAG_WINDOW].copy()
    c4b = eligible.sort_values("M_r").head(4)["objid"].tolist()
    dx, dy = _projected_kpc(eligible["RA"], eligible["Dec"], bgg["RA"], bgg["Dec"], host["z"].median())
    eligible["r_kpc"] = np.hypot(dx, dy)
    c4c = eligible.sort_values("r_kpc").head(4)["objid"].tolist()
    return bgg, c4b, c4c


def select_groups(sample: dict, pc_gals: pd.DataFrame):
    cg_groups = sample["CG4" + co.GRSUFF]
    cg_gals = sample["CG4" + co.GASUFF]
    z_median = float(cg_groups["z_group"].median())
    host_of = (
        cg_gals[["objid", "Group"]]
        .merge(pc_gals[["objid", "Group"]].rename(columns={"Group": "lim"}), on="objid", how="left")
        .groupby("Group")["lim"]
        .agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else np.nan)
    )
    n_host = pc_gals.groupby("Group").size()
    candidates = cg_groups.loc[cg_groups["Class"] == "Embedded", ["Group", "z_group"]].copy()
    candidates["lim"] = candidates["Group"].map(host_of)
    candidates["n_host"] = candidates["lim"].map(n_host)
    candidates = candidates.loc[candidates["n_host"] >= MIN_HOST_MEMBERS]
    candidates["dz"] = (candidates["z_group"] - z_median).abs()
    chosen = candidates.sort_values(["dz", "Group"]).iloc[0]
    return {
        "cg4_group": int(chosen["Group"]),
        "lim_host": int(chosen["lim"]),
        "n_host_members": int(chosen["n_host"]),
        "cg4_z_group": float(chosen["z_group"]),
        "cg4_sample_median_z": z_median,
        "n_embedded_candidates": int(len(candidates)),
        "selection_rule": (
            f"Embedded CG4 (corrected Zheng--Shen label) with host richness >= {MIN_HOST_MEMBERS}, "
            "minimising |z_group - median z_group(CG4)|"
        ),
    }


def _draw_group(ax, members, origin, z, marks, title, style_note=None, show_labels=True):
    dx, dy = _projected_kpc(members["RA"], members["Dec"], origin["RA"], origin["Dec"], z)
    lum_scale = 10 ** (-0.4 * (members["M_r"].to_numpy(float) - members["M_r"].min()))
    size = 18 + 60 * lum_scale
    highlighted = set()
    for objids, *_rest in marks.values():
        highlighted.update(int(value) for value in objids)
    other = ~members["objid"].astype("int64").isin(highlighted).to_numpy()
    if other.any():
        ax.scatter(dx[other], dy[other], s=size[other], facecolor="0.75",
                   edgecolor="0.45", linewidth=0.7, zorder=2,
                   label="other host members" if show_labels else None)
    for key, (objids, colour, marker, label, ms) in marks.items():
        sel = members["objid"].isin(objids).to_numpy()
        marker_path = CENTRED_TRIANGLE if marker == "^" else marker
        ax.scatter(dx[sel], dy[sel], s=size[sel] + ms, facecolor="none",
                   edgecolor=colour, marker=marker_path, linewidth=1.4,
                   zorder=3 + (key == "CG4"), label=label if show_labels else None)
    ax.axhline(0, color="0.85", lw=0.6, zorder=1)
    ax.axvline(0, color="0.85", lw=0.6, zorder=1)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel(r"$\Delta x$ (kpc)", fontsize=10)
    ax.tick_params(labelsize=9, direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.set_aspect("equal")
    ax.invert_xaxis()  # east to the left
    if style_note:
        ax.text(0.02, 0.02, style_note, transform=ax.transAxes, fontsize=7, va="bottom")
    return dx, dy, size


def run_schematic_figure(sample: dict, output_dir: str | None = None) -> dict:
    pc_gals = pd.read_csv(os.path.join(co.DATA_PATH, "PC_Gals.csv"))
    pc_gals["objid"] = pc_gals["objid"].astype("int64")
    choice = select_groups(sample, pc_gals)

    host = pc_gals.loc[pc_gals["Group"] == choice["lim_host"]].copy()
    cg_members = sample["CG4" + co.GASUFF]
    cg_objids = cg_members.loc[cg_members["Group"] == choice["cg4_group"], "objid"].astype("int64").tolist()
    bgg, c4b, c4c = _would_be_quartets(host)
    choice.update({
        "host_bgg_objid": int(bgg["objid"]),
        "cg4_member_objids": [int(v) for v in cg_objids],
        "would_be_control4b_objids": [int(v) for v in c4b],
        "would_be_control4c_objids": [int(v) for v in c4c],
        "cg4_bgg_is_host_bgg": bool(int(bgg["objid"]) in cg_objids and
                                   int(cg_members.loc[(cg_members["Group"] == choice["cg4_group"]) & (cg_members["rank_M"] == 1), "objid"].iloc[0]) == int(bgg["objid"])),
        "n_cg4_in_would_be_control4b": int(len(set(cg_objids) & set(c4b))),
        "n_cg4_in_would_be_control4c": int(len(set(cg_objids) & set(c4c))),
    })

    result = {"status": "ok", **choice}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        style = plt.style.context("default")  # earlier modules may set a seaborn style
        style.__enter__()
        fig, ax = plt.subplots(figsize=(6.6, 5.7))
        z_cg = host["z"].median()
        marks = {
            "C4B": (c4b, "#0072B2", "s", r"would-be Control$_{4B}$ quartet", 210),
            "C4C": (c4c, "#D55E00", "^", r"would-be Control$_{4C}$ quartet", 330),
            "CG4": (cg_objids, "black", "o", r"CG$_4$ members", 110),
        }
        _draw_group(
            ax, host, bgg, z_cg, marks,
            f"Embedded CG$_4$ {choice['cg4_group']} in Lim host {choice['lim_host']}\n"
            f"($N_{{\\rm host}}={choice['n_host_members']}$, $z={choice['cg4_z_group']:.3f}$)",
        )
        # Symmetric limits covering the full host.
        dx, dy = _projected_kpc(host["RA"], host["Dec"], bgg["RA"], bgg["Dec"], z_cg)
        half = 1.08 * max(np.abs(dx).max(), np.abs(dy).max())
        ax.set_xlim(half, -half)
        ax.set_ylim(-half, half)
        ax.set_ylabel(r"$\Delta y$ (kpc)", fontsize=11)
        ax.set_xlabel(r"$\Delta x$ (kpc)", fontsize=11)
        ax.set_title(ax.get_title(), fontsize=10.5, pad=8)
        ax.tick_params(labelsize=10)
        inset = ax.inset_axes([0.055, 0.545, 0.405, 0.405])
        _draw_group(inset, host, bgg, z_cg, marks, "", show_labels=False)
        inset.set_xlim(120, -120)
        inset.set_ylim(-120, 120)
        inset.set_xlabel("")
        inset.set_ylabel("")
        inset.set_title(r"central $\pm120$ kpc", fontsize=7, pad=1)
        inset.tick_params(labelsize=6, length=2, pad=1, top=True, right=True)
        ax.indicate_inset_zoom(inset, edgecolor="0.35", linewidth=0.8)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, 0.005), handletextpad=0.4, columnspacing=1.25)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        path = os.path.join(output_dir, "fig_sample_schematic.pdf")
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        style.__exit__(None, None, None)
        result["figure"] = os.path.basename(path)
        result["half_width_kpc"] = float(half)
    return safe_json(result)
