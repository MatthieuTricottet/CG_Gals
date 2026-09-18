"""Schematic of the sample definitions (Fig. 1, gary-r2 A7).

Two Lim--Tempel hosts of embedded CG4 systems (corrected Zheng--Shen label:
the compact group contributes less than half of its host luminosity), drawn
on a common projected physical scale relative to the host BGG.  Every host
member is a grey disc whose area is proportional to its r-band luminosity;
the would-be Control4B quartet (four brightest members within 3 mag of the
BGG) and Control4C quartet (BGG plus its three nearest projected companions
within 3 mag) are overlaid on the members they select, as are the CG4
members.  A third, narrower column holds one zoom panel per host that
enlarges the boxed window around its compact group.

Panel (a) -- the common configuration: the compact group *is* the
BGG-centred core of its host, so both quartets contain CG4 galaxies and are
excluded from the controls.  Chosen among embedded CG4s with host richness
>= MIN_HOST_MEMBERS as the one closest to the CG4 median redshift.

Panel (b) -- the other configuration: the compact group is an off-centre
substructure and the host's BGG-centred core is made of other members, so
the Control4C quartet contains no CG4 galaxy and is retained in the final
control sample, whereas the Control4B quartet contains the compact group's
brightest member and is excluded.  Chosen among embedded CG4s whose host
Control4C quartet is in the final sample (host richness >= MIN_HOST_MEMBERS)
as the one with host richness closest to panel (a), so that the two panels
differ mainly in where the compact group sits.

Exclusion status is read from the final samples, never recomputed here.
Positions only; no image cutouts.

Run ``python src/schematic_figure.py`` to regenerate the figure from the
cached processed sample and refresh ``results.json['schematic_figure']``
(the full pipeline does the same through ``main.py``).
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from matplotlib.ticker import MaxNLocator

try:
    import config as co
    import sample_construction as sc
    from extended_stats import safe_json
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from . import sample_construction as sc
    from .extended_stats import safe_json

MIN_HOST_MEMBERS = 8
MAG_WINDOW = sc.DMAG_MAX
INSET_MIN_HALF_KPC = 120.0  # inset window never smaller than this
INSET_PAD_KPC = 60.0  # margin around the outermost CG4 member inside the zoom window
SCALE_PADDING = 1.15  # common half-width = SCALE_PADDING x largest host extent
TEXT_CORNERS = ("top-left", "bottom-left", "bottom-right", "top-right")
# Nominal font sizes (pt); the figure prints at ~0.9 of nominal at \textwidth.
FONT = {"title": 9.5, "label": 10.5, "ticks": 9.5, "legend": 9, "status": 8.5, "zoom_ticks": 7.5, "zoom_title": 8.5}
# Marker geometry in points^2 (matplotlib ``s``): luminosity discs span
# DISC_AREA_MIN..DISC_AREA_MIN+DISC_AREA_RANGE; the overlay markers are fixed.
DISC_AREA_MIN, DISC_AREA_RANGE = 14.0, 96.0
OVERLAY_AREA = {"C4B": 230.0, "C4C": 430.0, "CG4": 250.0}
# Equilateral triangle centred on its centroid (matplotlib's '^' is centred
# on its bounding box, which would offset it from the disc it encloses).
CENTRED_TRIANGLE = Path(
    np.array([[0.0, 2.0 / 3.0], [-1.0 / np.sqrt(3), -1.0 / 3.0],
              [1.0 / np.sqrt(3), -1.0 / 3.0], [0.0, 2.0 / 3.0]])
)
COLOURS = {"C4B": "#0072B2", "C4C": "#D55E00", "CG4": "black"}
MARKERS = {"C4B": "s", "C4C": CENTRED_TRIANGLE, "CG4": "o"}
LEGEND_LABELS = {
    "host": r"host member (area $\propto L_r$)",
    "CG4": r"CG$_4$ member",
    "C4B": r"Control$_{4B}$ quartet: 4 brightest",
    "C4C": r"Control$_{4C}$ quartet: BGG + 3 nearest",
}


def _projected_kpc(ra, dec, ra0, dec0, z):
    """Small-angle projected physical offsets (kpc) relative to (ra0, dec0)."""

    kpc_per_arcmin = Planck15.kpc_proper_per_arcmin(float(z)).value
    dx = (np.asarray(ra, float) - ra0) * np.cos(np.deg2rad(dec0)) * 60.0 * kpc_per_arcmin
    dy = (np.asarray(dec, float) - dec0) * 60.0 * kpc_per_arcmin
    return dx, dy


def _would_be_quartets(host: pd.DataFrame):
    """Would-be Control4B and Control4C quartets of one Lim host (objid lists).

    Control4C comes from the pipeline's own selector (Delta_m <= 3 filter,
    then great-circle distance ranking), so the schematic cannot drift from
    the sample construction; Control4B is the four brightest members within
    Delta_m <= 3 of the BGG.
    """

    host = host.sort_values("M_r").reset_index(drop=True)
    bgg = host.iloc[0]
    eligible = host.loc[host["M_r"] <= bgg["M_r"] + MAG_WINDOW]
    c4b = eligible.sort_values("M_r").head(4)["objid"].astype("int64").tolist()
    c4c = sc.select_control4c_quartets(host)["objid"].astype("int64").tolist()
    return bgg, c4b, c4c


def _cg4_hosts(sample: dict, pc_gals: pd.DataFrame) -> pd.DataFrame:
    """One row per CG4 group: class, redshift, host Lim group and its richness."""

    cg_groups = sample["CG4" + co.GRSUFF]
    cg_gals = sample["CG4" + co.GASUFF]
    host_of = (
        cg_gals[["objid", "Group"]]
        .merge(pc_gals[["objid", "Group"]].rename(columns={"Group": "lim"}), on="objid", how="left")
        .groupby("Group")["lim"]
        .agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else np.nan)
    )
    table = cg_groups[["Group", "Class", "z_group"]].copy()
    table["lim"] = table["Group"].map(host_of)
    table["n_host"] = table["lim"].map(pc_gals.groupby("Group").size())
    return table


def select_groups(sample: dict, pc_gals: pd.DataFrame) -> dict:
    """Choose the two hosts; flat keys describe panel (a), ``panel_b`` the other."""

    table = _cg4_hosts(sample, pc_gals)
    z_median = float(sample["CG4" + co.GRSUFF]["z_group"].median())
    embedded = table.loc[(table["Class"] == "Embedded") & (table["n_host"] >= MIN_HOST_MEMBERS)].copy()

    embedded["dz"] = (embedded["z_group"] - z_median).abs()
    chosen_a = embedded.sort_values(["dz", "Group"]).iloc[0]

    retained_c4c = set(sample["Control4C" + co.GRSUFF]["Group"].astype("int64"))
    candidates_b = embedded.loc[embedded["lim"].astype("int64").isin(retained_c4c)].copy()
    candidates_b["dn"] = (candidates_b["n_host"] - chosen_a["n_host"]).abs()
    chosen_b = candidates_b.sort_values(["dn", "dz", "Group"]).iloc[0]

    return {
        "cg4_group": int(chosen_a["Group"]),
        "lim_host": int(chosen_a["lim"]),
        "n_host_members": int(chosen_a["n_host"]),
        "cg4_z_group": float(chosen_a["z_group"]),
        "cg4_sample_median_z": z_median,
        "n_embedded_candidates": int(len(embedded)),
        "selection_rule": (
            f"Embedded CG4 (corrected Zheng--Shen label) with host richness >= {MIN_HOST_MEMBERS}, "
            "minimising |z_group - median z_group(CG4)|"
        ),
        "panel_b": {
            "cg4_group": int(chosen_b["Group"]),
            "lim_host": int(chosen_b["lim"]),
            "n_host_members": int(chosen_b["n_host"]),
            "cg4_z_group": float(chosen_b["z_group"]),
            "n_candidates": int(len(candidates_b)),
            "candidate_cg4_groups": [int(v) for v in candidates_b.sort_values("Group")["Group"]],
            "selection_rule": (
                f"Embedded CG4 with host richness >= {MIN_HOST_MEMBERS} whose host Control4C "
                "quartet is in the final Control4C sample (contains no CG4 galaxy), "
                "minimising |N_host - N_host(panel a)|, then |z_group - median z_group(CG4)|"
            ),
        },
    }


def _panel_data(sample: dict, pc_gals: pd.DataFrame, cg4_group: int, lim_host: int) -> dict:
    """Everything one panel needs: members, offsets, quartets and their status."""

    host = pc_gals.loc[pc_gals["Group"] == lim_host].copy()
    cg_gals = sample["CG4" + co.GASUFF]
    cg_groups = sample["CG4" + co.GRSUFF]
    cg_objids = cg_gals.loc[cg_gals["Group"] == cg4_group, "objid"].astype("int64").tolist()
    cg_bgg = int(cg_gals.loc[(cg_gals["Group"] == cg4_group) & (cg_gals["rank_M"] == 1), "objid"].iloc[0])
    bgg, c4b, c4c = _would_be_quartets(host)
    z = float(host["z"].median())
    dx, dy = _projected_kpc(host["RA"], host["Dec"], bgg["RA"], bgg["Dec"], z)
    host["dx"], host["dy"] = dx, dy
    in_cg = host["objid"].isin(cg_objids).to_numpy()
    retained = {
        key: lim_host in set(sample[name + co.GRSUFF]["Group"].astype("int64"))
        for key, name in (("C4B", "Control4B"), ("C4C", "Control4C"))
    }
    return {
        "host": host,
        "bgg": bgg,
        "z": z,
        "cg_objids": cg_objids,
        "quartets": {"C4B": c4b, "C4C": c4c},
        "retained": retained,
        "meta": {
            "cg4_group": int(cg4_group),
            "lim_host": int(lim_host),
            "n_host_members": int(len(host)),
            "cg4_z_group": float(cg_groups.loc[cg_groups["Group"] == cg4_group, "z_group"].iloc[0]),
            "host_bgg_objid": int(bgg["objid"]),
            "cg4_member_objids": [int(v) for v in cg_objids],
            "would_be_control4b_objids": [int(v) for v in c4b],
            "would_be_control4c_objids": [int(v) for v in c4c],
            "cg4_bgg_is_host_bgg": bool(cg_bgg == int(bgg["objid"])),
            "n_cg4_in_would_be_control4b": int(len(set(cg_objids) & set(c4b))),
            "n_cg4_in_would_be_control4c": int(len(set(cg_objids) & set(c4c))),
            "control4b_retained": bool(retained["C4B"]),
            "control4c_retained": bool(retained["C4C"]),
            "cg4_centre_offset_kpc": float(np.hypot(dx[in_cg].mean(), dy[in_cg].mean())),
            "cg4_max_pair_separation_kpc": float(max(
                np.hypot(dx[in_cg][i] - dx[in_cg][j], dy[in_cg][i] - dy[in_cg][j])
                for i in range(in_cg.sum()) for j in range(i + 1, in_cg.sum())
            )),
            "host_half_extent_kpc": float(max(np.abs(dx).max(), np.abs(dy).max())),
        },
    }


def _status_text(key: str, n_cg4: int, retained: bool) -> str:
    label = r"Control$_{4B}$" if key == "C4B" else r"Control$_{4C}$"
    if retained:
        return f"{label}: retained"
    if n_cg4 == 4:
        return f"{label}: excluded (= CG$_4$)"
    if n_cg4 == 0:
        return f"{label}: excluded"
    plural = "galaxy" if n_cg4 == 1 else "galaxies"
    return f"{label}: excluded ({n_cg4} CG$_4$ {plural})"


def _draw_members(ax, panel: dict, lum_ref: float, overlay_scale: float = 1.0):
    """Luminosity discs for all members, overlay markers for the selected ones."""

    host = panel["host"]
    area = DISC_AREA_MIN + DISC_AREA_RANGE * host["Lum"].to_numpy(float) / lum_ref
    ax.scatter(host["dx"], host["dy"], s=area, facecolor="0.78", edgecolor="0.45",
               linewidth=0.6, zorder=2)
    sets = {"C4B": panel["quartets"]["C4B"], "C4C": panel["quartets"]["C4C"], "CG4": panel["cg_objids"]}
    for key in ("C4B", "C4C", "CG4"):
        sel = host["objid"].isin(sets[key]).to_numpy()
        ax.scatter(host["dx"][sel], host["dy"][sel], s=OVERLAY_AREA[key] * overlay_scale,
                   facecolor="none", edgecolor=COLOURS[key], marker=MARKERS[key],
                   linewidth=1.3, zorder=3 + (key == "CG4"))
    ax.axhline(0, color="0.85", lw=0.6, zorder=1)
    ax.axvline(0, color="0.85", lw=0.6, zorder=1)
    ax.set_aspect("equal")
    ax.tick_params(direction="in", top=True, right=True)


def _inset_window(panel: dict, min_half: float = INSET_MIN_HALF_KPC, pad: float = INSET_PAD_KPC):
    """Centre and half-width (kpc) of the window enclosing all CG4 members."""

    host = panel["host"]
    sel = host["objid"].isin(panel["cg_objids"]).to_numpy()
    cx = 0.5 * (host["dx"][sel].max() + host["dx"][sel].min())
    cy = 0.5 * (host["dy"][sel].max() + host["dy"][sel].min())
    reach = max(np.abs(host["dx"][sel] - cx).max(), np.abs(host["dy"][sel] - cy).max())
    return float(cx), float(cy), float(max(min_half, reach + pad))


def _members_axes_fraction(panel: dict, half: float):
    """Member positions in axes fraction, x axis inverted (east to the left)."""

    fx = (half - panel["host"]["dx"].to_numpy(float)) / (2 * half)
    fy = (panel["host"]["dy"].to_numpy(float) + half) / (2 * half)
    return fx, fy


def _rect_clearance(rect, fx, fy):
    """Smallest distance (axes fraction) from the points to the rectangle."""

    x0, y0, w, h = rect
    ddx = np.maximum(np.maximum(x0 - fx, fx - (x0 + w)), 0)
    ddy = np.maximum(np.maximum(y0 - fy, fy - (y0 + h)), 0)
    return float(np.hypot(ddx, ddy).min())


def _rects_gap(r1, r2):
    """Distance between two axes-fraction rectangles (0 when they overlap)."""

    gx = max(r1[0] - (r2[0] + r2[2]), r2[0] - (r1[0] + r1[2]), 0)
    gy = max(r1[1] - (r2[1] + r2[3]), r2[1] - (r1[1] + r1[3]), 0)
    return float(np.hypot(gx, gy))


def _place_text(fig, ax, text, panel, half, avoid_rect, min_clear_kpc):
    """Draw ``text`` in the free corner farthest from members and ``avoid_rect``.

    Corners are tried in ``TEXT_CORNERS`` order and the first whose rendered
    bounding box overlaps neither the zoom box nor any member (clearance at
    least ``min_clear_kpc``) wins; otherwise the corner with the largest
    clearance.
    """

    fx, fy = _members_axes_fraction(panel, half)
    renderer = fig.canvas.get_renderer()
    pad = 0.03
    best = None
    for corner in TEXT_CORNERS:
        ha = "left" if corner.endswith("left") else "right"
        va = "bottom" if corner.startswith("bottom") else "top"
        x = pad if ha == "left" else 1 - pad
        y = pad if va == "bottom" else 1 - pad
        artist = ax.text(x, y, text, transform=ax.transAxes, fontsize=FONT["status"], ha=ha, va=va, zorder=6,
                         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        bb = artist.get_window_extent(renderer).transformed(ax.transAxes.inverted())
        rect = [bb.x0, bb.y0, bb.width, bb.height]
        clearance = min(_rect_clearance(rect, fx, fy), _rects_gap(rect, avoid_rect)) * 2 * half
        if clearance >= min_clear_kpc:
            return corner, float(clearance)
        if best is None or clearance > best[1]:
            best = (corner, float(clearance), artist)
        artist.remove()
    corner, clearance, artist = best
    ax.add_artist(artist)
    return corner, clearance


def off_centre_embedded_summary(sample: dict, pc_gals: pd.DataFrame) -> dict:
    """Embedded CG4s with no member in their host's would-be Control4C quartet.

    These are the systems Fig. 1b stands for.  For each one the summary records
    whether the host's Control4C / Control4B quartets survive in the final
    samples and whether the would-be Control4B quartet contains the compact
    group's own brightest member (the reason its exclusion never lifts).
    """

    table = _cg4_hosts(sample, pc_gals)
    cg_gals = sample["CG4" + co.GASUFF]
    # The exclusion uses the full compact-group catalogue (split groups
    # included), so "another compact group in the core" is judged on it too.
    full_cg = pd.read_csv(os.path.join(co.DATA_PATH, "CG4_Gals.csv"))
    cg_group_of = dict(zip(full_cg["objid"].astype("int64"), full_cg["Group"].astype(int)))
    retained_c4c = set(sample["Control4C" + co.GRSUFF]["Group"].astype("int64"))
    retained_c4b = set(sample["Control4B" + co.GRSUFF]["Group"].astype("int64"))
    rows = []
    for _, row in table.loc[table["Class"] == "Embedded"].iterrows():
        host = pc_gals.loc[pc_gals["Group"] == int(row["lim"])]
        bgg, c4b, c4c = _would_be_quartets(host)
        members = cg_gals.loc[cg_gals["Group"] == row["Group"]]
        objids = set(members["objid"].astype("int64"))
        if objids & set(c4c):
            continue
        cg_bgg = int(members.loc[members["rank_M"] == 1, "objid"].iloc[0])
        other_cg4_in_core = sorted({cg_group_of[o] for o in c4c if o in cg_group_of} - {int(row["Group"])})
        rows.append({
            "cg4_group": int(row["Group"]),
            "lim_host": int(row["lim"]),
            "n_host_members": int(row["n_host"]),
            "control4c_retained": int(row["lim"]) in retained_c4c,
            "control4b_retained": int(row["lim"]) in retained_c4b,
            "other_cg4_in_would_be_control4c": other_cg4_in_core,
            "cg4_bgg_in_would_be_control4b": cg_bgg in set(c4b),
        })
    return {
        "n": len(rows),
        "cg4_groups": [r["cg4_group"] for r in rows],
        "n_host_control4c_retained": sum(r["control4c_retained"] for r in rows),
        "n_host_control4c_with_other_cg4": sum(bool(r["other_cg4_in_would_be_control4c"]) for r in rows),
        "n_host_control4b_retained": sum(r["control4b_retained"] for r in rows),
        "n_cg4_bgg_in_would_be_control4b": sum(r["cg4_bgg_in_would_be_control4b"] for r in rows),
        "systems": rows,
    }


def run_schematic_figure(sample: dict, output_dir: str | None = None) -> dict:
    pc_gals = pd.read_csv(os.path.join(co.DATA_PATH, "PC_Gals.csv"))
    pc_gals["objid"] = pc_gals["objid"].astype("int64")
    choice = select_groups(sample, pc_gals)

    panels = {
        "a": _panel_data(sample, pc_gals, choice["cg4_group"], choice["lim_host"]),
        "b": _panel_data(sample, pc_gals, choice["panel_b"]["cg4_group"], choice["panel_b"]["lim_host"]),
    }
    half = SCALE_PADDING * max(p["meta"]["host_half_extent_kpc"] for p in panels.values())
    windows = {k: _inset_window(p) for k, p in panels.items()}
    zoom_half = float(np.ceil(max(w[2] for w in windows.values()) / 10.0) * 10.0)
    # Overlay markers are ~16 pt wide; on a ~2.8 in panel that is ~8% of the
    # side, so require at least that much clearance around the annotations.
    min_clear_kpc = 0.08 * 2 * half

    result = {"status": "ok", **choice}
    result.update(panels["a"]["meta"])
    result["panel_b"].update(panels["b"]["meta"])
    result["half_width_kpc"] = float(half)
    result["inset_half_kpc"] = zoom_half
    result["off_centre_embedded"] = off_centre_embedded_summary(sample, pc_gals)
    for k in ("a", "b"):
        target = result if k == "a" else result["panel_b"]
        target["inset_centre_kpc"] = [round(windows[k][0], 1), round(windows[k][1], 1)]
        target["quartet_status"] = {
            key: _status_text(key, panels[k]["meta"][f"n_cg4_in_would_be_control4{key[-1].lower()}"],
                              panels[k]["retained"][key])
            for key in ("C4B", "C4C")
        }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        style = plt.style.context("default")  # earlier modules may set a seaborn style
        style.__enter__()
        fig = plt.figure(figsize=(8.4, 3.75))
        grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.5], wspace=0.12, left=0.07, right=0.995,
                                top=0.86, bottom=0.2)
        axes = [fig.add_subplot(grid[0, 0])]
        axes.append(fig.add_subplot(grid[0, 1], sharey=axes[0]))
        zoom_grid = grid[0, 2].subgridspec(2, 1, hspace=0.32)
        zooms = {"a": fig.add_subplot(zoom_grid[0, 0]), "b": fig.add_subplot(zoom_grid[1, 0])}
        lum_ref = max(p["host"]["Lum"].max() for p in panels.values())
        titles = {
            "a": "(a) Lim host {lim} ($N={n}$, $z={z:.3f}$)\nembedded CG$_4$ {cg} is the host core",
            "b": "(b) Lim host {lim} ($N={n}$, $z={z:.3f}$)\nembedded CG$_4$ {cg} is off-centre",
        }
        for ax, k in zip(axes, ("a", "b")):
            panel, meta = panels[k], panels[k]["meta"]
            target = result if k == "a" else result["panel_b"]
            _draw_members(ax, panel, lum_ref)
            ax.set_xlim(half, -half)
            ax.set_ylim(-half, half)
            ax.set_xlabel(r"$\Delta x$ (kpc)", fontsize=FONT["label"])
            ax.tick_params(labelsize=FONT["ticks"])
            ax.set_title(titles[k].format(lim=meta["lim_host"], n=meta["n_host_members"],
                                          z=meta["cg4_z_group"], cg=meta["cg4_group"]),
                         fontsize=FONT["title"], pad=6)
            cx, cy, _ = windows[k]
            ax.add_patch(Rectangle((cx - zoom_half, cy - zoom_half), 2 * zoom_half, 2 * zoom_half,
                                   facecolor="none", edgecolor="0.35", linewidth=0.8, zorder=5))
            box = [(half - cx - zoom_half) / (2 * half), (cy - zoom_half + half) / (2 * half),
                   zoom_half / half, zoom_half / half]
            zoom = zooms[k]
            _draw_members(zoom, panel, lum_ref, overlay_scale=1.15)
            zoom.set_xlim(cx + zoom_half, cx - zoom_half)
            zoom.set_ylim(cy - zoom_half, cy + zoom_half)
            zoom.xaxis.set_major_locator(MaxNLocator(3))
            zoom.yaxis.set_major_locator(MaxNLocator(3))
            zoom.tick_params(labelsize=FONT["zoom_ticks"], length=2.5, pad=2)
            zoom.set_title(rf"({k}) zoom: CG$_4$ {meta['cg4_group']}", fontsize=FONT["zoom_title"], pad=3)
            for spine in zoom.spines.values():
                spine.set_edgecolor("0.35")
            status = "\n".join(target["quartet_status"][key] for key in ("C4B", "C4C"))
            target["status_corner"], target["status_clearance_kpc"] = _place_text(
                fig, ax, status, panel, half, box, min_clear_kpc
            )
        axes[0].set_ylabel(r"$\Delta y$ (kpc)", fontsize=FONT["label"])
        plt.setp(axes[1].get_yticklabels(), visible=False)
        handles = [
            Line2D([], [], linestyle="", marker="o", markersize=7.5, markerfacecolor="0.78",
                   markeredgecolor="0.45", markeredgewidth=0.6, label=LEGEND_LABELS["host"]),
        ]
        for key in ("CG4", "C4B", "C4C"):
            handles.append(Line2D([], [], linestyle="", marker=MARKERS[key],
                                  markersize=np.sqrt(OVERLAY_AREA[key]) * 0.72,
                                  markerfacecolor="none", markeredgecolor=COLOURS[key],
                                  markeredgewidth=1.3, label=LEGEND_LABELS[key]))
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=FONT["legend"],
                   bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, columnspacing=1.4)
        path = os.path.join(output_dir, "fig_sample_schematic.pdf")
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        style.__exit__(None, None, None)
        result["figure"] = os.path.basename(path)
    return safe_json(result)


def refresh_results_entry(output_dir: str = co.FIGURES_PATH, results_path: str = co.RESULTS) -> dict:
    """Regenerate the figure from the cached sample and update results.json."""

    import pickle

    with open(os.path.join(co.DATA_PATH, co.PROCESS_SAMPLES), "rb") as fh:
        sample = pickle.load(fh)
    entry = run_schematic_figure(sample, output_dir=output_dir)
    with open(results_path, encoding="utf-8") as fh:
        data = json.load(fh)
    data["schematic_figure"] = entry
    tmp = results_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
        fh.write("\n")
    os.replace(tmp, results_path)
    return entry


if __name__ == "__main__":
    print(json.dumps(refresh_results_entry(), indent=1))
