"""Selection and missing-data diagnostics for the extended analyses."""

from __future__ import annotations

import os

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

try:
    from extended_data import ensure_galaxy_frame
    from extended_stats import (
        holm_correction,
        safe_json,
        standardized_mean_difference,
        two_sample_summary,
    )
except ModuleNotFoundError:  # pragma: no cover
    from .extended_data import ensure_galaxy_frame
    from .extended_stats import (
        holm_correction,
        safe_json,
        standardized_mean_difference,
        two_sample_summary,
    )


AVAILABILITY = {
    "morphology": ["elliptical"],
    "sSFR": ["quenched"],
    "stellar_mass": ["logMstar"],
    "colours": ["u_minus_r", "u_minus_g", "g_minus_r", "r_minus_i"],
    "spectral_lines": ["h_alpha_eqw", "h_beta_eqw", "oiii_5007_eqw", "nii_6584_eqw"],
    "velocity_data": ["V_norm"],
    "group_scale_quantities": [
        "R_scale",
        "velocity_dispersion",
        "log_group_luminosity",
        "dominance",
    ],
}

AVAILABILITY_LABELS = {
    "morphology": "secure GZ class",
    "sSFR": "sSFR class",
    "stellar_mass": "stellar mass",
    "colours": "SDSS colour columns",
    "spectral_lines": "BPT lines",
    "velocity_data": "velocity data",
    "group_scale_quantities": "group-scale covariates",
}

AVAILABILITY_NOTES = {
    "denominator": "Each fraction uses the final per-sample galaxy count as its denominator.",
    "morphology": (
        "The secure GZ class row counts galaxies classified as elliptical/smooth or "
        "spiral/features; uncertain Galaxy Zoo morphologies remain in the morphology table."
    ),
    "colours": (
        "The SDSS colour-columns row counts complete broad photometric columns in the "
        "final merged frame. The stricter colour-analysis matched subset is reported "
        "separately from the colour module."
    ),
}

GROUP_SCALE_AUDIT_COLUMNS = [
    "R_scale",
    "velocity_dispersion",
    "log_group_mass",
    "log_group_luminosity",
    "dominance",
]

MATCHING_AUDIT_COLUMNS = [
    "logMstar",
    "z_numeric",
    "rank",
    "log_group_luminosity",
    "velocity_dispersion",
]


def _nearest_angular(frame):
    values = np.full(len(frame), np.nan)
    positions = {index: position for position, index in enumerate(frame.index)}
    for _, group in frame.groupby("group_uid", observed=True):
        clean = group[["RA", "Dec"]].apply(
            lambda column: np.asarray(column, dtype=float)
        )
        if len(group) < 2 or not np.isfinite(clean.to_numpy()).all():
            continue
        ra = np.deg2rad(clean["RA"].to_numpy())
        dec = np.deg2rad(clean["Dec"].to_numpy())
        delta_ra = ra[:, None] - ra[None, :]
        delta_dec = dec[:, None] - dec[None, :]
        a = (
            np.sin(delta_dec / 2) ** 2
            + np.cos(dec[:, None]) * np.cos(dec[None, :]) * np.sin(delta_ra / 2) ** 2
        )
        angular = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        np.fill_diagonal(angular, np.inf)
        nearest_arcsec = np.min(angular, axis=1) * 180 / np.pi * 3600
        for index, value in zip(group.index, nearest_arcsec):
            values[positions[index]] = value
    return values


def _plot_availability(availability_counts, path):
    samples = ["CG4", "Control4B", "Control4C", "RG4"]
    quantities = list(AVAILABILITY)
    matrix = np.array(
        [
            [
                availability_counts.get(sample, {})
                .get(quantity, {})
                .get("fraction", 0)
                for quantity in quantities
            ]
            for sample in samples
        ]
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(
        np.arange(len(quantities)),
        [AVAILABILITY_LABELS.get(value, value.replace("_", " ")) for value in quantities],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(np.arange(len(samples)), samples)
    for row in range(len(samples)):
        for column in range(len(quantities)):
            item = availability_counts.get(samples[row], {}).get(quantities[column], {})
            n_available = item.get("n_available", 0)
            n_total = item.get("n_total", 0)
            ax.text(
                column,
                row,
                f"{n_available}/{n_total}\n({100 * matrix[row, column]:.0f}%)",
                ha="center",
                va="center",
                color="white" if matrix[row, column] < 0.55 else "black",
                fontsize=9,
            )
    fig.colorbar(image, ax=ax, label="Available fraction of final sample")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def _plot_colour_bias(frame, path):
    if "colour_matched" not in frame:
        return None
    matched = frame.loc[frame["colour_matched"], "logMstar"].dropna()
    unmatched = frame.loc[~frame["colour_matched"], "logMstar"].dropna()
    if matched.empty or unmatched.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(
        matched,
        bins=25,
        density=True,
        histtype="step",
        linewidth=2,
        label="Colour matched",
    )
    ax.hist(
        unmatched,
        bins=25,
        density=True,
        histtype="step",
        linewidth=2,
        label="Unmatched",
    )
    ax.set_xlabel(r"$\log(M_\star/M_\odot)$")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def _group_scale_audit(frame):
    fractions = {}
    for column in GROUP_SCALE_AUDIT_COLUMNS:
        if column in frame:
            fractions[column] = float(frame[column].notna().mean())
        else:
            fractions[column] = 0.0
    used_columns = AVAILABILITY["group_scale_quantities"]
    return {
        "availability_columns": used_columns,
        "column_available_fraction": fractions,
        "missing_or_sparse_columns": [
            column for column, fraction in fractions.items() if fraction < 0.65
        ],
    }


def _sample_size_audit(frame):
    """Return per-sample denominators for the main diagnostic families."""

    rows = {}
    colour_columns = [
        column
        for column in ["u_minus_r", "u_minus_g", "g_minus_r", "r_minus_i"]
        if column in frame
    ]
    bpt_columns = [column for column in ["log_NII_Ha", "log_OIII_Hb"] if column in frame]
    matching_columns = [column for column in MATCHING_AUDIT_COLUMNS if column in frame]

    for sample_name, part in frame.groupby("sample", observed=True):
        if "morphology" in part:
            morphology_table = int(part["morphology"].notna().sum())
        elif {"elliptical", "spiral"}.issubset(part.columns):
            morphology_table = int((part["elliptical"].notna() | part["spiral"].notna()).sum())
        else:
            morphology_table = 0

        row = {
            "total_galaxies": int(len(part)),
            "sSFR_table_N": int(part["quenched"].notna().sum()) if "quenched" in part else 0,
            "morphology_table_N": morphology_table,
            "secure_morphology_N": (
                int((part["elliptical"].notna() | part["spiral"].notna()).sum())
                if {"elliptical", "spiral"}.issubset(part.columns)
                else 0
            ),
            "stellar_mass_N": int(part["logMstar"].notna().sum()) if "logMstar" in part else 0,
            "all_colours_N": (
                int(part[colour_columns].notna().all(axis=1).sum())
                if colour_columns
                else 0
            ),
            "satellites_BGG_distance_N": (
                int(
                    (
                        part.get("is_satellite", 0).eq(1)
                        & part.get("dist2BGG_kpc", np.nan).notna()
                    ).sum()
                )
                if {"is_satellite", "dist2BGG_kpc"}.issubset(part.columns)
                else 0
            ),
            "matching_complete_case_N": (
                int(part[matching_columns].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).sum())
                if matching_columns
                else 0
            ),
            "BPT_AGN_classifiable_N": (
                int(part[bpt_columns].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).sum())
                if bpt_columns
                else 0
            ),
        }
        rows[sample_name] = row
    return rows


def run_selection_diagnostics(data, output_dir: str | None = None):
    """Quantify availability, colour-selection, and close-neighbour differences."""

    frame = ensure_galaxy_frame(data)
    if frame.empty or "sample" not in frame:
        return {"status": "skipped", "reason": "no_galaxy_samples"}
    availability = {}
    availability_counts = {}
    for sample_name, part in frame.groupby("sample", observed=True):
        availability[sample_name] = {}
        availability_counts[sample_name] = {}
        for quantity, columns in AVAILABILITY.items():
            existing = [column for column in columns if column in part]
            available = part[existing].notna().all(axis=1) if existing else None
            n_available = int(available.sum()) if available is not None else 0
            n_total = int(len(part))
            fraction = float(n_available / n_total) if n_total else 0.0
            availability[sample_name][quantity] = fraction
            availability_counts[sample_name][quantity] = (
                {
                    "n_available": n_available,
                    "n_total": n_total,
                    "fraction": fraction,
                }
            )

    missingness = {}
    tests = []
    keys = []
    for quantity, columns in AVAILABILITY.items():
        existing = [column for column in columns if column in frame]
        if not existing:
            continue
        available = frame[existing].notna().all(axis=1)
        cg = available[frame["is_CG4"] == 1]
        control = available[frame["is_CG4"] == 0]
        table = [
            [int(cg.sum()), int((~cg).sum())],
            [int(control.sum()), int((~control).sum())],
        ]
        p_value = float(stats.fisher_exact(table).pvalue)
        missingness[quantity] = {
            "cg4_available_fraction": float(cg.mean()),
            "control_available_fraction": float(control.mean()),
            "fisher_p": p_value,
        }
        tests.append(p_value)
        keys.append(quantity)
    for key, adjusted in zip(keys, holm_correction(tests)):
        missingness[key]["p_adj"] = adjusted

    colour_columns = [
        column
        for column in ["u_minus_r", "u_minus_g", "g_minus_r", "r_minus_i"]
        if column in frame
    ]
    frame = frame.copy()
    frame["colour_matched"] = (
        frame[colour_columns].notna().all(axis=1) if colour_columns else False
    )
    matched_unmatched = {}
    for column in ["logMstar", "z_numeric", "rank"]:
        if column in frame:
            matched_unmatched[column] = two_sample_summary(
                frame.loc[frame["colour_matched"], column],
                frame.loc[~frame["colour_matched"], column],
            )
            matched_unmatched[column]["standardized_mean_difference"] = (
                standardized_mean_difference(
                    frame.loc[frame["colour_matched"], column],
                    frame.loc[~frame["colour_matched"], column],
                )
            )
    for column in ["quenched", "starforming", "elliptical", "spiral"]:
        if column in frame:
            matched = frame.loc[frame["colour_matched"], column].dropna()
            unmatched = frame.loc[~frame["colour_matched"], column].dropna()
            if len(matched) and len(unmatched):
                table = [
                    [int(matched.sum()), int(len(matched) - matched.sum())],
                    [int(unmatched.sum()), int(len(unmatched) - unmatched.sum())],
                ]
                matched_unmatched[column] = {
                    "status": "ok",
                    "matched_fraction": float(matched.mean()),
                    "unmatched_fraction": float(unmatched.mean()),
                    "fisher_p": float(stats.fisher_exact(table).pvalue),
                }

    angular_result = {"status": "skipped", "reason": "missing_RA_Dec"}
    if {"RA", "Dec", "group_uid"}.issubset(frame.columns):
        frame["nearest_angular_separation_arcsec"] = _nearest_angular(frame)
        angular_result = two_sample_summary(
            frame.loc[frame["is_CG4"] == 1, "nearest_angular_separation_arcsec"],
            frame.loc[frame["is_CG4"] == 0, "nearest_angular_separation_arcsec"],
        )
        angular_result["fibre_collision_threshold_arcsec"] = 55
        angular_result["cg4_fraction_below_threshold"] = float(
            (
                frame.loc[frame["is_CG4"] == 1, "nearest_angular_separation_arcsec"]
                < 55
            ).mean()
        )
        angular_result["control_fraction_below_threshold"] = float(
            (
                frame.loc[frame["is_CG4"] == 0, "nearest_angular_separation_arcsec"]
                < 55
            ).mean()
        )

    result = {
        "status": "ok",
        "availability_by_sample": availability,
        "availability_counts_by_sample": availability_counts,
        "availability_labels": AVAILABILITY_LABELS,
        "availability_notes": AVAILABILITY_NOTES,
        "sample_size_audit": _sample_size_audit(frame),
        "group_scale_column_audit": _group_scale_audit(frame),
        "missingness_comparisons": missingness,
        "matched_unmatched_comparisons": matched_unmatched,
        "nearest_angular_separation": angular_result,
        "strong_selection_bias": any(
            value.get("p_adj", 1) < 0.05
            and abs(
                value["cg4_available_fraction"] - value["control_available_fraction"]
            )
            >= 0.1
            for value in missingness.values()
        ),
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result["availability_figure"] = _plot_availability(
            availability_counts,
            os.path.join(output_dir, "fig_data_availability_by_sample.pdf"),
        )
        result["colour_bias_figure"] = _plot_colour_bias(
            frame, os.path.join(output_dir, "fig_colour_matched_selection_bias.pdf")
        )
    return safe_json(result)
