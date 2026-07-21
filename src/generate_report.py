# region Imports

import os
import json
from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateNotFound,
    TemplateSyntaxError,
)
import subprocess

import numpy as np
from scipy.interpolate import interp1d

try:
    import config as co
    from utils import labels_utils as lu
    from utils import graphics_utils as gu
except ModuleNotFoundError:  # pragma: no cover
    from . import config as co
    from .utils import labels_utils as lu
    from .utils import graphics_utils as gu


# os.makedirs("output", exist_ok=True)

# endregion

RESULT_BLOCK_KEYS = {
    "extended_specialness",
    "phase_space_segregation",
    "morphology_dominance",
    "luminosity_function",
    "sSFR_robustness",
    "cg4_pc_quartet_overlap",
}


def _has_non_ascii(path):
    """Return whether the file contains any non-ASCII byte."""

    try:
        with open(path, "rb") as f:
            data = f.read()
        return any(b > 0x7F for b in data)
    except Exception:
        return False


def initialise_json(build=False):
    """Create the output directory and initialize the results file as an object."""
    file = co.RESULTS_BUILD if build else co.RESULTS
    os.makedirs(co.OUTPUT_PATH, exist_ok=True)
    with open(file, "w") as f:
        f.write("{\n")  # start an object, not a list
    # first entry: write the FIGURES_PATH field
    if not build:
        append_json("SUBFIGURES_PATH", co.SUBFIGURES_PATH)
        append_json("BIB_FILE", co.BIB_FILE)
        append_json("Z_MIN", co.Z_MIN)
        append_json("Z_MAX", co.Z_MAX)
        append_json("R_MAX", co.R_MAX)
        append_json("DATA_RELEASE", co.DATA_RELEASE)
        append_json("sSFR_status", co.sSFR_status)
        append_json("NosSFR_LABEL", co.NosSFR_LABEL)
        append_json("Morphologies", co.Morphologies)
        append_json("DOMINATIION_CRITERIA", co.DOMINATIION_CRITERIA)


def append_json(key: str, value, build=False):
    """Append a single JSON key/value pair, followed by a comma+newline."""
    file = co.RESULTS_BUILD if build else co.RESULTS

    # auto-encode interp1d and NumPy types
    if isinstance(value, interp1d):
        value = encode_interp1d(value)
    value = _jsonable(value)

    with open(file, "a") as f:
        f.write(f'"{key}": ')
        json.dump(value, f)
        f.write(",\n")


def finalize_json(build=False):
    """Remove trailing comma and close the object."""
    file = co.RESULTS_BUILD if build else co.RESULTS
    with open(file, "rb+") as f:
        f.seek(-2, os.SEEK_END)  # back over the last comma+newline
        f.truncate()  # remove them
        f.write(b"\n}")  # close the object


def _load_json(path):
    """Safe JSON loader: returns {} if the file doesn't exist or can't be read."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[Info] optional JSON not found: {path}")
        return {}
    except Exception as e:
        print(f"[Error] reading {path}: {e}")
        return {}


def deep_merge(base, overlay):
    """Return a recursive dictionary merge where overlay values take precedence.

    Nested dictionaries are merged without deleting keys that exist only in the
    base mapping. Lists and scalar values are treated as atomic values: when a
    key exists in both inputs, the overlay value replaces the base value.
    Neither input dictionary is modified.
    """

    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _get_path(mapping, dotted_path):
    """Return a nested value addressed by a dotted path, or None if absent."""

    value = mapping
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _validate_render_context(ctx):
    """Fail loudly when the paper context is missing required analysis blocks."""

    required_paths = [
        "CG4_Groups_nonsplit_N",
        "pval_MSresiduals_Control4B_Gals",
        "CG4_Gals_N_Elliptical",
        "Colour_matching_summary",
        "extended_specialness",
        "extended_specialness.specialness_models",
        "extended_specialness.specialness_models.elliptical_all",
        "extended_specialness.specialness_models.spiral_all",
        "extended_specialness.matched_controls",
        "extended_specialness.matched_controls.effects.elliptical_fraction",
        "extended_specialness.matched_controls.effects.spiral_fraction",
        "extended_specialness.morphology_robustness",
        "extended_specialness.morphology_dominance",
        "morphology_dominance",
        "extended_specialness.sample_size_audit",
        "phase_space_segregation",
    ]
    missing = [path for path in required_paths if _get_path(ctx, path) is None]

    morphology_robustness = _get_path(ctx, "extended_specialness.morphology_robustness")
    if isinstance(morphology_robustness, dict) and morphology_robustness.get("status") == "ok":
        for path in [
            "extended_specialness.morphology_robustness.concentration_index",
            "extended_specialness.morphology_robustness.cg_class_split",
            "extended_specialness.morphology_robustness.adjusted_models_with_close_flag",
            "extended_specialness.morphology_robustness.exact_tests_after_excluding_close",
        ]:
            if _get_path(ctx, path) is None:
                missing.append(path)

    size_analysis = _get_path(ctx, "extended_specialness.size_analysis")
    if isinstance(size_analysis, dict) and size_analysis.get("status") == "ok":
        for path in [
            "extended_specialness.size_analysis.adjusted",
            "extended_specialness.size_analysis.matched",
            "extended_specialness.size_analysis.availability_audit",
            "extended_specialness.size_analysis.verdicts",
            "extended_specialness.size_analysis.petrosian",
            "extended_specialness.size_analysis.concentration",
        ]:
            if _get_path(ctx, path) is None:
                missing.append(path)

    if missing:
        joined = "\n  - ".join(missing)
        raise KeyError(f"Render context missing required keys:\n  - {joined}")

    _validate_consistent_table_counts(ctx)


def _validate_consistent_table_counts(ctx):
    """Validate table-level count identities before rendering the paper."""

    samples = ["CG4", "Control4B", "Control4C", "RG4"]
    statuses = ["Quenched", "Starforming", "NosSFR"]
    morphs = ["Elliptical", "Spiral", "Uncertain"]
    sample_totals = {}
    problems = []
    for sample in samples:
        all_counts = {
            status: _get_path(ctx, f"{sample}_Gals_N{status}") for status in statuses
        }
        bgg_counts = {
            status: _get_path(ctx, f"{sample}_BGG_N{status}") for status in statuses
        }
        sat_counts = {
            status: _get_path(ctx, f"{sample}_Sat_N{status}") for status in statuses
        }
        if all(value is not None for value in [*all_counts.values(), *bgg_counts.values(), *sat_counts.values()]):
            for status in statuses:
                if int(all_counts[status]) != int(bgg_counts[status]) + int(sat_counts[status]):
                    problems.append(
                        f"{sample} {status}: all={all_counts[status]}, "
                        f"BGG+sat={int(bgg_counts[status]) + int(sat_counts[status])}"
                    )
            sample_totals[sample] = int(sum(all_counts.values()))

        morph_counts = {
            morph: _get_path(ctx, f"{sample}_Gals_N_{morph}") for morph in morphs
        }
        if sample in sample_totals and all(value is not None for value in morph_counts.values()):
            morph_total = int(sum(morph_counts.values()))
            if morph_total != sample_totals[sample]:
                problems.append(
                    f"{sample}: morphology total={morph_total}, "
                    f"sSFR total={sample_totals[sample]}"
                )

    tidal_total = _get_path(ctx, "extended_specialness.tidal_indices.n_galaxies_with_pairs")
    if tidal_total is not None and sample_totals:
        ssfr_total = int(sum(sample_totals.values()))
        if int(tidal_total) != ssfr_total:
            problems.append(
                f"pairwise-analysis total={tidal_total}, sSFR sample total={ssfr_total}"
            )

    if problems:
        joined = "\n  - ".join(problems)
        raise AssertionError(f"Render context has inconsistent table counts:\n  - {joined}")


def _write_json(path, data):
    """Write JSON atomically so the persisted build context is never partial."""

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(_jsonable(data), f)
        f.write("\n")
    os.replace(tmp_path, path)


def _load_referee_values():
    """Referee-revision sensitivity values (referee/values/*.json).

    Each file becomes ``referee[<stem>]`` in the render context, so the
    manuscript cites sensitivity numbers without hand-copied literals.
    Absent directory or files simply yield an empty mapping; template
    blocks must guard with ``<% if referee.get(...) %>``.
    """

    directory = os.path.join(co.BASE_PATH, "referee", "values")
    values = {}
    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            if name.endswith(".json"):
                values[name[:-5]] = _load_json(os.path.join(directory, name))
    return values


def _build_render_context(build_data, results_data):
    """Construct the single coherent Jinja context from both result files."""

    render_data = deep_merge(build_data, results_data)
    for key in RESULT_BLOCK_KEYS:
        if isinstance(results_data.get(key), dict):
            render_data[key] = results_data[key]
    ctx = dict(render_data)

    # Template aliases. They intentionally point at the same merged data so
    # template sections cannot accidentally see different JSON worlds.
    ctx["r"] = render_data
    ctx["build"] = render_data
    ctx["results"] = results_data
    ctx["results_build"] = build_data

    ctx["samples"] = ["CG4", "Control4B", "Control4C", "RG4"]
    ctx["statuses"] = ["Quenched", "Starforming"]
    ctx["morphs"] = ["Elliptical", "Spiral", "Uncertain"]

    ctx["quantities"] = ["sSFR", "M_r", "lgm"]
    ctx["qdefs"] = [
        {"key": "sSFR", "label": r"$\log_{10}(\mathrm{sSFR}\,[\mathrm{yr}^{-1}])$"},
        {"key": "M_r", "label": r"$M_r$"},
        {
            "key": "lgm",
            "label": r"$\log(M_\star/M_\odot)$",
        },
    ]
    ctx["lu"] = lu
    ctx["gu"] = gu

    ctx["extended_specialness"] = render_data.get("extended_specialness", {})
    ctx["phase_space_segregation"] = render_data.get(
        "phase_space_segregation",
        ctx["extended_specialness"].get("phase_space_segregation", {}),
    )
    ctx["referee"] = _load_referee_values()
    _validate_render_context(ctx)
    return ctx, render_data


def generate_report():
    """
    Generate a LaTeX report with bibliography:
    pdflatex → bibtex → pdflatex ×2.
    """
    # 1) Load both JSONs
    build_data = _load_json(co.RESULTS_BUILD)  # produced only when you rebuild
    results_data = _load_json(co.RESULTS)  # produced every run

    if not results_data and not build_data:
        print("[Error] Neither results nor build JSON could be loaded.")
        return

    # 2) Deep-merge for the template and persist the coherent build context.
    try:
        ctx, render_data = _build_render_context(build_data, results_data)
        _write_json(co.RESULTS_BUILD, render_data)
        print(f"[Info] merged render context written to: {co.RESULTS_BUILD}")
    except KeyError as e:
        print(f"[Error] {e}")
        return

    # 3) Render LaTeX via Jinja2
    env = Environment(
        loader=FileSystemLoader(co.TEMPLATE_PATH),
        undefined=StrictUndefined,
        block_start_string="<%",
        block_end_string="%>",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="<#",
        comment_end_string="#>",
    )
    env.filters["fmt"] = _format_number
    env.filters["pct"] = _format_percent
    try:
        template = env.get_template(co.TEMPLATE_FILE)
        rendered_tex = template.render(ctx)
    except (TemplateNotFound, TemplateSyntaxError) as e:
        print(f"[Error] template problem: {e}")
        return
    except Exception as e:
        print(f"[Error] rendering template: {e}")
        return

    # 4) Write out the .tex file (unchanged)
    report_dir = co.REPORT_PATH
    tex_file = co.REPORT_FILE
    tex_path = os.path.join(report_dir, tex_file)
    os.makedirs(report_dir, exist_ok=True)
    try:
        with open(tex_path, "w") as f:
            f.write(rendered_tex)
        print(f"[Info] LaTeX source written to: {tex_path}")
    except Exception as e:
        print(f"[Error] writing {tex_file}: {e}")
        return

    basename, _ = os.path.splitext(tex_file)

    # 5) Helper to run commands without immediate exception
    def run_proc(cmd, *, cwd=None, description=None):
        """Run a subprocess and print its stdout/stderr in a compact build log."""

        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
        print(f"[Info] {(description or ' '.join(cmd))} -> {proc.returncode}")
        if proc.stdout:
            print(proc.stdout)
        if proc.returncode != 0 and proc.stderr:
            print(proc.stderr)
        return proc

    # 6) Compile sequence
    try:
        # 6.1 Initial pdflatex
        # proc = run_proc(
        #     ["pdflatex", "-interaction=nonstopmode", "-output-directory", report_dir, tex_file],
        #     cwd=report_dir,
        #     description="pdflatex (1)"
        # )
        proc = run_proc(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                "-synctex=0",
                "-output-directory",
                report_dir,
                tex_file,
            ],
            cwd=report_dir,
            description="pdflatex (1, quiet)",
        )
        if proc.returncode != 0:
            return

        # 6.2 BibTeX
        def _has(cmd):
            """Return whether a command is available on the current PATH."""

            try:
                return (
                    subprocess.run(
                        [cmd, "--version"], capture_output=True, text=True
                    ).returncode
                    == 0
                )
            except Exception:
                return False

        def _has_bibtex8_utf8():
            """Return whether bibtex8 can find the requested UTF-8 CS file."""

            if not _has("bibtex8"):
                return False
            local_csf = os.path.join(report_dir, "utf8c.csf")
            if os.path.exists(local_csf):
                return True
            try:
                return (
                    subprocess.run(
                        ["kpsewhich", "utf8c.csf"],
                        capture_output=True,
                        text=True,
                        check=False,
                    ).returncode
                    == 0
                )
            except Exception:
                return False

        bib_cmd = (
            ["bibtex8", "-W", "-c", "utf8c.csf", basename]
            if _has_bibtex8_utf8()
            else ["bibtex", basename]
        )
        proc = run_proc(bib_cmd, cwd=report_dir, description=" ".join(bib_cmd))
        if proc.returncode != 0:
            blg = os.path.join(report_dir, f"{basename}.blg")
            print(
                f"[Error] Bibliography failed. {os.path.basename(blg)} follows (if present):"
            )
            try:
                print(open(blg, "r", encoding="utf-8", errors="replace").read())
            except FileNotFoundError:
                print("[Error] .blg not found.")
            return
        # 6.3 Two more pdflatex runs
        for i in (1, 2):
            # proc = run_proc(
            #     ["pdflatex", "-interaction=nonstopmode",
            #      "-output-directory", report_dir, tex_file],
            #     description=f"pdflatex pass #{i+1}"
            # )
            proc = run_proc(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-file-line-error",
                    "-synctex=0",
                    "-output-directory",
                    report_dir,
                    tex_file,
                ],
                cwd=report_dir,
                description=f"pdflatex pass #{i + 1}",
            )
            if proc.returncode != 0:
                print(proc.stderr)
                return

        print(f"[Success] PDF generated at {os.path.join(report_dir, basename)}.pdf")

    except Exception as e:
        print(f"[Error] build pipeline aborted: {e}")


def _jsonable(obj):
    """Make common SciPy/NumPy types JSON-serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def _format_number(value, digits=2):
    """Format optional finite values safely for the LaTeX template."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    digits = int(digits)
    if abs(number) < 0.5 * 10 ** (-digits):
        number = 0.0
    return f"{number:.{digits}f}"


def _format_percent(value, digits=1):
    """Format an optional fraction as a percentage without a percent sign."""

    try:
        return _format_number(100 * float(value), digits)
    except (TypeError, ValueError):
        return "n/a"


def encode_interp1d(f):
    """
    Turn an interp1d into a JSON-friendly dict that:
      - can reconstruct f
      - is easy to render in LaTeX (points list)
    """
    # robustly get x/y even across SciPy versions
    x = np.asarray(getattr(f, "x", getattr(f, "_x", None)))
    y = np.asarray(getattr(f, "y", getattr(f, "_y", None)))
    if x is None or y is None:
        raise ValueError("Could not access x/y from interp1d object")

    # y can be 1D (shape n) or ND (shape n, ...)
    points = []
    if y.ndim == 1:
        for xi, yi in zip(x, y):
            points.append({"x": float(xi), "y": float(yi)})
    else:
        # keep vector-valued y as lists per row
        for i, xi in enumerate(x):
            points.append({"x": float(xi), "y": _jsonable(y[i])})

    fill_value = getattr(f, "fill_value", None)
    if isinstance(fill_value, np.ndarray):
        fill_value = fill_value.tolist()

    return {
        "__type__": "interp1d",
        "kind": getattr(f, "kind", "linear"),
        "bounds_error": bool(getattr(f, "bounds_error", False)),
        "fill_value": fill_value,
        # keep raw arrays for easy reconstruction:
        "x": x.tolist(),
        "y": y.tolist(),
        # keep point pairs for easy LaTeX/Jinja iteration:
        "points": points,
    }


def decode_interp1d(d):
    """Rebuild an interp1d from the dict we wrote."""
    if d.get("__type__") != "interp1d":
        raise ValueError("Not an interp1d payload")
    return interp1d(
        np.array(d["x"]),
        np.array(d["y"]),
        kind=d.get("kind", "linear"),
        fill_value=d.get("fill_value", None),
        bounds_error=d.get("bounds_error", False),
    )
