# region Imports

import os
import json
from jinja2 import Environment, FileSystemLoader, TemplateNotFound, TemplateSyntaxError
import os, subprocess, re

import numpy as np
from scipy.interpolate import interp1d

import config as co

# os.makedirs("output", exist_ok=True)

#endregion

def _has_non_ascii(path):
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
        f.write("{\n")     # start an object, not a list
    # first entry: write the FIGURES_PATH field
    if not build:
        append_json('SUBFIGURES_PATH', co.SUBFIGURES_PATH)
        append_json('BIB_FILE', co.BIB_FILE)
        append_json('Z_MIN', co.Z_MIN)
        append_json('Z_MAX', co.Z_MAX)
        append_json('R_MAX', co.R_MAX)
        append_json('DATA_RELEASE', co.DATA_RELEASE)
        append_json('sSFR_status', co.sSFR_status)
        append_json('Morphologies', co.Morphologies)
        append_json('sSFR_THRESHOLD', co.sSFR_THRESHOLD)
        append_json('sSFR_QUENCHED', co.sSFR_QUENCHED)
  

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
        f.seek(-2, os.SEEK_END)      # back over the last comma+newline
        f.truncate()                 # remove them
        f.write(b"\n}")              # close the object


# import os
import json
import subprocess
from jinja2 import Environment, FileSystemLoader, TemplateNotFound, TemplateSyntaxError

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

def generate_report():
    """
    Generate a LaTeX report with bibliography:
    pdflatex → bibtex → pdflatex ×2.
    """
    # 1) Load both JSONs
    build_data   = _load_json(co.RESULTS_BUILD)   # produced only when you rebuild
    results_data = _load_json(co.RESULTS)         # produced every run

    if not results_data and not build_data:
        print("[Error] Neither results nor build JSON could be loaded.")
        return

    # 2) Merge for the template (per-run results take precedence)
    ctx = {**build_data, **results_data}

    # 3) Render LaTeX via Jinja2
    env = Environment(
        loader=FileSystemLoader(co.TEMPLATE_PATH),
        block_start_string='<%', block_end_string='%>',
        variable_start_string='<<', variable_end_string='>>',
        comment_start_string='<#', comment_end_string='#>'
    )
    try:
        template = env.get_template(co.TEMPLATE_FILE)
        rendered_tex = template.render(ctx)
    except (TemplateNotFound, TemplateSyntaxError) as e:
        print(f"[Error] template problem: {e}")
        return
    except Exception as e:
        print(f"[Error] rendering template: {e}")
        return

    # 4) Write out the .tex file
    report_dir = co.REPORT_PATH
    tex_file   = co.REPORT_FILE
    tex_path   = os.path.join(report_dir, tex_file)
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
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
        print(f"[Info] {(description or ' '.join(cmd))} -> {proc.returncode}")
        if proc.stdout: print(proc.stdout)
        if proc.returncode != 0 and proc.stderr: print(proc.stderr)
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
                "-output-directory", report_dir,
                tex_file,
            ],
            cwd=report_dir,
            description="pdflatex (1, quiet)"
        )
        if proc.returncode != 0:
            return

        # 6.2 BibTeX
        def _has(cmd):
            try:
                return subprocess.run([cmd, "--version"], capture_output=True, text=True).returncode == 0
            except Exception:
                return False

        bib_cmd = ["bibtex8", "-W", "-c", "utf8c.csf", basename] if _has("bibtex8") else ["bibtex", basename]
        proc = run_proc(bib_cmd, cwd=report_dir, description=" ".join(bib_cmd))
        if proc.returncode != 0:
            blg = os.path.join(report_dir, f"{basename}.blg")
            print(f"[Error] Bibliography failed. {os.path.basename(blg)} follows (if present):")
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
                    "-output-directory", report_dir,
                    tex_file,
                ],
                cwd=report_dir,
                description=f"pdflatex pass #{i+1}"
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