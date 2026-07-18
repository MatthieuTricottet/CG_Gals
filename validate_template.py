"""Static validation of the Jinja2/LaTeX paper template.

Usage:
    python validate_template.py src/paper_template/paper_template.tex

Checks (in order):
  1. The template parses under the pipeline's Jinja2 delimiters
     (blocks ``<% %>``, variables ``<< >>``, comments ``<# #>``), so a
     partially deleted guarded block is caught before any render.
  2. Every ``\\ref``/``\\pageref`` target has a matching ``\\label``
     (set(ref) - set(label) must be empty). Unreferenced labels are
     reported as info only, since figures may be cited by position.
  3. ``\\begin{env}``/``\\end{env}`` counts balance for every environment.

Exit status is 0 when all checks pass, 1 otherwise.
"""

import re
import sys

from jinja2 import Environment, TemplateSyntaxError

FAILED = False


def fail(message):
    global FAILED
    FAILED = True
    print(f"FAIL: {message}")


def ok(message):
    print(f"PASS: {message}")


def check_jinja_parse(source):
    env = Environment(
        block_start_string="<%",
        block_end_string="%>",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="<#",
        comment_end_string="#>",
    )
    try:
        env.parse(source)
    except TemplateSyntaxError as e:
        fail(f"Jinja2 parse error at line {e.lineno}: {e.message}")
        return
    ok("Jinja2 template parses with the pipeline delimiters")


def check_refs_labels(source):
    labels = set(re.findall(r"\\label\{([^}]+)\}", source))
    # lstlisting-style labels appear as an optional-argument key.
    labels |= set(re.findall(r"label=\{([^}]+)\}", source))
    refs = set(re.findall(r"\\(?:page)?ref\{([^}]+)\}", source))
    dangling = sorted(refs - labels)
    if dangling:
        fail(f"{len(dangling)} \\ref without \\label: {', '.join(dangling)}")
    else:
        ok(f"all {len(refs)} \\ref targets have a matching \\label")
    unreferenced = sorted(labels - refs)
    if unreferenced:
        print(f"INFO: {len(unreferenced)} labels never \\ref'd: "
              f"{', '.join(unreferenced)}")


def check_environment_balance(source):
    begins = re.findall(r"\\begin\{([^}]+)\}", source)
    ends = re.findall(r"\\end\{([^}]+)\}", source)
    problems = []
    for env_name in sorted(set(begins) | set(ends)):
        n_begin = begins.count(env_name)
        n_end = ends.count(env_name)
        if n_begin != n_end:
            problems.append(f"{env_name}: {n_begin} begin / {n_end} end")
    if problems:
        fail("unbalanced environments: " + "; ".join(problems))
    else:
        ok(f"all {len(set(begins))} environment kinds are begin/end balanced")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    with open(sys.argv[1], encoding="utf-8") as f:
        source = f.read()
    check_jinja_parse(source)
    check_refs_labels(source)
    check_environment_balance(source)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
