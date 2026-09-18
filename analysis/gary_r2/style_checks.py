"""gary-r2 style rules, checked by grep on the rendered manuscript.

Usage: python analysis/gary_r2/style_checks.py [output/paper/paper.tex]

Rules (Phase 3 of the gary-r2 prompt), applied to the rendered LaTeX outside
Appendix A (SDSS data query) and the Data availability section:
  * zero occurrences: "audit", "provenance", "guardrail", "equality cases",
    ".py"/".csv" file names, "must not be read", "rather than equivalence";
  * at most one occurrence each: "estimand", "does not establish", "nominally";
  * no one-sentence paragraphs in the body (main text, outside floats);
  * main-text word count (texcount "Words in text") within +5% of the
    pre-revision count.
Exit status 1 when any rule fails.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "output" / "paper" / "paper.tex"
PRE_MAIN_WORDS = 7277  # texcount "Words in text" of the main text at tag pre-gary-r2

ZERO = ["audit", "provenance", "guardrail", "equality cases", "must not be read",
        "rather than equivalence"]
FILE_NAMES = re.compile(r"[\w\\_-]+\.(py|csv)\b")
AT_MOST_ONE = ["estimand", "does not establish", "nominally"]


def strip_exempt(src: str) -> str:
    """Remove Appendix A, the Data availability section, comments, and the bibliography."""

    out = []
    for line in src.splitlines():
        if line.lstrip().startswith("%"):
            continue
        out.append(re.sub(r"(?<!\\)%.*$", "", line))
    text = "\n".join(out)
    # Data availability section
    text = re.sub(r"\\section\*\{Data availability\}.*?(?=\\begin\{acknowledgements\})", "", text, flags=re.S)
    # Appendix A = first \section after \begin{appendix} up to the next \section
    m = re.search(r"\\begin\{appendix\}\s*\\section\{[^}]*\}.*?(?=\\section\{)", text, flags=re.S)
    if m:
        text = text[: m.start()] + "\\begin{appendix}\n" + text[m.end():]
    return text


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PAPER
    src = path.read_text()
    text = strip_exempt(src)
    failed = False
    print(f"style checks on {path}")
    for term in ZERO:
        n = len(re.findall(re.escape(term), text, flags=re.I))
        print(f"  {'FAIL' if n else 'PASS'}: '{term}' x{n} (max 0)")
        failed |= n > 0
    files = FILE_NAMES.findall(text)
    n = len(files)
    print(f"  {'FAIL' if n else 'PASS'}: .py/.csv file names x{n} (max 0)")
    failed |= n > 0
    for term in AT_MOST_ONE:
        n = len(re.findall(re.escape(term), text, flags=re.I))
        print(f"  {'FAIL' if n > 1 else 'PASS'}: '{term}' x{n} (max 1)")
        failed |= n > 1

    # one-sentence paragraphs in the body (main text, outside floats/equations)
    main = src[src.index("\\maketitle"):src.index("\\begin{appendix}")]
    main = main[: main.index("\\section*{Data availability}")]
    main = re.sub(r"\\begin\{(figure\*?|table\*?|equation|itemize)\}.*?\\end\{\1\}", "", main, flags=re.S)
    main = re.sub(r"\\\[.*?\\\]", " ", main, flags=re.S)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", main)]
    short = []
    for para in paragraphs:
        if not para or para.startswith("\\") and "\n" not in para and len(para) < 120:
            continue
        body = re.sub(r"\\(section|subsection|paragraph)\*?\{[^}]*\}(\\label\{[^}]*\})*", "", para).strip()
        if not body:
            continue
        # count sentence terminators outside math
        plain = re.sub(r"\\\(.*?\\\)", "M", body)
        plain = re.sub(r"\$[^$]*\$", "M", plain)
        plain = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "X", plain)
        plain = plain.replace("e.g.", "eg").replace("i.e.", "ie").replace("et al.", "etal").replace("Sect.", "Sect").replace("Fig.", "Fig").replace("Figs.", "Figs").replace("Sects.", "Sects").replace("App.", "App").replace("vs.", "vs")
        n_sent = len(re.findall(r"[.!?](\s|$)", plain))
        if n_sent <= 1 and len(plain.split()) > 3:
            short.append(body[:90].replace("\n", " "))
    print(f"  {'FAIL' if short else 'PASS'}: one-sentence body paragraphs x{len(short)}")
    for item in short:
        print("      -", item)
    failed |= bool(short)

    # word count
    main_tex = src[src.index("\\maketitle"):src.index("\\begin{appendix}")]
    main_tex = main_tex[: main_tex.index("\\begin{acknowledgements}")]
    tmp = ROOT / "results" / "diagnostics" / "gary_r2" / "_main_text_for_texcount.tex"
    tmp.write_text(main_tex)
    try:
        res = subprocess.run(["texcount", "-merge", str(tmp)], capture_output=True, text=True).stdout
        words = int(re.search(r"Words in text:\s*(\d+)", res).group(1))
        ratio = words / PRE_MAIN_WORDS - 1
        print(f"  {'PASS' if ratio <= 0.05 else 'FAIL'}: main-text words {words} vs {PRE_MAIN_WORDS} before ({100*ratio:+.1f}%, max +5%)")
        failed |= ratio > 0.05
    except Exception as exc:  # texcount missing
        print(f"  SKIP: texcount unavailable ({exc})")
    finally:
        tmp.unlink(missing_ok=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
