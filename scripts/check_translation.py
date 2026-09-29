"""Checks that the English translation of the Turkish text changed no number (post-freeze).

The translation is a presentation change only (CLAUDE.md, Analysis Freeze). This script
turns "nothing but the wording changed" into mechanical checks. Every mode exits non-zero
on the first failed check.

TOKENS
------
A token is a number, a commit hash or a file path, in text order:
  * path   : a name ending in a known extension, with optional directories, globs and
             brace alternatives (`bench_*{,_aligned}_publication_aligned.csv`);
  * hash   : 7-40 lowercase hex characters containing at least one letter and one digit;
  * number : digits with optional decimal/thousands groups and exponent; a leading
             sign (−, +, or - not preceded by a letter or digit) is part of the value,
             also across the Turkish prefix percent sign ("−%13.1" and "−13.1%" are
             the same token, −13.1).
Paths and hashes are removed before numbers are read, so their digits are not counted
twice. The same tokenizer is applied to both texts, so a spelling difference that is
not a number (a Turkish suffix, "%80" vs "80%") does not affect the check.

MODES
-----
package  Regenerated number package vs the committed one at --base (default 59158c5,
         the freeze commit). The full token SEQUENCE must be identical (same values, same
         order); only the two header lines that record the generating commit and time
         are excluded. The line count must match. The primary-family CSV that script 18
         also writes must be byte-identical to the one at --base, as git stores it
         (line endings normalized by .gitattributes).
log      outputs/experiment_log_en.md vs outputs/experiment_log.md. The translation
         starts with a header block that ends with a line consisting of `<!-- end of
         translation header -->`; line n of the original is line n + offset of the
         translation. Checks: (1) line-for-line structure: blank lines, heading levels,
         table rows with the same number of cells, code fences; (2) per stage (each
         level-1 heading starts a stage), the MULTISET of tokens is equal. Missing and
         extra tokens are reported.
ast      For each script, the syntax tree at --base and in the working tree are compared
         with every string constant blanked (inside f-strings: the formatted values are
         kept in order, with their format specs). An empty difference proves that only
         string literals (messages, labels, docstrings) changed.
terms    Side-by-side listing of the lines whose Turkish original contains one of the
         status terms (önceden, ön-kayıt, keşifsel, post hoc, görüldükten sonra, ...),
         for the log and the package. Also lists every English line containing
         "pre-regist" with its original, so a drift towards a stronger claim is visible.

Usage (from the repo root):
  python scripts/check_translation.py package --gpr-alignment publication
  python scripts/check_translation.py log
  python scripts/check_translation.py ast scripts/01_build_targets.py ...
  python scripts/check_translation.py terms --gpr-alignment publication

Runtime: a few seconds.
"""
import argparse
import ast
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import alignment

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
FREEZE_COMMIT = "59158c5"
PACKAGE = "paper_numbers.md"
PACKAGE_CSV = "primary_family_tests.csv"
LOG_TR = OUT_DIR / "experiment_log.md"
LOG_EN = OUT_DIR / "experiment_log_en.md"
HEADER_END = "<!-- end of translation header -->"

EXT = r"(?:py|csv|json|md|txt|xlsx|pdf|svg|png|sha256|\{[\w,]+\})"
PATH_RE = re.compile(r"[\w\-*{},/.]*?[\w\-*{}]\." + EXT + r"(?![\w])")
HASH_RE = re.compile(r"(?<![\w])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}(?![\w])")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*(?:[eE][−+-]?\d+)?")
# Header lines of the package that legitimately change on every regeneration.
PACKAGE_VOLATILE = re.compile(r"^- \*\*(Üretildiği commit|Üretim tarihi|Generated from "
                              r"commit|Generated at):\*\*")
TERMS_TR = re.compile(r"önceden|onceden|ön-kayıt|ön kayıt|on-kayit|keşifsel|kesifsel|"
                      r"KEŞİFSEL|KESIFSEL|post hoc|görüldükten sonra|görülmeden|"
                      r"sonradan", re.IGNORECASE)


def tokens(text):
    """Numbers, hashes and paths of `text`, in order, as (kind, value) pairs."""
    found = []
    for kind, rx in (("path", PATH_RE), ("hash", HASH_RE)):
        for m in rx.finditer(text):
            found.append((m.start(), kind, m.group()))
        text = rx.sub(lambda m: " " * len(m.group()), text)
    for m in NUM_RE.finditer(text):
        i, v = m.start(), m.group()
        j = i - 1 if i and text[i - 1] == "%" else i  # Turkish "−%13.1" = "−13.1%"
        prev = text[j - 1] if j else ""
        prev2 = text[j - 2] if j > 1 else ""
        if prev in "−+" or (prev == "-" and not prev2.isalnum()):
            i, v = j - 1, prev.replace("-", "−") + v
        found.append((i, "num", v))
    return [(k, v) for _, k, v in sorted(found)]


def git_show(ref, rel):
    return subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ROOT, check=True,
                          capture_output=True).stdout


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
def check_package(base, al):
    if al != "publication":
        fail("the number package exists only in the publication-aligned version")
    md = alignment.out(PACKAGE, al)
    csv = alignment.out(PACKAGE_CSV, al)
    old = git_show(base, md.relative_to(ROOT).as_posix()).decode("utf-8").splitlines()
    new = md.read_text(encoding="utf-8").splitlines()
    if len(old) != len(new):
        fail(f"package line count {len(new)} != {len(old)} at {base}")
    seq = {}
    for name, lines in (("old", old), ("new", new)):
        excl = [i for i, ln in enumerate(lines) if PACKAGE_VOLATILE.match(ln)]
        if len(excl) != 2:
            fail(f"{name} package: expected 2 volatile header lines, found {len(excl)}")
        seq[name] = [(i + 1, k, v) for i, ln in enumerate(lines) if i not in excl
                     for k, v in tokens(ln)]
    a, b = seq["old"], seq["new"]
    for j, (x, y) in enumerate(zip(a, b)):
        if x[1:] != y[1:]:
            fail(f"token {j + 1} differs: old line {x[0]} {x[1:]} vs new line {y[0]} "
                 f"{y[1:]}\n  old: {old[x[0] - 1][:160]}\n  new: {new[y[0] - 1][:160]}")
    if len(a) != len(b):
        fail(f"token count {len(b)} != {len(a)}")
    kinds = Counter(k for _, k, _ in a)
    print(f"[OK] package: {len(a)} tokens identical in value and order "
          f"({kinds['num']} numbers, {kinds['hash']} hashes, {kinds['path']} paths), "
          f"{len(new)} lines, base {base}")
    # Compare as git stores the file: .gitattributes normalizes line endings to LF, while
    # pandas writes CRLF on Windows. hash-object applies the same normalization.
    rel = csv.relative_to(ROOT).as_posix()
    blob = lambda *a: subprocess.run(["git", *a], cwd=ROOT, check=True, capture_output=True,
                                     text=True).stdout.strip()
    if blob("rev-parse", f"{base}:{rel}") != blob("hash-object", f"--path={rel}", rel):
        fail(f"{csv.name} is not byte-identical to {base} (as stored by git)")
    print(f"[OK] {csv.name}: byte-identical to {base} (as stored by git)")


# ---------------------------------------------------------------------------
def shape(line):
    s = line.strip()
    if not s:
        return "blank"
    if s.startswith("```"):
        return "fence"
    m = re.match(r"(#+) ", s)
    if m:
        return f"h{len(m.group(1))}"
    if s.startswith("|"):
        return f"table{len(re.findall(r'(?<!\\)\|', s))}"
    return "text"


def check_log():
    tr = LOG_TR.read_text(encoding="utf-8").splitlines()
    en_all = LOG_EN.read_text(encoding="utf-8").splitlines()
    if HEADER_END not in en_all:
        fail(f"{LOG_EN.name}: header end marker not found")
    off = en_all.index(HEADER_END) + 1
    en = en_all[off:]
    if len(en) != len(tr):
        fail(f"line count after the header {len(en)} != original {len(tr)}")
    bad = [(i + 1, shape(a), shape(b)) for i, (a, b) in enumerate(zip(tr, en))
           if shape(a) != shape(b)]
    for n, sa, sb in bad[:20]:
        print(f"  structure: original line {n} is {sa}, translation line {n + off} is {sb}")
    if bad:
        fail(f"{len(bad)} lines differ in structure")
    print(f"[OK] structure: {len(tr)} lines, line n of the original = line n + {off} of "
          "the translation")
    starts = [i for i, ln in enumerate(tr) if shape(ln) == "h1"] + [len(tr)]
    if starts[0] != 0:
        starts = [0] + starts
    problems = 0
    for s, e in zip(starts, starts[1:]):
        ca = Counter(tokens("\n".join(tr[s:e])))
        cb = Counter(tokens("\n".join(en[s:e])))
        miss, extra = ca - cb, cb - ca
        title = tr[s][:70]
        if miss or extra:
            problems += 1
            print(f"  [DIFF] lines {s + 1}-{e} ({title})")
            if miss:
                print(f"     missing in translation: {sorted(miss.elements())}")
            if extra:
                print(f"     extra in translation:   {sorted(extra.elements())}")
        else:
            print(f"  [OK] lines {s + 1}-{e}: {sum(ca.values())} tokens ({title})")
    if problems:
        fail(f"{problems} stages differ in numbers, hashes or paths")
    print("[OK] every stage: identical multiset of numbers, hashes and paths")


# ---------------------------------------------------------------------------
class _Blank(ast.NodeTransformer):
    """Blanks string constants. Inside an f-string only the formatted values are kept, in
    order and with their format specs (a spec such as :.3f decides a printed number), so
    moving a literal "%" from before a value to after it is not a difference but swapping
    two values is."""

    def visit_JoinedStr(self, node):
        vals = [self.visit_FormattedValue(v) for v in node.values
                if isinstance(v, ast.FormattedValue)]
        return ast.copy_location(ast.JoinedStr(values=vals), node)

    def visit_FormattedValue(self, node):
        node.value = self.visit(node.value)
        return node  # format_spec is left intact

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(""), node)
        return node


def _stmts(src):
    tree = _Blank().visit(ast.parse(src))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(
                node, (ast.FunctionDef, ast.ClassDef, ast.If, ast.For, ast.While,
                       ast.With, ast.Try)):
            out.append((node.lineno, ast.dump(node)))
    return ast.dump(tree), out


def check_ast(base, paths):
    diff_files = 0
    for p in paths:
        rel = Path(p).resolve().relative_to(ROOT).as_posix()
        old = git_show(base, rel).decode("utf-8")
        new = (ROOT / rel).read_text(encoding="utf-8")
        (to, so), (tn, sn) = _stmts(old), _stmts(new)
        if to == tn:
            print(f"[OK] {rel}: syntax tree identical with string constants blanked")
            continue
        diff_files += 1
        ca, cb = Counter(d for _, d in so), Counter(d for _, d in sn)
        print(f"[DIFF] {rel}: non-string changes in these statements (new line numbers):")
        for ln, d in sn:
            if cb[d] > ca[d]:
                print(f"     line {ln}")
        gone = [ln for ln, d in so if ca[d] > cb[d]]
        if gone:
            print(f"     replaced statements at old lines {gone}")
    if diff_files:
        fail(f"{diff_files} file(s) with changes beyond string constants")


# ---------------------------------------------------------------------------
def check_terms(base, al):
    tr = LOG_TR.read_text(encoding="utf-8").splitlines()
    en_all = LOG_EN.read_text(encoding="utf-8").splitlines()
    off = en_all.index(HEADER_END) + 1
    md = alignment.out(PACKAGE, al)
    old = git_show(base, md.relative_to(ROOT).as_posix()).decode("utf-8").splitlines()
    new = md.read_text(encoding="utf-8").splitlines()
    for name, a, b, o in (("experiment_log", tr, en_all, off), ("package", old, new, 0)):
        print(f"\n=== {name}: lines with a status term ===")
        for i, ln in enumerate(a):
            if TERMS_TR.search(ln):
                print(f"TR {i + 1}: {ln.strip()}\nEN {i + 1 + o}: {b[i + o].strip()}\n")
        print(f"=== {name}: English lines containing 'pre-regist' ===")
        for i, ln in enumerate(b[o:]):
            if "pre-regist" in ln.lower():
                print(f"EN {i + 1 + o}: {ln.strip()}\nTR {i + 1}: {a[i].strip()}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    for mode in ("package", "terms"):
        s = sub.add_parser(mode)
        alignment.add_argument(s)
        s.add_argument("--base", default=FREEZE_COMMIT)
    sub.add_parser("log")
    s = sub.add_parser("ast")
    s.add_argument("paths", nargs="+")
    s.add_argument("--base", default=FREEZE_COMMIT)
    args = ap.parse_args()
    if args.mode == "package":
        check_package(args.base, args.gpr_alignment)
    elif args.mode == "log":
        check_log()
    elif args.mode == "ast":
        check_ast(args.base, args.paths)
    else:
        check_terms(args.base, args.gpr_alignment)


if __name__ == "__main__":
    main()
