"""Brand guard: find user-visible brand strings the rebrand missed.

Called by ``hack/check-brand.sh``. This is a separate program rather than a grep
pipeline because the distinction that matters cannot be made with grep: a brand
name in a **string literal** reaches a user (a log line, an error message, a CLI
``--help`` entry, an HTML fragment in an API response), while the same name in a
**comment or docstring** reaches only a developer reading the source.

Grepping both produces a list dominated by docstrings, and the person running it
learns to ignore the output — which is how a real leak survives. So this walks
tokens, keeps only strings, and drops docstrings by position.

Comments are deliberately *not* reported: leaving "GPUStack" in a comment that
explains upstream behaviour is often more accurate than renaming it, and the
upstream name in a comment is not a use of the mark in commerce.
"""

import ast
import io
import sys
import tokenize
from pathlib import Path
from typing import List, Set, Tuple

BRAND = "GPUStack"

# Substrings that make an occurrence legitimate inside a string literal:
#   * compatibility identifiers that must keep the upstream spelling;
#   * attribution, which Apache-2.0 §4(c) requires be retained verbatim;
#   * Kubernetes API groups and label keys, which are a contract with the
#     operator's CRDs rather than branding;
#   * the name of the ``gpustack-operator`` component, which is a real binary
#     and chart dependency — renaming it in a log line would misdescribe it.
ALLOWED = (
    "X-GPUStack",
    "GPUSTACK_",
    "gpustack:",
    "gpustack.ai",
    "GPUStack authors",
    "GPUStack operator",
    "GPUStack Operator",
    "Based on",
    "derivative",
    "upstream",
    "Apache",
)

# Directories whose contents are historical or generated.
SKIP_DIRS = {"migrations", "__pycache__", "ui", "third_party"}

# PEP 701 (Python 3.12) stopped emitting f-strings as a single STRING token and
# splits them into FSTRING_START / FSTRING_MIDDLE / FSTRING_END. A checker that
# only looks at STRING therefore passes on 3.12 while missing every brand string
# that sits inside an f-string — which is most of them, since that is how they
# get the product name interpolated. This was observed for real: the same tree
# reported 0 findings on 3.12 and 5 on 3.9.
#
# Handle both tokenizations so the result does not depend on which interpreter
# CI happens to have.
STRING_TOKEN_TYPES = {tokenize.STRING}
for _name in ("FSTRING_MIDDLE",):
    if hasattr(tokenize, _name):
        STRING_TOKEN_TYPES.add(getattr(tokenize, _name))
_IS_FSTRING_MIDDLE = hasattr(tokenize, "FSTRING_MIDDLE")


def _docstring_nodes(tree: ast.AST) -> Set[int]:
    """Line numbers of every docstring, so they can be excluded."""
    lines: Set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and (isinstance(first.value.value, str))
        ):
            # A docstring can span many lines; exclude all of them.
            for line in range(first.lineno, (first.end_lineno or first.lineno) + 1):
                lines.add(line)
    return lines


def scan_file(path: Path) -> List[Tuple[int, str]]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [(e.lineno or 0, f"<syntax error: {e.msg}>")]
    docstring_lines = _docstring_nodes(tree)

    findings: List[Tuple[int, str]] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type not in STRING_TOKEN_TYPES:
                continue
            # An FSTRING_MIDDLE is by definition inside an f-string, carries no
            # quote prefix, and holds only the literal text between
            # placeholders.
            is_fstring_part = _IS_FSTRING_MIDDLE and tok.type == tokenize.FSTRING_MIDDLE
            # Documentation is out of scope for both checks below: it reaches a
            # developer reading the source, not a user. Attribute docstrings (a
            # string following an assignment) are not AST docstrings, so the
            # position check alone cannot catch them.
            if tok.start[0] in docstring_lines:
                continue
            if not is_fstring_part and tok.string.startswith(('"""', "'''")):
                continue
            # A branding placeholder in a string that is *not* an f-string is the
            # worst failure mode here: it compiles, passes review, and renders
            # "{branding.PRODUCT_NAME}" verbatim to a customer. Cheap to catch,
            # impossible to notice otherwise.
            if (
                not is_fstring_part
                and "branding." in tok.string
                and not tok.string.lstrip().startswith(
                    ("f", "F", "rf", "fr", "Rf", "fR")
                )
            ):
                findings.append(
                    (
                        tok.start[0],
                        f"<branding placeholder in a non-f-string: {tok.string.strip()[:110]}>",
                    )
                )
                continue
            if BRAND not in tok.string:
                continue
            if any(marker in tok.string for marker in ALLOWED):
                continue
            findings.append((tok.start[0], tok.string.strip()[:150]))
    except (tokenize.TokenError, IndentationError):
        pass
    return findings


def main(root: str) -> int:
    base = Path(root)
    total = 0
    for path in sorted(base.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name == "branding.py":
            # The single source of truth is allowed to name the upstream project.
            continue
        for lineno, text in scan_file(path):
            print(f"  {path.relative_to(base)}:{lineno}: {text}")
            total += 1
    if total:
        print(
            f"\n{total} finding(s). A brand string belongs in gpustack/branding.py; "
            "a\nbranding placeholder outside an f-string is a bug that renders "
            "verbatim.\nAdd a line to ALLOWED only if it is a compatibility "
            "identifier,\nattribution, a Kubernetes API group, or the name of an "
            "upstream component."
        )
        return 1
    print("  no user-visible brand strings outside gpustack/branding.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "gpustack"))
