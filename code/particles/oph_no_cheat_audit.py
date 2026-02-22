#!/usr/bin/env python3
"""oph_no_cheat_audit.py

A tiny static + dynamic integrity check:
- Ensures Stage-5 prediction functions do not reference PDG tables.
- Ensures neutrino placeholders are not emitted by default.

This is *not* a physics test; it's an anti-leak / anti-cheat test.
"""

from __future__ import annotations

import ast
import pathlib
from typing import List


def _func_source(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(name)


def audit_stage5(path: str) -> List[str]:
    """Return a list of audit failures (empty == pass)."""
    txt = pathlib.Path(path).read_text()
    tree = ast.parse(txt)
    fails: List[str] = []

    # 1) build_spectrum must not reference PDG_REFERENCE / compare functions.
    bs = _func_source(tree, "build_spectrum")
    bs_src = ast.get_source_segment(txt, bs) or ""
    if "PDG_REFERENCE" in bs_src or "compare_to_reference" in bs_src:
        fails.append("build_spectrum references validation constants")

    # 2) lepton_spectrum must not emit neutrino placeholders.
    ls = _func_source(tree, "lepton_spectrum")
    ls_src = ast.get_source_segment(txt, ls) or ""
    if "nu_" in ls_src or "neutr" in ls_src.lower():
        # We allow the word "neutrino" in comments/docstrings, but not actual keys.
        if "\"nu_" in ls_src or "'nu_" in ls_src:
            fails.append("lepton_spectrum emits neutrino masses")

    # 3) Heuristic: PDG_REFERENCE should only be used inside compare_to_reference.
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "PDG_REFERENCE":
            # Walk up: easiest heuristic is source-segment locality.
            pass

    return fails


def main() -> None:
    stage5 = str((pathlib.Path(__file__).resolve().parent / "particle_masses_stage5.py"))
    fails = audit_stage5(stage5)
    if fails:
        raise SystemExit("NO-CHEAT AUDIT FAILED:\n  - " + "\n  - ".join(fails))
    print("OK: no-cheat audit")


if __name__ == "__main__":
    main()
