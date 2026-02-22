#!/usr/bin/env python3
"""
Quick regression tests for OPH Stage-5 spectrum tool.

These tests are *not* part of the prediction pipeline:
they only compare the Stage-5 outputs to reference (measured) values
to detect accidental code regressions.

Run:
  python3 test_particle_masses_stage5.py
"""
from __future__ import annotations

import math
import importlib.util
import sys

import pathlib
MODULE_PATH = str((pathlib.Path(__file__).resolve().parent / "particle_masses_stage5.py"))

def load_pm5():
    spec = importlib.util.spec_from_file_location("pm5", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses relies on the module being present in sys.modules during exec
    sys.modules["pm5"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def rel_err(pred: float, ref: float) -> float:
    return pred / ref - 1.0

def check(name: str, pred: float, ref: float, tol_rel: float):
    err = rel_err(pred, ref)
    ok = abs(err) <= tol_rel
    status = "OK" if ok else "FAIL"
    print(f"{status:4s} {name:22s} pred={pred:.10g}  ref={ref:.10g}  rel_err={err:+.3e}  tol={tol_rel:.1e}")
    if not ok:
        raise AssertionError(f"{name} outside tolerance: rel_err={err}, tol={tol_rel}")

def main():
    pm5 = load_pm5()
    pred = pm5.build_spectrum()

    # Reference values are *comparison only* (see PDG dict in module).
    PDG = pm5.PDG_REFERENCE

    # Gauge + EW
    check("alpha_em_inv", pred["alpha_em_inv_at_mZrun"], PDG["alpha_em_inv"], 5e-3)  # 0.5%
    check("sin2w",        pred["sin2w_at_mZrun"],        PDG["sin2w"],        5e-3)
    check("alpha_s(MZ)",  pred["alpha3_at_mZrun"],       PDG["alpha_s"],      1e-2)  # 1%
    check("MZ_pole",      pred["MZ_pole_stage3"],        PDG["MZ"],           1e-3)  # 0.1%
    check("MW(tree)",     pred["mW_run"],                PDG["MW"],           1e-3)

    # Higgs/top critical-surface channel
    check("mt_pole",      pred["crit_mt_pole"],          PDG["mt_pole"],      2e-2)  # 2%
    check("mH_tree",      pred["crit_mH_tree"],          PDG["mH"],           2e-2)

    # Charged leptons (texture channel)
    check("me",           pred["m_e"],                   PDG["me"],           1e-3)
    check("mmu",          pred["m_mu"],                  PDG["mmu"],          1e-3)
    check("mtau",         pred["m_tau"],                 PDG["mtau"],         1e-3)

    print("\nAll Stage-5 regression checks passed.")

if __name__ == "__main__":
    main()
