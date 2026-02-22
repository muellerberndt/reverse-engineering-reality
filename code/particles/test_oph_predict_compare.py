#!/usr/bin/env python3
"""Smoke tests for oph_predict_compare.py.

These tests validate:
- predictions do not depend on PDG_REF (no-cheat)
- neutrino masses depend on log_dim_H (capacity constant)
- hadron path executes and returns finite C constants on tiny demo settings

Run:
  python3 test_oph_predict_compare.py
"""

from __future__ import annotations

import math

import oph_predict_compare as pc


def assert_finite(x: float, name: str) -> None:
    if not (isinstance(x, (int, float)) and math.isfinite(float(x))):
        raise AssertionError(f"{name} not finite: {x}")


def main() -> None:
    P = pc.P_DEFAULT
    H1 = 1.0e122
    H2 = 1.0e120

    # Base prediction
    pred1 = pc.build_predictions(P, H1, with_hadrons=False, np_json=None, loops=4, hadron_profile='demo', hadron_overrides={})
    for k in ["mZ_run", "alpha3_at_mZrun", "m_e", "m_u"]:
        assert k in pred1
        assert_finite(float(pred1[k]), k)

    # Neutrinos must exist and scale with log_dim_H
    for k in ["m_nu_e", "m_nu_mu", "m_nu_tau", "rho_Lambda_1_4"]:
        assert k in pred1
        assert_finite(float(pred1[k]), k)

    pred2 = pc.build_predictions(P, H2, with_hadrons=False, np_json=None, loops=4, hadron_profile='demo', hadron_overrides={})
    # capacity larger => smaller rho^(1/4)
    if not (pred2["rho_Lambda_1_4"] > pred1["rho_Lambda_1_4"]):
        raise AssertionError("rho_Lambda_1_4 should increase when log_dim_H decreases")

    # No-cheat mutation test
    pc._assert_pdg_not_used(P, H1, loops=4)

    # Hadron demo path (ultra tiny settings to keep runtime low)
    predH = pc.build_predictions(
        P, H1,
        with_hadrons=True,
        np_json=None,
        loops=2,
        hadron_profile='demo',
        hadron_overrides={
            'L': 2, 'T': 4, 'therm': 0, 'sweeps': 1, 'every': 1,
            'kappas': [0.120],
            'beta1': 5.7, 'beta2': 6.0,
            'eps': 0.05,
        },
    )
    assert predH["np_error"] is None, predH["np_error"]
    C = predH["np_constants"]
    for k in ['C_p','C_n','C_pi','C_rho']:
        if k not in C:
            raise AssertionError(f"missing {k} in np_constants")
        assert_finite(float(C[k]), k)

    # Masses (GeV) should be finite (though not physical at this tiny lattice).
    for k in ['m_p','m_n','m_pi','m_rho']:
        if predH[k] is not None:
            assert_finite(float(predH[k]), k)

    print("OK: oph_predict_compare smoke tests passed")


if __name__ == '__main__':
    main()
