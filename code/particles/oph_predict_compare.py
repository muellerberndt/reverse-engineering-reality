#!/usr/bin/env python3
"""oph_predict_compare.py

OPH predictor + (optional) internally-computed hadrons + PDG comparison.

Hard constraints ("no cheating")
------------------------------
- Physical inputs: ONLY the 2 OPH constants
    P          : pixel area constant
    log_dim_H  : screen-capacity constant
- PDG/reference values are used ONLY for reporting/validation. They must not affect
  any prediction path (a runtime mutation-test enforces this).

What is predicted
-----------------
- Full Stage-5 charged SM sector (masses + couplings) from OPH, given P.
- Neutrinos: a Stage-6 minimal model that *uses log_dim_H* (dark-energy scale).
- Optional hadrons: proton/pion ratios C_X := m_X/Λ^(3) from an internal collar
  lattice computation (quenched gauge + Wilson valence, continuum 2-point O(a^2)
  extrapolation). This is numerically expensive at high precision.

Precision reality check
-----------------------
This tool is internally computed and self-consistent, but *PDG-level* hadron
precision requires:
- larger volumes (finite-volume control),
- many configurations (statistics),
- more lattice spacings (continuum extrapolation),
- and ultimately unquenched fermions.

The code exposes only numerical-method knobs for the hadron part; these are not
physical fit parameters.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import particle_masses_stage5 as st5  # charged SM (Stage-5)
import oph_qcd  # 4-loop MSbar running
import oph_lattice_su3_quenched_v5 as lat  # internal collar lattice (v5)


# -----------------------------
# Defaults (can be overridden)
# -----------------------------
P_DEFAULT = float(getattr(st5, "P_DEFAULT", 1.63094))
LOG_DIM_H_DEFAULT = 1.0e122


# -----------------------------
# PDG reference table (validation-only)
# -----------------------------
PDG_REF: Dict[str, float] = {
    # gauge + Higgs
    "MZ_pole": 91.1876,
    "MW": 80.377,
    "mH": 125.25,
    # couplings at mZ (scheme caveats)
    "alpha_em_inv": 127.952,
    "sin2w": 0.23122,
    "alpha_s": 0.1179,
    # charged leptons
    "m_e": 0.00051099895,
    "m_mu": 0.1056583755,
    "m_tau": 1.77686,
    # top pole (orientation)
    "mt_pole": 172.7,
    # proton/neutron
    "m_p": 0.9382720813,
    "m_n": 0.9395654133,
}


# -----------------------------
# No-cheat enforcement
# -----------------------------

def _sig(pred: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        float(pred["mZ_run"]),
        float(pred["alpha3_at_mZrun"]),
        float(pred["v"]),
        float(pred.get("m_nu_tau", 0.0)),
    )


def _assert_pdg_not_used(P: float, log_dim_H: float, loops: int) -> None:
    """Mutate PDG_REF and confirm predictions are invariant."""
    base = build_predictions(P, log_dim_H, with_hadrons=False, np_json=None,
                             loops=loops, hadron_profile="demo", hadron_overrides={})
    s0 = _sig(base)

    # mutate the reporting table wildly (and restore after)
    _old = {k: PDG_REF.get(k) for k in ['MZ_pole','mH','alpha_s']}
    PDG_REF['MZ_pole'] = 999.0
    PDG_REF['mH'] = 999.0
    PDG_REF['alpha_s'] = 0.5

    again = build_predictions(P, log_dim_H, with_hadrons=False, np_json=None,
                              loops=loops, hadron_profile="demo", hadron_overrides={})
    s1 = _sig(again)

    # restore
    for k,v in _old.items():
        if v is None: PDG_REF.pop(k, None)
        else: PDG_REF[k] = v

    if any(abs(a - b) > 1e-12 for a, b in zip(s0, s1)):
        raise AssertionError("PDG leakage detected: predictions changed after PDG_REF mutation")


# -----------------------------
# QCD: step down Λ_MSbar
# -----------------------------

def step_down_lambda_msbar(mu: float, alpha_s: float, m_b: float, m_c: float, loops: int = 4) -> Dict[int, float]:
    """Compute Λ_MSbar^(nf) for n_f=5→4→3 using LO threshold matching."""
    Lam5 = oph_qcd.lambda_msbar_from_alpha(mu, alpha_s, n_f=5, loops=loops)
    a_mb_5 = oph_qcd.alpha_from_lambda_msbar(m_b, Lam5, n_f=5, loops=loops)
    Lam4 = oph_qcd.lambda_msbar_from_alpha(m_b, a_mb_5, n_f=4, loops=loops)
    a_mc_4 = oph_qcd.alpha_from_lambda_msbar(m_c, Lam4, n_f=4, loops=loops)
    Lam3 = oph_qcd.lambda_msbar_from_alpha(m_c, a_mc_4, n_f=3, loops=loops)
    return {5: Lam5, 4: Lam4, 3: Lam3}


# -----------------------------
# Neutrinos (Stage-6 minimal model)
# -----------------------------

def neutrino_masses_dark_energy(log_dim_H: float, epsilon: float) -> Dict[str, float]:
    """Minimal OPH neutrino model using the capacity constant.

    Uses the OPH/paper relation (in Planck units) for the observed CC scale:
      ρ_Λ ≈ (3/8) M_P^4 / log_dim_H

    Then sets a hierarchical triplet anchored at ρ_Λ^{1/4}:
      m_ν3 = ρ_Λ^{1/4}
      m_ν2 = ε m_ν3
      m_ν1 = ε^2 m_ν3

    This introduces **no empirical inputs**, only (P, log_dim_H) via ε and log_dim_H.
    """
    if log_dim_H <= 0:
        raise ValueError("log_dim_H must be >0")
    Mpl = float(getattr(st5, "E_PLANCK_GEV", 1.22089e19))
    rho = (3.0 / 8.0) * (Mpl ** 4) / float(log_dim_H)
    m0 = rho ** 0.25
    return {
        "m_nu_e": m0 * (epsilon ** 2),
        "m_nu_mu": m0 * epsilon,
        "m_nu_tau": m0,
        "rho_Lambda_1_4": m0,
    }


# -----------------------------
# Optional hadrons: internal collar computation of C_X
# -----------------------------

def load_np_constants(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "C" in obj and isinstance(obj["C"], dict):
        obj = obj["C"]
    out: Dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("C_"):
                out[k] = float(v)
    return out


def richardson_c0(C1: float, a1: float, C2: float, a2: float) -> float:
    num = C2 * (a1 * a1) - C1 * (a2 * a2)
    den = (a1 * a1) - (a2 * a2)
    return num / den


def _pick(res: Dict[str, Any], key_base: str) -> float:
    # Prefer chiral-extrapolated values when present.
    if f"{key_base}_chiral" in res and math.isfinite(float(res[f"{key_base}_chiral"])):
        return float(res[f"{key_base}_chiral"])
    return float(res.get(f"{key_base}_0", float("nan")))


def compute_np_constants_internal(profile: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Compute C_p and C_pi with a 2-point (β1,β2) continuum extrapolation."""
    profiles = {
        "demo":   dict(beta1=5.7, beta2=6.0, L=2, T=4, therm=1, sweeps=2, every=1, seed=0, kappas=[0.120, 0.125], nf=0, c=0.30, eps=0.05),
        "quick":  dict(beta1=5.7, beta2=6.1, L=4, T=8, therm=10, sweeps=30, every=5, seed=0, kappas=[0.120, 0.125], nf=0, c=0.30, eps=0.01),
        "serious":dict(beta1=5.8, beta2=6.2, L=6, T=12, therm=50, sweeps=200, every=10, seed=0, kappas=[0.120, 0.125], nf=0, c=0.30, eps=0.01),
    }
    if profile not in profiles:
        raise ValueError(f"unknown hadron profile: {profile}")
    p = {**profiles[profile], **overrides}

    r1 = lat.run(p["beta1"], p["L"], p["T"], p["therm"], p["sweeps"], p["every"], p["seed"],
                 kappas=list(p["kappas"]), nf=int(p["nf"]), c_flow=float(p["c"]), eps_flow=float(p["eps"]))
    r2 = lat.run(p["beta2"], p["L"], p["T"], p["therm"], p["sweeps"], p["every"], p["seed"] + 1,
                 kappas=list(p["kappas"]), nf=int(p["nf"]), c_flow=float(p["c"]), eps_flow=float(p["eps"]))

    a1 = float(r1["aLambda_msbar"])
    a2 = float(r2["aLambda_msbar"])

    Cp1, Cp2 = _pick(r1, 'C_p'), _pick(r2, 'C_p')
    Cpi1, Cpi2 = _pick(r1, 'C_pi'), _pick(r2, 'C_pi')
    Crho1, Crho2 = _pick(r1, 'C_rho'), _pick(r2, 'C_rho')

    C_cont = {
        'C_p': richardson_c0(Cp1, a1, Cp2, a2) if all(map(math.isfinite, [a1, a2, Cp1, Cp2])) else float('nan'),
        'C_pi': richardson_c0(Cpi1, a1, Cpi2, a2) if all(map(math.isfinite, [a1, a2, Cpi1, Cpi2])) else float('nan'),
        'C_rho': richardson_c0(Crho1, a1, Crho2, a2) if all(map(math.isfinite, [a1, a2, Crho1, Crho2])) else float('nan'),
    }

    C_out = {
        'C_p': float(C_cont['C_p']),
        'C_n': float(C_cont['C_p']),  # isospin-symmetric in this demo
        'C_pi': float(C_cont['C_pi']),
        'C_rho': float(C_cont['C_rho']),
    }

    meta = {
        "profile": profile,
        "params": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in p.items()},
        'point1': {'beta': r1['beta'], 'aLambda_msbar': a1, 'C_p': Cp1, 'C_pi': Cpi1, 'C_rho': Crho1},
        'point2': {'beta': r2['beta'], 'aLambda_msbar': a2, 'C_p': Cp2, 'C_pi': Cpi2, 'C_rho': Crho2},
        "C_continuum": C_cont,
    }
    return {"C": C_out, "meta": meta}


# -----------------------------
# PDG comparison
# -----------------------------

@dataclass(frozen=True)
class Row:
    name: str
    pred: Optional[float]
    ref: Optional[float]
    rel_err: Optional[float]
    status: str
    note: str = ""


def compare(pred: Dict[str, Any], ref: Dict[str, float]) -> List[Row]:
    rows: List[Row] = []

    def add(name: str, pred_val: Optional[float], ref_key: Optional[str], status: str, note: str = "") -> None:
        ref_val = ref.get(ref_key, None) if ref_key else None
        rel = None
        if pred_val is not None and ref_val is not None and ref_val != 0.0:
            rel = (pred_val - ref_val) / ref_val
        rows.append(Row(name, pred_val, ref_val, rel, status, note))

    add("MZ_pole", float(pred["MZ_pole_stage3"]), "MZ_pole", "pred", "Stage-3 Δρ pole map")
    add("MW", float(pred["mW_run"]), "MW", "pred", "tree/run at μ=mZ_run")
    add("alpha_em_inv(mZ)", float(pred["alpha_em_inv_at_mZrun"]), "alpha_em_inv", "pred", "scheme-sensitive")
    add("sin2w(mZ)", float(pred["sin2w_at_mZrun"]), "sin2w", "pred", "scheme-sensitive")
    add("alpha_s(mZ)", float(pred["alpha3_at_mZrun"]), "alpha_s", "pred", "scheme-sensitive")

    add("mt_pole", float(pred["crit_mt_pole"]), "mt_pole", "pred", "critical-surface channel")
    add("mH", float(pred["crit_mH_tree"]), "mH", "pred", "critical-surface channel")

    add("m_e", float(pred["m_e"]), "m_e", "pred", "Koide+Z6")
    add("m_mu", float(pred["m_mu"]), "m_mu", "pred", "Koide+Z6")
    add("m_tau", float(pred["m_tau"]), "m_tau", "pred", "Koide+Z6")

    # neutrinos: PDG refs are optional / typically bounds
    add("m_nu_e", float(pred["m_nu_e"]), "m_nu_e", "pred", "Stage-6 capacity model")
    add("m_nu_mu", float(pred["m_nu_mu"]), "m_nu_mu", "pred", "Stage-6 capacity model")
    add("m_nu_tau", float(pred["m_nu_tau"]), "m_nu_tau", "pred", "Stage-6 capacity model")

    for q in ["u", "d", "s", "c", "b", "t"]:
        k = f"m_{q}"
        add(k, float(pred[k]), k, "pred", "texture (scale/scheme caveats)")

    for h in ['m_p','m_n','m_pi','m_rho']:
        add(h, pred.get(h, None), h, "pred" if pred.get(h, None) is not None else "pending",
            "from C_X * Λ^(3)" if pred.get(h, None) is not None else "enable --with-hadrons")

    return rows


def print_table(rows: List[Row]) -> None:
    print("\n=== OPH predictions vs PDG (validation-only) ===\n")
    hdr = f"{'quantity':<16} {'pred':>14} {'PDG':>14} {'rel.err':>12}  status  note"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        p = "—" if r.pred is None else f"{r.pred:.6g}"
        q = "—" if r.ref is None else f"{r.ref:.6g}"
        e = "—" if r.rel_err is None else f"{r.rel_err:+.3e}"
        print(f"{r.name:<16} {p:>14} {q:>14} {e:>12}  {r.status:<7} {r.note}")


# -----------------------------
# Main orchestration
# -----------------------------

def build_predictions(P: float, log_dim_H: float, with_hadrons: bool, np_json: Optional[str],
                      loops: int, hadron_profile: str, hadron_overrides: Dict[str, Any]) -> Dict[str, Any]:

    # 1) Stage-5 charged sector
    s5 = st5.build_spectrum(P=float(P), log_dim_H=float(log_dim_H))

    # 2) Λ_MSbar^(3) from α_s(mZ_run) + OPH thresholds
    mu = float(s5["mZ_run"])
    a3 = float(s5["alpha3_at_mZrun"])
    mb = float(s5["m_b"])
    mc = float(s5["m_c"])
    Lam = step_down_lambda_msbar(mu, a3, m_b=mb, m_c=mc, loops=int(loops))
    Lam3 = float(Lam[3])

    pred: Dict[str, Any] = dict(s5)
    # Massless (in the SM): included for completeness
    pred['m_gamma'] = 0.0
    pred['m_gluon'] = 0.0
    pred['m_graviton'] = 0.0

    pred["P"] = float(P)
    pred["log_dim_H"] = float(log_dim_H)
    pred["Lambda_MSbar"] = Lam
    pred["Lambda_MSbar_3"] = Lam3

    # 3) Neutrinos (capacity model)
    eps = float(st5.defect_epsilon_Z6())
    pred.update(neutrino_masses_dark_energy(log_dim_H=float(log_dim_H), epsilon=eps))

    # 4) Optional hadrons
    if with_hadrons:
        try:
            np_block = {"C": load_np_constants(np_json)} if np_json else compute_np_constants_internal(hadron_profile, hadron_overrides)
            C = np_block["C"]
            pred["np_meta"] = np_block.get("meta")
            pred["np_error"] = None
        except Exception as e:
            C = {}
            pred["np_meta"] = None
            pred["np_error"] = str(e)

        pred["np_constants"] = C
        for name, key in [('m_p','C_p'), ('m_n','C_n'), ('m_pi','C_pi'), ('m_rho','C_rho')]:
            pred[name] = float(C[key]) * Lam3 if key in C and math.isfinite(float(C[key])) else None
    else:
        pred["np_constants"] = None
        pred["np_meta"] = None
        pred["np_error"] = None
        pred["m_p"] = pred["m_n"] = pred["m_pi"] = pred["m_rho"] = None

    return pred


def main() -> None:
    ap = argparse.ArgumentParser(description="OPH predictor + PDG comparison (no-cheat)")
    ap.add_argument("--P", type=float, default=P_DEFAULT, help="pixel area constant")
    ap.add_argument("--log-dim-H", type=float, default=LOG_DIM_H_DEFAULT, help="screen capacity log(dim H)")
    ap.add_argument("--loops", type=int, default=4, choices=[1, 2, 3, 4], help="loop order for Λ extraction")

    ap.add_argument("--with-hadrons", action="store_true", help="compute hadrons internally (slow)")
    ap.add_argument("--np-json", type=str, default=None, help="load precomputed C_X JSON (no PDG)")
    ap.add_argument("--hadron-profile", type=str, default="demo", choices=["demo", "quick", "serious"],
                    help="numerical settings for collar lattice")

    # Optional numeric overrides (NOT physical inputs):
    ap.add_argument("--hadron-L", type=int, default=None)
    ap.add_argument("--hadron-T", type=int, default=None)
    ap.add_argument("--hadron-sweeps", type=int, default=None)
    ap.add_argument("--hadron-therm", type=int, default=None)
    ap.add_argument("--hadron-every", type=int, default=None)
    ap.add_argument("--hadron-kappas", type=str, default=None)
    ap.add_argument("--hadron-beta1", type=float, default=None)
    ap.add_argument("--hadron-beta2", type=float, default=None)
    ap.add_argument("--hadron-c", type=float, default=None)
    ap.add_argument("--hadron-eps", type=float, default=None)
    ap.add_argument("--hadron-nf", type=int, default=None)

    ap.add_argument("--compare", action="store_true", help="print PDG comparison")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--pdg-json", type=str, default=None, help="extend/override PDG table for reporting")

    args = ap.parse_args()

    # PDG overrides (reporting only)
    ref = dict(PDG_REF)
    if args.pdg_json:
        with open(args.pdg_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    ref[k] = float(v)
                except Exception:
                    pass

    had_over: Dict[str, Any] = {}
    for k, v in {
        "L": args.hadron_L,
        "T": args.hadron_T,
        "sweeps": args.hadron_sweeps,
        "therm": args.hadron_therm,
        "every": args.hadron_every,
        "beta1": args.hadron_beta1,
        "beta2": args.hadron_beta2,
        "c": args.hadron_c,
        "eps": args.hadron_eps,
        "nf": args.hadron_nf,
    }.items():
        if v is not None:
            had_over[k] = v
    if args.hadron_kappas:
        had_over["kappas"] = [float(x) for x in args.hadron_kappas.split(",") if x.strip()]

    pred = build_predictions(
        P=float(args.P),
        log_dim_H=float(args.log_dim_H),
        with_hadrons=bool(args.with_hadrons),
        np_json=args.np_json,
        loops=int(args.loops),
        hadron_profile=str(args.hadron_profile),
        hadron_overrides=had_over,
    )

    # no-cheat test (fast: no hadrons)
    _assert_pdg_not_used(float(args.P), float(args.log_dim_H), int(args.loops))

    if args.compare:
        rows = compare(pred, ref)
        print_table(rows)

    if args.json:
        print(json.dumps({
            "inputs": {"P": float(args.P), "log_dim_H": float(args.log_dim_H), "loops": int(args.loops)},
            "predictions": pred,
            "pdg_reference_used_for_reporting": ref,
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()