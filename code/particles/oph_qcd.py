#!/usr/bin/env python3
"""
oph_qcd.py (v2)
===============

Compact, dependency-free QCD running and Λ_MSbar extraction.

Key correction vs v1
--------------------
The RG-invariant scale Λ is *not* defined by integrating α all the way to ∞
with a truncated β(α) (that introduces a scheme-dependent constant error at ≥2 loops).

Instead, the standard MS-bar definition uses the subtracted integral at small coupling:

Let a := α_s/(4π), and (MS-bar):
    d a / d ln μ^2 = -β0 a^2 - β1 a^3 - β2 a^4 - β3 a^5 - ...

Then Λ is defined by:
    ln(μ^2/Λ^2) =  1/(β0 a)
                + (β1/β0^2) ln(β0 a)
                + ∫_0^a dx [ 1/β(x) + 1/(β0 x^2) - β1/(β0^2 x) ]      (*)

The bracketed integrand is finite at x→0 because the singular pieces are subtracted.
At 2 loops the integral term vanishes and (*) reproduces the familiar closed form:
    Λ = μ exp[-1/(2β0 a)] (β0 a)^(-β1/(2β0^2)).

This is the conventional Λ_MSbar used by PDG and perturbative QCD (no fitting; it is a
definition once the scheme and truncation order are chosen).

What this module provides
-------------------------
- 1..4 loop MS-bar β-coefficients for SU(3) QCD as functions of n_f.
- Λ_MSbar^(n_f) from α_s(μ) using the standard definition above.
- A monotone inversion α_s(μ) from Λ_MSbar (for self-tests / consistency).

No external data are used.
"""
from __future__ import annotations

import math
from typing import Tuple, Callable


# ----------------------------
# 4-loop MS-bar β coefficients
# ----------------------------
# Convention:
#   a = α_s/(4π)
#   d a / d ln μ^2 = - Σ_{k>=0} β_k a^{k+2}

def beta_coeffs_msbar(n_f: int) -> Tuple[float, float, float, float]:
    """Return (β0,β1,β2,β3) for SU(3) QCD in MS-bar with a=α/(4π)."""
    nf = float(n_f)
    z3 = 1.2020569031595942854  # ζ(3)
    b0 = 11.0 - (2.0/3.0)*nf
    b1 = 102.0 - (38.0/3.0)*nf
    b2 = 2857.0/2.0 - (5033.0/18.0)*nf + (325.0/54.0)*(nf**2)
    b3 = (
        149753.0/6.0 + 3564.0*z3
        + (-1078361.0/162.0 - (6508.0/27.0)*z3)*nf
        + (50065.0/162.0 + (6472.0/81.0)*z3)*(nf**2)
        + (1093.0/729.0)*(nf**3)
    )
    return (b0, b1, b2, b3)


def beta_a(a: float, n_f: int, loops: int = 4) -> float:
    """β(a) = d a / d ln μ^2 (negative for QCD at small a)."""
    if a <= 0:
        raise ValueError("a must be >0")
    b0, b1, b2, b3 = beta_coeffs_msbar(n_f)
    out = -b0 * a*a
    if loops >= 2:
        out += -b1 * a**3
    if loops >= 3:
        out += -b2 * a**4
    if loops >= 4:
        out += -b3 * a**5
    return out


# ----------------------------
# Tiny numerical integration
# ----------------------------

def _simpson(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Composite Simpson rule on [a,b] with even n."""
    if n % 2:
        raise ValueError("n must be even")
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        x = a + i*h
        s += f(x) * (4 if i % 2 else 2)
    return s * h / 3.0


# ----------------------------
# Λ_MSbar from α_s(μ)
# ----------------------------

def lambda_msbar_from_alpha(mu: float, alpha: float, n_f: int, loops: int = 4,
                            n: int = 20000) -> float:
    """
    Compute Λ_MSbar^(n_f) from α_s(μ), using the standard MS-bar definition.

    Steps:
      - a = α/(4π)
      - compute F(a) := ln(μ^2/Λ^2) via (*)
      - Λ = μ * exp(-F/2)

    n controls Simpson resolution for the finite integral; 20k panels is cheap and stable.
    """
    if mu <= 0:
        raise ValueError("mu must be >0")
    if not (1 <= loops <= 4):
        raise ValueError("loops must be 1..4")
    if alpha <= 0:
        raise ValueError("alpha must be >0")
    if n % 2:
        n += 1

    a = alpha / (4.0 * math.pi)
    b0, b1, *_ = beta_coeffs_msbar(n_f)

    # 1-loop + 2-loop closed pieces:
    F = 1.0/(b0*a)
    if loops >= 2:
        F += (b1/(b0*b0)) * math.log(b0*a)

    if loops >= 3:
        # Finite integral piece in (*). Start at ε to avoid 0/0 in floating arithmetic.
        eps = max(1e-8, a*1e-6)
        lo = eps
        hi = a

        def integrand(x: float) -> float:
            return (1.0/beta_a(x, n_f=n_f, loops=loops)
                    + 1.0/(b0*x*x)
                    - (b1/(b0*b0)) * (1.0/x))

        F += _simpson(integrand, lo, hi, n)

    # Λ from ln(μ^2/Λ^2)=F
    return mu * math.exp(-0.5 * F)


def alpha_from_lambda_msbar(mu: float, lam: float, n_f: int, loops: int = 4) -> float:
    """
    Invert Λ(α) numerically by bisection (monotone in α for perturbative α).
    """
    if mu <= 0 or lam <= 0:
        raise ValueError("mu, lam must be >0")
    lo, hi = 1e-4, 1.0
    f_lo = lambda_msbar_from_alpha(mu, lo, n_f=n_f, loops=loops) - lam
    f_hi = lambda_msbar_from_alpha(mu, hi, n_f=n_f, loops=loops) - lam
    # Expand hi if needed
    it = 0
    while f_lo * f_hi > 0 and it < 30:
        hi *= 1.5
        f_hi = lambda_msbar_from_alpha(mu, hi, n_f=n_f, loops=loops) - lam
        it += 1
    if f_lo * f_hi > 0:
        raise RuntimeError("Could not bracket α for inversion.")

    for _ in range(80):
        mid = 0.5*(lo+hi)
        f_mid = lambda_msbar_from_alpha(mu, mid, n_f=n_f, loops=loops) - lam
        if f_lo * f_mid > 0:
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return 0.5*(lo+hi)
