#!/usr/bin/env python3
"""oph_lattice_su3_quenched_v5.py

Minimal SU(3) lattice collar (quenched gauge) with *internally computed* QCD scale.

What changed vs v4 (precision + correctness):
- **Euclidean gamma matrices**: projectors (1±γ_μ)/2 are now true projectors.
  (v4 used Minkowski-signature spatial gammas => invalid Wilson operator).
- **GF coupling normalization fixed**: energy density from average plaquette must be
  scaled by 12*N to match continuum E(t)=1/4 F^2 at tree level.
- **Finite-volume GF tree correction δ(c)** included (Fodor et al., 2012). citeturn5view0
- **GF → MS-bar matching** through NNLO coefficients k1,k2 for t^2<E(t)> (Harlander &
  Neumann, 2016). citeturn12view0
- **aΛ_MSbar** extracted with the *standard* 4-loop MS-bar definition (oph_qcd.py).

Important scope notes
---------------------
This is a correctness-focused, compact reference implementation.
- Gauge sector is **quenched** (n_f=0) by default. Hadron correlators use Wilson valence
  quarks, so hadron masses are *not* expected to match real-world QCD without dynamical
  fermions.
- No PDG inputs are used here (PDG is only for later comparison).

Outputs
-------
JSON with:
  - aLambda_msbar : dimensionless a*Λ_MSbar^{(n_f)} at μ=1/(cL)
  - am_pi, am_p   : lattice masses for the chosen κ (or two κ's)
  - C_pi, C_p     : ratios m/Λ (dimensionless)
  - optional chiral extrapolation from two κ values (linear in m_pi^2).
"""

from __future__ import annotations

import argparse
import json
import math
import numpy as np

from typing import Dict, Tuple, List

# Local import (no external data): 4-loop Λ_MSbar definition
import oph_qcd


# ----------------------------
# Euclidean gamma matrices
# ----------------------------

def _gamma_euclidean() -> List[np.ndarray]:
    """Return γ_0..γ_3 with {γ_μ,γ_ν}=2δ_μν and γ_μ^2=1."""
    I2 = np.eye(2, dtype=np.complex128)
    Z2 = np.zeros((2, 2), dtype=np.complex128)
    s1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    s3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # Start with Minkowski Dirac basis, then Wick-rotate spatial gammas: γ_k^E = i γ_k^M.
    g0M = np.block([[I2, Z2], [Z2, -I2]])
    g1M = np.block([[Z2, s1], [-s1, Z2]])
    g2M = np.block([[Z2, s2], [-s2, Z2]])
    g3M = np.block([[Z2, s3], [-s3, Z2]])

    g0 = g0M
    g1 = 1j * g1M
    g2 = 1j * g2M
    g3 = 1j * g3M
    return [g0, g1, g2, g3]


def _gamma5(g: List[np.ndarray]) -> np.ndarray:
    return g[0] @ g[1] @ g[2] @ g[3]


GAMMA = _gamma_euclidean()
G5 = _gamma5(GAMMA)
PPLUS = 0.5 * (np.eye(4, dtype=np.complex128) + GAMMA[0])  # positive parity projector


# ----------------------------
# SU(3) utilities
# ----------------------------

def su3_project(U: np.ndarray) -> np.ndarray:
    """Project a 3x3 complex matrix to SU(3) using polar decomposition."""
    X = U
    # Polar: X = (X (X†X)^(-1/2)) * det^{-1/3}
    H = X.conj().T @ X
    w, v = np.linalg.eigh(H)
    w = np.maximum(w, 1e-14)
    Hm12 = v @ np.diag(1.0 / np.sqrt(w)) @ v.conj().T
    Uu = X @ Hm12
    det = np.linalg.det(Uu)
    Uu = Uu / det ** (1.0 / 3.0)
    return Uu


def random_su3(rng: np.random.Generator) -> np.ndarray:
    M = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) / math.sqrt(2)
    Q, _ = np.linalg.qr(M)
    return su3_project(Q)


def dagger(U: np.ndarray) -> np.ndarray:
    """Hermitian conjugate on the last two indices (supports batching)."""
    return np.conjugate(np.swapaxes(U, -1, -2))


# ----------------------------
# Lattice geometry helpers
# ----------------------------

def shift(x: Tuple[int, int, int, int], mu: int, s: int, L: int, T: int) -> Tuple[int, int, int, int]:
    t, x1, x2, x3 = x
    if mu == 0:
        return ((t + s) % T, x1, x2, x3)
    if mu == 1:
        return (t, (x1 + s) % L, x2, x3)
    if mu == 2:
        return (t, x1, (x2 + s) % L, x3)
    return (t, x1, x2, (x3 + s) % L)


def plaquette_at(U: np.ndarray, x: Tuple[int, int, int, int], mu: int, nu: int, L: int, T: int) -> np.ndarray:
    t, x1, x2, x3 = x
    U1 = U[t, x1, x2, x3, mu]
    x_mu = shift(x, mu, +1, L, T)
    U2 = U[x_mu[0], x_mu[1], x_mu[2], x_mu[3], nu]
    x_nu = shift(x, nu, +1, L, T)
    U3 = dagger(U[x_nu[0], x_nu[1], x_nu[2], x_nu[3], mu])
    U4 = dagger(U[t, x1, x2, x3, nu])
    return U1 @ U2 @ U3 @ U4


def measure_plaquette(U: np.ndarray, L: int, T: int) -> float:
    tot = 0.0
    cnt = 0
    for t in range(T):
        for x1 in range(L):
            for x2 in range(L):
                for x3 in range(L):
                    x = (t, x1, x2, x3)
                    for mu in range(4):
                        for nu in range(mu + 1, 4):
                            P = plaquette_at(U, x, mu, nu, L, T)
                            tot += float(np.real(np.trace(P)) / 3.0)
                            cnt += 1
    return tot / max(cnt, 1)


# ----------------------------
# Pure-gauge update (simple Metropolis)
# ----------------------------

def staple(U: np.ndarray, x: Tuple[int, int, int, int], mu: int, L: int, T: int) -> np.ndarray:
    t, x1, x2, x3 = x
    S = np.zeros((3, 3), dtype=np.complex128)
    for nu in range(4):
        if nu == mu:
            continue
        # forward
        x_nu = shift(x, nu, +1, L, T)
        x_mu = shift(x, mu, +1, L, T)
        U_nu = U[t, x1, x2, x3, nu]
        U_mu_nu = U[x_nu[0], x_nu[1], x_nu[2], x_nu[3], mu]
        U_nu_mu = dagger(U[x_mu[0], x_mu[1], x_mu[2], x_mu[3], nu])
        S += U_nu @ U_mu_nu @ U_nu_mu

        # backward
        x_mnu = shift(x, nu, -1, L, T)
        U_nu_dag = dagger(U[x_mnu[0], x_mnu[1], x_mnu[2], x_mnu[3], nu])
        U_mu_mnu = U[x_mnu[0], x_mnu[1], x_mnu[2], x_mnu[3], mu]
        x_mnu_mu = shift(x_mnu, mu, +1, L, T)
        U_nu_mnu_mu = U[x_mnu_mu[0], x_mnu_mu[1], x_mnu_mu[2], x_mnu_mu[3], nu]
        S += U_nu_dag @ U_mu_mnu @ U_nu_mnu_mu
    return S


def sweep_metropolis(U: np.ndarray, beta: float, rng: np.random.Generator, L: int, T: int, step: float = 0.24) -> None:
    for t in range(T):
        for x1 in range(L):
            for x2 in range(L):
                for x3 in range(L):
                    x = (t, x1, x2, x3)
                    for mu in range(4):
                        S = staple(U, x, mu, L, T)
                        U_old = U[t, x1, x2, x3, mu]
                        R = np.eye(3, dtype=np.complex128) + step * (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
                        U_new = su3_project(R @ U_old)
                        dS = -beta / 3.0 * (
                            np.real(np.trace(U_new @ S)) - np.real(np.trace(U_old @ S))
                        )
                        if dS < 0 or rng.random() < math.exp(-dS):
                            U[t, x1, x2, x3, mu] = U_new


# ----------------------------
# Gradient flow + GF coupling
# ----------------------------

def flow_step(U: np.ndarray, eps: float, L: int, T: int) -> np.ndarray:
    """A crude Wilson flow step: U <- Proj( exp(-eps * staple_antiherm) U )."""
    out = U.copy()
    for t in range(T):
        for x1 in range(L):
            for x2 in range(L):
                for x3 in range(L):
                    x = (t, x1, x2, x3)
                    for mu in range(4):
                        S = staple(out, x, mu, L, T)
                        A = 0.5 * (S - dagger(S))
                        out[t, x1, x2, x3, mu] = su3_project((np.eye(3) - eps * A) @ out[t, x1, x2, x3, mu])
    return out


def theta3(q: float, n_terms: int = 50) -> float:
    """Jacobi theta_3(0,q) = 1 + 2 Σ_{n>=1} q^{n^2}."""
    s = 1.0
    for n in range(1, n_terms + 1):
        s += 2.0 * (q ** (n * n))
    return s


def delta_tree(c: float) -> float:
    """Tree-level finite-volume correction δ(c)=δ_a+δ_e (Fodor et al. 2012).""" 
    # δ_a = -(π^2/3) c^4 when t=(cL)^2/8.
    da = -(math.pi ** 2 / 3.0) * (c ** 4)
    q = math.exp(-1.0 / (c * c))
    th = theta3(q)
    de = (th ** 4) - 1.0
    return da + de


def k1k2_harlander(n_f: int, CA: float = 3.0, CF: float = 4.0 / 3.0, TR: float = 0.5) -> Tuple[float, float]:
    """NNLO coefficients for K_E(α)=1+k1 α + k2 α^2 at μ=1/sqrt(8t).""" 
    nf = float(n_f)
    # Eq. (38) in Harlander & Neumann 2016 (numbers already include analytic constants).
    k1 = 8.0 * (0.045741114 * CA + 0.001888798 * TR * nf)
    k2 = 8.0 * (
        -0.0136423 * (CA ** 2)
        + TR * nf * (0.006440134 * CF - 0.0086884 * CA)
        + (TR ** 2) * (nf ** 2) * 0.000936117
    )
    return k1, k2


def alpha_msbar_from_g2(g2_gf: float, n_f: int) -> float:
    """Invert g_GF^2 = 4π α (1 + k1 α + k2 α^2) for α."""
    if g2_gf <= 0:
        raise ValueError("g2_gf must be >0")
    k1, k2 = k1k2_harlander(n_f)
    y = g2_gf / (4.0 * math.pi)

    # Newton on f(a)=a(1+k1 a + k2 a^2)-y
    a = max(1e-8, y)
    for _ in range(50):
        f = a * (1.0 + k1 * a + k2 * a * a) - y
        df = (1.0 + 2.0 * k1 * a + 3.0 * k2 * a * a)
        step = f / df
        a2 = a - step
        if a2 <= 0:
            a2 = a * 0.5
        if abs(step) < 1e-14:
            break
        a = a2
    return float(a)


def gf_coupling_msbar_aLambda(U: np.ndarray, L: int, T: int, c: float, n_f: int,
                             eps_flow: float = 0.01, n_steps: int | None = None) -> Tuple[float, float, float]:
    """Return (g2_GF, α_MSbar(μ), aΛ_MSbar) from flowed plaquette at t=(cL)^2/8."""
    t_target = (c * L) ** 2 / 8.0
    if n_steps is None:
        n_steps = max(1, int(round(t_target / eps_flow)))
    eps = t_target / n_steps

    Uf = U
    for _ in range(n_steps):
        Uf = flow_step(Uf, eps, L, T)

    pl = measure_plaquette(Uf, L, T)

    # E ≈ 12*N*(1 - <P>) in lattice units (a=1). Here N=3.
    E_lat = 12.0 * 3.0 * max(0.0, (1.0 - pl))
    t2E = (t_target ** 2) * E_lat

    # g_GF^2 definition with δ(c) correction at tree level.
    N = 3
    norm = (128.0 * math.pi ** 2) / (3.0 * (N * N - 1.0))
    delta = delta_tree(c)
    g2 = norm * t2E / (1.0 + delta)

    # Convert to MS-bar α_s at μ=1/sqrt(8t)=1/(cL) in lattice units.
    alpha = alpha_msbar_from_g2(g2, n_f)
    mu_lat = 1.0 / (c * L)
    aLambda = oph_qcd.lambda_msbar_from_alpha(mu_lat, alpha, n_f=n_f, loops=4)
    return float(g2), float(alpha), float(aLambda)


# ----------------------------
# Wilson Dirac operator + CG
# ----------------------------

def apply_D(U: np.ndarray, psi: np.ndarray, kappa: float, L: int, T: int) -> np.ndarray:
    """Wilson Dirac operator (r=1, m0 absorbed into κ)."""
    out = psi.copy()
    for mu in range(4):
        g = GAMMA[mu]
        # forward hop
        psi_f = np.roll(psi, -1, axis=mu)  # axis 0 is time
        # multiply by link U_mu(x)
        U_mu = U[..., mu, :, :]
        tmp_f = np.einsum('...ab,...sb->...sa', U_mu, psi_f)
        out -= kappa * np.einsum('ij,...ja->...ia', (np.eye(4) - g), tmp_f)

        # backward hop
        psi_b = np.roll(psi, +1, axis=mu)
        U_b = np.roll(U_mu, +1, axis=mu)
        tmp_b = np.einsum('...ab,...sb->...sa', dagger(U_b), psi_b)
        out -= kappa * np.einsum('ij,...ja->...ia', (np.eye(4) + g), tmp_b)
    return out


def cg_solve(U: np.ndarray, kappa: float, src: np.ndarray, L: int, T: int,
             tol: float = 1e-10, maxiter: int = 500) -> np.ndarray:
    """CG on normal equations: (D†D) x = D† b (robust for Wilson)."""
    # Build operator A = D†D and rhs = D† src.
    def D(x: np.ndarray) -> np.ndarray:
        return apply_D(U, x, kappa, L, T)

    def Dh(x: np.ndarray) -> np.ndarray:
        # γ5-hermiticity: D† = γ5 D γ5
        x2 = np.einsum('ij,...ja->...ia', G5, x)
        y = D(x2)
        return np.einsum('ij,...ja->...ia', G5, y)

    b = Dh(src)
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = float(np.vdot(r, r).real)
    if rs == 0.0:
        return x

    for _ in range(maxiter):
        Ap = Dh(D(p))
        pAp = float(np.vdot(p, Ap).real)
        if pAp <= 0:
            break
        a = rs / pAp
        x = x + a * p
        r = r - a * Ap
        rs_new = float(np.vdot(r, r).real)
        if rs_new < tol * tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def prop_from_point(U: np.ndarray, kappa: float, L: int, T: int, x0=(0, 0, 0, 0),
                    tol: float = 1e-10, maxiter: int = 600) -> np.ndarray:
    """Point-to-all propagator S(x;0) as array [t,x,y,z, spin_sink, spin_src, col_sink, col_src]."""
    t0, x1_0, x2_0, x3_0 = x0
    S = np.zeros((T, L, L, L, 4, 4, 3, 3), dtype=np.complex128)
    for s0 in range(4):
        for c0 in range(3):
            src = np.zeros((T, L, L, L, 4, 3), dtype=np.complex128)
            src[t0, x1_0, x2_0, x3_0, s0, c0] = 1.0
            psi = cg_solve(U, kappa, src, L, T, tol=tol, maxiter=maxiter)
            S[..., :, s0, :, c0] = psi  # sink spin/color live in psi axes
    # Rearrange to [.., s_sink, s_src, c_sink, c_src]
    S = np.transpose(S, (0, 1, 2, 3, 4, 5, 6, 7))
    return S


# ----------------------------
# Hadron correlators (zero-momentum)
# ----------------------------

def pion_corr(S: np.ndarray, L: int, T: int) -> np.ndarray:
    """C_pi(t)=Σ_x Tr[S(x) S(x)†] (uses γ5-hermiticity)."""
    C = np.zeros(T, dtype=np.float64)
    for t in range(T):
        Q = S[t]  # [x,y,z, s,s0,c,c0]
        C[t] = float(np.vdot(Q, Q).real)
    return C


def rho_corr(S: np.ndarray, L: int, T: int) -> np.ndarray:
    """Vector-meson (ρ) correlator using γ_k bilinear and γ5-hermiticity.

    C_ρ(t)=Σ_x Σ_{k=1..3} Tr[ γ_k S(x,0) (γ_k γ5) S(x,0)† γ5 ], averaged over k.

    This differs from the pion correlator (which reduces to Tr[S S†]) and
    provides a nontrivial additional mass channel.
    """
    C = np.zeros(T, dtype=np.float64)
    for t in range(T):
        acc = 0.0 + 0.0j
        for x1 in range(L):
            for x2 in range(L):
                for x3 in range(L):
                    Q = S[t, x1, x2, x3]  # [s,s0,c,c0]
                    for k in (1, 2, 3):
                        gk = GAMMA[k]
                        gkg5 = gk @ G5
                        for a in range(3):
                            for b in range(3):
                                Sab = Q[:, :, a, b]
                                tmp = gk @ Sab @ gkg5 @ Sab.conj().T @ G5
                                acc += np.trace(tmp)
        C[t] = float((acc.real) / 3.0)
    return C

def proton_corr_direct(S: np.ndarray, L: int, T: int) -> np.ndarray:
    """Very small but structurally correct local proton correlator (direct term only).

    N_α(x)=ε^{abc}(u^a(x)^T Cγ5 d^b(x)) u^c_α(x).

    We compute C(t)=Σ_x Tr[P+ G(x)], with
      G_{αα'}(x)=ε^{abc}ε^{a'b'c'} S^{cc'}_{αα'}(x)
                * Tr[ (Cγ5) S^{bb'}(x) (γ5 C) (S^{aa'}(x))^T ]

    Exchange term (from identical u quarks) is omitted for compactness.
    """
    C = np.zeros(T, dtype=np.float64)
    Cmat = (GAMMA[2] @ GAMMA[0])  # Euclidean charge conjugation choice
    Gamma = Cmat @ G5
    GammaB = G5 @ Cmat

    # Levi-Civita nonzero permutations for SU(3)
    perms = [
        (0, 1, 2, +1), (1, 2, 0, +1), (2, 0, 1, +1),
        (0, 2, 1, -1), (2, 1, 0, -1), (1, 0, 2, -1),
    ]

    for t in range(T):
        acc = 0.0 + 0.0j
        for x1 in range(L):
            for x2 in range(L):
                for x3 in range(L):
                    Q = S[t, x1, x2, x3]  # [s,s0,c,c0]
                    # Sum over sink/source color epsilons
                    for a, b, c, sgn1 in perms:
                        for ap, bp, cp, sgn2 in perms:
                            Sa = Q[:, :, a, ap]
                            Sb = Q[:, :, b, bp]
                            Sc = Q[:, :, c, cp]
                            D = np.trace(Gamma @ Sb @ GammaB @ Sa.T)
                            G = Sc * D
                            acc += (sgn1 * sgn2) * np.trace(PPLUS @ G)
        C[t] = float(acc.real)
    return C


def effective_mass_log(C: np.ndarray) -> float:
    """Crude effective mass from log ratios.

    For noisy baryon correlators the sign may fluctuate; we therefore use |Re C|
    for a robust mass estimate (this affects only the overlap/sign, not the
    exponential falloff when a plateau exists).
    """
    T = len(C)
    vals = []
    for t in range(1, T - 2):
        ct = float(abs(np.real(C[t])))
        ct1 = float(abs(np.real(C[t + 1])))
        if ct > 0 and ct1 > 0:
            vals.append(math.log(ct / ct1))
    if not vals:
        return float('nan')
    vals.sort()
    return float(vals[len(vals) // 2])


def mass_from_corr(C: np.ndarray) -> float:
    m = effective_mass_log(C)
    return m


# ----------------------------
# End-to-end run
# ----------------------------

def run(beta: float, L: int, T: int, therm: int, sweeps: int, every: int, seed: int,
        kappas: List[float], nf: int, c_flow: float, eps_flow: float) -> Dict[str, float]:

    rng = np.random.default_rng(seed)
    U = np.empty((T, L, L, L, 4, 3, 3), dtype=np.complex128)
    for t in range(T):
        for x1 in range(L):
            for x2 in range(L):
                for x3 in range(L):
                    for mu in range(4):
                        U[t, x1, x2, x3, mu] = np.eye(3, dtype=np.complex128)

    # Thermalize
    for _ in range(therm):
        sweep_metropolis(U, beta, rng, L, T)

    # Accumulators
    aL_list: List[float] = []
    g2_list: List[float] = []
    a_list: List[float] = []

    corr_pi = [np.zeros(T, dtype=np.float64) for _ in kappas]
    corr_p = [np.zeros(T, dtype=np.float64) for _ in kappas]
    corr_rho = [np.zeros(T, dtype=np.float64) for _ in kappas]
    n_meas = 0

    for sw in range(1, sweeps + 1):
        sweep_metropolis(U, beta, rng, L, T)
        if sw % every != 0:
            continue

        g2, alpha, aL = gf_coupling_msbar_aLambda(U, L, T, c=c_flow, n_f=nf, eps_flow=eps_flow)
        aL_list.append(aL)
        g2_list.append(g2)
        a_list.append(alpha)

        # Hadron correlators at each κ on this gauge field
        for i, kappa in enumerate(kappas):
            S = prop_from_point(U, kappa, L, T)
            corr_pi[i] += pion_corr(S, L, T)
            corr_p[i] += proton_corr_direct(S, L, T)
            corr_rho[i] += rho_corr(S, L, T)
        n_meas += 1

    if n_meas == 0:
        raise RuntimeError("No measurements taken; increase sweeps or reduce every.")

    # Ensemble averages
    aLambda = float(np.mean(aL_list))
    g2_GF = float(np.mean(g2_list))
    alpha_ms = float(np.mean(a_list))

    out: Dict[str, float] = {
        "beta": float(beta),
        "L": float(L),
        "T": float(T),
        "nf": float(nf),
        "c_flow": float(c_flow),
        "n_meas": float(n_meas),
        "g2_GF": g2_GF,
        "alpha_msbar_at_mu": alpha_ms,
        "mu_lat": float(1.0 / (c_flow * L)),
        "aLambda_msbar": aLambda,
    }

    am_pi: List[float] = []
    am_p: List[float] = []
    am_rho: List[float] = []

    for i, kappa in enumerate(kappas):
        Cpi = corr_pi[i] / n_meas
        Cp = corr_p[i] / n_meas
        Cr = corr_rho[i] / n_meas
        mpi = mass_from_corr(Cpi)
        mp = mass_from_corr(Cp)
        mr = mass_from_corr(Cr)
        am_pi.append(float(mpi))
        am_p.append(float(mp))
        am_rho.append(float(mr))
        out[f"kappa_{i}"] = float(kappa)
        out[f"am_pi_{i}"] = float(mpi)
        out[f"am_p_{i}"] = float(mp)
        out[f"am_rho_{i}"] = float(mr)
        out[f"C_pi_{i}"] = float(mpi / aLambda) if (aLambda > 0 and math.isfinite(mpi)) else float('nan')
        out[f"C_p_{i}"] = float(mp / aLambda) if (aLambda > 0 and math.isfinite(mp)) else float('nan')
        out[f"C_rho_{i}"] = float(mr / aLambda) if (aLambda > 0 and math.isfinite(mr)) else float('nan')

    # If two kappas: linear chiral extrapolation mp = m0 + b m_pi^2.
    if len(kappas) >= 2 and all(math.isfinite(x) for x in am_pi[:2] + am_p[:2]):
        x1, x2 = am_pi[0] ** 2, am_pi[1] ** 2
        y1, y2 = am_p[0], am_p[1]
        if abs(x2 - x1) > 1e-12:
            m0 = (y1 * x2 - y2 * x1) / (x2 - x1)
            out["am_p_chiral"] = float(m0)
            out["C_p_chiral"] = float(m0 / aLambda)

            # optional rho chiral extrapolation: m_rho = m0 + b m_pi^2
            y1r, y2r = am_rho[0], am_rho[1]
            m0r = (y1r * x2 - y2r * x1) / (x2 - x1)
            out["am_rho_chiral"] = float(m0r)
            out["C_rho_chiral"] = float(m0r / aLambda)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--beta', type=float, default=5.7)
    ap.add_argument('--L', type=int, default=4)
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--therm', type=int, default=10)
    ap.add_argument('--sweeps', type=int, default=30)
    ap.add_argument('--every', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--kappas', type=str, default='0.120,0.125', help='comma-separated κ values')
    ap.add_argument('--nf', type=int, default=0, help='active flavours for MS-bar β (0=quenched)')
    ap.add_argument('--c', dest='c_flow', type=float, default=0.3, help='flow parameter in t=(cL)^2/8')
    ap.add_argument('--eps', dest='eps_flow', type=float, default=0.01, help='flow integrator step size')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    kappas = [float(x) for x in args.kappas.split(',') if x.strip()]
    out = run(args.beta, args.L, args.T, args.therm, args.sweeps, args.every, args.seed,
              kappas=kappas, nf=args.nf, c_flow=args.c_flow, eps_flow=args.eps_flow)

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print("=== OPH collar lattice (v5) ===")
        for k in sorted(out.keys()):
            print(f"{k}: {out[k]}")


if __name__ == '__main__':
    main()