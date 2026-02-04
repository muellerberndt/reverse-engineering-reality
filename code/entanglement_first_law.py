#!/usr/bin/env python3
"""
Entanglement First Law Numerical Verification

This script verifies the entanglement first law:
    δS = δ⟨K⟩

where:
- S is the von Neumann entropy of the reduced state
- K = -log(ρ) is the modular Hamiltonian
- δ denotes variation under small perturbations

This identity is the "load-bearing" step in deriving Einstein's equation
from entanglement equilibrium in Observer Patch Holography.

Reference: STRING_THEORY.md, Appendix A
"""

import numpy as np

def random_density_matrix(dim):
    """Generate a random density matrix of given dimension."""
    # Generate random complex matrix
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    # Make it positive semidefinite via A†A
    rho = A @ A.conj().T
    # Normalize trace to 1
    rho = rho / np.trace(rho)
    return rho

def von_neumann_entropy(rho):
    """Compute von Neumann entropy S = -Tr(ρ log ρ)."""
    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(rho)
    # Filter out zeros/negatives (numerical noise)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    # Compute -Σ λ log λ
    return -np.sum(eigenvalues * np.log(eigenvalues))

def partial_trace_B(rho_AB, dim_A, dim_B):
    """Compute partial trace over subsystem B: ρ_A = Tr_B(ρ_AB)."""
    # Reshape to tensor with indices [a, b, a', b']
    rho_tensor = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
    # Trace over B indices (1 and 3)
    rho_A = np.trace(rho_tensor, axis1=1, axis2=3)
    return rho_A

def modular_hamiltonian(rho):
    """Compute modular Hamiltonian K = -log(ρ)."""
    # Use eigendecomposition for numerical stability
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    # Regularize small eigenvalues
    eigenvalues = np.maximum(eigenvalues, 1e-15)
    # K = -log(ρ) = V @ diag(-log(λ)) @ V†
    log_eigenvalues = -np.log(eigenvalues)
    K = eigenvectors @ np.diag(log_eigenvalues) @ eigenvectors.conj().T
    return K

def verify_first_law(dim_A=4, dim_B=4, epsilon_values=None, seed=42):
    """
    Verify the entanglement first law: δS_A ≈ δ⟨K_A⟩

    Parameters:
    -----------
    dim_A : int
        Dimension of subsystem A
    dim_B : int
        Dimension of subsystem B
    epsilon_values : list
        Perturbation strengths to test
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    results : list of tuples
        (epsilon, delta_S, delta_K, relative_error)
    """
    np.random.seed(seed)

    if epsilon_values is None:
        epsilon_values = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]

    dim_AB = dim_A * dim_B

    # Generate reference state ρ_AB(0)
    rho_AB_0 = random_density_matrix(dim_AB)

    # Compute reduced state ρ_A(0)
    rho_A_0 = partial_trace_B(rho_AB_0, dim_A, dim_B)

    # Compute modular Hamiltonian K_A = -log(ρ_A(0))
    K_A = modular_hamiltonian(rho_A_0)

    # Reference entropy
    S_A_0 = von_neumann_entropy(rho_A_0)

    # Generate perturbation direction (random Hermitian, traceless)
    delta_rho = random_density_matrix(dim_AB) - rho_AB_0
    delta_rho = delta_rho - np.trace(delta_rho) / dim_AB * np.eye(dim_AB)

    results = []

    for eps in epsilon_values:
        # Perturbed state: ρ_AB(ε) = ρ_AB(0) + ε * δρ (normalized)
        rho_AB_eps = rho_AB_0 + eps * delta_rho
        # Ensure positive semidefinite and normalized
        eigenvalues, eigenvectors = np.linalg.eigh(rho_AB_eps)
        eigenvalues = np.maximum(eigenvalues, 0)  # Project to positive
        rho_AB_eps = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
        rho_AB_eps = rho_AB_eps / np.trace(rho_AB_eps)

        # Compute reduced state ρ_A(ε)
        rho_A_eps = partial_trace_B(rho_AB_eps, dim_A, dim_B)

        # Compute ΔS_A = S(ρ_A(ε)) - S(ρ_A(0))
        S_A_eps = von_neumann_entropy(rho_A_eps)
        delta_S = S_A_eps - S_A_0

        # Compute Δ⟨K_A⟩ = Tr[(ρ_A(ε) - ρ_A(0)) K_A]
        delta_rho_A = rho_A_eps - rho_A_0
        delta_K_expectation = np.real(np.trace(delta_rho_A @ K_A))

        # Relative error
        if abs(delta_K_expectation) > 1e-15:
            rel_error = abs(delta_S - delta_K_expectation) / abs(delta_K_expectation)
        else:
            rel_error = float('inf')

        results.append((eps, delta_S, delta_K_expectation, rel_error))

    return results

def main():
    print("=" * 70)
    print("ENTANGLEMENT FIRST LAW VERIFICATION")
    print("δS_A ≈ δ⟨K_A⟩  (should agree to O(ε) as ε → 0)")
    print("=" * 70)
    print()
    print("Setup: Random bipartite state on 4×4 Hilbert space (2 qubits ⊗ 2 qubits)")
    print("       Perturb by ε, compare entropy change vs modular expectation change")
    print()

    results = verify_first_law()

    print(f"{'ε':<12} {'ΔS_A':<16} {'Δ⟨K_A⟩':<16} {'Rel. Error':<12}")
    print("-" * 56)

    for eps, delta_S, delta_K, rel_err in results:
        print(f"{eps:<12.6f} {delta_S:<16.8f} {delta_K:<16.8f} {rel_err:<12.4%}")

    print()
    print("-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    print("""
As ε → 0, the relative error goes to zero approximately linearly in ε.
This confirms the entanglement first law: δS = δ⟨K⟩ + O(ε²)

This identity is the key step in deriving Einstein's equation from
entanglement equilibrium in OPH:

  δS_bulk = δ⟨K⟩  →  (with geometric K)  →  ∫λ δ⟨T_kk⟩

Combined with area variation from Raychaudhuri:

  δ(A/4G) = -∫λ R_kk

Setting δS_gen = 0 gives:  R_kk = 8πG ⟨T_kk⟩  →  Einstein's equation
""")

    # Additional test with different dimensions
    print("=" * 70)
    print("SCALING CHECK: Different Hilbert space dimensions")
    print("=" * 70)
    print()

    for (dA, dB) in [(2, 2), (3, 3), (4, 4), (5, 5)]:
        results = verify_first_law(dim_A=dA, dim_B=dB, epsilon_values=[0.01, 0.001])
        print(f"dim_A={dA}, dim_B={dB}:")
        for eps, delta_S, delta_K, rel_err in results:
            print(f"  ε={eps:.4f}: Rel. Error = {rel_err:.4%}")
        print()

if __name__ == "__main__":
    main()
