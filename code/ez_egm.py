"""
Endogenous Grid Method (EGM) for Epstein-Zin Preferences.

This module implements the EZ-EGM algorithm described in the paper. The key insight
is that the transformation W = V^{1-ρ} converts the Epstein-Zin recursion into a
form where EGM applies directly.

Four operators: ezegm, ezvfi, bellman, howard
Two solvers: solve_ezegm, solve_ezvfi

Grid structure (CRITICAL):
    - m_grid: 1D array of cash-on-hand values, SAME for all z states
    - z_grid: 1D array of income states
    - c[i, j] = c(m_grid[i], z_grid[j]) - proper cross product
    - V[i, j] = V(m_grid[i], z_grid[j]) - proper cross product

This is the correct structure: c and V are defined on m_grid × z_grid.

Key features:
    1. Epstein-Zin preferences with separated risk aversion and EIS
    2. V stored and interpolated (not W) - V has better boundary behavior
    3. W = V^{1-ρ} computed only internally for Euler equation in EGM
    4. Both EGM and VFI precompute μ(a,z) on the a-grid (key optimization)
    5. Howard acceleration with convergence-based stopping

Implementation:
    - Store and interpolate V directly (V→0 as m→0 is a natural boundary)
    - EGM uses a_grid internally, outputs on m_grid
    - VFI works directly on m_grid

Certainty equivalent μ formulas (both are mathematically equivalent):
    - EGM (W-space): μ = (E[W^θ])^{1/θ} where θ = (1-γ)/(1-ρ)
    - VFI (V-space): μ = (E[V^{1-γ}])^{1/(1-γ)}
    Since W = V^{1-ρ}, we have W^θ = V^{(1-ρ)θ} = V^{1-γ}, so both give the same μ.

Notation:
    - m = cash-on-hand (state), m_grid is 1D
    - z = income state, z_grid is 1D
    - a = end-of-period assets (internal to EGM)
    - c = consumption, c[i,j] = c(m_grid[i], z_grid[j])
    - V = value function, V[i,j] = V(m_grid[i], z_grid[j])
    - Transition: m' = Ra + z'

Extrapolation strategy:
    - Below grid (m < m_min): Constraint binds, c = m, V = (β*μ)^{1/(1-ρ)}
    - Above grid (m > m_max): Linear extrapolation
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import quantecon as qe

from utils import BISECTION_TOL, EPS, golden_section_search

jax.config.update("jax_enable_x64", True)


# =============================================================================
# V ↔ W Conversion Helpers
# =============================================================================


def _V_to_W(V: jnp.ndarray, ρ: float) -> jnp.ndarray:
    """Convert value function V to transformed value W = V^{1-ρ}."""
    V_safe = jnp.maximum(V, EPS)
    return V_safe ** (1 - ρ)


def _W_to_V(W: jnp.ndarray, ρ: float) -> jnp.ndarray:
    """Convert transformed value W back to V = W^{1/(1-ρ)}."""
    W_safe = jnp.maximum(W, EPS)
    return W_safe ** (1 / (1 - ρ))


# =============================================================================
# Model Definition
# =============================================================================


class EZModel(NamedTuple):
    """
    Epstein-Zin preferences model primitives.

    Parameters:
        β: Discount factor
        R: Gross interest rate
        ρ: Governs EIS (EIS = 1/ρ)
        γ: Risk aversion
        θ: Auxiliary parameter θ = (1-γ)/(1-ρ)
        m_grid: Cash-on-hand grid (1D, same for all z states)
        a_grid: Asset grid (1D, used internally by EGM)
        z_grid: Income grid (1D, state values)
        Q: Markov transition matrix for income
    """

    β: float
    R: float
    ρ: float  # EIS = 1/ρ
    γ: float  # Risk aversion
    θ: float  # (1-γ)/(1-ρ)
    m_grid: jnp.ndarray  # 1D, shape (n_m,) - SAME for all z
    a_grid: jnp.ndarray  # 1D, shape (n_a,) - for EGM internal use
    z_grid: jnp.ndarray  # 1D, shape (n_z,)
    Q: jnp.ndarray  # shape (n_z, n_z)


def create_ez_model(
    β: float = 0.96,
    R: float = 1.02,
    ρ: float = 2 / 3,  # EIS = 1.5 (Bansal-Yaron 2004)
    γ: float = 10.0,  # Risk aversion (Bansal-Yaron 2004)
    m_min: float = 0.0,
    m_max: float = 20.0,
    m_size: int = 100,
    a_size: int = 100,  # For EGM internal grid
    z_rho: float = 0.95,  # Income persistence (Storesletten et al. 2004)
    z_sigma: float = 0.1,
    z_size: int = 7,
    grid_type: str = "exp",  # "uniform", "exp", or "double_exp"
) -> EZModel:
    """
    Create an Epstein-Zin model.

    Default parameters follow standard calibrations:
        - γ = 10, EIS = 1.5: Bansal & Yaron (2004)
        - β = 0.96, R = 1.02: Standard annual calibration
        - z_rho = 0.95: Storesletten, Telmer & Yaron (2004)

    With γ > ρ, agent prefers early resolution of uncertainty.
    θ = (1-10)/(1-2/3) = -27, consistent with early resolution preference.

    Grid types:
        - "uniform": Evenly spaced
        - "exp": Exponential spacing (more points near zero)
        - "double_exp": Double-exponential (even more concentration near zero)

    Args:
        m_min: Minimum cash-on-hand (0 = borrowing constraint)
        m_max: Maximum cash-on-hand
        m_size: Number of m grid points
        a_size: Number of asset grid points (for EGM internal use)

    Returns:
        EZModel with m_grid (1D) and a_grid (1D)

    Raises:
        ValueError: If ρ = 1 (log utility requires special handling not implemented)
    """
    # Validate parameters
    if abs(ρ - 1.0) < 1e-10:
        raise ValueError(
            "ρ = 1 (log utility / unit EIS) is not supported. "
            "This case requires special handling of the Epstein-Zin recursion."
        )

    # Compute auxiliary parameter
    θ = (1 - γ) / (1 - ρ)

    # Income process first (needed to compute m_max)
    mc = qe.tauchen(n=z_size, rho=z_rho, sigma=z_sigma)
    z_grid = jnp.exp(mc.state_values)
    Q = jax.device_put(mc.P)
    z_max = float(z_grid[-1])

    # m_grid: extends to R * m_max + z_max so we cover next-period states
    m_grid_max = R * m_max + z_max

    # a_grid: for EGM internal use (end-of-period assets)
    # Must extend to m_grid_max since max savings is a = m - c ≈ m at high m
    # This ensures VFI never extrapolates μ beyond the grid
    a_max = m_grid_max
    if grid_type == "uniform":
        a_grid = jnp.linspace(0.0, a_max, a_size)
    elif grid_type == "exp":
        x = np.linspace(0, np.log(a_max + 1), a_size)
        a_grid = jnp.array(np.exp(x) - 1)
    elif grid_type == "double_exp":
        x = np.linspace(0, 1, a_size)
        a_grid = jnp.array(a_max * (np.exp(5 * x) - 1) / (np.exp(5) - 1))
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")
    if grid_type == "uniform":
        m_grid = jnp.linspace(m_min, m_grid_max, m_size)
    elif grid_type == "exp":
        x = np.linspace(0, np.log(m_grid_max - m_min + 1), m_size)
        m_grid = jnp.array(m_min + np.exp(x) - 1)
    elif grid_type == "double_exp":
        x = np.linspace(0, 1, m_size)
        m_grid = jnp.array(
            m_min + (m_grid_max - m_min) * (np.exp(5 * x) - 1) / (np.exp(5) - 1)
        )
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")

    return EZModel(β, R, ρ, γ, θ, m_grid, a_grid, z_grid, Q)


# =============================================================================
# Four Operators: ezegm, ezvfi, bellman, howard
# =============================================================================


@jax.jit
def ezegm(
    c_in: jnp.ndarray,
    V_in: jnp.ndarray,
    model: EZModel,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    One step of EZ-EGM (optimized).

    Takes c, V on rectangular (m_grid, z_grid), returns c, V on same grid.
    The endogenous grid is purely internal to this operator.

    Key optimizations:
    1. Batch interpolation to avoid vmap overhead
    2. Direct μ computation avoiding extra interpolation pass
    3. Fused operations to reduce memory traffic

    Args:
        c_in: consumption, shape (n_m, n_z), c_in[i,j] = c(m_grid[i], z_grid[j])
        V_in: value function, shape (n_m, n_z)
        model: EZModel instance

    Returns:
        c_out: consumption on (m_grid, z_grid), shape (n_m, n_z)
        V_out: value function on (m_grid, z_grid), shape (n_m, n_z)
    """
    β, R, ρ, γ, θ, m_grid, a_grid, z_grid, Q = model
    n_m, n_a, n_z = len(m_grid), len(a_grid), len(z_grid)

    # =========================================================================
    # Step 1: Compute next-period m' = R*a + z' for all (a, z') pairs
    # Then interpolate c and V at (m', z')
    # =========================================================================
    m_next = R * a_grid[:, None] + z_grid[None, :]  # shape (n_a, n_z)

    # Batch interpolation: for each z', interpolate c and V at m_next[:,z']
    # Using searchsorted for vectorized linear interpolation
    def batch_interp_column(k):
        """Interpolate c and V for z'=k at all a grid points."""
        m_q = m_next[:, k]
        c_k = jnp.interp(m_q, m_grid, c_in[:, k])
        V_k = jnp.interp(m_q, m_grid, V_in[:, k])
        return c_k, V_k

    # Vectorize over z' columns
    c_next, V_next = jax.vmap(batch_interp_column)(jnp.arange(n_z))
    c_next = jnp.maximum(c_next.T, EPS)  # shape (n_a, n_z)
    V_next = jnp.maximum(V_next.T, EPS)  # shape (n_a, n_z)

    # =========================================================================
    # Step 2: Compute W and expectations on the a-grid
    # =========================================================================
    W_next = V_next ** (1 - ρ)  # inline _V_to_W

    # Certainty equivalent μ(a, z) = (E[W^θ | z])^{1/θ}
    W_theta = W_next**θ
    E_W_theta = W_theta @ Q.T  # shape (n_a, n_z)
    E_W_theta = jnp.maximum(E_W_theta, EPS)
    μ = E_W_theta ** (1 / θ)

    # Euler integrand expectation
    euler_integrand = (W_next ** (θ - 1)) * (c_next ** (-ρ))
    E_euler = euler_integrand @ Q.T  # shape (n_a, n_z)

    # =========================================================================
    # Step 3: Invert Euler equation to get c on endogenous grid
    # =========================================================================
    rhs = jnp.maximum(β * R * (μ ** (1 - θ)) * E_euler, EPS)
    c_endog = rhs ** (-1 / ρ)  # shape (n_a, n_z)

    # =========================================================================
    # Step 4: Endogenous grid m_endog = c + a, with constraint boundary
    # =========================================================================
    m_endog = c_endog + a_grid[:, None]  # shape (n_a, n_z)

    # Prepend (m=0, c=0, μ=μ[0]) for constraint boundary
    m_endog = jnp.vstack([jnp.zeros(n_z), m_endog])  # shape (n_a+1, n_z)
    c_endog = jnp.vstack([jnp.zeros(n_z), c_endog])  # shape (n_a+1, n_z)
    μ_endog = jnp.vstack([μ[0, :], μ])  # shape (n_a+1, n_z) - prepend μ at a=0

    # =========================================================================
    # Step 5: Interpolate c AND μ from endogenous grid to m_grid
    # Key insight: interpolate both in a single pass to avoid extra vmap
    # =========================================================================
    def interp_to_m_grid(j):
        """For income state z_j, interpolate c and μ onto m_grid."""
        c_j = jnp.interp(m_grid, m_endog[:, j], c_endog[:, j])
        μ_j = jnp.interp(m_grid, m_endog[:, j], μ_endog[:, j])
        return c_j, μ_j

    c_out, μ_out = jax.vmap(interp_to_m_grid)(jnp.arange(n_z))
    c_out = jnp.maximum(c_out.T, EPS)  # shape (n_m, n_z)
    μ_out = jnp.maximum(μ_out.T, EPS)  # shape (n_m, n_z)

    # =========================================================================
    # Step 6: Compute V directly from c and μ
    # =========================================================================
    W_out = (1 - β) * (c_out ** (1 - ρ)) + β * μ_out
    V_out = jnp.maximum(W_out, EPS) ** (1 / (1 - ρ))  # inline _W_to_V

    return c_out, V_out


@jax.jit
def ezegm_accurate(
    c_in: jnp.ndarray,
    V_in: jnp.ndarray,
    model: EZModel,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    One step of EZ-EGM (accurate version).

    Unlike ezegm which interpolates μ from endogenous grid, this version:
    1. Gets consumption c on m_grid via standard EGM interpolation
    2. Recomputes μ exactly at a=m-c using full expectation formula

    More accurate but slower due to recomputing expectations.

    Accuracy: ezegm_accurate > ezegm
    Speed: ezegm > ezegm_accurate
    """
    β, R, ρ, γ, θ, m_grid, a_grid, z_grid, Q = model
    n_m, n_a, n_z = len(m_grid), len(a_grid), len(z_grid)

    # =========================================================================
    # Steps 1-4: Same as ezegm - compute c on endogenous grid
    # =========================================================================
    m_next = R * a_grid[:, None] + z_grid[None, :]

    def batch_interp_column(k):
        m_q = m_next[:, k]
        c_k = jnp.interp(m_q, m_grid, c_in[:, k])
        V_k = jnp.interp(m_q, m_grid, V_in[:, k])
        return c_k, V_k

    c_next, V_next = jax.vmap(batch_interp_column)(jnp.arange(n_z))
    c_next = jnp.maximum(c_next.T, EPS)
    V_next = jnp.maximum(V_next.T, EPS)

    W_next = V_next ** (1 - ρ)

    # μ on endogenous grid (still needed for Euler inversion)
    W_theta = W_next**θ
    E_W_theta = W_theta @ Q.T
    E_W_theta = jnp.maximum(E_W_theta, EPS)
    μ = E_W_theta ** (1 / θ)

    euler_integrand = (W_next ** (θ - 1)) * (c_next ** (-ρ))
    E_euler = euler_integrand @ Q.T

    rhs = jnp.maximum(β * R * (μ ** (1 - θ)) * E_euler, EPS)
    c_endog = rhs ** (-1 / ρ)

    m_endog = c_endog + a_grid[:, None]
    m_endog = jnp.vstack([jnp.zeros(n_z), m_endog])
    c_endog = jnp.vstack([jnp.zeros(n_z), c_endog])

    # =========================================================================
    # Step 5: Interpolate ONLY c from endogenous grid (not μ)
    # =========================================================================
    def interp_c_to_m_grid(j):
        return jnp.interp(m_grid, m_endog[:, j], c_endog[:, j])

    c_out = jax.vmap(interp_c_to_m_grid)(jnp.arange(n_z))
    c_out = jnp.maximum(c_out.T, EPS)

    # =========================================================================
    # Step 6: Recompute μ exactly at a=m-c for each (m, z) point
    # =========================================================================
    a_vals = jnp.maximum(m_grid[:, None] - c_out, 0.0)  # shape (n_m, n_z)

    def compute_mu_exact_for_z(j):
        """Compute μ exactly at each m point for income state z_j."""
        Q_j = Q[j, :]

        def compute_mu_at_m(a_val):
            m_next_vals = R * a_val + z_grid  # shape (n_z,)

            def interp_V_at_zprime(k):
                return jnp.interp(m_next_vals[k], m_grid, V_in[:, k])

            V_next_all = jax.vmap(interp_V_at_zprime)(jnp.arange(n_z))
            V_next_all = jnp.maximum(V_next_all, EPS)

            W_next_all = V_next_all ** (1 - ρ)
            W_theta_all = W_next_all**θ
            E_W_theta = jnp.maximum(jnp.sum(W_theta_all * Q_j), EPS)
            return E_W_theta ** (1 / θ)

        return jax.vmap(compute_mu_at_m)(a_vals[:, j])

    μ_out = jax.vmap(compute_mu_exact_for_z)(jnp.arange(n_z))
    μ_out = jnp.maximum(μ_out.T, EPS)  # shape (n_m, n_z)

    # =========================================================================
    # Step 7: Compute V from c and exact μ
    # =========================================================================
    W_out = (1 - β) * (c_out ** (1 - ρ)) + β * μ_out
    V_out = jnp.maximum(W_out, EPS) ** (1 / (1 - ρ))

    return c_out, V_out


@jax.jit
def ezvfi(
    V_in: jnp.ndarray,
    model: EZModel,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    One step of VFI with golden section search. Returns (V, c).

    Works on rectangular (m_grid, z_grid) cross product.
    V_in[i, j] = V(m_grid[i], z_grid[j]).

    Key optimization: Precompute μ(a, z) on the a-grid, then interpolate
    during golden search.

    Note: Uses V-space μ = (E[V^{1-γ}])^{1/(1-γ)}, which is mathematically
    equivalent to the W-space formula μ = (E[W^θ])^{1/θ} used in ezegm.
    """
    β, R, ρ, γ, θ, m_grid, a_grid, z_grid, Q = model
    n_a, n_z = len(a_grid), len(z_grid)

    # =========================================================================
    # Step 1: Precompute μ(a, z) on the a-grid
    # =========================================================================
    m_next = R * a_grid[:, None] + z_grid[None, :]  # shape (n_a, n_z)

    def interp_for_zprime(k):
        """For z'_k, interpolate V at all a values."""
        m_queries = m_next[:, k]  # shape (n_a,)
        return jnp.interp(m_queries, m_grid, V_in[:, k], right="extrapolate")

    V_next_all = jax.vmap(interp_for_zprime)(jnp.arange(n_z)).T  # shape (n_a, n_z)
    V_next_all = jnp.maximum(V_next_all, EPS)

    # μ = (E[V^{1-γ}])^{1/(1-γ)}
    V_1_minus_gamma = V_next_all ** (1 - γ)
    E_V = V_1_minus_gamma @ Q.T  # shape (n_a, n_z)
    E_V = jnp.maximum(E_V, EPS)
    μ_precomputed = E_V ** (1 / (1 - γ))  # shape (n_a, n_z)

    # =========================================================================
    # Step 2: Golden section search at each (m, z) point
    # =========================================================================

    def solve_for_z(j):
        """Solve for all m values given income state z_j."""
        μ_j = μ_precomputed[:, j]  # μ on a-grid for this z

        def solve_at_m(m):
            def objective(c_val):
                c_val = jnp.maximum(c_val, EPS)
                a = jnp.maximum(m - c_val, 0.0)

                # Interpolate μ from precomputed a-grid
                μ = jnp.interp(a, a_grid, μ_j)
                μ = jnp.maximum(μ, EPS)

                # V = [(1-β) c^{1-ρ} + β μ^{1-ρ}]^{1/(1-ρ)}
                return ((1 - β) * (c_val ** (1 - ρ)) + β * (μ ** (1 - ρ))) ** (
                    1 / (1 - ρ)
                )

            c_opt, V_opt = golden_section_search(
                objective, EPS, jnp.maximum(m - EPS, 2 * EPS), tol=BISECTION_TOL
            )
            return V_opt, c_opt

        V_j, c_j = jax.vmap(solve_at_m)(m_grid)  # Use common m_grid
        return V_j, c_j

    results = jax.vmap(solve_for_z)(jnp.arange(n_z))
    return results[0].T, results[1].T  # shape (n_m, n_z)


@jax.jit
def ezvfi_accurate(
    V_in: jnp.ndarray,
    model: EZModel,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    One step of VFI with golden section search (accurate version).

    Unlike ezvfi which precomputes μ on a-grid and interpolates, this version
    computes μ exactly at each trial consumption during golden search.
    More accurate but slower due to recomputing expectations at each trial.

    Accuracy: ezvfi_accurate > ezvfi
    Speed: ezvfi > ezvfi_accurate
    """
    β, R, ρ, γ, θ, m_grid, a_grid, z_grid, Q = model
    n_z = len(z_grid)

    # =========================================================================
    # Golden section search at each (m, z) point - compute μ exactly each time
    # =========================================================================

    def solve_for_z(j):
        """Solve for all m values given income state z_j."""
        Q_j = Q[j, :]

        def solve_at_m(m):
            def compute_mu_exact(a_val):
                """Compute μ exactly by interpolating V' at each z' then taking expectation."""
                a_val = jnp.maximum(a_val, 0.0)
                m_next = R * a_val + z_grid  # shape (n_z,)

                def interp_V_at_zprime(k):
                    return jnp.interp(m_next[k], m_grid, V_in[:, k])

                V_next = jax.vmap(interp_V_at_zprime)(jnp.arange(n_z))
                V_next = jnp.maximum(V_next, EPS)

                # μ = (E[V^{1-γ}])^{1/(1-γ)}
                V_1_minus_gamma = V_next ** (1 - γ)
                E_V = jnp.sum(V_1_minus_gamma * Q_j)
                E_V = jnp.maximum(E_V, EPS)
                return E_V ** (1 / (1 - γ))

            def objective(c_val):
                c_val = jnp.maximum(c_val, EPS)
                a = jnp.maximum(m - c_val, 0.0)

                # Compute μ exactly at this a value
                μ = compute_mu_exact(a)
                μ = jnp.maximum(μ, EPS)

                # V = [(1-β) c^{1-ρ} + β μ^{1-ρ}]^{1/(1-ρ)}
                return ((1 - β) * (c_val ** (1 - ρ)) + β * (μ ** (1 - ρ))) ** (
                    1 / (1 - ρ)
                )

            c_opt, V_opt = golden_section_search(
                objective, EPS, jnp.maximum(m - EPS, 2 * EPS), tol=BISECTION_TOL
            )
            return V_opt, c_opt

        V_j, c_j = jax.vmap(solve_at_m)(m_grid)
        return V_j, c_j

    results = jax.vmap(solve_for_z)(jnp.arange(n_z))
    return results[0].T, results[1].T  # shape (n_m, n_z)


@jax.jit
def bellman(
    c: jnp.ndarray,
    V_in: jnp.ndarray,
    model: EZModel,
) -> jnp.ndarray:
    """
    One Bellman update: given c and V, compute updated V. Returns V.

    Works on rectangular (m_grid, z_grid) cross product.
    """
    β, R, ρ, γ, θ, m_grid, a_grid, z_grid, Q = model
    n_m, n_z = len(m_grid), len(z_grid)

    c_vals = jnp.maximum(c, EPS)
    a_vals = jnp.maximum(m_grid[:, None] - c_vals, 0.0)  # shape (n_m, n_z)

    # Next-period m' for each (m, z, z') triple
    m_next_3d = R * a_vals[:, :, None] + z_grid[None, None, :]  # shape (n_m, n_z, n_z)

    # Interpolate V at next-period states
    def interp_for_zprime(k):
        """For z'_k, interpolate V at all (m, z) pairs."""
        queries = m_next_3d[:, :, k]  # shape (n_m, n_z)
        return jnp.interp(
            queries.ravel(), m_grid, V_in[:, k], right="extrapolate"
        ).reshape(n_m, n_z)

    V_next_all = jax.vmap(interp_for_zprime)(jnp.arange(n_z))  # shape (n_z, n_m, n_z)
    V_next_all = jnp.maximum(V_next_all, EPS)

    # μ = (E[V^{1-γ}])^{1/(1-γ)}
    V_1_minus_gamma = V_next_all ** (1 - γ)
    E_V = jnp.einsum("jk,kij->ij", Q, V_1_minus_gamma)  # shape (n_m, n_z)
    E_V = jnp.maximum(E_V, EPS)
    μ = E_V ** (1 / (1 - γ))

    # V = [(1-β) c^{1-ρ} + β μ^{1-ρ}]^{1/(1-ρ)}
    return ((1 - β) * (c_vals ** (1 - ρ)) + β * (μ ** (1 - ρ))) ** (1 / (1 - ρ))


@partial(jax.jit, static_argnums=(3,))
def howard(
    c: jnp.ndarray,
    V_in: jnp.ndarray,
    model: EZModel,
    n: int = 10,
    tol: float = 1e-8,
) -> jnp.ndarray:
    """Apply bellman n times with convergence-based stopping. Returns V."""

    def cond_fn(state):
        V, i, err = state
        return (i < n) & (err >= tol)

    def body_fn(state):
        V_old, i, _ = state
        V_new = bellman(c, V_old, model)
        err = jnp.max(jnp.abs(V_new - V_old))
        return V_new, i + 1, err

    V_final, _, _ = jax.lax.while_loop(cond_fn, body_fn, (V_in, 0, tol + 1.0))
    return V_final


# =============================================================================
# Time Iteration Operator (Coeurdacier et al. style)
# =============================================================================


def _bisection_solve(f, a, b, tol=1e-10, max_iter=100):
    """
    Find root of f on [a, b] using bisection.
    Assumes f(a) and f(b) have opposite signs.
    """
    f_a = f(a)

    def cond(state):
        a, b, f_a, i = state
        return ((b - a) > tol) & (i < max_iter)

    def body(state):
        a, b, f_a, i = state
        mid = (a + b) / 2
        f_mid = f(mid)
        # If f_a and f_mid have same sign, root is in [mid, b]
        use_right = f_a * f_mid > 0
        new_a = jnp.where(use_right, mid, a)
        new_b = jnp.where(use_right, b, mid)
        new_f_a = jnp.where(use_right, f_mid, f_a)
        return new_a, new_b, new_f_a, i + 1

    a_out, b_out, _, _ = jax.lax.while_loop(cond, body, (a, b, f_a, 0))
    return (a_out + b_out) / 2


@jax.jit
def ezti(
    c_in: jnp.ndarray,
    V_in: jnp.ndarray,
    model: EZModel,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    One step of Time Iteration (accurate version) using V-space formulation.

    Computes μ/Ξ by interpolating c'/V' at each candidate a during bisection,
    then taking expectations. More accurate than ezti_fast but slower.

    Uses V-space certainty equivalent: μ = (E[V^{1-γ}])^{1/(1-γ)}
    Euler equation: c^{-ρ} = βR μ^{γ-ρ} E[V'^{ρ-γ} c'^{-ρ}]
    Bellman: V = [(1-β) c^{1-ρ} + β μ^{1-ρ}]^{1/(1-ρ)}

    Accuracy: EGM ≈ TI > VFI
    Speed: EGM > VFI > TI
    """
    β, R, ρ, γ, θ, m_grid, a_grid, z_grid, Q = model
    n_z = len(z_grid)

    # Precompute c'(a, z') and V'(a, z') on the a-grid (no W conversion)
    m_next = R * a_grid[:, None] + z_grid[None, :]

    def interp_for_zprime(k):
        m_queries = m_next[:, k]
        c_interp = jnp.interp(m_queries, m_grid, c_in[:, k])
        V_interp = jnp.interp(m_queries, m_grid, V_in[:, k])
        return c_interp, V_interp

    results = jax.vmap(interp_for_zprime)(jnp.arange(n_z))
    c_next_on_a = jnp.maximum(results[0].T, EPS)
    V_next_on_a = jnp.maximum(results[1].T, EPS)

    def solve_for_z(j):
        Q_j = Q[j, :]

        def solve_at_m(m):
            def compute_mu_xi(a_val):
                """Compute μ/Ξ by interpolating c'/V' then taking expectations."""
                a_val = jnp.maximum(a_val, 0.0)

                def interp_at_zprime(k):
                    c_next = jnp.interp(a_val, a_grid, c_next_on_a[:, k])
                    V_next = jnp.interp(a_val, a_grid, V_next_on_a[:, k])
                    return jnp.maximum(c_next, EPS), jnp.maximum(V_next, EPS)

                c_next_all, V_next_all = jax.vmap(interp_at_zprime)(jnp.arange(n_z))

                # μ = (E[V^{1-γ}])^{1/(1-γ)}
                V_1_minus_gamma = V_next_all ** (1 - γ)
                E_V = jnp.maximum(jnp.sum(V_1_minus_gamma * Q_j), EPS)
                μ = E_V ** (1 / (1 - γ))

                # Ξ = E[V'^{ρ-γ} c'^{-ρ}]
                Xi_integrand = (V_next_all ** (ρ - γ)) * (c_next_all ** (-ρ))
                Xi = jnp.maximum(jnp.sum(Xi_integrand * Q_j), EPS)

                return μ, Xi

            def euler_residual(c_val):
                c_val = jnp.maximum(c_val, EPS)
                a = jnp.maximum(m - c_val, 0.0)
                μ, Xi = compute_mu_xi(a)
                lhs = c_val ** (-ρ)
                # Euler: c^{-ρ} = βR μ^{γ-ρ} Ξ
                rhs = β * R * (jnp.maximum(μ, EPS) ** (γ - ρ)) * jnp.maximum(Xi, EPS)
                return lhs - rhs

            residual_at_constraint = euler_residual(m)
            c_min, c_max = EPS, jnp.maximum(m - EPS, 2 * EPS)
            f_min, f_max = euler_residual(c_min), euler_residual(c_max)
            has_root = f_min * f_max < 0

            c_euler = _bisection_solve(euler_residual, c_min, c_max)
            c_opt = jnp.where(
                has_root & (residual_at_constraint <= 0),
                jnp.minimum(c_euler, m),
                m,
            )
            c_opt = jnp.maximum(c_opt, EPS)

            a_opt = jnp.maximum(m - c_opt, 0.0)
            μ_opt, _ = compute_mu_xi(a_opt)
            # V = [(1-β) c^{1-ρ} + β μ^{1-ρ}]^{1/(1-ρ)}
            V_opt = (
                (1 - β) * (c_opt ** (1 - ρ)) + β * (jnp.maximum(μ_opt, EPS) ** (1 - ρ))
            ) ** (1 / (1 - ρ))

            return c_opt, V_opt

        return jax.vmap(solve_at_m)(m_grid)

    results = jax.vmap(solve_for_z)(jnp.arange(n_z))
    return results[0].T, results[1].T


@jax.jit
def ezti_fast(
    c_in: jnp.ndarray,
    V_in: jnp.ndarray,
    model: EZModel,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    One step of Time Iteration (fast version) using V-space formulation.

    Precomputes μ/Ξ on the a-grid, then interpolates during bisection.
    Faster than ezti but less accurate due to μ/Ξ interpolation error.

    Uses V-space certainty equivalent: μ = (E[V^{1-γ}])^{1/(1-γ)}
    Euler equation: c^{-ρ} = βR μ^{γ-ρ} E[V'^{ρ-γ} c'^{-ρ}]
    Bellman: V = [(1-β) c^{1-ρ} + β μ^{1-ρ}]^{1/(1-ρ)}

    Accuracy: EGM > VFI > TI_fast
    Speed: EGM > TI_fast > VFI
    """
    β, R, ρ, γ, θ, m_grid, a_grid, z_grid, Q = model
    n_z = len(z_grid)

    # Precompute c' and V' on the a-grid (no W conversion)
    m_next = R * a_grid[:, None] + z_grid[None, :]

    def interp_for_zprime(k):
        m_queries = m_next[:, k]
        c_interp = jnp.interp(m_queries, m_grid, c_in[:, k])
        V_interp = jnp.interp(m_queries, m_grid, V_in[:, k])
        return c_interp, V_interp

    results = jax.vmap(interp_for_zprime)(jnp.arange(n_z))
    c_next_all = jnp.maximum(results[0].T, EPS)
    V_next_all = jnp.maximum(results[1].T, EPS)

    # Precompute μ = (E[V^{1-γ}])^{1/(1-γ)}
    V_1_minus_gamma = V_next_all ** (1 - γ)
    E_V = jnp.maximum(V_1_minus_gamma @ Q.T, EPS)
    μ_precomputed = E_V ** (1 / (1 - γ))

    # Precompute Ξ = E[V'^{ρ-γ} c'^{-ρ}]
    Xi_integrand = (V_next_all ** (ρ - γ)) * (c_next_all ** (-ρ))
    Xi_precomputed = Xi_integrand @ Q.T

    def solve_for_z(j):
        μ_j = μ_precomputed[:, j]
        Xi_j = Xi_precomputed[:, j]

        def solve_at_m(m):
            def euler_residual(c_val):
                c_val = jnp.maximum(c_val, EPS)
                a = jnp.maximum(m - c_val, 0.0)
                μ = jnp.maximum(jnp.interp(a, a_grid, μ_j), EPS)
                Xi = jnp.maximum(jnp.interp(a, a_grid, Xi_j), EPS)
                lhs = c_val ** (-ρ)
                # Euler: c^{-ρ} = βR μ^{γ-ρ} Ξ
                rhs = β * R * (μ ** (γ - ρ)) * Xi
                return lhs - rhs

            residual_at_constraint = euler_residual(m)
            c_min, c_max = EPS, jnp.maximum(m - EPS, 2 * EPS)
            f_min, f_max = euler_residual(c_min), euler_residual(c_max)
            has_root = f_min * f_max < 0

            c_euler = _bisection_solve(euler_residual, c_min, c_max)
            c_opt = jnp.where(
                has_root & (residual_at_constraint <= 0),
                jnp.minimum(c_euler, m),
                m,
            )
            c_opt = jnp.maximum(c_opt, EPS)

            a_opt = jnp.maximum(m - c_opt, 0.0)
            μ_opt = jnp.maximum(jnp.interp(a_opt, a_grid, μ_j), EPS)
            # V = [(1-β) c^{1-ρ} + β μ^{1-ρ}]^{1/(1-ρ)}
            V_opt = ((1 - β) * (c_opt ** (1 - ρ)) + β * (μ_opt ** (1 - ρ))) ** (
                1 / (1 - ρ)
            )

            return c_opt, V_opt

        return jax.vmap(solve_at_m)(m_grid)

    results = jax.vmap(solve_for_z)(jnp.arange(n_z))
    return results[0].T, results[1].T


@partial(jax.jit, static_argnums=(3,))
def solve_ezti(
    model: EZModel,
    tol: float = 1e-6,
    max_iter: int = 1000,
    n_howard: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    Solve using Time Iteration (accurate version).

    Uses ezti which computes μ/Ξ by interpolating c'/V' then taking expectations.
    Uses V-space formulation: μ = (E[V^{1-γ}])^{1/(1-γ)}, exactly as in VFI.

    Returns (c, V, m_grid, n_iter) - check n_iter < max_iter for convergence.
    Ranking: Accuracy EGM ≈ TI > VFI, Speed EGM > VFI > TI
    """
    β, R, ρ, γ, θ, m_grid, a_grid, z_grid, Q = model
    n_z = len(z_grid)

    c_init = jnp.tile(m_grid[:, None], (1, n_z)) * 0.9
    V_init = c_init.copy()

    def cond_fn(state):
        c, V, i, err = state
        return (err >= tol) & (i < max_iter)

    def body_fn(state):
        c_old, V_old, i, _ = state
        c_new, V_new = ezti(c_old, V_old, model)
        V_new = jax.lax.cond(
            n_howard > 1,
            lambda V: howard(c_new, V, model, n_howard - 1),
            lambda V: V,
            V_new,
        )
        err = jnp.max(jnp.abs(c_new - c_old))
        return c_new, V_new, i + 1, err

    c_out, V_out, n_iter, _ = jax.lax.while_loop(
        cond_fn, body_fn, (c_init, V_init, 0, tol + 1.0)
    )
    return c_out, V_out, m_grid, n_iter


@partial(jax.jit, static_argnums=(3,))
def solve_ezti_fast(
    model: EZModel,
    tol: float = 1e-6,
    max_iter: int = 1000,
    n_howard: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    Solve using Time Iteration (fast version).

    Uses ezti_fast which precomputes μ/Ξ on a-grid then interpolates.
    Uses V-space formulation: μ = (E[V^{1-γ}])^{1/(1-γ)}, exactly as in VFI.

    Returns (c, V, m_grid, n_iter) - check n_iter < max_iter for convergence.
    Ranking: Accuracy EGM > VFI > TI_fast, Speed EGM > TI_fast > VFI
    """
    β, R, ρ, γ, θ, m_grid, a_grid, z_grid, Q = model
    n_z = len(z_grid)

    c_init = jnp.tile(m_grid[:, None], (1, n_z)) * 0.9
    V_init = c_init.copy()

    def cond_fn(state):
        c, V, i, err = state
        return (err >= tol) & (i < max_iter)

    def body_fn(state):
        c_old, V_old, i, _ = state
        c_new, V_new = ezti_fast(c_old, V_old, model)
        V_new = jax.lax.cond(
            n_howard > 1,
            lambda V: howard(c_new, V, model, n_howard - 1),
            lambda V: V,
            V_new,
        )
        err = jnp.max(jnp.abs(c_new - c_old))
        return c_new, V_new, i + 1, err

    c_out, V_out, n_iter, _ = jax.lax.while_loop(
        cond_fn, body_fn, (c_init, V_init, 0, tol + 1.0)
    )
    return c_out, V_out, m_grid, n_iter


# =============================================================================
# Four Solvers: solve_ezegm, solve_ezvfi, solve_ezti, solve_ezti_fast
# =============================================================================


@partial(jax.jit, static_argnums=(3,))
def solve_ezegm(
    model: EZModel,
    tol: float = 1e-6,
    max_iter: int = 1000,
    n_howard: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    Solve using EGM. n_howard=1 is 1:1 ratio, n_howard=K is K:1.

    Returns (c, V, m_grid, n_iter) where:
        - c[i, j] = c(m_grid[i], z_grid[j]) - proper cross product
        - V[i, j] = V(m_grid[i], z_grid[j]) - proper cross product
        - m_grid is 1D, same for all z states
        - n_iter: iterations used (check n_iter < max_iter for convergence)
    """
    β, R, ρ, γ, θ, m_grid, a_grid, z_grid, Q = model
    n_z = len(z_grid)

    # Initialize: consume-all policy, V = c
    c_init = jnp.tile(m_grid[:, None], (1, n_z)) * 0.9
    V_init = c_init.copy()

    def cond_fn(state):
        c, V, i, err = state
        return (err >= tol) & (i < max_iter)

    def body_fn(state):
        c_old, V_old, i, _ = state

        # EGM step
        c_new, V_new = ezegm(c_old, V_old, model)

        # Howard: apply bellman (n_howard - 1) extra times
        V_new = jax.lax.cond(
            n_howard > 1,
            lambda V: howard(c_new, V, model, n_howard - 1),
            lambda V: V,
            V_new,
        )

        err = jnp.max(jnp.abs(c_new - c_old))
        return c_new, V_new, i + 1, err

    c_out, V_out, n_iter, _ = jax.lax.while_loop(
        cond_fn, body_fn, (c_init, V_init, 0, tol + 1.0)
    )

    return c_out, V_out, m_grid, n_iter


@partial(jax.jit, static_argnums=(3,))
def solve_ezvfi(
    model: EZModel,
    tol: float = 1e-6,
    max_iter: int = 1000,
    n_howard: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    Solve using VFI with golden section search.

    Returns (c, V, m_grid, n_iter) where:
        - c[i, j] = c(m_grid[i], z_grid[j]) - proper cross product
        - V[i, j] = V(m_grid[i], z_grid[j]) - proper cross product
        - m_grid is 1D, same for all z states
        - n_iter: iterations used (check n_iter < max_iter for convergence)
    """
    β, R, ρ, γ, θ, m_grid, a_grid, z_grid, Q = model
    n_z = len(z_grid)

    c_init = jnp.tile(m_grid[:, None], (1, n_z)) * 0.5
    V_init = c_init.copy()

    def cond_fn(state):
        V, c, i, err = state
        return (err >= tol) & (i < max_iter)

    def body_fn(state):
        V_old, c_old, i, _ = state
        V_new, c_new = ezvfi(V_old, model)

        # Howard: apply bellman (n_howard - 1) extra times
        V_new = jax.lax.cond(
            n_howard > 1,
            lambda V: howard(c_new, V, model, n_howard - 1),
            lambda V: V,
            V_new,
        )

        err = jnp.max(jnp.abs(V_new - V_old))
        return V_new, c_new, i + 1, err

    V_out, c_out, n_iter, _ = jax.lax.while_loop(
        cond_fn, body_fn, (V_init, c_init, 0, tol + 1.0)
    )
    return c_out, V_out, m_grid, n_iter


@partial(jax.jit, static_argnums=(3,))
def solve_ezegm_accurate(
    model: EZModel,
    tol: float = 1e-6,
    max_iter: int = 1000,
    n_howard: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    Solve using EGM (accurate version).

    Uses ezegm_accurate which recomputes μ exactly at a=m-c after EGM step.
    More accurate than standard EGM but slower.

    Returns (c, V, m_grid, n_iter) - check n_iter < max_iter for convergence.
    """
    β, R, ρ, γ, θ, m_grid, a_grid, z_grid, Q = model
    n_z = len(z_grid)

    c_init = jnp.tile(m_grid[:, None], (1, n_z)) * 0.9
    V_init = c_init.copy()

    def cond_fn(state):
        c, V, i, err = state
        return (err >= tol) & (i < max_iter)

    def body_fn(state):
        c_old, V_old, i, _ = state
        c_new, V_new = ezegm_accurate(c_old, V_old, model)

        V_new = jax.lax.cond(
            n_howard > 1,
            lambda V: howard(c_new, V, model, n_howard - 1),
            lambda V: V,
            V_new,
        )

        err = jnp.max(jnp.abs(c_new - c_old))
        return c_new, V_new, i + 1, err

    c_out, V_out, n_iter, _ = jax.lax.while_loop(
        cond_fn, body_fn, (c_init, V_init, 0, tol + 1.0)
    )

    return c_out, V_out, m_grid, n_iter


@partial(jax.jit, static_argnums=(3,))
def solve_ezvfi_accurate(
    model: EZModel,
    tol: float = 1e-6,
    max_iter: int = 1000,
    n_howard: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    Solve using VFI (accurate version).

    Uses ezvfi_accurate which computes μ exactly during golden search.
    More accurate than standard VFI but slower.

    Returns (c, V, m_grid, n_iter) - check n_iter < max_iter for convergence.
    """
    β, R, ρ, γ, θ, m_grid, a_grid, z_grid, Q = model
    n_z = len(z_grid)

    c_init = jnp.tile(m_grid[:, None], (1, n_z)) * 0.5
    V_init = c_init.copy()

    def cond_fn(state):
        V, c, i, err = state
        return (err >= tol) & (i < max_iter)

    def body_fn(state):
        V_old, c_old, i, _ = state
        V_new, c_new = ezvfi_accurate(V_old, model)

        V_new = jax.lax.cond(
            n_howard > 1,
            lambda V: howard(c_new, V, model, n_howard - 1),
            lambda V: V,
            V_new,
        )

        err = jnp.max(jnp.abs(V_new - V_old))
        return V_new, c_new, i + 1, err

    V_out, c_out, n_iter, _ = jax.lax.while_loop(
        cond_fn, body_fn, (V_init, c_init, 0, tol + 1.0)
    )
    return c_out, V_out, m_grid, n_iter


# =============================================================================
# Euler Equation Error
# =============================================================================


def compute_euler_errors(
    c_policy: jnp.ndarray,
    V_policy: jnp.ndarray,
    m_grid: jnp.ndarray,
    model: EZModel,
    n_test: int = 500,
    constraint_tol: float = 0.01,  # 1% of grid range
) -> jnp.ndarray:
    """
    Compute Euler equation errors at interior (unconstrained) points.

    The Euler equation holds as EQUALITY only at interior points where a > 0.
    At the constraint (a = 0, c = m), the Euler equation holds as an INEQUALITY,
    so we exclude those points from error computation.

    Near the constraint boundary, even small savings have elevated Euler errors
    due to the kink in the policy function. We use a relative threshold
    (default 1% of grid range) to exclude points near the constraint.

    Args:
        c_policy: consumption, shape (n_m, n_z), on m_grid × z_grid
        V_policy: value function, shape (n_m, n_z)
        m_grid: 1D cash-on-hand grid (same for all z)
        model: EZModel instance
        n_test: number of test points
        constraint_tol: relative threshold for constrained points (fraction of grid range)

    Returns:
        log10 Euler errors at UNCONSTRAINED test points (averaged over income states)
    """
    β, R, ρ, γ, θ, _, a_grid, z_grid, Q = model
    n_z = len(z_grid)

    # Test in the interior of the grid (avoid boundary issues)
    m_min_test = jnp.percentile(m_grid, 10)
    m_max_test = jnp.percentile(m_grid, 90)
    m_test = jnp.linspace(m_min_test, m_max_test, n_test)

    # Absolute threshold for constrained points (relative to grid scale)
    a_threshold = constraint_tol * (m_grid[-1] - m_grid[0])

    def errors_for_z(j):
        # Interpolate c at test points (m_grid is 1D, same for all z)
        c_vals = jnp.interp(m_test, m_grid, c_policy[:, j], right="extrapolate")
        c_vals = jnp.maximum(c_vals, EPS)
        a_vals = jnp.maximum(m_test - c_vals, 0.0)

        # Identify constrained points (where a ≈ 0, Euler holds as inequality)
        constrained = a_vals < a_threshold

        m_next = R * a_vals[:, None] + z_grid[None, :]

        def interp_k(k):
            c_next = jnp.interp(
                m_next[:, k], m_grid, c_policy[:, k], right="extrapolate"
            )
            V_next = jnp.interp(
                m_next[:, k], m_grid, V_policy[:, k], right="extrapolate"
            )
            return c_next, V_next

        results = jax.vmap(interp_k)(jnp.arange(n_z))
        c_next_all = jnp.maximum(results[0].T, EPS)
        V_next_all = jnp.maximum(results[1].T, EPS)

        W_next_all = _V_to_W(V_next_all, ρ)

        euler_integrand = (W_next_all ** (θ - 1)) * (c_next_all ** (-ρ))
        E_integrand = euler_integrand @ Q[j, :]
        W_theta = W_next_all**θ
        E_W_theta = jnp.maximum(W_theta @ Q[j, :], EPS)
        μ = E_W_theta ** (1 / θ)

        rhs = β * R * (μ ** (1 - θ)) * E_integrand
        c_euler = rhs ** (-1 / ρ)
        error = jnp.abs(1 - c_euler / c_vals)
        log_error = jnp.log10(jnp.maximum(error, 1e-16))

        # Set constrained points to NaN (will be excluded from statistics)
        log_error = jnp.where(constrained, jnp.nan, log_error)
        return log_error

    all_errors = jax.vmap(errors_for_z)(jnp.arange(n_z))
    # Average over z, ignoring NaN (constrained points)
    return jnp.nanmean(all_errors, axis=0)


# =============================================================================
# Ergodic Distribution Simulation
# =============================================================================


def simulate_ergodic(
    c_policy: jnp.ndarray,
    model: EZModel,
    n_agents: int = 10000,
    n_periods: int = 500,
    burn_in: int = 200,
    seed: int = 42,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate the model to obtain the ergodic distribution of (m, z).

    Args:
        c_policy: consumption policy, shape (n_m, n_z)
        model: EZModel instance
        n_agents: number of agents to simulate
        n_periods: total simulation periods
        burn_in: periods to discard before collecting distribution
        seed: random seed

    Returns:
        m_sim: simulated cash-on-hand values, shape (n_agents * (n_periods - burn_in),)
        z_sim: simulated income state indices, shape (n_agents * (n_periods - burn_in),)
    """
    R = model.R
    m_grid = model.m_grid
    z_grid = model.z_grid
    Q = model.Q
    n_z = len(z_grid)

    key = jax.random.PRNGKey(seed)

    # Initialize agents at median wealth and random income state
    m_init = jnp.ones(n_agents) * jnp.median(m_grid)
    key, subkey = jax.random.split(key)
    z_idx_init = jax.random.randint(subkey, (n_agents,), 0, n_z, dtype=jnp.int32)

    # Precompute cumulative transition probabilities for sampling
    Q_cumsum = jnp.cumsum(Q, axis=1)

    def simulate_step(carry, key_t):
        m_t, z_idx_t = carry

        # Get consumption from policy
        def get_c(i):
            return jnp.interp(m_t[i], m_grid, c_policy[:, z_idx_t[i]])

        c_t = jax.vmap(get_c)(jnp.arange(n_agents))
        c_t = jnp.clip(c_t, EPS, m_t)  # Ensure feasibility

        # End-of-period assets
        a_t = m_t - c_t

        # Draw next income state
        key1, _ = jax.random.split(key_t)
        u = jax.random.uniform(key1, (n_agents,))

        def next_z(i):
            return jnp.searchsorted(Q_cumsum[z_idx_t[i]], u[i]).astype(jnp.int32)

        z_idx_next = jax.vmap(next_z)(jnp.arange(n_agents))
        z_idx_next = jnp.clip(z_idx_next, 0, n_z - 1).astype(jnp.int32)

        # Next period cash-on-hand
        m_next = R * a_t + z_grid[z_idx_next]

        return (m_next, z_idx_next), (m_t, z_idx_t)

    # Run simulation
    keys = jax.random.split(key, n_periods)
    _, (m_history, z_history) = jax.lax.scan(simulate_step, (m_init, z_idx_init), keys)

    # Discard burn-in and flatten
    m_sim = m_history[burn_in:].flatten()
    z_sim = z_history[burn_in:].flatten()

    return m_sim, z_sim


def compute_euler_errors_weighted(
    c_policy: jnp.ndarray,
    V_policy: jnp.ndarray,
    m_grid: jnp.ndarray,
    model: EZModel,
    m_sim: jnp.ndarray,
    z_sim: jnp.ndarray,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    constraint_tol: float = 0.01,
) -> tuple[float, float]:
    """
    Compute Euler errors at points from the ergodic distribution.

    Args:
        c_policy: consumption, shape (n_m, n_z)
        V_policy: value function, shape (n_m, n_z)
        m_grid: 1D cash-on-hand grid
        model: EZModel instance
        m_sim: simulated m values from ergodic distribution
        z_sim: simulated z indices from ergodic distribution
        percentile_low: lower percentile to include (default 5th)
        percentile_high: upper percentile to include (default 95th)
        constraint_tol: relative threshold for constrained points

    Returns:
        mean_error: mean log10 Euler error
        max_error: max log10 Euler error
    """
    β, R, ρ, γ, θ, _, a_grid, z_grid, Q = model
    n_z = len(z_grid)

    # Filter to percentile range
    m_low = jnp.percentile(m_sim, percentile_low)
    m_high = jnp.percentile(m_sim, percentile_high)
    mask = (m_sim >= m_low) & (m_sim <= m_high)
    m_test = m_sim[mask]
    z_test = z_sim[mask]

    # Subsample if too many points (for speed)
    n_max = 5000
    if len(m_test) > n_max:
        idx = jnp.linspace(0, len(m_test) - 1, n_max).astype(int)
        m_test = m_test[idx]
        z_test = z_test[idx]

    a_threshold = constraint_tol * (m_grid[-1] - m_grid[0])

    def compute_error(i):
        m_i = m_test[i]
        z_i = z_test[i]

        # Interpolate c at this point
        c_val = jnp.interp(m_i, m_grid, c_policy[:, z_i])
        c_val = jnp.maximum(c_val, EPS)
        a_val = jnp.maximum(m_i - c_val, 0.0)

        # Check if constrained
        constrained = a_val < a_threshold

        # Next period states
        m_next = R * a_val + z_grid

        # Interpolate next-period c and V for all z' (vectorized)
        def interp_z(k):
            c_k = jnp.interp(m_next[k], m_grid, c_policy[:, k])
            V_k = jnp.interp(m_next[k], m_grid, V_policy[:, k])
            return c_k, V_k

        c_next, V_next = jax.vmap(interp_z)(jnp.arange(n_z))
        c_next = jnp.maximum(c_next, EPS)
        V_next = jnp.maximum(V_next, EPS)

        W_next = _V_to_W(V_next, ρ)

        # Euler equation components
        euler_integrand = (W_next ** (θ - 1)) * (c_next ** (-ρ))
        E_integrand = jnp.sum(euler_integrand * Q[z_i, :])
        W_theta = W_next**θ
        E_W_theta = jnp.maximum(jnp.sum(W_theta * Q[z_i, :]), EPS)
        μ = E_W_theta ** (1 / θ)

        rhs = β * R * (μ ** (1 - θ)) * E_integrand
        c_euler = rhs ** (-1 / ρ)
        error = jnp.abs(1 - c_euler / c_val)
        log_error = jnp.log10(jnp.maximum(error, 1e-16))

        # Return NaN if constrained
        return jnp.where(constrained, jnp.nan, log_error)

    errors = jax.vmap(compute_error)(jnp.arange(len(m_test)))

    mean_error = float(jnp.nanmean(errors))
    max_error = float(jnp.nanmax(errors))

    return mean_error, max_error


# =============================================================================
# Consumption-Equivalent Welfare Cost
# =============================================================================


def compute_welfare_cost(
    c_approx: jnp.ndarray,
    V_approx: jnp.ndarray,
    m_grid_approx: jnp.ndarray,
    c_true: jnp.ndarray,
    V_true: jnp.ndarray,
    m_grid_true: jnp.ndarray,
    model: EZModel,
    n_agents: int = 10000,
    n_periods: int = 500,
    burn_in: int = 200,
    seed: int = 42,
) -> float:
    """
    Compute consumption-equivalent welfare cost of using approximate policy.

    This calculates how much consumption (as fraction) an agent would pay
    to switch from the approximate policy to the true policy.

    Method: Simulate agents using both policies with SAME shocks, evaluate
    welfare using the TRUE value function at states visited. Compare the
    ergodic average values.

    Args:
        c_approx: approximate consumption policy, shape (n_m_approx, n_z)
        V_approx: approximate value function, shape (n_m_approx, n_z)
        m_grid_approx: grid for approximate policy
        c_true: true (high-accuracy) consumption policy, shape (n_m_true, n_z)
        V_true: true (high-accuracy) value function, shape (n_m_true, n_z)
        m_grid_true: grid for true policy
        model: EZModel instance (for other parameters)
        n_agents: number of agents for simulation
        n_periods: simulation length
        burn_in: periods to discard before computing average
        seed: random seed

    Returns:
        welfare_cost: consumption-equivalent cost (fraction, e.g., 0.001 = 0.1%)
    """
    R = model.R
    ρ = model.ρ
    z_grid = model.z_grid
    Q = model.Q
    n_z = len(z_grid)

    key = jax.random.PRNGKey(seed)

    # Use the approximate grid range for simulation
    m_grid = m_grid_approx

    # Initialize agents at median wealth
    m_init = jnp.ones(n_agents) * jnp.median(m_grid)
    key, subkey = jax.random.split(key)
    z_idx_init = jax.random.randint(subkey, (n_agents,), 0, n_z, dtype=jnp.int32)

    Q_cumsum = jnp.cumsum(Q, axis=1)

    def simulate_with_policy(c_policy, c_grid, V_eval, V_grid, key):
        """
        Simulate using c_policy (on c_grid), evaluate value using V_eval (on V_grid).
        Returns ergodic average of V_eval at visited states.
        """

        def step(carry, key_t):
            m_t, z_idx_t = carry

            # Get consumption from policy
            def get_c(i):
                return jnp.interp(m_t[i], c_grid, c_policy[:, z_idx_t[i]])

            c_t = jax.vmap(get_c)(jnp.arange(n_agents))
            c_t = jnp.clip(c_t, EPS, m_t)

            # Get value from evaluation function at current state
            def get_V(i):
                return jnp.interp(m_t[i], V_grid, V_eval[:, z_idx_t[i]])

            v_t = jax.vmap(get_V)(jnp.arange(n_agents))

            a_t = m_t - c_t

            # Draw next state
            key1, _ = jax.random.split(key_t)
            u = jax.random.uniform(key1, (n_agents,))

            def next_z(i):
                return jnp.searchsorted(Q_cumsum[z_idx_t[i]], u[i]).astype(jnp.int32)

            z_idx_next = jax.vmap(next_z)(jnp.arange(n_agents))
            z_idx_next = jnp.clip(z_idx_next, 0, n_z - 1).astype(jnp.int32)

            m_next = R * a_t + z_grid[z_idx_next]

            return (m_next, z_idx_next), v_t

        keys = jax.random.split(key, n_periods)
        _, v_history = jax.lax.scan(step, (m_init, z_idx_init), keys)

        # Ergodic average: mean over post-burn-in periods and agents
        return jnp.mean(v_history[burn_in:])

    # Compute values using SAME random seed for fair comparison
    key1, _ = jax.random.split(key)

    # Value under approximate policy, evaluated with TRUE value function
    V_approx_realized = simulate_with_policy(
        c_approx, m_grid_approx, V_true, m_grid_true, key1
    )

    # Value under true policy, evaluated with TRUE value function (same shocks)
    V_true_realized = simulate_with_policy(
        c_true, m_grid_true, V_true, m_grid_true, key1
    )

    # Consumption equivalent: λ = 1 - (V_approx/V_true)^(1-ρ)
    # If V_approx < V_true (worse policy), ratio < 1, so λ > 0 (positive cost)
    ratio = jnp.maximum(V_approx_realized / V_true_realized, EPS)
    welfare_cost = 1.0 - float(ratio ** (1 - ρ))

    return welfare_cost
