"""
Shared utilities for dynamic programming solvers.

This module provides common numerical routines used across VFI, OPI, PFI,
and EGM implementations, avoiding code duplication.

Tolerance Conventions
---------------------
The codebase uses a hierarchy of tolerances for different purposes:

    SOLVER_TOL = 1e-5
        Outer loop convergence for value/policy iteration.
        Controls when the solver declares convergence.

    GOLDEN_TOL = 1e-6
        Golden section search for continuous optimization.
        Determines precision of optimal consumption/savings choice.

    BISECTION_TOL = 1e-8
        Bisection/root-finding for Euler equation inversion.
        Tighter than golden search since root-finding is cheaper.

    POLICY_EVAL_TOL = 1e-8
        Inner loop for policy evaluation (PFI).
        Tighter than outer loop to ensure accurate policy values.

    EPS = 1e-10
        Numerical floor to prevent division by zero and invalid powers.
        Used in expressions like jnp.maximum(x, EPS) before exponentiation.

Rationale: Outer tolerance (1e-5) balances speed vs accuracy for typical use.
Inner tolerances (1e-8 to 1e-10) are tighter because inner loops are cheaper
and errors compound through outer iterations.
"""

import math

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

__all__ = [
    "SOLVER_TOL",
    "GOLDEN_TOL",
    "BISECTION_TOL",
    "POLICY_EVAL_TOL",
    "EPS",
    "golden_section_search",
    "crra_utility",
]

# =============================================================================
# Tolerance Constants
# =============================================================================

SOLVER_TOL = 1e-5  # Outer loop convergence
GOLDEN_TOL = 1e-6  # Golden section search precision
BISECTION_TOL = 1e-8  # Root-finding precision
POLICY_EVAL_TOL = 1e-8  # Policy evaluation convergence
EPS = 1e-10  # Numerical floor for stability


# =============================================================================
# Golden Section Search
# =============================================================================

# Golden ratio constants (Python floats to avoid JIT tracing overhead)
_PHI = (1 + math.sqrt(5)) / 2
_INVPHI = 1 / _PHI  # ≈ 0.618
_INVPHI2 = 1 / (_PHI * _PHI)  # ≈ 0.382


def golden_section_search(f, a, b, tol=GOLDEN_TOL):
    """
    Find argmax and maximum value of f on [a, b] using golden section search.

    Assumes f is unimodal (single peak) on [a, b]. Uses the golden ratio
    to efficiently narrow the search interval, requiring only one new
    function evaluation per iteration.

    Args:
        f: Objective function to maximize (must be unimodal)
        a: Left endpoint of search interval
        b: Right endpoint of search interval
        tol: Convergence tolerance (interval width)

    Returns:
        Tuple (x*, f(x*)) where x* is the approximate maximizer and f(x*) is
        the maximum value. Callers who only need x* can use: x_opt, _ = ...
    """
    h = b - a
    c = a + _INVPHI2 * h
    d = a + _INVPHI * h
    fc = f(c)
    fd = f(d)

    def cond(state):
        a, b, c, d, fc, fd = state
        return (b - a) > tol

    def body(state):
        a, b, c, d, fc, fd = state
        # If f(c) > f(d), maximum is in [a, d], else in [c, b]
        take_left = fc > fd

        a_new = jnp.where(take_left, a, c)
        b_new = jnp.where(take_left, d, b)

        # Reuse one interior point, compute new one
        c_new = jnp.where(take_left, a_new + _INVPHI2 * (b_new - a_new), d)
        d_new = jnp.where(take_left, c, a_new + _INVPHI * (b_new - a_new))

        fc_new = jnp.where(take_left, f(c_new), fd)
        fd_new = jnp.where(take_left, fc, f(d_new))

        return a_new, b_new, c_new, d_new, fc_new, fd_new

    a, b, c, d, fc, fd = jax.lax.while_loop(cond, body, (a, b, c, d, fc, fd))

    x_opt = jnp.where(fc > fd, c, d)
    f_opt = jnp.where(fc > fd, fc, fd)
    return x_opt, f_opt


# =============================================================================
# CRRA Utility
# =============================================================================


def crra_utility(c, γ):
    """
    CRRA (Constant Relative Risk Aversion) utility function.

    u(c) = c^(1-γ) / (1-γ)  if γ ≠ 1
         = log(c)           if γ = 1

    Handles the γ = 1 (log utility) case as a limit.

    Args:
        c: Consumption (scalar or array)
        γ: Risk aversion coefficient

    Returns:
        Utility value. Returns -inf for c ≤ 0.
    """
    return jnp.where(
        c > 0,
        jnp.where(jnp.abs(γ - 1.0) < EPS, jnp.log(c), c ** (1 - γ) / (1 - γ)),
        -jnp.inf,
    )
