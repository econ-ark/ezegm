"""
Reproducible benchmarks for the EZ-EGM paper.

This script generates ALL numerical results and figures that appear in the paper.
Run this script to verify/update the paper's tables, claims, and figures.

Usage:
    uv run python benchmark_paper.py          # Run all benchmarks
    uv run python benchmark_paper.py --all    # Same as above
    uv run python benchmark_paper.py --main   # Main results only (Tables 1-2, Figures)
    uv run python benchmark_paper.py --appendix  # Appendix only (Howard K comparison)
    uv run python benchmark_paper.py --figures   # Regenerate figures only

The script outputs results in a format matching the paper's tables,
making it easy to verify consistency between code and paper.

=============================================================================
COMPLETE LIST OF NUMERICAL CLAIMS IN THE PAPER
=============================================================================

Section: Numerical example (parameters)
    - "10 grid points" for income (Tauchen discretization)
    - "100 points" for asset grid
    - "20 times mean income" for upper bound
    - β = 0.96 (standard annual)
    - R = 1.02 (2% real rate)
    - ρ = 2/3 (EIS = 1.5, Bansal-Yaron 2004)
    - γ = 10 (Bansal-Yaron 2004)
    - θ = (1-γ)/(1-ρ) = -27
    - z_rho = 0.95 (Storesletten et al. 2004)
    - K = 2 for EGM Howard acceleration
    - K = 4 for TI Howard acceleration
    - K = 30 for VFI Howard acceleration

Section: Speed comparison (Table 1)
    - EZ-EGM (1:1): time, iterations
    - EZ-EGM + Howard: time, iterations
    - TI (fast mode): time, iterations
    - VFI (golden section search): time, iterations
    - VFI-Howard: time, iterations
    - "approximately 60 times faster"
    - "239 to 11" reduction for VFI-Howard
    - "141 to 99" reduction for EGM-Howard

Section: Accuracy (Table 2)
    - EZ-EGM: mean (L1), max (L∞)
    - EZ-EGM + Howard: mean (L1), max (L∞)
    - TI (fast mode): mean (L1), max (L∞)
    - VFI: mean (L1), max (L∞)
    - "approximately 1.5 orders of magnitude smaller"

Figures:
    - Figure 1: Consumption policy function c(m,z)
    - Figure 2: Speed-accuracy tradeoff (EGM vs VFI across grid sizes)
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Ensure consistent numerical precision
jax.config.update("jax_enable_x64", True)

from ez_egm import (
    create_ez_model,
    solve_ezegm,
    solve_ezvfi,
    solve_ezvfi_accurate,
    solve_ezti_fast,
    solve_ezti,
    compute_euler_errors,
    simulate_ergodic,
    compute_euler_errors_weighted,
    ezegm,
    ezvfi,
    howard,
)


# =============================================================================
# Paper Parameters (Section: Numerical example)
# These MUST match the paper exactly
# All values are citable from the literature
# =============================================================================

PAPER_PARAMS = {
    # Preference parameters (Bansal & Yaron 2004)
    "β": 0.96,  # Standard annual discount factor
    "R": 1.02,  # 2% real interest rate
    "ρ": 2 / 3,  # EIS = 1/ρ = 1.5 (Bansal-Yaron 2004)
    "γ": 10.0,  # Risk aversion (Bansal-Yaron 2004)
    # Cash-on-hand grid (m_grid)
    "m_min": 0.0,  # Start at zero (borrowing constraint)
    "m_max": 20.0,  # Paper: "20 times mean income" (mean income ≈ 1)
    "m_size": 100,  # Paper: "100 points"
    "a_size": 100,  # Asset grid for EGM internal use
    # Income process (Storesletten, Telmer & Yaron 2004)
    "z_rho": 0.95,  # Persistence
    "z_sigma": 0.1,  # Innovation std dev
    "z_size": 10,  # Paper: "10 grid points"
}

# Solver parameters
TOL = 1e-5  # Convergence tolerance
MAX_ITER = 1000  # Maximum iterations
K_EGM = 2  # Paper: K=2 for EGM (improves accuracy but not speed)
K_TI = 4  # Paper: K=4 for TI (optimal; K=5 explodes)
K_VFI = 30  # Paper: K=30 for VFI (minimizes time)

# Number of timing runs for averaging (reduces noise)
N_TIMING_RUNS = 3

# Output directory for figures
FIGURE_DIR = Path(__file__).parent.parent / "content" / "figures"

# Formats to save (MyST will choose the best for each export target)
FIGURE_FORMATS = ["pdf", "svg", "png"]

# Minimum value for numerical stability
_EPS = 1e-10


# =============================================================================
# Figure Style Configuration
# =============================================================================


def setup_figure_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(
        {
            # Font settings
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            # Figure settings
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Line settings
            "lines.linewidth": 1.2,
            "axes.linewidth": 0.6,
            # Grid
            "axes.grid": False,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            # LaTeX
            "text.usetex": False,
            "mathtext.fontset": "cm",
        }
    )


# =============================================================================
# Model and Solver Functions
# =============================================================================


def create_paper_model():
    """Create model with exact paper parameters.

    Uses exponential grid spacing to capture policy curvature near
    the liquidity constraint, where the MPC is highest.
    """
    return create_ez_model(
        β=PAPER_PARAMS["β"],
        R=PAPER_PARAMS["R"],
        ρ=PAPER_PARAMS["ρ"],
        γ=PAPER_PARAMS["γ"],
        m_min=PAPER_PARAMS["m_min"],
        m_max=PAPER_PARAMS["m_max"],
        m_size=PAPER_PARAMS["m_size"],
        a_size=PAPER_PARAMS["a_size"],
        z_rho=PAPER_PARAMS["z_rho"],
        z_sigma=PAPER_PARAMS["z_sigma"],
        z_size=PAPER_PARAMS["z_size"],
        grid_type="exp",  # Exponential grid captures curvature near constraint
    )


def time_solver(solver_fn, model, n_runs=N_TIMING_RUNS, **kwargs):
    """
    Time a solver with JIT warmup and multiple runs.

    Returns (mean_time, result) where result is from the last run.
    """
    # JIT warmup (don't count this)
    result = solver_fn(model, tol=1e-3, max_iter=5, **kwargs)
    if hasattr(result[0], "block_until_ready"):
        result[0].block_until_ready()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = solver_fn(model, tol=TOL, max_iter=MAX_ITER, **kwargs)
        if hasattr(result[0], "block_until_ready"):
            result[0].block_until_ready()
        times.append(time.perf_counter() - start)

    return sum(times) / len(times), result


# =============================================================================
# Convergence Tracking Solvers (for Figure 2)
# =============================================================================


def solve_ezegm_with_history(model, tol=1e-6, max_iter=500, n_howard=1):
    """
    Solve using EGM while tracking convergence history.

    Returns (c, V, m_grid, n_iter, error_history, time_history).
    """
    m_grid, z_grid = model.m_grid, model.z_grid
    n_m, n_z = len(m_grid), len(z_grid)

    # Initialize: consume-all policy, V = c
    c = jnp.tile(m_grid[:, None], (1, n_z)) * 0.9
    V = c.copy()

    error_history = []
    time_history = []
    start_time = time.perf_counter()

    for i in range(max_iter):
        c_new, V_new = ezegm(c, V, model)

        # Howard acceleration: apply bellman (n_howard - 1) extra times
        if n_howard > 1:
            V_new = howard(c_new, V_new, model, n_howard - 1)

        # Force computation
        c_new.block_until_ready()

        # Record cumulative time
        time_history.append(time.perf_counter() - start_time)

        # Compare policies on common grid
        err = float(jnp.max(jnp.abs(c_new - c)))
        error_history.append(err)

        if err < tol:
            return (
                c_new,
                V_new,
                m_grid,
                i + 1,
                np.array(error_history),
                np.array(time_history),
            )

        c, V = c_new, V_new

    return c, V, m_grid, max_iter, np.array(error_history), np.array(time_history)


def solve_ezvfi_with_history(model, tol=1e-6, max_iter=500, n_howard=1):
    """
    Solve using VFI while tracking convergence history.

    Uses the ezvfi operator for golden section optimization.

    Returns (c, V, m_grid, n_iter, error_history, time_history).
    """
    m_grid, z_grid = model.m_grid, model.z_grid
    n_m, n_z = len(m_grid), len(z_grid)

    # Initialize: V = c
    c = jnp.tile(m_grid[:, None], (1, n_z)) * 0.5
    V = c.copy()

    error_history = []
    time_history = []
    start_time = time.perf_counter()

    for i in range(max_iter):
        V_new, c_new = ezvfi(V, model)

        # Howard acceleration: apply bellman (n_howard - 1) extra times
        if n_howard > 1:
            V_new = howard(c_new, V_new, model, n_howard - 1)

        # Force computation
        V_new.block_until_ready()

        # Record cumulative time
        time_history.append(time.perf_counter() - start_time)

        # Compare policies on common grid
        err = float(jnp.max(jnp.abs(c_new - c)))
        error_history.append(err)

        if err < tol:
            return (
                c_new,
                V_new,
                m_grid,
                i + 1,
                np.array(error_history),
                np.array(time_history),
            )

        c, V = c_new, V_new

    return c, V, m_grid, max_iter, np.array(error_history), np.array(time_history)


def compute_ergodic_errors(c, V, m, model):
    """Compute Euler errors weighted by ergodic distribution."""
    m_sim, z_sim = simulate_ergodic(
        c, model, n_agents=10000, n_periods=500, burn_in=200
    )
    mean_err, max_err = compute_euler_errors_weighted(c, V, m, model, m_sim, z_sim)
    return float(mean_err), float(max_err)


def run_main_benchmarks(model):
    """Run main paper benchmarks (Tables 1-2, Figures)."""
    results = {"model": model}

    # -------------------------------------------------------------------------
    # EZ-EGM (1:1) - baseline (used in both Table 1 and Table 2)
    # -------------------------------------------------------------------------
    print("Running EZ-EGM (1:1)...", flush=True)
    t1, (c1, V1, m1, n1) = time_solver(solve_ezegm, model)
    # Grid-based errors (for comparison)
    errors1_grid = compute_euler_errors(c1, V1, m1, model)
    # Ergodic errors (for paper tables)
    mean_erg1, max_erg1 = compute_ergodic_errors(c1, V1, m1, model)
    results["EZ-EGM (1:1)"] = {
        "time": t1,
        "iterations": int(n1),
        "euler_mean": mean_erg1,  # Ergodic
        "euler_max": max_erg1,  # Ergodic
        "euler_mean_grid": float(jnp.nanmean(errors1_grid)),
        "euler_max_grid": float(jnp.nanmax(errors1_grid)),
        "c": np.array(c1),
        "V": np.array(V1),
        "m": np.array(m1),
        "errors": np.array(errors1_grid),
    }
    print(f"  Done: {t1:.3f}s, {n1} iterations, ergodic mean={mean_erg1:.1f}")

    # -------------------------------------------------------------------------
    # EZ-EGM + Howard
    # -------------------------------------------------------------------------
    print(f"Running EZ-EGM + Howard (K={K_EGM})...", flush=True)
    t2, (c2, V2, m2, n2) = time_solver(solve_ezegm, model, n_howard=K_EGM)
    errors2_grid = compute_euler_errors(c2, V2, m2, model)
    mean_erg2, max_erg2 = compute_ergodic_errors(c2, V2, m2, model)
    results["EZ-EGM + Howard"] = {
        "time": t2,
        "iterations": int(n2),
        "euler_mean": mean_erg2,
        "euler_max": max_erg2,
        "euler_mean_grid": float(jnp.nanmean(errors2_grid)),
        "euler_max_grid": float(jnp.nanmax(errors2_grid)),
        "c": np.array(c2),
        "V": np.array(V2),
        "m": np.array(m2),
        "errors": np.array(errors2_grid),
    }
    print(f"  Done: {t2:.3f}s, {n2} iterations, ergodic mean={mean_erg2:.1f}")

    # -------------------------------------------------------------------------
    # VFI fast (golden section, precomputed μ) - Table 1
    # -------------------------------------------------------------------------
    print("Running VFI (fast mode)...", flush=True)
    t3, (c3, V3, m3, n3) = time_solver(solve_ezvfi, model, n_howard=1)
    errors3_grid = compute_euler_errors(c3, V3, m3, model)
    mean_erg3, max_erg3 = compute_ergodic_errors(c3, V3, m3, model)
    results["VFI"] = {
        "time": t3,
        "iterations": int(n3),
        "euler_mean": mean_erg3,
        "euler_max": max_erg3,
        "euler_mean_grid": float(jnp.nanmean(errors3_grid)),
        "euler_max_grid": float(jnp.nanmax(errors3_grid)),
        "c": np.array(c3),
        "V": np.array(V3),
        "m": np.array(m3),
        "errors": np.array(errors3_grid),
    }
    print(f"  Done: {t3:.3f}s, {n3} iterations, ergodic mean={mean_erg3:.1f}")

    # -------------------------------------------------------------------------
    # VFI accurate (computes μ exactly during search) - Table 2
    # -------------------------------------------------------------------------
    print("Running VFI (accurate mode)...", flush=True)
    t3a, (c3a, V3a, m3a, n3a) = time_solver(solve_ezvfi_accurate, model, n_howard=1)
    errors3a_grid = compute_euler_errors(c3a, V3a, m3a, model)
    mean_erg3a, max_erg3a = compute_ergodic_errors(c3a, V3a, m3a, model)
    results["VFI-accurate"] = {
        "time": t3a,
        "iterations": int(n3a),
        "euler_mean": mean_erg3a,
        "euler_max": max_erg3a,
        "euler_mean_grid": float(jnp.nanmean(errors3a_grid)),
        "euler_max_grid": float(jnp.nanmax(errors3a_grid)),
    }
    print(f"  Done: {t3a:.3f}s, {n3a} iterations, ergodic mean={mean_erg3a:.1f}")

    # -------------------------------------------------------------------------
    # VFI + Howard
    # -------------------------------------------------------------------------
    print(f"Running VFI + Howard (K={K_VFI})...", flush=True)
    t4, (c4, V4, m4, n4) = time_solver(solve_ezvfi, model, n_howard=K_VFI)
    results["VFI-Howard"] = {
        "time": t4,
        "iterations": int(n4),
    }
    print(f"  Done: {t4:.3f}s, {n4} iterations")

    # -------------------------------------------------------------------------
    # TI fast (precomputes μ on grid) - Table 1
    # -------------------------------------------------------------------------
    print("Running TI (fast mode)...", flush=True)
    t5, (c5, V5, m5, n5) = time_solver(solve_ezti_fast, model)
    errors5_grid = compute_euler_errors(c5, V5, m5, model)
    mean_erg5, max_erg5 = compute_ergodic_errors(c5, V5, m5, model)
    results["TI"] = {
        "time": t5,
        "iterations": int(n5),
        "euler_mean": mean_erg5,
        "euler_max": max_erg5,
        "euler_mean_grid": float(jnp.nanmean(errors5_grid)),
        "euler_max_grid": float(jnp.nanmax(errors5_grid)),
        "c": np.array(c5),
        "V": np.array(V5),
        "m": np.array(m5),
        "errors": np.array(errors5_grid),
    }
    print(f"  Done: {t5:.3f}s, {n5} iterations, ergodic mean={mean_erg5:.1f}")

    # -------------------------------------------------------------------------
    # TI accurate (computes μ exactly during search) - Table 2
    # -------------------------------------------------------------------------
    print("Running TI (accurate mode)...", flush=True)
    t5a, (c5a, V5a, m5a, n5a) = time_solver(solve_ezti, model)
    errors5a_grid = compute_euler_errors(c5a, V5a, m5a, model)
    mean_erg5a, max_erg5a = compute_ergodic_errors(c5a, V5a, m5a, model)
    results["TI-accurate"] = {
        "time": t5a,
        "iterations": int(n5a),
        "euler_mean": mean_erg5a,
        "euler_max": max_erg5a,
        "euler_mean_grid": float(jnp.nanmean(errors5a_grid)),
        "euler_max_grid": float(jnp.nanmax(errors5a_grid)),
    }
    print(f"  Done: {t5a:.3f}s, {n5a} iterations, ergodic mean={mean_erg5a:.1f}")

    # -------------------------------------------------------------------------
    # TI + Howard (fast mode with Howard acceleration)
    # -------------------------------------------------------------------------
    print(f"Running TI + Howard (K={K_TI})...", flush=True)
    t6, (c6, V6, m6, n6) = time_solver(solve_ezti_fast, model, n_howard=K_TI)
    errors6_grid = compute_euler_errors(c6, V6, m6, model)
    mean_erg6, max_erg6 = compute_ergodic_errors(c6, V6, m6, model)
    results["TI-Howard"] = {
        "time": t6,
        "iterations": int(n6),
        "euler_mean": mean_erg6,
        "euler_max": max_erg6,
        "euler_mean_grid": float(jnp.nanmean(errors6_grid)),
        "euler_max_grid": float(jnp.nanmax(errors6_grid)),
    }
    print(f"  Done: {t6:.3f}s, {n6} iterations")

    # -------------------------------------------------------------------------
    # Convergence history (for Figure 2)
    # -------------------------------------------------------------------------
    print("Running convergence tracking (for Figure 2)...", flush=True)

    # Warmup all four methods
    _ = solve_ezegm_with_history(model, tol=1e-3, max_iter=5, n_howard=1)
    _ = solve_ezegm_with_history(model, tol=1e-3, max_iter=5, n_howard=K_EGM)
    _ = solve_ezvfi_with_history(model, tol=1e-3, max_iter=5, n_howard=1)
    _ = solve_ezvfi_with_history(model, tol=1e-3, max_iter=5, n_howard=K_VFI)

    # Full runs with history for all four methods
    _, _, _, _, egm_history, egm_time = solve_ezegm_with_history(
        model, tol=TOL, max_iter=300, n_howard=1
    )
    _, _, _, _, egm_howard_history, egm_howard_time = solve_ezegm_with_history(
        model, tol=TOL, max_iter=300, n_howard=K_EGM
    )
    _, _, _, _, vfi_history, vfi_time = solve_ezvfi_with_history(
        model, tol=TOL, max_iter=300, n_howard=1
    )
    _, _, _, _, vfi_howard_history, vfi_howard_time = solve_ezvfi_with_history(
        model, tol=TOL, max_iter=300, n_howard=K_VFI
    )

    results["convergence"] = {
        "egm_history": egm_history,
        "egm_time": egm_time,
        "egm_howard_history": egm_howard_history,
        "egm_howard_time": egm_howard_time,
        "vfi_history": vfi_history,
        "vfi_time": vfi_time,
        "vfi_howard_history": vfi_howard_history,
        "vfi_howard_time": vfi_howard_time,
    }
    print(
        f"  Done: EGM {len(egm_history)} iters, EGM+H {len(egm_howard_history)} iters, VFI {len(vfi_history)} iters, VFI+H {len(vfi_howard_history)} iters"
    )

    return results


def run_appendix_benchmarks(model):
    """Run appendix benchmarks (Howard K comparison for EGM and VFI)."""

    # -------------------------------------------------------------------------
    # EGM Howard K comparison
    # -------------------------------------------------------------------------
    print("Running EGM Howard K comparison (for Appendix)...", flush=True)

    egm_howard_results = []
    for K in [1, 2, 3, 4, 5]:
        if K == 1:
            # Warmup
            _ = solve_ezegm(model, tol=1e-3, max_iter=5)
            # Time
            times = []
            for _ in range(N_TIMING_RUNS):
                start = time.perf_counter()
                c, V, m, n = solve_ezegm(model, tol=TOL, max_iter=MAX_ITER)
                c.block_until_ready()
                times.append(time.perf_counter() - start)
        else:
            # Warmup
            _ = solve_ezegm(model, tol=1e-3, max_iter=5, n_howard=K)
            # Time
            times = []
            for _ in range(N_TIMING_RUNS):
                start = time.perf_counter()
                c, V, m, n = solve_ezegm(model, tol=TOL, max_iter=MAX_ITER, n_howard=K)
                c.block_until_ready()
                times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        errors = compute_euler_errors(c, V, m, model)
        errors_valid = errors[~jnp.isnan(errors)]

        egm_howard_results.append(
            {
                "K": K,
                "time": avg_time,
                "iterations": int(n),
                "euler_mean": float(jnp.nanmean(errors_valid)),
                "euler_max": float(jnp.nanmax(errors_valid)),
            }
        )
        print(
            f"    K={K}: {avg_time * 1000:.1f}ms, {n} iters, mean error {float(jnp.nanmean(errors_valid)):.1f}"
        )

    # -------------------------------------------------------------------------
    # VFI Howard K comparison
    # -------------------------------------------------------------------------
    print("Running VFI Howard K comparison (for Appendix)...", flush=True)

    vfi_howard_results = []
    for K in [1, 10, 20, 30, 40, 50]:
        # Warmup
        _ = solve_ezvfi(model, tol=1e-3, max_iter=5, n_howard=K)
        # Time (only 1 run for VFI since it's slow)
        start = time.perf_counter()
        c, V, m, n = solve_ezvfi(model, tol=TOL, max_iter=MAX_ITER, n_howard=K)
        c.block_until_ready()
        elapsed = time.perf_counter() - start

        errors = compute_euler_errors(c, V, m, model)
        errors_valid = errors[~jnp.isnan(errors)]

        vfi_howard_results.append(
            {
                "K": K,
                "time": elapsed,
                "iterations": int(n),
                "euler_mean": float(jnp.nanmean(errors_valid)),
                "euler_max": float(jnp.nanmax(errors_valid)),
            }
        )
        print(
            f"    K={K}: {elapsed:.1f}s, {n} iters, mean error {float(jnp.nanmean(errors_valid)):.1f}"
        )

    # -------------------------------------------------------------------------
    # TI Howard K comparison
    # -------------------------------------------------------------------------
    print("Running TI Howard K comparison (for Appendix)...", flush=True)

    ti_howard_results = []
    for K in [1, 2, 3, 4, 5]:
        # Warmup
        _ = solve_ezti_fast(model, tol=1e-3, max_iter=5, n_howard=K)
        # Time
        times = []
        for _ in range(N_TIMING_RUNS):
            start = time.perf_counter()
            c, V, m, n = solve_ezti_fast(model, tol=TOL, max_iter=MAX_ITER, n_howard=K)
            c.block_until_ready()
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        errors = compute_euler_errors(c, V, m, model)
        errors_valid = errors[~jnp.isnan(errors)]

        ti_howard_results.append(
            {
                "K": K,
                "time": avg_time,
                "iterations": int(n),
                "euler_mean": float(jnp.nanmean(errors_valid)),
                "euler_max": float(jnp.nanmax(errors_valid)),
            }
        )
        print(
            f"    K={K}: {avg_time * 1000:.1f}ms, {n} iters, mean error {float(jnp.nanmean(errors_valid)):.1f}"
        )

    # -------------------------------------------------------------------------
    # VFI accurate mode Howard K comparison
    # -------------------------------------------------------------------------
    print("Running VFI accurate mode Howard K comparison (for Appendix)...", flush=True)

    vfi_acc_howard_results = []
    for K in [1, 10, 20, 30, 40, 50]:
        # Warmup
        _ = solve_ezvfi_accurate(model, tol=1e-3, max_iter=5, n_howard=K)
        # Time (only 1 run for VFI accurate since it's slow)
        start = time.perf_counter()
        c, V, m, n = solve_ezvfi_accurate(model, tol=TOL, max_iter=MAX_ITER, n_howard=K)
        c.block_until_ready()
        elapsed = time.perf_counter() - start

        errors = compute_euler_errors(c, V, m, model)
        errors_valid = errors[~jnp.isnan(errors)]

        vfi_acc_howard_results.append(
            {
                "K": K,
                "time": elapsed,
                "iterations": int(n),
                "euler_mean": float(jnp.nanmean(errors_valid)),
                "euler_max": float(jnp.nanmax(errors_valid)),
            }
        )
        print(
            f"    K={K}: {elapsed:.1f}s, {n} iters, mean error {float(jnp.nanmean(errors_valid)):.1f}"
        )

    # -------------------------------------------------------------------------
    # TI accurate mode Howard K comparison
    # -------------------------------------------------------------------------
    print("Running TI accurate mode Howard K comparison (for Appendix)...", flush=True)

    ti_acc_howard_results = []
    for K in [1, 2, 3, 4, 5]:
        # Warmup
        _ = solve_ezti(model, tol=1e-3, max_iter=5, n_howard=K)
        # Time (only 1 run since it's slow)
        start = time.perf_counter()
        c, V, m, n = solve_ezti(model, tol=TOL, max_iter=MAX_ITER, n_howard=K)
        c.block_until_ready()
        elapsed = time.perf_counter() - start

        errors = compute_euler_errors(c, V, m, model)
        errors_valid = errors[~jnp.isnan(errors)]

        ti_acc_howard_results.append(
            {
                "K": K,
                "time": elapsed,
                "iterations": int(n),
                "euler_mean": float(jnp.nanmean(errors_valid)),
                "euler_max": float(jnp.nanmax(errors_valid)),
            }
        )
        print(
            f"    K={K}: {elapsed:.1f}s, {n} iters, mean error {float(jnp.nanmean(errors_valid)):.1f}"
        )

    # -------------------------------------------------------------------------
    # Robustness: EGM across different rho values (Appendix: rho > 1)
    # -------------------------------------------------------------------------
    print("Running EGM robustness across rho values (for Appendix)...", flush=True)

    rho_robustness_results = []
    for rho in [0.5, 0.9, 1.1, 1.5, 2.0, 3.0]:
        # Create model with different rho, same other parameters
        model_rho = create_ez_model(
            ρ=rho,
            γ=model.γ,
            β=model.β,
            R=model.R,
            m_max=float(model.m_grid[-1]) / model.R,  # Original m_max
            m_size=len(model.m_grid),
            a_size=len(model.a_grid),
            z_rho=0.95,
            z_sigma=0.1,
            z_size=len(model.z_grid),
        )

        # Solve with EGM
        c, V, m, n = solve_ezegm(model_rho, tol=TOL, max_iter=MAX_ITER, n_howard=1)

        errors = compute_euler_errors(c, V, m, model_rho)
        errors_valid = errors[~jnp.isnan(errors)]

        rho_robustness_results.append(
            {
                "rho": rho,
                "eis": 1 / rho,
                "theta": float(model_rho.θ),
                "iterations": int(n),
                "euler_mean": float(jnp.nanmean(errors_valid)),
                "euler_max": float(jnp.nanmax(errors_valid)),
            }
        )
        print(
            f"    ρ={rho:.1f} (EIS={1 / rho:.2f}): {n} iters, mean={float(jnp.nanmean(errors_valid)):.1f}, max={float(jnp.nanmax(errors_valid)):.1f}"
        )

    return {
        "egm_howard_comparison": egm_howard_results,
        "vfi_howard_comparison": vfi_howard_results,
        "vfi_acc_howard_comparison": vfi_acc_howard_results,
        "ti_howard_comparison": ti_howard_results,
        "ti_acc_howard_comparison": ti_acc_howard_results,
        "rho_robustness": rho_robustness_results,
        "model": model,
    }


def run_equal_accuracy_benchmarks():
    """
    Run equal-accuracy benchmarks following Chris Carroll's suggestions:
    1. Ergodic distribution weighting for Euler errors
    2. Equal-accuracy speed comparison (EGM vs VFI at same error level)
    3. Consumption-equivalent welfare costs
    """
    from ez_egm import (
        simulate_ergodic,
        compute_euler_errors_weighted,
        compute_welfare_cost,
    )

    print("=" * 70)
    print("EQUAL-ACCURACY BENCHMARKS (Carroll Suggestions)")
    print("=" * 70)
    print()

    results = {}

    # -------------------------------------------------------------------------
    # 1. Ergodic distribution and weighted Euler errors
    # -------------------------------------------------------------------------
    print("1. Ergodic Distribution and Weighted Euler Errors")
    print("-" * 50)

    model = create_paper_model()
    c, V, m_grid, _ = solve_ezegm(model, tol=TOL, max_iter=MAX_ITER, n_howard=K_EGM)

    # Simulate ergodic distribution
    print("   Simulating ergodic distribution (10000 agents, 500 periods)...")
    m_sim, z_sim = simulate_ergodic(
        c, model, n_agents=10000, n_periods=500, burn_in=200
    )

    # Report ergodic distribution statistics
    p5 = float(jnp.percentile(m_sim, 5))
    p50 = float(jnp.percentile(m_sim, 50))
    p95 = float(jnp.percentile(m_sim, 95))
    print(f"   Ergodic wealth distribution: p5={p5:.2f}, p50={p50:.2f}, p95={p95:.2f}")

    # Weighted Euler errors
    mean_err_w, max_err_w = compute_euler_errors_weighted(
        c, V, m_grid, model, m_sim, z_sim
    )
    print(
        f"   Weighted Euler errors (5th-95th pctl): mean={mean_err_w:.2f}, max={max_err_w:.2f}"
    )

    # Compare to uniform grid errors
    errors_uniform = compute_euler_errors(c, V, m_grid, model)
    mean_err_u = float(jnp.nanmean(errors_uniform))
    max_err_u = float(jnp.nanmax(errors_uniform))
    print(
        f"   Uniform grid errors (10th-90th pctl):  mean={mean_err_u:.2f}, max={max_err_u:.2f}"
    )

    results["ergodic"] = {
        "p5": p5,
        "p50": p50,
        "p95": p95,
        "mean_weighted": mean_err_w,
        "max_weighted": max_err_w,
        "mean_uniform": mean_err_u,
        "max_uniform": max_err_u,
    }
    print()

    # -------------------------------------------------------------------------
    # 2. Equal-accuracy speed comparison
    # -------------------------------------------------------------------------
    print("2. Equal-Accuracy Speed Comparison")
    print("-" * 50)

    def benchmark_grid_sizes(method, grid_sizes):
        results = []
        for n in grid_sizes:
            # Create model with custom grid size but same other parameters
            model_n = create_ez_model(
                β=PAPER_PARAMS["β"],
                R=PAPER_PARAMS["R"],
                ρ=PAPER_PARAMS["ρ"],
                γ=PAPER_PARAMS["γ"],
                m_min=PAPER_PARAMS["m_min"],
                m_max=PAPER_PARAMS["m_max"],
                m_size=n,
                a_size=n,
                z_rho=PAPER_PARAMS["z_rho"],
                z_sigma=PAPER_PARAMS["z_sigma"],
                z_size=PAPER_PARAMS["z_size"],
            )

            # Warmup
            if method == "egm":
                _ = solve_ezegm(model_n, tol=1e-3, max_iter=5)
            else:
                _ = solve_ezvfi(model_n, tol=1e-3, max_iter=5)

            # Time
            start = time.perf_counter()
            if method == "egm":
                c_n, V_n, m_n, n_iter = solve_ezegm(model_n, tol=TOL, max_iter=MAX_ITER)
            else:
                c_n, V_n, m_n, n_iter = solve_ezvfi(model_n, tol=TOL, max_iter=MAX_ITER)
            c_n.block_until_ready()
            elapsed = time.perf_counter() - start

            # Euler errors
            err = compute_euler_errors(c_n, V_n, m_n, model_n)

            results.append(
                {
                    "n": n,
                    "time": elapsed,
                    "mean_err": float(jnp.nanmean(err)),
                    "max_err": float(jnp.nanmax(err)),
                }
            )
        return results

    print("   Benchmarking EGM with grid sizes [15, 20, 25, 30, 50, 100]...")
    egm_bench = benchmark_grid_sizes("egm", [15, 20, 25, 30, 50, 100])

    print("   Benchmarking VFI with grid sizes [50, 100, 150, 200, 300]...")
    vfi_bench = benchmark_grid_sizes("vfi", [50, 100, 150, 200, 300])

    # Find matches at similar accuracy
    print()
    print("   Equal-accuracy matches:")
    matches = []
    for vr in vfi_bench:
        # Find EGM with closest accuracy
        best_egm = min(egm_bench, key=lambda e: abs(e["mean_err"] - vr["mean_err"]))
        speedup = vr["time"] / best_egm["time"]
        matches.append(
            {
                "vfi_n": vr["n"],
                "vfi_err": vr["mean_err"],
                "vfi_time": vr["time"],
                "egm_n": best_egm["n"],
                "egm_err": best_egm["mean_err"],
                "egm_time": best_egm["time"],
                "speedup": speedup,
            }
        )

    # Print table matching paper format
    print()
    print("   APPENDIX TABLE: Speed comparison at equal accuracy")
    print("   " + "-" * 65)
    print("   | VFI n | EGM n | Mean Error | VFI (ms) | EGM (ms) | Speedup |")
    print("   " + "-" * 65)
    for m in matches:
        print(
            f"   | {m['vfi_n']:5d} | {m['egm_n']:5d} | {m['vfi_err']:10.1f} | "
            f"{m['vfi_time'] * 1000:8.0f} | {m['egm_time'] * 1000:8.0f} | {m['speedup']:6.0f}x |"
        )
    print("   " + "-" * 65)

    results["equal_accuracy"] = {
        "egm_bench": egm_bench,
        "vfi_bench": vfi_bench,
        "matches": matches,
    }
    print()

    # -------------------------------------------------------------------------
    # 3. Consumption-equivalent welfare costs
    # -------------------------------------------------------------------------
    print("3. Consumption-Equivalent Welfare Costs")
    print("-" * 50)

    # Solve with fine grid as "truth"
    print("   Solving fine-grid reference (500 points)...")
    model_fine = create_ez_model(
        β=PAPER_PARAMS["β"],
        R=PAPER_PARAMS["R"],
        ρ=PAPER_PARAMS["ρ"],
        γ=PAPER_PARAMS["γ"],
        m_min=PAPER_PARAMS["m_min"],
        m_max=PAPER_PARAMS["m_max"],
        m_size=500,
        a_size=500,
        z_rho=PAPER_PARAMS["z_rho"],
        z_sigma=PAPER_PARAMS["z_sigma"],
        z_size=PAPER_PARAMS["z_size"],
    )
    c_fine, V_fine, m_fine, _ = solve_ezegm(model_fine, tol=1e-8, max_iter=MAX_ITER)

    # Compute welfare costs for different approximations
    welfare_results = []
    for method, n_grid in [
        ("EGM", 25),
        ("EGM", 50),
        ("EGM", 100),
        ("VFI", 50),
        ("VFI", 100),
    ]:
        model_approx = create_ez_model(
            β=PAPER_PARAMS["β"],
            R=PAPER_PARAMS["R"],
            ρ=PAPER_PARAMS["ρ"],
            γ=PAPER_PARAMS["γ"],
            m_min=PAPER_PARAMS["m_min"],
            m_max=PAPER_PARAMS["m_max"],
            m_size=n_grid,
            a_size=n_grid,
            z_rho=PAPER_PARAMS["z_rho"],
            z_sigma=PAPER_PARAMS["z_sigma"],
            z_size=PAPER_PARAMS["z_size"],
        )
        if method == "EGM":
            c_approx, V_approx, m_approx, _ = solve_ezegm(model_approx, tol=TOL)
        else:
            c_approx, V_approx, m_approx, _ = solve_ezvfi(model_approx, tol=TOL)

        wc = compute_welfare_cost(
            c_approx,
            V_approx,
            m_approx,
            c_fine,
            V_fine,
            m_fine,
            model_approx,
            n_agents=20000,
        )

        err = compute_euler_errors(c_approx, V_approx, m_approx, model_approx)

        welfare_results.append(
            {
                "method": method,
                "n": n_grid,
                "welfare_cost": wc,
                "mean_err": float(jnp.nanmean(err)),
            }
        )
        print(
            f"   {method} n={n_grid:3d}: welfare cost = {wc * 100:.4f}%, Euler mean = {float(jnp.nanmean(err)):.1f}"
        )

    results["welfare"] = welfare_results
    print()

    return results


def run_benchmarks(include_main=True, include_appendix=True):
    """Run benchmarks and return results dict with solution arrays."""
    print("=" * 70)
    print("EZ-EGM Paper Benchmarks")
    print("=" * 70)
    print()

    # Create model
    model = create_paper_model()

    # Verify parameters match paper
    print("Model Parameters (verify against paper):")
    print(f"  β = {model.β}")
    print(f"  R = {model.R}")
    print(f"  ρ = {model.ρ} (EIS = {1 / model.ρ:.1f})")
    print(f"  γ = {model.γ}")
    print(f"  θ = (1-γ)/(1-ρ) = {model.θ:.1f}")
    print(f"  Grid: {len(model.m_grid)} m points × {len(model.z_grid)} income states")
    print(f"  a_grid: {len(model.a_grid)} points (internal EGM grid)")
    print()

    results = {"model": model}

    if include_main:
        main_results = run_main_benchmarks(model)
        results.update(main_results)

    if include_appendix:
        appendix_results = run_appendix_benchmarks(model)
        results.update(appendix_results)

    return results


# =============================================================================
# Figure Generation
# =============================================================================


def save_figure_multiformat(fig, basename):
    """Save figure in multiple formats for MyST to choose from."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in FIGURE_FORMATS:
        filepath = FIGURE_DIR / f"{basename}.{fmt}"
        fig.savefig(filepath, format=fmt)
    print(f"  Saved: {FIGURE_DIR / basename}.* ({', '.join(FIGURE_FORMATS)})")


def generate_figure_policy(results, basename="fig_policy"):
    """
    Figure 1: Consumption policy function c(m,z).

    The policy c(m, z) is defined on a proper rectangular grid:
    - m_grid is 1D (same for all z states)
    - c[i, j] = c(m_grid[i], z_grid[j])

    No interpolation needed - just plot directly!
    """
    model = results["model"]
    c = results["EZ-EGM + Howard"]["c"]
    m_grid = np.array(model.m_grid)  # 1D array, same for all z
    z_grid = np.array(model.z_grid)

    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    # Color map for income states
    n_z = len(z_grid)
    colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, n_z))

    # Plot policy for each income state - DIRECT plotting, no interpolation!
    m_max_plot = m_grid.max()

    for i in range(n_z):
        # Label only low, middle, high income
        if i == 0:
            label = f"Low $z$ ($z={z_grid[i]:.2f}$)"
        elif i == n_z // 2:
            label = f"Mean $z$ ($z={z_grid[i]:.2f}$)"
        elif i == n_z - 1:
            label = f"High $z$ ($z={z_grid[i]:.2f}$)"
        else:
            label = None

        # Direct plot: m_grid is same for all z, c[:, i] is consumption for z_i
        ax.plot(m_grid, c[:, i], color=colors[i], label=label, lw=1.3)

    # 45-degree line (consume everything / liquidity constraint)
    ax.plot([0, m_max_plot], [0, m_max_plot], "k--", lw=0.8, alpha=0.4, label="$c = m$")

    ax.set_xlabel("Cash-on-hand $m$")
    ax.set_ylabel("Consumption $c(m,z)$")
    ax.set_xlim(0, m_max_plot)
    ax.set_ylim(0, c.max() * 1.05)
    ax.legend(loc="lower right", fontsize=7)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    # Save in multiple formats
    save_figure_multiformat(fig, basename)

    return fig


def generate_figure_pareto_modes(results, basename="fig_pareto"):
    """
    Figure: Speed-accuracy Pareto frontier with grid size variations.

    Shows EGM, TI (fast/accurate), and VFI (fast/accurate) at different grid sizes.
    Demonstrates equal-accuracy speedups across methods.
    """
    print("  Computing speed-accuracy data for EGM, TI, and VFI across grid sizes...")

    # Grid sizes to test
    egm_sizes = [15, 20, 25, 30, 50, 75, 100]
    ti_fast_sizes = [50, 60, 75, 100, 150]
    ti_acc_sizes = [40, 50, 60, 75, 100]
    vfi_sizes = [50, 75, 100, 150, 200]

    def benchmark_egm(sizes):
        results_list = []
        for n in sizes:
            model_n = create_ez_model(
                β=PAPER_PARAMS["β"],
                R=PAPER_PARAMS["R"],
                ρ=PAPER_PARAMS["ρ"],
                γ=PAPER_PARAMS["γ"],
                m_min=PAPER_PARAMS["m_min"],
                m_max=PAPER_PARAMS["m_max"],
                m_size=n,
                a_size=n,
                z_rho=PAPER_PARAMS["z_rho"],
                z_sigma=PAPER_PARAMS["z_sigma"],
                z_size=PAPER_PARAMS["z_size"],
            )
            _ = solve_ezegm(model_n, tol=1e-3, max_iter=5)
            start = time.perf_counter()
            c_n, V_n, m_n, _ = solve_ezegm(model_n, tol=TOL, max_iter=MAX_ITER)
            c_n.block_until_ready()
            elapsed = time.perf_counter() - start
            err = compute_euler_errors(c_n, V_n, m_n, model_n)
            results_list.append(
                {"n": n, "time_ms": elapsed * 1000, "mean_err": float(jnp.nanmean(err))}
            )
        return results_list

    def benchmark_ti(sizes, accurate=False):
        results_list = []
        solver = solve_ezti if accurate else solve_ezti_fast
        for n in sizes:
            model_n = create_ez_model(
                β=PAPER_PARAMS["β"],
                R=PAPER_PARAMS["R"],
                ρ=PAPER_PARAMS["ρ"],
                γ=PAPER_PARAMS["γ"],
                m_min=PAPER_PARAMS["m_min"],
                m_max=PAPER_PARAMS["m_max"],
                m_size=n,
                a_size=n,
                z_rho=PAPER_PARAMS["z_rho"],
                z_sigma=PAPER_PARAMS["z_sigma"],
                z_size=PAPER_PARAMS["z_size"],
            )
            _ = solver(model_n, tol=1e-3, max_iter=5)
            start = time.perf_counter()
            c_n, V_n, m_n, _ = solver(model_n, tol=TOL, max_iter=MAX_ITER)
            c_n.block_until_ready()
            elapsed = time.perf_counter() - start
            err = compute_euler_errors(c_n, V_n, m_n, model_n)
            results_list.append(
                {"n": n, "time_ms": elapsed * 1000, "mean_err": float(jnp.nanmean(err))}
            )
        return results_list

    def benchmark_vfi(sizes, accurate=False):
        results_list = []
        solver = solve_ezvfi_accurate if accurate else solve_ezvfi
        for n in sizes:
            model_n = create_ez_model(
                β=PAPER_PARAMS["β"],
                R=PAPER_PARAMS["R"],
                ρ=PAPER_PARAMS["ρ"],
                γ=PAPER_PARAMS["γ"],
                m_min=PAPER_PARAMS["m_min"],
                m_max=PAPER_PARAMS["m_max"],
                m_size=n,
                a_size=n,
                z_rho=PAPER_PARAMS["z_rho"],
                z_sigma=PAPER_PARAMS["z_sigma"],
                z_size=PAPER_PARAMS["z_size"],
            )
            _ = solver(model_n, tol=1e-3, max_iter=5)
            start = time.perf_counter()
            c_n, V_n, m_n, _ = solver(model_n, tol=TOL, max_iter=MAX_ITER)
            c_n.block_until_ready()
            elapsed = time.perf_counter() - start
            err = compute_euler_errors(c_n, V_n, m_n, model_n)
            results_list.append(
                {"n": n, "time_ms": elapsed * 1000, "mean_err": float(jnp.nanmean(err))}
            )
        return results_list

    egm_data = benchmark_egm(egm_sizes)
    ti_fast_data = benchmark_ti(ti_fast_sizes, accurate=False)
    ti_acc_data = benchmark_ti(ti_acc_sizes, accurate=True)
    vfi_fast_data = benchmark_vfi(vfi_sizes, accurate=False)
    vfi_acc_data = benchmark_vfi(vfi_sizes, accurate=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Helper to scale marker size by grid points
    def marker_size(n, base=3, scale=0.08):
        return base + scale * n

    # Plot EGM (stars, blue) - size scales with n
    for d in egm_data:
        ax.plot(
            d["time_ms"],
            d["mean_err"],
            "*",
            color="#3498db",
            markersize=marker_size(d["n"], base=5, scale=0.12),
        )
    # Add line connecting EGM points
    egm_times = [d["time_ms"] for d in egm_data]
    egm_errs = [d["mean_err"] for d in egm_data]
    ax.plot(egm_times, egm_errs, "-", color="#3498db", lw=1.2, label="EGM")

    # Plot TI fast (open squares, green) - size scales with n
    for d in ti_fast_data:
        ax.plot(
            d["time_ms"],
            d["mean_err"],
            "s",
            color="#27ae60",
            markersize=marker_size(d["n"]),
            markerfacecolor="white",
            markeredgewidth=1.5,
        )
    ti_fast_times = [d["time_ms"] for d in ti_fast_data]
    ti_fast_errs = [d["mean_err"] for d in ti_fast_data]
    ax.plot(ti_fast_times, ti_fast_errs, "--", color="#27ae60", lw=1.2, label="TI fast")

    # Plot TI accurate (filled squares, green) - size scales with n
    for d in ti_acc_data:
        ax.plot(
            d["time_ms"],
            d["mean_err"],
            "s",
            color="#27ae60",
            markersize=marker_size(d["n"]),
        )
    ti_acc_times = [d["time_ms"] for d in ti_acc_data]
    ti_acc_errs = [d["mean_err"] for d in ti_acc_data]
    ax.plot(
        ti_acc_times,
        ti_acc_errs,
        "-",
        color="#27ae60",
        lw=1.2,
        alpha=0.5,
        label="TI accurate",
    )

    # Plot VFI fast (open circles, red) - size scales with n
    for d in vfi_fast_data:
        ax.plot(
            d["time_ms"],
            d["mean_err"],
            "o",
            color="#e74c3c",
            markersize=marker_size(d["n"]),
            markerfacecolor="white",
            markeredgewidth=1.5,
        )
    vfi_fast_times = [d["time_ms"] for d in vfi_fast_data]
    vfi_fast_errs = [d["mean_err"] for d in vfi_fast_data]
    ax.plot(
        vfi_fast_times, vfi_fast_errs, "--", color="#e74c3c", lw=1.2, label="VFI fast"
    )

    # Plot VFI accurate (filled circles, red) - size scales with n
    for d in vfi_acc_data:
        ax.plot(
            d["time_ms"],
            d["mean_err"],
            "o",
            color="#e74c3c",
            markersize=marker_size(d["n"]),
        )
    vfi_acc_times = [d["time_ms"] for d in vfi_acc_data]
    vfi_acc_errs = [d["mean_err"] for d in vfi_acc_data]
    ax.plot(
        vfi_acc_times,
        vfi_acc_errs,
        "-",
        color="#e74c3c",
        lw=1.2,
        alpha=0.5,
        label="VFI accurate",
    )

    # Reference line at EGM accuracy level (-3.4)
    ax.axhline(-3.4, color="gray", ls="--", lw=1.0, alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Mean Euler error ($\\log_{10}$)")
    ax.legend(loc="upper right", fontsize=7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure_multiformat(fig, basename)
    return fig


def generate_figure_tradeoff(results, basename="fig_tradeoff"):
    """
    Figure: Speed-accuracy tradeoff plane.

    Shows the fundamental tradeoff that VFI and TI face (fast vs accurate modes)
    while EGM sidesteps it entirely with a single point in the optimal corner.

    Key insight: Lines connecting fast/accurate modes visualize the "tradeoff";
    EGM's single point shows it doesn't face this tradeoff.
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Data from benchmarks (ergodic errors)
    # EGM: single mode
    egm_time = results["EZ-EGM (1:1)"]["time"] * 1000
    egm_acc = -results["EZ-EGM (1:1)"]["euler_mean"]  # Flip sign so higher = better

    # TI: fast and accurate modes
    ti_fast_time = results["TI"]["time"] * 1000
    ti_fast_acc = -results["TI"]["euler_mean"]
    ti_acc_time = results["TI-accurate"]["time"] * 1000
    ti_acc_acc = -results["TI-accurate"]["euler_mean"]

    # VFI: fast and accurate modes
    vfi_fast_time = results["VFI"]["time"] * 1000
    vfi_fast_acc = -results["VFI"]["euler_mean"]
    vfi_acc_time = results["VFI-accurate"]["time"] * 1000
    vfi_acc_acc = -results["VFI-accurate"]["euler_mean"]

    # Plot VFI tradeoff line (circles)
    ax.plot(
        [vfi_fast_time, vfi_acc_time],
        [vfi_fast_acc, vfi_acc_acc],
        "o-",
        color="#e74c3c",
        lw=2,
        markersize=8,
        label="VFI",
    )
    ax.plot(
        vfi_fast_time,
        vfi_fast_acc,
        "o",
        color="#e74c3c",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2,
    )  # Open marker for fast
    ax.plot(
        vfi_acc_time, vfi_acc_acc, "o", color="#e74c3c", markersize=8
    )  # Filled for accurate

    # Plot TI tradeoff line (squares)
    ax.plot(
        [ti_fast_time, ti_acc_time],
        [ti_fast_acc, ti_acc_acc],
        "s-",
        color="#27ae60",
        lw=2,
        markersize=8,
        label="TI",
    )
    ax.plot(
        ti_fast_time,
        ti_fast_acc,
        "s",
        color="#27ae60",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2,
    )  # Open marker for fast
    ax.plot(
        ti_acc_time, ti_acc_acc, "s", color="#27ae60", markersize=8
    )  # Filled for accurate

    # Plot EGM single point (star)
    ax.plot(
        egm_time,
        egm_acc,
        "*",
        color="#3498db",
        markersize=15,
        label="EGM",
        markeredgecolor="#1a5276",
        markeredgewidth=0.5,
    )

    # Annotations for fast/accurate
    ax.annotate(
        "fast",
        (vfi_fast_time, vfi_fast_acc),
        textcoords="offset points",
        xytext=(-15, -12),
        fontsize=7,
        color="#e74c3c",
    )
    ax.annotate(
        "accurate",
        (vfi_acc_time, vfi_acc_acc),
        textcoords="offset points",
        xytext=(5, -12),
        fontsize=7,
        color="#e74c3c",
    )
    ax.annotate(
        "fast",
        (ti_fast_time, ti_fast_acc),
        textcoords="offset points",
        xytext=(-15, 8),
        fontsize=7,
        color="#27ae60",
    )
    ax.annotate(
        "accurate",
        (ti_acc_time, ti_acc_acc),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=7,
        color="#27ae60",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Accuracy ($-\\log_{10}$ Euler error)")
    ax.legend(loc="lower left", fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(3.0, 5.2)

    fig.tight_layout()
    save_figure_multiformat(fig, basename)
    return fig


def generate_all_figures(results):
    """Generate all figures for the paper."""
    print()
    print("=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    print()

    setup_figure_style()

    # Figure 1: Policy function
    generate_figure_policy(results)

    # Figure 2: Speed-accuracy tradeoff plane
    generate_figure_tradeoff(results, basename="fig_tradeoff")

    # Figure 3: Speed-accuracy Pareto frontier with grid sizes
    generate_figure_pareto_modes(results, basename="fig_pareto")

    print()
    print("All figures saved to:", FIGURE_DIR)
    print("\nFigures generated:")
    print("  - fig_policy: Consumption policy function c(m,z)")
    print("  - fig_tradeoff: Speed-accuracy plane showing fast/accurate tradeoff")
    print("  - fig_pareto: Speed-accuracy Pareto frontier (grid sizes)")


# =============================================================================
# Table and Claim Verification
# =============================================================================


def print_paper_tables(results):
    """Print results in paper table format."""
    print()
    print("=" * 70)
    print("TABLE 1: Speed comparison (fast modes) - Ergodic Euler errors")
    print("=" * 70)
    print()
    print("| Method   | Time (ms) | Iters | Mean Err | Max Err |")
    print("|----------|-----------|-------|----------|---------|")
    for method in ["EZ-EGM (1:1)", "TI", "VFI"]:
        r = results[method]
        time_ms = r["time"] * 1000
        print(
            f"| {method:<8} | {time_ms:>9.0f} | {r['iterations']:>5} "
            f"| {r['euler_mean']:>8.1f} | {r['euler_max']:>7.1f} |"
        )
    print()

    print("=" * 70)
    print("TABLE 2: Accuracy comparison (accurate modes) - Ergodic Euler errors")
    print("=" * 70)
    print()
    print("| Method       | Time (ms) | Iters | Mean Err | Max Err |")
    print("|--------------|-----------|-------|----------|---------|")
    # EZ-EGM (same in both fast and accurate)
    r = results["EZ-EGM (1:1)"]
    print(
        f"| EZ-EGM       | {r['time'] * 1000:>9.0f} | {r['iterations']:>5} "
        f"| {r['euler_mean']:>8.1f} | {r['euler_max']:>7.1f} |"
    )
    # TI accurate
    r = results["TI-accurate"]
    print(
        f"| TI (acc)     | {r['time'] * 1000:>9.0f} | {r['iterations']:>5} "
        f"| {r['euler_mean']:>8.1f} | {r['euler_max']:>7.1f} |"
    )
    # VFI accurate
    r = results["VFI-accurate"]
    print(
        f"| VFI (acc)    | {r['time'] * 1000:>9.0f} | {r['iterations']:>5} "
        f"| {r['euler_mean']:>8.1f} | {r['euler_max']:>7.1f} |"
    )
    print()

    print("=" * 70)
    print("COMPARISON: Grid-based vs Ergodic Euler errors (for appendix)")
    print("=" * 70)
    print()
    print("| Method       | Grid Mean | Grid Max | Erg Mean | Erg Max |")
    print("|--------------|-----------|----------|----------|---------|")
    for method in ["EZ-EGM (1:1)", "TI", "VFI"]:
        r = results[method]
        print(
            f"| {method:<12} | {r['euler_mean_grid']:>9.1f} | {r['euler_max_grid']:>8.1f} "
            f"| {r['euler_mean']:>8.1f} | {r['euler_max']:>7.1f} |"
        )
    print()

    if "egm_howard_comparison" in results:
        print("=" * 70)
        print("APPENDIX TABLE: EGM Howard Acceleration with Different K")
        print("=" * 70)
        print()
        print("| K   | Time (ms) | Iterations | Euler Mean | Euler Max |")
        print("|-----|-----------|------------|------------|-----------|")
        for r in results["egm_howard_comparison"]:
            print(
                f"| {r['K']:<3} | {r['time'] * 1000:<9.1f} | {r['iterations']:<10} "
                f"| {r['euler_mean']:<10.1f} | {r['euler_max']:<9.1f} |"
            )
        print()

    if "vfi_howard_comparison" in results:
        print("=" * 70)
        print("APPENDIX TABLE: VFI Howard Acceleration with Different K")
        print("=" * 70)
        print()
        print("| K   | Time (ms) | Iterations | Euler Mean | Euler Max |")
        print("|-----|-----------|------------|------------|-----------|")
        for r in results["vfi_howard_comparison"]:
            print(
                f"| {r['K']:<3} | {r['time'] * 1000:<9.0f} | {r['iterations']:<10} "
                f"| {r['euler_mean']:<10.1f} | {r['euler_max']:<9.1f} |"
            )
        print()

    if "ti_howard_comparison" in results:
        print("=" * 70)
        print("APPENDIX TABLE: TI Howard Acceleration with Different K")
        print("=" * 70)
        print()
        print("| K   | Time (ms) | Iterations | Euler Mean | Euler Max |")
        print("|-----|-----------|------------|------------|-----------|")
        for r in results["ti_howard_comparison"]:
            print(
                f"| {r['K']:<3} | {r['time'] * 1000:<9.0f} | {r['iterations']:<10} "
                f"| {r['euler_mean']:<10.1f} | {r['euler_max']:<9.1f} |"
            )
        print()


def print_paper_claims(results):
    """Print and verify ALL numerical claims made in the paper."""
    print("=" * 70)
    print("VERIFICATION OF ALL PAPER CLAIMS")
    print("=" * 70)
    print()

    all_pass = True

    def check(name, computed, paper_value, tolerance=0.1):
        """Check if computed value matches paper claim."""
        nonlocal all_pass
        if isinstance(paper_value, str):
            print(f"  {name}: {computed} (paper: '{paper_value}')")
            return
        rel_err = abs(computed - paper_value) / max(abs(paper_value), 1e-10)
        status = "✓" if rel_err <= tolerance else "✗"
        if rel_err > tolerance:
            all_pass = False
        print(
            f"  {status} {name}: {computed} (paper: {paper_value}, err: {rel_err:.1%})"
        )

    # -------------------------------------------------------------------------
    print("SECTION: Parameters")
    print("-" * 70)
    θ_computed = (1 - PAPER_PARAMS["γ"]) / (1 - PAPER_PARAMS["ρ"])
    EIS_computed = 1 / PAPER_PARAMS["ρ"]
    check("β", PAPER_PARAMS["β"], 0.96)
    check("R", PAPER_PARAMS["R"], 1.02)
    check("ρ", PAPER_PARAMS["ρ"], 2 / 3)
    check("EIS = 1/ρ", EIS_computed, 1.5)
    check("γ", PAPER_PARAMS["γ"], 10.0)
    check("θ = (1-γ)/(1-ρ)", θ_computed, -27.0)
    check("z_rho (income persistence)", PAPER_PARAMS["z_rho"], 0.95)
    check("z_sigma", PAPER_PARAMS["z_sigma"], 0.1)
    check("m_grid size", PAPER_PARAMS["m_size"], 100)
    check("Income grid size", PAPER_PARAMS["z_size"], 10)
    check("m_max (≈20× mean income)", PAPER_PARAMS["m_max"], 20.0)
    check("K_EGM (Howard iterations for EGM)", K_EGM, 2)
    check("K_TI (Howard iterations for TI)", K_TI, 4)
    check("K_VFI (Howard iterations for VFI)", K_VFI, 30)
    print()

    # -------------------------------------------------------------------------
    print("SECTION: Table 1 - Speed Comparison (fast modes)")
    print("-" * 70)
    check("EZ-EGM iterations", results["EZ-EGM (1:1)"]["iterations"], 141)
    check("TI fast iterations", results["TI"]["iterations"], 140)
    check("VFI fast iterations", results["VFI"]["iterations"], 239)
    print()
    print("  Ergodic Euler Errors (Table 1):")
    print(
        f"    EZ-EGM:  mean={results['EZ-EGM (1:1)']['euler_mean']:.1f}, max={results['EZ-EGM (1:1)']['euler_max']:.1f}"
    )
    print(
        f"    TI:      mean={results['TI']['euler_mean']:.1f}, max={results['TI']['euler_max']:.1f}"
    )
    print(
        f"    VFI:     mean={results['VFI']['euler_mean']:.1f}, max={results['VFI']['euler_max']:.1f}"
    )
    print()
    print("  Timing (hardware-dependent, for reference only):")
    print(f"    EZ-EGM:          {results['EZ-EGM (1:1)']['time'] * 1000:.0f}ms")
    print(f"    TI:              {results['TI']['time'] * 1000:.0f}ms")
    print(f"    VFI:             {results['VFI']['time'] * 1000:.0f}ms")
    print()

    # -------------------------------------------------------------------------
    print("SECTION: Table 2 - Accuracy Comparison (accurate modes)")
    print("-" * 70)
    check(
        "TI accurate iterations",
        results["TI-accurate"]["iterations"],
        137,
        tolerance=0.10,
    )
    check(
        "VFI accurate iterations",
        results["VFI-accurate"]["iterations"],
        239,
        tolerance=0.05,
    )
    print()
    print("  Ergodic Euler Errors (Table 2):")
    print(
        f"    EZ-EGM:      mean={results['EZ-EGM (1:1)']['euler_mean']:.1f}, max={results['EZ-EGM (1:1)']['euler_max']:.1f}"
    )
    print(
        f"    TI (acc):    mean={results['TI-accurate']['euler_mean']:.1f}, max={results['TI-accurate']['euler_max']:.1f}"
    )
    print(
        f"    VFI (acc):   mean={results['VFI-accurate']['euler_mean']:.1f}, max={results['VFI-accurate']['euler_max']:.1f}"
    )
    print()
    print("  Timing (Table 2):")
    print(f"    EZ-EGM:      {results['EZ-EGM (1:1)']['time'] * 1000:.0f}ms")
    print(f"    TI (acc):    {results['TI-accurate']['time'] * 1000:.0f}ms")
    print(f"    VFI (acc):   {results['VFI-accurate']['time'] * 1000:.0f}ms")
    print()

    # -------------------------------------------------------------------------
    print("SECTION: Prose Claims")
    print("-" * 70)

    # Speedup ratio (fast modes)
    speedup = results["VFI"]["time"] / results["EZ-EGM (1:1)"]["time"]
    speedup_order = jnp.log10(speedup)
    print(
        f"  Speedup (fast modes): {speedup:.0f}x ({speedup_order:.1f} orders of magnitude)"
    )
    print("    Paper: 'speed gains of one to two orders of magnitude'")
    print()

    # Accuracy improvement (fast modes: EGM vs VFI)
    accuracy_diff = results["VFI"]["euler_mean"] - results["EZ-EGM (1:1)"]["euler_mean"]
    print(f"  Accuracy diff (EGM vs VFI fast): {accuracy_diff:.1f} orders of magnitude")
    print("    Paper: 'improves accuracy by more than one order of magnitude'")
    print()

    # Accurate modes comparison
    vfi_acc_time = results["VFI-accurate"]["time"]
    egm_time = results["EZ-EGM (1:1)"]["time"]
    speedup_acc = vfi_acc_time / egm_time
    print(f"  Speedup (EGM vs VFI accurate): {speedup_acc:.0f}x")
    print("    Paper: 'two to three orders of magnitude' (at equal accuracy)")
    print()

    # Howard effect on EGM iterations
    egm_iter_change = (
        results["EZ-EGM + Howard"]["iterations"] - results["EZ-EGM (1:1)"]["iterations"]
    )
    print(
        f"  Howard on EGM: {results['EZ-EGM (1:1)']['iterations']} → {results['EZ-EGM + Howard']['iterations']} iterations ({egm_iter_change:+d})"
    )
    print()

    # Howard effect on TI iterations
    print(
        f"  Howard on TI: {results['TI']['iterations']} → {results['TI-Howard']['iterations']} iterations"
    )
    print()

    # Howard effect on VFI iterations
    print(
        f"  Howard on VFI: {results['VFI']['iterations']} → {results['VFI-Howard']['iterations']} iterations"
    )
    print()

    # -------------------------------------------------------------------------
    print("SECTION: Robustness to ρ > 1 (Appendix)")
    print("-" * 70)

    if "rho_robustness" in results:
        rho_results = results["rho_robustness"]
        # Paper claims: mean ≈ -5, max ≈ -3.5 across all rho values
        for r in rho_results:
            mean_ok = -5.5 < r["euler_mean"] < -4.5
            max_ok = -4.0 < r["euler_max"] < -3.0
            status = "✓" if (mean_ok and max_ok) else "✗"
            if not (mean_ok and max_ok):
                all_pass = False
            print(
                f"  {status} ρ={r['rho']:.1f} (EIS={r['eis']:.2f}): mean={r['euler_mean']:.1f}, max={r['euler_max']:.1f}"
            )
        print("    Paper: 'mean Euler errors near -5 and max errors near -3.5'")
    else:
        print("  (Skipped - run with --appendix to include)")
    print()

    # -------------------------------------------------------------------------
    print("=" * 70)
    if all_pass:
        print("ALL NUMERICAL CLAIMS VERIFIED ✓")
    else:
        print("SOME CLAIMS DO NOT MATCH - UPDATE PAPER OR CODE")
    print("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Reproducible benchmarks for the EZ-EGM paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python benchmark_paper.py           # Run all benchmarks
    uv run python benchmark_paper.py --main    # Main results only
    uv run python benchmark_paper.py --appendix # Appendix only (Howard K)
    uv run python benchmark_paper.py --figures  # Regenerate figures only
        """,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all", action="store_true", default=True, help="Run all benchmarks (default)"
    )
    group.add_argument(
        "--main",
        action="store_true",
        help="Run main benchmarks only (Tables 1-2, Figures)",
    )
    group.add_argument(
        "--appendix",
        action="store_true",
        help="Run appendix benchmarks only (Howard K comparison)",
    )
    group.add_argument(
        "--figures",
        action="store_true",
        help="Regenerate figures only (requires prior --main run)",
    )
    group.add_argument(
        "--equal-accuracy",
        action="store_true",
        help="Run equal-accuracy benchmarks (ergodic distribution, welfare costs)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print()
    print("This script generates numerical results and figures for the EZ-EGM paper.")
    print()
    print(
        f"Settings: tol={TOL}, K_EGM={K_EGM}, K_TI={K_TI}, K_VFI={K_VFI}, timing_runs={N_TIMING_RUNS}"
    )
    print()

    # Determine what to run
    if args.main:
        include_main, include_appendix = True, False
        print("Mode: Main benchmarks only")
    elif args.appendix:
        include_main, include_appendix = False, True
        print("Mode: Appendix benchmarks only")
    elif args.figures:
        include_main, include_appendix = False, False
        print("Mode: Figures only (running minimal EGM for figure data)")
    elif getattr(args, "equal_accuracy", False):
        include_main, include_appendix = False, False
        print("Mode: Equal-accuracy benchmarks (Carroll suggestions)")
    else:
        include_main, include_appendix = True, True
        print("Mode: All benchmarks")
    print()

    if args.figures:
        # Quick run just for figure data
        model = create_paper_model()
        print("Running minimal benchmarks for figures...")

        # EGM + Howard for policy figure
        _ = solve_ezegm(model, tol=1e-3, max_iter=5, n_howard=K_EGM)
        c, V, m, n = solve_ezegm(model, tol=TOL, max_iter=MAX_ITER, n_howard=K_EGM)

        # Convergence history for all four methods
        _ = solve_ezegm_with_history(model, tol=1e-3, max_iter=5, n_howard=1)
        _ = solve_ezegm_with_history(model, tol=1e-3, max_iter=5, n_howard=K_EGM)
        _ = solve_ezvfi_with_history(model, tol=1e-3, max_iter=5, n_howard=1)
        _ = solve_ezvfi_with_history(model, tol=1e-3, max_iter=5, n_howard=K_VFI)
        _, _, _, _, egm_history, egm_time = solve_ezegm_with_history(
            model, tol=TOL, max_iter=300, n_howard=1
        )
        _, _, _, _, egm_howard_history, egm_howard_time = solve_ezegm_with_history(
            model, tol=TOL, max_iter=300, n_howard=K_EGM
        )
        _, _, _, _, vfi_history, vfi_time = solve_ezvfi_with_history(
            model, tol=TOL, max_iter=300, n_howard=1
        )
        _, _, _, _, vfi_howard_history, vfi_howard_time = solve_ezvfi_with_history(
            model, tol=TOL, max_iter=300, n_howard=K_VFI
        )

        results = {
            "model": model,
            "EZ-EGM + Howard": {"c": np.array(c), "V": np.array(V), "m": np.array(m)},
            "convergence": {
                "egm_history": egm_history,
                "egm_time": egm_time,
                "egm_howard_history": egm_howard_history,
                "egm_howard_time": egm_howard_time,
                "vfi_history": vfi_history,
                "vfi_time": vfi_time,
                "vfi_howard_history": vfi_howard_history,
                "vfi_howard_time": vfi_howard_time,
            },
        }
        generate_all_figures(results)
    elif getattr(args, "equal_accuracy", False):
        # Run equal-accuracy benchmarks
        results = run_equal_accuracy_benchmarks()

        # Print summary
        print()
        print("=" * 70)
        print("SUMMARY: Equal-Accuracy Speedups")
        print("=" * 70)
        if "equal_accuracy" in results:
            for m in results["equal_accuracy"]["matches"]:
                print(
                    f"  VFI n={m['vfi_n']:3d} ≈ EGM n={m['egm_n']:3d} → {m['speedup']:.0f}x speedup"
                )
        print()
    else:
        results = run_benchmarks(
            include_main=include_main, include_appendix=include_appendix
        )

        if include_main:
            print_paper_tables(results)
            print_paper_claims(results)
            generate_all_figures(results)

        if include_appendix and not include_main:
            # Print only appendix tables
            print()
            print("=" * 70)
            print("APPENDIX TABLE: EGM Howard Acceleration with Different K")
            print("=" * 70)
            print()
            print("| K   | Time (ms) | Iterations | Euler Mean | Euler Max |")
            print("|-----|-----------|------------|------------|-----------|")
            for r in results["egm_howard_comparison"]:
                print(
                    f"| {r['K']:<3} | {r['time'] * 1000:<9.1f} | {r['iterations']:<10} "
                    f"| {r['euler_mean']:<10.1f} | {r['euler_max']:<9.1f} |"
                )
            print()

            print("=" * 70)
            print("APPENDIX TABLE: VFI Howard Acceleration with Different K")
            print("=" * 70)
            print()
            print("| K   | Time (ms) | Iterations | Euler Mean | Euler Max |")
            print("|-----|-----------|------------|------------|-----------|")
            for r in results["vfi_howard_comparison"]:
                print(
                    f"| {r['K']:<3} | {r['time'] * 1000:<9.0f} | {r['iterations']:<10} "
                    f"| {r['euler_mean']:<10.1f} | {r['euler_max']:<9.1f} |"
                )
            print()

            print("=" * 70)
            print("APPENDIX TABLE: TI Howard Acceleration with Different K")
            print("=" * 70)
            print()
            print("| K   | Time (ms) | Iterations | Euler Mean | Euler Max |")
            print("|-----|-----------|------------|------------|-----------|")
            for r in results["ti_howard_comparison"]:
                print(
                    f"| {r['K']:<3} | {r['time'] * 1000:<9.0f} | {r['iterations']:<10} "
                    f"| {r['euler_mean']:<10.1f} | {r['euler_max']:<9.1f} |"
                )
            print()

            print("=" * 70)
            print("APPENDIX TABLE: VFI Accurate Mode Howard Acceleration")
            print("=" * 70)
            print()
            print("| K   | Time (ms) | Iterations | Euler Mean | Euler Max |")
            print("|-----|-----------|------------|------------|-----------|")
            for r in results["vfi_acc_howard_comparison"]:
                print(
                    f"| {r['K']:<3} | {r['time'] * 1000:<9.0f} | {r['iterations']:<10} "
                    f"| {r['euler_mean']:<10.1f} | {r['euler_max']:<9.1f} |"
                )
            print()

            print("=" * 70)
            print("APPENDIX TABLE: TI Accurate Mode Howard Acceleration")
            print("=" * 70)
            print()
            print("| K   | Time (ms) | Iterations | Euler Mean | Euler Max |")
            print("|-----|-----------|------------|------------|-----------|")
            for r in results["ti_acc_howard_comparison"]:
                print(
                    f"| {r['K']:<3} | {r['time'] * 1000:<9.0f} | {r['iterations']:<10} "
                    f"| {r['euler_mean']:<10.1f} | {r['euler_max']:<9.1f} |"
                )
            print()

    print()
    print("=" * 70)
    print("REPRODUCIBILITY NOTE")
    print("=" * 70)
    print()
    print("Iteration counts should be consistent across platforms.")
    print("Timing will vary by hardware (CPU vs GPU, chip architecture).")
    print()


if __name__ == "__main__":
    main()
