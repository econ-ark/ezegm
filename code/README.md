# EZ-EGM Code

Implementation of the Endogenous Grid Method for Epstein-Zin Preferences.

## Files

| File | Description |
|------|-------------|
| `ez_egm.py` | Core EZ-EGM algorithm implementation |
| `utils.py` | Utility functions (golden section search, constants) |
| `benchmark_paper.py` | Generates all paper results (tables, figures) |

## ez_egm.py

Implements the EZ-EGM algorithm. Key functions:

- `create_ez_model()` - Create model with default Bansal-Yaron (2004) parameters
- `solve_ez_egm()` - Solve with 1:1 policy/value updates
- `solve_ez_egm_howard()` - Solve with Howard acceleration (K value updates per policy update)
- `solve_ez_vfi()` - Value function iteration baseline (golden section search)
- `compute_euler_errors()` - Compute normalized Euler equation errors

The algorithm stores V (not W = V^{1-ρ}) for numerical stability, converting internally as needed.

## benchmark_paper.py

Reproduces all numerical results in the paper.

```bash
uv run python code/benchmark_paper.py          # All benchmarks
uv run python code/benchmark_paper.py --main   # Tables 1-2, figures
uv run python code/benchmark_paper.py --appendix  # Howard K comparison
uv run python code/benchmark_paper.py --figures   # Figures only
```

Outputs timing comparisons, Euler errors, and generates figures: `fig_policy.pdf`, `fig_pareto.pdf`, `fig_tradeoff.pdf`.

