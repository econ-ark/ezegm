---
title: Online Appendix
---

+++ {"part": "abstract"}
+++

This online appendix supplements the main text with algorithmic details and additional benchmarks. It presents pseudocode for value function iteration (VFI) and time iteration (TI), documents the speed-accuracy tradeoff these methods face, reports results with Howard policy improvement acceleration, provides equal-accuracy comparisons that hold solution quality constant while measuring computational cost, compares Euler error evaluation methods on the exogenous grid versus the ergodic distribution, and discusses the case $\rho > 1$ (EIS $< 1$) where the transformed problem becomes a minimization.

(appendix-algorithms)=
# Appendix: Alternative algorithms

Both value function iteration (VFI) and time iteration (TI) require numerical search at each grid point, unlike EGM which inverts the Euler equation analytically. Both methods face an inherent tradeoff: the certainty equivalent $\mu$ must be evaluated at candidate consumption values during search. Two approaches exist: compute $\mu$ exactly at each candidate (`accurate` mode), or precompute $\mu$ on the asset grid and interpolate during search (`fast` mode).

(appendix-vfi)=


**Value function iteration.** VFI finds optimal consumption by maximizing the Bellman equation directly:
```{math}
V(m, z) = \max_c \left[ (1-\beta) c^{1-\rho} + \beta \mu(m-c, z)^{1-\rho} \right]^{1/(1-\rho)}
```
where $\mu(a, z) = \left( \mathbb{E}[V(Ra + y(z'))^{1-\gamma} | z] \right)^{1/(1-\gamma)}$ is the certainty equivalent. Golden-section search evaluates $\mu$ at many candidate consumption values.

:::{raw} latex
\begin{algorithm}
\caption{VFI for Epstein-Zin}\label{alg:vfi-ez}
\begin{algorithmic}[1]
\Require Exogenous grid $\{m_i\}_{i=1}^I$, asset grid $\{a_j\}_{j=1}^J$, initial guess $V^{(0)}$, mode $\in \{\texttt{accurate}, \texttt{fast}\}$
\Ensure Converged value $V(\cdot, \cdot)$ and policy $c(\cdot, \cdot)$
\State Given $V^{(n)}$ from iteration $n$
\If{mode = \texttt{fast}}
    \State Precompute $\mu_j = (\sum_\ell \pi_{k\ell} V^{(n)}(Ra_j + y(z'_\ell))^{1-\gamma})^{1/(1-\gamma)}$ for all $a_j$
\EndIf
\For{each income state $z_k$ and each $m_i$}
    \State Define objective $f(c) = [(1-\beta) c^{1-\rho} + \beta \mu(m_i - c, z_k)^{1-\rho}]^{1/(1-\rho)}$
    \If{mode = \texttt{accurate}}
        \State Compute $\mu$ by interpolating $V^{(n)}$ at $m' = R(m_i - c) + y(z')$, then taking expectation
    \Else
        \State Interpolate $\mu$ from precomputed $\{\mu_j\}$ at $a = m_i - c$
    \EndIf
    \State Maximize $f(c)$ via golden-section search to get $c^{(n+1)}(m_i, z_k)$ and $V^{(n+1)}(m_i, z_k)$
\EndFor
\State \textbf{Iterate} until $\|V^{(n)} - V^{(n-1)}\| < \varepsilon$
\end{algorithmic}
\end{algorithm}
:::

(appendix-ti)=
**Time iteration.** Time iteration (TI), introduced by {cite:t}`coleman1991equilibrium`, works directly with the Euler equation rather than the Bellman equation. Like EGM, TI exploits the first-order condition; unlike EGM, it cannot invert the Euler equation analytically and must use numerical root-finding. {cite:t}`rendahl2015inequality` proves that under standard conditions, TI converges to the same solution as VFI. {cite:t}`coeurdacier2020financial` extend TI to Epstein-Zin preferences, treating consumption, value, and expected value as simultaneous controls; our accurate mode follows this approach by interpolating decision rules at each candidate consumption during root-finding.

Given current guesses $(c^{(n)}, V^{(n)})$, TI finds consumption at each $(m, z)$ by solving the Euler equation via bisection, then updates the value function. The certainty equivalent uses $V$ directly: $\mu(a, z) = (\mathbb{E}[V(m', z')^{1-\gamma}])^{1/(1-\gamma)}$, exactly as in VFI.

:::{raw} latex
\begin{algorithm}
\caption{Time Iteration for Epstein-Zin}\label{alg:ti-ez}
\begin{algorithmic}[1]
\Require Exogenous grid $\{m_i\}_{i=1}^I$, asset grid $\{a_j\}_{j=1}^J$, initial guess $(c^{(0)}, V^{(0)})$, mode $\in \{\texttt{accurate}, \texttt{fast}\}$
\Ensure Converged policy $c(\cdot, \cdot)$ and value $V(\cdot, \cdot)$
\State Given $(c^{(n)}, V^{(n)})$ from iteration $n$
\State Precompute $c^{(n)}$ and $V^{(n)}$ at $m' = Ra_j + y(z'_\ell)$ for all $(a_j, z'_\ell)$
\If{mode = \texttt{fast}}
    \State Precompute $\mu_j = (\sum_\ell \pi_{k\ell} V^{(n)}(m'_{j\ell})^{1-\gamma})^{1/(1-\gamma)}$ for all $a_j$
\EndIf
\For{each income state $z_k$ and each $m_i$}
    \State Define Euler residual $r(c)$ using $\mu$ and marginal utilities
    \If{mode = \texttt{accurate}}
        \State Compute $\mu$ by interpolating $V^{(n)}$ at $m' = R(m_i - c) + y(z')$, then taking expectation
    \Else
        \State Interpolate $\mu$ from precomputed $\{\mu_j\}$ at $a = m_i - c$
    \EndIf
    \State Solve $r(c) = 0$ via bisection to get $c^{(n+1)}(m_i, z_k)$
    \State Update $V^{(n+1)}(m_i, z_k)$ using the Bellman equation
\EndFor
\State \textbf{Iterate} until $\|c^{(n)} - c^{(n-1)}\| < \varepsilon$
\end{algorithmic}
\end{algorithm}
:::

(appendix-tradeoff)=
# Appendix: Speed-accuracy tradeoff

:::{raw} latex
\begin{table}[ht]
\centering
\caption{Accurate-mode comparison: EZ-EGM vs numerical search ($K=1$)}
\label{tab:tradeoff}
\begin{tabular}{@{}lrrr@{}}
\toprule
Method & Time (ms) & Euler Mean & Euler Max \\
\midrule
EZ-EGM & 25 & $-4.8$ & $-3.4$ \\
TI (accurate) & 2371 & $-4.8$ & $-3.4$ \\
VFI (accurate) & 8385 & $-3.5$ & $-2.3$ \\
\bottomrule
\end{tabular}

{\footnotesize \textit{Note:} Grid size 100, $K=1$ (no Howard acceleration). Accurate modes compute $\mu$ exactly during search (no precomputation). Errors evaluated on uniform grid ($\bar{\varepsilon}_G$); see the main text for ergodic errors ($\bar{\varepsilon}_\pi$).}
\end{table}
:::

[](#tab:tradeoff) reveals two main findings. First, TI-accurate achieves the same accuracy as EZ-EGM (mean Euler error $-4.8$), confirming that when $\mu$ is computed exactly during search, time iteration matches EGM's precision. Second, this accuracy comes at substantial cost: TI-accurate is nearly 100 times slower than EZ-EGM, because bisection requires many $\mu$ evaluations per grid point, each involving interpolation and expectation. VFI-accurate is slower still (over 300 times) yet less accurate than both EGM and TI, reflecting the underlying disadvantage of optimizing over the Bellman equation rather than working with the Euler equation.

EGM's advantage is that it achieves accurate-mode precision at fast-mode speed by avoiding numerical search entirely. The main text compares EGM to the fast modes of VFI and TI, which represent practical implementations; this appendix shows that even the accurate modes cannot match EGM's speed.



(appendix-howard)=
# Appendix: Howard acceleration parameter

The main text reports baseline results without Howard acceleration ($K=1$). This appendix examines how the choice of $K$ affects performance in both fast and accurate modes.

Note on iteration counts: For EGM, each "policy iteration" consists of one Euler equation inversion (the EGM step) plus up to $K$ value function updates. For TI, each policy iteration consists of one bisection solve at every grid point plus up to $K$ value updates. For VFI, each "policy iteration" consists of one numerical optimization over the consumption grid plus up to $K$ value function updates. All three methods use early termination: if the value function converges before $K$ updates, the remaining updates are skipped. The EGM policy step is analytic and thus much cheaper than VFI's optimization or TI's bisection.

**EGM.**  EGM has no fast/accurate distinction because it evaluates $\mu$ exactly on the endogenous grid.

:::{raw} latex
\begin{table}[ht]
\centering
\caption{EZ-EGM: Effect of Howard acceleration parameter $K$}
\label{tab:howard-egm}
\begin{tabular}{@{}rrrcc@{}}
\toprule
$K$ & Time (ms) & Policy Iters & Euler Mean & Euler Max \\
\midrule
1 & 25 & 141 & $-4.8$ & $-3.4$ \\
2 & 49 & 99 & $-4.9$ & $-3.4$ \\
3 & 62 & 86 & $-4.9$ & $-3.4$ \\
4 & 70 & 78 & $-4.9$ & $-3.4$ \\
5 & 78 & 70 & $-4.9$ & $-3.4$ \\
\bottomrule
\end{tabular}
\end{table}
:::

For EZ-EGM, $K=1$ is fastest. Additional value iterations reduce policy iterations (from 141 to 70) but increase total time because the EGM policy step is already cheap and accurate. The slight accuracy improvement ($-4.8$ to $-4.9$) does not justify the 3x time increase.

**VFI.** VFI benefits substantially from Howard acceleration in both modes. The optimal $K \approx 30$-$40$ reduces time by 7x in fast mode and 13x in accurate mode.

:::{raw} latex
\begin{table}[ht]
\centering
\caption{VFI fast mode: Effect of Howard acceleration parameter $K$}
\label{tab:howard-vfi-fast}
\begin{tabular}{@{}rrrcc@{}}
\toprule
$K$ & Time (ms) & Policy Iters & Euler Mean & Euler Max \\
\midrule
1 & 1190 & 239 & $-3.3$ & $-2.4$ \\
10 & 247 & 31 & $-3.3$ & $-2.4$ \\
20 & 183 & 16 & $-3.3$ & $-2.4$ \\
30 & 181 & 11 & $-3.3$ & $-2.4$ \\
40 & 177 & 9 & $-3.3$ & $-2.4$ \\
50 & 193 & 8 & $-3.3$ & $-2.4$ \\
\bottomrule
\end{tabular}
\end{table}
:::

:::{raw} latex
\begin{table}[ht]
\centering
\caption{VFI accurate mode: Effect of Howard acceleration parameter $K$}
\label{tab:howard-vfi-accurate}
\begin{tabular}{@{}rrrcc@{}}
\toprule
$K$ & Time (ms) & Policy Iters & Euler Mean & Euler Max \\
\midrule
1 & 8385 & 239 & $-3.5$ & $-2.3$ \\
10 & 1312 & 31 & $-3.5$ & $-2.3$ \\
20 & 802 & 16 & $-3.5$ & $-2.3$ \\
30 & 641 & 11 & $-3.5$ & $-2.3$ \\
40 & 626 & 10 & $-3.5$ & $-2.3$ \\
50 & 650 & 10 & $-3.5$ & $-2.3$ \\
\bottomrule
\end{tabular}
\end{table}
:::

VFI-accurate benefits more from Howard than VFI-fast because each policy optimization is more expensive in accurate mode (computing $\mu$ exactly versus interpolating). At $K=40$, VFI-accurate (626ms) is faster than VFI-fast at $K=1$ (1190ms), though still less accurate ($-3.5$ versus $-3.3$). This reflects the cost structure: golden-section search is the bottleneck, and reducing policy iterations helps more when each policy step is expensive.

**TI.** TI shows a distinctive pattern: moderate $K$ helps, but larger values hurt performance. The optimal $K$ differs between modes.


:::{raw} latex
\begin{table}[ht]
\centering
\caption{TI fast mode: Effect of Howard acceleration parameter $K$}
\label{tab:howard-ti-fast}
\begin{tabular}{@{}rrrcc@{}}
\toprule
$K$ & Time (ms) & Policy Iters & Euler Mean & Euler Max \\
\midrule
1 & 252 & 140 & $-3.2$ & $-2.7$ \\
2 & 229 & 100 & $-3.2$ & $-2.7$ \\
3 & 213 & 88 & $-3.2$ & $-2.7$ \\
4 & 213 & 81 & $-3.2$ & $-2.7$ \\
5 & 481 & 174 & $-3.2$ & $-2.7$ \\
\bottomrule
\end{tabular}
\end{table}
:::

:::{raw} latex
\begin{table}[ht]
\centering
\caption{TI accurate mode: Effect of Howard acceleration parameter $K$}
\label{tab:howard-ti-accurate}
\begin{tabular}{@{}rrrcc@{}}
\toprule
$K$ & Time (ms) & Policy Iters & Euler Mean & Euler Max \\
\midrule
1 & 2371 & 141 & $-4.8$ & $-3.4$ \\
2 & 1716 & 98 & $-4.9$ & $-3.4$ \\
3 & 1540 & 86 & $-4.9$ & $-3.4$ \\
4 & 1427 & 78 & $-4.9$ & $-3.4$ \\
5 & 1290 & 70 & $-4.9$ & $-3.4$ \\
\bottomrule
\end{tabular}
\end{table}
:::

TI-fast is optimal at $K=3$-$4$, with time falling from 252ms to 213ms. Beyond $K=4$, performance degrades sharply: $K=5$ nearly doubles time to 481ms. This instability occurs because TI's value function updates can overshoot when the policy is held fixed, destabilizing convergence.

TI-accurate is optimal at $K=5$, with time falling from 2371ms to 1290ms, a 1.8x speedup. The higher optimal $K$ reflects the greater cost of each bisection step in accurate mode.

The contrast with VFI is instructive: VFI benefits from large $K$ (up to 30-50) because golden-section search produces noisier policy updates, requiring more value iterations to stabilize. TI's bisection produces more accurate policies, reducing the benefit of additional value iterations and causing instability at high $K$.




(appendix-equal-accuracy)=
# Appendix: Equal-accuracy comparison

The main text compares EZ-EGM and VFI at fixed grid size (100 points), where EGM is both faster and more accurate. A more informative comparison holds accuracy constant: what grid size does VFI require to match EGM's Euler errors, and how do solve times compare?

We find grid sizes where mean Euler errors are approximately equal:

:::{raw} latex
\begin{table}[ht]
\centering
\caption{Speed comparison at equal accuracy}
\label{tab:equal-accuracy}
\begin{tabular}{@{}rrrrrr@{}}
\toprule
VFI $n$ & EGM $n$ & Mean Error & VFI (ms) & EGM (ms) & Speedup \\
\midrule
50 & 20 & $-3.2$ & 672 & 5 & 145$\times$ \\
100 & 20 & $-3.3$ & 1214 & 5 & 261$\times$ \\
150 & 20 & $-3.4$ & 1793 & 5 & 386$\times$ \\
200 & 20 & $-3.5$ & 2281 & 5 & 490$\times$ \\
300 & 25 & $-3.6$ & 3433 & 5 & 627$\times$ \\
\bottomrule
\end{tabular}
\end{table}
:::

At equivalent accuracy levels, EZ-EGM is 150–630 times faster than VFI. The speedup grows with target accuracy: to achieve mean Euler error near $-3.6$, VFI requires 300 grid points while EGM needs only 25. Euler errors weighted by the ergodic distribution (5th–95th percentiles of simulated wealth) are similar to uniformly-weighted errors; the ergodic distribution concentrates wealth in the lower range (median around 3 versus grid maximum of 20), but both weighting schemes yield mean errors near $-4.9$.

These results use $\beta R = 0.98 < 1$, so agents converge to finite target wealth. For more impatient agents ($\beta R \approx 0.92$), target wealth falls but accuracy remains similar. For very patient agents ($\beta R \geq 1$), target wealth diverges and the grid must be extended accordingly; the accuracy comparison in that regime warrants separate investigation.

(appendix-euler-errors)=
# Appendix: Euler error evaluation methods

Two approaches exist for evaluating Euler equation errors: uniform grid evaluation and ergodic distribution weighting; we report both for completeness.

**Standard approach (Santos 2000).** {cite:t}`santos2000accuracy` establishes the theoretical foundation for using Euler equation residuals to bound policy function errors. The standard implementation evaluates errors on a uniform grid spanning the state space, typically excluding boundary regions where the constraint binds. For our baseline (100 grid points), we evaluate at the 10th–90th percentiles of the grid:
```{math}
\bar{\varepsilon}_G = \frac{1}{|G|} \sum_{(m,z) \in G} \varepsilon(m, z)
```
where $G \subset \mathcal{M} \times \mathcal{Z}$ is the set of interior grid points.

**Ergodic distribution approach.** An alternative evaluates errors at wealth levels agents actually visit in the long run. We simulate the ergodic distribution (10,000 agents, 500 periods, 200 burn-in) and compute errors at the 5th–95th percentiles of realized wealth:
```{math}
\bar{\varepsilon}_\pi = \frac{1}{|S|} \sum_{(m,z) \in S} \varepsilon(m, z)
```
where $S \sim \pi$ is a sample from the stationary distribution $\pi(m,z)$. This approach weights accuracy by economic relevance: errors at rarely-visited wealth levels matter less than errors where agents spend time.

:::{raw} latex
\begin{table}[ht]
\centering
\caption{Euler errors by evaluation method ($\log_{10}$)}
\label{tab:euler-methods}
\begin{tabular}{@{}lrrrr@{}}
\toprule
 & \multicolumn{2}{c}{$\bar{\varepsilon}_G$ (grid)} & \multicolumn{2}{c}{$\bar{\varepsilon}_\pi$ (ergodic)} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Method & Mean & Max & Mean & Max \\
\midrule
EZ-EGM & $-4.8$ & $-3.4$ & $-4.8$ & $-3.2$ \\
TI & $-3.2$ & $-2.7$ & $-3.6$ & $-2.7$ \\
VFI & $-3.3$ & $-2.4$ & $-3.3$ & $-2.3$ \\
\bottomrule
\end{tabular}

{\footnotesize \textit{Note:} $\bar{\varepsilon}_G$ evaluates at 10th-90th percentiles of grid ($m \in [0.4, 16.5]$). $\bar{\varepsilon}_\pi$ evaluates at 5th-95th percentiles of simulated wealth ($m \in [0.7, 9.6]$); median wealth $\approx 3$. Errors exclude constrained points where the Euler equation holds as inequality. All methods use fast mode (precomputed $\mu$).}
\end{table}
:::

For this calibration ($\beta R = 0.98 < 1$), the stationary distribution $\pi$ concentrates in the lower wealth range: the median is approximately $m = 3$ versus the grid maximum of 20. Despite this difference in evaluation regions, EGM yields similar error statistics under both $\bar{\varepsilon}_G$ and $\bar{\varepsilon}_\pi$. TI shows improved accuracy under $\bar{\varepsilon}_\pi$ ($-3.6$ versus $-3.2$), suggesting its errors concentrate at high wealth levels that agents rarely visit.

EGM achieves the best accuracy because it satisfies the Euler equation by construction at the endogenous grid points; interpolation introduces error, but linear interpolation on a fine grid (100 points) keeps this small. TI and VFI both require numerical search, which compounds interpolation error. VFI shows the worst max error under $\bar{\varepsilon}_\pi$ ($-2.3$), reflecting that its optimization-based approach is less precise at low wealth levels where the policy function has higher curvature and agents spend the most time.

(appendix-rho)=
# Appendix: Robustness to $\rho > 1$

The baseline uses $\rho = 2/3$ (EIS $= 1.5$), but the algorithm applies equally to $\rho > 1$ (EIS $< 1$). We verify this by solving the same model with $\rho \in \{0.5, 0.9, 1.1, 1.5, 2, 3\}$, holding $\gamma = 10$ fixed. Across all values, EZ-EGM achieves mean Euler errors near $-5$ and max errors near $-3.5$ (in $\log_{10}$ units).  The case $\rho > 1$ is empirically relevant: the meta-analysis of {cite:t}`havranek2015measuring` finds that micro estimates of the EIS often fall below unity.

When $\rho > 1$, the transformation $W = V^{1-\rho}$ is decreasing, so maximizing $V$ is equivalent to minimizing $W$. No algorithmic modification is required: the first-order condition $\partial W / \partial c = 0$ is sufficient and characterizes the optimum regardless of whether it is a maximum or minimum, and EGM inverts the Euler equation analytically. VFI is unaffected because it works with $V$ directly; the CES aggregator remains increasing in $c$ for all $\rho \neq 1$. Time iteration, like EGM, works with the Euler equation.