+++ {"part": "abstract"}
The endogenous grid method (EGM) accelerates dynamic programming by inverting the Euler equation, but it appears incompatible with Epstein-Zin preferences where the value function enters the Euler equation. This paper shows that a power transformation resolves the difficulty. The resulting algorithm requires no root-finding, achieves speed gains of one to two orders of magnitude over value function iteration, and improves accuracy by more than one order of magnitude. Holding accuracy constant, the speedup is two to three orders of magnitude. VFI and time iteration face a speed-accuracy tradeoff; EGM sidesteps it entirely.
+++

# Introduction

The endogenous grid method (EGM) of {cite:t}`carroll2006method` inverts the Euler equation to obtain consumption as a function of end-of-period assets, eliminating root-finding and accelerating computation by orders of magnitude. EGM is now widely used in structural estimation of consumption-savings problems, including models of precautionary saving and liquidity constraints in the tradition of {cite:t}`Deaton1991`. {cite:t}`baraborrillasfernandez2007` extend the method to multiple controls; {cite:t}`iskhakov2017endogenous` handle discrete-continuous choice.

Separately, Epstein-Zin preferences, developed by {cite:t}`epstein1989substitution`, {cite:t}`epstein1991substitution`, and {cite:t}`weil1989equity`, have become standard in macro-finance for their ability to separate risk aversion from intertemporal substitution. Building on the temporal lottery framework of {cite:t}`krepsporteus1978`, these preferences are theoretically well-grounded: {cite:t}`duffieepstein1992` establish existence and uniqueness in continuous time; {cite:t}`ren2020dynamic` provide modern discrete-time foundations. Yet EGM has not been applied to Epstein-Zin preferences. The apparent obstacle is that the value function enters the Euler equation. Existing approaches based on time iteration, such as {cite:t}`coeurdacier2020financial`, treat consumption, value, and expected value as simultaneous control variables, requiring numerical root-finding on a system of non-linear equations at every grid point in every iteration.

This paper shows that such computational expense is unnecessary. A power transformation converts the Epstein-Zin recursion into a form where EGM applies directly. The algorithm tracks two functions instead of one (consumption and transformed value) but requires no root-finding and maintains the standard EGM structure. The method extends to other recursive preferences used in dynamic models, including the risk-sensitive preferences of {cite:t}`hansensargent1995`.


# Model


```{prf:definition} Consumption-Savings Problem with Epstein-Zin Preferences
:label: def-model

An agent solves
$$V(m, z) = \max_c \left[ (1-\beta) c^{1-\rho} + \beta \left( \mathbb{E}_{z'|z} \left[ V(m', z')^{1-\gamma} \right] \right)^{\frac{1-\rho}{1-\gamma}} \right]^{\frac{1}{1-\rho}}$$
subject to $a = m - c \geq 0$ and $m' = Ra + y(z')$, where $m$ is cash-on-hand, $c$ consumption, $a$ end-of-period assets, $R$ the gross interest rate, $z$ the current income state, and $y(z')$ income as a function of the realized state $z'$.
```

The state $z$ follows an AR(1) process with transition probabilities $\pi_{k\ell} = \Pr(z' = z_\ell \mid z = z_k)$, so the expectation $\mathbb{E}_{z'|z}$ is taken over next-period states conditional on the current state. The parameter $\rho$ governs the elasticity of intertemporal substitution ($\text{EIS} = 1/\rho$), while $\gamma$ governs risk aversion.[^ez-convention] When $\rho = \gamma$, preferences collapse to standard CRRA expected utility.

[^ez-convention]: We follow the original {cite:t}`epstein1989substitution` parameterization. An alternative convention, common in the asset pricing literature (e.g., {cite:t}`bansalyaron2004`), defines the EIS directly as $\psi = 1/\rho$ and writes the recursion with $\psi$ in place of $1/\rho$. The auxiliary parameter is then $\theta = (1-\gamma)/(1-1/\psi)$, which equals our $(1-\gamma)/(1-\rho)$.

We assume $\rho \neq 1$ and $\gamma \neq 1$ for the power transformation developed below; the limiting cases use logarithmic transformations but the method is otherwise analogous. For the infinite-horizon problem, standard conditions ensure the value function is well-defined and the consumption policy converges.[^convergence]

[^convergence]: See {cite:t}`epstein1989substitution` for existence and uniqueness conditions. With $\theta = (1-\gamma)/(1-\rho)$, a sufficient condition is $\beta R^\theta < 1$. When $\theta < 0$ (as in our calibration with $\gamma > 1 > \rho$), this becomes $\beta < R^{|\theta|}$, which holds for typical parameterizations. The recursion also requires $V > 0$; this follows by induction from the terminal condition whenever consumption remains strictly positive.

## Transformation

```{prf:definition} Transformed Value Function
:label: def-transform

The *transformed value function* is $W(m, z) \equiv V(m, z)^{1-\rho}$.
```

This transformation simplifies the certainty equivalent to a power mean.[^w-transform] {cite:t}`meyergohde2019entropy` use the same transformation to analyze model uncertainty with recursive preferences.

[^w-transform]: When $\rho > 1$, the transformation $W = V^{1-\rho}$ is decreasing, so maximizing $V$ is equivalent to minimizing $W$. In the discussion below, the Euler equation characterizes the optimum in either case, and EGM inverts it directly. See [](#appendix-rho) for details.

```{prf:definition} Certainty Equivalent
:label: def-mu

Let $\theta \equiv (1-\gamma)/(1-\rho)$. The *certainty equivalent* of next-period transformed value is
$$\mu(a, z) \equiv \left( \mathbb{E}_{z'|z} \left[ W(m', z')^{\theta} \right] \right)^{1/\theta}$$
where $m' = Ra + y(z')$.
```

The case $\theta = 1$ (equivalently $\gamma = \rho$) recovers time-separable CRRA expected utility; when $\gamma > \rho$, the agent prefers early resolution of uncertainty; when $\gamma < \rho$, late resolution.[^resolution]

[^resolution]: See {cite:t}`backus2008recursive` for a comprehensive treatment.

Raising the Bellman equation in {prf:ref}`def-model` to the power $1-\rho$ yields the transformed Bellman equation:
```{math}
:label: eq-bellman-w
W(m, z) = \max_c \left[ (1-\beta) c^{1-\rho} + \beta \mu(m-c, z) \right]
```

The transformation succeeds because it converts the CES aggregator in {prf:ref}`def-model` into an additive structure. The original Bellman equation involves $V$ raised to fractional powers both inside and outside the expectation; the transformed equation {eq}`eq-bellman-w` is additive, with the nonlinearity isolated in $\mu$. This additive structure admits clean differentiation, yielding an Euler equation that depends on $W$ and $c$ but can be inverted for $c$ in closed form.

The first-order condition equates the marginal utility of consumption to the marginal value of savings:
```{math}
:label: eq-foc
(1-\beta)(1-\rho) c^{-\rho} = \beta \frac{\partial \mu(a, z)}{\partial a}
```
The envelope theorem, combined with the first-order condition, gives
```{math}
:label: eq-envelope
\frac{\partial W(m, z)}{\partial m} = (1-\beta)(1-\rho) c(m,z)^{-\rho}
```

Differentiating $\mu(a, z)$ with respect to $a$ yields:
```{math}
:label: eq-mu-prime
\frac{\partial \mu}{\partial a} = R \cdot \mu(a,z)^{1-\theta} \cdot \mathbb{E}_{z'|z}\left[ W(m', z')^{\theta-1} \cdot \frac{\partial W(m', z')}{\partial m'} \right]
```
Substituting {eq}`eq-mu-prime` into the FOC {eq}`eq-foc` and applying the envelope condition {eq}`eq-envelope` to the next-period marginal value, the Euler equation becomes
```{math}
:label: eq-euler
c^{-\rho} = \beta R \cdot \mu(a, z)^{1-\theta} \cdot \Xi(a, z)
```
where
```{math}
:label: eq-xi
\Xi(a, z) \equiv \mathbb{E}_{z'|z} \left[ W(m', z')^{\theta-1} \cdot c(m', z')^{-\rho} \right]
```

Given $W(\cdot, z')$ and $c(\cdot, z')$ for all $z'$, both $\mu(a,z)$ and $\Xi(a,z)$ depend only on end-of-period assets $a$ and the current state $z$. Inverting the Euler equation yields consumption as a function of $(a, z)$:

```{prf:proposition} Inverted Euler Equation
:label: prop-euler

The Euler equation for Epstein-Zin preferences can be inverted to yield
$$c(a, z) = \left( \beta R \cdot \mu(a, z)^{1-\theta} \cdot \Xi(a, z) \right)^{-1/\rho}$$
```

This closed-form expression is what enables EGM: given an exogenous grid over $a$, we compute $c(a, z)$ directly and recover $m = c + a$.

```{prf:remark} Limiting Cases
:label: rem-limiting

When $\rho \to 1$, the transformation becomes $W = \log V$ and the certainty equivalent becomes $\mu(a,z) = \frac{1}{1-\gamma}\log \mathbb{E}_{z'|z}[\exp((1-\gamma)W(m',z'))]$, yielding the risk-sensitive preferences of {cite:t}`hansensargent1995`; {cite:t}`tallarini2000risk` demonstrates the equivalence in business cycle models. When $\gamma \to 1$, the certainty equivalent of $V'$ becomes the geometric mean $\exp(\mathbb{E}_{z'|z}[\log V(m',z')])$; {cite:t}`giovanniniweil1989` show that this case yields myopic portfolio allocation.[^limiting-cases]
```

[^limiting-cases]: The logarithmic limit requires renormalization: formally, $\lim_{\rho \to 1}(V^{1-\rho} - 1)/(1-\rho) = \log V$ by L'Hôpital's rule. The Bellman equation and certainty equivalent formulae stated here are the limiting forms after this renormalization.

# Algorithm

One distinction from standard EGM emerges from the structure of {prf:ref}`prop-euler`: the Euler equation requires both the policy function $c(\cdot, z')$ and the value function $W(\cdot, z')$ evaluated at next-period states. In standard CRRA problems, the Euler equation depends only on $c(\cdot, z')$, so EGM can iterate on the policy function alone. Here, we must iterate until both functions converge simultaneously.

This makes the algorithm a hybrid of EGM and the Howard policy iteration technique introduced by {cite:t}`howard1960dynamic`. EGM provides the root-finding-free policy update, while tracking $W(\cdot)$ alongside $c(\cdot)$ ensures the value function remains consistent with the policy. Each iteration updates both functions, and convergence requires both to stabilize.

```{prf:remark} Storing V instead of W
:label: rem-storing-v

While the algorithm is derived using $W = V^{1-\rho}$, numerical implementation benefits from storing and interpolating $V$ directly, converting to $W$ only when needed. When $\rho > 1$, the transformation $W = V^{1-\rho}$ maps small values of $V$ to large values of $W$ (since the exponent is negative), making $W$ less stable for interpolation near the borrowing constraint. When $\rho < 1$, both are well-behaved. In either case, $V$ is the natural object to store. The algorithm below stores $(c, V)$, converts $V \to W$ for the Euler computation, and converts back after updating. The algorithm works for both $\rho < 1$ (EIS $> 1$) and $\rho > 1$ (EIS $< 1$): although $\theta$ changes sign at $\rho = 1$, the Euler equation and certainty equivalent formulas remain valid.[^rho-verification]

[^rho-verification]: See [](#appendix-rho) for numerical verification.
```

The algorithm proceeds as follows: Fix a grid of end-of-period asset values $\{a_j\}_{j=1}^J$ and an exogenous grid of cash-on-hand values $\{m_i\}_{i=1}^I$. Initialize with a policy such as $c^{(0)}(m, z) = m$ and a monotonically increasing, concave $V^{(0)}$ (e.g., $V^{(0)} = c^{(0)}$); the iteration will converge to the fixed point. For finite-horizon problems, the terminal condition is $c(m, z) = m$; for infinite-horizon, iterate until convergence.

:::{raw} latex
\begin{algorithm}
\caption{EGM for Epstein-Zin}\label{alg:egm-ez}
\begin{algorithmic}[1]
\Require Grid of end-of-period assets $\{a_j\}_{j=1}^J$, exogenous grid $\{m_i\}_{i=1}^I$, initial guess $(c^{(0)}, V^{(0)})$
\Ensure Converged policy $c(\cdot, \cdot)$ and value $V(\cdot, \cdot)$
\State Given $(c^{(n)}, V^{(n)})$ from iteration $n$, for each income state $z_k$:
\For{each $a_j$}
    \State Compute $m'_{j\ell} = R a_j + y(z'_\ell)$ for all $z'_\ell$
    \State Interpolate $c^{(n)}(m'_{j\ell}, z'_\ell)$ and $V^{(n)}(m'_{j\ell}, z'_\ell)$; compute $W^{(n)} = (V^{(n)})^{1-\rho}$
    \State Compute $\mu_j = \left( \sum_\ell \pi_{k\ell} W^{(n)}(m'_{j\ell}, z'_\ell)^{\theta} \right)^{1/\theta}$
    \State Compute $\Xi_j = \sum_\ell \pi_{k\ell} \left[ W^{(n)}(m'_{j\ell}, z'_\ell)^{\theta-1} \cdot c^{(n)}(m'_{j\ell}, z'_\ell)^{-\rho} \right]$
    \State Invert Euler: $c_j = \left( \beta R \cdot \mu_j^{1-\theta} \cdot \Xi_j \right)^{-1/\rho}$
    \State Recover grid: $m_j = c_j + a_j$
\EndFor
\State Append $(0, 0)$ at the constraint
\State Interpolate $c^{(n+1)}(\cdot, z_k)$ from $\{(m_j, c_j)\}$ to $\{m_i\}$
\For{each $m_i$}
    \State Compute $a_i = m_i - c^{(n+1)}(m_i, z_k)$
    \State Interpolate $\mu(a_i, z_k)$ from $\{\mu_j\}$
    \State Update: $W^{(n+1)}(m_i, z_k) = (1-\beta) c^{(n+1)}(m_i, z_k)^{1-\rho} + \beta \mu(a_i, z_k)$
    \State Convert: $V^{(n+1)}(m_i, z_k) = (W^{(n+1)}(m_i, z_k))^{1/(1-\rho)}$
\EndFor
\State \textbf{Iterate} until $\|c^{(n)} - c^{(n-1)}\| < \varepsilon$
\end{algorithmic}
\end{algorithm}
:::

The algorithm requires no root-finding: each iteration involves only interpolation, expectation computation, and closed-form inversions. In contrast, time iteration methods must solve a non-linear system at each grid point; here, the Euler inversion yields consumption directly. The cost per iteration is comparable to standard EGM; the only addition is tracking the value function alongside consumption. Convergence is monitored via $\|c^{(n)} - c^{(n-1)}\|$; since $V$ is uniquely determined by $c$ through the Bellman equation, policy convergence implies value convergence.

To handle the borrowing constraint, we augment the endogenous grid with the point $(m, c) = (0, 0)$ to anchor interpolation. For typical calibrations, the resulting grid is monotonic and interpolation proceeds directly. If non-monotonicity occurs (i.e., $m_j > m_{j+1}$ for some $j$ despite $a_j < a_{j+1}$), construct the consumption function via the upper envelope following {cite:t}`iskhakov2017endogenous`. In the constrained region where $a = 0$ binds, $c(m, z) = m$ and $W(m, z) = (1-\beta) m^{1-\rho} + \beta \mu(0, z)$.

# Benchmarks

We implement the algorithm for an infinite-horizon consumption-savings problem with stochastic income. Income follows an AR(1) process with persistence 0.95, consistent with estimates from labor market data (e.g., {cite:t}`storesletten2004cyclical`), discretized using the {cite:t}`tauchen1986finite` method with 10 grid points. The asset grid uses 100 exponentially-spaced points with an upper bound of 20 times mean income.

**Parameters.** $\beta = 0.96$, $R = 1.02$, $\gamma = 10$, and EIS $= 1.5$ (so $\rho = 2/3$), following {cite:t}`bansalyaron2004`. The meta-analysis of {cite:t}`havranek2015measuring` finds that micro estimates of the EIS often fall below unity, though macro and asset pricing calibrations typically use higher values.[^calibration] Since $\gamma > \rho$, the agent prefers early resolution of uncertainty. The auxiliary parameter is $\theta = (1-\gamma)/(1-\rho) = -27$.

[^calibration]: High $\gamma$ with low $\rho$ implies aversion to contemporaneous consumption risk but tolerance for intertemporal variation. Whether this reflects preferences or serves as a modeling device for asset pricing remains debated; the method here is agnostic on calibration.

## Speed

We compare three solution methods: EZ-EGM, time iteration (TI), and VFI. {cite:t}`rendahl2015inequality` proves that TI and VFI converge to the same solution under standard conditions, so the comparison is one of computational efficiency. For TI and VFI, we distinguish between *fast* mode (precompute $\mu$ on the asset grid, interpolate during search) and *accurate* mode (compute $\mu$ exactly at each candidate).[^appendix-alg] EGM has no such distinction: it evaluates $\mu$ on the endogenous grid, achieving accurate-mode precision without the computational cost.

[^appendix-alg]: [](#appendix-algorithms) details both algorithms and the tradeoff between modes.

:::{raw} latex
\begin{table}[ht]
\centering
\caption{Speed comparison (fast modes)}
\label{tab:speed}
\begin{tabular}{@{}lrrr@{}}
\toprule
Method & Time (ms) & Policy Iters & Euler Error \\
\midrule
EZ-EGM & 21 & 141 & $-4.8$ \\
TI (fast) & 237 & 140 & $-3.6$ \\
VFI (fast) & 1162 & 239 & $-3.3$ \\
\bottomrule
\end{tabular}

{\footnotesize \textit{Note:} Euler error is mean $\log_{10}$ error on the ergodic distribution (more negative = more accurate). Fast modes precompute $\mu$ on the asset grid. Timings on CPU; results vary by hardware. All methods use JAX.}
\end{table}
:::

The speed advantage of EZ-EGM comes from eliminating numerical search. Time iteration (TI), introduced by {cite:t}`coleman1991equilibrium` and extended to Epstein-Zin preferences by {cite:t}`coeurdacier2020financial`, requires bisection at every grid point; VFI requires golden-section search. EZ-EGM bypasses this entirely through analytic Euler inversion. At equal grid size (100 points), EZ-EGM is approximately 50 times faster than VFI and 10 times faster than TI. Moreover, EGM is more than one order of magnitude more accurate than both. [](#fig-policy) shows that the resulting policy functions are smooth and well-behaved.

:::{figure} ../figures/fig_policy.*
:label: fig-policy
:align: center

Consumption policy $c(m,z)$ for different income states. Higher income states (red) allow more consumption at each wealth level.
:::

## Accuracy

We assess accuracy using the normalized Euler equation error, a standard metric whose magnitude bounds the policy function error. {cite:t}`santos2000accuracy` establishes the theoretical foundation:
```{math}
\varepsilon(m, z) = \log_{10} \left| 1 - \frac{\tilde{c}(m, z)}{c(m, z)} \right|
```
where $\tilde{c}(m, z)$ is consumption implied by the Euler equation given the computed policy. We report mean (L1) and maximum (L$\infty$) errors following {cite:t}`maliarmaliar2014`. More negative values indicate higher accuracy; an error of $-5$ means the policy deviates from the Euler equation by approximately $10^{-5}$, or 0.001\%.

Errors should be evaluated at wealth levels agents actually visit, not arbitrary grid points. We simulate the ergodic distribution (10,000 agents, 500 periods) and compute errors at the 5th–95th percentiles of realized wealth. For this calibration, the ergodic distribution concentrates in the lower wealth range (median $m \approx 3$ versus grid maximum of 20), reflecting moderate impatience. Errors evaluated on the ergodic distribution are similar to those on a uniform grid, indicating the solution is accurate where agents spend time.[^euler-error]

[^appendix-euler]: [](#appendix-euler-errors) compares both evaluation methods in detail.

[^euler-error]: Points near the borrowing constraint are excluded, as the Euler equation holds as an inequality there; this is standard practice following {cite:t}`santos2000accuracy`.

:::{raw} latex
\begin{table}[ht]
\centering
\caption{Accuracy comparison (accurate modes)}
\label{tab:accuracy}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Method & Time (ms) & Policy Iters & Mean (L1) & Max (L$\infty$) \\
\midrule
EZ-EGM & 21 & 141 & $-4.8$ & $-3.2$ \\
TI (accurate) & 2301 & 141 & $-4.8$ & $-3.5$ \\
VFI (accurate) & 8293 & 239 & $-3.4$ & $-2.2$ \\
\bottomrule
\end{tabular}

{\footnotesize \textit{Note:} Errors evaluated on ergodic distribution. Accurate modes compute $\mu$ exactly during search. TI-accurate matches EGM accuracy but is 110$\times$ slower. VFI-accurate is 400$\times$ slower yet less accurate.}
\end{table}
:::

[](#tab:accuracy) shows an important result: TI can match EGM's accuracy (mean error $-4.8$), but only by computing $\mu$ exactly during search, which makes it 110 times slower. VFI-accurate is 400 times slower than EGM yet still 1.4 orders of magnitude less accurate ($-3.4$ versus $-4.8$). The fundamental difference is that EGM and TI work with the Euler equation directly while VFI optimizes the Bellman equation; Euler-based methods achieve higher precision when $\mu$ is computed exactly. But TI solves the Euler equation numerically via bisection, whereas EGM inverts it analytically. This is why EGM sidesteps the speed-accuracy tradeoff entirely: the endogenous grid places evaluation points exactly where the Euler equation is satisfied, achieving accurate-mode precision at fast-mode speed.

Consumption-equivalent welfare costs (the permanent consumption an agent would sacrifice to avoid using the approximate policy) are below 0.1\% for EGM, making numerical error economically negligible.[^welfare]

[^welfare]: Computed by simulating 20,000 agents under the approximate and high-accuracy policies with identical shocks, then comparing ergodic average values.

:::{figure} ../figures/fig_pareto.*
:label: fig-pareto
:align: center

Speed-accuracy Pareto frontier across grid sizes. Marker size reflects grid resolution $n$. The dashed line marks a reference accuracy level; EGM achieves this with small grids while VFI and TI require substantially more computation.
:::

**Equal-accuracy comparison.** A more informative comparison holds accuracy constant. [](#fig-pareto) shows that to match EGM's Euler errors with 20–25 grid points, VFI requires 50–300 points; the resulting speedup is 150–630 times.[^appendix-equal] This comparison isolates computational efficiency: both methods deliver the same solution quality, but EGM does so with far less computation. For applications requiring many model evaluations (estimation, uncertainty quantification, real-time decision support), such speedups translate directly into feasibility.

[^appendix-equal]: See [](#appendix-equal-accuracy) for details.

:::{figure} ../figures/fig_tradeoff.*
:label: fig-tradeoff
:align: center

Speed-accuracy tradeoff. VFI and TI face a tradeoff between fast modes (open markers) and accurate modes (filled markers). EGM sidesteps this tradeoff: a single mode achieves both speed and accuracy.
:::

**Speed-accuracy tradeoff.** Time iteration offers a middle ground between EGM and VFI. Like EGM, TI works with the Euler equation; like VFI, it requires numerical optimization. As [](#fig-tradeoff) illustrates, both VFI and TI face a speed-accuracy tradeoff: computing $\mu$ exactly at each candidate during search is accurate but slow, as [](#tab:accuracy) documents; precomputing $\mu$ on the asset grid and interpolating is fast but introduces approximation error, as [](#tab:speed) shows. The fast modes in [](#tab:speed) represent practical implementations. EGM sidesteps this tradeoff entirely: the endogenous grid places evaluation points exactly where the Euler equation is satisfied, achieving accurate-mode precision at fast-mode speed.

**Howard acceleration.** The baseline algorithm performs one policy update per value update. To accelerate convergence, fix the policy $c(\cdot, \cdot)$ and iterate on $W$ alone:
```{math}
W^{(k+1)}(m, z) = (1-\beta) c(m,z)^{1-\rho} + \beta \mu^{(k)}(m-c, z)
```
where $\mu^{(k)}$ uses $W^{(k)}$. After $K$ value iterations, the policy is updated via EGM. The optimal $K$ differs by method: VFI benefits from large $K$ (30 or more), TI performs best with small $K$ (3-5), while EGM is fastest at $K=1$ because its analytic policy step is already cheap.[^appendix-howard] With one additional value iteration ($K=2$), EGM becomes even more accurate while remaining faster than all VFI and TI configurations.

[^appendix-howard]: [](#appendix-howard) examines how $K$ affects each method in both fast and accurate modes.


# Conclusion

The endogenous grid method extends to Epstein-Zin preferences through the transformation $W = V^{1-\rho}$, which decouples the Euler equation from the value recursion.

The approach generalizes beyond Epstein-Zin. The method extends to the limiting case $\rho = 1$ (unit EIS) and to the risk-sensitive preferences of {cite:t}`hansensargent1995` used in robust control, both of which admit analogous transformations and closed-form Euler inversions.

The single-asset case presented here serves as a building block for richer models with multiple assets, portfolio choice, or labor supply decisions.

For practitioners solving consumption-savings models with recursive utility, the message is simple: EGM works. Where time iteration requires numerical root-finding at every grid point, EZ-EGM inverts the Euler equation analytically. The transformation is elementary, the implementation follows standard EGM patterns, and the speed gains are substantial.

+++ {"part": "acknowledgments"}
I am grateful to Christopher D. Carroll for years of mentorship, for detailed feedback on this paper, and for sustained support throughout my career. I also thank Matthew N. White for extensive discussions and collaboration that shaped this work. This work was supported by the Alfred P. Sloan Foundation [[G-2025-79177](https://sloan.org/grant-detail/g-2025-79177)].
+++

