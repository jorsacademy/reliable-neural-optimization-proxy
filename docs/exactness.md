# Exactness and certification contract

This document states the mathematical obligations behind the words **exact**, **feasible**, and **certified** in this repository.

## 1. Declared problem class

For a fixed number of units, the query is

\[
\min_x f(x)=\sum_i \left(\tfrac12 a_i x_i^2+b_i x_i\right)
\]

subject to

\[
\mathbf 1^\top x=d,\qquad \ell\le x\le u.
\]

The domain validator requires finite coefficients, \(a_i>0\), \(\ell_i<u_i\), and

\[
\sum_i\ell_i\le d\le\sum_i u_i.
\]

Strict convexity gives a unique primal optimum. The feasible set is nonempty, compact, and convex.

## 2. Exact structured oracle

With equality multiplier \(\nu\), minimizing the Lagrangian over the box gives

\[
x_i(\nu)=\operatorname{clip}\left(-\frac{b_i+\nu}{a_i},\ell_i,u_i\right).
\]

The balance function

\[
h(\nu)=\sum_i x_i(\nu)-d
\]

is continuous and nonincreasing. Breakpoints derived from the lower and upper bounds provide a finite bracket. Bisection stops only when the scaled balance residual is below the configured tolerance. Aggregate-lower and aggregate-upper boundary cases are handled explicitly.

After solving, a separate KKT audit checks balance, bounds, and the sign-aware stationarity conditions. The primal objective is also compared with the dual value. A materially negative primal-dual gap or infeasible oracle decision raises an exception.

“Exact” therefore means exact for the declared separable convex model up to explicit floating-point and bisection tolerances. It does not mean symbolic arithmetic.

## 3. Exact feasibility projection

For a raw prediction \(r\), repair solves

\[
\min_x \tfrac12\lVert x-r\rVert_2^2
\]

subject to the original box and balance constraints. The projection KKT system yields

\[
x_i(\lambda)=\operatorname{clip}(r_i-\lambda,\ell_i,u_i).
\]

The associated balance function is continuous and nonincreasing, so scalar bisection finds the unique projected decision. The implementation independently re-audits the result; failure to meet the scaled tolerance is an error, not an accepted approximation.

This guarantee is specific to one affine equality plus box constraints. It must not be generalized to arbitrary nonlinear, integer, or network-constrained models.

## 4. Dual lower bound

For any finite scalar \(\nu\), define

\[
g(\nu)=-\nu d+\sum_i\min_{\ell_i\le z_i\le u_i}
\left[\tfrac12a_i z_i^2+(b_i+\nu)z_i\right].
\]

The inner minimizer is available in closed form. Weak duality gives

\[
g(\nu)\le f(x^*).
\]

For any feasible candidate \(\hat x\),

\[
0\le f(\hat x)-f(x^*)\le f(\hat x)-g(\nu).
\]

The right-hand side is the repository's certified gap. The multiplier estimated from candidate marginal costs is used to tighten the bound, but its optimality is not required for validity. If the candidate is infeasible, the certificate is invalid and its gap is set to infinity.

## 5. Selective acceptance guarantee

Let

\[
T=\tau_{\mathrm{abs}}+\tau_{\mathrm{rel}}\max(1,|f(\hat x)|).
\]

The hybrid service accepts the repaired prediction only if the certificate is valid and

\[
f(\hat x)-g(\nu)\le T.
\]

Weak duality then implies

\[
f(\hat x)-f(x^*)\le T.
\]

Otherwise, the exact structured oracle supplies the returned decision. Under the declared numerical tolerances, every hybrid output is therefore either a feasible candidate with an explicit suboptimality envelope or the exact structured optimum.

## 6. Independent benchmark audit

The benchmark does not trust the certificate by construction. It solves every query with the exact oracle and verifies

\[
\text{actual gap}\le\text{certified gap}+\text{numerical allowance}.
\]

It also verifies the configured acceptance envelope for every accepted hybrid prediction. Any violation aborts the benchmark.

## 7. Numerical scope

Floating-point certification is not interval arithmetic. The implementation uses scale-aware tolerances and raises on material inconsistencies, but it does not provide formally machine-checked roundoff bounds. Claims should therefore state the tolerance, Python/NumPy versions, hardware, seed range, and commit SHA.

## 8. Excluded guarantees

The repository does not certify worst-case behavior over a continuous parameter region, robustness to corrupted input data, variable-size transfer, security of arbitrary external storage, or optimality for models outside the declared problem class. Extending the method requires a new projection/completion layer, a mathematically valid lower bound, and independent tests for the new formulation.
