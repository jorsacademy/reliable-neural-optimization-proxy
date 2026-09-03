# Reliable Neural Optimization Proxy

[![CI](https://github.com/jorsacademy/reliable-neural-optimization-proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/jorsacademy/reliable-neural-optimization-proxy/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-orange)](LICENSE)

A verification-first research implementation of a **neural optimization proxy** for repeatedly solved parametric convex dispatch problems.

The repository separates prediction from trust:

> A neural network may propose a decision, but feasibility is enforced by an exact repair layer and solution quality is accepted only when a valid primal-dual certificate is below a user-selected threshold. Otherwise, an exact optimizer takes over.

This produces an interpretable speed–quality control surface rather than an unconditional neural replacement for mathematical optimization.

## Research question

Can a lightweight neural proxy reduce repeated-solve work while preserving:

- exact primal feasibility;
- a deterministic upper bound on suboptimality;
- explicit selective fallback;
- auditable behavior under distribution shift;
- and an exact reference solution for every benchmark query?

The project does not assume that the proxy is faster than the structured oracle on the small instances used in CI. Runtime, fallback rate, and negative results are retained rather than hard-coded as a speedup claim.

## Parametric optimization problem

Each query is a strictly convex separable production-dispatch problem:

\[
\min_x\quad
f_\theta(x)=\sum_{i=1}^n
\left(\frac{1}{2}a_i x_i^2+b_i x_i\right)
\]

subject to

\[
\sum_{i=1}^n x_i=d,
\qquad
\ell_i\le x_i\le u_i.
\]

The parameter vector \(\theta\) contains quadratic costs, linear costs, lower bounds, upper bounds, and demand. Strict positivity of every \(a_i\) gives a unique optimum.

This model is deliberately compact. It is rich enough to expose the central reliability issues—equality feasibility, active bounds, optimality loss, dual certification, fallback, and distribution shift—while allowing every claim to be checked independently.

## Reliability architecture

```text
parametric dispatch query
          │
          ▼
canonical feature ordering
          │
          ▼
NumPy MLP raw prediction
          │
          ├── raw feasibility audit
          ▼
exact box + balance projection
          │
          ├── independent feasibility audit
          ▼
candidate-derived dual multiplier
          │
          ▼
valid Lagrangian lower bound
          │
          ▼
certified primal-dual gap
          │
          ├── gap <= user threshold ──► accept repaired proxy
          │
          └── gap > user threshold  ──► exact KKT oracle fallback
```

The neural network is not part of the proof of feasibility or quality. The proof obligations are discharged by deterministic optimization structure.

## Exact feasibility repair

Given a raw neural output \(r\), the repair layer solves

\[
\min_x \frac{1}{2}\|x-r\|_2^2
\]

subject to the original balance and box constraints.

The unique projection has the form

\[
x_i(\lambda)=\operatorname{clip}(r_i-\lambda,\ell_i,u_i).
\]

Because \(\sum_i x_i(\lambda)\) is monotone, a one-dimensional bisection finds the multiplier that satisfies the balance equality. The repaired vector is then checked by a separate feasibility auditor.

The projection is not a heuristic redistribution. It is the exact Euclidean projection for the declared feasible set.

## Deterministic optimality certificate

For any scalar multiplier \(\nu\), the Lagrangian dual function is

\[
g_\theta(\nu)=
-\nu d+
\sum_i
\min_{\ell_i\le z_i\le u_i}
\left[
\frac{1}{2}a_i z_i^2+(b_i+\nu)z_i
\right].
\]

The inner minimizer is available in closed form:

\[
z_i(\nu)=
\operatorname{clip}
\left(-\frac{b_i+\nu}{a_i},\ell_i,u_i\right).
\]

Weak duality gives

\[
g_\theta(\nu)\le f_\theta(x^*).
\]

For any feasible repaired prediction \(\hat x\), therefore,

\[
0\le f_\theta(\hat x)-f_\theta(x^*)
\le f_\theta(\hat x)-g_\theta(\nu).
\]

The right-hand side is the reported certificate. The multiplier is estimated from candidate marginal costs; it does not need to be dual-optimal for the bound to remain valid.

In the benchmark harness, every certificate is audited against the exact optimum. A certificate violation raises an exception rather than being summarized as a normal data point.

## Selective fallback guarantee

Let the configured acceptance allowance be

\[
\tau_{\mathrm{abs}}+\tau_{\mathrm{rel}}\max(1,|f_\theta(\hat x)|).
\]

The repaired proxy is accepted only when its certified gap does not exceed this allowance. Otherwise, the exact KKT oracle is called.

Consequently, every returned hybrid decision is either:

1. a feasible proxy decision with a certified suboptimality envelope; or
2. the exact structured optimum.

The system fails closed on invalid dimensions, non-finite values, incompatible checkpoints, projection failure, or certificate inconsistency.

## Neural proxy

The first version uses a two-hidden-layer `tanh` MLP implemented directly in NumPy and trained with Adam, gradient clipping, weight decay, and validation-based early stopping.

The implementation is intentionally small and inspectable. It avoids an external deep-learning runtime in the core package and keeps CI independent of GPU availability.

### Inputs

Each unit contributes:

- quadratic cost;
- linear cost;
- lower bound;
- upper bound;
- flexible capacity span;
- lower-bound fraction.

Global features include:

- demand;
- aggregate lower and upper production;
- aggregate flexible capacity;
- demand ratio;
- cost-distribution moments.

### Canonicalization

Units are sorted deterministically by intrinsic coefficients before feature extraction. Predictions are mapped back to the original order afterward. This makes the proxy invariant to arbitrary permutations of otherwise identical unit lists without claiming variable-size generalization.

### Training target

Training data are labelled by the exact KKT oracle. The network learns the optimal allocation in canonical order with standardized input and output spaces.

This is a supervised baseline. It is not presented as self-supervised E2ELR, dual-proxy learning, or a reproduction of a particular paper-scale architecture.

## Compared methods

The benchmark evaluates six methods on identical queries:

| Method | Feasible by construction | Quality certificate | Exact fallback |
| --- | --- | --- | --- |
| `raw_neural` | No | No | No |
| `projected_neural` | Yes | Yes | No |
| `proportional_baseline` | Yes | Yes | No |
| `merit_order_baseline` | Yes | Yes | No |
| `certified_hybrid` | Yes | Yes when proxy accepted | Yes |
| `exact_oracle` | Yes | Strong-duality check | Not applicable |

The two non-neural controls distinguish neural approximation quality from gains caused merely by applying a feasibility projection.

## Exact oracle

The structured optimizer solves the KKT balance equation

\[
\sum_i
\operatorname{clip}
\left(-\frac{b_i+\nu}{a_i},\ell_i,u_i\right)=d
\]

by monotone bisection. It returns:

- the unique primal optimum;
- the dual multiplier;
- primal and dual objectives;
- duality gap;
- balance residual;
- stationarity residual;
- iteration count and runtime.

Aggregate-lower and aggregate-upper edge cases are handled explicitly.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The runtime dependency is NumPy. No commercial solver, external API, network connection, or GPU is required.

## Quick start

Generate one deterministic query:

```bash
reliable-proxy generate \
  --units 8 \
  --regime in_distribution \
  --seed 42 \
  --output artifacts/instance.json
```

### 1. Build disjoint oracle-labelled datasets

```bash
reliable-proxy collect \
  --samples 1500 \
  --units 8 \
  --seed 2026 \
  --output artifacts/train.npz

reliable-proxy collect \
  --samples 300 \
  --units 8 \
  --seed 102026 \
  --output artifacts/validation.npz
```

### 2. Train the proxy

```bash
reliable-proxy train \
  artifacts/train.npz \
  artifacts/validation.npz \
  --hidden-dim 64 \
  --epochs 100 \
  --checkpoint artifacts/proxy.npz \
  --output-report artifacts/training.json
```

### 3. Solve with a certified acceptance threshold

```bash
reliable-proxy solve \
  --input artifacts/instance.json \
  --checkpoint artifacts/proxy.npz \
  --relative-gap-tolerance 0.01 \
  --absolute-gap-tolerance 1e-6 \
  --include-vectors
```

### 4. Benchmark one regime

```bash
reliable-proxy benchmark \
  --samples 200 \
  --units 8 \
  --regime high_demand \
  --seed 202026 \
  --checkpoint artifacts/proxy.npz \
  --relative-gap-tolerance 0.01 \
  --output-json artifacts/high-demand.json \
  --output-csv artifacts/high-demand.csv
```

## Frozen research protocol

The full protocol trains only on `in_distribution` queries and evaluates disjoint seed ranges for:

1. in-distribution demand and costs;
2. high demand;
3. low demand;
4. shifted cost coefficients;
5. shifted capacities;
6. combined cost, capacity, and high-demand shift.

```bash
reliable-proxy research \
  --train-samples 1500 \
  --validation-samples 300 \
  --evaluation-samples 200 \
  --units 8 \
  --epochs 100 \
  --seed 2026 \
  --checkpoint artifacts/research-proxy.npz \
  --output-report artifacts/research-report.json
```

Defaults are frozen in [`configs/research_v1.json`](configs/research_v1.json). The protocol is described in [`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## Reported metrics

Prediction and feasibility:

- raw feasibility rate;
- balance violation;
- lower- and upper-bound violation;
- projection correction norm;
- validation RMSE, MAE, and maximum absolute prediction error.

Optimization quality:

- exact objective;
- actual optimality gap where the decision is feasible;
- certified primal-dual gap;
- KKT stationarity residual;
- certificate violation count.

Selective reliability:

- proxy acceptance rate;
- exact fallback rate;
- maximum accepted actual gap;
- acceptance curves at several quality thresholds.

Runtime:

- neural inference;
- feasibility repair;
- certification;
- exact fallback;
- total query time.

Correctness, quality, coverage, and runtime are not collapsed into a single weighted leaderboard score.

## Repository structure

```text
src/reliable_proxy/
├── domain.py       typed dispatch model and JSON I/O
├── generator.py    deterministic in-distribution and shifted queries
├── oracle.py       exact KKT optimizer and dual function
├── projection.py   exact box-plus-balance Euclidean repair
├── audit.py        independent feasibility and KKT checks
├── certificate.py  candidate-derived dual lower bounds
├── features.py     canonical feature and decision transforms
├── dataset.py      oracle-labelled NPZ datasets
├── model.py        NumPy MLP, Adam, early stopping, checkpoints
├── training.py     training and validation diagnostics
├── proxy.py        certified acceptance and exact fallback
├── benchmark.py    solver-grounded method comparison
├── experiment.py   frozen distribution-shift protocol
└── cli.py          command-line workflows
```

## Tests and CI

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

The regression suite checks:

- strict domain validation and deterministic generation;
- JSON and NPZ round trips;
- exact primal-dual strong duality;
- hand-computable KKT solutions;
- lower- and upper-aggregate edge cases;
- exact projection feasibility and idempotence;
- certificate dominance over the true gap;
- permutation-stable feature handling;
- neural loss reduction and checkpoint round trips;
- guaranteed acceptance of an exact predictor;
- exact fallback under a zero-gap threshold;
- all benchmark controls and certificate curves;
- compact distribution-shift experiments;
- CLI collect, train, solve, and benchmark workflows.

GitHub Actions runs Python 3.11 and 3.12, dependency checks, Ruff, strict mypy, branch-aware coverage, and an end-to-end collect/train/solve/benchmark smoke test.

## Methodological boundaries

This repository does **not** claim:

- industrial economic-dispatch realism;
- AC or DC power-flow constraints;
- network, reserve, ramping, unit-commitment, or integer constraints;
- variable-size neural generalization;
- self-supervised training;
- a learned dual proxy;
- worst-case verification over a continuous parameter region;
- that average proxy accuracy implies safe deployment;
- that the NumPy MLP is state of the art;
- that small-instance runtime comparisons transfer to commercial solvers;
- universal speedup over the closed-form structured oracle.

The project isolates a smaller proposition: **feasible repair plus a valid dual lower bound can convert an approximate neural prediction into a selectively trusted hybrid optimization service.**

See [`docs/exactness.md`](docs/exactness.md) for the precise guarantee and [`docs/research_context.md`](docs/research_context.md) for its relation to current optimization-learning work.

## References

1. Chen, W., Tanneau, M., & Van Hentenryck, P. (2023). *End-to-End Feasible Optimization Proxies for Large-Scale Economic Dispatch*. arXiv:2304.11726. https://arxiv.org/abs/2304.11726
2. Chen, W., Zhao, H., Tanneau, M., & Van Hentenryck, P. (2024). Compact Optimality Verification for Optimization Proxies. *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, 7847–7863. https://proceedings.mlr.press/v235/chen24bj.html
3. Tanneau, M., & Van Hentenryck, P. (2024). Dual Lagrangian Learning for Conic Optimization. *Advances in Neural Information Processing Systems 37*. https://proceedings.neurips.cc/paper_files/paper/2024/hash/646d2edf873df99d36aaeeaf058acdb8-Abstract-Conference.html
4. Van Hentenryck, P. (2025). *Optimization Learning*. arXiv:2501.03443. https://arxiv.org/abs/2501.03443
5. Klamkin, M., Tanneau, M., & Van Hentenryck, P. (2025). *Self-Certifying Primal-Dual Optimization Proxies for Large-Scale Batch Economic Dispatch*. arXiv:2510.15850. https://arxiv.org/abs/2510.15850
6. Pareek, P., Jayakumar, A., Sundar, K., Misra, S., & Deka, D. (2025). Optimization Proxies using Limited Labeled Data and Training Time. *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267. https://proceedings.mlr.press/v267/pareek25a.html

## License

This project is source-available under the **PolyForm Noncommercial License 1.0.0**. Commercial use is not granted. It is not OSI Open Source. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
