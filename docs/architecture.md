# Architecture

The package separates approximation, mathematical repair, certification, and fallback into independent layers. A learned component is never used as the source of a feasibility or optimality claim.

```text
DispatchInstance
      │
      ├── canonical feature transform
      │          │
      │          ▼
      │       NumPy MLP
      │          │ raw decision
      │          ▼
      ├── exact Euclidean projection
      │          │ feasible candidate
      │          ▼
      ├── candidate-derived dual multiplier
      │          │
      │          ▼
      ├── valid Lagrangian lower bound
      │          │ certified gap
      │          ▼
      ├── acceptance gate ── accepted ──► return repaired candidate
      │          │
      │          └── rejected ──────────► exact KKT oracle
      │                                      │
      └──────────────────────────────────────┴──► independent audit/report
```

## Layer responsibilities

`domain.py` defines an immutable strictly convex dispatch query and validates the mathematical domain. `features.py` provides deterministic fixed-size canonicalization so arbitrary unit ordering does not affect the model input. `model.py` contains a small NumPy regressor and versioned checkpoint format.

`projection.py` solves the box-constrained balance projection exactly up to a declared scalar-bisection tolerance. `certificate.py` evaluates a feasible primal point against a valid dual lower bound. `proxy.py` applies the user-selected acceptance threshold and calls the exact oracle whenever the certificate is unavailable or too loose.

`oracle.py` is structurally independent from the neural model. It solves the original strictly convex problem through the KKT balance equation and checks primal feasibility, stationarity, and strong duality. `benchmark.py` computes the exact optimum for every evaluation query and fails closed if a reported certificate does not dominate the observed true gap.

## Trust boundaries

The trusted computing base for the reliability claim is deliberately small:

1. strict instance validation;
2. scalar projection bisection;
3. closed-form dual-function evaluation;
4. exact KKT oracle and numerical audits;
5. the acceptance/fallback gate.

Neural weights, training data, prediction accuracy, and distributional assumptions are outside that trusted base. A poor checkpoint can increase fallback frequency or runtime, but it cannot bypass the feasibility repair or certificate threshold.

## Data artifacts

Datasets and checkpoints use compressed NumPy archives with `allow_pickle=False` on load. Checkpoints carry a version, unit count, input/output dimensions, and feature-schema identifier. The loader rejects incompatible artifacts rather than adapting dimensions silently.
