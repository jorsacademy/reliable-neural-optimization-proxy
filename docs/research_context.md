# Research context

Optimization proxies learn a parameter-to-solution map for problem families that must be solved repeatedly. Their practical value depends on more than prediction error: a useful proxy must address feasibility, objective quality, distribution shift, and the cost of recovering when the prediction is unreliable.

This repository implements a deliberately small instance of that program. It combines four ideas:

1. supervised solution-map approximation;
2. an exact feasibility-repair layer specialized to the constraint geometry;
3. a primal-dual quality certificate derived from weak duality;
4. selective exact fallback when the certificate exceeds a user-defined budget.

## Relationship to prior work

End-to-end feasible proxy research has emphasized completion and repair layers that turn unconstrained network outputs into feasible decisions. Compact optimality-verification work studies how to certify or bound proxy quality rather than relying on average prediction loss. Dual-learning methods learn lower-bound information directly, while self-certifying primal-dual proxies jointly produce decisions and certificates. Limited-label studies examine the cost of obtaining optimizer-generated training data.

The present implementation is narrower. It does not learn a dual model and does not verify a whole neural network over a continuous parameter region. Instead, it exploits a separable convex problem whose feasibility projection, dual function, and exact optimum can all be implemented independently and audited on every query.

## What the repository contributes

The contribution is methodological and engineering-oriented:

- prediction is explicitly separated from trust;
- raw, repaired, certified, and fallback behavior are all observable;
- a valid certificate is required at inference time, not inferred from validation accuracy;
- the exact optimizer remains in the loop as a selective safety mechanism;
- every evaluation query has an exact reference solution;
- distribution shifts are reported separately;
- non-neural feasible baselines isolate the value of learning;
- failure paths are tested and fail closed.

## What it does not reproduce

The code is not a reproduction of E2ELR, compact mixed-integer verifier formulations, Dual Lagrangian Learning, or a paper-scale self-certifying primal-dual architecture. It does not claim state-of-the-art predictive accuracy or industrial economic-dispatch scale.

Those methods motivate the design, but the repository's first objective is to make the certificate logic and experimental contract easy to inspect.

## Natural extensions

Research extensions that preserve the same trust structure include:

- learning a dual multiplier or dual function while validating every lower bound;
- conformal calibration of acceptance thresholds under exchangeability assumptions;
- differentiable projection or completion layers for richer convex sets;
- interval or mixed-integer worst-case verification over bounded parameter regions;
- active data acquisition focused on fallback-heavy regions;
- variable-size graph encoders with schema-aware checkpoint validation;
- DC optimal power flow with network constraints and a solver-backed repair layer;
- batch scheduling of proxy and fallback queries under latency service levels.

Each extension must re-establish feasibility and lower-bound validity for its new mathematical model; neural accuracy alone is not sufficient.
