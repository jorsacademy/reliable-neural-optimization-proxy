# Changelog

All notable changes to this project are documented here.

## [0.1.0] - 2026-09-03

### Added

- typed strictly convex parametric dispatch domain and deterministic shifted generators;
- exact KKT primal-dual oracle with aggregate-bound edge handling;
- exact Euclidean box-plus-balance feasibility projection;
- candidate-derived dual lower bounds and per-query suboptimality certificates;
- selective neural acceptance with exact fallback;
- permutation-stable fixed-size feature representation;
- auditable NumPy MLP with Adam, clipping, early stopping, and versioned checkpoints;
- oracle-labelled dataset generation and disjoint train/validation workflows;
- proportional and merit-order non-neural baselines;
- solver-grounded benchmark with acceptance curves and certificate audits;
- six-regime frozen distribution-shift protocol;
- CLI, tests, Python 3.11/3.12 CI, documentation, citation metadata, and noncommercial licensing.

### Reliability contract

A neural prediction is never returned solely because its validation error is small. The decision is exactly repaired, checked for feasibility, and accepted only when a valid dual certificate satisfies the configured quality envelope. Otherwise the exact structured oracle supplies the final decision.
