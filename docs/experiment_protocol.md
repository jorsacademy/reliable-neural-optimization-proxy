# Frozen experiment protocol

The purpose of the protocol is to evaluate prediction, repair, certification, fallback, and distribution shift separately. It is not designed to manufacture a speedup result.

## Versioned configuration

The default protocol is stored in `configs/research_v1.json`. Any change to sample counts, seed ranges, feature schema, model architecture, acceptance thresholds, or generator regimes constitutes a new protocol version. Published results should record the configuration file, checkpoint hash, repository commit, Python version, NumPy version, and hardware.

## Data generation

All data are synthetic and generated deterministically from integer seeds. Training and validation use only the `in_distribution` regime. Seed intervals are disjoint:

- training starts at the root seed;
- validation starts at root seed plus 100,000;
- evaluation starts at root seed plus 200,000;
- each evaluation regime receives a separate 10,000-seed block.

The report stores the exact inclusive seed ranges. Rows are not randomly split after generation, so no query can occur in more than one split.

## Distribution-shift suites

Evaluation contains six independently reported regimes:

| Regime | Change relative to training |
| --- | --- |
| `in_distribution` | Same coefficient, capacity, and demand ranges |
| `high_demand` | Demand close to aggregate upper capacity |
| `low_demand` | Demand close to aggregate lower production |
| `cost_shift` | Larger quadratic and linear cost coefficients |
| `capacity_shift` | Larger capacities and lower-bound fractions |
| `combined_shift` | Joint cost, capacity, and high-demand shift |

These are controlled synthetic shifts, not claims about real-grid covariate shift.

## Training

The exact KKT oracle labels every training and validation query. The first release uses a fixed-size two-hidden-layer `tanh` MLP, standardized inputs and outputs, Adam, weight decay, gradient clipping, and validation early stopping. Hyperparameters are fixed before evaluation.

Model selection uses validation loss only. Test regimes are not used to choose epochs, thresholds, or architecture.

## Compared methods

Every query is evaluated by:

1. raw neural prediction;
2. projected neural prediction;
3. proportional feasible baseline;
4. merit-order feasible baseline;
5. certified selective hybrid;
6. exact KKT oracle.

The same exact optimum is used for all methods on a query. Non-neural baselines prevent attributing all gains from deterministic projection to the neural model.

## Primary metrics

Reliability metrics:

- final feasibility rate;
- certificate-oracle violation count;
- hybrid acceptance and fallback rates;
- maximum actual gap among accepted decisions;
- acceptance/fallback curves across frozen relative tolerances.

Optimization metrics:

- actual absolute and percentage gap;
- certified absolute and percentage gap;
- balance and bound violations;
- KKT stationarity residual.

Approximation metrics:

- validation RMSE, MAE, and maximum absolute error;
- raw prediction feasibility;
- projection correction norm.

Systems metrics:

- inference, repair, certificate, oracle, and total time.

Correctness is never averaged away. Any certificate violation is a failed experiment, not a small contribution to an aggregate score.

## Timing discipline

Runtime measurements are wall-clock observations and must be interpreted cautiously. Report warm-up policy, repetitions, hardware, thread settings, and dependency versions. The small structured oracle may be faster than the neural pipeline; such results are retained. The repository makes no default speedup claim.

## Statistical reporting

For paper-scale experiments, use multiple independent training seeds and report per-regime means, dispersion, and worst cases. Acceptance curves should be shown together with actual accepted gaps. A higher acceptance rate is not inherently better if the requested quality envelope is looser.

## Reproducibility checklist

Before reporting results:

- run `ruff check .`, `ruff format --check .`, `mypy src`, and `pytest`;
- archive the exact config and checkpoint;
- record the commit SHA and dependency versions;
- verify all seed ranges are disjoint;
- retain complete JSON/CSV outputs, including negative results;
- state the acceptance threshold in objective units and relative form;
- do not compare timing measurements obtained under different hardware or solver settings as though they were paired.
