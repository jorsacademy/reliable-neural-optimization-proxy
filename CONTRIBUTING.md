# Contributing

Contributions must preserve the separation between neural prediction and mathematical certification.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the complete gate before opening a pull request:

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

## Correctness requirements

Changes to the oracle, projection, dual function, certificate, or acceptance gate require tests for:

- valid and invalid dimensions;
- non-finite input rejection;
- aggregate-bound edge cases;
- independent feasibility auditing;
- primal-dual consistency;
- certificate dominance over the exact observed gap;
- fallback behavior when the certificate exceeds the threshold.

A learned component must not create feasibility or optimality claims that cannot be checked independently. New problem classes require a formulation-specific repair/completion method and a valid lower-bound construction.

## Experimental requirements

Do not tune on evaluation regimes. Use disjoint seeds, record configuration and commit identifiers, compare methods on identical instances, retain negative results, and distinguish prediction error from optimization quality and runtime.

Claims of speedup require repeated measurements, hardware and dependency disclosure, and an appropriate paired statistical analysis. Claims of global optimality require a valid proof or independent exact reference under the declared model.

## Scope and style

Keep runtime dependencies minimal. Public APIs should be typed. Avoid hidden global state. Randomness must be seeded. Checkpoint or feature-format changes require a schema-version change and migration note.

Documentation should say precisely what is guaranteed, under which tolerances, and what remains outside scope.
