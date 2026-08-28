# Contributing

Contributions are welcome. Keep extensions capability-driven so custom problems do not need to
implement features they cannot support.

## Development setup

```bash
python -m pip install -e ".[plots,notebooks,reports,dev]"
python -m pytest
```

Before submitting a change:

1. Add focused tests for new behavior or bug fixes.
2. Run `python -m pytest` and `python -m compileall -q src`.
3. Keep bundled notebook settings small enough for a quick local run.
4. Document new public inputs in `docs/input_reference.md`.
5. Do not commit generated files from `outputs/experiments/`.

New estimators expose `theta_` and `history_` after `fit`. New losses implement
`value_and_subgradient`. New oracles implement `solve` and declare only the capabilities they
actually provide.

