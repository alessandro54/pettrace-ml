# Test layout

- `unit/pettrace/` mirrors `src/pettrace/`: `test_x.py` tests `x.py` at the same relative path.
- `integration/pettrace/` covers workflows spanning multiple package modules.
- `integration/notebooks/` verifies notebook contracts without executing experiments.
- `integration/tools/` covers repository tooling and subprocess launchers.
- `integration/package/` covers distribution, dependency and publication policy boundaries.

Unit tests must not contact external services. Integration tests may exercise filesystem and
subprocess boundaries, but remote APIs remain mocked unless a test is explicitly introduced as an
external system test.
