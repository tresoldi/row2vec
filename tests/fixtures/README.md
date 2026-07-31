# Test fixtures

Pickled intermediate results written by `tests/test_fixtures.py`.

These are **outputs, not committed baselines**: the tests that read them write
them first in the same run, so they verify save/load round-tripping and
within-run determinism rather than comparing against a stored reference. They
are therefore not tracked in git (see `.gitignore`) — a tracked copy would churn
on every test run without ever being checked against anything.
