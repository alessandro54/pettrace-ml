"""Unified CLI dispatch."""

from pettrace import cli


def test_run_e1_forwards_only_experiment_arguments(monkeypatch):
    received = []
    monkeypatch.setattr("pettrace.experiments.e1.main", lambda argv: received.extend(argv) or 0)

    assert cli.main(["run", "e1", "--private-version", "v5"]) == 0
    assert received == ["--private-version", "v5"]
