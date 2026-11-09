"""Tests de la configuration CI (T07, T20) : le workflow doit exécuter les contrôles ET signaler
les échecs (jamais de ``continue-on-error`` qui masquerait une régression)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    return workflow.get("on") or workflow.get(True)  # PyYAML lit la clé « on » comme True


def _steps_text(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in job["steps"])


def test_workflow_runs_on_pull_requests_and_pushes_to_main():
    triggers = _triggers(_load("tests.yml"))
    assert "pull_request" in triggers and "push" in triggers
    assert triggers["push"]["branches"] == ["main"]


def test_workflow_runs_pytest_with_coverage_gate_on_two_python_versions():
    job = _load("tests.yml")["jobs"]["tests"]
    text = _steps_text(job)
    assert "pytest" in text and "--cov=app" in text and "--cov-fail-under=85" in text
    assert "not slow and not full_data" in text
    assert len(job["strategy"]["matrix"]["python-version"]) >= 2
    assert job["strategy"]["fail-fast"] is False


def test_workflow_runs_ruff_lint_and_format_check():
    text = _steps_text(_load("tests.yml")["jobs"]["lint"])
    assert "ruff check" in text and "ruff format --check" in text


@pytest.mark.parametrize("name", ["tests.yml", "experiments.yml"])
def test_workflows_never_mask_failures(name):
    workflow = _load(name)
    for job in workflow["jobs"].values():
        assert job.get("continue-on-error") in (None, False)
        for step in job["steps"]:
            assert step.get("continue-on-error") in (None, False), step.get("name")


def test_workflow_uploads_evidence_even_when_tests_fail_and_signals_failure():
    steps = _load("tests.yml")["jobs"]["tests"]["steps"]
    upload = next(s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact"))
    assert upload["if"] == "always()"
    assert any(s.get("if") == "failure()" and "::error" in str(s.get("run", "")) for s in steps)


def test_workflow_uses_least_privilege_permissions():
    assert _load("tests.yml")["permissions"] == {"contents": "read"}
    assert _load("experiments.yml")["permissions"] == {"contents": "read"}


def test_heavy_workflow_is_manual_or_scheduled_only():
    triggers = _triggers(_load("experiments.yml"))
    assert "workflow_dispatch" in triggers and "schedule" in triggers
    assert "pull_request" not in triggers and "push" not in triggers
