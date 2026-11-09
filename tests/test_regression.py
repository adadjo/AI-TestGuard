"""Tests de non-régression et de reproductibilité (T06, T16).

La référence (baseline) est un fichier VERSIONNÉ ; elle n'est jamais recalculée par les tests.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app import config as cfg
from app.regression import (
    DECISION_ACCEPT,
    DECISION_INVESTIGATE,
    DECISION_REJECT,
    build_baseline_record,
    compare_to_baseline,
    environment_differences,
    load_baseline,
    validate_baseline_record,
    write_baseline,
)
from app.synthetic import make_synthetic_transactions
from app.training import run_training_pipeline, train_model

BASELINE_CI = cfg.BASELINE_DIR / "baseline_ci.json"
REFERENCE = {
    "average_precision": 0.80,
    "roc_auc": 0.98,
    "recall": 0.70,
    "precision": 0.85,
    "fpr": 0.0005,
}


def _candidate(**changes):
    return {**REFERENCE, **changes}


# -------------------------------------------------------------- règles de décision
def test_identical_candidate_is_accepted():
    result = compare_to_baseline(_candidate(), REFERENCE)
    assert result["decision"] == DECISION_ACCEPT and not result["reasons"]


def test_improvement_within_reason_is_accepted():
    assert (
        compare_to_baseline(_candidate(recall=0.72, roc_auc=0.985), REFERENCE)["decision"]
        == DECISION_ACCEPT
    )


def test_small_degradation_requires_investigation():
    result = compare_to_baseline(_candidate(average_precision=0.80 - 0.015), REFERENCE)
    assert result["decision"] == DECISION_INVESTIGATE
    assert result["details"]["average_precision"]["status"] == "warn"


def test_large_degradation_is_rejected():
    result = compare_to_baseline(_candidate(recall=0.70 - 0.05), REFERENCE)
    assert result["decision"] == DECISION_REJECT and result["details"]["recall"]["status"] == "fail"


def test_a_single_failing_metric_is_enough_to_reject():
    result = compare_to_baseline(_candidate(fpr=0.0005 + 0.0003), REFERENCE)
    assert result["decision"] == DECISION_REJECT


def test_fpr_direction_is_inverted_lower_is_better():
    better = compare_to_baseline(_candidate(fpr=0.0001), REFERENCE)
    assert better["decision"] == DECISION_ACCEPT and better["details"]["fpr"]["degradation"] < 0


def test_suspiciously_large_gain_triggers_investigation():
    result = compare_to_baseline(_candidate(average_precision=0.80 + 0.08), REFERENCE)
    assert result["decision"] == DECISION_INVESTIGATE
    assert any("fuite" in reason for reason in result["reasons"])


def test_tolerance_boundaries_are_strict():
    warn = cfg.REGRESSION_TOLERANCES["recall"]["warn"]
    at_limit = compare_to_baseline(_candidate(recall=0.70 - warn + 1e-12), REFERENCE)
    over = compare_to_baseline(_candidate(recall=0.70 - warn - 1e-6), REFERENCE)
    assert at_limit["decision"] == DECISION_ACCEPT and over["decision"] == DECISION_INVESTIGATE


def test_comparison_fails_loudly_on_missing_or_nan_metrics():
    incomplete = {k: v for k, v in REFERENCE.items() if k != "recall"}
    with pytest.raises(KeyError):
        compare_to_baseline(incomplete, REFERENCE)
    with pytest.raises(ValueError):
        compare_to_baseline(_candidate(recall=float("nan")), REFERENCE)


# ------------------------------------------- la référence n'est jamais recalculée seule
def test_missing_baseline_is_an_error_and_is_never_recreated(tmp_path):
    target = tmp_path / "baseline.json"
    with pytest.raises(FileNotFoundError, match="jamais créée automatiquement"):
        load_baseline(target)
    assert not target.exists()


def test_write_baseline_refuses_to_overwrite_an_existing_reference(tmp_path):
    record = json.loads(BASELINE_CI.read_text(encoding="utf-8"))
    target = tmp_path / "baseline.json"
    write_baseline(record, target)
    with pytest.raises(FileExistsError, match="recalculée automatiquement"):
        write_baseline(record, target)
    write_baseline(record, target, overwrite=True)


def test_incomplete_baseline_record_is_refused():
    record = json.loads(BASELINE_CI.read_text(encoding="utf-8"))
    del record["seeds"]
    with pytest.raises(ValueError, match="seeds"):
        validate_baseline_record(record)


def test_ci_baseline_records_everything_required_by_the_assignment():
    record = load_baseline(BASELINE_CI)
    assert record["code_version"] and "git_commit" in record  # version du code
    assert record["model"]["params"] is not None  # paramètres du modèle
    assert record["environment"]["libraries"]["scikit-learn"]  # version des bibliothèques
    assert {"split", "model"} <= set(record["seeds"])  # graines aléatoires
    assert set(cfg.REGRESSION_TOLERANCES) <= set(record["validation_metrics"])  # métriques
    assert 0.0 < record["model"]["threshold"] <= 1.0  # seuil de décision
    assert record["dataset"]["n_rows"] > 0 and "n_fraud" in record["dataset"]  # jeu de données


def test_environment_differences_reports_changed_versions():
    record = load_baseline(BASELINE_CI)
    current = {"libraries": dict(record["environment"]["libraries"])}
    assert environment_differences(record["environment"], current) == {}
    current["libraries"]["scikit-learn"] = "0.0.1"
    diff = environment_differences(record["environment"], current)
    assert diff["scikit-learn"]["current"] == "0.0.1"


def test_build_baseline_record_is_complete():
    metrics = dict(REFERENCE)
    record = build_baseline_record(
        "logreg",
        {"C": 1.0},
        0.9,
        {"split": 1, "model": 1},
        {"n_rows": 10, "n_fraud": 1},
        metrics,
        "2026-01-01T00:00:00+00:00",
    )
    validate_baseline_record(record)
    assert record["model"]["threshold"] == 0.9


# --------------------------------------------------- T06/T16 : code courant vs référence figée
def _ci_run(overrides=None, shuffle_labels=False):
    record = load_baseline(BASELINE_CI)
    params = record["dataset"]["synthetic_params"]
    df = make_synthetic_transactions(**params)
    if shuffle_labels:
        df = df.assign(Class=np.random.default_rng(0).permutation(df["Class"].to_numpy()))
    model_params = {**record["model"]["params"], **(overrides or {})}
    run = run_training_pipeline(
        df,
        record["model"]["name"],
        model_params,
        max_fpr=record["model"]["max_fpr"],
        random_state=record["seeds"]["model"],
    )
    return record, {k: run.val_metrics[k] for k in cfg.REGRESSION_TOLERANCES}


def test_current_code_matches_the_frozen_ci_baseline():
    record, measured = _ci_run()
    result = compare_to_baseline(measured, record["validation_metrics"], record["tolerances"])
    assert result["decision"] == DECISION_ACCEPT, result["reasons"]


def test_ci_baseline_dataset_has_the_recorded_characteristics():
    record = load_baseline(BASELINE_CI)
    df = make_synthetic_transactions(**record["dataset"]["synthetic_params"])
    assert len(df) == record["dataset"]["n_rows"]
    assert int(df["Class"].sum()) == record["dataset"]["n_fraud"]


def test_neutral_refactoring_is_accepted_by_the_gate():
    record, measured = _ci_run(overrides={"max_iter": 5000})
    assert (
        compare_to_baseline(measured, record["validation_metrics"], record["tolerances"])[
            "decision"
        ]
        == DECISION_ACCEPT
    )


def test_a_degraded_model_is_rejected_by_the_gate():
    """Candidat volontairement défectueux (étiquettes mélangées) : doit être REJETÉ."""
    record, measured = _ci_run(shuffle_labels=True)
    result = compare_to_baseline(measured, record["validation_metrics"], record["tolerances"])
    assert result["decision"] == DECISION_REJECT


# ----------------------------------------------------------------- reproductibilité (T06)
@pytest.mark.parametrize("name", ["logreg", "tree"])
def test_two_trainings_with_same_seed_give_identical_probabilities(name, splits):
    a = train_model(splits.X_train, splits.y_train, name, random_state=7)
    b = train_model(splits.X_train, splits.y_train, name, random_state=7)
    pa, pb = a.predict_proba(splits.X_test)[:, 1], b.predict_proba(splits.X_test)[:, 1]
    ca07 = next(c for c in cfg.ACCEPTANCE_CRITERIA if c["id"] == "CA07")
    assert np.max(np.abs(pa - pb)) <= ca07["value"]
