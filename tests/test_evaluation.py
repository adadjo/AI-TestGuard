"""Tests des métriques sur données déséquilibrées (T21), du seuil (T10) et des critères."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import average_precision_score, roc_auc_score

from app import config as cfg
from app.evaluation import (
    bootstrap_compare,
    check_criteria,
    cost_summary,
    evaluate_model,
    meets_performance_criteria,
    qualification_verdict,
    segment_report,
    threshold_table,
    wilson_interval,
)
from app.training import select_candidate, select_threshold


def test_evaluate_model_matches_hand_computation():
    y_true = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    m = evaluate_model(y_true, y_pred)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 7)
    assert m["fnr"] == pytest.approx(1 / 3)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["n_alerts"] == 3


def test_evaluate_model_perfect_and_inverted_classifiers():
    y = [0, 0, 1, 1, 0, 1]
    perfect = evaluate_model(y, y, [0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    assert (perfect["recall"], perfect["precision"], perfect["fpr"]) == (1.0, 1.0, 0.0)
    assert perfect["roc_auc"] == 1.0 and perfect["average_precision"] == 1.0
    inverted = evaluate_model(y, [1 - v for v in y], [0.9, 0.8, 0.1, 0.2, 0.7, 0.3])
    assert (inverted["recall"], inverted["precision"], inverted["fpr"]) == (0.0, 0.0, 1.0)
    assert inverted["roc_auc"] == 0.0


def test_accuracy_is_misleading_on_imbalanced_data():
    """Un classifieur qui ne signale JAMAIS de fraude a une accuracy > 99,8 % et un rappel nul."""
    y_true = np.zeros(100_000, dtype=int)
    y_true[:170] = 1  # prévalence 0,17 %
    m = evaluate_model(y_true, np.zeros_like(y_true))
    assert m["accuracy"] > 0.998
    assert m["recall"] == 0.0 and m["precision"] == 0.0 and m["f1"] == 0.0
    assert m["fnr"] == 1.0


def test_undefined_metrics_are_nan_not_zero():
    m = evaluate_model([0, 0, 0, 0], [0, 1, 0, 0], [0.1, 0.9, 0.2, 0.1])
    assert math.isnan(m["recall"]) and math.isnan(m["fnr"]) and math.isnan(m["f1"])
    assert math.isnan(m["roc_auc"]) and math.isnan(m["average_precision"])
    assert m["fpr"] == pytest.approx(0.25)


def test_precision_is_zero_without_alerts():
    m = evaluate_model([1, 0, 1, 0], [0, 0, 0, 0])
    assert m["precision"] == 0.0 and m["n_alerts"] == 0


@pytest.mark.parametrize(
    ("y_true", "y_pred", "y_proba"),
    [
        ([1, 0], [1], None),
        ([], [], None),
        ([0, 2], [0, 1], None),
        ([0, 1], [0, np.nan], None),
        ([[0, 1]], [[0, 1]], None),
        ([0, 1], [0, 1], [0.2, 1.2]),
        ([0, 1], [0, 1], [0.2, np.nan]),
        ([0, 1], [0, 1], [0.2]),
    ],
    ids=["length", "empty", "label-2", "nan-pred", "2d", "proba>1", "proba-nan", "proba-length"],
)
def test_evaluate_model_rejects_invalid_inputs(y_true, y_pred, y_proba):
    with pytest.raises(ValueError):
        evaluate_model(y_true, y_pred, y_proba)


def test_ranking_metrics_match_scikit_learn():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.05).astype(int)
    p = np.clip(rng.normal(0.25 + 0.4 * y, 0.2), 0, 1)
    m = evaluate_model(y, (p >= 0.5).astype(int), p)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["average_precision"] == pytest.approx(average_precision_score(y, p))


def test_average_precision_of_random_scores_is_close_to_prevalence():
    rng = np.random.default_rng(1)
    y = (rng.random(20_000) < 0.01).astype(int)
    m = evaluate_model(y, np.zeros_like(y), rng.random(20_000))
    assert abs(m["average_precision"] - y.mean()) < 0.01
    assert abs(m["roc_auc"] - 0.5) < 0.1


# ----------------------------------------------------------------------- Wilson
def test_wilson_interval_known_values():
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-12) and high == pytest.approx(0.2775, abs=1e-3)
    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(0.7225, abs=1e-3) and high == pytest.approx(1.0, abs=1e-12)
    low, high = wilson_interval(50, 100)
    assert (low + high) / 2 == pytest.approx(0.5, abs=1e-9)


def test_wilson_interval_edge_cases():
    assert all(math.isnan(v) for v in wilson_interval(0, 0))
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


@settings(max_examples=60, deadline=None, derandomize=True)
@given(n=st.integers(1, 5000), data=st.data())
def test_property_wilson_interval_contains_the_estimate_and_stays_in_unit_range(n, data):
    k = data.draw(st.integers(0, n))
    low, high = wilson_interval(k, n)
    assert 0.0 <= low <= k / n <= high <= 1.0


# --------------------------------------------------------------- seuil de décision (T10)
def _scores(seed=0, n=3000, prevalence=0.05):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    p = np.clip(rng.normal(0.2 + 0.5 * y, 0.2), 0, 1)
    return y, p


def test_select_threshold_respects_the_fpr_constraint_and_maximises_recall():
    y, p = _scores()
    choice = select_threshold(y, p, max_fpr=0.02)
    assert choice.fpr <= 0.02
    negatives = y == 0
    best = max(
        ((p >= t) & (y == 1)).sum() / (y == 1).sum()
        for t in np.unique(p)
        if ((p >= t) & negatives).sum() / negatives.sum() <= 0.02
    )
    assert choice.recall == pytest.approx(best)


def test_select_threshold_is_monotonic_in_the_fpr_budget():
    y, p = _scores(seed=3)
    recalls = [select_threshold(y, p, max_fpr=b).recall for b in (0.005, 0.01, 0.05, 0.2)]
    assert recalls == sorted(recalls)


@pytest.mark.parametrize(
    ("y", "p", "budget"),
    [
        (np.zeros(10), np.linspace(0, 1, 10), 0.01),
        (np.ones(10), np.linspace(0, 1, 10), 0.01),
        (np.array([0, 1] * 5), np.full(10, np.nan), 0.01),
        (np.array([0, 1] * 5), np.linspace(0, 1.5, 10), 0.01),
        (np.array([0, 1] * 5), np.linspace(0, 1, 10), 0.0),
        (np.array([0, 1] * 5), np.linspace(0, 1, 10), 1.0),
        (np.array([0, 1, 0]), np.array([0.1, 0.2]), 0.01),
    ],
    ids=["no-fraud", "only-fraud", "nan", "above-1", "budget-0", "budget-1", "length"],
)
def test_select_threshold_rejects_invalid_inputs(y, p, budget):
    with pytest.raises(ValueError):
        select_threshold(y, p, budget)


def test_threshold_table_is_monotonic():
    y, p = _scores(seed=5)
    rows = threshold_table(y, p, np.linspace(0.05, 0.95, 10))
    fps = [r["fp"] for r in rows]
    fns = [r["fn"] for r in rows]
    assert fps == sorted(fps, reverse=True)
    assert fns == sorted(fns)
    assert all(r["tp"] + r["fn"] == int(y.sum()) for r in rows)


def test_cost_summary_adds_missed_fraud_and_false_alert_costs():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]
    amounts = [100.0, 40.0, 7.0, 3.0]
    c = cost_summary(y_true, y_pred, amounts, cost_false_alert=5.0)
    assert c["fraud_amount_total"] == 140.0 and c["fraud_amount_detected"] == 100.0
    assert c["fraud_amount_missed"] == 40.0 and c["false_alert_cost"] == 5.0
    assert c["total_cost"] == 45.0


# --------------------------------------------------------------------- bootstrap
def test_bootstrap_is_reproducible_ordered_and_detects_the_better_model():
    y, good = _scores(seed=7, n=2000)
    rng = np.random.default_rng(9)
    weak = np.clip(good * 0.3 + rng.random(len(good)) * 0.7, 0, 1)
    a = bootstrap_compare(y, {"good": good, "weak": weak}, n_boot=60, seed=1)
    b = bootstrap_compare(y, {"good": good, "weak": weak}, n_boot=60, seed=1)
    assert a == b
    ci = a["models"]["good"]["average_precision"]
    assert ci["low"] <= ci["mean"] <= ci["high"]
    diff = a["differences"]["good_minus_weak"]["roc_auc"]
    assert diff["mean"] > 0 and diff["prob_a_better"] > 0.95


def test_bootstrap_requires_both_classes():
    with pytest.raises(ValueError):
        bootstrap_compare([0, 0, 0], {"m": [0.1, 0.2, 0.3]}, n_boot=5)


# ----------------------------------------------------- critères d'acceptation et verdict
def test_acceptance_criteria_are_well_formed():
    ids = [c["id"] for c in cfg.ACCEPTANCE_CRITERIA]
    assert len(ids) == len(set(ids))
    for c in cfg.ACCEPTANCE_CRITERIA:
        assert c["op"] in {">=", "<=", ">", "<"}
        assert c["severity"] in {"critique", "majeur", "mineur"}
        assert c["rationale"]


def _all_good():
    """Mesures qui satisfont exactement tous les critères."""
    return {
        c["metric"]: (c["value"] + 1 if c["op"] == ">=" else c["value"])
        for c in cfg.ACCEPTANCE_CRITERIA
    }


def test_verdict_rules_critical_major_and_pass():
    measured = _all_good()
    rows = check_criteria(measured)
    assert qualification_verdict(rows)["verdict"].startswith("APTE")

    major = dict(measured, average_precision=0.1)
    assert qualification_verdict(check_criteria(major))["verdict"] == "À INVESTIGUER"

    critical = dict(measured, recall=0.1)
    verdict = qualification_verdict(check_criteria(critical))
    assert verdict["verdict"] == "REJETÉ" and "CA01" in verdict["failed"]["critique"]

    missing = {k: v for k, v in measured.items() if k != "recall"}
    assert qualification_verdict(check_criteria(missing))["verdict"] == "REJETÉ"


def test_criteria_flag_statistical_uncertainty():
    rows = check_criteria({"recall": 0.80}, intervals={"recall": (0.70, 0.88)})
    row = next(r for r in rows if r["metric"] == "recall")
    assert row["passed"] is True and row["ci_straddles_threshold"] is True


# ------------------------------------------------------------ segments (T23, Activité 7)
def test_segment_report_matches_per_group_metrics_and_flags_small_samples():
    y_true = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    groups = np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
    rows = {r["group"]: r for r in segment_report(y_true, y_pred, groups, min_positives=2)}
    assert rows["A"]["recall"] == pytest.approx(0.5) and rows["A"]["fpr"] == pytest.approx(1 / 3)
    assert rows["B"]["recall"] == pytest.approx(1.0) and rows["B"]["n_positive"] == 1
    assert rows["A"]["sufficient_sample"] is True and rows["B"]["sufficient_sample"] is False
    low, high = rows["A"]["recall_ci"]
    assert low <= 0.5 <= high


def test_segment_report_rejects_misaligned_groups():
    with pytest.raises(ValueError):
        segment_report([0, 1], [0, 1], ["A"])


def test_dataset_contract_contains_no_demographic_attribute():
    """Limite essentielle : sans variable de groupe, aucune conclusion d'équité n'est possible."""
    assert set(cfg.REQUIRED_COLUMNS) == {"Time", *cfg.V_COLUMNS, "Amount", "Class"}


# ------------------------------------------------ règle de sélection du modèle (T11)
GOOD = {
    "recall": 0.80,
    "fpr": 0.0005,
    "precision": 0.70,
    "average_precision": 0.70,
    "roc_auc": 0.97,
}


def test_meets_performance_criteria_checks_each_requirement():
    assert meets_performance_criteria(GOOD)
    for key, bad in (
        ("recall", 0.70),
        ("fpr", 0.002),
        ("precision", 0.40),
        ("average_precision", 0.60),
        ("roc_auc", 0.90),
    ):
        assert not meets_performance_criteria({**GOOD, key: bad}), key


def test_select_candidate_prefers_a_qualified_model_over_a_higher_pr_auc():
    """Régression de protocole : l'arbre a la meilleure PR-AUC mais une ROC-AUC trop faible."""
    tree = {**GOOD, "average_precision": 0.81, "roc_auc": 0.915}
    logreg = {**GOOD, "average_precision": 0.77}
    choice = select_candidate({"logreg": logreg, "tree": tree})
    assert choice["r1_pick"] == "tree" and choice["pick"] == "logreg"
    assert choice["qualified_on_validation"] == ["logreg"]


def test_select_candidate_takes_the_best_pr_auc_among_qualified_models():
    a, b = {**GOOD, "average_precision": 0.72}, {**GOOD, "average_precision": 0.78}
    assert select_candidate({"a": a, "b": b})["pick"] == "b"


def test_select_candidate_returns_no_pick_when_nobody_qualifies():
    weak = {**GOOD, "recall": 0.5}
    choice = select_candidate({"a": weak, "b": weak})
    assert choice["pick"] is None and choice["qualified_on_validation"] == []
    with pytest.raises(ValueError):
        select_candidate({})
