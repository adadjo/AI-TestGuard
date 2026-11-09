"""Tests de la surveillance (T22) : PSI, KS, rapport de dérive, alertes simulées."""

from __future__ import annotations

import numpy as np
import pytest

from app import config as cfg
from app.monitoring import (
    alert_rate,
    drift_report,
    ks_statistic,
    population_stability_index,
    window_report,
)
from app.preprocessing import preprocess_data
from app.robustness import apply_covariate_shift, reference_sigma
from app.synthetic import make_synthetic_transactions
from app.training import select_threshold, train_model


@pytest.fixture(scope="module")
def stream():
    """Flux synthétique : modèle appris sur 10 000 lignes ; référence et fenêtre i.i.d."""
    X, y = preprocess_data(make_synthetic_transactions(30_000, 0.02, seed=3))
    model = train_model(X.iloc[:10_000], y.iloc[:10_000])
    ref_X, cur_X = X.iloc[10_000:20_000], X.iloc[20_000:]
    ref_s = model.predict_proba(ref_X)[:, 1]
    cur_s = model.predict_proba(cur_X)[:, 1]
    thr = select_threshold(y.iloc[10_000:20_000], ref_s, max_fpr=0.01).threshold
    return {
        "model": model,
        "ref_X": ref_X,
        "cur_X": cur_X,
        "ref_s": ref_s,
        "cur_s": cur_s,
        "thr": thr,
        "sigma": reference_sigma(X.iloc[:10_000]),
    }


def test_psi_is_zero_for_identical_and_small_for_same_distribution():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=5000), rng.normal(size=5000)
    assert population_stability_index(a, a) == pytest.approx(0.0, abs=1e-12)
    assert population_stability_index(a, b) < 0.05


def test_psi_detects_mean_shift_and_scale_change():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=5000)
    assert population_stability_index(ref, rng.normal(1.0, 1.0, 5000)) > cfg.PSI_ALERT
    assert population_stability_index(ref, rng.normal(0.0, 2.0, 5000)) > cfg.PSI_WARN


def test_psi_edge_cases():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=1000)
    with_nan = np.concatenate([rng.normal(size=1000), [np.nan, np.inf]])
    assert np.isfinite(population_stability_index(ref, with_nan))
    assert population_stability_index(np.full(100, 3.0), np.full(100, 3.0)) == pytest.approx(0.0)
    assert population_stability_index(np.full(100, 3.0), np.full(100, 9.0)) > cfg.PSI_ALERT
    with pytest.raises(ValueError):
        population_stability_index([], [1.0])
    with pytest.raises(ValueError):
        population_stability_index([1.0, 2.0], [np.nan])


def test_psi_detects_a_shift_of_a_discrete_or_constant_reference():
    """Régression : une référence constante ou binaire rendait le PSI aveugle (valeur 0)."""
    rng = np.random.default_rng(4)
    binary_ref = (rng.random(4000) < 0.5).astype(float)
    binary_cur = (rng.random(4000) < 0.9).astype(float)
    assert population_stability_index(binary_ref, binary_cur) > cfg.PSI_ALERT
    assert population_stability_index(binary_ref, binary_ref) == pytest.approx(0.0)
    mostly_zero = np.concatenate([np.zeros(990), rng.uniform(1, 100, 100)])
    assert population_stability_index(mostly_zero, np.full(500, 50.0)) > cfg.PSI_ALERT


def test_ks_statistic_range():
    rng = np.random.default_rng(3)
    a = rng.normal(size=2000)
    same = ks_statistic(a, a)
    shifted = ks_statistic(a, a + 2.0)
    assert same["statistic"] == 0.0 and shifted["statistic"] > 0.5 and shifted["pvalue"] < 1e-6


def test_drift_report_levels(stream):
    stable = drift_report(stream["ref_X"], stream["cur_X"])
    assert {r["level"] for r in stable} == {"ok"}
    shifted = apply_covariate_shift(stream["cur_X"], stream["sigma"], 1.0, amount_factor=3.0)
    levels = {r["column"]: r["level"] for r in drift_report(stream["ref_X"], shifted)}
    assert levels["Amount"] == "alert" and "alert" in levels.values()


def test_window_report_is_quiet_on_a_stable_window(stream):
    rep = window_report(
        stream["ref_X"],
        stream["ref_s"],
        stream["cur_X"],
        stream["cur_s"],
        stream["thr"],
        float((stream["ref_s"] >= stream["thr"]).mean()),
    )
    assert rep["alerts"] == [] and rep["n_features_in_alert"] == 0


def test_window_report_raises_alerts_on_drifted_inputs(stream):
    shifted = apply_covariate_shift(stream["cur_X"], stream["sigma"], 1.0, amount_factor=3.0)
    scores = stream["model"].predict_proba(shifted)[:, 1]
    rep = window_report(
        stream["ref_X"],
        stream["ref_s"],
        shifted,
        scores,
        stream["thr"],
        float((stream["ref_s"] >= stream["thr"]).mean()),
        invalid_rate=0.05,
    )
    names = {a["name"] for a in rep["alerts"]}
    assert {"data_drift", "invalid_inputs"} <= names
    assert rep["n_features_in_alert"] > 0


def test_window_report_flags_a_collapse_of_the_alert_rate(stream):
    """Dérive de concept simulée (mutant M23) : plus aucune alerte alors que la référence en émettait."""
    ref_rate = float((stream["ref_s"] >= stream["thr"]).mean())
    collapsed = np.zeros(len(stream["cur_X"]))
    rep = window_report(
        stream["ref_X"], stream["ref_s"], stream["cur_X"], collapsed, stream["thr"], ref_rate
    )
    assert any(a["name"] == "alert_rate" for a in rep["alerts"])
    assert rep["alert_rate_z"] < -cfg.ALERT_RATE_Z


def test_alert_rate_rule_needs_both_a_significant_and_a_large_change(stream):
    """+40 % d'alertes : écart significatif (z > 3) mais variation relative < 50 % : pas d'alerte."""
    n = len(stream["cur_X"])
    scores = np.zeros(n)
    scores[: int(0.014 * n)] = 0.9
    rep = window_report(stream["ref_X"], stream["ref_s"], stream["cur_X"], scores, 0.5, 0.01)
    assert rep["alert_rate_z"] > cfg.ALERT_RATE_Z
    assert all(a["name"] != "alert_rate" for a in rep["alerts"])


def test_alert_rate_validations():
    assert alert_rate([0.1, 0.9, 0.95, 0.2], 0.9) == 0.5
    with pytest.raises(ValueError):
        alert_rate([], 0.5)
    with pytest.raises(ValueError):
        alert_rate([0.1, np.nan], 0.5)
