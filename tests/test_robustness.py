"""Tests de robustesse et de sensibilité (T14, T15) : scénarios obligatoires de l'Activité 5.

Pour chaque scénario, le comportement attendu est explicite : REJETER, TRAITER ou ALERTER.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app import config as cfg
from app.prediction import predict_fraud
from app.preprocessing import DataValidationError
from app.robustness import (
    apply_covariate_shift,
    count_invalid_rows,
    inject_missing,
    logreg_delta_bound,
    perturb_features,
    reference_sigma,
    sensitivity_report,
)


# --------------------------------------------------- scénario 1 : données manquantes (TRAITER + ALERTER)
def test_missing_values_are_imputed_and_flagged(caplog, trained_lr, splits, lr_threshold):
    holes = inject_missing(splits.X_test, 0.01, seed=1)
    assert holes.isna().to_numpy().any()
    with caplog.at_level(logging.WARNING, logger="aitestguard"):
        out = predict_fraud(trained_lr, holes, lr_threshold)
    assert np.isfinite(out["fraud_probability"]).all()
    assert out["fraud_probability"].between(0, 1).all()
    assert any("manquante" in r.message for r in caplog.records)


def test_excessive_missing_values_are_rejected(trained_lr, splits, lr_threshold):
    holes = inject_missing(splits.X_test, 0.20, seed=1)
    with pytest.raises(DataValidationError, match="manquantes"):
        predict_fraud(trained_lr, holes, lr_threshold)


# --------------------------------- scénario 2 : infinies / non numériques (REJETER)
@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_infinite_values_are_rejected(trained_lr, splits, lr_threshold, bad):
    X = splits.X_test.copy()
    X.loc[X.index[0], "V10"] = bad
    with pytest.raises(DataValidationError, match="infinies"):
        predict_fraud(trained_lr, X, lr_threshold)


def test_non_numeric_values_are_rejected(trained_lr, splits, lr_threshold):
    X = splits.X_test.copy()
    X["V10"] = X["V10"].astype(str)
    with pytest.raises(DataValidationError, match="numériques"):
        predict_fraud(trained_lr, X, lr_threshold)


# ------------------------------ scénario 3 : Amount extrême (ALERTER puis REJETER au-delà du contrat)
def test_extreme_amount_is_processed_with_an_alert(caplog, trained_lr, splits, lr_threshold):
    X = splits.X_test.head(50).copy()
    X.loc[X.index[0], "Amount"] = 5 * cfg.AMOUNT_ALERT_THRESHOLD
    with caplog.at_level(logging.WARNING, logger="aitestguard"):
        out = predict_fraud(trained_lr, X, lr_threshold)
    assert out["fraud_probability"].between(0, 1).all()
    assert any("seuil d'alerte" in r.message for r in caplog.records)


def test_absurd_amount_is_rejected(trained_lr, splits, lr_threshold):
    X = splits.X_test.head(50).copy()
    X.loc[X.index[0], "Amount"] = 1e9
    with pytest.raises(DataValidationError, match="hors contrat"):
        predict_fraud(trained_lr, X, lr_threshold)


# ------------------------------------------------------ scénario 4 : schéma invalide (REJETER)
def test_invalid_schema_is_rejected(trained_lr, splits, lr_threshold):
    with pytest.raises(DataValidationError, match="absentes"):
        predict_fraud(trained_lr, splits.X_test.drop(columns=["V1", "V2"]), lr_threshold)
    with pytest.raises(DataValidationError, match="inattendue"):
        predict_fraud(trained_lr, splits.X_test.assign(extra=1.0), lr_threshold)


# --------------------------------------------------- scénario 5 : perturbation limitée (SENSIBILITÉ)
def test_zero_perturbation_changes_nothing(trained_lr, splits, lr_threshold):
    sigma = reference_sigma(splits.X_train)
    same = perturb_features(splits.X_test, sigma, eps_v=0.0, eps_amount=0.0)
    pd.testing.assert_frame_equal(same, splits.X_test)
    report = sensitivity_report(trained_lr, splits.X_test, splits.y_test, same, lr_threshold)
    assert report["flip_rate"] == 0.0 and report["max_abs_dp"] == 0.0
    assert report["delta_recall"] == 0.0 and report["n_invalid"] == 0


def test_perturbation_protocol_matches_its_documentation(splits):
    sigma = reference_sigma(splits.X_train)
    noisy = perturb_features(splits.X_test, sigma, eps_v=0.10, eps_amount=0.05, seed=5)
    z = (noisy[cfg.V_COLUMNS] - splits.X_test[cfg.V_COLUMNS]) / sigma
    assert z.to_numpy().std() == pytest.approx(0.10, rel=0.10)
    assert (noisy["Time"] == splits.X_test["Time"]).all()
    assert (noisy["Amount"] >= 0).all()
    ratio = (noisy["Amount"] / splits.X_test["Amount"].replace(0, np.nan)).dropna()
    assert ratio.between(0.95, 1.05).all()


def test_perturbation_is_reproducible_and_does_not_modify_its_input(splits):
    sigma = reference_sigma(splits.X_train)
    snapshot = splits.X_test.copy(deep=True)
    a = perturb_features(splits.X_test, sigma, seed=3)
    b = perturb_features(splits.X_test, sigma, seed=3)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(splits.X_test, snapshot)
    with pytest.raises(ValueError):
        perturb_features(splits.X_test, sigma, eps_v=-0.1)


def test_logistic_regression_sensitivity_respects_its_theoretical_bound(trained_lr, splits):
    sigma = reference_sigma(splits.X_train)
    for eps in (0.01, 0.05, 0.25):
        noisy = perturb_features(splits.X_test, sigma, eps_v=eps, eps_amount=eps, seed=2)
        dp = np.abs(
            trained_lr.predict_proba(noisy)[:, 1] - trained_lr.predict_proba(splits.X_test)[:, 1]
        )
        assert (dp <= logreg_delta_bound(trained_lr, splits.X_test, noisy) + 1e-12).all()


def test_sensitivity_grows_with_the_noise_amplitude(trained_lr, splits, lr_threshold):
    sigma = reference_sigma(splits.X_train)
    means = []
    for eps in (0.01, 0.05, 0.25):
        noisy = perturb_features(splits.X_test, sigma, eps_v=eps, eps_amount=eps, seed=2)
        rep = sensitivity_report(trained_lr, splits.X_test, splits.y_test, noisy, lr_threshold)
        means.append(rep["mean_abs_dp"])
    assert means == sorted(means) and means[0] < means[-1]


def test_small_perturbation_flips_few_decisions(trained_lr, splits, lr_threshold):
    """On mesure la sensibilité : un petit bruit peut changer la classe près de la frontière."""
    sigma = reference_sigma(splits.X_train)
    noisy = perturb_features(splits.X_test, sigma, eps_v=0.05, eps_amount=0.05, seed=4)
    rep = sensitivity_report(trained_lr, splits.X_test, splits.y_test, noisy, lr_threshold)
    assert rep["n_flips"] == rep["flips_to_fraud"] + rep["flips_to_normal"]
    assert rep["flip_rate"] < 0.05


# ------------------------------------------------------ scénario 6 : décalage de distribution
def test_covariate_shift_degrades_or_moves_scores(trained_lr, splits, lr_threshold):
    sigma = reference_sigma(splits.X_train)
    shifted = apply_covariate_shift(splits.X_test, sigma, shift_sigma=1.0, amount_factor=2.0)
    base = predict_fraud(trained_lr, splits.X_test, lr_threshold)["fraud_probability"]
    moved = predict_fraud(trained_lr, shifted, lr_threshold)["fraud_probability"]
    assert not np.allclose(base.to_numpy(), moved.to_numpy())
    assert np.isfinite(moved).all() and moved.between(0, 1).all()


def test_helpers_validate_their_arguments(splits):
    with pytest.raises(ValueError):
        inject_missing(splits.X_test, 1.5)
    broken = splits.X_test.copy()
    broken.iloc[0, broken.columns.get_loc("Amount")] = -1.0
    broken.iloc[1, broken.columns.get_loc("V3")] = np.inf
    assert count_invalid_rows(broken) == 2
    assert count_invalid_rows(splits.X_test) == 0


# ------------------------------------------------------------- propriétés métamorphiques
def test_decision_tree_predictions_are_row_independent_and_time_invariant(trained_tree, splits):
    X = splits.X_test.head(300)
    base = trained_tree.predict_proba(X)[:, 1]
    perm = np.random.default_rng(0).permutation(len(X))
    np.testing.assert_array_equal(trained_tree.predict_proba(X.iloc[perm])[:, 1], base[perm])
    np.testing.assert_array_equal(
        trained_tree.predict_proba(X.assign(Time=X["Time"] * 3))[:, 1], base
    )


@settings(max_examples=30, deadline=None, derandomize=True)
@given(
    seed=st.integers(0, 10_000),
    scale=st.floats(0.0, 30.0),
    amount=st.floats(0.0, 2e5),
    n=st.integers(1, 20),
)
def test_property_any_in_contract_input_gives_a_valid_probability(
    trained_lr, seed, scale, amount, n
):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(0.0, scale, (n, 28)), columns=cfg.V_COLUMNS)
    df.insert(0, "Time", rng.uniform(0, 172_792, n))
    df["Amount"] = amount
    out = predict_fraud(trained_lr, df, 0.5)
    assert np.isfinite(out["fraud_probability"]).all()
    assert out["fraud_probability"].between(0, 1).all()
