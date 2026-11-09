"""Tests de la prédiction : contrat de sortie (T04), service et gestionnaire de requête (T18)."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from app import config as cfg
from app.prediction import PredictionError, handle_prediction_request, predict_fraud
from app.preprocessing import DataValidationError
from tests.conftest import StubModel


# ------------------------------------------------------------------------ T04 (énoncé)
def test_prediction_probabilities_are_valid(trained_lr, splits):
    probabilities = trained_lr.predict_proba(splits.X_test)[:, 1]

    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_model_class_order_is_0_then_1(trained_lr, trained_tree):
    assert list(trained_lr.classes_) == [0, 1]
    assert list(trained_tree.classes_) == [0, 1]


def test_predict_fraud_output_contract(trained_lr, splits, lr_threshold):
    out = predict_fraud(trained_lr, splits.X_test, lr_threshold)
    assert list(out.columns) == ["fraud_probability", "is_fraud"]
    assert len(out) == len(splits.X_test)
    assert out.index.equals(splits.X_test.index)
    assert set(out["is_fraud"].unique()) <= {0, 1}
    assert out["fraud_probability"].between(0, 1).all()
    assert np.isfinite(out["fraud_probability"]).all()


def test_predict_fraud_decision_is_probability_above_threshold(trained_lr, splits, lr_threshold):
    out = predict_fraud(trained_lr, splits.X_test, lr_threshold)
    expected = (out["fraud_probability"] >= lr_threshold).astype(int)
    assert (out["is_fraud"] == expected).all()


def test_threshold_boundary_is_inclusive(valid_df):
    model = StubModel(proba=[0.5, 0.7, 0.2])
    out = predict_fraud(model, valid_df.head(3), threshold=0.5)
    assert out["is_fraud"].tolist() == [1, 1, 0]  # p == seuil => fraude


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), "0.5", None, True])
def test_predict_fraud_rejects_invalid_threshold(trained_lr, valid_df, bad):
    with pytest.raises(ValueError, match="seuil"):
        predict_fraud(trained_lr, valid_df, bad)


def test_predict_fraud_rejects_invalid_inputs(trained_lr, valid_df):
    broken = valid_df.copy()
    broken.loc[broken.index[0], "V4"] = np.inf
    with pytest.raises(DataValidationError):
        predict_fraud(trained_lr, broken, 0.5)
    with pytest.raises(DataValidationError):
        predict_fraud(trained_lr, valid_df.drop(columns=["Amount"]), 0.5)
    with pytest.raises(DataValidationError):
        predict_fraud(trained_lr, valid_df.iloc[0:0], 0.5)


def test_predict_fraud_single_row(trained_lr, valid_df):
    out = predict_fraud(trained_lr, valid_df.iloc[[0]], 0.5)
    assert out.shape == (1, 2)


def test_predict_fraud_ignores_target_column_when_present(trained_lr, splits, lr_threshold):
    with_target = splits.X_test.assign(Class=splits.y_test)
    a = predict_fraud(trained_lr, with_target, lr_threshold)
    b = predict_fraud(trained_lr, splits.X_test, lr_threshold)
    pd.testing.assert_frame_equal(a, b)


def test_predict_fraud_is_deterministic(trained_lr, splits, lr_threshold):
    a = predict_fraud(trained_lr, splits.X_test, lr_threshold)
    b = predict_fraud(trained_lr, splits.X_test, lr_threshold)
    pd.testing.assert_frame_equal(a, b)


def test_predict_fraud_does_not_depend_on_column_order(trained_lr, splits, lr_threshold):
    shuffled = splits.X_test[list(reversed(cfg.FEATURE_COLUMNS))]
    a = predict_fraud(trained_lr, splits.X_test, lr_threshold)
    b = predict_fraud(trained_lr, shuffled, lr_threshold)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize(
    "model",
    [
        StubModel(proba=[np.nan, 0.2, 0.3]),
        StubModel(proba=[0.1, 1.2, 0.3]),
        StubModel(proba=[0.1, -0.2, 0.3]),
        StubModel(raw=np.zeros((2, 2))),
        StubModel(raw=np.zeros((3, 3))),
    ],
    ids=["nan", "above-1", "below-0", "wrong-length", "wrong-width"],
)
def test_predict_fraud_detects_model_contract_violations(valid_df, model):
    with pytest.raises(PredictionError):
        predict_fraud(model, valid_df.head(3), 0.5)


def test_prediction_is_independent_across_rows(trained_lr, splits, lr_threshold):
    """Test métamorphique : permuter les lignes permute simplement les prédictions."""
    X = splits.X_test.head(200)
    permuted = X.sample(frac=1.0, random_state=1)
    a = predict_fraud(trained_lr, X, lr_threshold)["fraud_probability"]
    b = predict_fraud(trained_lr, permuted, lr_threshold)["fraud_probability"]
    np.testing.assert_allclose(b.loc[X.index].to_numpy(), a.to_numpy(), atol=1e-12)


def test_time_has_no_influence_on_prediction(trained_lr, splits):
    """Propriété de conception : Time est exclu du modèle."""
    shifted = splits.X_test.assign(Time=splits.X_test["Time"] + 50_000.0)
    a = trained_lr.predict_proba(splits.X_test)[:, 1]
    b = trained_lr.predict_proba(shifted)[:, 1]
    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------ gestionnaire de requête (type API)
def test_handler_accepts_valid_request_as_json_text_bytes_or_dict(
    trained_lr, lr_threshold, valid_payload
):
    results = []
    for raw in (json.dumps(valid_payload), json.dumps(valid_payload).encode(), valid_payload):
        status, body = handle_prediction_request(raw, trained_lr, lr_threshold)
        assert status == 200
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["is_fraud"] in (0, 1)
        assert {"request_id", "threshold", "model_version"} <= set(body)
        results.append(body["fraud_probability"])
    assert len(set(results)) == 1


def test_handler_reports_missing_fields(trained_lr, lr_threshold, valid_payload):
    del valid_payload["V7"]
    status, body = handle_prediction_request(valid_payload, trained_lr, lr_threshold)
    assert (status, body["error"], body["fields"]) == (422, "missing_fields", ["V7"])


def test_handler_rejects_unexpected_fields_without_echoing_their_names(
    trained_lr, lr_threshold, valid_payload
):
    valid_payload["champ_pirate_12345"] = 1.0
    status, body = handle_prediction_request(valid_payload, trained_lr, lr_threshold)
    assert (status, body["error"], body["count"]) == (422, "unexpected_fields", 1)
    assert "pirate" not in json.dumps(body)


@pytest.mark.parametrize(
    "bad", ["abc", None, True, [1.0], {"x": 1}], ids=lambda v: type(v).__name__
)
def test_handler_rejects_wrong_types(trained_lr, lr_threshold, valid_payload, bad):
    valid_payload["V3"] = bad
    status, body = handle_prediction_request(valid_payload, trained_lr, lr_threshold)
    assert (status, body["error"]) == (422, "invalid_types")
    assert body["fields"] == ["V3"]


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_handler_rejects_non_finite_json_literals(trained_lr, lr_threshold, valid_payload, literal):
    text = json.dumps(valid_payload).replace(f'"V2": {valid_payload["V2"]}', f'"V2": {literal}')
    assert literal in text
    status, body = handle_prediction_request(text, trained_lr, lr_threshold)
    assert (status, body["error"]) == (400, "invalid_json")


def test_handler_rejects_float_overflow_to_infinity(trained_lr, lr_threshold, valid_payload):
    text = json.dumps(valid_payload).replace(f'"V2": {valid_payload["V2"]}', '"V2": 1e999')
    status, body = handle_prediction_request(text, trained_lr, lr_threshold)
    assert (status, body["error"], body["fields"]) == (422, "non_finite_values", ["V2"])


def test_handler_rejects_values_outside_the_data_contract(trained_lr, lr_threshold, valid_payload):
    valid_payload["Amount"] = -5.0
    status, body = handle_prediction_request(valid_payload, trained_lr, lr_threshold)
    assert (status, body["error"]) == (422, "contract_violation")


@pytest.mark.parametrize("raw", ["{pas du json", "", b"\xff\xfe", "[1, 2, 3]", "42", '"texte"'])
def test_handler_rejects_malformed_or_non_object_payloads(trained_lr, lr_threshold, raw):
    status, body = handle_prediction_request(raw, trained_lr, lr_threshold)
    assert status == 400
    assert body["error"] in {"invalid_json", "payload_must_be_object"}


def test_handler_rejects_oversized_payload(trained_lr, lr_threshold, valid_payload):
    huge = json.dumps({**valid_payload, "padding": "x" * (cfg.MAX_PAYLOAD_BYTES + 10)})
    status, body = handle_prediction_request(huge, trained_lr, lr_threshold)
    assert (status, body["error"]) == (413, "payload_too_large")


def test_handler_masks_internal_errors(lr_threshold, valid_payload):
    class Exploding:
        def predict_proba(self, X):
            raise RuntimeError("détail interne confidentiel")

    status, body = handle_prediction_request(valid_payload, Exploding(), lr_threshold)
    assert (status, body["error"]) == (500, "internal_error")
    assert "confidentiel" not in json.dumps(body)


def test_handler_request_ids_are_unique(trained_lr, lr_threshold, valid_payload):
    ids = {
        handle_prediction_request(valid_payload, trained_lr, lr_threshold)[1]["request_id"]
        for _ in range(20)
    }
    assert len(ids) == 20


def test_handler_never_logs_input_values(caplog, trained_lr, lr_threshold, valid_payload):
    sentinel = 123456.789
    valid_payload["Amount"] = sentinel
    valid_payload["V1"] = 31.41592653
    bad = {**valid_payload, "V2": "31.41592653xyz", "champ_secret_abc": 7.0}
    negative = {**valid_payload, "Amount": -sentinel}
    with caplog.at_level(logging.DEBUG, logger="aitestguard"):
        handle_prediction_request(valid_payload, trained_lr, lr_threshold)
        handle_prediction_request(bad, trained_lr, lr_threshold)
        handle_prediction_request(negative, trained_lr, lr_threshold)
        handle_prediction_request(json.dumps(valid_payload)[:-5], trained_lr, lr_threshold)
    assert caplog.records, "des journaux d'audit doivent exister"
    for forbidden in ("123456", "31.4159", "champ_secret_abc"):
        assert forbidden not in caplog.text
