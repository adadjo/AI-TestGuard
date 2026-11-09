"""Garde-fous de performance (T17) : détectent un changement d'ordre de grandeur.

Ce ne sont PAS les mesures d'acceptation (voir ``scripts/run_performance.py`` et son protocole) :
les seuils ci-dessous sont volontairement très larges pour rester stables sur un runner partagé.
"""

from __future__ import annotations

import time

import numpy as np

from app.prediction import handle_prediction_request, predict_fraud
from app.training import save_model


def _p95_ms(callable_, repeats: int = 40) -> float:
    callable_()  # échauffement
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_()
        times.append((time.perf_counter() - start) * 1000.0)
    return float(np.percentile(times, 95))


def test_single_prediction_latency_has_the_expected_order_of_magnitude(trained_lr, splits):
    row = splits.X_test.iloc[[0]]
    assert _p95_ms(lambda: predict_fraud(trained_lr, row, 0.5)) < 250.0


def test_request_handler_latency_has_the_expected_order_of_magnitude(
    trained_lr, lr_threshold, valid_payload
):
    assert (
        _p95_ms(lambda: handle_prediction_request(valid_payload, trained_lr, lr_threshold)) < 250.0
    )


def test_batch_throughput_has_the_expected_order_of_magnitude(trained_lr, splits):
    batch = splits.X_test.head(900)
    predict_fraud(trained_lr, batch, 0.5)
    start = time.perf_counter()
    predict_fraud(trained_lr, batch, 0.5)
    throughput = len(batch) / (time.perf_counter() - start)
    assert throughput > 500.0


def test_serialized_model_is_small(tmp_path, trained_lr, trained_tree):
    for name, model in (("lr", trained_lr), ("tree", trained_tree)):
        meta = save_model(model, 0.5, tmp_path / f"{name}.joblib")
        assert meta["size_bytes"] < 10 * 1024 * 1024
