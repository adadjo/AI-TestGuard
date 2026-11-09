"""Prédiction : fonction de service ``predict_fraud`` et gestionnaire de requête unitaire.

``handle_prediction_request`` représente la logique d'une API (sans framework) : limite de
taille, JSON strict, schéma strict (champs manquants ou supplémentaires refusés), types
stricts, puis score. Les journaux ne contiennent JAMAIS de valeurs d'entrée.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from typing import Any

import numpy as np
import pandas as pd

from app import __version__
from app import config as cfg
from app.preprocessing import DataValidationError, validate_data

logger = logging.getLogger("aitestguard.prediction")


class PredictionError(RuntimeError):
    """Le modèle a produit une sortie qui viole le contrat (non finie, hors [0, 1], taille)."""


def _check_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.floating)):
        raise ValueError("Le seuil doit être un nombre.")
    value = float(threshold)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Le seuil doit être un nombre fini compris entre 0 et 1.")
    return value


def predict_fraud(model: Any, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Score des transactions : probabilité de fraude et décision ``p >= seuil``.

    Valide l'entrée (contrat de données), vérifie la sortie du modèle (finie, dans [0, 1],
    même longueur) et conserve l'index de l'entrée. Une colonne ``Class`` éventuelle est ignorée.
    """
    thr = _check_threshold(threshold)
    validate_data(X, require_target=False)
    features = X[cfg.FEATURE_COLUMNS].astype("float64")
    raw = np.asarray(model.predict_proba(features))
    if raw.ndim != 2 or raw.shape[0] != len(features) or raw.shape[1] != 2:
        raise PredictionError("Forme de sortie inattendue pour predict_proba.")
    proba = raw[:, 1].astype("float64")
    if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any():
        raise PredictionError("Probabilité non valide produite par le modèle.")
    return pd.DataFrame(
        {"fraud_probability": proba, "is_fraud": (proba > thr).astype("int8")},
        index=X.index,
    )


def _reject_json_constant(name: str) -> Any:
    raise ValueError(f"Constante JSON non finie refusée : {name}")


def _error(status: int, code: str, request_id: str, **extra: Any) -> tuple[int, dict[str, Any]]:
    logger.warning("request_id=%s status=%d error=%s", request_id, status, code)
    return status, {"request_id": request_id, "error": code, **extra}


def handle_prediction_request(
    raw: bytes | str | dict,
    model: Any,
    threshold: float,
    max_bytes: int = cfg.MAX_PAYLOAD_BYTES,
) -> tuple[int, dict[str, Any]]:
    """Traite UNE transaction (charge utile JSON) et renvoie ``(code_http, corps)``.

    400 JSON invalide ; 413 charge utile trop grande ; 422 contrat violé ; 500 erreur interne.
    """
    request_id = uuid.uuid4().hex[:12]
    try:
        if isinstance(raw, (bytes, str)):
            size = len(raw if isinstance(raw, bytes) else raw.encode("utf-8"))
            if size > max_bytes:
                return _error(413, "payload_too_large", request_id)
            try:
                payload = json.loads(raw, parse_constant=_reject_json_constant)
            except ValueError:
                return _error(400, "invalid_json", request_id)
        else:
            payload = raw
        if not isinstance(payload, dict):
            return _error(400, "payload_must_be_object", request_id)

        expected = set(cfg.FEATURE_COLUMNS)
        missing = sorted(expected - set(payload))
        if missing:
            return _error(422, "missing_fields", request_id, fields=missing)
        n_unexpected = len(set(payload) - expected)
        if n_unexpected:
            return _error(422, "unexpected_fields", request_id, count=n_unexpected)
        bad_type = sorted(
            k
            for k in expected
            if isinstance(payload[k], bool) or not isinstance(payload[k], (int, float))
        )
        if bad_type:
            return _error(422, "invalid_types", request_id, fields=bad_type)
        non_finite = sorted(k for k in expected if not math.isfinite(float(payload[k])))
        if non_finite:
            return _error(422, "non_finite_values", request_id, fields=non_finite)

        frame = pd.DataFrame([{k: float(payload[k]) for k in cfg.FEATURE_COLUMNS}])
        try:
            validate_data(frame, require_target=False, allow_missing=False)
        except DataValidationError as exc:
            return _error(422, "contract_violation", request_id, detail=str(exc))

        scored = predict_fraud(model, frame, threshold)
        probability = float(scored["fraud_probability"].iloc[0])
        decision = int(scored["is_fraud"].iloc[0])
        logger.info("request_id=%s status=200 decision=%d", request_id, decision)
        return 200, {
            "request_id": request_id,
            "fraud_probability": round(probability, 6),
            "is_fraud": decision,
            "threshold": float(threshold),
            "model_version": __version__,
        }
    except Exception as exc:  # garde-fou : jamais de trace ni de données dans la réponse
        logger.error("request_id=%s status=500 error=%s", request_id, type(exc).__name__)
        return 500, {"request_id": request_id, "error": "internal_error"}
