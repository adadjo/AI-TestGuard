"""Surveillance après déploiement : dérive des données (PSI, KS), taux d'alertes, alertes simulées.

Métriques disponibles IMMÉDIATEMENT (sans étiquette) : distribution des entrées et du score,
taux de transactions signalées, taux d'entrées invalides, latence. Métriques qui exigent les
étiquettes (retardées : rétrofacturations, enquêtes) : rappel, FPR, précision, PR-AUC.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app import config as cfg


def _psi_edges(ref: np.ndarray, bins: int) -> np.ndarray:
    """Intervalles du PSI. Référence continue : déciles. Référence constante ou discrète
    (<= ``bins`` valeurs distinctes) : une classe par valeur + zones « en dessous / entre / au
    dessus », pour qu'un déplacement vers des valeurs jamais vues soit détecté."""
    uniques = np.unique(ref)
    if uniques.size <= bins:
        inner = np.concatenate([uniques, np.nextafter(uniques, np.inf)])
        return np.unique(np.concatenate([[-np.inf], inner, [np.inf]]))
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:  # presque toute la masse sur une seule valeur
        values, counts = np.unique(ref, return_counts=True)
        mode = values[np.argmax(counts)]
        return np.array([-np.inf, mode, np.nextafter(mode, np.inf), np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def population_stability_index(
    reference: Any, current: Any, bins: int = 10, eps: float = 1e-4
) -> float:
    """PSI entre une distribution de référence et une distribution courante.

    Règle usuelle : < 0,10 stable ; 0,10-0,25 dérive modérée ; > 0,25 dérive significative.
    Les valeurs non finies sont ignorées ; une distribution vide est refusée.
    """
    ref = np.asarray(reference, dtype="float64")
    cur = np.asarray(current, dtype="float64")
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        raise ValueError("Les distributions de référence et courante doivent être non vides.")
    edges = _psi_edges(ref, bins)
    expected = np.clip(np.histogram(ref, bins=edges)[0] / ref.size, eps, None)
    actual = np.clip(np.histogram(cur, bins=edges)[0] / cur.size, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def ks_statistic(reference: Any, current: Any) -> dict[str, float]:
    result = stats.ks_2samp(np.asarray(reference, float), np.asarray(current, float))
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
    psi_warn: float = cfg.PSI_WARN,
    psi_alert: float = cfg.PSI_ALERT,
) -> list[dict[str, Any]]:
    """PSI et KS par colonne, avec niveau ``ok`` / ``warn`` / ``alert``."""
    rows = []
    for col in columns or cfg.MODEL_FEATURES:
        psi = population_stability_index(reference[col], current[col])
        level = "alert" if psi > psi_alert else "warn" if psi > psi_warn else "ok"
        rows.append(
            {
                "column": col,
                "psi": psi,
                "level": level,
                **ks_statistic(reference[col], current[col]),
            }
        )
    return rows


def window_report(
    reference_X: pd.DataFrame,
    reference_scores: Any,
    current_X: pd.DataFrame,
    current_scores: Any,
    threshold: float,
    reference_alert_rate: float,
    invalid_rate: float = 0.0,
) -> dict[str, Any]:
    """Indicateurs d'une fenêtre de production SANS étiquette + alertes déclenchées."""
    drift = drift_report(reference_X, current_X)
    psi_values = [r["psi"] for r in drift]
    n_alert_cols = sum(r["level"] == "alert" for r in drift)
    score_psi = population_stability_index(reference_scores, current_scores)
    scores_arr = np.asarray(current_scores)
    n_alerts = int((scores_arr >= threshold).sum())
    rate = n_alerts / len(scores_arr)
    z_score = 0.0
    alerts: list[dict[str, str]] = []
    if n_alert_cols:
        alerts.append(
            {
                "name": "data_drift",
                "level": "alert",
                "message": f"{n_alert_cols} variable(s) avec PSI > {cfg.PSI_ALERT}",
            }
        )
    if score_psi > cfg.PSI_ALERT:
        alerts.append(
            {
                "name": "score_drift",
                "level": "alert",
                "message": f"PSI du score = {score_psi:.2f} > {cfg.PSI_ALERT}",
            }
        )
    elif score_psi > cfg.PSI_WARN:
        alerts.append(
            {
                "name": "score_drift",
                "level": "warn",
                "message": f"PSI du score = {score_psi:.2f} > {cfg.PSI_WARN}",
            }
        )
    if 0 < reference_alert_rate < 1:
        sigma = math.sqrt(len(scores_arr) * reference_alert_rate * (1 - reference_alert_rate))
        z_score = (n_alerts - len(scores_arr) * reference_alert_rate) / sigma
        relative = abs(rate - reference_alert_rate) / reference_alert_rate
        if abs(z_score) > cfg.ALERT_RATE_Z and relative > cfg.ALERT_RATE_RELATIVE_CHANGE:
            alerts.append(
                {
                    "name": "alert_rate",
                    "level": "alert",
                    "message": f"taux d'alertes {rate:.4%} vs référence {reference_alert_rate:.4%} "
                    f"(z = {z_score:+.1f})",
                }
            )
    if invalid_rate > 0.01:
        alerts.append(
            {
                "name": "invalid_inputs",
                "level": "alert",
                "message": f"{invalid_rate:.2%} d'entrées invalides",
            }
        )
    return {
        "n": len(current_X),
        "max_feature_psi": float(max(psi_values)),
        "n_features_in_alert": int(n_alert_cols),
        "score_psi": float(score_psi),
        "alert_rate": rate,
        "alert_rate_z": float(z_score),
        "invalid_rate": float(invalid_rate),
        "alerts": alerts,
    }


def alert_rate(scores: Any, threshold: float) -> float:
    """Part des transactions signalées (score >= seuil)."""
    arr = np.asarray(scores, dtype="float64")
    if arr.size == 0 or not math.isfinite(float(arr.sum())):
        raise ValueError("Scores vides ou non finis.")
    return float((arr >= threshold).mean())
