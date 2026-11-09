"""Perturbations réalistes et mesure de SENSIBILITÉ (Activité 5).

Protocole (documenté, jamais le même bruit arbitraire sur toutes les colonnes) :

* V1..V28 (composantes PCA) : bruit gaussien d'écart-type ``eps * sigma_j`` où ``sigma_j`` est
  l'écart-type de la composante j sur l'ENTRAÎNEMENT (les composantes n'ont pas la même échelle) ;
* Amount : bruit multiplicatif relatif uniforme ``Amount * (1 + u)``, ``u ~ U(-eps, eps)``,
  plancher à 0 (un montant est une grandeur positive) ;
* Time : non perturbé (exclu du modèle ; l'invariance est vérifiée par un test dédié).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app import config as cfg
from app.evaluation import evaluate_model
from app.prediction import predict_fraud

PERTURBATION_PROTOCOL = {
    "V1-V28": "bruit gaussien N(0, (eps x sigma_j)^2), sigma_j = ecart-type sur l'entrainement",
    "Amount": "bruit multiplicatif relatif U(-eps, eps), plancher a 0",
    "Time": "non perturbe (exclu du modele)",
    "amplitude_moderee": {"eps_v": 0.05, "eps_amount": 0.05},
}


def reference_sigma(X_train: pd.DataFrame) -> pd.Series:
    """Écart-type de chaque composante V sur l'entraînement (échelle de référence du bruit)."""
    return X_train[cfg.V_COLUMNS].std(ddof=0)


def perturb_features(
    X: pd.DataFrame,
    sigma_ref: pd.Series,
    eps_v: float = 0.05,
    eps_amount: float = 0.05,
    seed: int = cfg.RANDOM_STATE,
) -> pd.DataFrame:
    """Applique le protocole de bruit ci-dessus (reproductible par ``seed``, entrée intacte)."""
    if eps_v < 0 or eps_amount < 0:
        raise ValueError("Les amplitudes doivent être positives ou nulles.")
    rng = np.random.default_rng(seed)
    out = X.copy()
    scale = eps_v * sigma_ref[cfg.V_COLUMNS].to_numpy()
    out[cfg.V_COLUMNS] = (
        X[cfg.V_COLUMNS].to_numpy() + rng.standard_normal((len(X), len(cfg.V_COLUMNS))) * scale
    )
    u = rng.uniform(-eps_amount, eps_amount, size=len(X))
    out["Amount"] = np.clip(X["Amount"].to_numpy() * (1.0 + u), 0.0, None)
    return out


def inject_missing(
    X: pd.DataFrame, fraction: float, seed: int = cfg.RANDOM_STATE, columns: list[str] | None = None
) -> pd.DataFrame:
    """Met à NaN une fraction aléatoire des cellules des colonnes choisies."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction doit être comprise entre 0 et 1.")
    rng = np.random.default_rng(seed)
    cols = columns or cfg.MODEL_FEATURES
    out = X.copy()
    mask = rng.random((len(X), len(cols))) < fraction
    values = out[cols].to_numpy(dtype="float64", copy=True)
    values[mask] = np.nan
    out[cols] = values
    return out


def apply_covariate_shift(
    X: pd.DataFrame,
    sigma_ref: pd.Series,
    shift_sigma: float = 0.0,
    columns: list[str] | None = None,
    amount_factor: float = 1.0,
) -> pd.DataFrame:
    """Décalage de moyenne de ``shift_sigma`` écarts-types sur ``columns`` et mise à l'échelle
    de Amount : simule une dérive de distribution des entrées."""
    cols = columns or cfg.V_COLUMNS
    out = X.copy()
    out[cols] = X[cols].to_numpy() + shift_sigma * sigma_ref[cols].to_numpy()
    out["Amount"] = X["Amount"].to_numpy() * amount_factor
    return out


def count_invalid_rows(X: pd.DataFrame) -> int:
    """Nombre de lignes qui violent le contrat (non finies, hors domaine) -- ligne par ligne."""
    values = X[cfg.FEATURE_COLUMNS].to_numpy(dtype="float64")
    bad = np.isinf(values).any(axis=1)
    amount, time_ = X["Amount"].to_numpy(dtype="float64"), X["Time"].to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        bad |= (amount < 0) | (amount > cfg.AMOUNT_HARD_LIMIT) | (time_ < 0)
        bad |= (np.abs(X[cfg.V_COLUMNS].to_numpy(dtype="float64")) > cfg.V_ABS_HARD_LIMIT).any(
            axis=1
        )
    return int(bad.sum())


def sensitivity_report(
    model: Any, X: pd.DataFrame, y: Any, X_perturbed: pd.DataFrame, threshold: float
) -> dict[str, Any]:
    """Mesure l'effet d'une perturbation : variation des probabilités, changements de classe
    et variation des métriques au seuil gelé (SENSIBILITÉ, pas invariance universelle)."""
    n_invalid = count_invalid_rows(X_perturbed)
    base = predict_fraud(model, X, threshold)
    pert = predict_fraud(model, X_perturbed, threshold)
    dp = np.abs(pert["fraud_probability"].to_numpy() - base["fraud_probability"].to_numpy())
    flipped = base["is_fraud"].to_numpy() != pert["is_fraud"].to_numpy()
    m0 = evaluate_model(np.asarray(y), base["is_fraud"], base["fraud_probability"])
    m1 = evaluate_model(np.asarray(y), pert["is_fraud"], pert["fraud_probability"])
    return {
        "n": len(X),
        "n_invalid": n_invalid,
        "mean_abs_dp": float(dp.mean()),
        "p99_abs_dp": float(np.quantile(dp, 0.99)),
        "max_abs_dp": float(dp.max()),
        "n_flips": int(flipped.sum()),
        "flip_rate": float(flipped.mean()),
        "flips_to_fraud": int((flipped & (pert["is_fraud"].to_numpy() == 1)).sum()),
        "flips_to_normal": int((flipped & (pert["is_fraud"].to_numpy() == 0)).sum()),
        "delta_recall": float(m1["recall"] - m0["recall"]),
        "delta_precision": float(m1["precision"] - m0["precision"]),
        "delta_fpr": float(m1["fpr"] - m0["fpr"]),
        "delta_average_precision": float(m1["average_precision"] - m0["average_precision"]),
    }


def logreg_delta_bound(pipeline: Any, X: pd.DataFrame, X_perturbed: pd.DataFrame) -> np.ndarray:
    """Borne théorique de |p' - p| pour une régression logistique : la sigmoïde est
    1/4-lipschitzienne, donc |dp| <= 0,25 * ||w||_2 * ||dz||_2 (Cauchy-Schwarz)."""
    pre, model = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    dz = pre.transform(X_perturbed) - pre.transform(X)
    return 0.25 * np.linalg.norm(model.coef_.ravel()) * np.linalg.norm(dz, axis=1)
