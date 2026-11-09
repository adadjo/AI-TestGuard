"""Entraînement : pipeline scikit-learn, réglage sur validation, seuil, sauvegarde sécurisée.

Principes anti-fuite : toutes les transformations apprises (imputation, mise à l'échelle) sont
DANS le Pipeline et ajustées sur l'entraînement uniquement ; les hyperparamètres et le seuil
sont choisis sur la validation ; le jeu de test n'est utilisé qu'une fois, pour l'évaluation
finale du modèle gelé.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app import __version__
from app import config as cfg
from app.evaluation import evaluate_model, evaluate_pipeline, meets_performance_criteria
from app.preprocessing import (
    DataValidationError,
    Splits,
    preprocess_data,
    split_data,
)
from app.utils import dump_json, sha256_file

logger = logging.getLogger("aitestguard.training")


class TrainingError(ValueError):
    """Données d'entraînement inutilisables (une seule classe, longueurs différentes...)."""


class ModelLoadError(Exception):
    """Artefact de modèle introuvable ou invalide."""


class ModelIntegrityError(ModelLoadError):
    """Empreinte SHA-256 absente ou différente : l'artefact n'est pas chargé."""


def _log1p_nonneg(x: np.ndarray) -> np.ndarray:
    """log(1 + x) avec plancher à 0 (fonction de module : sérialisable par joblib)."""
    return np.log1p(np.clip(x, 0.0, None))


def build_preprocessor(include_time: bool = False) -> ColumnTransformer:
    """Imputation médiane + mise à l'échelle ; Amount passe aussi par log1p (distribution
    très asymétrique). ``keep_empty_features`` préserve les dimensions si une colonne est
    entièrement manquante à l'entraînement."""
    amount = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("log1p", FunctionTransformer(_log1p_nonneg, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    pca = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = [
        ("amount", amount, ["Amount"]),
        ("pca", pca, cfg.V_COLUMNS),
    ]
    if include_time:
        time_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("time", time_pipe, ["Time"]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_estimator(model_name: str, params: dict | None = None, random_state: int = 42) -> Any:
    params = dict(params or {})
    if model_name == "logreg":
        defaults = {"C": 1.0, "class_weight": "balanced", "max_iter": 2000}
        return LogisticRegression(**{**defaults, **params}, random_state=random_state)
    if model_name == "tree":
        defaults = {"max_depth": 6, "min_samples_leaf": 20, "class_weight": "balanced"}
        return DecisionTreeClassifier(**{**defaults, **params}, random_state=random_state)
    raise ValueError(f"Modèle inconnu : {model_name!r} (attendu : 'logreg' ou 'tree').")


def build_pipeline(
    model_name: str = "logreg",
    params: dict | None = None,
    random_state: int = cfg.RANDOM_STATE,
    include_time: bool = False,
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(include_time)),
            ("model", make_estimator(model_name, params, random_state)),
        ]
    )


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "logreg",
    params: dict | None = None,
    random_state: int = cfg.RANDOM_STATE,
    include_time: bool = False,
) -> Pipeline:
    """Ajuste le pipeline complet (transformations + modèle) sur (X, y) UNIQUEMENT."""
    if len(X) == 0 or len(X) != len(y):
        raise TrainingError("X et y doivent être non vides et de même longueur.")
    needed = cfg.MODEL_FEATURES + (["Time"] if include_time else [])
    absent = [c for c in needed if c not in X.columns]
    if absent:
        raise DataValidationError(f"Colonnes d'entraînement absentes : {absent}.")
    classes = set(np.unique(np.asarray(y)).tolist())
    if classes != {0, 1}:
        raise TrainingError("L'entraînement exige les deux classes (0 et 1).")
    pipeline = build_pipeline(model_name, params, random_state, include_time)
    pipeline.fit(X, y)
    return pipeline


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    grid: dict[str, list] | None = None,
    random_state: int = cfg.RANDOM_STATE,
) -> tuple[dict, Pipeline, list[dict]]:
    """Choisit les hyperparamètres par PR-AUC sur la VALIDATION (jamais sur le test)."""
    grid = cfg.GRIDS[model_name] if grid is None else grid
    keys = sorted(grid)
    best: tuple[float, dict, Pipeline] | None = None
    rows: list[dict] = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values, strict=True))
        pipe = train_model(X_train, y_train, model_name, params, random_state)
        proba = pipe.predict_proba(X_val)[:, 1]
        ap = float(average_precision_score(y_val, proba))
        auc = float(roc_auc_score(y_val, proba))
        rows.append({**params, "val_average_precision": ap, "val_roc_auc": auc})
        if best is None or ap > best[0] + 1e-12:
            best = (ap, params, pipe)
    assert best is not None
    return best[1], best[2], rows


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    fpr: float
    recall: float
    precision: float
    n_alerts: int


def select_threshold(
    y_true: Any, y_proba: Any, max_fpr: float = cfg.THRESHOLD_FPR_TARGET
) -> ThresholdChoice:
    """Seuil maximisant le rappel sous la contrainte FPR <= max_fpr (jeu de VALIDATION).

    Règle de décision du modèle : fraude si ``p >= seuil``. À rappel égal, le seuil le plus
    élevé est retenu (moins d'alertes).
    """
    yt = np.asarray(y_true).astype(int)
    proba = np.asarray(y_proba, dtype="float64")
    if yt.shape != proba.shape or yt.size == 0:
        raise ValueError("y_true et y_proba doivent être non vides et de même longueur.")
    if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any():
        raise ValueError("y_proba doit être finie et comprise entre 0 et 1.")
    if set(np.unique(yt).tolist()) != {0, 1}:
        raise ValueError("La sélection du seuil exige les deux classes.")
    if not 0.0 < max_fpr < 1.0:
        raise ValueError("max_fpr doit être dans ]0, 1[.")

    fpr, tpr, thresholds = roc_curve(yt, proba, drop_intermediate=False)
    feasible = (fpr <= max_fpr) & np.isfinite(thresholds)
    if not feasible.any():
        return ThresholdChoice(1.0, 0.0, 0.0, 0.0, 0)
    best_tpr = tpr[feasible].max()
    index = int(np.flatnonzero(feasible & (tpr == best_tpr))[0])
    threshold = float(min(max(thresholds[index], 0.0), 1.0))
    pred = (proba >= threshold).astype(int)
    metrics = evaluate_model(yt, pred, proba)
    return ThresholdChoice(
        threshold, metrics["fpr"], metrics["recall"], metrics["precision"], metrics["n_alerts"]
    )


def select_candidate(val_metrics: dict[str, dict]) -> dict[str, Any]:
    """Règle de sélection R3 (jeu de VALIDATION uniquement).

    Parmi les modèles qui satisfont les critères de performance CA01-CA05 sur la validation,
    retenir celui de meilleure PR-AUC. La règle R1 (meilleure PR-AUC seule, sans contrôler les
    critères) est conservée dans ``r1_pick`` pour comparaison : elle peut désigner un modèle qui
    ne satisfait pas les exigences métier au point de fonctionnement.
    """
    if not val_metrics:
        raise ValueError("Aucun modèle candidat.")
    r1 = max(val_metrics, key=lambda n: val_metrics[n]["average_precision"])
    qualified = sorted(n for n, m in val_metrics.items() if meets_performance_criteria(m))
    pick = max(qualified, key=lambda n: val_metrics[n]["average_precision"]) if qualified else None
    return {"r1_pick": r1, "qualified_on_validation": qualified, "pick": pick}


# ------------------------------------------------------------------------ persistance
def save_model(
    pipeline: Pipeline, threshold: float, path: str | Path, metadata: dict | None = None
) -> dict[str, Any]:
    """Sérialise {pipeline, seuil} avec joblib + fichier JSON compagnon (SHA-256, versions)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": float(threshold), "format_version": 1}, target)
    meta = {
        **(metadata or {}),
        "threshold": float(threshold),
        "features": cfg.MODEL_FEATURES,
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
        "code_version": __version__,
        "sklearn_version": sklearn.__version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    dump_json(meta, target.with_suffix(".json"))
    return meta


def load_model(path: str | Path, verify: bool = True) -> tuple[Pipeline, float, dict]:
    """Recharge un modèle. L'empreinte SHA-256 est vérifiée AVANT toute désérialisation
    (un fichier joblib/pickle peut exécuter du code arbitraire au chargement)."""
    target = Path(path)
    if not target.is_file():
        raise ModelLoadError(f"Artefact introuvable : {target.name}")
    meta_path = target.with_suffix(".json")
    meta: dict = {}
    if verify:
        if not meta_path.is_file():
            raise ModelIntegrityError("Fichier d'empreinte SHA-256 absent : chargement refusé.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("sha256") != sha256_file(target):
            raise ModelIntegrityError("Empreinte SHA-256 invalide : artefact altéré ou corrompu.")
        if meta.get("sklearn_version") not in (None, sklearn.__version__):
            logger.warning("Version de scikit-learn différente de celle de l'entraînement.")
    bundle = joblib.load(target)
    if not isinstance(bundle, dict) or not {"pipeline", "threshold"} <= set(bundle):
        raise ModelLoadError("Format d'artefact inattendu.")
    pipeline = bundle["pipeline"]
    if not hasattr(pipeline, "predict_proba"):
        raise ModelLoadError("L'artefact ne contient pas de modèle probabiliste.")
    return pipeline, float(bundle["threshold"]), meta


# ----------------------------------------------------------- orchestration de bout en bout
@dataclass
class PipelineRun:
    """Résultat d'une exécution complète, avec la PROVENANCE des données de chaque étape."""

    model_name: str
    pipeline: Pipeline
    params: dict
    threshold: float
    splits: Splits
    tuning: list[dict] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    train_metrics: dict = field(default_factory=dict)
    val_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)


def run_training_pipeline(
    df: pd.DataFrame,
    model_name: str = "logreg",
    params: dict | None = None,
    tune: bool = False,
    grid: dict[str, list] | None = None,
    random_state: int = cfg.RANDOM_STATE,
    max_fpr: float = cfg.THRESHOLD_FPR_TARGET,
    splits: Splits | None = None,
    evaluate_test: bool = True,
) -> PipelineRun:
    """Charge -> valide -> sépare -> ajuste (train) -> règle (val) -> évalue une fois (test)."""
    X, y = preprocess_data(df)
    assert y is not None
    splits = splits or split_data(X, y, random_state=random_state)

    tuning: list[dict] = []
    if tune:
        params, pipeline, tuning = tune_model(
            model_name,
            splits.X_train,
            splits.y_train,
            splits.X_val,
            splits.y_val,
            grid,
            random_state,
        )
    else:
        params = dict(params or {})
        pipeline = train_model(splits.X_train, splits.y_train, model_name, params, random_state)

    val_proba = pipeline.predict_proba(splits.X_val)[:, 1]
    choice = select_threshold(splits.y_val, val_proba, max_fpr)

    run = PipelineRun(
        model_name=model_name,
        pipeline=pipeline,
        params=params,
        threshold=choice.threshold,
        splits=splits,
        tuning=tuning,
        provenance={
            "fit_index": splits.X_train.index,
            "tuning_index": splits.X_val.index if tune else None,
            "threshold_index": splits.X_val.index,
            "test_index": splits.X_test.index,
            "test_used_for": "évaluation finale uniquement (modèle et seuil gelés)",
        },
    )
    run.train_metrics, _ = evaluate_pipeline(
        pipeline, splits.X_train, splits.y_train, run.threshold
    )
    run.val_metrics, _ = evaluate_pipeline(pipeline, splits.X_val, splits.y_val, run.threshold)
    if evaluate_test:
        run.test_metrics, _ = evaluate_pipeline(
            pipeline, splits.X_test, splits.y_test, run.threshold
        )
    return run
