"""Données : chargement sûr, validation du contrat, prétraitement sans état, partitions.

Contrat de données (résumé) -- voir aussi docs/plan_de_test.md :

* REJET explicite (``DataValidationError``) : entrée vide ou non DataFrame, colonne
  obligatoire absente ou dupliquée, colonne inattendue, type non numérique, valeur infinie,
  cible hors {0, 1}, Amount < 0 ou > AMOUNT_HARD_LIMIT, |Vi| > V_ABS_HARD_LIMIT, Time < 0,
  taux de valeurs manquantes > MAX_MISSING_FRACTION dans une colonne.
* CORRECTION documentée : valeurs manquantes <= MAX_MISSING_FRACTION par colonne,
  imputées par la médiane APPRISE SUR L'ENTRAÎNEMENT (dans le pipeline, jamais ici).
* ALERTE (traité mais signalé) : Amount > AMOUNT_ALERT_THRESHOLD ; valeurs manquantes.

Cette couche est SANS ÉTAT : aucune statistique n'y est apprise (pas de fuite possible).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app import config as cfg
from app.utils import safe_names, sha256_file

logger = logging.getLogger("aitestguard.preprocessing")


class DataValidationError(ValueError):
    """Donnée non conforme au contrat : rejet explicite (jamais de correction silencieuse)."""


class DataLoadError(Exception):
    """Fichier absent, illisible, vide, trop volumineux ou mal formé."""


@dataclass(frozen=True)
class ValidationReport:
    """Résultat d'une validation réussie (les violations lèvent une exception)."""

    n_rows: int
    n_missing_cells: int
    missing_by_column: dict[str, int]
    warnings: tuple[str, ...] = ()

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


@dataclass(frozen=True)
class Splits:
    """Trois partitions disjointes (entraînement / validation / test)."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    def sizes(self) -> dict[str, dict[str, int]]:
        return {
            name: {"n": len(y), "n_fraud": int(y.sum())}
            for name, y in (("train", self.y_train), ("val", self.y_val), ("test", self.y_test))
        }


def _is_numeric_not_bool(series: pd.Series) -> bool:
    return (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
        and not pd.api.types.is_complex_dtype(series)
    )


def validate_data(
    df: pd.DataFrame, require_target: bool = True, allow_missing: bool = True
) -> ValidationReport:
    """Contrôle le schéma, les types et les domaines de valeurs d'un DataFrame.

    Lève ``DataValidationError`` à la première catégorie de violation rencontrée. Les
    messages citent des NOMS DE COLONNES et des NOMBRES, jamais des valeurs brutes.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("L'entrée doit être un DataFrame pandas.")
    if df.shape[0] == 0:
        raise DataValidationError("Jeu de données vide : aucune observation.")

    duplicated = df.columns[df.columns.duplicated()]
    if len(duplicated) > 0:
        raise DataValidationError(f"Colonnes dupliquées : {safe_names(sorted(set(duplicated)))}.")

    required = list(cfg.FEATURE_COLUMNS) + ([cfg.TARGET_COLUMN] if require_target else [])
    absent = [c for c in required if c not in df.columns]
    if absent:
        raise DataValidationError(f"Colonnes obligatoires absentes : {absent}.")

    allowed = set(cfg.FEATURE_COLUMNS) | {cfg.TARGET_COLUMN}
    extra = [c for c in df.columns if c not in allowed]
    if extra:
        raise DataValidationError(f"{len(extra)} colonne(s) inattendue(s) : {safe_names(extra)}.")

    bad_types = [c for c in cfg.FEATURE_COLUMNS if not _is_numeric_not_bool(df[c])]
    if bad_types:
        raise DataValidationError(f"Types non numériques pour les colonnes : {bad_types}.")

    features = df[cfg.FEATURE_COLUMNS].astype("float64")
    values = features.to_numpy()

    infinite = [
        c
        for c, has_inf in zip(cfg.FEATURE_COLUMNS, np.isinf(values).any(axis=0), strict=True)
        if has_inf
    ]
    if infinite:
        raise DataValidationError(f"Valeurs infinies interdites dans : {infinite}.")

    if require_target:
        target = df[cfg.TARGET_COLUMN]
        if not (_is_numeric_not_bool(target) or pd.api.types.is_bool_dtype(target)):
            raise DataValidationError("La cible 'Class' doit être numérique (0/1).")
        n_bad = int((~target.isin([0, 1])).sum())
        if n_bad:
            raise DataValidationError(
                f"Cible 'Class' : {n_bad} valeur(s) hors de {{0, 1}} (NaN compris)."
            )

    missing = features.isna().sum()
    missing_by_column = {str(c): int(v) for c, v in missing.items() if v > 0}
    n_missing = int(missing.sum())
    if n_missing and not allow_missing:
        raise DataValidationError(
            f"Valeurs manquantes non autorisées dans ce canal : {sorted(missing_by_column)}."
        )
    n_rows = len(df)
    too_many = [c for c, v in missing_by_column.items() if v / n_rows > cfg.MAX_MISSING_FRACTION]
    if too_many:
        raise DataValidationError(
            f"Taux de valeurs manquantes > {cfg.MAX_MISSING_FRACTION:.0%} pour : {sorted(too_many)}."
        )

    amount = features["Amount"]
    if (amount < 0).any():
        raise DataValidationError("Montants négatifs (Amount < 0) : hors contrat.")
    if (amount > cfg.AMOUNT_HARD_LIMIT).any():
        raise DataValidationError(f"Amount > {cfg.AMOUNT_HARD_LIMIT:.0f} : hors contrat.")
    if (features["Time"] < 0).any():
        raise DataValidationError("Time négatif : hors contrat.")
    if (features[cfg.V_COLUMNS].abs() > cfg.V_ABS_HARD_LIMIT).to_numpy().any():
        raise DataValidationError(f"|Vi| > {cfg.V_ABS_HARD_LIMIT:.0f} : hors contrat.")

    warnings: list[str] = []
    n_high = int((amount > cfg.AMOUNT_ALERT_THRESHOLD).sum())
    if n_high:
        warnings.append(
            f"{n_high} montant(s) supérieur(s) au seuil d'alerte {cfg.AMOUNT_ALERT_THRESHOLD:.0f}."
        )
    if n_missing:
        warnings.append(
            f"{n_missing} valeur(s) manquante(s) : imputation par la médiane d'entraînement."
        )
    for message in warnings:
        logger.warning(message)

    return ValidationReport(
        n_rows=n_rows,
        n_missing_cells=n_missing,
        missing_by_column=missing_by_column,
        warnings=tuple(warnings),
    )


def preprocess_data(
    df: pd.DataFrame, require_target: bool = True, allow_missing: bool = True
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Valide puis prépare les variables (sans état) : colonnes ordonnées, float64.

    L'entrée n'est jamais modifiée. La mise à l'échelle et l'imputation, qui APPRENNENT
    des statistiques, sont dans le Pipeline scikit-learn et ajustées sur l'entraînement.
    """
    validate_data(df, require_target=require_target, allow_missing=allow_missing)
    X = df[cfg.FEATURE_COLUMNS].astype("float64").copy()
    y = df[cfg.TARGET_COLUMN].astype("int64").copy() if require_target else None
    return X, y


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Supprime les lignes strictement identiques (évite les doublons entre partitions)."""
    before = len(df)
    out = df.drop_duplicates(keep="first")
    return out, before - len(out)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = cfg.SPLIT["val"],
    test_size: float = cfg.SPLIT["test"],
    random_state: int = cfg.RANDOM_STATE,
) -> Splits:
    """Partition aléatoire STRATIFIÉE en entraînement / validation / test (avant tout fit)."""
    if not (0 < val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
        raise ValueError("val_size et test_size doivent être dans ]0, 1[ avec une somme < 1.")
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X et y doivent être alignés (même longueur, même index).")
    counts = y.value_counts()
    if len(counts) < 2 or int(counts.min()) < 6:
        raise ValueError("Au moins 6 observations par classe sont nécessaires (3 partitions).")

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp,
        y_tmp,
        test_size=val_size / (1.0 - test_size),
        stratify=y_tmp,
        random_state=random_state,
    )
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)


def temporal_split(
    X: pd.DataFrame, y: pd.Series, train_frac: float = 0.6, val_frac: float = 0.2
) -> Splits:
    """Partition chronologique (pas de mélange) : test = transactions les plus récentes."""
    if not (train_frac > 0 and val_frac > 0 and train_frac + val_frac < 1):
        raise ValueError("train_frac et val_frac doivent être > 0 avec une somme < 1.")
    order = np.argsort(X["Time"].to_numpy(), kind="stable")
    X_sorted, y_sorted = X.iloc[order], y.iloc[order]
    n = len(X_sorted)
    i_train, i_val = int(n * train_frac), int(n * (train_frac + val_frac))
    return Splits(
        X_sorted.iloc[:i_train],
        X_sorted.iloc[i_train:i_val],
        X_sorted.iloc[i_val:],
        y_sorted.iloc[:i_train],
        y_sorted.iloc[i_train:i_val],
        y_sorted.iloc[i_val:],
    )


def load_data(path: str | Path | None = None, max_bytes: int = cfg.MAX_FILE_BYTES) -> pd.DataFrame:
    """Charge un CSV avec gestion explicite des erreurs, puis valide le contrat."""
    target = Path(path) if path is not None else cfg.DATA_PATH
    if not target.exists():
        raise DataLoadError(f"Fichier introuvable : {target.name}")
    if not target.is_file():
        raise DataLoadError(f"Ce chemin n'est pas un fichier : {target.name}")
    if target.suffix.lower() != ".csv":
        raise DataLoadError("Extension non autorisée (attendu : .csv).")
    size = target.stat().st_size
    if size == 0:
        raise DataLoadError("Fichier vide.")
    if size > max_bytes:
        raise DataLoadError(f"Fichier trop volumineux ({size} octets > {max_bytes}).")
    try:
        df = pd.read_csv(target)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Fichier CSV illisible ({type(exc).__name__}).") from exc
    validate_data(df)
    return df


def dataset_fingerprint(df: pd.DataFrame, path: str | Path | None = None) -> dict[str, Any]:
    """Caractéristiques du jeu de données à enregistrer avec une référence."""
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    info: dict[str, Any] = {
        "n_rows": len(df),
        "n_columns": int(df.shape[1]),
        "n_fraud": int(df[cfg.TARGET_COLUMN].sum()),
        "fraud_rate": float(df[cfg.TARGET_COLUMN].mean()),
        "content_sha256": hashlib.sha256(row_hashes.tobytes()).hexdigest(),
    }
    if path is not None and Path(path).is_file():
        info["file_sha256"] = sha256_file(path)
    return info
